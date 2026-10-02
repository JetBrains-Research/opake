"""CUDA-graph capture of Opake's per-microbatch chunk kernel (prototype).

Use as ``clipped_grad(..., _chunk_compiler=CudaGraphChunk)`` (or the
``install()`` hooks below). The chunk kernel is
``chunk(kernel_clipping_norm, *chunk_args, **kwargs) -> (reduced, markers,
squared_reduced, squared_markers, diagnostics)``.

Correctness rules:

* Capture key = pytree structure + per-tensor (shape, stride, dtype, device,
  requires_grad) + every non-tensor leaf *value*. A changed Python value
  (for example a float threshold) triggers a new capture instead of replaying
  a graph that baked in the old one.
* Every tensor input is copied into its static buffer before each replay
  (one ``torch._foreach_copy_``), including the clipping-norm tensor, so a
  threshold that changes every step (adaptive clipping) is always current.
* Outputs are cloned on return: the next replay overwrites the graph's
  output buffers, and the microbatch accumulator keeps the first result.
* Capture follows the PyTorch CUDA-graph recipe: side-stream warmup runs,
  then capture into a pool shared by all keys (replays are sequential).
* ``check`` re-runs the eager kernel on the same inputs on the first replays
  of each key and every ``check_every`` replays after, and records whether
  the outputs are bitwise equal (``CudaGraphChunk.report()``).

Host syncs inside the kernel make capture fail loudly (CUDA forbids them
while capturing); nothing falls back silently.
"""

from __future__ import annotations

import time

import torch
import torch.utils._pytree as pytree

_STATS = {"captures": 0, "replays": 0, "capture_s": 0.0, "checks": 0, "check_mismatch": 0,
          "max_abs_diff": 0.0, "keys": []}


def _leaf_key(x):
    if isinstance(x, torch.Tensor):
        return ("T", tuple(x.shape), tuple(x.stride()), x.dtype, str(x.device), x.requires_grad)
    try:
        hash(x)
        return ("V", type(x).__name__, x)
    except TypeError:
        return ("V", type(x).__name__, repr(x))


class CudaGraphChunk:
    """Wrap a chunk kernel; capture once per input signature, then replay."""

    check = False
    check_first = 3
    check_every = 20
    warmup = 2
    _pool = None
    _warmed_up = False

    def __init__(self, fn):
        self.fn = fn
        self.graphs = {}
        self.replays = {}

    def __call__(self, *args, **kwargs):
        flat, spec = pytree.tree_flatten((args, kwargs))
        key = (str(spec), tuple(_leaf_key(x) for x in flat))
        entry = self.graphs.get(key)
        if entry is None:
            entry = self.graphs[key] = self._capture(flat, spec)
        graph, static_flat, static_out = entry
        idx = [i for i, x in enumerate(flat) if isinstance(x, torch.Tensor)]
        dst = [static_flat[i] for i in idx]
        src = [flat[i] for i in idx]
        torch._foreach_copy_(dst, src)
        graph.replay()
        out = pytree.tree_map_only(torch.Tensor, torch.clone, static_out)
        n = self.replays[key] = self.replays.get(key, 0) + 1
        _STATS["replays"] += 1
        if self.check and (n <= self.check_first or n % self.check_every == 0):
            self._check(args, kwargs, out)
        return out

    def _capture(self, flat, spec):
        t0 = time.perf_counter()
        static_flat = [x.clone() if isinstance(x, torch.Tensor) else x for x in flat]
        s_args, s_kwargs = pytree.tree_unflatten(static_flat, spec)
        # Warm up (lazy init, cuBLAS handles, autotuning) only before the first
        # capture. Later keys capture straight into the shared pool, reusing its
        # freed blocks: an eager warmup there would need a second activation-sized
        # allocation next to the pool (OOM at microbatch 4 on 40 GB).
        n_warm = 0 if CudaGraphChunk._warmed_up else self.warmup
        if n_warm:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(n_warm):
                    self.fn(*s_args, **s_kwargs)
            torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()  # release warmup blocks cached for the side stream
        if CudaGraphChunk._pool is None:
            CudaGraphChunk._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=CudaGraphChunk._pool):
            static_out = self.fn(*s_args, **s_kwargs)
        torch.cuda.synchronize()
        CudaGraphChunk._warmed_up = True
        _STATS["captures"] += 1
        _STATS["capture_s"] += time.perf_counter() - t0
        _STATS["keys"].append([k for k in (_leaf_key(x) for x in flat) if k[0] == "T"][-1][1])
        print(f"[cudagraph] captured key #{_STATS['captures']} in {time.perf_counter() - t0:.1f}s", flush=True)
        return graph, static_flat, static_out

    def _check(self, args, kwargs, out):
        """Graph vs eager, and eager vs eager (the kernel's own run-to-run noise)."""
        ref = self.fn(*args, **kwargs)
        ref2 = self.fn(*args, **kwargs)
        _STATS["checks"] += 1
        g_vs_e = _compare(out, ref)
        e_vs_e = _compare(ref2, ref)
        same = all(r["bitwise"] for r in g_vs_e.values())
        if not same:
            _STATS["check_mismatch"] += 1
        _STATS.setdefault("check_detail", []).append({"graph_vs_eager": g_vs_e, "eager_vs_eager": e_vs_e})
        fmt = lambda d: " ".join(f"{k}:{'=' if v['bitwise'] else f'{v[chr(114)+chr(101)+chr(108)]:.1e}'}" for k, v in d.items())  # noqa: E731
        print(f"[cudagraph] check #{_STATS['checks']}: graph-vs-eager [{fmt(g_vs_e)}] | eager-vs-eager [{fmt(e_vs_e)}]", flush=True)


def _compare(a_tree, b_tree):
    """Per output field (reduced, diagnostics...): bitwise flag and max relative diff."""
    names = ("reduced", "markers", "squared_reduced", "squared_markers", "diagnostics")
    res = {}
    for name, a, b in zip(names, a_tree, b_tree, strict=True):
        fa, _ = pytree.tree_flatten(a)
        fb, _ = pytree.tree_flatten(b)
        ta = [x for x in fa if isinstance(x, torch.Tensor) and x.numel()]
        tb = [x for x in fb if isinstance(x, torch.Tensor) and x.numel()]
        if not ta:
            continue
        bitwise = all(torch.equal(x, y) for x, y in zip(ta, tb, strict=True))
        diff2 = sum(float(((x.double() - y.double()) ** 2).sum()) for x, y in zip(ta, tb, strict=True))
        ref2 = sum(float((y.double() ** 2).sum()) for y in tb)
        rel = (diff2 / ref2) ** 0.5 if ref2 > 0 else (0.0 if diff2 == 0 else float("inf"))
        res[name] = {"bitwise": bitwise, "rel": rel}  # global ||a-b|| / ||b|| over the field
    return res


def report():
    s = dict(_STATS)
    s["keys"] = list(s["keys"])
    return s


_PATCHED = {}


def install(check: bool = False):
    """Use ``CudaGraphChunk`` as the chunk compiler wherever none is set.

    Patches ``clipped_grad`` / ``adaptive_clipped_grad`` in opake.dpsgd.clipping
    and ``clipped_grad`` in opake.api.engine.clipping (DPTrainer imports both),
    before the caller imports them. An explicit ``_chunk_compiler=None`` (how
    DPTrainer passes "no compile") is replaced; a real compiler is kept.
    """
    import opake.api.engine.clipping as ec
    import opake.dpsgd.clipping as dc

    CudaGraphChunk.check = check
    for mod, name in ((dc, "clipped_grad"), (dc, "adaptive_clipped_grad"), (ec, "clipped_grad")):
        orig = getattr(mod, name)
        _PATCHED[(mod.__name__, name)] = orig

        def wrapped(*a, _orig=orig, **k):
            if k.get("_chunk_compiler") is None:
                k["_chunk_compiler"] = CudaGraphChunk
            return _orig(*a, **k)

        setattr(mod, name, wrapped)
