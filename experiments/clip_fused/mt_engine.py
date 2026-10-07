"""Multi-tensor variant of the fused clip-and-sum (prototype, not packaged).

The packaged kernel (`opake.api.patches.kernels._clip_sum.fused_clip_sum`)
launches two kernels per parameter tensor plus a few small torch ops per
tensor, so a 392-leaf LoRA tree costs thousands of launches per call. This
version processes every leaf of one storage dtype in a single launch per pass,
the "multi-tensor apply" pattern of apex's `amp_C.multi_tensor_l2norm` and
PyTorch's `torch._foreach_*`. Each program looks up its leaf in a cached tile
table and reads the leaf's base address from a per-call pointer table.

Per dtype group (at most two: float32, bfloat16):
  K1  grid (tiles, B): float64 sum of squares of one 2048-element tile of one
      example of one leaf (sanitized on read, float32 squares).
  K1b grid (leaves, B): per-leaf sum of that leaf's tile partials.
Then: cross-leaf sum over leaves, norm, and production `_finalize_scale` per
storage dtype (unchanged from the packaged kernel).
  K2  grid (tiles,): scale, cast to storage, subnormal nudge, sum over
      examples b = 0..B-1 in fixed order, written into one flat output buffer
      per output dtype (leaves are views into it).

Semantics and rounding budget:
- The reduction keeps production's structure: float32 squares, float64
  partials per 2048-element tile, per-leaf sum, then a cross-leaf sum over at
  most n_leaves terms. That stays within `_norm_roundoff`'s assumptions.
- Per-element arithmetic is identical to the packaged kernel.
- Results equal the packaged kernel's except for float64 summation order in
  the norm, which never moved an fp32 scale in the attribution runs.

Install by monkeypatching `_clip_sum.fused_clip_sum`; the packaged
`fused_stream_clip_and_sum` keeps its support checks and strict/fallback rules.
`install_vmap_chunk(k)` separately makes the streaming path call
`torch.func.vmap(..., chunk_size=k)` (prototype of report Tier-2 #6).
"""

from __future__ import annotations

import importlib

import torch
import triton
import triton.language as tl

from opake.api.engine.clipping import _pytree as cp

_BLOCK = 2048
_TB = 1024  # tile partials summed per step in K1b


@triton.jit
def _mt_partial_sq_kernel(X0, PTRS, SIZES, TILE_LEAF, TILE_LOCAL, PARTIAL, T,
                          BLOCK: tl.constexpr):
    t = tl.program_id(0)
    b = tl.program_id(1)
    leaf = tl.load(TILE_LEAF + t)
    local = tl.load(TILE_LOCAL + t)
    n = tl.load(SIZES + leaf)
    base = tl.load(PTRS + leaf).to(tl.pointer_type(X0.dtype.element_ty))
    offs = local.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(base + b.to(tl.int64) * n + offs, mask=mask, other=0.0)
    x = tl.where((x - x) == 0.0, x, 0.0)  # NaN/Inf -> 0
    xf = x.to(tl.float32)
    tl.store(PARTIAL + b.to(tl.int64) * T + t, tl.sum((xf * xf).to(tl.float64), axis=0))


@triton.jit
def _mt_leaf_sum_kernel(PARTIAL, TILE_START, LEAF_COL, LEAF_SQ, T, L_TOTAL,
                        TB: tl.constexpr):
    leaf = tl.program_id(0)
    b = tl.program_id(1)
    t0 = tl.load(TILE_START + leaf)
    t1 = tl.load(TILE_START + leaf + 1)
    acc = tl.zeros((TB,), dtype=tl.float64)
    for start in range(t0, t1, TB):
        idx = start + tl.arange(0, TB)
        acc += tl.load(PARTIAL + b.to(tl.int64) * T + idx, mask=idx < t1, other=0.0)
    col = tl.load(LEAF_COL + leaf)
    tl.store(LEAF_SQ + b.to(tl.int64) * L_TOTAL + col, tl.sum(acc, axis=0))


@triton.jit
def _mt_apply_sum_kernel(X0, PTRS, SIZES, TILE_LEAF, TILE_LOCAL, SCALE, OUT,
                         OUT_OFFS, POSTSQ, B, T, BLOCK: tl.constexpr,
                         SM_NORMAL: tl.constexpr, WITH_POSTSQ: tl.constexpr):
    t = tl.program_id(0)
    leaf = tl.load(TILE_LEAF + t)
    local = tl.load(TILE_LOCAL + t)
    n = tl.load(SIZES + leaf)
    base = tl.load(PTRS + leaf).to(tl.pointer_type(X0.dtype.element_ty))
    out_base = OUT + tl.load(OUT_OFFS + leaf)
    offs = local.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for b in range(B):
        x = tl.load(base + b.to(tl.int64) * n + offs, mask=mask, other=0.0)
        x = tl.where((x - x) == 0.0, x, 0.0)
        s = tl.load(SCALE + b)
        prod = x.to(tl.float32) * s
        stored = prod.to(X0.dtype.element_ty)
        sf = stored.to(tl.float32)
        if X0.dtype.element_ty == tl.bfloat16:
            sa = tl.abs(sf)
            sf = tl.where((sa <= SM_NORMAL) & (sa > tl.abs(prod)), 0.0, sf)
        acc += sf
        if WITH_POSTSQ:
            tl.store(POSTSQ + b.to(tl.int64) * T + t, tl.sum(sf * sf, axis=0))
    tl.store(out_base + offs, acc.to(OUT.dtype.element_ty), mask=mask)


_PLANS: dict = {}


def _plan(leaves, device):
    """Tile tables per storage dtype, cached by the tree's per-example shapes."""
    key = (device, tuple((leaf.dtype, tuple(leaf.shape[1:])) for leaf in leaves))
    plan = _PLANS.get(key)
    if plan is not None:
        return plan
    groups = []
    for dtype in sorted({leaf.dtype for leaf in leaves}, key=str):
        idx = [i for i, leaf in enumerate(leaves) if leaf.dtype == dtype]
        sizes, tile_leaf, tile_local, tile_start, out_offs = [], [], [], [0], []
        offset = 0
        for j, i in enumerate(idx):
            n = leaves[i][0].numel()
            tiles = (n + _BLOCK - 1) // _BLOCK
            sizes.append(n)
            tile_leaf += [j] * tiles
            tile_local += list(range(tiles))
            tile_start.append(tile_start[-1] + tiles)
            out_offs.append(offset)
            offset += n
        dev = lambda v, dt: torch.tensor(v, dtype=dt, device=device)  # noqa: E731
        groups.append({
            "dtype": dtype, "idx": idx, "total": offset, "T": len(tile_leaf),
            "sizes": dev(sizes, torch.int64), "tile_leaf": dev(tile_leaf, torch.int32),
            "tile_local": dev(tile_local, torch.int32), "tile_start": dev(tile_start, torch.int64),
            "leaf_col": dev(idx, torch.int32), "out_offs": dev(out_offs, torch.int64),
            "offsets": out_offs, "sizes_host": sizes,
        })
    plan = {"groups": groups, "n_leaves": len(leaves)}
    _PLANS[key] = plan
    return plan


def mt_fused_clip_sum(leaves, clipping_norm, *, with_postsq: bool, out_dtypes=None):
    """Same contract as `_clip_sum.fused_clip_sum`."""
    batch = leaves[0].shape[0]
    device = leaves[0].device
    leaves = [leaf if leaf.is_contiguous() else leaf.contiguous() for leaf in leaves]
    acc_dtype = cp._resolve_compute_dtype_for_reduction(leaves, None)
    sq_dtype = cp._sq_accum_dtype(leaves)
    roundoff = cp._norm_roundoff(acc_dtype, sq_dtype, len(leaves),
                                 cp._reduction_terms([leaf[0] for leaf in leaves]))
    plan = _plan(leaves, device)
    n_leaves = plan["n_leaves"]

    leaf_sq = torch.empty((batch, n_leaves), dtype=sq_dtype, device=device)
    ptrs = {}
    for g in plan["groups"]:
        ptrs[g["dtype"]] = torch.tensor([leaves[i].data_ptr() for i in g["idx"]],
                                        dtype=torch.int64, device=device)
        partial = torch.empty((batch, g["T"]), dtype=sq_dtype, device=device)
        x0 = leaves[g["idx"][0]]
        _mt_partial_sq_kernel[(g["T"], batch)](
            x0, ptrs[g["dtype"]], g["sizes"], g["tile_leaf"], g["tile_local"], partial,
            g["T"], BLOCK=_BLOCK)
        _mt_leaf_sum_kernel[(len(g["idx"]), batch)](
            partial, g["tile_start"], g["leaf_col"], leaf_sq, g["T"], n_leaves, TB=_TB)
    norm = torch.sqrt(leaf_sq.sum(1))  # cross-leaf: <= n_leaves terms

    bound = torch.clamp(torch.as_tensor(clipping_norm, dtype=norm.dtype, device=device), min=0.0)
    ratio = bound / norm
    reduced = [None] * n_leaves
    post_sq = None
    for g in plan["groups"]:
        dtype = g["dtype"]
        multiply_dtype = torch.promote_types(dtype, acc_dtype)
        assert multiply_dtype == torch.float32
        scale = cp._finalize_scale(ratio, dtype, multiply_dtype, roundoff,
                                   clamp_to_one=True).to(multiply_dtype).contiguous()
        out_dtype = (out_dtypes or {}).get(dtype, dtype)
        flat = torch.empty((g["total"],), dtype=out_dtype, device=device)
        post = (torch.empty((batch, g["T"]), dtype=torch.float32, device=device)
                if with_postsq else scale)
        smallest = float(torch.finfo(torch.bfloat16).smallest_normal) if dtype == torch.bfloat16 else 0.0
        x0 = leaves[g["idx"][0]]
        _mt_apply_sum_kernel[(g["T"],)](
            x0, ptrs[dtype], g["sizes"], g["tile_leaf"], g["tile_local"], scale, flat,
            g["out_offs"], post, batch, g["T"], BLOCK=_BLOCK, SM_NORMAL=smallest,
            WITH_POSTSQ=with_postsq, num_warps=4)
        for i, off, n in zip(g["idx"], g["offsets"], g["sizes_host"]):
            reduced[i] = flat[off:off + n].view(leaves[i].shape[1:])
        if with_postsq:
            s2 = post.sum(1)
            post_sq = s2 if post_sq is None else post_sq + s2
    return reduced, norm, post_sq, acc_dtype


_CS = importlib.import_module("opake.api.patches.kernels._clip_sum")
_ORIGINAL_FUSED = _CS.fused_clip_sum
_ORIGINAL_STREAM_FN = _CS.fused_stream_clip_and_sum
_MARKERS: dict = {}


def _shared_marker(leaf):
    """One 0-dim zero per (dtype, device). Markers are only read for .dtype."""
    k = (leaf.dtype, leaf.device)
    m = _MARKERS.get(k)
    if m is None:
        m = _MARKERS[k] = leaf.new_zeros(())
    return m


def mt_fused_stream_clip_and_sum(values, clipping_norm, *, scale, batch_size, reduce_leaf,
                                 compute_dtype, second_moment, return_aux, return_stats,
                                 strict):
    """Packaged wrapper with shared dtype markers; same checks and fallback."""
    from opake.api.engine.pytree import tree_flatten, tree_unflatten
    from opake.exceptions import ConfigurationError

    leaves, treedef = tree_flatten(values)
    reason = _CS.unsupported_reason(leaves, clipping_norm, scale, compute_dtype=compute_dtype,
                                    second_moment=second_moment, batch_size=batch_size)
    out_dtypes = None
    if reason is None:
        out_dtypes = _CS._output_dtypes(leaves, reduce_leaf)
        if out_dtypes is None:
            reason = "the requested output dtype is not supported"
    if reason is not None:
        if strict:
            raise ConfigurationError(*(f"clip_backend='triton' cannot run this call: {reason}.",))
        return _CS._stream_clip_and_sum(values, clipping_norm, scale=scale, batch_size=batch_size,
                                        reduce_leaf=reduce_leaf, compute_dtype=compute_dtype,
                                        second_moment=second_moment, return_aux=return_aux,
                                        return_stats=return_stats)
    reduced, norm, post_sq, acc_dtype = mt_fused_clip_sum(
        leaves, clipping_norm, with_postsq=return_aux, out_dtypes=out_dtypes)
    diagnostics = {}
    if return_aux or return_stats:
        diagnostics["norms"] = norm.to(acc_dtype).detach()
    if return_aux:
        diagnostics["clipped_norms"] = torch.sqrt(post_sq).detach()
    markers = [_shared_marker(leaf) for leaf in leaves]
    return (tree_unflatten(treedef, reduced), tree_unflatten(treedef, markers), (), (),
            diagnostics if (return_aux or return_stats) else ())


def install(shared_markers: bool = True):
    _CS.fused_clip_sum = mt_fused_clip_sum
    if shared_markers:
        _CS.fused_stream_clip_and_sum = mt_fused_stream_clip_and_sum


def uninstall():
    _CS.fused_clip_sum = _ORIGINAL_FUSED
    _CS.fused_stream_clip_and_sum = _ORIGINAL_STREAM_FN


_CF = importlib.import_module("opake.api.engine.clipping._clipped_fun")
_ORIGINAL_VMAP_VALUES = _CF._vmap_values


def install_vmap_chunk(chunk_size: int):
    """Streaming path: one vmap over the full batch, chunked inside vmap."""
    from torch.func import vmap

    def _vmap_values_chunked(fun_with_aux, in_dims, args, kwargs, return_aux):
        if return_aux:
            return vmap(fun_with_aux, in_dims=in_dims, out_dims=(0, 0),
                        randomness="same", chunk_size=chunk_size)(*args, **kwargs)

        def value_only(*a, **k):
            value, _ = fun_with_aux(*a, **k)
            return value

        return vmap(value_only, in_dims=in_dims, out_dims=0, randomness="same",
                    chunk_size=chunk_size)(*args, **kwargs), None

    _CF._vmap_values = _vmap_values_chunked


def uninstall_vmap_chunk():
    _CF._vmap_values = _ORIGINAL_VMAP_VALUES
