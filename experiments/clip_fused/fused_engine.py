"""Fused Triton clip+sum engine — drop-in replacement for _stream_clip_and_sum
(test-harness scope; promotion to opake-engine is Stage 2).

Supported (everything else falls back to the original streaming path and is
counted in ``STATS["fallback"]`` by reason):
  * flat clipping_norm (scalar or 0-dim tensor; no PerGroup)
  * fixed scale kind (no AUTO-S)
  * CUDA leaves of dtype float32 / bfloat16 only (fp16/fp64 -> fallback)
  * second_moment=False
  * compute_dtype=None; reduce_leaf output dtype is the storage dtype or fp32
    (probed; covers both the plain and the microbatch-chunk reducers)
  * no leaf requiring grad under grad mode (Triton outputs carry no autograd)
  * return_aux / return_stats supported: diagnostics "norms" and
    "clipped_norms" (the latter from the STORED values, as in production)

Semantics kept identical to the streaming path it replaces:
  * ONE per-example norm across the whole stacked tree: per-leaf fp32
    squares, fp64 per-2048-element tiles (same block structure as
    _leaf_sq_sum), fp64 cross-leaf accumulation in leaf order
  * ratio = clamp(C, 0) / norm in fp64, then the guard/clamp chain is
    production code: _pytree._finalize_scale(ratio, storage_dtype,
    multiply_dtype, roundoff, clamp_to_one=True) evaluated once PER STORAGE
    DTYPE present in the tree (the guard depends only on storage dtype, so
    this equals production's per-leaf evaluation exactly)
  * per element: sanitize NaN/Inf -> 0, multiply at f32, one cast to storage,
    subnormal round-up -> 0 (bf16; a no-op for fp32 storage), fp32 dim-0 sum,
    cast to output dtype
  * clipped_norms diagnostic: per-example sum of stored^2 at fp32

Launches per leaf: K1 partial sq (+1 tiny torch sum/add), K2 apply+sum
(+1 tiny torch sum/add when aux). No per-example loop on the host, no
tree-sized intermediates.
"""

from __future__ import annotations

from collections import Counter

import torch
import triton
import triton.language as tl

from opake.api.engine.clipping import _clipped_fun as CF
from opake.api.engine.clipping import _pytree as cp
from opake.api.engine.pytree import tree_flatten, tree_unflatten

_ORIGINAL_STREAM = CF._stream_clip_and_sum

_BLOCK = 2048
_SUPPORTED_DTYPES = (torch.float32, torch.bfloat16)
_MAX_N = 2**31 - _BLOCK  # int32 offsets inside a row

STATS: dict = {"fused": 0, "fallback": Counter()}


@triton.jit
def _partial_sq_kernel(X, PARTIAL, N, T, BLOCK: tl.constexpr):
    """PARTIAL[b, t] = sum over tile t of row b of sanitized(x)^2 (fp32 sq, fp64 acc).

    Tiles on grid axis 0 (2^31-1 limit), examples on axis 1 (65535 limit).
    """
    t = tl.program_id(0)
    b = tl.program_id(1)
    offs = t * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + b.to(tl.int64) * N + offs, mask=mask, other=0.0)
    x = tl.where((x - x) == 0.0, x, 0.0)  # NaN/Inf -> 0 (finite iff x-x==0)
    xf = x.to(tl.float32)
    partial = tl.sum((xf * xf).to(tl.float64), axis=0)
    tl.store(PARTIAL + b.to(tl.int64) * T + t, partial)


@triton.jit
def _apply_sum_kernel(X, SCALE, OUT, POSTSQ, N, B, T, BLOCK: tl.constexpr,
                      SM_NORMAL: tl.constexpr, WITH_POSTSQ: tl.constexpr):
    """One column slab, examples in fixed order b = 0..B-1 (deterministic)."""
    t = tl.program_id(0)
    offs = t * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for b in range(B):
        x = tl.load(X + b.to(tl.int64) * N + offs, mask=mask, other=0.0)
        x = tl.where((x - x) == 0.0, x, 0.0)
        s = tl.load(SCALE + b)
        prod = x.to(tl.float32) * s
        stored = prod.to(X.dtype.element_ty)
        sf = stored.to(tl.float32)
        if X.dtype.element_ty == tl.bfloat16:
            sa = tl.abs(sf)
            bad = (sa <= SM_NORMAL) & (sa > tl.abs(prod))
            sf = tl.where(bad, 0.0, sf)
        acc += sf
        if WITH_POSTSQ:
            tl.store(POSTSQ + b.to(tl.int64) * T + t, tl.sum(sf * sf, axis=0))
    tl.store(OUT + offs, acc.to(OUT.dtype.element_ty), mask=mask)


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _unsupported_reason(leaves, clipping_norm, scale, compute_dtype,
                        second_moment, batch_size, reduce_leaf):
    if hasattr(clipping_norm, "groups"):
        return "per_group"
    if scale.kind != "fixed":
        return "auto_s"
    if second_moment:
        return "second_moment"
    if compute_dtype is not None:
        return "compute_dtype"
    if not leaves:
        return "empty_tree"
    if batch_size > 65535:
        return "batch_too_large"  # K1 puts examples on grid axis 1 (65535 cap)
    grad_mode = torch.is_grad_enabled()
    for leaf in leaves:
        if not isinstance(leaf, torch.Tensor):
            return "non_tensor_leaf"
        if not leaf.is_cuda:
            return "non_cuda"
        if leaf.dtype not in _SUPPORTED_DTYPES:
            return f"dtype:{leaf.dtype}"
        if grad_mode and leaf.requires_grad:
            return "requires_grad"
        if leaf.dim() == 0 or leaf.shape[0] != batch_size:
            return "batch_mismatch"
        n = leaf.numel() // batch_size
        if n == 0:
            return "empty_leaf"
        if n > _MAX_N:
            return "leaf_too_large"
    return None


def _output_dtypes(leaves, reduce_leaf):
    """Probe the caller's reduce_leaf once per storage dtype.

    Non-chunk path: _sum_clipped_tensor -> output == storage dtype.
    Microbatch chunk path: torch.sum(dtype=_accumulation_dtype) -> fp32 for
    bf16 storage (the accumulator stays at fp32 across chunks). The kernel
    accumulates at fp32, so both are exact; anything else falls back.
    """
    targets = {}
    for dt in {leaf.dtype for leaf in leaves}:
        probe = reduce_leaf(torch.zeros((1,), dtype=dt, device=leaves[0].device))
        if probe.dtype not in (dt, torch.float32):
            return None
        targets[dt] = probe.dtype
    return targets


def _fused_impl(leaves, clipping_norm, *, with_postsq: bool, out_dtypes=None):
    """Core fused clip+sum on stacked leaves. No support checks (tests only)."""
    B = leaves[0].shape[0]
    dev = leaves[0].device
    acc_dtype = cp._resolve_compute_dtype_for_reduction(leaves, None)
    sq_dtype = cp._sq_accum_dtype(leaves)
    roundoff = cp._norm_roundoff(
        acc_dtype, sq_dtype, len(leaves),
        cp._reduction_terms([leaf[0] for leaf in leaves]),
    )

    # pass 1: tree-level per-example squared norm
    sq_total = None
    for leaf in leaves:
        x = leaf.contiguous()
        N = x.numel() // B
        T = _cdiv(N, _BLOCK)
        partial = torch.empty((B, T), dtype=sq_dtype, device=dev)
        _partial_sq_kernel[(T, B)](x, partial, N, T, BLOCK=_BLOCK)
        sq = partial.sum(1)
        sq_total = sq if sq_total is None else sq_total + sq
    norm = torch.sqrt(sq_total)

    # scalar chain: production code, once per storage dtype
    bound = torch.clamp(
        torch.as_tensor(clipping_norm, dtype=norm.dtype, device=norm.device),
        min=0.0,
    )
    ratio = bound / norm
    scales = {}
    for dt in {leaf.dtype for leaf in leaves}:
        multiply_dtype = torch.promote_types(dt, acc_dtype)
        if multiply_dtype != torch.float32:
            raise AssertionError(f"kernel multiplies at f32, got {multiply_dtype}")
        s = cp._finalize_scale(ratio, dt, multiply_dtype, roundoff,
                               clamp_to_one=True)
        scales[dt] = s.to(dtype=multiply_dtype).contiguous()

    # pass 2: apply + sum (+ stored-square diagnostics)
    post_sq = None
    reduced = []
    for leaf in leaves:
        x = leaf.contiguous()
        N = x.numel() // B
        T = _cdiv(N, _BLOCK)
        out_dt = (out_dtypes or {}).get(leaf.dtype, leaf.dtype)
        out = torch.empty(leaf.shape[1:], dtype=out_dt, device=dev)
        p2 = (torch.empty((B, T), dtype=torch.float32, device=dev)
              if with_postsq else scales[leaf.dtype])
        sm = (float(torch.finfo(torch.bfloat16).smallest_normal)
              if leaf.dtype == torch.bfloat16 else 0.0)
        _apply_sum_kernel[(T,)](x, scales[leaf.dtype], out, p2, N, B, T,
                                BLOCK=_BLOCK, SM_NORMAL=sm,
                                WITH_POSTSQ=with_postsq, num_warps=4)
        if with_postsq:
            s2 = p2.sum(1)
            post_sq = s2 if post_sq is None else post_sq + s2
        reduced.append(out)
    return reduced, norm, post_sq, acc_dtype


def fused_stream_clip_and_sum(
    values,
    clipping_norm,
    *,
    scale,
    batch_size,
    reduce_leaf,
    compute_dtype,
    second_moment,
    return_aux,
    return_stats,
):
    leaves, treedef = tree_flatten(values)
    reason = _unsupported_reason(leaves, clipping_norm, scale, compute_dtype,
                                 second_moment, batch_size, reduce_leaf)
    out_dtypes = None
    if reason is None:
        out_dtypes = _output_dtypes(leaves, reduce_leaf)
        if out_dtypes is None:
            reason = "output_dtype"
    if reason is not None:
        STATS["fallback"][reason] += 1
        return _ORIGINAL_STREAM(
            values, clipping_norm, scale=scale, batch_size=batch_size,
            reduce_leaf=reduce_leaf, compute_dtype=compute_dtype,
            second_moment=second_moment, return_aux=return_aux,
            return_stats=return_stats,
        )
    STATS["fused"] += 1
    reduced, norm, post_sq, acc_dtype = _fused_impl(
        leaves, clipping_norm, with_postsq=return_aux, out_dtypes=out_dtypes)

    diagnostics = {}
    if return_aux or return_stats:
        diagnostics["norms"] = norm.to(acc_dtype).detach()
    if return_aux:
        diagnostics["clipped_norms"] = torch.sqrt(post_sq).detach()
    markers = [leaf.new_zeros(()) for leaf in leaves]
    return (
        tree_unflatten(treedef, reduced),
        tree_unflatten(treedef, markers),
        (),
        (),
        diagnostics if (return_aux or return_stats) else (),
    )


def install() -> None:
    CF._stream_clip_and_sum = fused_stream_clip_and_sum


def uninstall() -> None:
    CF._stream_clip_and_sum = _ORIGINAL_STREAM
