"""Fused Triton clip+sum engine — drop-in replacement for _stream_clip_and_sum
(test-harness scope; promotion to opake-engine pending e2e results).

Scope of v0 (everything else falls back to the original streaming path):
  * flat scalar clipping_norm (no PerGroup)
  * fixed scale kind (no AUTO-S)
  * real floating-point leaves (no complex / non-float)
  * second_moment=False, return_aux=False, return_stats=False

Semantics kept identical to the streaming path it replaces:
  * ONE per-example norm across the whole stacked tree (cross-leaf fp64
    accumulation, per-leaf sanitized) — NOT per-leaf norms
  * ratio = C / norm in fp64 (no eps), tree-level roundoff guard constant,
    guard BEFORE min(1), non-finite -> 0, clamp >= 0 — computed in fp64 as
    the production scalar chain does, then multiplied per leaf at f32 with
    one cast to storage + bf16 subnormal nudge (inside K2)
  * reduce_leaf provided by the caller is used for the batch-dim sum
    (here: torch sum fp32-accumulate + cast back — matches
    _sum_clipped_tensor, with the known difference of not materializing
    the full clipped copy)
  * sanitize NaN/+-Inf -> 0 on read, in both passes (inside the kernels)

Launch shape per leaf: K1 (partial sq) -> [combine across leaves once] ->
K2 (scale+apply+sum). No per-example loop, no tree-sized intermediates
beyond the tiny [B, T] fp64 partials.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from opake.api.engine.clipping import _clipped_fun as CF
from opake.api.engine.clipping import _pytree as cp
from opake.api.engine.pytree import tree_flatten, tree_unflatten

_ORIGINAL_STREAM = CF._stream_clip_and_sum

_BLOCK = 2048


@triton.jit
def _partial_sq_kernel(X, PARTIAL, N, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    t = tl.program_id(1)
    offs = t * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + b.to(tl.int64) * N + offs, mask=mask, other=0.0)
    x = tl.where((x - x) == 0.0, x, 0.0)
    xf = x.to(tl.float32)
    partial = tl.sum((xf * xf).to(tl.float64), axis=0)
    T = tl.num_programs(1)
    tl.store(PARTIAL + b * T + t, partial)


@triton.jit
def _apply_sum_kernel(X, SCALE, OUT, N, B, BLOCK: tl.constexpr,
                      SM_NORMAL: tl.constexpr):
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
        if X.dtype.element_ty == tl.bfloat16:
            sa = tl.abs(stored.to(tl.float32))
            bad = (sa <= SM_NORMAL) & (sa > tl.abs(prod))
            stored = tl.where(bad, 0.0, stored)
        acc += stored.to(tl.float32)
    tl.store(OUT + offs, acc.to(OUT.dtype.element_ty), mask=mask)


def _supported(values, clipping_norm, scale, second_moment, return_aux,
               return_stats):
    if isinstance(clipping_norm, dict) or hasattr(clipping_norm, "groups"):
        return False  # PerGroup: defer
    if scale.kind != "fixed":
        return False
    if second_moment or return_aux or return_stats:
        return False  # diagnostics paths not implemented in the prototype
    leaves, _ = tree_flatten(values)
    if not leaves:
        return False
    for leaf in leaves:
        if not isinstance(leaf, torch.Tensor):
            return False
        if leaf.dtype.is_complex or not leaf.dtype.is_floating_point:
            return False
        if not leaf.is_cuda:
            return False
    return True


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
    if not _supported(values, clipping_norm, scale, second_moment,
                      return_aux, return_stats):
        return _ORIGINAL_STREAM(
            values, clipping_norm, scale=scale, batch_size=batch_size,
            reduce_leaf=reduce_leaf, compute_dtype=compute_dtype,
            second_moment=second_moment, return_aux=return_aux,
            return_stats=return_stats,
        )

    leaves, treedef = tree_flatten(values)
    acc_dtype = cp._resolve_compute_dtype_for_reduction(leaves, compute_dtype)
    sq_dtype = cp._sq_accum_dtype(leaves)  # fp64 on CUDA
    roundoff = cp._norm_roundoff(
        acc_dtype, sq_dtype, len(leaves),
        cp._reduction_terms([leaf[0] for leaf in leaves]),
    )
    B = batch_size

    # ---- pass 1: cross-leaf per-example squared norm, streaming per leaf
    sq_total = torch.zeros(B, dtype=sq_dtype, device=leaves[0].device)
    for leaf in leaves:
        N = leaf.numel() // B
        T = (N + _BLOCK - 1) // _BLOCK
        partial = torch.empty((B, T), dtype=sq_dtype, device=leaf.device)
        _partial_sq_kernel[(B, T)](leaf.contiguous(), partial, N, BLOCK=_BLOCK)
        sq_total += partial.sum(1)
        del partial
    norm = torch.sqrt(sq_total)

    # ---- scalar chain, fp64, on-device (no sync). kernel_clipping_norm
    # arrives as a 0-dim tensor from _prepare_kernel_clipping_norm.
    if isinstance(clipping_norm, torch.Tensor):
        C = torch.clamp(clipping_norm.reshape(()).to(sq_dtype), min=0.0)
    else:
        C = torch.clamp(
            torch.as_tensor(float(clipping_norm), dtype=sq_dtype,
                            device=sq_total.device),
            min=0.0,
        )
    ratio = C / norm
    # Guard shrink: production computes it PER LEAF from that leaf's storage
    # dtype inside _finalize_scale. One shared scale vector forces the most
    # conservative leaf (bf16 if present, else fp32) — equal for all-fp32
    # trees, slightly MORE shrink (bound-safe, small utility cost) on mixed
    # trees. Flagged for the packaged version (per-leaf scale or documented
    # conservative contract).
    any_bf16 = any(l.dtype == torch.bfloat16 for l in leaves)
    if any_bf16:
        store = torch.finfo(torch.bfloat16)
        rounding = store.eps / 2.0 + torch.finfo(torch.float32).eps
    else:
        store = torch.finfo(torch.float32)
        rounding = store.eps  # multiply_dtype == storage for all-fp32 trees
    relative = 1.0 - rounding - 2.0 * roundoff
    absolute = store.smallest_normal * store.eps
    scale_v = ratio * relative - absolute
    scale_v = torch.minimum(torch.ones_like(scale_v), scale_v)
    scale_v = torch.where(torch.isfinite(scale_v), scale_v,
                          torch.zeros_like(scale_v))
    scale_v = torch.clamp(scale_v, min=0.0)
    scale_f32 = scale_v.to(torch.float32)

    # ---- pass 2: apply + sum per leaf (no stored materialization)
    reduced = []
    for leaf in leaves:
        x = leaf.contiguous()
        N = leaf.numel() // B
        T = (N + _BLOCK - 1) // _BLOCK
        out = torch.empty(leaf.shape[1:], dtype=leaf.dtype,
                          device=leaf.device)
        sm = (torch.finfo(torch.bfloat16).smallest_normal
              if leaf.dtype == torch.bfloat16 else 0.0)
        _apply_sum_kernel[(T,)](x, scale_f32, out, N, B, BLOCK=_BLOCK,
                                SM_NORMAL=float(sm), num_warps=4)
        reduced.append(out)

    markers = [leaf.new_zeros(()) for leaf in leaves]
    return (
        tree_unflatten(treedef, reduced),
        tree_unflatten(treedef, markers),
        (),
        (),
        (),
    )
