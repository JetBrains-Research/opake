# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
# Reference copy of the per-tensor fused clip kernels as packaged at c13e3007.
# Loaded by run_train_variant.py (CB_OLD=1) as the within-session baseline.
"""Fused Triton clip-and-reduce for the streaming fixed-clipping path.

Drop-in replacement for
:func:`opake.api.engine.clipping._streaming._stream_clip_and_sum`, selected by
``clip_backend="triton"`` or ``"auto"`` on :func:`clipped_fun` /
:func:`clipped_grad`. Per tree it runs two kernels per leaf instead of the
per-leaf chain of sanitize, square, reduce, scale, cast and sum operations, and
it never materializes sanitized or clipped copies of the batch.

Semantics match the streaming path:

* One per-example norm across the whole stacked tree. Each leaf is sanitized
  (NaN/Inf to zero), squared in float32 and summed in float64 per 2048-element
  tile, the same block structure ``_leaf_sq_sum`` uses, then accumulated
  across leaves in leaf order. The reduction's rounding is therefore within the
  bound ``_norm_roundoff`` assumes.
* The ratio ``clamp(C, 0) / norm`` and the guarded, clamped scale come from
  :func:`opake.api.engine.clipping._pytree._finalize_scale`, evaluated once per
  storage dtype, which is exactly production's per-leaf evaluation.
* Each element is scaled in float32, cast once to its storage dtype, and
  subnormal results rounded up by that cast are zeroed (bfloat16; a no-op for
  float32 storage). Contributions are summed over the batch in float32.
* ``clipped_norms`` diagnostics are computed from the stored values.

Results differ from the streaming path only in the rounding of the batch sum:
examples are added in a fixed order ``b = 0..B-1``, unlike ``torch.sum``'s
reduction order, and for float32 leaves the compiler may fuse scale and add
into one FMA. Per-example norms, scales and stored values are unchanged.

Supported calls: scalar clipping norm, fixed scaling (not AUTO-S), no second
moment, ``compute_dtype=None``, CUDA float32/bfloat16 leaves that do not
require grad, an output dtype equal to the storage dtype or float32, and no
active ``torch.compile`` trace. Other calls fall back to the streaming path, or
raise when ``strict``.
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl

from opake.api.engine.clipping import _pytree as cp
from opake.api.engine.clipping._streaming import _stream_clip_and_sum
from opake.api.engine.pytree import tree_flatten, tree_unflatten
from opake.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

_BLOCK = 2048
_SUPPORTED_DTYPES = (torch.float32, torch.bfloat16)
_MAX_ROW = 2**31 - _BLOCK  # int32 offsets inside one example's row
_MAX_BATCH = 65535  # examples run on CUDA grid axis 1


@triton.jit
def _partial_sq_kernel(X, PARTIAL, N, T, BLOCK: tl.constexpr):
    """``PARTIAL[b, t]``: float64 sum of squares of tile ``t`` of row ``b``."""
    t = tl.program_id(0)
    b = tl.program_id(1)
    offs = t * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + b.to(tl.int64) * N + offs, mask=mask, other=0.0)
    x = tl.where((x - x) == 0.0, x, 0.0)  # NaN/Inf -> 0
    xf = x.to(tl.float32)
    partial = tl.sum((xf * xf).to(tl.float64), axis=0)
    tl.store(PARTIAL + b.to(tl.int64) * T + t, partial)


@triton.jit
def _apply_sum_kernel(
    X,
    SCALE,
    OUT,
    POSTSQ,
    N,
    B,
    T,
    BLOCK: tl.constexpr,
    SM_NORMAL: tl.constexpr,
    WITH_POSTSQ: tl.constexpr,
):
    """Scale, store-round and sum one column tile over examples ``0..B-1``."""
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
            sf = tl.where((sa <= SM_NORMAL) & (sa > tl.abs(prod)), 0.0, sf)
        acc += sf
        if WITH_POSTSQ:
            tl.store(POSTSQ + b.to(tl.int64) * T + t, tl.sum(sf * sf, axis=0))
    tl.store(OUT + offs, acc.to(OUT.dtype.element_ty), mask=mask)


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def unsupported_reason(
    leaves,
    clipping_norm,
    scale,
    *,
    compute_dtype,
    second_moment: bool,
    batch_size: int,
) -> str | None:
    """Return why this call cannot use the fused path, or ``None``."""
    if torch.compiler.is_compiling():
        return "the call is being traced by torch.compile"
    if hasattr(clipping_norm, "groups"):
        return "per-group clipping norms are not supported"
    if scale.kind != "fixed":
        return "only fixed clipping is supported (not AUTO-S)"
    if second_moment:
        return "second_moment=True is not supported"
    if compute_dtype is not None:
        return "an explicit compute_dtype is not supported"
    if not leaves:
        return "the value tree has no leaves"
    if batch_size > _MAX_BATCH:
        return f"batch size {batch_size} exceeds {_MAX_BATCH}"
    grad_mode = torch.is_grad_enabled()
    for leaf in leaves:
        if not isinstance(leaf, torch.Tensor):
            return f"non-tensor leaf of type {type(leaf).__name__}"
        if not leaf.is_cuda:
            return f"leaf on device {leaf.device}, not CUDA"
        if leaf.dtype not in _SUPPORTED_DTYPES:
            return f"leaf dtype {leaf.dtype} (supported: float32, bfloat16)"
        if grad_mode and leaf.requires_grad:
            return "a leaf requires grad"
        if leaf.dim() == 0 or leaf.shape[0] != batch_size:
            return "a leaf's leading dimension is not the batch"
        row = leaf.numel() // batch_size
        if row == 0:
            return "a leaf is empty"
        if row > _MAX_ROW:
            return f"a leaf has {row} elements per example (limit {_MAX_ROW})"
    return None


def _output_dtypes(leaves, reduce_leaf):
    """Probe ``reduce_leaf``'s output dtype per storage dtype.

    The full-batch reducer returns the storage dtype; the microbatch chunk
    reducer returns float32 for bfloat16 (its accumulator stays float32). Both
    are exact for a float32 accumulation. Anything else is unsupported.
    """
    targets = {}
    for dtype in {leaf.dtype for leaf in leaves}:
        probe = reduce_leaf(torch.zeros((1,), dtype=dtype, device=leaves[0].device))
        if probe.dtype not in (dtype, torch.float32):
            return None
        targets[dtype] = probe.dtype
    return targets


def fused_clip_sum(leaves, clipping_norm, *, with_postsq: bool, out_dtypes=None):
    """Fused clip-and-sum of stacked leaves. Callers check support first.

    Returns ``(reduced, norm, post_sq, acc_dtype)``: per-leaf batch sums, the
    float64 per-example norms, per-example squared stored norms (or ``None``),
    and the reduction dtype used for diagnostics.
    """
    batch = leaves[0].shape[0]
    device = leaves[0].device
    acc_dtype = cp._resolve_compute_dtype_for_reduction(leaves, None)
    sq_dtype = cp._sq_accum_dtype(leaves)
    roundoff = cp._norm_roundoff(
        acc_dtype,
        sq_dtype,
        len(leaves),
        cp._reduction_terms([leaf[0] for leaf in leaves]),
    )

    sq_total = None
    for leaf in leaves:
        x = leaf.contiguous()
        n = x.numel() // batch
        tiles = _cdiv(n, _BLOCK)
        partial = torch.empty((batch, tiles), dtype=sq_dtype, device=device)
        _partial_sq_kernel[(tiles, batch)](x, partial, n, tiles, BLOCK=_BLOCK)
        sq = partial.sum(1)
        sq_total = sq if sq_total is None else sq_total + sq
    norm = torch.sqrt(sq_total)

    bound = torch.clamp(
        torch.as_tensor(clipping_norm, dtype=norm.dtype, device=norm.device),
        min=0.0,
    )
    ratio = bound / norm
    scales = {}
    for dtype in {leaf.dtype for leaf in leaves}:
        multiply_dtype = torch.promote_types(dtype, acc_dtype)
        if multiply_dtype != torch.float32:  # guarded by unsupported_reason
            raise AssertionError(
                *(f"kernel multiplies in float32, not {multiply_dtype}",)
            )
        scale = cp._finalize_scale(
            ratio, dtype, multiply_dtype, roundoff, clamp_to_one=True
        )
        scales[dtype] = scale.to(dtype=multiply_dtype).contiguous()

    post_sq = None
    reduced = []
    for leaf in leaves:
        x = leaf.contiguous()
        n = x.numel() // batch
        tiles = _cdiv(n, _BLOCK)
        out_dtype = (out_dtypes or {}).get(leaf.dtype, leaf.dtype)
        out = torch.empty(leaf.shape[1:], dtype=out_dtype, device=device)
        post = (
            torch.empty((batch, tiles), dtype=torch.float32, device=device)
            if with_postsq
            else scales[leaf.dtype]
        )
        smallest_normal = (
            float(torch.finfo(torch.bfloat16).smallest_normal)
            if leaf.dtype == torch.bfloat16
            else 0.0
        )
        _apply_sum_kernel[(tiles,)](
            x,
            scales[leaf.dtype],
            out,
            post,
            n,
            batch,
            tiles,
            BLOCK=_BLOCK,
            SM_NORMAL=smallest_normal,
            WITH_POSTSQ=with_postsq,
            num_warps=4,
        )
        if with_postsq:
            s2 = post.sum(1)
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
    strict: bool,
):
    """``_stream_clip_and_sum`` with the fused kernels where supported.

    Unsupported calls run the streaming path, or raise
    :class:`~opake.exceptions.ConfigurationError` when ``strict``.
    """
    leaves, treedef = tree_flatten(values)
    reason = unsupported_reason(
        leaves,
        clipping_norm,
        scale,
        compute_dtype=compute_dtype,
        second_moment=second_moment,
        batch_size=batch_size,
    )
    out_dtypes = None
    if reason is None:
        out_dtypes = _output_dtypes(leaves, reduce_leaf)
        if out_dtypes is None:
            reason = "the requested output dtype is not supported"
    if reason is not None:
        if strict:
            raise ConfigurationError(
                *(f"clip_backend='triton' cannot run this call: {reason}.",)
            )
        logger.debug("clip_backend='auto' using the torch path: %s", reason)
        return _stream_clip_and_sum(
            values,
            clipping_norm,
            scale=scale,
            batch_size=batch_size,
            reduce_leaf=reduce_leaf,
            compute_dtype=compute_dtype,
            second_moment=second_moment,
            return_aux=return_aux,
            return_stats=return_stats,
        )

    reduced, norm, post_sq, acc_dtype = fused_clip_sum(
        leaves, clipping_norm, with_postsq=return_aux, out_dtypes=out_dtypes
    )
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
