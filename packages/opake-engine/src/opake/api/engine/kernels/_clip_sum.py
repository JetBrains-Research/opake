# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Fused Triton clip-and-reduce for the streaming fixed-clipping path.

Drop-in replacement for
:func:`opake.api.engine.clipping._streaming._stream_clip_and_sum`, selected by
``clip_backend="triton"`` or ``"auto"`` on :func:`clipped_fun` /
:func:`clipped_grad`. It never materializes sanitized or clipped copies of the
batch, and it handles all tensors of one storage dtype in one launch per pass
("multi-tensor apply"). Each program looks up its tensor in a cached tile table
and reads that tensor's address from a per-call pointer table. A call therefore
costs a few dozen CUDA launches, independent of the number of tensors.

Semantics match the streaming path:

* One per-example norm across the whole stacked tree. Entries are sanitized
  (NaN/Inf to zero), squared in float32 and summed in float64 per
  2048-element tile, the same block structure ``_leaf_sq_sum`` uses. Tile sums
  are then added per tensor, and per-tensor sums across tensors. The
  reduction's rounding is therefore within the bound ``_norm_roundoff``
  assumes.
* The ratio ``clamp(C, 0) / norm`` and the guarded, clamped scale come from
  :func:`opake.api.engine.clipping._pytree._finalize_scale`, evaluated once per
  storage dtype, which is exactly production's per-leaf evaluation.
* Each element is scaled in float32, cast once to its storage dtype, and
  subnormal results rounded up by that cast are zeroed (bfloat16; a no-op for
  float32 storage). Contributions are summed over the batch in float32.
* ``clipped_norms`` diagnostics are computed from the stored values.

Results differ from the streaming path only in the rounding of the batch sum:
examples are added in a fixed order ``b = 0..B-1``, unlike ``torch.sum``'s
reduction order, and for float32 tensors the compiler may fuse scale and add
into one FMA. Per-example norms, scales and stored values are unchanged.

The reduced tensors of one storage dtype are views into one flat buffer.

Supported calls: scalar clipping norm, fixed scaling (not AUTO-S), no second
moment, ``compute_dtype=None``, float32/bfloat16 tensors on a single CUDA device
that do not require grad, an output dtype equal to the storage dtype or float32,
and no active ``torch.compile`` trace. Other calls fall back to the streaming
path, or raise when ``strict``.
"""

from __future__ import annotations

import logging
from collections import OrderedDict

import torch
import triton
import triton.language as tl

from opake.api.engine.clipping import _pytree as cp
from opake.api.engine.clipping._streaming import _stream_clip_and_sum
from opake.api.engine.pytree import tree_flatten, tree_unflatten
from opake.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

_BLOCK = 2048  # elements per tile; the block size _leaf_sq_sum reduces over
_TILE_SUM_BLOCK = 1024  # tile partials read per step when summing a tensor
_SUPPORTED_DTYPES = (torch.float32, torch.bfloat16)
_MAX_ROW = 2**31 - _BLOCK  # elements per example in one tensor
_MAX_BATCH = 65535  # examples run on CUDA grid axis 1
_MAX_PLANS = 64  # cached tile tables, keyed by device and per-example shapes


@triton.jit
def _partial_sq_kernel(
    X0, PTRS, SIZES, TILE_LEAF, TILE_LOCAL, PARTIAL, T, BLOCK: tl.constexpr
):
    """``PARTIAL[b, t]``: float64 sum of squares of tile ``t`` of example ``b``.

    ``X0`` supplies the element type only; tensor addresses come from ``PTRS``.
    """
    t = tl.program_id(0)
    b = tl.program_id(1)
    leaf = tl.load(TILE_LEAF + t)
    n = tl.load(SIZES + leaf)
    base = tl.load(PTRS + leaf).to(tl.pointer_type(X0.dtype.element_ty))
    offs = tl.load(TILE_LOCAL + t).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(base + b.to(tl.int64) * n + offs, mask=mask, other=0.0)
    x = tl.where((x - x) == 0.0, x, 0.0)  # NaN/Inf -> 0
    xf = x.to(tl.float32)
    partial = tl.sum((xf * xf).to(tl.float64), axis=0)
    tl.store(PARTIAL + b.to(tl.int64) * T + t, partial)


@triton.jit
def _tensor_sum_kernel(
    PARTIAL, TILE_START, LEAF_COL, LEAF_SQ, T, N_LEAVES, TB: tl.constexpr
):
    """``LEAF_SQ[b, col]``: sum of one tensor's tile partials for example ``b``."""
    leaf = tl.program_id(0)
    b = tl.program_id(1)
    t0 = tl.load(TILE_START + leaf)
    t1 = tl.load(TILE_START + leaf + 1)
    acc = tl.zeros((TB,), dtype=tl.float64)
    for start in range(t0, t1, TB):
        idx = start + tl.arange(0, TB)
        acc += tl.load(PARTIAL + b.to(tl.int64) * T + idx, mask=idx < t1, other=0.0)
    col = tl.load(LEAF_COL + leaf)
    tl.store(LEAF_SQ + b.to(tl.int64) * N_LEAVES + col, tl.sum(acc, axis=0))


@triton.jit
def _apply_sum_kernel(
    X0,
    PTRS,
    SIZES,
    TILE_LEAF,
    TILE_LOCAL,
    SCALE,
    OUT,
    OUT_OFFS,
    POSTSQ,
    B,
    T,
    BLOCK: tl.constexpr,
    SM_NORMAL: tl.constexpr,
    WITH_POSTSQ: tl.constexpr,
):
    """Scale, store-round and sum one tile over examples ``0..B-1``."""
    t = tl.program_id(0)
    leaf = tl.load(TILE_LEAF + t)
    n = tl.load(SIZES + leaf)
    base = tl.load(PTRS + leaf).to(tl.pointer_type(X0.dtype.element_ty))
    out = OUT + tl.load(OUT_OFFS + leaf)
    offs = tl.load(TILE_LOCAL + t).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
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
    tl.store(out + offs, acc.to(OUT.dtype.element_ty), mask=mask)


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
        if leaf.device != leaves[0].device:
            return "the leaves span more than one device"
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


_PLANS: OrderedDict = OrderedDict()


def _plan(leaves, device):
    """Tile tables per storage dtype, cached by device and per-example shapes."""
    key = (device, tuple((leaf.dtype, tuple(leaf.shape[1:])) for leaf in leaves))
    plan = _PLANS.get(key)
    if plan is not None:
        _PLANS.move_to_end(key)
        return plan
    groups = []
    for dtype in sorted({leaf.dtype for leaf in leaves}, key=str):
        idx = [i for i, leaf in enumerate(leaves) if leaf.dtype == dtype]
        sizes = [leaves[i][0].numel() for i in idx]
        offsets = [0]
        for n in sizes[:-1]:
            offsets.append(offsets[-1] + n)
        on_device = lambda v, dt: torch.tensor(v, dtype=dt, device=device)  # noqa: E731
        tiles = on_device([_cdiv(n, _BLOCK) for n in sizes], torch.int64)
        tile_start = torch.cat([tiles.new_zeros(1), torch.cumsum(tiles, 0)])
        n_tiles = int(tile_start[-1])
        leaf_ids = torch.arange(len(idx), device=device)
        tile_leaf = torch.repeat_interleave(leaf_ids, tiles)
        tile_local = torch.arange(n_tiles, device=device) - tile_start[tile_leaf]
        groups.append(
            {
                "dtype": dtype,
                "idx": idx,
                "sizes_host": sizes,
                "offsets": offsets,
                "total": sum(sizes),
                "tiles": n_tiles,
                "sizes": on_device(sizes, torch.int64),
                "out_offs": on_device(offsets, torch.int64),
                "leaf_col": on_device(idx, torch.int32),
                "tile_start": tile_start,
                "tile_leaf": tile_leaf.to(torch.int32),
                "tile_local": tile_local.to(torch.int32),
            }
        )
    plan = {"groups": groups, "n_leaves": len(leaves)}
    _PLANS[key] = plan
    if len(_PLANS) > _MAX_PLANS:
        _PLANS.popitem(last=False)
    return plan


def fused_clip_sum(leaves, clipping_norm, *, with_postsq: bool, out_dtypes=None):
    """Fused clip-and-sum of stacked leaves. Callers check support first.

    Returns ``(reduced, norm, post_sq, acc_dtype)``: per-leaf batch sums, the
    float64 per-example norms, per-example squared stored norms (or ``None``),
    and the reduction dtype used for diagnostics.
    """
    batch = leaves[0].shape[0]
    device = leaves[0].device
    # Pointer tables address each tensor as a dense (batch, n) row block; the
    # list keeps any contiguous copies alive until the kernels are enqueued.
    leaves = [leaf if leaf.is_contiguous() else leaf.contiguous() for leaf in leaves]
    acc_dtype = cp._resolve_compute_dtype_for_reduction(leaves, None)
    sq_dtype = cp._sq_accum_dtype(leaves)
    roundoff = cp._norm_roundoff(
        acc_dtype,
        sq_dtype,
        len(leaves),
        cp._reduction_terms([leaf[0] for leaf in leaves]),
    )
    plan = _plan(leaves, device)
    n_leaves = plan["n_leaves"]

    with torch.cuda.device(device):
        leaf_sq = torch.empty((batch, n_leaves), dtype=sq_dtype, device=device)
        ptrs = {}
        for g in plan["groups"]:
            first = leaves[g["idx"][0]]
            ptrs[g["dtype"]] = torch.tensor(
                [leaves[i].data_ptr() for i in g["idx"]],
                dtype=torch.int64,
                device=device,
            )
            partial = torch.empty((batch, g["tiles"]), dtype=sq_dtype, device=device)
            _partial_sq_kernel[(g["tiles"], batch)](
                first,
                ptrs[g["dtype"]],
                g["sizes"],
                g["tile_leaf"],
                g["tile_local"],
                partial,
                g["tiles"],
                BLOCK=_BLOCK,
            )
            _tensor_sum_kernel[(len(g["idx"]), batch)](
                partial,
                g["tile_start"],
                g["leaf_col"],
                leaf_sq,
                g["tiles"],
                n_leaves,
                TB=_TILE_SUM_BLOCK,
            )
        norm = torch.sqrt(leaf_sq.sum(1))

        bound = torch.clamp(
            torch.as_tensor(clipping_norm, dtype=norm.dtype, device=device),
            min=0.0,
        )
        ratio = bound / norm
        reduced = [None] * n_leaves
        post_sq = None
        for g in plan["groups"]:
            dtype = g["dtype"]
            multiply_dtype = torch.promote_types(dtype, acc_dtype)
            if multiply_dtype != torch.float32:  # guarded by unsupported_reason
                raise AssertionError(
                    *(f"kernel multiplies in float32, not {multiply_dtype}",)
                )
            scale = cp._finalize_scale(
                ratio, dtype, multiply_dtype, roundoff, clamp_to_one=True
            )
            scale = scale.to(dtype=multiply_dtype).contiguous()
            out_dtype = (out_dtypes or {}).get(dtype, dtype)
            flat = torch.empty((g["total"],), dtype=out_dtype, device=device)
            post = (
                torch.empty((batch, g["tiles"]), dtype=torch.float32, device=device)
                if with_postsq
                else scale
            )
            smallest_normal = (
                float(torch.finfo(torch.bfloat16).smallest_normal)
                if dtype == torch.bfloat16
                else 0.0
            )
            _apply_sum_kernel[(g["tiles"],)](
                leaves[g["idx"][0]],
                ptrs[dtype],
                g["sizes"],
                g["tile_leaf"],
                g["tile_local"],
                scale,
                flat,
                g["out_offs"],
                post,
                batch,
                g["tiles"],
                BLOCK=_BLOCK,
                SM_NORMAL=smallest_normal,
                WITH_POSTSQ=with_postsq,
                num_warps=4,
            )
            for i, offset, n in zip(
                g["idx"], g["offsets"], g["sizes_host"], strict=True
            ):
                reduced[i] = flat[offset : offset + n].view(leaves[i].shape[1:])
            if with_postsq:
                s2 = post.sum(1)
                post_sq = s2 if post_sq is None else post_sq + s2
    return reduced, norm, post_sq, acc_dtype


_MARKERS: dict = {}


def _dtype_marker(leaf):
    """A shared 0-dim zero per (dtype, device).

    The accumulator only reads the markers' dtypes. Allocating one zero per
    leaf would cost one fill kernel per tensor per call.
    """
    key = (leaf.dtype, leaf.device)
    marker = _MARKERS.get(key)
    if marker is None:
        marker = _MARKERS[key] = leaf.new_zeros(())
    return marker


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
    markers = [_dtype_marker(leaf) for leaf in leaves]
    return (
        tree_unflatten(treedef, reduced),
        tree_unflatten(treedef, markers),
        (),
        (),
        diagnostics if (return_aux or return_stats) else (),
    )
