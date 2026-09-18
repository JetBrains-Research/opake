# Copyright (c) 2025 Opake Authors
# Copyright (c) 2024 LinkedIn Corporation (Liger Kernel)
# SPDX-License-Identifier: Apache-2.0 AND BSD-2-Clause
#
# Triton RMSNorm kernels derive from the Liger Kernel project (BSD-2-Clause,
# Copyright LinkedIn Corporation), which incorporated prior Unsloth Apache-2.0
# RMSNorm code. See:
# https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/rms_norm.py
# See ../../../../../NOTICE in this package for the full attribution.
"""RMSNorm Triton kernel with vmap(grad(...)) support (DP-SGD).

Implements Root Mean Square Layer Normalization from Zhang and Sennrich,
*Root Mean Square Layer Normalization* (https://arxiv.org/abs/1910.07467).
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from opake.exceptions import ConfigurationError, OperationError

from ._utils import (
    _MAX_PER_ROW_KERNEL_BLOCK_SIZE,
    _MIN_ROWS_FOR_BLOCK_KERNEL,
    calculate_settings,
    follow_autocast,
    torch_gpu_device,
    triton_cast,
)

_SAVED_TENSORS_WITH_WEIGHT = 3

try:
    _tv = tuple(int(p) for p in triton.__version__.split(".")[:3] if p.isdigit())
    if _tv >= (3, 0, 0):
        try:
            from triton.language.extra.libdevice import rsqrt
        except ModuleNotFoundError:
            from triton.language.extra.cuda.libdevice import rsqrt
    else:
        raise ImportError
except (ImportError, ValueError):
    rsqrt = tl.math.rsqrt

_STR_TO_CASTING = {
    "llama": 0,
    "gemma": 1,
    "none": -1,
}

_TORCH_TO_TRITON_DTYPES = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


def _casting_mode_int(casting_mode: str | int) -> int:
    if isinstance(casting_mode, int):
        if casting_mode not in _STR_TO_CASTING.values():
            raise ConfigurationError(*(f"Invalid casting_mode int: {casting_mode}",))
        return casting_mode
    if casting_mode not in _STR_TO_CASTING:
        raise ConfigurationError(*(f"Invalid casting_mode: {casting_mode}",))
    return _STR_TO_CASTING[casting_mode]


@triton.jit
def _rms_norm_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    W_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    n_cols,
    eps,
    offset,
    casting_mode: tl.constexpr,
    elementwise_affine: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # int64 stride math: ``row_idx * stride`` overflows int32 past 2^31 elements
    # at large microbatch (CUDA illegal access). Same pattern as ``cross_entropy.py``.
    y_base = Y_ptr + row_idx * triton_cast(Y_row_stride, tl.int64)
    x_base = X_ptr + row_idx * triton_cast(X_row_stride, tl.int64)
    rstd_base = RSTD_ptr + row_idx * triton_cast(RSTD_row_stride, tl.int64)

    X_row = tl.load(x_base + col_offsets, mask=mask, other=0)
    X_row_dtype = X_row.dtype
    if elementwise_affine:
        W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)

    if casting_mode == 0:
        X_row = X_row.to(tl.float32)

    if casting_mode == 1:
        if elementwise_affine:
            W_row = W_row.to(tl.float32)
        X_row = X_row.to(tl.float32)

    if casting_mode == -1:
        eps = eps.to(X_row_dtype)
        offset = offset.to(X_row_dtype)

    mean_square = tl.sum(X_row * X_row, axis=0) / n_cols
    row_rstd = rsqrt(mean_square + eps)
    tl.store(rstd_base, row_rstd)

    X_row = X_row * row_rstd

    if casting_mode == 0:
        X_row = X_row.to(X_row_dtype)

    Y_row = X_row * (offset + W_row) if elementwise_affine else X_row

    if casting_mode == 1:
        Y_row = Y_row.to(X_row_dtype)

    tl.store(y_base + col_offsets, Y_row, mask=mask)


@triton.jit
def _rms_norm_forward_block_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    W_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    n_rows,
    n_cols,
    eps,
    offset,
    rows_per_program,
    casting_mode: tl.constexpr,
    elementwise_affine: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Block variant of the forward: one program per SM, each looping
    ``rows_per_program`` rows — mirrors :func:`_rms_norm_backward_kernel`.

    Used for the small-hidden-dim, many-rows regime (small ``BLOCK_SIZE`` with
    large ``n_rows``), where the per-row kernel would launch one tiny program per
    row. The per-row body is identical to :func:`_rms_norm_forward_kernel`; only
    the grid/loop differs. The shared weight is loaded once (loop-invariant)."""
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    row_end = tl.minimum((row_block_id + 1) * rows_per_program, n_rows)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    if elementwise_affine:
        W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)
        if casting_mode == 1:
            W_row = W_row.to(tl.float32)

    # int64 stride math — same int32-overflow guard as the per-row kernel.
    y_stride64 = triton_cast(Y_row_stride, tl.int64)
    x_stride64 = triton_cast(X_row_stride, tl.int64)
    rstd_stride64 = triton_cast(RSTD_row_stride, tl.int64)

    for row_idx in range(row_start, row_end):
        y_base = Y_ptr + row_idx * y_stride64
        x_base = X_ptr + row_idx * x_stride64
        rstd_base = RSTD_ptr + row_idx * rstd_stride64

        X_row = tl.load(x_base + col_offsets, mask=mask, other=0)
        X_row_dtype = X_row.dtype

        if casting_mode == 0:
            X_row = X_row.to(tl.float32)
        if casting_mode == 1:
            X_row = X_row.to(tl.float32)
        if casting_mode == -1:
            eps_r = eps.to(X_row_dtype)
            offset_r = offset.to(X_row_dtype)
        else:
            eps_r = eps
            offset_r = offset

        mean_square = tl.sum(X_row * X_row, axis=0) / n_cols
        row_rstd = rsqrt(mean_square + eps_r)
        tl.store(rstd_base, row_rstd)

        X_row = X_row * row_rstd

        if casting_mode == 0:
            X_row = X_row.to(X_row_dtype)

        Y_row = X_row * (offset_r + W_row) if elementwise_affine else X_row

        if casting_mode == 1:
            Y_row = Y_row.to(X_row_dtype)

        tl.store(y_base + col_offsets, Y_row, mask=mask)


@triton.jit
def _rms_norm_backward_kernel(
    dY_ptr,
    dY_row_stride,
    dX_ptr,
    dX_row_stride,
    X_ptr,
    X_row_stride,
    X_dtype: tl.constexpr,
    W_ptr,
    W_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    dW_ptr,
    dW_row_stride,
    n_rows,
    n_cols,
    offset,
    rows_per_program,
    casting_mode: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    COMPUTE_DW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    row_end = tl.minimum((row_block_id + 1) * rows_per_program, n_rows)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    if COMPUTE_DW:
        dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    if HAS_WEIGHT:
        W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
        W_row = W_row + offset

    # int64 stride math — same int32-overflow guard as the forward kernel.
    dy_stride64 = triton_cast(dY_row_stride, tl.int64)
    dx_stride64 = triton_cast(dX_row_stride, tl.int64)
    x_stride64 = triton_cast(X_row_stride, tl.int64)
    rstd_stride64 = triton_cast(RSTD_row_stride, tl.int64)

    for row_idx in range(row_start, row_end):
        dy_base = dY_ptr + row_idx * dy_stride64
        dx_base = dX_ptr + row_idx * dx_stride64
        x_base = X_ptr + row_idx * x_stride64
        rstd_base = RSTD_ptr + row_idx * rstd_stride64

        dY_row = tl.load(dy_base + col_offsets, mask=mask, other=0.0)
        X_row = tl.load(x_base + col_offsets, mask=mask, other=0.0)
        rstd_row = tl.load(rstd_base)

        X_row = X_row.to(tl.float32)

        if casting_mode == 0:
            m = (dY_row * W_row).to(tl.float32) if HAS_WEIGHT else dY_row.to(tl.float32)
        elif casting_mode == 1:
            dY_row = dY_row.to(tl.float32)
            m = dY_row * W_row if HAS_WEIGHT else dY_row
        else:
            m = dY_row * W_row if HAS_WEIGHT else dY_row

        dX_row = rstd_row * m
        dX_row += rstd_row * (
            -(1 / n_cols) * rstd_row * rstd_row * tl.sum(m * X_row, axis=0) * X_row
        )

        if COMPUTE_DW:
            if casting_mode == 0:
                dW_row += dY_row * (X_row * rstd_row).to(X_dtype)
            else:
                dW_row += dY_row * (X_row * rstd_row)

        tl.store(dx_base + col_offsets, dX_row.to(X_dtype), mask=mask)

    if COMPUTE_DW:
        tl.store(dW_ptr + row_block_id * dW_row_stride + col_offsets, dW_row, mask=mask)


@triton.jit
def _rms_norm_weight_grad_partial_kernel(
    dY_ptr,
    dY_batch_stride,
    dY_row_stride,
    X_ptr,
    X_batch_stride,
    X_row_stride,
    X_dtype: tl.constexpr,
    RSTD_ptr,
    RSTD_batch_stride,
    RSTD_row_stride,
    partial_ptr,
    partial_batch_stride,
    partial_chunk_stride,
    n_rows_per_batch,
    n_cols,
    rows_per_program,
    casting_mode: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    row_start = chunk_idx * rows_per_program
    row_end = tl.minimum(row_start + rows_per_program, n_rows_per_batch)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    dY_batch = dY_ptr + batch_idx * triton_cast(dY_batch_stride, tl.int64)
    X_batch = X_ptr + batch_idx * triton_cast(X_batch_stride, tl.int64)
    RSTD_batch = RSTD_ptr + batch_idx * triton_cast(RSTD_batch_stride, tl.int64)
    dy_stride64 = triton_cast(dY_row_stride, tl.int64)
    x_stride64 = triton_cast(X_row_stride, tl.int64)
    rstd_stride64 = triton_cast(RSTD_row_stride, tl.int64)

    for row_idx in range(row_start, row_end):
        dY_row = tl.load(
            dY_batch + row_idx * dy_stride64 + col_offsets, mask=mask, other=0.0
        )
        X_row = tl.load(
            X_batch + row_idx * x_stride64 + col_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        rstd_row = tl.load(RSTD_batch + row_idx * rstd_stride64)

        if casting_mode == 0:
            dW_row += dY_row * (X_row * rstd_row).to(X_dtype)
        else:
            dW_row += dY_row.to(tl.float32) * (X_row * rstd_row)

    partial_base = (
        partial_ptr
        + batch_idx * partial_batch_stride
        + chunk_idx * partial_chunk_stride
    )
    tl.store(partial_base + col_offsets, dW_row, mask=mask)


@triton.jit
def _rms_norm_weight_grad_reduce_kernel(
    partial_ptr,
    partial_batch_stride,
    partial_chunk_stride,
    dW_ptr,
    dW_batch_stride,
    n_chunks,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    partial_batch = partial_ptr + batch_idx * partial_batch_stride

    for chunk_idx in range(n_chunks):
        dW_row += tl.load(
            partial_batch + chunk_idx * partial_chunk_stride + col_offsets,
            mask=mask,
            other=0.0,
        )

    tl.store(dW_ptr + batch_idx * dW_batch_stride + col_offsets, dW_row, mask=mask)


def _rms_norm_weight_grad_triton(
    dY: torch.Tensor,
    X: torch.Tensor,
    RSTD: torch.Tensor,
    W: torch.Tensor,
    vmap_batch_size: int,
    casting_mode: int,
    BLOCK_SIZE: int,
    num_warps: int,
) -> torch.Tensor:
    dim = dY.shape[-1]
    dY_3d = dY.contiguous().view(vmap_batch_size, -1, dim)
    X_3d = X.contiguous().view(vmap_batch_size, -1, dim)
    RSTD_2d = RSTD.contiguous().view(vmap_batch_size, -1)
    n_rows_per_batch = dY_3d.shape[1]
    sm_count = torch.cuda.get_device_properties(X.device).multi_processor_count
    n_chunks = min(n_rows_per_batch, max(1, math.ceil(sm_count / vmap_batch_size)))
    rows_per_program = math.ceil(n_rows_per_batch / n_chunks)
    partial = torch.empty(
        (vmap_batch_size, n_chunks, dim), dtype=torch.float32, device=W.device
    )
    dW = torch.empty((vmap_batch_size, dim), dtype=W.dtype, device=W.device)
    x_dtype_triton = _TORCH_TO_TRITON_DTYPES[X.dtype]

    with torch_gpu_device(X.device):
        _rms_norm_weight_grad_partial_kernel[(vmap_batch_size, n_chunks)](
            dY_3d,
            dY_3d.stride(0),
            dY_3d.stride(1),
            X_3d,
            X_3d.stride(0),
            X_3d.stride(1),
            x_dtype_triton,
            RSTD_2d,
            RSTD_2d.stride(0),
            RSTD_2d.stride(1),
            partial,
            partial.stride(0),
            partial.stride(1),
            n_rows_per_batch,
            dim,
            rows_per_program,
            casting_mode,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        _rms_norm_weight_grad_reduce_kernel[(vmap_batch_size,)](
            partial,
            partial.stride(0),
            partial.stride(1),
            dW,
            dW.stride(0),
            n_chunks,
            dim,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    return dW


def _rms_norm_forward_triton(
    X: torch.Tensor,
    W: torch.Tensor | None,
    eps: float,
    offset: float,
    casting_mode: int,
    row_mode: bool | None,
):
    """Returns (Y, X_2d, RSTD, BLOCK_SIZE, num_warps)."""
    shape = X.shape
    dim = shape[-1]
    X = X.contiguous().view(-1, dim)
    n_rows, n_cols = X.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    rstd_dtype = torch.float32 if casting_mode in (0, 1) else X.dtype
    RSTD = torch.empty(n_rows, dtype=rstd_dtype, device=X.device)

    elementwise_affine = W is not None
    W_contig = W.contiguous() if elementwise_affine else None

    def grid(meta):
        return (n_rows,)

    # Small hidden dim (``BLOCK_SIZE <= 256``) with many rows (``n_rows >= 32k``)
    # makes the per-row grid ``(n_rows,)`` launch one tiny program per row; the
    # block kernel runs one program per SM, each looping ``rows_per_program``
    # rows. Same math; ``row_mode`` forces the per-row path.
    use_block = not (
        BLOCK_SIZE > _MAX_PER_ROW_KERNEL_BLOCK_SIZE
        or n_rows < _MIN_ROWS_FOR_BLOCK_KERNEL
        or row_mode
    )

    with torch_gpu_device(X.device):
        if use_block:
            sm_count = (
                torch.cuda.get_device_properties(X.device).multi_processor_count
                if X.device.type == "cuda"
                else 1
            )
            rows_per_program = math.ceil(n_rows / sm_count)
            _rms_norm_forward_block_kernel[(sm_count,)](
                Y,
                Y.stride(0),
                X,
                X.stride(0),
                W_contig,
                W_contig.stride(0) if elementwise_affine else 0,
                RSTD,
                RSTD.stride(0),
                n_rows,
                n_cols,
                eps,
                offset,
                rows_per_program,
                casting_mode,
                elementwise_affine=elementwise_affine,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )
        else:
            _rms_norm_forward_kernel[grid](
                Y,
                Y.stride(0),
                X,
                X.stride(0),
                W_contig,
                W_contig.stride(0) if elementwise_affine else 0,
                RSTD,
                RSTD.stride(0),
                n_cols,
                eps,
                offset,
                casting_mode,
                elementwise_affine=elementwise_affine,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )

    return Y.view(*shape), X, RSTD, BLOCK_SIZE, num_warps


def _rms_norm_backward_triton(
    dY: torch.Tensor,
    X: torch.Tensor,
    W: torch.Tensor | None,
    RSTD: torch.Tensor,
    offset: float,
    casting_mode: int,
    BLOCK_SIZE: int,
    num_warps: int,
    in_place: bool,
    compute_dw: bool,
):
    shape = dY.shape
    dim = shape[-1]
    dY = dY.contiguous().view(-1, dim)
    n_rows, n_cols = dY.shape

    if n_cols > BLOCK_SIZE:
        raise OperationError(
            *(f"RMSNorm hidden dim {n_cols} exceeds fused block limit {BLOCK_SIZE}.",)
        )

    has_weight = W is not None
    if compute_dw and not has_weight:
        raise OperationError(*("Cannot compute an RMSNorm weight gradient without W.",))
    if X.device.type == "cuda":
        sm_count = torch.cuda.get_device_properties(X.device).multi_processor_count
    else:
        sm_count = 1

    _dW = (
        torch.empty((sm_count, n_cols), dtype=torch.float32, device=W.device)
        if compute_dw
        else None
    )
    rows_per_program = math.ceil(n_rows / sm_count)
    grid = (sm_count,)

    dX = dY if in_place else torch.zeros_like(dY)
    W_contig = W.contiguous() if has_weight else None
    dW_ptr = _dW if _dW is not None else dY
    x_dtype_triton = _TORCH_TO_TRITON_DTYPES[X.dtype]

    with torch_gpu_device(X.device):
        _rms_norm_backward_kernel[grid](
            dY,
            dY.stride(0),
            dX,
            dX.stride(0),
            X,
            X.stride(0),
            x_dtype_triton,
            W_contig,
            W_contig.stride(0) if has_weight else 0,
            RSTD,
            RSTD.stride(0),
            dW_ptr,
            _dW.stride(0) if _dW is not None else 0,
            n_rows,
            n_cols,
            offset,
            rows_per_program,
            casting_mode,
            HAS_WEIGHT=has_weight,
            COMPUTE_DW=compute_dw,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

    dX = dX.view(*shape)
    dW = _dW.sum(dim=0).to(W.dtype) if _dW is not None else None
    return dX, dW


def _empty_weight(device, dtype) -> torch.Tensor:
    """Placeholder when elementwise_affine is False (autograd needs a tensor)."""
    return torch.empty(0, device=device, dtype=dtype)


class _RMSNormBackward(torch.autograd.Function):
    """Backward as separate Function for vmap(grad(...))."""

    @staticmethod
    def forward(
        dY,
        X,
        W,
        RSTD,
        offset,
        casting_mode,
        BLOCK_SIZE,
        num_warps,
        in_place,
        has_weight,
        compute_dw,
    ):
        W_real = W if has_weight else None
        dX, dW = _rms_norm_backward_triton(
            dY,
            X,
            W_real,
            RSTD,
            offset,
            casting_mode,
            BLOCK_SIZE,
            num_warps,
            in_place,
            compute_dw,
        )
        return dX, dW if dW is not None else W.new_empty(0)

    @staticmethod
    def setup_context(ctx, inputs, output):
        del ctx

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Double backward not supported for RMSNorm")

    @staticmethod
    def vmap(
        info,
        in_dims,
        dY,
        X,
        W,
        RSTD,
        offset,
        casting_mode,
        BLOCK_SIZE,
        num_warps,
        in_place,
        has_weight,
        compute_dw,
    ):
        del info
        dy_b, x_b, w_b, r_b, *static_dims = in_dims
        if any(dim is not None for dim in static_dims):
            raise ConfigurationError(*("RMSNorm metadata must not be vmapped",))
        if w_b is not None:
            raise ConfigurationError(*("W must not be vmapped",))
        if dy_b != 0 or x_b != 0 or r_b != 0:
            raise ConfigurationError(*("dY, X, RSTD must be vmapped at dim 0",))

        H = X.shape[-1]
        head = dY.shape[:-1]
        B = dY.shape[0]
        dY_m = dY.reshape(-1, H)
        X_m = X.reshape(-1, H)
        R_m = RSTD.reshape(-1)
        W_use = W if has_weight else None
        dW_out = None
        if compute_dw:
            dW_out = _rms_norm_weight_grad_triton(
                dY,
                X,
                RSTD,
                W,
                B,
                casting_mode,
                BLOCK_SIZE,
                num_warps,
            )

        dX, _ = _rms_norm_backward_triton(
            dY_m,
            X_m,
            W_use,
            R_m,
            offset,
            casting_mode,
            BLOCK_SIZE,
            num_warps,
            in_place,
            False,
        )
        dX_out = dX.view(*head, H)
        if dW_out is not None:
            return (dX_out, dW_out), (dy_b, 0)
        return (dX_out, W.new_empty(0)), (dy_b, None)


class Opake_RMSNorm(torch.autograd.Function):
    """RMSNorm with Llama / Gemma casting modes (HuggingFace-aligned)."""

    @staticmethod
    def forward(X, W, eps, offset, casting_mode, in_place, row_mode):
        cm = _casting_mode_int(casting_mode)
        orig_shape = X.shape
        Y, X2d, RSTD, _, _ = _rms_norm_forward_triton(
            X, W, float(eps), float(offset), cm, row_mode
        )
        return (
            Y.view(orig_shape),
            RSTD.view(orig_shape[:-1]),
            X2d.view(orig_shape),
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        _X, W, _eps, offset, casting_mode, in_place, _row_mode = inputs
        _, RSTD, X_saved = output
        cm = _casting_mode_int(casting_mode)
        dim = X_saved.shape[-1]

        ctx.mark_non_differentiable(RSTD, X_saved)
        ctx.original_shape = X_saved.shape
        ctx.offset = float(offset)
        ctx.casting_mode = cm
        ctx.block_size, ctx.num_warps = calculate_settings(dim)
        ctx.in_place = bool(in_place)
        ctx.has_weight = W is not None
        ctx.compute_dw = W is not None and W.requires_grad

        if W is not None:
            ctx.save_for_backward(
                X_saved.reshape(-1, dim), W.contiguous(), RSTD.reshape(-1)
            )
        else:
            ctx.save_for_backward(X_saved.reshape(-1, dim), RSTD.reshape(-1))

    @staticmethod
    def backward(ctx, grad_output, _grad_rstd, _grad_x_saved):
        go = grad_output.contiguous()
        dim = go.shape[-1]
        go2 = go.reshape(-1, dim)
        saved = ctx.saved_tensors
        if len(saved) == _SAVED_TENSORS_WITH_WEIGHT:
            X_s, W, RSTD = saved
            W_arg = W
        else:
            X_s, RSTD = saved
            W_arg = _empty_weight(X_s.device, X_s.dtype)

        dX, dW = _RMSNormBackward.apply(
            go2,
            X_s,
            W_arg,
            RSTD,
            ctx.offset,
            ctx.casting_mode,
            ctx.block_size,
            ctx.num_warps,
            ctx.in_place,
            ctx.has_weight,
            ctx.compute_dw,
        )
        dX = dX.view(ctx.original_shape)
        return (
            dX,
            dW if ctx.compute_dw else None,
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def vmap(info, in_dims, X, W, eps, offset, casting_mode, in_place, row_mode):
        del info
        x_b, w_b = in_dims[0], in_dims[1]
        if w_b is not None:
            raise ConfigurationError(
                *("Opake_RMSNorm vmap: weight must not be batched",)
            )
        if x_b != 0:
            raise ConfigurationError(
                *("Opake_RMSNorm vmap: X must be vmapped at dim 0",)
            )
        cm = _casting_mode_int(casting_mode)
        shape = X.shape
        H = shape[-1]
        Xf = X.reshape(-1, H).contiguous()
        Yf, X2d, RSTD, _, _ = _rms_norm_forward_triton(
            Xf,
            W,
            float(eps),
            float(offset),
            cm,
            row_mode,
        )
        return (
            Yf.view(shape),
            RSTD.view(shape[:-1]),
            X2d.view(shape),
        ), (0, 0, 0)


def opake_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    offset: float = 0.0,
    casting_mode: str = "llama",
    *,
    in_place_backward: bool = False,
    row_mode: bool | None = None,
) -> torch.Tensor:
    """Public API: fused RMSNorm (CUDA only when Triton path is used)."""
    if not x.is_cuda:
        raise OperationError(*("opake_rms_norm Triton path requires CUDA",))
    x, weight = follow_autocast(x, weight)
    normalized, _, _ = Opake_RMSNorm.apply(
        x,
        weight,
        eps,
        offset,
        casting_mode,
        in_place_backward,
        row_mode,
    )
    return normalized
