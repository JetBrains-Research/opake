# Copyright (c) 2025 Opake Authors
# Copyright (c) 2024 LinkedIn Corporation (Liger Kernel)
# SPDX-License-Identifier: Apache-2.0 AND BSD-2-Clause
#
# Fused residual add + RMSNorm Triton kernels derive from the Liger Kernel
# project (BSD-2-Clause, Copyright LinkedIn Corporation). See:
# https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_add_rms_norm.py
# See ../../../../../NOTICE in this package for the full attribution.
"""Fused (hidden + residual) + RMSNorm with vmap(grad(...)) support (DP-SGD)."""

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
from .rms_norm import _rms_norm_weight_grad_triton

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
def _fused_add_rms_norm_forward_kernel(
    Y_ptr,
    Y_row_stride,
    S_ptr,
    S_row_stride,
    X_ptr,
    X_row_stride,
    R_ptr,
    R_row_stride,
    W_ptr,
    W_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    n_cols,
    eps,
    offset,
    casting_mode: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # int64 stride math: ``row_idx * stride`` overflows int32 at large microbatch.
    Y_ptr += row_idx * triton_cast(Y_row_stride, tl.int64)
    S_ptr += row_idx * triton_cast(S_row_stride, tl.int64)
    X_ptr += row_idx * triton_cast(X_row_stride, tl.int64)
    R_ptr += row_idx * triton_cast(R_row_stride, tl.int64)
    RSTD_ptr += row_idx * triton_cast(RSTD_row_stride, tl.int64)

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    R_row = tl.load(R_ptr + col_offsets, mask=mask, other=0)
    S_row = X_row + R_row
    tl.store(S_ptr + col_offsets, S_row, mask=mask)
    S_row_dtype = S_row.dtype
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)

    if casting_mode == 0:
        S_row = S_row.to(tl.float32)

    if casting_mode == 1:
        W_row = W_row.to(tl.float32)
        S_row = S_row.to(tl.float32)

    if casting_mode == -1:
        eps = eps.to(S_row_dtype)
        offset = offset.to(S_row_dtype)

    mean_square = tl.sum(S_row * S_row, axis=0) / n_cols
    row_rstd = rsqrt(mean_square + eps)
    tl.store(RSTD_ptr, row_rstd)

    S_row = S_row * row_rstd

    if casting_mode == 0:
        S_row = S_row.to(S_row_dtype)

    Y_row = S_row * (offset + W_row)

    if casting_mode == 1:
        Y_row = Y_row.to(S_row_dtype)

    tl.store(Y_ptr + col_offsets, Y_row, mask=mask)


@triton.jit
def _fused_add_rms_norm_backward_kernel(
    dY_ptr,
    dY_row_stride,
    dS_out_ptr,
    dS_out_row_stride,
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
    rows_per_program: tl.constexpr,
    casting_mode: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    COMPUTE_DW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    has_dS_out: tl.constexpr,
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
    ds_out_stride64 = triton_cast(dS_out_row_stride, tl.int64)

    for row_idx in range(row_start, row_end):
        dy_base = dY_ptr + row_idx * dy_stride64
        dx_base = dX_ptr + row_idx * dx_stride64
        x_base = X_ptr + row_idx * x_stride64
        rstd_base = RSTD_ptr + row_idx * rstd_stride64

        dY_row = tl.load(dy_base + col_offsets, mask=mask, other=0.0)
        X_row = tl.load(x_base + col_offsets, mask=mask, other=0.0)
        rstd_row = tl.load(rstd_base)

        X_row = X_row.to(tl.float32)

        if COMPUTE_DW:
            if casting_mode == 0:
                dW_row += dY_row * (X_row * rstd_row).to(X_dtype)
            else:
                dW_row += dY_row * (X_row * rstd_row)

        if casting_mode == 0:
            m = (dY_row * W_row).to(tl.float32) if HAS_WEIGHT else dY_row.to(tl.float32)
        elif casting_mode == 1:
            dY_row = dY_row.to(tl.float32)
            m = dY_row * W_row if HAS_WEIGHT else dY_row
        else:
            m = dY_row * W_row if HAS_WEIGHT else dY_row

        dot = tl.sum(m * X_row, axis=0)
        c = -(1.0 / n_cols) * rstd_row * rstd_row * rstd_row * dot
        dX_row = rstd_row * m + c * X_row

        if has_dS_out:
            ds_base = dS_out_ptr + row_idx * ds_out_stride64
            dS_out_row = tl.load(ds_base + col_offsets, mask=mask, other=0.0)
            dX_row += dS_out_row

        tl.store(dx_base + col_offsets, dX_row.to(X_dtype), mask=mask)

    if COMPUTE_DW:
        tl.store(dW_ptr + row_block_id * dW_row_stride + col_offsets, dW_row, mask=mask)


def _fused_add_rms_norm_forward_triton(
    X: torch.Tensor,
    R: torch.Tensor,
    W: torch.Tensor,
    eps: float,
    offset: float,
    casting_mode: int,
):
    """Returns (Y, S, RSTD, BLOCK_SIZE, num_warps)."""
    shape = X.shape
    dim = shape[-1]
    X = X.contiguous().view(-1, dim)
    R = R.contiguous().view(-1, dim)
    n_rows, n_cols = X.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    if not (
        BLOCK_SIZE > _MAX_PER_ROW_KERNEL_BLOCK_SIZE
        or n_rows < _MIN_ROWS_FOR_BLOCK_KERNEL
    ):
        raise OperationError(
            *(
                "Opake fused add RMSNorm: block kernel path not yet ported; use "
                "shapes that satisfy BLOCK_SIZE > 256 or n_rows < 32768.",
            )
        )

    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    S = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    rstd_dtype = torch.float32 if casting_mode in (0, 1) else X.dtype
    RSTD = torch.empty(n_rows, dtype=rstd_dtype, device=X.device)
    W_contig = W.contiguous()

    def grid(meta):
        return (n_rows,)

    with torch_gpu_device(X.device):
        _fused_add_rms_norm_forward_kernel[grid](
            Y,
            Y.stride(0),
            S,
            S.stride(0),
            X,
            X.stride(0),
            R,
            R.stride(0),
            W_contig,
            W_contig.stride(0),
            RSTD,
            RSTD.stride(0),
            n_cols,
            eps,
            offset,
            casting_mode,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=2,
        )

    return Y.view(*shape), S.view(*shape), RSTD, BLOCK_SIZE, num_warps


def _fused_add_rms_norm_backward_triton(
    dY: torch.Tensor,
    dS_out: torch.Tensor | None,
    S: torch.Tensor,
    W: torch.Tensor,
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
            *(f"fused add RMSNorm: hidden dim {n_cols} exceeds block {BLOCK_SIZE}.",)
        )

    has_dS = dS_out is not None
    # dY is unused when has_dS_out=False
    dS_out_2d = dS_out.contiguous().view(-1, dim) if has_dS else dY
    S = S.contiguous().view(-1, dim)

    if S.device.type == "cuda":
        sm_count = torch.cuda.get_device_properties(S.device).multi_processor_count
    else:
        sm_count = 1

    _dW = (
        torch.empty((sm_count, n_cols), dtype=torch.float32, device=W.device)
        if compute_dw
        else None
    )
    rows_per_program = math.ceil(n_rows / sm_count)
    grid = (sm_count,)

    dX = dY if in_place else torch.empty_like(dY)
    W_contig = W.contiguous()
    dW_ptr = _dW if _dW is not None else dY
    x_dtype_triton = _TORCH_TO_TRITON_DTYPES[S.dtype]

    with torch_gpu_device(S.device):
        _fused_add_rms_norm_backward_kernel[grid](
            dY,
            dY.stride(0),
            dS_out_2d,
            dS_out_2d.stride(0),
            dX,
            dX.stride(0),
            S,
            S.stride(0),
            x_dtype_triton,
            W_contig,
            W_contig.stride(0),
            RSTD,
            RSTD.stride(0),
            dW_ptr,
            _dW.stride(0) if _dW is not None else 0,
            n_rows,
            n_cols,
            offset,
            rows_per_program,
            casting_mode,
            HAS_WEIGHT=True,
            COMPUTE_DW=compute_dw,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            has_dS_out=has_dS,
        )

    dComb = dX.view(*shape)
    dW = _dW.sum(dim=0).to(W.dtype) if _dW is not None else None
    return dComb, dW


class _FusedAddRMSNormBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        dY,
        dS_out,
        S,
        W,
        RSTD,
        offset,
        casting_mode,
        BLOCK_SIZE,
        num_warps,
        in_place,
        compute_dw,
    ):
        dComb, dW = _fused_add_rms_norm_backward_triton(
            dY,
            dS_out,
            S,
            W,
            RSTD,
            offset,
            casting_mode,
            BLOCK_SIZE,
            num_warps,
            in_place,
            compute_dw,
        )
        return dComb, dW if dW is not None else W.new_empty(0)

    @staticmethod
    def setup_context(ctx, inputs, output):
        del ctx

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Double backward not supported for FusedAddRMSNorm")

    @staticmethod
    def vmap(
        info,
        in_dims,
        dY,
        dS_out,
        S,
        W,
        RSTD,
        offset,
        casting_mode,
        BLOCK_SIZE,
        num_warps,
        in_place,
        compute_dw,
    ):
        del info
        dy_b, ds_b, s_b, w_b, r_b, *static_dims = in_dims
        if any(dim is not None for dim in static_dims):
            raise ConfigurationError(*("Fused RMSNorm metadata must not be vmapped",))
        if w_b is not None:
            raise ConfigurationError(*("W must not be vmapped",))
        if dy_b != 0 or s_b != 0 or r_b != 0:
            raise ConfigurationError(*("dY, S, RSTD must be vmapped at dim 0",))

        H = S.shape[-1]
        head_dy = dY.shape[:-1]
        B = dY.shape[0]
        dY_m = dY.reshape(-1, H)
        if dS_out is None:
            dS_m = None
            if ds_b is not None:
                raise ConfigurationError(
                    *("dS_out in_dims must be None when dS_out is None",)
                )
        else:
            if ds_b != 0:
                raise ConfigurationError(
                    *("dS_out must be vmapped at dim 0 when provided",)
                )
            dS_m = dS_out.reshape(-1, H)
        S_m = S.reshape(-1, H)
        R_m = RSTD.reshape(-1)
        dW_out = None
        if compute_dw:
            dW_out = _rms_norm_weight_grad_triton(
                dY,
                S,
                RSTD,
                W,
                B,
                casting_mode,
                BLOCK_SIZE,
                num_warps,
            )

        dComb, _ = _fused_add_rms_norm_backward_triton(
            dY_m,
            dS_m,
            S_m,
            W,
            R_m,
            offset,
            casting_mode,
            BLOCK_SIZE,
            num_warps,
            in_place,
            False,
        )
        dComb_out = dComb.view(*head_dy, H)
        if dW_out is not None:
            return (dComb_out, dW_out), (dy_b, 0)
        return (dComb_out, W.new_empty(0)), (dy_b, None)


class Opake_FusedAddRMSNorm(torch.autograd.Function):
    """Residual add then RMSNorm (Llama / Gemma casting modes)."""

    @staticmethod
    def forward(X, R, W, eps, offset, casting_mode, in_place):
        cm = _casting_mode_int(casting_mode)
        orig_shape = X.shape
        Y, S, RSTD, _, _ = _fused_add_rms_norm_forward_triton(
            X, R, W, float(eps), float(offset), cm
        )
        return Y.view(orig_shape), S.view(orig_shape), RSTD.view(orig_shape[:-1])

    @staticmethod
    def setup_context(ctx, inputs, output):
        _X, _R, W, _eps, offset, casting_mode, in_place = inputs
        _, S, RSTD = output
        dim = S.shape[-1]

        ctx.mark_non_differentiable(RSTD)
        ctx.original_shape = S.shape
        ctx.offset = float(offset)
        ctx.casting_mode = _casting_mode_int(casting_mode)
        ctx.block_size, ctx.num_warps = calculate_settings(dim)
        ctx.in_place = bool(in_place)
        ctx.compute_dw = W.requires_grad
        ctx.save_for_backward(S.view(-1, dim), W.contiguous(), RSTD.reshape(-1))

    @staticmethod
    def backward(ctx, grad_Y, grad_S, _grad_RSTD):
        S_s, W, RSTD = ctx.saved_tensors
        if grad_Y is None:
            grad_Y = torch.zeros(ctx.original_shape, device=W.device, dtype=S_s.dtype)
        go = grad_Y.contiguous()
        dim = go.shape[-1]
        go2 = go.reshape(-1, dim)
        grad_S_arg = grad_S
        if grad_S is not None:
            grad_S_arg = grad_S.contiguous().reshape(-1, dim)
        dComb, dW = _FusedAddRMSNormBackward.apply(
            go2,
            grad_S_arg,
            S_s,
            W,
            RSTD,
            ctx.offset,
            ctx.casting_mode,
            ctx.block_size,
            ctx.num_warps,
            ctx.in_place,
            ctx.compute_dw,
        )
        dComb = dComb.view(ctx.original_shape)
        return (
            dComb,
            dComb,
            dW if ctx.compute_dw else None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def vmap(info, in_dims, X, R, W, eps, offset, casting_mode, in_place):
        del info
        x_b, r_b = in_dims[0], in_dims[1]
        if x_b is None or r_b is None or x_b != r_b:
            raise ConfigurationError(
                *("Opake_FusedAddRMSNorm vmap: X and R must share the same vmap dim",)
            )
        if x_b != 0:
            raise ConfigurationError(
                *("Opake_FusedAddRMSNorm vmap: use dim 0 for X and R",)
            )
        for i in range(2, 7):
            if in_dims[i] is not None:
                raise ConfigurationError(*("Only X and R may be batched under vmap",))
        cm = _casting_mode_int(casting_mode)
        shape = X.shape
        H = shape[-1]
        Xf = X.reshape(-1, H).contiguous()
        Rf = R.reshape(-1, H).contiguous()
        Yf, Sf, RSTD, _, _ = _fused_add_rms_norm_forward_triton(
            Xf, Rf, W, float(eps), float(offset), cm
        )
        return (Yf.view(shape), Sf.view(shape), RSTD.view(shape[:-1])), (0, 0, 0)


def opake_fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    offset: float = 0.0,
    casting_mode: str = "llama",
    *,
    in_place_backward: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused ``S = x + residual`` then Llama/Gemma RMSNorm; returns ``(norm(S), S)``."""
    if not x.is_cuda:
        raise OperationError(*("opake_fused_add_rms_norm Triton path requires CUDA",))
    x, residual, weight = follow_autocast(x, residual, weight)
    normalized, summed, _ = Opake_FusedAddRMSNorm.apply(
        x,
        residual,
        weight,
        eps,
        offset,
        casting_mode,
        in_place_backward,
    )
    return normalized, summed
