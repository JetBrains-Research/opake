# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
#
# Triton GeGLU kernels derive from the Unsloth project
# (Apache-2.0; https://github.com/unslothai/unsloth) and have been adapted to
# Opake's vmap-friendly new-style autograd dispatch. See NOTICE in the
# repository root.
"""GeGLU (exact and approx) kernels with vmap support for DP-SGD.

Implements the GEGLU feed-forward activation from Shazeer,
*GLU Variants Improve Transformer* (https://arxiv.org/abs/2002.05202).
"""

import math

import torch
import triton
import triton.language as tl

from opake.exceptions import ConfigurationError

from ._utils import (
    INT32_SAFETY_BUFFER,
    ensure_cuda_tensors,
    follow_autocast,
    torch_gpu_device,
    triton_tanh,
)

BLOCK_SIZE = 1024

_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


# ===== GeGLU Exact (erf-based) =====


@triton.jit
def _exact_forward_kernel(
    e,
    g,
    h,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    LONG_INDEXING: tl.constexpr,
):
    block_idx = tl.program_id(0)
    if LONG_INDEXING:
        offsets = block_idx.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(
            tl.int64
        )
        n_elements = tl.cast(n_elements, tl.int64)
    else:
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)

    # f = 1/2 * e * (1 + erf(1/sqrt(2) * e))
    f_row = 0.5 * e_row * (tl.math.erf(tl.math.rsqrt(2.0) * e_row) + 1.0)
    f_row = f_row.to(g_row.dtype)
    h_row = f_row * g_row

    tl.store(h + offsets, h_row, mask=mask)


@triton.jit
def _exact_backward_kernel(
    DW,
    e,
    g,
    de_out,
    dg_out,
    h_out,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    LONG_INDEXING: tl.constexpr,
    COMPUTE_H: tl.constexpr,
):
    """
    GeGLU (exact) backward pass: h = gelu(e) * g

    Following Unsloth's formula exactly:
    f = 1/2 * e * (1 + erf(1/sqrt(2) * e))
    h = f * g

    df/de = 1/2 * (1 + erf(1/sqrt(2) * e)) + 1/sqrt(2*pi) * e * exp(-1/2 * e^2)

    Note: dg_out = DW * f (grad for up input)
          de_out = DW * g * df/de (grad for gate input)

    When COMPUTE_H=True, also computes h and stores to h_out.
    """
    block_idx = tl.program_id(0)
    if LONG_INDEXING:
        offsets = block_idx.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(
            tl.int64
        )
        n_elements = tl.cast(n_elements, tl.int64)
    else:
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    DW_row = tl.load(DW + offsets, mask=mask, other=0)
    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)

    # Break e_row away for re-use
    # f = 1/2 * e * (1 + erf(1/sqrt(2) * e))
    f_partial_row = 0.5 * (tl.math.erf(tl.math.rsqrt(2.0) * e_row) + 1.0)
    f_row = f_partial_row * e_row
    f_row = f_row.to(DW_row.dtype)

    # df = DW * f (grad for up)
    df_row = DW_row * f_row
    # dg = DW * g (intermediate for computing de)
    dg_row = DW_row * g_row

    # df/de = 1/2 * (1 + erf(1/sqrt(2) * e)) + 1/sqrt(2*pi) * e * exp(-1/2 * e^2)
    t = 0.3989422804014327  # 1/sqrt(2*pi)
    df_de = f_partial_row + t * e_row * tl.exp(-0.5 * e_row * e_row)

    de_row = dg_row.to(tl.float32) * df_de
    de_row = de_row.to(DW_row.dtype)

    # Store: de_out = grad_gate, dg_out = grad_up
    tl.store(de_out + offsets, de_row, mask=mask)
    tl.store(dg_out + offsets, df_row, mask=mask)

    if COMPUTE_H:
        h_row = f_row * g_row
        tl.store(h_out + offsets, h_row, mask=mask)


class _GeGLUExactBackward(torch.autograd.Function):
    """Backward pass wrapped as autograd.Function for vmap(grad()) support."""

    @staticmethod
    def forward(grad_h_flat, gate_flat, up_flat):
        n_elements = gate_flat.numel()

        def grid(meta):
            return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        with torch_gpu_device(gate_flat.device):
            # In-place: gate_flat → grad_gate, up_flat → grad_up
            _exact_backward_kernel[grid](
                grad_h_flat,
                gate_flat,
                up_flat,
                gate_flat,
                up_flat,
                gate_flat,  # de→gate, dg→up, h_out unused
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
                LONG_INDEXING=0 if n_elements <= INT32_SAFETY_BUFFER else 1,
                COMPUTE_H=False,
            )

        return gate_flat, up_flat

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Double backward not supported for GeGLU Exact")

    @staticmethod
    def vmap(info, in_dims, grad_h_flat, gate_flat, up_flat):
        grad_h_bdim, gate_bdim, up_bdim = in_dims

        if not (grad_h_bdim == gate_bdim == up_bdim):
            # Mismatched batch dims would silently pair elements across examples.
            raise ConfigurationError(
                *(f"GeGLU backward vmap requires matching batch dims, got {in_dims}",)
            )

        batched_shape = gate_flat.shape
        grad_h_merged = grad_h_flat.reshape(-1)
        gate_merged = gate_flat.reshape(-1)
        up_merged = up_flat.reshape(-1)

        n_elements = gate_merged.numel()

        def grid(meta):
            return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        with torch_gpu_device(gate_merged.device):
            # In-place: gate_merged → grad_gate, up_merged → grad_up
            _exact_backward_kernel[grid](
                grad_h_merged,
                gate_merged,
                up_merged,
                gate_merged,
                up_merged,
                gate_merged,  # de→gate, dg→up, h_out unused
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
                LONG_INDEXING=0 if n_elements <= INT32_SAFETY_BUFFER else 1,
                COMPUTE_H=False,
            )

        return (
            (gate_merged.reshape(batched_shape), up_merged.reshape(batched_shape)),
            (gate_bdim, up_bdim),
        )


class Opake_GeGLU_Exact(torch.autograd.Function):
    """Exact GeGLU autograd kernel with a custom vmap rule."""

    @staticmethod
    def forward(gate, up):
        """New-style API forward without ctx parameter."""
        original_shape = gate.shape
        gate_flat = gate.reshape(-1)
        up_flat = up.reshape(-1)
        n_elements = gate_flat.numel()

        h = torch.empty_like(gate_flat)

        def grid(meta):
            return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        with torch_gpu_device(gate.device):
            _exact_forward_kernel[grid](
                gate_flat,
                up_flat,
                h,
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
                LONG_INDEXING=0 if n_elements <= INT32_SAFETY_BUFFER else 1,
            )

        return h.reshape(original_shape)

    @staticmethod
    def setup_context(ctx, inputs, output):
        gate, up = inputs
        ctx.save_for_backward(gate.reshape(-1), up.reshape(-1))
        ctx.original_shape = gate.shape
        ctx.n_elements = gate.numel()

    @staticmethod
    def backward(ctx, grad_h):
        if getattr(ctx, "_opake_backward_done", False):
            raise NotImplementedError("Repeated backward not supported for GeGLU Exact")
        ctx._opake_backward_done = True
        gate_flat, up_flat = ctx.saved_tensors
        grad_h_flat = grad_h.reshape(-1).contiguous()
        grad_gate_flat, grad_up_flat = _GeGLUExactBackward.apply(
            grad_h_flat, gate_flat, up_flat
        )
        return grad_gate_flat.reshape(ctx.original_shape), grad_up_flat.reshape(
            ctx.original_shape
        )

    @staticmethod
    def vmap(info, in_dims, gate, up):
        """Custom vmap rule for DP-SGD.

        Calls Triton forward kernel directly with merged batch dims.
        """
        gate_bdim, up_bdim = in_dims
        if gate_bdim != 0 or up_bdim != 0:
            raise ConfigurationError(*("Both gate and up should be batched at dim 0",))

        batched_shape = gate.shape
        gate_flat = gate.reshape(-1)
        up_flat = up.reshape(-1)
        n_elements = gate_flat.numel()

        h = torch.empty_like(gate_flat)

        def grid(meta):
            return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        with torch_gpu_device(gate.device):
            _exact_forward_kernel[grid](
                gate_flat,
                up_flat,
                h,
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
                LONG_INDEXING=0 if n_elements <= INT32_SAFETY_BUFFER else 1,
            )

        return h.reshape(batched_shape), gate_bdim


# ===== GeGLU Approx (tanh-based) =====


@triton.jit
def _approx_forward_kernel(
    e,
    g,
    h,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    LONG_INDEXING: tl.constexpr,
):
    block_idx = tl.program_id(0)
    if LONG_INDEXING:
        offsets = block_idx.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(
            tl.int64
        )
        n_elements = tl.cast(n_elements, tl.int64)
    else:
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    s = 0.7978845608028654  # math.sqrt(2 / math.pi)
    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)

    f_row = (
        0.5 * e_row * (triton_tanh(s * e_row * (1.0 + 0.044715 * e_row * e_row)) + 1.0)
    )
    f_row = f_row.to(g_row.dtype)
    h_row = f_row * g_row

    tl.store(h + offsets, h_row, mask=mask)


@triton.jit
def _approx_backward_kernel(
    DW,
    e,
    g,
    de_out,
    dg_out,
    h_out,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    LONG_INDEXING: tl.constexpr,
    COMPUTE_H: tl.constexpr,
):
    """
    GeGLU (approx tanh) backward pass: h = gelu_tanh(e) * g

    Following Unsloth's formula exactly:
    f = 1/2 * e * (1 + tanh( sqrt(2/pi) * x * (1 + 0.044715 * x^2 ) ))

    df/de = 1/2 * [1 + tanh( sqrt(2/pi) * x * (1 + 0.044715 * x^2 ) )] +
            1/2 * sech^2 [   sqrt(2/pi) * x * (1 + 0.044715 * x^2 )  ] *
                       ( sqrt(2/pi) * x * (1 + 0.044715 * x^2 * 3 ) )

    Note: dg_out = DW * f (grad for up input)
          de_out = DW * g * df/de (grad for gate input)

    When COMPUTE_H=True, also computes h and stores to h_out.
    """
    block_idx = tl.program_id(0)
    if LONG_INDEXING:
        offsets = block_idx.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(
            tl.int64
        )
        n_elements = tl.cast(n_elements, tl.int64)
    else:
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    DW_row = tl.load(DW + offsets, mask=mask, other=0)
    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)

    # See https://www.desmos.com/calculator/nqprfoni6x
    s = 0.7978845608028654  # math.sqrt(2 / math.pi)
    a = s * e_row  # a = sqrt(2 / pi) * x
    b = a * 0.044715 * e_row * e_row  # b = a * 0.044715 * x^2
    T = 1.0 + triton_tanh(a + b)
    T2 = 0.5 * T
    # Q = 0.5 * -T * (T - 2.0) * (a + 3.0 * b)
    Q2 = -T2 * (T - 2.0) * (a + 3.0 * b)
    df_de = T2 + Q2  # 1/2 * (T + Q)

    # f = 1/2 * e * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3 ) ))
    f_row = T2 * e_row
    f_row = f_row.to(DW_row.dtype)

    # df = DW * f (grad for up)
    df_row = DW_row * f_row
    # dg = DW * g (intermediate for computing de)
    dg_row = DW_row * g_row

    de_row = dg_row.to(tl.float32) * df_de
    de_row = de_row.to(DW_row.dtype)

    # Store: de_out = grad_gate, dg_out = grad_up
    tl.store(de_out + offsets, de_row, mask=mask)
    tl.store(dg_out + offsets, df_row, mask=mask)

    if COMPUTE_H:
        h_row = f_row * g_row
        tl.store(h_out + offsets, h_row, mask=mask)


class _GeGLUApproxBackward(torch.autograd.Function):
    """Backward pass wrapped as autograd.Function for vmap(grad()) support."""

    @staticmethod
    def forward(grad_h_flat, gate_flat, up_flat):
        n_elements = gate_flat.numel()

        def grid(meta):
            return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        with torch_gpu_device(gate_flat.device):
            # In-place: gate_flat → grad_gate, up_flat → grad_up
            _approx_backward_kernel[grid](
                grad_h_flat,
                gate_flat,
                up_flat,
                gate_flat,
                up_flat,
                gate_flat,  # de→gate, dg→up, h_out unused
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
                LONG_INDEXING=0 if n_elements <= INT32_SAFETY_BUFFER else 1,
                COMPUTE_H=False,
            )

        return gate_flat, up_flat

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Double backward not supported for GeGLU Approx")

    @staticmethod
    def vmap(info, in_dims, grad_h_flat, gate_flat, up_flat):
        grad_h_bdim, gate_bdim, up_bdim = in_dims

        if not (grad_h_bdim == gate_bdim == up_bdim):
            # Mismatched batch dims would silently pair elements across examples.
            raise ConfigurationError(
                *(f"GeGLU backward vmap requires matching batch dims, got {in_dims}",)
            )

        batched_shape = gate_flat.shape
        grad_h_merged = grad_h_flat.reshape(-1)
        gate_merged = gate_flat.reshape(-1)
        up_merged = up_flat.reshape(-1)

        n_elements = gate_merged.numel()

        def grid(meta):
            return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        with torch_gpu_device(gate_merged.device):
            # In-place: gate_merged → grad_gate, up_merged → grad_up
            _approx_backward_kernel[grid](
                grad_h_merged,
                gate_merged,
                up_merged,
                gate_merged,
                up_merged,
                gate_merged,  # de→gate, dg→up, h_out unused
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
                LONG_INDEXING=0 if n_elements <= INT32_SAFETY_BUFFER else 1,
                COMPUTE_H=False,
            )

        return (
            (gate_merged.reshape(batched_shape), up_merged.reshape(batched_shape)),
            (gate_bdim, up_bdim),
        )


class Opake_GeGLU_Approx(torch.autograd.Function):
    """Tanh-approximated GeGLU autograd kernel with a custom vmap rule."""

    @staticmethod
    def forward(gate, up):
        """New-style API forward without ctx parameter."""
        original_shape = gate.shape
        gate_flat = gate.reshape(-1)
        up_flat = up.reshape(-1)
        n_elements = gate_flat.numel()

        h = torch.empty_like(gate_flat)

        def grid(meta):
            return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        with torch_gpu_device(gate.device):
            _approx_forward_kernel[grid](
                gate_flat,
                up_flat,
                h,
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
                LONG_INDEXING=0 if n_elements <= INT32_SAFETY_BUFFER else 1,
            )

        return h.reshape(original_shape)

    @staticmethod
    def setup_context(ctx, inputs, output):
        gate, up = inputs
        ctx.save_for_backward(gate.reshape(-1), up.reshape(-1))
        ctx.original_shape = gate.shape
        ctx.n_elements = gate.numel()

    @staticmethod
    def backward(ctx, grad_h):
        if getattr(ctx, "_opake_backward_done", False):
            raise NotImplementedError(
                "Repeated backward not supported for GeGLU Approx"
            )
        ctx._opake_backward_done = True
        gate_flat, up_flat = ctx.saved_tensors
        grad_h_flat = grad_h.reshape(-1).contiguous()
        grad_gate_flat, grad_up_flat = _GeGLUApproxBackward.apply(
            grad_h_flat, gate_flat, up_flat
        )
        return grad_gate_flat.reshape(ctx.original_shape), grad_up_flat.reshape(
            ctx.original_shape
        )

    @staticmethod
    def vmap(info, in_dims, gate, up):
        """Custom vmap rule for DP-SGD.

        Calls Triton forward kernel directly with merged batch dims.
        """
        gate_bdim, up_bdim = in_dims
        if gate_bdim != 0 or up_bdim != 0:
            raise ConfigurationError(*("Both gate and up should be batched at dim 0",))

        batched_shape = gate.shape
        gate_flat = gate.reshape(-1)
        up_flat = up.reshape(-1)
        n_elements = gate_flat.numel()

        h = torch.empty_like(gate_flat)

        def grid(meta):
            return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        with torch_gpu_device(gate.device):
            _approx_forward_kernel[grid](
                gate_flat,
                up_flat,
                h,
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
                LONG_INDEXING=0 if n_elements <= INT32_SAFETY_BUFFER else 1,
            )

        return h.reshape(batched_shape), gate_bdim


# Convenience wrappers
def opake_geglu_exact(gate, up):
    """Apply exact GeGLU to matching CUDA tensors.

    Args:
        gate: Gate-projection tensor.
        up: Up-projection tensor with the same shape as ``gate``.

    Returns:
        The elementwise exact-GeGLU activation.
    """
    ensure_cuda_tensors(gate, up, fn_name="opake_geglu_exact")
    gate, up = follow_autocast(gate, up)
    return Opake_GeGLU_Exact.apply(gate, up)


def opake_geglu_approx(gate, up):
    """Apply tanh-approximated GeGLU to matching CUDA tensors.

    Args:
        gate: Gate-projection tensor.
        up: Up-projection tensor with the same shape as ``gate``.

    Returns:
        The elementwise approximate-GeGLU activation.
    """
    ensure_cuda_tensors(gate, up, fn_name="opake_geglu_approx")
    gate, up = follow_autocast(gate, up)
    return Opake_GeGLU_Approx.apply(gate, up)


def _triton_geglu_exact_forward(gate, up):
    """Direct Triton GeGLU (exact/erf) forward: h = gelu(gate) * up.

    Calls the Triton kernel directly without autograd wrapper.
    For use as a callback in LoRA_MLP.
    """
    original_shape = gate.shape
    gate_flat = gate.reshape(-1)
    up_flat = up.reshape(-1)
    n_elements = gate_flat.numel()

    h = torch.empty_like(gate_flat)

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_gpu_device(gate.device):
        _exact_forward_kernel[grid](
            gate_flat,
            up_flat,
            h,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            LONG_INDEXING=0 if n_elements <= INT32_SAFETY_BUFFER else 1,
        )

    return h.reshape(original_shape)


def _triton_geglu_exact_backward(grad_h, gate, up):
    """Direct Triton GeGLU (exact/erf) backward.

    Calls the Triton backward kernel directly without autograd wrapper.
    For use as a callback in LoRA_MLP.

    Args:
        grad_h: Gradient of output (same shape as gate/up)
        gate: Gate input from forward pass
        up: Up input from forward pass

    Returns:
        (grad_gate, grad_up) tuple
    """
    original_shape = gate.shape
    grad_h_flat = grad_h.reshape(-1)
    gate_flat = gate.reshape(-1)
    up_flat = up.reshape(-1)
    n_elements = gate_flat.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_gpu_device(gate.device):
        # In-place: gate_flat → grad_gate, up_flat → grad_up
        _exact_backward_kernel[grid](
            grad_h_flat,
            gate_flat,
            up_flat,
            gate_flat,
            up_flat,
            gate_flat,  # de→gate, dg→up, h_out unused
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            LONG_INDEXING=0 if n_elements <= INT32_SAFETY_BUFFER else 1,
            COMPUTE_H=False,
        )

    return gate_flat.reshape(original_shape), up_flat.reshape(original_shape)


def _triton_geglu_exact_backward_fused(dh, gate, up):
    """In-place GeGLU (exact) backward that also recomputes h.

    Overwrites: dh → h, gate → dgate, up → dup.
    Returns (h, dgate, dup) which alias the input buffers.
    """
    original_shape = gate.shape
    dh_flat = dh.reshape(-1)
    gate_flat = gate.reshape(-1)
    up_flat = up.reshape(-1)
    n_elements = gate_flat.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_gpu_device(gate.device):
        _exact_backward_kernel[grid](
            dh_flat,
            gate_flat,
            up_flat,
            gate_flat,
            up_flat,
            dh_flat,  # In-place: de→gate, dg→up, h→dh
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            LONG_INDEXING=0 if n_elements <= INT32_SAFETY_BUFFER else 1,
            COMPUTE_H=True,
        )

    return (
        dh_flat.reshape(original_shape),
        gate_flat.reshape(original_shape),
        up_flat.reshape(original_shape),
    )


def _triton_geglu_approx_forward(gate, up):
    """Direct Triton GeGLU (approx/tanh) forward: h = gelu_tanh(gate) * up.

    Calls the Triton kernel directly without autograd wrapper.
    For use as a callback in LoRA_MLP.
    """
    original_shape = gate.shape
    gate_flat = gate.reshape(-1)
    up_flat = up.reshape(-1)
    n_elements = gate_flat.numel()

    h = torch.empty_like(gate_flat)

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_gpu_device(gate.device):
        _approx_forward_kernel[grid](
            gate_flat,
            up_flat,
            h,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            LONG_INDEXING=0 if n_elements <= INT32_SAFETY_BUFFER else 1,
        )

    return h.reshape(original_shape)


def _triton_geglu_approx_backward(grad_h, gate, up):
    """Direct Triton GeGLU (approx/tanh) backward.

    Calls the Triton backward kernel directly without autograd wrapper.
    For use as a callback in LoRA_MLP.

    Args:
        grad_h: Gradient of output (same shape as gate/up)
        gate: Gate input from forward pass
        up: Up input from forward pass

    Returns:
        (grad_gate, grad_up) tuple
    """
    original_shape = gate.shape
    grad_h_flat = grad_h.reshape(-1)
    gate_flat = gate.reshape(-1)
    up_flat = up.reshape(-1)
    n_elements = gate_flat.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_gpu_device(gate.device):
        # In-place: gate_flat → grad_gate, up_flat → grad_up
        _approx_backward_kernel[grid](
            grad_h_flat,
            gate_flat,
            up_flat,
            gate_flat,
            up_flat,
            gate_flat,  # de→gate, dg→up, h_out unused
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            LONG_INDEXING=0 if n_elements <= INT32_SAFETY_BUFFER else 1,
            COMPUTE_H=False,
        )

    return gate_flat.reshape(original_shape), up_flat.reshape(original_shape)


def _triton_geglu_approx_backward_fused(dh, gate, up):
    """In-place GeGLU (approx) backward that also recomputes h.

    Overwrites: dh → h, gate → dgate, up → dup.
    Returns (h, dgate, dup) which alias the input buffers.
    """
    original_shape = gate.shape
    dh_flat = dh.reshape(-1)
    gate_flat = gate.reshape(-1)
    up_flat = up.reshape(-1)
    n_elements = gate_flat.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_gpu_device(gate.device):
        _approx_backward_kernel[grid](
            dh_flat,
            gate_flat,
            up_flat,
            gate_flat,
            up_flat,
            dh_flat,  # In-place: de→gate, dg→up, h→dh
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            LONG_INDEXING=0 if n_elements <= INT32_SAFETY_BUFFER else 1,
            COMPUTE_H=True,
        )

    return (
        dh_flat.reshape(original_shape),
        gate_flat.reshape(original_shape),
        up_flat.reshape(original_shape),
    )
