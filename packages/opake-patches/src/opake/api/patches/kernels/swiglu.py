# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
#
# Triton SwiGLU kernels derive from the Unsloth project
# (Apache-2.0; https://github.com/unslothai/unsloth) and have been adapted to
# Opake's vmap-friendly new-style autograd dispatch. See NOTICE in the
# repository root.
"""SwiGLU kernel with vmap support for DP-SGD.

Implements the SwiGLU feed-forward activation from Shazeer,
*GLU Variants Improve Transformer* (https://arxiv.org/abs/2002.05202).
"""

import torch
import triton
import triton.language as tl

from opake.exceptions import ConfigurationError

from ._utils import (
    INT32_SAFETY_BUFFER,
    ensure_cuda_tensors,
    follow_autocast,
    torch_gpu_device,
)

BLOCK_SIZE = 1024


@triton.jit
def _fg_kernel(
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

    # Compute in float32 for precision (matching Unsloth)
    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)

    # f = e * sigmoid(e)
    f_row = e_row * tl.sigmoid(e_row)
    f_row = f_row.to(g_row.dtype)  # Cast back to input dtype
    # h = f * g
    h_row = f_row * g_row

    tl.store(h + offsets, h_row, mask=mask)


@triton.jit
def _DWf_DW_dfg_kernel(
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
    SwiGLU backward pass: h = silu(e) * g = (e * sigmoid(e)) * g

    Following Unsloth's exact formula:
    - e = e.float()
    - se = 1.0 / (1.0 + torch.exp(-e))
    - f = (se * e).to(dtype)
    - h = f * g
    - df = DW * f
    - dg = DW * g
    - de = (dg.float() * se * (1.0 + e * (1.0 - se))).to(dtype)

    When COMPUTE_H=True, also computes h and stores to h_out.
    For in-place use: pass de_out=e, dg_out=g, h_out=DW (Unsloth pattern).
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
    # Compute in float32 for precision (matching Unsloth)
    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)

    # se = sigmoid(e) in float32
    se_row = tl.sigmoid(e_row)
    # f = (se * e).to(dtype)
    f_row = se_row * e_row
    f_row = f_row.to(DW_row.dtype)

    # df = DW * f (this is grad_up in our convention)
    df_row = DW_row * f_row

    # dg = DW * g (intermediate for computing de)
    dg_row = DW_row * g_row

    # de = (dg.float() * se * (1.0 + e * (1.0 - se))).to(dtype)
    de_row = dg_row.to(tl.float32) * se_row * (1.0 + e_row * (1.0 - se_row))
    de_row = de_row.to(DW_row.dtype)

    # Store derivatives to output buffers
    # de_out = grad_gate, dg_out = grad_up
    tl.store(de_out + offsets, de_row, mask=mask)
    tl.store(dg_out + offsets, df_row, mask=mask)

    # Optionally store h = silu(e) * g (recomputed, avoids saving h in forward)
    if COMPUTE_H:
        h_row = f_row * g_row
        tl.store(h_out + offsets, h_row, mask=mask)


class _SwiGLUBackward(torch.autograd.Function):
    """Backward pass wrapped as autograd.Function for vmap(grad()) support.

    When vmap(grad(fn)) runs backward, functorch intercepts .apply() and
    routes to vmap(), where tensors are regular and Triton kernels work.
    """

    @staticmethod
    def forward(grad_h_flat, gate_flat, up_flat):
        n_elements = gate_flat.numel()

        def grid(meta):
            return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        with torch_gpu_device(gate_flat.device):
            # In-place: gate_flat → grad_gate, up_flat → grad_up
            # Kernel reads e(gate), g(up) before writing de_out, dg_out
            _DWf_DW_dfg_kernel[grid](
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
        pass  # No double backward needed

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Double backward not supported for SwiGLU")

    @staticmethod
    def vmap(info, in_dims, grad_h_flat, gate_flat, up_flat):
        grad_h_bdim, gate_bdim, up_bdim = in_dims

        if not (grad_h_bdim == gate_bdim == up_bdim):
            # Mismatched batch dims would silently pair elements across examples.
            raise ConfigurationError(
                *(f"SwiGLU backward vmap requires matching batch dims, got {in_dims}",)
            )

        # Merge vmap batch dim into flat dim (element-wise kernel)
        batched_shape = gate_flat.shape
        grad_h_merged = grad_h_flat.reshape(-1)
        gate_merged = gate_flat.reshape(-1)
        up_merged = up_flat.reshape(-1)

        n_elements = gate_merged.numel()

        def grid(meta):
            return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        with torch_gpu_device(gate_merged.device):
            # In-place: gate_merged → grad_gate, up_merged → grad_up
            _DWf_DW_dfg_kernel[grid](
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


class Opake_SwiGLU(torch.autograd.Function):
    """SwiGLU autograd kernel with a custom vmap rule."""

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
            _fg_kernel[grid](
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
            raise NotImplementedError("Repeated backward not supported for SwiGLU")
        ctx._opake_backward_done = True
        gate_flat, up_flat = ctx.saved_tensors
        grad_h_flat = grad_h.reshape(-1).contiguous()
        grad_gate_flat, grad_up_flat = _SwiGLUBackward.apply(
            grad_h_flat, gate_flat, up_flat
        )
        return grad_gate_flat.reshape(ctx.original_shape), grad_up_flat.reshape(
            ctx.original_shape
        )

    @staticmethod
    def vmap(info, in_dims, gate, up):
        """Custom vmap rule for DP-SGD.

        Calls Triton forward kernel directly with merged batch dims.
        Tensors are regular here (unwrapped by functorch), Triton works.
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
            _fg_kernel[grid](
                gate_flat,
                up_flat,
                h,
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
                LONG_INDEXING=0 if n_elements <= INT32_SAFETY_BUFFER else 1,
            )

        return h.reshape(batched_shape), gate_bdim


def opake_swiglu(gate, up):
    """Convenience wrapper."""
    ensure_cuda_tensors(gate, up, fn_name="opake_swiglu")
    gate, up = follow_autocast(gate, up)
    return Opake_SwiGLU.apply(gate, up)


def _triton_swiglu_forward(gate, up):
    """Direct Triton SwiGLU forward: h = silu(gate) * up.

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
        _fg_kernel[grid](
            gate_flat,
            up_flat,
            h,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            LONG_INDEXING=0 if n_elements <= INT32_SAFETY_BUFFER else 1,
        )

    return h.reshape(original_shape)


def _triton_swiglu_backward(grad_h, gate, up):
    """Direct Triton SwiGLU backward.

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
        _DWf_DW_dfg_kernel[grid](
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


def _triton_swiglu_backward_fused(dh, gate, up):
    """In-place SwiGLU backward that also recomputes h (Unsloth pattern).

    Overwrites all three input buffers:
    - dh → h (activation output, recomputed from gate and up)
    - gate → dgate (gradient w.r.t. gate input)
    - up → dup (gradient w.r.t. up input)

    Returns (h, dgate, dup) which alias the input buffers.
    For use in LoRA_MLP backward to avoid saving h and allocating grad buffers.
    """
    original_shape = gate.shape
    dh_flat = dh.reshape(-1)
    gate_flat = gate.reshape(-1)
    up_flat = up.reshape(-1)
    n_elements = gate_flat.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_gpu_device(gate.device):
        _DWf_DW_dfg_kernel[grid](
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

    h = dh_flat.reshape(original_shape)
    dgate = gate_flat.reshape(original_shape)
    dup = up_flat.reshape(original_shape)
    return h, dgate, dup
