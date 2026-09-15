# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Fused grouped-GEMM MoE expert FFN — sparse (O(T*K)) perf path for ``opaque_moe``.

A re-derivation of Liger's ``LigerFusedMoEFunction`` made vmap- and
vmap(grad)-safe for Opaque's per-sample-gradient DP-SGD path. Same Function/vmap
contract as the dense :class:`~opaque.api.patches.kernels.moe.Opaque_MoE`
baseline, so ``opaque_moe`` dispatches CUDA+bf16/fp16 here and keeps the dense
torch path for CPU/fp32.

Strategy (per the MoE-port lessons): tokens are sorted by expert and run through
``torch._grouped_mm`` for the three mode-1 (offset-grouped along the token dim)
GEMMs — forward up-proj, forward down-proj, backward ``dx``. The mode-2
per-group weight grads (``dW1``/``dW2``) compute ``out[g] = A_g^T @ B_g`` with
the contraction grouped along the *token* axis. ``torch._grouped_mm`` expresses
this via its 2D×2D layout (``_grouped_mm(A.mT, B, offs)``); its 16-byte rule is
on matrix *strides* (the ``2I``/``H``/``I`` model dims), not the data-dependent
group sizes. A single custom Triton kernel :func:`_grouped_AtB` computes it
directly over the variable-size row groups as a fused variant (perf comparison
pending — #417). All reductions accumulate in fp32.

Per-sample weight grads under ``vmap(grad)`` (DP-SGD) use **virtual experts**:
sample ``b``'s tokens for real expert ``e`` are assigned to group ``b*E + e``, so
the grouped weight-grad naturally lands in a ``(B, E, ...)`` buffer — never
summed across the batch. The forward/dx GEMMs index the *shared* weights by the
real expert (``group % E``).
"""

# ``I`` is the per-expert intermediate dim throughout (paired with ``2I`` for the
# fused gate+up projection); single-letter tensor-shape names are intentional here.
# ruff: noqa: E741

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from ._grouped_moe import (
    _accumulate_grouped_AtB,
    _iter_route_plans,
    _plan_bytes,
    _route_plan,
    _route_subplan,
    _stream_grouped_weight_grads,
    _WeightGradPlan,
)
from ._moe_memory import (
    _workspace_budget_bytes,
    chunk_size,
    grouped_backward_bytes_per_route,
    grouped_forward_bytes_per_route,
)
from ._utils import active_cuda_dtype, cast_to_dtype, follow_autocast, torch_gpu_device

# ---------------------------------------------------------------------------
# Custom Triton kernel: grouped A^T @ B (the mode-2 per-group weight grad)
# ---------------------------------------------------------------------------


@triton.jit
def _grouped_AtB_kernel(
    A_ptr,  # (M, P)  rows grouped by expert via seg_offs
    B_ptr,  # (M, Q)
    seg_ptr,  # (G+1,) int32 — exclusive prefix offsets; group g is rows [seg[g], seg[g+1])
    out_ptr,  # (G, P, Q) — out[g] = A[group g]^T @ B[group g]
    P: tl.constexpr,
    Q: tl.constexpr,
    stride_a_m,
    stride_a_p: tl.constexpr,
    stride_b_m,
    stride_b_q: tl.constexpr,
    stride_o_g,
    stride_o_p,
    stride_o_q: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile over P (output rows)
    BLOCK_N: tl.constexpr,  # tile over Q (output cols)
    BLOCK_K: tl.constexpr,  # tile over the group's token rows (contraction)
):
    """Grid: (G * ceil(P/BLOCK_M), ceil(Q/BLOCK_N)). Early-exit on empty groups."""
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    n_p_tiles: tl.constexpr = (P + BLOCK_M - 1) // BLOCK_M
    # int64 guards expert/row offset overflow at large G*P*Q (mirrors Liger #1246).
    g = (pid0 // n_p_tiles).to(tl.int64)
    p_tile = pid0 % n_p_tiles

    seg_start = tl.load(seg_ptr + g)
    seg_end = tl.load(seg_ptr + g + 1)
    M_e = seg_end - seg_start
    if M_e == 0:
        return

    p_idx = p_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    q_idx = pid1 * BLOCK_N + tl.arange(0, BLOCK_N)
    p_mask = p_idx < P
    q_mask = q_idx < Q

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, M_e, BLOCK_K):
        k_off = k + tl.arange(0, BLOCK_K)
        k_mask = k_off < M_e
        rows = (seg_start + k_off).to(tl.int64)

        # A_blk[p, k] = A[row_k, p_idx_p]  -> (BLOCK_M, BLOCK_K)
        a_ptrs = A_ptr + rows[None, :] * stride_a_m + p_idx[:, None] * stride_a_p
        a_blk = tl.load(a_ptrs, mask=k_mask[None, :] & p_mask[:, None], other=0.0)
        # B_blk[k, q] = B[row_k, q_idx_q]  -> (BLOCK_K, BLOCK_N)
        b_ptrs = B_ptr + rows[:, None] * stride_b_m + q_idx[None, :] * stride_b_q
        b_blk = tl.load(b_ptrs, mask=k_mask[:, None] & q_mask[None, :], other=0.0)

        acc = tl.dot(a_blk, b_blk, acc=acc)

    o_ptrs = (
        out_ptr
        + g * stride_o_g
        + p_idx[:, None] * stride_o_p
        + q_idx[None, :] * stride_o_q
    )
    tl.store(
        o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=p_mask[:, None] & q_mask[None, :]
    )


def _grouped_AtB(A, B, ends, G, *, out=None):
    """Compute grouped ``A^T @ B``, optionally directly into a final output view."""
    A = A.contiguous()
    B = B.contiguous()
    seg_offs = torch.cat([ends.new_zeros(1), ends])
    P, Q = A.shape[1], B.shape[1]
    if out is None:
        out = torch.zeros(G, P, Q, dtype=torch.float32, device=A.device)
    else:
        out.zero_()
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32

    def grid(meta):
        return (G * triton.cdiv(P, meta["BLOCK_M"]), triton.cdiv(Q, meta["BLOCK_N"]))

    with torch_gpu_device(A.device):
        _grouped_AtB_kernel[grid](
            A,
            B,
            seg_offs,
            out,
            P=P,
            Q=Q,
            stride_a_m=A.stride(0),
            stride_a_p=A.stride(1),
            stride_b_m=B.stride(0),
            stride_b_q=B.stride(1),
            stride_o_g=out.stride(0),
            stride_o_p=out.stride(1),
            stride_o_q=out.stride(2),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
    return out


# ---------------------------------------------------------------------------
# Routing + grouped-GEMM helpers
# ---------------------------------------------------------------------------


def _grouped_mm(A, Bw, ends):
    """``torch._grouped_mm`` mode-1: ``A`` (M,Kc) grouped along M by ``ends`` @
    ``Bw`` (G,Kc,Nc) -> (M,Nc). ``ends`` = int32 cumulative group ends.

    ``Bw`` may be a non-contiguous ``.mT`` view — grouped_mm consumes it directly,
    avoiding a transient full-weight copy."""
    return torch._grouped_mm(A.contiguous(), Bw, offs=ends)


def _fused_moe_forward(x_flat, W1, W2, expert_of_row, tw_row, K):
    """Sparse grouped MoE forward with planner-bounded routing and activations."""
    N, H = x_flat.shape
    E = W1.shape[0]
    I = W1.shape[1] // 2
    dt = x_flat.dtype
    out = torch.zeros(N, H, dtype=torch.float32, device=x_flat.device)
    per_route = grouped_forward_bytes_per_route(H, I, x_flat.element_size())
    budget = _workspace_budget_bytes(x_flat.device)
    plan_chunk = chunk_size(
        expert_of_row.numel(),
        24,
        x_flat.device,
        fixed_bytes=out.numel() * out.element_size(),
        budget_bytes=budget,
    )
    for lo in range(0, expert_of_row.numel(), plan_chunk):
        hi = min(lo + plan_chunk, expert_of_row.numel())
        plan = _route_plan(expert_of_row, lo, hi, K, E)
        route_chunk = chunk_size(
            plan.routes.numel(),
            per_route,
            x_flat.device,
            fixed_bytes=out.numel() * out.element_size() + _plan_bytes(plan),
            budget_bytes=budget,
        )
        for rlo in range(0, plan.routes.numel(), route_chunk):
            rhi = min(rlo + route_chunk, plan.routes.numel())
            subplan = _route_subplan(plan, rlo, rhi, expert_of_row, K, E)
            sidx, tok_s, ends = (
                subplan.routes,
                subplan.tokens,
                subplan.real_ends,
            )
            x_s = x_flat[tok_s]
            gate_up = _grouped_mm(x_s, W1.mT, ends)
            g, u = gate_up[:, :I], gate_up[:, I:]
            h = F.silu(g.float()).to(dt) * u
            y = _grouped_mm(h, W2.mT, ends)
            yw = (y * tw_row[sidx].unsqueeze(-1)).float()
            out.index_add_(0, tok_s, yw)
    return out.to(dt)


def _fused_moe_backward(
    grad_flat,
    x_flat,
    W1,
    W2,
    real_eor,
    tw_row,
    K,
    n_groups,
    tokens_per_sample=None,
    compute_x_grad=True,
    compute_route_grad=True,
    compute_gate_wgrad=True,
    compute_down_wgrad=True,
    wgrad_out=None,
):
    """Manual grouped backward with planner-bounded routing and activations."""
    N, H = x_flat.shape
    I = W1.shape[1] // 2
    dt = x_flat.dtype
    E = W1.shape[0]
    dx = (
        torch.zeros(N, H, dtype=torch.float32, device=x_flat.device)
        if compute_x_grad
        else None
    )
    dtw = (
        torch.zeros(N * K, dtype=torch.float32, device=x_flat.device)
        if compute_route_grad
        else None
    )
    if wgrad_out is None:
        dW1 = W1.new_zeros(n_groups, 2 * I, H) if compute_gate_wgrad else None
        dW2 = W2.new_zeros(n_groups, H, I) if compute_down_wgrad else None
    else:
        dW1, dW2 = wgrad_out
        if dW1 is not None:
            dW1.zero_()
        if dW2 is not None:
            dW2.zero_()
    compute_wgrad = compute_gate_wgrad or compute_down_wgrad

    per_route = grouped_backward_bytes_per_route(
        H, I, x_flat.element_size(), backend_multiplier=1
    )
    budget = _workspace_budget_bytes(x_flat.device)
    metadata_per_route = 32 if tokens_per_sample is not None and n_groups != E else 24
    atomic_accumulator = (
        n_groups * 16 * 4 * int(compute_gate_wgrad + compute_down_wgrad)
    )
    plan_estimate = real_eor.numel() * metadata_per_route + (E + n_groups) * 4
    gate_grad_bytes = n_groups * 2 * I * H * 4 if compute_gate_wgrad else 0
    down_grad_bytes = n_groups * H * I * 4 if compute_down_wgrad else 0
    full_grad_bytes = gate_grad_bytes + down_grad_bytes
    inline_workspace = full_grad_bytes + max(gate_grad_bytes, down_grad_bytes)
    inline_wgrad = (
        compute_wgrad and plan_estimate + per_route + inline_workspace <= budget
    )
    cache_plan = atomic_accumulator + plan_estimate <= budget
    plan_chunk = (
        max(1, real_eor.numel())
        if cache_plan
        else chunk_size(
            real_eor.numel(),
            metadata_per_route,
            x_flat.device,
            fixed_bytes=atomic_accumulator,
            budget_bytes=budget,
        )
    )
    cached_plan = (
        _route_plan(
            real_eor,
            0,
            real_eor.numel(),
            K,
            E,
            tokens_per_sample=tokens_per_sample,
            n_groups=n_groups,
        )
        if cache_plan and real_eor.numel() > 0
        else None
    )
    gate_acc = (
        torch.zeros_like(dW1, dtype=torch.float32)
        if inline_wgrad and dW1 is not None
        else None
    )
    down_acc = (
        torch.zeros_like(dW2, dtype=torch.float32)
        if inline_wgrad and dW2 is not None
        else None
    )
    run_main = compute_route_grad or compute_x_grad or inline_wgrad

    if run_main:
        for plan in _iter_route_plans(
            cached_plan,
            real_eor,
            plan_chunk,
            K,
            E,
            tokens_per_sample,
            n_groups,
        ):
            route_chunk = chunk_size(
                plan.routes.numel(),
                per_route,
                x_flat.device,
                fixed_bytes=_plan_bytes(plan)
                + (inline_workspace if inline_wgrad else 0),
                budget_bytes=budget,
            )
            for rlo in range(0, plan.routes.numel(), route_chunk):
                rhi = min(rlo + route_chunk, plan.routes.numel())
                subplan = _route_subplan(
                    plan,
                    rlo,
                    rhi,
                    real_eor,
                    K,
                    E,
                    tokens_per_sample=tokens_per_sample,
                    n_groups=n_groups,
                )
                sidx, tok_s, ends = (
                    subplan.routes,
                    subplan.tokens,
                    subplan.real_ends,
                )
                x_s = x_flat[tok_s]
                gate_up = _grouped_mm(x_s, W1.mT, ends)
                g, u = gate_up[:, :I], gate_up[:, I:]
                sig = torch.sigmoid(g.float())
                silu = (g.float() * sig).to(dt)
                h = silu * u
                go_s = grad_flat[tok_s]
                tw_s = tw_row[sidx]
                if compute_route_grad:
                    y = _grouped_mm(h, W2.mT, ends)
                    dtw[sidx] = (go_s.float() * y.float()).sum(-1)
                if compute_x_grad or inline_wgrad:
                    dy = (tw_s.unsqueeze(-1) * go_s).to(dt)
                    if down_acc is not None:
                        _accumulate_grouped_AtB(
                            dy,
                            h,
                            subplan,
                            down_acc,
                            full_grad_bytes,
                            grouped_atb=_grouped_AtB,
                        )
                    if compute_x_grad or gate_acc is not None:
                        dh = _grouped_mm(dy, W2, ends)
                        dsilu = (sig * (1.0 + g.float() * (1.0 - sig))).to(dt)
                        dgu = torch.cat([dh * u * dsilu, dh * silu], dim=-1)
                        if compute_x_grad:
                            dx_s = _grouped_mm(dgu, W1, ends)
                            dx.index_add_(0, tok_s, dx_s.float())
                        if gate_acc is not None:
                            _accumulate_grouped_AtB(
                                dgu,
                                x_s,
                                subplan,
                                gate_acc,
                                full_grad_bytes,
                                grouped_atb=_grouped_AtB,
                            )

    dx = None if dx is None else dx.to(dt)
    dtw = None if dtw is None else dtw.reshape(N, K).to(dt)
    if not compute_wgrad:
        return dx, None, None, dtw
    if inline_wgrad:
        if dW1 is not None:
            dW1.copy_(gate_acc.to(W1.dtype))
        if dW2 is not None:
            dW2.copy_(down_acc.to(W2.dtype))
        return dx, dW1, dW2, dtw
    if run_main and real_eor.numel() > 0:
        del plan, subplan, sidx, tok_s, ends, x_s, gate_up, g, u, sig, silu
        del h, go_s, tw_s
        if compute_route_grad:
            del y
        if compute_x_grad:
            del dy, dh, dsilu, dgu, dx_s
    _stream_grouped_weight_grads(
        x_flat,
        grad_flat,
        (W1, W2),
        real_eor,
        tw_row,
        K,
        tokens_per_sample,
        n_groups,
        (dW1, dW2),
        _WeightGradPlan(per_route, budget, cached_plan, plan_chunk),
        grouped_atb=_grouped_AtB,
    )
    return dx, dW1, dW2, dtw


# ---------------------------------------------------------------------------
# Autograd Functions (two-Function pattern for vmap(grad) — see swiglu.py)
# ---------------------------------------------------------------------------


class _FusedMoEBackward(torch.autograd.Function):
    """Backward as an autograd.Function so ``vmap(grad)`` routes here (DP-SGD).
    No double backward."""

    @staticmethod
    def forward(
        grad_out,
        x,
        gate_up_proj,
        down_proj,
        top_k_index,
        top_k_weights,
        compute_x_grad,
        compute_route_grad,
        compute_gate_wgrad,
        compute_down_wgrad,
    ):
        K = top_k_index.shape[-1]
        E = gate_up_proj.shape[0]
        eor = top_k_index.reshape(-1)
        # Summed weight grads: group == real expert.
        dx, dW1, dW2, dtw = _fused_moe_backward(
            grad_out,
            x,
            gate_up_proj,
            down_proj,
            eor,
            top_k_weights.reshape(-1),
            K,
            n_groups=E,
            compute_x_grad=compute_x_grad,
            compute_route_grad=compute_route_grad,
            compute_gate_wgrad=compute_gate_wgrad,
            compute_down_wgrad=compute_down_wgrad,
        )
        return dx, dW1, dW2, dtw

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Double backward not supported for fused MoE")

    @staticmethod
    def vmap(
        info,
        in_dims,
        grad_out,
        x,
        gate_up_proj,
        down_proj,
        top_k_index,
        top_k_weights,
        compute_x_grad,
        compute_route_grad,
        compute_gate_wgrad,
        compute_down_wgrad,
    ):
        # DP-SGD contract: grad_out/x/top_k batched at 0; shared weights unbatched.
        B, T, H = x.shape
        K = top_k_index.shape[-1]
        E = gate_up_proj.shape[0]
        I = gate_up_proj.shape[1] // 2
        dx = torch.empty_like(x) if compute_x_grad else None
        dtw = torch.empty_like(top_k_weights) if compute_route_grad else None
        dW1 = gate_up_proj.new_empty(B, E, 2 * I, H) if compute_gate_wgrad else None
        dW2 = down_proj.new_empty(B, E, H, I) if compute_down_wgrad else None
        compute_wgrad = compute_gate_wgrad or compute_down_wgrad
        per_example = (
            T
            * K
            * grouped_backward_bytes_per_route(
                H, I, x.element_size(), backend_multiplier=1
            )
        )
        example_chunk = chunk_size(B, per_example, x.device)

        for blo in range(0, B, example_chunk):
            bhi = min(blo + example_chunk, B)
            current_batch = bhi - blo
            N = current_batch * T
            xf = x[blo:bhi].reshape(N, H)
            gf = grad_out[blo:bhi].reshape(N, H)
            tif = top_k_index[blo:bhi].reshape(N, K)
            twf = top_k_weights[blo:bhi].reshape(N, K)
            eor = tif.reshape(-1)
            wgrad_out = (
                None
                if not compute_wgrad
                else (
                    (
                        dW1[blo:bhi].reshape(current_batch * E, 2 * I, H)
                        if compute_gate_wgrad
                        else None
                    ),
                    (
                        dW2[blo:bhi].reshape(current_batch * E, H, I)
                        if compute_down_wgrad
                        else None
                    ),
                )
            )
            dx_chunk, _, _, dtw_chunk = _fused_moe_backward(
                gf,
                xf,
                gate_up_proj,
                down_proj,
                eor,
                twf.reshape(-1),
                K,
                n_groups=current_batch * E,
                tokens_per_sample=T,
                compute_x_grad=compute_x_grad,
                compute_route_grad=compute_route_grad,
                compute_gate_wgrad=compute_gate_wgrad,
                compute_down_wgrad=compute_down_wgrad,
                wgrad_out=wgrad_out,
            )
            if compute_x_grad:
                dx[blo:bhi] = dx_chunk.reshape(current_batch, T, H)
            if compute_route_grad:
                dtw[blo:bhi] = dtw_chunk.reshape(current_batch, T, K)

        x_result = dx if compute_x_grad else None
        route_result = dtw if compute_route_grad else None
        gate_result = dW1 if compute_gate_wgrad else None
        down_result = dW2 if compute_down_wgrad else None
        return (
            (x_result, gate_result, down_result, route_result),
            (
                0 if compute_x_grad else None,
                0 if compute_gate_wgrad else None,
                0 if compute_down_wgrad else None,
                0 if compute_route_grad else None,
            ),
        )


class Opaque_FusedMoE(torch.autograd.Function):
    """Sparse grouped-GEMM MoE expert FFN with vmap-grad support (DP-SGD). Same
    ``.apply`` signature as the dense ``Opaque_MoE`` and Liger's fused op."""

    @staticmethod
    def forward(x, gate_up_proj, down_proj, top_k_index, top_k_weights):
        x, gate_up_proj, down_proj, top_k_weights = cast_to_dtype(
            active_cuda_dtype(x), x, gate_up_proj, down_proj, top_k_weights
        )
        K = top_k_index.shape[-1]
        return _fused_moe_forward(
            x,
            gate_up_proj,
            down_proj,
            top_k_index.reshape(-1),
            top_k_weights.reshape(-1),
            K,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)
        ctx.compute_dtype = output.dtype

    @staticmethod
    def backward(ctx, grad_out):
        # needs_input_grad: (x, gate_up_proj, down_proj, top_k_index, top_k_weights).
        compute_x_grad = ctx.needs_input_grad[0]
        compute_gate_wgrad = ctx.needs_input_grad[1]
        compute_down_wgrad = ctx.needs_input_grad[2]
        compute_route_grad = ctx.needs_input_grad[4]
        x, gate_up_proj, down_proj, top_k_index, top_k_weights = ctx.saved_tensors
        x, gate_up_proj, down_proj, top_k_weights = cast_to_dtype(
            ctx.compute_dtype, x, gate_up_proj, down_proj, top_k_weights
        )
        dx, dW1, dW2, dtw = _FusedMoEBackward.apply(
            grad_out,
            x,
            gate_up_proj,
            down_proj,
            top_k_index,
            top_k_weights,
            compute_x_grad,
            compute_route_grad,
            compute_gate_wgrad,
            compute_down_wgrad,
        )
        # inputs: x, gate_up_proj, down_proj, top_k_index (int, no grad), top_k_weights
        return dx, dW1, dW2, None, dtw

    @staticmethod
    def vmap(info, in_dims, x, gate_up_proj, down_proj, top_k_index, top_k_weights):
        # Forward is token-independent: merge the vmap batch into the token dim.
        x, gate_up_proj, down_proj, top_k_weights = cast_to_dtype(
            active_cuda_dtype(x), x, gate_up_proj, down_proj, top_k_weights
        )
        B, T, H = x.shape
        K = top_k_index.shape[-1]
        N = B * T
        xf = x.reshape(N, H)
        tif = top_k_index.reshape(N, K)
        twf = top_k_weights.reshape(N, K)
        out = _fused_moe_forward(
            xf,
            gate_up_proj,
            down_proj,
            tif.reshape(-1),
            twf.reshape(-1),
            K,
        )
        return out.reshape(B, T, H), 0


def opaque_fused_moe(x, gate_up_proj, down_proj, top_k_index, top_k_weights):
    """Sparse grouped-GEMM MoE expert FFN (CUDA bf16/fp16). Autograd +
    ``vmap(grad)`` (DP-SGD) flow through the two-Function pair above."""
    x, top_k_weights = follow_autocast(x, top_k_weights)
    return Opaque_FusedMoE.apply(x, gate_up_proj, down_proj, top_k_index, top_k_weights)
