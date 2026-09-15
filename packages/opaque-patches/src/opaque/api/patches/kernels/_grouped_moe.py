# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Sparse grouped-GEMM MoE expert FFN — non-Triton (MPS/CPU) perf path.

The pure-PyTorch sibling of ``fused_moe.py``: same sparse O(T*K) strategy and the
same Function/vmap contract as the dense :class:`~opaque.api.patches.kernels.moe.Opaque_MoE`
baseline, but built on ``torch._grouped_mm`` instead of Triton so it runs on
Apple MPS (and CPU). ``opaque_moe`` dispatches here on non-CUDA hosts when
``torch._grouped_mm`` is available, avoiding the dense path's O(T*E) blowup
(every token through every expert).

Tokens are sorted by expert and run through ``torch._grouped_mm`` for the three
mode-1 (offset-grouped along the token dim) GEMMs — forward up-proj, forward
down-proj, backward ``dx``. The mode-2 per-group weight grads (``dW1``/``dW2``)
are ``out[g] = A_g^T @ B_g`` (contraction grouped along the *token* axis).
``torch._grouped_mm``'s 2D×2D layout expresses this — its 16-byte rule is on
matrix *strides*, not group sizes — and :func:`_grouped_AtB` submits every
group in one backend launch without copying offsets to the host. Bounded output
tiles accumulate in fp32 and are cast once into the final gradients.

Per-sample weight grads under ``vmap(grad)`` (DP-SGD) use **virtual experts**:
sample ``b``'s tokens for real expert ``e`` go to group ``b*E + e``, so the
grouped weight-grad lands per-sample in a ``(B, E, ...)`` buffer — never summed
across the batch. The forward/dx GEMMs index the shared weights by real expert.
"""

# ``I`` is the per-expert intermediate dim throughout (paired with ``2I`` for the
# fused gate+up projection); single-letter tensor-shape names are intentional.
# ruff: noqa: E741

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ._moe_memory import (
    _workspace_budget_bytes,
    chunk_size,
    grouped_backward_bytes_per_route,
    grouped_forward_bytes_per_route,
)


def grouped_mm_available() -> bool:
    """True when ``torch._grouped_mm`` exists (the sparse path's GEMM backend)."""
    return hasattr(torch, "_grouped_mm")


# ---------------------------------------------------------------------------
# Routing + grouped-GEMM helpers
# ---------------------------------------------------------------------------


def _grouped_mm(A, Bw, ends):
    """``torch._grouped_mm`` mode-1: ``A`` (M,Kc) grouped along M by ``ends`` @
    ``Bw`` (G,Kc,Nc) -> (M,Nc). ``ends`` = int32 cumulative group ends."""
    return torch._grouped_mm(A.contiguous(), Bw, offs=ends)


def _grouped_AtB(A, B, ends, G, *, out=None):
    """Compute every grouped ``A^T @ B`` in one backend launch."""
    result = torch._grouped_mm(A.float().mT, B.float(), offs=ends)
    if out is None:
        return result
    out.copy_(result)
    return out


def _route_sort(expert_of_row, n_groups):
    """Sort (token,k) rows by group id. Returns the sort permutation and the
    int32 cumulative-end offsets for :func:`_grouped_mm` (length ``n_groups``)."""
    sort_idx = torch.argsort(expert_of_row, stable=True)
    ends = torch.bincount(expert_of_row, minlength=n_groups).cumsum(0).to(torch.int32)
    return sort_idx, ends


@dataclass(frozen=True)
class _RoutePlan:
    """Bounded on-device ordering metadata shared by all expert matrices."""

    routes: torch.Tensor
    tokens: torch.Tensor
    real_ends: torch.Tensor
    group_order: torch.Tensor | None
    group_ends: torch.Tensor
    n_groups: int

    def grouped(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor if self.group_order is None else tensor[self.group_order]


def _route_plan(
    real_eor,
    lo,
    hi,
    K,
    E,
    *,
    tokens_per_sample=None,
    n_groups=None,
):
    """Build one stable route plan for a bounded contiguous route chunk."""
    local_sort, real_ends = _route_sort(real_eor[lo:hi], E)
    routes = lo + local_sort
    return _route_plan_from_routes(
        real_eor,
        routes,
        K,
        E,
        real_ends=real_ends,
        tokens_per_sample=tokens_per_sample,
        n_groups=n_groups,
    )


def _route_plan_from_routes(
    real_eor,
    routes,
    K,
    E,
    *,
    real_ends=None,
    tokens_per_sample=None,
    n_groups=None,
):
    """Attach group metadata to routes that are already real-expert sorted."""
    tokens = torch.div(routes, K, rounding_mode="floor")
    if real_ends is None:
        real_ends = (
            torch.bincount(real_eor[routes], minlength=E).cumsum(0).to(torch.int32)
        )
    if tokens_per_sample is None or n_groups == E:
        return _RoutePlan(
            routes,
            tokens,
            real_ends,
            None,
            real_ends,
            E,
        )

    groups = (
        torch.div(tokens, tokens_per_sample, rounding_mode="floor") * E
        + real_eor[routes]
    )
    group_order, group_ends = _route_sort(groups, n_groups)
    return _RoutePlan(
        routes,
        tokens,
        real_ends,
        group_order,
        group_ends,
        n_groups,
    )


def _route_subplan(
    plan,
    lo,
    hi,
    real_eor,
    K,
    E,
    *,
    tokens_per_sample=None,
    n_groups=None,
):
    return _route_plan_from_routes(
        real_eor,
        plan.routes[lo:hi],
        K,
        E,
        tokens_per_sample=tokens_per_sample,
        n_groups=n_groups,
    )


def _plan_bytes(plan):
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in (
            plan.routes,
            plan.tokens,
            plan.real_ends,
            plan.group_order,
            plan.group_ends,
        )
        if tensor is not None
    )


def _accumulate_grouped_AtB(
    A,
    B,
    plan,
    out,
    workspace_bytes,
    *,
    grouped_atb=_grouped_AtB,
):
    """Accumulate grouped products in bounded output tiles without route rescans."""
    A = plan.grouped(A)
    B = plan.grouped(B)
    del workspace_bytes
    out.add_(grouped_atb(A, B, plan.group_ends, plan.n_groups))


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


def _aligned_tile(value, limit, alignment=4):
    if value <= alignment:
        return value
    return min(value, max(alignment, limit - limit % alignment))


def _iter_route_plans(
    cached_plan,
    real_eor,
    plan_chunk,
    K,
    E,
    tokens_per_sample,
    n_groups,
):
    if cached_plan is not None:
        yield cached_plan
        return
    for lo in range(0, real_eor.numel(), plan_chunk):
        yield _route_plan(
            real_eor,
            lo,
            min(lo + plan_chunk, real_eor.numel()),
            K,
            E,
            tokens_per_sample=tokens_per_sample,
            n_groups=n_groups,
        )


@dataclass(frozen=True)
class _WeightGradPlan:
    per_route: int
    budget: int
    cached_route_plan: _RoutePlan | None
    plan_chunk: int


def _stream_grouped_weight_grads(
    x_flat,
    grad_flat,
    weights,
    real_eor,
    tw_row,
    K,
    tokens_per_sample,
    n_groups,
    outputs,
    config,
    *,
    grouped_atb=_grouped_AtB,
):
    """Accumulate bounded FP32 output tiles, scanning routes once per tile."""
    W1, W2 = weights
    dW1, dW2 = outputs
    per_route = config.per_route
    budget = config.budget
    cached_plan = config.cached_route_plan
    plan_chunk = config.plan_chunk
    E = W1.shape[0]
    I = W1.shape[1] // 2
    H = x_flat.shape[1]
    plan_bytes = _plan_bytes(cached_plan) if cached_plan is not None else 0
    if cached_plan is None:
        plan_bytes = (
            min(real_eor.numel(), plan_chunk)
            * (32 if tokens_per_sample is not None and n_groups != E else 24)
            + (E + n_groups) * 4
        )
    available = max(
        1,
        (budget - plan_bytes - per_route) // 2,
    )
    gate_bytes = n_groups * 2 * I * H * 4 if dW1 is not None else 0
    down_bytes = n_groups * H * I * 4 if dW2 is not None else 0

    def accumulate_tiles(
        gate_rows=None,
        gate_cols=None,
        down_rows=None,
        down_cols=None,
    ):
        gate_acc = (
            torch.zeros(
                n_groups,
                gate_rows.stop - gate_rows.start,
                gate_cols.stop - gate_cols.start,
                dtype=torch.float32,
                device=x_flat.device,
            )
            if gate_rows is not None
            else None
        )
        down_acc = (
            torch.zeros(
                n_groups,
                down_rows.stop - down_rows.start,
                down_cols.stop - down_cols.start,
                dtype=torch.float32,
                device=x_flat.device,
            )
            if down_rows is not None
            else None
        )
        acc_bytes = sum(
            acc.numel() * acc.element_size()
            for acc in (gate_acc, down_acc)
            if acc is not None
        )
        for plan in _iter_route_plans(
            cached_plan,
            real_eor,
            plan_chunk,
            K,
            E,
            tokens_per_sample,
            n_groups,
        ):
            current_plan_bytes = _plan_bytes(plan)
            route_chunk = chunk_size(
                plan.routes.numel(),
                per_route,
                x_flat.device,
                fixed_bytes=current_plan_bytes + 2 * acc_bytes,
                budget_bytes=budget,
            )
            for rlo in range(0, plan.routes.numel(), route_chunk):
                subplan = _route_subplan(
                    plan,
                    rlo,
                    min(rlo + route_chunk, plan.routes.numel()),
                    real_eor,
                    K,
                    E,
                    tokens_per_sample=tokens_per_sample,
                    n_groups=n_groups,
                )
                routes, tokens, ends = (
                    subplan.routes,
                    subplan.tokens,
                    subplan.real_ends,
                )
                xx = x_flat[tokens]
                gate_up = _grouped_mm(xx, W1.mT, ends)
                gate, up = gate_up[:, :I], gate_up[:, I:]
                sig = torch.sigmoid(gate.float())
                silu = (gate.float() * sig).to(x_flat.dtype)
                hidden = silu * up
                dy = (tw_row[routes].unsqueeze(-1) * grad_flat[tokens]).to(x_flat.dtype)
                if down_acc is not None:
                    _accumulate_grouped_AtB(
                        dy[:, down_rows],
                        hidden[:, down_cols],
                        subplan,
                        down_acc,
                        acc_bytes,
                        grouped_atb=grouped_atb,
                    )
                if gate_acc is not None:
                    dh = _grouped_mm(dy, W2, ends)
                    dsilu = (sig * (1.0 + gate.float() * (1.0 - sig))).to(x_flat.dtype)
                    dgu = torch.cat([dh * up * dsilu, dh * silu], dim=-1)
                    _accumulate_grouped_AtB(
                        dgu[:, gate_rows],
                        xx[:, gate_cols],
                        subplan,
                        gate_acc,
                        acc_bytes,
                        grouped_atb=grouped_atb,
                    )
        if gate_acc is not None:
            dW1[:, gate_rows, gate_cols].copy_(gate_acc.to(W1.dtype))
        if down_acc is not None:
            dW2[:, down_rows, down_cols].copy_(down_acc.to(W2.dtype))

    if gate_bytes + down_bytes <= available:
        accumulate_tiles(
            slice(0, 2 * I) if dW1 is not None else None,
            slice(0, H) if dW1 is not None else None,
            slice(0, H) if dW2 is not None else None,
            slice(0, I) if dW2 is not None else None,
        )
        return

    for out, P, Q, is_gate in (
        (dW1, 2 * I, H, True),
        (dW2, H, I, False),
    ):
        if out is None:
            continue
        max_elements = max(16 * n_groups, available // 4)
        q_rows = _aligned_tile(Q, max_elements // (n_groups * 4))
        p_rows = _aligned_tile(P, max_elements // (n_groups * q_rows))
        for p_lo in range(0, P, p_rows):
            p_slice = slice(p_lo, min(p_lo + p_rows, P))
            for q_lo in range(0, Q, q_rows):
                q_slice = slice(q_lo, min(q_lo + q_rows, Q))
                if is_gate:
                    accumulate_tiles(p_slice, q_slice, None, None)
                else:
                    accumulate_tiles(None, None, p_slice, q_slice)


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

    per_route = grouped_backward_bytes_per_route(H, I, x_flat.element_size())
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
                            dy, h, subplan, down_acc, full_grad_bytes
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
                                dgu, x_s, subplan, gate_acc, full_grad_bytes
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
    )
    return dx, dW1, dW2, dtw


# ---------------------------------------------------------------------------
# Autograd Functions (two-Function pattern for vmap(grad) — see moe.py)
# ---------------------------------------------------------------------------


class _GroupedMoEBackward(torch.autograd.Function):
    """Backward as an autograd.Function so ``vmap(grad)`` routes here (DP-SGD)."""

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
        raise NotImplementedError("Double backward not supported for grouped MoE")

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
        B, T, H = x.shape
        K = top_k_index.shape[-1]
        E = gate_up_proj.shape[0]
        I = gate_up_proj.shape[1] // 2
        dx = torch.empty_like(x) if compute_x_grad else None
        dtw = torch.empty_like(top_k_weights) if compute_route_grad else None
        dW1 = gate_up_proj.new_empty(B, E, 2 * I, H) if compute_gate_wgrad else None
        dW2 = down_proj.new_empty(B, E, H, I) if compute_down_wgrad else None
        compute_wgrad = compute_gate_wgrad or compute_down_wgrad
        per_example = T * K * grouped_backward_bytes_per_route(H, I, x.element_size())
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


class Opaque_GroupedMoE(torch.autograd.Function):
    """Sparse grouped-GEMM MoE expert FFN (non-Triton). Same ``.apply`` signature
    as the dense ``Opaque_MoE`` and the Triton ``Opaque_FusedMoE``."""

    @staticmethod
    def forward(x, gate_up_proj, down_proj, top_k_index, top_k_weights):
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

    @staticmethod
    def backward(ctx, grad_out):
        # needs_input_grad: (x, gate_up_proj, down_proj, top_k_index, top_k_weights).
        compute_x_grad = ctx.needs_input_grad[0]
        compute_gate_wgrad = ctx.needs_input_grad[1]
        compute_down_wgrad = ctx.needs_input_grad[2]
        compute_route_grad = ctx.needs_input_grad[4]
        dx, dW1, dW2, dtw = _GroupedMoEBackward.apply(
            grad_out,
            *ctx.saved_tensors,
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
