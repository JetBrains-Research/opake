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
matrix *strides*, not group sizes — but :func:`_grouped_AtB` does it with an
explicit per-group loop (G = E, or B*E for the per-sample DP path — both small)
so it stays safe inside the vmap rules. All reductions accumulate in fp32.

Per-sample weight grads under ``vmap(grad)`` (DP-SGD) use **virtual experts**:
sample ``b``'s tokens for real expert ``e`` go to group ``b*E + e``, so the
grouped weight-grad lands per-sample in a ``(B, E, ...)`` buffer — never summed
across the batch. The forward/dx GEMMs index the shared weights by real expert.
"""

# ``I`` is the per-expert intermediate dim throughout (paired with ``2I`` for the
# fused gate+up projection); single-letter tensor-shape names are intentional.
# ruff: noqa: E741

from __future__ import annotations

import torch
import torch.nn.functional as F

from ._moe_memory import (
    chunk_size,
    fp32_accumulator_rows,
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


def _grouped_AtB(A, B, seg_offs, G, *, out=None):
    """Compute grouped ``A^T @ B`` without full-size fp32 conversion copies."""
    P, Q = A.shape[1], B.shape[1]
    if out is None:
        out = torch.zeros(G, P, Q, dtype=A.dtype, device=A.device)
    bounds = seg_offs.tolist()
    for g in range(G):
        lo, hi = bounds[g], bounds[g + 1]
        if hi > lo:
            out[g] = (A[lo:hi].float().t() @ B[lo:hi].float()).to(A.dtype)
        else:
            out[g].zero_()
    return out


def _route_sort(expert_of_row, n_groups):
    """Sort (token,k) rows by group id. Returns the sort permutation and the
    int32 cumulative-end offsets for :func:`_grouped_mm` (length ``n_groups``)."""
    sort_idx = torch.argsort(expert_of_row, stable=True)
    ends = torch.bincount(expert_of_row, minlength=n_groups).cumsum(0).to(torch.int32)
    return sort_idx, ends


def _seg_offsets(group_of_row, n_groups):
    """Exclusive-prefix offsets (length ``n_groups+1``) for :func:`_grouped_AtB`."""
    counts = torch.bincount(group_of_row, minlength=n_groups)
    return torch.cat([counts.new_zeros(1), counts.cumsum(0)]).to(torch.int32)


def _fused_moe_forward(x_flat, W1, W2, expert_of_row, tw_row, K):
    """Sparse grouped MoE forward with planner-bounded routing and activations."""
    N, H = x_flat.shape
    E = W1.shape[0]
    I = W1.shape[1] // 2
    dt = x_flat.dtype
    out = torch.empty(N, H, dtype=dt, device=x_flat.device)
    per_route = grouped_forward_bytes_per_route(H, I, x_flat.element_size())
    route_chunk = chunk_size(
        expert_of_row.numel(),
        per_route,
        x_flat.device,
        fixed_bytes=out.numel() * out.element_size(),
        row_multiple=K,
    )
    route_chunk = min(route_chunk, fp32_accumulator_rows(N, H) * K)
    for lo in range(0, expert_of_row.numel(), route_chunk):
        hi = min(lo + route_chunk, expert_of_row.numel())
        local_sort, ends = _route_sort(expert_of_row[lo:hi], E)
        sidx = lo + local_sort
        tok_s = torch.div(sidx, K, rounding_mode="floor")
        token_lo, token_hi = lo // K, hi // K
        x_s = x_flat[tok_s]
        gate_up = _grouped_mm(x_s, W1.mT, ends)
        g, u = gate_up[:, :I], gate_up[:, I:]
        h = F.silu(g.float()).to(dt) * u
        y = _grouped_mm(h, W2.mT, ends)
        yw = (y * tw_row[sidx].unsqueeze(-1)).float()
        local_out = torch.zeros(
            token_hi - token_lo, H, dtype=torch.float32, device=x_flat.device
        )
        local_out.index_add_(0, tok_s - token_lo, yw)
        out[token_lo:token_hi] = local_out.to(dt)
    return out


def _route_groups(real_eor, routes, K, E, tokens_per_sample):
    experts = real_eor[routes]
    if tokens_per_sample is None:
        return experts
    tokens = torch.div(routes, K, rounding_mode="floor")
    samples = torch.div(tokens, tokens_per_sample, rounding_mode="floor")
    return samples * E + experts


def _weight_grad_terms(
    x_flat, grad_flat, W1, W2, tw_row, routes, K, expert, *, need_dgu=True
):
    """Recompute the routed terms needed by requested expert weight gradients."""
    I = W1.shape[1] // 2
    tokens = torch.div(routes, K, rounding_mode="floor")
    xx = x_flat[tokens]
    gate_up = F.linear(xx, W1[expert])
    gate, up = gate_up[:, :I], gate_up[:, I:]
    sig = torch.sigmoid(gate.float())
    silu = (gate.float() * sig).to(x_flat.dtype)
    hidden = silu * up
    go = grad_flat[tokens]
    dy = (tw_row[routes].unsqueeze(-1) * go).to(x_flat.dtype)
    if not need_dgu:
        return xx, hidden, dy, None
    dh = F.linear(dy, W2[expert].t())
    dsilu = (sig * (1.0 + gate.float() * (1.0 - sig))).to(x_flat.dtype)
    dgu = torch.cat([dh * up * dsilu, dh * silu], dim=-1)
    return xx, hidden, dy, dgu


def _group_route_chunks(
    real_eor, group, K, E, tokens_per_sample, chunk, fixed_bytes, per_route
):
    chunk = chunk_size(chunk, per_route, real_eor.device, fixed_bytes=fixed_bytes)
    for lo in range(0, real_eor.numel(), chunk):
        routes = torch.arange(
            lo, min(lo + chunk, real_eor.numel()), device=real_eor.device
        )
        groups = _route_groups(real_eor, routes, K, E, tokens_per_sample)
        yield routes[groups == group]


def _stream_grouped_weight_grads(
    x_flat,
    grad_flat,
    W1,
    W2,
    real_eor,
    tw_row,
    K,
    tokens_per_sample,
    n_groups,
    dW1,
    dW2,
    per_route,
):
    """Recompute bounded route and output tiles, one final group at a time."""
    E = W1.shape[0]
    I = W1.shape[1] // 2
    H = x_flat.shape[1]
    dW1_rows = chunk_size(2 * I, H * 4, x_flat.device) if dW1 is not None else 1
    dW2_rows = chunk_size(H, I * 4, x_flat.device) if dW2 is not None else 1
    combine = (
        dW1 is not None
        and dW2 is not None
        and chunk_size(2 * I, H * 4, x_flat.device, fixed_bytes=H * I * 4) == 2 * I
        and chunk_size(H, I * 4, x_flat.device, fixed_bytes=2 * I * H * 4) == H
    )

    for group in range(n_groups):
        expert = group % E
        if combine:
            dW1_acc = torch.zeros(2 * I, H, dtype=torch.float32, device=x_flat.device)
            dW2_acc = torch.zeros(H, I, dtype=torch.float32, device=x_flat.device)
            for routes in _group_route_chunks(
                real_eor,
                group,
                K,
                E,
                tokens_per_sample,
                real_eor.numel(),
                3 * I * H * 4,
                per_route,
            ):
                if routes.numel() == 0:
                    continue
                xx, hidden, dy, dgu = _weight_grad_terms(
                    x_flat, grad_flat, W1, W2, tw_row, routes, K, expert
                )
                dW1_acc.add_(dgu.float().t() @ xx.float())
                dW2_acc.add_(dy.float().t() @ hidden.float())
            dW1[group] = dW1_acc.to(W1.dtype)
            dW2[group] = dW2_acc.to(W2.dtype)
            continue

        if dW1 is not None:
            for out_lo in range(0, 2 * I, dW1_rows):
                out_hi = min(out_lo + dW1_rows, 2 * I)
                acc = torch.zeros(
                    out_hi - out_lo, H, dtype=torch.float32, device=x_flat.device
                )
                for routes in _group_route_chunks(
                    real_eor,
                    group,
                    K,
                    E,
                    tokens_per_sample,
                    real_eor.numel(),
                    acc.numel() * 4,
                    per_route,
                ):
                    if routes.numel() == 0:
                        continue
                    xx, _, _, dgu = _weight_grad_terms(
                        x_flat, grad_flat, W1, W2, tw_row, routes, K, expert
                    )
                    acc.add_(dgu[:, out_lo:out_hi].float().t() @ xx.float())
                dW1[group, out_lo:out_hi] = acc.to(W1.dtype)

        if dW2 is not None:
            for out_lo in range(0, H, dW2_rows):
                out_hi = min(out_lo + dW2_rows, H)
                acc = torch.zeros(
                    out_hi - out_lo, I, dtype=torch.float32, device=x_flat.device
                )
                for routes in _group_route_chunks(
                    real_eor,
                    group,
                    K,
                    E,
                    tokens_per_sample,
                    real_eor.numel(),
                    acc.numel() * 4,
                    per_route,
                ):
                    if routes.numel() == 0:
                        continue
                    _, hidden, dy, _ = _weight_grad_terms(
                        x_flat,
                        grad_flat,
                        W1,
                        W2,
                        tw_row,
                        routes,
                        K,
                        expert,
                        need_dgu=False,
                    )
                    acc.add_(dy[:, out_lo:out_hi].float().t() @ hidden.float())
                dW2[group, out_lo:out_hi] = acc.to(W2.dtype)


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
    dx = torch.empty(N, H, dtype=dt, device=x_flat.device) if compute_x_grad else None
    dtw = (
        torch.zeros(N * K, dtype=torch.float32, device=x_flat.device)
        if compute_route_grad
        else None
    )
    if wgrad_out is None:
        dW1 = W1.new_empty(n_groups, 2 * I, H) if compute_gate_wgrad else None
        dW2 = W2.new_empty(n_groups, H, I) if compute_down_wgrad else None
    else:
        dW1, dW2 = wgrad_out
    compute_wgrad = compute_gate_wgrad or compute_down_wgrad

    per_route = grouped_backward_bytes_per_route(H, I, x_flat.element_size())
    fixed_bytes = sum(
        buffer.numel() * buffer.element_size()
        for buffer in (dx, dtw)
        if buffer is not None
    )
    route_chunk = chunk_size(
        real_eor.numel(),
        per_route,
        x_flat.device,
        fixed_bytes=fixed_bytes,
        row_multiple=K,
    )
    route_chunk = min(route_chunk, fp32_accumulator_rows(N, H) * K)
    weight_grads_fit = (
        not compute_gate_wgrad
        or chunk_size(
            2 * I,
            H * 4,
            x_flat.device,
            fixed_bytes=H * I * 4 if compute_down_wgrad else 0,
        )
        == 2 * I
    ) and (
        not compute_down_wgrad
        or chunk_size(
            H,
            I * 4,
            x_flat.device,
            fixed_bytes=2 * I * H * 4 if compute_gate_wgrad else 0,
        )
        == H
    )
    fast_gate_wgrad = (
        compute_gate_wgrad and route_chunk == real_eor.numel() and weight_grads_fit
    )
    fast_down_wgrad = (
        compute_down_wgrad and route_chunk == real_eor.numel() and weight_grads_fit
    )
    need_dy = compute_x_grad or fast_gate_wgrad or fast_down_wgrad
    need_dgu = compute_x_grad or fast_gate_wgrad
    run_main = compute_route_grad or need_dy

    if run_main:
        for lo in range(0, real_eor.numel(), route_chunk):
            hi = min(lo + route_chunk, real_eor.numel())
            local_sort, ends = _route_sort(real_eor[lo:hi], E)
            sidx = lo + local_sort
            tok_s = torch.div(sidx, K, rounding_mode="floor")
            token_lo, token_hi = lo // K, hi // K
            dx_local = (
                torch.zeros(
                    token_hi - token_lo,
                    H,
                    dtype=torch.float32,
                    device=x_flat.device,
                )
                if compute_x_grad
                else None
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
            if need_dy:
                dy = (tw_s.unsqueeze(-1) * go_s).to(dt)
                if need_dgu:
                    dh = _grouped_mm(dy, W2, ends)
                    dsilu = (sig * (1.0 + g.float() * (1.0 - sig))).to(dt)
                    dgu = torch.cat([dh * u * dsilu, dh * silu], dim=-1)
                    if compute_x_grad:
                        dx_s = _grouped_mm(dgu, W1, ends)
                        dx_local.index_add_(0, tok_s - token_lo, dx_s.float())
            if compute_x_grad:
                dx[token_lo:token_hi] = dx_local.to(dt)

    dtw = None if dtw is None else dtw.reshape(N, K).to(dt)
    if not compute_wgrad:
        return dx, None, None, dtw
    if fast_gate_wgrad or fast_down_wgrad:
        group_sorted = _route_groups(real_eor, sidx, K, E, tokens_per_sample)
        vperm = torch.argsort(group_sorted, stable=True)
        seg = _seg_offsets(group_sorted, n_groups)
        if fast_gate_wgrad:
            _grouped_AtB(dgu[vperm], x_s[vperm], seg, n_groups, out=dW1)
        if fast_down_wgrad:
            _grouped_AtB(dy[vperm], h[vperm], seg, n_groups, out=dW2)
    else:
        if run_main and real_eor.numel() > 0:
            del local_sort, ends, sidx, tok_s, x_s, gate_up, g, u, sig, silu
            del h, go_s, tw_s
            if compute_route_grad:
                del y
            if need_dy:
                del dy
            if need_dgu:
                del dh, dsilu, dgu
            if compute_x_grad:
                del dx_s
            if x_flat.device.type == "cuda":
                torch.cuda.synchronize(x_flat.device)
            elif x_flat.device.type == "mps":
                torch.mps.synchronize()
        _stream_grouped_weight_grads(
            x_flat,
            grad_flat,
            W1,
            W2,
            real_eor,
            tw_row,
            K,
            tokens_per_sample,
            n_groups,
            dW1,
            dW2,
            per_route,
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
