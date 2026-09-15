# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Mixture-of-Experts expert-FFN kernel with vmap support for DP-SGD.

Signature mirrors Liger's ``LigerFusedMoEFunction.apply``. Layout matches HF v5
stacked-weight experts: ``x`` (..,H), ``gate_up_proj`` (E,2I,H), ``down_proj``
(E,H,I), ``top_k_index`` / ``top_k_weights`` (..,K) -> ``out`` (..,H).

Two ``autograd.Function``s (forward + backward), each with a ``vmap`` rule — the
Opaque pattern for ``vmap(grad)`` (DP-SGD per-sample gradients). The vmap rules
stay vectorized: per-token grads merge the vmap batch into the token dim;
per-sample expert-weight grads are written directly into their final batched
buffers (kept, NOT summed across samples). The expert loop is over the static
expert count.

This dense (every-token-through-every-expert) formulation is the correctness
baseline; a fused Triton grouped-GEMM kernel can replace the internals behind the
same Function/vmap contract for the sparse-compute speedup. (A pure-torch sparse
gather is *slower* here — per-expert gather/scatter overhead dominates — so the
sparsity win specifically needs the fused kernel.)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ._moe_expert_cast import cast_expert_banks
from ._moe_memory import (
    _workspace_budget_bytes,
    chunk_size,
    dense_backward_bytes_per_row,
    dense_routing_bytes_per_row,
    estimate_moe_workspace,
    fp32_accumulator_rows,
    use_grouped_route,
)

try:
    import triton  # noqa: F401

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


def _active_cuda_dtype(x):
    if x.is_cuda and torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype("cuda")
    return x.dtype


def _cast_to_dtype(dtype, *tensors):
    return tuple(
        tensor.to(dtype)
        if tensor.is_floating_point() and tensor.dtype != dtype
        else tensor
        for tensor in tensors
    )


def _follow_cuda_autocast(*tensors):
    """Cast caller-visible activations/router weights under CUDA autocast."""
    if not torch.is_autocast_enabled("cuda"):
        return tensors
    return _cast_to_dtype(torch.get_autocast_dtype("cuda"), *tensors)


def _expert_route(top_k_index, top_k_weights, expert, num_experts):
    """Return one expert's sparse route weights without expanding the expert axis."""
    if top_k_weights.shape[-1] == num_experts:
        return top_k_weights[..., expert : expert + 1]
    selected = top_k_index == expert
    return torch.where(selected, top_k_weights, 0).sum(dim=-1, keepdim=True)


def _moe_forward(x, gate_up_proj, down_proj, top_k_index, top_k_weights):
    """Dense-compute MoE forward with sparse, chunk-local routing weights."""
    intermediate = gate_up_proj.shape[1] // 2
    hidden = x.shape[-1]
    experts = gate_up_proj.shape[0]
    xf = x.reshape(-1, hidden)
    index = top_k_index.reshape(-1, top_k_index.shape[-1])
    weights = top_k_weights.reshape(-1, top_k_weights.shape[-1])
    out = torch.empty_like(x)
    out_flat = out.reshape(-1, hidden)
    per_row = dense_backward_bytes_per_row(
        hidden, intermediate, x.element_size()
    ) + dense_routing_bytes_per_row(top_k_index, top_k_weights, experts)
    rows = chunk_size(
        xf.shape[0],
        per_row,
        x.device,
        fixed_bytes=out.numel() * out.element_size(),
    )
    rows = min(rows, fp32_accumulator_rows(xf.shape[0], hidden))
    for lo in range(0, xf.shape[0], rows):
        hi = min(lo + rows, xf.shape[0])
        local_out = torch.zeros_like(xf[lo:hi], dtype=torch.float32)
        for e in range(experts):
            gate_up = F.linear(xf[lo:hi], gate_up_proj[e])
            h = (
                F.silu(gate_up[:, :intermediate].float()).to(x.dtype)
                * gate_up[:, intermediate:]
            )
            route = _expert_route(index[lo:hi], weights[lo:hi], e, experts)
            local_out.add_((F.linear(h, down_proj[e]) * route).float())
        out_flat[lo:hi] = local_out.to(x.dtype)
    return out


def _dense_weight_grad_terms(
    x, grad_out, route, gate_up_proj, down_proj, *, need_dgu=True
):
    """Recompute one dense expert's requested weight-gradient terms."""
    intermediate = gate_up_proj.shape[0] // 2
    gate_up = F.linear(x, gate_up_proj)
    gate, up = gate_up[:, :intermediate], gate_up[:, intermediate:]
    sig = torch.sigmoid(gate.float())
    silu = (gate.float() * sig).to(gate.dtype)
    hidden = silu * up
    dy = (route * grad_out).to(grad_out.dtype)
    if not need_dgu:
        return hidden, dy, None
    dh = F.linear(dy, down_proj.t())
    dsilu = (sig * (1.0 + gate.float() * (1.0 - sig))).to(gate.dtype)
    dgu = torch.cat([dh * up * dsilu, dh * silu], dim=-1)
    return hidden, dy, dgu


def _stream_dense_weight_grads(
    x,
    grad_out,
    top_k_index,
    top_k_weights,
    gate_up_proj,
    down_proj,
    dgate_up,
    ddown,
    per_row,
    dgate_rows,
    ddown_rows,
):
    """Accumulate requested dense expert gradients through bounded output tiles."""
    batch, tokens, hidden = x.shape
    intermediate = gate_up_proj.shape[1] // 2
    for b in range(batch):
        for expert in range(gate_up_proj.shape[0]):
            if dgate_up is not None:
                for out_lo in range(0, 2 * intermediate, dgate_rows):
                    out_hi = min(out_lo + dgate_rows, 2 * intermediate)
                    acc = torch.zeros(
                        out_hi - out_lo,
                        hidden,
                        dtype=torch.float32,
                        device=x.device,
                    )
                    token_chunk = chunk_size(
                        tokens,
                        per_row,
                        x.device,
                        fixed_bytes=acc.numel() * acc.element_size(),
                    )
                    for token_lo in range(0, tokens, token_chunk):
                        token_hi = min(token_lo + token_chunk, tokens)
                        xx = x[b, token_lo:token_hi]
                        route = _expert_route(
                            top_k_index[b, token_lo:token_hi],
                            top_k_weights[b, token_lo:token_hi],
                            expert,
                            gate_up_proj.shape[0],
                        )
                        _, _, dgu = _dense_weight_grad_terms(
                            xx,
                            grad_out[b, token_lo:token_hi],
                            route,
                            gate_up_proj[expert],
                            down_proj[expert],
                        )
                        acc.add_(dgu[:, out_lo:out_hi].float().t() @ xx.float())
                    dgate_up[b, expert, out_lo:out_hi] = acc.to(gate_up_proj.dtype)

            if ddown is not None:
                for out_lo in range(0, hidden, ddown_rows):
                    out_hi = min(out_lo + ddown_rows, hidden)
                    acc = torch.zeros(
                        out_hi - out_lo,
                        intermediate,
                        dtype=torch.float32,
                        device=x.device,
                    )
                    token_chunk = chunk_size(
                        tokens,
                        per_row,
                        x.device,
                        fixed_bytes=acc.numel() * acc.element_size(),
                    )
                    for token_lo in range(0, tokens, token_chunk):
                        token_hi = min(token_lo + token_chunk, tokens)
                        route = _expert_route(
                            top_k_index[b, token_lo:token_hi],
                            top_k_weights[b, token_lo:token_hi],
                            expert,
                            gate_up_proj.shape[0],
                        )
                        hidden_values, dy, _ = _dense_weight_grad_terms(
                            x[b, token_lo:token_hi],
                            grad_out[b, token_lo:token_hi],
                            route,
                            gate_up_proj[expert],
                            down_proj[expert],
                            need_dgu=False,
                        )
                        acc.add_(
                            dy[:, out_lo:out_hi].float().t() @ hidden_values.float()
                        )
                    ddown[b, expert, out_lo:out_hi] = acc.to(down_proj.dtype)


def _moe_backward(
    grad_out,
    x,
    gate_up_proj,
    down_proj,
    top_k_index,
    top_k_weights,
    batch_dims,
    compute_x_grad=True,
    compute_route_grad=True,
    compute_gate_wgrad=True,
    compute_down_wgrad=True,
):
    """Manual MoE backward with bounded activations and direct gradient writes."""
    intermediate = gate_up_proj.shape[1] // 2
    hidden = x.shape[-1]
    if batch_dims == 0:
        xb = x.reshape(-1, hidden).unsqueeze(0)
        gb = grad_out.reshape(-1, hidden).unsqueeze(0)
        ib = top_k_index.reshape(-1, top_k_index.shape[-1]).unsqueeze(0)
        wb = top_k_weights.reshape(-1, top_k_weights.shape[-1]).unsqueeze(0)
    else:
        batch = x.shape[0]
        xb = x.reshape(batch, -1, hidden)
        gb = grad_out.reshape(batch, -1, hidden)
        ib = top_k_index.reshape(batch, -1, top_k_index.shape[-1])
        wb = top_k_weights.reshape(batch, -1, top_k_weights.shape[-1])
    batch, tokens = xb.shape[:2]
    experts = gate_up_proj.shape[0]

    dx = (
        torch.zeros(xb.shape, dtype=torch.float32, device=x.device)
        if compute_x_grad
        else None
    )
    dtw = (
        torch.zeros(wb.shape, dtype=torch.float32, device=x.device)
        if compute_route_grad
        else None
    )
    dgate_up = (
        gate_up_proj.new_empty(batch, experts, 2 * intermediate, hidden)
        if compute_gate_wgrad
        else None
    )
    ddown = (
        down_proj.new_empty(batch, experts, hidden, intermediate)
        if compute_down_wgrad
        else None
    )

    per_row = dense_backward_bytes_per_row(
        hidden, intermediate, x.element_size()
    ) + dense_routing_bytes_per_row(top_k_index, top_k_weights, experts)
    gate_accumulator_bytes = 2 * intermediate * hidden * 4 if compute_gate_wgrad else 0
    down_accumulator_bytes = intermediate * hidden * 4 if compute_down_wgrad else 0
    dgate_rows = (
        chunk_size(2 * intermediate, hidden * 4, x.device) if compute_gate_wgrad else 1
    )
    ddown_rows = (
        chunk_size(hidden, intermediate * 4, x.device) if compute_down_wgrad else 1
    )
    weight_grads_fit = (
        not compute_gate_wgrad
        or chunk_size(
            2 * intermediate,
            hidden * 4,
            x.device,
            fixed_bytes=down_accumulator_bytes,
        )
        == 2 * intermediate
    ) and (
        not compute_down_wgrad
        or chunk_size(
            hidden,
            intermediate * 4,
            x.device,
            fixed_bytes=gate_accumulator_bytes,
        )
        == hidden
    )
    accumulator_bytes = (
        gate_accumulator_bytes + down_accumulator_bytes if weight_grads_fit else 0
    )
    fast_gate_wgrad = compute_gate_wgrad and weight_grads_fit
    fast_down_wgrad = compute_down_wgrad and weight_grads_fit
    need_dy = compute_x_grad or fast_gate_wgrad or fast_down_wgrad
    need_dgu = compute_x_grad or fast_gate_wgrad
    fixed_bytes = sum(
        buffer.numel() * buffer.element_size()
        for buffer in (dx, dtw)
        if buffer is not None
    )
    batch_chunk = chunk_size(
        batch,
        accumulator_bytes + per_row,
        x.device,
        fixed_bytes=fixed_bytes,
    )

    for blo in range(0, batch, batch_chunk):
        bhi = min(blo + batch_chunk, batch)
        current_batch = bhi - blo
        token_chunk = chunk_size(
            tokens,
            per_row * current_batch,
            x.device,
            fixed_bytes=fixed_bytes + accumulator_bytes * current_batch,
        )
        for e in range(experts):
            if compute_gate_wgrad and weight_grads_fit:
                dgate_acc = torch.zeros(
                    current_batch,
                    2 * intermediate,
                    hidden,
                    dtype=torch.float32,
                    device=x.device,
                )
            if compute_down_wgrad and weight_grads_fit:
                ddown_acc = torch.zeros(
                    current_batch,
                    hidden,
                    intermediate,
                    dtype=torch.float32,
                    device=x.device,
                )
            for tlo in range(0, tokens, token_chunk):
                thi = min(tlo + token_chunk, tokens)
                xx = xb[blo:bhi, tlo:thi]
                go = gb[blo:bhi, tlo:thi]
                index = ib[blo:bhi, tlo:thi]
                weights = wb[blo:bhi, tlo:thi]
                route = _expert_route(index, weights, e, experts)
                gate_up = F.linear(xx, gate_up_proj[e])
                gate, up = gate_up[..., :intermediate], gate_up[..., intermediate:]
                sig = torch.sigmoid(gate.float())
                silu = (gate.float() * sig).to(gate.dtype)
                h = silu * up
                if compute_route_grad:
                    y = F.linear(h, down_proj[e])
                    droute = (go.float() * y.float()).sum(dim=-1)
                    if weights.shape[-1] == experts:
                        dtw[blo:bhi, tlo:thi, e] = droute
                    else:
                        dtw[blo:bhi, tlo:thi].add_(
                            torch.where(index == e, droute.unsqueeze(-1), 0)
                        )
                if need_dy:
                    dy = (route * go).to(go.dtype)
                    if fast_down_wgrad:
                        ddown_acc.add_(
                            torch.einsum("bth,bti->bhi", dy.float(), h.float())
                        )
                    if need_dgu:
                        dh = F.linear(dy, down_proj[e].t())
                        dsilu = (sig * (1.0 + gate.float() * (1.0 - sig))).to(
                            gate.dtype
                        )
                        dgu = torch.cat([dh * up * dsilu, dh * silu], dim=-1)
                        if compute_x_grad:
                            dx[blo:bhi, tlo:thi].add_(
                                F.linear(dgu, gate_up_proj[e].t()).float()
                            )
                        if fast_gate_wgrad:
                            dgate_acc.add_(
                                torch.einsum("btj,bth->bjh", dgu.float(), xx.float())
                            )
            if compute_gate_wgrad and weight_grads_fit:
                dgate_up[blo:bhi, e] = dgate_acc.to(gate_up_proj.dtype)
            if compute_down_wgrad and weight_grads_fit:
                ddown[blo:bhi, e] = ddown_acc.to(down_proj.dtype)

    if (compute_gate_wgrad or compute_down_wgrad) and not weight_grads_fit:
        if batch > 0 and tokens > 0:
            del xx, go, index, weights, route, gate_up, gate, up, sig, silu, h
            if compute_route_grad:
                del y, droute
            if need_dy:
                del dy
            if need_dgu:
                del dh, dsilu, dgu
            if x.device.type == "cuda":
                torch.cuda.synchronize(x.device)
            elif x.device.type == "mps":
                torch.mps.synchronize()
        _stream_dense_weight_grads(
            xb,
            gb,
            ib,
            wb,
            gate_up_proj,
            down_proj,
            dgate_up,
            ddown,
            per_row,
            dgate_rows,
            ddown_rows,
        )

    if batch_dims == 0:
        return (
            None if dx is None else dx[0].to(x.dtype).reshape(x.shape),
            None if dgate_up is None else dgate_up[0],
            None if ddown is None else ddown[0],
            (
                None
                if dtw is None
                else dtw[0].to(top_k_weights.dtype).reshape(top_k_weights.shape)
            ),
        )
    return (
        None if dx is None else dx.to(x.dtype).reshape(x.shape),
        dgate_up,
        ddown,
        (
            None
            if dtw is None
            else dtw.to(top_k_weights.dtype).reshape(top_k_weights.shape)
        ),
    )


class _MoEBackward(torch.autograd.Function):
    """Backward as an autograd.Function so ``vmap(grad)`` routes here. No double backward."""

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
        return _moe_backward(
            grad_out,
            x,
            gate_up_proj,
            down_proj,
            top_k_index,
            top_k_weights,
            batch_dims=0,
            compute_x_grad=compute_x_grad,
            compute_route_grad=compute_route_grad,
            compute_gate_wgrad=compute_gate_wgrad,
            compute_down_wgrad=compute_down_wgrad,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Double backward not supported for MoE")

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
        dx, dgate_up, ddown, dtw = _moe_backward(
            grad_out,
            x,
            gate_up_proj,
            down_proj,
            top_k_index,
            top_k_weights,
            batch_dims=1,
            compute_x_grad=compute_x_grad,
            compute_route_grad=compute_route_grad,
            compute_gate_wgrad=compute_gate_wgrad,
            compute_down_wgrad=compute_down_wgrad,
        )
        x_result = dx if compute_x_grad else None
        route_result = dtw if compute_route_grad else None
        gate_result = dgate_up if compute_gate_wgrad else None
        down_result = ddown if compute_down_wgrad else None
        return (
            (x_result, gate_result, down_result, route_result),
            (
                0 if compute_x_grad else None,
                0 if compute_gate_wgrad else None,
                0 if compute_down_wgrad else None,
                0 if compute_route_grad else None,
            ),
        )


class Opaque_MoE(torch.autograd.Function):
    """MoE expert FFN with vmap-grad support (DP-SGD). Same ``.apply`` signature as
    Liger's fused op, so a Triton kernel can replace the dense internals."""

    @staticmethod
    def forward(x, gate_up_proj, down_proj, top_k_index, top_k_weights):
        compute_dtype = _active_cuda_dtype(x)
        x, top_k_weights = _cast_to_dtype(compute_dtype, x, top_k_weights)
        gate_up_proj, down_proj = cast_expert_banks(
            compute_dtype, gate_up_proj, down_proj
        )
        return _moe_forward(x, gate_up_proj, down_proj, top_k_index, top_k_weights)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)
        ctx.compute_dtype = output.dtype

    @staticmethod
    def backward(ctx, grad_out):
        compute_x_grad = ctx.needs_input_grad[0]
        compute_gate_wgrad = ctx.needs_input_grad[1]
        compute_down_wgrad = ctx.needs_input_grad[2]
        compute_route_grad = ctx.needs_input_grad[4]
        x, gate_up_proj, down_proj, top_k_index, top_k_weights = ctx.saved_tensors
        x, top_k_weights = _cast_to_dtype(ctx.compute_dtype, x, top_k_weights)
        gate_up_proj, down_proj = cast_expert_banks(
            ctx.compute_dtype, gate_up_proj, down_proj
        )
        dx, dgate_up, ddown, dtw = _MoEBackward.apply(
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
        return dx, dgate_up, ddown, None, dtw

    @staticmethod
    def vmap(info, in_dims, x, gate_up_proj, down_proj, top_k_index, top_k_weights):
        # Forward is token-independent: merge the vmap batch into the token dim.
        compute_dtype = _active_cuda_dtype(x)
        x, top_k_weights = _cast_to_dtype(compute_dtype, x, top_k_weights)
        gate_up_proj, down_proj = cast_expert_banks(
            compute_dtype, gate_up_proj, down_proj
        )
        return (
            _moe_forward(x, gate_up_proj, down_proj, top_k_index, top_k_weights),
            0,
        )


# Sparse compute pays off past a break-even expert count, but its gathered
# activations can use more memory than dense execution for large K/I/H. The
# dispatcher therefore combines this measured speed gate with the explicit
# workspace estimates below instead of routing on expert count alone.
_SPARSE_MOE_MIN_EXPERTS = 16


def opaque_moe(x, gate_up_proj, down_proj, top_k_index, top_k_weights, *, grouped=True):
    """MoE expert FFN. Autograd + ``vmap(grad)`` (DP-SGD) flow through the
    two-Function pair above.

    ``grouped`` (default ``True``) selects a **performance** grouped-GEMM path
    suited to the host; ``grouped=False`` forces the dense ``Opaque_MoE``
    **compat** path on every host. The patch layer wires ``grouped`` from the
    ``grouped_moe`` gate, so a dense run still keeps the vmap-safe experts
    ``forward`` installed for DP correctness.

    With ``grouped=True`` the dispatch is:
    - CUDA bf16/fp16 + Triton -> sparse grouped-GEMM Triton ``Opaque_FusedMoE``;
    - otherwise, when ``torch._grouped_mm`` is available, compare conservative
      dense/grouped workspace estimates against the internal device-memory budget
      and the grouped speed gate;
    - otherwise -> dense ``Opaque_MoE``.

    Every route chunks temporary activations to the same internal budget. The
    required trainable-expert per-example gradients remain unchunked outputs;
    frozen experts skip those gradients entirely.

    Both grouped-GEMM paths are performance variations — only ``Opaque_MoE`` is
    the always-correct vmap/DP-safe fallback. All are numerically equivalent
    within the documented dtype floor (see ``test_kernel_precision``).
    """
    if x.is_cuda:
        x, top_k_weights = _follow_cuda_autocast(x, top_k_weights)
    if grouped and top_k_weights.shape[-1] != gate_up_proj.shape[0]:
        if _TRITON_AVAILABLE and x.is_cuda:
            if x.dtype in (torch.bfloat16, torch.float16):
                from .fused_moe import Opaque_FusedMoE

                return Opaque_FusedMoE.apply(
                    x, gate_up_proj, down_proj, top_k_index, top_k_weights
                )
        else:
            from ._grouped_moe import Opaque_GroupedMoE, grouped_mm_available

            estimate = estimate_moe_workspace(
                x,
                gate_up_proj,
                down_proj,
                top_k_index,
                compute_gate_wgrad=gate_up_proj.requires_grad,
                compute_down_wgrad=down_proj.requires_grad,
            )
            if grouped_mm_available() and use_grouped_route(
                estimate,
                experts=gate_up_proj.shape[0],
                min_experts=_SPARSE_MOE_MIN_EXPERTS,
                budget_bytes=_workspace_budget_bytes(x.device),
            ):
                return Opaque_GroupedMoE.apply(
                    x, gate_up_proj, down_proj, top_k_index, top_k_weights
                )
    return Opaque_MoE.apply(x, gate_up_proj, down_proj, top_k_index, top_k_weights)


def torch_reference_moe(x, gate_up_proj, down_proj, top_k_index, top_k_weights):
    """Pure-PyTorch autograd-composed MoE reference (oracle for tests)."""
    return _moe_forward(x, gate_up_proj, down_proj, top_k_index, top_k_weights)
