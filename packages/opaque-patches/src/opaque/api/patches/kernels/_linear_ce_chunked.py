# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Portable memory-efficient linear cross-entropy (no Triton).

A pure-PyTorch, chunked, custom-autograd replacement for the Triton
``Opaque_LinearCrossEntropyLoss``. It never materializes the full
``(tokens, vocab)`` logit matrix — the forward streams an online log-sum-exp
over two-dimensional token/vocabulary tiles and the backward recomputes each
tile — so peak temporary memory on MPS/CPU (where Triton is unavailable) stays
within a private device-aware workspace budget. Linear projections retain the
input precision used by eager ``matmul`` while LSE and probability arithmetic
run in FP32.

Composes with ``vmap(grad(...))`` via ``generate_vmap_rule`` so the DP-SGD
per-example path works identically to the Triton kernel, and supports the same
feature surface: ``logit_softcapping`` (Gemma2), HF-style ``label_smoothing``,
and ``use_token_scaling`` (DFT). Grad-w.r.t.-weight is emitted only when the
weight requires grad (``needs_input_grad[1]``), so a frozen LoRA head skips it.

The autograd.Function operates on the already-shifted, flattened ``e`` /
``targets`` and returns the per-token loss; the label shift, ignore masking, and
reduction live in the wrappers (regular autograd). Keeping them outside the
Function is what preserves the memory win under ``functorch`` ``vmap(grad)`` —
folding the shift/sum into ``forward`` makes the transform retain the per-chunk
forward activations and defeats the streaming.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_MIB = 1024**2
_CPU_WORKSPACE_BYTES = 512 * _MIB
_MPS_FALLBACK_WORKSPACE_BYTES = 512 * _MIB
_MPS_MAX_WORKSPACE_BYTES = 1024 * _MIB
_MPS_FREE_MEMORY_FRACTION = 0.125
_CHUNK_VOCAB = 16384
_TILE_LIVE_VALUES = 6
_INVALID_CHUNK_VOCAB = "chunk_vocab must be positive"


@dataclass(frozen=True)
class _TilePlan:
    tokens: int
    vocab: int
    estimated_bytes: int
    budget_bytes: int
    batch_factor: int


def _workspace_budget_bytes(device: torch.device) -> int:
    """Return a conservative, hard-capped temporary-workspace budget."""
    if device.type != "mps":
        return _CPU_WORKSPACE_BYTES

    try:
        free_bytes = max(
            0,
            int(torch.mps.recommended_max_memory())
            - int(torch.mps.driver_allocated_memory()),
        )
    except (AttributeError, RuntimeError, TypeError):
        return _MPS_FALLBACK_WORKSPACE_BYTES
    return max(
        1,
        min(_MPS_MAX_WORKSPACE_BYTES, int(free_bytes * _MPS_FREE_MEMORY_FRACTION)),
    )


def _estimate_tile_bytes(
    tokens: int,
    vocab: int,
    hidden: int,
    itemsize: int,
    *,
    cast_hidden: bool,
    cast_weight: bool,
) -> int:
    """Conservatively estimate avoidable live storage for one 2-D tile."""
    tile = tokens * vocab * itemsize * _TILE_LIVE_VALUES
    hidden_work = tokens * hidden * itemsize * (1 + int(cast_hidden))
    weight_work = vocab * hidden * itemsize * int(cast_weight)
    return tile + hidden_work + weight_work


def _vmap_batch_factor(tensor: torch.Tensor) -> int:
    """Return the product of hidden physical vmap dimensions."""
    functorch = getattr(torch._C, "_functorch", None)
    if functorch is None:
        return 1
    factor = 1
    current = tensor
    while True:
        if functorch.is_gradtrackingtensor(current):
            current = functorch.get_unwrapped(current)
        elif functorch.is_batchedtensor(current):
            batch_dim = functorch.maybe_get_bdim(current)
            current = functorch.get_unwrapped(current)
            factor *= current.shape[batch_dim]
        else:
            return factor


def _tile_plan(
    e: torch.Tensor,
    weight: torch.Tensor,
    chunk_vocab: int | None = None,
    *,
    budget_bytes: int | None = None,
) -> _TilePlan:
    """Choose token/vocabulary tiles that fit the private workspace budget."""
    if chunk_vocab is not None and chunk_vocab <= 0:
        raise ValueError(_INVALID_CHUNK_VOCAB)
    tokens = e.shape[0]
    vocab, hidden = weight.shape
    vocab_cap = min(vocab, _CHUNK_VOCAB if chunk_vocab is None else chunk_vocab)
    budget = _workspace_budget_bytes(e.device) if budget_bytes is None else budget_bytes
    batch_factor = _vmap_batch_factor(e)
    tile_budget = max(1, budget // batch_factor)
    compute_dtype = _compute_dtype(e, weight)
    itemsize = torch.empty((), dtype=compute_dtype).element_size()
    mixed_dtypes = e.dtype != weight.dtype
    cast_hidden = mixed_dtypes and e.dtype != compute_dtype
    cast_weight = mixed_dtypes and weight.dtype != compute_dtype

    vocab_tile = max(1, vocab_cap)
    while (
        vocab_tile > 1
        and _estimate_tile_bytes(
            1,
            vocab_tile,
            hidden,
            itemsize,
            cast_hidden=cast_hidden,
            cast_weight=cast_weight,
        )
        > tile_budget
    ):
        vocab_tile = max(1, vocab_tile // 2)

    fixed_per_token = hidden * itemsize * (1 + int(cast_hidden))
    bytes_per_token = vocab_tile * itemsize * _TILE_LIVE_VALUES + fixed_per_token
    fixed_weight = vocab_tile * hidden * itemsize * int(cast_weight)
    available = max(bytes_per_token, tile_budget - fixed_weight)
    token_tile = max(1, available // max(bytes_per_token, 1))
    token_tile = min(max(tokens, 1), token_tile)
    estimated = batch_factor * _estimate_tile_bytes(
        token_tile,
        vocab_tile,
        hidden,
        itemsize,
        cast_hidden=cast_hidden,
        cast_weight=cast_weight,
    )
    return _TilePlan(token_tile, vocab_tile, estimated, budget, batch_factor)


def _softcap(logits: torch.Tensor, softcap: float | None) -> torch.Tensor:
    if softcap is None:
        return logits
    return softcap * torch.tanh(logits / softcap)


def _compute_dtype(e: torch.Tensor, weight: torch.Tensor) -> torch.dtype:
    """Stream in at least fp32: upcast bf16/fp16 inputs, preserve fp32/fp64."""
    return torch.promote_types(
        torch.promote_types(e.dtype, weight.dtype), torch.float32
    )


def _linear_chunk(e: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Match eager linear precision, then promote logits for CE statistics."""
    compute_dtype = _compute_dtype(e, weight)
    if e.dtype != weight.dtype:
        return e.to(compute_dtype) @ weight.to(compute_dtype).t()
    return (e @ weight.t()).to(compute_dtype)


def _stream_lse(
    e,
    weight,
    targets,
    softcap,
    token_tile,
    vocab_tile,
    logit_scale,
    need_logit_target=True,
    need_sum_logits=False,
):
    """One streamed pass over token and vocabulary tiles.

    Returns ``(lse, logit_target, sum_logits)`` each shaped ``(N,)``. Tile
    results are concatenated instead of assigned into a shared buffer so the
    generated vmap rule can add an outer per-example dimension safely.
    """
    N = e.shape[0]
    V = weight.shape[0]
    cdt = _compute_dtype(e, weight)
    if N == 0:
        empty = torch.empty(0, dtype=cdt, device=e.device)
        return (
            empty,
            empty if need_logit_target else None,
            empty if need_sum_logits else None,
        )

    lse_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] | None = [] if need_logit_target else None
    sum_parts: list[torch.Tensor] | None = [] if need_sum_logits else None
    zero = torch.zeros((), dtype=cdt, device=e.device)
    cast_hidden = e.dtype != weight.dtype and e.dtype != cdt

    for blo in range(0, N, token_tile):
        bhi = min(blo + token_tile, N)
        et = e[blo:bhi]
        tt = targets[blo:bhi]
        ef = et.to(cdt) if cast_hidden else et
        rows = bhi - blo
        m = torch.full((rows,), float("-inf"), dtype=cdt, device=e.device)
        s = torch.zeros(rows, dtype=cdt, device=e.device)
        logit_target = (
            torch.zeros(rows, dtype=cdt, device=e.device) if need_logit_target else None
        )
        sum_logits = (
            torch.zeros(rows, dtype=cdt, device=e.device) if need_sum_logits else None
        )

        for lo in range(0, V, vocab_tile):
            hi = min(lo + vocab_tile, V)
            lc = _softcap(_linear_chunk(ef, weight[lo:hi]) * logit_scale, softcap)
            cmax = torch.maximum(m, lc.max(-1).values)
            s = s * torch.exp(m - cmax) + torch.exp(lc - cmax[:, None]).sum(-1)
            m = cmax
            if need_sum_logits:
                sum_logits = sum_logits + lc.sum(-1)
            if need_logit_target:
                sel = (tt >= lo) & (tt < hi)
                idx = (tt - lo).clamp(0, hi - lo - 1)
                logit_target = logit_target + torch.where(
                    sel, lc.gather(1, idx[:, None]).squeeze(1), zero
                )

        lse_parts.append(m + torch.log(s))
        if need_logit_target:
            target_parts.append(logit_target)
        if need_sum_logits:
            sum_parts.append(sum_logits)

    return (
        torch.cat(lse_parts),
        torch.cat(target_parts) if need_logit_target else None,
        torch.cat(sum_parts) if need_sum_logits else None,
    )


def _per_token_loss(lse, logit_target, sum_logits, vocab, label_smoothing):
    """Per-token CE from the streamed stats (HF ``cross_entropy`` semantics)."""
    nll = lse - logit_target
    if label_smoothing:
        eps = label_smoothing
        # (1-eps)*nll + eps*(lse - mean_v logit_v)
        return (1.0 - eps) * nll + eps * (lse - sum_logits / vocab)
    return nll


class _ChunkedLinearCE(torch.autograd.Function):
    """Chunked linear + cross-entropy, per token.

    Operates on the pre-shifted, flattened ``e`` (N, D) and ``targets`` (N,) and
    returns the per-token loss (N,). The shift / ignore-mask / reduction live in
    the wrappers.
    """

    generate_vmap_rule = True

    # NOTE: do not override ``apply`` to project the output down to the loss.
    # ``torch.func.grad`` re-enters through
    # ``torch._functorch.autograd_function.custom_function_call_grad``, which
    # generates a single-level Function whose ``forward`` calls back into
    # ``custom_function_call`` and then hands whatever that returns to
    # ``setup_context`` as ``output``.  An ``apply`` that returns only
    # ``output[0]`` therefore makes the outer ``setup_context`` unpack the loss
    # tensor itself, which fails as soon as its length is not 3.  The
    # projection belongs at the call site, where no transform can see it.

    @staticmethod
    def forward(
        e,
        weight,
        targets,
        logit_softcapping,
        label_smoothing,
        use_token_scaling,
        logit_scale,
        token_tile,
        vocab_tile,
    ):
        softcap = logit_softcapping if logit_softcapping != 0 else None
        lse, logit_target, sum_logits = _stream_lse(
            e,
            weight,
            targets,
            softcap,
            token_tile,
            vocab_tile,
            logit_scale,
            need_sum_logits=float(label_smoothing) != 0.0,
        )
        loss = _per_token_loss(
            lse, logit_target, sum_logits, weight.shape[0], float(label_smoothing)
        )
        if use_token_scaling:
            # Detached confidence p_t = softmax(logits)[target] (DFT).
            token_weight = torch.exp(logit_target - lse).detach()
            loss = token_weight * loss
        else:
            token_weight = lse.new_empty(0)
        return loss, lse, token_weight

    @staticmethod
    def setup_context(ctx, inputs, output):
        (
            e,
            weight,
            targets,
            logit_softcapping,
            label_smoothing,
            use_token_scaling,
            logit_scale,
            token_tile,
            vocab_tile,
        ) = inputs
        _, lse, token_weight = output
        ctx.mark_non_differentiable(lse, token_weight)
        ctx.save_for_backward(e, weight, targets, lse, token_weight)
        ctx.softcap = logit_softcapping if logit_softcapping != 0 else None
        ctx.label_smoothing = float(label_smoothing)
        ctx.use_token_scaling = bool(use_token_scaling)
        ctx.logit_scale = float(logit_scale)
        ctx.token_tile = token_tile
        ctx.vocab_tile = vocab_tile

    @staticmethod
    def backward(ctx, grad_loss, _grad_lse, _grad_token_weight):
        e, weight, targets, lse, token_weight = ctx.saved_tensors
        softcap = ctx.softcap
        eps = ctx.label_smoothing
        compute_dc = ctx.needs_input_grad[1]
        N = e.shape[0]
        V = weight.shape[0]
        cdt = _compute_dtype(e, weight)
        row = grad_loss.to(cdt)
        if ctx.use_token_scaling:
            row = row * token_weight

        low_precision_linear = (
            e.dtype in {torch.float16, torch.bfloat16} and weight.dtype == e.dtype
        )
        vocab_ranges = [
            (lo, min(lo + ctx.vocab_tile, V)) for lo in range(0, V, ctx.vocab_tile)
        ]
        w_chunks: list[torch.Tensor | None] | None = (
            [None] * len(vocab_ranges) if compute_dc else None
        )
        grad_e_parts: list[torch.Tensor] = []

        for blo in range(0, N, ctx.token_tile):
            bhi = min(blo + ctx.token_tile, N)
            et = e[blo:bhi]
            tt = targets[blo:bhi]
            lt = lse[blo:bhi]
            rt = row[blo:bhi]
            ef = et if low_precision_linear else et.to(cdt)
            grad_et: torch.Tensor | None = None

            for vi, (lo, hi) in enumerate(vocab_ranges):
                wc = weight[lo:hi] if low_precision_linear else weight[lo:hi].to(cdt)
                lc = _softcap(_linear_chunk(ef, wc) * ctx.logit_scale, softcap)
                p = torch.exp(lc - lt[:, None])
                sel = (tt >= lo) & (tt < hi)
                idx = (tt - lo).clamp(0, hi - lo - 1)
                if eps:
                    p.sub_(eps / V)
                target_mass = sel[:, None].to(p.dtype) * (1.0 - eps)
                p.scatter_add_(1, idx[:, None], -target_mass)
                gl = p
                if softcap is not None:
                    gl.mul_(1.0 - (lc / softcap) ** 2)
                gl.mul_(ctx.logit_scale)
                gl.mul_(rt[:, None])
                linear_grad = gl.to(e.dtype) if low_precision_linear else gl

                if grad_et is None:
                    grad_et = (linear_grad @ wc).to(cdt)
                elif low_precision_linear:
                    grad_et.add_((linear_grad @ wc).to(cdt))
                else:
                    grad_et = torch.addmm(grad_et, linear_grad, wc)

                if compute_dc:
                    contribution = (linear_grad.t() @ ef).to(cdt)
                    previous = w_chunks[vi]
                    if previous is None:
                        w_chunks[vi] = contribution
                    elif low_precision_linear:
                        w_chunks[vi] = previous + contribution
                    else:
                        w_chunks[vi] = torch.addmm(previous, linear_grad.t(), ef)

            grad_e_parts.append(grad_et)

        if N == 0:
            grad_e = torch.zeros_like(e)
            grad_w = weight * e.sum().to(weight.dtype) if compute_dc else None
        else:
            grad_e = torch.cat(grad_e_parts).to(e.dtype)
            grad_w = torch.cat(w_chunks, dim=0).to(weight.dtype) if compute_dc else None
        return grad_e, grad_w, None, None, None, None, None, None, None


def linear_nll_sum_chunked(
    hidden_states,
    weight,
    labels,
    ignore_index=-100,
    logit_softcapping=0,
    label_smoothing=0.0,
    use_token_scaling=False,
    logit_scale=1.0,
    chunk_vocab=None,
):
    """Unreduced NLL sum over non-ignored tokens.

    Drop-in for ``Opaque_LinearCrossEntropyLoss.apply`` (same positional args,
    same return): HF-style label shift (position ``i`` predicts ``labels[i+1]``);
    the caller handles the mean reduction. Ignored positions contribute zero.
    """
    e = hidden_states[..., :-1, :].contiguous().flatten(0, -2)  # (N, D)
    targets = labels[..., 1:].contiguous().flatten()  # (N,)
    plan = _tile_plan(e, weight, chunk_vocab)
    nll, _lse, _token_weight = _ChunkedLinearCE.apply(
        e,
        weight,
        targets,
        logit_softcapping,
        label_smoothing,
        use_token_scaling,
        logit_scale,
        plan.tokens,
        plan.vocab,
    )
    valid = targets != ignore_index
    return torch.where(valid, nll, nll.new_zeros(())).sum()


def linear_cross_entropy_chunked(
    hidden_states,
    weight,
    labels,
    ignore_index=-100,
    logit_softcapping=0,
    label_smoothing=0.0,
    use_token_scaling=False,
    logit_scale=1.0,
    chunk_vocab=None,
):
    """Mean-reduced chunked linear CE — matches ``opaque_linear_cross_entropy_loss``.

    Divides the NLL sum by the count of non-ignored tokens.
    """
    nll_sum = linear_nll_sum_chunked(
        hidden_states,
        weight,
        labels,
        ignore_index,
        logit_softcapping,
        label_smoothing,
        use_token_scaling,
        logit_scale,
        chunk_vocab,
    )
    targets = labels[..., 1:].contiguous().flatten()
    n_valid = (targets != ignore_index).sum().clamp(min=1).to(nll_sum.dtype)
    return nll_sum / n_valid
