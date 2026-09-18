# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
#
# Triton cross-entropy kernel derives from the Unsloth project
# (Apache-2.0; https://github.com/unslothai/unsloth) and has been adapted to
# Opake's vmap-friendly new-style autograd dispatch. See NOTICE in the
# repository root.
"""Cross-entropy loss kernel with vmap support for DP-SGD."""

import torch
import triton
import triton.language as tl

from opake.exceptions import ConfigurationError

from ._utils import (
    _IGNORE_INDEX,
    MAX_FUSED_SIZE,
    calculate_settings,
    ensure_cuda_tensors,
    follow_autocast,
    torch_gpu_device,
    triton_cast,
    triton_tanh,
)


@triton.jit
def _cross_entropy_forward(
    logits_ptr,
    logits_row_stride,
    loss_ptr,
    logsumexp_ptr,
    sum_logits_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DO_SOFTCAPPING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    DO_LOGIT_SCALING: tl.constexpr,
    LOGIT_SCALE: tl.constexpr,
    DO_LABEL_SMOOTHING: tl.constexpr,
):
    row_idx = tl.program_id(0)
    logits_ptr += row_idx * triton_cast(logits_row_stride, tl.int64)
    loss_ptr += row_idx
    logsumexp_ptr += row_idx
    labels_ptr += row_idx

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int32)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(
        tl.float32
    )

    # Logit scaling for Cohere: s * x
    if DO_LOGIT_SCALING:
        logits = LOGIT_SCALE * logits
    # Logit softcapping for Gemma 2: t * tanh(x / t)
    if DO_SOFTCAPPING:
        logits = SOFTCAP * triton_tanh(logits / SOFTCAP)

    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))

    if label_idx != -100:  # noqa: PLR2004 - Triton JIT requires an inline constant
        x = tl.load(logits_ptr + label_idx).to(tl.float32)
        # Apply same transforms to the label logit
        if DO_LOGIT_SCALING:
            x = LOGIT_SCALE * x
        if DO_SOFTCAPPING:
            x = SOFTCAP * triton_tanh(x / SOFTCAP)
        loss = logsumexp - x
    else:
        loss = 0.0

    tl.store(loss_ptr, loss)
    tl.store(logsumexp_ptr, logsumexp)
    if DO_LABEL_SMOOTHING:
        sum_logits_ptr += row_idx
        sum_logits = tl.sum(tl.where(mask, logits, 0.0), 0)
        tl.store(sum_logits_ptr, sum_logits)


@triton.jit
def _chunked_cross_entropy_forward(
    logits_ptr,
    logits_row_stride,
    loss_ptr,
    logsumexp_ptr,
    sum_logits_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DO_SOFTCAPPING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    DO_LOGIT_SCALING: tl.constexpr,
    LOGIT_SCALE: tl.constexpr,
    DO_LABEL_SMOOTHING: tl.constexpr,
):
    """Chunked forward for vocab > MAX_FUSED_SIZE.

    Each program handles one (row, chunk). Computes per-chunk logsumexp.
    Chunk 0 also stores -x[label] as the initial loss value.
    Python-side then does: loss = logsumexp(per_chunk_lse) + initial_loss
    """
    row_idx = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    logits_ptr += row_idx * triton_cast(logits_row_stride, tl.int64)
    loss_ptr += row_idx
    logsumexp_ptr += row_idx * N_CHUNKS + chunk_idx
    labels_ptr += row_idx

    col_offsets = chunk_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int32)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(
        tl.float32
    )

    # Logit scaling for Cohere: s * x
    if DO_LOGIT_SCALING:
        logits = LOGIT_SCALE * logits
    # Logit softcapping for Gemma 2: t * tanh(x / t)
    if DO_SOFTCAPPING:
        logits = SOFTCAP * triton_tanh(logits / SOFTCAP)

    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))

    # Chunk 0 stores the -x[label] part of the loss
    if chunk_idx == 0:
        if label_idx != -100:  # noqa: PLR2004 - Triton JIT requires an inline constant
            x = tl.load(logits_ptr + label_idx).to(tl.float32)
            if DO_LOGIT_SCALING:
                x = LOGIT_SCALE * x
            if DO_SOFTCAPPING:
                x = SOFTCAP * triton_tanh(x / SOFTCAP)
            loss = -1.0 * x
        else:
            loss = 0.0
        tl.store(loss_ptr, loss)
    tl.store(logsumexp_ptr, logsumexp)
    # Cross-chunk accumulation of sum(logits) into a per-row buffer.
    # Each chunk contributes its partial sum; the buffer is pre-zeroed
    # in the Python wrapper so atomic_add composes correctly.
    if DO_LABEL_SMOOTHING:
        chunk_sum = tl.sum(tl.where(mask, logits, 0.0), 0)
        tl.atomic_add(sum_logits_ptr + row_idx, chunk_sum)


@triton.jit
def _cross_entropy_backward(
    logits_ptr,
    logits_row_stride,
    grad_logits_ptr,
    dlosses_ptr,
    dlosses_row_stride,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DO_SOFTCAPPING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    DO_LOGIT_SCALING: tl.constexpr,
    LOGIT_SCALE: tl.constexpr,
    LABEL_SMOOTHING: tl.constexpr,
):
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    logits_ptr += row_idx * triton_cast(logits_row_stride, tl.int64)
    grad_logits_ptr += row_idx.to(tl.int64) * VOCAB_SIZE
    logsumexp_ptr += row_idx
    labels_ptr += row_idx
    dlosses_ptr += row_idx * dlosses_row_stride

    col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int32)

    dloss = (
        tl.load(dlosses_ptr)
        if label_idx != -100  # noqa: PLR2004 - Triton JIT requires an inline constant
        else 0.0
    )

    x = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)

    # Apply logit scaling for Cohere: forward was s * x
    if DO_LOGIT_SCALING:
        x = x * LOGIT_SCALE

    # Apply logit softcapping for Gemma 2: forward was t * tanh(x / t)
    # d/dx [t * tanh(x/t)] = 1 - tanh^2(x/t)
    partial = x
    if DO_SOFTCAPPING:
        partial = triton_tanh(x / SOFTCAP)
        x = SOFTCAP * partial

    logsumexp = tl.load(logsumexp_ptr)
    y = tl.exp(x - logsumexp)
    # Gradient with optional label smoothing:
    #   ∂loss/∂z_j = p_j - (1-ls) * 1[j==t] - ls/V
    # Standard CE corresponds to ls = 0:
    #   ∂loss/∂z_j = p_j - 1[j==t]
    if LABEL_SMOOTHING > 0.0:
        y = tl.where(mask, y - (LABEL_SMOOTHING / VOCAB_SIZE), y)
        y = tl.where(
            col_offsets == label_idx,
            y - (1.0 - LABEL_SMOOTHING),
            y,
        )
    else:
        y = tl.where(
            col_offsets == label_idx,
            y - 1.0,
            y,
        )

    # Chain rule for logit scaling: d/dx [s * x] = s
    if DO_LOGIT_SCALING:
        y = y * LOGIT_SCALE

    # Chain rule for softcapping: d/dx [t * tanh(x/t)] = 1 - tanh^2(x/t)
    if DO_SOFTCAPPING:
        y = y * (1.0 - partial * partial)

    tl.store(grad_logits_ptr + col_offsets, dloss * y, mask=mask)


def _ce_forward_impl(
    logits_flat,
    labels_flat,
    n_rows,
    vocab_size,
    device,
    logit_softcapping=0,
    logit_scaling=0,
    label_smoothing=0.0,
):
    """Shared forward implementation for both standard and vmap paths.

    Returns (losses, logsumexp) both of shape (n_rows,).

    Args:
        logits_flat: Flattened logits with one row per target token.
        labels_flat: Flattened target-token indices.
        n_rows: Number of flattened token rows.
        vocab_size: Number of logits in each token row.
        device: Device on which to allocate kernel intermediates.
        logit_softcapping: Gemma 2 softcap value (0 = disabled).
        logit_scaling: Cohere logit scale value (0 = disabled).
        label_smoothing: ``F.cross_entropy``-style smoothing weight in
            [0, 1].  When > 0 the per-row NLL becomes
            ``(1-ls)*hard + ls*(lse - mean(z))`` matching
            ``F.cross_entropy(..., label_smoothing=ls)``.  Zero positions
            (``label == -100``) stay at loss 0.
    """
    div, mod = divmod(vocab_size, MAX_FUSED_SIZE)
    n_chunks = div + (mod != 0)

    DO_SOFTCAPPING = logit_softcapping != 0
    DO_LOGIT_SCALING = logit_scaling != 0
    ls = float(label_smoothing)
    if not (0.0 <= ls <= 1.0):
        raise ConfigurationError(
            *(f"label_smoothing must be in [0.0, 1.0]; got {label_smoothing!r}.",)
        )
    DO_LABEL_SMOOTHING = ls > 0.0

    losses = torch.empty(n_rows, dtype=torch.float32, device=device)
    # ``sum_logits`` accumulator: per-row sum of (post-scale/softcap)
    # logits, used to compute ``mean_z`` for the smoothed-CE formula.
    # Pre-zeroed because the chunked path uses ``atomic_add``.  A
    # one-element placeholder when smoothing is off keeps the kernel
    # signature uniform.
    sum_logits = (
        torch.zeros(n_rows, dtype=torch.float32, device=device)
        if DO_LABEL_SMOOTHING
        else torch.empty(1, dtype=torch.float32, device=device)
    )

    if n_chunks == 1:
        # Small vocab (<= 65536): single-chunk path
        logsumexp = torch.empty(n_rows, dtype=torch.float32, device=device)
        BLOCK_SIZE, num_warps = calculate_settings(vocab_size)

        with torch_gpu_device(device):
            _cross_entropy_forward[(n_rows,)](
                logits_flat,
                logits_flat.stride(0),
                losses,
                logsumexp,
                sum_logits,
                labels_flat,
                VOCAB_SIZE=vocab_size,
                BLOCK_SIZE=BLOCK_SIZE,
                DO_SOFTCAPPING=DO_SOFTCAPPING,
                SOFTCAP=logit_softcapping,
                DO_LOGIT_SCALING=DO_LOGIT_SCALING,
                LOGIT_SCALE=logit_scaling,
                DO_LABEL_SMOOTHING=DO_LABEL_SMOOTHING,
                num_warps=num_warps,
            )
    else:
        # Large vocab (> 65536): chunked path
        logsumexp = torch.empty((n_rows, n_chunks), dtype=torch.float32, device=device)

        with torch_gpu_device(device):
            _chunked_cross_entropy_forward[(n_rows, n_chunks)](
                logits_flat,
                logits_flat.stride(0),
                losses,
                logsumexp,
                sum_logits,
                labels_flat,
                VOCAB_SIZE=vocab_size,
                N_CHUNKS=n_chunks,
                BLOCK_SIZE=MAX_FUSED_SIZE,
                DO_SOFTCAPPING=DO_SOFTCAPPING,
                SOFTCAP=logit_softcapping,
                DO_LOGIT_SCALING=DO_LOGIT_SCALING,
                LOGIT_SCALE=logit_scaling,
                DO_LABEL_SMOOTHING=DO_LABEL_SMOOTHING,
                num_warps=32,
            )

        # Reduce per-chunk logsumexp to global logsumexp
        logsumexp = torch.logsumexp(logsumexp, dim=1)
        # loss = -x[label] + logsumexp (chunk 0 stored -x, now add logsumexp)
        losses += logsumexp
        losses.masked_fill_(labels_flat == _IGNORE_INDEX, 0)

    if DO_LABEL_SMOOTHING:
        # Smoothed CE: (1-ls)*hard + ls*(lse - mean_z), matching
        # ``F.cross_entropy(..., label_smoothing=ls)``.  Ignored positions
        # (label == -100) keep loss = 0 — the standard kernel already
        # zeroed those, but we re-mask after the smoothing rewrite.
        mean_z = sum_logits / vocab_size
        smoothed = (1.0 - ls) * losses + ls * (logsumexp - mean_z)
        losses = torch.where(
            labels_flat == _IGNORE_INDEX,
            torch.zeros_like(smoothed),
            smoothed,
        )

    return losses, logsumexp


class _CrossEntropyBackward(torch.autograd.Function):
    """Backward pass wrapped as autograd.Function for vmap(grad()) support."""

    @staticmethod
    def forward(
        grad_losses,
        logits,
        logsumexp,
        labels,
        vocab_size,
        logit_softcapping,
        logit_scaling,
        label_smoothing,
    ):
        original_shape = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        # The kernel reads saved logits and writes directly into the required
        # gradient output, avoiding a full-tensor clone before overwriting it.
        grad_logits = torch.empty_like(logits, memory_format=torch.contiguous_format)
        grad_logits_flat = grad_logits.reshape(-1, vocab_size)
        logsumexp_flat = logsumexp.reshape(-1)
        labels_flat = labels.reshape(-1)
        grad_losses_flat = grad_losses.reshape(-1)
        n_rows = logits_flat.shape[0]

        BLOCK_SIZE = 4096
        div, mod = divmod(vocab_size, BLOCK_SIZE)
        n_blocks = div + (mod != 0)

        grid = (n_rows, n_blocks)
        with torch_gpu_device(logits.device):
            _cross_entropy_backward[grid](
                logits_flat,
                logits_flat.stride(0),
                grad_logits_flat,
                grad_losses_flat,
                grad_losses_flat.stride(0) if grad_losses_flat.dim() > 0 else 0,
                logsumexp_flat,
                labels_flat,
                VOCAB_SIZE=vocab_size,
                BLOCK_SIZE=BLOCK_SIZE,
                DO_SOFTCAPPING=logit_softcapping != 0,
                SOFTCAP=logit_softcapping,
                DO_LOGIT_SCALING=logit_scaling != 0,
                LOGIT_SCALE=logit_scaling,
                LABEL_SMOOTHING=float(label_smoothing),
                num_warps=8,
            )

        return grad_logits.reshape(original_shape)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Double backward not supported for CrossEntropy")

    @staticmethod
    def vmap(
        info,
        in_dims,
        grad_losses,
        logits,
        logsumexp,
        labels,
        vocab_size,
        logit_softcapping,
        logit_scaling,
        label_smoothing,
    ):
        (
            grad_bdim,
            logits_bdim,
            lse_bdim,
            labels_bdim,
            vs_bdim,
            sc_bdim,
            ls_bdim,
            smooth_bdim,
        ) = in_dims

        assert vs_bdim is None, "vocab_size should not be batched"
        assert sc_bdim is None, "logit_softcapping should not be batched"
        assert ls_bdim is None, "logit_scaling should not be batched"
        assert smooth_bdim is None, "label_smoothing should not be batched"
        if logits_bdim != 0 or lse_bdim != 0 or labels_bdim != 0 or grad_bdim != 0:
            # Non-leading batch dims would pair rows with the wrong labels/LSE.
            raise ConfigurationError(
                *(
                    "CrossEntropy backward vmap requires all tensors batched at "
                    f"dim 0, got in_dims={in_dims}",
                )
            )

        original_shape = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        # Merge the vmap batch only in the independent writable output; saved
        # logits remain a read-only kernel input.
        grad_logits = torch.empty_like(logits, memory_format=torch.contiguous_format)
        grad_logits_flat = grad_logits.reshape(-1, vocab_size)
        logsumexp_flat = logsumexp.reshape(-1)
        labels_flat = labels.reshape(-1)
        grad_losses_flat = grad_losses.reshape(-1)
        n_rows = logits_flat.shape[0]

        BLOCK_SIZE = 4096
        div, mod = divmod(vocab_size, BLOCK_SIZE)
        n_blocks = div + (mod != 0)

        grid = (n_rows, n_blocks)
        with torch_gpu_device(logits.device):
            _cross_entropy_backward[grid](
                logits_flat,
                logits_flat.stride(0),
                grad_logits_flat,
                grad_losses_flat,
                grad_losses_flat.stride(0) if grad_losses_flat.dim() > 0 else 0,
                logsumexp_flat,
                labels_flat,
                VOCAB_SIZE=vocab_size,
                BLOCK_SIZE=BLOCK_SIZE,
                DO_SOFTCAPPING=logit_softcapping != 0,
                SOFTCAP=logit_softcapping,
                DO_LOGIT_SCALING=logit_scaling != 0,
                LOGIT_SCALE=logit_scaling,
                LABEL_SMOOTHING=float(label_smoothing),
                num_warps=8,
            )

        return grad_logits.reshape(original_shape), logits_bdim


class Opake_CrossEntropyLoss(torch.autograd.Function):
    """Per-token cross-entropy autograd kernel with a custom vmap rule."""

    @staticmethod
    def forward(
        logits, labels, logit_softcapping=0, logit_scaling=0, label_smoothing=0.0
    ):
        """New-style API forward without ctx parameter.

        Args:
            logits: Logits whose final dimension is the vocabulary axis.
            labels: Target token indices, with ``-100`` marking ignored tokens.
            logit_softcapping: Gemma 2 softcap value (0 = disabled).
            logit_scaling: Cohere logit scale value (0 = disabled).
            label_smoothing: ``F.cross_entropy``-style smoothing weight
                in [0, 1].  ``0.0`` (default) is standard CE.
        """
        original_shape = logits.shape[:-1]
        vocab_size = logits.shape[-1]

        logits_flat = logits.reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)
        n_rows = logits_flat.shape[0]
        device = logits.device

        losses, logsumexp = _ce_forward_impl(
            logits_flat,
            labels_flat,
            n_rows,
            vocab_size,
            device,
            logit_softcapping=logit_softcapping,
            logit_scaling=logit_scaling,
            label_smoothing=label_smoothing,
        )

        losses = losses.reshape(original_shape)
        logsumexp = logsumexp.reshape(original_shape)

        return losses, logsumexp

    @staticmethod
    def setup_context(ctx, inputs, output):
        logits, labels, logit_softcapping, logit_scaling, label_smoothing = inputs
        _losses, logsumexp = output
        ctx.save_for_backward(logits, logsumexp, labels)
        ctx.vocab_size = logits.shape[-1]
        ctx.original_shape = logits.shape[:-1]
        ctx.logit_softcapping = logit_softcapping
        ctx.logit_scaling = logit_scaling
        ctx.label_smoothing = float(label_smoothing)

    @staticmethod
    def backward(ctx, grad_losses, grad_logsumexp):
        logits, logsumexp, labels = ctx.saved_tensors
        grad_logits = _CrossEntropyBackward.apply(
            grad_losses,
            logits,
            logsumexp,
            labels,
            ctx.vocab_size,
            ctx.logit_softcapping,
            ctx.logit_scaling,
            ctx.label_smoothing,
        )
        return grad_logits, None, None, None, None

    @staticmethod
    def vmap(
        info, in_dims, logits, labels, logit_softcapping, logit_scaling, label_smoothing
    ):
        """Custom vmap rule for DP-SGD.

        Calls Triton forward kernel directly with merged batch dims.
        """
        logits_bdim, labels_bdim, sc_bdim, ls_bdim, smooth_bdim = in_dims

        if logits_bdim != 0:
            raise ConfigurationError(
                *(f"logits should be batched at dim 0, got {logits_bdim}",)
            )
        if labels_bdim != 0:
            raise ConfigurationError(
                *(f"labels should be batched at dim 0, got {labels_bdim}",)
            )
        assert sc_bdim is None, "logit_softcapping should not be batched"
        assert ls_bdim is None, "logit_scaling should not be batched"
        assert smooth_bdim is None, "label_smoothing should not be batched"

        batched_shape = logits.shape[:-1]
        vocab_size = logits.shape[-1]

        logits_flat = logits.reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)
        n_rows = logits_flat.shape[0]
        device = logits.device

        losses, logsumexp = _ce_forward_impl(
            logits_flat,
            labels_flat,
            n_rows,
            vocab_size,
            device,
            logit_softcapping=logit_softcapping,
            logit_scaling=logit_scaling,
            label_smoothing=label_smoothing,
        )

        losses = losses.reshape(batched_shape)
        logsumexp = logsumexp.reshape(batched_shape)

        return (losses, logsumexp), (logits_bdim, logits_bdim)


def opake_cross_entropy_loss(
    logits,
    labels,
    logit_softcapping=0,
    logit_scaling=0,
    label_smoothing=0.0,
):
    """Convenience wrapper.

    Args:
        logits: Logits whose final dimension is the vocabulary axis.
        labels: Target token indices, with ``-100`` marking ignored tokens.
        logit_softcapping: Gemma 2 softcap value (0 = disabled).
        logit_scaling: Cohere logit scale value (0 = disabled).
        label_smoothing: ``F.cross_entropy``-style smoothing weight in
            [0, 1].  Default 0.0 (standard CE).
    """
    ensure_cuda_tensors(logits, labels, fn_name="opake_cross_entropy_loss")
    (logits,) = follow_autocast(logits)
    losses, _ = Opake_CrossEntropyLoss.apply(
        logits, labels, logit_softcapping, logit_scaling, label_smoothing
    )
    return losses


class Opake_SelectiveLogSoftmax(torch.autograd.Function):
    """Per-token log-prob lookup ``log_softmax(logits)[indices]`` with vmap support.

    Reuses :func:`_ce_forward_impl` — the per-token NLL the chunked CE kernel
    already returns is ``-log p_target``, so the per-token log-prob is its
    negation. The backward kernel is reused unchanged: gradient of
    ``log p_t`` w.r.t. ``logits[v]`` equals ``-1 ×`` the gradient of
    ``NLL_t``, achieved by passing ``-grad_logp`` as the upstream
    ``grad_losses``.
    """

    @staticmethod
    def forward(logits, indices):
        original_shape = logits.shape[:-1]
        vocab_size = logits.shape[-1]
        logits_flat = logits.reshape(-1, vocab_size)
        indices_flat = indices.reshape(-1)
        n_rows = logits_flat.shape[0]
        losses, logsumexp = _ce_forward_impl(
            logits_flat,
            indices_flat,
            n_rows,
            vocab_size,
            logits.device,
        )
        log_p = (-losses).reshape(original_shape)
        logsumexp = logsumexp.reshape(original_shape)
        return log_p, logsumexp

    @staticmethod
    def setup_context(ctx, inputs, output):
        logits, indices = inputs
        _log_p, logsumexp = output
        ctx.save_for_backward(logits, logsumexp, indices)
        ctx.vocab_size = logits.shape[-1]

    @staticmethod
    def backward(ctx, grad_log_p, grad_logsumexp):
        logits, logsumexp, indices = ctx.saved_tensors
        grad_logits = _CrossEntropyBackward.apply(
            -grad_log_p,
            logits,
            logsumexp,
            indices,
            ctx.vocab_size,
            0,
            0,
            0.0,
        )
        return grad_logits, None

    @staticmethod
    def vmap(info, in_dims, logits, indices):
        logits_bdim, indices_bdim = in_dims
        if logits_bdim != 0:
            raise ConfigurationError(
                *(f"logits should be batched at dim 0, got {logits_bdim}",)
            )
        if indices_bdim != 0:
            raise ConfigurationError(
                *(f"indices should be batched at dim 0, got {indices_bdim}",)
            )
        batched_shape = logits.shape[:-1]
        vocab_size = logits.shape[-1]
        logits_flat = logits.reshape(-1, vocab_size)
        indices_flat = indices.reshape(-1)
        n_rows = logits_flat.shape[0]
        losses, logsumexp = _ce_forward_impl(
            logits_flat, indices_flat, n_rows, vocab_size, logits.device
        )
        log_p = (-losses).reshape(batched_shape)
        logsumexp = logsumexp.reshape(batched_shape)
        return (log_p, logsumexp), (logits_bdim, logits_bdim)


def opake_selective_log_softmax(logits, indices):
    """Per-token ``log_softmax(logits, dim=-1).gather(-1, indices[..., None]).squeeze(-1)``.

    Logits-light: routes through the chunked CE Triton kernel, never
    materialising a second ``(T, V)`` ``log_softmax`` tensor. Returns per-token
    logp at ``indices`` in ``[0, V)``.

    Indices set to the ignore sentinel ``-100`` return ``0`` (matching the CE
    kernel's ignore convention), so the standard masked :func:`sequence_logp`
    pattern is unaffected.
    """
    ensure_cuda_tensors(logits, indices, fn_name="opake_selective_log_softmax")
    (logits,) = follow_autocast(logits)
    log_p, _ = Opake_SelectiveLogSoftmax.apply(logits, indices)
    return log_p
