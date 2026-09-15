# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
#
# Triton fused linear+CE kernel derives from Apple's cut_cross_entropy
# project (Apache-2.0; https://github.com/apple/ml-cross-entropy, ICLR 2025).
# Simplified to drop bias / Kahan / filtering / VocabOrdering / dLSE / shift
# handling and adapted to Opaque's vmap-friendly new-style autograd dispatch.
# See NOTICE in the repository root.
"""Fused Linear + Cross-Entropy Loss with native vmap support.

Computes CE(hidden_states @ weight.T, labels) without materializing the full
logit matrix. Ported from Apple's cut_cross_entropy Triton kernels (ICLR 2025),
simplified to remove bias/Kahan/filtering/VocabOrdering/dLSE/shift and to
support our standard Opaque_Foo + _FooBackward vmap dispatch pattern.

Causal shift is handled by translating flat logical token indices to the
unshifted source storage. This lets vmap merge samples without materializing an
activation-sized shifted hidden-state copy.

Mathematical decomposition:
    CE(e, c, t) = -e·c[t] + log(Σ_v exp(e·c[v]))
                = neg_dot(e, c[t]) + LSE(e @ c.T)

Memory savings: O(B) + O(V) intermediate storage instead of O(B * V).
For LLaMA-3 (128K vocab), this avoids materializing ~1 GB of logits per sample.
"""

from __future__ import annotations

import os as _os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton import Config

from opaque.exceptions import ConfigurationError

from ._utils import (
    _build_flat_valids,
    b_bin_fn,
    ensure_cuda_tensors,
    follow_autocast,
    tl_lock_add,
    tl_logaddexp,
    tl_softcapping,
    tl_softcapping_grad,
)

# =============================================================================
# Autotune configs (subset of CCE defaults, sufficient for our use case)
# =============================================================================


def _get_autotune_configs():
    return [
        Config(
            {"BLOCK_B": 128, "BLOCK_V": 128, "BLOCK_D": 32}, num_warps=4, num_stages=4
        ),
        Config(
            {"BLOCK_B": 128, "BLOCK_V": 128, "BLOCK_D": 128}, num_warps=4, num_stages=2
        ),
        Config(
            {"BLOCK_B": 64, "BLOCK_V": 128, "BLOCK_D": 32}, num_warps=4, num_stages=4
        ),
        Config(
            {"BLOCK_B": 128, "BLOCK_V": 64, "BLOCK_D": 32}, num_warps=4, num_stages=4
        ),
        Config(
            {"BLOCK_B": 128, "BLOCK_V": 128, "BLOCK_D": 32}, num_warps=8, num_stages=3
        ),
        Config(
            {"BLOCK_B": 128, "BLOCK_V": 256, "BLOCK_D": 32}, num_warps=8, num_stages=3
        ),
        Config(
            {"BLOCK_B": 256, "BLOCK_V": 128, "BLOCK_D": 32}, num_warps=8, num_stages=3
        ),
        Config(
            {"BLOCK_B": 128, "BLOCK_V": 128, "BLOCK_D": 128}, num_warps=4, num_stages=4
        ),
        Config(
            {"BLOCK_B": 64, "BLOCK_V": 128, "BLOCK_D": 64}, num_warps=4, num_stages=4
        ),
        Config(
            {"BLOCK_B": 128, "BLOCK_V": 64, "BLOCK_D": 64}, num_warps=4, num_stages=4
        ),
    ]


# Use fixed best config (same as CCE default) — set CCE_AUTOTUNE=1 to enable search
_AUTOTUNE = _os.getenv("CCE_AUTOTUNE", "0") != "0"


# =============================================================================
# Forward Triton Kernel
# =============================================================================


def _linear_ce_forward_kernel(
    E,
    C,
    LSE,
    NegCorrectLogit,
    SumLogits,
    Locks,
    Valids,
    Targets,
    softcap,
    logit_scale,
    B,
    V,
    D,
    BMax,
    stride_eb,
    stride_ed,
    stride_cv,
    stride_cd,
    stride_vb,
    tokens_per_sequence,
    num_locks,
    # Meta-parameters
    B_BIN,
    HAS_VALIDS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GROUP_B: tl.constexpr,
    EVEN_D: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    label_smoothing: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_b = tl.cdiv(B, BLOCK_B)
    num_pid_v = tl.cdiv(V, BLOCK_V)
    num_pid_in_group = GROUP_B * num_pid_v
    group_id = pid // num_pid_in_group
    first_pid_b = group_id * GROUP_B
    group_size_b = min(num_pid_b - first_pid_b, GROUP_B)
    pid_b = (first_pid_b + ((pid % num_pid_in_group) % group_size_b)).to(tl.int64)
    pid_v = ((pid % num_pid_in_group) // group_size_b).to(tl.int64)

    offs_b = (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)).to(tl.int64)
    if HAS_VALIDS:
        offs_b = tl.load(Valids + stride_vb * offs_b, mask=offs_b < B, other=BMax).to(
            tl.int64
        )
    source_offs_b = offs_b + offs_b // tokens_per_sequence

    offs_v = (pid_v * BLOCK_V + tl.arange(0, BLOCK_V)).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
    e_ptrs = E + (source_offs_b[:, None] * stride_eb + offs_d[None, :] * stride_ed)
    c_ptrs = C + (offs_v[None, :] * stride_cv + offs_d[:, None] * stride_cd)

    # Tiled matmul: logits[b,v] = Σ_d E[b,d] * C[v,d]
    accum = tl.zeros((BLOCK_B, BLOCK_V), dtype=tl.float32)
    for d in range(tl.cdiv(D, BLOCK_D)):
        e_mask = offs_b[:, None] < BMax
        if not EVEN_D:
            e_mask = e_mask & (offs_d[None, :] < (D - d * BLOCK_D))
        e = tl.load(e_ptrs, mask=e_mask, other=0.0)

        c_mask = offs_v[None, :] < V
        if not EVEN_D:
            c_mask = c_mask & (offs_d[:, None] < (D - d * BLOCK_D))
        c = tl.load(c_ptrs, mask=c_mask, other=0.0)

        accum = tl.dot(e, c, accum, input_precision="ieee")
        e_ptrs += BLOCK_D * stride_ed
        c_ptrs += BLOCK_D * stride_cd

    tl.debug_barrier()

    accum = accum * logit_scale
    accum = accum.cast(E.dtype.element_ty, fp_downcast_rounding="rtne")
    logits = tl.where(offs_v[None, :] < V, accum, -float("inf"))
    if HAS_SOFTCAP:
        logits = tl_softcapping(logits, softcap)

    logits = logits.cast(tl.float32)

    # Store neg_correct_logit = -logits[b, target[b]] in logical token order.
    this_targets = tl.load(Targets + source_offs_b + 1, mask=offs_b < BMax, other=V + 1)
    direct_offs_b = (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)).to(tl.int64)
    neg_correct_logit_ptrs = NegCorrectLogit + direct_offs_b
    neg_correct_logit_ptrs = tl.broadcast_to(
        neg_correct_logit_ptrs[:, None], (BLOCK_B, BLOCK_V)
    )
    tl.store(
        neg_correct_logit_ptrs, -logits, mask=this_targets[:, None] == offs_v[None, :]
    )

    if label_smoothing > 0:
        row_sum = tl.sum(tl.where(offs_v[None, :] < V, logits, 0.0), axis=1)
        tl.atomic_add(SumLogits + direct_offs_b, row_sum, mask=direct_offs_b < B)

    # Per-block LSE: max + log(sum(exp(logits - max)))
    this_mx = tl.max(logits, axis=1)
    this_lse = this_mx + tl.log(tl.sum(tl.exp(logits - this_mx[:, None]), axis=1))

    o_mask = direct_offs_b < B
    lse_ptrs = LSE + direct_offs_b

    # Lock-based atomic logaddexp across V blocks
    this_locks = Locks + (pid_b // tl.cdiv(B, BLOCK_B * num_locks))
    while tl.atomic_cas(this_locks, 0, 1) == 1:
        pass

    lse = tl.load(lse_ptrs, mask=o_mask, other=0.0, eviction_policy="evict_last")
    lse = tl_logaddexp(lse, this_lse)
    tl.store(lse_ptrs, lse, mask=o_mask, eviction_policy="evict_last")

    tl.debug_barrier()
    tl.atomic_xchg(this_locks, 0)


_linear_ce_forward_kernel = triton.jit(_linear_ce_forward_kernel)
_linear_ce_forward_kernel = triton.heuristics(
    {
        "EVEN_D": lambda args: args["D"] % args["BLOCK_D"] == 0,
        "HAS_VALIDS": lambda args: args["Valids"] is not None,
        "HAS_SOFTCAP": lambda args: args["softcap"] is not None,
        "GROUP_B": lambda args: 8,
    }
)(_linear_ce_forward_kernel)

if _AUTOTUNE:
    _linear_ce_forward_kernel = triton.autotune(
        configs=_get_autotune_configs(),
        key=["V", "D", "B_BIN", "label_smoothing"],
        restore_value=["LSE", "SumLogits"],
    )(_linear_ce_forward_kernel)
else:
    # Fixed best config (matches CCE default)
    _linear_ce_forward_kernel = triton.heuristics(
        {
            k: (lambda args, _v=v: _v)
            for k, v in Config(
                {"BLOCK_B": 128, "BLOCK_V": 128, "BLOCK_D": 32},
                num_warps=4,
                num_stages=4,
            )
            .all_kwargs()
            .items()
        }
    )(_linear_ce_forward_kernel)


# =============================================================================
# Backward Triton Kernels
# =============================================================================


@triton.jit
def _mm_backward(
    do,
    da_ptrs,
    partial_mask_a,
    da_lock_ptr,
    n_locks,
    b_ptrs,
    partial_mask_b,
    stride_ad,
    stride_bd,
    D,
    BLOCK_D: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    d_inds = tl.arange(0, BLOCK_D)[None, :].to(tl.int64)

    b_ptrs = b_ptrs + d_inds * stride_bd
    da_ptrs = da_ptrs + d_inds * stride_ad

    for d in range(tl.cdiv(D, BLOCK_D)):
        mask = partial_mask_b if EVEN_D else partial_mask_b & (d_inds < D - d * BLOCK_D)

        b = tl.load(b_ptrs, mask=mask, other=0.0)
        da_i = tl.dot(do, b, input_precision="ieee").to(da_ptrs.dtype.element_ty)

        mask = partial_mask_a if EVEN_D else partial_mask_a & (d_inds < D - d * BLOCK_D)

        lock_offset = d // tl.cdiv(D, BLOCK_D * n_locks)
        this_da_lock_ptr = da_lock_ptr + lock_offset

        tl_lock_add(da_ptrs, da_i, mask, this_da_lock_ptr)

        b_ptrs += BLOCK_D * stride_bd
        da_ptrs += BLOCK_D * stride_ad


def _linear_ce_backward_kernel(
    E,
    C,
    LSE,
    dOut,
    Valids,
    softcap,
    logit_scale,
    Targets,
    dE,
    dELocks,
    dC,
    dCLocks,
    B,
    D,
    V,
    BMax,
    n_de_locks_0,
    n_de_locks_1,
    n_dc_locks_0,
    n_dc_locks_1,
    stride_eb,
    stride_ed,
    stride_cv,
    stride_cd,
    stride_vb,
    tokens_per_sequence,
    # Per-sample dC parameters
    tokens_per_sample,
    num_dc_samples,
    dc_sample_stride,
    dc_locks_sample_stride,
    B_BIN,
    BLOCK_B: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MM_BACK_BLOCK_D: tl.constexpr,
    GROUP_B: tl.constexpr,
    EVEN_D: tl.constexpr,
    MM_BACK_EVEN_D: tl.constexpr,
    ITEM_DO: tl.constexpr,
    HAS_VALIDS: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    COMPUTE_DC: tl.constexpr,
    COMPUTE_DE: tl.constexpr,
    PER_SAMPLE_DC: tl.constexpr,
    label_smoothing: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_b_chunks = tl.cdiv(B, BLOCK_B)
    num_v_chunks = tl.cdiv(V, BLOCK_V)
    num_v_in_group = GROUP_B * num_v_chunks
    group_id = pid // num_v_in_group
    first_pid_b = group_id * GROUP_B
    group_size_b = min(num_b_chunks - first_pid_b, GROUP_B)
    pid_b = (first_pid_b + ((pid % num_v_in_group) % group_size_b)).to(tl.int64)
    pid_v = ((pid % num_v_in_group) // group_size_b).to(tl.int64)

    offs_b = (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)).to(tl.int64)
    if HAS_VALIDS:
        offs_b = tl.load(Valids + stride_vb * offs_b, mask=offs_b < B, other=BMax).to(
            tl.int64
        )
    source_offs_b = offs_b + offs_b // tokens_per_sequence

    offs_v = (pid_v * BLOCK_V + tl.arange(0, BLOCK_V)).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
    e_ptrs = E + (source_offs_b[:, None] * stride_eb + offs_d[None, :] * stride_ed)
    c_ptrs = C + (offs_v[None, :] * stride_cv + offs_d[:, None] * stride_cd)

    # Recompute logits via tiled matmul E @ C^T
    accum = tl.zeros((BLOCK_B, BLOCK_V), dtype=tl.float32)
    for d in range(tl.cdiv(D, BLOCK_D)):
        e_mask = offs_b[:, None] < BMax
        if not EVEN_D:
            e_mask = e_mask & (offs_d[None, :] < (D - d * BLOCK_D))
        e = tl.load(e_ptrs, mask=e_mask, other=0.0)

        c_mask = offs_v[None, :] < V
        if not EVEN_D:
            c_mask = c_mask & (offs_d[:, None] < (D - d * BLOCK_D))
        c = tl.load(c_ptrs, mask=c_mask, other=0.0)

        accum = tl.dot(e, c, accum, input_precision="ieee")
        e_ptrs += BLOCK_D * stride_ed
        c_ptrs += BLOCK_D * stride_cd

    tl.debug_barrier()

    accum = accum * logit_scale
    accum = accum.cast(E.dtype.element_ty, fp_downcast_rounding="rtne")
    if HAS_SOFTCAP:
        accum = tl_softcapping(accum, softcap)

    # Load LSE for CE gradient computation
    if HAS_VALIDS:
        direct_offs_b = (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)).to(tl.int64)
        lse = tl.load(LSE + direct_offs_b, mask=direct_offs_b < B, other=float("inf"))
    else:
        lse = tl.load(LSE + offs_b, mask=offs_b < B, other=float("inf"))

    accum = accum.cast(tl.float32)
    d_accum = tl.exp(accum - lse[:, None])
    d_accum = tl.where(offs_v[None, :] < V, d_accum, 0.0)

    targets = tl.load(Targets + source_offs_b + 1, mask=offs_b < BMax, other=V + 1)
    is_target = targets[:, None] == offs_v[None, :]
    if label_smoothing > 0:
        d_accum = tl.where(offs_v[None, :] < V, d_accum - label_smoothing / V, d_accum)
        d_accum += tl.where(is_target, -(1.0 - label_smoothing), 0.0)
    else:
        d_accum += tl.where(is_target, -1.0, 0.0)

    # Scale by upstream gradient
    if ITEM_DO:
        d_out = tl.load(dOut)
    else:
        d_out = tl.load(dOut + offs_b, mask=offs_b < BMax, other=0.0)[:, None]

    d_accum = d_accum * d_out

    # Softcap and scalar-logit chain rule back to the raw E @ C.T dot product.
    if HAS_SOFTCAP:
        d_accum = tl_softcapping_grad(d_accum, accum, softcap)
    d_accum = d_accum * logit_scale

    d_accum = d_accum.cast(E.dtype.element_ty, fp_downcast_rounding="rtne")

    # dE = d_accum @ C via _mm_backward with locks
    if COMPUTE_DE:
        lock_offset = (pid_b // tl.cdiv(B, BLOCK_B * n_de_locks_0)) * n_de_locks_1
        _mm_backward(
            d_accum,
            dE + (offs_b[:, None] * stride_eb),
            offs_b[:, None] < BMax,
            dELocks + lock_offset,
            n_de_locks_1,
            C + offs_v[:, None] * stride_cv,
            offs_v[:, None] < V,
            stride_ed,
            stride_cd,
            D,
            MM_BACK_BLOCK_D,
            MM_BACK_EVEN_D,
        )

    # dC = d_accum^T @ E via _mm_backward with locks
    if COMPUTE_DC:
        if PER_SAMPLE_DC:
            # Per-sample dC: each token's contribution goes to its sample's dC buffer.
            # Compute sample_id from token position: sample_id = offs_b // tokens_per_sample
            sample_ids = offs_b // tokens_per_sample  # (BLOCK_B,)

            for s in range(num_dc_samples):
                s_mask = sample_ids == s  # (BLOCK_B,)
                count_s = tl.sum(s_mask.to(tl.int32))
                if count_s > 0:
                    masked_d = tl.where(
                        s_mask[:, None], d_accum, tl.zeros_like(d_accum)
                    )
                    dc_s_base = dC + s * dc_sample_stride
                    dc_locks_s_base = dCLocks + s * dc_locks_sample_stride

                    lock_offset = (
                        pid_v // tl.cdiv(V, BLOCK_V * n_dc_locks_0)
                    ) * n_dc_locks_1
                    _mm_backward(
                        tl.trans(masked_d),
                        dc_s_base + (offs_v[:, None] * stride_cv),
                        offs_v[:, None] < V,
                        dc_locks_s_base + lock_offset,
                        n_dc_locks_1,
                        E + (source_offs_b[:, None] * stride_eb),
                        offs_b[:, None] < BMax,
                        stride_cd,
                        stride_ed,
                        D,
                        MM_BACK_BLOCK_D,
                        MM_BACK_EVEN_D,
                    )
        else:
            lock_offset = (pid_v // tl.cdiv(V, BLOCK_V * n_dc_locks_0)) * n_dc_locks_1
            _mm_backward(
                tl.trans(d_accum),
                dC + (offs_v[:, None] * stride_cv),
                offs_v[:, None] < V,
                dCLocks + lock_offset,
                n_dc_locks_1,
                E + (source_offs_b[:, None] * stride_eb),
                offs_b[:, None] < BMax,
                stride_cd,
                stride_ed,
                D,
                MM_BACK_BLOCK_D,
                MM_BACK_EVEN_D,
            )


def _back_block_d(args) -> int:
    return 2 * args["BLOCK_D"]


_linear_ce_backward_kernel = triton.jit(_linear_ce_backward_kernel)
_linear_ce_backward_kernel = triton.heuristics(
    {
        "EVEN_D": lambda args: (args["D"] % args["BLOCK_D"]) == 0,
        "MM_BACK_BLOCK_D": lambda args: _back_block_d(args),
        "MM_BACK_EVEN_D": lambda args: (args["D"] % _back_block_d(args)) == 0,
        "HAS_VALIDS": lambda args: args["Valids"] is not None,
        "HAS_SOFTCAP": lambda args: args["softcap"] is not None,
        "ITEM_DO": lambda args: args["dOut"].numel() == 1,
        "GROUP_B": lambda args: 8,
        "COMPUTE_DC": lambda args: args["dC"] is not None,
        "COMPUTE_DE": lambda args: args["dE"] is not None,
        "PER_SAMPLE_DC": lambda args: args["num_dc_samples"] > 1,
    }
)(_linear_ce_backward_kernel)

if _AUTOTUNE:
    _linear_ce_backward_kernel = triton.autotune(
        configs=_get_autotune_configs(),
        key=["V", "D", "B_BIN", "label_smoothing"],
        reset_to_zero=["dE", "dC"],
    )(_linear_ce_backward_kernel)
else:
    _linear_ce_backward_kernel = triton.heuristics(
        {
            k: (lambda args, _v=v: _v)
            for k, v in Config(
                {"BLOCK_B": 128, "BLOCK_V": 128, "BLOCK_D": 32},
                num_warps=4,
                num_stages=4,
            )
            .all_kwargs()
            .items()
        }
    )(_linear_ce_backward_kernel)


# =============================================================================
# Python wrapper helpers
# =============================================================================


@dataclass(slots=True)
class LSEReturn:
    """Intermediate log-sum-exp values returned by the linear CE kernel."""

    lse: torch.Tensor
    neg_correct_logit: torch.Tensor
    sum_logits: torch.Tensor | None


def _per_token_nll_from_lse_ret(
    lse_ret: LSEReturn,
    vocab_size: int,
    label_smoothing: float,
) -> torch.Tensor:
    """Per-token NLL matching ``F.cross_entropy(..., label_smoothing=…)``."""
    hard = lse_ret.neg_correct_logit + lse_ret.lse
    ls = float(label_smoothing)
    if ls <= 0.0:
        return hard
    assert lse_ret.sum_logits is not None
    mean_z = lse_ret.sum_logits / vocab_size
    return (1.0 - ls) * hard + ls * (lse_ret.lse - mean_z)


def _token_scaling_weight(lse_ret: LSEReturn) -> torch.Tensor:
    """Detached per-token confidence weight ``p`` for DFT-style token scaling.

    ``p_t = softmax(logits)[target] = exp(logit_target − LSE) = exp(−CE_t)`` —
    the model's probability of the gold token — returned in the same
    valids-compacted order as the kernel's per-token NLL. Detached: it is a
    constant multiplier, so ``Σ p·nll`` has gradient ``Σ p·∂nll`` (no gradient
    flows through ``p``). Computed from the *hard* CE (``neg_correct_logit +
    lse``) so it is the true model confidence regardless of label smoothing.
    """
    hard = lse_ret.neg_correct_logit + lse_ret.lse
    return torch.exp(-hard).detach()


def _scatter_to_full(
    compact: torch.Tensor, valids: torch.Tensor | None, n_tokens: int
) -> torch.Tensor:
    """Scatter a valids-compacted per-token vector back to full length ``N``.

    The backward kernel indexes the upstream grad ``do`` by absolute (pre-shift)
    token position, whereas forward per-token quantities (NLL, ``p``) are
    compacted to non-ignored positions. Ignored positions get ``0`` (the kernel
    masks them anyway). ``valids is None`` means every token is valid (already
    full length).
    """
    if valids is None:
        return compact
    full = compact.new_zeros(n_tokens)
    full[valids.long()] = compact
    return full


def _saved_valids(valids: torch.Tensor | None, targets: torch.Tensor) -> torch.Tensor:
    """Represent the all-valid ``None`` fast path as an empty output tensor."""
    if valids is None:
        return targets.new_empty(0, dtype=torch.int32)
    return valids


def _restore_valids(
    saved_valids: torch.Tensor, lse: torch.Tensor, n_tokens: int
) -> torch.Tensor | None:
    """Restore the all-valid fast path from saved forward statistics."""
    if saved_valids.numel() == 0 and lse.numel() == n_tokens:
        return None
    return saved_valids


def _flatten_unshifted_sources(
    hidden_states: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return flat source storage and the logical tokens in each shifted sequence."""
    assert hidden_states.shape[:-1] == labels.shape
    tokens_per_sequence = labels.shape[-1] - 1
    assert tokens_per_sequence > 0
    return hidden_states.flatten(0, -2), labels.flatten(), tokens_per_sequence


def _forward_impl(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    valids: torch.Tensor | None,
    softcap: float | None,
    tokens_per_sequence: int,
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
) -> LSEReturn:
    """Launch forward kernel over logically shifted tokens in source storage."""
    assert e.is_contiguous()
    assert targets.is_contiguous()
    assert e.shape[1] == c.shape[1]
    source_tokens_per_sequence = tokens_per_sequence + 1
    assert e.shape[0] % source_tokens_per_sequence == 0
    assert targets.numel() == e.shape[0]
    n_tokens = e.shape[0] // source_tokens_per_sequence * tokens_per_sequence

    if valids is not None:
        assert valids.ndim == 1
        B = valids.numel()
    else:
        B = n_tokens

    V, D = c.shape
    ls = float(label_smoothing)

    lse = e.new_full((B,), -torch.inf, dtype=torch.float32)
    neg_correct_logit = e.new_full((B,), 0.0, dtype=torch.float32)
    sum_logits = e.new_zeros((B,), dtype=torch.float32)
    locks = e.new_full((triton.cdiv(B, 128),), 0, dtype=torch.uint32)

    def grid(META):
        return (triton.cdiv(B, META["BLOCK_B"]) * triton.cdiv(V, META["BLOCK_V"]),)

    _linear_ce_forward_kernel[grid](
        e,
        c,
        lse,
        neg_correct_logit,
        sum_logits,
        locks,
        valids,
        targets,
        softcap,
        logit_scale,
        B,
        V,
        D,
        n_tokens,  # BMax: logical shifted-token count
        e.stride(0),
        e.stride(1),
        c.stride(0),
        c.stride(1),
        1 if valids is None else valids.stride(0),
        tokens_per_sequence=tokens_per_sequence,
        num_locks=locks.size(0),
        B_BIN=b_bin_fn(B),
        label_smoothing=ls,
    )

    return LSEReturn(
        lse,
        neg_correct_logit,
        sum_logits if ls > 0.0 else None,
    )


def _backward_impl(
    do: torch.Tensor,
    e: torch.Tensor,
    c: torch.Tensor,
    lse: torch.Tensor,
    targets: torch.Tensor,
    valids: torch.Tensor | None,
    softcap: float | None,
    tokens_per_sequence: int,
    compute_de: bool,
    compute_dc: bool,
    num_dc_samples: int = 1,
    tokens_per_sample: int = 0,
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Launch backward kernel.

    Args:
        do: Upstream gradient for the per-token loss.
        e: Token hidden states.
        c: Output-projection weight matrix.
        lse: Per-token log-sum-exp values from the forward pass.
        targets: Target token indices.
        valids: Optional mask selecting valid token positions.
        softcap: Optional logit softcap value.
        tokens_per_sequence: Logical token count after shifting each sequence.
        compute_de: Whether to compute hidden-state gradients.
        compute_dc: Whether to compute output-projection gradients.
        num_dc_samples: When > 1, produces per-sample dC of shape (num_dc_samples, V, D).
            Tokens are split by sample_id = token_position // tokens_per_sample.
        tokens_per_sample: Number of tokens per sample (required when num_dc_samples > 1).
        label_smoothing: Cross-entropy label-smoothing weight.
        logit_scale: Scalar multiplier applied before optional softcapping.

    Returns (de, dc) cast to input dtype.
    """
    assert e.is_contiguous()
    assert targets.is_contiguous()
    assert c.shape[1] == e.shape[1]
    source_tokens_per_sequence = tokens_per_sequence + 1
    assert e.shape[0] % source_tokens_per_sequence == 0
    assert targets.numel() == e.shape[0]
    n_tokens = e.shape[0] // source_tokens_per_sequence * tokens_per_sequence

    B = valids.size(0) if valids is not None else n_tokens

    V, D = c.shape
    nd_locks = triton.cdiv(D, 64)

    de = (
        torch.zeros(n_tokens, D, dtype=torch.float32, device=e.device)
        if compute_de
        else None
    )

    if compute_dc:
        if num_dc_samples > 1:
            # Per-sample dC: (num_dc_samples, V, D)
            dc = torch.zeros(num_dc_samples, V, D, dtype=torch.float32, device=e.device)
            dc_sample_stride = V * D  # stride between samples (contiguous)
        else:
            dc = torch.zeros_like(c, dtype=torch.float32)
            dc_sample_stride = 0
    else:
        dc = None
        dc_sample_stride = 0

    if de is not None:
        de_locks = e.new_zeros((triton.cdiv(B, 128), nd_locks), dtype=torch.int32)
        de_lock_sizes = de_locks.size()
    else:
        de_locks = None
        de_lock_sizes = (None, None)

    if dc is not None:
        if num_dc_samples > 1:
            dc_locks = e.new_zeros(
                (num_dc_samples, triton.cdiv(V, 128), nd_locks), dtype=torch.int32
            )
            dc_lock_sizes = (triton.cdiv(V, 128), nd_locks)
            dc_locks_sample_stride = triton.cdiv(V, 128) * nd_locks
        else:
            dc_locks = e.new_zeros((triton.cdiv(V, 128), nd_locks), dtype=torch.int32)
            dc_lock_sizes = dc_locks.size()
            dc_locks_sample_stride = 0
    else:
        dc_locks = None
        dc_lock_sizes = (None, None)
        dc_locks_sample_stride = 0

    do = do.contiguous()
    lse = lse.contiguous()
    ls = float(label_smoothing)

    def grid(META):
        return (triton.cdiv(B, META["BLOCK_B"]) * triton.cdiv(V, META["BLOCK_V"]),)

    _linear_ce_backward_kernel[grid](
        e,
        c,
        lse,
        do,
        valids,
        softcap,
        logit_scale,
        targets,
        de,
        de_locks,
        dc,
        dc_locks,
        B,
        D,
        V,
        n_tokens,  # BMax: logical shifted-token count
        *de_lock_sizes,
        *dc_lock_sizes,
        e.stride(0),
        e.stride(1),
        c.stride(0),
        c.stride(1),
        1 if valids is None else valids.stride(0),
        tokens_per_sequence=tokens_per_sequence,
        tokens_per_sample=tokens_per_sample,
        num_dc_samples=num_dc_samples,
        dc_sample_stride=dc_sample_stride,
        dc_locks_sample_stride=dc_locks_sample_stride,
        B_BIN=b_bin_fn(B),
        label_smoothing=ls,
    )

    if de is not None:
        de = de.to(dtype=e.dtype)
    if dc is not None:
        dc = dc.to(dtype=c.dtype)

    return de, dc


# =============================================================================
# Backward autograd.Function (for vmap(grad()) dispatch)
# =============================================================================


class _LinearCEBackward(torch.autograd.Function):
    """Backward pass wrapped as autograd.Function for vmap(grad()) support."""

    @staticmethod
    def forward(
        grad_out,
        hidden_states,
        weight,
        labels,
        lse,
        saved_valids,
        token_weight,
        softcap,
        compute_dc,
        label_smoothing,
        use_token_scaling,
        logit_scale,
    ):
        e, targets, tokens_per_sequence = _flatten_unshifted_sources(
            hidden_states, labels
        )
        n_tokens = e.shape[0] // (tokens_per_sequence + 1) * tokens_per_sequence
        valids = _restore_valids(saved_valids, lse, n_tokens)

        do = grad_out
        if use_token_scaling:
            # The backward kernel reads ``do`` position-indexed (full length),
            # so scatter the saved valids-compacted confidence before scaling.
            do = grad_out * _scatter_to_full(token_weight, valids, n_tokens)

        de, dc = _backward_impl(
            do,
            e,
            weight,
            lse,
            targets,
            valids,
            softcap,
            tokens_per_sequence,
            compute_de=True,
            compute_dc=compute_dc,
            label_smoothing=label_smoothing,
            logit_scale=logit_scale,
        )
        if dc is None:
            dc = weight.new_zeros(weight.shape)
        return de, dc

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError(
            "Double backward not supported for LinearCrossEntropyLoss"
        )

    @staticmethod
    def vmap(
        info,
        in_dims,
        grad_out,
        hidden_states,
        weight,
        labels,
        lse,
        saved_valids,
        token_weight,
        softcap,
        compute_dc,
        label_smoothing,
        use_token_scaling,
        logit_scale,
    ):
        (
            grad_bdim,
            _h_bdim,
            w_bdim,
            _lab_bdim,
            lse_bdim,
            valids_bdim,
            token_weight_bdim,
            sc_bdim,
            dc_bdim,
            ls_bdim,
            uts_bdim,
            scale_bdim,
        ) = in_dims

        assert w_bdim is None, "weight should not be batched"
        assert lse_bdim is None, "saved LSE should be a merged tensor"
        assert valids_bdim is None, "saved valid indices should be a merged tensor"
        assert token_weight_bdim is None, "saved token weights should be merged"
        assert sc_bdim is None, "softcap should not be batched"
        assert dc_bdim is None, "compute_dc should not be batched"
        assert ls_bdim is None, "label_smoothing should not be batched"
        assert uts_bdim is None, "use_token_scaling should not be batched"
        assert scale_bdim is None, "logit_scale should not be batched"

        B_vmap = hidden_states.shape[0]
        D = hidden_states.shape[-1]
        e, targets, tokens_per_sequence = _flatten_unshifted_sources(
            hidden_states, labels
        )
        tokens_per_sample = (
            e.shape[0] // B_vmap // (tokens_per_sequence + 1) * tokens_per_sequence
        )
        n_tokens = B_vmap * tokens_per_sample
        valids = _restore_valids(saved_valids, lse, n_tokens)

        # Expand per-sample scalar grad to per-token grad
        if grad_bdim is not None:
            do = grad_out.repeat_interleave(
                torch.tensor(tokens_per_sample, device=grad_out.device)
            )
        else:
            do = grad_out.expand(n_tokens)

        if use_token_scaling:
            # Scale each token's upstream grad by its saved detached confidence,
            # scattered back to the full position-indexed length the kernel reads.
            do = do * _scatter_to_full(token_weight, valids, n_tokens)

        # Single merged backward with per-sample dC (if needed).
        de, dc = _backward_impl(
            do,
            e,
            weight,
            lse,
            targets,
            valids,
            softcap,
            tokens_per_sequence,
            compute_de=True,
            compute_dc=compute_dc,
            num_dc_samples=B_vmap if compute_dc else 1,
            tokens_per_sample=tokens_per_sample if compute_dc else 0,
            label_smoothing=label_smoothing,
            logit_scale=logit_scale,
        )

        # Reshape de from (B_vmap * tokens_per_sample, D) to (B_vmap, tokens_per_sample, D)
        # Don't pad here — backward() handles reshape + padding
        de = de.reshape(B_vmap, tokens_per_sample, D)

        if dc is None:
            # ``weight`` is frozen (``compute_dc=False``) — the per-sample
            # weight gradient is structurally all-zero.  Return a single
            # (V, D) zero tensor with ``out_dim=None`` so vmap broadcasts it
            # across the batch; allocating (B_vmap, V, D) here costs an
            # extra ``B_vmap * V * D`` floats and OOMs Mellum-scale LoRA on
            # 24 GB GPUs even at microbatch=4.
            return (de, weight.new_zeros(weight.shape)), (0, None)

        # dc is (B_vmap, V, D) from per-sample kernel
        return (de, dc), (0, 0)


# =============================================================================
# Main autograd.Function
# =============================================================================


class Opaque_LinearCrossEntropyLoss(torch.autograd.Function):
    """Fused linear projection + cross-entropy loss with vmap support.

    Computes the NLL sum (HF-style label shift: position ``i`` predicts
    ``labels[i+1]``) per valid token, optionally with label smoothing matching
    :func:`torch.nn.functional.cross_entropy`.  Does not materialize the full
    ``(batch * seq, vocab)`` logit matrix.

    Returns unreduced ``nll_sum`` — caller handles reduction.
    """

    # ``apply`` must not project the outputs down to the loss: a bare
    # ``torch.func.grad`` hands whatever ``apply`` returns to ``setup_context``
    # as ``output``.  Callers unpack ``(nll_sum, lse, valids, token_weight)``.

    @staticmethod
    def forward(
        hidden_states,
        weight,
        labels,
        ignore_index=-100,
        logit_softcapping=0,
        label_smoothing=0.0,
        use_token_scaling=False,
        logit_scale=1.0,
    ):
        softcap = logit_softcapping if logit_softcapping != 0 else None

        e, targets, tokens_per_sequence = _flatten_unshifted_sources(
            hidden_states, labels
        )
        valids = _build_flat_valids(labels[..., 1:], ignore_index)

        lse_ret = _forward_impl(
            e,
            weight,
            targets,
            valids,
            softcap,
            tokens_per_sequence,
            label_smoothing,
            logit_scale,
        )
        nll = _per_token_nll_from_lse_ret(lse_ret, weight.shape[0], label_smoothing)
        if use_token_scaling:
            # Weight each token's CE by its detached confidence p_t (DFT). ``p``
            # is already in the valids-compacted order of ``nll``.
            token_weight = _token_scaling_weight(lse_ret)
            nll = token_weight * nll
        else:
            token_weight = lse_ret.lse.new_empty(0)
        return nll.sum(), lse_ret.lse, _saved_valids(valids, targets), token_weight

    @staticmethod
    def setup_context(ctx, inputs, output):
        (
            hidden_states,
            weight,
            labels,
            _ignore_index,
            logit_softcapping,
            label_smoothing,
            use_token_scaling,
            logit_scale,
        ) = inputs

        _, lse, saved_valids, token_weight = output
        ctx.mark_non_differentiable(lse, saved_valids, token_weight)
        ctx.save_for_backward(
            hidden_states, weight, labels, lse, saved_valids, token_weight
        )
        ctx.softcap = logit_softcapping if logit_softcapping != 0 else None
        ctx.label_smoothing = float(label_smoothing)
        ctx.use_token_scaling = bool(use_token_scaling)
        ctx.logit_scale = float(logit_scale)

    @staticmethod
    def backward(ctx, grad_loss, _grad_lse, _grad_valids, _grad_token_weight):
        hidden_states, weight, labels, lse, saved_valids, token_weight = (
            ctx.saved_tensors
        )
        # needs_input_grad[1] = weight needs grad. In DP-SGD LoRA training,
        # weight is frozen → skip dC to save ~1/3 of backward kernel time.
        compute_dc = ctx.needs_input_grad[1]

        de, dc = _LinearCEBackward.apply(
            grad_loss,
            hidden_states,
            weight,
            labels,
            lse,
            saved_valids,
            token_weight,
            ctx.softcap,
            compute_dc,
            ctx.label_smoothing,
            ctx.use_token_scaling,
            ctx.logit_scale,
        )

        # de is (shifted_seq, D) — reshape and pad to match hidden_states shape
        shifted_shape = list(hidden_states.shape)
        shifted_shape[-2] -= 1
        de = de.reshape(shifted_shape)
        pad_shape = list(hidden_states.shape)
        pad_shape[-2] = 1
        de = torch.cat([de, de.new_zeros(pad_shape)], dim=-2)

        return de, dc, None, None, None, None, None, None

    @staticmethod
    def vmap(
        info,
        in_dims,
        hidden_states,
        weight,
        labels,
        ignore_index,
        logit_softcapping,
        label_smoothing,
        use_token_scaling,
        logit_scale,
    ):
        """Custom vmap rule for DP-SGD — single merged kernel call."""
        (
            h_bdim,
            w_bdim,
            lab_bdim,
            ii_bdim,
            sc_bdim,
            ls_bdim,
            uts_bdim,
            scale_bdim,
        ) = in_dims

        if h_bdim != 0:
            raise ConfigurationError(
                *(f"hidden_states should be batched at dim 0, got {h_bdim}",)
            )
        if lab_bdim != 0:
            raise ConfigurationError(
                *(f"labels should be batched at dim 0, got {lab_bdim}",)
            )
        assert w_bdim is None, "weight should not be batched"
        assert ii_bdim is None, "ignore_index should not be batched"
        assert sc_bdim is None, "logit_softcapping should not be batched"
        assert ls_bdim is None, "label_smoothing should not be batched"
        assert uts_bdim is None, "use_token_scaling should not be batched"
        assert scale_bdim is None, "logit_scale should not be batched"

        softcap = logit_softcapping if logit_softcapping != 0 else None

        B_vmap = hidden_states.shape[0]
        V = weight.shape[0]

        e, targets, tokens_per_sequence = _flatten_unshifted_sources(
            hidden_states, labels
        )
        tokens_per_sample = (
            e.shape[0] // B_vmap // (tokens_per_sequence + 1) * tokens_per_sequence
        )
        valids = _build_flat_valids(labels[..., 1:], ignore_index)

        # Single forward call for entire merged batch
        lse_ret = _forward_impl(
            e,
            weight,
            targets,
            valids,
            softcap,
            tokens_per_sequence,
            label_smoothing,
            logit_scale,
        )
        nll = _per_token_nll_from_lse_ret(lse_ret, V, label_smoothing)
        if use_token_scaling:
            token_weight = _token_scaling_weight(lse_ret)
            nll = token_weight * nll
        else:
            token_weight = lse_ret.lse.new_empty(0)

        # Split per-sample: scatter NLLs back to sample buckets
        if valids is not None:
            sample_ids = valids.long() // tokens_per_sample
            nll_sums = torch.zeros(B_vmap, device=nll.device, dtype=nll.dtype)
            nll_sums.scatter_add_(0, sample_ids, nll)
        else:
            nll_sums = nll.reshape(B_vmap, tokens_per_sample).sum(dim=1)

        return (
            nll_sums,
            lse_ret.lse,
            _saved_valids(valids, targets),
            token_weight,
        ), (0, None, None, None)


def opaque_linear_cross_entropy_loss(
    hidden_states,
    weight,
    labels,
    ignore_index=-100,
    logit_softcapping=0,
    label_smoothing=0.0,
    use_token_scaling=False,
    logit_scale=1.0,
):
    """Convenience wrapper for fused linear + cross-entropy loss.

    The kernel returns nll_sum (unreduced); this wrapper divides by the
    count of non-ignored tokens (mean reduction).

    ``logit_scale`` is applied to each logits tile before optional softcapping;
    backward applies its chain rule directly to the original weight.

    Args:
        hidden_states: (..., hidden_dim) embeddings from backbone
        weight: (vocab_size, hidden_dim) lm_head weight (pre-scaled if needed)
        labels: (...,) target token IDs (-100 = ignore)
        ignore_index: label value to ignore
        logit_softcapping: Gemma2 softcap value (0 = disabled)
        label_smoothing: same semantics as :func:`torch.nn.functional.cross_entropy`
        use_token_scaling: when True, weight each token's CE by its detached
            model confidence ``p_t = softmax(logits)[target]`` before summing
            (DFT). The weight is computed inside the kernel (no logits are
            materialised) and treated as a constant in the backward. Intended
            with ``label_smoothing=0``.
        logit_scale: scalar multiplier applied to logits before softcapping.

    Returns:
        loss: scalar tensor
    """
    ensure_cuda_tensors(
        hidden_states,
        weight,
        labels,
        fn_name="opaque_linear_cross_entropy_loss",
    )
    hidden_states, weight = follow_autocast(hidden_states, weight)
    nll_sum, _lse, _valids, _token_weight = Opaque_LinearCrossEntropyLoss.apply(
        hidden_states,
        weight,
        labels,
        ignore_index,
        logit_softcapping,
        label_smoothing,
        use_token_scaling,
        logit_scale,
    )
    n_valid = (labels[..., 1:] != ignore_index).sum().float().clamp(min=1)
    return nll_sum / n_valid
