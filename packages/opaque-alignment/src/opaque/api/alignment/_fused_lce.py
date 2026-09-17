# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Bridge to the opaque-patches fused linear cross-entropy kernel.

The fused alignment primitives — :func:`~opaque.api.alignment.sft.loss.fused_nll_loss`,
:func:`~opaque.api.alignment.sft.loss.fused_dft_loss`, and
:func:`~opaque.api.alignment.logprob.fused_sequence_logp` — share one accelerated
backend: the patches Triton ``Opaque_LinearCrossEntropyLoss`` kernel. It computes
the per-token cross-entropy ``Σ CE`` against ``hidden @ weight.T`` without
materialising the ``(T, V)`` logits and recomputes the LSE in its backward.
``opaque-patches`` is the optional ``opaque-alignment[patches]`` dependency, so
each primitive falls back to its eager (logits-materialising) counterpart when
the kernel is unavailable (e.g. CPU CI).

These are **per-example** helpers: call them on a single example ``(T, H)`` and
drive them with ``vmap(grad(...))`` (the ``clipped_grad`` DP-SGD path); the
kernel's merged forward/backward vmap rules then make the whole microbatch one
forward + one backward kernel launch. They call ``.apply`` directly rather than
wrapping in ``torch.vmap``: use ``vmap(grad(...))``, not ``grad(vmap(...))``,
because the kernel's manual vmap rule recomputes the forward with the raw
(non-autograd) ``_forward_impl``, so ``grad(vmap(...))`` silently yields *zero*
gradients.
"""

from __future__ import annotations

import importlib

import torch

# Patches is the optional ``opaque-alignment[patches]`` extra, not a runtime
# dependency. Resolve the kernel module dynamically so the bridge remains
# optional.
_LCE_KERNEL_PATH = "opaque.api.patches.kernels.linear_cross_entropy"
_CE_KERNEL_PATH = "opaque.api.patches.kernels.cross_entropy"
# Pure-PyTorch chunked kernel (Triton-free): the fused-CE path on MPS/CPU.
_LCE_CHUNKED_PATH = "opaque.api.patches.kernels._linear_ce_chunked"


def lce_available(hidden: torch.Tensor) -> bool:
    """True when a fused linear-CE kernel can run for ``hidden``.

    CUDA + half precision routes to the Triton kernel; any other host (MPS/CPU)
    routes to the pure-PyTorch chunked kernel, which streams the LSE in fp32 and
    works in any float dtype. ``opaque-patches`` must be importable either way
    (the ``[patches]`` extra); otherwise the caller uses its eager fallback.
    """
    if not hidden.is_floating_point():
        return False
    if hidden.is_cuda:
        if hidden.dtype not in (torch.float16, torch.bfloat16):
            return False
        path = _LCE_KERNEL_PATH
    else:
        path = _LCE_CHUNKED_PATH
    try:
        importlib.import_module(path)
    except Exception:
        return False
    return True


def selective_log_softmax_available(logits: torch.Tensor) -> bool:
    """True when the patches chunked CE kernel can power ``selective_log_softmax``.

    Needs CUDA and ``opaque-patches`` importable. The kernel casts inputs to fp32
    internally, so half precision is *not* required (unlike :func:`lce_available`).
    The eager fallback materialises a ``(T, V)`` ``log_softmax`` tensor and stays
    available everywhere this returns False.
    """
    if not logits.is_cuda:
        return False
    try:
        importlib.import_module(_CE_KERNEL_PATH)
    except Exception:
        return False
    return True


def import_ce_kernel():
    """Dynamically import the patches chunked-CE kernel module.

    Sole entry point alignment code uses to reach into the kernel. Keeps the
    static AST clean of ``import opaque.api.patches.*`` so the
    dependency-direction contract test stays green.
    """
    return importlib.import_module(_CE_KERNEL_PATH)


def linear_nll_sum(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    use_token_scaling: bool = False,
) -> torch.Tensor:
    """Per-example ``Σ CE`` (optionally DFT-token-scaled) via the patches kernel.

    Returns the unreduced sum of per-token cross-entropy over this example's
    non-ignored (``!= -100``) shifted tokens, computed from ``hidden @ weight.T``
    without materialising logits. ``use_token_scaling=True`` weights each token's
    CE by its detached model confidence ``softmax(logits)[target]`` (DFT).

    Call per example and drive with ``vmap(grad)``; see the module docstring for
    why this uses ``.apply`` directly rather than ``torch.vmap``.
    """
    if hidden.is_cuda:
        kernel_mod = importlib.import_module(_LCE_KERNEL_PATH)
        # We bypass the public wrapper, so apply its autocast policy ourselves:
        # under autocast ``hidden`` is half but the lm_head ``weight`` is fp32,
        # which would mismatch the kernel's ``tl.dot``. No-op when autocast off.
        hidden, weight = kernel_mod.follow_autocast(hidden, weight)
        nll_sum, _lse, _valids, _token_weight = (
            kernel_mod.Opaque_LinearCrossEntropyLoss.apply(
                hidden, weight, labels, -100, 0, 0.0, use_token_scaling, 1.0
            )
        )
        return nll_sum

    # Non-CUDA: the chunked kernel streams the matmul + LSE in fp32 itself, so a
    # mixed bf16-hidden / fp32-weight pair needs no follow_autocast reconciliation.
    chunked_mod = importlib.import_module(_LCE_CHUNKED_PATH)
    return chunked_mod.linear_nll_sum_chunked(
        hidden, weight, labels, -100, 0, 0.0, use_token_scaling
    )
