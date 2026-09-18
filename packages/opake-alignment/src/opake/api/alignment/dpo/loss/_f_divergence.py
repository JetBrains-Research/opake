# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""f-divergence log-ratio remaps for the DPO family.

The standard DPO objective (Rafailov et al., 2023) implicitly minimises the
*reverse* KL divergence between the policy and the reference. The DPO loss can
be generalised to other f-divergences by remapping each side's per-example
log-ratio ``g(logratio)`` *before* the preference logits are formed, i.e.::

    delta = g(chosen_logratio) - g(rejected_logratio)

This remapped ``delta`` then feeds any DPO variant (sigmoid, hinge, ipo, ...)
unchanged. The four supported f-divergences and their per-side remap ``g`` are
:

    reverse_kl        g(x) = x                          (identity — the DPO default)
    forward_kl        g(x) = 1 - exp(-x)
    js_divergence     g(x) = log(2) + logsigmoid(x)
    alpha_divergence  g(x) = (exp((alpha - 1) * x) - 1) / (alpha - 1)

The ``alpha_divergence`` remap at ``alpha == 1`` is a removable singularity
whose limit is the ``reverse_kl`` identity. Values within ``1e-6`` of one use
that limit, matching TRL's fallback.

The ``exp`` paths (``forward_kl`` and ``alpha_divergence``) clamp the exponent
through :func:`_cap_exp` with a dtype-aware cap (11.0 for fp16, 80.0 for
bf16/fp32) so that large log-ratios do not overflow to ``inf``.

The remap of example *i* depends only on example *i*'s log-ratio; there is no
cross-example aggregate. Dispatch on ``f_divergence_type`` / ``alpha`` uses
ordinary Python ``if``/``elif`` on *static* (non-tensor) arguments.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F

from opake.exceptions import ConfigurationError

__all__ = ["f_divergence_logits", "f_divergence_remap"]

FDivergence = Literal["reverse_kl", "forward_kl", "js_divergence", "alpha_divergence"]
_ALPHA_DIVERGENCE_LIMIT_TOLERANCE = 1e-6


def _cap_exp(x: torch.Tensor, cap: float | None = None) -> torch.Tensor:
    """``exp`` with an upper clamp on the exponent to avoid fp16/bf16 overflow.

    ``exp`` grows extremely fast: fp16 overflows above an exponent of roughly
    ``11.09``, while bf16 and fp32 overflow above roughly ``88.7``. Clamping the
    *exponent* (not the result) keeps the output finite while leaving the
    small/moderate range untouched.

    The default cap is chosen per dtype to stay safely below the dtype's
    overflow threshold:

    - float16: ``11.0`` (``exp(11) ~ 59,000``, fp16 max finite ~65,504)
    - bfloat16: ``80.0`` (``exp(80) ~ 5.5e34``, bf16 max finite ~3.4e38)
    - float32, float64: ``80.0``

    Args:
        x: Exponent tensor.
        cap: Upper bound applied to ``x`` before exponentiation. Defaults to
            a dtype-safe value (11.0 for fp16, 80.0 otherwise).

    Returns:
        ``exp(clamp(x, max=cap))``, same shape and dtype as ``x``.
    """
    if cap is None:
        cap = 11.0 if x.dtype == torch.float16 else 80.0
    return torch.exp(torch.clamp(x, max=cap))


def f_divergence_remap(
    logratio: torch.Tensor,
    *,
    f_divergence_type: FDivergence = "reverse_kl",
    alpha: float = 1.0,
) -> torch.Tensor:
    """Remap a per-example log-ratio under the chosen f-divergence.

    Applies the per-side remap ``g`` for the requested f-divergence:

    - ``reverse_kl``: ``g(x) = x`` (identity — grad-transparent).
    - ``forward_kl``: ``g(x) = 1 - _cap_exp(-x)``.
    - ``js_divergence``: ``g(x) = log(2) + logsigmoid(x)``.
    - ``alpha_divergence``:
      ``g(x) = (_cap_exp((alpha - 1) * x) - 1) / (alpha - 1)``;
      near ``alpha == 1`` this uses the identity (``reverse_kl``).

    The function is elementwise and works on batched ``(B,)`` inputs as well as
    0-dim scalars.

    Args:
        logratio: Per-example log-ratio ``log π(y | x) − log π_ref(y | x)``.
            Shape ``(B,)`` or ``()``.
        f_divergence_type: Which f-divergence remap to apply. One of
            ``"reverse_kl"``, ``"forward_kl"``, ``"js_divergence"``,
            ``"alpha_divergence"``. Defaults to ``"reverse_kl"``.
        alpha: The α parameter for ``alpha_divergence``. When ``alpha == 1``
            the function returns the identity (matching TRL's fallback).
            Ignored by the other divergences. Defaults to ``1.0``.

    Returns:
        Remapped log-ratio tensor of the same shape as ``logratio``.

    Raises:
        ValueError: If ``f_divergence_type`` is not a recognised divergence name.
    """
    if f_divergence_type == "reverse_kl":
        # Identity: returned unchanged so the gradient path is fully
        # transparent (d g/d x = 1), reproducing the standard DPO objective.
        return logratio
    if f_divergence_type == "forward_kl":
        return 1 - _cap_exp(-logratio)
    if f_divergence_type == "js_divergence":
        return math.log(2.0) + F.logsigmoid(logratio)
    if f_divergence_type == "alpha_divergence":
        if abs(alpha - 1.0) < _ALPHA_DIVERGENCE_LIMIT_TOLERANCE:
            # Limit of (exp((alpha-1)*x) - 1)/(alpha-1) as alpha->1.
            return logratio
        return (_cap_exp((alpha - 1) * logratio) - 1) / (alpha - 1)
    raise ConfigurationError(
        *(
            f"Unknown f_divergence_type {f_divergence_type!r}; expected one of "
            '"reverse_kl", "forward_kl", "js_divergence", "alpha_divergence".',
        )
    )


def f_divergence_logits(
    chosen_logratio: torch.Tensor,
    rejected_logratio: torch.Tensor,
    *,
    f_divergence_type: FDivergence = "reverse_kl",
    alpha: float = 1.0,
) -> torch.Tensor:
    """Form the remapped preference logits ``delta`` under an f-divergence.

    Computes::

        delta = f_divergence_remap(chosen_logratio) - f_divergence_remap(rejected_logratio)

    where both sides are remapped under the same ``f_divergence_type`` /
    ``alpha``. The resulting ``delta`` is the generalised drop-in replacement for
    the plain ``chosen_logratio - rejected_logratio`` term consumed by the DPO
    variants. Under ``reverse_kl`` this reduces exactly to that plain difference.

    The function is elementwise and works on batched ``(B,)`` inputs as well as
    0-dim scalars.

    Args:
        chosen_logratio: Per-example log-ratio for the chosen response. Shape
            ``(B,)`` or ``()``.
        rejected_logratio: Per-example log-ratio for the rejected response. Same
            shape as ``chosen_logratio``.
        f_divergence_type: Which f-divergence remap to apply to both sides.
            Defaults to ``"reverse_kl"``.
        alpha: The α parameter for ``alpha_divergence``. When ``alpha == 1``
            the function falls through to ``reverse_kl``. Ignored by the
            other divergences. Defaults to ``1.0``.

    Returns:
        Remapped preference-logit tensor of the same shape as the inputs.

    Raises:
        ValueError: Propagated from :func:`f_divergence_remap` for an unknown
            ``f_divergence_type``.
    """
    chosen = f_divergence_remap(
        chosen_logratio, f_divergence_type=f_divergence_type, alpha=alpha
    )
    rejected = f_divergence_remap(
        rejected_logratio, f_divergence_type=f_divergence_type, alpha=alpha
    )
    return chosen - rejected
