# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-example router statistics of mixture-of-experts models.

Algorithm-neutral helpers behind the MoE load-balancing release: the
per-example load fractions and mean router probabilities, their centred
form, the frozen-load surrogate and its structural sensitivity bound.  The
DP-SGD release that consumes them is :mod:`opaque.api.dpsgd.clipping`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from opaque.exceptions import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Sequence

_MIN_EXPERTS = 2


def _binary_mask(
    attention_mask: torch.Tensor | None,
    num_tokens: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Flat fp32 0/1 token mask (``attention_mask != 0``; all ones when absent)."""
    if attention_mask is None:
        return torch.ones(num_tokens, dtype=torch.float32, device=device)
    mask = (attention_mask != 0).reshape(-1).to(torch.float32)
    if mask.shape[0] != num_tokens:
        raise ConfigurationError(
            *(
                f"attention_mask covers {mask.shape[0]} tokens but the router "
                f"logits cover {num_tokens}.",
            )
        )
    return mask


def router_load_and_probs(
    router_logits: Sequence[torch.Tensor],
    attention_mask: torch.Tensor | None,
    *,
    top_k: int,
    num_layers: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-example load fractions, mean router probabilities and token count.

    Runs inside ``vmap(grad(...))`` on one example.  Load and router
    probability follow the Switch load-balancing loss (Fedus, Zoph, Shazeer,
    2022, eqs. 4-6) as pooled by Hugging Face: averaged over all layers and
    valid tokens with one common denominator.

    Args:
        router_logits: One ``(T, E)`` logits tensor per routed layer.
        attention_mask: Token validity (``None`` counts every position; read
            as ``attention_mask != 0``).
        top_k: Experts executed per token.
        num_layers: Expected number of captured layers; a mismatch raises.

    Returns:
        ``(h_layers, P, T_x)``: the ``(L, E)`` executed load fraction per
        layer, the ``(E,)`` mean router probability pooled over layers and
        valid tokens, and the valid-token count.  A fully masked row returns
        zeros.
    """
    if len(router_logits) != num_layers:
        raise ConfigurationError(
            *(
                f"expected router logits for {num_layers} layers, got "
                f"{len(router_logits)}; a duplicated capture would double the load.",
            )
        )
    first = router_logits[0]
    num_experts = first.shape[-1]
    num_tokens = first.reshape(-1, num_experts).shape[0]
    mask = _binary_mask(attention_mask, num_tokens, device=first.device)
    valid_tokens = mask.sum()
    denominator = valid_tokens.clamp(min=1.0)
    expert_ids = torch.arange(num_experts, device=first.device)

    loads = []
    prob_sum = torch.zeros(num_experts, dtype=torch.float32, device=first.device)
    for logits in router_logits:
        z = logits.reshape(-1, num_experts)
        if z.shape[0] != num_tokens:
            raise ConfigurationError(
                *("router logits disagree on the token count across layers.",)
            )
        probs = torch.softmax(z.float(), dim=-1)
        indices = torch.topk(probs, top_k, dim=-1).indices
        one_hot = (indices[..., None] == expert_ids).sum(dim=-2).to(torch.float32)
        loads.append((one_hot * mask[:, None]).sum(dim=0) / denominator)
        prob_sum = prob_sum + (probs * mask[:, None]).sum(dim=0)

    h_layers = torch.stack(loads)
    mean_probs = prob_sum / (len(router_logits) * denominator)
    has_tokens = valid_tokens > 0
    h_layers = torch.where(has_tokens, h_layers, torch.zeros_like(h_layers))
    mean_probs = torch.where(has_tokens, mean_probs, torch.zeros_like(mean_probs))
    return h_layers, mean_probs, valid_tokens


def centred_load(
    h_layers: torch.Tensor,
    *,
    top_k: int,
    valid_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """``h - top_k / E`` per layer; sums to zero over experts on a non-empty row.

    With ``valid_tokens`` (the ``T_x`` of :func:`router_load_and_probs`) a
    fully masked row returns an explicit zero instead of ``-top_k / E``.
    """
    num_experts = h_layers.shape[-1]
    centred = h_layers - top_k / num_experts
    if valid_tokens is None:
        return centred
    return torch.where(valid_tokens > 0, centred, torch.zeros_like(centred))


def load_balancing_surrogate(
    probs: torch.Tensor,
    f_tilde: torch.Tensor,
    token_weight: torch.Tensor | float,
    *,
    num_experts: int,
    top_k: int,
) -> torch.Tensor:
    """Per-example surrogate ``E * w_x * <f_tilde - top_k / E, P(x)>``.

    Summed over a batch with ``token_weight = T_x / mean_tokens`` and divided
    by the expected batch size, its gradient is the Switch load-balancing
    gradient at the frozen load ``f_tilde`` up to the public normalization
    (exact when the batch's token total equals ``normalize_by * mean_tokens``).
    Since ``sum_e P = 1`` the centring leaves the gradient unchanged and makes
    the value vanish at balance.
    """
    centred = f_tilde.to(probs.dtype) - top_k / num_experts
    return num_experts * token_weight * (centred * probs).sum()


def load_bound(
    *,
    top_k: int,
    num_experts: int,
    num_layers: int,
    mean_tokens: float,
    max_tokens: float,
) -> float:
    """Structural L2 bound ``Delta_L`` of one example's token-weighted load.

    For one example ``0 <= h^l_e <= 1`` and ``sum_e h^l_e = k`` on every
    layer, so ``||h^l - k/E||^2 <= k (1 - k/E)``; over ``L`` layers and with
    the token weight ``w_x <= max_tokens / mean_tokens``::

        Delta_L = (max_tokens / mean_tokens) * sqrt(k * L * (1 - k / E))

    The bound is attained when every token of every layer is routed to the
    same ``k`` experts, and it holds for adversarial inputs.
    """
    if num_experts < _MIN_EXPERTS or not 1 <= top_k < num_experts:
        raise ConfigurationError(
            *(
                "the router-load release requires 1 <= top_k < num_experts, "
                f"got top_k={top_k}, num_experts={num_experts}.",
            )
        )
    if num_layers < 1:
        raise ConfigurationError(*(f"num_layers must be >= 1, got {num_layers}.",))
    if not all(math.isfinite(t) and t > 0 for t in (mean_tokens, max_tokens)):
        raise ConfigurationError(
            *(
                "mean_tokens and max_tokens must be finite and positive, got "
                f"mean_tokens={mean_tokens}, max_tokens={max_tokens}.",
            )
        )
    return (max_tokens / mean_tokens) * math.sqrt(
        top_k * num_layers * (1.0 - top_k / num_experts)
    )


__all__ = [
    "centred_load",
    "load_balancing_surrogate",
    "load_bound",
    "router_load_and_probs",
]
