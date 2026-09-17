# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Top-k router helpers for stacked-expert MoE families.

:func:`moe_geometry` reads ``top_k``, ``num_experts`` and the number of
routed layers off a model as a mapping that unpacks into
:func:`opaque.dpsgd.clipping.moe_clipped_grad`.  :func:`install_fp32_router`
is the opt-in fp32-logit router swap: the stock router computes its logits
in the hidden-state dtype, so under bf16 they carry exact ties and the top-k
set recovered from the logits can differ from the executed one.  The swap is
an instance-level binding that :func:`remove_fp32_router` undoes.
"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, TypedDict

import torch
import torch.nn.functional as F

from opaque.exceptions import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch.nn as nn

_PREVIOUS_FORWARD_ATTR = "_opaque_fp32_router_previous_forward"
_ROUTER_ATTRS = ("top_k", "num_experts", "norm_topk_prob", "weight")


class MoeGeometry(TypedDict):
    """Routing geometry of a MoE model; unpacks into ``moe_clipped_grad``."""

    top_k: int
    num_experts: int
    num_layers: int


def is_router_module(module: nn.Module) -> bool:
    """Whether ``module`` is a stacked-expert top-k router.

    Matches a class name containing ``"TopKRouter"`` or the stock router
    attributes (``top_k``, ``num_experts``, ``norm_topk_prob``, ``weight``).
    """
    if "TopKRouter" in type(module).__name__:
        return True
    return all(hasattr(module, attr) for attr in _ROUTER_ATTRS)


def moe_geometry(model: nn.Module) -> MoeGeometry:
    """Read ``(top_k, num_experts, num_layers)`` off a MoE model.

    ``num_layers`` counts the router modules; ``top_k`` and ``num_experts``
    come from the routers and fall back to ``model.config``.

    Raises:
        ConfigurationError: no top-k router, routers that disagree, or
            neither routers nor config exposing the values.
    """
    routers = [module for module in model.modules() if is_router_module(module)]
    if not routers:
        raise ConfigurationError(
            *(
                f"{type(model).__name__} has no top-k router module; the MoE "
                "router-load release needs a mixture-of-experts model whose "
                "backbone records router logits.",
            )
        )
    shapes = {
        (getattr(r, "num_experts", None), getattr(r, "top_k", None)) for r in routers
    }
    if len(shapes) != 1:
        raise ConfigurationError(
            *(f"routers disagree on (num_experts, top_k): {sorted(shapes)}.",)
        )
    num_experts, top_k = next(iter(shapes))
    config = getattr(model, "config", None)
    if num_experts is None:
        num_experts = getattr(config, "num_experts", None)
        if num_experts is None:
            num_experts = getattr(config, "num_local_experts", None)
    if top_k is None:
        top_k = getattr(config, "num_experts_per_tok", None)
    if not num_experts or not top_k:
        raise ConfigurationError(
            *(
                f"{type(model).__name__}: neither the router modules nor the "
                "config expose (num_experts, top_k).",
            )
        )
    return MoeGeometry(
        top_k=int(top_k), num_experts=int(num_experts), num_layers=len(routers)
    )


def _fp32_router_forward(self, hidden_states: torch.Tensor):
    """fp32-logit router forward with the stock ``(logits, scores, indices)`` contract."""
    hidden_dim = self.weight.shape[-1]
    hidden_states = hidden_states.reshape(-1, hidden_dim)
    router_logits = F.linear(hidden_states.float(), self.weight.float())
    router_probs = torch.softmax(router_logits, dim=-1)
    router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
    if self.norm_topk_prob:
        router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
    router_scores = router_top_value.to(hidden_states.dtype)
    return router_logits, router_scores, router_indices


_fp32_router_forward.__opaque_fp32_router__ = True  # type: ignore[attr-defined]


def _is_router(module: nn.Module, router_cls: type | None) -> bool:
    if router_cls is not None:
        return type(module) is router_cls
    return is_router_module(module)


def _has_fp32_router(module: nn.Module) -> bool:
    forward = module.__dict__.get("forward")
    return getattr(forward, "__func__", None) is _fp32_router_forward


def install_fp32_router(
    model: nn.Module, *, router_cls: type | None = None
) -> Callable[[], None]:
    """Bind the fp32-logit forward on every router module of ``model``.

    Routers are matched by ``router_cls`` when given, otherwise by
    :func:`is_router_module`.  Idempotent per module.

    Returns:
        A callable that removes the swap from ``model`` again.

    Raises:
        ConfigurationError: when no module of ``model`` matches, so a
            requested fp32 router never silently installs nothing.
    """
    installed = 0
    for module in model.modules():
        if not _is_router(module, router_cls):
            continue
        installed += 1
        if _has_fp32_router(module):
            continue
        previous = module.__dict__.get("forward")
        module.__dict__[_PREVIOUS_FORWARD_ATTR] = previous
        module.forward = types.MethodType(_fp32_router_forward, module)
    if installed == 0:
        wanted = router_cls.__name__ if router_cls is not None else "a top-k router"
        raise ConfigurationError(
            *(
                f"install_fp32_router: {type(model).__name__} has no module "
                f"matching {wanted}; the fp32 router (router_fp32=True) has "
                "nothing to install on this model.",
            )
        )
    return lambda: remove_fp32_router(model)


def remove_fp32_router(model: nn.Module) -> None:
    """Undo :func:`install_fp32_router` on ``model`` (no-op when not installed)."""
    for module in model.modules():
        if not _has_fp32_router(module):
            continue
        previous = module.__dict__.pop(_PREVIOUS_FORWARD_ATTR, None)
        if previous is None:
            del module.__dict__["forward"]
        else:
            module.__dict__["forward"] = previous


def has_fp32_router(model: nn.Module) -> bool:
    """``True`` when at least one router module of ``model`` carries the swap."""
    return any(_has_fp32_router(module) for module in model.modules())


__all__ = [
    "MoeGeometry",
    "has_fp32_router",
    "install_fp32_router",
    "is_router_module",
    "moe_geometry",
    "remove_fp32_router",
]
