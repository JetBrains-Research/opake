# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Top-k router helpers for stacked-expert MoE families.

Two things live here.  :func:`moe_geometry` reads the routing geometry a
per-example MoE objective needs (``top_k``, ``num_experts`` and the number of
routed layers) off a model, as a plain mapping that unpacks into
:func:`opaque.dpsgd.clipping.moe_clipped_grad`.  :func:`install_fp32_router`
is the opt-in fp32-logit router swap: the stock Hugging Face router computes
its logits with ``F.linear`` in the hidden-state dtype and only the softmax
in fp32, so under bf16 the logits carry exact ties and the top-k set a load
statistic recovers from the logits can differ from the executed one.  The swap
computes the logits in fp32 (the precision Mellum 2.0 was pretrained with), so
ties disappear and the logits handed to the load statistics are the executed
ones.

The swap is an instance-level ``types.MethodType`` binding on each router
module, recorded so it can be removed again in-process; the class-level
forward is never touched.  It changes the executed routing function on the
small fraction of tokens that sit on a bf16 rounding tie; adapters served
through stock HF run bf16 routes, so the swap is opt-in and off by default.
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
    """Routing geometry of a mixture-of-experts model.

    A plain mapping so it unpacks into the keyword parameters of the
    per-example MoE clipper (``moe_clipped_grad(loss_fn, **geometry, ...)``)
    and round-trips through a configuration file unchanged.
    """

    top_k: int
    num_experts: int
    num_layers: int


def is_router_module(module: nn.Module) -> bool:
    """Whether ``module`` is a stacked-expert top-k router.

    The single router predicate of the package: a class name containing
    ``"TopKRouter"`` or the stock router attributes (``top_k``,
    ``num_experts``, ``norm_topk_prob``, ``weight``).  :func:`moe_geometry`
    and the fp32 installer count routers with it, so every consumer sees the
    same modules.
    """
    if "TopKRouter" in type(module).__name__:
        return True
    return all(hasattr(module, attr) for attr in _ROUTER_ATTRS)


def moe_geometry(model: nn.Module) -> MoeGeometry:
    """Read ``(top_k, num_experts, num_layers)`` off a MoE model.

    ``num_layers`` counts the router modules the backbone records logits
    for, so dense layers of a mixed model do not count.  ``top_k`` and
    ``num_experts`` are taken from the routers themselves (the values that
    execute) and fall back to ``model.config`` (``num_experts_per_tok`` and
    ``num_experts`` / ``num_local_experts``) for a router that does not
    expose them.

    Args:
        model: A Hugging Face MoE model whose backbone records router logits.

    Returns:
        A :class:`MoeGeometry` mapping.

    Raises:
        ConfigurationError: for a model without a top-k router, when the
            routers disagree on ``(num_experts, top_k)``, or when neither the
            routers nor the config expose them.
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
