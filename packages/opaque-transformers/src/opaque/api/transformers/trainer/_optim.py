"""Optimizer resolution for Trainer.

Two-layer surface, both routed through the same builder:

1. **Canonical opaque names** — every factory in
   :mod:`opaque.optimizers` (``adam``, ``adamw``, ``sgd``, ``rmsprop``,
   ``adagrad``, ``adafactor``, ``ademamix``, ``lion``, ``radam``,
   ``adadelta``, ``schedule_free``).
   The full opaque parameter surface is reachable through HF-canonical
   ``TrainingArguments`` fields (``learning_rate``, ``weight_decay``,
   ``adam_beta1``, ``adam_beta2``, ``adam_epsilon``) plus anything the
   chosen factory accepts via parsed ``optim_args`` (comma-separated
   ``key=value``), including ``noise_bias_correction``, ``decoupled_weight_decay``,
   ``update_rms_clip``, and other opaque optimizer knobs.  Opaque factories
   raise ``TypeError`` on unknown keys.

2. **HF compat aliases** — HF's ``OptimizerNames`` values that map
   cleanly onto an opaque factory (``adamw_torch`` → ``adamw``,
   ``adafactor`` → ``adafactor``, ``lion_32bit`` → ``lion``, …).  These
   accept the same HF fields and route to the same factories.
   Aliases are silently rewritten — Trainer never substitutes a
   different update rule for what the alias names; the HF spelling is
   honoured by selecting the opaque factory whose math matches.

Names that have no DP-aware mapping (8-bit, paged, GaLore, fused
``torch.optim`` subclasses) remain rejected with a redirect message.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import opaque.optimizers as opaque_opt
from opaque.exceptions import ConfigurationError, InputTypeError, OperationError

if TYPE_CHECKING:
    from collections.abc import Callable

GradientTransformation = Any  # torchopt.base.GradientTransformation, lazy

# ---------------------------------------------------------------------------
# Layer 1 — canonical opaque names
# ---------------------------------------------------------------------------

_OPAQUE_FACTORIES: dict[str, Callable[..., GradientTransformation]] = {
    "adam": opaque_opt.adam,
    "adamw": opaque_opt.adamw,
    "sgd": opaque_opt.sgd,
    "lion": opaque_opt.lion,
    "ademamix": opaque_opt.ademamix,
    "adafactor": opaque_opt.adafactor,
    "rmsprop": opaque_opt.rmsprop,
    "adagrad": opaque_opt.adagrad,
    "radam": opaque_opt.radam,
    "adadelta": opaque_opt.adadelta,
    "schedule_free": opaque_opt.schedule_free,
}

# Per-factory: which HF TrainingArguments fields apply, derived from each
# factory's signature.  Adafactor uses its own ``eps_grad`` / ``eps_root``
# pair rather than HF's ``adam_epsilon`` so it's omitted from EPS;
# Adam's signature has no ``decoupled_weight_decay`` (the decoupled
# variant is the separate ``adamw`` factory).
_APPLIES_BETAS = {"adam", "adamw", "ademamix", "lion", "radam"}
_APPLIES_EPS = {"adam", "adamw", "rmsprop", "adagrad", "radam", "adadelta"}
_APPLIES_WEIGHT_DECAY = {
    "adam",
    "adamw",
    "sgd",
    "rmsprop",
    "adagrad",
    "adafactor",
    "ademamix",
    "lion",
    "radam",
    "adadelta",
}

# ---------------------------------------------------------------------------
# Layer 2 — HF compat aliases
# ---------------------------------------------------------------------------

# HF name → (opaque factory key, base kwargs).  ``base kwargs`` apply
# only when the corresponding ``_APPLIES_*`` set contains the canonical
# name.  Use the empty dict for plain rewrites.
_HF_ALIASES: dict[str, tuple[str, dict[str, Any]]] = {
    "adamw_torch": ("adamw", {}),
    "adamw_torch_fused": ("adamw", {}),  # fused kernel ↦ functional path
    "adamw_hf": ("adamw", {}),
    "adafactor": ("adafactor", {}),
    "ademamix": ("ademamix", {}),
    "lion_32bit": ("lion", {}),
    # ``lion`` itself is a canonical opaque name; HF's enum still
    # surfaces it, route through the canonical entry above.
    "schedule_free_radam": ("schedule_free", {"base": "radam"}),
    # DP bias-corrected AdamW: a named shortcut for
    # ``optim="adamw", optim_args="noise_bias_correction=True"``.  Explicit
    # ``optim_args`` still win on merge (see ``resolve_optimizer_name``).
    "adamw-bc": ("adamw", {"noise_bias_correction": True}),
}


def supported_names() -> tuple[str, ...]:
    """Sorted union of canonical opaque names + HF aliases."""
    return tuple(sorted(set(_OPAQUE_FACTORIES) | set(_HF_ALIASES)))


def canonical_optimizer_names() -> tuple[str, ...]:
    """Sorted canonical opaque optimizer names (no HF aliases)."""
    return tuple(sorted(_OPAQUE_FACTORIES))


def normalize_optim(optim: Any) -> str:
    """Coerce HF's ``OptimizerNames`` enum to its string value."""
    return optim.value if hasattr(optim, "value") else str(optim)


def resolve_optimizer_name(optim: Any) -> tuple[str, dict[str, Any]]:
    """Resolve user-supplied ``optim`` to ``(canonical_name, base_kwargs)``.

    Whitelist-based: only canonical opaque names and the HF aliases we
    explicitly translate are accepted. Anything else raises ``ValueError``
    pointing at the supported set. No per-name redirection messages — the
    error lists the whole supported surface so users see their options.
    """
    name = normalize_optim(optim)
    if name in _OPAQUE_FACTORIES:
        return name, {}
    if name in _HF_ALIASES:
        canonical, base = _HF_ALIASES[name]
        return canonical, dict(base)
    raise ConfigurationError(
        *(
            f"optim={optim!r} is not supported by Trainer; "
            f"expected one of {supported_names()}.",
        )
    )


def _apply_top_level_fields(
    canonical: str,
    args: Any,
    kwargs: dict[str, Any],
) -> None:
    """Forward HF ``TrainingArguments`` fields to factory kwargs.

    Mutates ``kwargs`` in place.  Only fields applicable to the chosen
    factory are forwarded; unrelated fields are silently dropped (a
    factory raises TypeError if a user smuggles something via
    ``optim_args``).
    """
    if canonical in _APPLIES_BETAS:
        # ``ademamix`` needs a 3-tuple ``(β₁, β₂, β₃)``; the third
        # parameter is ademamix-specific (slow EMA half-life).  HF
        # surfaces only ``adam_beta1`` / ``adam_beta2`` so we default
        # β₃ to the opaque factory's default; users can override the
        # full tuple via ``optim_args``.
        if canonical == "ademamix":
            kwargs.setdefault("betas", (args.adam_beta1, args.adam_beta2, 0.9999))
        else:
            kwargs.setdefault("betas", (args.adam_beta1, args.adam_beta2))
    if canonical in _APPLIES_EPS:
        kwargs.setdefault("eps", args.adam_epsilon)
    if canonical in _APPLIES_WEIGHT_DECAY:
        kwargs.setdefault("weight_decay", args.weight_decay)


def _build_schedule_free(
    lr_schedule: Any,
    args: Any,
    extra_kwargs: dict[str, Any],
    base_kwargs: dict[str, Any],
) -> GradientTransformation:
    """Construct a ``schedule_free`` wrapper.

    ``optim_args="base=adamw,..."`` selects the inner factory; the
    remaining kwargs are split between the base factory and the
    schedule-free wrapper based on signature.  Missing ``base`` defaults
    to ``adamw`` (HF's ``schedule_free_adamw`` semantic).
    """
    pooled = {**base_kwargs, **extra_kwargs}
    base_name = pooled.pop("base", "adamw")
    if base_name not in _OPAQUE_FACTORIES or base_name == "schedule_free":
        raise ConfigurationError(
            *(
                f"schedule_free base={base_name!r} is not a supported "
                "opaque optimizer; pick from "
                f"{tuple(n for n in _OPAQUE_FACTORIES if n != 'schedule_free')}.",
            )
        )
    # schedule_free's own kwargs (e.g. ``beta``, ``weight_lr_power``)
    # vs. the base factory's kwargs are disambiguated by signature
    # introspection; opaque.optimizers.schedule_free raises on unknown
    # keys, so we hand it everything left after building the base.
    base_factory = _OPAQUE_FACTORIES[base_name]
    base_only = {}
    for key in list(pooled):
        # Base-factory-only knobs we know about.  Everything else
        # passes to schedule_free; if either factory doesn't recognise
        # a key, the TypeError surfaces the typo.
        if key in {
            "betas",
            "eps",
            "weight_decay",
            "decoupled_weight_decay",
            "update_rms_clip",
            "noise_bias_correction",
            "alpha",
            "momentum",
            "dampening",
            "nesterov",
            "beta1",
            "decay_rate",
            "eps_grad",
            "eps_root",
        }:
            base_only[key] = pooled.pop(key)
    # Apply HF TrainingArguments fields to the base factory.
    _apply_top_level_fields(base_name, args, base_only)
    base = base_factory(lr=lr_schedule, **base_only)
    return opaque_opt.schedule_free(base, **pooled)


def build_optimizer(
    args: Any,
    lr_schedule: Any,
    extra_kwargs: dict[str, Any] | None = None,
) -> GradientTransformation:
    """Construct the DP-aware opaque optimizer for ``args.optim``.

    ``extra_kwargs`` typically comes from ``args.optim_args``
    (already normalized to ``dict[str, Any] | None`` by
    :meth:`TrainingArguments.__post_init__`) and takes precedence over
    HF field defaults mapped in ``_apply_top_level_fields``.
    """
    canonical, base_kwargs = resolve_optimizer_name(args.optim)
    extra_kwargs = dict(extra_kwargs or {})
    if canonical == "schedule_free":
        return _build_schedule_free(lr_schedule, args, extra_kwargs, base_kwargs)
    factory = _OPAQUE_FACTORIES[canonical]
    kwargs = {**base_kwargs}
    _apply_top_level_fields(canonical, args, kwargs)
    kwargs.update(extra_kwargs)
    return factory(lr=lr_schedule, **kwargs)


def validate_functional_optimizer_cls_and_kwargs(
    optimizer_cls_and_kwargs: tuple[Any, ...],
) -> tuple[Callable[..., GradientTransformation], dict[str, Any]]:
    """Validate ``(factory, kwargs)`` for Trainer's functional optimizer path.

    The factory must **not** be a :class:`torch.optim.Optimizer` subclass and
    must be callable as ``factory(lr=lr_schedule, **kwargs)`` (same
    convention as :mod:`opaque.optimizers` and torchopt).  The returned
    object must expose ``init`` and ``update`` callables.
    """
    import torch.optim as torch_optim

    if (
        not isinstance(optimizer_cls_and_kwargs, tuple)
        or len(optimizer_cls_and_kwargs) != 2  # noqa: PLR2004 - factory/kwargs pair
    ):
        raise InputTypeError(
            *("optimizer_cls_and_kwargs must be a length-2 tuple (factory, kwargs).",)
        )
    factory, opt_kwargs = optimizer_cls_and_kwargs
    if not isinstance(opt_kwargs, dict):
        raise InputTypeError(
            *(
                "optimizer_cls_and_kwargs[1] must be dict[str, Any]; "
                f"got {type(opt_kwargs)!r}.",
            )
        )
    if isinstance(factory, type) and issubclass(factory, torch_optim.Optimizer):
        raise OperationError(
            *(
                "Trainer.optimizer_cls_and_kwargs rejects torch.optim.Optimizer "
                "subclasses: use a callable that returns a torchopt "
                "GradientTransformation (e.g. opaque.optimizers.adamw).",
            )
        )
    if not callable(factory):
        raise InputTypeError(
            *(
                f"optimizer_cls_and_kwargs[0] must be a callable factory; got {factory!r}.",
            )
        )

    def dummy_lr(_step: int) -> float:
        return 1e-4

    try:
        transform = factory(lr=dummy_lr, **opt_kwargs)
    except TypeError as exc:
        raise OperationError(
            *(
                "optimizer_cls_and_kwargs factory is not compatible with "
                "``factory(lr=lr_schedule, **kwargs)``.  Original error: "
                f"{exc}",
            )
        ) from exc
    init_fn = getattr(transform, "init", None)
    update_fn = getattr(transform, "update", None)
    if not callable(init_fn) or not callable(update_fn):
        raise OperationError(
            *(
                "optimizer_cls_and_kwargs factory must return an object with "
                "callable init and update (torchopt GradientTransformation); "
                f"got {type(transform)!r}.",
            )
        )
    return factory, dict(opt_kwargs)


__all__ = [
    "build_optimizer",
    "canonical_optimizer_names",
    "resolve_optimizer_name",
    "supported_names",
    "validate_functional_optimizer_cls_and_kwargs",
]
