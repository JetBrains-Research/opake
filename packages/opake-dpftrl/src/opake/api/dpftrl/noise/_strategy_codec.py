"""Factory-arg serialisation codec for MF strategies.

Strategies are pure recipes — frozen dataclasses carrying only their
factory arguments.  Serialization dumps every dataclass field by name
under a ``type`` discriminator; deserialization calls the matching
strategy factory with that dict.

Each strategy class registers itself via :func:`register_strategy`,
which indexes the class by name (used by :class:`MfGaussian`'s custom
serializer to resolve a strategy sub-dict back to its class) and
installs a ``state_dict`` / ``from_state_dict`` pair on the universal
:mod:`opake.serialization` registry so direct ``state_dict(strategy)``
works too.

Recipe-typed fields (for example the supported ``lr_schedule`` fields on
BandMF and BLT carrying a :mod:`opake.scheduling` recipe dataclass) round-trip
through a tagged sub-dict
``{"__opake_recipe__": "CosineSchedule", **fields}`` so the deserializer can
reconstruct the right subtype.  A user
passing a raw lambda for ``lr_schedule`` still raises — only
registered recipe classes round-trip cleanly.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import torch

from opake.exceptions import CheckpointError

#: Strategy-class registry, keyed by class name.  Populated by
#: :func:`register_strategy` at import time of each strategy module.
_STRATEGY_REGISTRY: dict[str, type] = {}

#: ``ClsName → factory_callable`` map.  Populated lazily.
_STRATEGY_FACTORIES: dict[str, Any] = {}

#: Magic key marking a tagged recipe sub-dict (Schedule, etc.) on the
#: wire.  The presence of this key in a dict-valued strategy field
#: tells :func:`deserialize_strategy` to reconstruct the recipe via
#: :func:`_resolve_recipe_class` rather than passing the dict through.
_RECIPE_TAG: str = "__opake_recipe__"


def _resolve_factory(cls_name: str):
    """Return the factory function for the given strategy class name."""
    if cls_name in _STRATEGY_FACTORIES:
        return _STRATEGY_FACTORIES[cls_name]
    from opake.api.dpftrl import noise as _noise

    factory_name = _factory_name_for(cls_name)
    factory = getattr(_noise, factory_name, None)
    if factory is None:
        raise CheckpointError(
            *(f"No factory function {factory_name!r} found for strategy {cls_name!r}",)
        )
    _STRATEGY_FACTORIES[cls_name] = factory
    return factory


def _factory_name_for(cls_name: str) -> str:
    """Map ``ClsName`` → ``snake_case_strategy`` factory name."""
    base = cls_name
    base = base.removesuffix("Strategy")
    if not base:
        raise CheckpointError(*(f"unexpected strategy class name: {cls_name!r}",))
    out_chars: list[str] = []
    for i, ch in enumerate(base):
        if ch.isupper():
            if i > 0:
                out_chars.append("_")
            out_chars.append(ch.lower())
        else:
            out_chars.append(ch)
    return "".join(out_chars) + "_strategy"


def _resolve_recipe_class(name: str) -> type | None:
    """Look up a registered recipe class by name.

    Currently scoped to :mod:`opake.scheduling.types` (where the
    Schedule recipe dataclasses are re-exported for ``isinstance``
    use); can be extended to other recipe namespaces as more callable
    dataclass families appear.  Returns ``None`` if the name is not
    found.
    """
    from opake.scheduling import types as _scheduling_types

    return getattr(_scheduling_types, name, None)


def _to_wire(value: Any) -> Any:
    """Coerce a single field value to a JSON/state-dict-friendly form.

    Callable recipe dataclasses (Schedule recipes from
    :mod:`opake.scheduling`) round-trip via a tagged nested sub-dict;
    raw callables (lambdas, free functions) raise — they have no
    recipe to serialize and must be re-supplied on the receiving side.

    Nesting is structured (not flat dotted paths) so multi-level
    recipes — e.g. :class:`~opake.scheduling.WithWarmup` wrapping a
    :class:`~opake.scheduling.CosineSchedule` — round-trip with
    recursive ``_to_wire`` / :func:`_from_wire` calls.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, tuple):
        return list(value)
    if callable(value) and not isinstance(value, type):
        cls = _resolve_recipe_class(type(value).__name__)
        if cls is type(value):
            from dataclasses import fields as _fields

            payload: dict[str, Any] = {_RECIPE_TAG: type(value).__name__}
            for f in _fields(value):
                payload[f.name] = _to_wire(getattr(value, f.name))
            return payload
        raise CheckpointError(
            *(
                "Cannot serialize a callable strategy field "
                f"(type={type(value).__name__}).  Pass an opake.scheduling "
                "recipe (e.g. cosine_schedule(...)) instead of a raw "
                "function/lambda, or re-supply the callable to the strategy "
                "factory after deserialization.",
            )
        )
    return value


def _from_wire(value: Any) -> Any:
    """Inverse of :func:`_to_wire` — unpack tagged recipes back into instances.

    Plain (non-tagged) values pass through unchanged.  Tagged recipes
    are reconstructed by class-name lookup; recursive ``_from_wire``
    handles nested recipes (e.g. a Schedule inside a WithWarmup).
    """
    if isinstance(value, dict) and _RECIPE_TAG in value:
        cls_name = value[_RECIPE_TAG]
        cls = _resolve_recipe_class(cls_name)
        if cls is None:
            raise CheckpointError(
                *(
                    f"Unknown recipe class {cls_name!r} on strategy field "
                    "(was the schedule defined in opake.scheduling?)",
                )
            )
        kwargs = {k: _from_wire(v) for k, v in value.items() if k != _RECIPE_TAG}
        return cls(**kwargs)
    return value


def serialize_strategy(s: Any) -> dict[str, Any]:
    """Emit ``{"type": ClsName, **factory_args}`` for the strategy."""
    out: dict[str, Any] = {"type": s.__class__.__name__}
    for f in fields(s):
        out[f.name] = _to_wire(getattr(s, f.name))
    return out


def deserialize_strategy(sd: dict[str, Any]) -> Any:
    """Deserialize a strategy dict by calling its factory.

    Tagged recipe sub-dicts (Schedule recipes etc.) are reconstructed
    via :func:`_from_wire` before reaching the factory so callers don't
    need to know about the wire format.
    """
    sd = dict(sd)
    t = sd.pop("type")
    cls = _STRATEGY_REGISTRY.get(t)
    if cls is None:
        raise CheckpointError(
            *(
                f"Unknown strategy type: {t!r} (registered: {sorted(_STRATEGY_REGISTRY)!r})",
            )
        )
    factory = _resolve_factory(t)
    kwargs = {k: _from_wire(v) for k, v in sd.items()}
    return factory(**kwargs)


def register_strategy(cls: type) -> type:
    """Register a strategy dataclass for serialization."""
    _STRATEGY_REGISTRY[cls.__name__] = cls
    from opake.serialization import register_serializer

    register_serializer(
        cls,
        lambda obj: serialize_strategy(obj),
        lambda _template, sd: deserialize_strategy(dict(sd)),
    )
    return cls


__all__ = [
    "deserialize_strategy",
    "register_strategy",
    "serialize_strategy",
]
