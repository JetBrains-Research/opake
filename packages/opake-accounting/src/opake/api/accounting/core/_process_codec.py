"""Internal nested codec for :class:`DpProcess` (used by :mod:`opake.serialization`).

Each process serializes as a nested dict carrying its concrete-class name in
the ``type`` field; inner :class:`DpProcess` fields recurse via the same
codec.  Non-``DpProcess`` field values are emitted only for primitives and
plain containers (tuple, list) — opaque values (e.g. ``torch.Tensor``)
are skipped on save and restored to the class default on load.

The codec knows only about :class:`DpProcess`.  Mechanisms that wrap
domain-specific sub-objects (strategies, encoders, …) register their own
custom ``(state_dict_fn, from_state_dict_fn)`` with
:func:`opake.serialization.register_serializer`; that registration
replaces the generic codec for that class.
"""

from __future__ import annotations

import dataclasses
from dataclasses import fields
from typing import Any

from opake.api.accounting.core._base import DpProcess
from opake.exceptions import CheckpointError

_PRIMITIVE_TYPES = (int, float, bool, str, type(None))


def _serialize_dp_process(p: Any) -> dict[str, Any]:
    """Serialize a DpProcess (or nested DpProcess sub-object) to a nested dict.

    Each field is dispatched by value:

    - Composition wrappers (``Composed`` / ``Repeated`` / ``CachedProcess``)
      → walked with an explicit stack: these are the only types that form
      unbounded spines (one node per training step), and a recursive walk
      overflows at a few hundred steps.  They have no custom serializers,
      so inlining them is wire-identical to the dispatch path.
    - Other nested ``DpProcess`` (depth-bounded mechanism internals) →
      dispatch via :mod:`opake.serialization` so custom serializers
      (e.g. ``MfGaussian``'s strategy-aware one) fire.
    - Primitive / tuple / list → emit verbatim.
    - Anything else → skip (load restores the class default).
    """
    from opake.api.accounting.core.composition.types import (
        CachedProcess,
        Composed,
        Repeated,
    )
    from opake.serialization import state_dict as opake_state_dict

    wrappers = (Composed, Repeated, CachedProcess)
    root: dict[str, Any] = {}
    stack: list[tuple[Any, dict[str, Any]]] = [(p, root)]
    while stack:
        node, out = stack.pop()
        out["type"] = node.__class__.__name__
        for f in fields(node):
            v = getattr(node, f.name)
            # Exact-type check: a hypothetical wrapper *subclass* must take
            # the dispatch path below so any custom serializer it registers
            # still fires (and its wire ``type`` name round-trips).
            if type(v) in wrappers:
                slot: dict[str, Any] = {}
                out[f.name] = slot
                stack.append((v, slot))  # iterate, don't recurse
            elif isinstance(v, DpProcess):
                out[f.name] = dict(opake_state_dict(v))
            elif isinstance(v, (_PRIMITIVE_TYPES, tuple, list)):
                out[f.name] = v
            # else: opaque — skip
    return root


def _generic_state_dict(obj: Any) -> dict[str, Any]:
    """The registry entry every ``DpProcess`` subclass gets by default."""
    return _serialize_dp_process(obj)


def _generic_from_state_dict(_template: Any, sd: Any) -> Any:
    """The registry entry every ``DpProcess`` subclass gets by default."""
    return _load_dp_process(dict(sd))


#: Composition-wrapper wire types and their DpProcess-typed field names —
#: the only shapes that form unbounded spines and need iterative loading.
_WRAPPER_CHILD_FIELDS: dict[str, tuple[str, ...]] = {
    "Composed": ("left", "right"),
    "Repeated": ("inner",),
    "CachedProcess": ("inner",),
}


def _load_dp_process(sd: dict[str, Any]) -> Any:
    """Deserialize a nested ``{"type": ..., ...}`` dict produced by
    :func:`_serialize_dp_process`.

    Composition wrappers are rebuilt with an explicit post-order stack
    (unbounded spine depth); everything else takes the depth-bounded
    :func:`_load_leaf` path, where nested ``{"type": X, ...}`` sub-dicts
    dispatch through the universal :mod:`opake.serialization` registry so
    custom serializers fire.
    """
    from opake.api.accounting.core._base import _PROCESS_REGISTRY
    from opake.serialization import lookup_serializer

    built: dict[int, Any] = {}
    stack: list[tuple[dict[str, Any], bool]] = [(sd, False)]
    while stack:
        node, expanded = stack.pop()
        t = node.get("type")
        if t is None:
            raise CheckpointError(
                *("missing required field 'type' for serialized DpProcess",)
            )
        child_fields = _WRAPPER_CHILD_FIELDS.get(t)
        if child_fields is None:
            # Non-wrapper node.  A class with a CUSTOM registered serializer
            # (e.g. MfGaussian's strategy-aware one) must load through it —
            # its wire format is not the generic field layout.  Everything
            # else takes the depth-bounded generic leaf path.
            cls = _PROCESS_REGISTRY.get(t)
            pair = lookup_serializer(cls) if cls is not None else None
            if pair is not None and pair[1] is not _generic_from_state_dict:
                built[id(node)] = pair[1](cls.__new__(cls), dict(node))
            else:
                built[id(node)] = _load_leaf(node)
            continue
        if not expanded:
            stack.append((node, True))
            for k in child_fields:
                child = node.get(k)
                if child is None and k not in node:
                    continue  # missing: handled by the defaults pass below
                if not isinstance(child, dict):
                    raise CheckpointError(
                        *(
                            f"field {k!r} of {t} must be a serialized DpProcess "
                            f"dict, got {type(child).__name__}",
                        )
                    )
                stack.append((child, False))
            continue
        cls = _PROCESS_REGISTRY[t]
        dataclass_cls: Any = cls
        field_names = {f.name for f in fields(dataclass_cls)}
        extra = set(node) - {"type", *field_names}
        if extra:
            raise CheckpointError(
                *(f"unexpected keys for {cls.__name__}: {sorted(extra)!r}",)
            )
        kwargs: dict[str, Any] = {}
        for f in fields(dataclass_cls):
            if f.name in node:
                raw = node[f.name]
                if f.name in child_fields:
                    # Lookup, not pop: an aliased sub-dict (the same object
                    # appearing as two children) must serve both parents.
                    kwargs[f.name] = built[id(raw)]
                else:
                    kwargs[f.name] = _coerce_field(cls, f.name, raw)
            elif f.default is not dataclasses.MISSING:
                kwargs[f.name] = f.default
            elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                kwargs[f.name] = f.default_factory()
            else:
                raise CheckpointError(
                    *(f"missing required field {f.name!r} for {cls.__name__}",)
                )
        built[id(node)] = dataclass_cls(**kwargs)
    return built[id(sd)]


def _load_leaf(sd: dict[str, Any]) -> Any:
    """Depth-bounded load of a non-wrapper node (the original recursive path)."""
    from opake.api.accounting.core._base import _PROCESS_REGISTRY
    from opake.serialization import from_state_dict as opake_from_state_dict

    sd = dict(sd)
    t = sd.pop("type")
    cls = _PROCESS_REGISTRY.get(t)
    if cls is None:
        raise CheckpointError(*(f"Unknown DpProcess type: {t}",))
    if not dataclasses.is_dataclass(cls):
        raise CheckpointError(*(f"DpProcess type {t!r} is not a dataclass",))

    dataclass_cls: Any = cls
    kwargs: dict[str, Any] = {}
    for f in fields(dataclass_cls):
        if f.name in sd:
            raw = sd.pop(f.name)
            if isinstance(raw, dict) and "type" in raw:
                inner_cls = _PROCESS_REGISTRY.get(raw["type"])
                if inner_cls is None:
                    raise CheckpointError(
                        *(
                            f"Unknown nested DpProcess type {raw['type']!r} "
                            f"in field {f.name!r} of {cls.__name__}",
                        )
                    )
                template = inner_cls.__new__(inner_cls)
                kwargs[f.name] = opake_from_state_dict(template, raw)
            else:
                kwargs[f.name] = _coerce_field(cls, f.name, raw)
        elif f.default is not dataclasses.MISSING:
            kwargs[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            kwargs[f.name] = f.default_factory()
        else:
            raise CheckpointError(
                *(f"missing required field {f.name!r} for {cls.__name__}",)
            )

    if sd:
        raise CheckpointError(*(f"unexpected keys for {cls.__name__}: {sorted(sd)!r}",))

    return dataclass_cls(**kwargs)


def _coerce_field(cls: type[Any], fname: str, val: Any) -> Any:
    from typing import get_origin, get_type_hints

    try:
        hints = get_type_hints(cls)
    except Exception:
        return val
    t = hints.get(fname)
    if t is None:
        return val
    origin = get_origin(t)
    if origin is tuple and isinstance(val, list):
        return tuple(val)
    return val


__all__: list[str] = []
