"""Generic Python container walker (no third-party deps).

Handles the language-level container types — dataclasses, NamedTuples,
tuples, lists, mappings, primitives — that are universally serializable
without knowing anything about the leaf type. Specific concrete leaf
types (``torch.Tensor``, ``numpy.ndarray``, custom state objects) live
in the registry instead, so the dispatcher consults the registry first
and falls back here for anything generic.

Keeping this module torch-free is what allows ``opake-base`` to ship
without a torch dependency. The torch and NumPy leaf handlers are
registered by ``opake-engine`` and pure ndarray handlers are also
needed by ``opake-accounting``; both register against the registry
on import.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any, cast

WalkSave = Callable[[Any, str, dict[str, Any]], None]
WalkLoad = Callable[[Any, Mapping[str, Any], str], Any]

# Python primitives we serialise verbatim. Anything that is not a
# registered type AND not one of these AND not a generic container is
# unrecognised, and an unrecognised leaf is an error rather than a
# silent drop — see :func:`_unrecognized_leaf`.
_PRIMITIVES = (int, float, bool, str, type(None))


def _is_named_tuple_instance(obj: Any) -> bool:
    return isinstance(obj, tuple) and hasattr(obj, "_fields")


def _unrecognized_leaf(obj: Any, prefix: str, verb: str) -> TypeError:
    """Build the fail-closed error for a leaf no handler claims."""
    where = f"at {prefix!r}" if prefix else "at the root"
    return TypeError(
        f"Cannot {verb} {type(obj).__module__}.{type(obj).__qualname__} {where}: "
        "it is not a registered type, not a generic container "
        "(dataclass / NamedTuple / tuple / list / mapping) and not a primitive. "
        "Register a handler with `register_serializer`, or — if the value "
        "carries no run state and the template reproduces it — declare it "
        "inert with `register_template_restored`."
    )


def walk_save(state: Any, prefix: str, out: dict[str, Any], recurse: WalkSave) -> None:
    """Save generic Python containers; unrecognised leaves raise ``TypeError``."""
    if dataclasses.is_dataclass(state) and not isinstance(state, type):
        for f in dataclasses.fields(state):
            sub = f"{prefix}.{f.name}" if prefix else f.name
            recurse(getattr(state, f.name), sub, out)
        return
    if _is_named_tuple_instance(state):
        named_tuple = cast("Any", state)
        for f in named_tuple._fields:
            sub = f"{prefix}.{f}" if prefix else f
            recurse(getattr(state, f), sub, out)
        return
    if isinstance(state, tuple):
        for i, v in enumerate(state):
            recurse(v, f"{prefix}[{i}]", out)
        return
    if isinstance(state, list):
        for i, v in enumerate(state):
            recurse(v, f"{prefix}[{i}]", out)
        return
    if isinstance(state, Mapping):
        for k, v in state.items():
            sub = f"{prefix}.{k}" if prefix else str(k)
            recurse(v, sub, out)
        return
    if isinstance(state, _PRIMITIVES):
        out[prefix] = state
        return
    raise _unrecognized_leaf(state, prefix, "serialize")


def walk_load(
    template: Any,
    sd: Mapping[str, Any],
    prefix: str,
    recurse: WalkLoad,
) -> Any:
    """Rebuild generic Python containers from ``sd``; unrecognised leaves raise."""
    if dataclasses.is_dataclass(template) and not isinstance(template, type):
        replacements: dict[str, Any] = {}
        for f in dataclasses.fields(template):
            sub = f"{prefix}.{f.name}" if prefix else f.name
            replacements[f.name] = recurse(getattr(template, f.name), sd, sub)
        return dataclasses.replace(template, **replacements)
    if _is_named_tuple_instance(template):
        fields = cast("Any", template)._fields
        return type(template)(
            *(
                recurse(getattr(template, f), sd, f"{prefix}.{f}" if prefix else f)
                for f in fields
            )
        )
    if isinstance(template, tuple):
        return tuple(recurse(v, sd, f"{prefix}[{i}]") for i, v in enumerate(template))
    if isinstance(template, list):
        return [recurse(v, sd, f"{prefix}[{i}]") for i, v in enumerate(template)]
    if isinstance(template, Mapping):
        return {
            k: recurse(v, sd, f"{prefix}.{k}" if prefix else str(k))
            for k, v in template.items()
        }
    if isinstance(template, _PRIMITIVES):
        return sd.get(prefix, template)
    raise _unrecognized_leaf(template, prefix, "restore")


__all__ = ["WalkLoad", "WalkSave", "walk_load", "walk_save"]
