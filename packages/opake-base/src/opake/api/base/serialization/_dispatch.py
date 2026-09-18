"""Dispatcher: registry-first, generic-structural fallback.

For each visited node the dispatcher consults the registry by exact
type, then by ``__mro__`` so subclasses reach their base class handler
(``nn.Parameter`` resolves to the ``torch.Tensor`` pair). On a miss the
generic structural walker handles the node if it is a container or a
primitive, and raises ``TypeError`` otherwise — an unrecognised leaf is
never dropped silently. An error raised by a registered handler on load
is annotated with the key being restored.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import _structural
from ._registry import resolve_serializer

if TYPE_CHECKING:
    from collections.abc import Mapping


def _join_path(prefix: str, rel_key: str) -> str:
    """Path join matching the structural walker's conventions.

    Dot segments separate field names; bracket-segments (``[i]``) attach
    directly without a leading dot. An empty ``rel_key`` collapses to
    ``prefix`` itself (a registered handler may emit ``{"": value}`` to
    occupy the prefix slot — used by leaf handlers like ``torch.Tensor``).
    """
    if not prefix:
        return rel_key
    if not rel_key:
        return prefix
    if rel_key.startswith("["):
        return prefix + rel_key
    return f"{prefix}.{rel_key}"


def _subdict(sd: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    """Slice ``sd`` to keys nested under ``prefix`` (dot + bracket segments)."""
    if not prefix:
        return dict(sd)
    out: dict[str, Any] = {}
    plen = len(prefix)
    for k, v in sd.items():
        if k == prefix:
            out[""] = v
        elif k.startswith(prefix) and len(k) > plen:
            sep = k[plen]
            if sep == ".":
                out[k[plen + 1 :]] = v
            elif sep == "[":
                out[k[plen:]] = v
    return out


def _walk_save(obj: Any, prefix: str, out: dict[str, Any]) -> None:
    handlers = resolve_serializer(type(obj))
    if handlers is not None:
        rel = handlers[0](obj)
        for rk, rv in rel.items():
            out[_join_path(prefix, rk)] = rv
        return
    _structural.walk_save(obj, prefix, out, _walk_save)


def _walk_load(template: Any, sd: Mapping[str, Any], prefix: str) -> Any:
    handlers = resolve_serializer(type(template))
    if handlers is not None:
        try:
            return handlers[1](template, _subdict(sd, prefix))
        except Exception as exc:
            # Handlers see only their slice, so only the dispatcher can
            # name the leaf that failed.
            exc.add_note(
                f"while restoring {prefix!r}" if prefix else "while restoring the root"
            )
            raise
    return _structural.walk_load(template, sd, prefix, _walk_load)


def state_dict(obj: Any) -> dict[str, Any]:
    """Serialise ``obj`` to a flat ``dict[str, Any]``.

    Registered types use their exact-type handler; everything else is
    walked structurally (dataclass, NamedTuple, tuple, list, mapping,
    primitives). Opaque non-containers are dropped — the load is
    template-driven and reads them back from the template.
    """
    out: dict[str, Any] = {}
    _walk_save(obj, "", out)
    return out


def from_state_dict(template: Any, sd: Mapping[str, Any]) -> Any:
    """Rebuild from ``sd`` using ``template`` for shape and omitted leaves.

    For types registered with :func:`register_serializer`, the
    ``template`` argument selects the handler; handlers may ignore it
    when the dict is self-describing (e.g. ``DpProcess`` subclasses).
    Generic Python container shapes come from the template; missing
    leaves keep the template's value (forward compatibility when new
    fields appear).
    """
    return _walk_load(template, sd, "")


__all__ = ["from_state_dict", "state_dict"]
