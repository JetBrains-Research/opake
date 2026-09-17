"""Public types for the serialization contract."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

#: Flat ``dict[str, Any]`` keyed by dotted / bracketed paths suitable for
#: ``torch.save`` / ``torch.load`` round-trips. Values are leaf objects
#: (tensors, ndarrays, primitives) — never nested dicts.
SerializedState = dict[str, Any]

#: ``state_dict_fn`` returns relative-key paths under the object root
#: (no leading dot). The dispatcher rewrites them to absolute paths.
StateDictFn = Callable[[Any], dict[str, Any]]

#: ``from_state_dict_fn`` receives a slice of the flat state dict whose
#: keys are relative to the object's root.
FromStateDictFn = Callable[[Any, Mapping[str, Any]], Any]


@runtime_checkable
class Serializer(Protocol):
    """Pair of save / load callables registered with the dispatcher."""

    def state_dict(self, obj: Any) -> dict[str, Any]: ...

    def from_state_dict(self, template: Any, sd: Mapping[str, Any]) -> Any: ...


__all__ = [
    "FromStateDictFn",
    "SerializedState",
    "Serializer",
    "StateDictFn",
]
