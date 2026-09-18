"""Serialization registry, dispatcher, and contract types.

This is the foundation seam every other ``opake-*`` wheel uses: each
wheel registers its concrete leaf types with the registry on import.
The dispatcher consults the registry first (exact type, then
``__mro__``), then falls back to a generic Python container walker
(dataclass / NamedTuple / tuple / list / mapping / primitives). A leaf no
handler claims raises ``TypeError``; use
:func:`register_template_restored` to declare one intentionally inert.

User-facing entry points are re-exported on the ``opake.serialization``
façade.
"""

from __future__ import annotations

from opake.api.base.serialization._dispatch import (
    from_state_dict,
    state_dict,
)
from opake.api.base.serialization._registry import (
    lookup_serializer,
    register_serializer,
    register_template_restored,
    resolve_serializer,
)
from opake.api.base.serialization._types import (
    FromStateDictFn,
    SerializedState,
    Serializer,
    StateDictFn,
)

__all__ = [
    # Dispatcher
    "state_dict",
    "from_state_dict",
    # Registry
    "register_serializer",
    "register_template_restored",
    "lookup_serializer",
    "resolve_serializer",
    # Contract types
    "Serializer",
    "SerializedState",
    "StateDictFn",
    "FromStateDictFn",
]
