"""Functional serialization for Opake explicit state objects.

Flattens tensor trees, NumPy arrays, dataclasses, named tuples,
sequences, and string-keyed dicts into a flat ``dict[str, Any]``
suitable for ``torch.save`` / ``torch.load``.

Restore is template-driven: supply a freshly-initialised object of the
same shape as at save time; each path present in the dict overwrites
the corresponding leaf. Missing paths keep the template (forward
compatibility when new fields appear).

Sub-packages may register custom serializers with
:func:`register_serializer`. Lookup is by exact type and then by
``__mro__``, so a subclass of a registered type (a custom
``torch.Tensor`` subclass, say) uses the nearest base class handler.
``torch.Tensor``, ``numpy.ndarray``, and ``torch.nn.Parameter``
handlers register automatically when ``opake-engine`` is loaded;
``nn.Parameter`` gets its own exact-type handler (preserving the
subclass and ``requires_grad``) rather than relying on the ``__mro__``
fallback. ``opake-accounting`` registers PLD process types; stack
wheels register their state objects.

A leaf that is neither registered nor a generic container nor a
primitive raises ``TypeError`` on both save and restore rather than
being dropped. Declare the genuinely inert ones — vendor structure
handles and the like — with :func:`register_template_restored`.
"""

from __future__ import annotations

from opake.api.base.serialization import (
    FromStateDictFn,
    SerializedState,
    Serializer,
    StateDictFn,
    from_state_dict,
    lookup_serializer,
    register_serializer,
    register_template_restored,
    resolve_serializer,
    state_dict,
)

__all__ = [
    "FromStateDictFn",
    "SerializedState",
    "Serializer",
    "StateDictFn",
    "from_state_dict",
    "lookup_serializer",
    "register_serializer",
    "register_template_restored",
    "resolve_serializer",
    "state_dict",
]
