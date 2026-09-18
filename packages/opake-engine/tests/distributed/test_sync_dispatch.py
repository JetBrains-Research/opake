"""Fail-closed ``sync()`` dispatch: MRO resolution + raise on unregistered.

The dispatcher used to require an exact-type registry key, so every
``ClippedGradAux`` / ``ClipState`` subclass that did not register itself by
hand (notably the AUTO-S family) hit a ``TypeError`` on the only public sync
entry point.  Resolution now walks ``__mro__``; an unregistered type still
raises rather than silently passing through unsynchronized.
"""

from __future__ import annotations

import pytest

# Importing the clipping package runs its __init__, which imports
# ``_distributed`` and registers the marker / aux sync handlers.  Import it
# explicitly so the registry-resolution assertions below do not depend on any
# other test having called ``sync()`` first (test order is not guaranteed).
import opake.api.engine.clipping  # noqa: F401
from opake.api.engine.clipping._auto import (
    AutoClippedFunAux,
    AutoClippedGradAux,
    AutoClipState,
)
from opake.api.engine.clipping._clipped_fun import ClippedFunAux, FixedClipState
from opake.api.engine.clipping._clipped_grad import ClippedGradAux
from opake.api.engine.distributed._state import (
    _resolve_sync_fn,
    register_sync_type,
    sync,
)
from opake.distributed import sync as facade_sync


def test_auto_s_states_dispatch_single_process() -> None:
    # Single-process sync is an identity, but it must reach a handler at all.
    for obj in (
        AutoClipState(),
        AutoClippedFunAux(),
        AutoClippedGradAux(),
        FixedClipState(),
    ):
        assert type(facade_sync(obj)) is type(obj)


def test_auto_state_resolves_to_marker_handler() -> None:
    assert _resolve_sync_fn(AutoClipState) is not None
    assert _resolve_sync_fn(AutoClippedGradAux) is not None


def test_subclass_resolves_to_registered_base() -> None:
    class _CustomGradAux(ClippedGradAux):
        pass

    # Not registered directly; must resolve through the ClippedGradAux base.
    assert _resolve_sync_fn(_CustomGradAux) is _resolve_sync_fn(ClippedGradAux)
    assert type(sync(_CustomGradAux())) is _CustomGradAux


def test_unregistered_type_raises() -> None:
    class _Nope:
        pass

    with pytest.raises(TypeError, match="No sync function registered for _Nope"):
        sync(_Nope())


def test_exact_registration_wins_over_base() -> None:
    class _Special(ClippedFunAux):
        pass

    sentinel = object()

    def _handler(_obj: object) -> object:
        return sentinel

    register_sync_type(_Special, _handler)
    try:
        assert _resolve_sync_fn(_Special) is _handler
    finally:
        from opake.api.engine.distributed._state import _SYNC_REGISTRY

        _SYNC_REGISTRY.pop(_Special, None)
