"""AUTO-S clipping on the :mod:`opake.dpsgd.clipping` import surface."""

from __future__ import annotations

import opake.dpsgd  # noqa: F401  (registers package)


def test_auto_clipped_grad_is_internal_impl():
    from opake.api.engine.clipping import auto_clipped_grad as internal
    from opake.dpsgd.clipping import auto_clipped_grad as public

    assert internal is public


def test_auto_clipped_grad_root_hoist():
    from opake.dpsgd import auto_clipped_grad as root
    from opake.dpsgd.clipping import auto_clipped_grad as mod

    assert root is mod


def test_auto_clipped_fun_is_internal_impl():
    from opake.api.engine.clipping.fun import auto_clipped_fun as internal
    from opake.dpsgd.clipping.fun import auto_clipped_fun as public

    assert internal is public


def test_auto_types_match_internal():
    from opake.api.engine.clipping.types import (
        AutoClippedFunAux as IntFunAux,
    )
    from opake.api.engine.clipping.types import (
        AutoClippedGradAux as IntGradAux,
    )
    from opake.api.engine.clipping.types import (
        AutoClipState as IntState,
    )
    from opake.dpsgd.clipping.types import (
        AutoClippedFunAux as PubFunAux,
    )
    from opake.dpsgd.clipping.types import (
        AutoClippedGradAux as PubGradAux,
    )
    from opake.dpsgd.clipping.types import (
        AutoClipState as PubState,
    )

    assert IntState is PubState
    assert IntFunAux is PubFunAux
    assert IntGradAux is PubGradAux


def test_auto_state_default_marker_equality():
    from opake.dpsgd.clipping.types import AutoClipState

    assert AutoClipState() == AutoClipState()
