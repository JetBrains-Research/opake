"""Callable contracts shared by clipping factories and public type façades."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

from opake.api.engine.types import ClippedPytree, SecondMomentClippingOutput

if TYPE_CHECKING:
    from opake.api.engine.clipping._clipped_fun import ClippingStats, FixedClipState
    from opake.api.engine.clipping._clipped_grad import ClippedGradAux


ClippedGradValue: TypeAlias = ClippedPytree | SecondMomentClippingOutput
"""Private gradient output before optional diagnostics are attached."""

ClippedGradResult: TypeAlias = (
    ClippedGradValue
    | tuple[ClippedGradValue, "ClippedGradAux"]
    | tuple[ClippedGradValue, "ClippingStats"]
)
"""Value returned by :class:`ClippedGradFn` before its updated state."""


class ClippedGradFn(Protocol):
    """Callable returned by :func:`opake.dpsgd.clipping.clipped_grad`.

    The positional arguments mirror the wrapped loss function. ``state`` is
    passed by keyword and every invocation returns a new immutable
    :class:`FixedClipState` with the clipped gradient result.
    """

    def __call__(
        self,
        *args: Any,
        state: FixedClipState,
        **kwargs: Any,
    ) -> tuple[ClippedGradResult, FixedClipState]: ...


__all__ = ["ClippedGradFn", "ClippedGradResult"]
