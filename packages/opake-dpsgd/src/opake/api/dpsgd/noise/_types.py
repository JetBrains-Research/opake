"""Callable contracts shared by Gaussian noise and public type façades."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeAlias

from opake.api.engine.types import (
    ClippedPytree,
    NoisedPytree,
    SecondMomentClippingOutput,
    SecondMomentNoiseOutput,
)

if TYPE_CHECKING:
    from opake.api.dpsgd.noise._gaussian import GaussianNoiseState


GaussianNoiseInput: TypeAlias = ClippedPytree | SecondMomentClippingOutput
"""Accepted clipped single-stream or paired first/second-moment values."""

GaussianNoiseOutput: TypeAlias = NoisedPytree | SecondMomentNoiseOutput
"""Noised counterpart to :data:`GaussianNoiseInput`."""


class GaussianNoiseFn(Protocol):
    """Callable returned by :func:`opake.dpsgd.noise.gaussian_noise`.

    It accepts a clipped single or paired stream and returns its noised
    counterpart together with a new immutable :class:`GaussianNoiseState`.
    """

    def __call__(
        self,
        clipped_grads: GaussianNoiseInput,
        state: GaussianNoiseState,
    ) -> tuple[GaussianNoiseOutput, GaussianNoiseState]: ...


__all__ = ["GaussianNoiseFn", "GaussianNoiseInput", "GaussianNoiseOutput"]
