"""Callable contracts shared by Gaussian noise and public type façades."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeAlias

from opaque.api.engine.types import (
    ClippedPytree,
    NoisedPytree,
    SecondMomentNoiseOutput,
)

if TYPE_CHECKING:
    from opaque.api.dpsgd.noise._gaussian import GaussianNoiseState
    from opaque.api.dpsgd.noise._jme import JmeNoiseState


GaussianNoiseInput: TypeAlias = ClippedPytree
"""Accepted scalar or per-group clipped query value."""

GaussianNoiseOutput: TypeAlias = NoisedPytree
"""Noised counterpart to :data:`GaussianNoiseInput`."""


class GaussianNoiseFn(Protocol):
    """Callable returned by :func:`opaque.dpsgd.noise.gaussian_noise`.

    It accepts a clipped query and returns its noised counterpart together
    with a new immutable :class:`GaussianNoiseState`.
    """

    def __call__(
        self,
        clipped_grads: GaussianNoiseInput,
        state: GaussianNoiseState,
    ) -> tuple[GaussianNoiseOutput, GaussianNoiseState]: ...


JmeNoiseInput: TypeAlias = ClippedPytree
"""A normalized, globally aggregated scalar-clipping output."""

JmeNoiseOutput: TypeAlias = SecondMomentNoiseOutput
"""Noisy projected aggregate and its separately noised clean square."""


class JmeNoiseFn(Protocol):
    """Callable returned by :func:`opaque.dpsgd.noise.jme_noise`."""

    def __call__(
        self,
        clipped_grads: JmeNoiseInput,
        state: JmeNoiseState,
    ) -> tuple[JmeNoiseOutput, JmeNoiseState]: ...


__all__ = [
    "GaussianNoiseFn",
    "GaussianNoiseInput",
    "GaussianNoiseOutput",
    "JmeNoiseFn",
    "JmeNoiseInput",
    "JmeNoiseOutput",
]
