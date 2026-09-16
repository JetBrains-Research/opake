"""Public type definitions for :mod:`opaque.dpftrl.noise`.

Re-exports MF noise state types and strategy dataclasses for type
annotations, plus the :class:`MfStrategy` Protocol every strategy class
implements.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from opaque.api.dpftrl.noise._band_mf import BandMfStrategy
from opaque.api.dpftrl.noise._bisr import BisrStrategy
from opaque.api.dpftrl.noise._blt import BltStrategy
from opaque.api.dpftrl.noise._bsr import BsrStrategy
from opaque.api.dpftrl.noise._engine import MFNoiseState
from opaque.api.dpftrl.noise._identity import IdentityStrategy
from opaque.api.dpftrl.noise._lambda_cgd import LambdaCgdStrategy

if TYPE_CHECKING:
    import torch

    from opaque.api.dpftrl.noise._streaming_matrix import StreamingMatrix
    from opaque.random.types import RngKey


RawMfNoiseFactory = Callable[
    [Any, "MfStrategy"],
    tuple[
        Callable[..., tuple[Any, "MFNoiseState"]],
        "MFNoiseState",
        Callable[[int], float],
    ],
]


@runtime_checkable
class MfStrategy(Protocol):
    """Polymorphic recipe for an MF noise mechanism.

    Strategies are *recipes* — small frozen dataclasses carrying only
    factory args.  Derived quantities (Toeplitz coefficients, gram
    matrices, streaming matrices, sensitivity values) are computed on
    demand via the four query methods below, parameterized by the
    amplification context (``n_steps``, ``min_sep``,
    ``max_participations``) supplied by the wrapping amplifier.

    Strategies that don't read every kwarg (e.g. :class:`IdentityStrategy`
    ignores all three; :class:`BandMfStrategy` only reads ``n_steps``)
    accept-and-ignore the rest via ``**_``.
    """

    def coefficients(
        self,
        *,
        n_steps: int,
        min_sep: int,
        max_participations: int | None,
    ) -> torch.Tensor: ...

    def gram_matrix(
        self,
        *,
        n_steps: int,
        min_sep: int,
        max_participations: int | None,
    ) -> tuple[float, ...]: ...

    def streaming_matrix(
        self,
        *,
        n_steps: int,
        min_sep: int,
        max_participations: int | None,
    ) -> StreamingMatrix: ...

    def sensitivity(
        self,
        *,
        n_steps: int,
        min_sep: int,
        max_participations: int | None,
    ) -> float: ...


@runtime_checkable
class RawMfNoiseFactoryProvider(Protocol):
    """Optional runtime hook for strategies with dedicated noise builders."""

    def raw_noise_factory(
        self,
        grad_template: Any,
        *,
        n_steps: int,
        min_sep: int,
        max_participations: int | None,
        key: RngKey,
        compute_dtype: torch.dtype,
    ) -> (
        tuple[
            Callable[..., tuple[Any, MFNoiseState]],
            MFNoiseState,
            Callable[[int], float],
        ]
        | None
    ): ...


__all__ = [
    "BandMfStrategy",
    "BisrStrategy",
    "BltStrategy",
    "BsrStrategy",
    "IdentityStrategy",
    "LambdaCgdStrategy",
    "MFNoiseState",
    "MfStrategy",
    "RawMfNoiseFactoryProvider",
    "RawMfNoiseFactory",
]
