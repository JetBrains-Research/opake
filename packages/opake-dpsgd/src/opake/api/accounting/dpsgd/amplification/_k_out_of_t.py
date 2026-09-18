# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
#
# Block allocation accounting structure adapted in part from the ICML 2026
# reference implementation for "Efficient privacy loss accounting for
# subsampling and random allocation" (Apache-2.0), then reworked for Opake's
# k-out-of-t accounting API. See ../../../../../../NOTICE in this package for
# the full attribution.
"""Block and total k-out-of-t allocation over a declared DP-SGD horizon."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from opake.api.accounting.core import _native
from opake.api.accounting.core._horizon import DpHorizonProcess
from opake.api.accounting.core._pld_cache import pld_cache
from opake.api.accounting.core._random_allocation_cache import epoch_pld
from opake.api.accounting.core.mechanisms._nonprivate import NonPrivate
from opake.api.accounting.dpsgd.mechanisms._adaclip import AdaClip
from opake.api.accounting.dpsgd.mechanisms._gaussian import Gaussian
from opake.exceptions import ConfigurationError, InputTypeError

if TYPE_CHECKING:
    from opake.api.accounting.core._base import Pld

_Inner = Gaussian | AdaClip | NonPrivate
_Allocation = Literal["block", "total"]


@dataclass(frozen=True, slots=True)
class KOutOfT(DpHorizonProcess):
    """Accounting for block or total ``k``-out-of-``t`` allocation.

    Block allocation places each record once in each of ``k`` contiguous,
    nearly equal blocks.

    Total allocation chooses a uniform ``k``-subset of the ``t`` steps. The
    block PLD is a conservative upper bound for its full horizon.
    """

    inner: _Inner
    k: int
    n_steps: int
    allocation: _Allocation

    def __post_init__(self) -> None:
        if self.n_steps < 1:
            raise ConfigurationError(*(f"t must be >= 1, got {self.n_steps}",))
        if not 1 <= self.k <= self.n_steps:
            raise ConfigurationError(
                *(f"k must be in [1, t={self.n_steps}], got {self.k}",)
            )
        if self.allocation not in ("block", "total"):
            raise ConfigurationError(
                *(f"allocation must be 'block' or 'total', got {self.allocation!r}",)
            )

    @property
    def t(self) -> int:
        """Total allocation horizon."""
        return self.n_steps

    @property
    def block_sizes(self) -> tuple[int, ...]:
        """Sizes of the contiguous blocks used by the accounting reduction."""
        floor = self.n_steps // self.k
        num_ceil = self.n_steps - floor * self.k
        return (floor,) * (self.k - num_ceil) + (floor + 1,) * num_ceil

    def _noise_multiplier(self) -> float | None:
        match self.inner:
            case NonPrivate() | Gaussian(noise_multiplier=0):
                return None
            case Gaussian(noise_multiplier=nm):
                return nm
            case AdaClip(inner=NonPrivate() | Gaussian(noise_multiplier=0)):
                return None
            case AdaClip(inner=Gaussian()) as ac:
                return ac.effective_noise_multiplier
            case _:
                raise InputTypeError(
                    *("KOutOfT requires Gaussian, AdaClip(Gaussian), or NonPrivate",)
                )

    @pld_cache(maxsize=16)
    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
        max_conv_grid: int | None = None,
        seed: int | None = None,
        mc_resolution: float | None = None,
        mc_failure_probability: float | None = None,
    ) -> Pld:
        from opake.api.accounting.core.discretization import get_discretization

        config = get_discretization(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
            max_conv_grid=max_conv_grid,
            seed=seed,
            mc_resolution=mc_resolution,
            mc_failure_probability=mc_failure_probability,
        )
        noise_multiplier = self._noise_multiplier()
        if noise_multiplier is None:
            return _native.non_private_pld(config.to_native())
        return epoch_pld(
            noise_multiplier,
            self.n_steps,
            self.k,
            config,
        )


def k_out_of_t(
    inner: _Inner,
    *,
    k: int,
    t: int,
    allocation: _Allocation,
) -> KOutOfT:
    """Create a block or total k-out-of-t horizon process.

    ``allocation="block"`` exactly matches
    :class:`opake.dpsgd.sampling.KOutOfTSampler` with the same arguments.

    ``allocation="total"`` uses the block reduction as a valid conservative
    upper bound. The returned process accounts the complete ``t``-step run.
    """
    if not isinstance(inner, (Gaussian, AdaClip, NonPrivate)):
        raise InputTypeError(
            *(
                "k_out_of_t() requires Gaussian, AdaClip, or NonPrivate inner, got "
                f"{type(inner).__name__}.",
            )
        )
    if allocation not in ("block", "total"):
        raise ConfigurationError(
            *(f"allocation must be 'block' or 'total', got {allocation!r}",)
        )
    return KOutOfT(
        inner=inner,
        k=int(k),
        n_steps=int(t),
        allocation=allocation,
    )


__all__ = ["KOutOfT", "k_out_of_t"]
