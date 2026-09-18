"""Parallel Poisson subsampling mechanism for multi-worker training."""

from __future__ import annotations

from dataclasses import dataclass

from opake.api.accounting.core import _native
from opake.api.accounting.core._base import DpProcess, Pld
from opake.api.accounting.core._pld_cache import pld_cache
from opake.api.accounting.core.mechanisms._nonprivate import NonPrivate
from opake.api.accounting.dpsgd.amplification._poisson import Poisson, poisson
from opake.api.accounting.dpsgd.mechanisms._adaclip import AdaClip
from opake.api.accounting.dpsgd.mechanisms._gaussian import Gaussian
from opake.exceptions import ConfigurationError, InputTypeError


@dataclass(frozen=True, slots=True)
class ParallelPoisson(DpProcess):
    """Poisson-subsampled Gaussian mechanism under parallel worker execution."""

    inner: Poisson
    num_workers: int

    def __post_init__(self):
        if (
            isinstance(self.num_workers, bool)
            or not isinstance(self.num_workers, int)
            or self.num_workers < 1
        ):
            raise ConfigurationError(
                *(f"num_workers must be a positive integer, got {self.num_workers}",)
            )
        if self.inner.truncated_batch_size is not None:
            raise ConfigurationError(
                *(
                    "ParallelPoisson does not support truncated Poisson inner mechanisms.",
                )
            )

    @pld_cache(maxsize=8)
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

        native_cfg = config.to_native()

        match self.inner:
            case Poisson(inner=NonPrivate() | Gaussian(noise_multiplier=0)):
                return _native.non_private_pld(native_cfg)
            case Poisson(
                inner=Gaussian(noise_multiplier=nm),
                sample_rate=rate,
            ):
                return _native.parallel_poisson_gaussian_pld(
                    nm,
                    rate,
                    self.num_workers,
                    native_cfg,
                )
            case Poisson(
                inner=AdaClip(inner=NonPrivate() | Gaussian(noise_multiplier=0)),
            ):
                return _native.non_private_pld(native_cfg)
            case Poisson(
                inner=AdaClip(inner=Gaussian()) as ac,
                sample_rate=rate,
            ):
                return _native.parallel_poisson_gaussian_pld(
                    ac.effective_noise_multiplier,
                    rate,
                    self.num_workers,
                    native_cfg,
                )
            case _:
                raise InputTypeError(
                    *(
                        "ParallelPoisson requires a Poisson inner mechanism, got "
                        f"{type(self.inner).__name__}.",
                    )
                )


def parallel_poisson(
    inner: Gaussian | AdaClip | NonPrivate,
    sample_rate: float,
    num_workers: int,
) -> ParallelPoisson:
    """Poisson sampling under parallel worker execution.

    Args:
        inner: A :class:`Gaussian`, :class:`AdaClip`, or :class:`NonPrivate` mechanism.
        sample_rate: Probability of including each example, in (0, 1).
        num_workers: Number of parallel workers running Poisson sampling
            independently. Truncated Poisson accounting is not supported.

    Returns:
        A :class:`ParallelPoisson` process.

    Example::

        step = dpsgd_acc.parallel_poisson(
            dpsgd_acc.gaussian(1.1), sample_rate=0.01, num_workers=4,
        )
        eps = (step * 500).epsilon_at(1e-5)
    """
    if not 0 < float(sample_rate) < 1:
        raise ConfigurationError(
            *(
                f"parallel_poisson: sample_rate must be in (0, 1), got {sample_rate}. "
                "q=1 (full participation on every worker) is not supported here.",
            )
        )
    match inner:
        case Gaussian() | AdaClip() | NonPrivate():
            pass
        case _:
            raise InputTypeError(
                *(
                    "parallel_poisson() requires a Gaussian, AdaClip, or NonPrivate "
                    f"inner mechanism, got {type(inner).__name__}.",
                )
            )
    poisson_inner = poisson(inner=inner, sample_rate=sample_rate)
    return ParallelPoisson(inner=poisson_inner, num_workers=num_workers)
