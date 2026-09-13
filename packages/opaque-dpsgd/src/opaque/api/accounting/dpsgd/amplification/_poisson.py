"""Poisson-subsampled mechanism — standard DP-SGD step.

Plain Poisson accepts any ``DpProcess``. When ``truncated_batch_size`` and
``dataset_size`` are provided, the capped form requires a Gaussian-derived or
non-private base.
"""

from __future__ import annotations

from dataclasses import dataclass

from opaque.api.accounting.core import _native
from opaque.api.accounting.core._base import DpProcess, Pld
from opaque.api.accounting.core._pld_cache import pld_cache
from opaque.api.accounting.core.mechanisms._nonprivate import NonPrivate
from opaque.api.accounting.dpsgd.mechanisms._adaclip import AdaClip
from opaque.api.accounting.dpsgd.mechanisms._gaussian import Gaussian
from opaque.exceptions import ConfigurationError, InputTypeError

#: Mechanism types accepted by plain :func:`poisson`.
_Inner = DpProcess


@dataclass(frozen=True, slots=True)
class Poisson(DpProcess):
    """Poisson-subsampled mechanism (single step).

    Plain Poisson accepts any Opaque ``DpProcess``. ``sample_rate=1.0``
    returns the base mechanism without amplification. The capped form
    requires a Gaussian, AdaClip(Gaussian), or NonPrivate inner mechanism,
    with both ``truncated_batch_size`` and ``dataset_size`` set. Its guarantee
    uses the dataset-size-indexed add/remove adjacency of Ganesh (2025),
    https://arxiv.org/abs/2508.15089.
    """

    inner: _Inner
    sample_rate: float
    truncated_batch_size: int | None = None
    dataset_size: int | None = None

    def __post_init__(self):
        if not isinstance(self.inner, DpProcess):
            raise InputTypeError(
                *(
                    "Poisson requires a DpProcess inner mechanism, got "
                    f"{type(self.inner).__name__}.",
                )
            )

        sample_rate = float(self.sample_rate)
        if not 0 < sample_rate <= 1:
            raise ConfigurationError(
                *(
                    f"sample_rate must be in (0, 1], got {self.sample_rate}. "
                    "For q=1 (every example participates) there is no Poisson "
                    "amplification.",
                )
            )
        object.__setattr__(self, "sample_rate", sample_rate)
        if sample_rate == 1.0 and self.truncated_batch_size is not None:
            raise ConfigurationError(
                *(
                    "Poisson: sample_rate=1.0 requires plain Poisson "
                    "(truncated_batch_size=None). With q=1 the batch cap yields a "
                    "fixed-size full batch, which has no truncated-Poisson "
                    "analysis; use plain Poisson or sample_rate<1.",
                )
            )

        # Validate truncation pairing here (not only in the factory) so direct
        # construction and deserialization can't pass an unpaired
        # ``(truncated_batch_size, dataset_size)`` into
        # ``_native.truncated_poisson_gaussian_pld`` and fail at PLD time.
        if (self.truncated_batch_size is None) != (self.dataset_size is None):
            raise ConfigurationError(
                *(
                    "Poisson: truncated_batch_size and dataset_size must be set "
                    "together (both None for plain Poisson, both set for truncated).",
                )
            )
        if self.truncated_batch_size is not None:
            if type(self.truncated_batch_size) is not int:
                raise InputTypeError(
                    *(
                        "Poisson: truncated_batch_size must be an int, got "
                        f"{type(self.truncated_batch_size).__name__}.",
                    )
                )
            if type(self.dataset_size) is not int:
                raise InputTypeError(
                    *(
                        "Poisson: dataset_size must be an int, got "
                        f"{type(self.dataset_size).__name__}.",
                    )
                )
            if self.truncated_batch_size < 1:
                raise ConfigurationError(
                    *(
                        "Poisson: truncated_batch_size must be >= 1, got "
                        f"{self.truncated_batch_size}",
                    )
                )
            if self.dataset_size < 1:
                raise ConfigurationError(
                    *(f"Poisson: dataset_size must be >= 1, got {self.dataset_size}",)
                )
            if not isinstance(self.inner, (Gaussian, AdaClip, NonPrivate)):
                raise InputTypeError(
                    *(
                        "truncated Poisson requires a Gaussian, AdaClip(Gaussian), "
                        "or NonPrivate inner mechanism, got "
                        f"{type(self.inner).__name__}.",
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
        from opaque.api.accounting.core.discretization import get_discretization

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
        truncated = self.truncated_batch_size is not None
        if self.sample_rate == 1.0:
            return self.inner.pld(
                discretization=discretization,
                log_x_mass_truncation_bound=log_x_mass_truncation_bound,
                max_grid_size=max_grid_size,
                max_conv_grid=max_conv_grid,
                seed=seed,
                mc_resolution=mc_resolution,
                mc_failure_probability=mc_failure_probability,
            )

        if truncated:
            match self.inner:
                case NonPrivate() | Gaussian(noise_multiplier=0):
                    return _native.non_private_pld(native_cfg)
                case Gaussian(noise_multiplier=nm):
                    return _native.truncated_poisson_gaussian_pld(
                        nm,
                        self.sample_rate,
                        self.truncated_batch_size,
                        self.dataset_size,
                        native_cfg,
                    )
                case AdaClip(inner=NonPrivate() | Gaussian(noise_multiplier=0)):
                    return _native.non_private_pld(native_cfg)
                case AdaClip(inner=Gaussian()) as ac:
                    return _native.truncated_poisson_gaussian_pld(
                        ac.effective_noise_multiplier,
                        self.sample_rate,
                        self.truncated_batch_size,
                        self.dataset_size,
                        native_cfg,
                    )
                case _:
                    raise AssertionError

        return _native.poisson_pld(
            self.inner.pld(
                discretization=discretization,
                log_x_mass_truncation_bound=log_x_mass_truncation_bound,
                max_grid_size=max_grid_size,
                max_conv_grid=max_conv_grid,
                seed=seed,
                mc_resolution=mc_resolution,
                mc_failure_probability=mc_failure_probability,
            ),
            self.sample_rate,
        )


def poisson(
    inner: _Inner,
    sample_rate: float,
    *,
    truncated_batch_size: int | None = None,
    dataset_size: int | None = None,
) -> Poisson:
    """Poisson-subsampled mechanism (per-step DP-SGD factory).

    Returns a single-step process — compose externally with ``* num_steps``
    for full-training privacy.

    When ``truncated_batch_size`` is set, both it and ``dataset_size`` are
    required, and the analysis switches to the truncated Poisson-Gaussian
    PLD (production DP-SGD with capped batch size). This is Theorem 3.1 of
    Ganesh (2025), https://arxiv.org/abs/2508.15089: provisional samples above
    the cap must be truncated to a uniformly random subset. Its adjacency is
    indexed by the public larger-neighbor size ``dataset_size``; composed uses
    must retain that same dataset-size interpretation.

    Args:
        inner: The base Opaque :class:`DpProcess`. The capped form requires
            :func:`gaussian`, :func:`adaclip`, or
            :func:`opaque.accounting.nonprivate`.
        sample_rate: Probability of including each example
            (``E[batch_size] / |D|``), between zero and one inclusive.
            ``sample_rate=1.0`` means every example participates — no
            amplification; the step is equivalent to ``inner``.
        truncated_batch_size: Optional max batch-size cap; switches the
            analysis to truncated Poisson (requires ``sample_rate < 1``).
        dataset_size: Required when ``truncated_batch_size`` is set;
            the public size ``n`` of the larger dataset in the theorem's
            directed add/remove neighboring pair. It must exactly match the
            sampler's pre-sampling dataset size.

    Returns:
        A :class:`Poisson` process.

    Example::

        # Plain Poisson
        step = dpsgd_acc.poisson(dpsgd_acc.gaussian(1.1), sample_rate=0.01)
        training = step * 1000

        # Truncated Poisson (production)
        n, batch = 50_000, 250
        step = dpsgd_acc.poisson(
            dpsgd_acc.gaussian(0.8), batch / n,
            truncated_batch_size=batch, dataset_size=n,
        )
        eps = (step * 1000).epsilon_at(1e-5)
    """
    # Pairing + per-field bounds on truncated_batch_size / dataset_size are
    # validated in ``Poisson.__post_init__`` so direct construction,
    # deserialization, and this factory stay consistent.
    return Poisson(
        inner=inner,
        sample_rate=float(sample_rate),
        truncated_batch_size=truncated_batch_size,
        dataset_size=dataset_size,
    )
