"""Poisson amplification for DP-FTRL — whole-process accountant.

For a ``BandMfStrategy`` inner, the ``n_steps`` training rounds divide
into ``num_groups = ceil(n_steps / bands)`` independent groups
(``bands`` = strategy bandwidth).  Each group is a Poisson-subsampled
Gaussian.  For ``IdentityStrategy`` (encoder ``I``), ``bands == 1`` so
every round is its own group — ``num_groups == n_steps``.

The PLD is the ``num_groups``-fold composition of the per-group Poisson PLD.
When ``truncated_batch_size`` and
``dataset_size`` are paired (``IdentityStrategy`` only), each group
instead uses the truncated Poisson-Gaussian PLD — matching a per-step
batch cap on the runtime sampler.

References:
    - BandMF amplification: Choquette-Choo et al. (2023) https://arxiv.org/abs/2306.08153
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from opake.api.accounting.core import _native
from opake.api.accounting.core._horizon import DpHorizonProcess
from opake.api.accounting.core._pld_cache import pld_cache
from opake.api.accounting.core.discretization import get_discretization
from opake.api.accounting.dpftrl.mechanisms._mf_gaussian import MfGaussian
from opake.api.dpftrl.noise._band_mf import BandMfStrategy
from opake.api.dpftrl.noise._identity import IdentityStrategy
from opake.api.dpftrl.noise._schedule_fingerprint import strategy_cache_key
from opake.exceptions import ConfigurationError, InputTypeError

if TYPE_CHECKING:
    from opake.api.accounting.core._base import Pld

#: Mechanism types accepted by :func:`poisson`.
_Inner = MfGaussian


@dataclass(frozen=True, slots=True)
class CyclicPoisson(DpHorizonProcess):
    """Poisson-amplified MF mechanism — total privacy cost over ``n_steps``.

    For ``BandMfStrategy`` inner, ``num_groups = ceil(n_steps / bands)``
    where ``bands = inner.strategy.bands``.  For ``IdentityStrategy``,
    ``num_groups = n_steps``.

    BandMF requires cyclic sampling over fixed dataset partitions;
    ``sample_rate`` is conditional on the active partition. Identity uses
    whole-dataset Poisson sampling, optionally capped with
    ``truncated_batch_size`` and ``dataset_size``. At ``sample_rate=1.0``,
    each accounted release is a plain Gaussian without amplification.

    Named ``CyclicPoisson`` (not ``Poisson``) to avoid a class-name collision
    with :class:`opake.dpsgd.accounting.amplification.Poisson` in the
    serialization registry.  The user-facing factory is still
    :func:`poisson`.
    """

    inner: _Inner
    sample_rate: float
    n_steps: int
    truncated_batch_size: int | None = None
    dataset_size: int | None = None

    @property
    def min_sep(self) -> int:
        # Conservative bound; BandMF and identity derive sensitivity independently.
        return 1

    @property
    def max_participations(self) -> int:
        # Conservative bound over the full horizon.
        return self.n_steps

    def __post_init__(self):
        sample_rate = float(self.sample_rate)
        if not 0 < sample_rate <= 1:
            raise ConfigurationError(
                *(
                    f"sample_rate must be in (0, 1], got {self.sample_rate}. "
                    "For q=1 (every example participates) there is no Poisson "
                    "amplification — the per-step release is the plain Gaussian.",
                )
            )
        object.__setattr__(self, "sample_rate", sample_rate)
        if int(self.n_steps) < 1:
            raise ConfigurationError(*(f"n_steps must be >= 1, got {self.n_steps}",))
        object.__setattr__(self, "n_steps", int(self.n_steps))
        if sample_rate == 1.0 and self.truncated_batch_size is not None:
            raise ConfigurationError(
                *(
                    "CyclicPoisson: sample_rate=1.0 requires plain Poisson "
                    "(truncated_batch_size=None). With q=1 the batch cap yields a "
                    "fixed-size full batch, which has no truncated-Poisson "
                    "analysis; use sample_rate<1.",
                )
            )
        # Validate truncation pairing here (not only in the factory) so
        # direct construction and deserialization can't pass an unpaired
        # ``(truncated_batch_size, dataset_size)`` into
        # ``_native.truncated_poisson_gaussian_pld`` and fail at PLD time.
        if (self.truncated_batch_size is None) != (self.dataset_size is None):
            raise ConfigurationError(
                *(
                    "CyclicPoisson: truncated_batch_size and dataset_size must be set "
                    "together (both None for plain Poisson, both set for truncated).",
                )
            )
        if self.truncated_batch_size is not None:
            if int(self.truncated_batch_size) < 1:
                raise ConfigurationError(
                    *(
                        "CyclicPoisson: truncated_batch_size must be >= 1, got "
                        f"{self.truncated_batch_size}",
                    )
                )
            if int(self.dataset_size) < 1:
                raise ConfigurationError(
                    *(
                        f"CyclicPoisson: dataset_size must be >= 1, got {self.dataset_size}",
                    )
                )
            if not isinstance(self.inner.strategy, IdentityStrategy):
                raise ConfigurationError(
                    *(
                        "CyclicPoisson: truncated Poisson is only supported for "
                        "IdentityStrategy inner (BandMfStrategy per-group truncation "
                        "is not implemented). Use plain Poisson "
                        "(truncated_batch_size=None) with BandMfStrategy.",
                    )
                )

    def _pld_cache_key(self) -> tuple[object, ...]:
        return (
            "CyclicPoisson",
            self.inner.noise_multiplier,
            self.sample_rate,
            self.n_steps,
            self.truncated_batch_size,
            self.dataset_size,
            strategy_cache_key(self.inner.strategy, self.n_steps),
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
        """Return the PLD for the complete declared horizon."""
        config = get_discretization(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
            max_conv_grid=max_conv_grid,
            seed=seed,
            mc_resolution=mc_resolution,
            mc_failure_probability=mc_failure_probability,
        )

        s = self.inner.strategy
        if isinstance(s, BandMfStrategy):
            # Theorem 4 of https://arxiv.org/abs/2306.08153 reduces BandMF
            # under cyclic-Poisson sampling to ``num_groups`` *sensitivity-κ*
            # subsampled-Gaussian queries, where κ is the maximum column norm
            # of C.  The repeat participations are priced by the
            # ``self_compose(num_groups)`` below, so the normaliser here must
            # be the single-participation value — ask for it explicitly
            # rather than relying on the schema defaults, which describe the
            # degenerate ``min_sep=1`` limit (see :attr:`min_sep`).
            sensitivity = s.sensitivity(
                n_steps=self.n_steps,
                min_sep=self.n_steps,
                max_participations=1,
            )
            effective_nm = self.inner.noise_multiplier / sensitivity
            bands = s.bands
            num_groups = math.ceil(self.n_steps / bands) if bands > 0 else 0
        elif isinstance(s, IdentityStrategy):
            effective_nm = float(self.inner.noise_multiplier)
            num_groups = self.n_steps
        else:
            raise InputTypeError(
                *(
                    "Poisson requires a BandMfStrategy or IdentityStrategy "
                    f"inner.strategy, got {type(s).__name__}.",
                )
            )

        if effective_nm == 0:
            return _native.non_private_pld(config.to_native())

        native_cfg = config.to_native()
        if self.sample_rate == 1.0:
            # q=1 is no subsampling: the per-group release is the plain
            # Gaussian at the mechanism's effective noise multiplier.
            per_group_pld = _native.gaussian_pld(effective_nm, native_cfg)
        elif self.truncated_batch_size is not None:
            per_group_pld = _native.truncated_poisson_gaussian_pld(
                effective_nm,
                self.sample_rate,
                self.truncated_batch_size,
                self.dataset_size,
                native_cfg,
            )
        else:
            per_group_pld = _native.poisson_pld(
                _native.gaussian_pld(effective_nm, native_cfg),
                self.sample_rate,
            )
        return per_group_pld.self_compose(num_groups)


def poisson(
    inner: _Inner,
    sample_rate: float,
    *,
    n_steps: int,
    truncated_batch_size: int | None = None,
    dataset_size: int | None = None,
) -> CyclicPoisson:
    """Poisson amplification for DP-FTRL.

    Whole-process accountant: returns a :class:`DpProcess` covering all
    ``n_steps`` training rounds.  Compose nothing externally with
    ``* num_steps``.

    ``IdentityStrategy`` uses whole-dataset Poisson sampling. ``BandMfStrategy``
    requires cyclic Poisson sampling over fixed dataset partitions, as provided
    by :class:`opake.dpftrl.sampling.CyclicPoissonSampler`.
    Truncated Poisson (capped batches) is supported only for ``IdentityStrategy``.

    Args:
        inner: ``mf_gaussian(nm, BandMfStrategy(...))`` or
            ``mf_gaussian(nm, identity_strategy())``.
        sample_rate: Inclusion probability in ``(0, 1]``. For identity,
            this applies to every record each step; for BandMF, it is
            conditional on the record's partition being active. At ``1.0``,
            each accounted release is a plain Gaussian without amplification.
        n_steps: Total number of training rounds.  For ``BandMfStrategy``
            the cycle count is ``ceil(n_steps / bands)``; for
            ``IdentityStrategy`` it equals ``n_steps``.
        truncated_batch_size: Optional max batch-size cap; switches the
            per-step analysis to truncated Poisson.  ``IdentityStrategy``
            only.
        dataset_size: Required when ``truncated_batch_size`` is set; ``|D|``.

    Returns:
        A :class:`CyclicPoisson` process.

    Example::

        import opake.dpftrl.accounting as ftrl_acc
        from opake.dpftrl.noise import band_mf_strategy, identity_strategy

        # BandMF
        s = band_mf_strategy(bands=10)
        proc = ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(1.0, s),
            sample_rate=0.01, n_steps=1000,
        )

        # MF identity (DP-SGD-style baseline through the FTRL API)
        proc = ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(1.0, identity_strategy()),
            sample_rate=0.01, n_steps=1000,
        )
        eps = proc.epsilon_at(1e-5)
    """
    if not isinstance(inner, MfGaussian):
        raise InputTypeError(
            *(f"poisson() requires an MfGaussian inner, got {type(inner).__name__}.",)
        )
    if not isinstance(inner.strategy, (BandMfStrategy, IdentityStrategy)):
        raise InputTypeError(
            *(
                "poisson() requires inner.strategy to be BandMfStrategy or "
                f"IdentityStrategy, got {type(inner.strategy).__name__}.",
            )
        )
    # Bounds live in ``CyclicPoisson.__post_init__``.
    return CyclicPoisson(
        inner=inner,
        sample_rate=float(sample_rate),
        n_steps=int(n_steps),
        truncated_batch_size=truncated_batch_size,
        dataset_size=dataset_size,
    )
