"""Balls-in-Bins amplification for DP-FTRL — **total** multi-epoch cost.

In the Balls-in-Bins (BnB) sampling scheme, the dataset is randomly
partitioned into ``num_bins`` equally-sized bins.  The bin assignment
is fixed once at sampler init and reused across all ``n_steps //
num_bins`` epochs, so each example stays in its bin — required for the
dominating-pair analysis.

Lemma 3.2 of Choquette-Choo et al. (2024) is stated for elementwise
non-negative encoders. This implementation uses the lemma directly on that
domain and extends it conservatively to signed encoders by the row-wise
triangle inequality::

    P = (1/b) Σ_{i=1}^{b} N(m_i, σ²I)        m_i = Σ_{j=0}^{E-1} |C|[:, b·j + i]
    Q = N(0, σ²I)

where ``E = n_steps // num_bins`` is the per-bin participation count.

After Gram-matrix reduction (``G[i,j] = m_i · m_j``) the privacy loss only
depends on ``G``, ``num_bins`` and ``σ``.  Both dispatch paths feed this
construction:

- **Correlated-noise** (matrix-factorisation): MfGaussian wrapping
  ``BltStrategy`` / ``BsrStrategy`` / ``BisrStrategy`` /
  ``LambdaCgdStrategy`` — pass the strategy's pre-computed Gram matrix of
  the elementwise-absolute forward encoder.
- **MF identity** (uncorrelated noise): MfGaussian wrapping
  ``IdentityStrategy`` (encoder ``C = I``) gives orthogonal ``m_i`` with
  ``‖m_i‖² = E``, i.e. ``G = E · I_b`` (diagonal).  This feeds the same
  Lemma 3.2 dominating pair through a specialised MC primitive.

The reachable coefficient domains differ: BSR validates non-negative forward
coefficients and λ-CGD has non-negative geometric coefficients by construction,
while generic Toeplitz and custom BISR inputs are not assumed non-negative.
Consequently the native Toeplitz/BISR Gram builders apply ``|C|`` explicitly,
using the conservative signed extension even when the absolute value is a no-op.
For column-normalized BISR/λ-CGD they normalize each column first (the norm is
unchanged by absolute value), then take the absolute epoch-column sum; they
never normalize an already-aggregated bin mode.

The returned process represents the **total** privacy cost across
all ``n_steps`` rounds.  Do NOT compose further externally.

References:
    - Chua et al. (2025), "Balls-and-Bins Sampling for DP-SGD":
      https://arxiv.org/abs/2412.16802
    - Choquette-Choo et al. (2024), "Near Exact Privacy Amplification
      for Matrix Mechanisms"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from opaque.api.accounting.core import _native
from opaque.api.accounting.core._horizon import DpHorizonProcess
from opaque.api.accounting.core._pld_cache import pld_cache
from opaque.api.accounting.core._random_allocation_cache import epoch_pld
from opaque.api.accounting.dpftrl.mechanisms._mf_gaussian import MfGaussian
from opaque.api.dpftrl.noise._bisr import BisrStrategy
from opaque.api.dpftrl.noise._blt import BltStrategy
from opaque.api.dpftrl.noise._bsr import BsrStrategy
from opaque.api.dpftrl.noise._identity import IdentityStrategy
from opaque.api.dpftrl.noise._lambda_cgd import LambdaCgdStrategy
from opaque.api.dpftrl.noise._schedule_fingerprint import strategy_cache_key
from opaque.exceptions import ConfigurationError, InputTypeError

from ._balls_in_bins_transcript_cache import with_handle as _with_transcript_handle

if TYPE_CHECKING:
    from opaque.api.accounting.core._base import Pld

#: Mechanism types accepted by :func:`balls_in_bins`.
_Inner = MfGaussian
_MIN_NUM_BINS = 2

#: Strategy types whose Gram is needed at PLD time (the "correlated MF" set).
_CorrelatedStrategies = (BltStrategy, BsrStrategy, BisrStrategy, LambdaCgdStrategy)


@dataclass(frozen=True, slots=True)
class BallsInBins(DpHorizonProcess):
    """Balls-in-Bins amplified MF mechanism — **total** privacy cost.

    The returned PLD covers all ``n_steps`` training rounds (= ``num_bins``
    bins × ``n_steps // num_bins`` epochs).  Do NOT compose externally.

    For ``IdentityStrategy`` inner, the dispatch uses the deterministic
    random-allocation transform: with ``C = I`` the Gram is exactly
    ``num_epochs · I_b`` and the dominating pair collapses onto
    1-out-of-``num_bins`` random allocation at ``σ/√num_epochs``.

    Example (DP-λCGD)::

        training = ftrl_acc.balls_in_bins(
            ftrl_acc.mf_gaussian(nm, strategy),
            num_bins=steps_per_epoch,
            n_steps=steps_per_epoch * num_epochs,
        )
        eps = training.epsilon_at(1e-5)
    """

    inner: _Inner
    num_bins: int
    n_steps: int

    def __post_init__(self) -> None:
        if self.num_bins < _MIN_NUM_BINS:
            raise ConfigurationError(
                *(f"num_bins must be >= 2 for BnB amplification, got {self.num_bins}",)
            )
        if self.n_steps < 1:
            raise ConfigurationError(*(f"n_steps must be >= 1, got {self.n_steps}",))
        if self.n_steps % self.num_bins != 0:
            raise ConfigurationError(
                *(
                    f"n_steps ({self.n_steps}) must be a positive multiple of "
                    f"num_bins ({self.num_bins}); BnB analysis assumes integer epochs.",
                )
            )

    @property
    def num_epochs(self) -> int:
        """Per-bin participation count: ``n_steps // num_bins``."""
        return self.n_steps // self.num_bins

    @property
    def min_sep(self) -> int:
        # Each example participates once per epoch; consecutive participations
        # are exactly ``num_bins`` rounds apart.
        return self.num_bins

    @property
    def max_participations(self) -> int:
        # Once per epoch ⇒ ``num_epochs`` total per example.
        return self.num_epochs

    def _pld_cache_key(self) -> tuple[object, ...]:
        return (
            "BallsInBins",
            self.inner.noise_multiplier,
            self.num_bins,
            self.n_steps,
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
        if self.inner.noise_multiplier == 0:
            return _native.non_private_pld(native_cfg)

        s = self.inner.strategy

        # Identity (C = I) collapses onto random allocation, exactly.
        #
        # The mixture means ``m_i = Σ_j |C|[:, b·j+i]`` are then indicators of
        # disjoint step sets, so they are orthogonal with equal norm ``√E`` and
        # the gram is ``G = E·I_b``.  Projecting onto their span and rescaling
        # by ``1/√E`` turns the Lemma 3.2 pair into
        # ``P = (1/b)Σ N(e_i, σ_eff²I_b)``, ``Q = N(0, σ_eff²I_b)`` with
        # ``σ_eff = σ/√E`` — precisely the 1-out-of-b random allocation pair.
        #
        # So this path uses the deterministic PLD transform rather than Monte
        # Carlo: no 1/δ sample cost, reproducible across thread counts, and
        # tighter.  The sampler is unchanged — bins stay fixed across epochs.
        if isinstance(s, IdentityStrategy):
            sigma_eff = float(self.inner.noise_multiplier) / math.sqrt(self.num_epochs)
            return epoch_pld(
                sigma_eff,
                self.num_bins,
                1,
                config,
            )

        if isinstance(s, BltStrategy):
            # BLT is retuned for the declared horizon, so use that horizon's
            # Toeplitz first column for the complete-process Gram matrix.
            coefs = s.coefficients(
                n_steps=self.n_steps,
                min_sep=self.min_sep,
                max_participations=self.max_participations,
            )
            pinned = list(coefs.tolist())
            gram = tuple(
                _native.toeplitz_gram_matrix(
                    pinned,
                    self.n_steps,
                    self.min_sep,
                    self.max_participations,
                    False,
                )
            )
        else:
            # BSR / BiSR / λ-CGD are recipe-driven: ``gram_matrix(n_steps=K)``
            # is a closed-form K-row gram of the *same* mechanism (no
            # re-tuning), so post-processing applies directly. Their native
            # BnB builders construct means from |C|; for normalized families,
            # per-column normalization precedes the epoch sum.
            gram = s.gram_matrix(
                n_steps=self.n_steps,
                min_sep=self.min_sep,
                max_participations=self.max_participations,
            )
        config.warn_if_large_mc()
        pinned_gram = tuple(gram)
        result = _with_transcript_handle(
            pinned_gram,
            self.num_bins,
            config.resolved_num_mc_samples,
            config.seed,
            lambda handle: _native.bnb_pld_from_transcript_handle(
                handle,
                list(pinned_gram),
                self.num_bins,
                self.inner.noise_multiplier,
                native_cfg,
            ),
            native=_native,
        )
        if result is not None:
            return result
        return _native.bnb_mc_pld(
            list(pinned_gram),
            self.num_bins,
            self.inner.noise_multiplier,
            native_cfg,
        )


def balls_in_bins(
    inner: _Inner,
    *,
    num_bins: int,
    n_steps: int,
) -> BallsInBins:
    """Balls-in-Bins amplified MF mechanism — **total** privacy cost.

    Each epoch, the dataset is partitioned into ``num_bins`` bins (assignment
    fixed at sampler init and reused across epochs).  Every example
    participates exactly once per epoch.  The total round count is
    ``n_steps``; per-bin participation count is ``n_steps // num_bins`` and
    must divide evenly.

    Args:
        inner: ``mf_gaussian(nm, strategy)`` where ``strategy`` is one of
            ``BltStrategy``, ``BsrStrategy``, ``BisrStrategy``,
            ``LambdaCgdStrategy`` (correlated MF) or ``IdentityStrategy``
            (uncorrelated baseline).
        num_bins: Bins per epoch (k ≥ 2).
        n_steps: Total training rounds.  Must be a positive multiple of
            ``num_bins`` (per-bin participation = ``n_steps // num_bins``).

    Returns:
        A :class:`BallsInBins` process (total cost).

    Example::

        from opaque.dpftrl.noise import blt_strategy, identity_strategy

        # Correlated MF
        s = blt_strategy(n_steps=1000, min_sep=100, max_participations=10)
        training = ftrl_acc.balls_in_bins(
            ftrl_acc.mf_gaussian(1.0, s),
            num_bins=100, n_steps=1000,
        )

        # Identity baseline
        training = ftrl_acc.balls_in_bins(
            ftrl_acc.mf_gaussian(1.0, identity_strategy()),
            num_bins=100, n_steps=1000,
        )
        eps = training.epsilon_at(1e-5)
    """
    if not isinstance(inner, MfGaussian):
        raise InputTypeError(
            *(
                f"balls_in_bins() requires an MfGaussian inner, got {type(inner).__name__}.",
            )
        )
    if not isinstance(inner.strategy, (*_CorrelatedStrategies, IdentityStrategy)):
        raise InputTypeError(
            *(
                "balls_in_bins() requires inner.strategy in {BltStrategy, "
                "BsrStrategy, BisrStrategy, LambdaCgdStrategy, IdentityStrategy}, "
                f"got {type(inner.strategy).__name__}.",
            )
        )
    # num_bins/n_steps bounds live in ``BallsInBins.__post_init__``.
    return BallsInBins(inner=inner, num_bins=num_bins, n_steps=n_steps)
