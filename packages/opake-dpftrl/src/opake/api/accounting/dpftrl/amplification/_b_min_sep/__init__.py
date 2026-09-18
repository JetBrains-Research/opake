"""b-min-sep subsampling amplification for BandMF (warm-start, Monte Carlo PLD).

Dong & Ganesh, "Privacy Amplification for BandMF via b-Min-Sep Subsampling"
(arXiv:2602.09338). Uses Monte Carlo accounting with the paper's dynamic
program for the likelihood ratio (Section 5).

The runtime sampler should use :class:`opake.dpftrl.sampling.BMinSepSampler`
with the same ``bands`` and ``p`` as the accountant.  Read these off the
:class:`BMinSep` instance: ``bands`` via ``inner.strategy.bands`` (or the
equivalent ``min_sep`` property) and ``p`` via ``sampling_prob``.  The
conversion ``p = p_0 / (1 - p_0 * (bands - 1))`` is encapsulated by
:func:`participation_p_from_per_example_rate`, so runtime callers never
duplicate the formula.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from opake.api.accounting.core import _native
from opake.api.accounting.core._horizon import DpHorizonProcess
from opake.api.accounting.core._pld_cache import pld_cache
from opake.api.accounting.core.discretization import get_discretization
from opake.api.accounting.dpftrl.mechanisms._mf_gaussian import MfGaussian
from opake.api.dpftrl.noise._band_mf import BandMfStrategy
from opake.api.dpftrl.noise._schedule_fingerprint import strategy_cache_key
from opake.exceptions import ConfigurationError, InputTypeError

from ._transcript_cache import with_handle as _with_transcript_handle

if TYPE_CHECKING:
    from opake.api.accounting.core._base import Pld

_Inner = MfGaussian


def participation_p_from_per_example_rate(p0: float, bands: int) -> float:
    """Paper: ``p = p_0 / (1 - p_0 * (b - 1))``.

    Converts a target per-example participation rate ``p_0 = E[batch] / |D|``
    into the paper's per-iteration inclusion probability ``p`` that
    :class:`opake.dpftrl.sampling.BMinSepSampler` expects.  ``bands == 1``
    degenerates to ``p_0`` (plain Poisson, no min-sep constraint).
    """
    if not 0.0 < p0 < 1.0:
        raise ConfigurationError(
            *(f"per-example rate p_0 must be in (0, 1), got {p0}",)
        )
    if bands < 1:
        raise ConfigurationError(*(f"bands must be >= 1, got {bands}",))
    if bands == 1:
        return p0
    denom = 1.0 - p0 * (bands - 1)
    if denom <= 0:
        raise ConfigurationError(
            *(f"infeasible p_0={p0} for bands={bands}: need p_0 < 1/(bands-1)",)
        )
    return p0 / denom


@dataclass(frozen=True, slots=True)
class BMinSep(DpHorizonProcess):
    """Monte Carlo PLD for BandMF + warm-start b-min-sep subsampling."""

    inner: _Inner
    n_steps: int
    p0: float

    def __post_init__(self) -> None:
        if self.n_steps < 1:
            raise ConfigurationError(*(f"n_steps must be >= 1, got {self.n_steps}",))
        if not 0.0 < self.p0 < 1.0:
            raise ConfigurationError(
                *(f"per-example rate p_0 must be in (0, 1), got {self.p0}",)
            )

    @property
    def min_sep(self) -> int:
        # b-min-sep contract: each example participates at most once per
        # ``bands``-row window ⇒ min separation = ``bands``.
        return self.inner.strategy.bands

    @property
    def max_participations(self) -> int:
        # At most one participation per ``bands``-row window across
        # ``n_steps``.  Use ``ceil(n_steps / bands)``: a window starts at the
        # first user contribution and may extend past ``n_steps`` mid-window,
        # so the worst case is ``ceil`` not ``floor`` (e.g. bands=4,
        # n_steps=10 yields 3, not 2).
        bands = self.inner.strategy.bands
        return (self.n_steps + bands - 1) // bands

    @property
    def sampling_prob(self) -> float:
        # The runtime ``BMinSepSampler`` parameter — the paper's per-iteration
        # inclusion probability ``p``.  Equal to ``self.p0`` only when
        # ``bands == 1``; the conversion belongs with the privacy proof so
        # callers (runtime sampler builders) never re-derive it.
        return participation_p_from_per_example_rate(self.p0, self.inner.strategy.bands)

    def _pld_cache_key(self) -> tuple[object, ...]:
        return (
            "BMinSep",
            self.inner.noise_multiplier,
            self.n_steps,
            self.p0,
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
        """Return the full-horizon confidence-bounded Monte Carlo PLD."""
        s = self.inner.strategy
        if not isinstance(s, BandMfStrategy):
            raise InputTypeError(
                *(
                    "b_min_sep requires inner.strategy to be BandMfStrategy, got "
                    f"{type(s).__name__}.",
                )
            )
        bands = s.bands
        if bands < 1:
            raise ConfigurationError(
                *(
                    "BandMfStrategy inner must have non-empty coefficients (bands >= 1).",
                )
            )
        config = get_discretization(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
            max_conv_grid=max_conv_grid,
            seed=seed,
            mc_resolution=mc_resolution,
            mc_failure_probability=mc_failure_probability,
        )
        if self.inner.noise_multiplier == 0:
            return _native.non_private_pld(config.to_native())
        config.warn_if_large_mc()
        native_cfg = config.to_native()

        coefs = s.coefficients(
            n_steps=self.n_steps,
            min_sep=self.min_sep,
            max_participations=self.max_participations,
        ).tolist()
        # The warm-start recursion of https://arxiv.org/abs/2602.09338
        # (Section 5, Eq. 1) scores each participation against a *single*
        # column ``c_i`` of C and generates repeat participations itself by
        # hopping ``bands`` rows ahead.  The Rust MC is therefore handed the
        # raw coefficients at the raw σ, so the normaliser must be the
        # single-participation column norm — ask for it explicitly rather
        # than passing this process's schema, which describes the
        # multi-participation pattern the recursion already accounts for.
        sensitivity = s.sensitivity(
            n_steps=self.n_steps,
            min_sep=self.n_steps,
            max_participations=1,
        )
        effective_nm = self.inner.noise_multiplier / sensitivity
        p = self.sampling_prob

        # Always look up the cached corpus at ``self.n_steps`` (the full
        # horizon at which it was prepared).
        # ``_with_transcript_handle`` holds the per-cache lock around
        # both the lookup and the Rust call so a concurrent
        # ``_clear_all_native_caches()`` (e.g. from ``calibrate()``'s
        # finally clause on another thread) cannot drop the corpus
        # mid-use; on cache-miss it returns ``None`` and we fall through
        # to a fresh full-horizon warm MC calculation.
        result = _with_transcript_handle(
            tuple(coefs),
            self.n_steps,
            p,
            config.resolved_num_mc_samples,
            config.seed,
            lambda hid: _native.bandmf_b_min_sep_pld_from_transcript_handle(
                hid,
                coefs,
                p,
                effective_nm,
                native_cfg,
            ),
        )
        if result is not None:
            return result
        return _native.bandmf_b_min_sep_warm_mc_pld(
            coefs,
            self.n_steps,
            p,
            effective_nm,
            native_cfg,
        )


def b_min_sep(
    inner: _Inner,
    *,
    n_steps: int,
    p0: float,
) -> BMinSep:
    """BandMF privacy accounting under warm-start b-min-sep subsampling.

    Args:
        inner: ``mf_gaussian(nm, BandMfStrategy(...))`` — strategy
            coefficients (and band width) are read from
            ``inner.strategy.coefficients``.
        n_steps: Total number of training iterations ``n``.
        p0: Per-example participation rate per iteration
            (``E[batch] / |D|``).  Same ``p_0`` as cyclic Poisson /
            batch-size accounting.

    Returns:
        A :class:`BMinSep` process (asymmetric PLD from Monte Carlo).
    """
    if not isinstance(inner, MfGaussian):
        raise InputTypeError(
            *(f"b_min_sep() requires an MfGaussian inner, got {type(inner).__name__}.",)
        )
    if not isinstance(inner.strategy, BandMfStrategy):
        raise InputTypeError(
            *(
                "b_min_sep() requires inner.strategy to be BandMfStrategy, got "
                f"{type(inner.strategy).__name__}.",
            )
        )
    if inner.strategy.bands < 1:
        raise ConfigurationError(
            *("BandMfStrategy inner must have non-empty coefficients (bands >= 1).",)
        )

    return BMinSep(
        inner=inner,
        n_steps=n_steps,
        p0=float(p0),
    )
