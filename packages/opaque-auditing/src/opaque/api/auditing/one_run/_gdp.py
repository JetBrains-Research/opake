"""μ-GDP order-statistics audit method for one-run privacy auditing.

Mirrors :class:`opaque.accounting.Pld`'s metric surface: ``epsilon_at``,
``delta_at``, ``beta_at``, ``advantage``.  All four derive from a single
inferred μ̂ via :meth:`GdpMethod._mu_at`.  Constructed via
:meth:`OneRunEstimate.gdp`.

Reference: Xiang et al. (2025), https://arxiv.org/abs/2509.08704
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import scipy.special
import scipy.stats

from opaque.api.auditing.one_run._stats import (
    search_ceiling,
    validate_delta,
    validate_significance,
)
from opaque.exceptions import ConfigurationError, OperationError

if TYPE_CHECKING:
    from opaque.api.auditing.one_run._estimate import OneRunEstimate


_TOL_MU = 0.01
_EXP_OVERFLOW_LOG_THRESHOLD = 700
_EPSILON_SEARCH_TOLERANCE = 1e-12
_SADDLEPOINT_SEARCH_TOLERANCE = 1e-10

# Maximum number of ranks to compute exactly. Higher ranks (sorted by |L|)
# have lower error probability (more confident). Omitted lower ranks use the
# first computed rank's error probability. This is a lower bound on their
# v_k values and therefore gives an upper bound on the lower-tail probability.
# 2000 ranks × 10K grid ≈ 160 MB.
_MAX_EXACT_RANKS = 2000

# Search caps keep malformed or numerically degenerate inputs from looping.
_MAX_MU_BRACKET_DOUBLINGS = 60
_MAX_MU_BISECTION_ITERS = 100


# ---------------------------------------------------------------------------
# Audit method
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GdpMethod:
    """μ-GDP order-statistics audit method (Xiang et al. 2025)."""

    _estimate: OneRunEstimate
    grid_size: int = 10_000

    # ------------------------------------------------------------------
    # Primitive — inferred μ̂ from the order-statistics test
    # ------------------------------------------------------------------

    def _mu_at(self, significance: float, threshold: float | None) -> float:
        """Inferred μ̂ via binary search.  Independent of δ.

        Raises:
            RuntimeError: If no finite μ brings the p-value up to
                ``significance`` for the observed number of errors.
        """
        validate_significance(significance)
        m = self._estimate.n_in + self._estimate.n_out
        r, u = self._estimate._best_r_u(threshold)

        # Bracket: start from the (ε, δ)-DP ceiling (a generous over-estimate
        # asymptotically for μ-GDP) and expand while the p-value is still
        # below significance.
        mu_hi = search_ceiling(m, 0.0, significance)
        for _ in range(_MAX_MU_BRACKET_DOUBLINGS):
            if _p_value(m, r, u, mu_hi, self.grid_size) >= significance:
                break
            mu_hi *= 2.0
        else:
            raise OperationError(
                *(
                    f"cannot invert μ-GDP p-value for (m={m}, r={r}, u={u}) at "
                    f"significance={significance}: p-value stays below it for "
                    f"every μ up to {mu_hi:g}; use fewer canaries or a larger "
                    f"significance.",
                )
            )

        mu_lo = 0.0
        # Relative tolerance: at large mu_hi, _TOL_MU alone is below one ULP.
        for _ in range(_MAX_MU_BISECTION_ITERS):
            if mu_hi - mu_lo <= _TOL_MU * max(1.0, mu_hi):
                break
            mu_mid = (mu_lo + mu_hi) / 2.0
            if _p_value(m, r, u, mu_mid, self.grid_size) < significance:
                mu_lo = mu_mid
            else:
                mu_hi = mu_mid
        return mu_lo

    # ------------------------------------------------------------------
    # Pld-mirror surface
    # ------------------------------------------------------------------

    def epsilon_at(
        self,
        *,
        delta: float,
        significance: float = 0.05,
        threshold: float | None = None,
    ) -> float:
        """Epsilon lower bound at the given (δ, significance).

        Raises:
            ValueError: If ``delta <= 0``.
        """
        if delta <= 0:
            raise ConfigurationError(
                *(f"μ-GDP f-DP auditing requires delta > 0, got {delta}",)
            )
        validate_delta(delta)
        return _gdp_to_eps_delta(
            self._mu_at(significance, threshold),
            delta,
        )

    def delta_at(
        self,
        *,
        epsilon: float,
        significance: float = 0.05,
        threshold: float | None = None,
    ) -> float:
        """δ(ε) under the inferred μ̂-GDP guarantee.

        Closed form: δ(ε; μ) = Φ(μ/2 − ε/μ) − e^ε · Φ(−μ/2 − ε/μ).
        """
        if epsilon < 0:
            raise ConfigurationError(*(f"epsilon must be >= 0, got {epsilon}",))
        mu = self._mu_at(significance, threshold)
        if mu == 0.0:
            return 0.0
        a = mu / 2.0 - epsilon / mu
        b = -mu / 2.0 - epsilon / mu
        term1 = scipy.stats.norm.cdf(a)
        log_term2 = epsilon + scipy.stats.norm.logcdf(b)
        term2 = (
            math.exp(log_term2) if log_term2 < _EXP_OVERFLOW_LOG_THRESHOLD else math.inf
        )
        return float(max(0.0, term1 - term2))

    def beta_at(
        self,
        *,
        alpha: float,
        significance: float = 0.05,
        threshold: float | None = None,
    ) -> float:
        """f-DP Type-II error at α under the inferred μ̂-GDP.

        β(α; μ) = Φ(Φ⁻¹(1 − α) − μ).  Note: this is the *theoretical* β
        of the post-audit guarantee, distinct from
        :meth:`OneRunEstimate.beta_at` which is the empirical attack ROC.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ConfigurationError(*(f"alpha must be in [0, 1], got {alpha}",))
        mu = self._mu_at(significance, threshold)
        return float(scipy.stats.norm.cdf(scipy.stats.norm.ppf(1.0 - alpha) - mu))

    def advantage(
        self,
        *,
        significance: float = 0.05,
        threshold: float | None = None,
    ) -> float:
        """Total-variation advantage at the inferred μ̂-GDP.

        TV(μ) = 2 · Φ(μ/2) − 1.
        """
        mu = self._mu_at(significance, threshold)
        return float(2.0 * scipy.stats.norm.cdf(mu / 2.0) - 1.0)


# ---------------------------------------------------------------------------
# GDP → (ε, δ) conversion
# ---------------------------------------------------------------------------


def _gdp_to_eps_delta(mu: float, delta: float) -> float:
    """Convert μ-GDP to (ε, δ)-DP.  Returns ε.

    Uses the closed-form relation
        δ(ε) = Φ(μ/2 − ε/μ) − exp(ε)·Φ(−μ/2 − ε/μ)
    and binary-searches for the ε at which δ(ε) = *delta*.
    """
    if mu < 0.0:
        raise ConfigurationError(*(f"mu must be >= 0, got {mu}",))
    if not (0.0 < delta <= 1.0):
        raise ConfigurationError(*(f"delta must be in (0, 1], got {delta}",))
    if mu == 0.0:
        return 0.0
    if delta >= 1.0:
        # δ ≥ 1 ⇒ no privacy constraint; smallest ε is 0.
        return 0.0

    _norm_cdf = scipy.stats.norm.cdf
    _norm_logcdf = scipy.stats.norm.logcdf

    def _delta_at_eps(eps: float) -> float:
        a = mu / 2.0 - eps / mu
        b = -mu / 2.0 - eps / mu
        term1 = _norm_cdf(a)
        log_term2 = eps + _norm_logcdf(b)
        term2 = (
            math.exp(log_term2) if log_term2 < _EXP_OVERFLOW_LOG_THRESHOLD else math.inf
        )
        return term1 - term2

    eps_lo = 0.0
    eps_hi = mu * mu + 4.0 * mu
    while _delta_at_eps(eps_hi) > delta:
        eps_hi *= 2.0

    if _delta_at_eps(0.0) <= delta:
        return 0.0

    for _ in range(100):
        eps_mid = (eps_lo + eps_hi) / 2.0
        if _delta_at_eps(eps_mid) > delta:
            eps_lo = eps_mid
        else:
            eps_hi = eps_mid
        if eps_hi - eps_lo < _EPSILON_SEARCH_TOLERANCE:
            break

    return eps_lo


# ---------------------------------------------------------------------------
# Discretised base distribution pair for μ-GDP
# ---------------------------------------------------------------------------


class _BaseGrid(NamedTuple):
    """Discretised base distribution pair for μ-GDP.

    All arrays are sorted by ascending ``abs_privacy_loss``.  The grid
    is constructed in z-space (z = Φ⁻¹(y)) for numerical stability at
    all μ values.  ``F_y`` is the CDF of the mixture density in
    |L|-sorted space, regardless of the z-space parameterisation.
    """

    z: np.ndarray
    """z-space grid points (sorted by |L|, not by z-value)."""

    mass: np.ndarray
    """Mass per grid point: (φ(z) + φ(z−μ)) / 2 · Δz."""

    F_y: np.ndarray
    """CDF of the mixture density in |L|-sorted space."""

    abs_privacy_loss: np.ndarray
    """|L(z)| = |μ²/2 − μ·z| at each grid point."""


def _gdp_base_pair_grid(mu: float, num_points: int) -> _BaseGrid:
    """Build discretised base pair for μ-GDP.

    The grid lives in z-space where z = Φ⁻¹(y).  In this space:

    - P density in z:  φ(z)      (standard normal)
    - Q density in z:  φ(z − μ)  (shifted normal)
    - Mixture:  (φ(z) + φ(z − μ)) / 2
    - |L(z)| = |μ²/2 − μ·z|
    """
    if mu <= 0.0:
        raise ConfigurationError(*("mu must be > 0 for grid construction",))

    z_lo = -6.0
    z_hi = mu + 6.0
    z = np.linspace(z_lo, z_hi, num_points)
    dz = z[1] - z[0]

    phi_z = scipy.stats.norm.pdf(z)
    phi_z_mu = scipy.stats.norm.pdf(z - mu)

    mix_dz = (phi_z + phi_z_mu) / 2.0
    mass = mix_dz * dz

    # L(z) = ln(φ(z)/φ(z−μ)) = −μ·z + μ²/2  ⇒  |L(z)| = |μ²/2 − μ·z|
    abs_pl = np.abs(0.5 * mu * mu - mu * z)

    sort_idx = np.argsort(abs_pl, kind="stable")
    z_sorted = z[sort_idx]
    mass_sorted = mass[sort_idx]
    abs_pl_sorted = abs_pl[sort_idx]

    cum = np.cumsum(mass_sorted)
    F_y = np.empty_like(cum)
    F_y[0] = 0.0
    F_y[1:] = cum[:-1]

    return _BaseGrid(z_sorted, mass_sorted, F_y, abs_pl_sorted)


# ---------------------------------------------------------------------------
# Order-statistics + Chernoff p-value
# ---------------------------------------------------------------------------


def _p_value(
    n: int,
    r: int,
    u: int,
    mu: float,
    grid_size: int = 10_000,
) -> float:
    """P-value under μ-GDP (Xiang et al. 2025, Theorems 2–3).

    Builds a discretised base pair for the Gaussian trade-off function,
    computes the conditional error probability v_k for the top r' ranks
    via numerical integration, then applies a Chernoff tail bound on the
    sum of independent heterogeneous Bernoullis. For omitted lower ranks,
    monotonicity of Eq. 25 gives v_k >= v_{n-r'+1}; substituting that
    μ-dependent boundary value conservatively upper-bounds the lower tail.
    """
    if not math.isfinite(mu) or mu <= 0.0:
        return 1.0  # perfectly private / non-finite — can't reject

    grid = _gdp_base_pair_grid(mu, grid_size)
    r_prime = min(r, _MAX_EXACT_RANKS)
    v_k = _compute_v_k(n, r_prime, grid)
    trunc_v = float(v_k[0]) if r > r_prime else None
    return _chernoff_lower_tail(v_k, n_trunc=r - r_prime, u=u, trunc_v=trunc_v)


def _compute_v_k(n: int, r_prime: int, grid: _BaseGrid) -> np.ndarray:
    """v_k for ranks k = n − r' + 1, …, n.

    v_k = E[sigmoid(−|L(Y_{(k)})|)] where Y_{(k)} is the k-th order
    statistic of n i.i.d. draws from the mixture density f_Y
    (Xiang et al. 2025, Eq. 12).  Returns shape ``(r_prime,)`` in [0, 0.5].
    """
    k_vals = np.arange(n - r_prime + 1, n + 1, dtype=np.float64)

    log_C = (
        scipy.special.gammaln(n + 1)
        - scipy.special.gammaln(k_vals)
        - scipy.special.gammaln(n - k_vals + 1)
    )

    mass = grid.mass
    F = grid.F_y
    abs_pl = grid.abs_privacy_loss

    log_mass = np.log(np.maximum(mass, 1e-300))
    log_F = np.log(np.maximum(F, 1e-300))
    log_1mF = np.log(np.maximum(1.0 - F, 1e-300))
    log_sig = np.asarray(scipy.special.log_expit(-abs_pl))

    k = k_vals[:, None]

    # (k-1) · log(F_Y):  0 when k = 1 to avoid 0 · (-inf) = NaN
    term_F = np.where(k > 1, (k - 1) * log_F[None, :], 0.0)
    # (n-k) · log(1 - F_Y):  0 when k = n
    term_1mF = np.where(k < n, (n - k) * log_1mF[None, :], 0.0)

    log_integrand = (
        log_C[:, None] + log_mass[None, :] + term_F + term_1mF + log_sig[None, :]
    )
    log_v_k = scipy.special.logsumexp(log_integrand, axis=1)

    return np.clip(np.exp(log_v_k), 0.0, 0.5)


def _chernoff_lower_tail(
    v_k: np.ndarray,
    n_trunc: int,
    u: int,
    trunc_v: float | None = None,
) -> float:
    """Chernoff bound on P(Σ V_k ≤ u)  (Xiang et al. 2025, Theorem 3).

    V_k are independent Bernoulli(v_k) for the exactly-computed ranks,
    plus n_trunc Bernoullis bounded below by ``trunc_v``. Replacing omitted
    probabilities with this lower bound increases the lower-tail probability.
    Returns the minimised upper bound, or 1.0 if observed errors are not
    below expectation.
    """
    if n_trunc and trunc_v is None:
        raise ConfigurationError(*("trunc_v is required when ranks are truncated",))
    trunc_v = 0.0 if trunc_v is None else trunc_v
    expected = float(np.sum(v_k)) + n_trunc * trunc_v
    if u >= expected:
        return 1.0

    if u == 0:
        # λ → −∞ minimum: P = Π(1−v_k) · (1−trunc_v)^n_trunc
        log_pval = float(
            np.sum(np.log(np.maximum(1.0 - v_k, 1e-300)))
        ) + n_trunc * math.log1p(-trunc_v)
        return min(math.exp(log_pval), 1.0)

    # Bisect for λ* ∈ (−50, 0) where dκ/dλ = 0.
    # κ(λ) = −λu + Σ ln(1 − v_k + v_k·e^λ)
    #        + n_trunc·ln(1 − trunc_v + trunc_v·e^λ)
    def _kappa_deriv(lam: float) -> float:
        e_lam = math.exp(lam)
        exact = float(np.sum(v_k * e_lam / (1.0 + v_k * (e_lam - 1.0))))
        trunc = n_trunc * trunc_v * e_lam / (1.0 + trunc_v * (e_lam - 1.0))
        return -u + exact + trunc

    lam_lo, lam_hi = -50.0, 0.0
    if _kappa_deriv(lam_lo) >= 0:
        lam_star = lam_lo
    else:
        for _ in range(100):
            lam_mid = (lam_lo + lam_hi) / 2.0
            if _kappa_deriv(lam_mid) > 0:
                lam_hi = lam_mid
            else:
                lam_lo = lam_mid
            if lam_hi - lam_lo < _SADDLEPOINT_SEARCH_TOLERANCE:
                break
        lam_star = (lam_lo + lam_hi) / 2.0

    e_lam = math.exp(lam_star)
    kappa = (
        -lam_star * u
        + float(np.sum(np.log1p(v_k * (e_lam - 1.0))))
        + n_trunc * math.log1p(trunc_v * (e_lam - 1.0))
    )
    return min(math.exp(kappa), 1.0)
