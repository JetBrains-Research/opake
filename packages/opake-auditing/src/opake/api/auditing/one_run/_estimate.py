"""One-run privacy audit estimator.

The ``one_run()`` function precomputes the empirical ROC (TN/FN counts at
every threshold) from canary scores and returns a frozen ``OneRunEstimate``.
Epsilon estimation goes through a method object obtained from one of the
factory methods (``eps_delta()``, ``gdp()``); attack-side metrics
(``auc``, ``beta_at``) live directly on the estimate.

References:
    - Xiang et al. (2025), https://arxiv.org/abs/2509.08704
    - Steinke, Nasr, Jagielski (2023), https://arxiv.org/abs/2305.08846
    - Carlini et al. (2022), https://arxiv.org/abs/2112.03570
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import scipy.stats

from opake.api.auditing.one_run._eps_delta import EpsDeltaMethod
from opake.api.auditing.one_run._gdp import GdpMethod
from opake.api.auditing.one_run._roc import get_tn_fn_counts, tpr_at_given_fpr
from opake.exceptions import ConfigurationError

if TYPE_CHECKING:
    from opake.api.auditing._coin_flip import CanaryScores, CoinFlip
    from opake.random.types import RngKey

__all__ = ["OneRunEstimate", "one_run"]


# Minimum grid points for the GDP numerical integration.  Coarser grids
# are not merely inaccurate — they can silently erase detectable leakage:
# at grid_size=16 a fully separable 500/500 canary split audits as
# eps=0.  The floor is validated against the v_k order-statistic identity
# (Xiang et al. 2025, Eq. 12): at grid_size=1_000 the n=1 Bayes error
# v_1 = Phi(-mu/2) is reproduced with absolute error ~1.2e-6, and
# detected-leak configurations stay well away from the eps=0 collapse.
# This keeps the accepted-range cost roughly an order of magnitude below
# the 10_000-point default while remaining numerically meaningful.
_MIN_GRID_SIZE = 1_000


def one_run(scores: CanaryScores, *, coin_flip: CoinFlip) -> OneRunEstimate:
    """Build a one-run privacy estimate from canary scores.

    Joins scores to the coin-flip partition by canary identifier,
    precomputes the raw empirical ROC, and returns a frozen estimate.
    The join makes scoring order irrelevant: identifiers that do not
    cover the partition's canaries one-to-one raise instead of silently
    producing a meaningless estimate.

    Args:
        scores: Per-canary membership scores carrying canary
            identifiers, as returned by the scoring functions in
            verified mode (``coin_flip=`` + ``dataset=``).  Higher score
            = more likely a training member.  For scores computed
            outside the built-in scorers, attest identifiers explicitly
            with ``canary_scores(values, canary_indices=...)``.
        coin_flip: The :class:`~opake.auditing.CoinFlip` partition.

    Returns:
        A :class:`OneRunEstimate` with precomputed threshold structure.

    Raises:
        TypeError: If ``scores`` is a bare array without identifiers.
        ValueError: If the identifiers do not join one-to-one onto the
            partition's canaries, either partition is empty, or any
            score is NaN or infinite.

    Example::

        import opake.auditing as auditing
        from opake.random import key

        cf = auditing.coin_flip(dataset, num_canaries=1000, key=key(42))
        scores = auditing.loss_scores(loss_fn, params,
                                       batch_argnums=(1,),
                                       coin_flip=cf, dataset=dataset)
        estimate = auditing.one_run(scores, coin_flip=cf)
        print(estimate.eps_delta().epsilon_at(delta=1e-5))
    """
    in_scores, out_scores = coin_flip.split_scores(scores)

    if in_scores.size == 0 or out_scores.size == 0:
        raise ConfigurationError(*("Both in_scores and out_scores must be non-empty",))

    if not np.all(np.isfinite(in_scores)) or not np.all(np.isfinite(out_scores)):
        raise ConfigurationError(*("scores must contain only finite values",))

    thresholds, tn_counts, fn_counts = get_tn_fn_counts(in_scores, out_scores)

    tp_counts = (fn_counts[-1] - fn_counts)[::-1]
    fp_counts = (tn_counts[-1] - tn_counts)[::-1]

    return OneRunEstimate(
        n_in=len(in_scores),
        n_out=len(out_scores),
        thresholds=thresholds,
        tn_counts=tn_counts,
        fn_counts=fn_counts,
        tp_counts=tp_counts,
        fp_counts=fp_counts,
        in_scores=in_scores,
        out_scores=out_scores,
        canary_indices=coin_flip.canary_indices,
    )


@dataclasses.dataclass(frozen=True)
class OneRunEstimate:
    """Precomputed one-run audit estimate.

    Constructed by :func:`one_run`.  Holds the empirical ROC counts
    shared by every audit method.

    Epsilon estimation: call :meth:`eps_delta` or :meth:`gdp` to get the
    corresponding audit method, then ``.epsilon_at(delta=…)``.
    Attack-side metrics (:meth:`auc`, :meth:`beta_at`) are independent of
    the audit method and live directly on the estimate.
    """

    n_in: int
    n_out: int
    thresholds: np.ndarray
    tn_counts: np.ndarray
    fn_counts: np.ndarray
    tp_counts: np.ndarray
    fp_counts: np.ndarray
    in_scores: np.ndarray
    out_scores: np.ndarray
    #: Stable example identifiers: the dataset indices of the audited
    #: canaries, in the order the scores were split.  Always populated by
    #: :func:`one_run`; ``None`` only for directly-constructed estimates.
    canary_indices: np.ndarray | None = None

    def __repr__(self) -> str:
        auc = _auc_from_counts(self.tn_counts, self.fn_counts)
        return f"OneRunEstimate(n_in={self.n_in}, n_out={self.n_out}, auc={auc:.4f})"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _best_r_u(
        self,
        threshold: float | None = None,
    ) -> tuple[int, int]:
        """Compute (r, u) for the order-statistics test.

        ``r`` is the number of released guesses (always m = n_in + n_out);
        ``u`` is the number of errors (m minus the number of correct
        guesses).  If ``threshold`` is given, accuracy is evaluated there;
        otherwise the Pareto-optimal threshold maximising TP + TN is used.
        """
        m = self.n_in + self.n_out

        if threshold is not None:
            tp = int(np.sum(self.in_scores >= threshold))
            tn = int(np.sum(self.out_scores < threshold))
            return m, m - (tp + tn)

        correct = (self.n_in - self.fn_counts) + self.tn_counts
        best_c = int(np.max(correct))
        return m, m - best_c

    # ------------------------------------------------------------------
    # Audit methods
    # ------------------------------------------------------------------

    def eps_delta(self) -> EpsDeltaMethod:
        """(ε, δ)-DP order-statistics audit method (Xiang et al. 2025).

        Mechanism-agnostic.
        """
        return EpsDeltaMethod(_estimate=self)

    def gdp(self, *, grid_size: int = 10_000) -> GdpMethod:
        """μ-GDP order-statistics audit method (Xiang et al. 2025).

        Args:
            grid_size: Grid points for the order-statistics integration.
                Must be at least ``_MIN_GRID_SIZE``; coarser grids can
                resolve the p-value as if no leakage were present.

        Raises:
            ConfigurationError: If ``grid_size`` is below the minimum.
        """
        if grid_size < _MIN_GRID_SIZE:
            raise ConfigurationError(
                *(f"grid_size must be >= {_MIN_GRID_SIZE}, got {grid_size}",)
            )
        return GdpMethod(_estimate=self, grid_size=grid_size)

    # ------------------------------------------------------------------
    # Pld-mirror surface — dispatches to gdp() (paper-recommended default)
    # ------------------------------------------------------------------

    def epsilon_at(
        self,
        *,
        delta: float,
        significance: float = 0.05,
        threshold: float | None = None,
    ) -> float:
        """Epsilon lower bound from the default μ-GDP audit method.

        Shortcut for ``self.gdp().epsilon_at(...)``.  For non-Gaussian-DP
        mechanisms use ``self.eps_delta().epsilon_at(...)`` explicitly.
        Requires ``delta > 0`` — μ-GDP is incompatible with pure ε-DP.
        """
        return self.gdp().epsilon_at(
            delta=delta,
            significance=significance,
            threshold=threshold,
        )

    def delta_at(
        self,
        *,
        epsilon: float,
        significance: float = 0.05,
        threshold: float | None = None,
    ) -> float:
        """δ(ε) from the default μ-GDP audit method.

        Shortcut for ``self.gdp().delta_at(...)``.
        """
        return self.gdp().delta_at(
            epsilon=epsilon,
            significance=significance,
            threshold=threshold,
        )

    def beta_at(
        self,
        *,
        alpha: float,
        significance: float = 0.05,
        threshold: float | None = None,
    ) -> float:
        """f-DP Type-II error at α from the default μ-GDP audit method.

        Shortcut for ``self.gdp().beta_at(...)`` — *theoretical* β at the
        inferred μ̂.  For the *empirical* attack-ROC β (1 − TPR at given
        FPR), see :meth:`attack_beta_at`.
        """
        return self.gdp().beta_at(
            alpha=alpha,
            significance=significance,
            threshold=threshold,
        )

    def advantage(
        self,
        *,
        significance: float = 0.05,
        threshold: float | None = None,
    ) -> float:
        """Total-variation advantage from the default μ-GDP audit method.

        Shortcut for ``self.gdp().advantage(...)``.
        """
        return self.gdp().advantage(
            significance=significance,
            threshold=threshold,
        )

    # ------------------------------------------------------------------
    # Attack-side empirical metrics
    # ------------------------------------------------------------------

    def attack_auc(
        self,
        *,
        confidence: float | None = None,
        num_samples: int = 1000,
        key: RngKey | None = None,
    ) -> float | tuple[float, float]:
        """Area under the ROC curve for the membership inference attack.

        AUC = 0.5 means random guessing, AUC = 1.0 means perfect attack.

        When ``confidence`` is provided, returns a confidence interval
        as a ``(lower, upper)`` tuple instead of a point estimate.

        Args:
            confidence: If provided, return a symmetric CI at this level
                (e.g. 0.95 for 95% CI). Must be in (0, 1).
            num_samples: Number of resamples for CI. Default: 1000.
            key: RNG key for reproducible resampling. Required when
                ``confidence`` is provided.

        Returns:
            Float AUC if ``confidence`` is None, otherwise
            ``(lower, upper)`` tuple.
        """
        point = _auc_from_counts(self.tn_counts, self.fn_counts)

        if confidence is None:
            return point

        if not 0 < confidence < 1:
            raise ConfigurationError(
                *(f"confidence must be in (0, 1), got {confidence}",)
            )
        if key is None:
            raise ConfigurationError(
                *(
                    "attack_auc(confidence=...) requires an explicit RNG key for "
                    "bootstrap resampling",
                )
            )

        significance = 1 - confidence
        quantiles = (significance / 2, 1 - significance / 2)

        rng = np.random.default_rng(seed=key.seed)
        values = np.empty(num_samples)
        for i in range(num_samples):
            in_sample = rng.choice(self.in_scores, size=self.n_in)
            out_sample = rng.choice(self.out_scores, size=self.n_out)
            _, tn, fn = get_tn_fn_counts(in_sample, out_sample)
            values[i] = _auc_from_counts(tn, fn)

        # Bias-corrected bootstrap
        prop_less = (int(np.sum(values < point)) + 1) / (num_samples + 2)
        z0 = scipy.stats.norm.ppf(prop_less)
        z_q = scipy.stats.norm.ppf(quantiles)
        corrected = scipy.stats.norm.cdf(2 * z0 + z_q)

        ci = np.quantile(values, corrected, method="linear")
        return (float(ci[0]), float(ci[1]))

    def attack_beta_at(self, *, alpha: float | np.ndarray) -> float | np.ndarray:
        """Empirical attack β: 1 − TPR at FPR = ``alpha``.

        Interpolated from the attack's empirical ROC;
        independent of the audit method.  For the *theoretical* β under
        the inferred μ̂-GDP guarantee, use :meth:`beta_at`.

        Args:
            alpha: Type-I error rate(s) (false positive rate) in [0, 1].

        Returns:
            Empirical Type-II error rate(s) at the specified alpha(s).
        """
        tpr = tpr_at_given_fpr(alpha, self.tp_counts, self.fp_counts)
        return 1.0 - tpr


def _auc_from_counts(tn_counts: np.ndarray, fn_counts: np.ndarray) -> float:
    """Compute AUC from precomputed TN/FN count arrays."""
    tnr = tn_counts / tn_counts[-1]
    fnr = fn_counts / fn_counts[-1]
    return float(0.5 * np.dot(tnr[:-1] + tnr[1:], fnr[1:] - fnr[:-1]))
