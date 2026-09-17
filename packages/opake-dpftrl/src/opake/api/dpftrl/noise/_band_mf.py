"""BandMF strategy — banded Toeplitz MF mechanism.

Computes optimized banded Toeplitz coefficients on demand from the
strategy's recipe (``bands``, ``momentum``, ``lr_schedule``) and the
amplifier-supplied ``n_steps``.  ``lr_schedule`` is an
:data:`opake.scheduling.types.Schedule` (``Callable[[int], float]``) materialised
to a tensor at workload-coefficient build time.

References:
    - BandMF: https://arxiv.org/abs/2306.08153
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

import torch

from opake.api.dpftrl.noise._strategy_codec import register_strategy
from opake.exceptions import ConfigurationError

from ._schedule_fingerprint import materialize_schedule
from ._sensitivity import minsep_true_max_participations
from ._toeplitz import inverse_as_streaming_matrix, minsep_sensitivity_upper_bound
from ._toeplitz import optimize as optimize_toeplitz

if TYPE_CHECKING:
    from opake.api.engine.scheduling.types import Schedule

    from ._streaming_matrix import StreamingMatrix


def _momentum_workload_coef(
    momentum: float,
    n: int,
) -> torch.Tensor:
    """Compute Toeplitz workload coefficients for momentum-SGD.

    For momentum β, the workload matrix has entries
    ``W[t, s] = β ** (t - s)`` for ``s <= t``. Learning-rate schedules
    scale workload *rows* and are applied separately as per-step query
    weights by the optimizer objective.
    """
    if momentum < 0:
        raise ConfigurationError(*(f"momentum must be >= 0, got {momentum}",))
    if momentum == 0.0:
        warnings.warn(
            "momentum=0.0 produces an identity workload — MF noise will "
            "reduce to independent noise with no benefit over standard "
            "Gaussian (DP-SGD). This is useful for testing but not for "
            "production training.",
            stacklevel=3,
        )
        coef = torch.zeros(n, dtype=torch.float64)
        coef[0] = 1.0
        return coef

    return torch.tensor([momentum**i for i in range(n)], dtype=torch.float64)


# ---------------------------------------------------------------------------
# Cached coefficient computation
# ---------------------------------------------------------------------------


_lr_key = materialize_schedule


@lru_cache(maxsize=32)
def _band_mf_coefficients_cached(
    n_steps: int,
    bands: int,
    momentum: float,
    lr_key: tuple[float, ...] | None,
) -> torch.Tensor:
    """Run the BandMF Toeplitz optimization for the given recipe + horizon."""
    if n_steps < 1:
        raise ConfigurationError(*(f"n_steps must be >= 1, got {n_steps}",))
    if bands < 1 or bands > n_steps:
        raise ConfigurationError(
            *(f"bands must be in [1, n_steps={n_steps}], got {bands}",)
        )
    lr = torch.tensor(lr_key, dtype=torch.float64) if lr_key is not None else None
    workload_coef = _momentum_workload_coef(momentum, n_steps)
    return optimize_toeplitz(
        n_steps,
        bands,
        workload_coef=workload_coef,
        query_weights=lr,
    )


# ---------------------------------------------------------------------------
# Strategy dataclass and factory
# ---------------------------------------------------------------------------


@register_strategy
@dataclass(frozen=True, slots=True)
class BandMfStrategy:
    """BandMF banded Toeplitz strategy — recipe only.

    All derived data (coefficients, sensitivity, streaming matrix) is
    computed on demand via the strategy methods, keyed on the
    amplifier-supplied ``n_steps``.  No gram matrix: BandMF uses cyclic
    Poisson / b-min-sep amplification, not BnB.
    """

    bands: int
    momentum: float = 1.0
    lr_schedule: Schedule | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.bands < 1:
            raise ConfigurationError(*(f"bands must be >= 1, got {self.bands}",))
        if self.momentum < 0:
            raise ConfigurationError(*(f"momentum must be >= 0, got {self.momentum}",))

    def coefficients(self, *, n_steps: int, **_) -> torch.Tensor:
        return _band_mf_coefficients_cached(
            n_steps, self.bands, self.momentum, _lr_key(self.lr_schedule, n_steps)
        )

    def gram_matrix(self, **_) -> tuple[float, ...]:
        # BandMF uses Poisson / b-min-sep amplification, not BnB.
        raise NotImplementedError(
            "BandMfStrategy does not provide a gram matrix (uses Poisson / "
            "b-min-sep amplification, not BnB)."
        )

    def streaming_matrix(self, *, n_steps: int, **_) -> StreamingMatrix:
        coefs = self.coefficients(n_steps=n_steps)
        return inverse_as_streaming_matrix(coefs)

    def sensitivity(
        self,
        *,
        n_steps: int,
        min_sep: int = 1,
        max_participations: int | None = None,
    ) -> float:
        r"""L2 sensitivity of ``C`` under the declared participation schema.

        ``C`` is ``bands``-banded and lower triangular, so the columns of
        any two participations separated by at least ``bands`` rows are
        orthogonal and the schema sensitivity
        :math:`\max_{\pi} \lVert \sum_{j \in \pi} C_{[:,j]} \rVert`
        grows with the participation count — up to
        :math:`\kappa \sqrt{k'}` for ``min_sep >= bands``
        (https://arxiv.org/abs/2306.08153, Theorem 2).  Callers that want
        the single-participation column norm :math:`\kappa` must ask for
        it: ``min_sep=n_steps, max_participations=1``.

        Args:
            n_steps: Horizon ``n`` the coefficients are optimized for.
            min_sep: Minimum separation between an example's
                participations (``1`` ⇒ consecutive steps allowed).
            max_participations: Upper bound on participations per example
                (``None`` ⇒ ``ceil(n_steps / min_sep)``).

        Returns:
            The L2 sensitivity: exact for the non-negative, non-increasing
            coefficients the BandMF optimizer produces, and a safe upper
            bound for any recipe whose coefficients leave that domain.
        """
        coefs = self.coefficients(n_steps=n_steps)
        k = minsep_true_max_participations(
            n=n_steps, min_sep=min_sep, max_participations=max_participations
        )
        if k == 1:
            return float(coefs.norm())
        # Both this helper and the strict ``minsep_sensitivity_squared``
        # return a SQUARED sensitivity, hence the ``sqrt`` below.  The bound
        # is preferred over the strict form because the optimizer can emit
        # coefficients a few ulp below zero (e.g. ``momentum=0.0``), which
        # the strict form rejects; on the theorem domain the two agree
        # exactly, so this costs nothing where the strict form applies.
        sens_sq = minsep_sensitivity_upper_bound(
            coefs,
            min_sep=min_sep,
            max_participations=max_participations,
            n=n_steps,
        )
        return float(sens_sq.sqrt())


def band_mf_strategy(
    *,
    bands: int,
    momentum: float = 1.0,
    lr_schedule: Schedule | None = None,
) -> BandMfStrategy:
    """Create a BandMF strategy recipe.

    The factory only validates static recipe knobs.  The Toeplitz
    optimization happens lazily when a strategy method is first called
    with an amplifier-supplied ``n_steps``.

    Args:
        bands: Number of bands in the Toeplitz matrix (>= 1).
        momentum: Polyak momentum coefficient (default 1.0 = prefix-sum).
        lr_schedule: Optional :data:`opake.scheduling.types.Schedule`
            (``Callable[[int], float]``).  Materialised at ``[0, n_steps)``
            when the strategy first sees the amplifier's ``n_steps``. Custom
            schedules must be deterministic, side-effect-free, and remain
            immutable for the strategy's lifetime.

    Returns:
        A :class:`BandMfStrategy` recipe.
    """
    # Recipe bounds live in ``BandMfStrategy.__post_init__``.
    return BandMfStrategy(
        bands=bands,
        momentum=momentum,
        lr_schedule=lr_schedule,
    )


__all__ = ["BandMfStrategy", "band_mf_strategy"]
