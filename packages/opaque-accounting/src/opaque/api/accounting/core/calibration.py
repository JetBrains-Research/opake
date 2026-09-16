# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
#
# Calibration workflow and budget-driven privacy search adapted in part from
# Google DP Accounting (Apache-2.0;
# https://github.com/google/differential-privacy/tree/main/python/dp_accounting),
# then reworked for Opaque's process algebra and native PLD backend.
# See ../../../../../NOTICE in this package for the full attribution.
"""Binary search calibration for finding parameters that achieve target privacy.

This module provides a generic binary search framework.  Privacy budget
*budgets* (epsilon, delta, advantage, beta, risk) live in
:mod:`opaque.accounting._budgets` and are re-exported here for convenience.

Example::

    from opaque.accounting import calibration as cal
    import opaque.accounting as acc

    budget = cal.epsilon_budget(3.0, delta=1e-5)
    result = cal.calibrate(
        budget,
        lambda nm: acc.poisson(acc.gaussian(nm), sample_rate=0.01) * 1000,
        param_min=0.1, param_max=5.0,
    )

    print(f"Noise multiplier: {result.param:.3f}")
    print(f"Achieved epsilon: {result.achieved:.6f}")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

# Re-export budgets so ``from opaque.api.accounting.core.calibration import Budget``
# and ``from opaque.accounting import calibration as cal; cal.epsilon_budget(...)``
# work as convenience imports.
from opaque.api.accounting.core._budgets import (
    AdvantageBudget,
    BetaBudget,
    Budget,
    DeltaBudget,
    EpsilonBudget,
    RiskBudget,
    advantage_budget,
    beta_budget,
    delta_budget,
    epsilon_budget,
    risk_budget,
)
from opaque.api.accounting.core._native_cache import _clear_all_native_caches
from opaque.api.accounting.core.discretization import (
    _use_discretization,
    get_discretization,
)
from opaque.exceptions import CalibrationError

if TYPE_CHECKING:
    from collections.abc import Callable

    from opaque.api.accounting.core._base import DpProcess

# =============================================================================
# Calibration
# =============================================================================


@dataclass
class CalibrateResult:
    """Result from calibration binary search.

    Attributes:
        param: Found parameter value (e.g., noise_multiplier).
        achieved: Achieved metric value.
        target: Target metric value.
        iterations: Number of binary search iterations.
        converged: Always ``True`` for results returned by :func:`calibrate`.
        mc_failure_probability: Overall failure probability covering every
            adaptive Monte Carlo probe. Zero for analytic calibration.
    """

    param: float
    achieved: float
    target: float
    iterations: int
    converged: bool
    mc_failure_probability: float = 0.0

    @property
    def mc_confidence(self) -> float:
        """Confidence level covering all adaptive Monte Carlo probes."""
        return 1.0 - self.mc_failure_probability

    def __repr__(self) -> str:
        status = "converged" if self.converged else "max iterations"
        confidence = (
            f", mc_confidence={self.mc_confidence:.6f}"
            if self.mc_failure_probability > 0.0
            else ""
        )
        return (
            f"CalibrateResult(param={self.param:.6f}, "
            f"achieved={self.achieved:.6f}, target={self.target:.6f}, "
            f"iterations={self.iterations}, {status}{confidence})"
        )


def calibrate(
    budget: Budget,
    process: Callable[[float], DpProcess],
    param_min: float,
    param_max: float,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
    prefix: DpProcess | None = None,
) -> CalibrateResult:
    """Binary search for parameter achieving target privacy metric.

    Finds the value of a parameter (e.g. noise_multiplier, sample rate,
    or step count) such that the resulting DpProcess achieves a target
    privacy guarantee.

    **Metric Direction:**
    Each budget declares a ``decreasing`` property describing the metric
    *kind*: privacy-loss metrics (epsilon/delta/advantage) are safe
    at-or-below the target, privacy-gain metrics (beta/risk) at-or-above.
    The direction the metric moves as the calibrated *parameter* grows is
    derived automatically by probing the metric at both endpoints — a
    noise multiplier decreases privacy-loss metrics while a sample rate
    or step count increases them, and both directions are supported.  The
    metric must be monotone in the parameter over ``[param_min,
    param_max]``; flat or detectably non-monotone parameterizations raise
    :class:`~opaque.exceptions.CalibrationError`.

    Successful results are one-sided and privacy-safe.  A decreasing
    privacy-loss metric satisfies ``achieved <= target``; an increasing
    privacy-gain metric satisfies ``achieved >= target``.  In both cases,
    ``achieved`` and ``target`` are close under the requested relative
    tolerance.  If no safe endpoint reaches that tolerance, calibration
    raises instead of returning an under-noised parameter.

    **Parameters:**

    Args:
        budget: Calibration budget (created with epsilon_budget(), delta_budget(), etc.)
            - Must have: budget.value (float), budget.evaluate(process) → float
            - Common budgets: epsilon_budget(3.0, 1e-5), delta_budget(1e-5, 3.0), advantage_budget(0.1)
            - Budgets validate themselves: epsilon_budget(-1.0) raises
              PrivacyBudgetError

        process: Callable taking a float parameter and returning a DpProcess.
            Must be deterministic (same input → same process).
            Example: ``lambda nm: acc.poisson(acc.gaussian(nm), 0.01) * 1000``

            Important: If process() raises an exception, it propagates immediately.

        param_min: Lower bound for the search.  Exactly one endpoint of
            ``[param_min, param_max]`` must be privacy-safe for the
            target; which one is detected automatically by probing.
            - Example: a low noise_multiplier (unsafe end), or a low
              sample rate (safe end)

        param_max: Upper bound for the search (see ``param_min`` — the
            safe endpoint is auto-detected, not positional).
            - Example: a high noise_multiplier (safe end), or a high
              sample rate (unsafe end)

        tolerance: Positive, finite relative convergence tolerance
            - Uses ``math.isclose(achieved, target, rel_tol=tolerance, abs_tol=0)``
            - Default: 1e-6; discretization can limit attainable precision

        max_iterations: Positive maximum number of binary search iterations
            - Each iteration halves the search space
            - Exhaustion or stalled bounds raises CalibrationError

        prefix: Optional already-executed process; each probe evaluates
            ``prefix | process(param)``, so the budget is the total across
            both. Cached internally — its PLD is computed once per search.

    Returns:
        CalibrateResult with:
        - param: Found parameter value
        - achieved: Metric value at found parameter
        - target: Target metric value (for comparison)
        - iterations: Number of iterations performed
        - converged: always True for a successfully returned result
        - mc_failure_probability: overall failure probability covering all
          adaptive MC probes (zero for analytic calibration)

    Raises:
        CalibrationError: If tolerance is not finite and positive, or
            max_iterations is not positive.
        CalibrationError: If param_min >= param_max.
        CalibrationError: If bounds don't bracket the target (neither or both
            endpoints privacy-safe).
        CalibrationError: If budget evaluation returns NaN at the bounds, or infinity
            on the privacy-safe endpoint (infinity on the unsafe endpoint is
            accepted — the target is below it by definition).
        CalibrationError: If the metric is flat across the bounds, or an interior
            probe escapes the endpoint value envelope (non-monotone
            parameterization).
        CalibrationError: If no privacy-safe endpoint converges within
            max_iterations.
        Exception: If process() or budget.evaluate() raises an exception

    Native LRU caches registered via :mod:`opaque.api.accounting.core._native_cache`
    are cleared in ``finally`` so calibration probes do not retain Rust-backed
    memory after the search completes (success, early exit, or exception).

    **Examples:**

    **Example 1: Standard (ε, δ)-DP**::

        import opaque.accounting as acc
        from opaque.accounting import calibration as cal

        budget = cal.epsilon_budget(3.0, delta=1e-5)
        result = cal.calibrate(
            budget,
            lambda nm: acc.poisson(acc.gaussian(nm), sample_rate=0.01) * 1000,
            param_min=0.7,
            param_max=3.5,
        )

        print(f"Use noise_multiplier = {result.param:.4f}")
        print(f"Achieves epsilon = {result.achieved:.6f}")
        print(f"Converged: {result.converged}")

    **Example 2: Multi-phase training**::

        def multiphase(nm):
            phase1 = acc.poisson(acc.gaussian(nm), 0.01) * 500
            phase2 = acc.poisson(acc.gaussian(nm * 0.8), 0.01) * 500
            phase3 = acc.poisson(acc.gaussian(nm * 0.5), 0.01) * 500
            return phase1 | phase2 | phase3

        result = cal.calibrate(
            cal.epsilon_budget(5.0, delta=1e-5),
            multiphase,
            param_min=0.5,
            param_max=3.0,
            tolerance=0.01,
        )

    **Example 3: Calibrating a second stage over a prefix**::

        # SFT already ran; load its executed process from the saved accountant
        import json

        from opaque.accounting import Accountant
        from opaque.serialization import from_state_dict

        with open("sft_checkpoint/accountant.json") as f:
            sft = from_state_dict(Accountant(), json.load(f))

        # Find the DPO noise multiplier so that the *total* (SFT + DPO)
        # privacy cost hits the budget.
        result = cal.calibrate(
            cal.epsilon_budget(8.0, delta=1e-6),
            lambda nm: acc.poisson(acc.gaussian(nm), 0.02) * 2000,
            param_min=0.5,
            param_max=5.0,
            prefix=sft.process,
        )

    **Example 4: Different privacy metrics**::

        process = lambda nm: acc.poisson(acc.gaussian(nm), 0.01) * 1000

        # f-DP advantage
        result = cal.calibrate(cal.advantage_budget(0.1), process, 0.5, 2.0)

        # (α, β) error rates
        result = cal.calibrate(cal.beta_budget(0.05, alpha=0.01), process, 0.5, 2.0)

        # Bayes risk
        result = cal.calibrate(cal.risk_budget(0.1, prior=0.5), process, 0.5, 2.0)

    **Troubleshooting:**

    - *CalibrationError: bounds don't bracket budget*
      → Try expanding param_min/param_max range
    - *CalibrationError: budget evaluation returned infinity*
      → Privacy budget may be unreachable with your mechanism; try higher param_min or lower budget value
    - *CalibrationError: calibration did not converge*
      → Increase tolerance or max_iterations; check that param changes actually affect metric
    """
    if prefix is not None:
        from opaque.api.accounting.core.composition._cached import cached

        cached_prefix = cached(prefix)
        inner = process

        def process(param: float) -> DpProcess:
            return cached_prefix | inner(param)

    try:
        overall_config = get_discretization()
        probe_count = max_iterations + 2
        probe_config = replace(
            overall_config,
            mc_failure_probability=(
                overall_config.mc_failure_probability / probe_count
            ),
        )
        with _use_discretization(probe_config):
            result = _calibrate_impl(
                budget,
                process,
                param_min,
                param_max,
                tolerance,
                max_iterations,
            )

        # Re-evaluate the calibrated process under the runtime discretization.
        # The probe context uses stricter MC bounds; re-evaluate under overall_config
        # to get the achieved epsilon that will actually apply during training.
        with _use_discretization(overall_config):
            final_process = process(result.param)
            achieved = budget.evaluate(final_process)
            runtime_safe = (
                achieved <= budget.value
                if budget.decreasing
                else achieved >= budget.value
            )
            if not runtime_safe:
                result = _calibrate_impl(
                    budget,
                    process,
                    param_min,
                    param_max,
                    tolerance,
                    max_iterations,
                )
                final_process = process(result.param)
                achieved = result.achieved
            if not math.isclose(achieved, budget.value, rel_tol=tolerance, abs_tol=0.0):
                message = (
                    f"Calibration for {budget.name} did not converge under runtime "
                    f"discretization: target={budget.value}, achieved={achieved}, "
                    f"relative tolerance={tolerance}, param={result.param}."
                )
                raise CalibrationError(message)
            result.achieved = achieved

            pld_method = getattr(final_process, "pld", None)
            runtime_kwargs = {}
            if isinstance(budget, EpsilonBudget):
                runtime_kwargs["mc_resolution"] = min(
                    overall_config.mc_resolution,
                    budget.delta / 2.0,
                )
            final_pld = pld_method(**runtime_kwargs) if callable(pld_method) else None
            if final_pld is not None and final_pld.mc_failure_probability > 0.0:
                result.mc_failure_probability = overall_config.mc_failure_probability

            return result
    finally:
        _clear_all_native_caches()


def _calibrate_impl(
    budget: Budget,
    process: Callable[[float], DpProcess],
    param_min: float,
    param_max: float,
    tolerance: float,
    max_iterations: int,
) -> CalibrateResult:
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise CalibrationError(*(f"tolerance must be finite and > 0, got {tolerance}",))
    if max_iterations <= 0:
        raise CalibrationError(*(f"max_iterations must be > 0, got {max_iterations}",))
    if param_min >= param_max:
        raise CalibrationError(
            *(f"param_min ({param_min}) must be < param_max ({param_max})",)
        )

    # Check bounds bracket the budget
    proc_min = process(param_min)
    proc_max = process(param_max)
    val_min = budget.evaluate(proc_min)
    val_max = budget.evaluate(proc_max)

    # Validate bounds don't return nan
    if math.isnan(val_min) or math.isnan(val_max):
        raise CalibrationError(
            *(
                f"Budget evaluation returned NaN at bounds: "
                f"at param_min={param_min}: {val_min}, at param_max={param_max}: {val_max}. "
                f"This usually indicates an issue with the process() function.",
            )
        )

    if math.isinf(val_min) and math.isinf(val_max):
        raise CalibrationError(
            *(
                f"Budget evaluation returned infinity at both bounds: "
                f"at param_min={param_min}: {val_min}, at param_max={param_max}: {val_max}. "
                f"This typically means the privacy target is unreachable with these parameter bounds. "
                f"Try expanding the search range or checking that process() produces valid DpProcess objects.",
            )
        )

    # Metric-kind: privacy-loss metrics (epsilon/delta/advantage) are safe
    # at-or-below the target; privacy-gain metrics (beta/risk) at-or-above.
    # This is a property of the METRIC.  The direction the metric moves as
    # the calibrated PARAMETER grows is derived below by probing both
    # bracket ends (values already computed above) — noise multipliers make
    # privacy-loss metrics decrease, but sample rates / step counts make
    # them increase, and both are valid calibration targets.
    safe_when_below = budget.decreasing

    def is_safe(achieved: float) -> bool:
        if safe_when_below:
            return achieved <= budget.value
        return achieved >= budget.value

    if val_min == val_max:
        raise CalibrationError(
            *(
                f"Metric {budget.name} is flat over param range "
                f"[{param_min}, {param_max}] (value {val_min} at both ends); "
                f"cannot infer a monotone search direction. Check that the "
                f"parameter actually affects the process.",
            )
        )
    safe_at_max = (val_max < val_min) if safe_when_below else (val_max > val_min)

    if safe_at_max:
        safe_param, safe_value = param_max, val_max
        unsafe_param, unsafe_value = param_min, val_min
    else:
        safe_param, safe_value = param_min, val_min
        unsafe_param, unsafe_value = param_max, val_max

    # inf on the unsafe endpoint is fine (the target is < inf); inf on the
    # safe endpoint means the target is unreachable within the bracket.
    if math.isinf(safe_value):
        raise CalibrationError(
            *(
                f"Budget evaluation returned infinity on the privacy-safe endpoint: "
                f"at param_min={param_min}: {val_min}, at param_max={param_max}: {val_max}. "
                f"The privacy target is unreachable with these parameter bounds.",
            )
        )
    if not is_safe(safe_value):
        raise CalibrationError(
            *(
                f"Budget {budget.name}={budget.value:.6f} not bracketed: neither "
                f"endpoint of [{param_min}, {param_max}] is privacy-safe "
                f"(values {val_min:.6f}, {val_max:.6f}). "
                f"The target may be unreachable with these bounds.",
            )
        )
    if not math.isinf(unsafe_value) and is_safe(unsafe_value):
        raise CalibrationError(
            *(
                f"Budget {budget.name}={budget.value:.6f} not bracketed: both "
                f"endpoints of [{param_min}, {param_max}] are privacy-safe "
                f"(values {val_min:.6f}, {val_max:.6f}). If the metric is not "
                f"monotone in the parameter, calibration is unsupported.",
            )
        )

    # Monotonicity envelope: any interior probe of a monotone metric stays
    # inside the endpoint value range; escaping it means the parameterization
    # is not monotone and bisection would silently mis-calibrate.  Padded by
    # a small relative tolerance so PLD-grid noise near the endpoints (the
    # metric's numerical floor) cannot hard-fail a legitimate calibration.
    env_lo = min(val_min, val_max)
    env_hi = max(val_min, val_max)
    env_pad = 1e-9 * max(abs(env_lo), abs(env_hi), 1.0)

    iterations = 0

    if math.isclose(
        safe_value,
        budget.value,
        rel_tol=tolerance,
        abs_tol=0.0,
    ):
        return CalibrateResult(
            param=safe_param,
            achieved=safe_value,
            target=budget.value,
            iterations=iterations,
            converged=True,
        )

    for iteration in range(max_iterations):
        mid = (unsafe_param + safe_param) / 2
        if mid in (unsafe_param, safe_param):
            break
        iterations = iteration + 1
        proc = process(mid)
        current = budget.evaluate(proc)

        if math.isnan(current):
            raise CalibrationError(
                *(
                    f"Budget evaluation returned NaN at param={mid} "
                    f"(iteration {iterations}). Check that process() "
                    f"produces valid DpProcess objects for all parameter values.",
                )
            )

        # A monotone metric stays inside the endpoint value envelope.
        if not math.isinf(current) and not (
            env_lo - env_pad <= current <= env_hi + env_pad
        ):
            raise CalibrationError(
                *(
                    f"Metric {budget.name} is not monotone in the calibrated "
                    f"parameter: value {current:.6f} at param={mid} lies outside "
                    f"the bracket value range [{env_lo:.6f}, {env_hi:.6f}].",
                )
            )

        if is_safe(current):
            safe_param = mid
            safe_value = current
        else:
            unsafe_param = mid

        # Convergence is accepted only from the proven privacy-safe endpoint.
        if math.isclose(
            safe_value,
            budget.value,
            rel_tol=tolerance,
            abs_tol=0.0,
        ):
            return CalibrateResult(
                param=safe_param,
                achieved=safe_value,
                target=budget.value,
                iterations=iterations,
                converged=True,
            )

    raise CalibrationError(
        *(
            f"Calibration for {budget.name} did not converge: "
            f"target={budget.value}, relative tolerance={tolerance}, "
            f"iterations={iterations}, "
            f"final bracket=(unsafe={unsafe_param}, safe={safe_param}), "
            f"last safe achieved={safe_value} at param={safe_param}.",
        )
    )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Re-exported from budgets (convenience)
    "Budget",
    "EpsilonBudget",
    "DeltaBudget",
    "AdvantageBudget",
    "BetaBudget",
    "RiskBudget",
    "epsilon_budget",
    "delta_budget",
    "advantage_budget",
    "beta_budget",
    "risk_budget",
    # Calibration
    "calibrate",
    "CalibrateResult",
]
