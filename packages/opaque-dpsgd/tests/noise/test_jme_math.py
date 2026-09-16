"""Mathematical checks for projected JME calibration."""

from __future__ import annotations

import math
from decimal import Decimal, localcontext

import numpy as np
import pytest
from scipy.optimize import minimize, minimize_scalar

from opaque.api.dpsgd.noise._jme import (
    _PAPER_D1_ETA,
    _fixed_distance_square_sensitivity_sq,
    _joint_sensitivity_sq_raw,
    _paper_lambda,
    _resolve_jme_parameters,
)
from opaque.dpsgd.noise.types import JmeAllocation


def _square_displacement(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum((x * x - y * y) ** 2))


@pytest.mark.parametrize("ratio", [0.0, 0.1, 0.5, 2.0 / 3.0])
def test_collinear_envelope_witness(ratio: float):
    R = 1.7
    r = ratio * R
    x = np.array([R, 0.0])
    y = np.array([R - r, 0.0])
    expected = _fixed_distance_square_sensitivity_sq(r, R, dimension=2)
    assert _square_displacement(x, y) == pytest.approx(expected)


@pytest.mark.parametrize("ratio", [2.0 / 3.0, 0.9, math.sqrt(2.0), 1.9, 2.0])
def test_two_coordinate_envelope_witness(ratio: float):
    R = 2.3
    r = ratio * R
    s = math.sqrt(max(0.0, 4.0 * R * R - r * r))
    u = np.array([r / math.sqrt(2.0), r / math.sqrt(2.0)])
    v = np.array([s / math.sqrt(2.0), -s / math.sqrt(2.0)])
    x = (u + v) / 2.0
    y = (v - u) / 2.0
    assert np.linalg.norm(x) == pytest.approx(R)
    assert np.linalg.norm(y) == pytest.approx(R)
    assert np.linalg.norm(x - y) == pytest.approx(r)
    expected = _fixed_distance_square_sensitivity_sq(r, R, dimension=2)
    assert _square_displacement(x, y) == pytest.approx(expected)


def test_one_dimension_has_only_collinear_envelope():
    R = 1.0
    r = 1.0
    scalar = _fixed_distance_square_sensitivity_sq(r, R, dimension=1)
    vector = _fixed_distance_square_sensitivity_sq(r, R, dimension=2)
    assert scalar == pytest.approx(1.0)
    assert vector == pytest.approx(1.5)


@pytest.mark.parametrize("dimension", [1, 2, 100])
def test_paper_reference_recovers_unrestricted_sensitivity(dimension: int):
    R = 1.9
    lambda_ = _paper_lambda(aggregate_norm=R, dimension=dimension)
    sensitivity_sq = _joint_sensitivity_sq_raw(
        delta=2.0 * R,
        aggregate_norm=R,
        lambda_=lambda_,
        dimension=dimension,
    )
    assert sensitivity_sq == pytest.approx(4.0 * R * R)


def test_paper_one_dimensional_reciprocal_constant():
    R = 3.0
    lambda_ = _paper_lambda(aggregate_norm=R, dimension=1)
    assert lambda_ * R * R == pytest.approx(_PAPER_D1_ETA)
    assert pytest.approx((11.0 + 5.0 * math.sqrt(5.0)) / 8.0) == _PAPER_D1_ETA


def test_joint_sensitivity_is_stable_when_r_four_would_underflow():
    R_float = 1.1939526742415291e-94
    delta_float = 4.763389529578269e-96
    lambda_ = _paper_lambda(aggregate_norm=R_float, dimension=2)
    actual = _joint_sensitivity_sq_raw(
        delta=delta_float,
        aggregate_norm=R_float,
        lambda_=lambda_,
        dimension=2,
    )

    with localcontext() as context:
        context.prec = 80
        R = Decimal(str(R_float))
        delta = Decimal(str(delta_float))
        ratio = delta / R
        eta = Decimal("0.5")
        factor = ratio**2 + eta * ratio**2 * (Decimal(2) - ratio) ** 2
        expected = float(R**2 * factor)

    assert actual == pytest.approx(expected, rel=1e-15)


@pytest.mark.parametrize(
    ("ratio", "eta"),
    [
        (0.2, 0.1),
        (0.55, 4.0),
        (0.9, 0.5),
        (1.2, 2.0),
        (1.7, 0.2),
        (1.9, 8.0),
        (2.4, 1.0),
    ],
)
def test_closed_form_matches_numerical_optimizer_in_one_dimension(
    ratio: float,
    eta: float,
):
    R = 1.4
    D = min(ratio * R, 2.0 * R)
    lambda_ = eta / (R * R)

    def objective(r: float) -> float:
        return -(r * r + lambda_ * r * r * (2.0 * R - r) ** 2)

    numerical = -minimize_scalar(
        objective,
        bounds=(0.0, D),
        method="bounded",
        options={"xatol": 1e-13},
    ).fun
    exact = _joint_sensitivity_sq_raw(
        delta=D,
        aggregate_norm=R,
        lambda_=lambda_,
        dimension=1,
    )
    assert numerical == pytest.approx(exact, rel=5e-7, abs=5e-9)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("ratio", "eta"),
    [
        (0.3, 0.2),
        (0.65, 5.0),
        (0.9, 0.5),
        (1.4, 2.0),
        (1.9, 0.2),
        (1.9, 8.0),
    ],
)
def test_closed_form_matches_multivariate_numerical_optimizer(
    ratio: float,
    eta: float,
):
    R = 1.0
    D = ratio * R
    lambda_ = eta / (R * R)

    def objective(z: np.ndarray) -> float:
        x, y = z[:2], z[2:]
        return -(np.sum((x - y) ** 2) + lambda_ * np.sum((x * x - y * y) ** 2))

    constraints = (
        {"type": "ineq", "fun": lambda z: R * R - np.dot(z[:2], z[:2])},
        {"type": "ineq", "fun": lambda z: R * R - np.dot(z[2:], z[2:])},
        {
            "type": "ineq",
            "fun": lambda z: D * D - np.dot(z[:2] - z[2:], z[:2] - z[2:]),
        },
    )
    generator = np.random.default_rng(407)
    starts = []
    for _ in range(96):
        x = generator.normal(size=2)
        y = generator.normal(size=2)
        x *= R * generator.random() ** 0.5 / np.linalg.norm(x)
        y *= R * generator.random() ** 0.5 / np.linalg.norm(y)
        if np.linalg.norm(x - y) > D:
            y = x + D * (y - x) / np.linalg.norm(y - x)
            if np.linalg.norm(y) > R:
                y *= R / np.linalg.norm(y)
        starts.append(np.concatenate((x, y)))

    numerical = -math.inf
    for start in starts:
        result = minimize(
            objective,
            start,
            method="SLSQP",
            constraints=constraints,
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        violations = [float(constraint["fun"](result.x)) for constraint in constraints]
        if min(violations) >= -1e-8:
            numerical = max(numerical, -float(result.fun))

    exact = _joint_sensitivity_sq_raw(
        delta=D,
        aggregate_norm=R,
        lambda_=lambda_,
        dimension=2,
    )
    assert numerical == pytest.approx(exact, rel=2e-6, abs=2e-7)


@pytest.mark.parametrize("max_ratio", [1.01, 1.1, 1.5, 3.0])
@pytest.mark.parametrize("dimension", [1, 2, 20])
@pytest.mark.parametrize("distance_ratio", [0.2, 0.8, 1.5, 2.0])
def test_first_variance_cap_is_respected(
    max_ratio: float,
    dimension: int,
    distance_ratio: float,
):
    R = 1.3
    delta = distance_ratio * R
    parameters = _resolve_jme_parameters(
        noise_multiplier=1.0,
        delta=delta,
        aggregate_norm=R,
        allocation=JmeAllocation.first_variance_cap(max_ratio),
        dimension=dimension,
    )
    D = min(delta, 2.0 * R)
    max_stddev = math.sqrt(max_ratio) * D
    assert parameters.first_stddev <= max_stddev


def test_larger_feasible_lambda_reduces_second_stream_variance():
    R = 1.0
    delta = 0.8
    max_ratio = 1.4
    parameters = _resolve_jme_parameters(
        noise_multiplier=1.0,
        delta=delta,
        aggregate_norm=R,
        allocation=JmeAllocation.first_variance_cap(max_ratio),
        dimension=2,
    )
    larger_lambda = parameters.lambda_ * (1.0 + 1e-10)
    larger_sensitivity = _joint_sensitivity_sq_raw(
        delta=delta,
        aggregate_norm=R,
        lambda_=larger_lambda,
        dimension=2,
    )
    assert larger_sensitivity > max_ratio * delta * delta
