"""Cross-validation of projected JME against sampled Gaussian accounting."""

from __future__ import annotations

import math

import pytest

dp_accounting = pytest.importorskip("dp_accounting")

from dp_accounting.pld import privacy_loss_distribution as pld_lib  # noqa: E402

import opaque.dpsgd.accounting as dpsgd_acc  # noqa: E402
from opaque.api.dpsgd.noise._jme import (  # noqa: E402
    _fixed_distance_square_sensitivity_sq,
    _resolve_jme_parameters,
)
from opaque.dpsgd.noise.types import JmeAllocation  # noqa: E402


def _maximizing_distance(
    *,
    delta: float,
    aggregate_norm: float,
    lambda_: float,
) -> float:
    D = min(delta, 2.0 * aggregate_norm)
    interior = math.sqrt(2.0 * aggregate_norm**2 + 1.0 / lambda_)
    return min(D, interior)


@pytest.mark.parametrize(
    "allocation",
    [
        JmeAllocation.paper_reference(),
        JmeAllocation.first_variance_cap(1.4),
    ],
)
def test_maximizing_jme_pair_whitens_to_gaussian_multiplier(allocation):
    noise_multiplier = 0.9
    R = 1.0
    delta = 0.8
    parameters = _resolve_jme_parameters(
        noise_multiplier=noise_multiplier,
        delta=delta,
        aggregate_norm=R,
        allocation=allocation,
        dimension=2,
    )
    distance = _maximizing_distance(
        delta=delta,
        aggregate_norm=R,
        lambda_=parameters.lambda_,
    )
    square_displacement_sq = _fixed_distance_square_sensitivity_sq(
        distance,
        R,
        dimension=2,
    )
    whitened_distance_sq = (
        distance**2 / parameters.first_stddev**2
        + square_displacement_sq / parameters.second_stddev**2
    )
    assert whitened_distance_sq <= 1.0 / noise_multiplier**2
    assert whitened_distance_sq == pytest.approx(
        1.0 / noise_multiplier**2,
        rel=1e-12,
    )


@pytest.mark.parametrize(
    ("noise_multiplier", "sample_rate", "steps", "delta"),
    [
        (0.8, 0.001, 20, 1e-6),
        (1.2, 0.0005, 100, 1e-7),
    ],
)
def test_plain_poisson_jme_uses_sampled_gaussian_dominating_pld(
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    delta: float,
):
    ours = (
        dpsgd_acc.poisson(
            dpsgd_acc.gaussian(noise_multiplier),
            sample_rate=sample_rate,
        )
        * steps
    ).epsilon_at(delta)
    reference = pld_lib.from_gaussian_mechanism(
        noise_multiplier,
        sampling_prob=sample_rate,
    ).self_compose(steps)
    assert ours == pytest.approx(
        reference.get_epsilon_for_delta(delta),
        abs=6e-5,
    )
