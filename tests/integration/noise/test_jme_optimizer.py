"""JME-to-optimizer behavioral integration."""

from __future__ import annotations

import torch

from opaque.dpsgd.noise import jme_noise
from opaque.dpsgd.noise.types import JmeAllocation
from opaque.optimizers import adamw
from opaque.random import key
from opaque.types import clipped


def test_zero_noise_jme_matches_ordinary_adam_for_bounded_aggregate():
    params = torch.tensor([0.2, -0.4])
    aggregate = torch.tensor([0.3, -0.1])
    optimizer = adamw(
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    raw_state = optimizer.init(params)
    jme_state = optimizer.init(params)

    expected_updates, raw_state = optimizer.update(
        aggregate,
        raw_state,
        params=params,
    )
    noise_fn, noise_state = jme_noise(
        noise_multiplier=0.0,
        aggregate_norm=1.0,
        allocation=JmeAllocation.paper_reference(),
        key=key(0),
    )
    private_moments, noise_state = noise_fn(
        clipped(aggregate, max_norm=0.2),
        noise_state,
    )
    actual_updates, jme_state = optimizer.update(
        private_moments,
        jme_state,
        params=params,
    )

    torch.testing.assert_close(actual_updates, expected_updates)
    torch.testing.assert_close(jme_state[0].mu, raw_state[0].mu)
    torch.testing.assert_close(jme_state[0].nu, raw_state[0].nu)


def test_zero_noise_jme_uses_projected_adam_statistic_above_radius():
    params = torch.tensor([0.2, -0.4])
    aggregate = torch.tensor([3.0, 4.0])
    optimizer = adamw(lr=0.01, weight_decay=0.0)
    raw_state = optimizer.init(params)
    jme_state = optimizer.init(params)
    raw_updates, _ = optimizer.update(aggregate, raw_state, params=params)

    noise_fn, noise_state = jme_noise(
        noise_multiplier=0.0,
        aggregate_norm=1.0,
        allocation=JmeAllocation.paper_reference(),
        key=key(0),
    )
    private_moments, _ = noise_fn(
        clipped(aggregate, max_norm=0.2),
        noise_state,
    )
    projected_updates, _ = optimizer.update(
        private_moments,
        jme_state,
        params=params,
    )

    assert not torch.equal(projected_updates, raw_updates)

