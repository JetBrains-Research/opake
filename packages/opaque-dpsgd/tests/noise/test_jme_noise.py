"""Runtime tests for projected DP-SGD JME."""

from __future__ import annotations

import math

import pytest
import torch

from opaque.api.dpsgd.noise._jme import _resolve_jme_parameters
from opaque.dpsgd.clipping import (
    adaptive_clipped_grad,
    auto_clipped_grad,
    clipped_grad,
)
from opaque.dpsgd.noise import gaussian_noise, jme_noise
from opaque.dpsgd.noise.types import JmeAllocation, JmeNoiseState
from opaque.random import key
from opaque.serialization import from_state_dict, state_dict
from opaque.types import (
    NoisedPytree,
    PerGroup,
    SecondMomentNoiseOutput,
    clipped,
    noised,
)


def _zero_noise(
    *,
    aggregate_norm: float = 2.0,
    allocation: JmeAllocation | None = None,
):
    return jme_noise(
        noise_multiplier=0.0,
        aggregate_norm=aggregate_norm,
        allocation=allocation or JmeAllocation.paper_reference(),
        key=key(7),
    )


def test_square_is_formed_after_aggregate_cancellation():
    g = torch.tensor([0.6, -0.8])
    examples = torch.stack((g, -g))

    def loss_fn(params, example):
        return (params * example).sum()

    grad_fn, clip_state = clipped_grad(
        loss_fn,
        clipping_norm=1.0,
        normalize_by=2.0,
    )
    aggregate, _ = grad_fn(torch.zeros_like(g), examples, state=clip_state)
    noise_fn, noise_state = _zero_noise()
    output, _ = noise_fn(aggregate, noise_state)

    torch.testing.assert_close(output.noisy_grads.pytree, torch.zeros_like(g))
    torch.testing.assert_close(
        output.noisy_squared_grads.pytree,
        torch.zeros_like(g),
    )


def test_zero_noise_projects_then_squares():
    noise_fn, state = _zero_noise(aggregate_norm=2.0)
    output, next_state = noise_fn(
        clipped(torch.tensor([3.0, 4.0]), max_norm=0.25),
        state,
    )
    expected = torch.tensor([1.2, 1.6])
    torch.testing.assert_close(output.noisy_grads.pytree, expected)
    torch.testing.assert_close(output.noisy_squared_grads.pytree, expected.square())
    assert next_state._step_counter == 1


def test_zero_noise_leaves_bounded_aggregate_unchanged():
    aggregate = torch.tensor([0.3, -0.4])
    noise_fn, state = _zero_noise(aggregate_norm=1.0)
    output, _ = noise_fn(clipped(aggregate, max_norm=0.2), state)
    torch.testing.assert_close(output.noisy_grads.pytree, aggregate)
    torch.testing.assert_close(output.noisy_squared_grads.pytree, aggregate.square())


def test_projection_promotes_compute_dtype_for_float64_inputs():
    aggregate = torch.tensor([1.9e19], dtype=torch.float64)
    noise_fn, state = _zero_noise(aggregate_norm=1e30)
    output, _ = noise_fn(clipped(aggregate, max_norm=1e18), state)
    torch.testing.assert_close(output.noisy_grads.pytree, aggregate)
    torch.testing.assert_close(
        output.noisy_squared_grads.pytree,
        aggregate.square(),
    )


def test_extreme_nonzero_distance_ratio_keeps_representable_noise_scales():
    noise_fn, state = jme_noise(
        noise_multiplier=1.0,
        aggregate_norm=1e154,
        allocation=JmeAllocation.paper_reference(),
        key=key(1),
        compute_dtype=torch.float64,
    )
    output, _ = noise_fn(
        clipped(torch.zeros(2, dtype=torch.float64), max_norm=1e-200),
        state,
    )
    assert output.noisy_grads.noise_stddev > 0.0
    assert output.noisy_squared_grads.noise_stddev > 0.0
    assert output.noisy_grads.noise_stddev == pytest.approx(
        math.sqrt(3.0) * 1e-200,
        rel=1e-12,
    )
    assert output.noisy_squared_grads.noise_stddev == pytest.approx(
        math.sqrt(6.0) * 1e-46,
        rel=1e-12,
    )


@pytest.mark.parametrize("mode", ["fixed", "auto", "adaptive"])
def test_all_scalar_clipping_modes_feed_jme(mode: str):
    def loss_fn(params, example):
        return (params * example).sum()

    common = {
        "loss_fn": loss_fn,
        "normalize_by": 2.0,
    }
    if mode == "fixed":
        grad_fn, clip_state = clipped_grad(clipping_norm=1.0, **common)
    elif mode == "auto":
        grad_fn, clip_state = auto_clipped_grad(R=1.0, **common)
    else:
        grad_fn, clip_state = adaptive_clipped_grad(
            initial_clipping_norm=1.0,
            key=key(123),
            **common,
        )
    aggregate, _ = grad_fn(
        torch.zeros(2),
        torch.tensor([[0.4, -0.2], [0.1, 0.3]]),
        state=clip_state,
    )
    noise_fn, noise_state = _zero_noise(aggregate_norm=1.0)
    output, _ = noise_fn(aggregate, noise_state)
    assert isinstance(output, SecondMomentNoiseOutput)
    torch.testing.assert_close(
        output.noisy_squared_grads.pytree,
        output.noisy_grads.pytree.square(),
    )


@pytest.mark.parametrize(
    "allocation",
    [
        JmeAllocation.paper_reference(),
        JmeAllocation.first_variance_cap(1.5),
    ],
)
def test_noise_metadata_matches_calibration(allocation: JmeAllocation):
    noise_multiplier = 1.3
    R = 2.0
    delta = 0.4
    noise_fn, state = jme_noise(
        noise_multiplier=noise_multiplier,
        aggregate_norm=R,
        allocation=allocation,
        key=key(9),
        compute_dtype=torch.float64,
    )
    output, _ = noise_fn(
        clipped(torch.zeros(4, dtype=torch.float64), max_norm=delta),
        state,
    )
    expected = _resolve_jme_parameters(
        noise_multiplier=noise_multiplier,
        delta=delta,
        aggregate_norm=R,
        allocation=allocation,
        dimension=4,
    )
    assert output.noisy_grads.noise_stddev == expected.first_stddev
    assert output.noisy_squared_grads.noise_stddev == expected.second_stddev
    assert output.noisy_grads.max_norm == expected.first_max_norm
    assert output.noisy_squared_grads.max_norm == expected.second_max_norm


def test_output_and_state_types():
    noise_fn, state = _zero_noise()
    output, state = noise_fn(clipped(torch.zeros(2), max_norm=0.1), state)
    assert isinstance(output, SecondMomentNoiseOutput)
    assert isinstance(output.noisy_grads, NoisedPytree)
    assert isinstance(output.noisy_squared_grads, NoisedPytree)
    assert isinstance(state, JmeNoiseState)


def test_same_key_reproduces_both_streams():
    kwargs = {
        "noise_multiplier": 1.0,
        "aggregate_norm": 1.0,
        "allocation": JmeAllocation.paper_reference(),
        "key": key(42),
    }
    fn_a, state_a = jme_noise(**kwargs)
    fn_b, state_b = jme_noise(**kwargs)
    value = clipped(torch.zeros(128), max_norm=0.2)
    out_a, _ = fn_a(value, state_a)
    out_b, _ = fn_b(value, state_b)
    torch.testing.assert_close(out_a.noisy_grads.pytree, out_b.noisy_grads.pytree)
    torch.testing.assert_close(
        out_a.noisy_squared_grads.pytree,
        out_b.noisy_squared_grads.pytree,
    )


def test_first_and_second_rng_domains_are_distinct():
    noise_fn, state = jme_noise(
        noise_multiplier=1.0,
        aggregate_norm=1.0,
        allocation=JmeAllocation.paper_reference(),
        key=key(42),
    )
    output, _ = noise_fn(clipped(torch.zeros(128), max_norm=0.2), state)
    first = output.noisy_grads.pytree / output.noisy_grads.noise_stddev
    second = output.noisy_squared_grads.pytree / output.noisy_squared_grads.noise_stddev
    assert not torch.equal(first, second)


def test_jme_streams_do_not_alias_ordinary_gaussian():
    base_key = key(42)
    jme_fn, jme_state = jme_noise(
        noise_multiplier=1.0,
        aggregate_norm=1.0,
        allocation=JmeAllocation.paper_reference(),
        key=base_key,
    )
    value = clipped(torch.zeros(128), max_norm=0.2)
    jme_output, _ = jme_fn(value, jme_state)

    gaussian_fn, gaussian_state = gaussian_noise(
        noise_multiplier=1.0,
        key=base_key,
    )
    gaussian_output, _ = gaussian_fn(
        clipped(
            torch.zeros(128),
            max_norm=jme_output.noisy_grads.noise_stddev,
        ),
        gaussian_state,
    )
    assert not torch.equal(
        jme_output.noisy_grads.pytree,
        gaussian_output.pytree,
    )


def test_state_advances_once_per_paired_release():
    noise_fn, state = _zero_noise()
    value = clipped(torch.zeros(2), max_norm=0.2)
    for expected_step in range(1, 5):
        _, state = noise_fn(value, state)
        assert state._step_counter == expected_step


def test_state_round_trip_preserves_both_streams():
    kwargs = {
        "noise_multiplier": 1.1,
        "aggregate_norm": 1.0,
        "allocation": JmeAllocation.first_variance_cap(1.4),
        "key": key(99),
    }
    noise_fn, state = jme_noise(**kwargs)
    value = clipped(torch.zeros(16), max_norm=0.2)
    _, state = noise_fn(value, state)
    saved = state_dict(state)

    uninterrupted, uninterrupted_state = noise_fn(value, state)
    restored_fn, fresh_state = jme_noise(**kwargs)
    restored_state = from_state_dict(fresh_state, saved)
    restored, restored_state = restored_fn(value, restored_state)

    torch.testing.assert_close(
        restored.noisy_grads.pytree,
        uninterrupted.noisy_grads.pytree,
    )
    torch.testing.assert_close(
        restored.noisy_squared_grads.pytree,
        uninterrupted.noisy_squared_grads.pytree,
    )
    assert restored_state._step_counter == uninterrupted_state._step_counter
    assert restored_state._aggregate_norm == state._aggregate_norm
    assert restored_state._allocation == state._allocation
    assert restored_state._noise_multiplier == state._noise_multiplier


def test_state_from_another_configuration_is_rejected():
    noise_fn, _ = _zero_noise(aggregate_norm=1.0)
    _, wrong_state = _zero_noise(aggregate_norm=2.0)
    with pytest.raises(ValueError, match="different JME configuration"):
        noise_fn(clipped(torch.zeros(2), max_norm=0.1), wrong_state)


def test_state_from_another_noise_multiplier_is_rejected():
    noise_fn, _ = jme_noise(
        noise_multiplier=1.0,
        aggregate_norm=1.0,
        allocation=JmeAllocation.paper_reference(),
        key=key(0),
    )
    _, wrong_state = jme_noise(
        noise_multiplier=0.5,
        aggregate_norm=1.0,
        allocation=JmeAllocation.paper_reference(),
        key=key(0),
    )
    with pytest.raises(ValueError, match="different JME configuration"):
        noise_fn(clipped(torch.zeros(2), max_norm=0.1), wrong_state)


@pytest.mark.slow
def test_both_streams_have_calibrated_gaussian_statistics():
    noise_fn, state = jme_noise(
        noise_multiplier=1.2,
        aggregate_norm=1.0,
        allocation=JmeAllocation.first_variance_cap(1.5),
        key=key(123),
    )
    output, _ = noise_fn(clipped(torch.zeros(100_000), max_norm=0.1), state)
    first = output.noisy_grads.pytree.double()
    second = output.noisy_squared_grads.pytree.double()
    first_std = float(output.noisy_grads.noise_stddev)
    second_std = float(output.noisy_squared_grads.noise_stddev)
    assert float(first.mean()) == pytest.approx(0.0, abs=0.015 * first_std)
    assert float(second.mean()) == pytest.approx(0.0, abs=0.015 * second_std)
    assert float(first.std()) == pytest.approx(first_std, rel=0.02)
    assert float(second.std()) == pytest.approx(second_std, rel=0.02)
    correlation = float(torch.corrcoef(torch.stack((first, second)))[0, 1])
    assert abs(correlation) < 0.02


def test_nested_pytree_is_projected_globally():
    value = {"a": torch.tensor([3.0]), "nested": (torch.tensor([4.0]),)}
    noise_fn, state = _zero_noise(aggregate_norm=2.0)
    output, _ = noise_fn(clipped(value, max_norm=0.2), state)
    torch.testing.assert_close(output.noisy_grads.pytree["a"], torch.tensor([1.2]))
    torch.testing.assert_close(
        output.noisy_grads.pytree["nested"][0],
        torch.tensor([1.6]),
    )
    torch.testing.assert_close(
        output.noisy_squared_grads.pytree["nested"][0],
        torch.tensor([2.56]),
    )


def test_per_group_input_is_rejected():
    groups = PerGroup(groups={"w": "all"}, values={"all": 1.0})
    noise_fn, state = _zero_noise()
    with pytest.raises(TypeError, match="does not support PerGroup"):
        noise_fn(clipped({"w": torch.zeros(2)}, max_norm=groups), state)


def test_already_noised_input_is_rejected():
    noise_fn, state = _zero_noise()
    with pytest.raises(TypeError, match="has not already been noised"):
        noise_fn(
            noised(torch.zeros(2), max_norm=0.1, noise_stddev=0.2),
            state,
        )


@pytest.mark.parametrize("aggregate_norm", [0.0, -1.0, math.inf, math.nan])
def test_invalid_aggregate_norm_is_rejected(aggregate_norm: float):
    with pytest.raises(ValueError, match="finite and positive"):
        jme_noise(
            noise_multiplier=1.0,
            aggregate_norm=aggregate_norm,
            allocation=JmeAllocation.paper_reference(),
            key=key(0),
        )


def test_allocation_is_required_and_typed():
    with pytest.raises(TypeError, match="allocation must be a JmeAllocation"):
        jme_noise(
            noise_multiplier=1.0,
            aggregate_norm=1.0,
            allocation="paper",  # type: ignore[arg-type]
            key=key(0),
        )


@pytest.mark.parametrize("max_ratio", [1.0, 0.9, math.inf, math.nan])
def test_invalid_first_variance_cap_is_rejected(max_ratio: float):
    with pytest.raises(ValueError, match="max_first_variance_ratio"):
        JmeAllocation.first_variance_cap(max_ratio)
