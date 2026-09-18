"""clipped_sum: what one round releases, and when it refuses to release anything."""

import ifed
import pytest
import torch

from opaque.federated import clipped_sum


def _agent(values, rows=4.0):
    return ifed.AgentState(
        params={name: torch.tensor(value) for name, value in values.items()},
        metrics=ifed.MetricsBundle(scalars={"n": rows}, histograms={}),
    )


def _norm(params):
    return float(torch.sqrt(sum(value.pow(2).sum() for value in params.values())))


def test_clips_each_client_then_sums():
    strategy = clipped_sum(clipping_norm=2.0)
    out = strategy.aggregate(
        [
            _agent({"w": [10.0, 0.0]}),  # norm 10 -> [2, 0]
            _agent({"w": [0.0, 5.0]}),  # norm 5  -> [0, 2]
            _agent({"w": [0.3, 0.4]}),  # norm 0.5, under the threshold
        ]
    )
    assert out.params["w"] == pytest.approx([2.3, 2.4], abs=1e-5)


def test_clips_the_whole_pytree_jointly():
    """The threshold is on one client's whole update, not per tensor."""
    strategy = clipped_sum(clipping_norm=1.0)
    out = strategy.aggregate([_agent({"w": [3.0], "b": [4.0]})])
    assert _norm(out.params) == pytest.approx(1.0, abs=1e-5)
    assert out.params["w"] == pytest.approx([0.6], abs=1e-5)
    assert out.params["b"] == pytest.approx([0.8], abs=1e-5)


@pytest.mark.parametrize("clipping_norm", [0.5, 2.0, 7.0])
def test_one_client_moves_the_sum_by_at_most_the_threshold(clipping_norm):
    """Add-or-remove adjacency on clients: the released sum has sensitivity C."""
    generator = torch.Generator().manual_seed(7)
    cohort = [
        _agent({"w": (torch.randn(6, generator=generator) * 4).tolist()})
        for _ in range(5)
    ]
    strategy = clipped_sum(clipping_norm=clipping_norm)
    with_all = strategy.aggregate(cohort)
    without_last = strategy.aggregate(cohort[:-1])
    delta = {
        name: value - without_last.params[name]
        for name, value in with_all.params.items()
    }
    assert _norm(delta) <= clipping_norm + 1e-5


def test_releases_no_metrics():
    strategy = clipped_sum(clipping_norm=1.0)
    out = strategy.aggregate([_agent({"w": [1.0]}), _agent({"w": [2.0]})])
    assert out.metrics.scalars == {}
    assert out.metrics.histograms == {}


def test_finalize_carries_the_sum_out_untouched():
    strategy = clipped_sum(clipping_norm=4.0)
    aggregated = strategy.aggregate([_agent({"w": [1.0]}), _agent({"w": [2.0]})])
    state = strategy.finalize(aggregated, ifed.InterState(round=6))
    assert state.round == 7
    assert state.params["w"] == pytest.approx([3.0], abs=1e-5)


def test_a_client_reporting_no_rows_fails_the_round():
    strategy = clipped_sum(clipping_norm=1.0)
    with pytest.raises(ValueError, match="max_skipped"):
        strategy.aggregate([_agent({"w": [1.0]}), _agent({"w": [2.0]}, rows=0.0)])


def test_a_non_finite_gradient_fails_the_round():
    strategy = clipped_sum(clipping_norm=1.0)
    with pytest.raises(ValueError, match="max_skipped"):
        strategy.aggregate([_agent({"w": [1.0]}), _agent({"w": [float("nan")]})])


def test_equal_weights_and_no_skipping_are_not_negotiable():
    strategy = clipped_sum(clipping_norm=1.0)
    assert strategy.weighted is False
    assert strategy.max_skipped == 0.0
    assert strategy.clipping_norm == 1.0
    assert strategy.optimizer is None  # FedSGD: agents send raw gradients


@pytest.mark.parametrize("clipping_norm", [0.0, -1.0])
def test_rejects_a_non_positive_threshold(clipping_norm):
    with pytest.raises(ValueError, match="clipping_norm"):
        clipped_sum(clipping_norm=clipping_norm)
