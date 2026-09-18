"""clipped_grad: the round contracts it enforces, and one real federated round."""

import ifed
import pytest
import torch

from opaque.api.engine.clipping.types import FixedClipState
from opaque.api.engine.types import ClippedPytree
from opaque.federated import (
    Cohort,
    DataLoader,
    MinSepSampler,
    clipped_grad,
    clipped_sum,
)
from opaque.federated import (
    population as make_population,
)


class Points(ifed.Dataset):
    x = ifed.Float()
    y = ifed.Float()


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(1, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features).squeeze(-1)


ROWS = [{"x": 1.0, "y": 5.0}, {"x": 2.0, "y": 9.0}, {"x": 3.0, "y": 13.0}]


class Unusable:
    """A session that fails loudly: these rounds must be rejected before stepping."""

    def step(self, *args, **kwargs):
        raise AssertionError("grad_fn stepped a round it should have rejected")


def _loader(population, rounds=4, batch_size=2, bands=2):
    sampler = MinSepSampler(population, batch_size=batch_size, bands=bands)
    return DataLoader(population, batch_sampler=sampler, rounds=rounds)


@pytest.fixture
def population():
    return make_population("/hive")


@pytest.fixture
def grad_fn():
    fn, _ = clipped_grad(Unusable(), clipped_sum(clipping_norm=1.0))
    return fn


def test_returns_the_central_fixed_clip_state():
    _, clip_state = clipped_grad(Unusable(), clipped_sum(clipping_norm=1.0))
    assert isinstance(clip_state, FixedClipState)


def test_rejects_a_strategy_without_a_threshold():
    with pytest.raises(TypeError, match="clipped_sum"):
        clipped_grad(Unusable(), ifed.FedAvg())


def test_rejects_a_raw_cohort(grad_fn):
    with pytest.raises(ValueError, match="DataLoader"):
        grad_fn({}, Cohort(round=0, size=2), state=FixedClipState())


def test_rejects_a_cohort_from_another_loader(population, grad_fn):
    grad_fn_other, _ = clipped_grad(Unusable(), clipped_sum(clipping_norm=1.0))
    first = next(iter(_loader(population)))
    with pytest.raises(AssertionError):
        grad_fn_other({}, first, state=FixedClipState())  # binds, then steps
    with pytest.raises(ValueError, match="different DataLoader"):
        grad_fn_other({}, next(iter(_loader(population))), state=FixedClipState())


def test_rejects_a_changed_cohort_size(population, grad_fn):
    loader = _loader(population, batch_size=2)
    cohorts = iter(loader)
    with pytest.raises(AssertionError):
        grad_fn({}, next(cohorts), state=FixedClipState())
    loader.batch_sampler.batch_size = 4  # a mid-run edit the divisor would follow
    with pytest.raises(ValueError, match="size/separation changed"):
        grad_fn({}, next(cohorts), state=FixedClipState())


def test_rejects_an_out_of_order_round(population, grad_fn):
    cohorts = iter(_loader(population, rounds=4))
    with pytest.raises(AssertionError):
        grad_fn({}, next(cohorts), state=FixedClipState())
    next(cohorts)  # skip round 1
    with pytest.raises(ValueError, match="out-of-order"):
        grad_fn({}, next(cohorts), state=FixedClipState())


def _bridge_reason():
    try:
        import ifed_agent_abi

        ifed_agent_abi.get_library()
    except Exception as error:  # any failure means no local agent runtime
        return f"ifed agent bridge unavailable: {error}"
    return None


@pytest.mark.skipif(_bridge_reason() is not None, reason="no ifed agent bridge")
def test_one_real_federated_round(population):
    """Two local agents holding the same rows, so the clipped sum is exact."""
    clipping_norm = 0.01  # far below the raw gradient norm, so every client clips
    strategy = clipped_sum(clipping_norm=clipping_norm)
    plan = ifed.build_train(
        net=Tiny(),
        source=Points,
        target="y",
        features=["x"],
        loss=ifed.Loss.mse,
        batch_size=None,  # one client contribution = one full-batch gradient
        shuffle=False,
        strategy=strategy,
    )
    points = Points.from_rows(ROWS)
    store = ifed.LocalDatastore(datasets=[points, points])

    with ifed.session(plan, store) as run:
        params = plan.init_state.params
        grad_fn, clip_state = clipped_grad(run, strategy)
        cohorts = iter(_loader(population, rounds=2, batch_size=2, bands=2))
        grads, threaded = grad_fn(params, next(cohorts), state=clip_state)

    assert isinstance(grads, ClippedPytree)
    assert threaded is clip_state
    assert set(grads.pytree) == set(params)
    assert grads.max_norm == pytest.approx(clipping_norm / 2)
    # both clients clip to exactly `clipping_norm`, so the mean of the two does too
    norm = torch.sqrt(sum(value.pow(2).sum() for value in grads.pytree.values()))
    assert float(norm) == pytest.approx(clipping_norm, rel=1e-4)
