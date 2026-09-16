"""Tests for empty batch handling in clipping primitives.

Verifies that clipped_grad, adaptive_clipped_grad, and their distributed sync
functions handle batch_size=0 correctly — producing zero grads, empty aux
tensors, preserving adaptive clipping_norm, and avoiding DDP deadlocks.
"""

import pytest
import torch

from opaque.api.dpsgd.clipping._adaptive import (
    AdaptiveClippedGradAux,
    AdaptiveClipState,
    _compute_clipping_stats,
)
from opaque.api.engine.clipping import clipped_grad
from opaque.api.engine.clipping._clipped_grad import ClippedGradAux
from opaque.dpsgd.clipping import adaptive_clipped_grad
from opaque.random import key
from opaque.types import ClippedPytree


def _unwrap_clipped(value):
    assert isinstance(value, ClippedPytree)
    return value.pytree


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _simple_loss_fn(params, x, y):
    pred = x @ params
    return ((pred - y) ** 2).mean()


@pytest.fixture
def params():
    return torch.randn(10, requires_grad=False)


@pytest.fixture
def empty_batch():
    return torch.randn(0, 10), torch.randn(0)


@pytest.fixture
def normal_batch():
    return torch.randn(8, 10), torch.randn(8)


# ---------------------------------------------------------------------------
# _compute_clipping_stats
# ---------------------------------------------------------------------------


class TestComputeClippingStats:
    def test_empty_tensor_returns_zeros(self):
        num_clipped, total, clipping_rate = _compute_clipping_stats(
            torch.empty(0), clipping_norm=1.0
        )
        assert num_clipped == 0.0
        assert total == 0.0
        assert clipping_rate == 0.0

    def test_nonempty_tensor_reports_honest_total(self):
        norms = torch.tensor([0.5, 1.5, 2.5])
        num_clipped, total, rate = _compute_clipping_stats(norms, clipping_norm=1.0)
        assert total == 3.0
        assert num_clipped == 2.0
        assert abs(rate - 2.0 / 3.0) < 1e-6

    def test_all_below_clip_norm(self):
        norms = torch.tensor([0.1, 0.2, 0.3])
        num_clipped, _total, rate = _compute_clipping_stats(norms, clipping_norm=1.0)
        assert num_clipped == 0.0
        assert rate == 0.0


# ---------------------------------------------------------------------------
# clipped_grad with empty batch
# ---------------------------------------------------------------------------


class TestClippedGradEmptyBatch:
    def test_returns_zero_grads(self, params, empty_batch):
        grad_fn, clip_state = clipped_grad(
            _simple_loss_fn,
            argnums=0,
            batch_argnums=(1, 2),
            clipping_norm=1.0,
        )
        grads, _new_state = grad_fn(params, *empty_batch, state=clip_state)
        grads = _unwrap_clipped(grads)
        assert grads.shape == params.shape
        assert torch.all(grads == 0)

    def test_returns_empty_aux_tensors(self, params, empty_batch):
        grad_fn, clip_state = clipped_grad(
            _simple_loss_fn,
            argnums=0,
            batch_argnums=(1, 2),
            clipping_norm=1.0,
            return_aux=True,
        )
        (grads, aux), _ = grad_fn(params, *empty_batch, state=clip_state)
        grads = _unwrap_clipped(grads)
        assert grads.shape == params.shape
        assert torch.all(grads == 0)
        assert isinstance(aux, ClippedGradAux)
        assert aux.grad_norms.shape == (0,)
        assert aux.loss_values.shape == (0,)
        assert aux.clipped_grad_norms.shape == (0,)
        assert aux.clipping_rate == 0.0
        assert aux.batch_size == 0

    def test_state_unchanged(self, params, empty_batch):
        grad_fn, clip_state = clipped_grad(
            _simple_loss_fn,
            argnums=0,
            batch_argnums=(1, 2),
            clipping_norm=1.0,
        )
        _, new_state = grad_fn(params, *empty_batch, state=clip_state)
        assert new_state is clip_state

    def test_pytree_params_zero_grads(self, empty_batch):
        def loss_fn(params, x, y):
            pred = x @ params["w"] + params["b"]
            return ((pred - y) ** 2).mean()

        params = {"w": torch.randn(10), "b": torch.randn(1)}
        grad_fn, clip_state = clipped_grad(
            loss_fn,
            argnums=0,
            batch_argnums=(1, 2),
            clipping_norm=1.0,
        )
        grads, _ = grad_fn(params, *empty_batch, state=clip_state)
        grads = _unwrap_clipped(grads)
        assert isinstance(grads, dict)
        assert torch.all(grads["w"] == 0)
        assert torch.all(grads["b"] == 0)

    def test_microbatch_empty_batch(self, params, empty_batch):
        """Microbatch path also handles empty batches."""
        grad_fn, clip_state = clipped_grad(
            _simple_loss_fn,
            argnums=0,
            batch_argnums=(1, 2),
            clipping_norm=1.0,
            microbatch_size=4,
            return_aux=True,
        )
        (grads, aux), _ = grad_fn(params, *empty_batch, state=clip_state)
        grads = _unwrap_clipped(grads)
        assert grads.shape == params.shape
        assert torch.all(grads == 0)
        assert aux.grad_norms.shape == (0,)


# ---------------------------------------------------------------------------
# adaptive_clipped_grad with empty batch
# ---------------------------------------------------------------------------


class TestAdaptiveClippedGradEmptyBatch:
    def test_zero_grads_and_preserved_clipping_norm(self, params, empty_batch):
        grad_fn, clip_state = adaptive_clipped_grad(
            _simple_loss_fn,
            initial_clipping_norm=1.0,
            key=key(0),
            batch_argnums=(1, 2),
        )
        initial_cn = clip_state._current_clipping_norm
        grads, new_state = grad_fn(params, *empty_batch, state=clip_state)
        grads = _unwrap_clipped(grads)

        assert grads.shape == params.shape
        assert torch.all(grads == 0)
        assert new_state._current_clipping_norm == initial_cn
        assert new_state._batch_size == 0.0
        assert new_state._num_clipped == 0.0

    def test_step_still_increments(self, params, empty_batch):
        grad_fn, clip_state = adaptive_clipped_grad(
            _simple_loss_fn,
            initial_clipping_norm=1.0,
            key=key(0),
            batch_argnums=(1, 2),
        )
        assert clip_state._step == 0
        _, new_state = grad_fn(params, *empty_batch, state=clip_state)
        assert new_state._step == 1

    def test_empty_then_normal_batch(self, params, empty_batch, normal_batch):
        """After an empty batch, a normal batch still adapts correctly."""
        grad_fn, clip_state = adaptive_clipped_grad(
            _simple_loss_fn,
            initial_clipping_norm=1.0,
            key=key(0),
            batch_argnums=(1, 2),
        )
        initial_cn = clip_state._current_clipping_norm

        # Empty batch: no adaptation
        _, clip_state = grad_fn(params, *empty_batch, state=clip_state)
        assert clip_state._current_clipping_norm == initial_cn

        # Normal batch: adaptation happens
        _, clip_state = grad_fn(params, *normal_batch, state=clip_state)
        assert clip_state._next_clipping_norm != initial_cn
        assert clip_state._batch_size > 0

    def test_return_aux_empty_batch(self, params, empty_batch):
        grad_fn, clip_state = adaptive_clipped_grad(
            _simple_loss_fn,
            initial_clipping_norm=1.0,
            key=key(0),
            batch_argnums=(1, 2),
            return_aux=True,
        )
        (_grads, aux), _new_state = grad_fn(params, *empty_batch, state=clip_state)

        assert isinstance(aux, AdaptiveClippedGradAux)
        assert aux.grad_norms.shape == (0,)
        assert aux.loss_values.shape == (0,)
        assert aux.clipped_grad_norms.shape == (0,)
        assert aux.clipping_rate == 0.0
        assert aux.loss_aux is None

    def test_consecutive_empty_batches(self, params, empty_batch):
        """Multiple empty batches don't drift clipping_norm."""
        grad_fn, clip_state = adaptive_clipped_grad(
            _simple_loss_fn,
            initial_clipping_norm=1.0,
            key=key(0),
            batch_argnums=(1, 2),
        )
        initial_cn = clip_state._current_clipping_norm

        for _ in range(10):
            _, clip_state = grad_fn(params, *empty_batch, state=clip_state)

        assert clip_state._current_clipping_norm == initial_cn
        assert clip_state._next_clipping_norm == initial_cn
        assert clip_state._step == 10


def test_adaptive_invalid_normalize_by_fails_at_construction():
    with pytest.raises(ValueError, match="normalize_by must be > 0"):
        adaptive_clipped_grad(
            _simple_loss_fn,
            initial_clipping_norm=1.0,
            key=key(0),
            batch_argnums=(1, 2),
            normalize_by=0.0,
        )
    with pytest.raises(ValueError, match="normalize_by must be > 0"):
        adaptive_clipped_grad(
            _simple_loss_fn,
            initial_clipping_norm=1.0,
            key=key(0),
            batch_argnums=(1, 2),
            normalize_by=-1.0,
        )


def test_adaptive_legacy_second_moment_argument_is_rejected():
    with pytest.raises(TypeError, match="second_moment"):
        adaptive_clipped_grad(
            _simple_loss_fn,
            initial_clipping_norm=1.0,
            key=key(0),
            second_moment=True,
        )


# ---------------------------------------------------------------------------
# Per-group aux: clipping_rate presence tracks the config, not the batch
# ---------------------------------------------------------------------------


def _pg_loss_fn(params, x, y):
    pred = x @ params["w"] + params["b"]
    return ((pred - y) ** 2).mean()


class TestPerGroupEmptyBatchAuxSchema:
    def _per_group(self):
        from opaque.dpsgd.clipping import per_group

        params = {"w": torch.randn(10), "b": torch.randn(1)}
        return params, per_group(params, w=1.0, b=0.5)

    def test_fixed_per_group_empty_aux_rate_is_none(self, empty_batch):
        params, pg = self._per_group()
        grad_fn, clip_state = clipped_grad(
            _pg_loss_fn,
            batch_argnums=(1, 2),
            clipping_norm=pg,
            return_aux=True,
        )
        (_, aux), _ = grad_fn(params, *empty_batch, state=clip_state)
        assert aux.clipping_rate is None

    def test_fixed_scalar_empty_aux_rate_stays_zero(self, params, empty_batch):
        grad_fn, clip_state = clipped_grad(
            _simple_loss_fn,
            batch_argnums=(1, 2),
            clipping_norm=1.0,
            return_aux=True,
        )
        (_, aux), _ = grad_fn(params, *empty_batch, state=clip_state)
        assert aux.clipping_rate == 0.0

    def test_adaptive_per_group_empty_aux_rate_is_none(self, empty_batch):
        params, pg = self._per_group()
        grad_fn, clip_state = adaptive_clipped_grad(
            _pg_loss_fn,
            initial_clipping_norm=pg,
            key=key(0),
            batch_argnums=(1, 2),
            return_aux=True,
        )
        (_, aux), _ = grad_fn(params, *empty_batch, state=clip_state)
        assert aux.clipping_rate is None

    def test_adaptive_per_group_empty_matches_nonempty_presence(
        self, empty_batch, normal_batch
    ):
        params, pg = self._per_group()
        grad_fn, clip_state = adaptive_clipped_grad(
            _pg_loss_fn,
            initial_clipping_norm=pg,
            key=key(0),
            batch_argnums=(1, 2),
            return_aux=True,
        )
        (_, empty_aux), state = grad_fn(params, *empty_batch, state=clip_state)
        (_, full_aux), _ = grad_fn(params, *normal_batch, state=state)
        assert empty_aux.clipping_rate is None
        assert full_aux.clipping_rate is None

    def test_adaptive_scalar_empty_matches_nonempty_presence(
        self, params, empty_batch, normal_batch
    ):
        grad_fn, clip_state = adaptive_clipped_grad(
            _simple_loss_fn,
            initial_clipping_norm=1.0,
            key=key(0),
            batch_argnums=(1, 2),
            return_aux=True,
        )
        (_, empty_aux), state = grad_fn(params, *empty_batch, state=clip_state)
        (_, full_aux), _ = grad_fn(params, *normal_batch, state=state)
        assert isinstance(empty_aux.clipping_rate, float)
        assert isinstance(full_aux.clipping_rate, float)


# ---------------------------------------------------------------------------
# sync_adaptive_clip_state with all-empty ranks (unit test, no real DDP)
# ---------------------------------------------------------------------------


class TestSyncAdaptiveClipStateAllEmpty:
    """Unit-test the all-empty-ranks guard without spawning processes."""

    def test_total_zero_preserves_clipping_norm(self):
        from opaque.api.dpsgd.clipping._distributed import sync_adaptive_clip_state

        state = AdaptiveClipState(
            _current_clipping_norm=1.5,
            _next_clipping_norm=1.5,
            _step=5,
            _rng_key=key(0),
            _fraction_noise_std=0.05,
            _learning_rate=0.2,
            _target_quantile=0.5,
            _clipping_norm_min=0.01,
            _clipping_norm_max=100.0,
            _num_clipped=0.0,
            _batch_size=0.0,
        )
        # In non-distributed mode sync_adaptive_clip_state returns state as-is
        result = sync_adaptive_clip_state(state)
        assert result._current_clipping_norm == 1.5
        assert result._next_clipping_norm == 1.5


# ---------------------------------------------------------------------------
# sync_adaptive_clipped_grad_aux (non-distributed passthrough)
# ---------------------------------------------------------------------------


class TestSyncAdaptiveClippedGradAux:
    def test_empty_grad_norms_passthrough(self):
        from opaque.api.dpsgd.clipping._distributed import (
            sync_adaptive_clipped_grad_aux,
        )

        aux = AdaptiveClippedGradAux(
            loss_values=torch.empty(0),
            grad_norms=torch.empty(0),
            clipped_grad_norms=torch.empty(0),
            loss_aux=None,
            clipping_rate=0.0,
            batch_size=0,
        )
        result = sync_adaptive_clipped_grad_aux(aux)
        assert result.clipping_rate == 0.0
        assert result.grad_norms.shape == (0,)


# ---------------------------------------------------------------------------
# Empty-batch vs non-empty parity: dtype + pre_clipping_transform structure
# ---------------------------------------------------------------------------


def _dict_linear_loss(params, x, y):
    return ((x @ params["w"] - y) ** 2).mean()


class TestAdaptiveEmptyBatchStructureAndDtypeParity:
    """Empty adaptive steps must mirror non-empty structure and dtype."""

    @staticmethod
    def _params_and_batches():
        params = {"w": torch.randn(10, dtype=torch.bfloat16)}
        x = torch.randn(8, 10, dtype=torch.bfloat16)
        y = torch.randn(8, dtype=torch.bfloat16)
        return params, (x, y), (x[:0], y[:0])

    def test_dtype_override_applies_to_empty_batch(self):
        params, (x, y), (empty_x, empty_y) = self._params_and_batches()
        grad_fn, clip_state = adaptive_clipped_grad(
            _dict_linear_loss,
            batch_argnums=(1, 2),
            key=key(0),
            dtype=torch.float32,
        )
        full, _ = grad_fn(params, x, y, state=clip_state)
        empty, _ = grad_fn(params, empty_x, empty_y, state=clip_state)
        assert full.pytree["w"].dtype == torch.float32
        assert empty.pytree["w"].dtype == torch.float32
        assert torch.all(empty.pytree["w"] == 0)

    def test_transform_structure_applies_to_empty_batch(self):
        params, (x, y), (empty_x, empty_y) = self._params_and_batches()

        def transform(g):
            return {"q": g["w"]}

        grad_fn, clip_state = adaptive_clipped_grad(
            _dict_linear_loss,
            batch_argnums=(1, 2),
            key=key(0),
            pre_clipping_transform=transform,
            dtype=torch.float32,
        )
        full, _ = grad_fn(params, x, y, state=clip_state)
        empty, _ = grad_fn(params, empty_x, empty_y, state=clip_state)
        assert sorted(full.pytree) == sorted(empty.pytree) == ["q"]
        assert torch.all(empty.pytree["q"] == 0)
