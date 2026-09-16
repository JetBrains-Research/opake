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


# ---------------------------------------------------------------------------
# adaptive_clipped_grad(second_moment=True) empty-batch parity
# ---------------------------------------------------------------------------


class TestAdaptiveClippedGradEmptyBatchSecondMoment:
    """Empty batches must emit the same paired ``SecondMomentClippingOutput``
    shape that ``clipped_grad`` / ``auto_clipped_grad`` emit on non-empty
    batches, so paired-stream noise + optimizer dispatch stay stable across
    empty and non-empty Poisson steps.
    """

    def test_empty_batch_returns_paired_output(self, params, empty_batch):
        from opaque.types import SecondMomentClippingOutput

        grad_fn, clip_state = adaptive_clipped_grad(
            _simple_loss_fn,
            initial_clipping_norm=1.0,
            key=key(0),
            batch_argnums=(1, 2),
            second_moment=True,
        )
        grads, _ = grad_fn(params, *empty_batch, state=clip_state)
        assert isinstance(grads, SecondMomentClippingOutput)

        first = _unwrap_clipped(grads.grads)
        squared = _unwrap_clipped(grads.squared_grads)
        assert first.shape == params.shape
        assert squared.shape == params.shape
        assert torch.all(first == 0)
        assert torch.all(squared == 0)

    def test_empty_batch_max_norms_match_clipped_grad(self, params, empty_batch):
        """Both streams' max_norm values must equal what ``clipped_grad`` would
        attach: ``C/normalize_by`` for the first stream and ``C²/normalize_by``
        for the squared stream."""
        clip_norm = 0.7
        normalize_by = 4.0

        grad_fn, clip_state = adaptive_clipped_grad(
            _simple_loss_fn,
            initial_clipping_norm=clip_norm,
            key=key(0),
            batch_argnums=(1, 2),
            second_moment=True,
            normalize_by=normalize_by,
        )
        grads, _ = grad_fn(params, *empty_batch, state=clip_state)

        assert grads.grads.max_norm == pytest.approx(clip_norm / normalize_by)
        assert grads.squared_grads.max_norm == pytest.approx(
            (clip_norm * clip_norm) / normalize_by
        )

    def test_empty_batch_uses_next_clipping_norm(
        self, params, empty_batch, normal_batch
    ):
        """After a normal batch updates ``_next_clipping_norm``, the next
        empty batch must report the *updated* threshold (and its square) as
        the streams' bounds."""
        grad_fn, clip_state = adaptive_clipped_grad(
            _simple_loss_fn,
            initial_clipping_norm=1.0,
            key=key(0),
            batch_argnums=(1, 2),
            second_moment=True,
        )
        _, clip_state = grad_fn(params, *normal_batch, state=clip_state)
        next_cn = clip_state._next_clipping_norm
        assert next_cn != 1.0  # adapted

        grads, _ = grad_fn(params, *empty_batch, state=clip_state)
        assert grads.grads.max_norm == pytest.approx(next_cn)
        assert grads.squared_grads.max_norm == pytest.approx(next_cn * next_cn)

    def test_paired_output_consistent_across_empty_and_normal(
        self, params, empty_batch, normal_batch
    ):
        """Type stays ``SecondMomentClippingOutput`` on both empty and normal
        steps so downstream noise mechanism dispatch is stable."""
        from opaque.types import SecondMomentClippingOutput

        grad_fn, clip_state = adaptive_clipped_grad(
            _simple_loss_fn,
            initial_clipping_norm=1.0,
            key=key(0),
            batch_argnums=(1, 2),
            second_moment=True,
        )
        for batch in (empty_batch, normal_batch, empty_batch):
            grads, clip_state = grad_fn(params, *batch, state=clip_state)
            assert isinstance(grads, SecondMomentClippingOutput)

    def test_paired_output_drives_paired_noise_dispatch(
        self, params, empty_batch, normal_batch
    ):
        """End-to-end: gaussian_noise must emit ``SecondMomentNoiseOutput``
        on both empty and non-empty steps when adaptive clipping is in
        ``second_moment=True`` mode."""
        from opaque.dpsgd.noise import gaussian_noise
        from opaque.types import SecondMomentNoiseOutput

        grad_fn, clip_state = adaptive_clipped_grad(
            _simple_loss_fn,
            initial_clipping_norm=1.0,
            key=key(0),
            batch_argnums=(1, 2),
            second_moment=True,
        )
        noise_fn, noise_state = gaussian_noise(noise_multiplier=1.1, key=key(99))

        for batch in (empty_batch, normal_batch, empty_batch):
            grads, clip_state = grad_fn(params, *batch, state=clip_state)
            noisy, noise_state = noise_fn(grads, noise_state)
            assert isinstance(noisy, SecondMomentNoiseOutput)

    def test_per_group_empty_batch_paired(self, empty_batch):
        """PerGroup adaptive + ``second_moment=True`` empty batch produces
        paired output with PerGroup max_norms on both streams."""
        from opaque.types import PerGroup, SecondMomentClippingOutput

        def loss_fn(params, x, y):
            pred = x @ params["w"] + params["b"]
            return ((pred - y) ** 2).mean()

        params = {"w": torch.randn(10), "b": torch.randn(1)}
        groups = {"w": "weights", "b": "biases"}
        init = PerGroup(groups, {"weights": 1.0, "biases": 0.5})

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=init,
            key=key(0),
            batch_argnums=(1, 2),
            second_moment=True,
            normalize_by=2.0,
        )
        grads, _ = grad_fn(params, *empty_batch, state=clip_state)

        assert isinstance(grads, SecondMomentClippingOutput)
        first_mn = grads.grads.max_norm
        squared_mn = grads.squared_grads.max_norm
        assert isinstance(first_mn, PerGroup)
        assert isinstance(squared_mn, PerGroup)
        assert first_mn.values["weights"] == pytest.approx(1.0 / 2.0)
        assert first_mn.values["biases"] == pytest.approx(0.5 / 2.0)
        assert squared_mn.values["weights"] == pytest.approx(1.0 * 1.0 / 2.0)
        assert squared_mn.values["biases"] == pytest.approx(0.5 * 0.5 / 2.0)

    def test_invalid_normalize_by_fails_at_construction(self):
        """``normalize_by <= 0`` must fail at ``adaptive_clipped_grad`` factory
        time so empty-batch and non-empty-batch failure modes stay consistent
        (the empty-batch short-circuit uses ``normalize_by`` to compute the
        squared-stream ``max_norm`` without going through ``clipped_grad``,
        which is where the inner validation otherwise fires)."""
        with pytest.raises(ValueError, match="normalize_by must be > 0"):
            adaptive_clipped_grad(
                _simple_loss_fn,
                initial_clipping_norm=1.0,
                key=key(0),
                batch_argnums=(1, 2),
                second_moment=True,
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

    def test_return_aux_with_second_moment_empty_batch(self, params, empty_batch):
        """Combining ``return_aux=True`` and ``second_moment=True`` on an
        empty batch must still return the paired stream alongside the aux."""
        from opaque.types import SecondMomentClippingOutput

        grad_fn, clip_state = adaptive_clipped_grad(
            _simple_loss_fn,
            initial_clipping_norm=1.0,
            key=key(0),
            batch_argnums=(1, 2),
            return_aux=True,
            second_moment=True,
        )
        (grads, aux), _ = grad_fn(params, *empty_batch, state=clip_state)
        assert isinstance(grads, SecondMomentClippingOutput)
        assert isinstance(aux, AdaptiveClippedGradAux)
        assert aux.clipping_rate == 0.0
        assert aux.grad_norms.shape == (0,)


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
            return {"q": g["w"] + 7.0}

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

    def test_second_moment_streams_parity(self):
        from opaque.types import SecondMomentClippingOutput

        params, (x, y), (empty_x, empty_y) = self._params_and_batches()

        def transform(g):
            return {"q": g["w"] + 7.0}

        grad_fn, clip_state = adaptive_clipped_grad(
            _dict_linear_loss,
            batch_argnums=(1, 2),
            key=key(0),
            pre_clipping_transform=transform,
            dtype=torch.float32,
            second_moment=True,
        )
        full, _ = grad_fn(params, x, y, state=clip_state)
        empty, _ = grad_fn(params, empty_x, empty_y, state=clip_state)
        assert isinstance(full, SecondMomentClippingOutput)
        assert isinstance(empty, SecondMomentClippingOutput)
        for stream in (full, empty):
            assert sorted(stream.grads.pytree) == ["q"]
            assert sorted(stream.squared_grads.pytree) == ["q"]
        assert empty.grads.pytree["q"].dtype == torch.float32
        assert empty.squared_grads.pytree["q"].dtype == torch.float32
        assert torch.all(empty.grads.pytree["q"] == 0)
        assert torch.all(empty.squared_grads.pytree["q"] == 0)
        assert full.grads.max_norm == empty.grads.max_norm
        assert full.squared_grads.max_norm == empty.squared_grads.max_norm
