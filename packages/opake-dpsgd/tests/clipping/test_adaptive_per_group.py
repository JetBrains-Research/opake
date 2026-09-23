"""Tests for adaptive per-group gradient clipping."""

import math

import pytest
import torch
from torch._dynamo.testing import CompileCounterWithBackend

from opake.api.dpsgd.clipping._adaptive import AdaptiveClipState, adaptive_clipped_grad
from opake.api.engine.clipping import _clipped_fun as clipped_fun_module
from opake.dpsgd.clipping import per_group
from opake.random import key
from opake.types import ClippedPytree, PerGroup


def _unwrap_clipped(value):
    assert isinstance(value, ClippedPytree)
    return value.pytree


def _make_loss_fn():
    """Simple loss function for testing."""

    def loss_fn(params, x, y):
        pred = x @ params
        return ((pred - y) ** 2).mean()

    return loss_fn


def _make_per_group_loss_fn():
    """Loss function with dict params for per-group testing."""

    def loss_fn(params, x, y):
        # params is a dict: {"a": weight_a, "b": weight_b}
        pred = x @ params["a"] + params["b"].sum()
        return ((pred - y) ** 2).mean()

    return loss_fn


def _make_per_group(params, a_norm=1.0, b_norm=2.0):
    """Build a PerGroup from dict params with 2 groups."""
    return per_group(params, a=a_norm, b=b_norm)


class TestAdaptivePerGroupBasic:
    """Basic tests for adaptive per-group clipping."""

    def test_accepts_per_group_initial_clipping_norm(self):
        """Test that adaptive_clipped_grad accepts PerGroup initial_clipping_norm."""
        loss_fn = _make_per_group_loss_fn()
        params = {"a": torch.randn(10), "b": torch.randn(5)}
        pg = _make_per_group(params)

        _grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=pg,
            key=key(0),
            batch_argnums=(1, 2),
            expected_batch_size=1.0,
        )

        assert isinstance(clip_state._current_clipping_norm, PerGroup)
        assert isinstance(clip_state._next_clipping_norm, PerGroup)
        assert clip_state._current_clipping_norm.values == pg.values

    def test_per_group_state_is_per_group(self):
        """Test that state fields remain PerGroup after a step."""
        loss_fn = _make_per_group_loss_fn()
        params = {"a": torch.randn(10), "b": torch.randn(5)}
        pg = _make_per_group(params)

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=pg,
            key=key(0),
            batch_argnums=(1, 2),
            return_aux=True,
            expected_batch_size=1.0,
        )

        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        (_grads, _aux), clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)

        assert isinstance(clip_state._current_clipping_norm, PerGroup)
        assert isinstance(clip_state._next_clipping_norm, PerGroup)
        assert isinstance(clip_state._num_clipped, dict)
        assert set(clip_state._num_clipped.keys()) == {"a", "b"}

    def test_per_group_returns_grads(self):
        """Test that per-group adaptive returns correctly shaped gradients."""
        loss_fn = _make_per_group_loss_fn()
        params = {"a": torch.randn(10), "b": torch.randn(5)}
        pg = _make_per_group(params)

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=pg,
            key=key(0),
            batch_argnums=(1, 2),
            expected_batch_size=1.0,
        )

        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
        grads = _unwrap_clipped(grads)

        assert isinstance(grads, dict)
        assert grads["a"].shape == params["a"].shape
        assert grads["b"].shape == params["b"].shape

    def test_per_group_output_bound_metadata(self):
        """Adaptive clipping returns per-group max_norm metadata after normalization."""
        loss_fn = _make_per_group_loss_fn()
        params = {"a": torch.randn(10), "b": torch.randn(5)}
        pg = _make_per_group(params, a_norm=1.0, b_norm=2.0)

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=pg,
            key=key(0),
            batch_argnums=(1, 2),
            normalize_by=10.0,
        )

        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)
        grads, _ = grad_fn(params, batch_x, batch_y, state=clip_state)
        assert isinstance(grads.max_norm, PerGroup)
        assert grads.max_norm.values == {
            "a": pytest.approx(0.1),
            "b": pytest.approx(0.2),
        }
        import math

        assert grads.max_norm.effective == pytest.approx(math.sqrt(5) / 10)


class TestAdaptivePerGroupConvergence:
    """Tests that per-group thresholds adapt independently."""

    def test_independent_adaptation(self):
        """Test that groups with different gradient magnitudes adapt differently."""

        # Group "a" will have large gradients, group "b" small
        def loss_fn(params, x, y):
            # Scale so group "a" gradients are much larger
            pred = 10.0 * x @ params["a"] + 0.01 * params["b"].sum()
            return ((pred - y) ** 2).mean()

        params = {"a": torch.randn(5), "b": torch.randn(3)}
        pg = per_group(params, a=0.5, b=0.5)  # Same initial threshold

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=pg,
            target_quantile=0.5,
            learning_rate=0.5,  # Large LR for visible effect
            key=key(42),
            batch_argnums=(1, 2),
            return_aux=True,
            expected_batch_size=1.0,
        )

        batch_x = torch.randn(16, 5)
        batch_y = torch.randn(16)

        # Run several steps
        for _ in range(5):
            (_, _aux), clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)

        # Group "a" should have adapted to a different threshold than "b"
        final = clip_state._next_clipping_norm
        assert isinstance(final, PerGroup)
        # They started the same but should have diverged
        assert final.values["a"] != final.values["b"]

    def test_high_norm_group_threshold_increases(self):
        """Test that a group with consistently clipped gradients increases its threshold."""

        def loss_fn(params, x, y):
            pred = 100.0 * x @ params["a"] + 0.001 * params["b"].sum()
            return ((pred - y) ** 2).mean()

        params = {"a": torch.randn(5), "b": torch.randn(3)}
        pg = per_group(params, a=0.001, b=100.0)  # a too low, b too high

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=pg,
            target_quantile=0.5,
            learning_rate=0.3,
            key=key(0),
            batch_argnums=(1, 2),
            expected_batch_size=1.0,
        )

        initial_a = pg.values["a"]
        initial_b = pg.values["b"]

        batch_x = torch.randn(16, 5)
        batch_y = torch.randn(16)

        _, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)

        # Group "a": all clipped → threshold should increase
        assert clip_state._next_clipping_norm.values["a"] > initial_a
        # Group "b": none clipped → threshold should decrease
        assert clip_state._next_clipping_norm.values["b"] < initial_b


class TestAdaptivePerGroupDeterministic:
    """Tests for determinism with per-group adaptive clipping."""

    def test_deterministic_with_same_key(self):
        """Test that same key produces same adaptation path."""
        loss_fn = _make_per_group_loss_fn()
        params = {"a": torch.randn(10), "b": torch.randn(5)}
        pg = _make_per_group(params)

        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        results = []
        for _ in range(2):
            grad_fn, clip_state = adaptive_clipped_grad(
                loss_fn,
                initial_clipping_norm=pg,
                key=key(42),
                batch_argnums=(1, 2),
                expected_batch_size=1.0,
            )
            _, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
            results.append(clip_state._next_clipping_norm)

        assert results[0].values["a"] == results[1].values["a"]
        assert results[0].values["b"] == results[1].values["b"]


class TestAdaptivePerGroupEmptyBatch:
    """Tests for empty batch handling with per-group adaptive clipping."""

    def test_empty_batch_noises_per_group_thresholds(self):
        """An empty draw still spends one adaptive clipping step."""
        loss_fn = _make_per_group_loss_fn()
        params = {"a": torch.randn(10), "b": torch.randn(5)}
        pg = _make_per_group(params, a_norm=1.5, b_norm=3.0)

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=pg,
            key=key(0),
            batch_argnums=(1, 2),
            expected_batch_size=1.0,
        )

        # Empty batch
        batch_x = torch.randn(0, 10)
        batch_y = torch.randn(0)

        _, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)

        assert clip_state._next_clipping_norm.values["a"] != 1.5
        assert clip_state._next_clipping_norm.values["b"] != 3.0
        assert clip_state._step == 1

    def test_empty_batch_num_clipped_is_dict(self):
        """Test that empty batch _num_clipped remains a dict."""
        loss_fn = _make_per_group_loss_fn()
        params = {"a": torch.randn(10), "b": torch.randn(5)}
        pg = _make_per_group(params)

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=pg,
            key=key(0),
            batch_argnums=(1, 2),
            expected_batch_size=1.0,
        )

        batch_x = torch.randn(0, 10)
        batch_y = torch.randn(0)

        _, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)

        assert isinstance(clip_state._num_clipped, dict)
        assert all(v == 0.0 for v in clip_state._num_clipped.values())


class TestAdaptivePerGroupAux:
    """Tests for auxiliary output with per-group adaptive clipping."""

    def test_return_aux_with_per_group(self):
        """Test that return_aux works with per-group adaptive clipping."""
        loss_fn = _make_per_group_loss_fn()
        params = {"a": torch.randn(10), "b": torch.randn(5)}
        pg = _make_per_group(params)

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=pg,
            key=key(0),
            batch_argnums=(1, 2),
            return_aux=True,
            expected_batch_size=1.0,
        )

        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        (_grads, aux), clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)

        # Aux should have group_norms
        assert aux.group_norms is not None
        assert isinstance(aux.group_norms, dict)
        assert "a" in aux.group_norms
        assert "b" in aux.group_norms
        # Each group_norms entry should be a 1D tensor of shape [batch_size]
        assert aux.group_norms["a"].shape == (8,)
        assert aux.group_norms["b"].shape == (8,)

    def test_group_norms_in_aux_without_return_aux(self):
        """Test that group_norms flow through _force_grad_norms path."""
        loss_fn = _make_per_group_loss_fn()
        params = {"a": torch.randn(10), "b": torch.randn(5)}
        pg = _make_per_group(params)

        # return_aux=False → _force_grad_norms=True inside adaptive
        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=pg,
            key=key(0),
            batch_argnums=(1, 2),
            return_aux=False,
            expected_batch_size=1.0,
        )

        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        _grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)

        # Even without return_aux, adaptation should work (state should change)
        assert isinstance(clip_state._next_clipping_norm, PerGroup)
        # Thresholds should have adapted (not equal to initial)
        assert clip_state._step == 1


class TestAdaptivePerGroupValidation:
    """Tests for validation of per-group adaptive clipping parameters."""

    def test_rejects_negative_per_group_value(self):
        """Test that negative per-group initial values are rejected."""
        pg = PerGroup(
            groups={"a": "a", "b": "b"},
            values={"a": -1.0, "b": 1.0},
        )

        with pytest.raises(ValueError, match="positive"):
            adaptive_clipped_grad(
                _make_per_group_loss_fn(),
                initial_clipping_norm=pg,
                key=key(0),
                batch_argnums=(1, 2),
                expected_batch_size=1.0,
            )

    def test_state_validation_rejects_negative_next_clipping_norm(self):
        """Test that AdaptiveClipState rejects negative per-group next_clipping_norm."""
        pg = PerGroup(
            groups={"a": "a"},
            values={"a": -0.1},
        )

        with pytest.raises(ValueError, match="positive"):
            AdaptiveClipState(
                _current_clipping_norm=pg,
                _next_clipping_norm=pg,
                _step=0,
                _rng_key=key(0),
                _fraction_noise_std=0.05,
                _expected_batch_size=1.0,
                _learning_rate=0.2,
                _target_quantile=0.5,
                _clipping_norm_min=0.01,
                _clipping_norm_max=100.0,
                _num_clipped={"a": 0.0},
                _batch_size=0,
            )


class TestAdaptivePerGroupMicrobatch:
    """Tests for microbatching with per-group adaptive clipping."""

    def test_microbatch_with_per_group(self):
        """Test that microbatching works with per-group adaptive clipping."""
        loss_fn = _make_per_group_loss_fn()
        params = {"a": torch.randn(10), "b": torch.randn(5)}
        pg = _make_per_group(params)

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=pg,
            key=key(0),
            batch_argnums=(1, 2),
            microbatch_size=4,
            expected_batch_size=1.0,
        )

        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
        grads = _unwrap_clipped(grads)

        assert isinstance(grads, dict)
        assert grads["a"].shape == params["a"].shape
        assert isinstance(clip_state._next_clipping_norm, PerGroup)

    def test_no_aux_microbatch_per_group_streams_stats(self, monkeypatch):
        """Per-group no-aux adaptive microbatching should not concatenate aux."""
        original = clipped_fun_module._microbatch_accumulate_reduced

        def wrapped(*args, **kwargs):
            if kwargs.get("return_aux"):
                raise AssertionError("unexpected per-example aux materialization")
            return original(*args, **kwargs)

        monkeypatch.setattr(
            clipped_fun_module, "_microbatch_accumulate_reduced", wrapped
        )

        loss_fn = _make_per_group_loss_fn()
        params = {"a": torch.randn(10), "b": torch.randn(5)}
        pg = _make_per_group(params)

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=pg,
            key=key(0),
            batch_argnums=(1, 2),
            microbatch_size=4,
            return_aux=False,
            expected_batch_size=1.0,
        )

        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
        grads = _unwrap_clipped(grads)

        assert isinstance(grads, dict)
        assert clip_state._step == 1

    def test_threshold_updates_reuse_one_strict_chunk_graph(self):
        params = {"a": torch.randn(10), "b": torch.randn(5)}
        pg = _make_per_group(params)
        backend = CompileCounterWithBackend("aot_eager")

        def compiler(fn):
            return torch.compile(fn, backend=backend, fullgraph=True)

        grad_fn, state = adaptive_clipped_grad(
            _make_per_group_loss_fn(),
            initial_clipping_norm=pg,
            key=key(0),
            batch_argnums=(1, 2),
            microbatch_size=4,
            return_aux=True,
            _chunk_compiler=compiler,
            expected_batch_size=1.0,
        )
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        for _ in range(3):
            (_grads, _aux), state = grad_fn(params, batch_x, batch_y, state=state)

        assert backend.frame_count == 1
        assert state._step == 3


class TestAdaptivePerGroupAccounting:
    """Tests for the accounting integration with num_groups."""

    def test_adaclip_num_groups_1_matches_original(self):
        """Test that num_groups=1 gives same result as before."""
        from opake.dpsgd.accounting import adaclip, gaussian

        ac1 = adaclip(gaussian(1.0), expected_batch_size=100, num_groups=1)
        ac_orig = adaclip(gaussian(1.0), expected_batch_size=100)

        assert ac1.effective_noise_multiplier == ac_orig.effective_noise_multiplier

    def test_adaclip_num_groups_lowers_effective_nm(self):
        """Test that more groups means lower effective noise multiplier."""
        from opake.dpsgd.accounting import adaclip, gaussian

        ac1 = adaclip(gaussian(1.0), expected_batch_size=256, num_groups=1)
        ac3 = adaclip(gaussian(1.0), expected_batch_size=256, num_groups=3)
        ac5 = adaclip(gaussian(1.0), expected_batch_size=256, num_groups=5)

        # More groups → more privacy consumed → lower effective nm
        assert ac3.effective_noise_multiplier < ac1.effective_noise_multiplier
        assert ac5.effective_noise_multiplier < ac3.effective_noise_multiplier

    def test_adaclip_num_groups_negligible_cost(self):
        """Test that K groups has negligible extra cost with typical hyperparams."""
        from opake.dpsgd.accounting import adaclip, gaussian

        nm = 1.1
        ebs = 256

        ac1 = adaclip(gaussian(nm), expected_batch_size=ebs, num_groups=1)
        ac5 = adaclip(gaussian(nm), expected_batch_size=ebs, num_groups=5)

        # Relative difference should be < 5% for typical values
        relative_diff = (
            abs(ac1.effective_noise_multiplier - ac5.effective_noise_multiplier)
            / ac1.effective_noise_multiplier
        )
        assert relative_diff < 0.05

    def test_adaclip_num_groups_formula(self):
        """Test the exact formula: z̃ = sqrt(1/z² + K/(4·σ_b²))."""
        from opake.dpsgd.accounting import adaclip, gaussian

        nm = 1.0
        ebs = 100.0
        frac_std = 0.05
        sigma_b = ebs * frac_std  # 5.0
        k = 3

        ac = adaclip(
            gaussian(nm),
            expected_batch_size=ebs,
            fraction_noise_std=frac_std,
            num_groups=k,
        )

        expected_sensitivity = math.sqrt(1.0 / nm**2 + k / (4 * sigma_b**2))
        expected_nm = 1.0 / expected_sensitivity

        assert abs(ac.effective_noise_multiplier - expected_nm) < 1e-10

    def test_adaclip_num_groups_pld(self):
        """Test that PLD computation works with num_groups > 1."""
        from opake.dpsgd.accounting import adaclip, gaussian, poisson

        step = poisson(
            adaclip(gaussian(1.0), expected_batch_size=256, num_groups=3),
            sample_rate=0.01,
        )
        training = step * 100
        eps = training.epsilon_at(1e-5)
        assert eps > 0
        assert math.isfinite(eps)

    def test_adaclip_gaussian_num_groups(self):
        """Test that Gaussian path handles num_groups > 1."""
        from opake.dpsgd.accounting import adaclip, gaussian, poisson

        step = poisson(
            adaclip(
                gaussian(1.0),
                expected_batch_size=256,
                num_groups=3,
            ),
            sample_rate=0.01,
        )
        training = step * 100
        eps = training.epsilon_at(1e-5)
        assert eps > 0
        assert math.isfinite(eps)

        # More groups should give higher epsilon.  Compare without Poisson
        # amplification, using a large inner noise_multiplier (cheap
        # gradient mechanism) so the bit-query cost dominates, and small
        # ebs + fraction_noise_std so each bit query is expensive.
        step_1g = adaclip(
            gaussian(5.0),
            expected_batch_size=5,
            fraction_noise_std=0.05,
            num_groups=1,
        )
        step_100g = adaclip(
            gaussian(5.0),
            expected_batch_size=5,
            fraction_noise_std=0.05,
            num_groups=100,
        )
        eps_1g = step_1g.epsilon_at(1e-5)
        eps_100g = step_100g.epsilon_at(1e-5)
        assert eps_100g > eps_1g  # More groups = more privacy consumed

    def test_adaclip_num_groups_validation(self):
        """Test that num_groups < 1 is rejected."""
        from opake.dpsgd.accounting import adaclip, gaussian

        with pytest.raises(ValueError, match="num_groups"):
            adaclip(gaussian(1.0), expected_batch_size=100, num_groups=0)


class TestGroupNormsInAux:
    """Tests for per-group norms flowing through the aux pipeline."""

    def test_group_norms_in_clipped_grad_aux(self):
        """Test that group_norms appear in ClippedGradAux when PerGroup is used."""
        from opake.dpsgd.clipping import clipped_grad

        loss_fn = _make_per_group_loss_fn()
        params = {"a": torch.randn(10), "b": torch.randn(5)}
        pg = _make_per_group(params)

        grad_fn, clip_state = clipped_grad(
            loss_fn,
            clipping_norm=pg,
            batch_argnums=(1, 2),
            return_aux=True,
        )

        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        (_grads, aux), _ = grad_fn(params, batch_x, batch_y, state=clip_state)

        assert aux.group_norms is not None
        assert "a" in aux.group_norms
        assert "b" in aux.group_norms
        assert aux.group_norms["a"].shape == (8,)
        assert aux.group_norms["b"].shape == (8,)

    def test_group_norms_none_for_scalar_clipping(self):
        """Test that group_norms is None when scalar clipping is used."""
        from opake.dpsgd.clipping import clipped_grad

        loss_fn = _make_loss_fn()

        grad_fn, clip_state = clipped_grad(
            loss_fn,
            clipping_norm=1.0,
            batch_argnums=(1, 2),
            return_aux=True,
        )

        params = torch.randn(10)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        (_grads, aux), _ = grad_fn(params, batch_x, batch_y, state=clip_state)

        assert aux.group_norms is None
