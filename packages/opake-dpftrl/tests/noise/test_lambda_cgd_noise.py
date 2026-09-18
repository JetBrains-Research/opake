"""Tests for DP-lambda-CGD noise generation via PRNG replay."""

import math

import pytest
import torch

import opake.dpftrl.accounting as ftrl_acc
from opake.api.dpftrl.noise import _lambda_cgd as lambda_cgd_module
from opake.api.dpftrl.noise._engine import MF_GAUSSIAN_STREAM_FOLD
from opake.api.dpftrl.noise._lambda_cgd import LambdaCgdStrategy, lambda_cgd_strategy
from opake.dpftrl.noise import mf_gaussian_noise
from opake.random import fold_in, generator_from_key, key
from opake.serialization import from_state_dict, state_dict
from opake.types import NoisedPytree, clipped


def _make_noise(template, n_steps=100, lambda_=0.9, normalized=True, seed=42):
    """Helper: create lambda-CGD noise via the strategy + mf_gaussian_noise API.

    Uses ``noise_multiplier=1.0`` so realized stddev equals each call's
    ``ClippedPytree.max_norm``; tests pass ``max_norm=1.0`` to recover the
    historical ``stddev=1.0`` semantics.
    """
    strategy = lambda_cgd_strategy(lambda_=lambda_, normalized=normalized)
    return mf_gaussian_noise(
        template,
        strategy,
        n_steps=n_steps,
        min_sep=1,
        max_participations=1,
        noise_multiplier=1.0,
        key=key(seed),
    )


def _call(noise_fn, grad_pytree, state, *, max_norm=1.0):
    """Wrap ``grad_pytree`` as clipped, run noise, return (noisy_pytree, state)."""
    noisy_out, new_state = noise_fn(clipped(grad_pytree, max_norm=max_norm), state)
    assert isinstance(noisy_out, NoisedPytree)
    return noisy_out.pytree, new_state


class TestLambdaCgdNoise:
    def _make_template(self):
        return {"w": torch.zeros(10)}

    def test_basic_noise_generation(self):
        """Noise function returns correctly shaped output."""
        template = self._make_template()
        noise_fn, state = _make_noise(template)
        noised, new_state = _call(noise_fn, {"w": torch.zeros(10)}, state)
        assert noised["w"].shape == (10,)
        assert new_state._step_counter == 1

    def test_deterministic_with_same_key(self):
        """Same key produces identical noise sequences."""
        template = self._make_template()
        results = []
        for _ in range(2):
            noise_fn, state = _make_noise(template)
            noised, state = _call(noise_fn, {"w": torch.zeros(10)}, state)
            noisy2, state = _call(noise_fn, {"w": torch.zeros(10)}, state)
            results.append(torch.cat([noised["w"], noisy2["w"]]))
        torch.testing.assert_close(results[0], results[1])

    def test_different_keys_give_different_noise(self):
        """Different keys produce different sequences."""
        template = self._make_template()
        noise_fn1, state1 = _make_noise(template, seed=1)
        noise_fn2, state2 = _make_noise(template, seed=2)
        noisy1, _ = _call(noise_fn1, {"w": torch.zeros(10)}, state1)
        noisy2, _ = _call(noise_fn2, {"w": torch.zeros(10)}, state2)
        assert not torch.allclose(noisy1["w"], noisy2["w"])

    def test_lambda_zero_is_independent(self):
        """lambda=0 should produce independent noise at each step (DP-SGD)."""
        template = self._make_template()
        noise_fn, state = _make_noise(template, lambda_=0.0)

        noisy0, state = _call(noise_fn, {"w": torch.zeros(10)}, state)
        noisy1, state = _call(noise_fn, {"w": torch.zeros(10)}, state)

        assert noisy0["w"].std().item() > 0.1
        assert noisy1["w"].std().item() > 0.1

    def test_first_step_same_as_lambda_zero(self):
        """First step is z_0 regardless of lambda (no previous noise) -- unnormalized."""
        template = self._make_template()

        noise_fn_corr, state_corr = _make_noise(template, lambda_=0.9, normalized=False)
        noise_fn_ind, state_ind = _make_noise(template, lambda_=0.0, normalized=False)

        noisy_corr, _ = _call(noise_fn_corr, {"w": torch.zeros(10)}, state_corr)
        noisy_ind, _ = _call(noise_fn_ind, {"w": torch.zeros(10)}, state_ind)

        torch.testing.assert_close(noisy_corr["w"], noisy_ind["w"])

    def test_correlation_changes_second_step(self):
        """Second step with lambda>0 should differ from lambda=0."""
        template = self._make_template()

        noise_fn_corr, state_corr = _make_noise(template, lambda_=0.9)
        noise_fn_ind, state_ind = _make_noise(template, lambda_=0.0)

        _, state_corr = _call(noise_fn_corr, {"w": torch.zeros(10)}, state_corr)
        _, state_ind = _call(noise_fn_ind, {"w": torch.zeros(10)}, state_ind)

        noisy_corr, _ = _call(noise_fn_corr, {"w": torch.zeros(10)}, state_corr)
        noisy_ind, _ = _call(noise_fn_ind, {"w": torch.zeros(10)}, state_ind)

        assert not torch.allclose(noisy_corr["w"], noisy_ind["w"])

    def test_multi_param(self):
        """Works with multiple parameter tensors."""
        template = {"w1": torch.zeros(5), "w2": torch.zeros(3, 4)}
        noise_fn, state = _make_noise(template)
        noised, _new_state = _call(
            noise_fn,
            {"w1": torch.zeros(5), "w2": torch.zeros(3, 4)},
            state,
        )
        assert noised["w1"].shape == (5,)
        assert noised["w2"].shape == (3, 4)

    def test_noise_adds_to_grads(self):
        """Noise is added to the gradient, not overwriting it."""
        template = self._make_template()
        noise_fn, state = _make_noise(template)
        grad = {"w": torch.ones(10) * 5.0}
        noised, _ = _call(noise_fn, grad, state)
        noise_fn2, state2 = _make_noise(template)
        noise_only, _ = _call(noise_fn2, {"w": torch.zeros(10)}, state2)
        torch.testing.assert_close(noised["w"] - 5.0, noise_only["w"])

    def test_step_counter_increments(self):
        """Step counter increments with each call."""
        template = self._make_template()
        noise_fn, state = _make_noise(template)
        assert state._step_counter == 0
        assert state._inner_state is None
        _, state = _call(noise_fn, {"w": torch.zeros(10)}, state)
        assert state._step_counter == 1
        assert state._inner_state is None
        _, state = _call(noise_fn, {"w": torch.zeros(10)}, state)
        assert state._step_counter == 2
        assert state._inner_state is None

    def test_rejects_invalid_lambda(self):
        for value in (-0.1, 1.0, float("nan"), float("inf")):
            with pytest.raises(
                ValueError, match=r"lambda_ must be finite and in \[0, 1\)"
            ):
                lambda_cgd_strategy(lambda_=value)

    @pytest.mark.parametrize("lambda_", [0.0, 0.5])
    @pytest.mark.parametrize("normalized", [False, True])
    def test_prng_replay_uses_namespaced_draws(self, lambda_, normalized):
        template = (torch.zeros(3), torch.zeros(2, 4))
        n_steps = 4
        base = key(42)
        noise_fn, state = _make_noise(
            template, n_steps=n_steps, lambda_=lambda_, normalized=normalized
        )
        previous = tuple(torch.zeros_like(leaf) for leaf in template)

        for step in range(n_steps):
            generator = generator_from_key(
                fold_in(
                    base,
                    MF_GAUSSIAN_STREAM_FOLD,
                    "mf_gaussian_column",
                    step,
                )
            )
            current = tuple(
                torch.randn(leaf.shape, generator=generator) for leaf in template
            )
            scale = (
                math.sqrt(sum(lambda_ ** (2 * j) for j in range(n_steps - step)))
                if normalized
                else 1.0
            )
            expected = tuple(
                (now - lambda_ * prev) * scale
                for now, prev in zip(current, previous, strict=True)
            )
            actual, state = noise_fn(clipped(template, max_norm=1.0), state)

            torch.testing.assert_close(actual.pytree, expected, atol=0.0, rtol=0.0)
            row_norm = scale * math.sqrt(1.0 + (lambda_**2 if step else 0.0))
            assert actual.noise_stddev == pytest.approx(row_norm)
            if step == 0:
                caller_draw = torch.randn(
                    template[0].shape, generator=generator_from_key(fold_in(base, 0))
                )
                assert not torch.equal(actual.pytree[0], caller_draw * scale)
            previous = current


class TestLambdaCgdCheckpoint:
    @pytest.mark.parametrize("normalized", [False, True])
    def test_resume_preserves_draws_with_a_fresh_template(self, normalized):
        template = {"w": torch.zeros(8)}
        noise_fn, state = _make_noise(template, n_steps=5, normalized=normalized)
        for _ in range(2):
            _, state = _call(noise_fn, template, state, max_norm=2.0)

        resumed_fn, fresh = _make_noise(
            template, n_steps=5, normalized=normalized, seed=73
        )
        restored = from_state_dict(fresh, state_dict(state))
        assert restored == state

        for _ in range(3):
            expected, state = _call(noise_fn, template, state, max_norm=2.0)
            actual, restored = _call(resumed_fn, template, restored, max_norm=2.0)
            torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


_PARTICIPATION = {"n_steps": 100, "min_sep": 25, "max_participations": 4}


class TestLambdaCgdStrategy:
    def test_returns_correct_type(self):
        s = lambda_cgd_strategy(lambda_=0.9)
        assert isinstance(s, LambdaCgdStrategy)

    def test_sensitivity_positive(self):
        s = lambda_cgd_strategy(lambda_=0.9)
        assert s.sensitivity(**_PARTICIPATION) > 0

    def test_gram_matrix_present(self):
        s = lambda_cgd_strategy(lambda_=0.9)
        gram = s.gram_matrix(**_PARTICIPATION)
        assert gram is not None
        assert len(gram) == 25 * 25

    def test_lr_schedule_is_rejected_with_recalibration_guidance(self):
        with pytest.raises(
            ValueError, match=r"does not support lr_schedule.*recalibrate"
        ):
            lambda_cgd_strategy(lambda_=0.4, lr_schedule=lambda _step: 1.0)

    def test_legacy_none_schedule_state_loads(self):
        strategy = from_state_dict(
            lambda_cgd_strategy(lambda_=0.4),
            {
                "type": "LambdaCgdStrategy",
                "lambda_": 0.4,
                "normalized": False,
                "lr_schedule": None,
            },
        )

        assert strategy == lambda_cgd_strategy(lambda_=0.4, normalized=False)
        assert state_dict(strategy)["lr_schedule"] is None

    def test_legacy_non_none_schedule_state_is_rejected(self):
        with pytest.raises(
            ValueError, match=r"does not support lr_schedule.*recalibrate"
        ):
            from_state_dict(
                lambda_cgd_strategy(lambda_=0.4),
                {
                    "type": "LambdaCgdStrategy",
                    "lambda_": 0.4,
                    "normalized": False,
                    "lr_schedule": {
                        "__opake_recipe__": "ConstantSchedule",
                        "value": 0.1,
                    },
                },
            )

    def test_gram_uses_only_unweighted_native_path(self, monkeypatch):
        expected = (2.0, 0.5, 0.5, 1.0)

        class CapturingNative:
            def lambda_cgd_gram_matrix(self, *_args):
                return expected

            def lambda_cgd_gram_matrix_lr(self, *_args):
                raise AssertionError("weighted Gram path must not be called")

        monkeypatch.setattr(lambda_cgd_module, "_native", CapturingNative)
        lambda_cgd_module._lambda_cgd_gram_matrix_cached.cache_clear()
        strategy = lambda_cgd_strategy(lambda_=0.4)

        assert strategy.gram_matrix(
            n_steps=4, min_sep=2, max_participations=2
        ) == pytest.approx(expected)
        lambda_cgd_module._lambda_cgd_gram_matrix_cached.cache_clear()

    def test_normalized_single_participation_sensitivity_one(self):
        """Normalized + single participation -> sensitivity = 1.0."""
        s = lambda_cgd_strategy(lambda_=0.9)
        assert s.sensitivity(
            n_steps=100, min_sep=1, max_participations=1
        ) == pytest.approx(1.0, abs=1e-6)

    def test_momentum_not_accepted(self):
        """lambda_cgd_strategy does not accept momentum (use bisr_strategy for that)."""
        with pytest.raises(TypeError):
            lambda_cgd_strategy(lambda_=0.5, momentum=0.95)

    def test_unnormalized(self):
        s = lambda_cgd_strategy(lambda_=0.9, normalized=False)
        assert s.sensitivity(**_PARTICIPATION) > 0

    def test_internal_fields(self):
        s = lambda_cgd_strategy(lambda_=0.9)
        assert s.lambda_ == pytest.approx(0.9)
        assert s.normalized is True


class TestLambdaCgdPld:
    delta = 1e-5

    def test_lambda_cgd_pld(self):
        s = lambda_cgd_strategy(lambda_=0.9)
        eps = ftrl_acc.mf_gaussian(1.0, s, **_PARTICIPATION).epsilon_at(self.delta)
        assert eps > 0

    def test_lambda_cgd_bnb(self):
        s = lambda_cgd_strategy(lambda_=0.9)
        eps = ftrl_acc.balls_in_bins(
            ftrl_acc.mf_gaussian(1.0, s),
            num_bins=25,
            n_steps=100,
        ).epsilon_at(
            1e-2,
            mc_resolution=5e-3,
            mc_failure_probability=1e-2,
        )
        assert eps > 0
