"""Tests for BisrStrategy factory, runtime dispatch, and accounting equivalence."""

import dataclasses
import math

import pytest
import torch

import opaque.dpftrl.accounting as ftrl_acc
from opaque.api.accounting.dpftrl.amplification import _balls_in_bins
from opaque.api.dpftrl.noise import _bisr as bisr_module
from opaque.api.dpftrl.noise._bisr import BisrStrategy, _native, bisr_strategy
from opaque.api.dpftrl.noise._engine import _matrix_factorization_noise
from opaque.api.dpftrl.noise._toeplitz import (
    inverse_as_streaming_matrix,
    materialize_lower_triangular,
)
from opaque.dpftrl.noise import mf_gaussian_noise
from opaque.exceptions import CheckpointError, ConfigurationError
from opaque.pytree import tree_leaves, tree_map
from opaque.random import key
from opaque.serialization import from_state_dict, state_dict
from opaque.types import clipped

_PART = {"n_steps": 100, "min_sep": 25, "max_participations": 4}


def _assert_tree_close(actual, expected, *, atol=1e-6, rtol=1e-6):
    actual_leaves = tree_leaves(actual)
    expected_leaves = tree_leaves(expected)
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        torch.testing.assert_close(
            actual_leaf,
            expected_leaf,
            atol=atol,
            rtol=rtol,
        )


def _tensor_bytes(value) -> int:
    return sum(
        leaf.numel() * leaf.element_size()
        for leaf in state_dict(value).values()
        if isinstance(leaf, torch.Tensor)
    )


class TestBisrStrategy:
    def test_returns_correct_type(self):
        assert isinstance(bisr_strategy(bandwidth=4), BisrStrategy)

    def test_sensitivity_positive(self):
        assert bisr_strategy(bandwidth=4).sensitivity(**_PART) > 0

    def test_gram_matrix_present(self):
        gram = bisr_strategy(bandwidth=4).gram_matrix(**_PART)
        assert gram is not None
        assert len(gram) == 25 * 25

    def test_lr_schedule_is_rejected_with_recalibration_guidance(self):
        with pytest.raises(
            ValueError, match=r"does not support lr_schedule.*recalibrate"
        ):
            bisr_strategy(bandwidth=3, lr_schedule=lambda _step: 1.0)

    def test_legacy_none_schedule_state_loads(self):
        strategy = from_state_dict(
            bisr_strategy(bandwidth=3),
            {
                "type": "BisrStrategy",
                "bandwidth": 3,
                "normalized": False,
                "momentum": 0.3,
                "lr_schedule": None,
                "inv_coefficients": None,
            },
        )

        assert strategy == bisr_strategy(bandwidth=3, normalized=False, momentum=0.3)
        assert state_dict(strategy)["lr_schedule"] is None

    def test_legacy_non_none_schedule_state_is_rejected(self):
        with pytest.raises(
            ValueError, match=r"does not support lr_schedule.*recalibrate"
        ):
            from_state_dict(
                bisr_strategy(bandwidth=3),
                {
                    "type": "BisrStrategy",
                    "bandwidth": 3,
                    "normalized": False,
                    "momentum": 0.3,
                    "lr_schedule": {
                        "__opaque_recipe__": "ConstantSchedule",
                        "value": 0.1,
                    },
                    "inv_coefficients": None,
                },
            )

    def test_gram_uses_only_unweighted_native_path(self, monkeypatch):
        expected = (2.0, 0.5, 0.5, 1.0)

        class CapturingNative:
            def bisr_gram_matrix(self, *_args):
                return expected

            def bisr_gram_matrix_lr(self, *_args):
                raise AssertionError("weighted Gram path must not be called")

        monkeypatch.setattr(bisr_module, "_native", CapturingNative)
        bisr_module._bisr_gram_matrix_cached.cache_clear()
        strategy = bisr_strategy(bandwidth=3)

        assert strategy.gram_matrix(
            n_steps=4, min_sep=2, max_participations=2
        ) == pytest.approx(expected)
        bisr_module._bisr_gram_matrix_cached.cache_clear()

    def test_streaming_matrix_present(self):
        assert bisr_strategy(bandwidth=4).streaming_matrix(**_PART) is not None

    def test_exposes_runtime_noise_factory(self):
        strategy = bisr_strategy(bandwidth=4)
        template = {"w": torch.zeros(6)}
        raw = strategy.raw_noise_factory(
            template,
            n_steps=12,
            min_sep=3,
            max_participations=2,
            key=key(0),
            compute_dtype=torch.float32,
        )
        noise_fn, state, row_l2_at = raw
        assert callable(noise_fn)
        assert state._step_counter == 0
        assert row_l2_at(0) > 0

    def test_mf_gaussian_noise_uses_runtime_operator(self, monkeypatch):
        strategy = bisr_strategy(bandwidth=4)
        template = {"w": torch.zeros(6)}
        calls = {"count": 0}
        original = BisrStrategy.raw_noise_factory

        def fake_raw_noise_factory(self, *args, **kwargs):
            calls["count"] += 1
            return original(self, *args, **kwargs)

        monkeypatch.setattr(BisrStrategy, "raw_noise_factory", fake_raw_noise_factory)
        noise_fn, state = mf_gaussian_noise(
            template,
            strategy,
            n_steps=12,
            min_sep=3,
            max_participations=2,
            noise_multiplier=1.0,
            key=key(1),
        )
        out, _ = noise_fn(clipped({"w": torch.zeros(6)}, max_norm=1.0), state)
        assert calls["count"] == 1
        assert float(out.noise_stddev) > 0

    @pytest.mark.parametrize("bandwidth", [2, 4])
    @pytest.mark.parametrize("n_steps", [2, 6, 12])
    def test_runtime_operator_uses_full_horizon_strategy(self, bandwidth, n_steps):
        strategy = bisr_strategy(bandwidth=bandwidth, normalized=False, momentum=0.3)
        streaming = strategy.streaming_matrix(n_steps=n_steps)
        runtime_noise_fn, _, runtime_row_l2_at = strategy.raw_noise_factory(
            {"w": torch.zeros(1)},
            n_steps=n_steps,
            min_sep=1,
            max_participations=1,
            key=key(0),
            compute_dtype=torch.float32,
        )
        del runtime_noise_fn

        expected_dense = streaming.materialize(n_steps)
        runtime_row_l2 = torch.tensor(
            [runtime_row_l2_at(step) for step in range(n_steps)], dtype=torch.float64
        )
        expected_row_l2 = expected_dense.pow(2).sum(dim=1).sqrt()

        torch.testing.assert_close(runtime_row_l2, expected_row_l2)

        full_horizon_strategy_coefs = _native().bisr_strategy_coefficients(
            list(strategy._inv_coefs()), n_steps
        )
        assert len(full_horizon_strategy_coefs) == n_steps
        manual_streaming = inverse_as_streaming_matrix(
            torch.tensor(full_horizon_strategy_coefs, dtype=torch.float64)
        )
        torch.testing.assert_close(
            expected_dense, manual_streaming.materialize(n_steps)
        )

    @pytest.mark.parametrize("normalized", [False, True])
    def test_row_norms_closed_form_matches_probing(self, normalized):
        n_steps = 12
        strategy = bisr_strategy(bandwidth=3, normalized=normalized)
        streaming = strategy.streaming_matrix(n_steps=n_steps)
        closed_form = streaming.row_norms_squared(n_steps)
        probing = dataclasses.replace(
            streaming, row_norms_squared_fn=None
        ).row_norms_squared(n_steps)
        torch.testing.assert_close(closed_form, probing, atol=1e-10, rtol=1e-10)

    def test_matches_old_sensitivity(self):
        assert bisr_strategy(bandwidth=4).sensitivity(**_PART) > 0

    def test_with_momentum(self):
        assert bisr_strategy(bandwidth=4, momentum=0.95).sensitivity(**_PART) > 0

    def test_coefficients_match_lemma_one_alpha_one_specialization(self):
        beta = 0.4
        strategy = bisr_strategy(bandwidth=4, momentum=beta)
        r = [1.0]
        for j in range(1, strategy.bandwidth):
            r.append(((j - 1.5) / j) * r[-1])
        expected_inverse = tuple(
            sum(r[j] * beta**j * r[k - j] for j in range(k + 1))
            for k in range(strategy.bandwidth)
        )

        assert strategy._inv_coefs() == pytest.approx(expected_inverse)

    def test_rejects_bad_bandwidth(self):
        with pytest.raises(ValueError, match="bandwidth must be >= 2"):
            bisr_strategy(bandwidth=1)

    @pytest.mark.parametrize("momentum", [-0.1, 1.0, float("nan"), float("inf")])
    def test_rejects_invalid_momentum(self, momentum):
        with pytest.raises(
            ValueError, match=r"momentum must be finite and in \[0, 1\)"
        ):
            bisr_strategy(bandwidth=2, momentum=momentum)

    @pytest.mark.parametrize(
        "coefficients",
        [(0.0, 1.0), (1e-31, 1.0), (1.0, float("nan")), (1.0, float("inf"))],
    )
    def test_rejects_invalid_custom_inverse_coefficients(self, coefficients):
        with pytest.raises(ValueError, match="inv_coefficients"):
            bisr_strategy(bandwidth=2, inv_coefficients=coefficients)

    def test_coefficients_reject_nonpositive_horizon(self):
        with pytest.raises(ValueError, match="n_steps must be >= 1"):
            bisr_strategy(bandwidth=2).coefficients(n_steps=0)


class TestBisrBoundedRuntime:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize(
        ("n_steps", "bandwidth", "momentum", "normalized", "inverse"),
        [
            (12, 2, 0.0, False, None),
            (12, 4, 0.9, True, None),
            (3, 5, 0.3, True, None),
            (8, 3, 0.0, False, (2.0, -0.5, 0.25)),
            (8, 3, 0.0, True, (2.0, -0.5, 0.25)),
        ],
    )
    def test_direct_runtime_matches_dense_reference(
        self,
        dtype,
        n_steps,
        bandwidth,
        momentum,
        normalized,
        inverse,
    ):
        template = {
            "left": torch.zeros(7, dtype=dtype),
            "nested": {"right": torch.zeros(3, 2, dtype=dtype)},
        }
        strategy = bisr_strategy(
            bandwidth=bandwidth,
            momentum=momentum,
            normalized=normalized,
            inv_coefficients=inverse,
        )
        direct_fn, direct_state, direct_row_l2 = strategy.raw_noise_factory(
            template,
            n_steps=n_steps,
            min_sep=1,
            max_participations=None,
            key=key(795),
            compute_dtype=dtype,
        )
        dense_matrix = strategy.streaming_matrix(n_steps=n_steps)
        dense_fn, dense_state = _matrix_factorization_noise(
            template,
            dense_matrix,
            key=key(795),
            compute_dtype=dtype,
            n_steps=n_steps,
        )
        dense_row_l2 = dense_matrix.materialize(n_steps).square().sum(dim=1).sqrt()

        zeros = tree_map(torch.zeros_like, template)
        for step in range(n_steps):
            direct, direct_state = direct_fn(zeros, direct_state, stddev=1.25)
            dense, dense_state = dense_fn(zeros, dense_state, stddev=1.25)
            tolerance = 2e-5 if dtype == torch.float32 else 1e-10
            _assert_tree_close(
                direct,
                dense,
                atol=tolerance,
                rtol=tolerance,
            )
            assert direct_row_l2(step) == pytest.approx(
                float(dense_row_l2[step]),
                abs=1e-10,
                rel=1e-10,
            )

    @pytest.mark.parametrize(
        ("n_steps", "expected_histories"),
        [(1, 0), (3, 2), (16, 3), (256, 3)],
    )
    def test_runtime_state_bytes_are_bounded(self, n_steps, expected_histories):
        template = {
            "left": torch.zeros(11, dtype=torch.float32),
            "right": torch.zeros(7, dtype=torch.float32),
        }
        strategy = bisr_strategy(bandwidth=4)
        noise_fn, state, _ = strategy.raw_noise_factory(
            template,
            n_steps=n_steps,
            min_sep=1,
            max_participations=None,
            key=key(1),
            compute_dtype=torch.float32,
        )
        template_bytes = _tensor_bytes(template)
        expected_bytes = expected_histories * template_bytes

        assert len(state._inner_state.history) == expected_histories
        assert _tensor_bytes(state._inner_state) == expected_bytes

        zeros = tree_map(torch.zeros_like, template)
        for _ in range(min(n_steps, 6)):
            _, state = noise_fn(zeros, state, stddev=1.0)

        assert len(state._inner_state.history) == expected_histories
        assert _tensor_bytes(state._inner_state) == expected_bytes

    def test_compute_dtype_controls_history_dtype(self):
        template = {"w": torch.zeros(5, dtype=torch.float32)}
        noise_fn, state, _ = bisr_strategy(bandwidth=4).raw_noise_factory(
            template,
            n_steps=8,
            min_sep=1,
            max_participations=None,
            key=key(2),
            compute_dtype=torch.float64,
        )

        history_leaves = [
            value
            for value in state_dict(state._inner_state).values()
            if isinstance(value, torch.Tensor)
        ]
        assert history_leaves
        assert all(value.dtype == torch.float64 for value in history_leaves)

        output, state = noise_fn(template, state, stddev=1.0)
        assert output["w"].dtype == torch.float32
        history_leaves = [
            value
            for value in state_dict(state._inner_state).values()
            if isinstance(value, torch.Tensor)
        ]
        assert all(value.dtype == torch.float64 for value in history_leaves)

    @pytest.mark.parametrize("compute_dtype", [torch.complex64, torch.int64])
    def test_rejects_non_real_compute_dtype(self, compute_dtype):
        with pytest.raises(ConfigurationError, match="real floating-point"):
            bisr_strategy(bandwidth=3).raw_noise_factory(
                {"w": torch.zeros(4)},
                n_steps=5,
                min_sep=1,
                max_participations=None,
                key=key(2),
                compute_dtype=compute_dtype,
            )

    @pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_inputs_keep_float32_history_and_output_dtype(
        self, input_dtype
    ):
        template = {"w": torch.zeros(5, dtype=input_dtype)}
        noise_fn, state = mf_gaussian_noise(
            template,
            bisr_strategy(bandwidth=3),
            n_steps=5,
            noise_multiplier=1.0,
            key=key(8),
        )
        grads = clipped(template, max_norm=1.0)

        for _ in range(4):
            output, state = noise_fn(grads, state)
            assert output.pytree["w"].dtype == input_dtype
            history_leaves = [
                value
                for value in state_dict(state._inner_state).values()
                if isinstance(value, torch.Tensor)
            ]
            assert history_leaves
            assert all(value.dtype == torch.float32 for value in history_leaves)

    def test_raw_factory_does_not_construct_dense_streaming_state(self, monkeypatch):
        def fail_if_called(*_args, **_kwargs):
            raise AssertionError("production BISR must not build the dense reference")

        monkeypatch.setattr(BisrStrategy, "streaming_matrix", fail_if_called)
        noise_fn, state, row_l2_at = bisr_strategy(bandwidth=4).raw_noise_factory(
            {"w": torch.zeros(3)},
            n_steps=8,
            min_sep=1,
            max_participations=None,
            key=key(3),
            compute_dtype=torch.float32,
        )

        output, _ = noise_fn({"w": torch.zeros(3)}, state, stddev=1.0)
        assert output["w"].shape == (3,)
        assert row_l2_at(0) > 0

    @pytest.mark.parametrize("n_steps", [True, 2.5, 0, -1])
    def test_raw_factory_rejects_invalid_horizon(self, n_steps):
        with pytest.raises((TypeError, ValueError), match="n_steps"):
            bisr_strategy(bandwidth=2).raw_noise_factory(
                {"w": torch.zeros(1)},
                n_steps=n_steps,
                min_sep=1,
                max_participations=None,
                key=key(4),
                compute_dtype=torch.float32,
            )

    def test_new_state_dict_continues_bit_identically(self):
        n_steps = 10
        template = {
            "left": torch.zeros(5),
            "nested": {"right": torch.zeros(2, 3)},
        }
        strategy = bisr_strategy(bandwidth=4, momentum=0.9)
        continuous_fn, continuous_state, _ = strategy.raw_noise_factory(
            template,
            n_steps=n_steps,
            min_sep=1,
            max_participations=None,
            key=key(5),
            compute_dtype=torch.float64,
        )
        zeros = tree_map(torch.zeros_like, template)
        for _ in range(5):
            _, continuous_state = continuous_fn(
                zeros,
                continuous_state,
                stddev=0.75,
            )

        saved = state_dict(continuous_state)
        assert saved["_inner_state.layout_version"] == 1
        resumed_fn, resumed_template, _ = strategy.raw_noise_factory(
            template,
            n_steps=n_steps,
            min_sep=1,
            max_participations=None,
            key=key(999),
            compute_dtype=torch.float64,
        )
        resumed_state = from_state_dict(resumed_template, saved)

        for _ in range(5, n_steps):
            expected, continuous_state = continuous_fn(
                zeros,
                continuous_state,
                stddev=0.75,
            )
            actual, resumed_state = resumed_fn(
                zeros,
                resumed_state,
                stddev=0.75,
            )
            _assert_tree_close(actual, expected, atol=0.0, rtol=0.0)

    @pytest.mark.parametrize(
        (
            "source_kwargs",
            "target_kwargs",
            "source_n_steps",
            "target_n_steps",
            "source_dtype",
            "target_dtype",
        ),
        [
            pytest.param(
                {
                    "bandwidth": 3,
                    "normalized": True,
                    "inv_coefficients": (1.0, -0.5, 0.25),
                },
                {
                    "bandwidth": 3,
                    "normalized": False,
                    "inv_coefficients": (1.0, -0.5, 0.25),
                },
                8,
                8,
                torch.float32,
                torch.float32,
                id="normalized",
            ),
            pytest.param(
                {"bandwidth": 3, "normalized": False, "momentum": 0.0},
                {"bandwidth": 3, "normalized": False, "momentum": 0.9},
                8,
                8,
                torch.float32,
                torch.float32,
                id="momentum-derived-effective-coefficients",
            ),
            pytest.param(
                {
                    "bandwidth": 3,
                    "normalized": False,
                    "inv_coefficients": (1.0, -0.5, 0.25),
                },
                {
                    "bandwidth": 3,
                    "normalized": False,
                    "inv_coefficients": (1.0, -0.25, 0.125),
                },
                8,
                8,
                torch.float32,
                torch.float32,
                id="custom-effective-coefficients",
            ),
            pytest.param(
                {"bandwidth": 3, "normalized": True, "momentum": 0.3},
                {"bandwidth": 3, "normalized": True, "momentum": 0.3},
                6,
                9,
                torch.float32,
                torch.float32,
                id="horizon-with-same-ring-shape",
            ),
            pytest.param(
                {"bandwidth": 3, "normalized": True, "momentum": 0.3},
                {"bandwidth": 3, "normalized": True, "momentum": 0.3},
                8,
                8,
                torch.float32,
                torch.float64,
                id="compute-dtype",
            ),
        ],
    )
    def test_direct_and_serialized_restore_reject_execution_drift(
        self,
        source_kwargs,
        target_kwargs,
        source_n_steps,
        target_n_steps,
        source_dtype,
        target_dtype,
    ):
        template = {"w": torch.zeros(4)}
        source_fn, source_state, _ = bisr_strategy(**source_kwargs).raw_noise_factory(
            template,
            n_steps=source_n_steps,
            min_sep=1,
            max_participations=None,
            key=key(795),
            compute_dtype=source_dtype,
        )
        target_fn, target_template, _ = bisr_strategy(
            **target_kwargs
        ).raw_noise_factory(
            template,
            n_steps=target_n_steps,
            min_sep=1,
            max_participations=None,
            key=key(796),
            compute_dtype=target_dtype,
        )

        zeros = tree_map(torch.zeros_like, template)
        for _ in range(2):
            _, source_state = source_fn(zeros, source_state, stddev=1.0)
        saved = state_dict(source_state)

        # Runtime state is functional and publicly threaded by callers. Passing
        # it directly to a differently configured factory must fail even when
        # the bounded history has the same length and tensor shapes.
        with pytest.raises(CheckpointError):
            target_fn(zeros, source_state, stddev=1.0)

        # Template-driven restore must enforce the same execution identity; a
        # layout version and matching ring shape alone are insufficient.
        with pytest.raises(CheckpointError):
            from_state_dict(target_template, saved)

    def test_direct_state_rejects_broadcast_compatible_shape_drift(self):
        strategy = bisr_strategy(bandwidth=3, momentum=0.3)
        source_fn, source_state, _ = strategy.raw_noise_factory(
            {"w": torch.zeros(1)},
            n_steps=6,
            min_sep=1,
            max_participations=None,
            key=key(796),
            compute_dtype=torch.float32,
        )
        target_fn, _, _ = strategy.raw_noise_factory(
            {"w": torch.zeros(4)},
            n_steps=6,
            min_sep=1,
            max_participations=None,
            key=key(797),
            compute_dtype=torch.float32,
        )
        _, source_state = source_fn({"w": torch.zeros(1)}, source_state, stddev=1.0)

        with pytest.raises(CheckpointError, match="history leaf does not match"):
            target_fn({"w": torch.zeros(4)}, source_state, stddev=1.0)

    def test_restore_schema_validation_does_not_clone_template_history(
        self, monkeypatch
    ):
        template = {"w": torch.zeros(4)}
        strategy = bisr_strategy(bandwidth=3, momentum=0.3)
        noise_fn, state, _ = strategy.raw_noise_factory(
            template,
            n_steps=8,
            min_sep=1,
            max_participations=None,
            key=key(797),
            compute_dtype=torch.float32,
        )
        _, state = noise_fn(template, state, stddev=1.0)
        saved = state_dict(state)
        _, restore_template, _ = strategy.raw_noise_factory(
            template,
            n_steps=8,
            min_sep=1,
            max_participations=None,
            key=key(798),
            compute_dtype=torch.float32,
        )

        def reject_clone(_tensor, *_args, **_kwargs):
            raise AssertionError(
                "restore schema validation must not serialize/clone template tensors"
            )

        monkeypatch.setattr(torch.Tensor, "clone", reject_clone)
        restored = from_state_dict(restore_template, saved)

        assert restored._step_counter == state._step_counter
        assert restored._inner_state.step == state._inner_state.step

    def test_empty_pytree_state_round_trips_and_continues(self):
        strategy = bisr_strategy(bandwidth=3, momentum=0.3)
        continuous_fn, continuous_state, _ = strategy.raw_noise_factory(
            {},
            n_steps=5,
            min_sep=1,
            max_participations=None,
            key=key(798),
            compute_dtype=torch.float32,
        )
        for _ in range(2):
            output, continuous_state = continuous_fn({}, continuous_state, stddev=1.0)
            assert output == {}

        saved = state_dict(continuous_state)
        resumed_fn, resumed_template, _ = strategy.raw_noise_factory(
            {},
            n_steps=5,
            min_sep=1,
            max_participations=None,
            key=key(999),
            compute_dtype=torch.float32,
        )
        resumed_state = from_state_dict(resumed_template, saved)

        expected, _ = continuous_fn({}, continuous_state, stddev=1.0)
        actual, _ = resumed_fn({}, resumed_state, stddev=1.0)
        assert actual == expected == {}

    def test_extreme_normalized_coefficients_keep_unit_release_finite(self):
        template = {"w": torch.zeros(64, dtype=torch.float64)}
        extreme = bisr_strategy(
            bandwidth=2,
            normalized=True,
            inv_coefficients=(1e200, 0.0),
        )
        reference = bisr_strategy(
            bandwidth=2,
            normalized=True,
            inv_coefficients=(1.0, 0.0),
        )
        extreme_fn, extreme_state, extreme_row_l2 = extreme.raw_noise_factory(
            template,
            n_steps=1,
            min_sep=1,
            max_participations=None,
            key=key(799),
            compute_dtype=torch.float64,
        )
        reference_fn, reference_state, reference_row_l2 = reference.raw_noise_factory(
            template,
            n_steps=1,
            min_sep=1,
            max_participations=None,
            key=key(799),
            compute_dtype=torch.float64,
        )

        actual, _ = extreme_fn(template, extreme_state, stddev=1.0)
        expected, _ = reference_fn(template, reference_state, stddev=1.0)

        assert math.isfinite(extreme_row_l2(0))
        assert extreme_row_l2(0) == pytest.approx(reference_row_l2(0), rel=1e-14)
        assert extreme_row_l2(0) == pytest.approx(1.0, rel=1e-14)
        assert bool(torch.isfinite(actual["w"]).all())
        torch.testing.assert_close(actual["w"], expected["w"], atol=0.0, rtol=1e-14)

    def test_effective_coefficients_must_fit_compute_dtype(self):
        template = {"w": torch.zeros(4)}
        with pytest.raises(ConfigurationError, match="not representable"):
            bisr_strategy(
                bandwidth=2,
                normalized=False,
                inv_coefficients=(1e40, 0.0),
            ).raw_noise_factory(
                template,
                n_steps=1,
                min_sep=1,
                max_participations=None,
                key=key(800),
                compute_dtype=torch.float32,
            )

        normalized_fn, normalized_state, row_l2_at = bisr_strategy(
            bandwidth=2,
            normalized=True,
            inv_coefficients=(1e40, 0.0),
        ).raw_noise_factory(
            template,
            n_steps=1,
            min_sep=1,
            max_participations=None,
            key=key(800),
            compute_dtype=torch.float32,
        )
        output, _ = normalized_fn(template, normalized_state, stddev=1.0)
        assert bool(torch.isfinite(output["w"]).all())
        assert row_l2_at(0) == pytest.approx(1.0)

    @pytest.mark.parametrize("normalized", [False, True])
    @pytest.mark.parametrize("n_steps", [3, 4, 9])
    def test_legacy_dense_state_is_rejected(self, normalized, n_steps):
        template = {"w": torch.zeros(3)}
        strategy = bisr_strategy(bandwidth=4, normalized=normalized)
        _, legacy_state = _matrix_factorization_noise(
            template,
            strategy.streaming_matrix(n_steps=n_steps),
            key=key(6),
            compute_dtype=torch.float32,
            n_steps=n_steps,
        )
        _, bounded_template, _ = strategy.raw_noise_factory(
            template,
            n_steps=n_steps,
            min_sep=1,
            max_participations=None,
            key=key(6),
            compute_dtype=torch.float32,
        )

        with pytest.raises(CheckpointError, match="legacy BISR dense-history"):
            from_state_dict(bounded_template, state_dict(legacy_state))

    def test_unknown_incomplete_and_mismatched_state_is_rejected(self):
        template = {"w": torch.zeros(3)}
        strategy = bisr_strategy(bandwidth=3)
        noise_fn, bounded_template, _ = strategy.raw_noise_factory(
            template,
            n_steps=6,
            min_sep=1,
            max_participations=None,
            key=key(7),
            compute_dtype=torch.float32,
        )
        saved = state_dict(bounded_template)

        unknown = dict(saved)
        unknown["_inner_state.layout_version"] = 999
        with pytest.raises(CheckpointError, match="Unsupported BISR"):
            from_state_dict(bounded_template, unknown)

        incomplete = dict(saved)
        history_key = next(
            name for name in incomplete if name.startswith("_inner_state.history")
        )
        del incomplete[history_key]
        with pytest.raises(CheckpointError, match="fields do not match"):
            from_state_dict(bounded_template, incomplete)

        past_horizon = dict(saved)
        past_horizon["_inner_state.step"] = 7
        past_horizon["_step_counter"] = 7
        with pytest.raises(CheckpointError, match="within the configured horizon"):
            from_state_dict(bounded_template, past_horizon)

        mismatched = dict(saved)
        mismatched["_step_counter"] = 1
        restored = from_state_dict(bounded_template, mismatched)
        with pytest.raises(CheckpointError, match="step counters disagree"):
            noise_fn(template, restored, stddev=1.0)


class TestBisrPld:
    delta = 1e-5

    def test_bisr_pld(self):
        eps = ftrl_acc.mf_gaussian(1.0, bisr_strategy(bandwidth=4), **_PART).epsilon_at(
            self.delta
        )
        assert eps > 0

    def test_bisr_bnb(self):
        eps = ftrl_acc.balls_in_bins(
            ftrl_acc.mf_gaussian(1.0, bisr_strategy(bandwidth=4)),
            num_bins=25,
            n_steps=100,
        ).epsilon_at(
            1e-2,
            mc_resolution=5e-3,
            mc_failure_probability=1e-2,
        )
        assert eps > 0

    def test_bnb_uses_absolute_normalized_forward_encoder(self, monkeypatch):
        n_steps, num_bins = 4, 2
        strategy = bisr_strategy(
            bandwidth=2,
            normalized=True,
            inv_coefficients=(1.0, 1.0),
        )
        process = ftrl_acc.balls_in_bins(
            ftrl_acc.mf_gaussian(1.0, strategy),
            num_bins=num_bins,
            n_steps=n_steps,
        )

        actual_native = _balls_in_bins._native
        captured: dict[str, tuple[float, ...]] = {}
        sentinel = object()

        class CapturingNative:
            def __getattr__(self, name):
                return getattr(actual_native, name)

            def bnb_mc_pld(self, gram, *_):
                captured["gram"] = tuple(gram)
                return sentinel

        monkeypatch.setattr(_balls_in_bins, "_native", CapturingNative())
        assert process.pld() is sentinel

        encoder = materialize_lower_triangular(
            strategy.coefficients(n_steps=n_steps), n_steps
        )
        normalized_encoder = encoder / encoder.square().sum(dim=0).sqrt()
        grouped = torch.stack(
            [
                normalized_encoder[:, bin_index::num_bins].abs().sum(dim=1)
                for bin_index in range(num_bins)
            ],
            dim=1,
        )
        expected = grouped.T @ grouped

        actual = torch.tensor(captured["gram"], dtype=torch.float64).reshape(
            num_bins, num_bins
        )
        torch.testing.assert_close(actual, expected)
        assert actual[0, 1] > 0
