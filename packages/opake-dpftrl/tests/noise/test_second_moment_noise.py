"""Tests for private second-moment MF noise."""

import math

import pytest
import torch

from opake.api.dpftrl.noise._engine import MF_GAUSSIAN_STREAM_FOLD
from opake.api.engine.noise_allocation import paired_noise_stddevs
from opake.dpftrl.noise import (
    band_mf_strategy,
    bisr_strategy,
    blt_strategy,
    bsr_strategy,
    identity_strategy,
    lambda_cgd_strategy,
    mf_gaussian_noise,
)
from opake.dpftrl.noise.types import SecondMomentMFNoiseState
from opake.exceptions import ConfigurationError
from opake.pytree import tree_leaves
from opake.random import fold_in, generator_from_key, key
from opake.serialization import from_state_dict, state_dict
from opake.types import (
    NoisedPytree,
    PerGroup,
    SecondMomentClippingOutput,
    SecondMomentNoiseOutput,
    clipped,
)

_SENSITIVITY = 0.1


def _max_column_norm(strategy, *, n_steps: int) -> float:
    """Strategy's single-participation sensitivity = ``‖C‖_{1→2}``."""
    return strategy.sensitivity(n_steps=n_steps, min_sep=n_steps, max_participations=1)


def _row_l2_at_zero(strategy, *, n_steps: int, min_sep: int = 1) -> float:
    """First-step ``‖row_0(C^-1)‖`` — used to back out base σ from the
    realized σ that :class:`NoisedPytree.noise_stddev` now publishes.
    """
    streaming = strategy.streaming_matrix(
        n_steps=n_steps, min_sep=min_sep, max_participations=1
    )
    return float(streaming.row_norms_squared(n_steps).clamp_min(0.0).sqrt()[0])


def _paired(grads):
    """Build a SecondMomentClippingOutput directly from raw grads.

    Tests pass already-computed gradients (no clipping loop), so we
    construct the paired form by hand: each stream gets its own
    ``ClippedPytree`` with the appropriate max_norm bound.  The squared
    stream's payload is element-wise g² and its bound is the squared
    contribution bound.
    """
    grads_clipped = clipped(grads, max_norm=_SENSITIVITY)
    sq_pytree = {k: v * v for k, v in grads.items()}
    sq_clipped = clipped(sq_pytree, max_norm=_SENSITIVITY * _SENSITIVITY)
    return SecondMomentClippingOutput(grads=grads_clipped, squared_grads=sq_clipped)


def _clipped(grads):
    """Wrap raw grad pytree as ClippedPytree at the test's standard max_norm."""
    return clipped(grads, max_norm=_SENSITIVITY)


def test_lambda_cgd_streams_keep_namespaces_after_resume():
    template = {"w": torch.zeros(8)}
    strategy = lambda_cgd_strategy(lambda_=0.5, normalized=False)

    def make_noise(seed):
        return mf_gaussian_noise(
            template,
            strategy,
            n_steps=4,
            noise_multiplier=1.0,
            key=key(seed),
            second_moment_strategy=strategy,
        )

    noise_fn, state = make_noise(42)
    inputs = SecondMomentClippingOutput(
        grads=clipped(template, max_norm=1.0),
        squared_grads=clipped(template, max_norm=1.0),
    )
    _, state = noise_fn(inputs, state)
    resumed_fn, fresh = make_noise(73)
    restored = from_state_dict(fresh, state_dict(state))

    for step in range(1, 4):
        expected, state = noise_fn(inputs, state)
        actual, restored = resumed_fn(inputs, restored)
        for name, expected_stream, actual_stream in (
            ("first", expected.noisy_grads, actual.noisy_grads),
            ("second", expected.noisy_squared_grads, actual.noisy_squared_grads),
        ):
            parent = fold_in(key(42), f"opake.dpftrl.second_moment.{name}")
            current = torch.randn(
                (8,),
                generator=generator_from_key(
                    fold_in(
                        parent,
                        MF_GAUSSIAN_STREAM_FOLD,
                        "mf_gaussian_column",
                        step,
                    )
                ),
            ) * math.sqrt(2.0)
            previous = torch.randn(
                (8,),
                generator=generator_from_key(
                    fold_in(
                        parent,
                        MF_GAUSSIAN_STREAM_FOLD,
                        "mf_gaussian_column",
                        step - 1,
                    )
                ),
            ) * math.sqrt(2.0)
            torch.testing.assert_close(
                actual_stream.pytree["w"], current - 0.5 * previous
            )
            torch.testing.assert_close(
                actual_stream.pytree, expected_stream.pytree, atol=0.0, rtol=0.0
            )


class TestSecondMomentCalibration:
    """``mf_gaussian_noise`` consumes ``paired_noise_stddevs`` for σ allocation.

    The strategy norms enter as multipliers on the per-record bounds:
    ``Δ¹ = ζ · ‖C₁‖``, ``Δ² = ζ² · ‖C₂‖``.  These tests pin the closed
    form on representative inputs.
    """

    def test_paired_stddevs_with_strategy_norms(self):
        # Δ¹ = ζ · c1, Δ² = ζ² · c2.
        zeta, c1, c2 = 0.2, 2.0, 1.5
        nm = 3.0
        delta1 = zeta * c1
        delta2 = (zeta**2) * c2
        s_first, s_second = paired_noise_stddevs(nm, first=delta1, second=delta2)
        s_total = delta1 + delta2
        assert s_first == pytest.approx(nm * math.sqrt(delta1 * s_total))
        assert s_second == pytest.approx(nm * math.sqrt(delta2 * s_total))

    def test_mahalanobis_equality(self):
        zeta, c1, c2, nm = 0.5, 2.0, 1.0, 1.0
        delta1 = zeta * c1
        delta2 = (zeta**2) * c2
        s_first, s_second = paired_noise_stddevs(nm, first=delta1, second=delta2)
        mahal = (delta1 / s_first) ** 2 + (delta2 / s_second) ** 2
        assert mahal == pytest.approx(1.0 / nm**2, rel=1e-12)

    def test_squared_max_norm_couples_both_streams(self):
        """Increasing ``squared_max_norm`` shifts both σ's via S = Δ¹+Δ²."""
        a_first, a_second = paired_noise_stddevs(1.0, first=0.1, second=0.01)
        b_first, b_second = paired_noise_stddevs(1.0, first=0.1, second=0.04)
        assert b_first > a_first
        assert b_second > a_second

    def test_rejects_invalid(self):
        with pytest.raises(
            ConfigurationError, match="noise_multiplier must be non-negative"
        ):
            paired_noise_stddevs(-1.0, first=0.1, second=0.01)
        with pytest.raises(ValueError, match="first must be non-negative"):
            paired_noise_stddevs(1.0, first=-0.1, second=0.01)
        with pytest.raises(ValueError, match="second must be non-negative"):
            paired_noise_stddevs(1.0, first=0.1, second=-0.01)


class TestSecondMomentMFNoiseMatchesMfGaussianPld:
    """Joint paired Mahalanobis on encoded streams equals ``(‖C₁‖ / nm)²``.

    The dispatcher must translate the calibrated ``MfGaussian(nm, ‖C₁‖)``
    parameter into the appropriate ``paired_noise_stddevs`` effective
    multiplier (``nm / ‖C₁‖``) so the joint paired release has the same
    PLD ``gaussian_pld(nm / ‖C₁‖)`` as the single-stream MF release.

    Without the translation the runtime over-noises by a factor of
    ``‖C₁‖`` per stream — privacy is strictly stricter than the
    calibration target but utility suffers.
    """

    @pytest.fixture
    def grad_template(self):
        return {"w": torch.zeros(4, 3), "b": torch.zeros(4)}

    @pytest.mark.parametrize("nm", [0.5, 1.0, 2.0])
    def test_joint_mahalanobis_matches_mf_gaussian_pld(self, grad_template, nm):
        # BLT has ‖C₁‖ ≠ 1 robustly across platforms.  BandMF column-
        # normalises to ‖C‖ ≈ 1 (1.0 - O(eps) on x86 Linux but exactly 1.0
        # on Apple Silicon), so it's a poor regression target for the
        # ``nm / c1`` scaling fix this test guards.
        strategy = blt_strategy(momentum=0.9)
        second_strategy = blt_strategy(momentum=0.99)
        c1 = _max_column_norm(strategy, n_steps=50)
        c2 = _max_column_norm(second_strategy, n_steps=50)

        noise_fn, state = mf_gaussian_noise(
            grad_template,
            strategy,
            n_steps=50,
            min_sep=50,
            max_participations=1,
            noise_multiplier=nm,
            key=key(42),
            second_moment_strategy=second_strategy,
        )
        grads = {"w": torch.zeros(4, 3), "b": torch.zeros(4)}
        out, _ = noise_fn(_paired(grads), state)

        # ``noise_stddev`` is realized σ (= base · row_l2); recover base
        # σ to check the joint-PLD calibration identity.
        first_row_l2 = _row_l2_at_zero(strategy, n_steps=50, min_sep=50)
        second_row_l2 = _row_l2_at_zero(second_strategy, n_steps=50, min_sep=50)
        base_sigma_first = out.noisy_grads.noise_stddev / first_row_l2
        base_sigma_second = out.noisy_squared_grads.noise_stddev / second_row_l2
        # Encoded per-record sensitivities.
        delta1 = _SENSITIVITY * c1
        delta2 = (_SENSITIVITY**2) * c2
        mahal = (delta1 / base_sigma_first) ** 2 + (delta2 / base_sigma_second) ** 2

        # Target: the joint PLD must equal MfGaussian(nm, c1) PLD =
        # gaussian_pld(nm / c1), i.e. effective multiplier nm / c1, i.e.
        # joint Mahalanobis = (c1 / nm)².
        expected = (c1 / nm) ** 2
        assert mahal == pytest.approx(expected, rel=1e-10), (
            f"joint Mahalanobis {mahal} != (c1/nm)² = {expected} (c1={c1}, nm={nm})"
        )

    def test_first_stream_recovers_single_stream_in_small_squared_limit(
        self, grad_template
    ):
        """As Δ² / Δ¹ → 0, σ_first → single-stream MF σ = nm·ζ."""
        strategy = blt_strategy(momentum=0.9)
        second_strategy = blt_strategy(momentum=0.99)
        nm = 1.0
        # Build a paired input where the squared-stream sensitivity is
        # effectively negligible relative to the first.
        grads = {"w": torch.zeros(4, 3), "b": torch.zeros(4)}
        small_zeta = 1e-6
        first_clipped = clipped(grads, max_norm=small_zeta)
        sq_pytree = {k: v * v for k, v in grads.items()}
        sq_clipped = clipped(sq_pytree, max_norm=small_zeta * small_zeta)
        paired = SecondMomentClippingOutput(
            grads=first_clipped, squared_grads=sq_clipped
        )
        noise_fn, state = mf_gaussian_noise(
            grad_template,
            strategy,
            n_steps=50,
            min_sep=50,
            max_participations=1,
            noise_multiplier=nm,
            key=key(42),
            second_moment_strategy=second_strategy,
        )
        out, _ = noise_fn(paired, state)
        # Single-stream MF runtime σ on the noise tensor for ζ=small_zeta is
        # nm·ζ; the paired σ_first should converge to that as Δ²/Δ¹ → 0.
        # Recover base σ from realized σ (= base · row_l2) for the comparison.
        first_row_l2 = _row_l2_at_zero(strategy, n_steps=50, min_sep=50)
        base_sigma_first = out.noisy_grads.noise_stddev / first_row_l2
        single_sigma = nm * small_zeta
        assert base_sigma_first == pytest.approx(single_sigma, rel=1e-3)


class TestPairedPerGroupMahalanobis:
    """Per-group paired MF: joint encoded Mahalanobis equals ``(c1 / nm)²``."""

    @pytest.fixture
    def grad_template(self):
        return {"w": torch.zeros(3, 2), "b": torch.zeros(3)}

    def test_joint_budget_with_per_group_bounds(self, grad_template):
        strategy = band_mf_strategy(bands=4, momentum=0.9)
        second_strategy = band_mf_strategy(bands=4, momentum=0.99)
        nm = 1.1
        c1 = _max_column_norm(strategy, n_steps=50)
        c2 = _max_column_norm(second_strategy, n_steps=50)
        z = 0.04
        pg = PerGroup(
            groups={"w": "a", "b": "b"},
            values={"a": z, "b": z * 2},
        )
        sq = {k: v * v for k, v in {"w": torch.ones(3, 2), "b": torch.ones(3)}.items()}
        sq_pg = pg * pg
        paired = SecondMomentClippingOutput(
            grads=clipped({"w": torch.ones(3, 2), "b": torch.ones(3)}, max_norm=pg),
            squared_grads=clipped(sq, max_norm=sq_pg),
        )
        noise_fn, state = mf_gaussian_noise(
            grad_template,
            strategy,
            n_steps=50,
            min_sep=50,
            max_participations=1,
            noise_multiplier=nm,
            key=key(123),
            second_moment_strategy=second_strategy,
        )
        out, _ = noise_fn(paired, state)
        s1 = out.noisy_grads.noise_stddev
        s2 = out.noisy_squared_grads.noise_stddev
        assert isinstance(s1, PerGroup)
        assert isinstance(s2, PerGroup)
        # Recover base σ from realized σ (= base · row_l2) per stream.
        first_row_l2 = _row_l2_at_zero(strategy, n_steps=50, min_sep=50)
        second_row_l2 = _row_l2_at_zero(second_strategy, n_steps=50, min_sep=50)
        mahal = 0.0
        for param_key in ("w", "b"):
            d1 = pg.for_path(param_key) * c1
            d2 = sq_pg.for_path(param_key) * c2
            base_s1 = s1.for_path(param_key) / first_row_l2
            base_s2 = s2.for_path(param_key) / second_row_l2
            mahal += (d1 / base_s1) ** 2 + (d2 / base_s2) ** 2
        assert mahal == pytest.approx((c1 / nm) ** 2, rel=1e-9)


class TestSecondMomentMFNoise:
    @pytest.fixture
    def grad_template(self):
        return {"w": torch.zeros(4, 3), "b": torch.zeros(4)}

    def test_returns_correct_types(self, grad_template):
        strategy = band_mf_strategy(bands=5, momentum=0.9)
        second_strategy = band_mf_strategy(bands=5, momentum=0.99)
        noise_fn, state = mf_gaussian_noise(
            grad_template,
            strategy,
            n_steps=50,
            min_sep=50,
            max_participations=1,
            noise_multiplier=1.0,
            key=key(42),
            second_moment_strategy=second_strategy,
        )
        assert isinstance(state, SecondMomentMFNoiseState)

        grads = {"w": torch.randn(4, 3), "b": torch.randn(4)}
        output, new_state = noise_fn(_paired(grads), state)
        assert isinstance(output, SecondMomentNoiseOutput)
        assert isinstance(output.noisy_grads, NoisedPytree)
        assert isinstance(output.noisy_squared_grads, NoisedPytree)
        assert isinstance(new_state, SecondMomentMFNoiseState)

    def test_output_shapes_match_input(self, grad_template):
        strategy = band_mf_strategy(bands=5, momentum=0.9)
        second_strategy = band_mf_strategy(bands=5, momentum=0.99)
        noise_fn, state = mf_gaussian_noise(
            grad_template,
            strategy,
            n_steps=50,
            min_sep=50,
            max_participations=1,
            noise_multiplier=1.0,
            key=key(42),
            second_moment_strategy=second_strategy,
        )
        grads = {"w": torch.randn(4, 3), "b": torch.randn(4)}
        output, state = noise_fn(_paired(grads), state)
        assert output.noisy_grads.pytree["w"].shape == (4, 3)
        assert output.noisy_grads.pytree["b"].shape == (4,)
        assert output.noisy_squared_grads.pytree["w"].shape == (4, 3)
        assert output.noisy_squared_grads.pytree["b"].shape == (4,)

    def test_tuple_unpacking(self, grad_template):
        strategy = band_mf_strategy(bands=5, momentum=0.9)
        second_strategy = band_mf_strategy(bands=5, momentum=0.99)
        noise_fn, state = mf_gaussian_noise(
            grad_template,
            strategy,
            n_steps=50,
            min_sep=50,
            max_participations=1,
            noise_multiplier=1.0,
            key=key(42),
            second_moment_strategy=second_strategy,
        )
        grads = {"w": torch.randn(4, 3), "b": torch.randn(4)}
        noisy_g, noisy_sq = noise_fn(_paired(grads), state)[0]
        assert isinstance(noisy_g, NoisedPytree)
        assert isinstance(noisy_sq, NoisedPytree)

    def test_step_counter_increments(self, grad_template):
        strategy = band_mf_strategy(bands=5, momentum=0.9)
        second_strategy = band_mf_strategy(bands=5, momentum=0.99)
        noise_fn, state = mf_gaussian_noise(
            grad_template,
            strategy,
            n_steps=50,
            min_sep=50,
            max_participations=1,
            noise_multiplier=1.0,
            key=key(42),
            second_moment_strategy=second_strategy,
        )
        assert state._step_counter == 0
        grads = {"w": torch.randn(4, 3), "b": torch.randn(4)}
        _, state = noise_fn(_paired(grads), state)
        assert state._step_counter == 1
        _, state = noise_fn(_paired(grads), state)
        assert state._step_counter == 2

    def test_bisr_streams_remain_bounded_and_resume_exactly(self, grad_template):
        """Both paired BISR windows checkpoint independently past ring fill."""
        n_steps = 9
        first = bisr_strategy(bandwidth=4, momentum=0.9)
        second = bisr_strategy(bandwidth=3, momentum=0.99)
        noise_fn, state = mf_gaussian_noise(
            grad_template,
            first,
            n_steps=n_steps,
            noise_multiplier=0.8,
            key=key(795),
            second_moment_strategy=second,
            compute_dtype=torch.float64,
        )
        paired = _paired({"w": torch.zeros(4, 3), "b": torch.zeros(4)})

        for _ in range(5):
            _, state = noise_fn(paired, state)

        first_inner = state._first_state._inner_state
        second_inner = state._second_state._inner_state
        assert state._first_state._rng_key != state._second_state._rng_key
        assert len(first_inner.history) == 3
        assert len(second_inner.history) == 2
        assert first_inner.history is not second_inner.history
        assert first_inner.history[0] is not second_inner.history[0]
        assert any(
            not torch.equal(first_value, second_value)
            for first_value, second_value in zip(
                tree_leaves(first_inner.history[0]),
                tree_leaves(second_inner.history[0]),
                strict=True,
            )
        )

        saved = state_dict(state)
        assert saved["_first_state._inner_state.layout_version"] == 1
        assert saved["_second_state._inner_state.layout_version"] == 1

        resumed_fn, resumed_template = mf_gaussian_noise(
            grad_template,
            first,
            n_steps=n_steps,
            noise_multiplier=0.8,
            key=key(999),
            second_moment_strategy=second,
            compute_dtype=torch.float64,
        )
        resumed_state = from_state_dict(resumed_template, saved)

        for _ in range(5, n_steps):
            expected, state = noise_fn(paired, state)
            actual, resumed_state = resumed_fn(paired, resumed_state)
            expected_leaves = tree_leaves(expected.noisy_grads.pytree)
            actual_leaves = tree_leaves(actual.noisy_grads.pytree)
            for actual_leaf, expected_leaf in zip(
                actual_leaves, expected_leaves, strict=True
            ):
                torch.testing.assert_close(
                    actual_leaf, expected_leaf, atol=0.0, rtol=0.0
                )
            expected_sq_leaves = tree_leaves(expected.noisy_squared_grads.pytree)
            actual_sq_leaves = tree_leaves(actual.noisy_squared_grads.pytree)
            for actual_leaf, expected_leaf in zip(
                actual_sq_leaves, expected_sq_leaves, strict=True
            ):
                torch.testing.assert_close(
                    actual_leaf, expected_leaf, atol=0.0, rtol=0.0
                )

    def test_deterministic_with_same_key(self, grad_template):
        strategy = band_mf_strategy(bands=5, momentum=0.9)
        second_strategy = band_mf_strategy(bands=5, momentum=0.99)
        grads = {"w": torch.randn(4, 3), "b": torch.randn(4)}

        noise_fn1, state1 = mf_gaussian_noise(
            grad_template,
            strategy,
            n_steps=50,
            min_sep=50,
            max_participations=1,
            noise_multiplier=1.0,
            key=key(42),
            second_moment_strategy=second_strategy,
        )
        output1, _ = noise_fn1(_paired(grads), state1)

        noise_fn2, state2 = mf_gaussian_noise(
            grad_template,
            strategy,
            n_steps=50,
            min_sep=50,
            max_participations=1,
            noise_multiplier=1.0,
            key=key(42),
            second_moment_strategy=second_strategy,
        )
        output2, _ = noise_fn2(_paired(grads), state2)

        torch.testing.assert_close(
            output1.noisy_grads.pytree["w"], output2.noisy_grads.pytree["w"]
        )
        torch.testing.assert_close(
            output1.noisy_squared_grads.pytree["w"],
            output2.noisy_squared_grads.pytree["w"],
        )

    def test_different_keys_give_different_noise(self, grad_template):
        strategy = band_mf_strategy(bands=5, momentum=0.9)
        second_strategy = band_mf_strategy(bands=5, momentum=0.99)
        grads = {"w": torch.randn(4, 3), "b": torch.randn(4)}

        noise_fn1, state1 = mf_gaussian_noise(
            grad_template,
            strategy,
            n_steps=50,
            min_sep=50,
            max_participations=1,
            noise_multiplier=1.0,
            key=key(42),
            second_moment_strategy=second_strategy,
        )
        output1, _ = noise_fn1(_paired(grads), state1)

        noise_fn2, state2 = mf_gaussian_noise(
            grad_template,
            strategy,
            n_steps=50,
            min_sep=50,
            max_participations=1,
            noise_multiplier=1.0,
            key=key(99),
            second_moment_strategy=second_strategy,
        )
        output2, _ = noise_fn2(_paired(grads), state2)

        assert not torch.allclose(
            output1.noisy_grads.pytree["w"], output2.noisy_grads.pytree["w"]
        )

    @pytest.mark.parametrize("mechanism", ["band_mf", "blt", "bisr", "bsr", "identity"])
    def test_works_with_supported_mechanisms(self, grad_template, mechanism):
        if mechanism == "band_mf":
            strategy = band_mf_strategy(bands=5, momentum=0.9)
            second_strategy = band_mf_strategy(bands=5, momentum=0.99)
        elif mechanism == "blt":
            strategy = blt_strategy(momentum=0.9)
            second_strategy = blt_strategy(momentum=0.99)
        elif mechanism == "bisr":
            strategy = bisr_strategy(bandwidth=4, momentum=0.9)
            second_strategy = bisr_strategy(bandwidth=4, momentum=0.99)
        elif mechanism == "bsr":
            strategy = bsr_strategy(bandwidth=4, alpha=1.0, beta=0.9)
            second_strategy = bsr_strategy(bandwidth=4, alpha=1.0, beta=0.99)
        elif mechanism == "identity":
            strategy = identity_strategy()
            second_strategy = identity_strategy()

        noise_fn, state = mf_gaussian_noise(
            grad_template,
            strategy,
            n_steps=50,
            min_sep=50,
            max_participations=1,
            noise_multiplier=1.0,
            key=key(42),
            second_moment_strategy=second_strategy,
        )
        grads = {"w": torch.randn(4, 3), "b": torch.randn(4)}
        output, new_state = noise_fn(_paired(grads), state)
        assert output.noisy_grads.pytree["w"].shape == (4, 3)
        assert output.noisy_squared_grads.pytree["w"].shape == (4, 3)
        assert isinstance(new_state, SecondMomentMFNoiseState)

    def test_paired_input_requires_second_moment_strategy(self, grad_template):
        """Single-stream mf_gaussian_noise rejects paired-stream input."""
        strategy = lambda_cgd_strategy(lambda_=0.9)
        noise_fn, state = mf_gaussian_noise(
            grad_template,
            strategy,
            n_steps=50,
            min_sep=50,
            max_participations=1,
            noise_multiplier=1.0,
            key=key(42),
        )
        grads = {"w": torch.randn(4, 3), "b": torch.randn(4)}
        with pytest.raises(TypeError, match="second_moment_strategy"):
            noise_fn(_paired(grads), state)

    def test_single_input_rejected_when_second_moment_strategy_supplied(
        self, grad_template
    ):
        """Paired-stream mf_gaussian_noise rejects single-stream input."""
        strategy = lambda_cgd_strategy(lambda_=0.9)
        second_strategy = lambda_cgd_strategy(lambda_=0.999)
        noise_fn, state = mf_gaussian_noise(
            grad_template,
            strategy,
            n_steps=50,
            min_sep=50,
            max_participations=1,
            noise_multiplier=1.0,
            key=key(42),
            second_moment_strategy=second_strategy,
        )
        grads = {"w": torch.randn(4, 3), "b": torch.randn(4)}
        with pytest.raises(TypeError, match="SecondMomentClippingOutput"):
            noise_fn(_clipped(grads), state)

    def test_lambda_cgd_accepts_explicit_second_strategy(self, grad_template):
        strategy = lambda_cgd_strategy(lambda_=0.9)
        second_strategy = lambda_cgd_strategy(lambda_=0.999)
        noise_fn, state = mf_gaussian_noise(
            grad_template,
            strategy,
            n_steps=50,
            min_sep=50,
            max_participations=1,
            noise_multiplier=1.0,
            key=key(42),
            second_moment_strategy=second_strategy,
        )
        grads = {"w": torch.randn(4, 3), "b": torch.randn(4)}
        output, _ = noise_fn(_paired(grads), state)
        assert output.noisy_squared_grads.pytree["w"].shape == (4, 3)

    def test_squared_grads_are_noised_not_raw(self, grad_template):
        strategy = band_mf_strategy(bands=5, momentum=0.9)
        second_strategy = band_mf_strategy(bands=5, momentum=0.99)
        noise_fn, state = mf_gaussian_noise(
            grad_template,
            strategy,
            n_steps=50,
            min_sep=50,
            max_participations=1,
            noise_multiplier=1.0,
            key=key(42),
            second_moment_strategy=second_strategy,
        )
        grads = {"w": torch.ones(4, 3), "b": torch.ones(4)}
        output, _ = noise_fn(_paired(grads), state)
        raw_sq = grads["w"] ** 2
        assert not torch.allclose(
            output.noisy_squared_grads.pytree["w"], raw_sq, atol=1e-6
        )


def test_paired_state_has_a_registered_sync_handler():
    """``sync`` dispatches on type, and the paired state is not an ``MFNoiseState``.

    :class:`SecondMomentMFNoiseState` inherits from ``NoiseState``, so the MRO
    walk in :func:`opake.distributed.sync` does not find the single-stream
    handler and this type needs a registration of its own.  The documented
    distributed loop calls ``sync(noise_state)``, so an unregistered paired
    state makes every private-second-moment run raise ``TypeError``.  Dispatch
    happens whether or not a process group exists, which is why this needs no
    ranks.
    """
    from opake.distributed import sync

    _, state = mf_gaussian_noise(
        {"w": torch.zeros(4, 3), "b": torch.zeros(4)},
        identity_strategy(),
        n_steps=2,
        noise_multiplier=1.0,
        key=key(7),
        second_moment_strategy=identity_strategy(),
    )
    assert isinstance(state, SecondMomentMFNoiseState)
    assert sync(state) is state
