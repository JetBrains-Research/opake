"""Integration tests for AUTO-S clipping wired into ``mf_gaussian_noise``.

AUTO-S has a constant, data-independent per-record sensitivity bound::

    sup_g  || R · g / (||g|| + γ) || <= R    for any g.

That satisfies the only assumption MF privacy accounting requires of
clipping (constant per-step record sensitivity), so the same
``ClippedPytree`` output that flows into DP-SGD's :func:`gaussian_noise`
also flows into DP-FTRL's :func:`mf_gaussian_noise` for every strategy.  These
tests cover all six MF strategies — ``identity``, ``band_mf``, ``blt``,
``bisr``, ``bsr``, ``lambda_cgd`` — across:

1. **Scalar AUTO-S × MF** — the dispatcher's
   ``_validate_constant_max_norm`` latch passes across many steps and
   ``noise_stddev = nm · R / batch_size``.
2. **AUTO-S vs fixed-clipping equivalence** — at the same ``R`` both
   ``auto_clipped_grad`` and ``clipped_grad`` deliver the same
   ``ClippedPytree.max_norm`` and the same downstream calibration.
3. **Per-record sensitivity bound, end-to-end** — gradients pumped to
   1e6 still leave the released signal bounded by ``R / batch_size``.
4. **Per-group AUTO-S × MF** — once :class:`PerGroup` bounds are
   accepted by ``mf_gaussian_noise`` (PR #192), AUTO-S delivers a constant
   per-group bound that flows through the per-group noise allocator
   identically to fixed clipping.
5. **Second-moment AUTO-S × paired MF** — ``auto_clipped_grad(...,
   second_moment=True)`` returns a :class:`SecondMomentClippingOutput`
   with constant ``R / B`` and ``R² / B`` bounds; ``mf_gaussian_noise(...,
   second_moment_strategy=...)`` consumes it and the paired
   Mahalanobis allocation matches the equivalent fixed-clipping path.
6. **Negative regression guard** — adaptive clipping × MF still raises
   the constant-max_norm latch error, locking the only scenario the
   library legitimately rejects.

The (4) per-group cases are gated behind a runtime check (``mf_gaussian_noise``
on plain ``main`` rejects ``PerGroup`` bounds; once PR #192 lands the
gate falls open and the asserts run for real).
"""

from __future__ import annotations

import math

import pytest
import torch

from opake.dpftrl.clipping import auto_clipped_grad, clipped_grad
from opake.dpftrl.noise import (
    band_mf_strategy,
    bisr_strategy,
    blt_strategy,
    bsr_strategy,
    identity_strategy,
    lambda_cgd_strategy,
    mf_gaussian_noise,
)
from opake.dpsgd.clipping import adaptive_clipped_grad
from opake.random import key
from opake.types import (
    ClippedPytree,
    NoisedPytree,
    PerGroup,
    SecondMomentClippingOutput,
    SecondMomentNoiseOutput,
    clipped,
)

N_STEPS = 16
BATCH_SIZE = 8
N_FEATURES = 4
R = 0.5
NOISE_MULTIPLIER = 1.1


# ------------------------------------------------------------------ helpers


def _scalar_loss_fn(params, x, y):
    return ((x @ params - y) ** 2).mean()


def _per_group_loss_fn(params, x):
    return ((params["w"] * x).sum(dim=-1) - params["b"].sum()).pow(2).mean()


def _make_strategy(name: str):
    """Construct each MF strategy recipe (horizon is supplied at use time)."""
    if name == "identity":
        return identity_strategy()
    if name == "band_mf":
        return band_mf_strategy(bands=4, momentum=0.95)
    if name == "blt":
        return blt_strategy(max_buffers=4)
    if name == "bisr":
        return bisr_strategy(bandwidth=4)
    if name == "bsr":
        return bsr_strategy(bandwidth=4, alpha=1.0, beta=0.5)
    if name == "lambda_cgd":
        return lambda_cgd_strategy(lambda_=0.5)
    raise AssertionError(f"unknown strategy {name}")


_NOISE_PART = {"n_steps": N_STEPS, "min_sep": 1, "max_participations": 1}


def _row_l2_at_zero(strategy, *, n_steps, min_sep=1, max_participations=None) -> float:
    """First-step ``‖row_0(C^-1)‖`` — used to back out base σ from the
    realized σ that :class:`NoisedPytree.noise_stddev` publishes.

    Identity / λ-CGD step-0 / λ=0 cases all return 1.  λ-CGD has a
    PRNG-replay path (no ``streaming_matrix``); we synthesize the
    expected factor from the strategy's closed-form column norm.
    """
    from opake.dpftrl.noise.types import IdentityStrategy, LambdaCgdStrategy

    if isinstance(strategy, IdentityStrategy):
        return 1.0
    if isinstance(strategy, LambdaCgdStrategy):
        from opake.api.dpftrl.noise._lambda_cgd import _column_norm

        col = _column_norm(strategy.lambda_, n_steps, 0) if strategy.normalized else 1.0
        return col  # step-0 short-circuit: no z_{t-1} term so no sqrt(1+λ²) factor
    streaming = strategy.streaming_matrix(
        n_steps=n_steps, min_sep=min_sep, max_participations=max_participations
    )
    return float(streaming.row_norms_squared(n_steps).clamp_min(0.0).sqrt()[0])


_ALL_STRATEGY_NAMES = ("identity", "band_mf", "blt", "bisr", "bsr", "lambda_cgd")


def _per_group_supported_by_mf_noise() -> bool:
    """Probe whether the dispatcher accepts ``PerGroup`` bounds.

    On plain ``main`` ``mf_gaussian_noise`` rejects ``PerGroup`` bounds because the
    dispatcher only allows a constant per-step ``max_norm`` shape that older
    releases did not support for MF noise.  PR #192 removes that rejection.
    These tests
    light up automatically once #192 lands; until then they are skipped
    rather than failing the suite.
    """
    pg = PerGroup(groups={"w": "g"}, values={"g": 1.0})
    grads = clipped({"w": torch.zeros(2, 2)}, max_norm=pg)
    template = {"w": torch.zeros(2, 2)}
    noise_fn, state = mf_gaussian_noise(
        template, identity_strategy(), **_NOISE_PART, noise_multiplier=1.0, key=key(0)
    )
    try:
        noise_fn(grads, state)
    except TypeError as exc:
        if "PerGroup" in str(exc):
            return False
        raise
    return True


# ------------------------------------------------------------------ scalar


@pytest.mark.parametrize("strategy_name", _ALL_STRATEGY_NAMES)
class TestScalarAutoSxMf:
    """Scalar AUTO-S × ``mf_gaussian_noise`` for every MF strategy."""

    def test_constant_max_norm_latch_passes(self, strategy_name):
        """``_validate_constant_max_norm`` never fires across many steps.

        Under MF noise the per-step *realized* σ on ``NoisedPytree`` is
        ``base_σ · ‖row_t(C^-1)‖`` and varies by step (that's the whole
        point of MF — and why Adam BC needs it).  This test only pins
        the latch contract: ``max_norm`` stays constant across calls so
        the dispatcher never raises.  Per-step σ variation is exercised
        by ``packages/opake-dpftrl/tests/noise/test_realized_stddev.py``.
        """
        torch.manual_seed(0)
        params = torch.randn(N_FEATURES)
        grad_fn, clip_state = auto_clipped_grad(
            _scalar_loss_fn,
            argnums=0,
            batch_argnums=(1, 2),
            R=R,
            normalize_by=BATCH_SIZE,
        )
        noise_fn, noise_state = mf_gaussian_noise(
            params,
            _make_strategy(strategy_name),
            **_NOISE_PART,
            noise_multiplier=NOISE_MULTIPLIER,
            key=key(7),
        )
        seen_max_norms: list[float] = []
        for step in range(N_STEPS):
            scale = 100.0 if step % 2 == 0 else 0.001
            x = torch.randn(BATCH_SIZE, N_FEATURES) * scale
            y = torch.randn(BATCH_SIZE) * scale
            grads, clip_state = grad_fn(params, x, y, state=clip_state)
            assert isinstance(grads, ClippedPytree)
            seen_max_norms.append(float(grads.max_norm))
            noised, noise_state = noise_fn(grads, noise_state)
            assert isinstance(noised, NoisedPytree)
            assert float(noised.noise_stddev) > 0
        assert all(m == seen_max_norms[0] for m in seen_max_norms)

    def test_max_norm_and_stddev_match_calibration(self, strategy_name):
        """Base σ = ``noise_multiplier · R / batch_size``; realized σ on
        ``NoisedPytree.noise_stddev`` is ``base_σ · ‖row_0(C^-1)‖`` —
        recover base σ by dividing out the row-L2 factor.  For Identity
        and step-0 λ-CGD the factor is 1 so realized == base; for the
        correlated MF strategies it differs."""
        torch.manual_seed(0)
        params = torch.randn(N_FEATURES)
        grad_fn, clip_state = auto_clipped_grad(
            _scalar_loss_fn,
            argnums=0,
            batch_argnums=(1, 2),
            R=R,
            normalize_by=BATCH_SIZE,
        )
        strategy = _make_strategy(strategy_name)
        noise_fn, noise_state = mf_gaussian_noise(
            params,
            strategy,
            **_NOISE_PART,
            noise_multiplier=NOISE_MULTIPLIER,
            key=key(11),
        )
        x = torch.randn(BATCH_SIZE, N_FEATURES)
        y = torch.randn(BATCH_SIZE)
        grads, _ = grad_fn(params, x, y, state=clip_state)
        noised, _ = noise_fn(grads, noise_state)
        expected_max_norm = R / BATCH_SIZE
        expected_base_sigma = NOISE_MULTIPLIER * expected_max_norm
        assert float(grads.max_norm) == pytest.approx(expected_max_norm, abs=1e-9)
        assert float(noised.max_norm) == pytest.approx(expected_max_norm, abs=1e-9)
        # Compute the strategy's row-0 L2 factor analytically and back
        # out base σ from realized σ for the calibration check.
        row_l2 = _row_l2_at_zero(strategy, **_NOISE_PART)
        base_sigma = float(noised.noise_stddev) / row_l2
        assert base_sigma == pytest.approx(expected_base_sigma, abs=1e-9)

    def test_equivalent_to_fixed_clipping_at_same_R(self, strategy_name):
        """AUTO-S(R) and ``clipped_grad(C=R)`` deliver identical calibration."""
        torch.manual_seed(0)
        params = torch.randn(N_FEATURES)
        auto_fn, auto_state = auto_clipped_grad(
            _scalar_loss_fn,
            argnums=0,
            batch_argnums=(1, 2),
            R=R,
            normalize_by=BATCH_SIZE,
        )
        fixed_fn, fixed_state = clipped_grad(
            _scalar_loss_fn,
            argnums=0,
            batch_argnums=(1, 2),
            clipping_norm=R,
            normalize_by=BATCH_SIZE,
        )
        auto_noise_fn, auto_noise_state = mf_gaussian_noise(
            params,
            _make_strategy(strategy_name),
            **_NOISE_PART,
            noise_multiplier=NOISE_MULTIPLIER,
            key=key(13),
        )
        fixed_noise_fn, fixed_noise_state = mf_gaussian_noise(
            params,
            _make_strategy(strategy_name),
            **_NOISE_PART,
            noise_multiplier=NOISE_MULTIPLIER,
            key=key(13),
        )
        for _ in range(N_STEPS):
            x = torch.randn(BATCH_SIZE, N_FEATURES) * 5.0
            y = torch.randn(BATCH_SIZE) * 5.0
            ga, auto_state = auto_fn(params, x, y, state=auto_state)
            gf, fixed_state = fixed_fn(params, x, y, state=fixed_state)
            assert float(ga.max_norm) == pytest.approx(float(gf.max_norm), abs=1e-9)
            na, auto_noise_state = auto_noise_fn(ga, auto_noise_state)
            nf, fixed_noise_state = fixed_noise_fn(gf, fixed_noise_state)
            assert float(na.noise_stddev) == pytest.approx(
                float(nf.noise_stddev), abs=1e-9
            )

    def test_signal_norm_is_capped_by_R(self, strategy_name):
        """Even with gradients ×1e6, the released *signal* is capped by R/B.

        AUTO-S guarantees ``||tilde g_i|| <= R`` per example, hence
        ``||sum_i tilde g_i|| <= B·R`` and ``||clipped_grads.pytree|| <=
        R / normalize_by · batch_size``.  This test confirms the
        end-to-end clipped pytree (the signal feeding ``mf_gaussian_noise``)
        respects that bound regardless of input gradient magnitude.
        """
        torch.manual_seed(0)
        params = torch.randn(N_FEATURES)
        grad_fn, clip_state = auto_clipped_grad(
            _scalar_loss_fn,
            argnums=0,
            batch_argnums=(1, 2),
            R=R,
            normalize_by=BATCH_SIZE,
        )
        noise_fn, noise_state = mf_gaussian_noise(
            params,
            _make_strategy(strategy_name),
            **_NOISE_PART,
            noise_multiplier=NOISE_MULTIPLIER,
            key=key(17),
        )
        for _ in range(N_STEPS):
            x = torch.randn(BATCH_SIZE, N_FEATURES) * 1e6
            y = torch.randn(BATCH_SIZE) * 1e6
            grads, clip_state = grad_fn(params, x, y, state=clip_state)
            signal_norm = float(torch.linalg.vector_norm(grads.pytree))
            cap = (R / BATCH_SIZE) * BATCH_SIZE  # i.e. R
            assert signal_norm <= cap + 1e-5
            _, noise_state = noise_fn(grads, noise_state)


# ------------------------------------------------------------------ per-group


def _make_per_group() -> PerGroup:
    return PerGroup(groups={"w": "attn", "b": "mlp"}, values={"attn": 0.4, "mlp": 1.0})


@pytest.mark.parametrize("strategy_name", _ALL_STRATEGY_NAMES)
class TestPerGroupAutoSxMf:
    """Per-group AUTO-S × ``mf_gaussian_noise``.

    Skipped on plain ``main`` because ``mf_gaussian_noise`` rejects per-group
    bounds via ``_expect_clipped``; PR #192 removes that gate.
    """

    @pytest.fixture(autouse=True)
    def _gate(self):
        if not _per_group_supported_by_mf_noise():
            pytest.skip(
                "mf_gaussian_noise does not yet accept PerGroup bounds; "
                "test pending PR #192 (per-group MF noise)."
            )

    def test_per_group_constant_max_norm_latch_passes(self, strategy_name):
        torch.manual_seed(0)
        params = {"w": torch.randn(N_FEATURES), "b": torch.randn(2)}
        pg = _make_per_group()
        grad_fn, clip_state = auto_clipped_grad(
            _per_group_loss_fn,
            argnums=0,
            batch_argnums=1,
            R=pg,
            normalize_by=BATCH_SIZE,
        )
        noise_fn, noise_state = mf_gaussian_noise(
            params,
            _make_strategy(strategy_name),
            **_NOISE_PART,
            noise_multiplier=NOISE_MULTIPLIER,
            key=key(7),
        )
        for _ in range(N_STEPS):
            x = torch.randn(BATCH_SIZE, N_FEATURES) * 5.0
            grads, clip_state = grad_fn(params, x, state=clip_state)
            assert isinstance(grads, ClippedPytree)
            assert isinstance(grads.max_norm, PerGroup)
            noised, noise_state = noise_fn(grads, noise_state)
            assert isinstance(noised, NoisedPytree)
            assert isinstance(noised.noise_stddev, PerGroup)

    def test_per_group_equivalence_with_fixed_clipping(self, strategy_name):
        """At the same ``PerGroup R``, AUTO-S and fixed clipping match calibration."""
        torch.manual_seed(0)
        params = {"w": torch.randn(N_FEATURES), "b": torch.randn(2)}
        pg = _make_per_group()
        auto_fn, auto_state = auto_clipped_grad(
            _per_group_loss_fn,
            argnums=0,
            batch_argnums=1,
            R=pg,
            normalize_by=BATCH_SIZE,
        )
        fixed_fn, fixed_state = clipped_grad(
            _per_group_loss_fn,
            argnums=0,
            batch_argnums=1,
            clipping_norm=pg,
            normalize_by=BATCH_SIZE,
        )
        auto_noise_fn, auto_noise_state = mf_gaussian_noise(
            params,
            _make_strategy(strategy_name),
            **_NOISE_PART,
            noise_multiplier=NOISE_MULTIPLIER,
            key=key(13),
        )
        fixed_noise_fn, fixed_noise_state = mf_gaussian_noise(
            params,
            _make_strategy(strategy_name),
            **_NOISE_PART,
            noise_multiplier=NOISE_MULTIPLIER,
            key=key(13),
        )
        x = torch.randn(BATCH_SIZE, N_FEATURES) * 5.0
        ga, _ = auto_fn(params, x, state=auto_state)
        gf, _ = fixed_fn(params, x, state=fixed_state)
        assert isinstance(ga.max_norm, PerGroup)
        assert isinstance(gf.max_norm, PerGroup)
        assert ga.max_norm.values == pytest.approx(gf.max_norm.values)
        na, _ = auto_noise_fn(ga, auto_noise_state)
        nf, _ = fixed_noise_fn(gf, fixed_noise_state)
        # Noise standard deviations match group-by-group.
        assert isinstance(na.noise_stddev, PerGroup)
        assert isinstance(nf.noise_stddev, PerGroup)
        for g in na.noise_stddev.values:
            assert na.noise_stddev.values[g] == pytest.approx(
                nf.noise_stddev.values[g], abs=1e-9
            )


# ------------------------------------------------------------------ second-moment


@pytest.mark.parametrize("strategy_name", _ALL_STRATEGY_NAMES)
class TestSecondMomentAutoSxMf:
    """Second-moment AUTO-S × paired ``mf_gaussian_noise``.

    AUTO-S with ``second_moment=True`` produces a
    :class:`SecondMomentClippingOutput` with bounds ``R / B`` (first
    moment) and ``R² / B`` (second moment), both data-independent
    constants.  ``mf_gaussian_noise(..., second_moment_strategy=...)`` consumes
    it and applies the joint Mahalanobis allocation; the result must
    agree with the fixed-clipping path at the same ``R``.
    """

    def test_paired_release_runs_end_to_end(self, strategy_name):
        torch.manual_seed(0)
        params = torch.randn(N_FEATURES)
        grad_fn, clip_state = auto_clipped_grad(
            _scalar_loss_fn,
            argnums=0,
            batch_argnums=(1, 2),
            R=R,
            normalize_by=BATCH_SIZE,
            second_moment=True,
        )
        first_strategy = _make_strategy(strategy_name)
        # Avoid reusing the same strategy object's streaming state when the
        # mechanism allocates two independent streams.  ``identity`` and
        # ``lambda_cgd`` are stateless modulo their RNG keys, but for the
        # streaming-matrix strategies we want two independent factories.
        second_strategy = _make_strategy(strategy_name)
        noise_fn, noise_state = mf_gaussian_noise(
            params,
            first_strategy,
            **_NOISE_PART,
            noise_multiplier=NOISE_MULTIPLIER,
            key=key(19),
            second_moment_strategy=second_strategy,
        )
        for _ in range(N_STEPS):
            x = torch.randn(BATCH_SIZE, N_FEATURES) * 5.0
            y = torch.randn(BATCH_SIZE) * 5.0
            grads, clip_state = grad_fn(params, x, y, state=clip_state)
            assert isinstance(grads, SecondMomentClippingOutput)
            out, noise_state = noise_fn(grads, noise_state)
            assert isinstance(out, SecondMomentNoiseOutput)
        assert float(grads.grads.max_norm) == pytest.approx(R / BATCH_SIZE, abs=1e-9)
        assert float(grads.squared_grads.max_norm) == pytest.approx(
            (R * R) / BATCH_SIZE, abs=1e-9
        )

    def test_paired_calibration_matches_fixed_clipping(self, strategy_name):
        """AUTO-S(R) + paired MF == fixed-clipping(C=R) + paired MF."""
        torch.manual_seed(0)
        params = torch.randn(N_FEATURES)
        auto_fn, auto_state = auto_clipped_grad(
            _scalar_loss_fn,
            argnums=0,
            batch_argnums=(1, 2),
            R=R,
            normalize_by=BATCH_SIZE,
            second_moment=True,
        )
        fixed_fn, fixed_state = clipped_grad(
            _scalar_loss_fn,
            argnums=0,
            batch_argnums=(1, 2),
            clipping_norm=R,
            normalize_by=BATCH_SIZE,
            second_moment=True,
        )
        auto_noise_fn, auto_noise_state = mf_gaussian_noise(
            params,
            _make_strategy(strategy_name),
            **_NOISE_PART,
            noise_multiplier=NOISE_MULTIPLIER,
            key=key(23),
            second_moment_strategy=_make_strategy(strategy_name),
        )
        fixed_noise_fn, fixed_noise_state = mf_gaussian_noise(
            params,
            _make_strategy(strategy_name),
            **_NOISE_PART,
            noise_multiplier=NOISE_MULTIPLIER,
            key=key(23),
            second_moment_strategy=_make_strategy(strategy_name),
        )
        x = torch.randn(BATCH_SIZE, N_FEATURES) * 5.0
        y = torch.randn(BATCH_SIZE) * 5.0
        ga, _ = auto_fn(params, x, y, state=auto_state)
        gf, _ = fixed_fn(params, x, y, state=fixed_state)
        # Same per-stream bound.
        assert float(ga.grads.max_norm) == pytest.approx(
            float(gf.grads.max_norm), abs=1e-9
        )
        assert float(ga.squared_grads.max_norm) == pytest.approx(
            float(gf.squared_grads.max_norm), abs=1e-9
        )
        # Same paired Mahalanobis stddevs after the dispatcher.
        na, _ = auto_noise_fn(ga, auto_noise_state)
        nf, _ = fixed_noise_fn(gf, fixed_noise_state)
        assert float(na.noisy_grads.noise_stddev) == pytest.approx(
            float(nf.noisy_grads.noise_stddev), abs=1e-9
        )
        assert float(na.noisy_squared_grads.noise_stddev) == pytest.approx(
            float(nf.noisy_squared_grads.noise_stddev), abs=1e-9
        )


# ------------------------------------------------------------------ negative


class TestAdaptiveClippingRejected:
    """Adaptive × MF must keep raising the constant-max_norm latch error."""

    def test_adaptive_clipping_rejected_by_constant_latch(self):
        torch.manual_seed(0)
        params = torch.randn(N_FEATURES)
        grad_fn, clip_state = adaptive_clipped_grad(
            _scalar_loss_fn,
            argnums=0,
            batch_argnums=(1, 2),
            initial_clipping_norm=R,
            target_quantile=0.5,
            learning_rate=0.5,
            key=key(31),
            normalize_by=BATCH_SIZE,
        )
        noise_fn, noise_state = mf_gaussian_noise(
            params,
            band_mf_strategy(bands=4, momentum=0.95),
            **_NOISE_PART,
            noise_multiplier=NOISE_MULTIPLIER,
            key=key(37),
        )
        # First call: latches the initial threshold; succeeds.
        x = torch.randn(BATCH_SIZE, N_FEATURES) * 100.0
        y = torch.randn(BATCH_SIZE) * 100.0
        grads, clip_state = grad_fn(params, x, y, state=clip_state)
        _, noise_state = noise_fn(grads, noise_state)

        def drift_threshold():
            current_clip_state = clip_state
            current_noise_state = noise_state
            for _ in range(N_STEPS):
                x = torch.randn(BATCH_SIZE, N_FEATURES) * 100.0
                y = torch.randn(BATCH_SIZE) * 100.0
                grads, current_clip_state = grad_fn(
                    params, x, y, state=current_clip_state
                )
                _, current_noise_state = noise_fn(grads, current_noise_state)

        # Subsequent calls drift the threshold; the dispatcher must reject.
        with pytest.raises(ValueError, match="constant per-step sensitivity"):
            drift_threshold()


# ------------------------------------------------------------------ math sanity


class TestAutoSPerExampleCapMath:
    """Document the AUTO-S sensitivity math the rest of the suite relies on.

    These are unit-level checks that the per-example output norm is
    bounded by ``R`` (with the asymptotic supremum approached as
    ``||g|| -> infinity``), which is what justifies the constant
    sensitivity used by all the MF integration tests above.
    """

    def test_per_example_bound_is_R_in_supremum(self):
        from opake.dpftrl.clipping.fun import auto_scale_pytree

        gamma = 0.01
        for scale in (1e-3, 1.0, 1e3, 1e6):
            tensor = torch.randn(N_FEATURES) * scale
            scaled, _ = auto_scale_pytree({"w": tensor}, R=R, gamma=gamma)
            norm = float(torch.linalg.vector_norm(scaled["w"]))
            assert norm <= R + 1e-6
        # As ||g|| -> infty, ||tilde g|| -> R.
        big = torch.tensor([1e9, 0.0])
        scaled, _ = auto_scale_pytree({"w": big}, R=R, gamma=gamma)
        assert float(torch.linalg.vector_norm(scaled["w"])) == pytest.approx(
            R, rel=1e-6
        )

    def test_squared_per_example_bound_is_R_squared(self):
        """The second-moment stream is bounded by ``R²``."""
        from opake.dpftrl.clipping.fun import auto_scale_pytree

        big = torch.tensor([1e9, 0.0])
        scaled, _ = auto_scale_pytree({"w": big}, R=R, gamma=0.01)
        sq = scaled["w"].pow(2)
        sq_norm = float(torch.linalg.vector_norm(sq))
        assert sq_norm <= R * R + 1e-6
        # Triangle: sqrt(sum tilde_g_j^4) <= sum tilde_g_j^2 <= R².
        assert sq_norm <= float((scaled["w"].pow(2)).sum()) + 1e-6
        assert math.isfinite(sq_norm)
