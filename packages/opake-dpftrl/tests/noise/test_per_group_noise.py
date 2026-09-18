"""Tests for mf_gaussian_noise with PerGroup clipping bounds."""

from __future__ import annotations

import math

import pytest
import torch

from opake.api.dpftrl.noise._distributed import (
    fingerprint_per_group_max_norm,
    mf_per_group_sync_fingerprint_for_latch,
)
from opake.api.dpftrl.noise._engine import MFNoiseState, _matrix_factorization_noise
from opake.api.engine.noise_allocation import per_group_noise_stddev
from opake.dpftrl.clipping import clipped_grad
from opake.dpftrl.noise import (
    band_mf_strategy,
    bisr_strategy,
    blt_strategy,
    bsr_strategy,
    identity_strategy,
    lambda_cgd_strategy,
    mf_gaussian_noise,
)
from opake.pytree import tree_leaves
from opake.random import key
from opake.types import NoisedPytree, PerGroup, SecondMomentClippingOutput, clipped


def _make_pg_two_groups() -> PerGroup:
    """w -> attn (B=1), b -> mlp (B=2)."""
    return PerGroup(
        groups={"w": "attn", "b": "mlp"},
        values={"attn": 1.0, "mlp": 2.0},
    )


def test_legacy_unsigned_sync_fingerprint_is_normalized() -> None:
    state = MFNoiseState(
        _inner_state=None,
        _step_counter=1,
        _rng_key=key(5),
        _first_max_norm=1.0,
        _first_max_norm_sync_fingerprint=(1 << 64) - 1,
    )

    assert mf_per_group_sync_fingerprint_for_latch(state, 1.0) == -1


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


def _assert_per_group_stddev_matches_expected(grad_template, *, key_seed: int) -> None:
    # Identity strategy has row L2 ≡ 1, so the realized per-step σ
    # published on NoisedPytree.noise_stddev equals the base σ from
    # ``per_group_noise_stddev(pg, nm)`` exactly.  For correlated MF the
    # realized σ also folds in ``‖row_t(C^-1)‖`` — exercised by the
    # streaming-matrix-aware tests below.  This test pins the per-group
    # base-σ pass-through; Identity isolates that path.
    strategy = identity_strategy()
    nm = 1.5
    noise_fn, state = mf_gaussian_noise(
        grad_template,
        strategy,
        n_steps=20,
        noise_multiplier=nm,
        key=key(key_seed),
    )
    pg = _make_pg_two_groups()
    grads = clipped({"w": torch.randn(4, 3), "b": torch.randn(4)}, max_norm=pg)
    out, _ = noise_fn(grads, state)
    assert isinstance(out, NoisedPytree)
    assert isinstance(out.noise_stddev, PerGroup)
    expected = per_group_noise_stddev(pg, nm)
    assert out.noise_stddev.groups == expected.groups
    for g in expected.values:
        assert out.noise_stddev.values[g] == pytest.approx(expected.values[g])


class TestMfNoisePerGroupSingleStream:
    @pytest.fixture
    def grad_template(self):
        return {"w": torch.zeros(4, 3), "b": torch.zeros(4)}

    def test_returns_per_group_noise_stddev(self, grad_template):
        _assert_per_group_stddev_matches_expected(grad_template, key_seed=7)

    def test_single_stream_per_group_returns_per_group_stddev(self, grad_template):
        """Plan checklist name — same assertions as ``test_returns_per_group_noise_stddev``."""
        _assert_per_group_stddev_matches_expected(grad_template, key_seed=701)

    def test_mlp_group_has_larger_noise_stddev_than_attn(self, grad_template):
        strategy = identity_strategy()
        noise_fn, state = mf_gaussian_noise(
            grad_template,
            strategy,
            n_steps=20,
            noise_multiplier=1.0,
            key=key(42),
        )
        pg = _make_pg_two_groups()
        grads = clipped({"w": torch.zeros(4, 3), "b": torch.zeros(4)}, max_norm=pg)
        out, _ = noise_fn(grads, state)
        assert isinstance(out.noise_stddev, PerGroup)
        s_attn = out.noise_stddev.values["attn"]
        s_mlp = out.noise_stddev.values["mlp"]
        assert s_mlp == pytest.approx(s_attn * math.sqrt(2.0), rel=1e-9)

    def test_constant_max_norm_latch_pergroup_mismatch(self, grad_template):
        strategy = band_mf_strategy(bands=3, momentum=0.9)
        noise_fn, state = mf_gaussian_noise(
            grad_template,
            strategy,
            n_steps=10,
            noise_multiplier=1.0,
            key=key(0),
        )
        pg1 = PerGroup(
            groups={"w": "g", "b": "g"},
            values={"g": 1.0},
        )
        pg2 = PerGroup(
            groups={"w": "g", "b": "g"},
            values={"g": 2.0},
        )
        _, state = noise_fn(
            clipped({"w": torch.zeros(4, 3), "b": torch.zeros(4)}, max_norm=pg1),
            state,
        )
        with pytest.raises(ValueError, match="varying"):
            noise_fn(
                clipped({"w": torch.zeros(4, 3), "b": torch.zeros(4)}, max_norm=pg2),
                state,
            )

    def test_constant_max_norm_latch_kind_mismatch(self, grad_template):
        strategy = band_mf_strategy(bands=3, momentum=0.9)
        noise_fn, state = mf_gaussian_noise(
            grad_template,
            strategy,
            n_steps=10,
            noise_multiplier=1.0,
            key=key(1),
        )
        _, state = noise_fn(
            clipped({"w": torch.zeros(4, 3), "b": torch.zeros(4)}, max_norm=0.5),
            state,
        )
        pg = PerGroup(
            groups={"w": "g", "b": "g"},
            values={"g": 1.0},
        )
        with pytest.raises(ValueError, match="varying"):
            noise_fn(
                clipped({"w": torch.zeros(4, 3), "b": torch.zeros(4)}, max_norm=pg),
                state,
            )

    def test_per_group_non_dict_grads_raises(self, grad_template):
        strategy = identity_strategy()
        noise_fn, state = mf_gaussian_noise(
            grad_template,
            strategy,
            n_steps=5,
            noise_multiplier=1.0,
            key=key(2),
        )
        pg = PerGroup(groups={"w": "g"}, values={"g": 1.0})
        # Tuple grads have paths (0,), (1,) — not ("w",) — so lookup fails.
        bad = clipped((torch.zeros(3), torch.zeros(3)), max_norm=pg)
        with pytest.raises(KeyError):
            noise_fn(bad, state)

    def test_nested_per_group_noise(self):
        """MF PerGroup σ resolves nested ParamPaths via per_group()."""
        from opake.api.engine.clipping._per_group import per_group

        nested_template = {
            "layer": {"w": torch.zeros(4, 3), "b": torch.zeros(4)},
        }
        strategy = identity_strategy()
        noise_fn, state = mf_gaussian_noise(
            nested_template,
            strategy,
            n_steps=5,
            noise_multiplier=1.0,
            key=key(11),
        )
        pg = per_group(nested_template, w=0.5, b=2.0)
        grads = {
            "layer": {
                "w": torch.zeros(4, 3),
                "b": torch.zeros(4),
            },
        }
        out, _ = noise_fn(clipped(grads, max_norm=pg), state)
        assert isinstance(out.noise_stddev, PerGroup)
        assert ("layer", "w") in out.noise_stddev.groups
        assert ("layer", "b") in out.noise_stddev.groups
        assert out.noise_stddev.for_path(("layer", "b")) > out.noise_stddev.for_path(
            ("layer", "w")
        )
        assert not torch.allclose(out.pytree["layer"]["w"], grads["layer"]["w"])

    @pytest.mark.parametrize(
        "make_strategy",
        [
            lambda: band_mf_strategy(bands=3, momentum=0.9),
            lambda: blt_strategy(momentum=0.9),
            lambda: bisr_strategy(bandwidth=3, momentum=0.9),
            lambda: bsr_strategy(bandwidth=4, alpha=1.0, beta=0.9),
            lambda: lambda_cgd_strategy(lambda_=0.85),
            lambda: identity_strategy(),
        ],
    )
    def test_per_group_runs_all_strategies(self, grad_template, make_strategy):
        strategy = make_strategy()
        noise_fn, state = mf_gaussian_noise(
            grad_template,
            strategy,
            n_steps=15,
            min_sep=15,
            max_participations=1,
            noise_multiplier=0.8,
            key=key(99),
        )
        pg = PerGroup(
            groups={"w": "a", "b": "b"},
            values={"a": 0.5, "b": 1.5},
        )
        grads = clipped({"w": torch.randn(4, 3), "b": torch.randn(4)}, max_norm=pg)
        out, _ = noise_fn(grads, state)
        assert isinstance(out.noise_stddev, PerGroup)
        assert not torch.allclose(out.pytree["w"], grads.pytree["w"])

    def test_bisr_per_group_crosses_ring_fill_with_dense_row_stddevs(
        self, grad_template
    ):
        n_steps = 7
        bandwidth = 3
        nm = 0.8
        strategy = bisr_strategy(
            bandwidth=bandwidth,
            momentum=0.9,
            normalized=True,
        )
        dense_row_l2 = (
            strategy.streaming_matrix(n_steps=n_steps)
            .materialize(n_steps)
            .square()
            .sum(dim=1)
            .sqrt()
        )
        noise_fn, state = mf_gaussian_noise(
            grad_template,
            strategy,
            n_steps=n_steps,
            noise_multiplier=nm,
            key=key(795),
        )
        dense_fn, dense_state = _matrix_factorization_noise(
            grad_template,
            strategy.streaming_matrix(n_steps=n_steps),
            key=key(795),
            n_steps=n_steps,
        )
        pg = PerGroup(
            groups={"w": "attn", "b": "mlp"},
            values={"attn": 0.5, "mlp": 1.5},
        )
        grads = clipped(
            {"w": torch.zeros(4, 3), "b": torch.zeros(4)},
            max_norm=pg,
        )

        for step in range(n_steps):
            out, state = noise_fn(grads, state)
            base = per_group_noise_stddev(pg, nm)
            dense_tree, dense_state = dense_fn(
                grads.pytree,
                dense_state,
                stddev=base,
            )
            for actual, expected in zip(
                tree_leaves(out.pytree), tree_leaves(dense_tree), strict=True
            ):
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
            assert isinstance(out.noise_stddev, PerGroup)
            for group in base.values:
                assert out.noise_stddev.values[group] == pytest.approx(
                    base.values[group] * float(dense_row_l2[step]),
                    rel=1e-12,
                )
            assert len(state._inner_state.history) == bandwidth - 1

        assert state._step_counter == n_steps

    def test_constant_max_norm_latch_pergroup(self, grad_template):
        """Identical ``PerGroup`` across calls keeps the latch happy."""
        strategy = band_mf_strategy(bands=3, momentum=0.9)
        noise_fn, state = mf_gaussian_noise(
            grad_template,
            strategy,
            n_steps=10,
            noise_multiplier=1.0,
            key=key(3),
        )
        pg = PerGroup(
            groups={"w": "g", "b": "g"},
            values={"g": 1.0},
        )
        tree = {"w": torch.zeros(4, 3), "b": torch.zeros(4)}
        _, state = noise_fn(clipped(tree, max_norm=pg), state)
        _, state = noise_fn(clipped(tree, max_norm=pg), state)
        assert state._first_max_norm == pg
        assert state._first_max_norm_sync_fingerprint == fingerprint_per_group_max_norm(
            pg
        )

    def test_per_group_matches_isotropic_when_uniform(self):
        """Uniform ``B_g = B``: per-leaf stddevs match isotropic at ``effective``."""
        nm = 1.0
        B = 0.7
        pg = PerGroup(
            groups={"w": "g1", "b": "g2"},
            values={"g1": B, "g2": B},
        )
        eff = pg.effective
        sig_pg = per_group_noise_stddev(pg, nm)
        sigma_iso = nm * eff
        assert sig_pg.for_path("w") == pytest.approx(sigma_iso)
        assert sig_pg.for_path("b") == pytest.approx(sigma_iso)

    def test_per_group_unequal_bounds_strict_utility_win(self, grad_template):
        """Asymmetric group bounds: summed leaf variance under optimal per-group
        allocation is strictly below isotropic noise at ``max_norm.effective``."""
        nm = 1.0
        pg = PerGroup(
            groups={"w": "a", "b": "b"},
            values={"a": 1.0, "b": 4.0},
        )
        sig = per_group_noise_stddev(pg, nm)
        v_opt = sig.for_path("w") ** 2 * grad_template["w"].numel()
        v_opt += sig.for_path("b") ** 2 * grad_template["b"].numel()
        sigma_iso = nm * pg.effective
        n_w = grad_template["w"].numel()
        n_b = grad_template["b"].numel()
        v_iso = sigma_iso**2 * (n_w + n_b)
        assert v_opt < v_iso * 0.99


class TestMfNoisePerGroupPairedStream:
    @pytest.fixture
    def grad_template(self):
        return {"w": torch.zeros(4, 3), "b": torch.zeros(4)}

    def test_paired_returns_per_group_stddevs(self, grad_template):
        n_steps = 20
        strategy = band_mf_strategy(bands=4, momentum=0.9)
        second = band_mf_strategy(bands=4, momentum=0.99)
        nm = 1.2
        c1 = _max_column_norm(strategy, n_steps=n_steps)
        noise_fn, state = mf_gaussian_noise(
            grad_template,
            strategy,
            n_steps=n_steps,
            noise_multiplier=nm,
            key=key(11),
            second_moment_strategy=second,
        )
        zeta = 0.05
        pg = PerGroup(
            groups={"w": "a", "b": "b"},
            values={"a": zeta, "b": zeta * 2},
        )
        sq = {k: v * v for k, v in {"w": torch.ones(4, 3), "b": torch.ones(4)}.items()}
        sq_pg = pg * pg
        paired = SecondMomentClippingOutput(
            grads=clipped({"w": torch.ones(4, 3), "b": torch.ones(4)}, max_norm=pg),
            squared_grads=clipped(sq, max_norm=sq_pg),
        )
        out, _ = noise_fn(paired, state)
        assert isinstance(out.noisy_grads.noise_stddev, PerGroup)
        assert isinstance(out.noisy_squared_grads.noise_stddev, PerGroup)
        # Joint Mahalanobis on encoded sensitivities equals (c1 / nm)².
        # The published ``noise_stddev`` is the per-step *realized* σ
        # (= base σ · ‖row_t(C^-1)‖); divide each stream's row_l2 out to
        # recover base σ for the joint-PLD calibration identity.
        c2 = _max_column_norm(second, n_steps=n_steps)
        first_row_l2 = _row_l2_at_zero(strategy, n_steps=n_steps)
        second_row_l2 = _row_l2_at_zero(second, n_steps=n_steps)
        s1 = out.noisy_grads.noise_stddev
        s2 = out.noisy_squared_grads.noise_stddev
        mahal = 0.0
        for param_key in ("w", "b"):
            b1 = pg.for_path(param_key)
            b2 = sq_pg.for_path(param_key)
            d1 = b1 * c1
            d2 = b2 * c2
            base_s1 = s1.for_path(param_key) / first_row_l2
            base_s2 = s2.for_path(param_key) / second_row_l2
            mahal += (d1 / base_s1) ** 2 + (d2 / base_s2) ** 2
        assert mahal == pytest.approx((c1 / nm) ** 2, rel=1e-9)


class TestMfNoisePerGroupMahalanobisSingleStream:
    """Encoded Mahalanobis budget with per-group IID base stddev."""

    @pytest.fixture
    def grad_template(self):
        return {"w": torch.zeros(3), "b": torch.zeros(3)}

    def test_mahalanobis_equals_c1_over_nm_squared(self, grad_template):
        n_steps = 12
        strategy = band_mf_strategy(bands=3, momentum=0.9)
        nm = 0.75
        c1 = _max_column_norm(strategy, n_steps=n_steps)
        noise_fn, state = mf_gaussian_noise(
            grad_template,
            strategy,
            n_steps=n_steps,
            noise_multiplier=nm,
            key=key(5),
        )
        zeta = 0.08
        pg = PerGroup(
            groups={"w": "a", "b": "b"},
            values={"a": zeta, "b": zeta * 3},
        )
        grads = clipped({"w": torch.zeros(3), "b": torch.zeros(3)}, max_norm=pg)
        out, _ = noise_fn(grads, state)
        sigma = out.noise_stddev
        assert isinstance(sigma, PerGroup)
        # ``noise_stddev`` is the realized per-step σ (= base σ · row_l2);
        # divide row_l2 out to recover base σ for the calibration identity.
        row_l2 = _row_l2_at_zero(strategy, n_steps=n_steps)
        acc = 0.0
        for param_key in ("w", "b"):
            b_g = pg.for_path(param_key)
            sens = b_g * c1
            base_sigma = sigma.for_path(param_key) / row_l2
            acc += (sens / base_sigma) ** 2
        assert acc == pytest.approx((c1 / nm) ** 2, rel=1e-9)

    def test_pld_match_per_group_single_stream(self, grad_template):
        """Plan checklist name — same Mahalanobis check as
        ``test_mahalanobis_equals_c1_over_nm_squared``."""
        self.test_mahalanobis_equals_c1_over_nm_squared(grad_template)


class TestPerGroupPairedWithClippedGradAndMf:
    """End-to-end: ``clipped_grad(..., second_moment=True, PerGroup)`` → ``mf_gaussian_noise``."""

    def test_per_group_paired_with_mf(self):
        torch.manual_seed(0)
        params = {"w": torch.randn(3), "b": torch.randn(())}
        batch_size = 6
        x = torch.randn(batch_size, 3)
        y = torch.randn(batch_size)

        def loss_fn(p, x_, y_):
            return ((x_ @ p["w"] + p["b"] - y_) ** 2).mean()

        pg = PerGroup(
            groups={"w": "ga", "b": "gb"},
            values={"ga": 2.0, "gb": 2.0},
        )
        grad_fn, clip_state = clipped_grad(
            loss_fn,
            argnums=0,
            batch_argnums=(1, 2),
            clipping_norm=pg,
            normalize_by=batch_size,
            second_moment=True,
        )
        paired, clip_state = grad_fn(params, x, y, state=clip_state)
        assert isinstance(paired, SecondMomentClippingOutput)

        n_steps = 40
        grad_template = {"w": torch.zeros(3), "b": torch.zeros(())}
        strategy = band_mf_strategy(bands=4, momentum=0.9)
        second = band_mf_strategy(bands=4, momentum=0.99)
        nm = 1.0
        c1 = _max_column_norm(strategy, n_steps=n_steps)
        c2 = _max_column_norm(second, n_steps=n_steps)
        noise_fn, noise_state = mf_gaussian_noise(
            grad_template,
            strategy,
            n_steps=n_steps,
            noise_multiplier=nm,
            key=key(2026),
            second_moment_strategy=second,
        )
        out, _ = noise_fn(paired, noise_state)
        assert isinstance(out.noisy_grads.noise_stddev, PerGroup)
        assert isinstance(out.noisy_squared_grads.noise_stddev, PerGroup)
        s1 = out.noisy_grads.noise_stddev
        s2 = out.noisy_squared_grads.noise_stddev
        pg1 = paired.grads.max_norm
        assert isinstance(pg1, PerGroup)
        sq1 = paired.squared_grads.max_norm
        assert isinstance(sq1, PerGroup)
        # Recover base σ from realized σ (= base · row_l2) per stream.
        first_row_l2 = _row_l2_at_zero(strategy, n_steps=n_steps)
        second_row_l2 = _row_l2_at_zero(second, n_steps=n_steps)
        mahal = 0.0
        for param_key in ("w", "b"):
            d1 = pg1.for_path(param_key) * c1
            d2 = sq1.for_path(param_key) * c2
            base_s1 = s1.for_path(param_key) / first_row_l2
            base_s2 = s2.for_path(param_key) / second_row_l2
            mahal += (d1 / base_s1) ** 2 + (d2 / base_s2) ** 2
        assert mahal == pytest.approx((c1 / nm) ** 2, rel=1e-8)
