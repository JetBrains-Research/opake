"""Tests for AUTO-S automatic clipping (Bu et al., NeurIPS 2023)."""

import math

import pytest
import torch

from opaque.api.engine.clipping import auto_clipped_grad, clipped_grad
from opaque.api.engine.clipping._per_group import per_group
from opaque.api.engine.clipping.fun import auto_clipped_fun, auto_scale_pytree
from opaque.api.engine.clipping.types import AutoClippedGradAux, AutoClipState
from opaque.exceptions import ConfigurationError
from opaque.types import ClippedPytree, PerGroup


def _unwrap_clipped(value):
    assert isinstance(value, ClippedPytree)
    return value.pytree


class TestAutoScalePytree:
    """Tests for the low-level auto_scale_pytree primitive."""

    def test_formula_scalar(self):
        """Output should equal R * g / (||g|| + gamma)."""
        pytree = {"a": torch.tensor([3.0, 4.0])}  # norm = 5
        scaled, aux = auto_scale_pytree(pytree, R=1.0, gamma=0.01)

        expected_scale = 1.0 / (5.0 + 0.01)
        torch.testing.assert_close(scaled["a"], pytree["a"] * expected_scale)
        assert aux.norm.item() == pytest.approx(5.0)
        assert aux.group_norms is None

    def test_output_norm_clipped_by_R(self):
        """For any input, ||output|| must be <= R."""
        generator = torch.Generator().manual_seed(42)
        R = 0.7
        for _ in range(20):
            pytree = {
                "a": torch.randn(5, generator=generator) * 100,
                "b": torch.randn(3, generator=generator),
            }
            scaled, _ = auto_scale_pytree(pytree, R=R, gamma=0.01)
            total = torch.cat([scaled["a"], scaled["b"]])
            norm = float(torch.linalg.vector_norm(total))
            assert norm <= R + 1e-6, f"norm {norm} exceeds R={R}"

    def test_approaches_unit_projection_for_large_norms(self):
        """When ||g|| >> gamma, output approaches unit-norm projection."""
        pytree = {"a": torch.tensor([300.0, 400.0])}  # norm = 500
        scaled, _ = auto_scale_pytree(pytree, R=1.0, gamma=0.01)
        # Expected norm ≈ 500 / (500 + 0.01) ≈ 0.99998
        norm = float(torch.linalg.vector_norm(scaled["a"]))
        assert norm == pytest.approx(1.0, rel=1e-4)

    def test_zero_gradient_maps_to_zero(self):
        """g = 0 should yield output 0 (finite, no NaN)."""
        pytree = {"a": torch.zeros(4)}
        scaled, aux = auto_scale_pytree(pytree, R=1.0, gamma=0.01)
        torch.testing.assert_close(scaled["a"], torch.zeros(4))
        assert aux.norm.item() == 0.0

    def test_nan_sanitization(self):
        """NaN/Inf values should be replaced with 0 before scaling."""
        pytree = {"a": torch.tensor([float("nan"), float("inf"), 1.0])}
        scaled, _ = auto_scale_pytree(pytree, R=1.0, gamma=0.01)
        assert not torch.isnan(scaled["a"]).any()
        assert not torch.isinf(scaled["a"]).any()

    def test_rejects_zero_gamma(self):
        """gamma=0 is rejected (AUTO-V, undefined at zero gradient)."""
        with pytest.raises(ValueError, match="gamma must be positive"):
            auto_scale_pytree({"a": torch.tensor([1.0])}, R=1.0, gamma=0.0)

    def test_per_group_formula(self):
        """Per-group AUTO-S applies the formula within each group."""
        pytree = {
            "attn.q": torch.tensor([3.0]),
            "attn.k": torch.tensor([4.0]),  # attn group norm = 5
            "mlp.w": torch.tensor([6.0]),
        }
        pg = PerGroup(
            groups={"attn.q": "attn", "attn.k": "attn", "mlp.w": "mlp"},
            values={"attn": 2.0, "mlp": 3.0},
        )
        scaled, aux = auto_scale_pytree(pytree, R=pg, gamma=0.01)

        # attn: scale = 2 / (5 + 0.01)
        attn_scale = 2.0 / (5.0 + 0.01)
        torch.testing.assert_close(scaled["attn.q"], torch.tensor([3.0 * attn_scale]))
        torch.testing.assert_close(scaled["attn.k"], torch.tensor([4.0 * attn_scale]))

        # mlp: scale = 3 / (6 + 0.01)
        mlp_scale = 3.0 / (6.0 + 0.01)
        torch.testing.assert_close(scaled["mlp.w"], torch.tensor([6.0 * mlp_scale]))

        assert aux.group_norms is not None
        assert aux.group_norms["attn"].item() == pytest.approx(5.0)
        assert aux.group_norms["mlp"].item() == pytest.approx(6.0)

    def test_per_group_sensitivity_bound(self):
        """Per-group output norm is clipped by sqrt(sum R_k^2)."""
        pytree = {
            "a": torch.randn(5) * 1000,
            "b": torch.randn(5) * 1000,
        }
        pg = PerGroup(
            groups={"a": "g1", "b": "g2"},
            values={"g1": 0.5, "g2": 1.5},
        )
        scaled, _ = auto_scale_pytree(pytree, R=pg, gamma=0.01)
        total_norm = float(
            torch.linalg.vector_norm(torch.cat([scaled["a"], scaled["b"]]))
        )
        max_norm = math.sqrt(0.5**2 + 1.5**2)
        assert total_norm <= max_norm + 1e-4

    def test_nested_pytree_scales_groups(self):
        pytree = {
            "layer1": {"attn": torch.tensor([3.0, 4.0]), "mlp": torch.tensor([6.0])},
            "layer2": {"attn": torch.tensor([0.0, 0.0]), "mlp": torch.tensor([8.0])},
        }
        pg = per_group(pytree, attn=2.0, mlp=1.0)
        scaled, aux = auto_scale_pytree(pytree, R=pg, gamma=0.01)

        attn_scale = 2.0 / (5.0 + 0.01)
        mlp_scale = 1.0 / (10.0 + 0.01)
        torch.testing.assert_close(
            scaled["layer1"]["attn"], torch.tensor([3.0, 4.0]) * attn_scale
        )
        torch.testing.assert_close(
            scaled["layer1"]["mlp"], torch.tensor([6.0]) * mlp_scale
        )
        torch.testing.assert_close(
            scaled["layer2"]["mlp"], torch.tensor([8.0]) * mlp_scale
        )
        assert aux.group_norms is not None
        assert aux.group_norms["attn"].item() == pytest.approx(5.0)
        assert aux.group_norms["mlp"].item() == pytest.approx(10.0)

    def test_group_key_mismatch_raises(self):
        pytree = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
        pg = PerGroup(groups={"a": "g1"}, values={"g1": 1.0})
        with pytest.raises(
            ConfigurationError, match="must match the pytree tensor leaves"
        ):
            auto_scale_pytree(pytree, R=pg)


class TestAutoClipState:
    """Tests for AutoClipState marker behavior and factory validation."""

    def test_state_is_marker(self):
        assert AutoClipState() == AutoClipState()

    def test_rejects_non_positive_R(self):
        with pytest.raises(ValueError, match="positive"):
            auto_clipped_grad(lambda params, x: (params * x).sum(), R=0.0)

    def test_rejects_non_positive_gamma(self):
        with pytest.raises(ValueError, match="gamma"):
            auto_clipped_grad(lambda params, x: (params * x).sum(), R=1.0, gamma=0.0)


class TestAutoClippedGrad:
    """Tests for auto_clipped_grad."""

    def _loss(self, params, x, y):
        return ((x @ params - y) ** 2).mean()

    def test_basic_usage(self):
        grad_fn, state = auto_clipped_grad(
            self._loss, argnums=0, batch_argnums=(1, 2), R=1.0
        )
        params = torch.randn(10)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        grads, new_state = grad_fn(params, batch_x, batch_y, state=state)
        grads = _unwrap_clipped(grads)
        assert grads.shape == params.shape
        assert isinstance(new_state, AutoClipState)

    def test_stats_preserve_sanitized_auto_clipping_output(self):
        def loss(params, data):
            return torch.sqrt(data - params)

        grad_fn, state = auto_clipped_grad(
            loss,
            argnums=0,
            batch_argnums=1,
            R=1.0,
            return_stats=True,
        )

        (grads, stats), _ = grad_fn(
            torch.tensor(0.0),
            torch.tensor([1.0, -1.0]),
            state=state,
        )

        assert stats.batch_size == 2
        assert torch.isfinite(_unwrap_clipped(grads)).all()

    def test_state_is_fixed(self):
        """State does not change across steps (no adaptation)."""
        grad_fn, state = auto_clipped_grad(
            self._loss, argnums=0, batch_argnums=(1, 2), R=0.5
        )
        params = torch.randn(10)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        _, s1 = grad_fn(params, batch_x, batch_y, state=state)
        _, s2 = grad_fn(params, batch_x, batch_y, state=s1)
        assert s1 == s2  # dataclasses with same values

    def test_bound_via_output_metadata(self):
        """Output max_norm equals R / normalize_by."""
        grad_fn, state = auto_clipped_grad(
            self._loss,
            argnums=0,
            batch_argnums=(1, 2),
            R=2.5,
            normalize_by=10.0,
        )
        params = torch.randn(10)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)
        grads, _ = grad_fn(params, batch_x, batch_y, state=state)
        assert grads.max_norm == pytest.approx(0.25)

    def test_grad_norm_clipped(self):
        """Sum of per-example scaled gradients has clipped norm.

        With B examples, each scaled to norm <= R, the triangle-inequality
        max_norm is B * R.  We check this holds.
        """
        R = 0.3
        grad_fn, state = auto_clipped_grad(
            self._loss, argnums=0, batch_argnums=(1, 2), R=R
        )
        params = torch.randn(10)
        batch_size = 8
        batch_x = torch.randn(batch_size, 10) * 100  # force large gradients
        batch_y = torch.randn(batch_size) * 100

        grads, _ = grad_fn(params, batch_x, batch_y, state=state)
        grads = _unwrap_clipped(grads)
        norm = float(torch.linalg.vector_norm(grads))
        assert norm <= batch_size * R + 1e-5

    def test_normalize_by(self):
        """normalize_by scales the summed gradient."""
        params = torch.randn(10)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        fn1, s1 = auto_clipped_grad(
            self._loss, argnums=0, batch_argnums=(1, 2), R=1.0, normalize_by=1.0
        )
        fn2, s2 = auto_clipped_grad(
            self._loss, argnums=0, batch_argnums=(1, 2), R=1.0, normalize_by=4.0
        )
        g1, _ = fn1(params, batch_x, batch_y, state=s1)
        g2, _ = fn2(params, batch_x, batch_y, state=s2)
        g1 = _unwrap_clipped(g1)
        g2 = _unwrap_clipped(g2)
        torch.testing.assert_close(g1 / 4.0, g2)

    def test_return_aux(self):
        """return_aux=True produces AutoClippedGradAux with norms."""
        grad_fn, state = auto_clipped_grad(
            self._loss,
            argnums=0,
            batch_argnums=(1, 2),
            R=1.0,
            return_aux=True,
        )
        params = torch.randn(10)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        (_grads, aux), _ = grad_fn(params, batch_x, batch_y, state=state)
        assert isinstance(aux, AutoClippedGradAux)
        assert aux.grad_norms.shape == (8,)
        assert aux.clipped_grad_norms.shape == (8,)
        assert aux.loss_values.shape == (8,)
        assert aux.batch_size == 8
        # Each clipped norm must be <= R
        assert bool((aux.clipped_grad_norms <= 1.0 + 1e-5).all())

    def test_microbatching_equivalence(self):
        """Microbatched AUTO-S matches full-batch AUTO-S."""
        torch.manual_seed(0)
        params = torch.randn(10)
        batch_x = torch.randn(16, 10)
        batch_y = torch.randn(16)

        fn_full, s_full = auto_clipped_grad(
            self._loss, argnums=0, batch_argnums=(1, 2), R=0.5
        )
        fn_mb, s_mb = auto_clipped_grad(
            self._loss,
            argnums=0,
            batch_argnums=(1, 2),
            R=0.5,
            microbatch_size=4,
        )
        g_full, _ = fn_full(params, batch_x, batch_y, state=s_full)
        g_mb, _ = fn_mb(params, batch_x, batch_y, state=s_mb)
        g_full = _unwrap_clipped(g_full)
        g_mb = _unwrap_clipped(g_mb)
        torch.testing.assert_close(g_full, g_mb, rtol=1e-5, atol=1e-6)

    def test_empty_batch(self):
        """Empty batch yields zero gradient and preserves state."""
        grad_fn, state = auto_clipped_grad(
            self._loss, argnums=0, batch_argnums=(1, 2), R=1.0
        )
        params = torch.randn(10)
        batch_x = torch.empty(0, 10)
        batch_y = torch.empty(0)

        grads, new_state = grad_fn(params, batch_x, batch_y, state=state)
        grads = _unwrap_clipped(grads)
        assert grads.shape == params.shape
        torch.testing.assert_close(grads, torch.zeros_like(params))
        assert new_state == state

    def test_rejects_non_positive_R(self):
        with pytest.raises(ValueError, match="R must be positive"):
            auto_clipped_grad(self._loss, R=0.0, batch_argnums=(1, 2))

    def test_rejects_non_positive_gamma(self):
        with pytest.raises(ValueError, match="gamma"):
            auto_clipped_grad(self._loss, gamma=0.0, batch_argnums=(1, 2))

    def test_differs_from_fixed_clipping_at_small_norms(self):
        """At small ||g||, AUTO-S can differ from fixed clipping.

        This is the defining behavioral difference: AUTO-S continuously scales by
        R / (||g|| + gamma) even when ||g|| < R, unlike fixed clipping's
        min(1, R/||g||).
        """
        # Construct a batch where per-example gradients will be small.
        torch.manual_seed(0)
        params = torch.tensor([0.001, 0.001])

        def tiny_loss(p, x, y):
            # Loss with tiny gradient magnitude
            return 0.001 * ((x @ p - y) ** 2).mean()

        batch_x = torch.randn(4, 2) * 0.01
        batch_y = torch.randn(4) * 0.01

        fn_auto, s_auto = auto_clipped_grad(
            tiny_loss, argnums=0, batch_argnums=(1, 2), R=1.0, gamma=0.01
        )
        fn_fixed, s_fixed = clipped_grad(
            tiny_loss, argnums=0, batch_argnums=(1, 2), clipping_norm=1.0
        )
        g_auto, _ = fn_auto(params, batch_x, batch_y, state=s_auto)
        g_fixed, _ = fn_fixed(params, batch_x, batch_y, state=s_fixed)
        g_auto = _unwrap_clipped(g_auto)
        g_fixed = _unwrap_clipped(g_fixed)

        # Gradients should be materially different (AUTO-S amplifies small grads).
        assert not torch.allclose(g_auto, g_fixed, rtol=1e-3)

    def test_has_aux_passthrough(self):
        """loss_fn returning (loss, extra) is supported with has_aux=True."""

        def loss_with_aux(p, x, y):
            pred = x @ p
            return ((pred - y) ** 2).mean(), pred

        grad_fn, state = auto_clipped_grad(
            loss_with_aux,
            argnums=0,
            has_aux=True,
            batch_argnums=(1, 2),
            R=1.0,
            return_aux=True,
        )
        params = torch.randn(10)
        batch_x = torch.randn(8, 10)
        batch_y = torch.randn(8)

        (_grads, aux), _ = grad_fn(params, batch_x, batch_y, state=state)
        assert aux.loss_aux is not None  # predictions per example


class TestAutoClippedGradPerGroup:
    """Per-group AUTO-S."""

    def test_per_group_bound_metadata(self):
        """Per-group output max_norm is R / normalize_by."""
        params = {"a": torch.randn(4), "b": torch.randn(4)}
        pg = PerGroup(groups={"a": "g1", "b": "g2"}, values={"g1": 3.0, "g2": 4.0})
        grad_fn, state = auto_clipped_grad(
            lambda p, x: ((p["a"] + p["b"] - x) ** 2).mean(),
            argnums=0,
            batch_argnums=1,
            R=pg,
            normalize_by=2.0,
        )
        grads, _ = grad_fn(params, torch.randn(8, 4), state=state)
        assert isinstance(grads.max_norm, PerGroup)
        assert grads.max_norm.values == {
            "g1": pytest.approx(1.5),
            "g2": pytest.approx(2.0),
        }
        assert grads.max_norm.effective == pytest.approx(5.0 / 2.0)

    def test_per_group_applies_correct_scales(self):
        """Per-group AUTO-S scales each group by its own R_k."""

        # Two-parameter model: dict form needed for per-group.
        def loss_fn(params, x, y):
            pred = x @ params["w"] + params["b"]
            return ((pred - y) ** 2).mean()

        pg = PerGroup(
            groups={"w": "weights", "b": "biases"},
            values={"weights": 2.0, "biases": 0.5},
        )
        grad_fn, state = auto_clipped_grad(
            loss_fn,
            argnums=0,
            batch_argnums=(1, 2),
            R=pg,
            return_aux=True,
        )
        params = {"w": torch.randn(10) * 10, "b": torch.randn(1) * 10}
        batch_x = torch.randn(6, 10)
        batch_y = torch.randn(6)

        (_grads, aux), _ = grad_fn(params, batch_x, batch_y, state=state)
        assert aux.group_norms is not None
        assert set(aux.group_norms.keys()) == {"weights", "biases"}


class TestAutoClippedFun:
    """Tests for auto_clipped_fun (general-purpose scaling)."""

    def test_basic_scaling_and_sum(self):
        """Sum of scaled per-example outputs; each output clipped by R."""

        def per_example(x):
            return x  # identity per-example, batch sums

        fn, state = auto_clipped_fun(per_example, batch_argnums=0, R=1.0)
        batch = torch.randn(5, 3) * 10
        result, new_state = fn(batch, state=state)
        result = _unwrap_clipped(result)

        expected = torch.zeros(3)
        for i in range(5):
            scale = 1.0 / (torch.linalg.vector_norm(batch[i]) + 0.01)
            expected = expected + batch[i] * scale

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)
        assert new_state == state

    @pytest.mark.parametrize(
        "microbatch_size",
        [None, 2],
        ids=["full-batch", "microbatch"],
    )
    def test_rejects_empty_batch(self, microbatch_size):
        fn, state = auto_clipped_fun(
            lambda value: value,
            R=1.0,
            microbatch_size=microbatch_size,
        )
        with pytest.raises(ConfigurationError, match="requires a non-empty batch"):
            fn(torch.empty(0, 3), state=state)

    def test_return_stats_preserves_sanitized_output(self):
        fn, state = auto_clipped_fun(
            lambda x: x,
            batch_argnums=0,
            R=1.0,
            return_stats=True,
        )

        (result, stats), _ = fn(
            torch.tensor([[1.0], [float("nan")]]),
            state=state,
        )

        assert stats.batch_size == 2
        assert torch.isfinite(_unwrap_clipped(result)).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestAutoClippedGradEmptyBatchParity:
    """Empty-batch short-circuit must mirror non-empty structure and dtype."""

    @staticmethod
    def _loss(params, x, y):
        return ((x @ params["w"] - y) ** 2).mean()

    def test_dtype_and_transform_parity(self):
        params = {"w": torch.randn(10, dtype=torch.bfloat16)}
        x = torch.randn(8, 10, dtype=torch.bfloat16)
        y = torch.randn(8, dtype=torch.bfloat16)

        def transform(g):
            return {"q": g["w"] + 7.0}

        gf, state = auto_clipped_grad(
            self._loss,
            batch_argnums=(1, 2),
            R=1.0,
            pre_clipping_transform=transform,
            dtype=torch.float32,
            second_moment=True,
        )
        full, _ = gf(params, x, y, state=state)
        empty, _ = gf(params, x[:0], y[:0], state=state)
        assert sorted(full.grads.pytree) == sorted(empty.grads.pytree) == ["q"]
        assert (
            sorted(full.squared_grads.pytree)
            == sorted(empty.squared_grads.pytree)
            == ["q"]
        )
        assert empty.grads.pytree["q"].dtype == torch.float32
        assert empty.squared_grads.pytree["q"].dtype == torch.float32
        assert torch.all(empty.grads.pytree["q"] == 0)
        assert torch.all(empty.squared_grads.pytree["q"] == 0)
