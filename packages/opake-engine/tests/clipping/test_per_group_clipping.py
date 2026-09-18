"""Tests for per-group clipping through clip_pytree, clipped_fun, and clipped_grad."""

import pytest
import torch

from opake.api.engine.clipping import clipped_grad
from opake.api.engine.clipping._per_group import per_group
from opake.api.engine.clipping._pytree import clip_pytree
from opake.api.engine.clipping.types import FixedClipState
from opake.exceptions import ConfigurationError
from opake.types import ClippedPytree, PerGroup


def _unwrap_clipped(value):
    assert isinstance(value, ClippedPytree)
    return value.pytree


class TestClipPytreePerGroup:
    """Tests for clip_pytree with PerGroup clipping_norm."""

    def _make_pg(self, keys, group_map, values):
        """Helper to build a PerGroup from key lists."""
        groups = {k: group_map[k] for k in keys}
        return PerGroup(groups=groups, values=values)

    def test_no_clipping_when_within_bounds(self):
        """Gradients within group norms should pass through unchanged."""
        pytree = {
            "attn.q": torch.tensor([1.0, 0.0]),  # norm = 1
            "attn.k": torch.tensor([0.0, 1.0]),  # norm = 1
            "mlp.w": torch.tensor([0.5, 0.0]),  # norm = 0.5
        }
        pg = PerGroup(
            groups={"attn.q": "attn", "attn.k": "attn", "mlp.w": "mlp"},
            values={"attn": 10.0, "mlp": 10.0},  # Large bounds
        )
        clipped, _aux = clip_pytree(pytree, pg)
        for key in pytree:
            torch.testing.assert_close(clipped[key], pytree[key])

    def test_clips_groups_independently(self):
        """Each group should be clipped to its own norm max_norm."""
        # attn group: norm = sqrt(9 + 16) = 5, max_norm = 1 → scale = 0.2
        # mlp group: norm = 3, max_norm = 6 → no clipping (scale = 1)
        pytree = {
            "attn.q": torch.tensor([3.0]),
            "attn.k": torch.tensor([4.0]),
            "mlp.w": torch.tensor([3.0]),
        }
        pg = PerGroup(
            groups={"attn.q": "attn", "attn.k": "attn", "mlp.w": "mlp"},
            values={"attn": 1.0, "mlp": 6.0},
        )
        clipped, _aux = clip_pytree(pytree, pg)

        # attn group clipped: scale = 1/5
        torch.testing.assert_close(clipped["attn.q"], torch.tensor([0.6]))
        torch.testing.assert_close(clipped["attn.k"], torch.tensor([0.8]))

        # mlp group not clipped
        torch.testing.assert_close(clipped["mlp.w"], torch.tensor([3.0]))

    def test_aux_returns_global_norm(self):
        """aux.norm should be the global L2 norm across all groups."""
        pytree = {
            "a": torch.tensor([3.0]),
            "b": torch.tensor([4.0]),
        }
        pg = PerGroup(
            groups={"a": "g1", "b": "g2"},
            values={"g1": 10.0, "g2": 10.0},
        )
        _, aux = clip_pytree(pytree, pg)
        assert aux.norm.item() == pytest.approx(5.0)  # sqrt(9+16)

    def test_nan_sanitization(self):
        """NaN values should be replaced with 0 before clipping."""
        pytree = {
            "a": torch.tensor([float("nan"), 1.0]),
            "b": torch.tensor([2.0]),
        }
        pg = PerGroup(
            groups={"a": "g1", "b": "g2"},
            values={"g1": 10.0, "g2": 10.0},
        )
        clipped, _ = clip_pytree(pytree, pg)
        assert not torch.isnan(clipped["a"]).any()

    def test_return_zero(self):
        """return_zero should zero out the output regardless of inputs."""
        pytree = {"a": torch.tensor([5.0]), "b": torch.tensor([3.0])}
        pg = PerGroup(
            groups={"a": "g1", "b": "g2"},
            values={"g1": 10.0, "g2": 10.0},
        )
        clipped, _ = clip_pytree(pytree, pg, return_zero=True)
        torch.testing.assert_close(clipped["a"], torch.tensor([0.0]))
        torch.testing.assert_close(clipped["b"], torch.tensor([0.0]))

    def test_zero_norm_group(self):
        """Group with all-zero tensors should produce zero output without NaN."""
        pytree = {
            "a": torch.tensor([0.0, 0.0]),
            "b": torch.tensor([1.0]),
        }
        pg = PerGroup(
            groups={"a": "g1", "b": "g2"},
            values={"g1": 1.0, "g2": 10.0},
        )
        clipped, _ = clip_pytree(pytree, pg)
        assert not torch.isnan(clipped["a"]).any()
        torch.testing.assert_close(clipped["a"], torch.tensor([0.0, 0.0]))

    def test_nested_pytree_clips_groups(self):
        """Nested params clip by optree ParamPath — not silently skipped."""
        pytree = {
            "layer1": {
                "attn": torch.tensor([3.0, 4.0]),
                "mlp": torch.tensor([30.0]),
            },
            "layer2": {
                "attn": torch.tensor([5.0, 12.0]),
                "mlp": torch.tensor([40.0]),
            },
        }
        pg = per_group(pytree, attn=1.0, mlp=1.0)
        clipped, aux = clip_pytree(pytree, pg)

        attn_scale = 1.0 / (9.0 + 16.0 + 25.0 + 144.0) ** 0.5
        mlp_scale = 1.0 / 50.0
        torch.testing.assert_close(
            clipped["layer1"]["attn"], torch.tensor([3.0, 4.0]) * attn_scale
        )
        torch.testing.assert_close(
            clipped["layer2"]["attn"], torch.tensor([5.0, 12.0]) * attn_scale
        )
        torch.testing.assert_close(
            clipped["layer1"]["mlp"], torch.tensor([30.0]) * mlp_scale
        )
        torch.testing.assert_close(
            clipped["layer2"]["mlp"], torch.tensor([40.0]) * mlp_scale
        )
        assert aux.group_norms is not None
        assert aux.group_norms["attn"].item() == pytest.approx((194.0) ** 0.5)
        assert aux.group_norms["mlp"].item() == pytest.approx(50.0)

    def test_group_key_mismatch_raises(self):
        """Missing or extra PerGroup paths must raise, not skip clipping."""
        pytree = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
        pg_missing = PerGroup(groups={"a": "g1"}, values={"g1": 1.0})
        with pytest.raises(
            ConfigurationError, match="must match the pytree tensor leaves"
        ):
            clip_pytree(pytree, pg_missing)

        pg_extra = PerGroup(
            groups={"a": "g1", "b": "g2", "c": "g3"},
            values={"g1": 1.0, "g2": 1.0, "g3": 1.0},
        )
        with pytest.raises(
            ConfigurationError, match="must match the pytree tensor leaves"
        ):
            clip_pytree(pytree, pg_extra)


class TestFixedClipStatePerGroup:
    """Tests for fixed clipping marker state and factory validation."""

    def test_state_is_marker(self):
        assert FixedClipState() == FixedClipState()

    def test_clipped_grad_accepts_per_group_clipping_norm(self):
        pg = PerGroup(
            groups={"a": "g1", "b": "g2"},
            values={"g1": 2.0, "g2": 4.0},
        )
        _, state = clipped_grad(
            lambda params, data: params["a"].sum(), clipping_norm=pg
        )
        assert isinstance(state, FixedClipState)

    def test_validation_rejects_non_positive_group(self):
        pg = PerGroup(groups={"a": "g1"}, values={"g1": -1.0})
        with pytest.raises(ValueError, match="positive"):
            clipped_grad(lambda params, data: params["a"].sum(), clipping_norm=pg)


class TestClippedGradPerGroup:
    """Tests for clipped_grad with PerGroup clipping_norm."""

    def test_basic_per_group_clipping(self):
        """clipped_grad should clip per group when given PerGroup norm."""

        def loss(params, data):
            pred = params["w1"] * data + params["w2"] * data
            return (pred**2).mean()

        params = {"w1": torch.tensor(3.0), "w2": torch.tensor(4.0)}
        pg = per_group(params, w1=1.0, w2=1.0)

        grad_fn, clip_state = clipped_grad(
            loss, argnums=0, batch_argnums=1, clipping_norm=pg
        )

        data = torch.tensor([1.0, 2.0, 3.0])
        grads, _ = grad_fn(params, data, state=clip_state)
        grads = _unwrap_clipped(grads)

        assert isinstance(grads, dict)
        assert "w1" in grads
        assert "w2" in grads

    def test_output_bound_preserves_per_group_metadata(self):
        """The clipped output carries per-group max_norm metadata after normalization."""

        def loss(params, data):
            return (params["a"] * data).mean()

        params = {"a": torch.tensor(1.0), "b": torch.tensor(1.0)}
        pg = per_group(params, a=2.0, b=4.0)

        grad_fn, clip_state = clipped_grad(
            loss,
            argnums=0,
            batch_argnums=1,
            clipping_norm=pg,
            normalize_by=10.0,
        )

        grads, _ = grad_fn(params, torch.randn(8), state=clip_state)
        assert isinstance(grads.max_norm, PerGroup)
        assert grads.max_norm.groups == pg.groups
        assert grads.max_norm.values == {
            "a": pytest.approx(0.2),
            "b": pytest.approx(0.4),
        }

    def test_per_group_with_microbatch(self):
        """Per-group clipping should work with microbatching."""

        def loss(params, data):
            return ((params["w"] - data) ** 2).mean()

        params = {"w": torch.tensor(0.0)}
        pg = per_group(params, w=1.0)

        grad_fn, clip_state = clipped_grad(
            loss,
            argnums=0,
            batch_argnums=1,
            clipping_norm=pg,
            microbatch_size=2,
        )

        data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        grads, _ = grad_fn(params, data, state=clip_state)
        grads = _unwrap_clipped(grads)
        assert isinstance(grads, dict)

    def test_per_group_with_return_aux(self):
        """return_aux should work with per-group clipping."""

        def loss(params, data):
            return ((params["w"] - data) ** 2).mean()

        params = {"w": torch.tensor(0.0)}
        pg = per_group(params, w=1.0)

        grad_fn, clip_state = clipped_grad(
            loss,
            argnums=0,
            batch_argnums=1,
            clipping_norm=pg,
            return_aux=True,
        )

        data = torch.tensor([1.0, 2.0, 3.0])
        (grads, aux), _ = grad_fn(params, data, state=clip_state)
        grads = _unwrap_clipped(grads)
        assert isinstance(grads, dict)
        assert aux.grad_norms is not None

    def test_empty_batch_aux_rate_presence_tracks_schema(self):
        """Empty-batch aux mirrors the non-empty per-group ``clipping_rate=None``."""

        def loss(params, data):
            return ((params["w"] - data) ** 2).mean()

        params = {"w": torch.tensor(0.0)}
        pg = per_group(params, w=1.0)
        grad_fn, clip_state = clipped_grad(
            loss,
            argnums=0,
            batch_argnums=1,
            clipping_norm=pg,
            return_aux=True,
        )

        (_, empty_aux), _ = grad_fn(params, torch.empty(0), state=clip_state)
        assert empty_aux.clipping_rate is None
        assert empty_aux.batch_size == 0

        (_, full_aux), _ = grad_fn(params, torch.tensor([1.0, 2.0]), state=clip_state)
        assert full_aux.clipping_rate is None

        scalar_fn, scalar_state = clipped_grad(
            loss,
            argnums=0,
            batch_argnums=1,
            clipping_norm=1.0,
            return_aux=True,
        )
        (_, scalar_empty_aux), _ = scalar_fn(params, torch.empty(0), state=scalar_state)
        assert scalar_empty_aux.clipping_rate == 0.0

    def test_global_vs_per_group_single_group(self):
        """Single-group PerGroup should behave like global clipping."""

        def loss(params, data):
            return ((params["w"] - data) ** 2).mean()

        params = {"w": torch.tensor(0.0)}
        norm_val = 1.5

        # Global clipping
        grad_fn_g, cs_g = clipped_grad(
            loss, argnums=0, batch_argnums=1, clipping_norm=norm_val
        )
        # Per-group with single group
        pg = per_group(params, w=norm_val)
        grad_fn_pg, cs_pg = clipped_grad(
            loss, argnums=0, batch_argnums=1, clipping_norm=pg
        )

        data = torch.tensor([5.0, 10.0, -3.0])
        grads_g, _ = grad_fn_g(params, data, state=cs_g)
        grads_pg, _ = grad_fn_pg(params, data, state=cs_pg)
        grads_g = _unwrap_clipped(grads_g)
        grads_pg = _unwrap_clipped(grads_pg)

        torch.testing.assert_close(grads_g["w"], grads_pg["w"])

    def test_noise_multiplier_bound_arithmetic(self):
        """noise_multiplier * grads.max_norm should preserve PerGroup metadata."""

        def loss(params, data):
            return (params["a"] * data).mean()

        params = {"a": torch.tensor(1.0), "b": torch.tensor(1.0)}
        pg = per_group(params, a=1.0, b=2.0)

        grad_fn, clip_state = clipped_grad(
            loss, argnums=0, batch_argnums=1, clipping_norm=pg
        )

        noise_multiplier = 1.1
        grads, _ = grad_fn(params, torch.randn(4), state=clip_state)
        stddev = noise_multiplier * grads.max_norm
        assert isinstance(stddev, PerGroup)
        assert stddev.values == {"a": pytest.approx(1.1), "b": pytest.approx(2.2)}
