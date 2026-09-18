"""Tests for PerGroup type and per_group helper."""

import pytest
import torch

from opake.api.engine.clipping._per_group import per_group
from opake.exceptions import ConfigurationError
from opake.types import PerGroup


class TestPerGroup:
    """Tests for the PerGroup dataclass."""

    def test_basic_construction(self):
        pg = PerGroup(
            groups={"a": "g1", "b": "g2"},
            values={"g1": 1.0, "g2": 2.0},
        )
        # str keys normalize to one-segment ParamPaths
        assert pg.groups == {("a",): "g1", ("b",): "g2"}
        assert pg.values == {"g1": 1.0, "g2": 2.0}

    def test_effective(self):
        pg = PerGroup(
            groups={"a": "g1", "b": "g2"},
            values={"g1": 3.0, "g2": 4.0},
        )
        assert pg.effective == pytest.approx(5.0)  # sqrt(9 + 16)

    def test_effective_single_group(self):
        pg = PerGroup(groups={"a": "g1"}, values={"g1": 7.0})
        assert pg.effective == pytest.approx(7.0)

    def test_num_groups_counts_configured_groups(self):
        pg = PerGroup(
            groups={"a": "shared", "b": "shared", "c": "other"},
            values={"shared": 1.0, "other": 2.0},
        )
        assert pg.num_groups == 2

    def test_rmul(self):
        pg = PerGroup(groups={"a": "g1", "b": "g2"}, values={"g1": 1.0, "g2": 2.0})
        result = 3.0 * pg
        assert isinstance(result, PerGroup)
        assert result.values == {"g1": 3.0, "g2": 6.0}
        assert result.groups == pg.groups

    def test_mul(self):
        pg = PerGroup(groups={"a": "g1"}, values={"g1": 5.0})
        result = pg * 2.0
        assert result.values == {"g1": 10.0}

    def test_truediv(self):
        pg = PerGroup(groups={"a": "g1", "b": "g2"}, values={"g1": 6.0, "g2": 10.0})
        result = pg / 2.0
        assert result.values == {"g1": 3.0, "g2": 5.0}

    def test_for_path(self):
        pg = PerGroup(
            groups={"param_a": "attn", "param_b": "mlp"},
            values={"attn": 1.0, "mlp": 2.0},
        )
        assert pg.for_path("param_a") == 1.0
        assert pg.for_path("param_b") == 2.0

    def test_for_path_missing_raises(self):
        pg = PerGroup(groups={"a": "g1"}, values={"g1": 1.0})
        with pytest.raises(KeyError):
            pg.for_path("nonexistent")

    def test_frozen(self):
        pg = PerGroup(groups={"a": "g1"}, values={"g1": 1.0})
        with pytest.raises(AttributeError):
            pg.groups = {}

    @pytest.mark.parametrize(
        ("attribute", "key", "value"),
        [
            ("groups", ("a",), "g2"),
            ("values", "g1", 2.0),
        ],
    )
    def test_mappings_are_immutable(self, attribute, key, value):
        pg = PerGroup(groups={"a": "g1"}, values={"g1": 1.0})

        with pytest.raises(TypeError):
            getattr(pg, attribute)[key] = value

    def test_constructor_inputs_are_defensively_copied(self):
        groups = {"a": "g1"}
        values = {"g1": 1.0}
        pg = PerGroup(groups=groups, values=values)

        groups["a"] = "g2"
        values["g1"] = 2.0

        assert pg.groups == {("a",): "g1"}
        assert pg.values == {"g1": 1.0}

    def test_arithmetic_chain(self):
        """noise_multiplier * (clipping_norm / normalize_by) should work."""
        pg = PerGroup(groups={"a": "g1", "b": "g2"}, values={"g1": 2.0, "g2": 4.0})
        sensitivity = pg / 10.0  # normalize_by=10
        stddev = 1.1 * sensitivity  # noise_multiplier=1.1
        assert stddev.values["g1"] == pytest.approx(0.22)
        assert stddev.values["g2"] == pytest.approx(0.44)
        with pytest.raises(TypeError):
            stddev.values["g1"] = 1.0


class TestPerGroupHelper:
    """Tests for the per_group() factory."""

    def test_flat_dict_two_groups(self):
        params = {
            "layers.0.self_attn.q_proj.weight": torch.zeros(1),
            "layers.0.self_attn.k_proj.weight": torch.zeros(1),
            "layers.0.mlp.gate_proj.weight": torch.zeros(1),
            "layers.0.mlp.up_proj.weight": torch.zeros(1),
        }
        pg = per_group(params, self_attn=1.0, mlp=2.0)
        assert pg.groups[("layers.0.self_attn.q_proj.weight",)] == "self_attn"
        assert pg.groups[("layers.0.self_attn.k_proj.weight",)] == "self_attn"
        assert pg.groups[("layers.0.mlp.gate_proj.weight",)] == "mlp"
        assert pg.groups[("layers.0.mlp.up_proj.weight",)] == "mlp"
        assert pg.values == {"self_attn": 1.0, "mlp": 2.0}

    def test_patterns_dict_for_dotted_keys(self):
        params = {
            "layers.0.weight": torch.zeros(1),
            "layers.1.weight": torch.zeros(1),
        }
        pg = per_group(params, patterns={"layers.0": 0.5, "layers.1": 1.0})
        assert pg.groups[("layers.0.weight",)] == "layers.0"
        assert pg.groups[("layers.1.weight",)] == "layers.1"

    def test_patterns_merged_with_kwargs(self):
        params = {
            "layers.0.attn.weight": torch.zeros(1),
            "layers.0.mlp.weight": torch.zeros(1),
        }
        pg = per_group(params, patterns={"layers.0.attn": 0.5}, mlp=1.0)
        assert pg.groups[("layers.0.attn.weight",)] == "layers.0.attn"
        assert pg.groups[("layers.0.mlp.weight",)] == "mlp"

    def test_nested_dict(self):
        """Nested params compile to multi-segment optree paths."""
        params = {
            "layer1": {"attn": torch.zeros(1), "mlp": torch.zeros(1)},
            "layer2": {"attn": torch.zeros(1), "mlp": torch.zeros(1)},
        }
        pg = per_group(params, attn=1.0, mlp=2.0)
        assert pg.groups[("layer1", "attn")] == "attn"
        assert pg.groups[("layer1", "mlp")] == "mlp"
        assert pg.groups[("layer2", "attn")] == "attn"
        assert pg.groups[("layer2", "mlp")] == "mlp"

    def test_flat_named_parameters_are_one_segment_paths(self):
        params = {
            "layers.0.self_attn.weight": torch.zeros(1),
            "layers.0.mlp.weight": torch.zeros(1),
        }
        pg = per_group(params, self_attn=1.0, mlp=2.0)
        assert pg.groups[("layers.0.self_attn.weight",)] == "self_attn"
        assert ("layers", 0, "self_attn", "weight") not in pg.groups

    def test_no_match_raises(self):
        params = {"weight": torch.zeros(1)}
        with pytest.raises(ValueError, match="did not match any pattern"):
            per_group(params, attn=1.0)

    def test_multiple_matches_raises(self):
        params = {"self_attn_mlp": torch.zeros(1)}
        with pytest.raises(ValueError, match="matched multiple patterns"):
            per_group(params, self_attn=1.0, mlp=2.0)

    def test_no_patterns_raises(self):
        params = {"w": torch.zeros(1)}
        with pytest.raises(ValueError, match="At least one pattern"):
            per_group(params)

    def test_negative_value_raises(self):
        params = {"w": torch.zeros(1)}
        with pytest.raises(ValueError, match="must be positive"):
            per_group(params, w=-1.0)

    def test_zero_value_raises(self):
        params = {"w": torch.zeros(1)}
        with pytest.raises(ValueError, match="must be positive"):
            per_group(params, w=0.0)

    def test_many_fine_grained_groups(self):
        """Per-layer clipping norms."""
        params = {f"layers.{i}.weight": torch.zeros(1) for i in range(4)}
        norms = {f"layers.{i}": float(i + 1) for i in range(4)}
        pg = per_group(params, **norms)
        assert len(pg.values) == 4
        assert pg.for_path("layers.2.weight") == 3.0

    def test_fallback_catches_unmatched(self):
        """fallback catches params that don't match any explicit pattern."""
        params = {
            "layers.0.self_attn.q_proj.weight": torch.zeros(1),
            "layers.0.mlp.gate_proj.weight": torch.zeros(1),
            "layers.0.norm.weight": torch.zeros(1),
        }
        pg = per_group(params, self_attn=1.0, mlp=2.0, fallback=0.5)
        assert pg.groups[("layers.0.self_attn.q_proj.weight",)] == "self_attn"
        assert pg.groups[("layers.0.mlp.gate_proj.weight",)] == "mlp"
        assert pg.groups[("layers.0.norm.weight",)] == "fallback"
        assert pg.values["fallback"] == 0.5

    def test_unused_pattern_raises_even_with_fallback(self):
        params = {
            "layers.0.self_attn.q_proj.weight": torch.zeros(1),
            "layers.0.mlp.gate_proj.weight": torch.zeros(1),
        }
        with pytest.raises(
            ValueError,
            match=r"Patterns did not match any parameter: \['self_atten'\]",
        ):
            per_group(params, self_atten=1.0, fallback=0.5)

    def test_unused_pattern_error_includes_sample_display_paths(self):
        params = {
            "layers.0.self_attn.q_proj.weight": torch.zeros(1),
            "layers.0.mlp.gate_proj.weight": torch.zeros(1),
        }
        with pytest.raises(ConfigurationError) as exc_info:
            per_group(params, self_atten=1.0, fallback=0.5)

        assert "layers.0.self_attn.q_proj.weight" in str(exc_info.value)
        assert "allow_unused_patterns=True" in str(exc_info.value)

    def test_allow_unused_patterns_omits_unused_pattern_value(self):
        params = {"attn.weight": torch.zeros(1)}
        pg = per_group(
            params,
            attn=1.0,
            mlp=2.0,
            allow_unused_patterns=True,
        )
        assert pg.groups == {("attn.weight",): "attn"}
        assert pg.values == {"attn": 1.0}

    def test_fallback_not_in_values_when_unused(self):
        """If all params match explicit patterns, fallback is excluded from values."""
        params = {
            "attn.weight": torch.zeros(1),
            "mlp.weight": torch.zeros(1),
        }
        pg = per_group(params, attn=1.0, mlp=2.0, fallback=0.5)
        assert "fallback" not in pg.values
        assert pg.effective == pytest.approx((1.0**2 + 2.0**2) ** 0.5)

    def test_fallback_effective_includes_fallback(self):
        """Effective norm should include the fallback group."""
        params = {
            "attn.weight": torch.zeros(1),
            "unknown.weight": torch.zeros(1),
        }
        pg = per_group(params, attn=3.0, fallback=4.0)
        assert pg.effective == pytest.approx(5.0)  # sqrt(9 + 16)

    def test_no_match_error_suggests_fallback(self):
        """Error message suggests using fallback when param unmatched."""
        params = {"weight": torch.zeros(1)}
        with pytest.raises(ValueError, match="fallback"):
            per_group(params, attn=1.0)

    def test_fallback_only(self):
        """fallback as the sole group catches everything."""
        params = {"a": torch.zeros(1), "b": torch.zeros(1)}
        pg = per_group(params, fallback=1.0)
        assert pg.groups == {("a",): "fallback", ("b",): "fallback"}
        assert pg.values == {"fallback": 1.0}
