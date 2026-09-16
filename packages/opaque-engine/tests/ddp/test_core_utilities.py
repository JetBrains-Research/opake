"""Tests for opaque.distributed core utilities.

These tests focus on non-distributed behavior (when torch.distributed is not initialized).
For actual multi-device distributed tests, see ``test_collectives.py`` and
``test_profiler_sync.py`` in the ``ddp/`` folder.
"""

from dataclasses import dataclass

import pytest
import torch

from opaque.api.engine.distributed._state import (
    reduce_scalar,
    sync_object,
)

# ``_reduced_metadata`` is a private helper exercised directly by the
# tests below; reach for it via the impl-side path.
from opaque.api.engine.distributed.gradients import (
    _assert_public_metadata_equal,
)
from opaque.api.engine.distributed.gradients import (
    _reduced_metadata as _reduced_metadata,
)
from opaque.distributed import get_rank, get_world_size, is_distributed, sum_gradients
from opaque.distributed.collectives import all_reduce, all_reduce_, barrier
from opaque.distributed.gradients import (  # noqa: F401
    reduce_pytree,
    reduce_pytree_,
    sum_gradients_,
)
from opaque.exceptions import ConfigurationError
from opaque.types import (
    ClippedPytree,
    NoisedPytree,
    PerGroup,
    SecondMomentNoiseOutput,
)


class TestNonDistributed:
    """Tests for behavior when distributed is not initialized."""

    def test_is_distributed_false(self):
        """is_distributed() returns False when not initialized."""
        # Note: In CI, distributed might be initialized, so we just check it's callable
        result = is_distributed()
        assert isinstance(result, bool)

    def test_get_rank_returns_zero(self):
        """get_rank() returns 0 when not initialized."""
        # In non-distributed mode, rank is always 0
        # Note: If dist is initialized, this might return actual rank
        result = get_rank()
        assert isinstance(result, int)
        assert result >= 0

    def test_get_world_size_returns_one(self):
        """get_world_size() returns 1 when not initialized."""
        # In non-distributed mode, world size is always 1
        # Note: If dist is initialized, this might return actual world size
        result = get_world_size()
        assert isinstance(result, int)
        assert result >= 1

    def test_all_reduce_raises_without_init(self):
        """all_reduce() raises RuntimeError when not initialized."""
        tensor = torch.tensor([1.0, 2.0, 3.0])

        # Skip if distributed is initialized (e.g., in multi-GPU CI)
        if is_distributed():
            pytest.skip("Distributed already initialized")

        with pytest.raises(RuntimeError, match="not initialized"):
            all_reduce(tensor)

    def test_all_reduce_inplace_raises_without_init(self):
        """all_reduce_() raises RuntimeError when not initialized."""
        tensor = torch.tensor([1.0, 2.0, 3.0])

        if is_distributed():
            pytest.skip("Distributed already initialized")

        with pytest.raises(RuntimeError, match="not initialized"):
            all_reduce_(tensor)

    def test_barrier_no_op_without_init(self):
        """barrier() is no-op when not initialized."""
        # Skip if distributed is initialized
        if is_distributed():
            pytest.skip("Distributed already initialized")

        # Should not raise
        barrier()

    def test_reduce_scalar_preserves_integer_type_without_distribution(self):
        value = 2**24 + 1

        result = reduce_scalar(value, op="sum")

        assert isinstance(result, int)
        assert result == value

    def test_reduce_scalar_integer_mean_returns_float_without_distribution(self):
        result = reduce_scalar(2**24 + 1, op="mean")

        assert isinstance(result, float)
        assert result == float(2**24 + 1)

    @pytest.mark.parametrize("bad_dtype", [torch.int64, torch.bool, torch.complex64])
    def test_reduce_scalar_rejects_nonfloating_compute_dtype(self, bad_dtype):
        with pytest.raises(TypeError, match="real floating-point"):
            reduce_scalar(1.0, compute_dtype=bad_dtype)

    def test_reduce_scalar_rejects_compute_dtype_for_integer_values(self):
        with pytest.raises(TypeError, match="only supported for floating-point"):
            reduce_scalar(1, compute_dtype=torch.float64)


class TestAllReduceValidation:
    """Tests for all_reduce/all_reduce_ parameter validation."""

    def test_invalid_op_raises(self):
        """all_reduce() and all_reduce_() raise ValueError for invalid op."""
        tensor = torch.tensor([1.0])

        with pytest.raises(ValueError, match="Invalid reduction operation"):
            all_reduce(tensor, op="invalid_op")
        with pytest.raises(ValueError, match="Invalid reduction operation"):
            all_reduce_(tensor, op="invalid_op")

    def test_valid_operations(self):
        """all_reduce() and all_reduce_() accept all valid operations."""
        valid_ops = ["sum", "mean", "max", "min", "product"]

        # Just test that these don't raise ValueError during validation
        # (will still raise RuntimeError if not initialized)
        for op in valid_ops:
            tensor = torch.tensor([1.0])

            if is_distributed():
                # Actually execute if initialized
                all_reduce(tensor, op=op)
                all_reduce_(tensor, op=op)
                continue

            # Just check it gets past parameter validation
            with pytest.raises(RuntimeError, match="not initialized"):
                all_reduce(tensor, op=op)
            with pytest.raises(RuntimeError, match="not initialized"):
                all_reduce_(tensor, op=op)


class TestSyncObjectSchema:
    @dataclass(frozen=True)
    class _State:
        count: int
        label: str

    @dataclass(frozen=True)
    class _BooleanState:
        enabled: bool

    def test_requires_a_complete_field_schema(self):
        with pytest.raises(ValueError, match="missing fields"):
            sync_object(self._State(count=1, label="local"), {"count": "sum"})

    def test_rejects_unknown_schema_fields(self):
        with pytest.raises(ValueError, match="unknown fields"):
            sync_object(
                self._State(count=1, label="local"),
                {"count": "sum", "label": "local", "missing": "local"},
            )

    def test_callable_cannot_replace_boolean_field(self, monkeypatch):
        import opaque.api.engine.distributed._state as state_module

        monkeypatch.setattr(state_module, "is_distributed", lambda: True)
        with pytest.raises(TypeError, match="cannot update a bool field"):
            sync_object(
                self._BooleanState(enabled=True), {"enabled": lambda _value, _device: 1}
            )

    def test_rejects_callable_with_unsupported_signature(self):
        with pytest.raises(
            ConfigurationError, match=r"callable as fn\(value, device\)"
        ):
            sync_object(self._BooleanState(enabled=True), {"enabled": lambda _value: 1})

    def test_rejects_field_op_that_is_neither_name_nor_callable(self):
        with pytest.raises(ValueError, match="operation name or a callable"):
            sync_object(self._BooleanState(enabled=True), {"enabled": 1})

    def test_callable_signature_is_validated_before_any_field_runs(self, monkeypatch):
        import opaque.api.engine.distributed._state as state_module

        monkeypatch.setattr(state_module, "is_distributed", lambda: True)
        reduced: list[int] = []

        with pytest.raises(
            ConfigurationError, match=r"callable as fn\(value, device\)"
        ):
            sync_object(
                self._State(count=1, label="local"),
                {
                    "count": lambda value, _device: reduced.append(value) or 1,
                    "label": lambda _value: 1,
                },
            )

        assert reduced == []

    def test_callable_error_propagates_without_a_second_call(self, monkeypatch):
        import opaque.api.engine.distributed._state as state_module

        monkeypatch.setattr(state_module, "is_distributed", lambda: True)
        calls: list[int] = []
        failure = TypeError("callback body failed")

        def _failing(value, _device):
            calls.append(value)
            raise failure

        with pytest.raises(TypeError) as excinfo:
            sync_object(
                self._State(count=1, label="local"),
                {"count": _failing, "label": "local"},
            )

        assert excinfo.value is failure
        assert calls == [1]


class TestBoundedGradientAggregation:
    """Tests for wrapper-aware gradient aggregation outside distributed mode."""

    def test_sum_gradients_preserves_clipped_pytree(self):
        gradients = ClippedPytree({"w": torch.tensor([1.0, 2.0])}, max_norm=0.5)

        reduced = sum_gradients(gradients)

        assert isinstance(reduced, ClippedPytree)
        assert not isinstance(reduced, NoisedPytree)
        assert reduced.max_norm == gradients.max_norm
        assert reduced.pytree is not gradients.pytree
        torch.testing.assert_close(reduced.pytree["w"], gradients.pytree["w"])

    def test_sum_gradients_inplace_preserves_clipped_pytree(self):
        gradients = ClippedPytree({"w": torch.tensor([1.0, 2.0])}, max_norm=0.5)

        result = sum_gradients_(gradients)

        assert result is None
        assert gradients.max_norm == 0.5
        torch.testing.assert_close(gradients.pytree["w"], torch.tensor([1.0, 2.0]))

    def test_mean_gradients_preserves_clipped_pytree_outside_distributed(self):
        gradients = ClippedPytree({"w": torch.tensor([1.0])}, max_norm=0.5)

        reduced = reduce_pytree(gradients, op="mean")

        assert isinstance(reduced, ClippedPytree)
        assert reduced.max_norm == gradients.max_norm
        torch.testing.assert_close(reduced.pytree["w"], gradients.pytree["w"])

    def test_unsupported_clipped_reduction_raises(self):
        gradients = ClippedPytree({"w": torch.tensor([1.0])}, max_norm=0.5)

        with pytest.raises(TypeError, match="supports op='sum' or op='mean'"):
            reduce_pytree(gradients, op="max")

    def test_sum_gradients_preserves_noisy_pytree_outside_distributed(self):
        gradients = NoisedPytree(
            {"w": torch.tensor([1.0])}, max_norm=0.5, noise_stddev=1.0
        )

        reduced = sum_gradients(gradients)

        assert isinstance(reduced, NoisedPytree)
        assert reduced.max_norm == gradients.max_norm
        assert reduced.noise_stddev == gradients.noise_stddev
        torch.testing.assert_close(reduced.pytree["w"], gradients.pytree["w"])

    def test_clipped_metadata_scales_for_distributed_mean(self):
        gradients = ClippedPytree({"w": torch.tensor([1.0])}, max_norm=2.0)

        reduced = _reduced_metadata(gradients, "mean", world_size=4)

        assert isinstance(reduced, ClippedPytree)
        assert reduced.max_norm == pytest.approx(0.5)

    def test_noisy_metadata_scales_for_distributed_sum_and_mean(self):
        gradients = NoisedPytree(
            {"w": torch.tensor([1.0])}, max_norm=2.0, noise_stddev=0.5
        )

        summed = _reduced_metadata(gradients, "sum", world_size=4)
        averaged = _reduced_metadata(gradients, "mean", world_size=4)

        assert isinstance(summed, NoisedPytree)
        assert summed.max_norm == pytest.approx(2.0)
        assert summed.noise_stddev == pytest.approx(1.0)
        assert isinstance(averaged, NoisedPytree)
        assert averaged.max_norm == pytest.approx(0.5)
        assert averaged.noise_stddev == pytest.approx(0.25)

    def test_sum_gradients_preserves_second_moment_noise_output(self):
        out = SecondMomentNoiseOutput(
            noisy_grads=NoisedPytree(
                {"w": torch.tensor([1.0])}, max_norm=0.5, noise_stddev=0.2
            ),
            noisy_squared_grads=NoisedPytree(
                {"w": torch.tensor([4.0])}, max_norm=1.0, noise_stddev=0.3
            ),
        )
        reduced = sum_gradients(out)
        assert isinstance(reduced, SecondMomentNoiseOutput)
        assert reduced.noisy_grads.noise_stddev == pytest.approx(0.2)
        assert reduced.noisy_squared_grads.noise_stddev == pytest.approx(0.3)
        torch.testing.assert_close(
            reduced.noisy_grads.pytree["w"], out.noisy_grads.pytree["w"]
        )

    def test_reduce_pytree_rejects_non_tensor_leaves(self):
        with pytest.raises(TypeError, match="tensor leaves"):
            reduce_pytree({"w": 1.0})


class TestDistributedMetadataValidation:
    class _DuckTypedPerGroup:
        def __init__(self, values: dict[str, float]) -> None:
            self.groups = {"w": "weights", "b": "biases"}
            self.values = values

    @staticmethod
    def _record_metadata_calls(monkeypatch):
        import opaque.api.engine.distributed.gradients as gradients_module

        calls: list[tuple[str, str, object]] = []
        monkeypatch.setattr(gradients_module, "is_distributed", lambda: True)
        monkeypatch.setattr(
            gradients_module,
            "_assert_object_equal",
            lambda value, *, name: calls.append(("object", name, value)),
        )
        monkeypatch.setattr(
            gradients_module,
            "assert_scalar_equal",
            lambda value, *, name: calls.append(("scalar", name, value)),
        )
        return calls

    def test_per_group_metadata_uses_sorted_group_names(self, monkeypatch):
        calls = self._record_metadata_calls(monkeypatch)
        metadata = PerGroup(
            groups={"w": "weights", "b": "biases"},
            values={"weights": 2.0, "biases": 1.0},
        )

        _assert_public_metadata_equal(metadata, name="max_norm")

        assert calls == [
            ("object", "max_norm.kind", "per_group"),
            ("object", "max_norm.groups", {("w",): "weights", ("b",): "biases"}),
            ("object", "max_norm.values.keys", {"weights", "biases"}),
            ("scalar", "max_norm.values['biases']", 1.0),
            ("scalar", "max_norm.values['weights']", 2.0),
        ]

    def test_duck_typed_per_group_metadata_uses_sorted_group_names(self, monkeypatch):
        calls = self._record_metadata_calls(monkeypatch)
        metadata = self._DuckTypedPerGroup({"weights": 2.0, "biases": 1.0})

        _assert_public_metadata_equal(metadata, name="max_norm")

        assert calls == [
            ("object", "max_norm.kind", "scalar"),
            ("object", "max_norm.groups", {"w": "weights", "b": "biases"}),
            ("object", "max_norm.values.keys", {"weights", "biases"}),
            ("scalar", "max_norm.values['biases']", 1.0),
            ("scalar", "max_norm.values['weights']", 2.0),
        ]

    def test_none_and_scalar_metadata_follow_symmetric_paths(self, monkeypatch):
        calls = self._record_metadata_calls(monkeypatch)

        _assert_public_metadata_equal(None, name="optional")
        _assert_public_metadata_equal(1.5, name="scalar")

        assert calls == [
            ("object", "optional.kind", "none"),
            ("object", "optional", None),
            ("object", "scalar.kind", "scalar"),
            ("scalar", "scalar", 1.5),
        ]
