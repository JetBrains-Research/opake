"""Minimal NCCL DDP helpers + mp.spawn entrypoints (must live in this module for pickle)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist
from opaque_test_support import (
    cleanup_process_group as _cleanup_ddp,
)
from opaque_test_support import (
    setup_gloo as _setup_gloo,
)
from opaque_test_support import (
    setup_nccl as _setup_ddp,
)
from opaque_test_support import (
    spawn as _spawn,
)

_spawn_gloo = _spawn


@dataclass(frozen=True)
class _ScalarExactnessState:
    value: int


@dataclass(frozen=True)
class _CoreGlooState:
    total: int
    average: float
    minimum: int
    maximum: int
    product: int
    local_rank: int


@dataclass(frozen=True)
class _DuckTypedPerGroup:
    groups: dict[str, str]
    values: dict[str, float]


def _worker_reduce_scalar(rank: int, world_size: int, port: int) -> None:
    from opaque.api.engine.distributed._state import reduce_scalar

    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        value = float(rank + 1)
        synced = reduce_scalar(value, op="mean", device=device)
        expected_avg = sum(range(1, world_size + 1)) / world_size
        assert abs(synced - expected_avg) < 1e-5
    finally:
        _cleanup_ddp()


def _worker_all_reduce_values(rank: int, world_size: int, port: int) -> None:
    from opaque.distributed.collectives import all_reduce, all_reduce_

    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        base = torch.tensor([float(rank + 1), float(2 * (rank + 1))], device=device)

        result = all_reduce(base, op="sum")
        assert torch.allclose(
            base, torch.tensor([float(rank + 1), float(2 * (rank + 1))], device=device)
        )
        assert torch.allclose(result, torch.tensor([3.0, 6.0], device=device))

        averaged = base.clone()
        inplace_result = all_reduce_(averaged, op="mean")
        assert inplace_result is None
        assert torch.allclose(averaged, torch.tensor([1.5, 3.0], device=device))

        maximum = all_reduce(base, op="max")
        assert torch.allclose(maximum, torch.tensor([2.0, 4.0], device=device))

        minimum = all_reduce(base, op="min")
        assert torch.allclose(minimum, torch.tensor([1.0, 2.0], device=device))

        product = all_reduce(base, op="product")
        assert torch.allclose(product, torch.tensor([2.0, 8.0], device=device))
    finally:
        _cleanup_ddp()


def _worker_reduce_pytree(rank: int, world_size: int, port: int) -> None:
    from opaque.distributed.gradients import reduce_pytree, reduce_pytree_

    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        grads = {
            "w": torch.tensor([1.0, 2.0], device=device),
            "b": torch.tensor([0.5], device=device),
        }

        local_grads = {
            "w": grads["w"].clone(),
            "b": grads["b"].clone(),
        }

        result = reduce_pytree(grads, op="sum")
        assert torch.allclose(grads["w"], local_grads["w"])
        assert torch.allclose(grads["b"], local_grads["b"])
        assert torch.allclose(result["w"], torch.tensor([2.0, 4.0], device=device))
        assert torch.allclose(result["b"], torch.tensor([1.0], device=device))

        inplace_result = reduce_pytree_(grads, op="sum")
        assert inplace_result is None
        assert torch.allclose(grads["w"], torch.tensor([2.0, 4.0], device=device))
        assert torch.allclose(grads["b"], torch.tensor([1.0], device=device))
    finally:
        _cleanup_ddp()


def _worker_reduce_pytree_nested(rank: int, world_size: int, port: int) -> None:
    from opaque.distributed.gradients import reduce_pytree

    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        pytree = {
            "encoder": {
                "w": torch.tensor([[float(rank + 1), 1.0]], device=device),
                "b": torch.tensor([float(rank)], device=device),
            },
            "head": [torch.tensor([2.0 * (rank + 1)], device=device)],
        }

        original_w = pytree["encoder"]["w"].clone()
        original_b = pytree["encoder"]["b"].clone()
        original_head = pytree["head"][0].clone()

        result = reduce_pytree(pytree, op="sum")

        assert torch.allclose(pytree["encoder"]["w"], original_w)
        assert torch.allclose(pytree["encoder"]["b"], original_b)
        assert torch.allclose(pytree["head"][0], original_head)

        assert torch.allclose(
            result["encoder"]["w"],
            torch.tensor([[3.0, 2.0]], device=device),
        )
        assert torch.allclose(
            result["encoder"]["b"],
            torch.tensor([1.0], device=device),
        )
        assert torch.allclose(
            result["head"][0],
            torch.tensor([6.0], device=device),
        )
    finally:
        _cleanup_ddp()


def _worker_sync_profiler(rank: int, world_size: int, port: int) -> None:
    from opaque.api.engine.distributed._state import reduce_scalar
    from opaque.distributed import sync
    from opaque.profiling import PerfState, step_perf

    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        state = PerfState(device=device)

        with step_perf(device, batch_size=4) as perf:
            x = torch.randn(1024 + 512 * rank, device=device)
            _ = (x * x).sum()
        state = state.add(perf.result)

        synced_state = sync(state)
        assert synced_state is not state
        assert synced_state.num_steps == 1
        assert synced_state.last_step.batch_size == world_size * 4
        assert state.last_step.batch_size == 4
        assert synced_state.last_step.step_time_sec >= 0.0

        local_peak = float(synced_state.max_peak_memory_gb)
        peak_min = reduce_scalar(local_peak, op="min", device=device)
        peak_max = reduce_scalar(local_peak, op="max", device=device)
        assert abs(peak_max - peak_min) < 1e-6
    finally:
        _cleanup_ddp()


def _worker_second_moment_clip_gloo(rank: int, world_size: int, port: int) -> None:
    from opaque.distributed.gradients import reduce_pytree
    from opaque.types import ClippedPytree, SecondMomentClippingOutput

    _setup_gloo(rank, world_size, port)
    try:
        scale = 1.0 if rank == 0 else 10.0
        out = SecondMomentClippingOutput(
            grads=ClippedPytree({"w": torch.tensor([1.0, 2.0]) * scale}, max_norm=1.0),
            squared_grads=ClippedPytree(
                {"w": torch.tensor([3.0]) * scale}, max_norm=2.0
            ),
        )
        reduced = reduce_pytree(out, op="sum")
        assert isinstance(reduced, SecondMomentClippingOutput)
        assert torch.allclose(reduced.grads.pytree["w"], torch.tensor([11.0, 22.0]))
        assert torch.allclose(reduced.squared_grads.pytree["w"], torch.tensor([33.0]))
        assert abs(reduced.grads.max_norm - 1.0) < 1e-6
        assert abs(reduced.squared_grads.max_norm - 2.0) < 1e-6
    finally:
        _cleanup_ddp()


def _worker_second_moment_noise_gloo(rank: int, world_size: int, port: int) -> None:
    from opaque.distributed.gradients import reduce_pytree
    from opaque.types import NoisedPytree, SecondMomentNoiseOutput

    _setup_gloo(rank, world_size, port)
    try:
        scale = 1.0 if rank == 0 else 10.0
        out = SecondMomentNoiseOutput(
            noisy_grads=NoisedPytree(
                {"w": torch.tensor([1.0]) * scale},
                max_norm=1.0,
                noise_stddev=0.5,
            ),
            noisy_squared_grads=NoisedPytree(
                {"w": torch.tensor([2.0]) * scale},
                max_norm=2.0,
                noise_stddev=0.25,
            ),
        )
        reduced = reduce_pytree(out, op="sum")
        assert isinstance(reduced, SecondMomentNoiseOutput)
        assert torch.allclose(reduced.noisy_grads.pytree["w"], torch.tensor([11.0]))
        assert torch.allclose(
            reduced.noisy_squared_grads.pytree["w"], torch.tensor([22.0])
        )
        assert abs(reduced.noisy_grads.noise_stddev - 0.5 * (2.0**0.5)) < 1e-6
        assert abs(reduced.noisy_squared_grads.noise_stddev - 0.25 * (2.0**0.5)) < 1e-6
    finally:
        _cleanup_ddp()


def _paired_clipping_fixture(
    device: torch.device | str,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    params = {
        "linear": {
            "weight": torch.tensor([0.25, -0.5, 0.75], device=device),
        },
        "bias": torch.tensor(0.1, device=device),
    }
    x = torch.arange(24, dtype=torch.float32, device=device).reshape(8, 3) / 10.0
    y = torch.linspace(-0.4, 0.6, 8, device=device)
    return params, x, y


def _paired_clipping_loss(
    params: dict, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    prediction = x @ params["linear"]["weight"] + params["bias"]
    return (prediction - y).square()


def _worker_second_moment_clipping_parity_gloo(
    rank: int,
    world_size: int,
    port: int,
    out_path: str,
) -> None:
    from opaque.api.engine.clipping import clipped_grad
    from opaque.distributed import sum_gradients
    from opaque.pytree import tree_map
    from opaque.types import SecondMomentClippingOutput

    _setup_gloo(rank, world_size, port)
    try:
        params, x, y = _paired_clipping_fixture("cpu")
        grad_fn, clip_state = clipped_grad(
            _paired_clipping_loss,
            clipping_norm=0.7,
            batch_argnums=(1, 2),
            normalize_by=len(x),
            second_moment=True,
        )
        assert len(x) % world_size == 0
        shard_size = len(x) // world_size
        shard = slice(rank * shard_size, (rank + 1) * shard_size)
        local, _ = grad_fn(params, x[shard], y[shard], state=clip_state)
        reduced = sum_gradients(local)

        assert isinstance(reduced, SecondMomentClippingOutput)
        if rank == 0:
            torch.save(
                {
                    "grads": tree_map(
                        lambda tensor: tensor.cpu(), reduced.grads.pytree
                    ),
                    "squared_grads": tree_map(
                        lambda tensor: tensor.cpu(), reduced.squared_grads.pytree
                    ),
                    "max_norm": reduced.grads.max_norm,
                    "squared_max_norm": reduced.squared_grads.max_norm,
                },
                out_path,
            )
    finally:
        _cleanup_ddp()


def _worker_core_collectives_gloo(rank: int, world_size: int, port: int) -> None:
    """Exercise backend-neutral engine primitives through a live Gloo group."""
    from opaque.api.engine.distributed._state import gather_tensors, sync_object
    from opaque.distributed import (
        all_reduce,
        barrier,
        gather_for_metrics,
        get_rank,
        get_world_size,
        is_distributed,
        is_main_process,
        num_processes,
        process_index,
        reduce_pytree,
        sync,
    )
    from opaque.distributed.collectives import all_reduce_
    from opaque.profiling import PerfState, step_perf
    from opaque.types import ClippedPytree, NoisedPytree, PerGroup

    _setup_gloo(rank, world_size, port)
    try:
        assert is_distributed()
        assert get_rank() == process_index() == rank
        assert get_world_size() == num_processes() == world_size
        assert is_main_process() is (rank == 0)

        base = torch.tensor([float(rank + 1), float(2 * (rank + 1))])
        expected = {
            "sum": torch.tensor([3.0, 6.0]),
            "mean": torch.tensor([1.5, 3.0]),
            "max": torch.tensor([2.0, 4.0]),
            "min": torch.tensor([1.0, 2.0]),
            "product": torch.tensor([2.0, 8.0]),
        }
        for op, value in expected.items():
            torch.testing.assert_close(all_reduce(base, op=op), value)
        torch.testing.assert_close(
            base, torch.tensor([float(rank + 1), float(2 * (rank + 1))])
        )

        inplace = base.clone()
        assert all_reduce_(inplace, op="mean") is None
        torch.testing.assert_close(inplace, expected["mean"])

        gathered_scalars = gather_for_metrics(torch.tensor(float(rank)))
        torch.testing.assert_close(gathered_scalars, torch.tensor([0.0, 1.0]))
        gathered_ragged = gather_tensors(torch.full((rank + 1, 1), float(rank)))
        torch.testing.assert_close(
            gathered_ragged,
            torch.tensor([[0.0], [1.0], [1.0]]),
        )

        local_tree = {
            "weight": torch.tensor([float(rank + 1)]),
            "bias": torch.tensor([float(rank)]),
        }
        reduced_tree = reduce_pytree(local_tree, op="sum")
        torch.testing.assert_close(reduced_tree["weight"], torch.tensor([3.0]))
        torch.testing.assert_close(reduced_tree["bias"], torch.tensor([1.0]))
        torch.testing.assert_close(
            local_tree["weight"], torch.tensor([float(rank + 1)])
        )

        clipped = ClippedPytree({"w": torch.tensor([float(rank + 1)])}, max_norm=2.0)
        averaged = reduce_pytree(clipped, op="mean")
        assert isinstance(averaged, ClippedPytree)
        assert averaged.max_norm == 1.0
        torch.testing.assert_close(averaged.pytree["w"], torch.tensor([1.5]))

        noised = NoisedPytree(
            {"w": torch.tensor([float(rank + 1)])},
            max_norm=2.0,
            noise_stddev=0.5,
        )
        summed_noised = reduce_pytree(noised, op="sum")
        assert isinstance(summed_noised, NoisedPytree)
        assert summed_noised.max_norm == 2.0
        assert summed_noised.noise_stddev == pytest.approx(0.5 * 2**0.5)
        torch.testing.assert_close(summed_noised.pytree["w"], torch.tensor([3.0]))

        per_group_norm = PerGroup(
            groups={"w": "weights"},
            values={"weights": 1.0},
        )
        per_group_reduced = reduce_pytree(
            ClippedPytree(
                {"w": torch.tensor([float(rank + 1)])},
                max_norm=per_group_norm,
            ),
            op="sum",
        )
        assert per_group_reduced.max_norm == per_group_norm
        torch.testing.assert_close(per_group_reduced.pytree["w"], torch.tensor([3.0]))

        mismatched_per_group_norm = PerGroup(
            groups=per_group_norm.groups,
            values={"weights": float(rank + 1)},
        )
        with pytest.raises(
            RuntimeError,
            match=r"ClippedPytree\.max_norm\.values\['weights'\] mismatch",
        ):
            reduce_pytree(
                ClippedPytree(
                    {"w": torch.tensor([float(rank + 1)])},
                    max_norm=mismatched_per_group_norm,
                ),
                op="sum",
            )

        reordered_mismatch = PerGroup(
            groups={"w": "weights", "b": "biases"},
            values=(
                {"weights": 1.0, "biases": 2.0}
                if rank == 0
                else {"biases": 1.0, "weights": 2.0}
            ),
        )
        with pytest.raises(
            RuntimeError,
            match=r"ClippedPytree\.max_norm\.values\['(?:biases|weights)'\] mismatch",
        ):
            reduce_pytree(
                ClippedPytree(
                    {
                        "w": torch.tensor([float(rank + 1)]),
                        "b": torch.tensor([float(rank + 1)]),
                    },
                    max_norm=reordered_mismatch,
                ),
                op="sum",
            )

        reordered_duck_typed_metadata = _DuckTypedPerGroup(
            groups={"w": "weights", "b": "biases"},
            values=(
                {"weights": 1.0, "biases": 2.0}
                if rank == 0
                else {"biases": 1.0, "weights": 2.0}
            ),
        )
        with pytest.raises(
            RuntimeError,
            match=r"ClippedPytree\.max_norm\.values\['(?:biases|weights)'\] mismatch",
        ):
            reduce_pytree(
                ClippedPytree(
                    {
                        "w": torch.tensor([float(rank + 1)]),
                        "b": torch.tensor([float(rank + 1)]),
                    },
                    max_norm=reordered_duck_typed_metadata,
                ),
                op="sum",
            )

        mixed_metadata = per_group_norm if rank == 0 else 1.0
        with pytest.raises(
            RuntimeError,
            match=r"ClippedPytree\.max_norm\.kind mismatch",
        ):
            reduce_pytree(
                ClippedPytree(
                    {"w": torch.tensor([float(rank + 1)])},
                    max_norm=mixed_metadata,
                ),
                op="sum",
            )

        synced = sync_object(
            _CoreGlooState(
                total=rank + 1,
                average=float(rank + 1),
                minimum=rank + 1,
                maximum=rank + 1,
                product=rank + 1,
                local_rank=rank,
            ),
            {
                "total": "sum",
                "average": "mean",
                "minimum": "min",
                "maximum": "max",
                "product": "product",
                "local_rank": "local",
            },
        )
        assert synced == _CoreGlooState(
            total=3,
            average=1.5,
            minimum=1,
            maximum=2,
            product=2,
            local_rank=rank,
        )

        perf_state = PerfState(torch.device("cpu"))
        with step_perf(torch.device("cpu"), batch_size=rank + 1) as step:
            _ = torch.arange(64, dtype=torch.float32).square().sum()
        perf_state = perf_state.add(step.perf)
        synced_perf_state = sync(perf_state)
        assert synced_perf_state is not perf_state
        assert synced_perf_state.num_steps == 1
        assert synced_perf_state.total_samples == 3
        assert synced_perf_state.last_step is not None
        assert synced_perf_state.last_step.batch_size == 3
        assert synced_perf_state.last_step.step_time_sec >= 0.0

        barrier()
        token = torch.tensor([1.0])
        dist.all_reduce(token, op=dist.ReduceOp.SUM)
        assert token.item() == float(world_size)
    finally:
        _cleanup_ddp()


def _worker_gather_optional_ragged(rank: int, world_size: int, port: int) -> None:
    from opaque.api.engine.distributed._state import gather_pytree, gather_tensors

    _setup_gloo(rank, world_size, port)
    try:
        optional = (
            None if rank == 0 else {"value": torch.tensor([[1.0, 2.0]]), "aux": None}
        )
        gathered_optional = gather_pytree(optional)
        assert torch.equal(
            gathered_optional["value"],
            torch.tensor([[1.0, 2.0]]),
        )
        assert gathered_optional["aux"] is None

        ragged = torch.full((rank * 2, 2), float(rank))
        gathered_ragged = gather_tensors(ragged)
        assert torch.equal(
            gathered_ragged,
            torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
        )

        incompatible_dtype = (
            torch.ones((1, 2), dtype=torch.float32)
            if rank == 0
            else torch.ones((1, 2), dtype=torch.int64)
        )
        with pytest.raises(TypeError, match="matching dtypes"):
            gather_tensors(incompatible_dtype)

        incompatible_shape = (
            torch.ones((1, 2), dtype=torch.float32)
            if rank == 0
            else torch.ones((1, 3), dtype=torch.float32)
        )
        with pytest.raises(ValueError, match="non-concatenated dimensions"):
            gather_tensors(incompatible_shape)

        incompatible_rank = (
            torch.ones((1, 2), dtype=torch.float32)
            if rank == 0
            else torch.ones((1, 2, 1), dtype=torch.float32)
        )
        with pytest.raises(ValueError, match="matching tensor ranks"):
            gather_tensors(incompatible_rank)

        incompatible_structure = (
            {"value": torch.ones((1, 2))} if rank == 0 else [torch.ones((1, 2))]
        )
        with pytest.raises(TypeError, match="matching pytree structures"):
            gather_pytree(incompatible_structure)
    finally:
        _cleanup_ddp()


def _worker_scalar_exactness_gloo(rank: int, world_size: int, port: int) -> None:
    from opaque.api.engine.distributed._state import (
        assert_scalar_equal,
        reduce_scalar,
        sync_object,
    )

    _setup_gloo(rank, world_size, port)
    try:
        integer_value = 2**24 + rank
        assert reduce_scalar(integer_value, op="min") == 2**24
        assert reduce_scalar(integer_value, op="max") == 2**24 + 1
        assert reduce_scalar(integer_value, op="sum") == 2 * 2**24 + 1

        float_value = 1.0 + rank * 2**-30
        assert (
            reduce_scalar(
                float_value,
                op="max",
                compute_dtype=torch.float64,
            )
            == 1.0 + 2**-30
        )

        with pytest.raises(RuntimeError, match="integer"):
            assert_scalar_equal(integer_value, name="integer")
        with pytest.raises(RuntimeError, match="float64"):
            assert_scalar_equal(
                float_value,
                name="float64",
                atol=0.0,
                rtol=0.0,
                compute_dtype=torch.float64,
            )
        with pytest.raises(RuntimeError, match=r"_ScalarExactnessState\.value"):
            sync_object(
                _ScalarExactnessState(integer_value),
                field_ops={"value": "assert_equal"},
            )
    finally:
        _cleanup_ddp()


def _worker_sync_aux_empty_batch(rank: int, world_size: int, port: int) -> None:
    """Rank 0 draws an empty batch; rank 1 draws examples. Must not hang."""
    from opaque.api.engine.clipping._clipped_grad import ClippedGradAux
    from opaque.api.engine.clipping._distributed import sync_clipped_grad_aux
    from opaque.distributed import sync

    _setup_gloo(rank, world_size, port)
    try:
        if rank == 0:
            aux = ClippedGradAux(
                loss_values=torch.empty(0),
                grad_norms=torch.empty(0),
                clipped_grad_norms=torch.empty(0),
                loss_aux=None,
                clipping_rate=0.0,
                batch_size=0,
                group_norms=None,
            )
        else:
            aux = ClippedGradAux(
                loss_values=torch.tensor([1.0, 2.0, 3.0]),
                grad_norms=torch.tensor([0.4, 1.2, 0.8]),
                clipped_grad_norms=torch.tensor([0.4, 1.0, 0.8]),
                loss_aux=None,
                clipping_rate=1.0 / 3.0,
                batch_size=3,
                group_norms=None,
            )

        synced = sync_clipped_grad_aux(aux)
        # Also exercise the type-dispatched sync path used by trainers.
        synced2 = sync(aux)

        assert synced.batch_size == 3
        assert synced2.batch_size == 3
        assert synced.grad_norms.shape[0] == 3
        assert abs(synced.clipping_rate - (1.0 / 3.0)) < 1e-5
        # A follow-up collective must still succeed (proves no desync).
        token = torch.tensor([float(rank + 1)])
        dist.all_reduce(token, op=dist.ReduceOp.SUM)
        assert abs(token.item() - sum(range(1, world_size + 1))) < 1e-5
    finally:
        _cleanup_ddp()


def _worker_sync_aux_empty_vs_per_group(rank: int, world_size: int, port: int) -> None:
    """Empty rank has group_norms=None; nonempty has per-group dict.

    After ParamPath-keyed PerGroup, aux ``group_norms`` are still keyed by
    group name, but empty batches still omit the dict. Sync must not hang.
    """
    from opaque.api.engine.clipping._clipped_grad import ClippedGradAux
    from opaque.api.engine.clipping._distributed import sync_clipped_grad_aux

    _setup_gloo(rank, world_size, port)
    try:
        if rank == 0:
            aux = ClippedGradAux(
                loss_values=torch.empty(0),
                grad_norms=torch.empty(0),
                clipped_grad_norms=torch.empty(0),
                loss_aux=None,
                clipping_rate=0.0,
                batch_size=0,
                group_norms=None,
            )
        else:
            aux = ClippedGradAux(
                loss_values=torch.tensor([1.0, 2.0]),
                grad_norms=torch.tensor([0.5, 1.5]),
                clipped_grad_norms=torch.tensor([0.5, 1.0]),
                loss_aux=None,
                clipping_rate=0.5,
                batch_size=2,
                group_norms={
                    "attn": torch.tensor([0.5, 1.5]),
                    "mlp": torch.tensor([0.2, 0.3]),
                },
            )

        synced = sync_clipped_grad_aux(aux)
        assert synced.batch_size == 2
        assert synced.group_norms is not None
        assert set(synced.group_norms) == {"attn", "mlp"}
        assert synced.group_norms["attn"].shape[0] == 2
        token = torch.tensor([1.0])
        dist.all_reduce(token, op=dist.ReduceOp.SUM)
        assert abs(token.item() - float(world_size)) < 1e-5
    finally:
        _cleanup_ddp()


def _worker_sync_schema_contracts_gloo(rank: int, world_size: int, port: int) -> None:
    """Exercise schema mismatches without leaving Gloo ranks desynchronized."""
    from opaque.api.engine.clipping._clipped_grad import ClippedGradAux
    from opaque.api.engine.clipping._distributed import sync_clipped_grad_aux
    from opaque.api.engine.distributed._state import reduce_scalar, sync_object
    from opaque.api.engine.profiling._distributed import (
        sync_perf_state,
        sync_perf_tracker,
    )
    from opaque.profiling import PerfState, PerfTracker, StepPerf

    @dataclass(frozen=True)
    class _CompatibleAux(ClippedGradAux):
        pass

    @dataclass(frozen=True)
    class _UnsupportedAux(ClippedGradAux):
        extra: torch.Tensor | None = None

    _setup_gloo(rank, world_size, port)
    try:
        aux = _CompatibleAux(
            loss_values=torch.tensor([float(rank)]),
            grad_norms=torch.tensor([1.0]),
            clipped_grad_norms=torch.tensor([1.0]),
            clipping_rate=0.0,
            batch_size=1,
        )
        synced_aux = sync_clipped_grad_aux(aux)
        assert type(synced_aux) is _CompatibleAux
        assert synced_aux.batch_size == world_size

        with pytest.raises(TypeError, match="synchronization schema"):
            sync_clipped_grad_aux(_UnsupportedAux(extra=torch.tensor([float(rank)])))

        with pytest.raises(RuntimeError, match="clipping_rate presence mismatch"):
            sync_clipped_grad_aux(
                ClippedGradAux(
                    grad_norms=torch.empty(0),
                    clipped_grad_norms=torch.empty(0),
                    clipping_rate=None if rank == 0 else 0.0,
                    batch_size=0,
                )
            )

        ordered_tracker = PerfTracker(torch.device("cpu"), warmup_steps=0)
        for name in ("train", "eval") if rank == 0 else ("eval", "train"):
            stage = ordered_tracker[name]
            stage.num_steps = 1
        synced_tracker = sync_perf_tracker(ordered_tracker)
        assert tuple(synced_tracker.stages) == ("eval", "train")

        mismatched_tracker = PerfTracker(torch.device("cpu"), warmup_steps=0)
        mismatched_tracker["train" if rank == 0 else "eval"].num_steps = 1
        with pytest.raises(RuntimeError, match="stage schema mismatch"):
            sync_perf_tracker(mismatched_tracker)

        state = PerfState(
            device=torch.device("cpu"),
            last_step=(
                None if rank == 0 else StepPerf(step_time_sec=1.0, batch_size=1)
            ),
        )
        with pytest.raises(RuntimeError, match="StepPerf presence mismatch"):
            sync_perf_state(state)

        # A field callable that fails after reducing must reduce exactly once,
        # even when its signature would also accept a single argument.
        reductions: list[int] = []

        def _reduce_then_fail(value: int, device: torch.device | None = None) -> int:
            reductions.append(reduce_scalar(value, op="sum", device=device))
            raise TypeError("field callable failed after a collective")

        with pytest.raises(TypeError, match="failed after a collective"):
            sync_object(
                _ScalarExactnessState(1), field_ops={"value": _reduce_then_fail}
            )
        assert reductions == [world_size]

        token = torch.tensor([1.0])
        dist.all_reduce(token, op=dist.ReduceOp.SUM)
        assert token.item() == float(world_size)
    finally:
        _cleanup_ddp()


def _worker_per_group_clipped_grad_one_rank_empty_gloo(
    rank: int, world_size: int, port: int
) -> None:
    """Per-group clipped_grad with rank 0 empty: sync(aux) must not desync (#805)."""
    from opaque.api.engine.clipping import clipped_grad
    from opaque.api.engine.clipping._per_group import per_group
    from opaque.distributed import sync
    from opaque.types import PerGroup

    _setup_gloo(rank, world_size, port)
    try:

        def loss_fn(params, x, y):
            prediction = x @ params["weight"] + params["bias"]
            return ((prediction - y) ** 2).mean()

        params = {
            "weight": torch.tensor([0.25, -0.5]),
            "bias": torch.tensor([0.1]),
        }
        bounds = per_group(params, weight=1.0, bias=0.5)
        grad_fn, clip_state = clipped_grad(
            loss_fn,
            batch_argnums=(1, 2),
            clipping_norm=bounds,
            return_aux=True,
        )

        local_batch_size = 0 if rank == 0 else 3
        x = torch.arange(local_batch_size * 2, dtype=torch.float32).reshape(
            local_batch_size, 2
        )
        y = torch.linspace(0.0, 1.0, local_batch_size)

        (local_grads, local_aux), clip_state = grad_fn(params, x, y, state=clip_state)
        assert local_aux.clipping_rate is None
        assert isinstance(local_grads.max_norm, PerGroup)

        synced_aux = sync(local_aux)
        # Trainer-style dispatch through sync(*payloads) must agree.
        synced_aux2 = sync(local_aux)

        assert synced_aux.clipping_rate is None
        assert synced_aux2.clipping_rate is None
        assert synced_aux.batch_size == 3
        assert synced_aux.grad_norms.shape[0] == 3
        assert synced_aux.group_norms is not None
        assert set(synced_aux.group_norms) == {"weight", "bias"}
        assert synced_aux.group_norms["weight"].shape[0] == 3

        # A follow-up collective must still succeed (proves no desync).
        token = torch.tensor([float(rank + 1)])
        dist.all_reduce(token, op=dist.ReduceOp.SUM)
        assert abs(token.item() - sum(range(1, world_size + 1))) < 1e-5
    finally:
        _cleanup_ddp()
