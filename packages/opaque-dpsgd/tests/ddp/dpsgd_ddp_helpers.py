"""Minimal NCCL DDP helpers + mp.spawn entrypoints (must live in this module for pickle)."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn as nn
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

from opaque.distributed import sum_gradients, sync
from opaque.dpsgd.clipping import adaptive_clipped_grad, clipped_grad
from opaque.dpsgd.noise import gaussian_noise
from opaque.functional import make_functional
from opaque.pytree import tree_leaves
from opaque.random import key
from opaque.types import clipped

_spawn_gloo = _spawn


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def _fixed_sd_clipped() -> dict[str, torch.Tensor]:
    torch.manual_seed(4242)
    m = SimpleModel()
    return m.state_dict()


def _worker_dp_training_step(rank: int, world_size: int, port: int) -> None:
    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        model = SimpleModel().to(device)
        func_model, params = make_functional(model)

        def loss_fn(params, x, y):
            pred = func_model(params, x)
            return ((pred - y) ** 2).mean()

        grad_fn, clip_state = clipped_grad(
            loss_fn, clipping_norm=1.0, batch_argnums=(1, 2)
        )
        noise_fn, noise_state = gaussian_noise(noise_multiplier=1.1, key=key(0))

        batch_size = 8
        x = torch.randn(batch_size, 10, device=device)
        y = torch.randn(batch_size, 1, device=device)

        grads, clip_state = grad_fn(params, x, y, state=clip_state)
        summed_grads = sum_gradients(grads)
        for grad, summed in zip(
            tree_leaves(grads), tree_leaves(summed_grads), strict=False
        ):
            assert grad is not summed
        noisy_grads, noise_state = noise_fn(summed_grads, noise_state)

        for grad in tree_leaves(noisy_grads.pytree):
            assert grad.device == device
            assert not torch.isnan(grad).any()
            assert not torch.isinf(grad).any()
    finally:
        _cleanup_ddp()


def _worker_dp_parity(rank: int, world_size: int, port: int, out_path: str) -> None:
    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        model = SimpleModel().to(device)
        model.load_state_dict(_fixed_sd_clipped())
        func_model, params, _frozen = make_functional(model, partition_trainable=True)

        def loss_fn(params, x, y):
            pred = func_model(params, x)
            return ((pred - y) ** 2).mean()

        grad_fn, clip_state = clipped_grad(
            loss_fn, clipping_norm=1.0, batch_argnums=(1, 2)
        )
        noise_fn, noise_state = gaussian_noise(noise_multiplier=1.1, key=key(0))

        x_full = torch.arange(80, dtype=torch.float32, device=device).reshape(8, 10)
        y_full = torch.arange(8, dtype=torch.float32, device=device).reshape(8, 1) * 0.1
        sl = slice(rank * 4, (rank + 1) * 4)
        x = x_full[sl]
        y = y_full[sl]

        grads, _ = grad_fn(params, x, y, state=clip_state)
        summed = sum_gradients(grads)
        noised, _ = noise_fn(summed, noise_state)
        if rank == 0:
            torch.save(noised.pytree["fc1.weight"].cpu(), out_path)
    finally:
        _cleanup_ddp()


def _worker_shared_noise_is_deterministic(
    rank: int, world_size: int, port: int
) -> None:
    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        grads = {
            "weight": torch.zeros(10, 5, device=device),
            "bias": torch.zeros(5, device=device),
        }
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(0))
        noised, _ = noise_fn(clipped(grads, max_norm=1.0), state)

        gathered = [
            torch.zeros_like(noised.pytree["weight"]) for _ in range(world_size)
        ]
        dist.all_gather(gathered, noised.pytree["weight"])
        if rank == 0:
            for other in gathered[1:]:
                assert torch.equal(gathered[0], other)
    finally:
        _cleanup_ddp()


def _worker_sync_adaptive_clip_state(rank: int, world_size: int, port: int) -> None:
    from opaque.api.dpsgd.clipping._adaptive import AdaptiveClipState

    _setup_ddp(rank, world_size, port)
    try:
        state = AdaptiveClipState(
            _current_clipping_norm=float(rank + 1),
            _next_clipping_norm=float(rank + 1),
            _step=100,
            _rng_key=key(42),
            _fraction_noise_std=0.05,
            _learning_rate=0.2,
            _target_quantile=0.5,
            _clipping_norm_min=0.01,
            _clipping_norm_max=100.0,
            _num_clipped=float(3 * (rank + 1)),
            _batch_size=8 * (rank + 1),
        )
        synced = sync(state)
        expected_bs = sum(8 * (r + 1) for r in range(world_size))
        assert synced._batch_size == expected_bs
    finally:
        _cleanup_ddp()


def _worker_adaptive_clipping(rank: int, world_size: int, port: int) -> None:
    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        model = SimpleModel().to(device)
        func_model, params = make_functional(model)

        def loss_fn(params, x, y):
            pred = func_model(params, x)
            return ((pred - y) ** 2).mean()

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            batch_argnums=(1, 2),
            initial_clipping_norm=0.1,
            key=key(0),
        )

        batch_size = 8
        x = torch.randn(batch_size, 10, device=device)
        y = torch.randn(batch_size, 1, device=device)

        grads, new_state = grad_fn(params, x, y, state=clip_state)
        new_state = sync(new_state)

        # Full advertised distributed chain: grad_fn -> sync -> sum_gradients
        # (mirrors the docstring example in _adaptive.py, issue #415).
        summed = sum_gradients(grads)

        assert new_state._current_clipping_norm > 0
        assert new_state._step == 1
        assert grads is not None
        # ``summed`` is a ClippedPytree; its real tensors live under ``.pytree``.
        assert all(torch.isfinite(leaf).all() for leaf in tree_leaves(summed.pytree))
    finally:
        _cleanup_ddp()


def _worker_adaptive_clipping_uneven_batches(
    rank: int, world_size: int, port: int
) -> None:
    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        model = SimpleModel().to(device)
        func_model, params = make_functional(model)

        def loss_fn(params, x, y):
            pred = func_model(params, x)
            return ((pred - y) ** 2).mean()

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            batch_argnums=(1, 2),
            initial_clipping_norm=0.1,
            key=key(0),
        )

        local_batch_size = 4 if rank == 0 else 7
        x = torch.randn(local_batch_size, 10, device=device)
        y = torch.randn(local_batch_size, 1, device=device)

        _grads, new_state = grad_fn(params, x, y, state=clip_state)
        synced = sync(new_state)

        assert synced._batch_size == 11
        assert synced._next_clipping_norm > 0
    finally:
        _cleanup_ddp()


def _worker_sync_aux_adaptive_clipping(rank: int, world_size: int, port: int) -> None:
    from opaque.api.engine.distributed._state import reduce_scalar
    from opaque.distributed import sync

    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        model = SimpleModel().to(device)
        func_model, params = make_functional(model)

        def loss_fn(params, x, y):
            pred = func_model(params, x)
            return ((pred - y) ** 2).mean()

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            batch_argnums=(1, 2),
            initial_clipping_norm=0.1,
            key=key(0),
            return_aux=True,
        )

        local_batch_size = 3 if rank == 0 else 5
        x = torch.randn(local_batch_size, 10, device=device)
        y = torch.randn(local_batch_size, 1, device=device)

        (_grads, aux), new_state = grad_fn(params, x, y, state=clip_state)
        synced_aux = sync(aux)

        expected_n = sum(3 if r == 0 else 5 for r in range(world_size))
        assert synced_aux.loss_values.shape[0] == expected_n
        assert synced_aux.grad_norms.shape[0] == expected_n
        assert synced_aux.clipped_grad_norms.shape[0] == expected_n

        local_clipped = float(
            (aux.grad_norms > new_state._current_clipping_norm).sum().item()
        )
        local_total = float(aux.grad_norms.numel())
        global_clipped = reduce_scalar(local_clipped, op="sum", device=device)
        global_total = reduce_scalar(local_total, op="sum", device=device)
        expected_rate = global_clipped / max(1.0, global_total)
        assert abs(synced_aux.clipping_rate - expected_rate) < 1e-6
    finally:
        _cleanup_ddp()


def _worker_cpu_gloo_training_contract(rank: int, world_size: int, port: int) -> None:
    """Exercise normal and empty-rank DP-SGD chains on CPU."""
    _setup_gloo(rank, world_size, port)
    try:
        torch.manual_seed(123)
        model = SimpleModel()
        func_model, params = make_functional(model)

        def loss_fn(params, x, y):
            prediction = func_model(params, x)
            return ((prediction - y) ** 2).mean()

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            batch_argnums=(1, 2),
            initial_clipping_norm=0.1,
            key=key(17),
        )
        local_batch_size = 3 if rank == 0 else 5
        x = torch.randn(local_batch_size, 10)
        y = torch.randn(local_batch_size, 1)
        grads, clip_state = grad_fn(params, x, y, state=clip_state)
        synced_clip_state = sync(clip_state)
        assert synced_clip_state._batch_size == 8

        summed_grads = sum_gradients(grads)
        noise_fn, noise_state = gaussian_noise(noise_multiplier=1.1, key=key(23))
        noised_grads, noise_state = noise_fn(summed_grads, noise_state)
        synced_noise_state = sync(noise_state)
        assert synced_noise_state._step_counter == 1

        first_leaf = tree_leaves(noised_grads.pytree)[0]
        gathered = [torch.zeros_like(first_leaf) for _ in range(world_size)]
        dist.all_gather(gathered, first_leaf)
        assert all(torch.equal(gathered[0], value) for value in gathered[1:])

        empty_grad_fn, empty_clip_state = adaptive_clipped_grad(
            loss_fn,
            batch_argnums=(1, 2),
            initial_clipping_norm=0.1,
            key=key(31),
            return_aux=True,
        )
        empty_batch_size = 0 if rank == 0 else 2
        empty_x = torch.randn(empty_batch_size, 10)
        empty_y = torch.randn(empty_batch_size, 1)
        (empty_grads, empty_aux), empty_clip_state = empty_grad_fn(
            params, empty_x, empty_y, state=empty_clip_state
        )
        synced_empty_clip_state = sync(empty_clip_state)
        synced_empty_aux = sync(empty_aux)
        assert synced_empty_clip_state._batch_size == 2
        assert synced_empty_aux.batch_size == 2
        assert synced_empty_aux.loss_values.shape == (2,)

        summed_empty_grads = sum_gradients(empty_grads)
        noised_empty_grads, noise_state = noise_fn(summed_empty_grads, noise_state)
        synced_noise_state = sync(noise_state)
        assert synced_noise_state._step_counter == 2
        empty_leaf = tree_leaves(noised_empty_grads.pytree)[0]
        gathered_empty = [torch.zeros_like(empty_leaf) for _ in range(world_size)]
        dist.all_gather(gathered_empty, empty_leaf)
        assert all(
            torch.equal(gathered_empty[0], value) for value in gathered_empty[1:]
        )

        token = torch.tensor([float(rank + 1)])
        dist.all_reduce(token, op=dist.ReduceOp.SUM)
        assert token.item() == sum(range(1, world_size + 1))
    finally:
        _cleanup_ddp()


def _worker_per_group_adaptive_state_gloo(
    rank: int, world_size: int, port: int
) -> None:
    """Synchronize the per-group adaptive-clipping branch with unequal shards."""
    from opaque.api.dpsgd.clipping._adaptive import AdaptiveClipState
    from opaque.distributed import sync
    from opaque.random import key
    from opaque.types import PerGroup

    _setup_gloo(rank, world_size, port)
    try:
        bounds = PerGroup(
            groups={"weight": "weights", "bias": "biases"},
            values={"weights": 1.0, "biases": 0.5},
        )
        state = AdaptiveClipState(
            _current_clipping_norm=bounds,
            _next_clipping_norm=bounds,
            _step=1,
            _rng_key=key(41),
            _fraction_noise_std=1e-12,
            _learning_rate=0.2,
            _target_quantile=0.5,
            _clipping_norm_min=0.01,
            _clipping_norm_max=100.0,
            _num_clipped={
                "weights": float(rank + 1),
                "biases": float(2 - rank),
            },
            _batch_size=3 * (rank + 1),
        )
        synced = sync(state)

        assert synced._batch_size == 9.0
        assert synced._num_clipped == {"weights": 3.0, "biases": 3.0}
        assert isinstance(synced._next_clipping_norm, PerGroup)
        assert synced._next_clipping_norm.values["weights"] != bounds.values["weights"]
        assert synced._next_clipping_norm.values["biases"] < bounds.values["biases"]

        gathered = [None] * world_size
        dist.all_gather_object(
            gathered,
            tuple(sorted(synced._next_clipping_norm.values.items())),
        )
        assert all(value == gathered[0] for value in gathered[1:])

        token = torch.tensor([1.0])
        dist.all_reduce(token, op=dist.ReduceOp.SUM)
        assert token.item() == float(world_size)
    finally:
        _cleanup_ddp()


def _worker_per_group_adaptive_training_gloo(
    rank: int, world_size: int, port: int
) -> None:
    """Run the per-group adaptive DP-SGD chain across unequal local batches."""
    from opaque.distributed import sum_gradients, sync
    from opaque.dpsgd.clipping import adaptive_clipped_grad, per_group
    from opaque.dpsgd.noise import gaussian_noise
    from opaque.pytree import tree_leaves
    from opaque.random import key
    from opaque.types import PerGroup

    _setup_gloo(rank, world_size, port)
    try:
        params = {
            "weight": torch.tensor([0.25, -0.5]),
            "bias": torch.tensor([0.1]),
        }
        bounds = per_group(params, weight=1.0, bias=0.5)

        def loss_fn(current_params, x, y):
            prediction = x @ current_params["weight"] + current_params["bias"].sum()
            return (prediction - y).square()

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=bounds,
            fraction_noise_std=0.05,
            key=key(71),
            batch_argnums=(1, 2),
            return_aux=True,
        )
        local_batch_size = 2 if rank == 0 else 3
        x = (
            torch.arange(local_batch_size * 2, dtype=torch.float32).reshape(
                local_batch_size, 2
            )
            + rank
        )
        y = torch.linspace(0.0, 1.0, local_batch_size)

        (local_grads, local_aux), clip_state = grad_fn(params, x, y, state=clip_state)
        reduced = sum_gradients(local_grads)
        synced_state, synced_aux = sync(clip_state, local_aux)

        noise_fn, noise_state = gaussian_noise(noise_multiplier=1.1, key=key(73))
        noised, noise_state = noise_fn(reduced, noise_state)
        synced_noise_state = sync(noise_state)

        assert isinstance(reduced.max_norm, PerGroup)
        assert isinstance(noised.noise_stddev, PerGroup)
        assert synced_state._batch_size == 5.0
        assert synced_aux.batch_size == 5
        assert synced_aux.group_norms is not None
        assert set(synced_aux.group_norms) == {"weight", "bias"}
        assert synced_noise_state._step_counter == 1

        first_leaf = tree_leaves(noised.pytree)[0]
        gathered = [torch.zeros_like(first_leaf) for _ in range(world_size)]
        dist.all_gather(gathered, first_leaf)
        assert all(torch.equal(gathered[0], value) for value in gathered[1:])
    finally:
        _cleanup_ddp()


def _worker_noise_seed_out_of_int64_range_gloo(
    rank: int, world_size: int, port: int
) -> None:
    """Sync a shared key whose seed does not fit a signed 64-bit reduction.

    ``RngKey.seed`` is canonicalized to unsigned 64-bit, so roughly half of all
    ``fold_in``-derived keys set the top bit and fall outside the signed
    ``int64`` domain a scalar reduction can carry.  The documented
    per-rank-stream recipe produces exactly such keys.
    """
    import pytest

    from opaque.random import fold_in

    _setup_gloo(rank, world_size, port)
    try:
        shared = fold_in(key(5), 1)
        assert int(shared.seed) > 2**63 - 1, (
            "fixture must use a seed outside the int64 domain"
        )

        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=shared)
        noised, state = noise_fn(_shared_clipped(), state)
        synced = sync(state)
        assert synced._step_counter == 1

        # A per-rank key must still be rejected, and by seed rather than by
        # some artifact of the encoding.
        per_rank_fn, per_rank_state = gaussian_noise(
            noise_multiplier=1.0, key=fold_in(shared, rank)
        )
        _, per_rank_state = per_rank_fn(_shared_clipped(), per_rank_state)
        with pytest.raises(RuntimeError, match="seed"):
            sync(per_rank_state)

        # The shared key was accepted, so every rank drew the same noise on the
        # same input — the property `sync` exists to enforce.
        first_leaf = tree_leaves(noised.pytree)[0]
        gathered = [torch.zeros_like(first_leaf) for _ in range(world_size)]
        dist.all_gather(gathered, first_leaf)
        assert all(torch.equal(gathered[0], value) for value in gathered[1:])
    finally:
        _cleanup_ddp()


def _shared_clipped() -> dict[str, torch.Tensor]:
    """A rank-independent clipped pytree, so any cross-rank difference is noise."""
    return clipped({"w": torch.ones(3)}, max_norm=1.0)


def _worker_summed_noise_scaling_gloo(rank: int, world_size: int, port: int) -> None:
    """Pin the two aggregation regimes `reduce_pytree`'s metadata cannot tell apart.

    ``gradients.py`` states that summing noised local queries scales
    ``noise_stddev`` by ``sqrt(world_size)``.  That holds only when the summands
    are *independent* — per-rank keys.  ``sync_gaussian_noise_state`` enforces
    the opposite (seed equality), and summing identical draws multiplies the
    realized noise by ``world_size`` while the published metadata still claims
    ``sqrt(world_size)``.

    Neither is a defect in the reduction: ``NoisedPytree`` carries no provenance
    that would let it detect which regime it is in.  It is a sharp edge, so both
    sides are measured here rather than left to the docstring.
    """
    import pytest

    from opaque.distributed import reduce_pytree
    from opaque.random import fold_in

    _setup_gloo(rank, world_size, port)
    try:
        shared_key = key(11)
        sigma = 1.0
        samples = 20_000
        wide = clipped({"w": torch.zeros(samples)}, max_norm=1.0)

        # Independent per-rank streams: the advertised sqrt(W) is realized.
        indep_fn, indep_state = gaussian_noise(
            noise_multiplier=sigma, key=fold_in(shared_key, rank)
        )
        indep, _ = indep_fn(wide, indep_state)
        indep_sum = reduce_pytree(indep, op="sum")
        realized = indep_sum.pytree["w"].std().item()
        claimed = indep_sum.noise_stddev
        assert claimed == pytest.approx(sigma * math.sqrt(world_size))
        assert realized == pytest.approx(claimed, rel=0.05), (
            f"independent summands realized {realized}, metadata claims {claimed}"
        )

        # Shared key: every rank drew the same noise, so summing multiplies it
        # by W.  The metadata still reports sqrt(W) — this is the trap.
        shared_fn, shared_state = gaussian_noise(noise_multiplier=sigma, key=shared_key)
        shared, _ = shared_fn(wide, shared_state)
        shared_sum = reduce_pytree(shared, op="sum")
        shared_realized = shared_sum.pytree["w"].std().item()
        assert shared_sum.noise_stddev == pytest.approx(sigma * math.sqrt(world_size))
        assert shared_realized == pytest.approx(sigma * world_size, rel=0.05), (
            "identical summands must scale linearly, not by sqrt(W)"
        )
    finally:
        _cleanup_ddp()


def _worker_per_group_adaptive_one_rank_empty_gloo(
    rank: int, world_size: int, port: int
) -> None:
    """Adaptive per-group clipping derives empty-rank reductions from the schema."""
    from opaque.distributed import sum_gradients, sync
    from opaque.dpsgd.clipping import adaptive_clipped_grad, per_group
    from opaque.random import key
    from opaque.types import PerGroup

    _setup_gloo(rank, world_size, port)
    try:
        params = {
            "bias": torch.tensor([0.1]),
            "weight": torch.tensor([0.25, -0.5]),
        }
        bounds = per_group(
            params,
            patterns={"weight": 1.0, "bias": 0.5},
        )
        assert tuple(bounds.values) == ("weight", "bias")
        assert tuple(dict.fromkeys(bounds.groups.values())) == ("bias", "weight")

        def loss_fn(current_params, x, y):
            prediction = x @ current_params["weight"] + current_params["bias"]
            return (prediction - y).square().mean()

        grad_fn, clip_state = adaptive_clipped_grad(
            loss_fn,
            initial_clipping_norm=bounds,
            fraction_noise_std=1e-12,
            key=key(97),
            batch_argnums=(1, 2),
            return_aux=True,
        )
        local_batch_size = 0 if rank == 0 else 2
        x = torch.tensor([[100.0, 0.0], [100.0, 0.0]])[:local_batch_size]
        y = torch.full((local_batch_size,), 25.09)

        (local_grads, local_aux), clip_state = grad_fn(params, x, y, state=clip_state)
        assert local_aux.clipping_rate is None

        reduced = sum_gradients(local_grads)
        assert isinstance(reduced.max_norm, PerGroup)
        synced_state, synced_aux = sync(clip_state, local_aux)

        assert synced_aux.clipping_rate is None
        assert synced_aux.batch_size == 2
        assert synced_aux.group_norms is not None
        assert set(synced_aux.group_norms) == {"weight", "bias"}
        assert synced_state._batch_size == 2.0
        assert synced_state._num_clipped == {"bias": 0.0, "weight": 2.0}
        assert isinstance(synced_state._next_clipping_norm, PerGroup)
        assert synced_state._next_clipping_norm.values["weight"] > 1.0
        assert synced_state._next_clipping_norm.values["bias"] < 0.5

        gathered_thresholds = [None] * world_size
        dist.all_gather_object(
            gathered_thresholds,
            tuple(sorted(synced_state._next_clipping_norm.values.items())),
        )
        assert all(value == gathered_thresholds[0] for value in gathered_thresholds[1:])

        # A follow-up collective must still succeed (proves no desync).
        token = torch.tensor([float(rank + 1)])
        dist.all_reduce(token, op=dist.ReduceOp.SUM)
        assert abs(token.item() - sum(range(1, world_size + 1))) < 1e-5
    finally:
        _cleanup_ddp()


# ---------------------------------------------------------------------------
# MoE router-load release: rank-local means reduced, noised once (gloo/CPU)
# ---------------------------------------------------------------------------

_MOE_E, _MOE_K, _MOE_L, _MOE_D, _MOE_T = 8, 2, 2, 6, 12


def _moe_fixture() -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(11)
    params = {
        "router": torch.randn(_MOE_L, _MOE_D, _MOE_E, generator=g),
        "head": torch.randn(_MOE_D, generator=g),
    }
    x = torch.randn(4, _MOE_T, _MOE_D, generator=g)
    lengths = torch.tensor([_MOE_T, 9, _MOE_T, 5])
    mask = (torch.arange(_MOE_T)[None, :] < lengths[:, None]).long()
    y = torch.randn(4, generator=g)
    return params, x, mask, y


def _moe_loss(params: dict, x: torch.Tensor, mask: torch.Tensor, y: torch.Tensor):
    logits = [x @ params["router"][layer] for layer in range(_MOE_L)]
    m = mask.float()
    pred = x @ params["head"]
    loss = ((pred - y) ** 2 * m).sum() / m.sum().clamp(min=1.0)
    return loss, logits, mask


def _moe_factory(**overrides):
    from opaque.dpsgd.clipping import moe_clipped_grad
    from opaque.random import key

    kwargs = {
        "clipping_norm": 0.7,
        "normalize_by": 4.0,
        "batch_argnums": (1, 2, 3),
        "noise_multiplier": 1.0,
        "ratio": 0.5,
        "key": key(7),
        "top_k": _MOE_K,
        "num_experts": _MOE_E,
        "num_layers": _MOE_L,
        "max_tokens": _MOE_T,
        "alpha": 0.2,
    }
    kwargs.update(overrides)
    return moe_clipped_grad(_moe_loss, **kwargs)


def _expect_release_state_mismatch(state) -> None:
    from opaque.distributed import sync
    from opaque.exceptions import OperationError

    message = None
    try:
        sync(state)
    except OperationError as exc:
        message = str(exc)
    assert message is not None, "a mismatched release state was not rejected"
    assert "release state" in message


def _worker_moe_sync_ratio_mismatch_gloo(rank: int, world_size: int, port: int) -> None:
    """Ranks built with different ratios must fail on every rank, not reduce."""
    _setup_gloo(rank, world_size, port)
    try:
        params, x, mask, y = _moe_fixture()
        shard = slice(2 * rank, 2 * rank + 2)
        grad_fn, state = _moe_factory(ratio=0.5 if rank == 0 else 0.25)
        _, state = grad_fn(params, x[shard], mask[shard], y[shard], state=state)
        _expect_release_state_mismatch(state)
    finally:
        _cleanup_ddp()


def _worker_moe_sync_pending_disagreement_gloo(
    rank: int, world_size: int, port: int
) -> None:
    """A rank that has not drawn its step must not leave the others blocked."""
    _setup_gloo(rank, world_size, port)
    try:
        params, x, mask, y = _moe_fixture()
        grad_fn, state = _moe_factory()
        if rank == 0:
            _, state = grad_fn(params, x[:2], mask[:2], y[:2], state=state)
        _expect_release_state_mismatch(state)
    finally:
        _cleanup_ddp()


def _worker_moe_sync_gloo(rank: int, world_size: int, port: int, out_path: str) -> None:
    from opaque.distributed import sync
    from opaque.exceptions import OperationError

    _setup_gloo(rank, world_size, port)
    try:
        params, x, mask, y = _moe_fixture()
        shard = slice(2 * rank, 2 * rank + 2)
        grad_fn, state = _moe_factory()
        grads, state = grad_fn(params, x[shard], mask[shard], y[shard], state=state)
        assert state._pending
        assert state.step == 0
        # A second step before the sync must fail loudly rather than drop
        # the pending release.
        try:
            grad_fn(params, x[shard], mask[shard], y[shard], state=state)
        except OperationError:
            pass
        else:
            raise AssertionError("pending release was silently overwritten")
        synced = sync(state)
        assert not synced._pending
        assert synced.step == 1
        assert sync(synced) is synced
        reduced = {}
        for name, value in grads.pytree.items():
            total = value.clone()
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
            reduced[name] = total
        if rank == 0:
            torch.save({"grads": reduced, "f_tilde": synced.f_tilde}, out_path)
    finally:
        _cleanup_ddp()
