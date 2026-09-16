"""Minimal NCCL DDP helpers + mp.spawn entrypoints (must live in this module for pickle)."""

from __future__ import annotations

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
from opaque.dpftrl.clipping import auto_clipped_grad, clipped_grad
from opaque.dpftrl.noise import (
    band_mf_strategy,
    bisr_strategy,
    identity_strategy,
    mf_gaussian_noise,
)
from opaque.functional import make_functional
from opaque.pytree import tree_leaves, tree_map
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


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(6, 12)
        self.fc2 = nn.Linear(12, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class AutoSimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def _fixed_sd_mf() -> dict[str, torch.Tensor]:
    torch.manual_seed(1313)
    return SimpleModel().state_dict()


def _fixed_sd_tiny() -> dict[str, torch.Tensor]:
    torch.manual_seed(9191)
    return TinyModel().state_dict()


def _worker_identity_mf_three_steps(rank: int, world_size: int, port: int) -> None:
    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        batch_size = 32
        param_dim = 64
        grad_template = {"weight": torch.zeros(batch_size, param_dim, device=device)}
        noise_fn, state = mf_gaussian_noise(
            grad_template,
            identity_strategy(),
            n_steps=3,
            noise_multiplier=1.0,
            key=key(0),
        )
        step_noise_values = []
        step_stds = []
        for _step in range(3):
            grads = clipped(
                {"weight": torch.zeros(batch_size, param_dim, device=device)},
                max_norm=1.0,
            )
            noised, state = noise_fn(grads, state)
            step_noise_values.append(noised.pytree["weight"].clone())
            step_stds.append(noised.pytree["weight"].std().item())

        assert not torch.allclose(step_noise_values[0], step_noise_values[1])
        assert not torch.allclose(step_noise_values[1], step_noise_values[2])

        for step_idx, std in enumerate(step_stds):
            assert 0.8 < std < 1.2, f"Step {step_idx}: std {std} out of range"

        for step_idx, noise_val in enumerate(step_noise_values):
            assert torch.isfinite(noise_val).all(), f"Step {step_idx}: non-finite noise"

        for step_idx, noise_val in enumerate(step_noise_values):
            gathered = [torch.zeros_like(noise_val) for _ in range(world_size)]
            dist.all_gather(gathered, noise_val)
            if rank == 0:
                for other in gathered[1:]:
                    assert torch.allclose(gathered[0], other, atol=1e-6), (
                        f"Step {step_idx}: rank 0 and other ranks disagree"
                    )

    finally:
        _cleanup_ddp()


def _worker_mf_shared_noise(rank: int, world_size: int, port: int) -> None:
    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        grad_template = {"weight": torch.zeros(4, device=device)}
        noise_fn, state = mf_gaussian_noise(
            grad_template,
            identity_strategy(),
            n_steps=1,
            noise_multiplier=1.0,
            key=key(0),
        )
        grads = {"weight": torch.zeros(4, device=device)}
        noised, _ = noise_fn(clipped(grads, max_norm=1.0), state)

        w = noised.pytree["weight"]
        gathered = [torch.zeros_like(w) for _ in range(world_size)]
        dist.all_gather(gathered, w)
        if rank == 0:
            for other in gathered[1:]:
                assert torch.equal(gathered[0], other)
    finally:
        _cleanup_ddp()


def _worker_mf_clip_three_steps(rank: int, world_size: int, port: int) -> None:
    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        model = SimpleModel().to(device)
        model.load_state_dict(_fixed_sd_mf())
        func_model, params, _frozen = make_functional(model, partition_trainable=True)

        def loss_fn(params, x, y):
            pred = func_model(params, x)
            return ((pred - y) ** 2).mean()

        grad_fn, clip_state = clipped_grad(
            loss_fn, clipping_norm=1.0, batch_argnums=(1, 2)
        )
        tmpl = tree_map(lambda t: torch.zeros_like(t, device=device), params)
        noise_fn, noise_state = mf_gaussian_noise(
            tmpl,
            identity_strategy(),
            n_steps=4,
            noise_multiplier=1.1,
            key=key(0),
        )
        x = torch.randn(4, 10, device=device)
        y = torch.randn(4, 1, device=device)
        for _ in range(3):
            grads, clip_state = grad_fn(params, x, y, state=clip_state)
            summed = sum_gradients(grads)
            _noised, noise_state = noise_fn(summed, noise_state)
            noise_state = sync(noise_state)
            assert torch.isfinite(_noised.pytree["fc1.weight"]).all()
    finally:
        _cleanup_ddp()


def _worker_mf_parity(rank: int, world_size: int, port: int, out_path: str) -> None:
    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        model = SimpleModel().to(device)
        model.load_state_dict(_fixed_sd_mf())
        func_model, params, _frozen = make_functional(model, partition_trainable=True)

        def loss_fn(params, x, y):
            pred = func_model(params, x)
            return ((pred - y) ** 2).mean()

        grad_fn, clip_state = clipped_grad(
            loss_fn, clipping_norm=1.0, batch_argnums=(1, 2)
        )
        tmpl = tree_map(lambda t: torch.zeros_like(t, device=device), params)
        noise_fn, noise_state = mf_gaussian_noise(
            tmpl,
            identity_strategy(),
            n_steps=4,
            noise_multiplier=1.1,
            key=key(0),
        )
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


def _worker_auto_mf(rank: int, world_size: int, port: int) -> None:
    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        torch.manual_seed(777 + rank)
        model = AutoSimpleModel().to(device)
        func_model, params, _frozen = make_functional(model, partition_trainable=True)

        def loss_fn(params, x, y):
            pred = func_model(params, x)
            return ((pred - y) ** 2).mean()

        grad_fn, clip_state = auto_clipped_grad(
            loss_fn,
            batch_argnums=(1, 2),
            R=1.0,
            normalize_by=4.0,
        )
        tmpl = tree_map(lambda t: torch.zeros_like(t, device=device), params)
        noise_fn, noise_state = mf_gaussian_noise(
            tmpl,
            band_mf_strategy(bands=2, momentum=0.9),
            n_steps=4,
            noise_multiplier=0.6,
            key=key(22),
        )
        x = torch.randn(4, 8, device=device)
        y = torch.randn(4, 1, device=device)
        for _ in range(2):
            grads, clip_state = grad_fn(params, x, y, state=clip_state)
            summed = sum_gradients(grads)
            noised, noise_state = noise_fn(summed, noise_state)
            assert torch.isfinite(noised.pytree["fc1.weight"]).all()
            noise_state = sync(noise_state)
    finally:
        _cleanup_ddp()


def _worker_band_parity(rank: int, world_size: int, port: int, out_path: str) -> None:
    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        model = TinyModel().to(device)
        model.load_state_dict(_fixed_sd_tiny())
        func_model, params, _frozen = make_functional(model, partition_trainable=True)

        def loss_fn(params, x, y):
            pred = func_model(params, x)
            return ((pred - y) ** 2).mean()

        grad_fn, clip_state = clipped_grad(
            loss_fn, clipping_norm=1.0, batch_argnums=(1, 2)
        )
        tmpl = tree_map(lambda t: torch.zeros_like(t, device=device), params)
        noise_fn, noise_state = mf_gaussian_noise(
            tmpl,
            band_mf_strategy(bands=2, momentum=0.9),
            n_steps=4,
            noise_multiplier=0.85,
            key=key(5),
        )
        x_full = torch.arange(48, dtype=torch.float32, device=device).reshape(8, 6)
        y_full = (
            torch.arange(8, dtype=torch.float32, device=device).reshape(8, 1) * 0.05
        )
        sl = slice(rank * 4, (rank + 1) * 4)
        grads, _ = grad_fn(params, x_full[sl], y_full[sl], state=clip_state)
        summed = sum_gradients(grads)
        noised, _ = noise_fn(summed, noise_state)
        if rank == 0:
            torch.save(noised.pytree["fc1.weight"].cpu(), out_path)
    finally:
        _cleanup_ddp()


def _worker_cpu_gloo_training_contract(rank: int, world_size: int, port: int) -> None:
    """Exercise normal and empty-rank DP-FTRL chains on CPU."""
    _setup_gloo(rank, world_size, port)
    try:
        torch.manual_seed(321)
        model = SimpleModel()
        func_model, params, _frozen = make_functional(model, partition_trainable=True)

        def loss_fn(params, x, y):
            prediction = func_model(params, x)
            return ((prediction - y) ** 2).mean()

        grad_fn, clip_state = clipped_grad(
            loss_fn, clipping_norm=1.0, batch_argnums=(1, 2)
        )
        local_batch_size = 3 if rank == 0 else 5
        x = torch.randn(local_batch_size, 10)
        y = torch.randn(local_batch_size, 1)
        grads, _ = grad_fn(params, x, y, state=clip_state)
        summed_grads = sum_gradients(grads)

        template = tree_map(torch.zeros_like, params)
        noise_fn, noise_state = mf_gaussian_noise(
            template,
            identity_strategy(),
            n_steps=2,
            noise_multiplier=1.1,
            key=key(29),
        )
        noise_steps = []
        for _ in range(2):
            noised_grads, noise_state = noise_fn(summed_grads, noise_state)
            noise_state = sync(noise_state)
            first_leaf = tree_leaves(noised_grads.pytree)[0]
            gathered = [torch.zeros_like(first_leaf) for _ in range(world_size)]
            dist.all_gather(gathered, first_leaf)
            assert all(torch.equal(gathered[0], value) for value in gathered[1:])
            noise_steps.append(first_leaf)

        assert not torch.equal(noise_steps[0], noise_steps[1])

        empty_grad_fn, empty_clip_state = clipped_grad(
            loss_fn,
            clipping_norm=1.0,
            batch_argnums=(1, 2),
            return_aux=True,
        )
        empty_batch_size = 0 if rank == 0 else 2
        empty_x = torch.randn(empty_batch_size, 10)
        empty_y = torch.randn(empty_batch_size, 1)
        (empty_grads, empty_aux), _ = empty_grad_fn(
            params, empty_x, empty_y, state=empty_clip_state
        )
        synced_empty_aux = sync(empty_aux)
        assert synced_empty_aux.batch_size == 2
        assert synced_empty_aux.loss_values.shape == (2,)

        summed_empty_grads = sum_gradients(empty_grads)
        empty_noise_fn, empty_noise_state = mf_gaussian_noise(
            template,
            identity_strategy(),
            n_steps=1,
            noise_multiplier=1.1,
            key=key(37),
        )
        noised_empty_grads, empty_noise_state = empty_noise_fn(
            summed_empty_grads, empty_noise_state
        )
        synced_empty_noise_state = sync(empty_noise_state)
        assert synced_empty_noise_state._step_counter == 1
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


def _worker_per_group_mf_state_gloo(rank: int, world_size: int, port: int) -> None:
    """Exercise per-group MF sensitivity-state synchronization and mismatch checks."""
    from dataclasses import replace

    import pytest

    from opaque.api.dpftrl.noise._distributed import fingerprint_per_group_max_norm
    from opaque.distributed import sync
    from opaque.dpftrl.noise import identity_strategy, mf_gaussian_noise
    from opaque.random import key
    from opaque.types import PerGroup, clipped

    _setup_gloo(rank, world_size, port)
    try:
        bounds = PerGroup(
            groups={"weight": "weights", "bias": "biases"},
            values={"weights": 1.0, "biases": 0.5},
        )
        gradients = clipped(
            {
                "weight": torch.full((2,), float(rank + 1)),
                "bias": torch.tensor([float(rank)]),
            },
            max_norm=bounds,
        )
        noise_fn, state = mf_gaussian_noise(
            {"weight": torch.zeros(2), "bias": torch.zeros(1)},
            identity_strategy(),
            n_steps=1,
            noise_multiplier=1.0,
            key=key(53),
        )
        noised, state = noise_fn(gradients, state)
        synced = sync(state)

        assert isinstance(noised.noise_stddev, PerGroup)
        assert synced._first_max_norm == bounds
        assert (
            synced._first_max_norm_sync_fingerprint
            == fingerprint_per_group_max_norm(bounds)
        )

        mismatched_bounds = PerGroup(
            groups=bounds.groups,
            values={"weights": 2.0, "biases": 0.5},
        )
        mismatched = replace(
            synced,
            _first_max_norm=mismatched_bounds,
            _first_max_norm_sync_fingerprint=(
                fingerprint_per_group_max_norm(bounds)
                if rank == 0
                else fingerprint_per_group_max_norm(mismatched_bounds)
            ),
        )
        with pytest.raises(
            RuntimeError,
            match=r"MFNoiseState\._first_max_norm_sync_fingerprint mismatch",
        ):
            sync(mismatched)

        token = torch.tensor([1.0])
        dist.all_reduce(token, op=dist.ReduceOp.SUM)
        assert token.item() == float(world_size)
    finally:
        _cleanup_ddp()


def _worker_bisr_bounded_state_gloo(rank: int, world_size: int, port: int) -> None:
    """Keep BISR releases and each local history slot identical across ranks."""
    _setup_gloo(rank, world_size, port)
    try:
        n_steps = 7
        bandwidth = 3
        template = {
            "weight": torch.zeros(6),
            "nested": {"bias": torch.zeros(3)},
        }
        noise_fn, state = mf_gaussian_noise(
            template,
            bisr_strategy(bandwidth=bandwidth, momentum=0.9),
            n_steps=n_steps,
            noise_multiplier=0.8,
            key=key(795),
            compute_dtype=torch.float64,
        )
        gradients = clipped(tree_map(torch.zeros_like, template), max_norm=1.0)

        for step in range(n_steps):
            noised, state = noise_fn(gradients, state)
            state = sync(state)
            assert state._step_counter == step + 1
            assert state._inner_state.step == step + 1
            assert len(state._inner_state.history) == bandwidth - 1

            for value in tree_leaves(noised.pytree):
                gathered = [torch.zeros_like(value) for _ in range(world_size)]
                dist.all_gather(gathered, value)
                assert all(torch.equal(gathered[0], other) for other in gathered[1:])

            # The distributed schema intentionally treats the inner matrix
            # state as local, so verify the fixed window itself, not just the
            # public step counter validated by ``sync``.
            for history_tree in state._inner_state.history:
                for value in tree_leaves(history_tree):
                    gathered = [torch.zeros_like(value) for _ in range(world_size)]
                    dist.all_gather(gathered, value)
                    assert all(
                        torch.equal(gathered[0], other) for other in gathered[1:]
                    )
    finally:
        _cleanup_ddp()


def _worker_auto_band_mf_gloo(rank: int, world_size: int, port: int) -> None:
    """Run AUTO-S gradients through two band-MF releases on CPU/Gloo."""
    from opaque.distributed import sum_gradients, sync
    from opaque.dpftrl.clipping import auto_clipped_grad
    from opaque.dpftrl.noise import band_mf_strategy, mf_gaussian_noise
    from opaque.pytree import tree_leaves, tree_map
    from opaque.random import key

    _setup_gloo(rank, world_size, port)
    try:
        params = {
            "weight": torch.tensor([0.25, -0.5]),
            "bias": torch.tensor([0.1]),
        }

        def loss_fn(current_params, x, y):
            prediction = x @ current_params["weight"] + current_params["bias"].sum()
            return (prediction - y).square()

        grad_fn, clip_state = auto_clipped_grad(
            loss_fn,
            R=1.0,
            batch_argnums=(1, 2),
        )
        noise_fn, noise_state = mf_gaussian_noise(
            tree_map(torch.zeros_like, params),
            band_mf_strategy(bands=2, momentum=0.9),
            n_steps=2,
            noise_multiplier=0.8,
            key=key(83),
        )

        releases = []
        for step in range(2):
            local_batch_size = 2 if rank == 0 else 3
            x = (
                torch.arange(local_batch_size * 2, dtype=torch.float32).reshape(
                    local_batch_size, 2
                )
                + rank
                + step
            )
            y = torch.linspace(0.0, 1.0, local_batch_size) + step
            local_grads, clip_state = grad_fn(params, x, y, state=clip_state)
            noised, noise_state = noise_fn(sum_gradients(local_grads), noise_state)
            releases.append(tree_leaves(noised.pytree)[0])
            noise_state = sync(noise_state)
            assert noise_state._step_counter == step + 1
            assert torch.isfinite(releases[-1]).all()

        assert not torch.equal(releases[0], releases[1])
        gathered = [torch.zeros_like(releases[-1]) for _ in range(world_size)]
        dist.all_gather(gathered, releases[-1])
        assert all(torch.equal(gathered[0], value) for value in gathered[1:])
    finally:
        _cleanup_ddp()
