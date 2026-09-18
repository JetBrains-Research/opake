"""Sampler-active DP-SGD parity: 1-GPU vs 2-GPU sharded.

Unlike :mod:`tests.integration.transformers.test_dpsgd_short_run_parity`
(which slices the *same* fixed batch on every rank and shares one noise
key, exercising the bit-exact algebraic identity
``sum_gradients(per_record_clip(shard)) == per_record_clip(full)``), this
test runs a real :class:`opake.dpsgd.sampling.PoissonSampler` on each
rank with ``fold_in(key(seed), rank)``.  The set of training examples
seen at each step diverges between 1-GPU and 2-GPU, so the trajectories
intentionally drift.  A 2% bound on the final eval loss gives ~4×
headroom over the observed drift while still catching real bugs
(missing gradient reduction, wrong noise scaling, mis-keyed sampler,
etc.).

This test validates the same single-vs-multi-GPU training behavior at a
small scale.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from opake_test_support import cleanup_process_group, setup_nccl, spawn

from opake.api.engine.clipping import clipped_grad
from opake.distributed import local_shard, sum_gradients
from opake.dpsgd.noise import gaussian_noise
from opake.dpsgd.sampling import PoissonSampler
from opake.functional import make_functional
from opake.random import fold_in, key

pytestmark = [pytest.mark.slow, pytest.mark.cuda]

# Dataset & training hyperparameters.  Picked so the run is fast (<30s on
# 2× A10G), the Poisson expected batch is ~16 per step, and the model
# learns enough that eval loss is visibly non-trivial.
_N_TRAIN = 256
_N_EVAL = 64
_INPUT_DIM = 16
_OUTPUT_DIM = 4
_SAMPLE_RATE = 0.0625  # E[batch] = 16
_N_STEPS = 30
_LR = 0.05
_CLIP = 1.0
_NOISE_MULTIPLIER = 0.3
_SEED = 13


class _Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(_INPUT_DIM, 32)
        self.fc2 = nn.Linear(32, _OUTPUT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.tanh(self.fc1(x)))


def _reference_state_dict() -> dict[str, torch.Tensor]:
    torch.manual_seed(2025)
    return _Mlp().state_dict()


def _build_dataset(
    n: int, generator_seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(generator_seed)
    x = torch.randn(n, _INPUT_DIM, device=device, generator=g)
    # Linear target plus tanh nonlinearity in the model — train can never
    # reach zero loss, so eval loss stays bounded below by a healthy
    # constant even at full batch.
    target_w = torch.randn(_INPUT_DIM, _OUTPUT_DIM, device=device, generator=g)
    y = x @ target_w + 0.1 * torch.randn(n, _OUTPUT_DIM, device=device, generator=g)
    return x, y


def _eval_loss(
    fmodel,
    params: dict[str, torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    with torch.no_grad():
        pred = fmodel(params, x)
        return float(((pred - y) ** 2).mean())


def _run_single(device: torch.device) -> float:
    model = _Mlp().to(device)
    model.load_state_dict({k: v.to(device) for k, v in _reference_state_dict().items()})
    x_train, y_train = _build_dataset(_N_TRAIN, generator_seed=101, device=device)
    x_eval, y_eval = _build_dataset(_N_EVAL, generator_seed=202, device=device)
    fmodel, params, _frozen = make_functional(
        model, disable_autograd_tracking=True, partition_trainable=True
    )

    def per_example_loss(p, x_one, y_one):
        return ((fmodel(p, x_one) - y_one) ** 2).mean()

    grad_fn, clip_state = clipped_grad(
        per_example_loss, argnums=0, batch_argnums=(1, 2), clipping_norm=_CLIP
    )
    noise_fn, noise_state = gaussian_noise(
        noise_multiplier=_NOISE_MULTIPLIER, key=key(_SEED + 999)
    )

    sampler = PoissonSampler(
        list(range(_N_TRAIN)),
        sample_rate=_SAMPLE_RATE,
        n_steps=_N_STEPS,
        key=key(_SEED),
    )
    bs_expected = _N_TRAIN * _SAMPLE_RATE
    for batch_indices in sampler:
        if not batch_indices:
            continue
        idx = torch.tensor(batch_indices, device=device, dtype=torch.long)
        xb = x_train[idx]
        yb = y_train[idx]
        grads, clip_state = grad_fn(params, xb, yb, state=clip_state)
        noised, noise_state = noise_fn(grads, noise_state)
        for name, value in params.items():
            params[name] = value - _LR * (noised.pytree[name] / bs_expected)

    return _eval_loss(fmodel, params, x_eval, y_eval)


def _worker_ddp(rank: int, world_size: int, port: int, out_path: str) -> None:
    setup_nccl(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        model = _Mlp().to(device)
        model.load_state_dict(
            {k: v.to(device) for k, v in _reference_state_dict().items()}
        )

        x_train_full, y_train_full = _build_dataset(
            _N_TRAIN, generator_seed=101, device=device
        )
        x_eval, y_eval = _build_dataset(_N_EVAL, generator_seed=202, device=device)

        global_indices = list(range(_N_TRAIN))
        shard_indices = local_shard(global_indices, rank=rank, world_size=world_size)
        x_train = x_train_full[torch.tensor(shard_indices, device=device)]
        y_train = y_train_full[torch.tensor(shard_indices, device=device)]

        fmodel, params, _frozen = make_functional(
            model, disable_autograd_tracking=True, partition_trainable=True
        )

        def per_example_loss(p, x_one, y_one):
            return ((fmodel(p, x_one) - y_one) ** 2).mean()

        grad_fn, clip_state = clipped_grad(
            per_example_loss,
            argnums=0,
            batch_argnums=(1, 2),
            clipping_norm=_CLIP,
        )
        # Same noise key across ranks — replicated noise is fine for the
        # parity check (the privacy story has noise added only on rank 0
        # then broadcast; here we just want every rank to land on the same
        # updated params after each step).
        noise_fn, noise_state = gaussian_noise(
            noise_multiplier=_NOISE_MULTIPLIER, key=key(_SEED + 999)
        )

        # Per-rank sampler with fold_in(seed, rank).  The shard size is
        # _N_TRAIN // world_size; the per-example probability stays the
        # same so the world-summed expected batch matches the 1-GPU one.
        sampler = PoissonSampler(
            list(range(len(shard_indices))),
            sample_rate=_SAMPLE_RATE,
            n_steps=_N_STEPS,
            key=fold_in(key(_SEED), rank),
        )
        bs_expected = _N_TRAIN * _SAMPLE_RATE
        for batch_indices in sampler:
            if batch_indices:
                idx = torch.tensor(batch_indices, device=device, dtype=torch.long)
                xb = x_train[idx]
                yb = y_train[idx]
                grads, clip_state = grad_fn(params, xb, yb, state=clip_state)
            else:
                # Empty local batch: still need to participate in the
                # cross-rank sum_gradients collective with a zero-shaped
                # contribution.  We pass a one-element xb/yb and zero out
                # the resulting grad so its contribution is null.
                xb = x_train[:1] * 0
                yb = y_train[:1] * 0
                grads, clip_state = grad_fn(params, xb, yb, state=clip_state)
                for name in grads.pytree:
                    grads.pytree[name] = grads.pytree[name] * 0
            pooled = sum_gradients(grads)
            noised, noise_state = noise_fn(pooled, noise_state)
            for name, value in params.items():
                params[name] = value - _LR * (noised.pytree[name] / bs_expected)

        eval_loss = _eval_loss(fmodel, params, x_eval, y_eval)
        if rank == 0:
            Path(out_path).write_text(json.dumps({"eval_loss": float(eval_loss)}))
    finally:
        cleanup_process_group()


def _run(world_size: int) -> float:
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"Requires >= {world_size} CUDA devices")
    if world_size == 1:
        return _run_single(torch.device("cuda:0"))
    with tempfile.TemporaryDirectory() as tmp:
        out = str(Path(tmp) / "metrics.json")
        spawn(world_size, _worker_ddp, out)
        return float(json.loads(Path(out).read_text())["eval_loss"])


def test_dpsgd_sampler_short_run_1_vs_2_gpu_parity() -> None:
    # Real Poisson sampler with fold_in(seed, rank) on each rank — the
    # set of examples seen at each step differs between 1-GPU and 2-GPU,
    # so eval loss trajectories drift.  2% bound = sampler-induced drift
    # over 30 steps must not exceed the noise floor of "still learning
    # the same task" (observed locally on 2× H100: ~0.47%, so ~4×
    # headroom).  See module docstring for the algebraic-identity
    # companion that uses the *same* batch + noise key (bound = 1e-4).
    one = _run(1)
    two = _run(2)
    assert one > 0
    assert two > 0
    rel = abs(two - one) / one
    print(
        f"\nDP-SGD sampler parity: eval_1gpu={one:.6f}, eval_2gpu={two:.6f}, rel={rel:.4%}"
    )
    assert rel < 0.02, f"sampler-active eval relative delta {rel:.4f} exceeds 2%"
