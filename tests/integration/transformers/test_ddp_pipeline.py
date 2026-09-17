"""DDP end-to-end DP-pipeline integration on a Qwen2 model.

Spawns a DDP process group, builds a patched LoRA Qwen2-0.5B on each
rank, and runs a single DP step (clip → noise → cross-rank
``sum_gradients`` → manual update) under both DP-SGD and DP-FTRL noise
mechanisms. This is a one-step smoke — not a training run.

Combines DDP × patches × Qwen2 × DP-SGD/DP-FTRL. Multi-GPU primitive
coverage lives under ``packages/opake-*/tests/ddp/``; this file
verifies the user-facing pipeline holds together with HF-Hub model
weights and the patches enabled.

Markers:

- ``slow`` — first run downloads Qwen2-0.5B from HF Hub.
- ``cuda`` — DDP needs NCCL + multi-GPU.

Skipped automatically when fewer than 2 CUDA devices are visible
(``world_size = 2``).
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("peft")

from opake_test_support import (
    cleanup_process_group as _cleanup_ddp,
)
from opake_test_support import (
    setup_nccl as _setup_ddp,
)
from opake_test_support import (
    spawn as _spawn,
)
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from opake.api.engine.clipping import clipped_grad
from opake.distributed import sum_gradients
from opake.dpftrl.noise import identity_strategy, mf_gaussian_noise
from opake.dpsgd.noise import gaussian_noise
from opake.functional import make_functional
from opake.patches import apply_model_patches
from opake.random import fold_in, key

QWEN2_REPO = "Qwen/Qwen2-0.5B"

pytestmark = [pytest.mark.slow, pytest.mark.cuda]


def _has_two_gpus() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() >= 2


def _build_patched_qwen2_lora(device: torch.device):
    config = AutoConfig.from_pretrained(QWEN2_REPO)
    config.num_hidden_layers = 2
    model = AutoModelForCausalLM.from_config(config).to(device)
    lora = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
    )
    model = get_peft_model(model, lora)
    apply_model_patches(
        model,
        performance=False,
        compat=True,
        lora=True,
        eager_attention=True,
    )
    return model


def _tokenize(device: torch.device, rank: int, world_size: int):
    tokenizer = AutoTokenizer.from_pretrained(QWEN2_REPO)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Slice the corpus by rank — each worker sees a different shard.
    corpus = [
        "Hello world.",
        "Another short example.",
        "Third sample.",
        "Final one.",
        "DP-SGD with patches.",
        "DP-FTRL with patches.",
        "Cross-rank reduction.",
        "Distributed training smoke.",
    ]
    shard = corpus[rank::world_size][:2]
    inputs = tokenizer(
        shard,
        return_tensors="pt",
        padding=True,
        max_length=16,
        truncation=True,
    ).to(device)
    return inputs["input_ids"], inputs["attention_mask"], inputs["input_ids"].clone()


def _run_dp_step(model, ids, mask, lbls, *, noise_kind: str, k):
    fmodel, trainable, frozen = make_functional(
        model, disable_autograd_tracking=True, partition_trainable=True
    )

    def per_example_loss(params, frozen_params, x_ids, x_mask, x_lbls):
        all_params = {**frozen_params, **params}
        return fmodel(all_params, x_ids, attention_mask=x_mask, labels=x_lbls).loss

    grad_fn, clip_state = clipped_grad(
        per_example_loss,
        argnums=0,
        batch_argnums=(2, 3, 4),
        clipping_norm=1.0,
    )
    local_grads, _ = grad_fn(trainable, frozen, ids, mask, lbls, state=clip_state)

    # Cross-rank sum of clipped per-example gradients.
    pooled = sum_gradients(local_grads)

    if noise_kind == "dpsgd":
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=k)
    elif noise_kind == "dpftrl":
        strategy = identity_strategy()
        noise_fn, state = mf_gaussian_noise(
            pooled,
            strategy,
            n_steps=1,
            noise_multiplier=1.0,
            key=k,
        )
    else:
        raise ValueError(f"unknown noise_kind={noise_kind!r}")

    noised, _ = noise_fn(pooled, state)
    return trainable, noised


def _assert_finite_step(trainable, noised, batch_size: int) -> None:
    lr = 0.01
    for name, value in trainable.items():
        updated = value - lr * (noised.pytree[name] / batch_size)
        assert torch.isfinite(updated).all(), f"non-finite param after step: {name}"
        assert updated.shape == value.shape


def _worker_dpsgd(rank: int, world_size: int, port: int) -> None:
    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        model = _build_patched_qwen2_lora(device)
        ids, mask, lbls = _tokenize(device, rank, world_size)

        # Same seed across ranks — identical noise draw, deterministic test.
        trainable, noised = _run_dp_step(
            model,
            ids,
            mask,
            lbls,
            noise_kind="dpsgd",
            k=key(0),
        )
        # batch_size is the *aggregated* batch across ranks
        _assert_finite_step(trainable, noised, batch_size=ids.shape[0] * world_size)
    finally:
        _cleanup_ddp()


def _worker_dpftrl(rank: int, world_size: int, port: int) -> None:
    _setup_ddp(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        model = _build_patched_qwen2_lora(device)
        ids, mask, lbls = _tokenize(device, rank, world_size)

        trainable, noised = _run_dp_step(
            model,
            ids,
            mask,
            lbls,
            noise_kind="dpftrl",
            k=fold_in(key(0), 1),
        )
        _assert_finite_step(trainable, noised, batch_size=ids.shape[0] * world_size)
    finally:
        _cleanup_ddp()


def test_ddp_dpsgd_step_qwen2():
    """DDP + DP-SGD step on a patched LoRA Qwen2 — full pipeline smoke."""
    if not _has_two_gpus():
        pytest.skip("DDP test requires ≥2 CUDA devices")
    _spawn(world_size=2, fn=_worker_dpsgd)


def test_ddp_dpftrl_step_qwen2():
    """DDP + DP-FTRL step (identity strategy) on a patched LoRA Qwen2."""
    if not _has_two_gpus():
        pytest.skip("DDP test requires ≥2 CUDA devices")
    _spawn(world_size=2, fn=_worker_dpftrl)
