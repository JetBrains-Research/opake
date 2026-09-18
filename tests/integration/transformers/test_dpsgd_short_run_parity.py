"""Short DP-SGD runs: single-GPU vs 2-GPU parity (synthetic LoRA causal LM)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("peft")

from opake_test_support import cleanup_process_group, setup_nccl, spawn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, LlamaConfig

from opake.api.engine.clipping import clipped_grad
from opake.distributed import sum_gradients
from opake.dpsgd.noise import gaussian_noise
from opake.functional import make_functional
from opake.patches import apply_model_patches
from opake.random import key

pytestmark = [pytest.mark.slow, pytest.mark.cuda]


def _wrap_in_lora_and_patch(model: torch.nn.Module) -> torch.nn.Module:
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


def _build_synthetic_lora_model() -> torch.nn.Module:
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rope_theta=10000.0,
    )
    return _wrap_in_lora_and_patch(AutoModelForCausalLM.from_config(config))


def _reference_state_dict() -> dict[str, torch.Tensor]:
    torch.manual_seed(2025)
    return _build_synthetic_lora_model().state_dict()


def _batch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(424242)
    input_ids = torch.randint(0, 128, (4, 8), device=device, generator=g)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return input_ids, attention_mask, labels


def _eval_loss(
    fmodel,
    trainable: dict[str, torch.Tensor],
    frozen: dict[str, torch.Tensor],
    ids: torch.Tensor,
    mask: torch.Tensor,
    lbls: torch.Tensor,
) -> float:
    with torch.no_grad():
        all_p = {**frozen, **trainable}
        return float(fmodel(all_p, ids, attention_mask=mask, labels=lbls).loss)


def _grad_norm(noised: Any) -> float:
    sq = 0.0
    for t in noised.pytree.values():
        if isinstance(t, torch.Tensor):
            sq += float(t.detach().float().pow(2).sum().item())
    return sq**0.5


def _run_single() -> dict[str, float]:
    device = torch.device("cuda:0")
    model = _build_synthetic_lora_model().to(device)
    model.load_state_dict({k: v.to(device) for k, v in _reference_state_dict().items()})
    ids, mask, lbls = _batch(device)
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
    noise_fn, noise_state = gaussian_noise(noise_multiplier=0.3, key=key(99))
    lr = 0.02
    train_losses: list[float] = []
    last_gnorm = 0.0
    for _ in range(5):
        train_losses.append(_eval_loss(fmodel, trainable, frozen, ids, mask, lbls))
        grads, clip_state = grad_fn(
            trainable, frozen, ids, mask, lbls, state=clip_state
        )
        noised, noise_state = noise_fn(grads, noise_state)
        last_gnorm = _grad_norm(noised)
        bs = ids.shape[0]
        for name, value in trainable.items():
            trainable[name] = value - lr * (noised.pytree[name] / bs)

    eval_loss = _eval_loss(fmodel, trainable, frozen, ids, mask, lbls)
    return {
        "train_loss": float(train_losses[-1]),
        "eval_loss": float(eval_loss),
        "final_grad_norm": float(last_gnorm),
    }


def _worker_ddp_short(rank: int, world_size: int, port: int, out_path: str) -> None:
    setup_nccl(rank, world_size, port)
    try:
        device = torch.device(f"cuda:{rank}")
        model = _build_synthetic_lora_model().to(device)
        model.load_state_dict(
            {k: v.to(device) for k, v in _reference_state_dict().items()}
        )
        ids_f, mask_f, lbls_f = _batch(torch.device("cuda:0"))
        # identical logical batch on all ranks (deterministic generator seed)
        ids = ids_f.to(device)
        mask = mask_f.to(device)
        lbls = lbls_f.to(device)
        sl = slice(rank * 2, (rank + 1) * 2)
        ids = ids[sl]
        mask = mask[sl]
        lbls = lbls[sl]

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
        noise_fn, noise_state = gaussian_noise(noise_multiplier=0.3, key=key(99))
        lr = 0.02
        train_losses: list[float] = []
        last_gnorm = 0.0
        full_ids, full_mask, full_lbls = _batch(torch.device("cuda:0"))
        for _ in range(5):
            train_losses.append(
                _eval_loss(
                    fmodel,
                    trainable,
                    frozen,
                    full_ids.to(device),
                    full_mask.to(device),
                    full_lbls.to(device),
                )
            )
            grads, clip_state = grad_fn(
                trainable, frozen, ids, mask, lbls, state=clip_state
            )
            pooled = sum_gradients(grads)
            noised, noise_state = noise_fn(pooled, noise_state)
            last_gnorm = _grad_norm(noised)
            bs = 4
            for name, value in trainable.items():
                trainable[name] = value - lr * (noised.pytree[name] / bs)

        eval_loss = _eval_loss(
            fmodel,
            trainable,
            frozen,
            full_ids.to(device),
            full_mask.to(device),
            full_lbls.to(device),
        )
        if rank == 0:
            payload = {
                "train_loss": float(train_losses[-1]),
                "eval_loss": float(eval_loss),
                "final_grad_norm": float(last_gnorm),
            }
            Path(out_path).write_text(json.dumps(payload))
    finally:
        cleanup_process_group()


def _run(world_size: int) -> dict[str, float]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"Requires >= {world_size} CUDA devices")
    if world_size == 1:
        return _run_single()
    with tempfile.TemporaryDirectory() as tmp:
        out = str(Path(tmp) / "metrics.json")
        spawn(world_size, _worker_ddp_short, out)
        return json.loads(Path(out).read_text())


def test_dpsgd_short_run_1_vs_2_gpu_parity() -> None:
    # Algebraic-identity parity: both runs slice the *same* fixed batch and
    # use the *same* fixed noise key, so per-record clipping + sum_gradients
    # + replicated noise must agree to FP32 summation-order precision over
    # the 5-step rollout.  Tolerance ~1e-4 leaves ample headroom over the
    # observed ~0% drift while still catching real bugs (missing reduce,
    # wrong noise scaling, sampler-keying error).  Sampler-induced
    # trajectory drift between ranks is covered by
    # ``tests/integration/ddp/test_dpsgd_sampler_short_run_parity.py``.
    one = _run(1)
    two = _run(2)
    assert one["eval_loss"] > 0
    assert two["eval_loss"] > 0
    rel = abs(two["eval_loss"] - one["eval_loss"]) / one["eval_loss"]
    assert rel < 1e-4, f"eval relative delta {rel:.2e} exceeds 1e-4"
