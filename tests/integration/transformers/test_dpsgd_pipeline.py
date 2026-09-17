"""DP-SGD end-to-end integration: clip → ``gaussian_noise`` → manual update.

Verifies the full DP-SGD pipeline runs cleanly on a patched LoRA HF
model. Patches are part of the framework's normal usage and apply
throughout, so the tests aren't "patches integration" — they're DP
training integration that happens to enable patches.

Two model sources, both go through the same assertion path:

- A tiny synthetic ``LlamaConfig`` (fast — runs in the PR gate).
- Real Qwen2-0.5B from HuggingFace Hub (marked ``slow``; downloads on
  first run, then cached).

Skipped when ``transformers`` / ``peft`` aren't installed.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("peft")

from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
)

from opake.api.engine.clipping import clipped_grad
from opake.dpsgd.noise import gaussian_noise
from opake.functional import make_functional
from opake.patches import apply_model_patches
from opake.random import key

QWEN2_REPO = "Qwen/Qwen2-0.5B"


def _wrap_in_lora_and_patch(model):
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


def _build_synthetic_lora_model():
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


def _build_qwen2_lora_model():
    config = AutoConfig.from_pretrained(QWEN2_REPO)
    config.num_hidden_layers = 2  # trim for CPU speed
    return _wrap_in_lora_and_patch(AutoModelForCausalLM.from_config(config))


def _run_dpsgd_step(model, input_ids, attention_mask, labels):
    fmodel, trainable, frozen = make_functional(
        model, disable_autograd_tracking=True, partition_trainable=True
    )

    def per_example_loss(params, frozen_params, ids, mask, lbls):
        all_params = {**frozen_params, **params}
        return fmodel(all_params, ids, attention_mask=mask, labels=lbls).loss

    grad_fn, clip_state = clipped_grad(
        per_example_loss,
        argnums=0,
        batch_argnums=(2, 3, 4),
        clipping_norm=1.0,
    )
    grads, _ = grad_fn(
        trainable, frozen, input_ids, attention_mask, labels, state=clip_state
    )

    noise_fn, noise_state = gaussian_noise(noise_multiplier=1.0, key=key(0))
    noised, _ = noise_fn(grads, noise_state)

    batch_size = input_ids.shape[0]
    lr = 0.01
    for name, value in trainable.items():
        updated = value - lr * (noised.pytree[name] / batch_size)
        assert torch.isfinite(updated).all(), f"non-finite param after step: {name}"
        assert updated.shape == value.shape


def test_dpsgd_step_synthetic():
    """DP-SGD step on a tiny synthetic LlamaConfig + LoRA + patches."""
    batch_size = 4
    seq_len = 8
    input_ids = torch.randint(0, 128, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    _run_dpsgd_step(_build_synthetic_lora_model(), input_ids, attention_mask, labels)


@pytest.mark.slow
@pytest.mark.cuda
def test_dpsgd_step_qwen2():
    """DP-SGD step on Qwen2-0.5B + LoRA + patches.

    Exercises the full HF stack — attention masks built by the
    tokenizer, the model's actual RoPE θ, the actual intermediate
    sizes — so it catches issues the synthetic case can't.
    ``slow`` because the first run downloads the model from HF Hub
    (cached afterward); ``cuda`` because Qwen2 forward+vmap+grad is
    too slow on CPU to be worth the runtime cost.
    """
    tokenizer = AutoTokenizer.from_pretrained(QWEN2_REPO)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(
        ["Hello world.", "Another short example.", "Third sample.", "Final one."],
        return_tensors="pt",
        padding=True,
        max_length=16,
        truncation=True,
    )
    _run_dpsgd_step(
        _build_qwen2_lora_model(),
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["input_ids"].clone(),
    )
