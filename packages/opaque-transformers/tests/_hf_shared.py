"""Shared helpers for opaque-transformers validation + distributed tests.

Contains pure-Python utilities (``MODEL_CONFIGS``, ``STANDARD_LORA_CONFIG``,
``build_text_batch``, ``build_lm_dataset``, ``has_min_gpu_memory``,
``gpu_memory_gate_reason``, ``load_model_with_lora``,
``run_dp_training_step``) used by tests under ``validation/`` and
``distributed/``.  Session-scoped fixtures live in the sibling ``conftest.py``,
which calls ``opaque.patches.apply_runtime_patches(compat=True)`` so global HF
runtime compat shims are active for the whole test tree (Trainer also applies
them on construction).
"""

from __future__ import annotations

import copy
import functools
from typing import Any

import torch
from opaque_test_support import get_default_gpu_device

# =============================================================================
# Model Testing Utilities
# =============================================================================

STANDARD_LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
}

MODEL_CONFIGS = {
    "qwen2-0.5b": {
        "model_id": "Qwen/Qwen2-0.5B",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "min_mem_gb": 12,
        "max_length": 2048,
        "trust_remote_code": False,
        "batch_size": 8,
        "accum_steps": 32,
    },
    "qwen2-1.5b": {
        "model_id": "Qwen/Qwen2-1.5B",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "min_mem_gb": 12,
        "max_length": 1024,
        "trust_remote_code": False,
        "batch_size": 8,
        "accum_steps": 16,
    },
    "tinyllama-1.1b": {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "min_mem_gb": 12,
        "max_length": 1024,
        "trust_remote_code": False,
        "batch_size": 8,
        "accum_steps": 16,
    },
    "phi3-mini": {
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "target_modules": ["qkv_proj"],
        "min_mem_gb": 16,
        "max_length": 2048,
        "trust_remote_code": True,
        "batch_size": 8,
        "accum_steps": 16,
    },
}


def build_text_batch(batch_size):
    """Create deterministic text batch for testing."""
    base_texts = [
        "Hello world test",
        "Another example",
        "Third sample",
        "Final one",
        "Short prompt",
        "Yet another example",
        "Testing a longer input",
        "Final sample in batch",
    ]
    if batch_size <= len(base_texts):
        return base_texts[:batch_size]
    repeats = (batch_size + len(base_texts) - 1) // len(base_texts)
    return (base_texts * repeats)[:batch_size]


def build_lm_dataset(
    texts: list[str],
    tokenizer: Any,
    max_length: int = 32,
) -> Any:
    """Return a tokenised causal-LM ``Dataset`` ready for ``default_data_collator``.

    Each example carries pre-padded ``input_ids`` (length ``max_length``,
    pad token id used for filler), ``attention_mask`` (1 on real tokens,
    0 on pads), and ``labels`` (copy of ``input_ids`` with pad
    positions set to -100 so the causal-LM loss skips them).  Designed
    for HF's ``default_data_collator``, which simply stacks the
    pre-shaped tensors by key without any further padding.

    Used by Trainer integration tests to build minimal HF-shaped
    datasets without inventing trainer-side abstractions.
    """
    from datasets import Dataset  # local import: optional dep at module level.

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0

    rows = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)[:max_length]
        n = len(ids)
        input_ids = ids + [pad_id] * (max_length - n)
        attention_mask = [1] * n + [0] * (max_length - n)
        labels = list(ids) + [-100] * (max_length - n)
        rows.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )
    return Dataset.from_list(rows)


# =============================================================================
# Lightweight GPT-2 factory for CPU mechanics tests
# =============================================================================
#
# The Trainer validation suite exercises *training / eval / checkpoint
# mechanics*, not pretrained-GPT-2 semantics — assertions are of the form
# ``training_loss > 0``, ``generate() runs``, checkpoints round-trip, runs are
# bit-identical, DP noise is applied, etc. None of that needs the 124M-param
# pretrained weights. Constructing real GPT-2 per test (``from_pretrained`` ≈
# 0.8s each) plus running fwd/bwd on 124M params on CPU dominated the suite.
#
# ``make_gpt2()`` returns a *tiny* (2-layer, 128-dim ≈ 7M-param) randomly
# initialised GPT-2 deep-copied from a cached template, paired with the real
# GPT-2 tokenizer. Keeping ``vocab_size=50257`` means the real tokenizer's ids
# stay in range, so datasets built with it work unchanged. The template is
# seeded, so every call yields identical starting weights — preserving the
# "two models start equal" property that ``from_pretrained`` provided to the
# reproducibility/determinism tests.


@functools.lru_cache(maxsize=1)
def gpt2_tokenizer():
    """Cached real GPT-2 tokenizer with ``pad_token`` set to ``eos``."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok


@functools.lru_cache(maxsize=1)
def _tiny_gpt2_template():
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained("gpt2")
    cfg.n_layer = 2
    cfg.n_embd = 128
    cfg.n_head = 2
    # vocab_size deliberately left at 50257 so the real tokenizer stays valid.
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(cfg)
    return model


def make_gpt2_model():
    """Fresh tiny GPT-2 (random init) with ``pad_token_id`` set.

    Drop-in for a bare ``AutoModelForCausalLM.from_pretrained("gpt2")``.
    """
    model = copy.deepcopy(_tiny_gpt2_template())
    model.config.pad_token_id = gpt2_tokenizer().pad_token_id
    return model


def make_gpt2():
    """Fresh tiny GPT-2 (random init) + cached tokenizer.

    Drop-in replacement for ``AutoModelForCausalLM.from_pretrained("gpt2")`` +
    ``AutoTokenizer.from_pretrained("gpt2")`` in CPU mechanics tests.
    """
    return make_gpt2_model(), gpt2_tokenizer()


def has_min_gpu_memory(min_gb, device=None):
    """Check if GPU has minimum required memory."""
    gpu_device = get_default_gpu_device() if device is None else torch.device(device)

    if gpu_device is None:
        return False

    required_bytes = int(min_gb * (1024**3))

    if gpu_device.type == "cuda":
        try:
            free_bytes, _total_from_driver = torch.cuda.mem_get_info(0)
            if free_bytes > 0:
                return free_bytes >= required_bytes
        except Exception:
            pass
        try:
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            return total_bytes >= required_bytes
        except Exception:
            return False

    if gpu_device.type == "mps":
        try:
            recommended_bytes = None
            allocated_bytes = None

            if hasattr(torch.mps, "recommended_max_memory"):
                candidate = int(torch.mps.recommended_max_memory())
                if candidate > 0:
                    recommended_bytes = candidate

            if hasattr(torch.mps, "current_allocated_memory"):
                candidate = int(torch.mps.current_allocated_memory())
                if candidate >= 0:
                    allocated_bytes = candidate

            if recommended_bytes is not None and allocated_bytes is not None:
                free_bytes = max(0, recommended_bytes - allocated_bytes)
                return free_bytes >= required_bytes

            if recommended_bytes is not None:
                return recommended_bytes >= required_bytes
        except Exception:
            return False
        return False

    return False


def gpu_memory_gate_reason(min_gb, device=None):
    """Return standardized skip reason for GPU memory gating."""
    gpu_device = get_default_gpu_device() if device is None else torch.device(device)

    if gpu_device is None:
        return f"Requires GPU with >= {min_gb}GB memory"

    if gpu_device.type == "cuda":
        try:
            free_bytes, _total_from_driver = torch.cuda.mem_get_info(0)
            free_gb = free_bytes / (1024**3)
            return (
                f"Requires >= {min_gb}GB free CUDA memory "
                f"(currently {free_gb:.2f}GB free)"
            )
        except Exception:
            return f"Requires CUDA GPU with >= {min_gb}GB memory"

    if gpu_device.type == "mps":
        try:
            recommended = None
            allocated = None
            if hasattr(torch.mps, "recommended_max_memory"):
                recommended = int(torch.mps.recommended_max_memory())
            if hasattr(torch.mps, "current_allocated_memory"):
                allocated = int(torch.mps.current_allocated_memory())

            if (
                recommended
                and recommended > 0
                and allocated is not None
                and allocated >= 0
            ):
                free_gb = max(0, recommended - allocated) / (1024**3)
                return (
                    f"Requires >= {min_gb}GB free MPS memory "
                    f"(estimated {free_gb:.2f}GB free)"
                )

            if recommended and recommended > 0:
                recommended_gb = recommended / (1024**3)
                return (
                    f"Requires >= {min_gb}GB MPS recommended memory "
                    f"(currently {recommended_gb:.2f}GB)"
                )
        except Exception:
            pass

        return f"Requires MPS memory introspection and >= {min_gb}GB available"

    return f"Requires GPU with >= {min_gb}GB memory"


def load_model_with_lora(
    model_config, device="cuda", dtype=torch.float16, lora_config=None
):
    """Load HuggingFace model with LoRA adapters."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_id = model_config["model_id"]
    trust_remote_code = model_config["trust_remote_code"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    ).to(device)

    if lora_config is None:
        lora_config = STANDARD_LORA_CONFIG.copy()

    lora_config["target_modules"] = model_config["target_modules"]
    model = get_peft_model(model, LoraConfig(**lora_config))

    return model, tokenizer


def run_dp_training_step(
    model,
    tokenizer,
    batch_size,
    max_length,
    accum_steps,
    training_steps=3,
    learning_rate=1e-3,
    clipping_norm=1.0,
):
    """Run DP-SGD training with clipped gradients and gradient accumulation.

    Returns:
        tuple: ``(accumulated, state)`` where ``accumulated`` is a
        :class:`~opaque.types.ClippedPytree` for the last training step
        (access tensor leaves via ``accumulated.pytree``) and ``state`` is
        the final clipping state.
    """
    from opaque.api.engine.clipping import clipped_grad
    from opaque.functional import make_functional
    from opaque.pytree import tree_map
    from opaque.types import clipped

    device = next(model.parameters()).device

    texts = build_text_batch(batch_size)
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        max_length=max_length,
        truncation=True,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    labels = input_ids.clone()

    fmodel, trainable, frozen = make_functional(
        model,
        disable_autograd_tracking=True,
        partition_trainable=True,
    )

    def per_example_loss(
        trainable_params, frozen_params, ids_single, mask_single, labels_single
    ):
        all_params = {**frozen_params, **trainable_params}
        outputs = fmodel(
            all_params, ids_single, attention_mask=mask_single, labels=labels_single
        )
        return outputs.loss

    grad_fn, clip_state = clipped_grad(
        per_example_loss,
        argnums=0,
        batch_argnums=(2, 3, 4),
        clipping_norm=clipping_norm,
    )

    state = clip_state
    trainable_params = trainable
    last_accumulated = None

    for _step in range(training_steps):
        accumulated_pytree = None
        accumulated_max_norm = None

        for _ in range(accum_steps):
            grads, state = grad_fn(
                trainable_params,
                frozen,
                input_ids,
                attention_mask,
                labels,
                state=state,
            )

            if accumulated_pytree is None:
                accumulated_pytree = tree_map(
                    lambda x: x.detach().clone(), grads.pytree
                )
                accumulated_max_norm = grads.max_norm
            else:
                accumulated_pytree = tree_map(
                    lambda x, y: x + y, accumulated_pytree, grads.pytree
                )

        scale = 1.0 / float(accum_steps)
        accumulated = clipped(
            tree_map(lambda x, s=scale: x * s, accumulated_pytree),
            max_norm=accumulated_max_norm,
        )

        trainable_params = tree_map(
            lambda p, g: p - learning_rate * g,
            trainable_params,
            accumulated.pytree,
        )

        last_accumulated = accumulated

    return last_accumulated, state
