# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Reusable test-only fixtures and helpers.

This module is deliberately outside published packages. It contains generic
test infrastructure shared by package-local test suites; feature-specific
assertions and multiprocessing worker entry points remain with their owners.
"""

from __future__ import annotations

import os
import socket
from contextlib import contextmanager
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

if TYPE_CHECKING:
    from datetime import timedelta

_HF_TOKEN_ENVIRONMENT_VARIABLES = (
    "HF_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "HUGGINGFACE_TOKEN",
)


def get_default_device() -> torch.device:
    """Return the requested device, or CUDA, MPS, then CPU."""
    requested = os.environ.get("OPAKE_TEST_DEVICE", "").strip().lower()
    if requested:
        if requested == "cpu":
            return torch.device("cpu")
        if requested == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            raise RuntimeError(
                "OPAKE_TEST_DEVICE=cuda requested but CUDA is unavailable"
            )
        if requested == "mps":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            raise RuntimeError("OPAKE_TEST_DEVICE=mps requested but MPS is unavailable")
        raise RuntimeError(
            f"Invalid OPAKE_TEST_DEVICE={requested!r}. Expected one of: cpu, cuda, mps"
        )

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_default_gpu_device() -> torch.device | None:
    """Return CUDA, MPS, or ``None`` when no accelerator is available."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return None


def set_random_seed(seed: int = 42) -> None:
    """Seed every available Torch device for reproducible tests."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def find_free_port() -> int:
    """Reserve an available local port for a temporary process group."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _configure_process_group_environment(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)


def setup_nccl(rank: int, world_size: int, port: int) -> None:
    """Initialize an NCCL process group on the rank's CUDA device."""
    if not dist.is_available() or not torch.cuda.is_available():
        raise RuntimeError("DDP requires CUDA and torch.distributed support")
    _configure_process_group_environment(rank, world_size, port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def setup_gloo(
    rank: int,
    world_size: int,
    port: int,
    *,
    timeout: timedelta | None = None,
) -> None:
    """Initialize a CPU/Gloo process group, with an optional test timeout."""
    _configure_process_group_environment(rank, world_size, port)
    kwargs: dict[str, Any] = {
        "backend": "gloo",
        "rank": rank,
        "world_size": world_size,
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
    dist.init_process_group(**kwargs)


def cleanup_process_group() -> None:
    """Destroy the active process group, if a worker initialized one."""
    if dist.is_initialized():
        dist.destroy_process_group()


def spawn(world_size: int, fn: Any, *args: Any) -> None:
    """Run a module-level worker function once per rank."""
    mp.spawn(
        fn,
        args=(world_size, find_free_port(), *args),
        nprocs=world_size,
        join=True,
    )


def has_hf_token() -> bool:
    """Return whether a Hugging Face token is available for gated-model tests."""
    return any(os.getenv(name) for name in _HF_TOKEN_ENVIRONMENT_VARIABLES)


requires_hf_auth = pytest.mark.skipif(
    not has_hf_token(),
    reason="HF token not set (test loads a gated HuggingFace model)",
)


@pytest.fixture(scope="module")
def qwen2_config():
    """Build a tiny, structurally faithful Qwen2 configuration."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained("Qwen/Qwen2-0.5B")
    config.num_hidden_layers = 2
    config.hidden_size = 128
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = config.hidden_size // config.num_attention_heads
    config.intermediate_size = 256
    return config


@pytest.fixture(scope="module")
def qwen2_tokenizer():
    """Return the Qwen2 tokenizer used by compatibility tests."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")


def prepare_lora_model(
    config: Any,
    target_modules: list[str] | None = None,
    *,
    apply_patches: bool = False,
) -> Any:
    """Create a Qwen-compatible LoRA model, optionally with Opake patches."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]
    model = get_peft_model(
        AutoModelForCausalLM.from_config(config),
        LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.0,
        ),
    )
    if apply_patches:
        from opake.patches import apply_model_patches

        apply_model_patches(
            model,
            performance=False,
            compat=True,
            lora=True,
            activation=False,
            rms_norm=False,
            rope=False,
            cross_entropy=False,
            eager_attention=True,
        )
    return model


def run_clipped_grad_test(
    model: Any,
    tokenizer: Any,
    device: torch.device | None = None,
    *,
    apply_patches: bool = True,
) -> tuple[Any, Any]:
    """Run the real clipped-gradient path and return its gradients and state."""
    from opake.api.engine.clipping import clipped_grad
    from opake.functional import make_functional

    if device is None:
        device = next(model.parameters()).device
    if apply_patches:
        from opake.patches import apply_model_patches

        apply_model_patches(model, compat=True, performance=False)

    texts = ["Hello world test", "Another example", "Third sample", "Final one"]
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, max_length=16, truncation=True
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    labels = input_ids.clone()

    fmodel, trainable, frozen = make_functional(
        model, disable_autograd_tracking=True, partition_trainable=True
    )

    def per_example_loss(trainable_params, frozen_params, ids, mask, lbls):
        all_params = {**frozen_params, **trainable_params}
        outputs = fmodel(all_params, ids, attention_mask=mask, labels=lbls)
        return outputs.loss

    grad_fn, clip_state = clipped_grad(
        per_example_loss, argnums=0, batch_argnums=(2, 3, 4), clipping_norm=1.0
    )
    return grad_fn(
        trainable, frozen, input_ids, attention_mask, labels, state=clip_state
    )


@contextmanager
def fast_mc_accounting():
    """Temporarily lower Monte Carlo accounting resolution for smoke tests."""
    import opake.accounting as acc
    from opake.accounting import discretization

    original = discretization._default_config
    fast_config = replace(
        acc.get_discretization(),
        mc_resolution=5e-3,
        mc_failure_probability=1e-2,
    )
    acc.set_discretization(**asdict(fast_config))
    try:
        yield
    finally:
        discretization._default_config = original
