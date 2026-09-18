# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Single-GPU validation tests for real HuggingFace models with LoRA + DP-SGD.

Tests use shared model configs and utilities from tests/conftest.py to avoid duplication.
"""

import pytest
from _hf_shared import (
    MODEL_CONFIGS,
    get_default_gpu_device,
    gpu_memory_gate_reason,
    has_min_gpu_memory,
    load_model_with_lora,
    run_dp_training_step,
)

from opake.types import ClippedPytree

transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")


@pytest.mark.slow
@pytest.mark.cuda
class TestRealModelsSingleGPU:
    """Single-GPU validation with real pretrained weights using shared utilities."""

    @pytest.mark.parametrize(
        "model_key",
        ["qwen2-0.5b", "qwen2-1.5b", "tinyllama-1.1b", "phi3-mini"],
        ids=lambda k: MODEL_CONFIGS[k]["model_id"],
    )
    def test_model_lora_dp_training(self, model_key):
        """Run DP-SGD training with LoRA using shared test utilities."""
        config = MODEL_CONFIGS[model_key]
        gpu_device = get_default_gpu_device()

        if gpu_device is None:
            pytest.skip("No GPU available (CUDA or MPS)")

        if not has_min_gpu_memory(config["min_mem_gb"], device=gpu_device):
            pytest.skip(gpu_memory_gate_reason(config["min_mem_gb"], device=gpu_device))

        # Load model using shared utility
        model, tokenizer = load_model_with_lora(config, device=str(gpu_device))

        # Run training using shared utility
        grads, _state = run_dp_training_step(
            model,
            tokenizer,
            batch_size=config["batch_size"],
            max_length=config["max_length"],
            accum_steps=config["accum_steps"],
            training_steps=3,
            learning_rate=1e-3,
            clipping_norm=1.0,
        )

        # Verify training produced clipped gradients with preserved metadata
        assert isinstance(grads, ClippedPytree)
        assert len(grads.pytree) > 0
