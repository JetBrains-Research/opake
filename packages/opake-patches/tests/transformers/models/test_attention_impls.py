# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Attention-implementation validation.

The per-family suites run under ``sdpa`` (the transformers default). This file
keeps an explicit cross-check that representative families — one dense (llama),
one MoE (mellum) — work under DP ``vmap(grad)`` on BOTH ``eager`` (the
O(N²) reference) and ``sdpa``.
"""

import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _test_utils import (
    assert_forward_no_grad,
    assert_vmap_grad,
    build_moe_model,
    get_tiny_config_kwargs,
)

IMPLS = ["eager", "sdpa"]


def _build_llama(device, impl="sdpa"):
    from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM

    from opake.patches import apply_model_patches

    config = LlamaConfig(**get_tiny_config_kwargs())
    config._attn_implementation = impl
    model = LlamaForCausalLM(config).to(device)
    apply_model_patches(model, eager_attention=True)
    return model


def _sdpa_backends():
    from torch.nn.attention import SDPBackend

    return [
        pytest.param(SDPBackend.MATH, id="math"),
        pytest.param(SDPBackend.EFFICIENT_ATTENTION, id="efficient"),
        pytest.param(SDPBackend.CUDNN_ATTENTION, id="cudnn"),
        # Flash isn't selectable under vmap ("No available kernel"); xfail
        # non-strict so it auto-passes if a future torch enables it.
        pytest.param(
            SDPBackend.FLASH_ATTENTION,
            id="flash",
            marks=pytest.mark.xfail(
                reason="flash not selected under vmap", strict=False
            ),
        ),
    ]


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="SDPA backends need CUDA")
@pytest.mark.parametrize("backend", _sdpa_backends())
def test_sdpa_backends_under_vmap(backend, device):
    """Every selectable SDPA backend works under DP vmap(grad). MATH is
    vmap-native; efficient/cudnn run via the per-example-loop fallback until the
    upstream batching-rule patch lands; flash isn't selected (xfail)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    # The fused SDPA kernels (efficient/cudnn/flash) only provide a bf16 path on
    # Ampere+ (sm>=80); on older GPUs (e.g. Turing/T4, sm_75) they raise
    # "No available kernel". The MATH backend is pure-PyTorch and runs anywhere.
    if backend is not SDPBackend.MATH and device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device)
        if major < 8:
            pytest.skip(
                "fused bf16 SDPA (efficient/cudnn/flash) requires CUDA sm>=80; "
                f"this GPU is sm_{major}x"
            )

    model = _build_llama(device, "sdpa")
    with sdpa_kernel([backend]):
        assert_vmap_grad(model, device, dtype=torch.bfloat16)


@pytest.mark.parametrize("impl", IMPLS)
def test_llama_attention_impl(impl, device):
    from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM

    from opake.patches import apply_model_patches

    config = LlamaConfig(**get_tiny_config_kwargs())
    config._attn_implementation = impl
    model = LlamaForCausalLM(config).to(device)
    apply_model_patches(model, eager_attention=True)
    assert_forward_no_grad(model, device)
    assert_vmap_grad(model, device)


@pytest.mark.parametrize("impl", IMPLS)
def test_mellum_attention_impl(impl, device):
    model, _ = build_moe_model(
        "mellum",
        device,
        attn_impl=impl,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
    )
    assert_forward_no_grad(model, device)
    assert_vmap_grad(model, device)
