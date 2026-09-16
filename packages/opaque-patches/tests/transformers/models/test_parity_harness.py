# SPDX-FileCopyrightText: 2026 JetBrains
# SPDX-License-Identifier: Apache-2.0
"""The parity harness must hand back a genuinely unpatched reference.

``parity_model_patches`` rebinds module-level names in
``transformers.models.<family>.modeling_<family>``.  Those are plain module
attributes, so unless they are restored the first parity test to touch a family
leaves every later "upstream" reference model running Opaque's own mask
builders and attention — and the suite then compares the patched path against
itself.
"""

from __future__ import annotations

import sys

import pytest
import torch

from ._test_utils import (
    build_patched_model_pair,
    get_tiny_config_kwargs,
    parity_model_patches,
)

# Spelled out rather than imported from ``_test_utils`` so this test still runs
# against a build where the restore logic is absent.
_MODULE_PATCH_NAMES = (
    "create_causal_mask",
    "create_sliding_window_causal_mask",
    "create_recurrent_attention_mask",
    "repeat_kv",
    "eager_attention_forward",
    "apply_rotary_pos_emb",
)

# gemma2 additionally rewrites the process-wide ``ALL_ATTENTION_FUNCTIONS``
# "sdpa" entry, and mistral rebinds the sliding-window mask builder, so between
# them they cover both leak routes.
_FAMILIES = ("mistral", "gemma2", "qwen3_next")


def _imports(family: str):
    module = pytest.importorskip(f"transformers.models.{family}")
    config_cls = getattr(module, f"{family.capitalize()}Config", None)
    model_cls = getattr(module, f"{family.capitalize()}ForCausalLM", None)
    if family == "gemma2":
        config_cls = module.Gemma2Config
        model_cls = module.Gemma2ForCausalLM
    if family == "qwen3_next":
        config_cls = module.Qwen3NextConfig
        model_cls = module.Qwen3NextForCausalLM
    if config_cls is None or model_cls is None:  # pragma: no cover - guard
        pytest.skip(f"{family} config/model classes unavailable")
    return config_cls, model_cls


def _config_kwargs(model_cls) -> dict:
    kwargs = get_tiny_config_kwargs()
    if model_cls.__name__ == "Qwen3NextForCausalLM":
        kwargs.update(
            num_hidden_layers=4,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=64,
            use_cache=False,
        )
    return kwargs


def _reference_logits(config_cls, model_cls, device) -> torch.Tensor:
    """Build a fresh *unpatched* model and run it."""
    torch.manual_seed(0)
    upstream, _ = build_patched_model_pair(
        config_cls, model_cls, device, config_kwargs=_config_kwargs(model_cls)
    )
    upstream.eval()
    input_ids = (
        torch.arange(10, device=device).unsqueeze(0) % upstream.config.vocab_size
    )
    with torch.no_grad():
        return upstream(input_ids=input_ids).logits


@pytest.mark.parametrize("family", _FAMILIES)
def test_reference_output_survives_a_patch_cycle(family, device):
    """Two reference builds must agree across an intervening patch/unpatch."""
    config_cls, model_cls = _imports(family)

    before = _reference_logits(config_cls, model_cls, device)

    torch.manual_seed(0)
    _, patchable = build_patched_model_pair(
        config_cls, model_cls, device, config_kwargs=_config_kwargs(model_cls)
    )
    with parity_model_patches(patchable):
        pass

    after = _reference_logits(config_cls, model_cls, device)

    assert torch.equal(before, after), (
        f"{family}: the upstream reference changed after a patch/unpatch cycle "
        f"(max abs diff {(before - after).abs().max().item():.3e}) — module-level "
        "patches leaked, so parity assertions would compare Opaque against itself"
    )


@pytest.mark.parametrize("family", _FAMILIES)
def test_module_level_names_are_restored(family, device):
    """The rebindable module attributes are the same objects afterwards."""
    config_cls, model_cls = _imports(family)
    torch.manual_seed(0)
    _, patchable = build_patched_model_pair(
        config_cls, model_cls, device, config_kwargs=_config_kwargs(model_cls)
    )
    modeling_module = sys.modules[type(patchable).__module__]

    before = {
        name: getattr(modeling_module, name)
        for name in _MODULE_PATCH_NAMES
        if hasattr(modeling_module, name)
    }
    registry = getattr(modeling_module, "ALL_ATTENTION_FUNCTIONS", None)
    sdpa_before = registry.get("sdpa") if registry is not None else None

    with parity_model_patches(patchable):
        pass

    for name, original in before.items():
        assert getattr(modeling_module, name) is original, (
            f"{family}: {name} was not restored after parity_model_patches"
        )
    if sdpa_before is not None:
        assert modeling_module.ALL_ATTENTION_FUNCTIONS is registry, (
            f"{family}: ALL_ATTENTION_FUNCTIONS was not restored after "
            "parity_model_patches"
        )
        assert registry.get("sdpa") is sdpa_before, (
            f"{family}: ALL_ATTENTION_FUNCTIONS['sdpa'] was not restored"
        )
