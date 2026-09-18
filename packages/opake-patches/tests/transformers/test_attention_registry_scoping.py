# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""An attention-interface override must stay inside the family that asked for it.

HuggingFace ships a single ``AttentionInterface`` instance that every
``modeling_X`` module imports, so writing ``ALL_ATTENTION_FUNCTIONS["sdpa"]``
through one family's module reroutes every other family too.  Gemma2 is the
only family opake overrides SDPA for (its ``softcap`` needs a chunked
implementation), and its shim used to land in that shared instance — which
mattered for families that forward ``softcap`` into the interface: vaultgemma
defaults ``attn_logit_softcapping`` to 50.0 and was therefore pulled into
gemma2's implementation.
"""

import pytest

pytest.importorskip("transformers")

import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import (
    ALL_ATTENTION_FUNCTIONS as SHARED_ATTENTION_FUNCTIONS,
)
from transformers.models.gemma2 import modeling_gemma2
from transformers.models.llama import modeling_llama
from transformers.models.mistral import modeling_mistral

from opake.api.patches.transformers.components import attention as opake_attention
from opake.api.patches.transformers.models.gemma2 import apply_gemma2_family_patches
from opake.api.patches.transformers.models.llama import apply_llama_family_patches
from opake.api.patches.transformers.models.mistral import apply_mistral_family_patches
from opake.api.patches.transformers.runtime.masking import (
    vmap_create_compact_sdpa_sliding_window_causal_mask,
)
from opake.patches import apply_runtime_patches

_TINY_CONFIG = {
    "vocab_size": 64,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 16,
}


@pytest.fixture
def patched_gemma2_family():
    """Install opake's runtime patches plus the gemma2 family patches."""
    apply_runtime_patches()
    apply_gemma2_family_patches(eager_attention=True)


@pytest.fixture
def softcap_fallback_spy(monkeypatch):
    """Record every call that reaches gemma2's softcap eager fallback."""
    calls = []
    real = opake_attention.vmap_eager_attention_forward_gemma2

    def spy(*args, **kwargs):
        calls.append(kwargs.get("softcap"))
        return real(*args, **kwargs)

    # The shim resolves the fallback through its own module globals.
    monkeypatch.setattr(
        opake_attention, "vmap_eager_attention_forward_gemma2", spy, raising=True
    )
    return calls


def test_shared_attention_registry_keeps_stock_sdpa(patched_gemma2_family):
    """Family overrides never replace the process-wide SDPA implementation."""
    assert SHARED_ATTENTION_FUNCTIONS["sdpa"] is sdpa_attention_forward
    assert (
        modeling_llama.ALL_ATTENTION_FUNCTIONS["sdpa"]
        is not opake_attention.vmap_sdpa_attention_forward_gemma2
    )


def test_default_gqa_sdpa_override_is_family_scoped():
    apply_runtime_patches()
    apply_llama_family_patches(eager_attention=True)

    assert SHARED_ATTENTION_FUNCTIONS["sdpa"] is sdpa_attention_forward
    assert modeling_llama.ALL_ATTENTION_FUNCTIONS is not SHARED_ATTENTION_FUNCTIONS
    assert (
        modeling_llama.ALL_ATTENTION_FUNCTIONS["sdpa"]
        is opake_attention.vmap_sdpa_attention_forward
    )


def test_gemma2_module_gets_a_private_interface(patched_gemma2_family):
    """Gemma2 keeps the vmap-safe shim, on an interface of its own."""
    assert modeling_gemma2.ALL_ATTENTION_FUNCTIONS is not SHARED_ATTENTION_FUNCTIONS
    assert (
        modeling_gemma2.ALL_ATTENTION_FUNCTIONS["sdpa"]
        is opake_attention.vmap_sdpa_attention_forward_gemma2
    )
    # Keys gemma2 does not override still come from the class-wide mapping.
    assert (
        modeling_gemma2.ALL_ATTENTION_FUNCTIONS["flex_attention"]
        is SHARED_ATTENTION_FUNCTIONS["flex_attention"]
    )


def test_mistral_scopes_compact_sliding_window_sdpa():
    """Mistral binds both halves of the compact no-padding SDPA path locally."""
    apply_runtime_patches()
    apply_mistral_family_patches(eager_attention=True)

    assert modeling_mistral.ALL_ATTENTION_FUNCTIONS is not SHARED_ATTENTION_FUNCTIONS
    assert (
        modeling_mistral.ALL_ATTENTION_FUNCTIONS["sdpa"]
        is opake_attention.vmap_sdpa_attention_forward_sliding_window
    )
    assert (
        modeling_mistral.create_sliding_window_causal_mask
        is vmap_create_compact_sdpa_sliding_window_causal_mask
    )


def test_gemma2_softcap_avoids_the_eager_fallback(
    patched_gemma2_family, softcap_fallback_spy
):
    """Gemma2's SDPA shim keeps softcapping on the chunked path."""
    config = modeling_gemma2.Gemma2Config(**_TINY_CONFIG)
    config._attn_implementation = "sdpa"
    model = modeling_gemma2.Gemma2ForCausalLM(config).eval()

    with torch.no_grad():
        model(input_ids=torch.tensor([[1, 2, 3, 4]]))

    assert softcap_fallback_spy == []
    assert config.attn_logit_softcapping is not None


def test_softcap_family_is_not_rerouted_through_gemma2(
    patched_gemma2_family, softcap_fallback_spy
):
    """A softcap-forwarding family opake never patched keeps HF's SDPA path."""
    modeling_vaultgemma = pytest.importorskip(
        "transformers.models.vaultgemma.modeling_vaultgemma"
    )

    config = modeling_vaultgemma.VaultGemmaConfig(**_TINY_CONFIG)
    # The reason this family is the canary: it forwards a non-None softcap.
    assert config.attn_logit_softcapping is not None
    config._attn_implementation = "sdpa"
    model = modeling_vaultgemma.VaultGemmaForCausalLM(config).eval()

    assert modeling_vaultgemma.ALL_ATTENTION_FUNCTIONS["sdpa"] is sdpa_attention_forward

    with torch.no_grad():
        logits = model(input_ids=torch.tensor([[1, 2, 3, 4]])).logits

    assert softcap_fallback_spy == []
    assert torch.isfinite(logits).all()
