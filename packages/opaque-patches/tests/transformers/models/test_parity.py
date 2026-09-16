# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Patched-vs-upstream numerical parity suite.

For each registered model family, constructs two tiny models with identical
weights — one patched, one unpatched — and compares:

1. Forward logits (eager and SDPA attention implementations)
2. Per-parameter gradients from a standard backward pass
3. vmap(grad) clipped gradients against a runtime-compatible reference

Feature-variant parity tests cover:
- Sliding window attention (Ministral, Mistral)
- Attention logit softcapping (Gemma2, Gemma3)
- KV cache during generation
- PEFT LoRA adapters

Tolerances are dtype-specific:
- fp32 (CPU/MPS): rtol=1e-5, atol=1e-6
- bf16 (CUDA): rtol=1e-2, atol=1e-2
- Softcapping families (fp32): rtol=1e-4, atol=1e-5
"""

from __future__ import annotations

import gc
import importlib
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _test_utils import (
    assert_logits_differ,
    assert_parity_forward,
    assert_parity_grad,
    assert_parity_vmap_grad,
    build_patched_model_pair,
    get_tiny_config_kwargs,
    parity_model_patches,
)

from opaque.api.patches.transformers import supported_families

# ---------------------------------------------------------------------------
# Family → (Config, ForCausalLM, extra config kwargs) mapping
# ---------------------------------------------------------------------------

# Families that have attention logit softcapping (relaxed fp32 tolerances).
_SOFTCAP_FAMILIES = {"gemma2", "gemma3"}

# Families that support sliding window attention.
_SLIDING_WINDOW_FAMILIES = {"ministral", "mistral"}

# Sliding-window probe geometry. The window must be strictly smaller than the
# sequence for the look-back limit to bind — a window wider than the sequence
# reduces to a plain causal mask and tests nothing, which is what
# ``_NON_BINDING_SLIDING_WINDOW`` serves as the control for.
_SLIDING_WINDOW = 4
_SLIDING_WINDOW_SEQ_LEN = 16
_NON_BINDING_SLIDING_WINDOW = 64

# Families that are eager-only (SDPA not supported by HF).
_EAGER_ONLY_FAMILIES = {"gpt_oss", "deepseek_v4"}

# MoE families — need extra config kwargs.
_MOE_FAMILIES = {
    "mellum",
    "mixtral",
    "qwen3_moe",
    "qwen3_5_moe",
    "qwen3_next",
    "hunyuan_v1_moe",
}

# Families that cannot be constructed with tiny configs for parity tests —
# their config/model classes have hard requirements (e.g. heterogeneous
# layer types, gated model downloads) that tiny configs can't satisfy.
_PARITY_SKIP_FAMILIES = {
    "qwen3_5_moe",  # requires heterogeneous layer_types
}

# These families require runtime compatibility shims even for an ordinary
# forward, so no strict-upstream parity reference is available.
_STRICT_FORWARD_PARITY_SKIP_FAMILIES = {
    "deepseek_v4",
    "gpt_oss",
    "qwen3_next",
}

# Known, test-specific upstream compatibility gaps. Strict xfails prevent these
# exemptions from remaining after Transformers or the patches resolve them.
_BACKWARD_GRAD_XFAIL_REASONS = {
    "qwen3_next": (
        "Transformers Qwen3-Next calls Cache.get_seq_length on a cache containing "
        "only linear-attention layers"
    ),
    "deepseek_v4": (
        "the required eager-attention compatibility shim omits the upstream "
        "self-attention sink gradient"
    ),
    "gpt_oss": (
        "the required eager-attention compatibility shim omits the upstream "
        "self-attention sink gradient"
    ),
}

_VMAP_GRAD_XFAIL_REASONS: dict[str, str] = {}

# Families whose Config / ForCausalLM class name does not follow the standard
# pattern (FamilyConfig, FamilyForCausalLM). Map family → explicit module
# attribute names.
_EXCEPTIONS: dict[str, tuple[str, str]] = {
    "gemma3": ("Gemma3TextConfig", "Gemma3ForCausalLM"),
    "gpt2": ("GPT2Config", "GPT2LMHeadModel"),
    "hunyuan_v1_moe": ("HunYuanMoEV1Config", "HunYuanMoEV1ForCausalLM"),
    "qwen3_5_moe": ("Qwen3_5MoeConfig", "Qwen3_5MoeForCausalLM"),
}


def _resolve_family_imports(family: str):
    """Return ``(config_cls, model_cls)`` for a registered family.

    Uses the standard naming convention (FamilyConfig, FamilyForCausalLM)
    and falls back to ``_EXCEPTIONS`` for non-standard names.
    """
    cfg_mod = importlib.import_module(
        f"transformers.models.{family}.configuration_{family}"
    )
    model_mod = importlib.import_module(
        f"transformers.models.{family}.modeling_{family}"
    )
    module_attrs, class_forwards = _PRISTINE_FAMILY_STATE[family]
    for name, value in module_attrs.items():
        setattr(model_mod, name, value)
    for cls, forward in class_forwards.items():
        cls.forward = forward

    if family in _EXCEPTIONS:
        config_cls = getattr(cfg_mod, _EXCEPTIONS[family][0])
        model_cls = getattr(model_mod, _EXCEPTIONS[family][1])
    else:
        config_cls = next(
            getattr(cfg_mod, n)
            for n in dir(cfg_mod)
            if n.endswith("Config") and "PreTrained" not in n
        )
        model_cls = next(
            getattr(model_mod, n) for n in dir(model_mod) if n.endswith("ForCausalLM")
        )
    return config_cls, model_cls


def _extra_config_kwargs(family: str) -> dict:
    """Return extra config kwargs for families that need them."""
    extra = {}
    if family in _MOE_FAMILIES:
        extra.update(
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=64,
        )
    if family in {"gemma", "gemma2", "gemma3", "ministral", "glm4", "hunyuan_v1_moe"}:
        extra["head_dim"] = 16
    return extra


def _base_config_kwargs(family: str) -> dict:
    """Return tiny config kwargs, filtered for family compatibility.

    GPT2 lacks ``num_key_value_heads`` and ``rope_theta``; the base
    ``get_tiny_config_kwargs()`` includes those.
    """
    kwargs = get_tiny_config_kwargs()
    if family == "gpt2":
        kwargs.pop("num_key_value_heads", None)
        kwargs.pop("rope_theta", None)
        # Model preparation disables stochastic dropout for DP transforms. Match
        # that deterministic contract in the upstream parity reference instead of
        # comparing against GPT-2's nonzero training defaults.
        kwargs.update(attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0)
    return kwargs


def _parity_dtype(device: torch.device) -> torch.dtype | None:
    """Exercise CUDA kernels at bf16 precision while retaining CPU/MPS coverage."""
    return torch.bfloat16 if device.type == "cuda" else None


# ===========================================================================
# Core parity tests — parameterized over all registered families
# ===========================================================================


def _get_families():
    """Return list of registered families, excluding those that can't be tested."""
    import contextlib

    # Import all model modules to trigger registration.
    with contextlib.suppress(Exception):
        from opaque.api.patches.transformers.models import (  # noqa: F401
            apply_cohere2_patches,
            apply_cohere_patches,
            apply_deepseek_v4_patches,
            apply_exaone4_patches,
            apply_gemma2_patches,
            apply_gemma3_patches,
            apply_gemma_patches,
            apply_glm4_patches,
            apply_gpt2_patches,
            apply_gpt_oss_patches,
            apply_granite_patches,
            apply_hunyuan_v1_moe_patches,
            apply_llama_patches,
            apply_mellum_patches,
            apply_ministral_patches,
            apply_mistral_patches,
            apply_mixtral_patches,
            apply_olmo2_patches,
            apply_olmo3_patches,
            apply_phi3_patches,
            apply_qwen2_patches,
            apply_qwen3_5_moe_patches,
            apply_qwen3_moe_patches,
            apply_qwen3_next_patches,
            apply_qwen3_patches,
            apply_smollm3_patches,
        )
    return sorted(
        family
        for family in set(supported_families()) - _PARITY_SKIP_FAMILIES
        if importlib.util.find_spec(f"transformers.models.{family}") is not None
    )


FAMILIES = _get_families()


_FAMILY_MODULE_PATCH_NAMES = (
    "create_causal_mask",
    "create_sliding_window_causal_mask",
    "create_recurrent_attention_mask",
    "repeat_kv",
    "eager_attention_forward",
    "apply_rotary_pos_emb",
    "ALL_ATTENTION_FUNCTIONS",
)


def _snapshot_pristine_family_state(families=FAMILIES):
    """Capture upstream globals before any package test applies process-wide patches."""
    state = {}
    for family in families:
        model_mod = importlib.import_module(
            f"transformers.models.{family}.modeling_{family}"
        )
        module_attrs = {
            name: getattr(model_mod, name)
            for name in _FAMILY_MODULE_PATCH_NAMES
            if hasattr(model_mod, name)
        }
        class_forwards = {
            value: value.forward
            for value in vars(model_mod).values()
            if isinstance(value, type)
            and value.__module__ == model_mod.__name__
            and hasattr(value, "forward")
        }
        state[family] = module_attrs, class_forwards
    return state


_PRISTINE_FAMILY_STATE = _snapshot_pristine_family_state()


def _restore_family_state(state):
    for family, (module_attrs, class_forwards) in state.items():
        model_mod = importlib.import_module(
            f"transformers.models.{family}.modeling_{family}"
        )
        for name, value in module_attrs.items():
            setattr(model_mod, name, value)
        for cls, forward in class_forwards.items():
            cls.forward = forward


@pytest.fixture(autouse=True)
def _isolate_parity_family_state(request):
    """Keep pristine parity references from altering later package tests."""
    from opaque.api.patches.transformers._family import _reset_patched_families

    callspec = getattr(request.node, "callspec", None)
    family = callspec.params.get("family") if callspec is not None else None
    previous_state = _snapshot_pristine_family_state((family,)) if family else {}
    try:
        yield
    finally:
        _restore_family_state(previous_state)
        # The restored globals may differ from what the idempotency cache recorded
        # during the parity case. Force later tests to inspect and patch them anew.
        _reset_patched_families()
        if torch.backends.mps.is_available():
            gc.collect()
            torch.mps.empty_cache()


_FORWARD_PARITY_CASES = [
    pytest.param(
        family,
        impl,
        id=f"{impl}-{family}",
        marks=(
            pytest.mark.slow
            if (family, impl) in {("cohere", "eager"), ("hunyuan_v1_moe", "eager")}
            else ()
        ),
    )
    for impl in ("eager", "sdpa")
    for family in FAMILIES
]


def _family_params(*slow_families: str, xfail_reasons: dict[str, str] | None = None):
    params = []
    for family in FAMILIES:
        marks = []
        if family in slow_families:
            marks.append(pytest.mark.slow)
        if xfail_reasons is not None and family in xfail_reasons:
            marks.append(pytest.mark.xfail(strict=True, reason=xfail_reasons[family]))
        params.append(pytest.param(family, marks=marks))
    return params


@pytest.mark.parametrize(("family", "impl"), _FORWARD_PARITY_CASES)
def test_forward_logits_parity(family, impl, device):
    """Patched and unpatched models produce identical forward logits."""
    if family in _STRICT_FORWARD_PARITY_SKIP_FAMILIES:
        pytest.skip(f"{family} requires runtime compatibility shims")
    config_cls, model_cls = _resolve_family_imports(family)
    extra = _extra_config_kwargs(family)
    softcapping = family in _SOFTCAP_FAMILIES
    if impl == "sdpa" and family in _EAGER_ONLY_FAMILIES:
        pytest.skip(f"{family} is eager-only (SDPA not supported by HF)")
    try:
        config_kwargs = _base_config_kwargs(family)
        config_kwargs.update(extra)
        assert_parity_forward(
            model_cls,
            config_cls,
            device,
            attn_impl=impl,
            config_kwargs=config_kwargs,
            softcapping=softcapping,
            label=f"{family} [{impl}]",
            dtype=_parity_dtype(device),
        )
    except Exception as e:
        raise AssertionError(f"{family} [{impl}] forward parity failed") from e


@pytest.mark.parametrize(
    "family",
    _family_params("cohere", xfail_reasons=_BACKWARD_GRAD_XFAIL_REASONS),
)
def test_backward_grads_parity(family, device):
    """Patched and unpatched models produce identical per-parameter gradients."""
    config_cls, model_cls = _resolve_family_imports(family)
    extra = _extra_config_kwargs(family)
    softcapping = family in _SOFTCAP_FAMILIES
    try:
        config_kwargs = _base_config_kwargs(family)
        config_kwargs.update(extra)
        assert_parity_grad(
            model_cls,
            config_cls,
            device,
            config_kwargs=config_kwargs,
            softcapping=softcapping,
            label=f"{family}",
            dtype=_parity_dtype(device),
        )
    except Exception as e:
        raise AssertionError(f"{family} backward grad parity failed") from e


# ===========================================================================
# Feature-variant parity tests
# ===========================================================================


@pytest.mark.parametrize(
    "use_attention_mask", [True, False], ids=["padding_mask", "no_padding_mask"]
)
@pytest.mark.parametrize("family", sorted(_SLIDING_WINDOW_FAMILIES))
def test_sliding_window_parity(family, use_attention_mask, device):
    """Parity with a binding sliding window (window 4, sequence 16)."""
    config_cls, model_cls = _resolve_family_imports(family)
    config_kwargs = _base_config_kwargs(family)
    config_kwargs.update(_extra_config_kwargs(family))
    config_kwargs["sliding_window"] = _SLIDING_WINDOW
    label = f"{family} [sliding_window]"
    try:
        logits, _ = assert_parity_forward(
            model_cls,
            config_cls,
            device,
            config_kwargs=config_kwargs,
            label=label,
            dtype=_parity_dtype(device),
            seq_len=_SLIDING_WINDOW_SEQ_LEN,
            use_attention_mask=use_attention_mask,
        )
    except Exception as e:
        raise AssertionError(f"{family} sliding window parity failed") from e

    # Vacuity guard: a non-binding window (ministral rejects sliding_window=None).
    logits_open_window, _ = assert_parity_forward(
        model_cls,
        config_cls,
        device,
        config_kwargs={**config_kwargs, "sliding_window": _NON_BINDING_SLIDING_WINDOW},
        label=f"{family} [non_binding_sliding_window]",
        dtype=_parity_dtype(device),
        seq_len=_SLIDING_WINDOW_SEQ_LEN,
        use_attention_mask=use_attention_mask,
    )
    assert_logits_differ(logits, logits_open_window, label=label)


@pytest.mark.parametrize("family", sorted(_SOFTCAP_FAMILIES))
def test_softcapping_parity(family, device):
    """Parity for families with attention logit softcapping."""
    config_cls, model_cls = _resolve_family_imports(family)
    extra = _extra_config_kwargs(family)
    extra["attn_logit_softcapping"] = 50.0
    try:
        config_kwargs = _base_config_kwargs(family)
        config_kwargs.update(extra)
        assert_parity_forward(
            model_cls,
            config_cls,
            device,
            config_kwargs=config_kwargs,
            softcapping=True,
            label=f"{family} [softcapping]",
            dtype=_parity_dtype(device),
        )
        assert_parity_grad(
            model_cls,
            config_cls,
            device,
            config_kwargs=config_kwargs,
            softcapping=True,
            label=f"{family} [softcapping]",
            dtype=_parity_dtype(device),
        )
    except Exception as e:
        raise AssertionError(f"{family} softcapping parity failed") from e


@pytest.mark.parametrize(
    "family",
    [pytest.param("llama", marks=pytest.mark.slow), "qwen2"],
)
def test_kv_cache_parity(family, device):
    """Patched and unpatched models generate identical tokens with KV cache."""
    config_cls, model_cls = _resolve_family_imports(family)
    base_kwargs = _base_config_kwargs(family)
    unpatched, patched = build_patched_model_pair(
        config_cls,
        model_cls,
        device,
        config_kwargs=base_kwargs,
    )
    dtype = _parity_dtype(device)
    if dtype is not None:
        unpatched = unpatched.to(dtype)
        patched = patched.to(dtype)
    unpatched.eval()
    patched.eval()

    torch.manual_seed(0)
    vocab = unpatched.config.vocab_size
    input_ids = torch.randint(0, vocab, (1, 5), device=device)

    gen_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": 3,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": unpatched.config.pad_token_id,
    }
    out_ref = unpatched.generate(**gen_kwargs)
    with parity_model_patches(patched, {"eager_attention": True}):
        out_test = patched.generate(**gen_kwargs)

    assert out_ref.shape == out_test.shape, (
        f"{family} KV cache shape mismatch: {out_ref.shape} vs {out_test.shape}"
    )
    assert torch.equal(out_ref, out_test), f"{family} KV cache output mismatch"


@pytest.mark.parametrize("family", ["llama", "qwen2"])
def test_lora_forward_parity(family, device):
    """Patched model with LoRA produces same forward logits as unpatched model with LoRA.

    Gradient parity through LoRA weights is not tested here: the opaque patches
    replace the base model's attention/MLP backends (eager attention, fused MLP,
    etc.) with custom autograd functions. Even when forward outputs are bit-close,
    the backward Jacobian can accumulate different rounding paths, so LoRA
    gradient parity depends on the exact kernel implementation. Forward parity is
    the meaningful check — it verifies the patched forward is semantically
    equivalent for LoRA fine-tuning.
    """
    try:
        import peft
    except ImportError:
        pytest.skip("peft not installed")

    config_cls, model_cls = _resolve_family_imports(family)
    base_kwargs = _base_config_kwargs(family)
    unpatched, patched = build_patched_model_pair(
        config_cls,
        model_cls,
        device,
        config_kwargs=base_kwargs,
    )
    dtype = _parity_dtype(device)
    if dtype is not None:
        unpatched = unpatched.to(dtype)
        patched = patched.to(dtype)

    # Apply LoRA to both models with identical config.
    lora_config = peft.LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
    )
    unpatched = peft.get_peft_model(unpatched, lora_config)
    patched = peft.get_peft_model(patched, lora_config)

    torch.manual_seed(0)
    vocab = unpatched.config.vocab_size
    input_ids = torch.randint(0, vocab, (2, 8), device=device)
    attention_mask = torch.ones_like(input_ids)

    unpatched.eval()
    patched.eval()
    with torch.no_grad():
        logits_ref = unpatched(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits
        from opaque.patches import apply_model_patches

        apply_model_patches(patched, eager_attention=True, lora=True)
        logits_test = patched(input_ids=input_ids, attention_mask=attention_mask).logits

    assert logits_ref.shape == logits_test.shape
    rtol, atol = (1e-2, 1e-2) if dtype is torch.bfloat16 else (1e-4, 1e-5)
    assert torch.allclose(logits_test, logits_ref, rtol=rtol, atol=atol), (
        f"{family} LoRA forward mismatch: max diff "
        f"{(logits_test - logits_ref).abs().max().item():.2e}"
    )


@pytest.mark.parametrize(
    "family",
    _family_params("cohere", xfail_reasons=_VMAP_GRAD_XFAIL_REASONS),
)
def test_vmap_grad_parity(family, device):
    """Patched and runtime-compatible reference vmap(grad) values match."""
    config_cls, model_cls = _resolve_family_imports(family)
    extra = _extra_config_kwargs(family)
    softcapping = family in _SOFTCAP_FAMILIES
    try:
        config_kwargs = _base_config_kwargs(family)
        config_kwargs.update(extra)
        assert_parity_vmap_grad(
            model_cls,
            config_cls,
            device,
            config_kwargs=config_kwargs,
            softcapping=softcapping,
            label=f"{family}",
            dtype=_parity_dtype(device),
            batch=2 if family in _MOE_FAMILIES else 4,
            seq=4 if family in _MOE_FAMILIES else 12,
        )
    except Exception as e:
        raise AssertionError(f"{family} vmap grad parity failed") from e


# ===========================================================================
# Coverage guard
# ===========================================================================


def test_parity_families_covered():
    """All registered families have at least one parity test passing."""
    families = {
        family
        for family in set(supported_families()) - _PARITY_SKIP_FAMILIES
        if importlib.util.find_spec(f"transformers.models.{family}") is not None
    }
    missing = families - set(FAMILIES)
    assert not missing, (
        f"Registered families missing from parity suite: {sorted(missing)}"
    )
