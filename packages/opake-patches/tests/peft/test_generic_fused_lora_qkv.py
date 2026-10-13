import copy
import sys
import types

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    Cohere2Config,
    DynamicCache,
    Gemma2Config,
    GemmaConfig,
    Glm4Config,
    GraniteConfig,
    LlamaConfig,
    MistralConfig,
    Qwen2Config,
)

from opake.api.patches.peft import apply_peft_model_patches
from opake.api.patches.peft.components.qkv import (
    _fused_qkv_attention_forward,
    _resolve_fused_qkv_forward_factory,
)
from opake.api.patches.transformers.components.attention import (
    vmap_sdpa_attention_forward_gemma2,
    vmap_sdpa_attention_forward_sliding_window,
)

RTOL = 1e-4
ATOL = 1e-5


def _tiny_config(config_type):
    kwargs = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 32,
        "attention_dropout": 0.0,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    if config_type in (Cohere2Config, Gemma2Config, Qwen2Config):
        kwargs.update(
            sliding_window=3,
            layer_types=["sliding_attention", "full_attention"],
        )
    if config_type is MistralConfig:
        kwargs["sliding_window"] = 3
    if config_type is Qwen2Config:
        kwargs["use_sliding_window"] = True
    if config_type is Gemma2Config:
        kwargs["attn_logit_softcapping"] = 0.5
    return config_type(**kwargs)


def _attention_case(config_type, layer_index=0):
    model = AutoModelForCausalLM.from_config(_tiny_config(config_type)).eval()
    model.config._attn_implementation = "eager"
    return model, model.model.layers[layer_index].self_attn


def _cohere2_lora_model(device="cpu"):
    model = AutoModelForCausalLM.from_config(_tiny_config(Cohere2Config))
    model = get_peft_model(
        model,
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj"],
        ),
    )
    return model.to(device).eval()


def _reference_projector(attn):
    calls = []

    def project(self, hidden_states):
        calls.append(True)
        return (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )

    attn._opake_fused_qkv = types.MethodType(project, attn)
    return calls


def _embeddings(model, hidden_states):
    positions = torch.arange(1, hidden_states.shape[-2] + 1).unsqueeze(0)
    return model.model.rotary_emb(hidden_states, positions)


@pytest.mark.parametrize(
    ("config_type", "layer_index"),
    [
        pytest.param(LlamaConfig, 0, id="llama"),
        pytest.param(MistralConfig, 0, id="mistral"),
        pytest.param(GemmaConfig, 0, id="gemma"),
        pytest.param(Gemma2Config, 0, id="gemma2-sliding"),
        pytest.param(Gemma2Config, 1, id="gemma2-full"),
        pytest.param(GraniteConfig, 0, id="granite"),
        pytest.param(Cohere2Config, 0, id="cohere2-sliding"),
        pytest.param(Cohere2Config, 1, id="cohere2-full"),
        pytest.param(Qwen2Config, 0, id="qwen2-sliding"),
        pytest.param(Qwen2Config, 1, id="qwen2-full"),
        pytest.param(Glm4Config, 0, id="glm4"),
    ],
)
def test_generic_pipeline_matches_family_forward(config_type, layer_index, monkeypatch):
    torch.manual_seed(7)
    model, attn = _attention_case(config_type, layer_index)
    projection_calls = _reference_projector(attn)
    factory = _resolve_fused_qkv_forward_factory(attn)
    assert factory is not None
    policy = factory.keywords["policy"]

    module = sys.modules[type(attn).__module__]
    attention_calls = []
    rope_calls = []
    original_attention = module.eager_attention_forward
    original_rope = module.apply_rotary_pos_emb

    def record_attention(*args, **kwargs):
        attention_calls.append(kwargs)
        return original_attention(*args, **kwargs)

    def record_rope(*args, **kwargs):
        rope_calls.append(True)
        return original_rope(*args, **kwargs)

    monkeypatch.setattr(module, "eager_attention_forward", record_attention)
    monkeypatch.setattr(module, "apply_rotary_pos_emb", record_rope)

    hidden_states = torch.randn(1, 6, model.config.hidden_size)
    position_embeddings = _embeddings(model, hidden_states)
    with torch.no_grad():
        expected = attn(hidden_states, position_embeddings, attention_mask=None)
        native_rope_calls = len(rope_calls)
        actual = _fused_qkv_attention_forward(
            attn, hidden_states, position_embeddings, policy=policy
        )

    assert projection_calls == [True]
    assert attention_calls[0] == attention_calls[1]
    assert len(rope_calls) == 2 * native_rope_calls
    if config_type is Cohere2Config:
        assert native_rope_calls == (1 if layer_index == 0 else 0)
    if config_type is Qwen2Config:
        assert attn.sliding_window == (3 if layer_index == 0 else None)
    if config_type is Gemma2Config:
        assert attention_calls[1]["softcap"] == 0.5
    torch.testing.assert_close(actual[0], expected[0], rtol=RTOL, atol=ATOL)
    if expected[1] is not None:
        torch.testing.assert_close(actual[1], expected[1], rtol=RTOL, atol=ATOL)


def test_mistral_compact_window_matches_explicit_mask(monkeypatch):
    torch.manual_seed(11)
    model, attn = _attention_case(MistralConfig)
    model.config._attn_implementation = "sdpa"
    _reference_projector(attn)
    factory = _resolve_fused_qkv_forward_factory(attn)
    hidden_states = 3 * torch.randn(1, 7, model.config.hidden_size)
    position_embeddings = _embeddings(model, hidden_states)

    causal = torch.ones(7, 7, dtype=torch.bool).tril()
    window = causal & torch.ones(7, 7, dtype=torch.bool).triu(
        diagonal=1 - model.config.sliding_window
    )
    with torch.no_grad():
        expected = attn(
            hidden_states, position_embeddings, attention_mask=window[None, None]
        )[0]
        full = attn(
            hidden_states, position_embeddings, attention_mask=causal[None, None]
        )[0]
    assert (expected - full).abs().max() > 1e-5

    module = sys.modules[type(attn).__module__]
    scoped_attention = type(module.ALL_ATTENTION_FUNCTIONS)()
    scoped_attention["sdpa"] = vmap_sdpa_attention_forward_sliding_window
    monkeypatch.setattr(module, "ALL_ATTENTION_FUNCTIONS", scoped_attention)
    with torch.no_grad():
        actual = _fused_qkv_attention_forward(
            attn,
            hidden_states,
            position_embeddings,
            policy=factory.keywords["policy"],
        )[0]
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


def test_gemma2_sdpa_preserves_softcap(monkeypatch):
    torch.manual_seed(13)
    model, attn = _attention_case(Gemma2Config, layer_index=1)
    model.config._attn_implementation = "sdpa"
    _reference_projector(attn)
    factory = _resolve_fused_qkv_forward_factory(attn)

    module = sys.modules[type(attn).__module__]
    scoped_attention = type(module.ALL_ATTENTION_FUNCTIONS)()
    scoped_attention["sdpa"] = vmap_sdpa_attention_forward_gemma2
    monkeypatch.setattr(module, "ALL_ATTENTION_FUNCTIONS", scoped_attention)

    hidden_states = 4 * torch.randn(1, 6, model.config.hidden_size)
    position_embeddings = _embeddings(model, hidden_states)
    with torch.no_grad():
        expected = attn(hidden_states, position_embeddings, attention_mask=None)[0]
        actual = _fused_qkv_attention_forward(
            attn,
            hidden_states,
            position_embeddings,
            policy=factory.keywords["policy"],
        )[0]
        attn.attn_logit_softcapping = None
        uncapped = attn(hidden_states, position_embeddings, attention_mask=None)[0]

    assert (expected - uncapped).abs().max() > 1e-5
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


def test_cohere2_nope_adapter_grads_match_upstream():
    torch.manual_seed(23)
    model = _cohere2_lora_model()
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=0.03)

    attn = model.base_model.model.model.layers[1].self_attn
    assert attn.sliding_window is None
    projection_calls = _reference_projector(attn)
    factory = _resolve_fused_qkv_forward_factory(attn)
    hidden_states = torch.randn(1, 6, model.config.hidden_size)
    position_embeddings = model.base_model.model.model.rotary_emb(
        hidden_states, torch.arange(1, 7).unsqueeze(0)
    )
    adapter_params = [
        parameter
        for name, parameter in attn.named_parameters()
        if "lora_A" in name or "lora_B" in name
    ]

    expected = attn(hidden_states, position_embeddings, attention_mask=None)[0]
    expected_grads = torch.autograd.grad(expected.square().mean(), adapter_params)
    actual = _fused_qkv_attention_forward(
        attn,
        hidden_states,
        position_embeddings,
        policy=factory.keywords["policy"],
    )[0]
    actual_grads = torch.autograd.grad(actual.square().mean(), adapter_params)

    assert projection_calls == [True]
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=RTOL, atol=ATOL)


def test_cohere2_nope_cached_decode_matches_upstream():
    torch.manual_seed(29)
    model, attn = _attention_case(Cohere2Config, layer_index=1)
    _reference_projector(attn)
    factory = _resolve_fused_qkv_forward_factory(attn)
    policy = factory.keywords["policy"]
    prompt = torch.randn(1, 5, model.config.hidden_size)
    step = torch.randn(1, 1, model.config.hidden_size)

    def run(forward):
        cache = DynamicCache(config=model.config)
        outputs = []
        for hidden_states, positions, cache_position in (
            (prompt, torch.arange(1, 6), torch.arange(5)),
            (step, torch.tensor([6]), torch.tensor([5])),
        ):
            embeddings = model.model.rotary_emb(hidden_states, positions.unsqueeze(0))
            output, _ = forward(
                hidden_states,
                embeddings,
                attention_mask=None,
                past_key_values=cache,
                cache_position=cache_position,
            )
            outputs.append(output)
        return outputs

    with torch.no_grad():
        expected = run(attn)
        actual = run(
            lambda *args, **kwargs: _fused_qkv_attention_forward(
                attn, *args, policy=policy, **kwargs
            )
        )
    for actual_output, expected_output in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_output, expected_output, rtol=RTOL, atol=ATOL)


def test_cohere2_router_keeps_cpu_fallback():
    torch.manual_seed(19)
    model = _cohere2_lora_model()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 6))
    with torch.no_grad():
        expected = model(input_ids).logits
    apply_peft_model_patches(model)
    for layer in model.base_model.model.model.layers:
        assert layer.self_attn._opake_lora_qkv_patched
    with torch.no_grad():
        actual = model(input_ids).logits
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cohere2_fused_route_matches_unfused_logits_and_adapter_grads(
    monkeypatch,
):
    pytest.importorskip("triton")
    torch.manual_seed(17)
    reference = _cohere2_lora_model("cuda")
    for name, parameter in reference.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=0.03)
    fused = copy.deepcopy(reference)

    input_ids = torch.randint(0, reference.config.vocab_size, (2, 6), device="cuda")
    ref_params = {
        name: parameter
        for name, parameter in reference.named_parameters()
        if "self_attn" in name and ("lora_A" in name or "lora_B" in name)
    }
    expected = reference(input_ids).logits
    expected_grads = torch.autograd.grad(
        expected.float().square().mean(), tuple(ref_params.values())
    )

    apply_peft_model_patches(fused)
    for layer in fused.base_model.model.model.layers:
        assert layer.self_attn._opake_lora_qkv_patched

    from opake.api.patches.kernels.lora import Opake_LoRA_QKV

    original_apply = Opake_LoRA_QKV.apply
    fused_calls = []

    def record_fusion(*args):
        fused_calls.append(True)
        return original_apply(*args)

    monkeypatch.setattr(Opake_LoRA_QKV, "apply", record_fusion)
    actual = fused(input_ids).logits
    fused_params = dict(fused.named_parameters())
    actual_grads = torch.autograd.grad(
        actual.float().square().mean(),
        tuple(fused_params[name] for name in ref_params),
    )

    assert fused_calls
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=1e-3)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-2, atol=2e-3)
