"""Public surface for downstream users to register their own model families.

Validates:

- Built-in kinds work via ``make_apply_model_patches`` (smoke).
- Custom kinds register, then resolve by name.
- Raw callables are accepted directly (no registration needed).
- Unknown string kinds raise a useful error listing the registered options.
- ``family_name`` is exposed at the package level and detects HF families.
"""

from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("transformers")

from opaque.api.patches.transformers._registry import _FAMILY_REGISTRY
from opaque.patches.transformers import (
    family_name,
    make_apply_family_patches,
    make_apply_model_patches,
    register_activation_kind,
    register_family,
    register_fused_add_rms_kind,
    register_rms_norm_kind,
    supported_families,
)


@pytest.fixture
def _restore_registry():
    """Snapshot/restore the family registry around tests that mutate it."""
    snapshot = dict(_FAMILY_REGISTRY)
    yield
    _FAMILY_REGISTRY.clear()
    _FAMILY_REGISTRY.update(snapshot)


def _passthrough_factory(orig):
    """Trivial factory used as a stand-in for a real Triton kernel patch."""
    return orig


# ----------------------------------------------------------------------------
# make_apply_family_patches / make_apply_model_patches
# ----------------------------------------------------------------------------


def test_built_in_kinds_compose_into_a_callable():
    fam = make_apply_family_patches(
        family="public_api_test_a",
        module_path="transformers.models.llama.modeling_llama",
    )
    apply = make_apply_model_patches(
        family="public_api_test_a",
        family_apply=fam,
        module_path="transformers.models.llama.modeling_llama",
        classes={
            "mlp": "LlamaMLP",
            "rms_norm": "LlamaRMSNorm",
            "decoder_layer": "LlamaDecoderLayer",
            "causal_lm": "LlamaForCausalLM",
        },
        activation_kind="swiglu",
        rms_norm_kind="llama",
        fused_add_rms_kind="llama",
    )
    assert callable(apply)
    assert apply.__name__ == "apply_public_api_test_a_patches"
    assert apply._opaque_family == "public_api_test_a"


def test_unknown_kind_raises_with_registered_set_in_message():
    fam = make_apply_family_patches(
        family="public_api_test_unknown",
        module_path="transformers.models.llama.modeling_llama",
    )
    with pytest.raises(ValueError, match="Unknown kind 'nope'"):
        make_apply_model_patches(
            family="public_api_test_unknown",
            family_apply=fam,
            module_path="transformers.models.llama.modeling_llama",
            classes={},
            activation_kind="nope",
        )


# ----------------------------------------------------------------------------
# Custom registration
# ----------------------------------------------------------------------------


def test_register_activation_kind_then_use_by_name():
    register_activation_kind("public_api_glu", _passthrough_factory)
    fam = make_apply_family_patches(
        family="public_api_test_b",
        module_path="transformers.models.llama.modeling_llama",
    )
    apply = make_apply_model_patches(
        family="public_api_test_b",
        family_apply=fam,
        module_path="transformers.models.llama.modeling_llama",
        classes={"mlp": "LlamaMLP"},
        activation_kind="public_api_glu",
    )
    assert callable(apply)


def test_register_rms_norm_kind_then_use_by_name():
    register_rms_norm_kind("public_api_norm", _passthrough_factory)
    fam = make_apply_family_patches(
        family="public_api_test_c",
        module_path="transformers.models.llama.modeling_llama",
    )
    apply = make_apply_model_patches(
        family="public_api_test_c",
        family_apply=fam,
        module_path="transformers.models.llama.modeling_llama",
        classes={"rms_norm": "LlamaRMSNorm"},
        rms_norm_kind="public_api_norm",
    )
    assert callable(apply)


def test_register_fused_add_rms_kind_then_use_by_name():
    register_fused_add_rms_kind("public_api_fused", _passthrough_factory)
    fam = make_apply_family_patches(
        family="public_api_test_d",
        module_path="transformers.models.llama.modeling_llama",
    )
    apply = make_apply_model_patches(
        family="public_api_test_d",
        family_apply=fam,
        module_path="transformers.models.llama.modeling_llama",
        classes={"decoder_layer": "LlamaDecoderLayer"},
        fused_add_rms_kind="public_api_fused",
    )
    assert callable(apply)


def test_raw_callable_accepted_without_registration():
    """Power-user shortcut: pass a factory callable directly, skip the registry."""
    fam = make_apply_family_patches(
        family="public_api_test_e",
        module_path="transformers.models.llama.modeling_llama",
    )
    apply = make_apply_model_patches(
        family="public_api_test_e",
        family_apply=fam,
        module_path="transformers.models.llama.modeling_llama",
        classes={"mlp": "LlamaMLP", "rms_norm": "LlamaRMSNorm"},
        activation_kind=_passthrough_factory,  # not registered, raw callable
        rms_norm_kind=_passthrough_factory,  # raw, also unregistered
    )
    assert callable(apply)


def test_activation_kwarg_gates_family_selected_activation_patch(monkeypatch):
    """The public fine-grained knob is ``activation``; the family-selected
    ``activation_kind`` decides which concrete kernel factory is installed."""
    module_name = "public_api_fake_activation_module"
    mod = types.ModuleType(module_name)

    class FakeMLP:
        def forward(self, x):
            return x

    mod.FakeMLP = FakeMLP
    monkeypatch.setitem(sys.modules, module_name, mod)

    calls = []

    def activation_factory(original):
        calls.append(original)

        def forward(self, x):
            return original(self, x)

        return forward

    fam = make_apply_family_patches(
        family="public_api_test_activation_gate",
        module_path=module_name,
    )
    apply = make_apply_model_patches(
        family="public_api_test_activation_gate",
        family_apply=fam,
        module_path=module_name,
        classes={"mlp": "FakeMLP"},
        activation_kind=activation_factory,
    )

    apply(performance=True, compat=True, activation=False)
    assert calls == []
    assert not hasattr(FakeMLP.forward, "__opaque_patched__")

    apply(performance=True, compat=True, activation=True)
    assert len(calls) == 1
    assert getattr(FakeMLP.forward, "__opaque_patched__", False)


def test_cross_entropy_sets_loss_function_on_model_instance(monkeypatch):
    import torch

    from opaque.api.patches.transformers.components.cross_entropy import (
        _opaque_causal_lm_loss,
    )

    module_name = "public_api_fake_loss_function_module"
    mod = types.ModuleType(module_name)

    def original_loss(*args, **kwargs):
        return None

    class FakeForCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.loss_function = original_loss

        def forward(self, *args, **kwargs):
            return None

    mod.FakeForCausalLM = FakeForCausalLM
    monkeypatch.setitem(sys.modules, module_name, mod)

    fam = make_apply_family_patches(
        family="public_api_test_loss_function_patch",
        module_path=module_name,
    )
    apply = make_apply_model_patches(
        family="public_api_test_loss_function_patch",
        family_apply=fam,
        module_path=module_name,
        classes={"causal_lm": "FakeForCausalLM"},
    )

    patched = FakeForCausalLM()
    untouched = FakeForCausalLM()

    apply(patched, performance=True, compat=False, cross_entropy=False)
    assert patched.loss_function is original_loss

    apply(patched, performance=True, compat=False, cross_entropy=True)
    assert patched.loss_function is _opaque_causal_lm_loss
    assert untouched.loss_function is original_loss


def test_fused_linear_cross_entropy_follows_performance(monkeypatch):
    """The conditional forward follows ``performance`` and supports overrides."""
    import torch

    from opaque.api.patches.transformers.components.cross_entropy import (
        _opaque_causal_lm_loss,
    )

    # The non-fused CE loss_function kernel is Triton-only; its default install
    # gates on the kernel runtime. Simulate it so the opt-in semantics are tested
    # device-independently.
    monkeypatch.setattr(
        "opaque.api.engine.device.fused_kernels_available", lambda: True
    )

    def _fresh_module(
        suffix,
        *,
        fused_linear_cross_entropy=True,
        chunked_linear_cross_entropy=False,
    ):
        module_name = f"public_api_fake_fused_ce_module_{suffix}"
        mod = types.ModuleType(module_name)

        def original_loss(*args, **kwargs):
            return None

        class FakeForCausalLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.loss_function = original_loss

            def forward(self, *args, **kwargs):
                return None

        mod.FakeForCausalLM = FakeForCausalLM
        monkeypatch.setitem(sys.modules, module_name, mod)

        fam = make_apply_family_patches(
            family=f"public_api_test_fused_linear_ce_{suffix}",
            module_path=module_name,
        )
        apply = make_apply_model_patches(
            family=f"public_api_test_fused_linear_ce_{suffix}",
            family_apply=fam,
            module_path=module_name,
            classes={"causal_lm": "FakeForCausalLM"},
            fused_linear_cross_entropy=fused_linear_cross_entropy,
            chunked_linear_cross_entropy=chunked_linear_cross_entropy,
        )
        return mod, FakeForCausalLM, original_loss, apply

    # The wrapper follows the normal performance bucket.
    _, FakeA, _original_loss, apply_a = _fresh_module("a")
    instance_a = FakeA()
    apply_a(instance_a, performance=True, compat=False)
    assert instance_a.loss_function is _opaque_causal_lm_loss
    assert getattr(FakeA.forward, "__opaque_patched__", False)

    # Turning off performance keeps both CE patches off by default.
    _, FakeB, original_loss_b, apply_b = _fresh_module("b")
    instance_b = FakeB()
    apply_b(instance_b, performance=False, compat=False)
    assert instance_b.loss_function is original_loss_b
    assert not hasattr(FakeB.forward, "__opaque_patched__")

    # Per-concern overrides remain available in both directions.
    _, FakeC, original_loss_c, apply_c = _fresh_module("c")
    instance_c = FakeC()
    apply_c(
        instance_c,
        performance=True,
        compat=False,
        cross_entropy=False,
        fused_linear_cross_entropy=False,
    )
    assert instance_c.loss_function is original_loss_c
    assert not hasattr(FakeC.forward, "__opaque_patched__")

    _, FakeOverride, _, apply_override = _fresh_module("override")
    instance_override = FakeOverride()
    apply_override(
        instance_override,
        performance=False,
        compat=False,
        fused_linear_cross_entropy=True,
    )
    assert getattr(FakeOverride.forward, "__opaque_patched__", False)

    # Families outside the measured backward-error envelope keep the original
    # forward even when a caller requests the fused loss.
    _, FakeD, _, apply_d = _fresh_module("d", fused_linear_cross_entropy=False)
    instance_d = FakeD()
    apply_d(
        instance_d,
        performance=True,
        compat=False,
        fused_linear_cross_entropy=True,
    )
    assert not hasattr(FakeD.forward, "__opaque_patched__")

    # A family can reject the CUDA kernel while retaining the portable chunked
    # loss-only forward.
    chunk_widths = []

    def record_chunk_width(original, *, force_chunked=False):
        chunk_widths.append(force_chunked)
        return original

    monkeypatch.setattr(
        "opaque.api.patches.transformers._factory._make_fused_ce_causal_lm_forward",
        record_chunk_width,
    )
    patched = []
    monkeypatch.setattr(
        "opaque.api.patches.transformers._factory._patch_forward",
        lambda *args, **kwargs: patched.append((args, kwargs)),
    )
    _, FakeE, _, apply_e = _fresh_module(
        "e",
        fused_linear_cross_entropy=False,
        chunked_linear_cross_entropy=True,
    )
    instance_e = FakeE()
    apply_e(
        instance_e,
        performance=True,
        compat=False,
        fused_linear_cross_entropy=True,
        chunked_linear_cross_entropy=1024,
    )
    assert len(patched) == 1
    patched[0][0][1](lambda *args, **kwargs: None)
    assert chunk_widths == [1024]

    # ``True`` preserves an integer family default rather than falling back to
    # the global chunk width.
    _, FakeFamilyDefault, _, apply_family_default = _fresh_module(
        "family_default_chunk",
        fused_linear_cross_entropy=False,
        chunked_linear_cross_entropy=2048,
    )
    apply_family_default(
        FakeFamilyDefault(),
        performance=True,
        compat=False,
        fused_linear_cross_entropy=True,
        chunked_linear_cross_entropy=True,
    )
    assert len(patched) == 2
    patched[1][0][1](lambda *args, **kwargs: None)
    assert chunk_widths == [1024, 2048]

    # An explicit runtime setting can enable the portable path for a family
    # whose default is disabled.
    _, FakeOptIn, _, apply_opt_in = _fresh_module(
        "opt_in_chunk",
        fused_linear_cross_entropy=False,
        chunked_linear_cross_entropy=False,
    )
    apply_opt_in(
        FakeOptIn(),
        performance=True,
        compat=False,
        fused_linear_cross_entropy=True,
        chunked_linear_cross_entropy=512,
    )
    assert len(patched) == 3
    patched[2][0][1](lambda *args, **kwargs: None)
    assert chunk_widths == [1024, 2048, 512]

    from opaque.exceptions import ConfigurationError

    _, FakeZero, _, apply_zero = _fresh_module(
        "zero_chunk",
        fused_linear_cross_entropy=False,
        chunked_linear_cross_entropy=0,
    )
    apply_zero(
        FakeZero(),
        performance=True,
        compat=False,
        fused_linear_cross_entropy=True,
    )
    assert len(patched) == 3

    with pytest.raises(ConfigurationError, match="non-negative integer"):
        _fresh_module(
            "invalid_chunk",
            fused_linear_cross_entropy=False,
            chunked_linear_cross_entropy=-1,
        )


# ----------------------------------------------------------------------------
# family_name detection
# ----------------------------------------------------------------------------


def test_family_name_for_known_hf_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
    )
    assert family_name(model) == "qwen2"


def test_family_name_returns_none_for_non_hf_object():
    import torch

    assert family_name(torch.nn.Linear(2, 2)) is None


# ----------------------------------------------------------------------------
# make_apply_family_patches accepts overrides for the runtime replacements
# ----------------------------------------------------------------------------


def test_family_factory_default_replacements_are_opaque_vmap_safe():
    """Default closures bind opaque's vmap-safe implementations."""
    from opaque.api.patches.transformers._family import (
        _opaque_apply_rotary_pos_emb,
        apply_module_masking_patch,
        vmap_eager_attention_forward,
        vmap_repeat_kv,
        vmap_sdpa_attention_forward,
    )

    apply = make_apply_family_patches(
        family="public_api_test_family_defaults",
        module_path="some.nonexistent.module",
    )
    # Sanity check that the closure references the expected defaults
    closure = {c.cell_contents for c in (apply.__closure__ or ())}
    assert vmap_repeat_kv in closure
    assert vmap_eager_attention_forward in closure
    assert vmap_sdpa_attention_forward in closure
    assert _opaque_apply_rotary_pos_emb in closure
    assert apply_module_masking_patch in closure


def test_family_factory_accepts_custom_replacement_callables():
    """Power-user override: a custom architecture provides its own
    vmap-safe ``repeat_kv`` replacement."""
    sentinel_calls: list[tuple] = []

    def my_repeat_kv(*args, **kw):
        sentinel_calls.append((args, kw))
        return None

    apply = make_apply_family_patches(
        family="public_api_test_family_override",
        module_path="some.nonexistent.module",
        repeat_kv_replacement=my_repeat_kv,
    )
    assert callable(apply)
    closure = {c.cell_contents for c in (apply.__closure__ or ())}
    assert my_repeat_kv in closure


def test_family_factory_accepts_none_to_skip_concern():
    """Pass ``None`` per slot to skip that concern entirely (e.g. a
    non-RoPE architecture)."""
    apply = make_apply_family_patches(
        family="public_api_test_family_skip_rope",
        module_path="some.nonexistent.module",
        rope_replacement=None,
        masking_module_patcher=None,
    )
    closure = {c.cell_contents for c in (apply.__closure__ or ())}
    assert None in closure  # one of the skip-this slots


def test_family_factory_tracks_idempotency_per_enabled_concern(monkeypatch):
    from opaque.api.patches.transformers._family import _reset_patched_families

    _reset_patched_families()
    # rope only installs where the Triton runtime exists; simulate it so this
    # concern-tracking test is device-independent.
    monkeypatch.setattr(
        "opaque.api.engine.device.fused_kernels_available", lambda: True
    )
    module_name = "public_api_fake_family_module"
    mod = types.ModuleType(module_name)

    def original_repeat_kv(x):
        return x

    def original_eager_attention_forward(x):
        return x

    def original_rope(x):
        return x

    def replacement_repeat_kv(x):
        return x

    def replacement_eager_attention_forward(x):
        return x

    def replacement_rope(x):
        return x

    mod.repeat_kv = original_repeat_kv
    mod.eager_attention_forward = original_eager_attention_forward
    mod.apply_rotary_pos_emb = original_rope
    monkeypatch.setitem(sys.modules, module_name, mod)

    apply = make_apply_family_patches(
        family="public_api_test_family_concern_tracking",
        module_path=module_name,
        repeat_kv_replacement=replacement_repeat_kv,
        eager_attention_replacement=replacement_eager_attention_forward,
        rope_replacement=replacement_rope,
        masking_module_patcher=None,
    )

    apply(performance=False, compat=False)
    assert mod.repeat_kv is original_repeat_kv
    assert mod.eager_attention_forward is original_eager_attention_forward
    assert mod.apply_rotary_pos_emb is original_rope

    apply(performance=False, compat=True)
    assert mod.repeat_kv is replacement_repeat_kv
    assert mod.eager_attention_forward is replacement_eager_attention_forward
    assert mod.apply_rotary_pos_emb is original_rope

    apply(performance=True, compat=False)
    assert mod.apply_rotary_pos_emb is replacement_rope


# ----------------------------------------------------------------------------
# register_family — user-defined families dispatch through the router
# ----------------------------------------------------------------------------


@pytest.mark.usefixtures("_restore_registry")
def test_register_family_makes_it_visible_in_supported_families():
    name = "public_api_test_router_visibility"
    fam = make_apply_family_patches(family=name, module_path="some.fake.module")
    apply = make_apply_model_patches(
        family=name,
        family_apply=fam,
        module_path="some.fake.module",
        classes={},
        activation_kind="swiglu",
    )
    register_family(name, apply)
    assert name in supported_families()


@pytest.mark.usefixtures("_restore_registry")
def test_register_family_routes_via_apply_transformers_model_patches():
    """``apply_transformers_model_patches`` (the dispatcher used by
    :func:`opaque.patches.apply_model_patches`) consults the registry,
    so user-registered families are reached without any source-code
    changes inside opaque-patches."""
    from opaque.api.patches.transformers._router import apply_transformers_model_patches

    name = "public_api_test_router_dispatch"
    calls: list[dict] = []

    def my_apply_fn(
        model: object | None = None,
        *,
        performance: bool = True,
        compat: bool = True,
        kernels: bool | None = None,
        **kwargs: object,
    ) -> None:
        calls.append(
            {
                "model": model,
                "performance": performance,
                "kwargs": {"kernels": kernels, **kwargs},
            }
        )

    register_family(name, my_apply_fn)

    # Build a stub model that detect_family() will resolve to ``name``
    # via its config.model_type.
    class _Cfg:
        model_type = name

    class _Model:
        config = _Cfg()

    m = _Model()
    apply_transformers_model_patches(m, performance=False, compat=True, foo="bar")
    assert len(calls) == 1
    assert calls[0]["model"] is m
    assert calls[0]["performance"] is False
    assert calls[0]["kwargs"]["foo"] == "bar"


def test_non_hf_model_without_registered_family_warns_and_skips(caplog):
    """Custom non-HF modules remain supported without model-level patches."""
    from opaque.api.patches.transformers._router import apply_transformers_model_patches

    class _Model:
        pass

    apply_transformers_model_patches(_Model())

    assert "no registered apply function" in caplog.text
    assert any(record.levelname == "WARNING" for record in caplog.records)


def test_unregistered_hf_family_raises_when_compat_is_enabled():
    """HF models cannot silently skip the default vmap-safety patches."""
    from transformers import PretrainedConfig, PreTrainedModel

    from opaque.api.patches.transformers._router import apply_transformers_model_patches
    from opaque.exceptions import ConfigurationError

    class _Config(PretrainedConfig):
        model_type = "definitely-not-registered-xyz"

    class _Model(PreTrainedModel):
        config_class = _Config

    with pytest.raises(ConfigurationError, match="require a registered"):
        apply_transformers_model_patches(_Model(_Config()))


@pytest.mark.parametrize("mixed", [False, True], ids=["peft", "mixed-peft"])
def test_peft_wrapped_unregistered_hf_family_raises(mixed):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import PretrainedConfig, PreTrainedModel

    from opaque.api.patches.transformers._router import apply_transformers_model_patches
    from opaque.exceptions import ConfigurationError

    class _Config(PretrainedConfig):
        model_type = "definitely-not-registered-peft-xyz"

    class _Model(PreTrainedModel):
        config_class = _Config

        def __init__(self):
            super().__init__(_Config())
            self.linear = torch.nn.Linear(2, 2)

    model = get_peft_model(
        _Model(),
        LoraConfig(target_modules=["linear"]),
        mixed=mixed,
    )

    with pytest.raises(ConfigurationError, match="require a registered"):
        apply_transformers_model_patches(model)


def test_peft_wrapped_custom_model_warns_and_skips(caplog):
    import torch
    from peft import LoraConfig, get_peft_model

    from opaque.api.patches.transformers._router import apply_transformers_model_patches

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

        def forward(self, inputs):
            return self.linear(inputs)

    model = get_peft_model(_Model(), LoraConfig(target_modules=["linear"]))
    apply_transformers_model_patches(model)

    assert "no registered apply function" in caplog.text
