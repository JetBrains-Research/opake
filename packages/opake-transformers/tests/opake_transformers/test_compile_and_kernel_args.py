"""Trainer compile, kernel, and precision wiring."""

from __future__ import annotations

import logging

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch._dynamo.testing import CompileCounterWithBackend
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
)

from opake.api.transformers.trainer._distributed import DDPState
from opake.api.transformers.trainer._dp_trainer import _compile_strict_chunk
from opake.exceptions import ConfigurationError
from opake.transformers.trainer import DPTrainer, TrainingArguments

# ----------------------------------------------------------------------------
# Tiny shared trainer helper
# ----------------------------------------------------------------------------


def _args(tmp_path, **overrides) -> TrainingArguments:
    defaults = {
        "output_dir": str(tmp_path),
        "per_device_train_batch_size": 1,
        "max_steps": 1,
        "num_train_epochs": 1,
        "save_strategy": "no",
        "use_cpu": True,
        "privacy_target_epsilon": 10.0,
        "privacy_noise_multiplier": 1.0,
    }
    defaults.update(overrides)
    return TrainingArguments(**defaults)


def _tiny_trainer(tmp_path, **arg_overrides) -> tuple[DPTrainer, nn.Module]:
    model = nn.Linear(4, 2)
    args = _args(tmp_path, **arg_overrides)
    trainer = DPTrainer(
        model=model,
        args=args,
        train_dataset=[{"x": torch.zeros(4)}],
        eval_dataset=None,
    )
    return trainer, model


class _TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(16, 4)
        self.head = nn.Linear(4, 16)

    def forward(self, input_ids, **_):
        logits = self.head(self.embed(input_ids))
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), input_ids.reshape(-1)
        )
        return {"loss": loss, "logits": logits}


class _UnregisteredConfig(PretrainedConfig):
    model_type = "definitely-not-registered-xyz"


class _UnregisteredHFModel(PreTrainedModel):
    config_class = _UnregisteredConfig

    def __init__(self) -> None:
        super().__init__(_UnregisteredConfig())
        self.linear = nn.Linear(4, 2)


def test_default_trainer_rejects_unregistered_hf_family(tmp_path):
    """The default compatibility path must not skip an unknown HF family."""
    with pytest.raises(ConfigurationError, match="require a registered"):
        DPTrainer(
            model=_UnregisteredHFModel(),
            args=_args(tmp_path),
            train_dataset=[{"x": torch.zeros(4)}],
            eval_dataset=None,
        )


def test_default_trainer_rejects_unregistered_hf_family_with_lora(tmp_path):
    model = get_peft_model(
        _UnregisteredHFModel(),
        LoraConfig(target_modules=["linear"]),
    )

    with pytest.raises(ConfigurationError, match="require a registered"):
        DPTrainer(
            model=model,
            args=_args(tmp_path),
            train_dataset=[{"x": torch.zeros(4)}],
            eval_dataset=None,
        )


# ----------------------------------------------------------------------------
# torch_compile flag wiring
# ----------------------------------------------------------------------------


def test_torch_compile_default_false_does_not_compile(tmp_path):
    """When torch_compile is unset, the trainer must not pull torch.compile
    onto the loss closure (zero-overhead default)."""
    trainer, _ = _tiny_trainer(tmp_path)
    assert trainer.args.torch_compile is False


def test_torch_compile_true_accepted(tmp_path):
    """torch_compile=True initializes without error; backend/mode default
    to inductor/default at the closure-building site."""
    trainer, _ = _tiny_trainer(tmp_path, torch_compile=True)
    assert trainer.args.torch_compile is True


@pytest.mark.slow
def test_torch_compile_runs_poisson_training_strictly(tmp_path, monkeypatch, caplog):
    generator = torch.Generator().manual_seed(0)
    dataset = [
        {"input_ids": torch.randint(0, 16, (4,), generator=generator)}
        for _ in range(32)
    ]

    def collate(batch):
        return {"input_ids": torch.stack([example["input_ids"] for example in batch])}

    args = _args(
        tmp_path,
        per_device_train_batch_size=3,
        max_steps=3,
        torch_compile=True,
        torch_compile_backend="aot_eager",
        report_to=[],
        logging_strategy="no",
        disable_tqdm=True,
    )
    trainer = DPTrainer(
        model=_TinyLM(),
        args=args,
        train_dataset=dataset,
        data_collator=collate,
    )
    torch._dynamo.reset()
    counter = CompileCounterWithBackend("aot_eager")
    original_compile = torch.compile

    def compile_with_counter(fn, *, backend, mode, fullgraph):
        assert backend == "aot_eager"
        return original_compile(fn, backend=counter, mode=mode, fullgraph=fullgraph)

    monkeypatch.setattr(torch, "compile", compile_with_counter)
    with caplog.at_level(logging.WARNING):
        result = trainer.train()

    assert result.global_step == 3
    assert counter.frame_count > 0
    assert not any(
        "Strict DP gradient compilation failed" in record.message
        for record in caplog.records
    )


class _ClipStateTrainer(DPTrainer):
    def _inner_training_loop(self, ctx, **kwargs):
        result = super()._inner_training_loop(ctx, **kwargs)
        self.final_clip_state = ctx.clip_state
        return result


def _train_llama(tmp_path, *, mode, compiled, fused_ce=True):
    torch.manual_seed(1060)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        pad_token_id=0,
    )
    model = LlamaForCausalLM(config)
    generator = torch.Generator().manual_seed(1060)
    dataset = []
    for _ in range(32):
        ids = torch.randint(1, 32, (8,), generator=generator)
        dataset.append(
            {
                "input_ids": ids,
                "labels": ids.clone(),
                "attention_mask": torch.ones_like(ids),
            }
        )
    args = _args(
        tmp_path,
        per_device_train_batch_size=3,
        max_steps=2,
        privacy_target_epsilon=None,
        torch_compile=compiled,
        torch_compile_backend="aot_eager",
        clipping_mode=mode,
        clipping_kwargs={"gamma": 0.01} if mode == "auto" else {},
        performance_kernels_config=(
            None if fused_ce else {"fused_linear_cross_entropy": False}
        ),
        report_to=[],
        logging_strategy="no",
        disable_tqdm=True,
    )
    trainer = _ClipStateTrainer(model=model, args=args, train_dataset=dataset)
    assert trainer._fused_forward_uses_marker == fused_ce
    result = trainer.train()
    assert result.global_step == 2
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    return result.training_loss, weights, trainer.final_clip_state


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["fixed", "adaptive", "auto"])
def test_torch_compile_trains_fused_llama(tmp_path, caplog, mode):
    torch._dynamo.reset()
    eager_loss, eager_weights, eager_state = _train_llama(
        tmp_path / "eager", mode=mode, compiled=False
    )
    with caplog.at_level(logging.WARNING):
        compiled_loss, compiled_weights, compiled_state = _train_llama(
            tmp_path / "compiled", mode=mode, compiled=True
        )

    fallback_warnings = [
        record
        for record in caplog.records
        if "Strict DP gradient compilation failed" in record.message
    ]
    assert len(fallback_warnings) <= 1
    assert compiled_loss == pytest.approx(eager_loss, rel=1e-5, abs=1e-6)
    assert compiled_state == eager_state
    assert compiled_weights.keys() == eager_weights.keys()
    for name in eager_weights:
        torch.testing.assert_close(
            compiled_weights[name], eager_weights[name], rtol=1e-5, atol=1e-6
        )


@pytest.mark.slow
def test_torch_compile_auto_clipping_uses_strict_graph_without_fused_ce(
    tmp_path, monkeypatch, caplog
):
    torch._dynamo.reset()
    counter = CompileCounterWithBackend("aot_eager")
    original_compile = torch.compile

    def compile_with_counter(fn, *, backend, mode, fullgraph):
        assert backend == "aot_eager"
        assert fullgraph
        return original_compile(fn, backend=counter, mode=mode, fullgraph=fullgraph)

    monkeypatch.setattr(torch, "compile", compile_with_counter)
    with caplog.at_level(logging.WARNING):
        _train_llama(tmp_path, mode="auto", compiled=True, fused_ce=False)

    assert counter.frame_count > 0
    assert not any(
        "Strict DP gradient compilation failed" in record.message
        for record in caplog.records
    )


def test_torch_compile_with_backend_and_mode(tmp_path):
    trainer, _ = _tiny_trainer(
        tmp_path,
        torch_compile=True,
        torch_compile_backend="aot_eager",
        torch_compile_mode="reduce-overhead",
    )
    assert trainer.args.torch_compile_backend == "aot_eager"
    assert trainer.args.torch_compile_mode == "reduce-overhead"


def test_torch_compile_invalid_mode_rejected_at_args(tmp_path):
    with pytest.raises(ValueError, match="torch_compile_mode"):
        _args(tmp_path, torch_compile_mode="nonsense")


def test_torch_compile_with_auto_find_microbatch_size_rejected(tmp_path):
    with pytest.raises(
        ConfigurationError, match=r"torch_compile.*auto_find_microbatch_size"
    ):
        _args(tmp_path, torch_compile=True, auto_find_microbatch_size=True)


def test_torch_compile_with_explicit_no_autofind_accepted(tmp_path):
    trainer, _ = _tiny_trainer(
        tmp_path,
        torch_compile=True,
        auto_find_microbatch_size=False,
    )
    assert trainer.args.torch_compile is True
    assert trainer.args.auto_find_microbatch_size is False


def test_torch_compile_with_gradient_checkpointing_rejected(tmp_path):
    with pytest.raises(
        ConfigurationError, match=r"torch_compile.*gradient_checkpointing"
    ):
        _args(tmp_path, torch_compile=True, gradient_checkpointing=True)


def test_torch_compile_rejects_model_with_checkpointing_already_enabled(tmp_path):
    model = _TinyLM()
    model.is_gradient_checkpointing = True

    with pytest.raises(ConfigurationError, match="already has gradient checkpointing"):
        DPTrainer(
            model=model,
            args=_args(tmp_path, torch_compile=True),
            train_dataset=[{"input_ids": torch.zeros(4, dtype=torch.long)}],
        )


# ----------------------------------------------------------------------------
# strict chunk compilation
# ----------------------------------------------------------------------------


def test_strict_chunk_compiler_requests_fullgraph_with_automatic_shapes(monkeypatch):
    compile_calls = []

    def fake_compile(fn, *, backend, mode, fullgraph):
        compile_calls.append((fn, backend, mode, fullgraph))
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)

    def chunk(params, x, y):
        return params + x.sum() + y.sum()

    compiled = _compile_strict_chunk(
        chunk, backend="aot_eager", mode="default", on_fallback=lambda _: None
    )
    torch.testing.assert_close(
        compiled(torch.tensor(1.0), torch.ones(3), torch.ones(3)),
        torch.tensor(7.0),
    )
    assert compile_calls == [(chunk, "aot_eager", "default", True)]


def test_strict_chunk_compiler_handles_batch_sizes():
    torch._dynamo.reset()
    backend = CompileCounterWithBackend("aot_eager")
    fallbacks = []

    def chunk(x):
        return x.sin().sum(dim=0)

    compiled = _compile_strict_chunk(
        chunk, backend=backend, mode="default", on_fallback=fallbacks.append
    )
    generator = torch.Generator().manual_seed(0)
    for batch_size in (4, 2, 3, 1):
        x = torch.randn(batch_size, 5, generator=generator)
        torch.testing.assert_close(compiled(x), chunk(x))

    assert not fallbacks
    assert backend.frame_count > 0


@pytest.mark.parametrize("failure_at", [1, 3])
def test_strict_chunk_compiler_falls_back_once(monkeypatch, failure_at):
    failure = torch._dynamo.exc.Unsupported("graph break")
    compiled_calls = 0
    eager_calls = 0
    fallbacks = []

    def fake_compile(fn, *, backend, mode, fullgraph):
        def compiled(x):
            nonlocal compiled_calls
            compiled_calls += 1
            if compiled_calls == failure_at:
                raise failure
            return x + 10

        return compiled

    def chunk(x):
        nonlocal eager_calls
        eager_calls += 1
        return x + 1

    monkeypatch.setattr(torch, "compile", fake_compile)
    compiled = _compile_strict_chunk(
        chunk, backend="aot_eager", mode="default", on_fallback=fallbacks.append
    )
    for value in range(1, failure_at):
        assert compiled(value) == value + 10
    assert compiled(failure_at) == failure_at + 1
    assert compiled(failure_at + 1) == failure_at + 2
    assert compiled_calls == failure_at
    assert eager_calls == 2
    assert fallbacks == [failure]


@pytest.mark.parametrize("failure_type", [RuntimeError, torch.OutOfMemoryError])
def test_strict_chunk_compiler_does_not_retry_other_failures(monkeypatch, failure_type):
    fallbacks = []
    eager_calls = 0

    def fake_compile(fn, *, backend, mode, fullgraph):
        def compiled(x):
            raise failure_type("not a Dynamo failure")

        return compiled

    def chunk(x):
        nonlocal eager_calls
        eager_calls += 1
        return x

    monkeypatch.setattr(torch, "compile", fake_compile)
    compiled = _compile_strict_chunk(
        chunk, backend="aot_eager", mode="default", on_fallback=fallbacks.append
    )
    with pytest.raises(failure_type, match="not a Dynamo failure"):
        compiled(torch.ones(2))
    assert eager_calls == 0
    assert not fallbacks


def test_grad_compiler_warns_once_across_chunks(tmp_path, monkeypatch, caplog):
    failure = torch._dynamo.exc.Unsupported("graph break")

    def fake_compile(fn, *, backend, mode, fullgraph):
        def compiled(x):
            raise failure

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    trainer, _ = _tiny_trainer(
        tmp_path, torch_compile=True, torch_compile_backend="aot_eager"
    )
    compiler = trainer._grad_compiler()
    assert compiler is not None
    with caplog.at_level(logging.WARNING):
        assert compiler(lambda x: x + 1)(1) == 2
        assert compiler(lambda x: x + 2)(1) == 3

    warnings = [
        record
        for record in caplog.records
        if "Strict DP gradient compilation failed" in record.message
    ]
    assert len(warnings) == 1


def _make_fake_distributed(trainer):
    trainer._ddp = DDPState(
        is_distributed=True,
        rank=0,
        local_rank=0,
        world_size=2,
        backend="gloo",
        device=torch.device("cpu"),
    )


def test_sibling_compile_failure_raises_before_gradient_collective(
    tmp_path, monkeypatch
):
    trainer, _ = _tiny_trainer(tmp_path)
    _make_fake_distributed(trainer)

    def sibling_failed(flags, *, op):
        flags[1] = 1.0

    monkeypatch.setattr(torch.distributed, "all_reduce", sibling_failed)

    with pytest.raises(RuntimeError, match="failed on a sibling rank"):
        trainer._synchronize_grad_failure(None)


def test_local_compile_failure_is_synchronized_then_reraised(tmp_path, monkeypatch):
    trainer, _ = _tiny_trainer(tmp_path)
    _make_fake_distributed(trainer)
    calls = []
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda flags, *, op: calls.append(flags.clone()),
    )
    failure = torch._dynamo.exc.Unsupported("strict graph failed")

    with pytest.raises(torch._dynamo.exc.Unsupported, match="strict graph failed"):
        trainer._synchronize_grad_failure(failure)

    assert calls[0].tolist() == [0.0, 1.0]


# ----------------------------------------------------------------------------
# use_performance_kernels — wiring through to apply_model_patches
# ----------------------------------------------------------------------------


def test_use_performance_kernels_default_keeps_kv_cache_and_compat_on(
    tmp_path, monkeypatch
):
    """Default-off ``use_performance_kernels`` still applies compat and the
    ``performance`` bucket (kv_cache); only the Triton ``kernels`` group is
    disabled."""
    calls: list[dict] = []

    def _spy(model, **kwargs):
        calls.append({"model": model, "kwargs": kwargs})

    monkeypatch.setattr("opake.patches.apply_model_patches", _spy)

    _tiny_trainer(tmp_path)  # use_performance_kernels default is False
    assert len(calls) == 1
    assert calls[0]["kwargs"]["performance"] is True
    assert calls[0]["kwargs"]["kernels"] is False
    assert calls[0]["kwargs"]["compat"] is True
    assert "fused_linear_cross_entropy" not in calls[0]["kwargs"]


def test_use_performance_kernels_true_enables_kernels_group(tmp_path, monkeypatch):
    """``use_performance_kernels=True`` flips ``kernels`` on at the
    ``apply_model_patches`` call (alongside the always-on ``performance``
    and ``compat`` umbrellas)."""
    calls: list[dict] = []

    def _spy(model, **kwargs):
        calls.append({"model": model, "kwargs": kwargs})

    # _performance_kernels.py imports apply_model_patches lazily inside the function body,
    # so patch the source location rather than the consumer's module.
    monkeypatch.setattr("opake.patches.apply_model_patches", _spy)

    _trainer, model = _tiny_trainer(tmp_path, use_performance_kernels=True)
    assert len(calls) == 1
    assert calls[0]["model"] is model
    assert calls[0]["kwargs"]["performance"] is True
    assert calls[0]["kwargs"]["kernels"] is True
    assert calls[0]["kwargs"]["compat"] is True


def test_performance_kernels_config_forwards_opake_keys_as_is(tmp_path, monkeypatch):
    """``performance_kernels_config`` is a flat dict forwarded as-is to
    ``apply_model_patches`` kwargs — no key translation, opake-patches
    keys used directly."""
    calls: list[dict] = []

    def _spy(model, **kwargs):
        calls.append({"kwargs": kwargs})

    monkeypatch.setattr("opake.patches.apply_model_patches", _spy)

    _tiny_trainer(
        tmp_path,
        use_performance_kernels=True,
        performance_kernels_config={
            "rope": True,
            "rms_norm": True,
            "fused_linear_cross_entropy": True,
            "chunked_linear_cross_entropy": 2048,
        },
    )
    assert len(calls) == 1
    assert calls[0]["kwargs"]["rope"] is True
    assert calls[0]["kwargs"]["rms_norm"] is True
    assert calls[0]["kwargs"]["fused_linear_cross_entropy"] is True
    assert calls[0]["kwargs"]["chunked_linear_cross_entropy"] == 2048


def test_performance_kernels_config_can_disable_automatic_fused_loss(
    tmp_path, monkeypatch
):
    calls: list[dict] = []

    def _spy(model, **kwargs):
        calls.append({"kwargs": kwargs})

    monkeypatch.setattr("opake.patches.apply_model_patches", _spy)

    trainer, _model = _tiny_trainer(
        tmp_path,
        performance_kernels_config={"fused_linear_cross_entropy": False},
    )
    assert calls[0]["kwargs"]["fused_linear_cross_entropy"] is False
    assert trainer._fused_forward_uses_marker is False


def test_performance_kernels_config_can_disable_kv_cache(tmp_path, monkeypatch):
    """``kv_cache`` stays on by default but can be opted out via the config
    dict for models whose forward depends on HF's DynamicCache."""
    calls: list[dict] = []

    def _spy(model, **kwargs):
        calls.append({"kwargs": kwargs})

    monkeypatch.setattr("opake.patches.apply_model_patches", _spy)

    _tiny_trainer(tmp_path, performance_kernels_config={"kv_cache": False})
    assert len(calls) == 1
    assert calls[0]["kwargs"]["kv_cache"] is False
    assert calls[0]["kwargs"]["performance"] is True


# ----------------------------------------------------------------------------
# Precision: bf16 is the only mixed-precision mode; fp16 is unsupported
# ----------------------------------------------------------------------------


def test_bf16_trainer_enables_autocast(tmp_path):
    """bf16=True sets the autocast dtype (no loss scaler — bf16's wider
    exponent range needs none)."""
    trainer, _ = _tiny_trainer(tmp_path, bf16=True)
    assert trainer._amp_dtype == torch.bfloat16


def test_fp32_trainer_has_no_autocast(tmp_path):
    trainer, _ = _tiny_trainer(tmp_path)
    assert trainer._amp_dtype is None


def test_fp16_training_is_rejected(tmp_path):
    """fp16 training (autocast + dynamic loss scaling) is unsupported."""
    with pytest.raises(TypeError):
        _tiny_trainer(tmp_path, fp16=True)
