"""Trainer-level wiring for ``torch_compile``, ``use_performance_kernels``, and
compute-precision flags.

These tests target the *plumbing* — compile / kernel features behave
correctly when flags flip — without running full training (which would
require a complete
data collator, sampler, dataset, accountant). The actual training/eval
behavior is covered by the broader trainer suite.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch._dynamo.testing import CompileCounterWithBackend
from transformers import PretrainedConfig, PreTrainedModel

from opaque.api.transformers.trainer._distributed import DDPState
from opaque.api.transformers.trainer._trainer import _compile_strict_chunk
from opaque.exceptions import ConfigurationError
from opaque.transformers.trainer import Trainer, TrainingArguments

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


def _tiny_trainer(tmp_path, **arg_overrides) -> tuple[Trainer, nn.Module]:
    model = nn.Linear(4, 2)
    args = _args(tmp_path, **arg_overrides)
    trainer = Trainer(
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
        Trainer(
            model=_UnregisteredHFModel(),
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


def test_torch_compile_runs_poisson_training_strictly(tmp_path):
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
    trainer = Trainer(
        model=_TinyLM(),
        args=args,
        train_dataset=dataset,
        data_collator=collate,
    )

    result = trainer.train()

    assert result.global_step == 3


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
        Trainer(
            model=model,
            args=_args(tmp_path, torch_compile=True),
            train_dataset=[{"input_ids": torch.zeros(4, dtype=torch.long)}],
        )


# ----------------------------------------------------------------------------
# strict chunk compilation
# ----------------------------------------------------------------------------


def test_strict_chunk_compiler_requests_dynamic_fullgraph(monkeypatch):
    compile_calls = []

    def fake_compile(fn, *, backend, mode, fullgraph, dynamic):
        compile_calls.append((fn, backend, mode, fullgraph, dynamic))
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)

    def chunk(params, x, y):
        return params + x.sum() + y.sum()

    compiled = _compile_strict_chunk(
        chunk,
        backend="aot_eager",
        mode="default",
    )
    torch.testing.assert_close(
        compiled(torch.tensor(1.0), torch.ones(3), torch.ones(3)),
        torch.tensor(7.0),
    )
    assert compile_calls == [(chunk, "aot_eager", "default", True, True)]


def test_strict_chunk_compiler_reuses_dynamic_graph_across_batch_sizes():
    torch._dynamo.reset()
    backend = CompileCounterWithBackend("aot_eager")

    def chunk(x):
        return x.sin().sum(dim=0)

    compiled = _compile_strict_chunk(chunk, backend=backend, mode="default")
    generator = torch.Generator().manual_seed(0)
    for batch_size in (4, 2, 3, 1):
        x = torch.randn(batch_size, 5, generator=generator)
        torch.testing.assert_close(compiled(x), chunk(x))

    # Symbolic dimensions specialize at size one on supported PyTorch versions.
    assert 1 <= backend.frame_count <= 2


def test_strict_chunk_compiler_propagates_lazy_compile_failure(monkeypatch):
    failure = torch._dynamo.exc.Unsupported("graph break")

    def fake_compile(fn, *, backend, mode, fullgraph, dynamic):
        def compiled(*args, **kwargs):
            raise failure

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    compiled = _compile_strict_chunk(
        lambda x: x,
        backend="aot_eager",
        mode="default",
    )
    with pytest.raises(torch._dynamo.exc.Unsupported, match="graph break"):
        compiled(torch.ones(2))


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

    monkeypatch.setattr("opaque.patches.apply_model_patches", _spy)

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
    monkeypatch.setattr("opaque.patches.apply_model_patches", _spy)

    _trainer, model = _tiny_trainer(tmp_path, use_performance_kernels=True)
    assert len(calls) == 1
    assert calls[0]["model"] is model
    assert calls[0]["kwargs"]["performance"] is True
    assert calls[0]["kwargs"]["kernels"] is True
    assert calls[0]["kwargs"]["compat"] is True


def test_performance_kernels_config_forwards_opaque_keys_as_is(tmp_path, monkeypatch):
    """``performance_kernels_config`` is a flat dict forwarded as-is to
    ``apply_model_patches`` kwargs — no key translation, opaque-patches
    keys used directly."""
    calls: list[dict] = []

    def _spy(model, **kwargs):
        calls.append({"kwargs": kwargs})

    monkeypatch.setattr("opaque.patches.apply_model_patches", _spy)

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

    monkeypatch.setattr("opaque.patches.apply_model_patches", _spy)

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

    monkeypatch.setattr("opaque.patches.apply_model_patches", _spy)

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
