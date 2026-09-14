"""Tests for compute-precision flags on Trainer.

Covers the full-cast precision path:
- ``bf16=True`` casts the model to ``torch.bfloat16`` at ``__init__``.
- ``fp16=True`` is unsupported (not a field) and raises ``TypeError``;
  bf16 is the only mixed-precision mode (no loss scaler needed).
- ``tf32=True``/``False`` flips both ``torch.backends.cuda.matmul.allow_tf32``
  and ``torch.backends.cudnn.allow_tf32``; ``None`` leaves them alone.
- ``self._train_dtype`` captures the effective post-cast dtype for the
  eval-time precision context.
"""

from __future__ import annotations

import pytest
import torch

from opaque.transformers.trainer import Trainer, TrainingArguments


def _args(tmp_path, **overrides) -> TrainingArguments:
    defaults = {
        "output_dir": str(tmp_path),
        "per_device_train_batch_size": 1,
        "max_steps": 1,
        "num_train_epochs": 1,
        "save_strategy": "no",
        "use_cpu": True,
        # Synthetic ``nn.Linear`` fixture; not in opaque's family
        # registry — silence the "no detectable family" info log.
        "use_compat_patches": False,
        "privacy_target_epsilon": 10.0,
        "privacy_noise_multiplier": 1.0,
    }
    defaults.update(overrides)
    return TrainingArguments(**defaults)


def _tiny_trainer(tmp_path, **arg_overrides) -> tuple[Trainer, torch.nn.Module]:
    model = torch.nn.Linear(4, 2)
    args = _args(tmp_path, **arg_overrides)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=[{"x": torch.zeros(4)}],
        eval_dataset=None,
    )
    return trainer, model


# ----------------------------------------------------------------------------
# bf16 full cast
# ----------------------------------------------------------------------------


def test_bf16_true_enables_autocast(tmp_path):
    """bf16=True enables autocast (HF parity); model params stay fp32."""
    trainer, model = _tiny_trainer(tmp_path, bf16=True)
    assert next(model.parameters()).dtype == torch.float32
    assert trainer._train_dtype == torch.float32
    assert trainer._amp_dtype == torch.bfloat16


def test_bf16_default_false_no_autocast(tmp_path):
    trainer, model = _tiny_trainer(tmp_path)
    assert next(model.parameters()).dtype == torch.float32
    assert trainer._train_dtype == torch.float32
    assert trainer._amp_dtype is None


def test_bf16_preserves_pre_cast_model(tmp_path):
    """If the caller pre-cast the model to bf16, autocast is still set,
    but we don't undo their cast."""
    model = torch.nn.Linear(4, 2).to(dtype=torch.bfloat16)
    args = _args(tmp_path, bf16=True)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=[{"x": torch.zeros(4)}],
        eval_dataset=None,
    )
    assert next(model.parameters()).dtype == torch.bfloat16
    assert trainer._train_dtype == torch.bfloat16
    assert trainer._amp_dtype == torch.bfloat16


# ----------------------------------------------------------------------------
# fp16 is unsupported (bf16 is the only mixed-precision mode)
# ----------------------------------------------------------------------------


def test_fp16_is_rejected(tmp_path):
    """fp16 training is not a supported field — passing it raises TypeError."""
    import pytest

    with pytest.raises(TypeError):
        _tiny_trainer(tmp_path, fp16=True)


# ----------------------------------------------------------------------------
# tf32 global flag
# ----------------------------------------------------------------------------


def test_tf32_none_leaves_flags_alone(tmp_path):
    """Default tf32=None must not touch torch.backends flags."""
    before_matmul = torch.backends.cuda.matmul.allow_tf32
    before_cudnn = torch.backends.cudnn.allow_tf32
    _tiny_trainer(tmp_path)  # tf32=None (default)
    # CPU host: even if we wanted to flip, the gate skips non-cuda devices.
    # Either way, the flags are unchanged.
    assert torch.backends.cuda.matmul.allow_tf32 == before_matmul
    assert torch.backends.cudnn.allow_tf32 == before_cudnn


def test_tf32_no_op_on_cpu(tmp_path):
    """tf32=True on a CPU-only run must NOT raise (gated on cuda device)."""
    before_matmul = torch.backends.cuda.matmul.allow_tf32
    before_cudnn = torch.backends.cudnn.allow_tf32
    _tiny_trainer(tmp_path, tf32=True)  # use_cpu=True is in defaults
    # Gate at __init__ checks ``self._device.type == "cuda"``; we're on cpu,
    # so the global flags should not have flipped.
    assert torch.backends.cuda.matmul.allow_tf32 == before_matmul
    assert torch.backends.cudnn.allow_tf32 == before_cudnn


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("tf32_value", [True, False])
def test_tf32_flips_flags_on_cuda(tmp_path, tf32_value):
    """When the resolved device is cuda, tf32 flips both backends to the
    requested value.  Skipped on hosts without real CUDA — the gate
    inside ``__init__`` checks ``self._device.type == "cuda"``, and we
    can't fake that without faking device-side model placement too.
    """
    saved_matmul = torch.backends.cuda.matmul.allow_tf32
    saved_cudnn = torch.backends.cudnn.allow_tf32
    try:
        # Set to opposite of tf32_value so we can observe the flip.
        torch.backends.cuda.matmul.allow_tf32 = not tf32_value
        torch.backends.cudnn.allow_tf32 = not tf32_value
        model = torch.nn.Linear(4, 2)
        args = _args(tmp_path, tf32=tf32_value, use_cpu=False)
        Trainer(
            model=model,
            args=args,
            train_dataset=[{"x": torch.zeros(4)}],
            eval_dataset=None,
        )
        assert torch.backends.cuda.matmul.allow_tf32 is bool(tf32_value)
        assert torch.backends.cudnn.allow_tf32 is bool(tf32_value)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = saved_matmul
        torch.backends.cudnn.allow_tf32 = saved_cudnn
