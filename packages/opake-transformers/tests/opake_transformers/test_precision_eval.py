"""Tests for the eval-only dtype context manager.

Covers:
- ``bf16_full_eval=True``: model is cast to bf16 inside the eval context
  and restored to the training dtype on exit.
- ``bf16_full_eval=True`` + ``bf16=True``: no-op (model already bf16);
  the model stays bf16 after the context exits.
- ``fp16_full_eval`` is unsupported (not a field) and raises ``TypeError``;
  ``bf16_full_eval`` is the only full-cast eval mode.
"""

from __future__ import annotations

import pytest
import torch

from opake.api.transformers.trainer._precision import eval_dtype
from opake.transformers.trainer import TrainingArguments


def _args(**overrides) -> TrainingArguments:
    defaults = {
        "output_dir": None,
        "save_strategy": "no",
        "privacy_target_epsilon": 10.0,
        "privacy_noise_multiplier": 1.0,
        "use_cpu": True,
        # Synthetic ``nn.Linear`` fixture; not in opake's family
        # registry — silence the "no detectable family" info log.
        "use_compat_patches": False,
    }
    defaults.update(overrides)
    return TrainingArguments(**defaults)


# ----------------------------------------------------------------------------
# bf16_full_eval cast + restore
# ----------------------------------------------------------------------------


def test_bf16_full_eval_casts_inside_and_restores_on_exit():
    model = torch.nn.Linear(4, 2)
    train_dtype = torch.float32
    args = _args(bf16_full_eval=True)
    assert next(model.parameters()).dtype == torch.float32
    with eval_dtype(model, args, train_dtype):
        assert next(model.parameters()).dtype == torch.bfloat16
    # Restored after context exit.
    assert next(model.parameters()).dtype == torch.float32


def test_bf16_full_eval_with_bf16_training_is_noop():
    """If the model is already bf16, the context is a no-op."""
    model = torch.nn.Linear(4, 2).to(dtype=torch.bfloat16)
    train_dtype = torch.bfloat16
    args = _args(bf16=True, bf16_full_eval=True)
    with eval_dtype(model, args, train_dtype):
        assert next(model.parameters()).dtype == torch.bfloat16
    assert next(model.parameters()).dtype == torch.bfloat16


def test_bf16_full_eval_false_does_nothing():
    model = torch.nn.Linear(4, 2)
    train_dtype = torch.float32
    args = _args()  # bf16_full_eval default False
    with eval_dtype(model, args, train_dtype):
        assert next(model.parameters()).dtype == torch.float32
    assert next(model.parameters()).dtype == torch.float32


def test_bf16_full_eval_restores_even_on_exception():
    """Training dtype is restored even when the eval body raises."""
    model = torch.nn.Linear(4, 2)
    train_dtype = torch.float32
    args = _args(bf16_full_eval=True)
    with eval_dtype(model, args, train_dtype):
        assert next(model.parameters()).dtype == torch.bfloat16
        with pytest.raises(RuntimeError, match="boom"):
            raise RuntimeError("boom")
    # The finally clause must have restored fp32.
    assert next(model.parameters()).dtype == torch.float32


# ----------------------------------------------------------------------------
# fp16_full_eval is unsupported (bf16_full_eval is the only full-cast mode)
# ----------------------------------------------------------------------------


def test_fp16_full_eval_is_rejected():
    """fp16_full_eval is not a supported field — passing it raises TypeError."""
    with pytest.raises(TypeError):
        _args(fp16_full_eval=True)
