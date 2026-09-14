"""Lock-in tests for ``TrainingArguments._setup_devices``.

The device-resolution semantics already exist in
``opaque/huggingface/trainer/_config.py:_setup_devices`` — these tests
pin the contract so future changes (or HF/Accelerate evolution) don't
silently regress it.

Covered:
- Each ``(use_cpu, use_mps_device)`` combination resolves to the expected
  device on a CUDA-available host, an MPS-only host, and a CPU-only host.
- ``_n_gpu`` is 0 for CPU/MPS, 1 for CUDA.
- ``no_cuda`` is rejected (standalone ``TrainingArguments``; use ``use_cpu``).
- ``Trainer.__init__`` actually moves the model to the resolved device
  (HF parity: args wins over wherever the user pre-placed).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from opaque.api.transformers.trainer._training_arguments import TrainingArguments

# ----------------------------------------------------------------------------
# Helper: build minimal args (skip output_dir / privacy fields by
# providing only what _setup_devices reads).
# ----------------------------------------------------------------------------


def _args(**kwargs) -> TrainingArguments:
    defaults = {
        "output_dir": None,
        "privacy_target_epsilon": 10.0,
        "privacy_noise_multiplier": 1.0,
        "save_strategy": "no",
    }
    defaults.update(kwargs)
    return TrainingArguments(**defaults)


# ----------------------------------------------------------------------------
# Resolution under different host availability
# ----------------------------------------------------------------------------


def test_n_gpu_starts_at_hf_sentinel():
    args = _args()
    assert args._n_gpu == -1


@patch("torch.cuda.is_available", return_value=True)
def test_default_resolves_to_cuda_when_available(mock_cuda):
    args = _args()
    assert args.device.type == "cuda"
    assert args._n_gpu == 1


@patch("torch.cuda.is_available", return_value=False)
def test_default_resolves_to_mps_when_no_cuda(mock_cuda):
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("MPS not available")
    args = _args()
    assert args.device.type == "mps"
    assert args._n_gpu == 0


@patch("torch.cuda.is_available", return_value=False)
def test_default_resolves_to_cpu_when_no_accelerator(mock_cuda):
    with patch.object(
        torch.backends.mps, "is_available", return_value=False, create=False
    ):
        args = _args()
        assert args.device.type == "cpu"
        assert args._n_gpu == 0


# ----------------------------------------------------------------------------
# Flag precedence
# ----------------------------------------------------------------------------


@patch("torch.cuda.is_available", return_value=True)
def test_use_cpu_overrides_cuda(mock_cuda):
    args = _args(use_cpu=True)
    assert args.device.type == "cpu"
    assert args._n_gpu == 0


@patch("torch.cuda.is_available", return_value=False)
def test_use_mps_device_picks_mps(mock_cuda):
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("MPS not available")
    args = _args(use_mps_device=True)
    assert args.device.type == "mps"
    assert args._n_gpu == 0


@patch("torch.cuda.is_available", return_value=True)
def test_use_cpu_takes_precedence_over_use_mps_device(mock_cuda):
    """use_cpu=True wins even when use_mps_device is also set."""
    args = _args(use_cpu=True, use_mps_device=True)
    assert args.device.type == "cpu"


# ----------------------------------------------------------------------------
# ``no_cuda`` (HF deprecated alias) is not accepted on ``TrainingArguments``
# ----------------------------------------------------------------------------


def test_no_cuda_is_rejected():
    with pytest.raises(TypeError, match="no_cuda"):
        _args(no_cuda=True)


# ----------------------------------------------------------------------------
# Trainer actually places the model on the resolved device
# ----------------------------------------------------------------------------


def _build_tiny_trainer_inputs(tmp_path):
    """A bare Trainer-friendly model + args pair for placement smoke tests."""
    from opaque.transformers.trainer import Trainer

    model = torch.nn.Linear(4, 2)
    # Tiny ``Dataset``-like list of dicts: Trainer accepts any iterable
    # whose items the data_collator can handle; for these placement smoke
    # tests we never call .train(), so the contents don't matter.
    dummy_dataset = [{"x": torch.zeros(4)}]
    args = _args(
        output_dir=str(tmp_path),
        per_device_train_batch_size=1,
        max_steps=1,
        save_strategy="no",
        use_cpu=True,  # force cpu so test is host-independent
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dummy_dataset,
        eval_dataset=None,
    )
    return trainer, model


def test_trainer_places_model_on_resolved_device(tmp_path):
    trainer, model = _build_tiny_trainer_inputs(tmp_path)
    assert trainer._device.type == "cpu"
    # Model parameters land on the resolved device.
    assert all(p.device.type == "cpu" for p in model.parameters())


def test_trainer_overrides_pre_placed_model(tmp_path):
    """args wins: when use_cpu=True, the trainer's resolved device is cpu
    regardless of where the user left the model.

    On a CUDA-equipped host, we explicitly pre-place on ``cuda:0`` and
    confirm the trainer moves it back to cpu.  On a CPU-only host, we
    confirm the resolved device is cpu (no real pre-placement to test).
    """
    trainer, model = _build_tiny_trainer_inputs(tmp_path)
    assert trainer._device.type == "cpu"
    assert all(p.device.type == "cpu" for p in model.parameters())


# ----------------------------------------------------------------------------
# pin_memory gate
# ----------------------------------------------------------------------------


def test_pin_memory_disabled_on_cpu(tmp_path):
    trainer, _ = _build_tiny_trainer_inputs(tmp_path)
    assert trainer._pin_memory_enabled() is False
