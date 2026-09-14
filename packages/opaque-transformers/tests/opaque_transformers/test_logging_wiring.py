# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""``log_level`` / ``log_level_replica`` wiring + ``privacy_noise_radius`` removal.

Regression coverage for issue #388: the log-level fields were accepted and
documented but never consumed, and ``privacy_noise_radius`` was inert.
"""

from __future__ import annotations

import logging

import pytest
import torch
from transformers.utils import logging as hf_logging

from opaque.api.transformers.trainer._distributed import DDPState, apply_logging
from opaque.transformers.trainer import Trainer, TrainingArguments


@pytest.fixture
def restore_logging():
    """Save/restore the process-global logging state apply_logging mutates."""
    saved_verbosity = hf_logging.get_verbosity()
    opaque_logger = logging.getLogger("opaque")
    saved_level = opaque_logger.level
    try:
        yield
    finally:
        hf_logging.set_verbosity(saved_verbosity)
        opaque_logger.setLevel(saved_level)


def _ddp(is_distributed=False, rank=0, local_rank=0, world_size=1):
    return DDPState(
        is_distributed=is_distributed,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=None,
        device=torch.device("cpu"),
    )


def _args(tmp_path, **kw):
    return TrainingArguments(
        privacy_noise_multiplier=1.0,
        output_dir=str(tmp_path),
        use_cpu=True,
        **kw,
    )


def test_main_rank_uses_log_level(tmp_path, restore_logging):
    raw = apply_logging(_args(tmp_path, log_level="error"), _ddp())
    assert raw == 40
    assert hf_logging.get_verbosity() == 40
    assert logging.getLogger("opaque").level == 40


def test_passive_leaves_levels_unchanged(tmp_path, restore_logging):
    logging.getLogger("opaque").setLevel(logging.DEBUG)  # sentinel
    hf_logging.set_verbosity(10)  # sentinel (detail/debug)
    raw = apply_logging(_args(tmp_path, log_level="passive"), _ddp())
    assert raw == -1
    assert logging.getLogger("opaque").level == logging.DEBUG  # untouched
    assert hf_logging.get_verbosity() == 10  # transformers verbosity untouched


def test_replica_uses_log_level_replica(tmp_path, restore_logging):
    args = _args(tmp_path, log_level="passive", log_level_replica="error")
    ddp = _ddp(is_distributed=True, rank=1, local_rank=1, world_size=2)
    assert apply_logging(args, ddp) == 40
    assert logging.getLogger("opaque").level == 40


def test_main_rank_ignores_replica(tmp_path, restore_logging):
    args = _args(tmp_path, log_level="error", log_level_replica="debug")
    ddp = _ddp(is_distributed=True, rank=0, local_rank=0, world_size=2)
    assert apply_logging(args, ddp) == 40  # main -> log_level, not replica


def test_invalid_log_level_rejected(tmp_path):
    with pytest.raises(ValueError, match="log_level="):
        _args(tmp_path, log_level="bogus")


def test_privacy_noise_radius_removed(tmp_path):
    with pytest.raises(TypeError):
        _args(tmp_path, privacy_noise_radius=3.0)


def test_hf_baseline_respects_use_cpu_for_runtime_defaults(tmp_path):
    from transformers import TrainingArguments as HFTrainingArguments

    opaque = TrainingArguments.from_hf(
        HFTrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            report_to=[],
            per_device_train_batch_size=1,
        ),
        privacy_noise_multiplier=1.0,
        clipping_norm=1.0,
    )
    assert opaque.use_cpu is True


def test_apply_logging_runs_at_trainer_construction(tmp_path, restore_logging):
    """The wiring call site (``Trainer.__init__`` -> ``apply_logging``) fires.

    Guards the fix against silently reverting to "accepted but never consumed":
    deleting the ``apply_logging`` call in ``Trainer.__init__`` makes this fail.
    """
    model = torch.nn.Linear(4, 2)
    dummy_dataset = [{"x": torch.zeros(4)}]
    args = _args(
        tmp_path,
        log_level="error",
        per_device_train_batch_size=1,
        max_steps=1,
        save_strategy="no",
    )
    Trainer(model=model, args=args, train_dataset=dummy_dataset, eval_dataset=None)
    assert hf_logging.get_verbosity() == 40  # log_level="error" applied in __init__
