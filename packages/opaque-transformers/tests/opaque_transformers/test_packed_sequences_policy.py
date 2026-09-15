# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""``DPTrainer`` installs its own ``packed_sequences`` policy and restores the previous one."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from opaque.patches import packed_sequences, set_packed_sequences
from opaque.transformers.trainer import DPTrainer, TrainingArguments


@pytest.fixture(autouse=True)
def _reset_policy():
    previous = packed_sequences()
    set_packed_sequences(None)
    yield
    set_packed_sequences(previous)


def _trainer(tmp_path, **overrides) -> DPTrainer:
    args = TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=1,
        max_steps=1,
        save_strategy="no",
        use_cpu=True,
        privacy_target_epsilon=10.0,
        privacy_noise_multiplier=1.0,
        **overrides,
    )
    return DPTrainer(
        model=nn.Linear(4, 2),
        args=args,
        train_dataset=[{"x": torch.zeros(4)}],
        eval_dataset=None,
    )


@pytest.mark.parametrize("policy", [None, True, False])
def test_run_policy_is_the_argument_and_the_previous_value_comes_back(tmp_path, policy):
    foreign = policy is not True
    set_packed_sequences(foreign)
    trainer = _trainer(tmp_path, packed_sequences=policy)

    previous = trainer._install_packed_sequences_policy()
    assert previous is foreign
    assert packed_sequences() is policy

    trainer._restore_packed_sequences_policy(previous)
    assert packed_sequences() is foreign


def test_default_never_probes_the_batch(tmp_path):
    assert _trainer(tmp_path).args.packed_sequences is False


def test_train_once_restores_the_previous_policy_on_failure(tmp_path, monkeypatch):
    set_packed_sequences(True)
    trainer = _trainer(tmp_path)
    seen = {}

    def boom(self, **kwargs):
        seen["during"] = packed_sequences()
        raise RuntimeError("boom")

    monkeypatch.setattr(DPTrainer, "_train_once_with_policy", boom)
    with pytest.raises(RuntimeError, match="boom"):
        trainer._train_once(
            resume_from_checkpoint=None,
            microbatch_size_override=None,
            ignore_keys_for_eval=None,
        )
    assert seen["during"] is False
    assert packed_sequences() is True
