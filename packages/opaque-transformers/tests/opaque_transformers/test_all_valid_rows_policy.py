# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""``DPTrainer`` installs its own ``all_valid_rows`` policy and restores the previous one."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from opaque.exceptions import ConfigurationError
from opaque.patches import all_valid_rows, set_all_valid_rows
from opaque.transformers.trainer import DPTrainer, TrainingArguments


@pytest.fixture(autouse=True)
def _reset_policy():
    previous = all_valid_rows()
    set_all_valid_rows(None)
    yield
    set_all_valid_rows(previous)


def _trainer(tmp_path, **overrides) -> DPTrainer:
    kwargs = {
        "output_dir": str(tmp_path),
        "per_device_train_batch_size": 1,
        "max_steps": 1,
        "save_strategy": "no",
        "use_cpu": True,
        "privacy_target_epsilon": 10.0,
        "privacy_noise_multiplier": 1.0,
    }
    kwargs.update(overrides)
    args = TrainingArguments(**kwargs)
    return DPTrainer(
        model=nn.Linear(4, 2),
        args=args,
        train_dataset=[{"x": torch.zeros(4)}],
        eval_dataset=None,
    )


_NON_PRIVATE = {"privacy_target_epsilon": None, "privacy_noise_multiplier": 0.0}


@pytest.mark.parametrize("policy", [None, True, False])
def test_run_policy_is_the_argument_and_the_previous_value_comes_back(tmp_path, policy):
    foreign = policy is not True
    set_all_valid_rows(foreign)
    private = {} if policy is not None else _NON_PRIVATE
    trainer = _trainer(tmp_path, all_valid_rows=policy, **private)

    previous = trainer._install_all_valid_rows_policy()
    assert previous is foreign
    assert all_valid_rows() is policy

    trainer._restore_all_valid_rows_policy(previous)
    assert all_valid_rows() is foreign


def test_default_never_probes_the_batch(tmp_path):
    assert _trainer(tmp_path).args.all_valid_rows is False


def test_private_run_rejects_the_probe(tmp_path):
    with pytest.raises(ConfigurationError, match="not a private mode"):
        _trainer(tmp_path, all_valid_rows=None)
    with pytest.raises(ConfigurationError, match="not a private mode"):
        _trainer(
            tmp_path,
            all_valid_rows=None,
            privacy_noise_multiplier=None,
            privacy_target_epsilon=4.0,
        )


def test_non_private_run_accepts_the_probe(tmp_path):
    trainer = _trainer(tmp_path, all_valid_rows=None, **_NON_PRIVATE)
    assert trainer.args.all_valid_rows is None


def test_train_once_restores_the_previous_policy_on_failure(tmp_path, monkeypatch):
    set_all_valid_rows(True)
    trainer = _trainer(tmp_path)
    seen = {}

    def boom(self, **kwargs):
        seen["during"] = all_valid_rows()
        raise RuntimeError("boom")

    monkeypatch.setattr(DPTrainer, "_train_once_with_policy", boom)
    with pytest.raises(RuntimeError, match="boom"):
        trainer._train_once(
            resume_from_checkpoint=None,
            microbatch_size_override=None,
            ignore_keys_for_eval=None,
        )
    assert seen["during"] is False
    assert all_valid_rows() is True
