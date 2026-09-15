# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""``DPTrainer`` with the MoE router-load release (``router_load=True``).

A tiny random-init Mellum 2.0 model (E = 8, k = 2, L = 2) runs a few DP-SGD
steps: the clipper is the MoE one, the accountant wraps ``moe_aux``, the
monitor is logged, and a checkpoint resume continues the release state.
"""

from __future__ import annotations

import importlib
import logging
import math

import pytest
import torch
from torch.utils.data import Dataset

pytest.importorskip("transformers")

from opaque.api.transformers.trainer._dp_trainer import DPTrainer
from opaque.dpsgd.accounting.mechanisms.types import MoeAux
from opaque.dpsgd.clipping.types import MoeClipState
from opaque.exceptions import CheckpointError, ConfigurationError
from opaque.transformers import TrainingArguments

E, K, L, VOCAB, SEQ = 8, 2, 2, 128, 12


def _tiny_mellum():
    try:
        mod = importlib.import_module("transformers.models.mellum.modeling_mellum")
        cfg_mod = importlib.import_module(
            "transformers.models.mellum.configuration_mellum"
        )
    except ModuleNotFoundError:
        pytest.skip("mellum is unavailable in this transformers")
    if not hasattr(mod, "MellumExperts"):
        pytest.skip("mellum: stacked experts module absent")
    torch.manual_seed(0)
    config = cfg_mod.MellumConfig(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=L,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rope_theta=10000.0,
        num_experts=E,
        num_experts_per_tok=K,
        moe_intermediate_size=32,
        router_aux_loss_coef=0.05,
    )
    config._attn_implementation = "eager"
    return mod.MellumForCausalLM(config)


class _RaggedDS(Dataset):
    def __init__(self, n: int = 32) -> None:
        g = torch.Generator().manual_seed(1)
        self._ids = torch.randint(3, VOCAB, (n, SEQ), generator=g)
        self._len = torch.randint(5, SEQ + 1, (n,), generator=g)

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        mask = (torch.arange(SEQ) < self._len[i]).long()
        ids = torch.where(mask.bool(), self._ids[i], torch.zeros_like(self._ids[i]))
        labels = torch.where(mask.bool(), ids, torch.full_like(ids, -100))
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def _collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def _filtered_std_after_drift(
    first: float, scales: list[float], beta: float = 0.99
) -> float:
    """Recursive noise std of the filter after releases at ``first * scale`` each."""
    var, decay = 0.0, 1.0
    for scale in scales:
        var = beta**2 * var + (1 - beta) ** 2 * (first * scale) ** 2
        decay *= beta
    return math.sqrt(var) / (1 - decay)


def _args(output_dir, **overrides):
    kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": 4,
        "max_steps": 3,
        "privacy_noise_multiplier": 1.0,
        "clipping_norm": 1.0,
        "router_load": True,
        "router_load_ratio": 0.5,
        "router_load_max_tokens": SEQ,
        "learning_rate": 1e-3,
        "optim": "sgd",
        "report_to": [],
        "eval_strategy": "no",
        "logging_strategy": "steps",
        "logging_steps": 1,
        "save_strategy": "no",
        "disable_tqdm": True,
        "use_cpu": True,
        "seed": 0,
        "dataloader_num_workers": 0,
        "use_performance_kernels": False,
    }
    kwargs.update(overrides)
    return TrainingArguments(**kwargs)


class TestArguments:
    def test_requires_max_tokens(self):
        with pytest.raises(ConfigurationError, match="router_load_max_tokens"):
            _args("/tmp/x", router_load_max_tokens=None)

    def test_rejects_adaptive_clipping(self):
        with pytest.raises(ConfigurationError, match="adaptive"):
            _args("/tmp/x", clipping_mode="adaptive")

    def test_rejects_non_gaussian_mechanism(self):
        with pytest.raises(ConfigurationError, match="gaussian"):
            _args("/tmp/x", privacy_noise_mechanism="mf_identity")

    def test_rejects_bad_ratio_and_unknown_kwargs(self):
        with pytest.raises(ConfigurationError, match="router_load_ratio"):
            _args("/tmp/x", router_load_ratio=0.0)
        with pytest.raises(ConfigurationError, match="router_load_kwargs"):
            _args("/tmp/x", router_load_kwargs={"lam": 1.0})

    def test_off_by_default_ignores_the_rest(self):
        args = TrainingArguments(
            output_dir="/tmp/x",
            report_to=[],
            use_cpu=True,
            privacy_noise_multiplier=1.0,
        )
        assert args.router_load is False
        assert args.router_load_ratio == pytest.approx(0.02)


class TestTrain:
    def test_trains_with_the_moe_clipper_and_joint_accountant(self, tmp_path):
        model = _tiny_mellum()
        trainer = DPTrainer(
            model=model,
            args=_args(tmp_path),
            train_dataset=_RaggedDS(),
            data_collator=_collate,
        )
        assert trainer._moe_geometry == {"top_k": K, "num_experts": E, "num_layers": L}
        assert trainer._router_load_alpha == pytest.approx(0.05)
        out = trainer.train()
        assert out.global_step == 3
        assert "privacy_epsilon" in out.metrics
        logged = [
            row for row in trainer.state.log_history if "router_load_imbalance" in row
        ]
        assert len(logged) == 3
        assert all(row["router_load_noise_std"] > 0 for row in logged)
        # The accountant prices the joint release, not the plain Gaussian.
        process = trainer._accountant.process
        assert "MoeAux" in repr(process)
        assert MoeAux.__name__ in repr(process)

    def test_custom_loss_func_is_rejected(self, tmp_path):
        with pytest.raises(ConfigurationError, match="compute_loss_func"):
            DPTrainer(
                model=_tiny_mellum(),
                args=_args(tmp_path),
                train_dataset=_RaggedDS(),
                data_collator=_collate,
                compute_loss_func=lambda output, labels: output["loss"],
            )

    def test_trains_with_the_fused_routes_off(self, tmp_path):
        trainer = DPTrainer(
            model=_tiny_mellum(),
            args=_args(
                tmp_path,
                max_steps=1,
                performance_kernels_config={"fused_linear_cross_entropy": False},
            ),
            train_dataset=_RaggedDS(),
            data_collator=_collate,
        )
        assert trainer.train().global_step == 1

    def test_alpha_override_and_geometry_from_model(self, tmp_path):
        model = _tiny_mellum()
        trainer = DPTrainer(
            model=model,
            args=_args(tmp_path, router_load_kwargs={"alpha": 0.2, "filter_beta": 0.9}),
            train_dataset=_RaggedDS(),
            data_collator=_collate,
        )
        assert trainer._router_load_alpha == pytest.approx(0.2)
        trainer.train()

    def test_dense_model_is_rejected(self, tmp_path):
        from transformers import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig(
            vocab_size=VOCAB,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            pad_token_id=0,
        )
        config._attn_implementation = "eager"
        with pytest.raises(ConfigurationError, match="no top-k router"):
            DPTrainer(
                model=LlamaForCausalLM(config),
                args=_args(tmp_path),
                train_dataset=_RaggedDS(),
                data_collator=_collate,
            )

    def test_resume_continues_the_release_state(self, tmp_path):
        """Resuming from step 2 reproduces an uninterrupted run's steps 3 and 4."""
        common = {
            "save_strategy": "steps",
            "save_steps": 2,
            "lr_scheduler": "constant",
        }
        reference = DPTrainer(
            model=_tiny_mellum(),
            args=_args(tmp_path / "reference", max_steps=4, **common),
            train_dataset=_RaggedDS(),
            data_collator=_collate,
        )
        reference.train()

        first = DPTrainer(
            model=_tiny_mellum(),
            args=_args(tmp_path / "run", max_steps=2, **common),
            train_dataset=_RaggedDS(),
            data_collator=_collate,
        )
        first.train()
        ckpt = tmp_path / "run" / "checkpoint-2"
        assert ckpt.is_dir()
        # A fresh model instance: the weights come back from the checkpoint.
        resumed = DPTrainer(
            model=_tiny_mellum(),
            args=_args(tmp_path / "run", max_steps=4, **common),
            train_dataset=_RaggedDS(),
            data_collator=_collate,
        )
        out = resumed.train(resume_from_checkpoint=str(ckpt))
        assert out.global_step == 4

        def rows(trainer):
            return [
                row
                for row in trainer.state.log_history
                if "router_load_imbalance" in row
            ]

        ref_rows, res_rows = rows(reference), rows(resumed)
        # The resumed trainer restores the checkpoint's log history (steps 1
        # and 2) and appends the two resumed steps.
        assert len(ref_rows) == 4
        assert len(res_rows) == 4
        for row, ref in zip(res_rows[2:], ref_rows[2:], strict=True):
            assert row["router_load_noise_std"] == pytest.approx(
                ref["router_load_noise_std"]
            )
            assert row["router_load_imbalance"] == pytest.approx(
                ref["router_load_imbalance"], rel=1e-4
            )
        # Known noise keeps falling: the resumed releases continue the filter
        # rather than starting a fresh one.
        assert (
            res_rows[2]["router_load_noise_std"] < ref_rows[0]["router_load_noise_std"]
        )

    def test_resume_with_a_changed_ratio_warns_and_uses_the_current_one(
        self, tmp_path, caplog
    ):
        common = {"save_strategy": "steps", "save_steps": 2, "lr_scheduler": "constant"}
        DPTrainer(
            model=_tiny_mellum(),
            args=_args(tmp_path / "run", max_steps=2, **common),
            train_dataset=_RaggedDS(),
            data_collator=_collate,
        ).train()
        resumed = DPTrainer(
            model=_tiny_mellum(),
            args=_args(
                tmp_path / "run", max_steps=3, router_load_ratio=0.125, **common
            ),
            train_dataset=_RaggedDS(),
            data_collator=_collate,
        )
        with caplog.at_level(logging.WARNING):
            resumed.train(resume_from_checkpoint=str(tmp_path / "run" / "checkpoint-2"))
        assert any("router_load_ratio" in r.getMessage() for r in caplog.records)
        rows = [
            row for row in resumed.state.log_history if "router_load_noise_std" in row
        ]
        # The third release ran at the current ratio (0.125 vs 0.5: twice the
        # load noise), which is what the accountant priced, and the telemetry
        # tracks the mixed history exactly.  The first row is one release's
        # own std, so it seeds the recursion.
        first = rows[0]["router_load_noise_std"]
        assert rows[2]["router_load_noise_std"] == pytest.approx(
            _filtered_std_after_drift(first, [1.0, 1.0, 2.0]), rel=1e-6
        )


class TestResumeToggle:
    def _run(self, output_dir, **overrides):
        args = _args(
            output_dir,
            max_steps=2,
            save_strategy="steps",
            save_steps=2,
            lr_scheduler="constant",
            **overrides,
        )
        DPTrainer(
            model=_tiny_mellum(),
            args=args,
            train_dataset=_RaggedDS(),
            data_collator=_collate,
        ).train()
        return str(output_dir / "checkpoint-2")

    def _resume(self, output_dir, ckpt, **overrides):
        args = _args(output_dir, max_steps=4, lr_scheduler="constant", **overrides)
        DPTrainer(
            model=_tiny_mellum(),
            args=args,
            train_dataset=_RaggedDS(),
            data_collator=_collate,
        ).train(resume_from_checkpoint=ckpt)

    def test_release_cannot_be_switched_off_by_a_resume(self, tmp_path):
        ckpt = self._run(tmp_path)
        with pytest.raises(CheckpointError, match="cannot be switched on or off"):
            self._resume(tmp_path, ckpt, router_load=False)

    def test_release_cannot_be_switched_on_by_a_resume(self, tmp_path):
        ckpt = self._run(tmp_path, router_load=False)
        with pytest.raises(CheckpointError, match="cannot be switched on or off"):
            self._resume(tmp_path, ckpt)


def test_state_type_is_registered_for_sync():
    from opaque.api.engine.distributed._state import _SYNC_REGISTRY

    assert MoeClipState in _SYNC_REGISTRY
