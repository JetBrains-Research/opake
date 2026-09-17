# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""``DPTrainer`` and ``SFTTrainer`` with the MoE router auxiliary loss.

A tiny random-init Mellum 2.0 model (E = 8, k = 2, L = 2) runs a few DP-SGD
steps with ``router_aux_loss_coef`` set: the clipper is the MoE one, the
accountant wraps ``moe_aux``, the model config is synced the TRL way, the
monitor is logged, and a checkpoint resume continues the release state.
"""

from __future__ import annotations

import importlib
import json
import logging
import math
import types

import pytest
import torch
from torch.utils.data import Dataset

pytest.importorskip("transformers")

from opaque.api.transformers.trainer._dp_trainer import DPTrainer
from opaque.dpsgd.accounting.mechanisms.types import MoeAux
from opaque.dpsgd.clipping.types import MoeClipState
from opaque.exceptions import CheckpointError, ConfigurationError
from opaque.functional import make_functional
from opaque.transformers import TrainingArguments
from opaque.transformers.trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

E, K, L, VOCAB, SEQ = 8, 2, 2, 128, 12
COEF = 0.05


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
        router_aux_loss_coef=0.001,
    )
    config._attn_implementation = "eager"
    return mod.MellumForCausalLM(config)


def _tiny_llama():
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
    return LlamaForCausalLM(config)


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


_COMMON = {
    "per_device_train_batch_size": 4,
    "max_steps": 3,
    "privacy_noise_multiplier": 1.0,
    "privacy_accounting": True,
    "clipping_norm": 1.0,
    "router_aux_loss_coef": COEF,
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


def _args(output_dir, **overrides):
    kwargs = {
        **_COMMON,
        "output_dir": str(output_dir),
        "router_aux_kwargs": {"max_tokens": SEQ, "ratio": 0.5},
    }
    kwargs.update(overrides)
    return TrainingArguments(**kwargs)


def _rows(trainer, key="router_aux_imbalance"):
    return [row for row in trainer.state.log_history if key in row]


def _text_config(model):
    return model.config.get_text_config()


class TestArguments:
    def test_requires_max_tokens(self):
        with pytest.raises(ConfigurationError, match="max_tokens"):
            _args("/tmp/x", router_aux_kwargs={"ratio": 0.5})

    def test_rejects_adaptive_clipping(self):
        with pytest.raises(ConfigurationError, match="adaptive"):
            _args("/tmp/x", clipping_mode="adaptive")

    def test_rejects_non_gaussian_mechanism(self):
        with pytest.raises(ConfigurationError, match="gaussian"):
            _args("/tmp/x", privacy_noise_mechanism="mf_identity")

    def test_rejects_bad_values(self):
        with pytest.raises(ConfigurationError, match="ratio"):
            _args("/tmp/x", router_aux_kwargs={"max_tokens": SEQ, "ratio": 0.0})
        with pytest.raises(ConfigurationError, match="router_aux_kwargs"):
            _args("/tmp/x", router_aux_kwargs={"max_tokens": SEQ, "alpha": 1.0})
        with pytest.raises(ConfigurationError, match="max_tokens"):
            _args("/tmp/x", router_aux_kwargs={"max_tokens": 0})
        with pytest.raises(ConfigurationError, match="router_aux_loss_coef"):
            _args("/tmp/x", router_aux_loss_coef=float("nan"))

    def test_off_by_default_needs_no_bound(self):
        args = TrainingArguments(
            output_dir="/tmp/x",
            report_to=[],
            use_cpu=True,
            privacy_noise_multiplier=1.0,
        )
        assert args.router_aux_loss_coef == 0.0
        assert args.router_aux_kwargs == {}
        assert _args("/tmp/x", router_aux_loss_coef=0.0, router_aux_kwargs={})

    def test_sft_config_derives_the_bound_from_max_length(self):
        cfg = SFTConfig(
            output_dir="/tmp/x",
            report_to=[],
            use_cpu=True,
            privacy_noise_multiplier=1.0,
            router_aux_loss_coef=0.01,
            max_length=64,
        )
        assert cfg.router_aux_kwargs == {"max_tokens": 64}
        explicit = SFTConfig(
            output_dir="/tmp/x",
            report_to=[],
            use_cpu=True,
            privacy_noise_multiplier=1.0,
            router_aux_loss_coef=0.01,
            max_length=64,
            router_aux_kwargs={"max_tokens": 16, "ratio": 0.1},
        )
        assert explicit.router_aux_kwargs == {"max_tokens": 16, "ratio": 0.1}


class TestDPTrainer:
    def test_trains_with_the_moe_clipper_and_joint_accountant(self, tmp_path):
        model = _tiny_mellum()
        trainer = DPTrainer(
            model=model,
            args=_args(tmp_path),
            train_dataset=_RaggedDS(),
            data_collator=_collate,
        )
        assert trainer._moe_geometry == {"top_k": K, "num_experts": E, "num_layers": L}
        assert trainer._router_aux_loss_coef == pytest.approx(COEF)
        assert trainer._router_aux_ratio == pytest.approx(0.5)
        # Config sync: the model's own auxiliary term is off for the run and
        # router logits are requested explicitly by the per-example forward.
        assert _text_config(model).output_router_logits is False
        assert _text_config(model).router_aux_loss_coef == 0.0
        out = trainer.train()
        assert out.global_step == 3
        assert "privacy_epsilon" in out.metrics
        logged = _rows(trainer)
        assert len(logged) == 3
        assert all(row["router_aux_noise_std"] > 0 for row in logged)
        # The accountant prices the joint release, not the plain Gaussian.
        assert MoeAux.__name__ in repr(trainer._accountant.process)

    def test_save_writes_the_model_own_router_flags(self, tmp_path):
        model = _tiny_mellum()
        _text_config(model).output_router_logits = True
        trainer = DPTrainer(
            model=model,
            args=_args(tmp_path, max_steps=1),
            train_dataset=_RaggedDS(),
            data_collator=_collate,
        )
        trainer.train()
        trainer.save_model(str(tmp_path / "saved"))
        saved = json.loads((tmp_path / "saved" / "config.json").read_text())
        assert saved["output_router_logits"] is True
        assert saved["router_aux_loss_coef"] == pytest.approx(0.001)
        # The live config stays synced for the rest of the run.
        assert _text_config(model).output_router_logits is False
        assert _text_config(model).router_aux_loss_coef == 0.0

    def test_zero_coefficient_turns_the_model_aux_term_off(self, tmp_path):
        model = _tiny_mellum()
        _text_config(model).output_router_logits = True
        trainer = DPTrainer(
            model=model,
            args=_args(
                tmp_path, max_steps=1, router_aux_loss_coef=0.0, router_aux_kwargs={}
            ),
            train_dataset=_RaggedDS(),
            data_collator=_collate,
        )
        assert trainer._router_aux_enabled is False
        assert _text_config(model).output_router_logits is False
        assert _text_config(model).router_aux_loss_coef == 0.0
        assert trainer.train().global_step == 1
        assert not _rows(trainer)
        assert MoeAux.__name__ not in repr(trainer._accountant.process)

    def test_dense_model_ignores_the_coefficient(self, tmp_path):
        trainer = DPTrainer(
            model=_tiny_llama(),
            args=_args(tmp_path, max_steps=1),
            train_dataset=_RaggedDS(),
            data_collator=_collate,
        )
        assert trainer._router_aux_enabled is False
        assert trainer.train().global_step == 1
        assert not _rows(trainer)

    def test_custom_loss_func_trains(self, tmp_path):
        """A custom loss keeps the logits; the aux-free marker carries the request."""
        seen: list[tuple[bool, bool]] = []

        def custom_loss(output, labels):
            seen.append(
                (
                    output.get("router_logits") is not None,
                    output.get("aux_loss") is None,
                )
            )
            return output["loss"]

        trainer = DPTrainer(
            model=_tiny_mellum(),
            args=_args(tmp_path, max_steps=1),
            train_dataset=_RaggedDS(),
            data_collator=_collate,
            compute_loss_func=custom_loss,
        )
        assert trainer._router_aux_forward_marker is True
        assert trainer.train().global_step == 1
        assert len(_rows(trainer)) == 1
        assert any(has_router for has_router, _ in seen)
        assert all(aux_free for _, aux_free in seen)

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
        assert len(_rows(trainer)) == 1

    def test_kwargs_reach_the_clipper(self, tmp_path):
        trainer = DPTrainer(
            model=_tiny_mellum(),
            args=_args(
                tmp_path,
                max_steps=1,
                router_aux_kwargs={
                    "max_tokens": SEQ,
                    "ratio": 0.25,
                    "mean_tokens": 8,
                    "filter_beta": 0.9,
                },
            ),
            train_dataset=_RaggedDS(),
            data_collator=_collate,
        )
        assert trainer._router_aux_ratio == pytest.approx(0.25)
        assert trainer._router_aux_factory_kwargs == {
            "max_tokens": SEQ,
            "mean_tokens": 8,
            "filter_beta": 0.9,
        }
        assert trainer.train().global_step == 1
        # The same ratio prices the joint release.
        assert "ratio=0.25" in repr(trainer._accountant.process)

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

        ref_rows, res_rows = _rows(reference), _rows(resumed)
        # The resumed trainer restores the checkpoint's log history (steps 1
        # and 2) and appends the two resumed steps.
        assert len(ref_rows) == 4
        assert len(res_rows) == 4
        for row, ref in zip(res_rows[2:], ref_rows[2:], strict=True):
            assert row["router_aux_noise_std"] == pytest.approx(
                ref["router_aux_noise_std"]
            )
            assert row["router_aux_imbalance"] == pytest.approx(
                ref["router_aux_imbalance"], rel=1e-4
            )
        # Known noise keeps falling: the resumed releases continue the filter
        # rather than starting a fresh one.
        assert res_rows[2]["router_aux_noise_std"] < ref_rows[0]["router_aux_noise_std"]

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
                tmp_path / "run",
                max_steps=3,
                router_aux_kwargs={"max_tokens": SEQ, "ratio": 0.125},
                **common,
            ),
            train_dataset=_RaggedDS(),
            data_collator=_collate,
        )
        with caplog.at_level(logging.WARNING):
            resumed.train(resume_from_checkpoint=str(tmp_path / "run" / "checkpoint-2"))
        assert any("router_aux_ratio" in r.getMessage() for r in caplog.records)
        rows = _rows(resumed, "router_aux_noise_std")
        # The third release ran at the current ratio (0.125 vs 0.5: twice the
        # load noise), which is what the accountant priced, and the telemetry
        # tracks the mixed history exactly.  The first row is one release's
        # own std, so it seeds the recursion.
        first = rows[0]["router_aux_noise_std"]
        assert rows[2]["router_aux_noise_std"] == pytest.approx(
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
        trainer = DPTrainer(
            model=_tiny_mellum(),
            args=args,
            train_dataset=_RaggedDS(),
            data_collator=_collate,
        )
        trainer.train(resume_from_checkpoint=ckpt)
        return trainer

    def test_release_cannot_be_switched_off_by_a_resume(self, tmp_path):
        ckpt = self._run(tmp_path)
        with pytest.raises(CheckpointError, match="cannot be switched on or off"):
            self._resume(tmp_path, ckpt, router_aux_loss_coef=0.0)

    def test_release_cannot_be_switched_on_by_a_resume(self, tmp_path):
        ckpt = self._run(tmp_path, router_aux_loss_coef=0.0)
        with pytest.raises(CheckpointError, match="cannot be switched on or off"):
            self._resume(tmp_path, ckpt)

    def test_changed_coefficient_warns_and_continues(self, tmp_path, caplog):
        ckpt = self._run(tmp_path)
        with caplog.at_level(logging.WARNING):
            trainer = self._resume(tmp_path, ckpt, router_aux_loss_coef=0.2)
        assert trainer._router_aux_loss_coef == pytest.approx(0.2)
        assert any("router_aux_loss_coef" in r.getMessage() for r in caplog.records)
        assert len(_rows(trainer)) == 4


def _stub_tokenizer():
    return types.SimpleNamespace(
        pad_token_id=0,
        pad_token="<pad>",
        eos_token="</s>",
        save_pretrained=lambda *a, **k: None,
    )


def _sft_dataset():
    datasets = pytest.importorskip("datasets")
    g = torch.Generator().manual_seed(2)
    rows = []
    for _ in range(16):
        n = int(torch.randint(5, SEQ + 1, (1,), generator=g))
        rows.append({"input_ids": torch.randint(3, VOCAB, (n,), generator=g).tolist()})
    return datasets.Dataset.from_list(rows)


def _sft_args(output_dir, **overrides):
    kwargs = {
        **_COMMON,
        "output_dir": str(output_dir),
        "max_steps": 2,
        "max_length": SEQ,
        "router_aux_kwargs": {"ratio": 0.5},
    }
    kwargs.update(overrides)
    return SFTConfig(**kwargs)


class TestSFTTrainer:
    @pytest.mark.parametrize(
        ("loss_type", "log_completion_metrics"),
        [
            ("nll", True),
            ("nll", False),
            ("chunked_nll", False),
            ("dft", True),
            ("dft", False),
        ],
    )
    def test_trains_on_every_loss_path(
        self, tmp_path, loss_type, log_completion_metrics
    ):
        model = _tiny_mellum()
        trainer = SFTTrainer(
            model=model,
            args=_sft_args(
                tmp_path,
                loss_type=loss_type,
                log_completion_metrics=log_completion_metrics,
            ),
            train_dataset=_sft_dataset(),
            processing_class=_stub_tokenizer(),
        )
        assert trainer._router_aux_enabled
        assert trainer._router_aux_factory_kwargs["max_tokens"] == SEQ
        assert _text_config(model).output_router_logits is False
        out = trainer.train()
        assert out.global_step == 2
        assert torch.isfinite(torch.tensor(out.training_loss))
        rows = _rows(trainer)
        assert len(rows) == 2
        assert all(row["router_aux_noise_std"] > 0 for row in rows)
        # TRL's raw metric rides along with the same name.
        assert all(row["aux_loss"] > 0 for row in rows)
        assert MoeAux.__name__ in repr(trainer._accountant.process)
        # The completion telemetry rides next to the release when logits exist.
        eager = log_completion_metrics and loss_type != "chunked_nll"
        assert all(("entropy" in row) == eager for row in rows)


def _dpo_pairs():
    datasets = pytest.importorskip("datasets")
    g = torch.Generator().manual_seed(3)
    prompt_len = 3
    rows = []
    for _ in range(8):
        prompt = torch.randint(3, VOCAB, (prompt_len,), generator=g).tolist()
        n_c = int(torch.randint(2, SEQ - prompt_len + 1, (1,), generator=g))
        n_r = int(torch.randint(2, SEQ - prompt_len + 1, (1,), generator=g))
        rows.append(
            {
                "chosen_input_ids": prompt
                + torch.randint(3, VOCAB, (n_c,), generator=g).tolist(),
                "rejected_input_ids": prompt
                + torch.randint(3, VOCAB, (n_r,), generator=g).tolist(),
                "chosen_completion_mask": [0] * prompt_len + [1] * n_c,
                "rejected_completion_mask": [0] * prompt_len + [1] * n_r,
            }
        )
    return datasets.Dataset.from_list(rows)


def _dpo_args(output_dir, **overrides):
    kwargs = {
        **_COMMON,
        "output_dir": str(output_dir),
        "max_steps": 2,
        "max_length": SEQ,
        "router_aux_kwargs": {"ratio": 0.5},
    }
    kwargs.update(overrides)
    return DPOConfig(**kwargs)


class TestDPOTrainer:
    def test_pair_token_bound_from_max_length(self, tmp_path):
        """Both sides carry the prompt, so the pair bound is twice ``max_length``."""
        assert _dpo_args(tmp_path).router_aux_kwargs == {
            "ratio": 0.5,
            "max_tokens": 2 * SEQ,
        }
        explicit = _dpo_args(tmp_path, router_aux_kwargs={"max_tokens": 5})
        assert explicit.router_aux_kwargs == {"max_tokens": 5}
        off = _dpo_args(tmp_path, router_aux_loss_coef=0.0)
        assert off.router_aux_kwargs == {"ratio": 0.5}

    @pytest.mark.parametrize(
        ("log_completion_metrics", "use_weighting"),
        [(True, False), (False, False), (False, True)],
    )
    def test_trains_on_every_loss_path(
        self, tmp_path, log_completion_metrics, use_weighting
    ):
        model = _tiny_mellum()
        trainer = DPOTrainer(
            model=model,
            ref_model=_tiny_mellum(),
            args=_dpo_args(
                tmp_path,
                log_completion_metrics=log_completion_metrics,
                use_weighting=use_weighting,
            ),
            train_dataset=_dpo_pairs(),
            processing_class=_stub_tokenizer(),
        )
        assert trainer._router_aux_enabled
        assert trainer._router_aux_factory_kwargs["max_tokens"] == 2 * SEQ
        assert _text_config(model).output_router_logits is False
        # WPO consumes logits, so it keeps the eager path; the other two rows
        # are eligible for the fused log-prob path, which the metrics gate
        # switches off per step when the completion telemetry is on.
        assert trainer._use_fused_logp is (not use_weighting)
        out = trainer.train()
        assert out.global_step == 2
        assert torch.isfinite(torch.tensor(out.training_loss))
        rows = _rows(trainer)
        assert len(rows) == 2
        assert all(row["router_aux_noise_std"] > 0 for row in rows)
        # TRL's raw metric rides along with the same name.
        assert all(row["aux_loss"] > 0 for row in rows)
        # The reward telemetry rides next to the release on every path.
        assert all(any(key.startswith("rewards/") for key in row) for row in rows)
        assert MoeAux.__name__ in repr(trainer._accountant.process)

    def test_seam_joins_the_pair_along_the_token_axis(self, tmp_path):
        """The clipper sees one example: both sides' router logits and masks."""
        trainer = DPOTrainer(
            model=_tiny_mellum(),
            args=_dpo_args(tmp_path, loss_type="simpo"),
            train_dataset=_dpo_pairs(),
            processing_class=_stub_tokenizer(),
        )
        batch = next(iter(trainer.get_train_dataloader()))
        tensors = {k: v for k, v in batch.items() if torch.is_tensor(v)}
        fmodel, params = make_functional(trainer._model)

        def seam(example):
            return trainer.compute_per_example_loss_and_router_logits(
                fmodel, params, example
            )

        def side(example, name):
            out = fmodel(
                params,
                input_ids=example[f"{name}_input_ids"],
                attention_mask=example[f"{name}_attention_mask"],
                output_router_logits=True,
                router_aux_loss=False,
            )
            assert out.aux_loss is None
            return out.router_logits

        loss, router_logits, mask, aux = torch.func.vmap(seam)(tensors)
        chosen = torch.func.vmap(lambda ex: side(ex, "chosen"))(tensors)
        rejected = torch.func.vmap(lambda ex: side(ex, "rejected"))(tensors)
        assert torch.isfinite(loss).all()
        assert len(router_logits) == L
        for layer in range(L):
            torch.testing.assert_close(
                router_logits[layer],
                torch.cat([chosen[layer], rejected[layer]], dim=1),
            )
        assert torch.equal(
            mask,
            torch.cat(
                [tensors["chosen_attention_mask"], tensors["rejected_attention_mask"]],
                dim=1,
            ),
        )
        assert aux
        assert all(torch.is_tensor(value) for value in aux.values())


def test_state_type_is_registered_for_sync():
    from opaque.api.engine.distributed._state import _SYNC_REGISTRY

    assert MoeClipState in _SYNC_REGISTRY
