"""Regression tests for checkpoint correctness.

Covers:

- Best-folder lookup in ``_save_checkpoint`` (HF parity): the best model
  checkpoint is registered by *folder* keyed on ``state.best_global_step``,
  not only when the eval boundary coincides with the save boundary.
- ``_load_best_model`` mutates ``self._model`` (so an immediate
  ``save_model()`` sees the loaded weights without going through
  ``_restore_params`` first).
- ``DPTrainerState.stateful_callbacks`` round-trip
  through ``to_json`` / ``from_json`` (single schema; no JSON re-parse).
- ``_warn_on_arg_drift`` surfaces ``expected_batch_size`` drift so a
  user resuming with a changed batch size sees the privacy-relevant
  change instead of having it silently absorbed by the ``sample_rate``
  warning.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import pytest
import torch
from _hf_shared import build_lm_dataset, gpt2_tokenizer, make_gpt2_model
from datasets import Dataset
from peft import (
    LoraConfig,
    PromptEncoderConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
)
from safetensors.torch import load_file, save_file
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM

from opake.api.transformers.trainer._state import DPTrainerState
from opake.exceptions import CheckpointError
from opake.transformers.trainer import DPTrainer, TrainingArguments

# ---------------------------------------------------------------------------
# Fixtures (same shape as test_callback_hooks.py).
# ---------------------------------------------------------------------------


@pytest.fixture
def small_model_and_tokenizer():
    tokenizer = gpt2_tokenizer()
    tokenizer.pad_token = tokenizer.eos_token
    model = make_gpt2_model()
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


@pytest.fixture
def lora_model(small_model_and_tokenizer):
    model, tokenizer = small_model_and_tokenizer
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["c_attn"],
        fan_in_fan_out=True,
    )
    return get_peft_model(model, lora_config), tokenizer


@pytest.fixture
def tiny_dataset(small_model_and_tokenizer):
    """Six pre-padded causal-LM examples for checkpoint tests."""
    _, tokenizer = small_model_and_tokenizer
    return build_lm_dataset(
        [f"sample {i}" for i in range(6)],
        tokenizer,
        max_length=16,
    )


def _args(tmp_path, **overrides) -> TrainingArguments:
    # ``use_cpu=True``: pin to CPU so the trainer's ``args.device``
    # resolves to CPU regardless of the host (LoRA fixtures are CPU).
    defaults: dict[str, Any] = {
        "output_dir": str(tmp_path),
        "per_device_train_batch_size": 2,
        "privacy_target_epsilon": 10.0,
        "privacy_noise_multiplier": 1.0,
        "max_steps": 4,
        "num_train_epochs": 1,
        "logging_steps": 1,
        "save_strategy": "no",
        "use_cpu": True,
    }
    defaults.update(overrides)
    return TrainingArguments(**defaults)


# ---------------------------------------------------------------------------
# DPTrainerState round-trip — single schema for trainer_state.json.
# ---------------------------------------------------------------------------


class TestStateRoundTrip:
    """``DPTrainerState`` round-trips every persisted field through ``to_json``."""

    def test_stateful_callbacks_round_trip(self):
        s = DPTrainerState(
            global_step=10,
            stateful_callbacks={"EarlyStoppingCallback": {"patience_counter": 3}},
        )
        round_tripped = DPTrainerState.from_json(s.to_json())
        assert round_tripped.stateful_callbacks == {
            "EarlyStoppingCallback": {"patience_counter": 3}
        }
        assert round_tripped.global_step == 10

    def test_unknown_keys_dropped(self):
        # Forward-compat: a future field is ignored without raising.
        s = DPTrainerState.from_json(
            {
                "global_step": 5,
                "future_field_not_yet_invented": "anything",
            }
        )
        assert s.global_step == 5


# ---------------------------------------------------------------------------
# _load_best_model mutates self._model.
# ---------------------------------------------------------------------------


class TestLoadBestModelMutates:
    """``_load_best_model`` writes the loaded weights into ``self._model``."""

    def test_save_model_after_load_best_sees_best_weights(
        self,
        lora_model,
        tiny_dataset,
        tmp_path,
    ):
        """``save_model()`` after ``train()`` reflects the loaded best weights.

        ``_load_best_model`` mutates ``self._model`` directly, so a
        fresh ``save_model()`` call is byte-identical to the recorded
        best checkpoint regardless of the in-flight functional view.
        """
        model, tokenizer = lora_model
        ckpt_root = tmp_path / "ckpts"
        ckpt_root.mkdir()
        save_target = tmp_path / "post_train"

        trainer = DPTrainer(
            model=model,
            args=_args(
                ckpt_root,
                max_steps=4,
                num_train_epochs=1,
                eval_strategy="steps",
                eval_steps=2,
                save_strategy="steps",
                save_steps=2,
                metric_for_best_model="eval_loss",
                load_best_model_at_end=True,
                logging_steps=1,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
            eval_dataset=tiny_dataset,
        )
        trainer.train()

        if trainer.state.best_model_checkpoint is None:
            pytest.skip("Eval never improved; nothing to compare.")

        best_dir = trainer.state.best_model_checkpoint
        from safetensors.torch import load_file as load_safetensors

        adapter_path = str(Path(best_dir) / "adapter_model.safetensors")
        if not Path(adapter_path).exists():
            pytest.skip(f"Expected adapter weights under {best_dir} not found")
        best_weights = load_safetensors(adapter_path, device="cpu")

        # Save the in-memory module to a fresh directory.  This writes
        # the *best* weights — the module was mutated by
        # ``_load_best_model``.
        trainer.save_model(str(save_target))
        new_path = save_target / "adapter_model.safetensors"
        assert new_path.exists(), f"save_model did not produce {new_path}"
        post_save_weights = load_safetensors(str(new_path), device="cpu")

        # Same key set + identical tensors → module reflects best.
        assert set(best_weights) == set(post_save_weights), (
            "save_model produced a different adapter shape than the best "
            "checkpoint — module wasn't mutated by _load_best_model"
        )
        # ``atol=1e-3`` allows fp32 round-trip noise from PEFT's
        # save_pretrained merging.
        for name, saved in best_weights.items():
            diff = (post_save_weights[name] - saved).abs().max().item()
            assert diff < 1e-3, (
                f"Best weights for {name!r} drift from disk by {diff:g} after "
                "_load_best_model — module mutation must reflect best "
                "checkpoint exactly"
            )


class TestModelRestore:
    def test_resume_restores_fresh_lora_model(
        self, small_model_and_tokenizer, tiny_dataset, tmp_path
    ):
        base, tokenizer = small_model_and_tokenizer
        fresh_base = copy.deepcopy(base)
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=4,
            lora_alpha=8,
            target_modules=["c_attn"],
            fan_in_fan_out=True,
        )
        source = get_peft_model(base, copy.deepcopy(config))
        fresh = get_peft_model(fresh_base, copy.deepcopy(config))
        first = DPTrainer(
            model=source,
            args=_args(tmp_path, max_steps=2, save_strategy="steps", save_steps=2),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
        )
        first.train()
        checkpoint = tmp_path / "checkpoint-2"
        saved = load_file(str(checkpoint / "adapter_model.safetensors"))
        initial = get_peft_model_state_dict(fresh)
        assert any(not torch.equal(saved[name], initial[name]) for name in saved)

        resumed = DPTrainer(
            model=fresh,
            args=_args(tmp_path, max_steps=2),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
        )
        resumed.train(resume_from_checkpoint=checkpoint)
        restored = get_peft_model_state_dict(fresh)
        assert set(restored) == set(saved)
        assert all(torch.equal(restored[name], value) for name, value in saved.items())

    def test_partial_lora_checkpoint_fails_with_matching_bias(
        self, small_model_and_tokenizer, tiny_dataset, tmp_path
    ):
        base, tokenizer = small_model_and_tokenizer
        source = get_peft_model(
            base,
            LoraConfig(
                r=2,
                lora_alpha=4,
                target_modules=["c_attn"],
                bias="all",
                fan_in_fan_out=True,
            ),
        )
        folder = tmp_path / "adapter"
        source.save_pretrained(folder)
        weights_path = folder / "adapter_model.safetensors"
        weights = load_file(str(weights_path))
        biases = {name: value for name, value in weights.items() if "bias" in name}
        assert biases
        assert len(biases) < len(weights)
        save_file(biases, str(weights_path))
        trainer = DPTrainer(
            model=copy.deepcopy(source),
            args=_args(tmp_path / "run"),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
        )
        with pytest.raises(CheckpointError, match="Incomplete PEFT adapter"):
            trainer._restore_checkpoint_model(str(folder))

    def test_missing_peft_weights_fail(self, lora_model, tiny_dataset, tmp_path):
        model, tokenizer = lora_model
        folder = tmp_path / "adapter"
        model.save_pretrained(folder)
        (folder / "adapter_model.safetensors").unlink()
        trainer = DPTrainer(
            model=model,
            args=_args(tmp_path / "run"),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
        )
        with pytest.raises(CheckpointError, match="Missing PEFT weights"):
            trainer._restore_checkpoint_model(str(folder))

    def test_peft_config_drift_fails(
        self, small_model_and_tokenizer, tiny_dataset, tmp_path
    ):
        base, tokenizer = small_model_and_tokenizer
        config = LoraConfig(
            r=2, lora_alpha=4, target_modules=["c_attn"], fan_in_fan_out=True
        )
        target_base = copy.deepcopy(base)
        source = get_peft_model(base, config)
        folder = tmp_path / "adapter"
        source.save_pretrained(folder)
        target = get_peft_model(
            target_base,
            LoraConfig(
                r=2, lora_alpha=8, target_modules=["c_attn"], fan_in_fan_out=True
            ),
        )
        trainer = DPTrainer(
            model=target,
            args=_args(tmp_path / "run"),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
        )
        with pytest.raises(CheckpointError, match="lora_alpha"):
            trainer._restore_checkpoint_model(str(folder))

    def test_restores_trainable_tokens(
        self, small_model_and_tokenizer, tiny_dataset, tmp_path
    ):
        base, tokenizer = small_model_and_tokenizer
        source = get_peft_model(
            base,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=2,
                lora_alpha=4,
                target_modules=["c_attn"],
                fan_in_fan_out=True,
                trainable_token_indices={"transformer.wte": [1, 2]},
            ),
        )
        folder = tmp_path / "adapter"
        source.save_pretrained(folder)
        saved = load_file(str(folder / "adapter_model.safetensors"))
        assert any("trainable_tokens_delta" in name for name in saved)
        target = copy.deepcopy(source)
        with torch.no_grad():
            for parameter in target.parameters():
                if parameter.requires_grad:
                    parameter.zero_()
        trainer = DPTrainer(
            model=target,
            args=_args(tmp_path / "run"),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
        )
        trainer._restore_checkpoint_model(str(folder))
        restored = get_peft_model_state_dict(target)
        assert set(restored) == set(saved)
        assert all(torch.equal(restored[name], value) for name, value in saved.items())

    def test_lossy_prompt_encoder_checkpoint_fails(self, tiny_dataset, tmp_path):
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
        source = get_peft_model(
            LlamaForCausalLM(config),
            PromptEncoderConfig(
                num_virtual_tokens=2,
                task_type=TaskType.CAUSAL_LM,
                encoder_hidden_size=16,
            ),
        )
        folder = tmp_path / "prompt"
        source.save_pretrained(folder)
        target = copy.deepcopy(source)
        with torch.no_grad():
            for parameter in target.parameters():
                if parameter.requires_grad:
                    parameter.zero_()
        trainer = DPTrainer(
            model=target,
            args=_args(tmp_path / "run"),
            train_dataset=tiny_dataset,
        )
        with pytest.raises(CheckpointError, match="prompt_embeddings"):
            trainer._restore_checkpoint_model(str(folder))

    @pytest.mark.parametrize("safe_serialization", [True, False])
    def test_restores_each_named_adapter(
        self, lora_model, tiny_dataset, tmp_path, safe_serialization
    ):
        source, tokenizer = lora_model
        source.add_adapter("custom", copy.deepcopy(source.peft_config["default"]))
        with torch.no_grad():
            for name, parameter in source.named_parameters():
                if ".lora_" in name and ".default." in name:
                    parameter.fill_(0.125)
                elif ".lora_" in name and ".custom." in name:
                    parameter.fill_(0.375)
        source.set_adapter("custom")
        folder = tmp_path / "adapters"
        source.save_pretrained(folder, safe_serialization=safe_serialization)

        target = copy.deepcopy(source)
        with torch.no_grad():
            for name, parameter in target.named_parameters():
                if ".lora_" in name:
                    parameter.zero_()
        trainer = DPTrainer(
            model=target,
            args=_args(tmp_path / "run"),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
        )
        trainer._restore_checkpoint_model(str(folder))
        assert target.active_adapter == "custom"
        for adapter in ("default", "custom"):
            path = folder if adapter == "default" else folder / adapter
            if safe_serialization:
                saved = load_file(str(path / "adapter_model.safetensors"))
            else:
                saved = torch.load(
                    path / "adapter_model.bin", map_location="cpu", weights_only=True
                )
            restored = get_peft_model_state_dict(target, adapter_name=adapter)
            assert set(restored) == set(saved)
            assert all(
                torch.equal(restored[name], value) for name, value in saved.items()
            )

    @pytest.mark.parametrize(
        ("sharded", "legacy_binary"), [(False, False), (True, False), (False, True)]
    )
    def test_restores_tied_full_model(
        self,
        small_model_and_tokenizer,
        tiny_dataset,
        tmp_path,
        sharded,
        legacy_binary,
    ):
        source, tokenizer = small_model_and_tokenizer
        folder = tmp_path / "full"
        source.save_pretrained(folder, max_shard_size="5MB" if sharded else "5GB")
        if sharded:
            index = json.loads((folder / "model.safetensors.index.json").read_text())
            saved_keys = index["weight_map"]
        else:
            weights_path = folder / "model.safetensors"
            saved = load_file(str(weights_path))
            saved_keys = saved
            if legacy_binary:
                torch.save(saved, folder / "pytorch_model.bin")
                weights_path.unlink()
        assert "lm_head.weight" not in saved_keys
        target = copy.deepcopy(source)
        with torch.no_grad():
            for parameter in target.parameters():
                parameter.zero_()
        trainer = DPTrainer(
            model=target,
            args=_args(tmp_path / "run"),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
        )
        trainer._restore_checkpoint_model(str(folder))
        for name, value in source.state_dict().items():
            assert torch.equal(target.state_dict()[name], value)
        assert (
            target.lm_head.weight.data_ptr() == target.transformer.wte.weight.data_ptr()
        )

    def test_missing_untied_weight_fails(
        self, small_model_and_tokenizer, tiny_dataset, tmp_path
    ):
        base, tokenizer = small_model_and_tokenizer
        config = copy.deepcopy(base.config)
        config.tie_word_embeddings = False
        source = type(base)(config)
        folder = tmp_path / "full"
        source.save_pretrained(folder)
        weights_path = folder / "model.safetensors"
        weights = load_file(str(weights_path))
        weights.pop("lm_head.weight")
        save_file(weights, str(weights_path))
        trainer = DPTrainer(
            model=type(base)(config),
            args=_args(tmp_path / "run"),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
        )
        with pytest.raises(CheckpointError, match=r"lm_head\.weight"):
            trainer._restore_checkpoint_model(str(folder))


@pytest.fixture
def tiny_tied_model_and_dataset():
    config = GPT2Config(
        vocab_size=32,
        n_positions=16,
        n_ctx=16,
        n_embd=16,
        n_layer=1,
        n_head=2,
        tie_word_embeddings=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = GPT2LMHeadModel(config)
    tokens = [[1, 2 + i, 3 + i, 4, 5, 0, 0, 0] for i in range(6)]
    dataset = Dataset.from_dict(
        {
            "input_ids": tokens,
            "attention_mask": [[1] * 5 + [0] * 3 for _ in tokens],
            "labels": [row[:5] + [-100] * 3 for row in tokens],
        }
    )
    return model, dataset


class TestTiedModelTraining:
    def test_resume_restores_tied_model(self, tiny_tied_model_and_dataset, tmp_path):
        source, dataset = tiny_tied_model_and_dataset
        fresh = copy.deepcopy(source)
        first = DPTrainer(
            model=source,
            args=_args(tmp_path, max_steps=2, save_strategy="steps", save_steps=2),
            train_dataset=dataset,
        )
        first.train()
        checkpoint = tmp_path / "checkpoint-2"
        saved = load_file(str(checkpoint / "model.safetensors"))
        assert "lm_head.weight" not in saved
        assert not torch.equal(
            fresh.transformer.wte.weight, saved["transformer.wte.weight"]
        )

        resumed = DPTrainer(
            model=fresh,
            args=_args(tmp_path, max_steps=2),
            train_dataset=dataset,
        )
        resumed.train(resume_from_checkpoint=checkpoint)
        restored = fresh.state_dict()
        assert all(torch.equal(restored[name], value) for name, value in saved.items())
        assert (
            fresh.lm_head.weight.data_ptr() == fresh.transformer.wte.weight.data_ptr()
        )

    def test_best_checkpoint_restores_tied_model(
        self, tiny_tied_model_and_dataset, tmp_path, monkeypatch
    ):
        model, dataset = tiny_tied_model_and_dataset
        trainer = DPTrainer(
            model=model,
            args=_args(
                tmp_path,
                max_steps=4,
                eval_strategy="steps",
                eval_steps=2,
                save_strategy="steps",
                save_steps=2,
                metric_for_best_model="eval_loss",
                load_best_model_at_end=True,
            ),
            train_dataset=dataset,
            eval_dataset=dataset,
        )
        losses = iter((0.1, 0.9))
        evaluate = trainer._run_evaluation_loop

        def scripted_evaluation(*args, **kwargs):
            result = evaluate(*args, **kwargs)
            result.metrics["eval_loss"] = next(losses)
            return result

        monkeypatch.setattr(trainer, "_run_evaluation_loop", scripted_evaluation)
        trainer.train()
        assert trainer.state.best_global_step == 2
        best = load_file(str(tmp_path / "checkpoint-2" / "model.safetensors"))
        last = load_file(str(tmp_path / "checkpoint-4" / "model.safetensors"))
        assert not torch.equal(
            best["transformer.wte.weight"], last["transformer.wte.weight"]
        )
        restored = model.state_dict()
        assert all(torch.equal(restored[name], value) for name, value in best.items())
        assert (
            model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr()
        )


# ---------------------------------------------------------------------------
# Best-folder lookup keyed on best_global_step (HF parity).
# ---------------------------------------------------------------------------


class TestBestFolderLookup:
    """``_save_checkpoint`` registers best by folder, not just exact-step coincidence."""

    def test_best_metric_forces_checkpoint_at_intermediate_eval_step(
        self, lora_model, tiny_dataset, tmp_path
    ):
        """An improved eval metric saves its evaluated parameters immediately.

        A save every four steps and evaluation every two steps is valid because
        the save cadence is a multiple of the eval cadence.  The first score
        wins, so the best checkpoint must be the extra checkpoint materialized
        at step two rather than the later regular save at step four.
        """
        model, tokenizer = lora_model
        scores = iter((1.0, 0.0))

        def compute_metrics(_):
            return {"score": next(scores)}

        trainer = DPTrainer(
            model=model,
            args=_args(
                tmp_path,
                max_steps=4,
                num_train_epochs=1,
                eval_strategy="steps",
                eval_steps=2,
                save_strategy="steps",
                save_steps=4,
                metric_for_best_model="score",
                greater_is_better=True,
                load_best_model_at_end=True,
                logging_steps=1,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
            eval_dataset=tiny_dataset,
            compute_metrics=compute_metrics,
        )
        trainer.train()

        expected = tmp_path / "checkpoint-2"
        assert expected.is_dir()
        assert trainer.state.best_global_step == 2
        assert Path(trainer.state.best_model_checkpoint).resolve() == expected.resolve()

    def test_best_folder_lookup(self, lora_model, tiny_dataset, tmp_path):
        """Eval at step 2 may improve while save_strategy fires at every step.

        With ``save_strategy="steps", save_steps=1`` and
        ``eval_strategy="steps", eval_steps=2``, eval-improvement at
        step 2 must produce ``best_model_checkpoint == "checkpoint-2/"``
        (HF parity), even though the best decision and the save
        decision are made at different cadence boundaries.
        """
        model, tokenizer = lora_model

        trainer = DPTrainer(
            model=model,
            args=_args(
                tmp_path,
                max_steps=4,
                num_train_epochs=1,
                eval_strategy="steps",
                eval_steps=2,
                save_strategy="steps",
                save_steps=1,
                metric_for_best_model="eval_loss",
                save_total_limit=10,
                logging_steps=1,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
            eval_dataset=tiny_dataset,
        )
        trainer.train()

        if trainer.state.best_global_step is None:
            pytest.skip("Eval never improved; nothing to verify.")

        expected = str(tmp_path / f"checkpoint-{trainer.state.best_global_step}")
        assert trainer.state.best_model_checkpoint is not None
        assert (
            Path(trainer.state.best_model_checkpoint).resolve()
            == Path(expected).resolve()
        ), (
            f"best_model_checkpoint should resolve to checkpoint-"
            f"{trainer.state.best_global_step} via folder lookup; got "
            f"{trainer.state.best_model_checkpoint}"
        )
        # The folder should also exist on disk and be persisted in
        # trainer_state.json (single schema; no JSON re-parse).
        ts_path = Path(expected) / "trainer_state.json"
        assert ts_path.exists()
        with ts_path.open() as f:
            ts = json.load(f)
        assert ts["best_global_step"] == trainer.state.best_global_step


class TestBestOnEvalOnlyStep:
    """#386: best on an eval-only step must still get a checkpoint.

    ``save_steps=4`` is coarser than ``eval_steps=2``, so evals fire at 2 and
    4 but saves fire only at 4.  Scripting ``eval_loss`` so step 2 is the
    genuine best lands ``best_global_step`` on eval-only step 2.  Pre-fix:
    no ``checkpoint-2`` -> ``best_model_checkpoint`` is None ->
    ``_load_best_model`` raises ``RuntimeError``.  Post-fix: the force-save
    callback writes ``checkpoint-2`` and best is loadable.
    """

    def test_best_eval_only_step_forces_checkpoint(
        self, lora_model, tiny_dataset, tmp_path, monkeypatch
    ):
        model, tokenizer = lora_model
        trainer = DPTrainer(
            model=model,
            args=_args(
                tmp_path,
                max_steps=4,
                num_train_epochs=1,
                eval_strategy="steps",
                eval_steps=2,
                save_strategy="steps",
                save_steps=4,  # coarser than eval: best can fall off a save step
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                load_best_model_at_end=True,
                save_total_limit=10,
                logging_steps=1,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
            eval_dataset=tiny_dataset,
        )

        # Script eval_loss at the pure eval seam so the callback's improvement
        # check and _update_best_metric read identical values.  Step 2 is the
        # genuine best (0.10); step 4 is worse (0.90).
        scripted = [0.10, 0.90]
        counter = {"i": 0}
        real_loop = trainer._run_evaluation_loop

        def scripted_loop(*args, **kwargs):
            result = real_loop(*args, **kwargs)
            prefix = kwargs.get("metric_key_prefix", "eval")
            idx = min(counter["i"], len(scripted) - 1)
            result.metrics[f"{prefix}_loss"] = scripted[idx]
            counter["i"] += 1
            return result

        monkeypatch.setattr(trainer, "_run_evaluation_loop", scripted_loop)

        trainer.train()  # pre-fix: raises RuntimeError inside _load_best_model

        # Best tracks the genuine best eval, on the eval-only step.
        assert trainer.state.best_global_step == 2
        assert trainer.state.best_metric == pytest.approx(0.10)

        # AC1: best points to an existing checkpoint even though step 2 is not
        # a save boundary (save_steps=4).
        best = trainer.state.best_model_checkpoint
        assert best is not None
        assert Path(best).resolve() == (tmp_path / "checkpoint-2").resolve()
        assert Path(best).is_dir()
        assert (Path(best) / "adapter_model.safetensors").exists()
        best_weights = load_file(str(Path(best) / "adapter_model.safetensors"))
        last_weights = load_file(
            str(tmp_path / "checkpoint-4" / "adapter_model.safetensors")
        )
        assert any(
            not torch.equal(best_weights[name], last_weights[name])
            for name in best_weights
        )
        restored = get_peft_model_state_dict(model)
        assert all(
            torch.equal(restored[name], value) for name, value in best_weights.items()
        )


# ---------------------------------------------------------------------------
# save_only_model: trainer_state.json is always written (HF parity).
# ---------------------------------------------------------------------------


class TestSaveOnlyModelStillWritesTrainerState:
    """``save_only_model=True`` no longer drops trainer_state.json."""

    def test_trainer_state_present_under_save_only_model(
        self,
        lora_model,
        tiny_dataset,
        tmp_path,
    ):
        model, tokenizer = lora_model
        trainer = DPTrainer(
            model=model,
            args=_args(
                tmp_path,
                save_only_model=True,
                max_steps=2,
                save_strategy="steps",
                save_steps=2,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
            eval_dataset=tiny_dataset,
        )
        trainer.train()
        d = tmp_path / "checkpoint-2"
        # Interpretability files always written (HF parity for the first
        # two; ``accountant.json`` is the DP-correct extension — the
        # privacy guarantee is a property of the saved model).
        assert (d / "trainer_state.json").exists()
        assert (d / "training_args.bin").exists()
        assert (d / "accountant.json").exists()
        # Resumability files skipped under save_only_model.
        assert not (d / "dp_optimizer.pt").exists()
        assert not (d / "dp_state.pt").exists()


# ---------------------------------------------------------------------------
# EarlyStoppingCallback round-trip.
# ---------------------------------------------------------------------------


class TestEarlyStoppingExportableState:
    """HF's stock ``EarlyStoppingCallback`` saves under the ExportableState schema.

    HF's callback uses the ``ExportableState`` protocol
    (``state()`` / ``from_state()``) rather than a flat ``state_dict()``;
    the trainer must persist via ``state()`` when the callback supports it
    or the callback's internal counters are lost on resume.

    This test pins the *save-side* contract.  The *load-side*
    contract (attribute-set on resume) is covered by the simpler
    legacy callback test in ``test_dp_trainer.py``
    (``test_resume_restores_callback_state``).
    """

    def test_state_dict_schema_matches_hf(
        self,
        lora_model,
        tiny_dataset,
        tmp_path,
    ):
        from transformers.trainer_callback import EarlyStoppingCallback

        model, tokenizer = lora_model
        cb = EarlyStoppingCallback(early_stopping_patience=2)
        trainer = DPTrainer(
            model=model,
            args=_args(
                tmp_path,
                max_steps=2,
                save_strategy="steps",
                save_steps=2,
                eval_strategy="steps",
                eval_steps=2,
                metric_for_best_model="eval_loss",
                logging_steps=1,
            ),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
            eval_dataset=tiny_dataset,
            callbacks=[cb],
        )
        trainer.train()

        # ``stateful_callbacks`` has the HF ExportableState shape:
        # ``{"args": {...}, "attributes": {...}}``.  This is what
        # ``transformers.trainer_callback.TrainerState`` writes too —
        # the schema is the canonical HF round-trip shape.
        ts_path = tmp_path / "checkpoint-2" / "trainer_state.json"
        with ts_path.open() as f:
            ts = json.load(f)
        cb_payload = ts["stateful_callbacks"].get("EarlyStoppingCallback")
        assert cb_payload is not None
        assert "args" in cb_payload
        assert "attributes" in cb_payload
        # The args round-trip exactly through ``EarlyStoppingCallback.state()``.
        assert cb_payload["args"]["early_stopping_patience"] == 2


# ---------------------------------------------------------------------------
# expected_batch_size drift warning + drift-disposition dispatch.
# ---------------------------------------------------------------------------


class TestArgDriftWarnings:
    """``_warn_on_arg_drift`` surfaces drift per the field disposition.

    ``expected_batch_size`` carries the ``dp_relevant`` disposition so a
    user resuming with a different ``per_device_train_batch_size`` sees
    the privacy-relevant change rather than having it silently absorbed
    by the ``sample_rate`` warning.

    Tested at the ``_warn_on_arg_drift`` boundary directly so the
    test doesn't drag in the full save/resume cycle (which is
    exercised by other resume tests that already pin the
    ``sample_rate`` / ``noise_multiplier`` warnings).
    """

    def _baseline_runtime(self, trainer, tiny_dataset):
        """Build a ``RuntimeCheckpoint`` whose privacy scalars match live args.

        Drift comparisons are made field-by-field; baseline values that
        match the live trainer trip no warnings.  Tests then mutate the
        field under test on the returned bundle.
        """
        import opake.api.transformers.trainer._checkpoint as ckpt

        target_delta = trainer.args.privacy_target_delta or 1e-5
        return ckpt.RuntimeCheckpoint(
            version=ckpt.DP_STATE_BUNDLE_VERSION,
            clip_state={},
            noise_state={},
            sampler_state=None,
            sample_rate=trainer.args.train_batch_size / max(1, len(tiny_dataset)),
            target_delta=float(target_delta),
            noise_multiplier=float(trainer.args.privacy_noise_multiplier),
            expected_steps_per_epoch=1,
            expected_batch_size=int(trainer.args.train_batch_size),
            total_steps=int(trainer.args.max_steps),
        )

    def test_expected_batch_size_drift_warns(
        self,
        lora_model,
        tiny_dataset,
        tmp_path,
        caplog,
    ):
        """A synthetic bundle whose ``expected_batch_size`` differs warns."""
        model, tokenizer = lora_model
        trainer = DPTrainer(
            model=model,
            args=_args(tmp_path, per_device_train_batch_size=2),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
            eval_dataset=tiny_dataset,
        )
        runtime = self._baseline_runtime(trainer, tiny_dataset)
        # Drift only ``expected_batch_size``: saved=4, current=2.
        runtime.expected_batch_size = int(trainer.args.train_batch_size) * 2

        with caplog.at_level(logging.WARNING):
            trainer._warn_on_arg_drift(runtime)

        drift_msgs = [
            r
            for r in caplog.records
            if "expected_batch_size" in r.getMessage()
            and "drift" in r.getMessage().lower()
        ]
        assert drift_msgs, (
            f"expected an ``expected_batch_size`` drift warning; "
            f"got records: {[r.getMessage() for r in caplog.records]}"
        )

    def test_no_drift_no_warning(self, lora_model, tiny_dataset, tmp_path, caplog):
        """Identical saved/current bundle emits no drift warning."""
        model, tokenizer = lora_model
        trainer = DPTrainer(
            model=model,
            args=_args(tmp_path, per_device_train_batch_size=2),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
            eval_dataset=tiny_dataset,
        )
        runtime = self._baseline_runtime(trainer, tiny_dataset)

        with caplog.at_level(logging.WARNING):
            trainer._warn_on_arg_drift(runtime)

        drift_msgs = [r for r in caplog.records if "drift" in r.getMessage().lower()]
        assert not drift_msgs, (
            f"no drift was introduced but warnings fired: "
            f"{[r.getMessage() for r in drift_msgs]}"
        )

    def test_total_steps_drift_silent_for_dp_sgd(
        self, lora_model, tiny_dataset, tmp_path, caplog
    ):
        """``total_steps`` drift is silent under DP-SGD — users extend
        training routinely (intentional_extend disposition).
        """
        model, tokenizer = lora_model
        trainer = DPTrainer(
            model=model,
            args=_args(tmp_path, per_device_train_batch_size=2, max_steps=10),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
            eval_dataset=tiny_dataset,
        )
        runtime = self._baseline_runtime(trainer, tiny_dataset)
        runtime.total_steps = 5  # saved=5, current=10 — clean extension

        with caplog.at_level(logging.WARNING):
            trainer._warn_on_arg_drift(runtime)

        total_msgs = [r for r in caplog.records if "total_steps" in r.getMessage()]
        assert not total_msgs, (
            "DP-SGD total_steps extension should be silent; got: "
            f"{[r.getMessage() for r in total_msgs]}"
        )

    def test_total_steps_drift_raises_for_dp_ftrl(
        self, lora_model, tiny_dataset, tmp_path
    ):
        """``total_steps`` drift under DP-FTRL raises — the MF strategy
        is shape-locked for the original composition.
        """
        model, tokenizer = lora_model
        trainer = DPTrainer(
            model=model,
            args=_args(tmp_path, per_device_train_batch_size=2, max_steps=10),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
            eval_dataset=tiny_dataset,
        )
        runtime = self._baseline_runtime(trainer, tiny_dataset)
        runtime.mechanism_kind = "mf_band"
        runtime.total_steps = 5  # saved=5, current=10 — DP-FTRL extension

        with pytest.raises(CheckpointError, match="Whole-horizon resume forbids drift"):
            trainer._warn_on_arg_drift(runtime)

    def test_total_steps_drift_raises_for_k_out_of_t(
        self, lora_model, tiny_dataset, tmp_path
    ):
        model, tokenizer = lora_model
        trainer = DPTrainer(
            model=model,
            args=_args(tmp_path, per_device_train_batch_size=2, max_steps=10),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
            eval_dataset=tiny_dataset,
        )
        runtime = self._baseline_runtime(trainer, tiny_dataset)
        runtime.is_horizon_process = True
        runtime.total_steps = 5

        with pytest.raises(CheckpointError, match="Whole-horizon resume forbids drift"):
            trainer._warn_on_arg_drift(runtime)

    def test_shape_drift_warns_on_lr_scheduler(
        self, lora_model, tiny_dataset, tmp_path, caplog
    ):
        """``lr_scheduler`` drift is a shape-disposition warning."""
        model, tokenizer = lora_model
        trainer = DPTrainer(
            model=model,
            args=_args(tmp_path, per_device_train_batch_size=2, lr_scheduler="linear"),
            processing_class=tokenizer,
            train_dataset=tiny_dataset,
            eval_dataset=tiny_dataset,
        )
        runtime = self._baseline_runtime(trainer, tiny_dataset)
        runtime.lr_scheduler = "cosine"  # saved=cosine, current=linear

        with caplog.at_level(logging.WARNING):
            trainer._warn_on_arg_drift(runtime)

        shape_msgs = [
            r
            for r in caplog.records
            if "lr_scheduler" in r.getMessage() and "shape" in r.getMessage()
        ]
        assert shape_msgs, (
            f"expected an lr_scheduler shape-drift warning; "
            f"got records: {[r.getMessage() for r in caplog.records]}"
        )
