"""Matched balancing controls and independently held-out public SFT partitions."""

from dataclasses import replace

import pytest
import torch
from datasets import Dataset
from examples.moe_privacy.sft_data import prepare_partitions
from examples.moe_privacy.sft_run import SFTExperimentConfig, training_arguments


class PublicTokenizer:
    eos_token_id = 2

    def __call__(self, text, **kwargs):
        return {"input_ids": [3 + len(word) % 16 for word in text.split()]}


def test_magicoder_partitions_are_disjoint_complete_and_reproducible(monkeypatch):
    raw = Dataset.from_list(
        [
            {
                "problem": f"Implement task {i}",
                "solution": "def solve(): return 1",
                "lang": "python",
            }
            for i in range(32)
        ]
        + [{"problem": "C++ task", "solution": "int main() {}", "lang": "cpp"}]
        + [{"problem": "Too long", "solution": "code " * 100, "lang": "python"}]
    )
    monkeypatch.setattr(
        "examples.moe_privacy.sft_data.load_dataset", lambda *a, **k: {"train": raw}
    )
    config = replace(
        SFTExperimentConfig(),
        dataset_format="magicoder_oss",
        dataset_languages=("python",),
        train_sequences=8,
        validation_sequences=4,
        test_sequences=4,
        diagnostic_sequences=4,
        expected_batch_size=4,
        sequence_length=32,
        require_full_answers=True,
    )
    first = prepare_partitions(config, PublicTokenizer())
    again = prepare_partitions(config, PublicTokenizer())
    assert first[-1] == again[-1]
    groups = [
        set(first[-1][name]["prompt_hashes"])
        for name in ("train", "validation", "test", "diagnostic")
    ]
    assert len(set.union(*groups)) == sum(map(len, groups)) == 20
    assert [len(part) for part in first[:-1]] == [8, 4, 4, 4]
    for part in first[:-1]:
        assert all(row["input_ids"][-1] == 2 for row in part)
    assert first[-1]["test"]["truncated_answers"] == 0
    assert first[-1]["selection"]["language_filtered"] == 1
    assert first[-1]["selection"]["overlength_filtered"] == 1
    excluded = first[-1]["train"]["prompt_hashes"][0]
    filtered = prepare_partitions(
        replace(config, excluded_train_prompt_hashes=(excluded,)), PublicTokenizer()
    )
    assert len(filtered[0]) == config.train_sequences
    assert excluded not in filtered[-1]["train"]["prompt_hashes"]
    assert filtered[-1]["train_sha256"] != first[-1]["train_sha256"]
    for name in ("validation", "test", "diagnostic"):
        assert filtered[-1][name] == first[-1][name]


def test_nonprivate_balanced_control_has_no_noise_or_privacy_claim(tmp_path):
    config = SFTExperimentConfig()
    args = training_arguments(config, "reference_aux", tmp_path, device="cpu")
    assert args.privacy_noise_multiplier == 0.0
    assert not args.privacy_accounting
    assert args.privacy_target_epsilon is None
    assert args.router_aux_loss_coef == config.router_aux_loss_coef
    assert args.router_aux_kwargs["ratio"] == config.load_noise_ratio
    assert (
        args.clipping_norm
        == training_arguments(config, "dp_aux", tmp_path, device="cpu").clipping_norm
    )


def test_nonprivate_balanced_control_executes_real_updates(tmp_path):
    from examples.moe_privacy.run import (
        ExperimentConfig,
        GrammarDataset,
        PublicMetricsRecorder,
        collate,
        make_model,
    )
    from examples.moe_privacy.sft_adapters import configure_expert_lora
    from examples.moe_privacy.sft_run import DetailedPublicEvaluation

    from opaque.transformers import DPTrainer

    tiny = ExperimentConfig(
        train_sequences=16,
        validation_sequences=4,
        sequence_length=8,
        expected_batch_size=4,
        microbatch_size=1,
        steps=2,
        eval_every=2,
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=4,
        top_k=2,
        moe_intermediate_size=8,
    )
    model, metadata = configure_expert_lora(make_model(tiny, 0))
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    config = replace(
        SFTExperimentConfig(),
        train_sequences=16,
        expected_batch_size=4,
        sequence_length=8,
        steps=2,
        eval_every=2,
    )
    args = training_arguments(config, "reference_aux", tmp_path, device="cpu")
    args.eval_strategy = "no"
    args.seed, args.data_seed = 0, 1
    validation = GrammarDataset(tiny, 4, 1)
    recorder = PublicMetricsRecorder(tmp_path / "metrics.jsonl")
    trainer = DPTrainer(
        model=model,
        args=args,
        train_dataset=GrammarDataset(tiny, 16, 0),
        eval_dataset=validation,
        data_collator=collate,
        callbacks=[
            recorder,
            DetailedPublicEvaluation(validation, collate, recorder, tmp_path, 2),
        ],
    )
    trainer.evaluate()
    trainer.train()
    trainer.evaluate()
    trainer.evaluate()
    assert trainer.state.global_step == 2
    assert [row["step"] for row in recorder.rows] == [0, 2, 2]
    assert all(
        0 <= row["eval_teacher_forced_token_accuracy"] <= 1 for row in recorder.rows
    )
    assert all(row["eval_global_load_cv_mean"] >= 0 for row in recorder.rows)
    assert (tmp_path / "validation-records-000002.jsonl").is_file()
    assert trainer.state.privacy_resolved_noise_multiplier == 0.0
    assert trainer._router_aux_enabled
    current = dict(model.named_parameters())
    assert any(not torch.equal(before[n], current[n]) for n in metadata["router_names"])
    for name, parameter in current.items():
        if not parameter.requires_grad:
            torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)


@pytest.mark.parametrize(
    "changes",
    [
        {"filter_beta": -0.1},
        {"test_sequences": -1},
        {"diagnostic_sequences": -1},
        {"dataset_format": "unknown"},
        {"excluded_train_prompt_hashes": ["not-a-sha256"]},
    ],
)
def test_campaign_configuration_rejects_invalid_values(changes):
    with pytest.raises(ValueError, match="must"):
        replace(SFTExperimentConfig(), **changes)


def test_balanced_nonprivate_tracking_does_not_claim_dp(tmp_path):
    from examples.moe_privacy.tracking import ExperimentTracker

    tracker = ExperimentTracker(
        tmp_path,
        config={"target_epsilon": 8.0, "router_aux_loss_coef": 0.2},
        arm="reference_aux",
        seed=0,
        experiment="pretrained_moe_sft",
    )
    assert tracker.config["private"] is False
    assert tracker.config["balancing_enabled"] is True
    assert tracker.config["target_epsilon"] is None
    assert tracker.config["router_aux_loss_coef"] == 0.2
    assert tracker.config["balancing_kind"] == "lagged_unnoised"
