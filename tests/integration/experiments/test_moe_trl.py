"""Native TRL integration in fresh interpreters, isolated from Opaque fixtures."""

import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest


@pytest.mark.slow
@pytest.mark.parametrize(
    "config_name",
    ["sft_trl_baselines.json", "sft_trl_aux0001.json"],
    ids=["coef0.2", "coef0.001"],
)
@pytest.mark.parametrize("case", ["off", "balanced", "cache", "runner", "guards"])
def test_native_trl(case, config_name, tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            case,
            config_name,
            str(tmp_path),
        ],
        env={
            **os.environ,
            "WANDB_MODE": "disabled",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _fixture(path, config_name):
    import torch
    from datasets import Dataset
    from examples.moe_privacy import sft_run
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import MellumConfig, MellumForCausalLM, PreTrainedTokenizerFast

    torch.set_num_threads(1)
    torch.manual_seed(0)
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    vocab.update({f"t{i}": i for i in range(4, 32)})
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    model_config = MellumConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        mlp_layer_types=["sparse", "sparse"],
        attention_dropout=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_cache=False,
        router_aux_loss_coef=0.017,
    )
    model_config._attn_implementation = "eager"
    model = MellumForCausalLM(model_config).float().cpu()
    checkpoint = path / "checkpoint"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    config_path = Path(sft_run.__file__).parent / "configs" / config_name
    config = replace(
        sft_run.SFTExperimentConfig(**json.loads(config_path.read_text())),
        name=f"tiny-native-trl-test-{config_path.stem}",
        model_id=str(checkpoint),
        train_sequences=8,
        validation_sequences=2,
        test_sequences=2,
        diagnostic_sequences=1,
        sequence_length=8,
        expected_batch_size=4,
        microbatch_size=2,
        eval_batch_size=2,
        steps=2,
        eval_every=1,
        lora_rank=2,
        lora_alpha=4,
    )
    rows = [
        {"input_ids": [1, 4, 5, 6, 2], "completion_mask": [0, 0, 1, 1, 1]},
        {"input_ids": [1, 7, 8, 2], "completion_mask": [0, 0, 0, 1]},
        {"input_ids": [1, 9, 10, 11, 12, 2], "completion_mask": [0, 0, 1, 1, 1, 1]},
        {"input_ids": [1, 13, 14, 15, 2], "completion_mask": [0, 0, 0, 1, 1]},
    ]
    dataset = Dataset.from_list(rows * 2)
    return config, tokenizer, dataset


def _trainer(config, model, tokenizer, dataset, path, *, balanced):
    from examples.moe_privacy import trl_run
    from transformers.trainer_callback import PrinterCallback
    from trl import SFTTrainer

    trainer = SFTTrainer(
        model=model,
        args=trl_run.training_arguments(
            config,
            "trl_reference_aux" if balanced else "trl_reference",
            path,
            device="cpu",
        ),
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=trl_run.CompletionCollator(
            config.sequence_length, tokenizer.pad_token_id
        ),
    )
    trainer.remove_callback(PrinterCallback)
    assert type(trainer) is SFTTrainer
    assert not trainer.model_accepts_loss_kwargs
    assert trainer.compute_loss_func is None
    trl_run.assert_native_model(model)
    return trainer


def _nonzero_adapters(model):
    import torch

    generator = torch.Generator().manual_seed(17)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("lora_B"):
                parameter.normal_(std=0.03, generator=generator)


def _ce(output, batch):
    import torch.nn.functional as F

    return F.cross_entropy(
        output.logits[:, :-1].float().reshape(-1, output.logits.shape[-1]),
        batch["labels"][:, 1:].reshape(-1),
        ignore_index=-100,
    )


def _loss_case(path, config_name, *, balanced):
    import torch
    from examples.moe_privacy import trl_run
    from transformers.models.mellum.modeling_mellum import load_balancing_loss_func

    config, tokenizer, dataset = _fixture(path, config_name)
    model, metadata = trl_run.load_model(config, 0, device="cpu", balanced=balanced)
    _nonzero_adapters(model)
    trainer = _trainer(
        config, model, tokenizer, dataset, path / "trainer", balanced=balanced
    )
    coefficient = config.router_aux_loss_coef if balanced else 0.0
    assert (
        model.config.router_aux_loss_coef == model.router_aux_loss_coef == coefficient
    )
    assert model.config.output_router_logits is balanced
    assert trainer.args.router_aux_loss_coef == coefficient
    assert trainer.aux_loss_enabled is balanced
    batches = [trainer.data_collator([dataset[i], dataset[i + 1]]) for i in (0, 2)]
    assert batches[0]["labels"].tolist() == [
        [-100, -100, 5, 6, 2, -100, -100, -100],
        [-100, -100, -100, 2, -100, -100, -100, -100],
    ]
    parameters = [p for p in model.parameters() if p.requires_grad]
    router = dict(model.named_parameters())[metadata["router_names"][0]]
    for batch in batches:
        loss, output = trainer.compute_loss(model, dict(batch), return_outputs=True)
        ce = _ce(output, batch)
        if balanced:
            native_aux = load_balancing_loss_func(
                output.router_logits,
                model.num_experts,
                model.num_experts_per_tok,
                batch["attention_mask"],
            )
            torch.testing.assert_close(output.aux_loss, native_aux)
            assert output.aux_loss.item() > 0
            aux_gradient = torch.autograd.grad(
                output.aux_loss, router, retain_graph=True
            )[0]
            assert aux_gradient.norm().item() > 1e-6
            task_gradient = torch.autograd.grad(ce, router, retain_graph=True)[0]
            combined_gradient = torch.autograd.grad(loss, router, retain_graph=True)[0]
            contribution = combined_gradient - task_gradient
            assert contribution.norm().item() > coefficient * 1e-6
            torch.testing.assert_close(
                contribution, coefficient * aux_gradient, atol=1e-8, rtol=1e-4
            )
            expected = ce + coefficient * output.aux_loss
        else:
            assert output.aux_loss is None
            expected = ce
        torch.testing.assert_close(loss, expected)
        expected_grads = torch.autograd.grad(expected, parameters, retain_graph=True)
        actual_grads = torch.autograd.grad(loss, parameters)
        for actual, wanted in zip(actual_grads, expected_grads, strict=True):
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, wanted)

    reference_loss = sum(model(**batch).loss / len(batches) for batch in batches)
    expected_grads = torch.autograd.grad(reference_loss, parameters)
    model.zero_grad(set_to_none=True)
    trainer.current_gradient_accumulation_steps = len(batches)
    total_tokens = sum(batch["labels"][:, 1:].ne(-100).sum() for batch in batches)
    actual_loss = sum(
        trainer.training_step(model, dict(batch), num_items_in_batch=total_tokens)
        for batch in batches
    )
    torch.testing.assert_close(actual_loss, reference_loss.detach())
    for parameter, expected in zip(parameters, expected_grads, strict=True):
        torch.testing.assert_close(parameter.grad, expected, atol=1e-7, rtol=1e-5)


def _cache_case(path, config_name):
    import torch
    from examples.moe_privacy import trl_run
    from examples.moe_privacy.sft_adapters import export_trainable, load_trainable
    from transformers import AutoModelForCausalLM
    from transformers.models.mellum.modeling_mellum import MellumTopKRouter

    config, tokenizer, dataset = _fixture(path, config_name)
    cached, metadata = trl_run.load_model(config, 0, device="cpu", balanced=True)
    native = AutoModelForCausalLM.from_pretrained(
        config.model_id, attn_implementation="eager"
    )
    uncached, _ = trl_run._configure_native_model(
        native, config, 0, balanced=True, cache_experts=False
    )
    models = (cached, uncached)
    for model in models:
        _nonzero_adapters(model)
        model.gradient_checkpointing_enable({"use_reentrant": False})
        model.train()
    batch = trl_run.CompletionCollator(8, 0)([dataset[0], dataset[1]])
    outputs = [model(**batch) for model in models]
    torch.testing.assert_close(outputs[0].logits, outputs[1].logits)
    torch.testing.assert_close(outputs[0].loss, outputs[1].loss)
    gradients = [
        torch.autograd.grad(
            output.loss, [p for p in model.parameters() if p.requires_grad]
        )
        for output, model in zip(outputs, models, strict=True)
    ]
    for left, right in zip(*gradients, strict=True):
        torch.testing.assert_close(left, right, atol=1e-7, rtol=1e-5)
    router = next(
        module for module in cached.modules() if isinstance(module, MellumTopKRouter)
    )
    hidden = torch.randn(2, 8, 16).bfloat16()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = router(hidden)
    expected = router.forward.__wrapped__(hidden.float())
    assert actual[0].dtype == torch.float32
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right)

    before = {name: p.detach().clone() for name, p in cached.named_parameters()}
    orders = []
    for index, model in enumerate(models):
        trainer = _trainer(
            config, model, tokenizer, dataset, path / f"train-{index}", balanced=True
        )
        orders.append(
            [batch["input_ids"].tolist() for batch in trainer.get_train_dataloader()]
        )
        result = trainer.train()
        assert result.global_step == 2
        assert model.is_gradient_checkpointing
    assert orders[0] == orders[1]
    for (name, left), (_, right) in zip(
        cached.named_parameters(), uncached.named_parameters(), strict=True
    ):
        assert torch.isfinite(left).all()
        torch.testing.assert_close(left, right, atol=1e-7, rtol=1e-5)
        if not left.requires_grad:
            assert torch.equal(left, before[name])
    for names in (metadata["router_names"], metadata["expert_adapter_names"]):
        assert any(
            not torch.equal(dict(cached.named_parameters())[name], before[name])
            for name in names
        )
    assert all(p.dtype == torch.float32 for p in cached.parameters() if p.requires_grad)
    export_trainable(cached, path / "export")
    restored, _ = trl_run.load_model(config, 0, device="cpu", balanced=True)
    load_trainable(restored, path / "export")
    cached.eval()
    restored.eval()
    torch.testing.assert_close(cached(**batch).logits, restored(**batch).logits)


def _runner_case(path, config_name):
    import math

    import torch
    from examples.moe_privacy import trl_campaign, trl_run
    from examples.moe_privacy.sft_adapters import load_trainable
    from examples.moe_privacy.sft_data import fingerprint

    config, tokenizer, dataset = _fixture(path, config_name)
    validation = dataset.select([0, 1])
    test = dataset.select([2, 3])
    diagnostic = dataset.select([4])
    data = {
        name: {"sha256": fingerprint(partition)}
        for name, partition in (
            ("train", dataset),
            ("validation", validation),
            ("test", test),
            ("diagnostic", diagnostic),
        )
    }
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            trl_run,
            "prepare_partitions",
            lambda config, tokenizer: (dataset, validation, test, diagnostic, data),
        )
        result = path / "result"
        summary = trl_run.run_experiment(
            config, "trl_reference_aux", 0, result, device="cpu"
        )
        assert summary["status"] == "completed"
        assert summary["experiment"] == "native_trl_moe_sft"
        assert summary["config"] == asdict(config)
        assert summary["test_evaluated"] is True
        assert summary["privacy"]["private"] is False
        assert summary["privacy"]["epsilon"] is None
        assert summary["privacy"]["target_epsilon"] is None
        assert summary["privacy"]["delta"] is None
        assert summary["privacy"]["sample_rate"] is None
        assert summary["privacy"]["noise_multiplier"] == 0
        assert summary["privacy"]["load_release"] is False
        assert summary["privacy"]["load_noise_ratio"] is None
        assert summary["privacy"]["steps"] == config.steps
        assert summary["privacy"]["balancing"] == "native_current_batch"
        assert (
            summary["execution"]["task_loss_reduction"] == trl_run.TASK_LOSS_REDUCTION
        )
        assert summary["execution"]["trainer_backend"] == "trl.SFTTrainer"
        assert summary["execution"]["opaque_patches"] is False
        assert summary["execution"]["physical_batch_size"] == config.microbatch_size
        assert (
            summary["execution"]["effective_batch_size"] == config.expected_batch_size
        )
        assert summary["execution"]["gradient_accumulation_steps"] == 2
        assert summary["execution"]["batch_clipping_norm"] == 1.0
        assert (
            summary["execution"]["router_aux_loss_coef"] == config.router_aux_loss_coef
        )
        assert summary["execution"]["aux_loss_scope"] == "native_current_physical_batch"
        assert (
            summary["execution"]["aux_loss_accumulation"]
            == "mean_of_physical_batch_aux_losses"
        )
        assert summary["execution"]["global_batch_aux_equivalent"] is False
        assert (
            summary["measurements"]["test_nll"]
            == summary["test"]["eval_nll_token_mean"]
        )
        assert (
            summary["measurements"]["test_token_accuracy"]
            == summary["test"]["eval_teacher_forced_token_accuracy"]
        )
        assert json.loads((result / "summary.json").read_text()) == json.loads(
            json.dumps(summary)
        )
        rows = [
            json.loads(line)
            for line in (result / "metrics.jsonl").read_text().splitlines()
        ]
        assert [row["step"] for row in rows] == [0, 1, 2]
        for row in rows:
            assert row["eval_loss"] == row["eval_nll_token_mean"]
            assert all(math.isfinite(value) for value in row.values())
        restored, _ = trl_run.load_model(config, 0, device="cpu", balanced=True)
        load_trainable(restored, result / "trainable")
        restored.train()
        collator = trl_run.CompletionCollator(8, tokenizer.pad_token_id)
        batch = collator(list(validation))
        output = restored(**batch)
        metrics = trl_run.evaluate_public(restored, validation, collator, batch_size=2)
        assert restored.training
        torch.testing.assert_close(
            torch.tensor(metrics["eval_nll_token_mean"]), _ce(output, batch)
        )
        torch.testing.assert_close(
            output.loss - _ce(output, batch),
            config.router_aux_loss_coef * output.aux_loss,
            atol=5e-7,
            rtol=1e-4,
        )
        assert (
            output.loss.item()
            > metrics["eval_nll_token_mean"] + config.router_aux_loss_coef / 2
        )
        assert summary["final"]["eval_loss"] == pytest.approx(
            metrics["eval_nll_token_mean"]
        )

        artifacts = {
            name: (result / name).read_bytes()
            for name in (
                "summary.json",
                "execution.json",
                "metrics.jsonl",
                "trainable/adapter_spec.json",
                "trainable/trainable.safetensors",
            )
        }
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            trl_run.run_experiment(config, "trl_reference_aux", 0, result, device="cpu")
        assert all(
            (result / name).read_bytes() == value for name, value in artifacts.items()
        )

        config_path = path / "config.json"
        config_path.write_text(json.dumps(asdict(config)))
        development = path / "development"
        patch.setattr(
            sys,
            "argv",
            [
                "trl_run",
                "--config",
                str(config_path),
                "--arm",
                "trl_reference",
                "--seed",
                "0",
                "--output-dir",
                str(development),
                "--device",
                "cpu",
                "--development-only",
                "--wandb-mode",
                "disabled",
            ],
        )
        trl_run.main()
        dev_summary = json.loads((development / "summary.json").read_text())
        assert dev_summary["test_evaluated"] is False
        assert dev_summary["test"] == {}
        assert dev_summary["measurements"]["test_nll"] is None
        assert dev_summary["privacy"]["balancing"] == "off"
        assert dev_summary["execution"]["router_aux_loss_coef"] == 0.0
        assert not (development / "test-records.jsonl").exists()
        trl_campaign.validate_result(
            summary, dev_summary, summary["config"], "trl_reference_aux"
        )
        trl_campaign.validate_result(
            dev_summary, summary, dev_summary["config"], "trl_reference"
        )


def _guards_case(path, config_name):
    from examples.moe_privacy import trl_run
    from transformers.models.mellum.modeling_mellum import (
        MellumExperts,
        MellumForCausalLM,
        MellumTopKRouter,
    )
    from trl import SFTConfig

    config, _, _ = _fixture(path, config_name)
    large = replace(
        config,
        train_sequences=8192,
        expected_batch_size=32,
        microbatch_size=4,
        steps=256,
    )
    for arm in trl_run.ARMS:
        args = trl_run.training_arguments(large, arm, path / arm, device="cpu")
        assert type(args) is SFTConfig
        assert args.per_device_train_batch_size == 4
        assert args.gradient_accumulation_steps == 8
        assert args.max_steps == 256
        assert args.seed == args.data_seed == 0
        assert args.max_grad_norm == 1.0
        assert args.optim == "sgd"
        assert args.lr_scheduler_type == "constant"
        assert args.learning_rate == 0.05
        assert args.weight_decay == 0.0
        assert args.gradient_checkpointing_kwargs == {"use_reentrant": False}
        assert args.loss_type == "nll"
        assert args.completion_only_loss
        assert not args.packing
        assert not args.padding_free
        assert args.eval_strategy == args.save_strategy == "no"
        assert args.dataset_kwargs == {"skip_prepare_dataset": True}
        assert args.train_sampling_strategy == "random"
        assert not hasattr(args, "privacy_noise_multiplier")
    with pytest.raises(ValueError, match="divisible"):
        trl_run.training_arguments(
            replace(large, expected_batch_size=33),
            "trl_reference",
            path / "bad",
            device="cpu",
        )
    with pytest.raises(ValueError, match="unknown experiment arm"):
        trl_run.training_arguments(config, "dp", path / "bad", device="cpu")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        trl_run.run_experiment(config, "trl_reference", 0, path, device="cpu")
    for cls in (MellumExperts, MellumForCausalLM, MellumTopKRouter):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(cls.forward, "__opaque_patched__", True, raising=False)
            with pytest.raises(RuntimeError, match="fresh process"):
                trl_run.load_model(config, 0, device="cpu")


if __name__ == "__main__":
    case, config_name, directory = sys.argv[1:]
    path = Path(directory)
    if case in ("off", "balanced"):
        _loss_case(path, config_name, balanced=case == "balanced")
    else:
        {"cache": _cache_case, "runner": _runner_case, "guards": _guards_case}[case](
            path, config_name
        )
