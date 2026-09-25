"""Nonprivate native TRL controls for completion-only Mellum expert-LoRA SFT.

These use shuffled batches, physical-batch token means and ordinary accumulated
batch-gradient clipping, not Opaque's Poisson/example-mean/per-record path. The
native auxiliary loss is computed on each physical batch, not the effective
accumulated batch. This is not a comparison isolating only privacy noise.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import time
from dataclasses import asdict
from functools import wraps
from pathlib import Path

import torch
import torch.utils.checkpoint
from examples.moe_privacy import LEARNING_TARGET
from examples.moe_privacy.run import (
    PublicMetricsRecorder,
    execution_metadata,
    resolve_device,
    synchronize_device,
    write_json,
)
from examples.moe_privacy.sft_adapters import configure_expert_lora, export_trainable
from examples.moe_privacy.sft_data import CompletionCollator, prepare_partitions
from examples.moe_privacy.sft_metrics import evaluate_public
from examples.moe_privacy.sft_run import Progress, SFTExperimentConfig
from examples.moe_privacy.tracking import (
    ExperimentTracker,
    TrackingOptions,
    add_tracking_arguments,
    options_from_args,
)
from torch.nn.utils import parametrize
from transformers import AutoModelForCausalLM, AutoTokenizer, modeling_utils
from transformers.models.mellum import modeling_mellum
from transformers.trainer_callback import PrinterCallback
from trl import SFTConfig, SFTTrainer

ARMS = ("trl_reference", "trl_reference_aux")
TASK_LOSS_REDUCTION = "mean_of_physical_batch_completion_token_means"
EXPERIMENT = "native_trl_moe_sft"


def training_arguments(config, arm, output_dir, *, device="cuda"):
    """Build actual TRL arguments with a fixed public shuffle seed of zero."""
    if arm not in ARMS:
        message = f"unknown experiment arm: {arm}"
        raise ValueError(message)
    if config.expected_batch_size % config.microbatch_size:
        message = "expected_batch_size must be divisible by microbatch_size"
        raise ValueError(message)
    target = resolve_device(device)
    args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.microbatch_size,
        gradient_accumulation_steps=config.expected_batch_size
        // config.microbatch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        max_steps=config.steps,
        learning_rate=config.learning_rate,
        optim="sgd",
        lr_scheduler_type="constant",
        warmup_steps=0,
        weight_decay=0.0,
        max_grad_norm=config.clipping_norm,
        router_aux_loss_coef=config.router_aux_loss_coef
        if arm == "trl_reference_aux"
        else 0.0,
        loss_type="nll",
        completion_only_loss=True,
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=config.sequence_length,
        packing=False,
        padding_free=False,
        shuffle_dataset=False,
        train_sampling_strategy="random",
        seed=0,
        data_seed=0,
        accelerator_config={"use_seedable_sampler": True},
        average_tokens_across_devices=False,
        use_cpu=target.type == "cpu",
        bf16=target.type == "cuda",
        fp16=False,
        tf32=False,
        use_liger_kernel=False,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_drop_last=False,
        remove_unused_columns=False,
        eval_strategy="no",
        logging_strategy="no",
        logging_nan_inf_filter=False,
        save_strategy="no",
        report_to=[],
        disable_tqdm=True,
    )
    if args.world_size != 1 or args.n_gpu > 1:
        message = "native TRL controls require one process and one visible device"
        raise ValueError(message)
    return args


def assert_native_model(model=None):
    """Fail closed on global or instance Opaque forward contamination."""
    candidates = list(vars(modeling_mellum).values())
    candidates += [
        torch.utils.checkpoint.checkpoint,
        modeling_utils.checkpoint,
    ]
    candidates += [
        getattr(value, "forward", None)
        for value in candidates
        if isinstance(value, type)
    ]
    if model is not None:
        candidates += [module.forward for module in model.modules()]
    for candidate in candidates:
        while callable(candidate):
            if hasattr(candidate, "__opaque_patched__") or getattr(
                candidate, "__module__", ""
            ).startswith("opaque."):
                message = "native TRL requires a fresh process without Opaque patches"
                raise RuntimeError(message)
            candidate = getattr(candidate, "__wrapped__", None)


def _cached_expert_forward(original):
    @wraps(original)
    def forward(*args, **kwargs):
        # Scope to this expert module, including each checkpoint recomputation.
        with parametrize.cached():
            return original(*args, **kwargs)

    return forward


def _fp32_router_forward(original):
    @wraps(original)
    def forward(hidden_states):
        with torch.autocast(hidden_states.device.type, enabled=False):
            return original(hidden_states.float())

    return forward


def _configure_native_model(model, config, seed, *, balanced=False, cache_experts=True):
    assert_native_model(model)
    if not isinstance(model, modeling_mellum.MellumForCausalLM):
        message = "native TRL controls require MellumForCausalLM"
        raise TypeError(message)
    if type(seed) is not int or not 0 <= seed < 2**32:
        message = "model seed must be an integer in [0, 2**32)"
        raise ValueError(message)
    coefficient = config.router_aux_loss_coef if balanced else 0.0
    model.config.use_cache = False
    model.config.output_router_logits = balanced
    model.config.router_aux_loss_coef = coefficient
    model.router_aux_loss_coef = coefficient
    # Native Mellum adds a batch-mean auxiliary term that does not honor
    # num_items_in_batch. Let Trainer divide the ENTIRE loss by accumulation,
    # rather than globally normalizing only CE and summing unscaled auxiliaries.
    model.accepts_loss_kwargs = False
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model, trainability = configure_expert_lora(
            model, rank=config.lora_rank, alpha=config.lora_alpha, apply_patches=False
        )
    for module in model.modules():
        if isinstance(module, modeling_mellum.MellumExperts) and cache_experts:
            module.forward = _cached_expert_forward(module.forward)
        elif isinstance(module, modeling_mellum.MellumTopKRouter):
            module.forward = _fp32_router_forward(module.forward)
    assert_native_model(model)
    return model, trainability


def load_model(config, seed, *, device="cuda", balanced=False):
    """Load the pinned native backbone, identical FP32 adapters and FP32 routers."""
    assert_native_model()
    target = resolve_device(device)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        dtype=torch.bfloat16 if target.type == "cuda" else torch.float32,
        attn_implementation="eager",
        trust_remote_code=False,
    )
    model, trainability = _configure_native_model(
        model, config, seed, balanced=balanced
    )
    return model.to(target), trainability


class PublicEvaluation(PublicMetricsRecorder):
    """Use task-only public NLL, never the trainer's CE-plus-aux evaluation loss."""

    def __init__(self, config, dataset, collator, output_dir, device, *, tracker):
        super().__init__(output_dir / "metrics.jsonl", device, tracker=tracker)
        self.config = config
        self.dataset = dataset
        self.collator = collator
        self.output_dir = output_dir

    def evaluate(self, model, args, state, control):
        metrics = evaluate_public(
            model,
            self.dataset,
            self.collator,
            batch_size=self.config.eval_batch_size,
            output_path=self.output_dir
            / f"validation-records-{state.global_step:06d}.jsonl"
            if self.config.detailed_metrics
            else None,
        )
        metrics["eval_loss"] = metrics["eval_nll_token_mean"]
        self.on_evaluate(args, state, control, metrics=metrics)

    def on_step_end(self, args, state, control, *, model, **kwargs):
        super().on_step_end(args, state, control, **kwargs)
        if (
            state.global_step % self.config.eval_every == 0
            or state.global_step == args.max_steps
        ):
            self.evaluate(model, args, state, control)


def _execution_settings(config, args):
    return {
        "trainer_backend": "trl.SFTTrainer",
        "physical_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": config.expected_batch_size,
        "batch_clipping_norm": args.max_grad_norm,
        "sampling_mode": "shuffled_without_replacement",
        "shuffle_seed": args.data_seed,
        "task_loss_reduction": TASK_LOSS_REDUCTION,
        "router_aux_loss_coef": args.router_aux_loss_coef,
        "aux_loss_scope": "native_current_physical_batch",
        "aux_loss_accumulation": "mean_of_physical_batch_aux_losses",
        "global_batch_aux_equivalent": False,
        "router_precision": "fp32_native_forward_autocast_disabled",
        "expert_weight_cache": "per_expert_module_forward_parametrize_cached",
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_checkpointing_kwargs": args.gradient_checkpointing_kwargs,
        "optimizer": "sgd",
        "learning_rate": args.learning_rate,
        "lr_scheduler": "constant",
        "weight_decay": args.weight_decay,
        "loss_type": args.loss_type,
        "opaque_patches": False,
    }


def _provenance():
    paths = [Path(__file__)] + [
        Path(__file__).with_name(name)
        for name in (
            "sft_run.py",
            "sft_data.py",
            "sft_adapters.py",
            "sft_metrics.py",
            "tracking.py",
        )
    ]
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }
    return {
        "source_revision": os.environ.get("OPAQUE_SOURCE_REVISION"),
        "runner_sha256": hashes[Path(__file__).name],
        "sft_sources_sha256": hashes,
        "python": sys.version,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "trl",
                "accelerate",
                "datasets",
                "safetensors",
                "tokenizers",
            )
        },
    }


def run_experiment(
    config: SFTExperimentConfig,
    arm: str,
    seed: int,
    output_dir: Path,
    *,
    device="cuda",
    tracking: TrackingOptions | None = None,
    evaluate_test=True,
):
    """Run one genuine nonprivate TRL arm and export only adapters and routers."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        message = f"refusing to overwrite experiment output: {output_dir}"
        raise FileExistsError(message)
    args = training_arguments(config, arm, output_dir, device=device)
    assert_native_model()
    target = resolve_device(device)
    settings = _execution_settings(config, args)
    with ExperimentTracker(
        output_dir,
        config={**asdict(config), **settings, "target_epsilon": None},
        arm=arm,
        seed=seed,
        experiment=EXPERIMENT,
        options=tracking,
    ) as tracker:
        output_dir.mkdir(parents=True)
        tracker.start()
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id, revision=config.model_revision, trust_remote_code=False
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        train, validation, test, _, data = prepare_partitions(config, tokenizer)
        write_json(output_dir / "data.json", data)
        model, trainability = load_model(
            config, seed, device=device, balanced=arm == "trl_reference_aux"
        )
        collator = CompletionCollator(config.sequence_length, tokenizer.pad_token_id)
        recorder = PublicEvaluation(
            config, validation, collator, output_dir, target, tracker=tracker
        )
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=train,
            data_collator=collator,
            processing_class=tokenizer,
            callbacks=[recorder, Progress()],
        )
        trainer.remove_callback(PrinterCallback)
        assert_native_model(model)
        if args.device != target or any(p.device != target for p in model.parameters()):
            message = "model/trainer placement differs from the requested device"
            raise RuntimeError(message)
        if trainer.model_accepts_loss_kwargs or trainer.compute_loss_func is not None:
            message = "TRL must accumulate the complete native physical-batch loss"
            raise RuntimeError(message)
        write_json(output_dir / "trainability.json", trainability)
        write_json(output_dir / "execution.json", settings)
        recorder.evaluate(model, args, trainer.state, trainer.control)
        synchronize_device(target)
        started = time.monotonic()
        result = trainer.train()
        synchronize_device(target)
        seconds = time.monotonic() - started
        if trainer.state.global_step != config.steps:
            message = f"incomplete training: {trainer.state.global_step}/{config.steps}"
            raise RuntimeError(message)
        if not math.isfinite(result.training_loss):
            message = "nonfinite native training loss"
            raise RuntimeError(message)
        initial, final = recorder.rows[0], recorder.rows[-1]
        test_metrics = (
            evaluate_public(
                model,
                test,
                collator,
                batch_size=config.eval_batch_size,
                output_path=output_dir / "test-records.jsonl",
            )
            if len(test) and evaluate_test
            else {}
        )
        recorder._record_memory()
        export_trainable(model, output_dir / "trainable")
        improvement = (initial["eval_loss"] - final["eval_loss"]) / initial["eval_loss"]
        summary = {
            "schema_version": 1,
            "status": "completed",
            "experiment": EXPERIMENT,
            "arm": arm,
            "model_seed": seed,
            "config": asdict(config),
            "provenance": _provenance(),
            "data": data,
            "trainability": trainability,
            "execution": {
                **execution_metadata(
                    target,
                    peak_cuda_memory_allocated_bytes=recorder.peak_cuda_memory_allocated_bytes,
                ),
                **settings,
            },
            "privacy": {
                "private": False,
                "epsilon": None,
                "delta": None,
                "target_epsilon": None,
                "noise_multiplier": 0.0,
                "steps": config.steps,
                "sample_rate": None,
                "load_release": False,
                "load_noise_ratio": None,
                "balancing": "native_current_batch"
                if arm == "trl_reference_aux"
                else "off",
                "scope": "nonprivate control; no differential privacy guarantee",
                "raw_training_telemetry_released": False,
            },
            "initial": initial,
            "final": final,
            "test": test_metrics,
            "test_evaluated": bool(test_metrics),
            "measurements": {
                name: test_metrics.get(key)
                for name, key in {
                    "test_nll": "eval_nll_token_mean",
                    "test_token_accuracy": "eval_teacher_forced_token_accuracy",
                    "test_load_cv": "eval_global_load_cv_mean",
                    "test_pooled_load_cv": "eval_pooled_load_cv",
                    "test_pooled_effective_experts": "eval_pooled_effective_experts",
                    "test_pooled_max_mean_load": "eval_pooled_max_to_mean_load",
                    "test_max_mean_load": "eval_global_max_to_mean_load",
                    "test_effective_experts": "eval_global_effective_experts_mean",
                }.items()
            },
            "training_seconds": seconds,
            "seconds_per_update": seconds / config.steps,
            "cost": {
                "training_seconds": seconds,
                "seconds_per_update": seconds / config.steps,
            },
            "relative_loss_improvement": improvement,
            "learning_target": LEARNING_TARGET,
            "learning_target_met": improvement >= LEARNING_TARGET,
        }
        write_json(output_dir / "summary.json", summary)
        tracker.complete(summary)
        return summary


def main():
    """Run one native control with explicit public-data tracking."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--development-only", action="store_true")
    add_tracking_arguments(parser)
    args = parser.parse_args()
    run_experiment(
        SFTExperimentConfig(**json.loads(args.config.read_text())),
        args.arm,
        args.seed,
        args.output_dir,
        device=args.device,
        tracking=options_from_args(args),
        evaluate_test=not args.development_only,
    )


if __name__ == "__main__":
    main()
