"""Matched privacy/balancing controls for completion-only pretrained MoE SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import secrets
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from examples.moe_privacy import LEARNING_TARGET
from examples.moe_privacy.run import (
    PublicMetricsRecorder,
    epsilon_for_run,
    execution_metadata,
    move_batch,
    provenance,
    resolve_device,
    routing_metrics,
    synchronize_device,
    write_json,
)
from examples.moe_privacy.sft_data import (
    MIN_SEQUENCE_LENGTH,
    CompletionCollator,
    prepare_partitions,
)
from examples.moe_privacy.tracking import (
    ExperimentTracker,
    TrackingOptions,
    add_tracking_arguments,
    options_from_args,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from transformers.trainer_callback import PrinterCallback

from opaque.transformers import DPTrainer, TrainingArguments

REVISION_LENGTH = 40
SHA256_LENGTH = 64
ARMS = ("reference", "reference_aux", "dp", "dp_aux")
PRIVATE_ARMS = ("dp", "dp_aux")
BALANCED_ARMS = ("reference_aux", "dp_aux")


@dataclass(frozen=True)
class SFTExperimentConfig:
    """Public hyperparameters; secret sampling/noise randomness is never saved."""

    name: str = "mellum2-code-sft-pilot"
    model_id: str = "JetBrains/Mellum2-12B-A2.5B-Base"
    model_revision: str = "271755e48ab6b2ed0ef224eaaabe2d25275fb8ee"
    dataset_id: str = "HuggingFaceH4/CodeAlpaca_20K"
    dataset_revision: str = "798c567f69c8f4b12fc191015e59ee34e9afe00d"
    dataset_format: str = "code_alpaca"
    dataset_languages: tuple[str, ...] = ()
    require_full_answers: bool = False
    excluded_train_prompt_hashes: tuple[str, ...] = ()
    train_sequences: int = 4096
    validation_sequences: int = 1024
    test_sequences: int = 0
    diagnostic_sequences: int = 0
    sequence_length: int = 256
    expected_batch_size: int = 32
    microbatch_size: int = 1
    eval_batch_size: int = 4
    steps: int = 256
    eval_every: int = 128
    learning_rate: float = 0.05
    clipping_norm: float = 0.1
    target_epsilon: float = 8.0
    delta: float = 1e-5
    router_aux_loss_coef: float = 0.05
    load_noise_ratio: float = 0.02
    filter_beta: float = 0.99
    lora_rank: int = 4
    lora_alpha: int = 8
    data_seed: int = 20260917
    validation_seed: int = 20260918
    gradient_checkpointing: bool = False
    detailed_metrics: bool = False

    def __post_init__(self):
        for name in (
            "train_sequences",
            "validation_sequences",
            "sequence_length",
            "expected_batch_size",
            "microbatch_size",
            "eval_batch_size",
            "steps",
            "eval_every",
            "lora_rank",
            "lora_alpha",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                message = f"{name} must be a positive integer"
                raise ValueError(message)
        if self.sequence_length < MIN_SEQUENCE_LENGTH:
            message = "sequence_length must be at least 4"
            raise ValueError(message)
        if not self.microbatch_size <= self.expected_batch_size <= self.train_sequences:
            message = "require microbatch <= expected batch <= train size"
            raise ValueError(message)
        for name in (
            "learning_rate",
            "clipping_norm",
            "target_epsilon",
            "router_aux_loss_coef",
            "load_noise_ratio",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                message = f"{name} must be finite and positive"
                raise ValueError(message)
        if not 0 < self.delta < 1 / self.train_sequences:
            message = "delta must be positive and below 1 / train size"
            raise ValueError(message)
        if not 0 <= self.filter_beta < 1:
            message = "filter_beta must be in [0, 1)"
            raise ValueError(message)
        if self.dataset_format not in ("code_alpaca", "magicoder_oss"):
            message = "dataset_format must be code_alpaca or magicoder_oss"
            raise ValueError(message)
        for name in ("test_sequences", "diagnostic_sequences"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                message = f"{name} must be a nonnegative integer"
                raise ValueError(message)
        if not isinstance(self.dataset_languages, (list, tuple)) or any(
            not isinstance(language, str) or not language
            for language in self.dataset_languages
        ):
            message = "dataset_languages must contain nonempty language names"
            raise ValueError(message)
        if not isinstance(self.excluded_train_prompt_hashes, (list, tuple)) or any(
            not isinstance(digest, str)
            or len(digest) != SHA256_LENGTH
            or any(c not in "0123456789abcdef" for c in digest)
            for digest in self.excluded_train_prompt_hashes
        ):
            message = "excluded_train_prompt_hashes must be SHA-256 hex digests"
            raise ValueError(message)
        if self.excluded_train_prompt_hashes and self.dataset_format != "magicoder_oss":
            message = "benchmark exclusions are supported for magicoder_oss partitions"
            raise ValueError(message)
        for name in ("model_revision", "dataset_revision"):
            revision = getattr(self, name)
            if len(revision) != REVISION_LENGTH or any(
                c not in "0123456789abcdef" for c in revision
            ):
                message = f"{name} must be an immutable 40-character revision"
                raise ValueError(message)


def training_arguments(config, arm, output_dir, *, device="cuda"):
    """Use the same joint clipping/accounting path as the tiny correctness gate."""
    target = resolve_device(device)
    if arm not in ARMS:
        message = f"unknown experiment arm: {arm}"
        raise ValueError(message)
    private = arm in PRIVATE_ARMS
    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.expected_batch_size,
        microbatch_size=config.microbatch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        max_steps=config.steps,
        learning_rate=config.learning_rate,
        optim="sgd",
        lr_scheduler="constant",
        weight_decay=0.0,
        clipping_mode="fixed",
        clipping_norm=config.clipping_norm,
        sampling_mode="poisson",
        privacy_noise_mechanism="gaussian",
        privacy_noise_multiplier=None if private else 0.0,
        privacy_target_epsilon=config.target_epsilon if private else None,
        privacy_target_delta=config.delta,
        privacy_accounting=private,
        noise_calibration_kwargs={"tolerance": 1e-4},
        router_aux_loss_coef=config.router_aux_loss_coef
        if arm in BALANCED_ARMS
        else 0.0,
        router_aux_kwargs={
            "max_tokens": config.sequence_length,
            "mean_tokens": config.sequence_length,
            "ratio": config.load_noise_ratio,
            "filter_beta": config.filter_beta,
        }
        if arm in BALANCED_ARMS
        else {},
        seed=secrets.randbits(32),
        data_seed=secrets.randbits(32),
        use_cpu=target.type == "cpu",
        bf16=target.type == "cuda",
        tf32=False,
        use_performance_kernels=False,
        performance_kernels_config={"router_fp32": True},
        gradient_checkpointing=config.gradient_checkpointing,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        eval_strategy="steps",
        eval_steps=config.eval_every,
        prediction_loss_only=True,
        eval_accumulation_steps=1,
        logging_strategy="no",
        save_strategy="no",
        report_to=[],
        disable_tqdm=True,
    )


@torch.no_grad()
def evaluate_routing(model, dataset, collator, *, batch_size=4):
    """Evaluate routing on all public input tokens, including the masked prompt."""
    device = next(model.parameters()).device
    config = model.config
    experts, top_k = config.num_experts, config.num_experts_per_tok
    layers = sum(kind == "sparse" for kind in config.mlp_layer_types)
    loads = torch.zeros(layers, experts, device=device, dtype=torch.float64)
    training = model.training
    model.eval()
    try:
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            for start in range(0, len(dataset), batch_size):
                rows = [
                    dataset[i]
                    for i in range(start, min(start + batch_size, len(dataset)))
                ]
                batch = move_batch(collator(rows), device)
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_router_logits=True,
                    router_aux_loss=False,
                )
                if out.router_logits is None or len(out.router_logits) != layers:
                    message = "router outputs do not match actual sparse layers"
                    raise RuntimeError(message)
                for layer, logits in enumerate(out.router_logits):
                    logits = logits.reshape(*batch["input_ids"].shape, experts)
                    selected = logits.float().topk(top_k, dim=-1).indices
                    counts = torch.nn.functional.one_hot(selected, experts).sum(-2)
                    loads[layer] += (
                        counts * batch["attention_mask"].unsqueeze(-1)
                    ).sum((0, 1))
    finally:
        model.train(training)
    return {
        f"eval_{k}": v for k, v in routing_metrics(loads.cpu().numpy()[None]).items()
    }


class DetailedPublicEvaluation(TrainerCallback):
    """Measure public routing and answer quality at the trainer's evaluation points."""

    def __init__(self, dataset, collator, recorder, output_dir, batch_size):
        self.dataset = dataset
        self.collator = collator
        self.recorder = recorder
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.metrics_by_step = {}

    def on_evaluate(self, args, state, control, *, model, **kwargs):
        from examples.moe_privacy.sft_metrics import evaluate_public

        step = state.global_step
        if step in self.metrics_by_step:
            self.recorder.rows[-1].update(self.metrics_by_step[step])
            return
        metrics = evaluate_public(
            model,
            self.dataset,
            self.collator,
            batch_size=self.batch_size,
            output_path=self.output_dir / f"validation-records-{step:06d}.jsonl",
        )
        self.recorder.rows[-1].update(metrics)
        self.recorder._record_memory()
        if self.recorder.tracker is not None:
            self.recorder.tracker.log_metrics({"step": step, **metrics})
        with (self.output_dir / "public_metrics.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": step, **metrics}, allow_nan=False) + "\n")
        self.metrics_by_step[step] = metrics


class Progress(TrainerCallback):
    """Report optimizer progress, without emitting raw training telemetry."""

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        print(
            json.dumps(
                {
                    "event": "optimizer_step",
                    "step": state.global_step,
                    "steps": args.max_steps,
                    "seconds": time.monotonic() - self.started,
                }
            ),
            flush=True,
        )


def run_experiment(
    config,
    arm,
    seed,
    output_dir,
    *,
    device="cuda",
    tracking: TrackingOptions | None = None,
    evaluate_test=True,
):
    """Fine-tune expert adapters and routers, exporting weights but never RNG state."""
    with ExperimentTracker(
        output_dir,
        config=asdict(config),
        arm=arm,
        seed=seed,
        experiment="pretrained_moe_sft",
        options=tracking,
    ) as tracker:
        summary = _run_experiment(
            config,
            arm,
            seed,
            output_dir,
            device=device,
            tracker=tracker,
            evaluate_test=evaluate_test,
        )
        tracker.complete(summary)
        return summary


def load_model(config, seed, *, device="cuda"):
    """Reconstruct the pinned trainable scope without loading any optimizer state."""
    from examples.moe_privacy.sft_adapters import configure_expert_lora

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=False,
    )
    model.config.use_cache = False
    model.config.router_aux_loss_coef = 0.0
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model, trainability = configure_expert_lora(
            model, rank=config.lora_rank, alpha=config.lora_alpha
        )
    model.to(resolve_device(device))
    return model, trainability


def _run_experiment(
    config, arm, seed, output_dir, *, tracker, device="cuda", evaluate_test=True
):
    from examples.moe_privacy.sft_adapters import export_trainable

    target = resolve_device(device)
    if target.type != "cuda":
        message = (
            "the pretrained 12B experiment requires CUDA; use run.py for CPU checks"
        )
        raise ValueError(message)
    if output_dir.exists():
        message = f"refusing to overwrite experiment output: {output_dir}"
        raise FileExistsError(message)
    arguments = training_arguments(config, arm, output_dir, device=device)
    output_dir.mkdir(parents=True)
    tracker.start()
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train, validation, test, _, data = prepare_partitions(config, tokenizer)
    write_json(output_dir / "data.json", data)
    print(
        json.dumps(
            {"event": "data_ready", "train": len(train), "validation": len(validation)}
        ),
        flush=True,
    )
    model, trainability = load_model(config, seed, device=device)
    collator = CompletionCollator(config.sequence_length, tokenizer.pad_token_id)
    recorder = PublicMetricsRecorder(
        output_dir / "metrics.jsonl", target, tracker=tracker
    )
    callbacks = [recorder, Progress()]
    if config.detailed_metrics:
        callbacks.append(
            DetailedPublicEvaluation(
                validation, collator, recorder, output_dir, config.eval_batch_size
            )
        )
    trainer = DPTrainer(
        model=model,
        args=arguments,
        train_dataset=train,
        eval_dataset=validation,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    trainer.remove_callback(PrinterCallback)
    if arguments.device != target or any(
        p.device != target for p in model.parameters()
    ):
        message = "model/trainer placement differs from the requested GPU"
        raise RuntimeError(message)
    write_json(output_dir / "trainability.json", trainability)
    print(
        json.dumps({"event": "initial_evaluation_started", "records": len(validation)}),
        flush=True,
    )
    trainer.evaluate()
    initial_routing = (
        {}
        if config.detailed_metrics
        else evaluate_routing(
            model, validation, collator, batch_size=config.eval_batch_size
        )
    )
    tracker.log_metrics({"step": trainer.state.global_step, **initial_routing})
    print(
        json.dumps(
            {"event": "initial_evaluation", "loss": recorder.rows[0]["eval_loss"]}
        ),
        flush=True,
    )
    synchronize_device(target)
    started = time.monotonic()
    logger = logging.getLogger(DPTrainer.__module__)
    previous = logger.level
    try:
        logger.setLevel(logging.WARNING)
        trainer.train()
    finally:
        logger.setLevel(previous)
    synchronize_device(target)
    seconds = time.monotonic() - started
    if trainer.state.global_step != config.steps:
        message = f"incomplete training: {trainer.state.global_step}/{config.steps}"
        raise RuntimeError(message)
    if recorder.rows[-1]["step"] != config.steps:
        trainer.evaluate()
    noise = float(trainer.state.privacy_resolved_noise_multiplier)
    epsilon = (
        epsilon_for_run(config, arm, noise, config.steps)
        if arm in PRIVATE_ARMS
        else None
    )
    if epsilon is not None and (
        not math.isfinite(epsilon) or epsilon > config.target_epsilon
    ):
        message = f"accounted epsilon {epsilon} exceeds target {config.target_epsilon}"
        raise RuntimeError(message)
    if trainer.state.privacy_sample_rate != config.expected_batch_size / len(train):
        message = "trainer sample rate differs from independent accounting"
        raise RuntimeError(message)
    initial, final = recorder.rows[0], recorder.rows[-1]
    initial.update(initial_routing)
    if not config.detailed_metrics:
        final.update(
            evaluate_routing(
                model, validation, collator, batch_size=config.eval_batch_size
            )
        )
    test_metrics = {}
    if len(test) and evaluate_test:
        from examples.moe_privacy.sft_metrics import evaluate_public

        test_metrics = evaluate_public(
            model,
            test,
            collator,
            batch_size=config.eval_batch_size,
            output_path=output_dir / "test-records.jsonl",
        )
        recorder._record_memory()
    recorder.path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in recorder.rows
        )
    )
    export_trainable(model, output_dir / "trainable")
    improvement = (initial["eval_loss"] - final["eval_loss"]) / initial["eval_loss"]
    source = provenance()
    source["sft_sources_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            Path(__file__),
            Path(__file__).with_name("sft_data.py"),
            Path(__file__).with_name("sft_adapters.py"),
        )
    }
    summary = {
        "schema_version": 1,
        "status": "completed",
        "experiment": "pretrained_moe_sft",
        "arm": arm,
        "model_seed": seed,
        "config": asdict(config),
        "provenance": source,
        "data": data,
        "trainability": trainability,
        "execution": execution_metadata(
            target,
            peak_cuda_memory_allocated_bytes=recorder.peak_cuda_memory_allocated_bytes,
        ),
        "privacy": {
            "private": arm in PRIVATE_ARMS,
            "unit": "one_preprocessed_prompt_answer_pair",
            "adjacency": "add_remove",
            "epsilon": epsilon,
            "delta": config.delta if arm in PRIVATE_ARMS else None,
            "target_epsilon": config.target_epsilon if arm in PRIVATE_ARMS else None,
            "noise_multiplier": noise,
            "steps": config.steps,
            "sample_rate": config.expected_batch_size / len(train),
            "load_release": arm in BALANCED_ARMS,
            "load_noise_ratio": config.load_noise_ratio if arm == "dp_aux" else None,
            "balancing": "lagged_noisy"
            if arm == "dp_aux"
            else "lagged_unnoised"
            if arm == "reference_aux"
            else "off",
            "scope": "one released fine-tuning run; not pretraining or the multi-arm campaign",
            "raw_training_telemetry_released": False,
        },
        "initial": initial,
        "final": final,
        "test": test_metrics,
        "test_evaluated": bool(test_metrics),
        "measurements": {
            "test_nll": test_metrics.get("eval_nll_token_mean"),
            "test_token_accuracy": test_metrics.get(
                "eval_teacher_forced_token_accuracy"
            ),
            "test_load_cv": test_metrics.get("eval_global_load_cv_mean"),
            "test_pooled_load_cv": test_metrics.get("eval_pooled_load_cv"),
            "test_pooled_effective_experts": test_metrics.get(
                "eval_pooled_effective_experts"
            ),
            "test_pooled_max_mean_load": test_metrics.get(
                "eval_pooled_max_to_mean_load"
            ),
            "test_max_mean_load": test_metrics.get("eval_global_max_to_mean_load"),
            "test_effective_experts": test_metrics.get(
                "eval_global_effective_experts_mean"
            ),
        },
        "training_seconds": seconds,
        "seconds_per_update": seconds / config.steps,
        "relative_loss_improvement": improvement,
        "learning_target": LEARNING_TARGET,
        "learning_target_met": improvement >= LEARNING_TARGET,
    }
    write_json(output_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "event": "completed",
                "arm": arm,
                "steps": config.steps,
                "epsilon": epsilon,
                "initial_loss": initial["eval_loss"],
                "final_loss": final["eval_loss"],
                "training_seconds": seconds,
                "output_dir": str(output_dir),
            }
        ),
        flush=True,
    )
    return summary


def main():
    """Run one pretrained SFT arm in a fresh output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--development-only",
        action="store_true",
        help="keep final-test records reserved but do not evaluate them in diagnostic runs",
    )
    add_tracking_arguments(parser)
    args = parser.parse_args()
    config = SFTExperimentConfig(**json.loads(args.config.read_text()))
    run_experiment(
        config,
        args.arm,
        args.seed,
        args.output_dir,
        device=args.device,
        tracking=options_from_args(args),
        evaluate_test=not args.development_only,
    )


if __name__ == "__main__":
    main()
