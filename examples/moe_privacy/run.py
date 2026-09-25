"""Synthetic, sequence-level DP-SGD on a fully trainable Mellum MoE.

Run from the repository root with ``python -m examples.moe_privacy.run``.
Only public validation metrics and weights are exported, never trainer/RNG state.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import resource
import secrets
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from examples.moe_privacy import ARMS, LEARNING_TARGET
from examples.moe_privacy.tracking import (
    ExperimentTracker,
    TrackingOptions,
    add_tracking_arguments,
    options_from_args,
)
from torch.utils.data import Dataset
from transformers import TrainerCallback
from transformers.models.mellum.configuration_mellum import MellumConfig
from transformers.models.mellum.modeling_mellum import MellumForCausalLM
from transformers.trainer_callback import PrinterCallback

import opaque.dpsgd.accounting as accounting
from opaque.patches import apply_model_patches
from opaque.transformers import DPTrainer, TrainingArguments

PR_REVISION = "bde08c91247cc2fe127e776bc72e402d8e7fef83"
ROOT = Path(__file__).resolve().parents[2]
_MIN_SEQUENCE_LENGTH = 4
_GRAMMAR_VOCAB_SIZE = 104
_MELLUM_EVAL_OUTPUTS = 2


@dataclass(frozen=True)
class ExperimentConfig:
    """Public, validated parameters shared by every comparison arm."""

    name: str = "pilot"
    train_sequences: int = 4096
    validation_sequences: int = 1024
    sequence_length: int = 32
    expected_batch_size: int = 32
    microbatch_size: int = 1
    steps: int = 256
    eval_every: int = 32
    learning_rate: float = 0.25
    clipping_norm: float = 1.0
    target_epsilon: float = 8.0
    delta: float = 1e-5
    router_aux_loss_coef: float = 0.05
    load_noise_ratio: float = 0.02
    filter_beta: float = 0.99
    data_seed: int = 20260917
    validation_seed: int = 20260918
    transition_probability: float = 0.9
    vocab_size: int = 128
    hidden_size: int = 64
    intermediate_size: int = 128
    num_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    num_experts: int = 8
    top_k: int = 2
    moe_intermediate_size: int = 32

    def __post_init__(self) -> None:
        positive_ints = (
            "train_sequences",
            "validation_sequences",
            "sequence_length",
            "expected_batch_size",
            "microbatch_size",
            "steps",
            "eval_every",
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "num_experts",
            "top_k",
            "moe_intermediate_size",
        )
        for field in positive_ints:
            value = getattr(self, field)
            if type(value) is not int or value <= 0:
                message = f"{field} must be a positive integer"
                raise ValueError(message)
        for field in (
            "learning_rate",
            "clipping_norm",
            "target_epsilon",
            "router_aux_loss_coef",
            "load_noise_ratio",
        ):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0:
                message = f"{field} must be finite and positive"
                raise ValueError(message)
        if not 0 < self.delta < 1 / self.train_sequences:
            message = "delta must be in (0, 1 / train_sequences)"
            raise ValueError(message)
        if not 0 < self.filter_beta < 1:
            message = "filter_beta must be in (0, 1)"
            raise ValueError(message)
        if not 0 < self.transition_probability <= 1:
            message = "transition_probability must be in (0, 1]"
            raise ValueError(message)
        if not self.microbatch_size <= self.expected_batch_size <= self.train_sequences:
            message = (
                "require microbatch_size <= expected_batch_size <= train_sequences"
            )
            raise ValueError(message)
        if self.top_k >= self.num_experts:
            message = "top_k must be smaller than num_experts: this is a sparse MoE"
            raise ValueError(message)
        if (
            self.sequence_length < _MIN_SEQUENCE_LENGTH
            or self.vocab_size < _GRAMMAR_VOCAB_SIZE
        ):
            message = "the four-domain grammar requires sequence_length >= 4 and vocab_size >= 104"
            raise ValueError(message)
        if (
            self.hidden_size % self.num_attention_heads
            or (self.hidden_size // self.num_attention_heads) % 2
        ):
            message = "hidden_size must yield an even integer attention head size"
            raise ValueError(message)
        if self.num_attention_heads % self.num_key_value_heads:
            message = "num_key_value_heads must divide num_attention_heads"
            raise ValueError(message)
        for value in (self.data_seed, self.validation_seed):
            if type(value) is not int or not 0 <= value < 2**32:
                message = "public data seeds must be integers in [0, 2**32)"
                raise ValueError(message)
        if self.data_seed == self.validation_seed:
            message = "training and public validation must use different seeds"
            raise ValueError(message)

    @classmethod
    def load(cls, path: Path) -> ExperimentConfig:
        return cls(**json.loads(path.read_text()))


class GrammarDataset(Dataset):
    """One independent record per fixed-length, domain-tagged token sequence."""

    def __init__(self, config: ExperimentConfig, size: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        domains = torch.randint(4, (size,), generator=generator)
        state = torch.randint(24, (size,), generator=generator)
        offsets = torch.tensor([1, 5, 7, 11])[domains]
        self.ids = torch.empty(size, config.sequence_length, dtype=torch.long)
        self.ids[:, 0] = 1
        self.ids[:, 1] = 3 + domains
        for position in range(2, config.sequence_length):
            self.ids[:, position] = 8 + 24 * domains + state
            follow_rule = (
                torch.rand(size, generator=generator) < config.transition_probability
            )
            random_next = torch.randint(24, (size,), generator=generator)
            state = torch.where(follow_rule, (state + offsets) % 24, random_next)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ids = self.ids[index]
        return {
            "input_ids": ids,
            "labels": ids.clone(),
            "attention_mask": torch.ones_like(ids),
        }

    def fingerprint(self) -> str:
        return hashlib.sha256(self.ids.numpy().tobytes()).hexdigest()


def collate(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack sequences; DPTrainer wraps this with empty-batch handling."""
    return {name: torch.stack([row[name] for row in rows]) for name in rows[0]}


def resolve_device(device: str = "cpu") -> torch.device:
    """Select CPU or the first visible CUDA device, never silently falling back."""
    if device not in ("cpu", "cuda"):
        message = "device must be 'cpu' or 'cuda'"
        raise ValueError(message)
    if device == "cuda" and not torch.cuda.is_available():
        message = (
            "CUDA was requested but is unavailable. Use a CUDA-enabled PyTorch "
            "build on a visible GPU, or choose --device cpu."
        )
        raise RuntimeError(message)
    return torch.device("cuda:0" if device == "cuda" else "cpu")


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    """Move all inputs together without changing their dtypes."""
    return {name: value.to(device) for name, value in batch.items()}


def synchronize_device(device: torch.device) -> None:
    """Wait for CUDA work before reading wall-clock timings."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def execution_metadata(
    device: torch.device, *, peak_cuda_memory_allocated_bytes: int = 0
) -> dict[str, Any]:
    """Describe the actual tensor device, without recording any random state."""
    synchronize_device(device)
    cuda = device.type == "cuda"
    return {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if cuda
        else platform.processor() or platform.machine(),
        "cuda_version": torch.version.cuda,
        "peak_cuda_memory_allocated_bytes": max(
            peak_cuda_memory_allocated_bytes, torch.cuda.max_memory_allocated(device)
        )
        if cuda
        else None,
    }


def make_model(
    config: ExperimentConfig, seed: int, *, device: str = "cpu"
) -> MellumForCausalLM:
    """Initialize and patch on CPU, then move identical FP32 weights to the device."""
    target_device = resolve_device(device)
    if type(seed) is not int or not 0 <= seed < 2**32:
        message = "model seed must be an integer in [0, 2**32)"
        raise ValueError(message)
    if target_device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(seed)
        model_config = MellumConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=max(128, config.sequence_length),
            num_experts=config.num_experts,
            num_experts_per_tok=config.top_k,
            moe_intermediate_size=config.moe_intermediate_size,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            rope_theta=10000.0,
            router_aux_loss_coef=0.0,
            output_router_logits=True,
            use_cache=False,
        )
        model_config._attn_implementation = "eager"
        model = MellumForCausalLM(model_config).float().cpu()
        apply_model_patches(
            model,
            performance=False,
            peft=False,
            router_fp32=True,
            router_aux_loss=False,
        )
    if not all(parameter.requires_grad for parameter in model.parameters()):
        message = "the experiment must train every model parameter"
        raise RuntimeError(message)
    return model.to(target_device)


def training_arguments(
    config: ExperimentConfig, arm: str, output_dir: Path, *, device: str = "cpu"
) -> TrainingArguments:
    """Wire one arm with fresh, unpublished sampling and noise seeds."""
    target_device = resolve_device(device)
    if arm not in ARMS:
        message = f"unknown arm: {arm}"
        raise ValueError(message)
    private = arm != "reference"
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.expected_batch_size,
        microbatch_size=config.microbatch_size,
        per_device_eval_batch_size=config.microbatch_size,
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
        router_aux_loss_coef=config.router_aux_loss_coef if arm == "dp_aux" else 0.0,
        router_aux_kwargs={
            "max_tokens": config.sequence_length,
            "mean_tokens": config.sequence_length,
            "ratio": config.load_noise_ratio,
            "filter_beta": config.filter_beta,
        }
        if arm == "dp_aux"
        else {},
        # Never serialize these arguments: both seeds are private randomness.
        seed=secrets.randbits(32),
        data_seed=secrets.randbits(32),
        use_cpu=target_device.type == "cpu",
        bf16=False,
        tf32=False,
        use_performance_kernels=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        eval_strategy="steps",
        eval_steps=config.eval_every,
        eval_accumulation_steps=1,
        logging_strategy="no",
        save_strategy="no",
        report_to=[],
        disable_tqdm=True,
    )
    if arguments.device != target_device:
        message = (
            f"trainer resolved {arguments.device}, but requested {target_device}; "
            "refusing device fallback"
        )
        raise RuntimeError(message)
    return arguments


def routing_counts(
    logits: Any, labels: torch.Tensor, config: ExperimentConfig
) -> torch.Tensor:
    """Reduce public router outputs to per-sequence assignment counts."""
    if not isinstance(logits, tuple) or len(logits) != _MELLUM_EVAL_OUTPUTS:
        message = "expected Mellum (logits, router_logits) eval outputs"
        raise RuntimeError(message)
    _, routers = logits
    if not isinstance(routers, tuple) or len(routers) != config.num_layers:
        message = "missing Mellum router logits; check pinned Transformers/patches"
        raise RuntimeError(message)
    batch_size, tokens = labels.shape
    mask = labels.ne(-100)
    counts = []
    for router in routers:
        router = router.reshape(batch_size, tokens, config.num_experts)
        indices = router.float().topk(config.top_k, dim=-1).indices
        assignments = torch.nn.functional.one_hot(indices, config.num_experts).sum(-2)
        counts.append((assignments * mask.unsqueeze(-1)).sum(1))
    return torch.stack(counts, dim=1)


def routing_metrics(counts: np.ndarray) -> dict[str, float]:
    """Summarize expert-assignment shares, imbalance, and normalized entropy."""
    loads = np.asarray(counts, dtype=np.float64).sum(axis=0)
    totals = loads.sum(axis=-1, keepdims=True)
    if not np.isfinite(loads).all() or (loads < 0).any() or (totals <= 0).any():
        message = "public routing counts must be finite, nonnegative and nonempty"
        raise ValueError(message)
    shares = loads / totals
    entropy = -(shares * np.log(np.maximum(shares, np.finfo(float).tiny))).sum(-1)
    result = {
        "load_cv": float((shares.std(-1) / shares.mean(-1)).mean()),
        "max_expert_share": float(shares.max()),
        "routing_entropy": float((entropy / np.log(shares.shape[-1])).mean()),
        "active_experts_mean": float((loads > 0).sum(-1).mean()),
    }
    for layer, values in enumerate(shares):
        for expert, share in enumerate(values):
            result[f"layer_{layer}_expert_{expert}_share"] = float(share)
    return result


@torch.no_grad()
def evaluate_routing(
    model, dataset: GrammarDataset, config: ExperimentConfig
) -> dict[str, float]:
    """Measure routing on public validation data, restoring the model's mode."""
    was_training = model.training
    model.eval()
    counts = []
    device = next(model.parameters()).device
    try:
        for start in range(0, len(dataset), config.microbatch_size):
            batch = collate(
                [
                    dataset[index]
                    for index in range(
                        start, min(start + config.microbatch_size, len(dataset))
                    )
                ]
            )
            batch = move_batch(batch, device)
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_router_logits=True,
                router_aux_loss=False,
            )
            counts.append(
                routing_counts(
                    (output.logits, output.router_logits), batch["labels"], config
                )
            )
    finally:
        model.train(was_training)
    return {
        f"eval_{key}": value
        for key, value in routing_metrics(torch.cat(counts).cpu().numpy()).items()
    }


def epsilon_for_run(
    config: ExperimentConfig, arm: str, noise: float, steps: int
) -> float | None:
    """Independently account the actual gradient and optional load releases."""
    if arm == "reference":
        return None
    if arm not in ARMS or not math.isfinite(noise) or noise <= 0:
        message = "private accounting requires a known arm and positive finite noise"
        raise ValueError(message)
    mechanism = accounting.gaussian(noise)
    if arm == "dp_aux":
        mechanism = accounting.moe_aux(mechanism, ratio=config.load_noise_ratio)
    process = accounting.poisson(
        mechanism, config.expected_batch_size / config.train_sequences
    )
    return float((process * steps).epsilon_at(config.delta))


def write_json(path: Path, value: Any) -> None:
    """Write a finite-valued JSON artifact."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


class PublicMetricsRecorder(TrainerCallback):
    """Persist only public evaluation metrics, never raw training telemetry."""

    def __init__(
        self,
        path: Path,
        device: torch.device | str = "cpu",
        *,
        tracker: ExperimentTracker | None = None,
    ) -> None:
        self.path = path
        self.device = torch.device(device)
        self.tracker = tracker
        self.train_start: float | None = None
        self.rows: list[dict[str, Any]] = []
        self.peak_cuda_memory_allocated_bytes = 0
        synchronize_device(self.device)
        self._record_memory()
        self.start = time.monotonic()

    def _record_memory(self) -> None:
        if self.device.type == "cuda":
            self.peak_cuda_memory_allocated_bytes = max(
                self.peak_cuda_memory_allocated_bytes,
                torch.cuda.max_memory_allocated(self.device),
            )

    def on_train_begin(self, args, state, control, **kwargs):
        self._record_memory()
        self.train_start = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        # The trainer resets CUDA's peak counter for each training/eval phase.
        self._record_memory()
        if self.tracker is not None and self.train_start is not None:
            self.tracker.log_progress(
                state.global_step, args.max_steps, time.monotonic() - self.train_start
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        synchronize_device(self.device)
        self._record_memory()
        row = {
            "step": state.global_step,
            "elapsed_seconds": time.monotonic() - self.start,
            **{
                key: float(value)
                for key, value in metrics.items()
                if key.startswith("eval_")
            },
        }
        with self.path.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        self.rows.append(row)
        if self.tracker is not None:
            self.tracker.log_metrics(row)


def provenance() -> dict[str, Any]:
    """Record runtime versions and code identity without credentials or RNG state."""
    revision = (
        subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if (ROOT / ".git").exists()
        else None
    )
    versions = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
    }
    return {
        "pr_revision": PR_REVISION,
        "code_revision": revision.stdout.strip()
        if revision and revision.returncode == 0
        else None,
        "source_revision": os.environ.get("OPAQUE_SOURCE_REVISION"),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version,
        "dependencies": versions,
    }


def run_experiment(
    config: ExperimentConfig,
    arm: str,
    seed: int,
    output_dir: Path,
    *,
    checks: bool = False,
    device: str = "cpu",
    tracking: TrackingOptions | None = None,
) -> dict[str, Any]:
    """Train one arm and export a non-overwriting, weights-only result bundle."""
    with ExperimentTracker(
        output_dir,
        config=asdict(config),
        arm=arm,
        seed=seed,
        experiment="synthetic_moe",
        options=tracking,
    ) as tracker:
        summary = _run_experiment(
            config, arm, seed, output_dir, checks=checks, device=device, tracker=tracker
        )
        tracker.complete(summary)
        return summary


def _run_experiment(
    config: ExperimentConfig,
    arm: str,
    seed: int,
    output_dir: Path,
    *,
    tracker: ExperimentTracker,
    checks: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    target_device = resolve_device(device)
    if arm not in ARMS:
        message = f"unknown arm: {arm}"
        raise ValueError(message)
    if output_dir.exists():
        message = f"refusing to overwrite experiment output: {output_dir}"
        raise FileExistsError(message)
    torch.set_num_threads(1)
    arguments = training_arguments(config, arm, output_dir, device=device)
    synchronize_device(target_device)
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)
    model = make_model(config, seed, device=device)
    train = GrammarDataset(config, config.train_sequences, config.data_seed)
    validation = GrammarDataset(
        config, config.validation_sequences, config.validation_seed
    )
    output_dir.mkdir(parents=True)
    tracker.start()
    if checks:
        from examples.moe_privacy.checks import run_checks

        write_json(output_dir / "checks.json", run_checks(config, seed, device=device))
    recorder = PublicMetricsRecorder(
        output_dir / "metrics.jsonl", target_device, tracker=tracker
    )
    trainer = DPTrainer(
        model=model,
        args=arguments,
        train_dataset=train,
        eval_dataset=validation,
        data_collator=collate,
        callbacks=[recorder],
    )
    if any(parameter.device != target_device for parameter in model.parameters()):
        message = "model placement differs from the requested execution device"
        raise RuntimeError(message)
    trainer.remove_callback(PrinterCallback)
    trainer.evaluate()
    initial_routing = evaluate_routing(model, validation, config)
    tracker.log_metrics({"step": trainer.state.global_step, **initial_routing})
    synchronize_device(target_device)
    started = time.monotonic()
    training_logger = logging.getLogger(DPTrainer.__module__)
    previous_level = training_logger.level
    try:
        training_logger.setLevel(logging.WARNING)
        trainer.train()
    finally:
        training_logger.setLevel(previous_level)
    synchronize_device(target_device)
    training_seconds = time.monotonic() - started
    if trainer.state.global_step != config.steps:
        message = f"incomplete run: {trainer.state.global_step}/{config.steps} steps"
        raise RuntimeError(message)
    if recorder.rows[-1]["step"] != trainer.state.global_step:
        trainer.evaluate()
    noise = float(trainer.state.privacy_resolved_noise_multiplier)
    epsilon = epsilon_for_run(config, arm, noise, trainer.state.global_step)
    if epsilon is not None and (
        not math.isfinite(epsilon) or epsilon > config.target_epsilon
    ):
        message = f"accounted epsilon {epsilon} exceeds target {config.target_epsilon}"
        raise RuntimeError(message)
    if trainer.state.privacy_sample_rate != config.expected_batch_size / len(train):
        message = "trainer sample rate differs from experiment accounting"
        raise RuntimeError(message)
    initial, final = recorder.rows[0], recorder.rows[-1]
    initial.update(initial_routing)
    final.update(evaluate_routing(model, validation, config))
    recorder.path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in recorder.rows
        )
    )
    improvement = (initial["eval_loss"] - final["eval_loss"]) / initial["eval_loss"]
    # A weights-only export deliberately excludes optimizer, sampler and noise keys.
    model.save_pretrained(output_dir / "model")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    summary = {
        "schema_version": 1,
        "status": "completed",
        "arm": arm,
        "model_seed": seed,
        "config": asdict(config),
        "provenance": provenance(),
        "execution": execution_metadata(
            next(model.parameters()).device,
            peak_cuda_memory_allocated_bytes=recorder.peak_cuda_memory_allocated_bytes,
        ),
        "data": {
            "kind": "synthetic_public_four_domain_grammar",
            "train_sha256": train.fingerprint(),
            "validation_sha256": validation.fingerprint(),
        },
        "privacy": {
            "private": arm != "reference",
            "unit": "one_sequence",
            "adjacency": "add_remove",
            "epsilon": epsilon,
            "delta": config.delta if arm != "reference" else None,
            "target_epsilon": config.target_epsilon if arm != "reference" else None,
            "steps": trainer.state.global_step,
            "sample_rate": trainer.state.privacy_sample_rate,
            "noise_multiplier": noise,
            "load_release": arm == "dp_aux",
            "load_noise_ratio": config.load_noise_ratio if arm == "dp_aux" else None,
            "normalization": config.expected_batch_size,
            "max_tokens": config.sequence_length,
            "mean_tokens": config.sequence_length,
            "randomness": "fresh_unpublished_noise_and_sampling_seeds",
        },
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initial": initial,
        "final": final,
        "relative_ce_improvement": improvement,
        "learning_target_met": improvement >= LEARNING_TARGET,
        "training_seconds": training_seconds,
        "peak_rss_mib": rss / (1024**2 if sys.platform == "darwin" else 1024),
        "checks_run": checks,
        "scope": "synthetic feasibility experiment; per-run accounting, not a privacy proof or user-level DP",
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    """Run the configured arm from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="public model-initialization seed, not a noise seed",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--checks",
        action="store_true",
        help="run the real-model correctness gate before training",
    )
    add_tracking_arguments(parser)
    args = parser.parse_args()
    summary = run_experiment(
        ExperimentConfig.load(args.config),
        args.arm,
        args.seed,
        args.output_dir,
        checks=args.checks,
        device=args.device,
        tracking=options_from_args(args),
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "execution": summary["execution"],
                "epsilon": summary["privacy"]["epsilon"],
                "initial_ce": summary["initial"]["eval_loss"],
                "final_ce": summary["final"]["eval_loss"],
                "learning_target_met": summary["learning_target_met"],
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
