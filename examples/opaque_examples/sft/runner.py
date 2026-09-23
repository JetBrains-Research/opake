"""Reusable orchestration for differentially private supervised fine-tuning."""

# ruff: noqa: TRY003

from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .data import prepare_magicoder_dataset

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from .config import SFTJobConfig


_IGNORE_INDEX = -100


def _report_stage(stage: str) -> None:
    """Emit a sanitized, immediately flushed remote progress marker."""
    print(f"Opaque SFT runner stage: {stage}", flush=True)


@dataclass(frozen=True)
class SFTRunSummary:
    """JSON-safe outcome of a completed SFT job."""

    global_step: int
    train_count: int
    eval_count: int
    train_loss: float | str
    eval_loss: float | str
    final_epsilon: float | str
    resolved_delta: float
    noise_multiplier: float
    converged_physical_microbatch_size: int | None
    throughput_samples_per_second: float | None
    peak_memory_gb: float | None
    checkpoint_artifact_ids: tuple[str, ...]
    run_references: dict[str, dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON contract used by orchestration adapters."""
        return {
            "global_step": self.global_step,
            "train_examples": self.train_count,
            "eval_examples": self.eval_count,
            "train_loss": self.train_loss,
            "eval_loss": self.eval_loss,
            "final_epsilon": self.final_epsilon,
            "resolved_delta": self.resolved_delta,
            "noise_multiplier": self.noise_multiplier,
            "converged_microbatch_size": self.converged_physical_microbatch_size,
            "throughput_samples_per_second": self.throughput_samples_per_second,
            "peak_memory_gb": self.peak_memory_gb,
            "checkpoint_artifact_ids": list(self.checkpoint_artifact_ids),
            "run_references": self.run_references,
        }


class _TrainerModelSaver:
    """Expose final inference serialization through DPTrainer's publication API."""

    _TRAINING_ONLY_FILES = ("accountant.json", "training_args.bin")

    def __init__(self, trainer: Any) -> None:
        self._trainer = trainer

    def save_pretrained(self, target: str | os.PathLike[str]) -> None:
        destination = Path(target)
        self._trainer.save_model(str(destination))
        for filename in self._TRAINING_ONLY_FILES:
            (destination / filename).unlink(missing_ok=True)


@dataclass
class _RunnerDependencies:
    torch: Any
    load_dataset: Callable[..., Any]
    auto_tokenizer: Any
    auto_model_for_causal_lm: Any
    lora_config: Callable[..., Any]
    sft_config: Callable[..., Any]
    sft_trainer: Callable[..., Any]
    apply_runtime_patches: Callable[[], None]
    apply_model_patches: Callable[[Any], None]
    write_final_bundle: Callable[..., Path]
    wandb: Any | None


@dataclass(frozen=True)
class _LegacyDatasetManifest:
    source_id: str
    source_revision: str | None
    original_fingerprint: str
    dataset_text_field: str
    split_seed: int
    eval_fraction: float
    train_cap: int | None
    eval_cap: int | None
    source_count: int
    split_train_count: int
    split_eval_count: int
    final_train_count: int
    final_eval_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_revision": self.source_revision,
            "original_fingerprint": self.original_fingerprint,
            "dataset_text_field": self.dataset_text_field,
            "split_seed": self.split_seed,
            "eval_fraction": self.eval_fraction,
            "train_cap": self.train_cap,
            "eval_cap": self.eval_cap,
            "source_count": self.source_count,
            "split_train_count": self.split_train_count,
            "split_eval_count": self.split_eval_count,
            "final_train_count": self.final_train_count,
            "final_eval_count": self.final_eval_count,
        }


@dataclass(frozen=True)
class _PreparedInput:
    train_dataset: Any
    eval_dataset: Any
    delta: float
    manifest: Any


def _load_dependencies() -> _RunnerDependencies:
    _report_stage("import-ml-dependencies")
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from opaque.patches import apply_model_patches, apply_runtime_patches
    from opaque.transformers.trl import SFTConfig, SFTTrainer

    from .artifacts import write_final_bundle

    try:
        import wandb
    except ImportError:
        wandb = None

    dependencies = _RunnerDependencies(
        torch=torch,
        load_dataset=load_dataset,
        auto_tokenizer=AutoTokenizer,
        auto_model_for_causal_lm=AutoModelForCausalLM,
        lora_config=LoraConfig,
        sft_config=SFTConfig,
        sft_trainer=SFTTrainer,
        apply_runtime_patches=apply_runtime_patches,
        apply_model_patches=apply_model_patches,
        write_final_bundle=write_final_bundle,
        wandb=wandb,
    )
    _report_stage("import-ml-dependencies-complete")
    return dependencies


def _select_cap(dataset: Any, cap: int | None) -> Any:
    if cap is None or cap >= len(dataset):
        return dataset
    return dataset.select(range(cap))


def _prepare_legacy_text_dataset(dataset: Any, config: SFTJobConfig) -> _PreparedInput:
    if config.dataset_text_field not in dataset.column_names:
        raise ValueError(f"dataset is missing text field {config.dataset_text_field!r}")
    source_count = len(dataset)
    split = dataset.train_test_split(
        test_size=config.eval_fraction,
        seed=config.split_seed,
        shuffle=True,
    )
    split_train = split["train"]
    split_eval = split["test"]
    train_dataset = _select_cap(split_train, config.max_train_samples)
    eval_dataset = _select_cap(split_eval, config.max_eval_samples)
    if len(train_dataset) == 0 or len(eval_dataset) == 0:
        raise ValueError("effective train and eval datasets must both be non-empty")

    delta = (
        len(train_dataset) ** -1.1
        if config.target_delta is None
        else config.target_delta
    )
    manifest = _LegacyDatasetManifest(
        source_id=config.dataset,
        source_revision=config.dataset_revision,
        original_fingerprint=str(getattr(dataset, "_fingerprint", "unknown")),
        dataset_text_field=config.dataset_text_field,
        split_seed=config.split_seed,
        eval_fraction=config.eval_fraction,
        train_cap=config.max_train_samples,
        eval_cap=config.max_eval_samples,
        source_count=source_count,
        split_train_count=len(split_train),
        split_eval_count=len(split_eval),
        final_train_count=len(train_dataset),
        final_eval_count=len(eval_dataset),
    )
    return _PreparedInput(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        delta=delta,
        manifest=manifest,
    )


def _load_and_prepare_dataset(
    config: SFTJobConfig, dependencies: _RunnerDependencies
) -> Any:
    dataset = dependencies.load_dataset(
        path=config.dataset,
        name=config.dataset_config,
        split=config.dataset_split,
        revision=config.dataset_revision,
        streaming=False,
    )
    if config.prompt_field is None:
        return _prepare_legacy_text_dataset(dataset, config)
    assert config.completion_field is not None
    return prepare_magicoder_dataset(
        dataset,
        source_id=config.dataset,
        source_revision=config.dataset_revision,
        prompt_field=config.prompt_field,
        completion_field=config.completion_field,
        eval_fraction=config.eval_fraction,
        split_seed=config.split_seed,
        train_cap=config.max_train_samples,
        eval_cap=config.max_eval_samples,
        target_delta=config.target_delta,
    )


def _label_smoothed_ce(outputs: Any, labels: Any) -> Any:
    import torch.nn.functional as functional

    logits = outputs.logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    mask = shift_labels != _IGNORE_INDEX
    safe_labels = shift_labels.clamp(min=0)
    log_probabilities = functional.log_softmax(logits, dim=-1)
    negative_log_likelihood = -log_probabilities.gather(
        -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    smooth_loss = -log_probabilities.mean(dim=-1)
    loss = 0.9 * negative_log_likelihood + 0.1 * smooth_loss
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def _report_to(config: SFTJobConfig, *, wandb_available: bool) -> list[str]:
    if config.no_wandb or not wandb_available:
        return []
    if not os.environ.get("WANDB_MODE"):
        os.environ["WANDB_MODE"] = (
            "online" if os.environ.get("WANDB_API_KEY") else "offline"
        )
    return ["wandb"]


def _build_training_args(
    config: SFTJobConfig,
    prepared: Any,
    work_dir: Path,
    dependencies: _RunnerDependencies,
) -> Any:
    arguments: dict[str, Any] = {
        "output_dir": str(work_dir),
        "overwrite_output_dir": True,
        "dataset_text_field": config.dataset_text_field,
        "completion_only_loss": config.completion_only_loss,
        "assistant_only_loss": config.assistant_only_loss,
        "chat_template_path": config.chat_template_path,
        "eos_token": config.eos_token,
        "loss_type": config.loss_type,
        "log_completion_metrics": config.log_completion_metrics,
        "max_length": config.max_length,
        "per_device_train_batch_size": config.logical_batch_size,
        "microbatch_size": config.physical_microbatch_size,
        "num_train_epochs": config.num_train_epochs,
        "max_steps": config.max_steps if config.max_steps is not None else -1,
        "learning_rate": config.learning_rate,
        "logging_steps": config.logging_steps,
        "save_strategy": "steps" if config.save_steps is not None else "no",
        "eval_strategy": "steps" if config.eval_steps > 0 else "no",
        "eval_steps": config.eval_steps if config.eval_steps > 0 else None,
        "eval_on_start": config.eval_on_start,
        "per_device_eval_batch_size": (
            config.per_device_eval_batch_size
            or config.physical_microbatch_size
            or config.logical_batch_size
        ),
        "seed": config.seed,
        "use_cpu": not dependencies.torch.cuda.is_available(),
        "bf16": config.bf16,
        "report_to": _report_to(config, wandb_available=dependencies.wandb is not None),
        "run_name": os.environ.get("WANDB_NAME") or os.environ.get("RUN_NAME"),
        "clipping_norm": config.max_grad_norm,
        "privacy_noise_multiplier": config.noise_multiplier,
        "privacy_target_epsilon": config.target_epsilon,
        "privacy_target_delta": prepared.delta,
        "optim": "adamw",
        "optim_args": (
            "noise_bias_correction=True" if config.noise_bias_correction else None
        ),
        "use_performance_kernels": config.use_performance_kernels,
        "gradient_checkpointing": config.gradient_checkpointing,
        "activation_offloading": config.activation_offloading,
        "auto_find_microbatch_size": config.auto_microbatch_backoff,
        "skip_memory_metrics": True,
    }
    if config.save_steps is not None:
        arguments["save_steps"] = config.save_steps
    return dependencies.sft_config(**arguments)


def _build_trainer(
    config: SFTJobConfig,
    prepared: Any,
    work_dir: Path,
    callbacks: Sequence[Any],
    dependencies: _RunnerDependencies,
) -> tuple[Any, Any]:
    dependencies.apply_runtime_patches()
    tokenizer = dependencies.auto_tokenizer.from_pretrained(
        config.model, revision=config.model_revision
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = dependencies.auto_model_for_causal_lm.from_pretrained(
        config.model,
        revision=config.model_revision,
        torch_dtype=(
            dependencies.torch.bfloat16 if config.bf16 else dependencies.torch.float32
        ),
        attn_implementation=config.attn_implementation,
    )
    dependencies.apply_model_patches(model)
    peft_config = dependencies.lora_config(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.lora_target_modules),
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    training_args = _build_training_args(config, prepared, work_dir, dependencies)
    trainer = dependencies.sft_trainer(
        model=model,
        args=training_args,
        train_dataset=prepared.train_dataset,
        eval_dataset=prepared.eval_dataset,
        processing_class=tokenizer,
        compute_loss_func=_label_smoothed_ce if config.compute_loss_func else None,
        peft_config=peft_config,
        callbacks=list(callbacks),
    )
    return trainer, tokenizer


def _number(*values: Any) -> float | None:
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def _required_number(name: str, *values: Any) -> float:
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        return float(value)
    raise RuntimeError(f"trainer did not expose required {name}")


def _json_number(value: float) -> float | str:
    if math.isfinite(value):
        return value
    if math.isnan(value):
        return "nan"
    return "infinity" if value > 0 else "-infinity"


def _last_log_metric(state: Any, key: str) -> Any:
    for entry in reversed(getattr(state, "log_history", ())):
        if key in entry:
            return entry[key]
    return None


def _checkpoint_artifact_ids(callbacks: Sequence[Any]) -> tuple[str, ...]:
    identifiers: list[str] = []
    configured = os.environ.get("OPAQUE_CHECKPOINT_ARTIFACT_IDS")
    if configured:
        try:
            parsed = json.loads(configured)
        except json.JSONDecodeError:
            parsed = configured.split(",")
        if isinstance(parsed, str):
            parsed = [parsed]
        if isinstance(parsed, list):
            identifiers.extend(
                str(value).strip() for value in parsed if str(value).strip()
            )
    for callback in callbacks:
        values = getattr(callback, "checkpoint_artifact_ids", ())
        if values is None:
            continue
        if isinstance(values, str):
            values = (values,)
        identifiers.extend(str(value) for value in values if str(value))
    return tuple(dict.fromkeys(identifiers))


def _run_references(wandb: Any | None) -> dict[str, dict[str, str]]:
    references: dict[str, dict[str, str]] = {}
    wandb_reference: dict[str, str] = {}
    run = getattr(wandb, "run", None) if wandb is not None else None
    run_id = getattr(run, "id", None) or os.environ.get("WANDB_RUN_ID")
    run_url = getattr(run, "url", None) or os.environ.get("WANDB_RUN_URL")
    if run_id:
        wandb_reference["run_id"] = str(run_id)
    if run_url:
        wandb_reference["url"] = str(run_url)
    if wandb_reference:
        references["wandb"] = wandb_reference

    zenml_reference: dict[str, str] = {}
    zenml_run_id = os.environ.get("ZENML_RUN_ID")
    zenml_run_url = os.environ.get("ZENML_RUN_URL")
    if zenml_run_id:
        zenml_reference["run_id"] = zenml_run_id
    if zenml_run_url:
        zenml_reference["url"] = zenml_run_url
    if zenml_reference:
        references["zenml"] = zenml_reference
    return references


def _extract_summary(
    trainer: Any,
    train_output: Any,
    train_metrics: dict[str, Any],
    eval_metrics: dict[str, Any],
    prepared: Any,
    callbacks: Sequence[Any],
    run_references: dict[str, dict[str, str]],
) -> SFTRunSummary:
    state = trainer.state
    final_epsilon = _required_number(
        "final privacy epsilon",
        train_metrics.get("privacy_epsilon"),
        _last_log_metric(state, "privacy_epsilon"),
    )
    train_loss = _required_number(
        "training loss",
        train_metrics.get("train_loss"),
        getattr(train_output, "training_loss", None),
    )
    eval_loss = _required_number("evaluation loss", eval_metrics.get("eval_loss"))
    resolved_delta = _required_number(
        "resolved privacy delta",
        getattr(state, "privacy_resolved_delta", None),
        train_metrics.get("privacy_delta"),
    )
    noise_multiplier = _required_number(
        "resolved noise multiplier",
        getattr(state, "privacy_resolved_noise_multiplier", None),
        train_metrics.get("privacy_noise_multiplier"),
    )
    throughput = _number(
        train_metrics.get("train_samples_per_second"),
        train_metrics.get("samples_per_second"),
        train_metrics.get("avg_samples_per_second"),
    )
    peak_memory = _number(
        train_metrics.get("train_max_peak_memory_gb"),
        train_metrics.get("max_peak_memory_gb"),
        train_metrics.get("memory_peak_gb"),
        _last_log_metric(state, "max_peak_memory_gb"),
        _last_log_metric(state, "memory_peak_gb"),
    )
    if peak_memory is None:
        memory_bytes = _number(
            train_metrics.get("train_mem_gpu_peaked_delta"),
            train_metrics.get("train_mem_cpu_peaked_delta"),
        )
        if memory_bytes is not None:
            peak_memory = memory_bytes / 1024**3
    converged_microbatch = getattr(state, "converged_microbatch_size", None)
    return SFTRunSummary(
        global_step=int(
            getattr(train_output, "global_step", getattr(state, "global_step", 0))
        ),
        train_count=len(prepared.train_dataset),
        eval_count=len(prepared.eval_dataset),
        train_loss=_json_number(train_loss),
        eval_loss=_json_number(eval_loss),
        final_epsilon=_json_number(final_epsilon),
        resolved_delta=resolved_delta,
        noise_multiplier=noise_multiplier,
        converged_physical_microbatch_size=(
            int(converged_microbatch) if converged_microbatch is not None else None
        ),
        throughput_samples_per_second=throughput,
        peak_memory_gb=peak_memory,
        checkpoint_artifact_ids=_checkpoint_artifact_ids(callbacks),
        run_references=run_references,
    )


def _privacy_manifest(trainer: Any, summary: SFTRunSummary) -> dict[str, Any]:
    state = trainer.state
    return {
        "epsilon": summary.final_epsilon,
        "delta": summary.resolved_delta,
        "noise_multiplier": summary.noise_multiplier,
        "calibration_converged": getattr(state, "privacy_calibration_converged", None),
        "target_epsilon_reached": bool(
            getattr(state, "privacy_target_epsilon_reached", False)
        ),
        "sample_rate": _number(getattr(state, "privacy_sample_rate", None)),
        "total_steps": getattr(state, "privacy_total_steps", None),
    }


def _finish_wandb(wandb: Any | None) -> None:
    if wandb is None:
        return
    with contextlib.suppress(Exception):
        wandb.finish()


def _bind_callbacks(
    callbacks: Sequence[Any],
    *,
    config: SFTJobConfig,
    prepared: Any,
    source_commit_sha: str,
    run_references: dict[str, dict[str, str]],
) -> None:
    """Publish dataset-derived metadata to orchestration-specific callbacks."""
    for callback in callbacks:
        bind = getattr(callback, "bind_sft_run", None)
        if callable(bind):
            bind(
                config=config,
                dataset_manifest=prepared.manifest,
                resolved_delta=prepared.delta,
                source_commit_sha=source_commit_sha,
                run_references=run_references,
            )


def run_sft(
    config: SFTJobConfig,
    output_dir: str | os.PathLike[str],
    *,
    callbacks: Sequence[Any] = (),
    resume_from_checkpoint: str | os.PathLike[str] | bool | None = None,
    source_commit_sha: str | None = None,
) -> SFTRunSummary:
    """Train, evaluate, and atomically publish one private SFT bundle."""
    output_path = Path(output_dir)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"output directory already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dependencies = _load_dependencies()
    staging_dir: Path | None = None
    try:
        _report_stage("prepare-dataset")
        prepared = _load_and_prepare_dataset(config, dependencies)
        _report_stage("prepare-dataset-complete")
        resolved_source_sha = (
            source_commit_sha
            if source_commit_sha is not None
            else os.environ.get("OPAQUE_SOURCE_COMMIT_SHA", "unknown")
        ).strip() or "unknown"
        _bind_callbacks(
            callbacks,
            config=config,
            prepared=prepared,
            source_commit_sha=resolved_source_sha,
            run_references=_run_references(dependencies.wandb),
        )
        with tempfile.TemporaryDirectory(prefix="opaque-sft-work-") as work_root:
            _report_stage("build-model-and-trainer")
            trainer, tokenizer = _build_trainer(
                config,
                prepared,
                Path(work_root) / "trainer",
                callbacks,
                dependencies,
            )
            _report_stage("build-model-and-trainer-complete")
            _report_stage("train")
            train_output = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            _report_stage("train-complete")
            train_metrics = dict(train_output.metrics)
            _report_stage("evaluate")
            eval_metrics = dict(trainer.evaluate())
            _report_stage("evaluate-complete")
            run_references = _run_references(dependencies.wandb)
            summary = _extract_summary(
                trainer,
                train_output,
                train_metrics,
                eval_metrics,
                prepared,
                callbacks,
                run_references,
            )
            privacy = _privacy_manifest(trainer, summary)
            _report_stage("write-final-bundle")
            staging_dir = Path(
                tempfile.mkdtemp(
                    prefix=f".{output_path.name}.staging-",
                    dir=output_path.parent,
                )
            )
            bundle_path = Path(
                dependencies.write_final_bundle(
                    staging_dir,
                    model=_TrainerModelSaver(trainer),
                    tokenizer=tokenizer,
                    config=config,
                    dataset_manifest=prepared.manifest,
                    summary=summary,
                    train_metrics=train_metrics,
                    eval_metrics=eval_metrics,
                    privacy=privacy,
                    source_commit_sha=resolved_source_sha,
                    run_references=run_references,
                )
            )
            if bundle_path != staging_dir:
                raise RuntimeError("write_final_bundle returned an unexpected path")
            if output_path.exists() or output_path.is_symlink():
                raise FileExistsError(
                    f"output directory appeared during training: {output_path}"
                )
            bundle_path.replace(output_path)
            staging_dir = None
            _report_stage("write-final-bundle-complete")
            return summary
    finally:
        if staging_dir is not None:
            shutil.rmtree(staging_dir, ignore_errors=True)
        _finish_wandb(dependencies.wandb)


__all__ = ["SFTRunSummary", "run_sft"]
