"""Shared configuration and argument parsing for SFT examples."""

from __future__ import annotations

import argparse
import copy
import math
from dataclasses import MISSING, dataclass, fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import NoReturn


@dataclass(frozen=True, slots=True)
class SFTJobConfig:
    """Validated inputs for one SFT training job."""

    model_name_or_path: str | None = None
    model_revision: str | None = "main"
    dataset_name: str | None = None
    dataset_revision: str | None = "main"
    dataset_config: str | None = None
    dataset_split: str = "train"
    dataset_text_field: str = "text"
    prompt_field: str | None = None
    completion_field: str | None = None
    completion_only_loss: bool = False
    packing: bool = False
    loss_type: str = "nll"
    compute_loss_func: bool = False
    assistant_only_loss: bool = False
    eos_token: str | None = None
    log_completion_metrics: bool = False
    max_length: int = 512
    max_train_samples: int | None = 256
    max_eval_samples: int | None = 64
    eval_fraction: float = 0.01
    split_seed: int = 42
    num_train_epochs: float = 3.0
    max_steps: int | None = 8
    learning_rate: float = 1e-5
    logical_batch_size: int = 16
    physical_microbatch_size: int | None = None
    auto_microbatch_backoff: bool = False
    target_epsilon: float | None = 8.0
    target_delta: float | None = None
    noise_multiplier: float | None = None
    max_grad_norm: float = 1.0
    lora_rank: int = 4
    lora_alpha: float = 8.0
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    chat_template_path: str | None = None
    bf16: bool = True
    attn_implementation: str = "sdpa"
    use_performance_kernels: bool = True
    gradient_checkpointing: bool = False
    activation_offloading: bool = False
    chunked_nll: bool = False
    save_steps: int | None = None
    eval_steps: int = 10
    eval_on_start: bool = True
    per_device_eval_batch_size: int | None = None
    logging_steps: int = 1
    noise_bias_correction: bool = False
    no_wandb: bool = False
    seed: int = 42
    output_dir: str = "trainer_output/sft"

    def __post_init__(self) -> None:
        _validate_sft_config(self)

    @property
    def model(self) -> str:
        assert self.model_name_or_path is not None
        return self.model_name_or_path

    @property
    def model_name(self) -> str:
        assert self.model_name_or_path is not None
        return self.model_name_or_path

    @property
    def dataset(self) -> str:
        assert self.dataset_name is not None
        return self.dataset_name

    @property
    def batch_size(self) -> int:
        return self.logical_batch_size

    @property
    def max_physical_batch_size(self) -> int | None:
        return self.physical_microbatch_size

    @property
    def microbatch_size(self) -> int | None:
        return self.physical_microbatch_size

    @property
    def epochs(self) -> float:
        return self.num_train_epochs

    @property
    def epsilon(self) -> float | None:
        return self.target_epsilon

    @property
    def delta(self) -> float | None:
        return self.target_delta

    @property
    def lora_r(self) -> int:
        return self.lora_rank

    @property
    def lora_modules(self) -> tuple[str, ...]:
        return self.lora_target_modules

    @property
    def clipping_norm(self) -> float:
        return self.max_grad_norm

    @property
    def auto_find_microbatch_size(self) -> bool:
        return self.auto_microbatch_backoff

    @property
    def log_steps(self) -> int:
        return self.logging_steps

    @property
    def max_samples(self) -> int | None:
        return self.max_train_samples


def _require_nonempty_string(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        _raise_config_error(f"{name} must be a non-empty string")


def _require_positive_number(name: str, value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        _raise_config_error(f"{name} must be a positive finite number")


def _require_positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _raise_config_error(f"{name} must be a positive integer")


def _require_nonnegative_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _raise_config_error(f"{name} must be a non-negative integer")


def _require_fraction(name: str, value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 < value < 1
    ):
        _raise_config_error(f"{name} must be strictly between 0 and 1")


def _raise_config_error(message: str) -> NoReturn:
    raise ValueError(message)


def _validate_sft_config(config: SFTJobConfig) -> None:
    _require_nonempty_string("model_name_or_path", config.model_name_or_path)
    _require_nonempty_string("dataset_name", config.dataset_name)
    for name, value in (
        ("model_revision", config.model_revision),
        ("dataset_revision", config.dataset_revision),
        ("dataset_config", config.dataset_config),
        ("chat_template_path", config.chat_template_path),
        ("eos_token", config.eos_token),
        ("output_dir", config.output_dir),
    ):
        if value is not None:
            _require_nonempty_string(name, value)

    if (config.prompt_field is None) != (config.completion_field is None):
        _raise_config_error("prompt_field and completion_field must be set together")
    if config.prompt_field is not None:
        _require_nonempty_string("prompt_field", config.prompt_field)
        _require_nonempty_string("completion_field", config.completion_field)
        if config.prompt_field == config.completion_field:
            _raise_config_error("prompt_field and completion_field must be different")
        if not config.completion_only_loss:
            _raise_config_error(
                "completion_only_loss must be enabled for prompt/completion data"
            )
        if config.packing:
            _raise_config_error("packing must be disabled for prompt/completion data")
    else:
        _require_nonempty_string("dataset_text_field", config.dataset_text_field)

    _require_nonempty_string("dataset_split", config.dataset_split)
    if config.loss_type not in {"nll", "dft", "chunked_nll"}:
        _raise_config_error("loss_type must be one of: nll, dft, chunked_nll")
    if config.compute_loss_func and config.loss_type != "nll":
        _raise_config_error("compute_loss_func is only supported with loss_type=nll")
    if config.chunked_nll != (config.loss_type == "chunked_nll"):
        _raise_config_error("chunked_nll must match loss_type=chunked_nll")

    for name, value in (
        ("max_length", config.max_length),
        ("logical_batch_size", config.logical_batch_size),
        ("lora_rank", config.lora_rank),
        ("logging_steps", config.logging_steps),
    ):
        _require_positive_integer(name, value)
    for name, value in (
        ("physical_microbatch_size", config.physical_microbatch_size),
        ("max_train_samples", config.max_train_samples),
        ("max_eval_samples", config.max_eval_samples),
        ("max_steps", config.max_steps),
        ("save_steps", config.save_steps),
        ("per_device_eval_batch_size", config.per_device_eval_batch_size),
    ):
        if value is not None:
            _require_positive_integer(name, value)
    _require_nonnegative_integer("eval_steps", config.eval_steps)

    if (
        config.physical_microbatch_size is not None
        and config.physical_microbatch_size > config.logical_batch_size
    ):
        _raise_config_error(
            "physical_microbatch_size must not exceed logical_batch_size"
        )

    _require_fraction("eval_fraction", config.eval_fraction)
    if not 0.0 <= config.lora_dropout < 1.0:
        _raise_config_error(
            "lora_dropout must be greater than or equal to 0 and less than 1"
        )
    for name, value in (
        ("num_train_epochs", config.num_train_epochs),
        ("learning_rate", config.learning_rate),
        ("max_grad_norm", config.max_grad_norm),
        ("lora_alpha", config.lora_alpha),
    ):
        _require_positive_number(name, value)

    if config.target_epsilon is not None:
        _require_positive_number("target_epsilon", config.target_epsilon)
    if config.target_delta is not None:
        _require_fraction("target_delta", config.target_delta)
    if config.noise_multiplier is not None and (
        not math.isfinite(config.noise_multiplier) or config.noise_multiplier < 0
    ):
        _raise_config_error("noise_multiplier must be a non-negative finite number")
    if config.target_epsilon is None and config.noise_multiplier is None:
        _raise_config_error("target_epsilon or noise_multiplier must be configured")
    if config.target_epsilon is not None and config.noise_multiplier is not None:
        _raise_config_error(
            "target_epsilon must be cleared when noise_multiplier is fixed"
        )

    if (
        not isinstance(config.lora_target_modules, tuple)
        or not config.lora_target_modules
    ):
        _raise_config_error("lora_target_modules must contain at least one module")
    for target in config.lora_target_modules:
        _require_nonempty_string("lora_target_modules", target)
    if config.attn_implementation not in {"eager", "sdpa"}:
        _raise_config_error("attn_implementation must be one of: eager, sdpa")


_FIELD_ALIASES = {
    "model": "model_name_or_path",
    "model_name": "model_name_or_path",
    "dataset": "dataset_name",
    "batch_size": "logical_batch_size",
    "physical_batch_size": "physical_microbatch_size",
    "max_physical_batch_size": "physical_microbatch_size",
    "epochs": "num_train_epochs",
    "epsilon": "target_epsilon",
    "delta": "target_delta",
    "lora_r": "lora_rank",
    "max_samples": "max_train_samples",
}


def _normalize_overrides(overrides: Mapping[str, object]) -> dict[str, object]:
    config_fields = {field.name for field in fields(SFTJobConfig)}
    normalized: dict[str, object] = {}
    unknown: list[str] = []
    for supplied_name, value in overrides.items():
        name = _FIELD_ALIASES.get(supplied_name, supplied_name)
        if name not in config_fields:
            unknown.append(supplied_name)
            continue
        if name in normalized:
            _raise_config_error(f"SFT configuration field {name!r} was supplied twice")
        if name == "lora_target_modules" and isinstance(value, list):
            value = tuple(value)
        normalized[name] = value
    if unknown:
        noun = "field" if len(unknown) == 1 else "fields"
        names = ", ".join(repr(name) for name in sorted(unknown))
        _raise_config_error(f"Unknown SFT configuration {noun}: {names}")
    if "loss_type" in normalized and "chunked_nll" not in normalized:
        normalized["chunked_nll"] = normalized["loss_type"] == "chunked_nll"
    elif "chunked_nll" in normalized and "loss_type" not in normalized:
        normalized["loss_type"] = "chunked_nll" if normalized["chunked_nll"] else "nll"
    return normalized


def resolve_sft_config(values: Mapping[str, object]) -> SFTJobConfig:
    """Overlay supplied values on defaults and return a validated SFT config."""
    resolved: dict[str, object] = {}
    for config_field in fields(SFTJobConfig):
        if config_field.default is MISSING:
            _raise_config_error(
                f"SFTJobConfig field {config_field.name!r} has no default"
            )
        resolved[config_field.name] = copy.deepcopy(config_field.default)
    resolved.update(_normalize_overrides(values))
    return SFTJobConfig(**resolved)


def build_sft_parser() -> argparse.ArgumentParser:
    """Build the shared SFT command-line parser without parsing process arguments."""
    parser = argparse.ArgumentParser(
        description="Configure an Opaque differentially private SFT job.",
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--model", "--model-name", "--model-name-or-path", dest="model_name_or_path"
    )
    parser.add_argument("--model-revision")
    parser.add_argument("--dataset", "--dataset-name", dest="dataset_name")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--dataset-config")
    parser.add_argument("--dataset-split")
    parser.add_argument("--dataset-text-field")
    parser.add_argument("--prompt-field")
    parser.add_argument("--completion-field")
    parser.add_argument("--completion-only-loss", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--completion-only", dest="completion_only_loss", action="store_true"
    )
    parser.add_argument("--packing", action=argparse.BooleanOptionalAction)
    parser.add_argument("--loss-type", choices=["nll", "dft", "chunked_nll"])
    parser.add_argument("--compute-loss-func", action="store_true")
    parser.add_argument("--assistant-only-loss", action="store_true")
    parser.add_argument("--eos-token")
    parser.add_argument("--log-completion-metrics", action="store_true")
    parser.add_argument("--max-length", type=int)
    parser.add_argument(
        "--max-samples",
        "--max-train-samples",
        "--num-train-samples",
        dest="max_train_samples",
        type=int,
    )
    parser.add_argument(
        "--max-eval-samples",
        "--num-eval-samples",
        dest="max_eval_samples",
        type=int,
    )
    parser.add_argument("--eval-fraction", type=float)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument(
        "--epochs", "--num-train-epochs", dest="num_train_epochs", type=float
    )
    parser.add_argument("--max-steps", "--stop-at-step", dest="max_steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument(
        "--batch-size", "--logical-batch-size", dest="logical_batch_size", type=int
    )
    parser.add_argument(
        "--max-physical-batch-size",
        "--physical-microbatch-size",
        "--microbatch-size",
        dest="physical_microbatch_size",
        type=int,
    )
    parser.add_argument(
        "--auto-microbatch-backoff", action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--auto-find-microbatch-size",
        dest="auto_microbatch_backoff",
        action="store_true",
    )
    parser.add_argument(
        "--epsilon", "--target-epsilon", dest="target_epsilon", type=float
    )
    parser.add_argument(
        "--no-target-epsilon", dest="target_epsilon", action="store_const", const=None
    )
    parser.add_argument("--delta", "--target-delta", dest="target_delta", type=float)
    parser.add_argument("--noise-multiplier", type=float)
    parser.add_argument(
        "--max-grad-norm", "--clipping-norm", dest="max_grad_norm", type=float
    )
    parser.add_argument("--lora-r", "--lora-rank", dest="lora_rank", type=int)
    parser.add_argument("--lora-alpha", type=float)
    parser.add_argument("--lora-dropout", type=float)
    parser.add_argument(
        "--lora-target-modules",
        "--lora-modules",
        dest="lora_target_modules",
        nargs="+",
    )
    parser.add_argument("--chat-template-path")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction)
    parser.add_argument("--attn-implementation")
    parser.add_argument(
        "--performance-kernels",
        dest="use_performance_kernels",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--activation-offloading", action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--chunked-nll", action=argparse.BooleanOptionalAction)
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--eval-steps", type=int)
    parser.add_argument("--eval-on-start", action=argparse.BooleanOptionalAction)
    parser.add_argument("--per-device-eval-batch-size", type=int)
    parser.add_argument(
        "--logging-steps", "--log-steps", dest="logging_steps", type=int
    )
    parser.add_argument(
        "--noise-bias-correction", action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir")
    return parser


def parse_sft_config(argv: Sequence[str]) -> SFTJobConfig:
    """Parse only the provided arguments and return a validated SFT config."""
    parser = build_sft_parser()
    arguments = vars(parser.parse_args(list(argv)))
    if "noise_multiplier" in arguments and "target_epsilon" not in arguments:
        arguments["target_epsilon"] = None
    try:
        return resolve_sft_config(arguments)
    except ValueError as error:
        parser.error(str(error))
