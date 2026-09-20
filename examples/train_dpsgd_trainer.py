"""End-to-end DP LoRA training with Opake's HuggingFace Trainer.

Covers both DP-SGD (Gaussian) and DP-FTRL (matrix-factorization) via the
``--noise-mechanism`` flag — the Trainer accepts both surfaces through
the same ``TrainingArguments``.

USAGE:

  # Quick smoke test, SmolLM2-135M on KExercises (DP-SGD)
  python examples/train_dpsgd_trainer.py --preset smoke

  # Full production-style configuration on Mellum-4b + KStack (DP-SGD)
  python examples/train_dpsgd_trainer.py --preset mellum-kstack

  # DP-FTRL with banded MF (Mellum-shaped defaults: bands=16, sampler=b_min_sep)
  python examples/train_dpsgd_trainer.py --preset mellum-kstack \
      --noise-mechanism mf_band

  # DP-FTRL with BLT (balls-in-bins sampling)
  python examples/train_dpsgd_trainer.py --preset smoke \
      --noise-mechanism mf_blt --noise-mechanism-kwargs max_buffers=10

  # Save Trainer checkpoints every eval interval
  python examples/train_dpsgd_trainer.py --preset smoke --save-steps 10

  # Disable W&B/HF reporting callbacks
  python examples/train_dpsgd_trainer.py --preset smoke --no-wandb

  # Block k-out-of-t allocation
  python examples/train_dpsgd_trainer.py --preset smoke --num-epochs 2 \\
      --noise-mechanism gaussian --sampler k_out_of_t --k 2 \
      --allocation block
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
import time
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from opake.device import device_capabilities
from opake.transformers.trainer import Trainer, TrainingArguments

log = logging.getLogger(__name__)


def _select_device() -> tuple[torch.device, str]:
    """Select best available device, honouring ``LOCAL_RANK`` under torchrun.

    Each torchrun rank reads ``LOCAL_RANK`` to pin itself to a distinct CUDA
    device; without this every rank lands on ``cuda:0`` and NCCL crashes with
    ``Duplicate GPU detected`` at the first collective.  Mirrors the manual-
    loop example (``examples/train_dpsgd.py:_select_device``).
    """
    if torch.cuda.is_available():
        local_rank_env = os.environ.get("LOCAL_RANK")
        if local_rank_env is not None:
            local_rank = int(local_rank_env)
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            return device, torch.cuda.get_device_name(local_rank)
        device = torch.device("cuda")
        return device, torch.cuda.get_device_name(0)
    if torch.backends.mps.is_available():
        return torch.device("mps"), "Apple Silicon"
    return torch.device("cpu"), "CPU"


def _resolve_trainer_dtype(
    requested_name: str,
    device: torch.device,
) -> tuple[str, torch.dtype, str | None]:
    """Resolve dtype for Trainer, honouring bf16 wherever the device runs it.

    Trainer's full-cast precision supports float32 everywhere and bf16 wherever
    the device can actually execute it — CUDA (Ampere+), Apple Silicon (MPS) on a
    recent PyTorch, and CPU (functional but slow, used under ``use_cpu=True``).
    We ask :func:`device_capabilities` rather than hard-coding a per-device table
    so MPS bf16 stays enabled as PyTorch's Metal support advances, instead of
    being silently downgraded to fp32.
    """
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    requested_dtype = dtype_map[requested_name]
    if requested_name == "float32" or device_capabilities(device).supports_bf16:
        return requested_name, requested_dtype, None

    # bf16 requested but this device can't run it → fall back to fp32.
    reason = (
        f"Requested dtype '{requested_name}' is not supported on "
        f"{device.type} for Trainer; using 'float32' instead."
    )
    return "float32", torch.float32, reason


def _kernel_mode_summary(device: torch.device, dtype_name: str) -> tuple[str, str]:
    """Return concise status of kernel optimization mode for this run."""
    if os.environ.get("OPAKE_NO_PATCH", "0") == "1":
        return "disabled", "OPAKE_NO_PATCH=1"
    if os.environ.get("OPAKE_NO_KERNEL_PATCH", "0") == "1":
        return "disabled", "OPAKE_NO_KERNEL_PATCH=1"
    if device.type != "cuda":
        return "disabled", f"device={device.type} (Triton kernels are CUDA-only)"
    if importlib.util.find_spec("triton") is None:
        return "disabled", "triton package not installed"
    if dtype_name != "bfloat16":
        return "partial", f"dtype={dtype_name} (fused CE prefers bf16 here)"
    return "enabled", "applied by Trainer"


def _print_runtime_mode_report(
    device: torch.device,
    device_label: str,
    dtype_name: str,
    dtype: torch.dtype,
    dtype_warning: str | None,
) -> None:
    """Print active runtime mode so fallback behavior is explicit."""
    kernel_mode, kernel_reason = _kernel_mode_summary(device, dtype_name)

    print("\nRuntime mode:")
    print(f"  Device: {device} ({device_label})")
    print(f"  Dtype: {dtype_name} ({dtype})")
    if dtype_warning:
        print(f"  Dtype fallback: {dtype_warning}")
    print(f"  Kernel optimizations: {kernel_mode} ({kernel_reason})")

    if device.type == "cpu":
        print("  Note: CPU path prioritizes correctness over throughput.")
    elif device.type == "mps":
        print(
            "  Note: Apple Silicon (MPS) runs bf16 with eager kernels; the "
            "Triton fused kernels fall back to pure-PyTorch equivalents."
        )


def _load_streaming_subset(
    dataset_name: str,
    dataset_subset: str | None,
    dataset_split: str,
    dataset_text_field: str,
    total_needed: int,
) -> Dataset:
    """Stream only required rows, then materialize to an in-memory Dataset."""
    print("  Streaming source dataset and materializing required subset...")
    stream_ds = load_dataset(
        dataset_name,
        name=dataset_subset,
        split=dataset_split,
        streaming=True,
    )

    rows = list(stream_ds.take(total_needed))
    if rows and dataset_text_field not in rows[0]:
        raise ValueError(
            f"Text field '{dataset_text_field}' not found in streamed sample. "
            f"Available fields: {list(rows[0].keys())}"
        )
    if len(rows) < total_needed:
        raise ValueError(
            f"Stream ended after {len(rows)} examples, but {total_needed} are "
            "required (train + eval)."
        )
    return Dataset.from_list(rows)


def _provided_dests(parser: argparse.ArgumentParser) -> set[str]:
    provided: set[str] = set()
    argv_tokens = sys.argv[1:]
    for action in parser._actions:
        if not action.option_strings:
            continue
        for opt in action.option_strings:
            if any(
                token == opt or token.startswith(f"{opt}=") for token in argv_tokens
            ):
                provided.add(action.dest)
                break
    return provided


def _require_configured(parser, args, required=("model_name", "dataset")):
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        flags = ", ".join("--" + name.replace("_", "-") for name in missing)
        parser.error(
            f"missing required configuration: {flags}. "
            f"Pass them directly or select a --preset (e.g. --preset smoke)."
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with logical groups."""
    parser = argparse.ArgumentParser(
        description="DP-SGD LoRA training for causal LMs using Trainer"
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["smoke", "mellum-kstack", "mellum2-kstack"],
        default=None,
        help="Optional preset that fills in any unset arguments (CLI args take "
        "precedence). Omit it to configure the run directly (at least "
        "--model-name and --dataset).",
    )

    model_group = parser.add_argument_group("model", "Model and tokenizer settings")
    model_group.add_argument("--model-name", type=str, default=None)
    model_group.add_argument(
        "--attention",
        type=str,
        choices=["eager", "sdpa"],
        default="sdpa",
        help="Attention implementation.",
    )
    model_group.add_argument(
        "--sdpa-backend",
        type=str,
        choices=["flash", "efficient", "cudnn", "math"],
        default=None,
        help="Force a specific CUDA SDPA backend.",
    )
    model_group.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "bfloat16"],
        help="Model precision. Trainer supports float32 and bf16.",
    )

    data_group = parser.add_argument_group("data", "Dataset and tokenization settings")
    data_group.add_argument("--dataset", type=str, default=None)
    data_group.add_argument(
        "--dataset-subset",
        "--dataset-name",
        dest="dataset_subset",
        type=str,
        default=None,
        help="Optional HF load_dataset name/subset argument.",
    )
    data_group.add_argument("--dataset-split", type=str, default="train")
    data_group.add_argument("--dataset-text-field", type=str, default="text")
    data_group.add_argument("--num-train-samples", type=int, default=5000)
    data_group.add_argument(
        "--num-eval-samples",
        "--num-eval-samples-alt",
        dest="num_eval_samples",
        type=int,
        default=1000,
    )
    data_group.add_argument("--max-seq-len", type=int, default=512)

    train_group = parser.add_argument_group("training", "Training loop settings")
    train_group.add_argument(
        "--output-dir", type=str, default="trainer_output/dp_trainer_causal_lm"
    )
    train_group.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Expected Poisson batch size, i.e. Trainer logical batch.",
    )
    train_group.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Evaluation batch size (default: microbatch_size or batch_size).",
    )
    train_group.add_argument("--num-epochs", type=int, default=3)
    train_group.add_argument("--learning-rate", type=float, default=1.0e-5)
    train_group.add_argument(
        "--optimizer",
        type=str,
        default="adafactor",
        choices=[
            "sgd",
            "adam",
            "adamw",
            "adamw-bc",
            "adagrad",
            "rmsprop",
            "lion",
            "ademamix",
            "adafactor",
            "radam",
            "adadelta",
            "schedule_free",
        ],
        help="Torchopt-backed optimizer name passed to TrainingArguments.optim.",
    )
    train_group.add_argument("--weight-decay", type=float, default=0.01)
    train_group.add_argument(
        "--max-grad-norm",
        type=float,
        default=None,
        help=(
            "HF TrainingArguments.max_grad_norm forwarded verbatim. In "
            "Trainer this field is inert (DP per-example clipping owns "
            "the clip path; HF's clip_grad_norm_ is never called), but "
            "exposing the knob lets validators confirm ε is unaffected "
            "when users carry over a non-DP recipe that sets it. Default: "
            "leave HF's own default in place (1.0)."
        ),
    )
    train_group.add_argument("--log-steps", type=int, default=1)
    train_group.add_argument("--eval-steps", type=int, default=10)
    train_group.add_argument(
        "--eval-on-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run an evaluation pass at step 0 before training begins, "
        "providing a pre-training anchor for the eval curve. "
        "``--no-eval-on-start`` skips it.",
    )
    train_group.add_argument(
        "--save-steps",
        type=int,
        default=None,
        help="Save Trainer checkpoints every N steps. Default: disabled.",
    )
    train_group.add_argument(
        "--save-strategy",
        type=str,
        default=None,
        choices=["no", "steps", "epoch", "best"],
        help=(
            "Checkpoint save strategy forwarded to TrainingArguments. "
            "Default: inferred ('steps' when --save-steps is set, else 'no')."
        ),
    )
    train_group.add_argument(
        "--save-total-limit",
        type=int,
        default=None,
        help=(
            "Keep at most N most-recent checkpoints (best is protected). "
            "Default: unbounded."
        ),
    )
    train_group.add_argument(
        "--save-only-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Save only model weights (no optimizer / DP runtime / accountant). "
            "Such a checkpoint is a weights-only export and is NOT resumable."
        ),
    )
    train_group.add_argument(
        "--load-best-model-at-end",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reload the best-metric checkpoint at the end of training. "
            "Requires eval and save strategies to match."
        ),
    )
    train_group.add_argument(
        "--metric-for-best-model",
        type=str,
        default=None,
        help="Metric used to select the best checkpoint (e.g. eval_loss).",
    )
    train_group.add_argument(
        "--greater-is-better",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Whether a larger metric is better. Default: inferred from "
            "--metric-for-best-model (False for *loss metrics)."
        ),
    )
    train_group.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help=(
            "Resume from a checkpoint directory, or pass 'auto'/'true' to "
            "auto-detect the latest checkpoint under --output-dir."
        ),
    )
    train_group.add_argument(
        "--stop-at-step",
        type=int,
        default=None,
        help="Stop the training loop after this many optimizer steps "
        "(early-stop knob, not a privacy-accounting target — privacy is "
        "calibrated from target_epsilon × steps × sample_rate regardless).",
    )
    train_group.add_argument("--seed", type=int, default=42)
    train_group.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="Optional separate seed for Poisson sampling.",
    )
    train_group.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    train_group.add_argument(
        "--activation-offloading",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use Trainer activation_offloading.",
    )
    train_group.add_argument(
        "--auto-find-microbatch-size",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Let Trainer retry with smaller physical microbatches on OOM.",
    )
    train_group.add_argument(
        "--torch-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable torch.compile on the DP per-example loss closure.",
    )
    train_group.add_argument(
        "--torch-compile-backend",
        type=str,
        default=None,
        help="torch.compile backend (default: inductor when --torch-compile).",
    )
    train_group.add_argument(
        "--torch-compile-mode",
        type=str,
        default=None,
        choices=[
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
        help="torch.compile mode (default: 'default' when --torch-compile).",
    )
    train_group.add_argument(
        "--bf16-full-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Cast the model to bf16 for the final post-training evaluation. "
            "Only applies outside the training loop (mid-training eval still "
            "uses the training dtype)."
        ),
    )
    train_group.add_argument(
        "--use-performance-kernels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable opake's CUDA + Triton kernel patches "
            "(rope / rms_norm / activation / cross_entropy). Requires "
            "CUDA + Triton at runtime."
        ),
    )
    train_group.add_argument(
        "--lr-scheduler",
        type=str,
        default="constant",
        help=(
            "HF-style LR scheduler name forwarded to "
            "TrainingArguments.lr_scheduler.  Supported: constant, "
            "constant_with_warmup, linear, cosine, polynomial, "
            "inverse_sqrt, cosine_with_restarts, cosine_with_min_lr, "
            "cosine_warmup_with_min_lr, warmup_stable_decay.  Compose "
            "warmup with any of the above via --warmup-steps (e.g. "
            "`--lr-scheduler linear --warmup-steps 0.1` is "
            "with_warmup(linear, ...))."
        ),
    )
    train_group.add_argument(
        "--warmup-steps",
        type=float,
        default=0,
        help=(
            "Warmup length: a number of steps when >= 1, or a fraction of "
            "the total training steps when between 0 and 1."
        ),
    )
    train_group.add_argument(
        "--lr-scheduler-kwargs",
        type=str,
        default=None,
        help=(
            "JSON object of scheduler-specific kwargs (e.g. "
            "'{\"num_cycles\": 0.5}' for cosine, "
            "'{\"min_lr\": 1e-6}' for cosine_with_min_lr).  Empty ⇒ "
            "scheduler defaults."
        ),
    )

    lora_group = parser.add_argument_group("lora", "LoRA adapter settings")
    lora_group.add_argument("--lora-r", type=int, default=4)
    lora_group.add_argument("--lora-alpha", type=int, default=8)
    lora_group.add_argument(
        "--lora-modules",
        type=str,
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    dp_group = parser.add_argument_group("dp", "DP-SGD clipping and noise")
    dp_group.add_argument(
        "--clipping-mode",
        type=str,
        choices=["fixed", "adaptive", "auto"],
        default="adaptive",
    )
    dp_group.add_argument(
        "--clipping-norm",
        type=float,
        default=1.0,
        help=(
            "Per-example DP clip bound (positive float).  Pass 'inf' to disable "
            "clipping entirely — only meaningful for a non-private baseline "
            "(--noise-multiplier 0)."
        ),
    )
    dp_group.add_argument("--target-clipping-rate", type=float, default=0.5)
    dp_group.add_argument("--clipping-norm-max", type=float, default=10.0)
    dp_group.add_argument("--auto-clipping-gamma", type=float, default=0.01)
    dp_group.add_argument(
        "--microbatch-size",
        type=int,
        default=None,
        help="Physical batch size. Use 0 to mean no microbatching.",
    )
    dp_group.add_argument(
        "--sampler",
        type=str,
        choices=[
            "auto",
            "poisson",
            "k_out_of_t",
            "b_min_sep",
            "balls_in_bins",
        ],
        default="auto",
        help=(
            "Per-step participation pattern.  ``auto`` (default) lets the "
            "trainer pick the canonical sampler for the chosen "
            "``--noise-mechanism`` (poisson for gaussian / mf_identity, "
            "b_min_sep for mf_band, balls_in_bins for mf_blt / mf_bisr / "
            "mf_bsr / mf_lambda_cgd).  Gaussian also accepts "
            "``k_out_of_t`` (block or total-horizon allocation; pair with "
            "``--k`` and ``--allocation``). Explicit overrides are validated "
            "by ``TrainingArguments`` against the mechanism allow-list."
        ),
    )
    dp_group.add_argument(
        "--k",
        type=int,
        default=None,
        metavar="K",
        help=(
            "For ``--sampler k_out_of_t``: each training example participates "
            "in exactly K optimizer steps. Required when that sampler is selected."
        ),
    )
    dp_group.add_argument(
        "--allocation",
        choices=["block", "total"],
        default=None,
        help=(
            "For ``--sampler k_out_of_t``: ``block`` assigns every example once "
            "in each of K nearly equal blocks; ``total`` chooses K steps uniformly "
            "from the complete horizon and uses the conservative block accountant."
        ),
    )
    dp_group.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        help=(
            "Optional upper bound on the per-step Poisson batch. Routes "
            "through the truncated form of the Poisson accountant; ε is "
            "higher than unbounded Poisson (Gan'25 pessimistic bound) but "
            "valid under a guaranteed bounded batch size."
        ),
    )
    dp_group.add_argument(
        "--noise-mechanism",
        type=str,
        choices=[
            "gaussian",
            "mf_band",
            "mf_blt",
            "mf_bisr",
            "mf_bsr",
            "mf_lambda_cgd",
            "mf_identity",
        ],
        default="gaussian",
        help=(
            "DP-SGD: 'gaussian' (default).  "
            "DP-FTRL: 'mf_band', 'mf_blt', 'mf_bisr', 'mf_bsr', "
            "'mf_lambda_cgd', 'mf_identity'.  Strategy kwargs auto-fill "
            "from Mellum-shaped defaults; override via --noise-mechanism-kwargs."
        ),
    )
    dp_group.add_argument(
        "--noise-mechanism-kwargs",
        type=str,
        default=None,
        help=(
            "JSON object or 'key=value,...' string forwarded to the strategy "
            "factory (e.g. 'bands=16' for mf_band, "
            "'bandwidth=8,alpha=1.0,beta=0.9' for mf_bsr).  Empty ⇒ keep "
            "per-mechanism defaults."
        ),
    )
    dp_group.add_argument(
        "--per-group-clipping",
        type=str,
        nargs="+",
        default=None,
        metavar="PATTERN=NORM",
    )

    privacy_group = parser.add_argument_group("privacy", "Privacy accounting")
    privacy_group.add_argument("--target-epsilon", type=float, default=8.0)
    privacy_group.add_argument("--target-delta", type=float, default=None)
    privacy_group.add_argument(
        "--noise-multiplier",
        type=float,
        default=None,
        help=(
            "Fixed noise multiplier (skips calibration).  Set to 0 for a "
            "non-private baseline: the chosen mechanism/sampler are kept, no "
            "noise is added, and the accountant reports epsilon=inf."
        ),
    )
    privacy_group.add_argument("--calibration-min", type=float, default=0.11)
    privacy_group.add_argument("--calibration-max", type=float, default=3.5)
    privacy_group.add_argument("--calibration-tolerance", type=float, default=1e-3)

    tracking_group = parser.add_argument_group("tracking", "Experiment tracking")
    tracking_group.add_argument("--no-wandb", action="store_true")
    tracking_group.add_argument(
        "--wandb-project",
        type=str,
        default=os.environ.get("WANDB_PROJECT", "opake"),
    )
    tracking_group.add_argument(
        "--wandb-run-name",
        type=str,
        default=os.environ.get("WANDB_NAME") or os.environ.get("RUN_NAME"),
    )
    tracking_group.add_argument(
        "--wandb-entity",
        type=str,
        default=os.environ.get("WANDB_ENTITY"),
    )
    tracking_group.add_argument(
        "--wandb-tags",
        type=str,
        nargs="+",
        default=None,
        metavar="TAG",
        help=(
            "Space-separated W&B tags applied to the run. Forwarded "
            "as the ``WANDB_TAGS`` environment variable (CSV); equivalent "
            "to setting ``WANDB_TAGS=tag1,tag2`` before invocation."
        ),
    )

    hub_group = parser.add_argument_group("hub", "Hugging Face Hub publishing")
    hub_group.add_argument(
        "--push-to-hub",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Upload the final model + tokenizer + DP model card to the Hub.",
    )
    hub_group.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="Target Hub repo id (e.g. 'org/name'). Required when --push-to-hub.",
    )
    hub_group.add_argument(
        "--hub-private-repo",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Create the Hub repo as private (default: account default).",
    )
    hub_group.add_argument(
        "--hub-revision",
        type=str,
        default=None,
        help="Branch / revision name on the Hub to push to (default: main).",
    )

    args = parser.parse_args()
    provided_dests = _provided_dests(parser)

    def _set(name: str, value: Any) -> None:
        if name not in provided_dests:
            setattr(args, name, value)

    if args.preset == "smoke":
        _set("model_name", "HuggingFaceTB/SmolLM2-135M")
        _set("dataset", "JetBrains/KExercises")
        _set("dataset_text_field", "solution")
        _set("num_train_samples", 256)
        _set("num_eval_samples", 64)
        _set("num_epochs", 1)
        _set("batch_size", 16)
        _set("log_steps", 5)
        _set("eval_steps", 5)
        _set("target_epsilon", 3.0)
        _set("learning_rate", 1e-5)
        _set("lora_r", 4)
        _set("lora_alpha", 8)
        _set("max_seq_len", 512)
        _set("lora_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
        _set("dtype", "bfloat16")
    elif args.preset == "mellum-kstack":
        _set("model_name", "JetBrains/Mellum-4b-base")
        _set("dataset", "JetBrains/KStack")
        _set("dataset_text_field", "content")
        _set("num_train_samples", 50000)
        _set("num_eval_samples", 1000)
        _set("num_epochs", 3)
        _set("batch_size", 128)
        _set("log_steps", 2)
        _set("eval_steps", 10)
        _set("target_epsilon", 10.0)
        _set("learning_rate", 5e-5)
        _set("lora_r", 16)
        _set("lora_alpha", 32)
        _set("max_seq_len", 1024)
        _set(
            "lora_modules",
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        _set("dtype", "bfloat16")
        # microbatch_size=16 chunks the per-rank logical batch (=128) into
        # 8 vmap passes of 16 examples each — fits on a single H200 for
        # the 4B LLaMA shape.  auto_find=True halves and retries on an
        # unexpected mid-training CUDA-OOM.
        _set("microbatch_size", 16)
        _set("auto_find_microbatch_size", True)
    elif args.preset == "mellum2-kstack":
        # Mellum2-12B-A2.5B (MoE) + KStack at ε=10. LoRA on attention projections
        # only; the routed experts are stacked nn.Parameter weights PEFT can't adapt.
        _set("model_name", "JetBrains/Mellum2-12B-A2.5B-Base")
        _set("dataset", "JetBrains/KStack")
        _set("dataset_text_field", "content")
        _set("num_train_samples", 50000)
        _set("num_eval_samples", 1000)
        _set("num_epochs", 3)
        _set("batch_size", 128)
        _set("log_steps", 2)
        _set("eval_steps", 10)
        _set("target_epsilon", 10.0)
        _set("learning_rate", 5e-5)
        _set("lora_r", 16)
        _set("lora_alpha", 32)
        _set("max_seq_len", 1024)
        _set("lora_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
        _set("dtype", "bfloat16")
        _set("microbatch_size", 16)
        _set("auto_find_microbatch_size", True)

    _require_configured(parser, args)

    if args.microbatch_size == 0:
        args.microbatch_size = None

    if args.push_to_hub and not args.hub_model_id:
        parser.error("--push-to-hub requires --hub-model-id (e.g. 'org/name')")

    if args.per_group_clipping:
        parsed: dict[str, float] = {}
        fallback_value = None
        for item in args.per_group_clipping:
            if "=" not in item:
                parser.error(
                    f"--per-group-clipping values must be PATTERN=NORM, got '{item}'"
                )
            pattern, value = item.split("=", 1)
            if pattern == "fallback":
                fallback_value = float(value)
            else:
                parsed[pattern] = float(value)
        args.per_group_clipping = parsed
        if fallback_value is not None:
            args.clipping_norm = fallback_value

    _validate_sampler_cli(parser, args)

    return args


def _validate_sampler_cli(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    sampler = args.sampler
    if sampler == "k_out_of_t" and args.k is None:
        parser.error("--sampler k_out_of_t requires --k K")
    if sampler == "k_out_of_t" and args.allocation is None:
        parser.error("--sampler k_out_of_t requires --allocation")
    if sampler == "k_out_of_t":
        if args.noise_mechanism != "gaussian":
            parser.error(
                f"--sampler {sampler} is only supported with --noise-mechanism gaussian"
            )
        if args.max_batch_size is not None:
            parser.error(
                "--max-batch-size (truncated Poisson) is incompatible with "
                f"--sampler {sampler}"
            )
    if args.k is not None and sampler != "k_out_of_t":
        parser.error("--k is only used with --sampler k_out_of_t")
    if args.allocation is not None and sampler != "k_out_of_t":
        parser.error("--allocation is only used with --sampler k_out_of_t")


def _sampling_kwargs_for_trainer(args: argparse.Namespace) -> dict[str, Any]:
    sk: dict[str, Any] = {}
    if args.max_batch_size is not None:
        sk["truncated_batch_size"] = args.max_batch_size
    if args.sampler == "k_out_of_t":
        sk["k"] = int(args.k)
        sk["allocation"] = args.allocation
    return sk


def _resolve_trainer_batching(args: argparse.Namespace) -> int:
    """Return the per-rank logical Poisson batch.

    Trainer no longer supports gradient accumulation; ``--batch-size``
    is the per-rank logical batch (vmap chunk size = logical batch by
    default; ``auto_find_microbatch_size`` may shrink the internal chunk
    on OOM, but the logical batch is privacy-fixed).
    """
    return args.batch_size


def _configure_reporting(args: argparse.Namespace) -> list[str]:
    """Set W&B env defaults and return TrainingArguments.report_to."""
    if args.no_wandb:
        return []

    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    if args.wandb_tags:
        os.environ["WANDB_TAGS"] = ",".join(args.wandb_tags)
    if not os.environ.get("WANDB_MODE"):
        os.environ["WANDB_MODE"] = (
            "online" if os.environ.get("WANDB_API_KEY") else "offline"
        )
    return ["wandb"]


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    if args.eval_batch_size is None:
        args.eval_batch_size = args.microbatch_size or args.batch_size

    print("=" * 80)
    print("DP-SGD LoRA Training for Causal Language Models with Trainer")
    print("=" * 80)

    device, device_label = _select_device()
    dtype_name, torch_dtype, dtype_warning = _resolve_trainer_dtype(args.dtype, device)
    args.dtype = dtype_name
    _print_runtime_mode_report(
        device, device_label, dtype_name, torch_dtype, dtype_warning
    )

    per_rank_logical_batch = _resolve_trainer_batching(args)
    report_to = _configure_reporting(args)

    if args.sdpa_backend is not None and device.type == "cuda":
        backends = {
            "flash": torch.backends.cuda.enable_flash_sdp,
            "efficient": torch.backends.cuda.enable_mem_efficient_sdp,
            "cudnn": torch.backends.cuda.enable_cudnn_sdp,
            "math": torch.backends.cuda.enable_math_sdp,
        }
        for name, setter in backends.items():
            setter(name == args.sdpa_backend)
        print(f"SDPA backend forced: {args.sdpa_backend}")

    print(f"\nLoading model: {args.model_name}...")
    config = AutoConfig.from_pretrained(args.model_name)
    for attr in (
        "attn_pdrop",
        "resid_pdrop",
        "embd_pdrop",
        "attention_dropout",
        "hidden_dropout",
        "dropout",
        "attn_dropout",
        "ffn_dropout",
    ):
        if hasattr(config, attr):
            setattr(config, attr, 0.0)
    if args.gradient_checkpointing and hasattr(config, "use_cache"):
        config.use_cache = False

    use_eager = args.attention == "eager" or device.type == "mps"
    model_kwargs = {
        "config": config,
        "dtype": torch_dtype,
        "trust_remote_code": True,
    }
    if use_eager:
        print("Attention: eager")
        model_kwargs["attn_implementation"] = "eager"
    else:
        backend_label = args.sdpa_backend or "auto"
        print(f"Attention: sdpa (backend={backend_label})")

    start_time = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    except TypeError as exc:
        if "unexpected keyword argument 'dtype'" not in str(exc):
            raise
        model_kwargs.pop("dtype")
        model_kwargs["torch_dtype"] = torch_dtype
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    if getattr(model, "loss_type", None) is None:
        model.loss_type = "ForCausalLM"
    print(f"Model loaded in {time.time() - start_time:.1f}s")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"\nLoading dataset: {args.dataset}...")
    if args.dataset_subset:
        print(f"  Subset: {args.dataset_subset}")
    print(f"  Split: {args.dataset_split}")
    print(f"  Text field: {args.dataset_text_field}")

    total_needed = args.num_train_samples + args.num_eval_samples
    dataset = _load_streaming_subset(
        dataset_name=args.dataset,
        dataset_subset=args.dataset_subset,
        dataset_split=args.dataset_split,
        dataset_text_field=args.dataset_text_field,
        total_needed=total_needed,
    )
    print(f"  Total examples in dataset: {len(dataset)}")

    if len(dataset) > 0:
        sample_text = dataset[0][args.dataset_text_field]
        print("\n  Sample data (first example):")
        print(f"    Text length: {len(sample_text)} chars")
        print(f"    Preview: {sample_text[:200]}...")

    print(
        f"\nPreparing {args.num_eval_samples} eval + "
        f"{args.num_train_samples} train samples..."
    )
    eval_dataset = dataset.take(args.num_eval_samples)
    train_dataset = dataset.skip(args.num_eval_samples).take(args.num_train_samples)

    def tokenize_function(examples):
        return tokenizer(
            examples[args.dataset_text_field],
            truncation=True,
            max_length=args.max_seq_len,
        )

    print(f"\nTokenizing (max_seq_len={args.max_seq_len})...")
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing eval",
    )
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train",
    )
    print(
        f"Prepared datasets: {len(train_dataset)} train samples, "
        f"{len(eval_dataset)} eval samples"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    if args.wandb_run_name is None:
        model_short = args.model_name.split("/")[-1]
        args.wandb_run_name = (
            f"trainer_{model_short}_n{args.num_train_samples}_e{args.num_epochs}_"
            f"b{args.batch_size}_eps{args.target_epsilon}_lr{args.learning_rate}"
        )

    if args.save_strategy is not None:
        save_strategy = args.save_strategy
    else:
        save_strategy = "steps" if args.save_steps is not None else "no"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        report_to=report_to,
        run_name=args.wandb_run_name,
        logging_dir=os.path.join(args.output_dir, "runs"),
        logging_strategy="steps",
        logging_steps=args.log_steps,
        eval_strategy="steps" if args.num_eval_samples > 0 else "no",
        eval_steps=args.eval_steps,
        eval_on_start=args.eval_on_start,
        save_strategy=save_strategy,
        save_steps=args.save_steps or args.eval_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=args.save_only_model,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        max_steps=args.stop_at_step if args.stop_at_step is not None else -1,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=per_rank_logical_batch,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optim=args.optimizer,
        lr_scheduler=args.lr_scheduler,
        lr_scheduler_kwargs=args.lr_scheduler_kwargs,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        data_seed=args.data_seed,
        use_cpu=device.type == "cpu",
        use_mps_device=device.type == "mps",
        bf16=dtype_name == "bfloat16",
        bf16_full_eval=args.bf16_full_eval,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        activation_offloading=args.activation_offloading,
        microbatch_size=args.microbatch_size,
        auto_find_microbatch_size=args.auto_find_microbatch_size,
        torch_compile=args.torch_compile,
        torch_compile_backend=args.torch_compile_backend,
        torch_compile_mode=args.torch_compile_mode,
        use_performance_kernels=args.use_performance_kernels,
        remove_unused_columns=True,
        include_tokens_per_second=True,
        include_num_input_tokens_seen="all",
        privacy_target_epsilon=args.target_epsilon,
        privacy_target_delta=args.target_delta,
        privacy_noise_multiplier=args.noise_multiplier,
        noise_calibration_kwargs={
            "min": args.calibration_min,
            "max": args.calibration_max,
            "tolerance": args.calibration_tolerance,
        },
        clipping_mode=args.clipping_mode,
        clipping_norm=(
            {"fallback": float(args.clipping_norm), **dict(args.per_group_clipping)}
            if args.per_group_clipping
            else args.clipping_norm
        ),
        clipping_kwargs=(
            {
                "target_quantile": args.target_clipping_rate,
                "clipping_norm_max": args.clipping_norm_max,
            }
            if args.clipping_mode == "adaptive"
            else {"gamma": args.auto_clipping_gamma}
            if args.clipping_mode == "auto"
            else {}
        ),
        sampling_mode=args.sampler,
        sampling_kwargs=_sampling_kwargs_for_trainer(args),
        privacy_noise_mechanism=args.noise_mechanism,
        privacy_noise_mechanism_kwargs=args.noise_mechanism_kwargs or {},
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_private_repo=args.hub_private_repo,
        hub_revision=args.hub_revision,
    )

    # Opake's ``TrainingArguments`` is a standalone dataclass and does
    # NOT declare ``max_grad_norm`` — HF's per-step clip_grad_norm_ is
    # replaced by the per-example DP clipping path.  We still let users
    # pass ``--max-grad-norm`` to validate that the field is inert: we
    # stamp it onto the args post-construction so any downstream code
    # path that read ``args.max_grad_norm`` would surface it.  Trainer
    # currently has zero references to it, so ε must remain bit-exact
    # against a baseline that did not set the flag.
    if args.max_grad_norm is not None:
        object.__setattr__(training_args, "max_grad_norm", args.max_grad_norm)

    print("\nTrainer argument summary:")
    print(f"  Output dir: {training_args.output_dir}")
    print(f"  Optimizer: {training_args.optim}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Clipping mode: {training_args.clipping_mode}")
    print(f"  Clip norm (clipping_norm): {training_args.clipping_norm}")
    print(f"  Noise mechanism: {training_args.privacy_noise_mechanism}")
    print(f"  Sampling mode: {training_args.sampling_mode}")
    if training_args.sampling_kwargs:
        print(f"  Sampling kwargs: {training_args.sampling_kwargs}")
    print(f"  Target epsilon: {training_args.privacy_target_epsilon}")
    print(f"  Target delta: {training_args.privacy_target_delta or 'auto'}")
    print(f"  Save strategy: {training_args.save_strategy}")
    print(f"  torch_compile: {training_args.torch_compile}")
    if training_args.torch_compile:
        print(
            f"    backend={training_args.torch_compile_backend or 'inductor'} "
            f"mode={training_args.torch_compile_mode or 'default'}"
        )
    print(f"  bf16_full_eval: {training_args.bf16_full_eval}")
    print(f"  use_performance_kernels: {training_args.use_performance_kernels}")
    print(f"  Push to Hub: {training_args.push_to_hub}")
    if training_args.push_to_hub:
        print(f"  Hub model id: {training_args.hub_model_id}")
        print(f"  Hub private repo: {training_args.hub_private_repo}")
        print(f"  Hub revision: {training_args.hub_revision or 'main'}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if args.num_eval_samples > 0 else None,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # Resolve --resume-from-checkpoint: 'auto'/'true' -> bool True (let the
    # trainer auto-detect the latest checkpoint under output_dir); a path
    # string is passed through verbatim; absent -> None (fresh run).
    resume_arg: str | bool | None = None
    if args.resume_from_checkpoint is not None:
        token = args.resume_from_checkpoint.strip().lower()
        if token in ("auto", "true", "latest"):
            resume_arg = True
        elif token in ("false", "no", "none", ""):
            resume_arg = None
        else:
            resume_arg = args.resume_from_checkpoint

    print("\n" + "=" * 80)
    print("Starting Trainer training...")
    print(f"  resume_from_checkpoint: {resume_arg!r}")
    print("=" * 80)
    train_output = trainer.train(resume_from_checkpoint=resume_arg)

    print("\n" + "=" * 80)
    print("Training Complete")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset} ({len(train_dataset)} train samples)")
    print("\nTraining metrics:")
    for key_name in sorted(train_output.metrics):
        print(f"  {key_name}: {train_output.metrics[key_name]}")

    if args.num_eval_samples > 0:
        print("\nRunning final evaluation...")
        eval_metrics = trainer.evaluate()
        for key_name in sorted(eval_metrics):
            print(f"  {key_name}: {eval_metrics[key_name]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
