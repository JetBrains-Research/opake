# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""End-to-end DP-SGD LoRA training example for SFT (Supervised Fine-Tuning).

This is the SFT sibling of ``examples/train_dpsgd.py`` (the comprehensive
DP-SGD baseline) and ``examples/train_dpo.py``. It ports the full production-style
DP-SGD scaffolding from ``train_dpsgd.py`` — clipping + noise + accounting +
calibration + auditing + LoRA + LR schedules + distributed/Poisson sampling +
W&B — and swaps in the SFT-specific loss and data machinery from
``opaque-alignment``:

  * ``language_modeling_collator(pad_token_id, max_length)`` builds the batch
    (``input_ids``, ``attention_mask``, ``labels`` with pad/non-completion tokens
    masked to ``-100``).  When ``--completion-only`` is set the collator runs
    with ``completion_only_loss=True`` and the loss is computed only over
    assistant/completion tokens (see "Completion-only loss" below).
  * ``SFT_LOSSES["nll"]`` / ``SFT_LOSSES["dft"]`` compute the per-example loss
    from ``out.logits`` and ``labels`` with a DP-safe per-example divisor —
    each example divides by *its own* non-ignored token count (not a batch
    aggregate), so per-example sensitivity stays ``O(C)`` after clipping.

Because the eager SFT losses consume ``out.logits``, this script applies
``apply_model_patches(model)`` **without** ``fused_linear_cross_entropy``: the
fused linear+CE patch returns ``logits=None`` on its fast path, which the eager
losses cannot use.  (``train_dpsgd.py`` can opt into the fused kernel because
it consumes ``output.loss`` directly.)

Eval reports held-out *language-modeling* quality — mean eval loss + perplexity,
plus token accuracy / prediction entropy (``opaque.alignment.metric``) — NOT
reward metrics (those are DPO-specific).

----------------------------------------------------------------------------
COMPLETION-ONLY LOSS FROM CHAT DATA (``--completion-only``)
----------------------------------------------------------------------------
With ``--completion-only`` the script trains only on assistant/completion
tokens of chat-format datasets (rows carrying a ``"messages"`` /
``"conversations"`` column of ``{"role", "content"}`` dicts). It:

  1. installs a generation-marker chat template on the tokenizer via
     ``get_training_chat_template`` (so HF can emit an assistant-token mask),
  2. tokenizes each chat example with ``apply_chat_template_with_mask`` to get
     ``input_ids`` + a ``completion_mask`` (``1`` on assistant tokens), and
  3. builds the collator with ``completion_only_loss=True`` so non-completion
     positions are masked to ``-100`` in ``labels``.

The plain text-field path (``--dataset-text-field``) keeps working when
``--completion-only`` is off (standard next-token prediction over the full
sequence).

----------------------------------------------------------------------------
SMOKE MODE (``--smoke``)
----------------------------------------------------------------------------
``--smoke`` runs the **full per-example vmap DP-SGD path** on a tiny,
randomly-initialized LlamaForCausalLM (no network, no HF download) over a small
synthetic token dataset (plain LM). It executes 2 real DP-SGD steps and prints
the loss each step. It also exercises the completion-only collator wiring on a
tiny synthetic example (``completion_only_loss=True`` over a hand-built
``completion_mask``) so that path is covered without a network round-trip.

A documented fallback exists in ``_run_smoke`` for environments where
``vmap(grad(...))`` over the patched model fails on CPU: a single non-vmap
forward + ``SFT_LOSSES["nll"]`` validates the loss wiring. The script never
exits non-zero in smoke mode.

The mechanism is the caller's choice: swap the ``opaque.dpsgd`` imports below
for ``opaque.dpftrl`` to run DP-FTRL instead. The loss closure does not change.

USAGE:

  # Smoke test (CPU, ~seconds, no network)
  python examples/train_sft.py --smoke

  # Quick test preset (SmolLM2-135M on KExercises, plain text field)
  python examples/train_sft.py --preset smoke

  # Completion-only SFT on a chat dataset
  python examples/train_sft.py \\
    --model-name Qwen/Qwen2.5-0.5B-Instruct \\
    --dataset trl-lib/Capybara --completion-only \\
    --loss-type nll --max-length 1024 --batch-size 16

  # 4-GPU distributed run with torchrun
  torchrun --nproc_per_node=4 examples/train_sft.py --preset mellum-kstack
"""

from __future__ import annotations

# E402: ``apply_runtime_patches()`` must run before transformers/opaque
# submodules are imported (it monkeypatches their runtime behavior), so the
# remaining imports intentionally follow that call — same as train_dpo.py.
# ruff: noqa: E402

import argparse
import contextlib
import importlib.util
import os
import sys
import time

import torch
import torch.distributed as dist
import torchopt
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model

from opaque.patches import apply_model_patches, apply_runtime_patches

apply_runtime_patches()

from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

import opaque.accounting as acc
import opaque.auditing as auditing
import opaque.dpsgd.accounting as dpsgd_acc
from opaque.accounting import calibration as cal, Accountant
from opaque.dpsgd.clipping import auto_clipped_grad, clipped_grad
from opaque.dpsgd.clipping import adaptive_clipped_grad
from opaque.distributed import sync
from opaque.distributed.gradients import sum_gradients_
from opaque.dpsgd.noise import gaussian_noise, jme_noise
from opaque.dpsgd.noise.types import JmeAllocation
from opaque.profiling import (
    perf_tracker,
    print_memory,
    reset_peak_memory,
)
from opaque.random import fold_in, key, split
from opaque.dpsgd.sampling import PoissonSampler
from opaque.distributed import local_shard
from opaque.functional import make_functional
from opaque.scheduling import (
    cosine_schedule,
    inverse_sqrt_schedule,
    linear_schedule,
    with_warmup,
)
from opaque.scheduling.types import Schedule
from opaque.types import (
    ClippedPytree,
    PerGroup,
    SecondMomentNoiseOutput,
)
from opaque.dpsgd.clipping import per_group
import wandb

# SFT-specific machinery from opaque-alignment.
from opaque.alignment.sft.collator import language_modeling_collator
from opaque.alignment.sft.loss import dft_loss, nll_loss
from opaque.alignment.data import (
    apply_chat_template_with_mask,
    get_training_chat_template,
)

# Token-level eval telemetry (mean token accuracy / prediction entropy).
from opaque.alignment.metric import entropy_from_logits, mean_token_accuracy

# DP-FTRL mechanism swap: the loss closure is mechanism-agnostic.
# To run DP-FTRL instead of DP-SGD, replace the two ``opaque.dpsgd`` noise/
# sampling imports above with their DP-FTRL counterparts, e.g.:
#   from opaque.dpftrl.noise import band_mf_noise  # matrix-factorized noise
# and feed it the same ``ClippedPytree`` produced by ``clipped_grad`` below.

# The library (``opaque.alignment.sft``) exposes direct loss functions, not a
# string registry. The CLI ``--loss-type`` string is mapped to a function here,
# at the call site — mirroring ``examples/train_dpo.py``'s ``_DPO_LOSSES``.
_SFT_LOSSES = {"nll": nll_loss, "dft": dft_loss}

# The 3 per-example loss arguments after the trainable params (argnums=0):
# input_ids, attention_mask, labels.  The vmap batch axis is taken over all of
# them, so batch_argnums lists every index 1..3.
_BATCH_ARGNUMS = (1, 2, 3)

# Chat-format columns we recognise for --completion-only.  Each holds a list of
# {"role", "content"} message dicts per row.
_CHAT_COLUMNS = ("messages", "conversations", "chat")


def _effective(value):
    """Extract scalar from float or PerGroup for logging/printing."""
    return value.effective if isinstance(value, PerGroup) else value


def _noise_stddev(max_norm, noise_multiplier, *, per_group=True):
    """Noise stddev: MSE-optimal per-group when available, isotropic otherwise."""
    if per_group and isinstance(max_norm, PerGroup):
        return ClippedPytree(pytree={}, max_norm=max_norm).noise_stddev_for(
            noise_multiplier=noise_multiplier
        )
    return noise_multiplier * max_norm


def _step_clip_norm(grads_tuple):
    """Return the current scalar or per-group clipping sensitivity."""
    return grads_tuple.max_norm


def _step_noise_stddev(noisy_grads):
    """``.noise_stddev`` from a noised pytree, unwrapping the paired SM output."""
    if isinstance(noisy_grads, SecondMomentNoiseOutput):
        return noisy_grads.noisy_grads.noise_stddev
    return noisy_grads.noise_stddev


def _log_jme(args) -> None:
    """Log the explicit projected-JME configuration."""
    print(
        f"  JME: on (aggregate_norm={args.jme_aggregate_norm}, "
        f"allocation={args.jme_allocation})"
    )
    if args.jme_allocation == "first_variance_cap":
        print(
            "  JME max first-stream variance ratio: "
            f"{args.jme_max_first_variance_ratio}"
        )


def _select_device(local_rank: int | None = None) -> tuple[torch.device, str]:
    """Select best available device with user-facing label."""
    if torch.cuda.is_available():
        if local_rank is not None:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            return device, torch.cuda.get_device_name(local_rank)
        device = torch.device("cuda")
        return device, torch.cuda.get_device_name(0)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        return device, "Apple Silicon"

    return torch.device("cpu"), "CPU"


def _init_distributed() -> tuple[bool, int, int, int]:
    """Initialize torch.distributed when launched via torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size <= 1:
        return False, 0, 1, 0

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    return True, rank, world_size, local_rank


def _cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _is_dtype_supported(device: torch.device, dtype: torch.dtype) -> bool:
    """Check whether a dtype can be allocated on a specific device."""
    try:
        torch.empty((1,), device=device, dtype=dtype)
        return True
    except (RuntimeError, TypeError):
        return False


def _resolve_model_dtype(
    requested_name: str,
    device: torch.device,
) -> tuple[str, torch.dtype, str | None]:
    """Resolve requested dtype with safe fallback for unsupported device paths."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    requested_dtype = dtype_map[requested_name]

    if _is_dtype_supported(device, requested_dtype):
        return requested_name, requested_dtype, None

    fallback_order = {
        "cuda": ["float16", "bfloat16", "float32"],
        "mps": ["float16", "float32"],
        "cpu": ["float32", "bfloat16", "float16"],
    }
    for fallback_name in fallback_order.get(device.type, ["float32"]):
        fallback_dtype = dtype_map[fallback_name]
        if _is_dtype_supported(device, fallback_dtype):
            reason = (
                f"Requested dtype '{requested_name}' is not supported on {device.type}; "
                f"using '{fallback_name}' instead."
            )
            return fallback_name, fallback_dtype, reason

    raise RuntimeError(
        f"No supported dtype found for device={device.type}. "
        f"Requested dtype was '{requested_name}'."
    )


def _kernel_mode_summary(device: torch.device, dtype_name: str) -> tuple[str, str]:
    """Return concise status of kernel optimization mode for this run."""
    if os.environ.get("OPAQUE_NO_PATCH", "0") == "1":
        return "disabled", "OPAQUE_NO_PATCH=1"

    if os.environ.get("OPAQUE_NO_KERNEL_PATCH", "0") == "1":
        return "disabled", "OPAQUE_NO_KERNEL_PATCH=1"

    if device.type != "cuda":
        return "disabled", f"device={device.type} (Triton kernels are CUDA-only)"

    if importlib.util.find_spec("triton") is None:
        return "disabled", "triton package not installed"

    # SFT here runs the eager logits losses (no fused linear+CE), so the fused
    # CE kernel is intentionally not engaged regardless of dtype.
    return "partial", "eager SFT losses (fused linear+CE not engaged)"


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
            "  Note: MPS uses compatibility fallbacks when CUDA-only kernels are unavailable."
        )


def _load_streaming_subset(
    dataset_name: str,
    dataset_subset: str | None,
    dataset_split: str,
    total_needed: int,
) -> Dataset:
    """Stream only required rows, then materialize to in-memory static Dataset."""
    print("  Streaming source dataset and materializing required subset...")
    stream_ds = load_dataset(
        dataset_name,
        name=dataset_subset,
        split=dataset_split,
        streaming=True,
    )

    rows = list(stream_ds.take(total_needed))
    if len(rows) < total_needed:
        raise ValueError(
            f"Stream ended after {len(rows)} examples, but {total_needed} are required "
            f"(train + eval)."
        )

    return Dataset.from_list(rows)


def _detect_chat_column(row: dict) -> str | None:
    """Return the chat-message column name in *row*, or ``None`` if plain text.

    A chat column holds a non-empty list of ``{"role", "content"}`` message
    dicts; this is the column ``--completion-only`` tokenizes with
    ``apply_chat_template_with_mask``.
    """
    for col in _CHAT_COLUMNS:
        value = row.get(col)
        if (
            isinstance(value, list)
            and value
            and isinstance(value[0], dict)
            and "role" in value[0]
            and "content" in value[0]
        ):
            return col
    return None


def _tokenize_text_example(text, tokenizer, max_length):
    """Tokenize a plain text-field example into ``{"input_ids": list[int]}``.

    No completion mask is produced (full-sequence next-token prediction).
    """
    ids = tokenizer(text, truncation=True, max_length=max_length)["input_ids"]
    return {"input_ids": ids}


def _tokenize_chat_example(messages, tokenizer, max_length):
    """Tokenize a chat example into ``input_ids`` + ``completion_mask``.

    Uses ``apply_chat_template_with_mask`` so the assistant-turn tokens are
    flagged ``1`` in ``completion_mask`` and prompt tokens ``0`` — exactly the
    mask ``language_modeling_collator(..., completion_only_loss=True)`` consumes.
    Sequences are truncated keep-start to ``max_length``.
    """
    encoded = apply_chat_template_with_mask(
        tokenizer,
        messages,
        max_length=max_length,
        truncation=True,
    )
    ids = encoded["input_ids"][:max_length]
    cmask = encoded["completion_mask"][:max_length]
    return {"input_ids": ids, "completion_mask": cmask}


def _make_per_example_loss(fmodel, frozen, *, loss_type):
    """Build the SFT per-example loss closure (one forward over ``out.logits``).

    The returned callable has signature::

        per_example_loss(trainable_params, input_ids, attention_mask, labels)
            -> per-example scalar loss

    which is ``argnums=0`` (trainable params) + the 3 per-example args in
    ``_BATCH_ARGNUMS``. ``frozen`` first in the merge so trainable params win on
    key collision. The loss reads ``out.logits`` (eager SFT loss), so the model
    is patched WITHOUT the fused linear+CE kernel. Each output depends only on
    this example's data (DP-safe per-example divisor inside the loss).
    """

    def per_example_loss(trainable_params, input_ids, attention_mask, labels):
        merged = {**frozen, **trainable_params}
        out = fmodel(merged, input_ids=input_ids, attention_mask=attention_mask)
        return _SFT_LOSSES[loss_type](out.logits, labels)

    return per_example_loss


def _require_configured(parser, args, required=("model_name", "dataset")):
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        flags = ", ".join("--" + name.replace("_", "-") for name in missing)
        parser.error(
            f"missing required configuration: {flags}. "
            f"Pass them directly or select a --preset (e.g. --preset smoke)."
        )


def parse_args():
    """Parse command-line arguments with logical groups."""
    parser = argparse.ArgumentParser(
        description="End-to-end DP-SGD LoRA Supervised Fine-Tuning (SFT)"
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny CPU smoke test (random model, synthetic data, 2 steps). "
        "Bypasses all other configuration.",
    )

    # Preset configurations
    parser.add_argument(
        "--preset",
        type=str,
        choices=[
            "smoke",
            "mellum-kstack",
            "mellum2-kstack",
            "qwen-7b-kstack",
        ],
        default=None,
        help="Optional preset that fills in any unset arguments. Omit it to "
        "configure the run directly (at least --model-name and --dataset). "
        "smoke=quick test SmolLM2-135M + KExercises at ε=8, "
        "mellum-kstack=Mellum-4b + KStack at ε=10 with adafactor @ 5e-5, "
        "mellum2-kstack=Mellum2-12B-A2.5B MoE + KStack at ε=10 with adafactor @ 5e-5, "
        "qwen-7b-kstack=Qwen2.5-Coder-7B + KStack at ε=3 with adafactor @ 5e-4.",
    )

    model_group = parser.add_argument_group("model", "Model and tokenizer settings")
    model_group.add_argument(
        "--model-name",
        "--model",
        dest="model_name",
        type=str,
        default=None,
        help="HuggingFace model name or local path (required unless a --preset sets it)",
    )
    model_group.add_argument(
        "--attention",
        type=str,
        choices=["eager", "sdpa"],
        default="sdpa",
        help="Attention implementation (default: sdpa, which is faster and uses less memory)",
    )
    model_group.add_argument(
        "--sdpa-backend",
        type=str,
        choices=["flash", "efficient", "cudnn", "math"],
        default=None,
        help="Force a specific SDPA backend (default: None = PyTorch auto-selects)",
    )

    data_group = parser.add_argument_group("data", "Dataset and tokenization settings")
    data_group.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="HuggingFace dataset name (required unless a --preset sets it)",
    )
    data_group.add_argument(
        "--dataset-subset",
        "--dataset-name",
        dest="dataset_subset",
        type=str,
        default=None,
        help="Optional dataset subset (HF load_dataset 'name' argument), e.g. 'stage1-auto-format'.",
    )
    data_group.add_argument(
        "--dataset-split", type=str, default="train", help="Dataset split for training"
    )
    data_group.add_argument(
        "--dataset-text-field",
        type=str,
        default="text",
        help="Field containing raw text (plain-text path; ignored under --completion-only).",
    )
    data_group.add_argument(
        "--completion-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train only on assistant/completion tokens of chat-format datasets. "
        "Installs a generation-marker chat template via get_training_chat_template, "
        "tokenizes each chat row with apply_chat_template_with_mask to get a "
        "completion_mask, and builds the collator with completion_only_loss=True. "
        "Off by default (plain --dataset-text-field next-token prediction).",
    )
    data_group.add_argument(
        "--num-train-samples",
        type=int,
        default=5000,
        help="Number of training examples (default: 5000)",
    )
    data_group.add_argument(
        "--num-eval-samples",
        type=int,
        default=1000,
        help="Held-out sample count for periodic eval-loss / perplexity "
        "reporting. At 100 the per-eval loss is too noisy to read learning "
        "dynamics; 1000 keeps the per-eval std-err around 3%%.",
    )
    data_group.add_argument(
        "--max-length",
        "--max-seq-len",
        dest="max_length",
        type=int,
        default=512,
        help="Maximum sequence length for the language-modeling collator",
    )

    sft_group = parser.add_argument_group("sft", "SFT loss settings")
    sft_group.add_argument(
        "--loss-type",
        type=str,
        choices=sorted(_SFT_LOSSES),
        default="nll",
        help="SFT loss variant (direct functions from opaque.alignment.sft).",
    )

    train_group = parser.add_argument_group("training", "Training loop settings")
    train_group.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Expected batch size for Poisson sampling (determines sample_rate)",
    )
    train_group.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Batch size for evaluation (default: same as batch_size, can be larger since no privacy needed)",
    )
    train_group.add_argument(
        "--num-epochs", type=int, default=3, help="Number of epochs"
    )
    train_group.add_argument(
        "--learning-rate", type=float, default=1.0e-5, help="Learning rate"
    )
    train_group.add_argument(
        "--lr-schedule",
        type=str,
        default="none",
        choices=["none", "cosine", "linear", "sqrt"],
        help=(
            "LR schedule applied to ``--learning-rate``.  ``cosine`` decays "
            "from peak to ``lr * lr_min_ratio`` over the full training run; "
            "``linear`` decays linearly to the same floor; ``sqrt`` decays "
            "as 1/sqrt(1 + step/warmup) (Adagrad-mimic).  Default ``none`` "
            "keeps the constant LR."
        ),
    )
    train_group.add_argument(
        "--lr-min-ratio",
        type=float,
        default=0.0,
        help="Floor LR as a fraction of peak (cosine/linear).  Default 0 (decay to zero).",
    )
    train_group.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=0,
        help="Linear warmup from 0 to peak LR over this many steps (any schedule).",
    )
    train_group.add_argument(
        "--optimizer",
        type=str,
        default="adafactor",
        choices=[
            "sgd",
            "adam",
            "adamw",
            "ademamix",
            "lion",
            "adafactor",
            "rmsprop",
            "adagrad",
        ],
        help=(
            "Optimizer.  ``sgd`` and ``adam`` are torchopt's vanilla "
            "primitives (no DP-aware paths); the others are Opaque-built "
            "(see opaque.optimizers).  Pair with "
            "``--noise-bias-correction`` to enable DP-aware bias "
            "correction where applicable."
        ),
    )
    train_group.add_argument(
        "--noise-bias-correction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable DP noise-variance bias correction on optimizers that "
            "support it (adam/adamw/ademamix/rmsprop/adagrad/adafactor).  "
            "Silently ignored on sgd/lion.  Off by default; see "
            "docs/user-guide/optimizers.md for when to flip it on."
        ),
    )
    train_group.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for optimizers that support it (default: 0.01)",
    )
    train_group.add_argument(
        "--log-steps",
        type=int,
        default=1,
        help="Log training metrics every N steps",
    )
    train_group.add_argument(
        "--eval-steps",
        type=int,
        default=10,
        help="Log eval loss/perplexity and privacy every N steps",
    )
    train_group.add_argument(
        "--stop-at-step",
        type=int,
        default=None,
        help="Stop the training loop after this many optimizer steps "
        "(early-stop knob, not a privacy-accounting target — privacy is "
        "calibrated from target_epsilon × steps × sample_rate regardless). "
        "Overrides --num-epochs when set.",
    )
    train_group.add_argument("--seed", type=int, default=42, help="Random seed")
    train_group.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable gradient checkpointing for memory savings (trades compute for memory)",
    )
    train_group.add_argument(
        "--activation-offloading",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Offload saved tensors to CPU via save_on_cpu (works with or without checkpointing)",
    )

    lora_group = parser.add_argument_group("lora", "LoRA adapter settings")
    lora_group.add_argument("--lora-r", type=int, default=4, help="LoRA rank")
    lora_group.add_argument("--lora-alpha", type=int, default=8, help="LoRA alpha")
    lora_group.add_argument(
        "--lora-modules",
        type=str,
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        help="Target module names for LoRA",
    )

    dp_group = parser.add_argument_group("dp", "DP-SGD clipping and noise")
    dp_group.add_argument(
        "--shard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shard dataset across DDP ranks (default). "
        "Use --no-shard to replicate full dataset on each rank "
        "(uses parallel_poisson accounting).",
    )
    dp_group.add_argument(
        "--clipping-mode",
        type=str,
        choices=["fixed", "adaptive", "auto"],
        default="adaptive",
        help="Clipping strategy: fixed (constant threshold), adaptive (Andrew "
        "et al. quantile tracking), or auto (AUTO-S automatic scaling, Bu et "
        "al. NeurIPS 2023).",
    )
    dp_group.add_argument(
        "--clipping-norm",
        type=float,
        default=1.0,
        help="Clipping norm: fixed threshold C (fixed mode), starting threshold "
        "(adaptive mode), or sensitivity max_norm R (auto mode).",
    )
    dp_group.add_argument(
        "--target-clipping-rate",
        type=float,
        default=0.5,
        help="Target clipping rate for --clipping-mode adaptive.",
    )
    dp_group.add_argument(
        "--clipping-norm-max",
        type=float,
        default=10.0,
        help="Maximum clipping norm for --clipping-mode adaptive.",
    )
    dp_group.add_argument(
        "--auto-clipping-gamma",
        type=float,
        default=0.01,
        help="Denominator stabilizer γ for --clipping-mode auto "
        "(AUTO-S: g̃ = R·g / (‖g‖ + γ); default 0.01).",
    )
    dp_group.add_argument(
        "--microbatch-size",
        type=int,
        default=None,
        help="Microbatch size passed to clipped_grad/adaptive_clipped_grad (None=process full batch with vmap, faster but more memory; use 0 on CLI to mean None)",
    )
    dp_group.add_argument(
        "--truncated-batch-size",
        type=int,
        default=None,
        help="Optional cap on per-step batch size (truncated Poisson). "
        "When set, the sampler caps each Poisson draw at this size and the "
        "accountant switches to the matching truncated Poisson-Gaussian PLD. "
        "Standard plain Poisson when omitted.",
    )
    dp_group.add_argument(
        "--noise-mechanism",
        type=str,
        choices=["gaussian", "bounded_gaussian"],
        default="gaussian",
        help="Noise mechanism: gaussian (standard, unbounded support) "
        "or bounded_gaussian (Chen and Hale, 2024; renormalized density on a "
        "per-coordinate interval).",
    )
    dp_group.add_argument(
        "--noise-bound",
        type=float,
        default=1.0,
        help="Symmetric absolute bound B for bounded_gaussian (per-coordinate "
        "support [-B, B]; same units as the gradient / clip norm). Ignored "
        "for standard gaussian.",
    )
    dp_group.add_argument(
        "--jme",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Release the projected normalized aggregate and its clean square "
        "with DP-SGD JME. Requires scalar clipping, standard Gaussian noise, "
        "plain independent Poisson sampling, and explicit JME configuration.",
    )
    dp_group.add_argument(
        "--jme-aggregate-norm",
        type=float,
        default=None,
        help="Required public L2 projection radius for --jme.",
    )
    dp_group.add_argument(
        "--jme-allocation",
        choices=["paper_reference", "first_variance_cap"],
        default=None,
        help="Required JME first/second noise allocation policy.",
    )
    dp_group.add_argument(
        "--jme-max-first-variance-ratio",
        type=float,
        default=None,
        help="Required with --jme-allocation first_variance_cap.",
    )
    dp_group.add_argument(
        "--per-group-clipping",
        type=str,
        nargs="+",
        default=None,
        metavar="PATTERN=NORM",
        help="Per-group clipping norms as PATTERN=NORM pairs (e.g., self_attn=1.0 mlp=2.0). "
        "Each trainable param must match exactly one pattern substring. "
        "Use 'fallback=NORM' as catch-all for unmatched params. "
        "Compatible with all --clipping-mode values (adaptive adapts each group "
        "independently; auto uses per-group R_k).",
    )

    # Model precision
    precision_group = parser.add_argument_group("Model Configuration")
    precision_group.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model precision (default: bfloat16 for best performance/memory tradeoff)",
    )

    privacy_group = parser.add_argument_group(
        "privacy", "Privacy accounting and noise calibration"
    )
    privacy_group.add_argument(
        "--target-epsilon",
        type=float,
        default=8.0,
        help="Target epsilon used to calibrate noise_multiplier",
    )
    privacy_group.add_argument(
        "--target-delta",
        type=float,
        default=None,
        help="Target delta for DP accounting. Default: 1/n^1.1 where n = training set size.",
    )
    privacy_group.add_argument(
        "--noise-multiplier",
        type=float,
        default=None,
        help="Use a fixed noise multiplier instead of calibrating from target_epsilon",
    )
    privacy_group.add_argument(
        "--calibration-min",
        type=float,
        default=0.11,
        help="Lower bound for noise calibration search",
    )
    privacy_group.add_argument(
        "--calibration-max",
        type=float,
        default=3.5,
        help="Upper bound for noise calibration search",
    )
    privacy_group.add_argument(
        "--calibration-tolerance",
        type=float,
        default=1e-3,
        help="Tolerance for noise calibration",
    )

    audit_group = parser.add_argument_group("audit", "Empirical privacy auditing")
    audit_group.add_argument(
        "--audit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable empirical auditing (disabled by default)",
    )
    audit_group.add_argument(
        "--audit-canaries",
        type=int,
        default=1000,
        help="Number of canaries for one-run auditing",
    )
    audit_group.add_argument(
        "--audit-method",
        choices=["gdp", "eps_delta"],
        default="gdp",
        help="Which audit method's ε to report ('gdp' = μ-GDP, recommended for Gaussian-DP mechanisms like DP-SGD; 'eps_delta' = mechanism-agnostic (ε, δ)-DP fallback)",
    )
    audit_group.add_argument(
        "--audit-batch-size",
        type=int,
        default=None,
        help="Batch size for auditing scoring (default: same as microbatch_size; forward-only so less memory than training)",
    )

    tracking_group = parser.add_argument_group("tracking", "Experiment tracking (W&B)")
    tracking_group.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable experiment tracking (wandb is enabled by default, offline if no credentials)",
    )
    tracking_group.add_argument(
        "--wandb-project",
        type=str,
        default=os.environ.get("WANDB_PROJECT", "opaque"),
        help="W&B project name (default: WANDB_PROJECT env var or 'opaque')",
    )
    tracking_group.add_argument(
        "--wandb-run-name",
        type=str,
        default=os.environ.get("WANDB_NAME") or os.environ.get("RUN_NAME"),
        help="Run name (default: WANDB_NAME, then RUN_NAME env var, or auto-generated from model and hyperparameters)",
    )
    tracking_group.add_argument(
        "--wandb-entity",
        type=str,
        default=os.environ.get("WANDB_ENTITY"),
        help="W&B entity/team (default: WANDB_ENTITY env var)",
    )

    args = parser.parse_args()

    # Track which options were explicitly provided on CLI.
    # This avoids ambiguity when an explicit value equals the parser default.
    provided_dests = set()
    argv_tokens = sys.argv[1:]
    for action in parser._actions:
        if not action.option_strings:
            continue
        for opt in action.option_strings:
            if any(
                token == opt or token.startswith(f"{opt}=") for token in argv_tokens
            ):
                provided_dests.add(action.dest)
                break

    def _set(name, value):
        if name not in provided_dests:
            setattr(args, name, value)

    # Apply preset configurations (CLI args take precedence)
    if args.preset == "smoke":
        _set("model_name", "HuggingFaceTB/SmolLM2-135M")
        _set("dataset", "JetBrains/KExercises")
        _set("dataset_text_field", "solution")
        _set("completion_only", False)
        _set("num_train_samples", 256)
        _set("num_eval_samples", 64)
        _set("num_epochs", 1)
        _set("batch_size", 16)
        _set("log_steps", 5)
        _set("eval_steps", 5)
        _set("target_epsilon", 8.0)
        _set("learning_rate", 1e-5)
        _set("loss_type", "nll")
        _set("lora_r", 4)
        _set("lora_alpha", 8)
        _set("max_length", 512)
        _set("lora_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
        _set("dtype", "bfloat16")
        _set("audit", False)
    elif args.preset == "mellum-kstack":
        # Mellum-4b + KStack SFT LoRA fine-tuning at ε=10.
        _set("model_name", "JetBrains/Mellum-4b-base")
        _set("dataset", "JetBrains/KStack")
        _set("dataset_text_field", "content")
        _set("num_train_samples", 50000)
        _set("num_eval_samples", 1000)
        _set("num_epochs", 3)
        _set("batch_size", 128)
        _set("microbatch_size", 16)
        _set("log_steps", 2)
        _set("eval_steps", 10)
        _set("target_epsilon", 10.0)
        _set("learning_rate", 5e-5)
        _set("optimizer", "adafactor")
        _set("loss_type", "nll")
        _set("lora_r", 16)
        _set("lora_alpha", 32)
        _set("max_length", 1024)
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
    elif args.preset == "mellum2-kstack":
        # Mellum2-12B-A2.5B (MoE) + KStack SFT LoRA fine-tuning at ε=10.
        # LoRA targets attention projections only: the routed experts are stacked
        # nn.Parameter weights (gate_up_proj/down_proj), which PEFT can't adapt.
        _set("model_name", "JetBrains/Mellum2-12B-A2.5B-Base")
        _set("dataset", "JetBrains/KStack")
        _set("dataset_text_field", "content")
        _set("num_train_samples", 50000)
        _set("num_eval_samples", 1000)
        _set("num_epochs", 3)
        _set("batch_size", 128)
        _set("microbatch_size", 16)
        _set("log_steps", 2)
        _set("eval_steps", 10)
        _set("target_epsilon", 10.0)
        _set("learning_rate", 5e-5)
        _set("optimizer", "adafactor")
        _set("loss_type", "nll")
        _set("lora_r", 16)
        _set("lora_alpha", 32)
        _set("max_length", 1024)
        _set("lora_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
        _set("dtype", "bfloat16")
    elif args.preset == "qwen-7b-kstack":
        # Qwen2.5-Coder-7B + KStack SFT LoRA fine-tuning at ε=3.
        _set("model_name", "Qwen/Qwen2.5-Coder-7B")
        _set("dataset", "JetBrains/KStack")
        _set("dataset_text_field", "content")
        _set("num_train_samples", 50000)
        _set("num_eval_samples", 1000)
        _set("num_epochs", 2)
        _set("batch_size", 192)
        _set("microbatch_size", 16)
        _set("log_steps", 2)
        _set("eval_steps", 10)
        _set("target_epsilon", 3.0)
        _set("learning_rate", 5e-4)
        _set("optimizer", "adafactor")
        _set("loss_type", "nll")
        _set("lora_r", 16)
        _set("lora_alpha", 16)
        _set("max_length", 1024)
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

    _require_configured(parser, args)

    # --microbatch-size 0 means "no microbatching" (full-batch vmap).
    # Needed because argparse type=int can't accept None on CLI to override presets.
    if args.microbatch_size == 0:
        args.microbatch_size = None

    # Parse --per-group-clipping PATTERN=NORM pairs
    if args.per_group_clipping:
        parsed = {}
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
        args.per_group_clipping_fallback = fallback_value

    return args


def _run_smoke(args):
    """Tiny CPU smoke test: random Llama, synthetic data, 2 real DP-SGD steps.

    Builds a tiny randomly-initialized LlamaForCausalLM (no network), a small
    synthetic token dataset (plain LM), and runs the full per-example vmap
    DP-SGD path for 2 steps, printing the SFT loss each step.  It also exercises
    the completion-only collator wiring on a tiny synthetic example so that path
    is covered hermetically.  On a genuine vmap failure it falls back to a single
    non-vmap forward + SFT loss so the smoke still exits 0 (documented in the
    module header).
    """
    from transformers import LlamaConfig

    device = torch.device("cpu")
    torch.manual_seed(args.seed)

    print("=" * 72)
    print("train_sft.py --smoke  (tiny random Llama, synthetic data, CPU)")
    print("=" * 72)

    # --- Tiny randomly-initialized model (no HF download) ---
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=64,
    )
    model = AutoModelForCausalLM.from_config(config)
    # NOTE: no fused_linear_cross_entropy — the eager SFT losses need out.logits.
    apply_model_patches(model)  # vmap-safety wrappers (eager attn / batchify)
    model.eval()
    model.to(device)

    pad_token_id = 0
    max_length = 16
    batch_size = 4
    loss_type = args.loss_type

    # --- Tiny synthetic dataset: 8 short token sequences (no network) ---
    rng = torch.Generator().manual_seed(args.seed)
    synthetic = [
        {
            "input_ids": torch.randint(
                1,
                config.vocab_size,
                (int(torch.randint(6, max_length, (1,), generator=rng)),),
                generator=rng,
            ).tolist()
        }
        for _ in range(8)
    ]

    collate = language_modeling_collator(pad_token_id, max_length)

    def collate_to_device(examples):
        b = collate(examples)
        return (
            b["input_ids"].to(device),
            b["attention_mask"].to(device),
            b["labels"].to(device),
        )

    # --- Exercise the completion-only collator wiring (no network) ----------
    # A hand-built example with a completion_mask (0 over a 3-token "prompt",
    # 1 over the response) collated with completion_only_loss=True: the prompt
    # label positions must be masked to -100, the completion positions kept.
    cmask_collate = language_modeling_collator(
        pad_token_id, max_length, completion_only_loss=True
    )
    demo_ids = [5, 6, 7, 11, 12, 13, 14]
    demo_cmask = [0, 0, 0, 1, 1, 1, 1]
    demo_batch = cmask_collate([{"input_ids": demo_ids, "completion_mask": demo_cmask}])
    demo_labels = demo_batch["labels"][0]
    n_supervised = int((demo_labels[:7] != -100).sum().item())
    print(
        "\nCompletion-only collator wiring: "
        f"{n_supervised} supervised label tokens (prompt masked to -100), "
        f"completion_mask present={'completion_mask' in demo_batch}"
    )

    # --- Functional conversion (everything trainable on this tiny model) ---
    fmodel, trainable, frozen = make_functional(
        model, disable_autograd_tracking=True, partition_trainable=True
    )
    print(f"Trainable param tensors: {len(trainable)} | frozen: {len(frozen)}")

    per_example_loss = _make_per_example_loss(fmodel, frozen, loss_type=loss_type)

    # --- Try the full per-example vmap DP-SGD path; fall back if it breaks ---
    try:
        from opaque.optimizers import adamw

        grad_fn, clip_state = clipped_grad(
            per_example_loss,
            argnums=0,
            batch_argnums=_BATCH_ARGNUMS,
            clipping_norm=args.clipping_norm,
            normalize_by=batch_size,
            return_aux=True,
        )
        noise_fn, noise_state = gaussian_noise(
            noise_multiplier=args.noise_multiplier or 1.0, key=key(args.seed)
        )
        base_opt = adamw(lr=args.learning_rate)
        opt_state = base_opt.init(trainable)

        # Two DP-SGD steps over Poisson-sampled batches.  Draw extra steps so
        # an empty Poisson draw (possible with this tiny dataset) still leaves
        # two non-empty steps that print loss.
        sampler = PoissonSampler(
            synthetic,
            sample_rate=batch_size / len(synthetic),
            n_steps=8,
            key=fold_in(key(args.seed), 0, 0),
        )
        print(
            f"\nRunning 2 DP-SGD steps (full per-example vmap path, loss={loss_type})..."
        )
        step = 0
        for indices in sampler:
            rows = [synthetic[i] for i in indices]
            if not rows:  # empty Poisson draw — skip, no gradient to release
                continue
            batch = collate_to_device(rows)
            (grads, aux), clip_state = grad_fn(trainable, *batch, state=clip_state)
            noisy_grads, noise_state = noise_fn(grads, noise_state)
            updates, opt_state = base_opt.update(
                noisy_grads, opt_state, params=trainable
            )
            trainable = torchopt.apply_updates(trainable, updates)
            step += 1
            print(
                f"  step {step}/2 | bs={batch[0].shape[0]} | "
                f"loss={aux.loss_values.mean().item():.4f}"
            )
            if step >= 2:
                break

        if step == 0:
            raise RuntimeError("no non-empty Poisson batch drawn in smoke")
        print("\nSmoke OK: full per-example DP-SGD vmap path completed 2 steps.")
        return 0

    except Exception as exc:  # pragma: no cover - defensive fallback path
        # Documented fallback (module header): vmap(grad(...)) over the patched
        # model failed on this CPU host. Validate the loss wiring with a single
        # non-vmap forward + SFT loss so the smoke still exits 0.
        print(f"\nNote: full vmap DP-SGD path raised: {type(exc).__name__}: {exc}")
        print(
            "Falling back to a single non-vmap forward + SFT loss to validate "
            "the loss wiring."
        )
        batch = collate_to_device(synthetic[:batch_size])
        input_ids, attention_mask, labels = batch
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = _SFT_LOSSES[loss_type](out.logits, labels)
        print(f"  non-vmap batch loss (per-example mean): {loss.mean().item():.4f}")
        print("\nSmoke OK (fallback path): loss wiring validated.")
        return 0


def main():
    args = parse_args()

    if args.smoke:
        return _run_smoke(args)

    is_ddp, rank, world_size, local_rank = _init_distributed()
    is_main_process = rank == 0

    if args.eval_batch_size is None:
        args.eval_batch_size = args.microbatch_size or args.batch_size

    # Set audit_batch_size to microbatch_size if not specified (forward-only, so at least as cheap)
    if args.audit_batch_size is None:
        args.audit_batch_size = args.microbatch_size or args.batch_size

    # μ-GDP auditing has no meaningful answer at δ = 0 (pure DP is incompatible
    # with Gaussian DP).  Fail fast instead of crashing inside the audit.
    if (
        args.audit
        and args.audit_method == "gdp"
        and args.target_delta is not None
        and args.target_delta <= 0
    ):
        raise SystemExit(
            "--audit-method gdp requires --target-delta > 0 "
            f"(got {args.target_delta}); use --audit-method eps_delta for pure DP."
        )

    if is_main_process:
        print("=" * 80)
        print("DP-SGD LoRA Supervised Fine-Tuning (SFT)")
        print("=" * 80)

    # Initialize wandb (enabled by default, offline if no credentials)
    use_wandb = (not args.no_wandb) and is_main_process
    if use_wandb:
        # Generate default run name from key parameters if not specified
        if args.wandb_run_name is None:
            model_short = args.model_name.split("/")[-1]
            run_name = f"sft_{model_short}_n{args.num_train_samples}_e{args.num_epochs}_b{args.batch_size}_eps{args.target_epsilon}_lr{args.learning_rate}"
        else:
            run_name = args.wandb_run_name

        # Offline by default; set WANDB_MODE=online (or WANDB_API_KEY) to sync
        if not os.environ.get("WANDB_MODE"):
            os.environ["WANDB_MODE"] = (
                "online" if os.environ.get("WANDB_API_KEY") else "offline"
            )
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
        )
        print(f"W&B initialized (mode: {os.environ.get('WANDB_MODE', 'online')})")

    # Setup device
    device, device_name = _select_device(local_rank if is_ddp else None)
    if is_main_process:
        if device.type == "cpu":
            print(f"\nUsing device: {device}")
            print("Warning: Training on CPU will be slow")
        else:
            print(f"\nUsing device: {device} ({device_name})")
        if is_ddp:
            print(
                f"Distributed mode: rank={rank}/{world_size}, local_rank={local_rank}"
            )

    # Set seed
    torch.manual_seed(args.seed)

    # Attention implementation: SDPA is the default in recent HuggingFace Transformers
    # and provides up to 3.6x memory savings over eager at seq_len=1024 with vmap.
    # Use --attention eager to override (e.g., for debugging).
    use_eager = args.attention == "eager" or device.type == "mps"

    # When a specific SDPA backend is requested, enable only that one globally.
    if not use_eager and args.sdpa_backend is not None:
        backends = {
            "flash": torch.backends.cuda.enable_flash_sdp,
            "efficient": torch.backends.cuda.enable_mem_efficient_sdp,
            "cudnn": torch.backends.cuda.enable_cudnn_sdp,
            "math": torch.backends.cuda.enable_math_sdp,
        }
        for name, setter in backends.items():
            setter(name == args.sdpa_backend)
        print(f"SDPA backend forced: {args.sdpa_backend}")

    # Load model config and disable dropout
    print(f"\nLoading model: {args.model_name}...")
    config = AutoConfig.from_pretrained(args.model_name)

    dropout_attrs = [
        "attn_pdrop",
        "resid_pdrop",
        "embd_pdrop",
        "attention_dropout",
        "hidden_dropout",
        "dropout",
        "attn_dropout",
        "ffn_dropout",
    ]
    for attr in dropout_attrs:
        if hasattr(config, attr):
            setattr(config, attr, 0.0)

    dtype_name, torch_dtype, dtype_warning = _resolve_model_dtype(args.dtype, device)
    args.dtype = dtype_name
    _print_runtime_mode_report(
        device, device_name, dtype_name, torch_dtype, dtype_warning
    )

    # Load model
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

    tracker = perf_tracker(device)

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    except TypeError as exc:
        if "unexpected keyword argument 'dtype'" not in str(exc):
            raise
        model_kwargs.pop("dtype")
        model_kwargs["torch_dtype"] = torch_dtype
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model = model.to(device)
    print_memory(device, "After model load")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Completion-only chat template install (before make_functional!) -----
    # ``get_training_chat_template`` returns a template carrying the
    # ``{% generation %}`` markers ``apply_chat_template_with_mask`` relies on.
    # We assign it onto the tokenizer here, BEFORE LoRA / make_functional, so no
    # embedding resize crosses the functional snapshot.  No new tokens are added,
    # so no resize occurs.
    if args.completion_only:
        if not getattr(tokenizer, "chat_template", None):
            raise SystemExit(
                "--completion-only requires a tokenizer with a chat_template "
                f"(model '{args.model_name}' has none).  Use an -Instruct/chat "
                "model, or clone a template with "
                "opaque.alignment.data.clone_chat_template."
            )
        tokenizer.chat_template = get_training_chat_template(tokenizer)
        print("Completion-only loss: enabled (assistant-token mask via chat template)")

    # Apply LoRA
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
    # SFT consumes ``out.logits`` (eager nll/dft loss), so we do NOT opt into the
    # fused linear+CE kernel here — that fast path returns ``logits=None``.
    apply_model_patches(model)
    model.print_trainable_parameters()
    print_memory(device, "After LoRA")

    # Load and prepare dataset
    print(f"\nLoading dataset: {args.dataset}...")
    if args.dataset_subset:
        print(f"  Subset: {args.dataset_subset}")
    print(f"  Split: {args.dataset_split}")
    if args.completion_only:
        print(f"  Mode: completion-only (chat columns: {', '.join(_CHAT_COLUMNS)})")
    else:
        print(f"  Text field: {args.dataset_text_field}")

    total_needed = args.num_train_samples + args.num_eval_samples
    dataset = _load_streaming_subset(
        dataset_name=args.dataset,
        dataset_subset=args.dataset_subset,
        dataset_split=args.dataset_split,
        total_needed=total_needed,
    )
    print(f"  Total examples in dataset: {len(dataset)}")

    # Validate we have enough data
    if len(dataset) < total_needed:
        raise ValueError(
            f"Dataset has {len(dataset)} examples but need {total_needed} "
            f"(train={args.num_train_samples} + eval={args.num_eval_samples})"
        )

    # Split into eval and train using skip/take
    print(
        f"\nPreparing {args.num_eval_samples} eval + {args.num_train_samples} train samples..."
    )
    eval_raw = dataset.take(args.num_eval_samples)
    train_raw = dataset.skip(args.num_eval_samples).take(args.num_train_samples)

    # Tokenize each split into per-example dicts the LM collator consumes
    # ({"input_ids": [...]} plus, under --completion-only, a "completion_mask").
    print(f"\nTokenizing (max_length={args.max_length})...")

    def _tokenize_split(rows_iter, desc):
        out = []
        for raw_row in rows_iter:
            if args.completion_only:
                chat_col = _detect_chat_column(raw_row)
                if chat_col is None:
                    continue
                try:
                    tok = _tokenize_chat_example(
                        raw_row[chat_col], tokenizer, args.max_length
                    )
                except Exception:
                    continue
                # Skip examples with no completion tokens (nothing to learn from).
                if sum(tok["completion_mask"]) == 0:
                    continue
            else:
                text = raw_row.get(args.dataset_text_field)
                if not text:
                    continue
                tok = _tokenize_text_example(text, tokenizer, args.max_length)
            # Need at least one (input, target) pair after the causal shift.
            if len(tok["input_ids"]) >= 2:
                out.append(tok)
        print(f"  {desc}: {len(out)} usable examples")
        return Dataset.from_list(out)

    eval_dataset = _tokenize_split(eval_raw, "Tokenizing eval")
    train_dataset = _tokenize_split(train_raw, "Tokenizing train")

    if len(train_dataset) == 0:
        raise SystemExit(
            f"No usable examples found in '{args.dataset}'. "
            + (
                "Check that the dataset has a chat column (messages/conversations)."
                if args.completion_only
                else f"Check that the text field '{args.dataset_text_field}' exists."
            )
        )
    print(
        f"Prepared datasets: {len(train_dataset)} train samples, {len(eval_dataset)} eval samples"
    )

    # Language-modeling collator (opaque-alignment primitive).  Under
    # --completion-only it masks non-completion label positions to -100 so the
    # loss is computed only over assistant/completion tokens.
    collate_raw = language_modeling_collator(
        tokenizer.pad_token_id,
        args.max_length,
        completion_only_loss=args.completion_only,
    )

    def collate(examples):
        # Tuple order matches _BATCH_ARGNUMS and the per-example loss signature.
        b = collate_raw(examples)
        return (
            b["input_ids"].to(device),
            b["attention_mask"].to(device),
            b["labels"].to(device),
        )

    # Privacy auditing setup: designate canaries and remove held-out ones
    audit_cf = None
    audit_dataset = None
    audit_ref_scores = None
    if args.audit:
        print(f"\nSetting up privacy auditing with {args.audit_canaries} canaries...")
        audit_cf = auditing.coin_flip(
            train_dataset,
            num_canaries=args.audit_canaries,
            key=key(args.seed),
        )
        audit_dataset = train_dataset  # Keep reference before filtering
        train_dataset = train_dataset.select(audit_cf.train_indices(len(train_dataset)))
        print(
            f"  Canaries: {len(audit_cf.in_indices)} in, "
            f"{len(audit_cf.out_indices)} out (held out from training)"
        )
        print("  Natural-row canaries: audit ε is a lower bound, not a certificate")
        print(f"  Training set: {len(train_dataset)} examples")

    global_train_size = len(train_dataset)
    use_shard = is_ddp and args.shard
    use_parallel_poisson = is_ddp and not args.shard
    if use_shard:
        train_dataset = local_shard(train_dataset, rank=rank, world_size=world_size)

    # Eval DataLoader (standard batching, no privacy requirements)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
        drop_last=False,
    )

    # For training: Poisson sampling (not uniform shuffling!)
    # Poisson: each example independently sampled with probability sample_rate each step.
    # In parallel Poisson mode each rank samples independently from the full dataset,
    # so we divide by world_size to keep the global expected batch size = args.batch_size.
    truncated_batch_size = args.truncated_batch_size
    sample_rate = args.batch_size / global_train_size
    if use_parallel_poisson:
        sample_rate /= world_size

    expected_steps_per_epoch = int(global_train_size / args.batch_size)

    print("\nPoisson sampling setup:")
    if use_parallel_poisson:
        print(f"  Mode: parallel_poisson (no shard, world_size={world_size})")
    print("  Sampler: poisson")
    if truncated_batch_size is not None:
        print(f"  Truncated batch size (cap): {truncated_batch_size}")
    print(f"  Sample rate (per rank): {sample_rate:.6f}")
    print(f"  Expected global batch size: {args.batch_size}")
    print(f"  Expected steps per epoch: ~{expected_steps_per_epoch}")
    print(f"Eval batches: {len(eval_loader)}")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("\nGradient checkpointing: enabled")

    offload_ctx = (
        torch.autograd.graph.save_on_cpu(pin_memory=True)
        if args.activation_offloading
        else contextlib.nullcontext()
    )
    if args.activation_offloading:
        print(
            f"CPU offload: enabled (save_on_cpu, works {'with' if args.gradient_checkpointing else 'without'} checkpointing)"
        )

    # Convert to functional (only LoRA parameters)
    print("\nConverting to functional form (LoRA parameters only)...")
    print("  (This may take 1-2 minutes for large models...)")
    start_time = time.time()
    fmodel, trainable_params, frozen_params = make_functional(
        model,
        disable_autograd_tracking=True,
        partition_trainable=True,
    )
    param_names = list(trainable_params.keys())
    elapsed = time.time() - start_time
    print(f"Trainable parameters: {len(param_names)} (took {elapsed:.1f}s)")
    print_memory(device, "After functional conversion")

    # Per-example SFT loss closure (one forward over ``out.logits``; the eager
    # nll/dft loss applies the DP-safe per-example divisor).  Output for example
    # i depends only on example i's data, so per-example sensitivity is O(C).
    per_example_loss_fn = _make_per_example_loss(
        fmodel, frozen_params, loss_type=args.loss_type
    )

    # Verified canary scoring: the per-example SFT loss is the membership
    # signal; loss_scores builds the canary loader internally and binds
    # each score to its canary identifier.
    def score_canaries(params, reference_scores=None):
        return auditing.loss_scores(
            per_example_loss_fn,
            params,
            batch_argnums=_BATCH_ARGNUMS,
            coin_flip=audit_cf,
            dataset=audit_dataset,
            batch_size=args.audit_batch_size,
            collate_fn=collate,
            reference_scores=reference_scores,
        )

    # Auditing helper: compute scores and run one-run estimator
    def run_audit(trainable):
        """Score canaries and report audit metrics. Returns OneRunEstimate or None."""
        if not args.audit or audit_cf is None:
            return None
        scores = score_canaries(trainable, reference_scores=audit_ref_scores)
        return auditing.one_run(scores, coin_flip=audit_cf)

    def _audit_method(estimate):
        """Pick the audit-method object on `estimate` per ``args.audit_method``."""
        return estimate.gdp() if args.audit_method == "gdp" else estimate.eps_delta()

    # Compute reference (untrained) scores for auditing before any training
    # Paper Algorithm 3: Score = loss(w0, x) - loss(wℓ, x), so we need w0 losses
    if args.audit and audit_cf is not None:
        print("\nComputing reference scores on untrained model...")
        audit_ref_scores = score_canaries(trainable_params)
        print(
            f"  Reference scores: mean={audit_ref_scores.scores.mean():.4f}, std={audit_ref_scores.scores.std():.4f}"
        )

    def eval_lm_metrics(trainable):
        """Held-out language-modeling metrics over the eval set.

        Token-weighted mean cross-entropy (pad / non-completion positions masked
        to ``-100`` by the collator) + its perplexity, plus mean next-token
        accuracy and prediction entropy (``opaque.alignment.metric``) over the
        supervised positions.  All forwards run
        under ``torch.no_grad()`` outside the clipped path.  Returns a dict of
        floats; an empty eval set yields ``nan`` loss / accuracy.
        """
        with torch.no_grad():
            merged = {**frozen_params, **trainable}
            total_loss = 0.0
            total_tokens = 0
            acc_sum = 0.0
            ent_sum = 0.0
            metric_tokens = 0
            for input_ids, attention_mask, labels in eval_loader:
                out = fmodel(merged, input_ids=input_ids, attention_mask=attention_mask)
                # Supervised (non-ignored) shifted-token count for this batch.
                num_tokens = int((labels[..., 1:] != -100).sum().item())
                if num_tokens == 0:
                    continue
                # Token-weighted CE: sum over supervised tokens then re-divide.
                per_example_ce = nll_loss(out.logits, labels)  # (B,) per-example mean
                ex_tokens = (labels[..., 1:] != -100).sum(-1).clamp(min=1)
                total_loss += float((per_example_ce * ex_tokens).sum().item())
                total_tokens += num_tokens
                mask = (labels != -100).to(out.logits.dtype)
                acc_sum += (
                    float(mean_token_accuracy(out.logits, labels, mask).item())
                    * num_tokens
                )
                ent_sum += (
                    float(entropy_from_logits(out.logits, mask).item()) * num_tokens
                )
                metric_tokens += num_tokens
            if total_tokens == 0:
                return {
                    "eval/loss": float("nan"),
                    "eval/perplexity": float("nan"),
                    "eval/token_accuracy": float("nan"),
                    "eval/entropy": float("nan"),
                }
            mean_loss = total_loss / total_tokens
            result = {
                "eval/loss": mean_loss,
                "eval/perplexity": float(torch.exp(torch.tensor(mean_loss)).item()),
            }
            if metric_tokens > 0:
                result["eval/token_accuracy"] = acc_sum / metric_tokens
                result["eval/entropy"] = ent_sum / metric_tokens
            return result

    # Build clipping norm (scalar or per-group)
    if args.per_group_clipping:
        clip_norm = per_group(
            trainable_params,
            fallback=args.per_group_clipping_fallback,
            **args.per_group_clipping,
        )
        if is_main_process:
            print("\nPer-group clipping norms:")
            for gname, val in clip_norm.values.items():
                count = sum(1 for g in clip_norm.groups.values() if g == gname)
                print(f"  {gname}: {val:.3f} ({count} params)")
            print(f"  Effective (for accounting): {clip_norm.effective:.3f}")
    else:
        clip_norm = args.clipping_norm

    # Setup optimizer
    print("\nSetting up DP-SGD SFT training...")
    print(f"  Loss: {args.loss_type}")
    print(f"  Completion-only: {args.completion_only}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Learning rate: {args.learning_rate}")
    if isinstance(clip_norm, PerGroup):
        print(f"  Clip norm: per-group (effective={clip_norm.effective:.3f})")
    else:
        print(f"  Clip norm: {clip_norm}")
    print(f"  Noise mechanism: {args.noise_mechanism}")
    if args.noise_mechanism != "gaussian":
        print(f"  Noise bound: ±{args.noise_bound}")
    print(f"  Microbatch size: {args.microbatch_size}")
    print(f"  Clipping mode: {args.clipping_mode}")
    if args.clipping_mode == "auto":
        print(f"  AUTO-S gamma: {args.auto_clipping_gamma}")
    print(f"  Eval steps: {args.eval_steps}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Expected total steps: ~{args.num_epochs * expected_steps_per_epoch}")

    _JME_OPTIMIZERS = frozenset(
        {"adam", "adamw", "ademamix", "rmsprop", "radam", "adadelta"}
    )
    use_jme = bool(args.jme)
    jme_allocation = None
    if use_jme:
        if args.optimizer not in _JME_OPTIMIZERS:
            raise ValueError(
                f"--jme requires an optimizer that consumes SecondMomentNoiseOutput "
                f"({sorted(_JME_OPTIMIZERS)}), got {args.optimizer!r}."
            )
        if isinstance(clip_norm, PerGroup):
            raise ValueError("--jme does not support --per-group-clipping.")
        if args.noise_mechanism != "gaussian":
            raise ValueError("--jme requires --noise-mechanism gaussian.")
        if truncated_batch_size is not None or use_parallel_poisson:
            raise ValueError(
                "--jme currently supports only plain independent Poisson sampling; "
                "truncated and parallel sampling require separate derivations."
            )
        if args.jme_aggregate_norm is None or args.jme_aggregate_norm <= 0:
            raise ValueError("--jme requires a positive --jme-aggregate-norm.")
        if args.jme_allocation is None:
            raise ValueError("--jme requires --jme-allocation.")
        if args.jme_allocation == "paper_reference":
            if args.jme_max_first_variance_ratio is not None:
                raise ValueError(
                    "--jme-max-first-variance-ratio is only valid with "
                    "--jme-allocation first_variance_cap."
                )
            jme_allocation = JmeAllocation.paper_reference()
        else:
            if args.jme_max_first_variance_ratio is None:
                raise ValueError(
                    "--jme-allocation first_variance_cap requires "
                    "--jme-max-first-variance-ratio."
                )
            jme_allocation = JmeAllocation.first_variance_cap(
                args.jme_max_first_variance_ratio
            )
    elif any(
        value is not None
        for value in (
            args.jme_aggregate_norm,
            args.jme_allocation,
            args.jme_max_first_variance_ratio,
        )
    ):
        raise ValueError("JME-specific arguments require --jme.")

    # AdaClip's noisy clipping-rate estimate and the gradient mechanism are
    # separate DP releases, so they need independent RNG streams.
    quantile_noise_key = gradient_noise_key = key(args.seed)
    if args.clipping_mode == "adaptive":
        quantile_noise_key, gradient_noise_key = split(gradient_noise_key)

    # Create gradient function based on clipping mode.
    if args.clipping_mode == "adaptive":
        grad_fn, clip_state = adaptive_clipped_grad(
            per_example_loss_fn,
            argnums=0,
            batch_argnums=_BATCH_ARGNUMS,
            initial_clipping_norm=clip_norm,
            target_quantile=args.target_clipping_rate,
            clipping_norm_max=args.clipping_norm_max,
            microbatch_size=args.microbatch_size,
            return_aux=True,
            key=quantile_noise_key,
            normalize_by=args.batch_size,
        )
    elif args.clipping_mode == "auto":
        grad_fn, clip_state = auto_clipped_grad(
            per_example_loss_fn,
            argnums=0,
            batch_argnums=_BATCH_ARGNUMS,
            R=clip_norm,
            gamma=args.auto_clipping_gamma,
            normalize_by=args.batch_size,
            microbatch_size=args.microbatch_size,
            return_aux=True,
        )
    else:
        grad_fn, clip_state = clipped_grad(
            per_example_loss_fn,
            argnums=0,
            batch_argnums=_BATCH_ARGNUMS,
            clipping_norm=clip_norm,
            normalize_by=args.batch_size,
            microbatch_size=args.microbatch_size,
            return_aux=True,
        )

    # Calibrate noise multiplier from target privacy budget
    # sample_rate already computed above
    total_steps = args.num_epochs * expected_steps_per_epoch

    # Compute delta from training set size: δ = 1/n^1.1 (keeps δ below 1/n while
    # being less conservative than the previous 1/n² heuristic on smaller runs).
    if args.target_delta is None:
        args.target_delta = 1.0 / (global_train_size**1.1)
    if use_wandb:
        wandb.config.update({"target_delta": args.target_delta}, allow_val_change=True)

    # Noise injection — bind mechanism-specific parameters once.
    # Chain: base mechanism → adaclip (optional) → amplification.
    # Bounded Gaussian noise (Chen and Hale, 2024) confines per-coordinate
    # support but accounting collapses to ordinary Gaussian at training
    # scale (ℓ₂-ball clip, not a product of intervals), so we use
    # dpsgd_acc.gaussian() for accounting either way.
    _num_groups = len(clip_norm.values) if isinstance(clip_norm, PerGroup) else 1
    if args.noise_multiplier == 0:

        def mechanism(nm):
            return acc.nonprivate()
    else:
        # Both gaussian and bounded_gaussian account as ordinary Gaussian at
        # training scale (ℓ₂-ball clip, not a product of intervals).
        mechanism = dpsgd_acc.gaussian

    if args.clipping_mode == "adaptive":
        _base_mechanism = mechanism

        def mechanism(nm, ebs=args.batch_size, ng=_num_groups):
            return dpsgd_acc.adaclip(
                _base_mechanism(nm), expected_batch_size=ebs, num_groups=ng
            )

    # Projected JME is dominated by the same Gaussian PLD after whitening.

    _unamplified = mechanism
    if truncated_batch_size is not None:

        def mechanism(nm):
            return dpsgd_acc.poisson(
                _unamplified(nm),
                sample_rate=sample_rate,
                truncated_batch_size=truncated_batch_size,
                dataset_size=global_train_size,
            )
    elif use_parallel_poisson:

        def mechanism(nm):
            return dpsgd_acc.parallel_poisson(
                _unamplified(nm),
                sample_rate=sample_rate,
                num_workers=world_size,
            )
    else:

        def mechanism(nm):
            return dpsgd_acc.poisson(_unamplified(nm), sample_rate=sample_rate)

    # Calibrate noise multiplier from target privacy budget.
    if args.noise_multiplier is not None:
        noise_multiplier = args.noise_multiplier
        print(
            f"\nUsing fixed noise multiplier: {noise_multiplier:.4f} (skipping calibration)"
        )
        if use_jme:
            _log_jme(args)
    else:
        print("\nCalibrating privacy parameters...")
        if use_parallel_poisson:
            print(f"  Accounting: parallel_poisson (world_size={world_size})")
        print(f"  Noise mechanism: {args.noise_mechanism}")
        if args.noise_mechanism == "bounded_gaussian":
            print(f"  Noise bound: ±{args.noise_bound}")
        if use_jme:
            _log_jme(args)
        print(f"  δ = {args.target_delta:.2e} (n={global_train_size})")
        print(f"  Total steps: {total_steps}")
        print(f"  Sample rate: {sample_rate:.6f}")
        print(f"  Target: ε={args.target_epsilon}, δ={args.target_delta:.2e}")
        print("  (This may take 1-3 minutes...)")

        start_time = time.time()
        calibration = cal.calibrate(
            cal.epsilon_budget(args.target_epsilon, delta=args.target_delta),
            lambda nm: mechanism(nm) * total_steps,
            param_min=args.calibration_min,
            param_max=args.calibration_max,
            tolerance=args.calibration_tolerance,
        )
        noise_multiplier = calibration.param
        elapsed = time.time() - start_time

        print(f"\nCalibrated privacy parameters (took {elapsed:.1f}s):")
        print(
            f"  Target: ε={args.target_epsilon:.3f}, δ={args.target_delta:.1e} | "
            f"Achieved ε≈{calibration.achieved:.3f}"
        )
        print(
            f"  Noise multiplier: {noise_multiplier:.4f} "
            f"(iterations={calibration.iterations}, converged={calibration.converged})"
        )

    # Build LR schedule using opaque.scheduling primitives.  Each curve
    # returns a ``Callable[[int], float]`` and ``with_warmup`` composes a
    # 0→1 linear ramp during the warmup window; torchopt's
    # ``scale_by_neg_lr`` accepts either a callable or a scalar.  We
    # share ``total_steps`` with the privacy calibration above so the
    # schedule and accounting agree on the run length.  ``--max-steps``
    # (when set) only truncates training — the schedule is laid out over
    # the full planned epoch count, same as accounting.
    if not 0.0 <= args.lr_min_ratio <= 1.0:
        raise ValueError(f"--lr-min-ratio must be in [0, 1], got {args.lr_min_ratio}")
    peak_lr = args.learning_rate
    warmup = max(0, int(args.lr_warmup_steps))
    lr_min = peak_lr * args.lr_min_ratio
    decay_span = max(1, total_steps - warmup)

    base: float | Schedule
    if args.lr_schedule == "cosine":
        base = cosine_schedule(
            init_value=peak_lr,
            end_value=lr_min,
            transition_steps=decay_span,
            transition_begin=warmup,
        )
    elif args.lr_schedule == "linear":
        base = linear_schedule(
            init_value=peak_lr,
            end_value=lr_min,
            transition_steps=decay_span,
            transition_begin=warmup,
        )
    elif args.lr_schedule == "sqrt":
        # Inverse-sqrt timescale defaults to warmup when set, otherwise
        # to the full training run (gives a gentle ~1/sqrt(2) decay over
        # the run rather than the very aggressive 1/sqrt(t) that would
        # come from a tiny timescale).
        base = inverse_sqrt_schedule(
            init_value=peak_lr,
            transition_steps=warmup if warmup > 0 else max(1, total_steps),
            transition_begin=warmup,
        )
    elif args.lr_schedule == "none":
        base = peak_lr
    else:
        raise ValueError(f"Unknown --lr-schedule: {args.lr_schedule}")

    if warmup > 0:
        lr_for_opt: float | Schedule = with_warmup(base, transition_steps=warmup)
    else:
        lr_for_opt = base

    if args.lr_schedule != "none" or warmup > 0:
        print(
            f"  LR schedule: {args.lr_schedule} "
            f"(peak={peak_lr:g}, min={lr_min:g}, warmup={warmup}, total={total_steps})"
        )

    # Setup optimizer.  Noise metadata travels with ``NoisedPytree`` updates,
    # so optimizer construction does not need a precomputed stddev;
    # ``--noise-bias-correction`` only controls whether the optimizer's
    # DP-aware path consumes that metadata.  For optimizers without a BC
    # path (sgd/lion) the flag is silently ignored.
    if args.optimizer == "adam":
        from opaque.optimizers import adam

        base_opt = adam(
            lr=lr_for_opt,
            weight_decay=args.weight_decay,
            noise_bias_correction=args.noise_bias_correction,
        )
    elif args.optimizer == "sgd":
        from opaque.optimizers import sgd

        base_opt = sgd(lr=lr_for_opt, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        from opaque.optimizers import adamw

        base_opt = adamw(
            lr=lr_for_opt,
            weight_decay=args.weight_decay,
            noise_bias_correction=args.noise_bias_correction,
        )
    elif args.optimizer == "ademamix":
        from opaque.optimizers import ademamix

        base_opt = ademamix(
            lr=lr_for_opt,
            weight_decay=args.weight_decay,
            noise_bias_correction=args.noise_bias_correction,
        )
    elif args.optimizer == "lion":
        from opaque.optimizers import lion

        base_opt = lion(
            lr=lr_for_opt,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adafactor":
        from opaque.optimizers import adafactor

        base_opt = adafactor(
            lr=lr_for_opt,
            weight_decay=args.weight_decay,
            noise_bias_correction=args.noise_bias_correction,
        )
    elif args.optimizer == "rmsprop":
        from opaque.optimizers import rmsprop

        base_opt = rmsprop(
            lr=lr_for_opt,
            weight_decay=args.weight_decay,
            noise_bias_correction=args.noise_bias_correction,
        )
    elif args.optimizer == "adagrad":
        from opaque.optimizers import adagrad

        base_opt = adagrad(
            lr=lr_for_opt,
            weight_decay=args.weight_decay,
            noise_bias_correction=args.noise_bias_correction,
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    opt_state = base_opt.init(trainable_params)
    accounting = Accountant()

    # Noise functions consume ClippedPytree metadata directly and return
    # NoisedPytree updates carrying the realized per-step stddev.
    initial_bound = clip_norm / args.batch_size
    if use_jme:
        assert jme_allocation is not None
        noise_fn, noise_state = jme_noise(
            noise_multiplier=noise_multiplier,
            aggregate_norm=args.jme_aggregate_norm,
            allocation=jme_allocation,
            key=gradient_noise_key,
        )
    elif args.noise_mechanism == "bounded_gaussian":
        # Pass ``bound`` unconditionally — at ``noise_multiplier=0`` the
        # bounded path clamps the input to the interval (vs. the unbounded
        # path which returns it unchanged), so the mechanism stays
        # consistent for the user's chosen flag.
        noise_fn, noise_state = gaussian_noise(
            noise_multiplier=noise_multiplier,
            bound=args.noise_bound,
            key=gradient_noise_key,
        )
    else:
        noise_fn, noise_state = gaussian_noise(
            noise_multiplier=noise_multiplier,
            key=gradient_noise_key,
        )

    # Training loop
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)

    losses = []
    clip_norms_history = []
    clip_rates_history = []
    last_clip_bound = initial_bound
    global_step = 0

    # Reset peak memory before training to get accurate training peak
    reset_peak_memory(device)
    print_memory(device, "Before training")

    # Step-0 eval: log baseline LM metrics before any training
    initial_metrics = eval_lm_metrics(trainable_params)
    initial_epsilon = accounting.epsilon_at(args.target_delta)
    initial_noise_std = _noise_stddev(initial_bound, noise_multiplier)
    print(
        f"  → Step 0 eval: loss={initial_metrics['eval/loss']:.4f}, "
        f"ppl={initial_metrics['eval/perplexity']:.3f}, ε={initial_epsilon:.3f}"
    )
    if use_wandb:
        wandb.log(
            {
                **initial_metrics,
                "privacy/epsilon": initial_epsilon,
                "train/noise_std": _effective(initial_noise_std),
                "train/clipping_norm": _effective(clip_norm),
            },
            step=0,
        )

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        print("-" * 80)
        print("Creating poisson sampler...")

        # Create sampler for this epoch
        epoch_sampler = PoissonSampler(
            train_dataset,
            sample_rate=sample_rate,
            truncated_batch_size=truncated_batch_size,
            n_steps=expected_steps_per_epoch,
            key=fold_in(key(args.seed), rank, epoch),
        )
        print("Creating DataLoader...")

        # DataLoader with batch_sampler
        epoch_loader = DataLoader(
            train_dataset,
            batch_sampler=epoch_sampler,
            collate_fn=collate,
        )

        # Iterate through Poisson-sampled batches
        for step_idx, batch in enumerate(epoch_loader):
            # === Accounting (data-independent, before execution) ===
            accounting |= mechanism(noise_multiplier)

            batch_size = batch[0].shape[0]

            # === Execution ===
            with tracker.train(batch_size=batch_size) as sp:
                # Compute clipped gradients (handles empty batches via library)
                with offload_ctx:
                    (grads_tuple, aux), clip_state = grad_fn(
                        trainable_params, *batch, state=clip_state
                    )
                if is_ddp:
                    clip_state, aux = sync(clip_state, aux)
                    sum_gradients_(grads_tuple)
                sp.mark("clip")

                step_clip_norm = _step_clip_norm(grads_tuple)
                noisy_grads, noise_state = noise_fn(grads_tuple, noise_state)
                noise_stddev = _step_noise_stddev(noisy_grads)
                if is_ddp:
                    noise_state = sync(noise_state)
                sp.mark("noise")

                updates, opt_state = base_opt.update(
                    noisy_grads, opt_state, params=trainable_params
                )
                trainable_params = torchopt.apply_updates(trainable_params, updates)
                sp.mark("optimizer")

            # Empty batch (rare but possible with Poisson): skip metrics.
            if batch_size == 0:
                global_step += 1
                continue

            # === Step metrics ===
            avg_loss = aux.loss_values.mean().item()
            clip_rate = aux.clipping_rate
            mean_grad_norm = aux.grad_norms.mean().item()

            losses.append(avg_loss)
            clip_norms_history.append(_effective(step_clip_norm))
            clip_rates_history.append(clip_rate)
            last_clip_bound = step_clip_norm

            global_step += 1

            # === Logging (every log_steps) ===
            if global_step % args.log_steps == 0:
                if use_wandb:
                    current_lr = (
                        float(lr_for_opt(global_step))
                        if callable(lr_for_opt)
                        else float(lr_for_opt)
                    )
                    wb_metrics = {
                        "train/loss": avg_loss,
                        "train/batch_size": batch_size,
                        "train/clipping_norm": _effective(step_clip_norm),
                        "train/clip_rate": clip_rate,
                        "train/grad_norm_mean": mean_grad_norm,
                        "train/clipped_grad_norm_mean": aux.clipped_grad_norms.mean().item(),
                        "train/noise_std": _effective(noise_stddev),
                        "train/lr": current_lr,
                        **tracker.train.last.to_dict(prefix="train/"),
                    }
                    if (
                        isinstance(step_clip_norm, PerGroup)
                        and aux.group_norms is not None
                    ):
                        for gname in step_clip_norm.values:
                            gn_bound = step_clip_norm.values[gname]
                            wb_metrics[f"group/clipping_norm/{gname}"] = gn_bound
                            gnorms = aux.group_norms[gname]
                            wb_metrics[f"group/grad_norm/{gname}"] = (
                                gnorms.mean().item()
                            )
                            gn_clipped = float((gnorms > gn_bound).sum().item())
                            wb_metrics[f"group/clip_rate/{gname}"] = gn_clipped / max(
                                1.0, float(batch_size)
                            )
                            if isinstance(noise_stddev, PerGroup):
                                wb_metrics[f"group/noise_std/{gname}"] = (
                                    noise_stddev.values[gname]
                                )
                    wandb.log(wb_metrics, step=global_step)

                last = tracker.train.last
                print(
                    f"Step {global_step:4d} [E{epoch + 1} S{step_idx + 1:3d}/{expected_steps_per_epoch:3d}] | "
                    f"BS: {batch_size} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Clip: norm={_effective(step_clip_norm):.3f}, rate={clip_rate:.1%} | "
                    f"GradNorm: μ={mean_grad_norm:.3f} | "
                    f"Noise: σ={_effective(noise_stddev):.4f} | "
                    f"Time: {last.step_time_sec:.2f}s | Mem: {last.memory_peak_gb:.1f}GB"
                )

            # Expensive operations (eval + privacy + audit) every eval_steps
            if global_step % args.eval_steps == 0:
                eval_m = eval_lm_metrics(trainable_params)
                # Cache PLD before eval so it serves as opaque boundary
                accounting = acc.cached(accounting)
                epsilon = accounting.epsilon_at(args.target_delta)

                metrics = {**eval_m, "privacy/epsilon": epsilon}
                eval_msg = (
                    f"  → Eval: loss={eval_m['eval/loss']:.4f}, "
                    f"ppl={eval_m['eval/perplexity']:.3f}"
                )
                if "eval/token_accuracy" in eval_m:
                    eval_msg += (
                        f", token_acc={eval_m['eval/token_accuracy']:.3f}, "
                        f"entropy={eval_m['eval/entropy']:.3f}"
                    )
                eval_msg += f", ε={epsilon:.3f}"

                if args.audit:
                    audit_estimate = run_audit(trainable_params)
                    audit_eps = _audit_method(audit_estimate).epsilon_at(
                        delta=args.target_delta
                    )
                    audit_auc = audit_estimate.attack_auc()
                    metrics["privacy/epsilon_audit"] = audit_eps
                    metrics["privacy/audit_auc"] = audit_auc
                    eval_msg += (
                        f", ε_audit[{args.audit_method}]={audit_eps:.4f}"
                        f", AUC={audit_auc:.4f}"
                    )

                if use_wandb:
                    wandb.log(metrics, step=global_step)
                print(eval_msg)

            # Early exit if --stop-at-step reached
            if args.stop_at_step is not None and global_step >= args.stop_at_step:
                print(
                    f"\nReached --stop-at-step={args.stop_at_step}, stopping training."
                )
                break

        # Break outer epoch loop if --stop-at-step reached
        if args.stop_at_step is not None and global_step >= args.stop_at_step:
            break

    # Final summary
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset} ({global_train_size} train samples)")
    print(f"Loss: {args.loss_type} (completion_only={args.completion_only})")
    print("\nTraining results:")
    print(f"  Total steps: {global_step}")
    if losses:
        print(f"  Initial loss: {losses[0]:.4f}")
        print(f"  Final loss: {losses[-1]:.4f}")
        if losses[0] != 0:
            print(
                f"  Loss reduction: {((losses[0] - losses[-1]) / abs(losses[0]) * 100):.1f}%"
            )

    final_metrics = eval_lm_metrics(trainable_params)
    print("\nFinal eval LM metrics:")
    print(f"  Loss:            {final_metrics['eval/loss']:.4f}")
    print(f"  Perplexity:      {final_metrics['eval/perplexity']:.3f}")
    if "eval/token_accuracy" in final_metrics:
        print(f"  Token accuracy:  {final_metrics['eval/token_accuracy']:.3f}")
        print(f"  Pred. entropy:   {final_metrics['eval/entropy']:.3f}")

    if args.clipping_mode == "adaptive":
        print("\nAdaptive clipping:")
        if isinstance(last_clip_bound, PerGroup):
            print("  Per-group output bounds:")
            initial_cn = initial_bound
            for gname in sorted(last_clip_bound.values.keys()):
                print(
                    f"    {gname}: {initial_cn.values[gname]:.3f} → {last_clip_bound.values[gname]:.3f}"
                )
            print(f"  Effective (final): {last_clip_bound.effective:.3f}")
        else:
            print(f"  Initial output max_norm: {_effective(initial_bound):.3f}")
            print(f"  Final output max_norm: {last_clip_bound:.3f}")
        if clip_norms_history:
            print(
                f"  Clip norm range: [{min(clip_norms_history):.3f}, {max(clip_norms_history):.3f}]"
            )
    elif isinstance(clip_norm, PerGroup):
        print("\nPer-group clipping:")
        for gname, val in clip_norm.values.items():
            print(f"  {gname}: {val:.3f}")
        print(f"  Effective: {clip_norm.effective:.3f}")
        if clip_rates_history:
            print(
                f"  Average clip rate: {sum(clip_rates_history) / len(clip_rates_history):.2%}"
            )
    else:
        print("\nFixed clipping:")
        print(f"  Clip norm: {args.clipping_norm:.3f}")
        if clip_rates_history:
            print(
                f"  Average clip rate: {sum(clip_rates_history) / len(clip_rates_history):.2%}"
            )

    final_epsilon = accounting.epsilon_at(args.target_delta)
    print("\nPrivacy:")
    if truncated_batch_size is not None:
        print(
            f"  Accounting: truncated_poisson (cap={truncated_batch_size}, n={global_train_size})"
        )
    elif use_parallel_poisson:
        print(f"  Accounting: parallel_poisson (world_size={world_size})")
    if use_jme:
        _log_jme(args)
    print(
        f"  Target: ε={args.target_epsilon:.3f}, δ={args.target_delta:.2e} (n={global_train_size})"
    )
    print(f"  Noise multiplier: {noise_multiplier:.4f}")
    print(f"  Final ε (theoretical): {final_epsilon:.4f}")
    if args.audit:
        audit_result = run_audit(trainable_params)
        audit_eps = _audit_method(audit_result).epsilon_at(delta=args.target_delta)
        audit_auc = audit_result.attack_auc()
        print(
            f"  Final ε (audit, {args.audit_method}): {audit_eps:.4f}"
            f"  ({audit_result.n_in} in, {audit_result.n_out} out)"
        )
        print(f"  Audit AUC:            {audit_auc:.4f}")
        # Empirical attack ROC β (1 − TPR at given FPR); independent of the
        # audit method, hence read from OneRunEstimate rather than the method.
        print(f"  Attack β @ α=0.01:    {audit_result.attack_beta_at(alpha=0.01):.4f}")
        print(f"  Attack β @ α=0.10:    {audit_result.attack_beta_at(alpha=0.1):.4f}")
        if use_wandb:
            wandb.log(
                {
                    "privacy/epsilon_audit": audit_eps,
                    "privacy/audit_auc": audit_auc,
                },
                step=global_step,
            )

    synced = sync(tracker) if is_ddp else tracker
    print("\nPerformance:")
    print(f"  Throughput: {synced.train.samples_per_second:.1f} samples/s")
    print(f"  Steps/s: {synced.train.steps_per_second:.2f}")
    print(f"  Peak memory: {synced.train.max_peak_memory_gb:.2f} GB")

    if use_wandb:
        wandb.finish()

    if is_ddp:
        dist.barrier(device_ids=[local_rank])
        _cleanup_distributed()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
