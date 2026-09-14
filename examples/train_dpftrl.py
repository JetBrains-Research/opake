"""DP-FTRL training with matrix factorization noise for causal language models.

This script implements DP-FTRL (Follow-The-Regularized-Leader) with correlated
noise from matrix factorization mechanisms. Unlike standard DP-SGD, the noise
is correlated across steps, yielding better privacy/utility tradeoffs when
combined with the correct optimizer.

KEY DIFFERENCES FROM DP-SGD (train_dpsgd.py):

  1. Optimizer: SGD with Polyak momentum (default), or one of the
     Opaque-built v-using optimizers (``adamw``, ``ademamix``).  For
    adaptive optimizers, pair with ``--second-moment`` to activate a
    private squared-gradient stream.  The optimizer consumes the
    privately-estimated ``g²`` stream alongside standard noised gradients.
    ``lion`` is also exposed but has no v, so ``--second-moment`` is
    rejected for it.

  2. Clipping: Fixed scalar norm or fixed per-group norms (``--per-group-clipping``).
     Adaptive clipping is not supported (sensitivity must stay constant across
     the MF run for the single-shot privacy proof).

  3. LR schedule: Constant by default (--warmup-frac 0); optional linear warmup
     → constant, fully predetermined before training (fixed linear map).

  4. Accounting: Single-shot from MF encoder sensitivity. The declared
     full-horizon epsilon is reported throughout training.

  5. Noise: Fixed stddev, correlated across steps via C^{-1} streaming
     multiplication. Cannot change noise level mid-training.
    With ``--second-moment`` (Adam-family only): two independent MF noise
    streams, calibrated via joint first+second moment sensitivity.

MECHANISMS:

  band_mf   — Banded Toeplitz (Choquette-Choo et al., 2023)
               O(bands × d) memory, uses Poisson (cyclic) sampling.
  blt       — Buffered Linear Toeplitz (Choquette-Choo et al., 2024)
               O(buffers × d) memory, near-optimal, handles multi-epoch.
  lambda_cgd — DP-λCGD (Kalinin et al., 2026), bandwidth-2 correlated noise
               via PRNG replay. Single hyperparam λ, zero extra memory.
  bisr      — BISR (Kalinin et al., ICLR 2026), generalises λCGD to
               arbitrary bandwidth p. Coefficients from inverse square root.
  bsr       — BSR (Kalinin & Lampert, NeurIPS 2024), closed-form banded square
               root for the paper workload (α, β); no L-BFGS. BnB sampling.
  identity  — DP-SGD baseline via MF API (C^{-1} = I, independent noise).
               Same training loop for fair comparison.

USAGE:

  # Quick smoke test (~2 minutes, SmolLM2-135M on KExercises)
  python examples/train_dpftrl.py --preset smoke

  # BandMF + b-min-sep on Mellum (default mechanism: b=64, momentum=0.95)
  python examples/train_dpftrl.py --preset mellum-kstack

  # 4-GPU distributed run with torchrun (sharded, same global batch as 1-GPU)
  torchrun --nproc_per_node=4 examples/train_dpftrl.py --preset mellum-kstack

  # BLT on Mellum (near-optimal correlated noise; heavier calibration solve)
  python examples/train_dpftrl.py --preset mellum-kstack --mechanism blt

  # DP-λCGD with Balls-in-Bins sampling (bandwidth-2 correlated noise, λ=0.9)
  python examples/train_dpftrl.py --preset mellum-kstack --mechanism lambda_cgd --lambda_ 0.9

  # BISR with bandwidth=4, Balls-in-Bins sampling
  python examples/train_dpftrl.py --preset mellum-kstack --mechanism bisr --bisr-bandwidth 4

  # BSR (closed-form): workload α via --bsr-alpha (paper default 1.0); optimizer WD is separate (--weight-decay, default 0)
  python examples/train_dpftrl.py --preset smoke --mechanism bsr --bsr-bandwidth 8 --bsr-alpha 1.0

  # DP-SGD baseline for fair comparison (same loop, independent noise)
  python examples/train_dpftrl.py --preset mellum-kstack --mechanism identity

  # Non-DP baseline (no noise, no privacy accounting, same loop)
  python examples/train_dpftrl.py --preset mellum-kstack --mechanism none

    # Adam-family without private second moments (single-stream MF noise)
  python examples/train_dpftrl.py --preset smoke --optimizer adamw

    # DP-Adam with private second moments (two MF noise streams)
    python examples/train_dpftrl.py --preset smoke --optimizer adamw --second-moment
    python examples/train_dpftrl.py --preset smoke --optimizer adamw --second-moment --mechanism blt
    python examples/train_dpftrl.py --preset smoke --optimizer adamw --second-moment --beta1 0.9 --beta2 0.999

    # AdEMAMix with private second moments — slow EMA captures long-range gradient signal
    python examples/train_dpftrl.py --preset smoke --optimizer ademamix --second-moment

    # Lion under MF noise (no private second moment — lion has no second moment)
  python examples/train_dpftrl.py --preset smoke --optimizer lion

REFERENCES:

  - BandMF: https://arxiv.org/abs/2306.08153
  - BLT: https://arxiv.org/abs/2404.16706
  - DP-λCGD: https://arxiv.org/abs/2601.22334
  - BISR: https://arxiv.org/abs/2505.12128
  - BSR: https://arxiv.org/abs/2405.13763
  - DP-FTRL: https://arxiv.org/abs/2103.00039
    - Private second moments: https://arxiv.org/abs/2502.06597
"""

# ruff: noqa: E402

import argparse
import contextlib
import os
import sys
import time

import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from opaque.device import sdpa_autocast_under_vmap_broken
from opaque.patches import apply_model_patches, apply_runtime_patches

apply_runtime_patches()

import torchopt

import opaque.accounting as acc
import opaque.auditing as auditing
import opaque.dpftrl.accounting as dpftrl_acc
from opaque.accounting import Accountant
from opaque.accounting import calibration as cal
from opaque.distributed import local_shard, sync
from opaque.distributed.gradients import sum_gradients_
from opaque.dpftrl.clipping import auto_clipped_grad, clipped_grad, per_group
from opaque.dpftrl.noise import (
    band_mf_strategy,
    bisr_strategy,
    blt_strategy,
    bsr_strategy,
    identity_strategy,
    lambda_cgd_strategy,
    mf_gaussian_noise,
)
from opaque.dpftrl.sampling import (
    BallsInBinsSampler,
    BMinSepSampler,
    CyclicPoissonSampler,
    SequentialBatchSampler,
)
from opaque.functional import empty_collate, make_functional
from opaque.profiling import (
    perf_tracker,
    print_memory,
    reset_peak_memory,
)
from opaque.random import fold_in, key
from opaque.scheduling import (
    cosine_schedule,
    inverse_sqrt_schedule,
    linear_schedule,
    with_warmup,
)
from opaque.scheduling.types import Schedule
from opaque.types import (
    PerGroup,
    SecondMomentClippingOutput,
    SecondMomentNoiseOutput,
)

try:
    import wandb
except ImportError:
    wandb = None


# ---------------------------------------------------------------------------
# Shared utilities (same as train_dpsgd.py)
# ---------------------------------------------------------------------------


def _compile_grad_fn(grad_fn, *, backend, mode):
    """Caller-applied ``torch.compile`` of the DP grad transform.

    Compiles the whole ``vmap(grad(loss))`` + clipping step (functorch *inside*
    the compiled region — the supported, fusing pattern; ``vmap(grad)`` *outside*
    a compiled loss is the unsupported ``grad(compiled_fn)`` case). Tries
    ``fullgraph=True`` first and lazily falls back to ``fullgraph=False`` on the
    first graph break (the failure is lazy — it surfaces on first execution).
    """
    full = torch.compile(grad_fn, backend=backend, mode=mode, fullgraph=True)
    fallback = []

    def wrapper(*args, **kwargs):
        if fallback:
            return fallback[0](*args, **kwargs)
        try:
            return full(*args, **kwargs)
        except Exception as e:
            print(
                f"WARNING: torch.compile(fullgraph=True) failed "
                f"({type(e).__name__}: {e}); falling back to fullgraph=False."
            )
            fallback.append(
                torch.compile(grad_fn, backend=backend, mode=mode, fullgraph=False)
            )
            return fallback[0](*args, **kwargs)

    return wrapper


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
        return torch.device("mps"), "Apple Silicon"
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
    try:
        torch.empty((1,), device=device, dtype=dtype)
        return True
    except (RuntimeError, TypeError):
        return False


def _resolve_model_dtype(
    requested_name: str,
    device: torch.device,
) -> tuple[str, torch.dtype, str | None]:
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
            reason = f"Requested dtype '{requested_name}' not supported on {device.type}; using '{fallback_name}'."
            return fallback_name, fallback_dtype, reason
    raise RuntimeError(
        f"No supported dtype for device={device.type}. Requested '{requested_name}'."
    )


def _load_streaming_subset(
    dataset_name: str,
    dataset_subset: str | None,
    dataset_split: str,
    dataset_text_field: str,
    total_needed: int,
) -> Dataset:
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
            f"Text field '{dataset_text_field}' not found. Available: {list(rows[0].keys())}"
        )
    if len(rows) < total_needed:
        raise ValueError(
            f"Stream ended after {len(rows)} examples, need {total_needed}."
        )
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def make_lr_schedule(
    base_lr: float,
    total_steps: int,
    *,
    kind: str = "none",
    min_ratio: float = 0.0,
    warmup_steps: int = 0,
) -> Schedule:
    """Build an :data:`opaque.scheduling.types.Schedule` callable.

    The same callable is handed both to the torchopt optimizer (via
    ``scale_by_schedule`` machinery) and to the MF noise strategy's
    ``lr_schedule`` argument (for BandMF / BLT — see
    :func:`opaque.dpftrl.noise._band_mf._momentum_workload_coef`).  Both
    consumers query identical per-step LRs.

    Args:
        base_lr: Peak learning rate.
        total_steps: Total number of training steps.
        kind: Decay curve — ``none`` (constant), ``cosine``, ``linear``,
            ``sqrt`` (inverse-sqrt).
        min_ratio: Floor LR as a fraction of peak (cosine / linear only;
            sqrt decays to zero asymptotically).  Must be in [0, 1].
        warmup_steps: Linear warmup from 0 → peak LR over this many steps
            (any kind).

    Returns:
        A ``Callable[[int], float]`` returning the LR at any non-negative
        step.
    """
    if not 0.0 <= min_ratio <= 1.0:
        raise ValueError(f"min_ratio must be in [0, 1], got {min_ratio}")
    warmup = max(0, int(warmup_steps))
    decay_span = max(1, total_steps - warmup)
    lr_min = base_lr * min_ratio

    if kind == "cosine":
        base = cosine_schedule(
            init_value=base_lr,
            end_value=lr_min,
            transition_steps=decay_span,
            transition_begin=warmup,
        )
    elif kind == "linear":
        base = linear_schedule(
            init_value=base_lr,
            end_value=lr_min,
            transition_steps=decay_span,
            transition_begin=warmup,
        )
    elif kind == "sqrt":
        # Inverse-sqrt timescale defaults to warmup when set, otherwise
        # to the full run.
        base = inverse_sqrt_schedule(
            init_value=base_lr,
            transition_steps=warmup if warmup > 0 else max(1, total_steps),
            transition_begin=warmup,
        )
    elif kind == "none":
        base = lambda _step: base_lr  # noqa: E731
    else:
        raise ValueError(f"Unknown LR schedule kind: {kind!r}")

    return with_warmup(base, transition_steps=warmup) if warmup > 0 else base


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _require_configured(parser, args, required=("model_name", "dataset")):
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        flags = ", ".join("--" + name.replace("_", "-") for name in missing)
        parser.error(
            f"missing required configuration: {flags}. "
            f"Pass them directly or select a --preset (e.g. --preset smoke)."
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="DP-FTRL training with matrix factorization noise"
    )

    parser.add_argument(
        "--preset",
        type=str,
        choices=["smoke", "mellum-kstack", "mellum2-kstack"],
        default=None,
        help="Optional preset that fills in any unset arguments. Omit it to "
        "configure the run directly (at least --model-name and --dataset).",
    )

    # Model
    model_g = parser.add_argument_group("model")
    model_g.add_argument("--model-name", type=str, default=None)
    model_g.add_argument(
        "--attention", type=str, choices=["eager", "sdpa"], default="sdpa"
    )
    model_g.add_argument(
        "--sdpa-backend",
        type=str,
        choices=["flash", "efficient", "cudnn", "math"],
        default=None,
    )
    model_g.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )

    # Data
    data_g = parser.add_argument_group("data")
    data_g.add_argument("--dataset", type=str, default=None)
    data_g.add_argument(
        "--dataset-subset", dest="dataset_subset", type=str, default=None
    )
    data_g.add_argument("--dataset-split", type=str, default="train")
    data_g.add_argument("--dataset-text-field", type=str, default="text")
    data_g.add_argument("--num-train-samples", type=int, default=5000)
    data_g.add_argument("--num-eval-samples", type=int, default=100)
    data_g.add_argument("--max-seq-len", type=int, default=512)

    # Training
    train_g = parser.add_argument_group("training")
    train_g.add_argument("--batch-size", type=int, default=16)
    train_g.add_argument("--eval-batch-size", type=int, default=None)
    train_g.add_argument("--num-epochs", type=int, default=3)
    train_g.add_argument("--learning-rate", type=float, default=5e-4)
    train_g.add_argument(
        "--optimizer",
        type=str,
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
        default="sgd",
        help=(
            "Optimizer.  ``sgd`` is the canonical DP-FTRL baseline (sgd, "
            "Polyak momentum).  ``adam`` / ``adamw`` / ``ademamix`` are "
            "Adam-family adaptive optimizers; pair with ``--second-moment`` "
            "to activate a private squared-gradient stream (``adamw`` / "
            "``ademamix`` only — the others fall back to single-stream).  "
            "``lion`` is sign-of-momentum; works under MF noise but has "
            "no ``v`` so ``--second-moment`` is auto-disabled.  "
            "``adafactor`` / ``rmsprop`` / ``adagrad`` are second-moment-"
            "only optimizers (no first-moment EMA, so the MF workload is "
            "the identity — equivalent in noise structure to DP-SGD; the "
            "MF correlation is wasted unless paired with ``--mechanism "
            "identity``).  ``--noise-bias-correction`` is plumbed through "
            "for the optimizers that support it; under MF noise the "
            "optimizer reads the per-step *realized* σ from "
            "``NoisedPytree.noise_stddev`` so the second-moment EMA "
            "debias is correct for every strategy."
        ),
    )
    train_g.add_argument(
        "--second-moment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Activate private second-moment noise: ``mf_gaussian_noise`` produces a "
            "privately-estimated ``g²`` stream alongside noised gradients, "
            "and Opaque optimizers consume it automatically.  Joint noise "
            "uses sensitivity-proportional Mahalanobis allocation: privacy "
            "accounting is the underlying MF mechanism at the same noise "
            "multiplier — no extra cost.  Requires an Adam-family optimizer "
            "(``adamw`` or ``ademamix``) with a second moment to consume "
            "the paired stream.  Off by default."
        ),
    )
    train_g.add_argument(
        "--momentum",
        type=float,
        default=0.95,
        help="Polyak momentum for SGD (default: 0.95 per BandMF paper)",
    )
    train_g.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Optimizer weight decay: sgd L2-style coefficient, or "
        "adamw decoupled WD (default 0).",
    )
    train_g.add_argument(
        "--noise-bias-correction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable DP noise-variance bias correction on optimizers that "
            "support it (adam/adamw/ademamix/rmsprop/adagrad/adafactor).  "
            "Silently ignored on sgd/lion.  Under MF noise the optimizer "
            "now reads the per-step *realized* σ "
            "(= base σ · ‖row_t(C^-1)‖) from ``NoisedPytree.noise_stddev`` "
            "(see ``test_realized_stddev.py`` for the bug fix) so the "
            "second-moment EMA debias is correct for every MF strategy.  "
            "Off by default; flip to ``--noise-bias-correction`` when the "
            "training-time gradient distribution justifies it (see "
            "docs/user-guide/optimizers.md)."
        ),
    )
    train_g.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="Adam first moment decay (default: 0.9). Ignored for --optimizer sgd.",
    )
    train_g.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="Adam second moment decay (default: 0.999). Ignored for --optimizer sgd.",
    )
    train_g.add_argument(
        "--adam-eps",
        type=float,
        default=1e-8,
        help="Adam epsilon (default: 1e-8). Ignored for --optimizer sgd.",
    )
    train_g.add_argument(
        "--lr-schedule",
        type=str,
        choices=["none", "cosine", "linear", "sqrt"],
        default="none",
        help=(
            "LR decay curve applied to ``--learning-rate``.  ``cosine`` "
            "decays smoothly from peak to ``--learning-rate * "
            "--lr-min-ratio``; ``linear`` decays linearly to the same "
            "floor; ``sqrt`` decays as inverse-sqrt (Adagrad-mimic).  "
            "Default ``none`` keeps the LR constant.  Only ``band_mf`` / "
            "``blt`` mechanisms model the schedule in their noise "
            "correlation; for ``bsr`` / ``bisr`` / ``lambda_cgd`` / "
            "``identity`` / ``none`` the optimizer-side schedule is "
            "applied but the noise stays tuned for the constant-LR "
            "workload — privacy still holds, utility may be suboptimal "
            "(startup warning is printed)."
        ),
    )
    train_g.add_argument(
        "--lr-min-ratio",
        type=float,
        default=0.0,
        help=(
            "Floor LR as a fraction of peak for cosine/linear schedules "
            "(default 0 = decay to zero)."
        ),
    )
    train_g.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=0,
        help=(
            "Linear warmup from 0 to peak LR over this many steps "
            "(applied to any schedule)."
        ),
    )
    train_g.add_argument("--log-steps", type=int, default=1)
    train_g.add_argument("--eval-steps", type=int, default=10)
    train_g.add_argument("--max-steps", type=int, default=None)
    train_g.add_argument("--seed", type=int, default=42)
    train_g.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False
    )
    train_g.add_argument(
        "--activation-offloading", action=argparse.BooleanOptionalAction, default=False
    )

    # LoRA
    lora_g = parser.add_argument_group("lora")
    lora_g.add_argument("--lora-r", type=int, default=4)
    lora_g.add_argument("--lora-alpha", type=int, default=8)
    lora_g.add_argument(
        "--lora-modules",
        type=str,
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # DP / MF mechanism
    dp_g = parser.add_argument_group("dp", "DP-FTRL mechanism and clipping")
    dp_g.add_argument(
        "--mechanism",
        type=str,
        default="band_mf",
        choices=["band_mf", "blt", "lambda_cgd", "bisr", "bsr", "identity", "none"],
        help="MF mechanism: band_mf, blt, lambda_cgd, bisr, bsr, identity, none (non-DP).",
    )
    dp_g.add_argument(
        "--clipping-norm", type=float, default=0.9, help="Fixed clipping norm"
    )
    dp_g.add_argument(
        "--clipping-mode",
        type=str,
        choices=["fixed", "auto"],
        default="fixed",
        help="Clipping mode: 'fixed' (clipped_grad, threshold = --clipping-norm) "
        "or 'auto' (AUTO-S smooth scaling, sensitivity bound = --clipping-norm). "
        "AUTO-S satisfies MF's constant per-record sensitivity invariant; both "
        "modes compose with --per-group-clipping and --second-moment.",
    )
    dp_g.add_argument(
        "--auto-gamma",
        type=float,
        default=0.01,
        help="AUTO-S denominator stabilizer γ (only used with --clipping-mode auto).",
    )
    dp_g.add_argument(
        "--per-group-clipping",
        type=str,
        nargs="+",
        default=None,
        metavar="PATTERN=NORM",
        help="Per-group clipping norms as PATTERN=NORM pairs (e.g. q_proj=0.9 v_proj=0.5 "
        "for --preset smoke SmolLM2 LoRA, or q_proj=0.5 fallback=1.0 for Mellum presets). "
        "Each trainable param must match exactly one pattern substring. "
        "Use 'fallback=NORM' as catch-all.  Incompatible with adaptive clipping; "
        "MF ``mf_gaussian_noise`` uses the same Mahalanobis allocation as DP-SGD Gaussian.",
    )
    dp_g.add_argument("--microbatch-size", type=int, default=None)
    dp_g.add_argument(
        "--kernel-patches",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("OPAQUE_NO_KERNEL_PATCH", "0") != "1",
        help="Apply the opaque Triton speed kernels (rope/rms_norm/activation/"
        "fused-CE) to the model. On by default (auto-falls back to eager on "
        "non-CUDA hosts). --no-kernel-patches forces the eager baseline; the "
        "compat vmap-safety wrappers — including the load-bearing MoE experts "
        "patch — plus kv_cache and PEFT kernels stay on. Default also follows "
        "OPAQUE_NO_KERNEL_PATCH=1.",
    )
    dp_g.add_argument(
        "--torch-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="torch.compile the DP per-example grad step (vmap(grad) + clipping). "
        "Off by default; a large speedup on GPU/MPS once warm (the first step "
        "pays the compile cost).",
    )
    dp_g.add_argument(
        "--torch-compile-backend",
        type=str,
        default="inductor",
        help="torch.compile backend (default: inductor).",
    )
    dp_g.add_argument(
        "--torch-compile-mode",
        type=str,
        default="default",
        choices=[
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
        help="torch.compile mode (default: 'default').",
    )
    dp_g.add_argument(
        "--bands",
        type=int,
        default=8,
        help="Band count for band_mf mechanism and BandMF subsampling.",
    )
    dp_g.add_argument(
        "--band-mf-sampling",
        type=str,
        choices=["poisson", "b_min_sep"],
        default="poisson",
        help="BandMF data subsampling: poisson (default) or b_min_sep (Dong & Ganesh 2026).",
    )
    dp_g.add_argument(
        "--truncated-batch-size",
        type=int,
        default=None,
        help="Optional cap on per-step batch size (truncated Poisson). "
        "Matched privacy accounting exists only for --mechanism identity. "
        "With --mechanism band_mf --band-mf-sampling poisson the accountant "
        "rejects the combination at calibration time. Silently ignored for "
        "--mechanism band_mf --band-mf-sampling b_min_sep and for blt, "
        "lambda_cgd, bisr, bsr, none (no Poisson sampling).",
    )
    dp_g.add_argument(
        "--mc-resolution",
        type=float,
        default=1e-5,
        help="Maximum unresolved Monte Carlo mass for b_min_sep and BnB accounting.",
    )
    dp_g.add_argument(
        "--mc-failure-probability",
        type=float,
        default=1e-6,
        help="Failure probability of the simultaneous Monte Carlo confidence bound.",
    )
    dp_g.add_argument(
        "--max-buffers",
        type=int,
        default=10,
        help="Maximum BLT buffers to try (higher = better noise, slower init).",
    )
    dp_g.add_argument(
        "--lambda_",
        type=float,
        default=0.9,
        help="Correlation coefficient for lambda_cgd mechanism (0=DP-SGD, higher=more correlation).",
    )
    dp_g.add_argument(
        "--bisr-bandwidth",
        type=int,
        default=4,
        help="Bandwidth for BISR mechanism (>= 2). Higher = better utility, more PRNG replays.",
    )
    dp_g.add_argument(
        "--bsr-bandwidth",
        type=int,
        default=8,
        help="Bandwidth p for BSR mechanism (>= 1). Closed-form coefficients; no optimizer.",
    )
    dp_g.add_argument(
        "--bsr-alpha",
        type=float,
        default=1.0,
        help="BSR workload α in (0,1] (paper); must satisfy α>β where β comes from "
        "--momentum (SGD) or --beta1 (Adam). Ignored unless --mechanism bsr.",
    )

    # Privacy
    priv_g = parser.add_argument_group("privacy")
    priv_g.add_argument("--target-epsilon", type=float, default=3.0)
    priv_g.add_argument("--target-delta", type=float, default=None)
    priv_g.add_argument(
        "--noise-multiplier",
        type=float,
        default=None,
        help="Fixed noise multiplier (skip calibration)",
    )
    priv_g.add_argument("--calibration-min", type=float, default=0.1)
    priv_g.add_argument("--calibration-max", type=float, default=20.0)
    priv_g.add_argument("--calibration-tolerance", type=float, default=1e-3)

    audit_g = parser.add_argument_group("audit", "Empirical privacy auditing")
    audit_g.add_argument(
        "--audit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable empirical auditing (disabled by default)",
    )
    audit_g.add_argument(
        "--audit-canaries",
        type=int,
        default=1000,
        help="Number of canaries for one-run auditing",
    )
    audit_g.add_argument(
        "--audit-method",
        choices=["gdp", "eps_delta"],
        default="gdp",
        help="Which audit method's ε to report ('gdp' = μ-GDP, recommended for Gaussian-DP mechanisms like DP-FTRL; 'eps_delta' = mechanism-agnostic (ε, δ)-DP fallback)",
    )
    audit_g.add_argument(
        "--audit-batch-size",
        type=int,
        default=None,
        help="Batch size for auditing scoring (default: same as training batch size; forward-only so less memory than training)",
    )

    # W&B
    track_g = parser.add_argument_group("tracking")
    track_g.add_argument("--no-wandb", action="store_true")
    track_g.add_argument(
        "--wandb-project", type=str, default=os.environ.get("WANDB_PROJECT", "opaque")
    )
    track_g.add_argument(
        "--wandb-run-name",
        type=str,
        default=os.environ.get("WANDB_NAME") or os.environ.get("RUN_NAME"),
    )
    track_g.add_argument(
        "--wandb-entity", type=str, default=os.environ.get("WANDB_ENTITY")
    )

    args = parser.parse_args()

    # Track which options were explicitly provided on CLI
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
        _set("learning_rate", 5e-4)
        _set("lora_r", 4)
        _set("lora_alpha", 8)
        _set("max_seq_len", 512)
        _set("lora_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
        _set("dtype", "bfloat16")
    elif args.preset == "mellum-kstack":
        _set("model_name", "JetBrains/Mellum-4b-base")
        _set("dataset", "JetBrains/KStack")
        _set("dataset_text_field", "content")
        _set("num_train_samples", 500000)
        _set("num_eval_samples", 1000)
        _set("num_epochs", 8)
        _set("batch_size", 256)
        _set("log_steps", 10)
        _set("eval_steps", 25)
        _set("target_epsilon", 3.0)
        _set("learning_rate", 2e-3)
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
        _set("microbatch_size", 16)
        _set("bands", 64)
        _set("mechanism", "band_mf")
        _set("band_mf_sampling", "b_min_sep")
        # ~5% of total steps as warmup; the preset's defaults give the same
        # number of steps as the old ``warmup_frac=0.05``.
        _set("lr_warmup_steps", 0)
    elif args.preset == "mellum2-kstack":
        # Mellum2-12B-A2.5B (MoE) + KStack under DP-FTRL (band-MF) at ε=3. LoRA on
        # attention projections only; routed experts are stacked nn.Parameter weights.
        _set("model_name", "JetBrains/Mellum2-12B-A2.5B-Base")
        _set("dataset", "JetBrains/KStack")
        _set("dataset_text_field", "content")
        _set("num_train_samples", 500000)
        _set("num_eval_samples", 1000)
        _set("num_epochs", 8)
        _set("batch_size", 256)
        _set("log_steps", 10)
        _set("eval_steps", 25)
        _set("target_epsilon", 3.0)
        _set("learning_rate", 2e-3)
        _set("lora_r", 16)
        _set("lora_alpha", 32)
        _set("max_seq_len", 1024)
        _set("lora_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
        _set("dtype", "bfloat16")
        _set("microbatch_size", 8)
        _set("bands", 64)
        _set("mechanism", "band_mf")
        _set("band_mf_sampling", "b_min_sep")
        _set("lr_warmup_steps", 0)

    _require_configured(parser, args)

    if args.microbatch_size == 0:
        args.microbatch_size = None
    if args.eval_batch_size is None:
        args.eval_batch_size = args.microbatch_size or args.batch_size
    if args.audit_batch_size is None:
        args.audit_batch_size = args.eval_batch_size

    # μ-GDP auditing has no meaningful answer at δ = 0 (pure DP is incompatible
    # with Gaussian DP).  Fail fast instead of crashing inside the audit.
    if args.audit and args.audit_method == "gdp" and args.target_delta <= 0:
        raise SystemExit(
            "--audit-method gdp requires --target-delta > 0 "
            f"(got {args.target_delta}); use --audit-method eps_delta for pure DP."
        )

    if args.per_group_clipping:
        parsed: dict[str, float] = {}
        fallback_value = None
        for item in args.per_group_clipping:
            if "=" not in item:
                parser.error(
                    f"--per-group-clipping values must be PATTERN=NORM, got '{item}'"
                )
            pattern, value = item.split("=", 1)
            try:
                norm = float(value)
            except ValueError:
                parser.error(
                    f"--per-group-clipping norm must be a number, got '{value}' "
                    f"in '{item}'"
                )
            if pattern == "fallback":
                fallback_value = norm
            else:
                parsed[pattern] = norm
        args.per_group_clipping = parsed
        args.per_group_clipping_fallback = fallback_value

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    is_ddp, rank, world_size, local_rank = _init_distributed()
    is_main_process = rank == 0

    if is_main_process:
        print("=" * 80)
        print("DP-FTRL Training (Matrix Factorization Noise)")
        print("=" * 80)

    # --- W&B ---
    use_wandb = wandb is not None and (not args.no_wandb) and is_main_process
    if use_wandb:
        if args.wandb_run_name is None:
            model_short = args.model_name.split("/")[-1]
            args.wandb_run_name = (
                f"ftrl-{args.mechanism}_{model_short}_eps{args.target_epsilon}"
                f"_lr{args.learning_rate}_m{args.momentum}"
            )
        if not os.environ.get("WANDB_MODE"):
            os.environ["WANDB_MODE"] = (
                "online" if os.environ.get("WANDB_API_KEY") else "offline"
            )
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args),
        )
        print(f"W&B initialized (mode: {os.environ.get('WANDB_MODE', 'online')})")

    # --- Device ---
    device, device_name = _select_device(local_rank if is_ddp else None)
    if is_main_process:
        print(f"\nDevice: {device} ({device_name})")
        if is_ddp:
            print(
                f"Distributed mode: rank={rank}/{world_size}, local_rank={local_rank}"
            )

    torch.manual_seed(args.seed)

    # --- Attention ---
    # Force eager on MPS only while the live torch has the autocast-under-vmap
    # SDPA bug (probe-gated, so it auto-drops once pytorch/pytorch#187282 lands).
    use_eager = args.attention == "eager" or (
        device.type == "mps" and sdpa_autocast_under_vmap_broken(device.type)
    )
    if not use_eager and args.sdpa_backend is not None:
        backends = {
            "flash": torch.backends.cuda.enable_flash_sdp,
            "efficient": torch.backends.cuda.enable_mem_efficient_sdp,
            "cudnn": torch.backends.cuda.enable_cudnn_sdp,
            "math": torch.backends.cuda.enable_math_sdp,
        }
        for name, setter in backends.items():
            setter(name == args.sdpa_backend)

    # --- Model ---
    print(f"\nLoading model: {args.model_name}...")
    config = AutoConfig.from_pretrained(args.model_name)
    for attr in [
        "attn_pdrop",
        "resid_pdrop",
        "embd_pdrop",
        "attention_dropout",
        "hidden_dropout",
        "dropout",
        "attn_dropout",
        "ffn_dropout",
    ]:
        if hasattr(config, attr):
            setattr(config, attr, 0.0)

    dtype_name, torch_dtype, dtype_warning = _resolve_model_dtype(args.dtype, device)
    args.dtype = dtype_name
    print(f"  Dtype: {dtype_name}")
    if dtype_warning:
        print(f"  Warning: {dtype_warning}")

    model_kwargs = {"config": config, "dtype": torch_dtype, "trust_remote_code": True}
    if use_eager:
        model_kwargs["attn_implementation"] = "eager"

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

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- LoRA ---
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
    # Loss is the only consumer of the forward output in this script, so the
    # default opts into the fused linear+CE kernel (skips ``lm_head`` and returns
    # ``logits=None`` on the fast path — see ``apply_model_patches`` docs).
    # ``--no-kernel-patches`` passes ``kernels=False`` for an eager baseline: it
    # disables only the Triton speed kernels and fused linear-CE, while the
    # ``compat`` vmap-safety wrappers (the load-bearing MoE experts patch,
    # kv_cache, PEFT kernels) stay on.
    if args.kernel_patches:
        apply_model_patches(model, kernels=True, fused_linear_cross_entropy=True)
    else:
        print("Kernel patches: DISABLED (eager baseline; compat/MoE/PEFT stay on)")
        apply_model_patches(model, kernels=False, fused_linear_cross_entropy=False)
    model.print_trainable_parameters()
    print_memory(device, "After LoRA")

    # --- Data ---
    print(f"\nLoading dataset: {args.dataset}...")
    total_needed = args.num_train_samples + args.num_eval_samples
    dataset = _load_streaming_subset(
        args.dataset,
        args.dataset_subset,
        args.dataset_split,
        args.dataset_text_field,
        total_needed,
    )

    eval_dataset = dataset.take(args.num_eval_samples)
    train_dataset = dataset.skip(args.num_eval_samples).take(args.num_train_samples)

    def tokenize_function(examples):
        return tokenizer(
            examples[args.dataset_text_field],
            truncation=True,
            max_length=args.max_seq_len,
        )

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
    print(f"Prepared: {len(train_dataset)} train, {len(eval_dataset)} eval")

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

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    @empty_collate
    def collate(examples):
        batch = data_collator(examples)
        return (batch["input_ids"].to(device),)

    global_train_size = len(train_dataset)

    # BLT uses fixed iteration order (sequential DataLoader, drop_last=True),
    # so shuffle once to randomize which examples land in which batch.
    # λ-CGD and BISR use BnB sampling which randomizes assignment itself.
    # Shuffle BEFORE sharding so every rank sees the same global order;
    # local_shard then carves the deterministic contiguous slice.
    if args.mechanism == "blt":
        train_dataset = train_dataset.shuffle(seed=args.seed)

    # Sharded DDP: trim to a multiple of ``world_size`` so every shard has
    # equal length (keeps ``sync()`` / ``sum_gradients_()`` in lockstep).
    if is_ddp:
        trimmed_size = (len(train_dataset) // world_size) * world_size
        if trimmed_size < len(train_dataset):
            train_dataset = train_dataset.select(range(trimmed_size))
        train_dataset = local_shard(train_dataset, rank=rank, world_size=world_size)

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
        drop_last=False,
    )

    # --- Sampling ---
    # ``sample_rate`` is the per-example global Poisson probability and is
    # unchanged under DDP: disjoint shards mean each example is still drawn
    # with probability ``sample_rate`` across ranks.
    sample_rate = args.batch_size / global_train_size
    expected_steps_per_epoch = global_train_size // args.batch_size
    # ``total_steps`` is the full training horizon (``num_epochs *
    # steps_per_epoch``).  ``--max-steps`` is an early-termination knob
    # (semantically "stop at step N"), NOT a horizon override: the sampler
    # and every privacy / accounting / strategy object below are sized
    # against the full horizon so the executed stream, calibration, MF
    # workload coefficients, and the LR schedule all describe the same
    # un-truncated training run.  Only the training loop terminates early
    # when ``global_step >= args.max_steps``.
    total_steps = args.num_epochs * expected_steps_per_epoch
    stop_at_step = (
        min(total_steps, args.max_steps) if args.max_steps is not None else total_steps
    )
    # The global expected batch is split across ranks, so the per-rank
    # batch_size for non-Poisson samplers (BLT sequential and BnB) is
    # reduced.  Poisson samplers handle this implicitly via ``sample_rate``
    # on the shard.
    if is_ddp:
        # BLT (SequentialBatchSampler) and BnB consume a fixed per-rank
        # batch size; if ``--batch-size`` is not divisible by
        # ``world_size`` we'd silently train on a different global batch
        # than the one accounting / clipping / LR schedule were sized
        # against.  Reject the run early instead of masking the drift.
        if args.batch_size < world_size or args.batch_size % world_size != 0:
            raise ValueError(
                f"--batch-size ({args.batch_size}) must be a positive "
                f"multiple of world_size ({world_size}) under DDP so the "
                f"per-rank non-Poisson sampler reproduces the global batch."
            )
        per_rank_batch_size = args.batch_size // world_size
    else:
        per_rank_batch_size = args.batch_size

    # Sampler keys fold in ``rank`` so each shard draws independent
    # examples; in single-rank mode ``rank == 0`` and the keys reduce to
    # the non-distributed values.  BnB / sequential samplers consume a
    # base key that is similarly fold_in(seed, rank) so each shard gets
    # its own deterministic partition.
    base_sampler_key = fold_in(key(args.seed), rank) if is_ddp else key(args.seed)

    # Create ONE sampler spanning the full ``total_steps`` stream.  The
    # samplers are resumable streams (their ``_consumed`` cursor persists
    # across ``__iter__`` calls), so a single object carries its
    # participation contract — fixed BnB partition, b-min-sep cooldown,
    # cyclic band phase — across every epoch boundary of the run.
    # Rebuilding per epoch would redraw the randomized samplers' Markov /
    # partition state at each boundary, silently violating the
    # min-separation contract the accounting below assumes over
    # ``total_steps``.  Constructor validation (bin counts / divisibility
    # / empty dataset) still fires here at config time, before model
    # loading.
    if args.mechanism == "band_mf":
        p0 = sample_rate  # E[batch]/|D| per iteration (same as ``dpftrl_acc.poisson`` regime)
        sampling_prob = 0.0
        p_bms = 0.0
        if args.band_mf_sampling == "poisson":
            sampling_prob = args.batch_size * args.bands / global_train_size
            if sampling_prob > 1.0:
                raise ValueError(
                    f"poisson sampling_prob = {sampling_prob:.4f} > 1.0. "
                    f"Reduce --bands ({args.bands}) or --batch-size ({args.batch_size})."
                )
            train_sampler = CyclicPoissonSampler(
                train_dataset,
                sample_rate=sampling_prob,
                bands=args.bands,
                n_steps=total_steps,
                truncated_batch_size=args.truncated_batch_size,
                key=base_sampler_key,
            )
        else:
            if args.bands > 1 and p0 * (args.bands - 1) >= 1.0:
                raise ValueError(
                    f"b_min_sep requires p_0 < 1/(bands-1); got p_0={p0:.6f}, bands={args.bands}."
                )
            denom = 1.0 - p0 * max(0, args.bands - 1)
            p_bms = p0 / denom if args.bands > 1 else p0
            if p_bms > 1.0:
                raise ValueError(
                    f"b_min_sep per-iteration p = {p_bms:.4f} > 1.0; reduce batch size or bands."
                )
            train_sampler = BMinSepSampler(
                train_dataset,
                bands=args.bands,
                sampling_prob=p_bms,
                n_steps=total_steps,
                key=base_sampler_key,
            )

    elif args.mechanism == "blt":
        # Deterministic fixed order cycling for the whole run: min_sep =
        # steps/epoch and max_participations = num_epochs — the BLT
        # accounting contract — are enforced by the sampler itself.
        train_sampler = SequentialBatchSampler(
            train_dataset,
            batch_size=per_rank_batch_size,
            n_steps=total_steps,
        )

    elif args.mechanism in ("lambda_cgd", "bisr", "bsr"):
        # One BnB sampler for the run: the bin assignment is drawn once and
        # round-robins over ``total_steps`` slots (per-bin participation =
        # num_epochs), as BnB privacy accounting requires (Lemma 3.2 of
        # Choquette-Choo et al. 2024).  Under DDP each rank partitions its
        # disjoint shard into the same number of bins; combined across
        # ranks every global example still appears in exactly one bin so
        # BnB privacy holds unchanged.  We deliberately use the un-folded
        # ``key(args.seed)`` (rather than ``base_sampler_key``, which is
        # rank-folded for randomized samplers): together with the equal
        # shard sizes guaranteed above this gives every rank the same
        # empty/non-empty bin pattern within its local index space, so all
        # ranks yield an identical number of batches and the cross-rank
        # collectives in the training loop stay in lockstep.
        train_sampler = BallsInBinsSampler(
            train_dataset,
            num_bins=expected_steps_per_epoch,
            n_steps=total_steps,
            key=key(args.seed),
        )

    else:  # identity, none
        train_sampler = CyclicPoissonSampler(
            train_dataset,
            sample_rate=sample_rate,
            n_steps=total_steps,
            truncated_batch_size=args.truncated_batch_size,
            key=base_sampler_key,
        )

    if is_main_process:
        print("\nSampling:")
        if is_ddp:
            print(f"  DDP mode: sharded (world_size={world_size})")
            print(f"  Per-rank shard size: {len(train_dataset)}")
            print(f"  Per-rank batch (non-Poisson): {per_rank_batch_size}")
        print(f"  Mechanism: {args.mechanism}")
        if args.mechanism == "band_mf":
            if args.band_mf_sampling == "poisson":
                print(f"  Sampler: poisson (bands={args.bands}, q={sampling_prob:.6f})")
            else:
                print(
                    f"  Sampler: b_min_sep (bands={args.bands}, p_0={p0:.6f}, p={p_bms:.6f})"
                )
        elif args.mechanism == "blt":
            print("  Sampler: sequential (fixed order, drop_last=True)")
            print(f"  min_sep: {expected_steps_per_epoch} (= steps/epoch)")
            print(f"  max_participations: {args.num_epochs}")
        elif args.mechanism == "lambda_cgd":
            print(
                f"  Sampler: balls-in-bins (k={expected_steps_per_epoch}, fixed partition, reused across epochs)"
            )
            print(f"  DP-λCGD: λ={args.lambda_}")
        elif args.mechanism == "bisr":
            print(
                f"  Sampler: balls-in-bins (k={expected_steps_per_epoch}, fixed partition, reused across epochs)"
            )
            print(f"  BISR bandwidth: {args.bisr_bandwidth}")
        elif args.mechanism == "bsr":
            print(
                f"  Sampler: balls-in-bins (k={expected_steps_per_epoch}, fixed partition, reused across epochs)"
            )
            print(f"  BSR bandwidth: {args.bsr_bandwidth} (closed-form coefficients)")
        else:
            print(f"  Sampler: poisson (q={sample_rate:.6f})")
        print(f"  Expected batch size: {args.batch_size}")
        print(f"  Expected steps/epoch: {expected_steps_per_epoch}")

    # --- Gradient checkpointing ---
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if is_main_process:
            print("\nGradient checkpointing: enabled")

    offload_ctx = (
        torch.autograd.graph.save_on_cpu(pin_memory=True)
        if args.activation_offloading
        else contextlib.nullcontext()
    )

    # --- Functional conversion ---
    print("\nConverting to functional form...")
    t0 = time.time()
    fmodel, trainable_params, frozen_params = make_functional(
        model,
        disable_autograd_tracking=True,
        partition_trainable=True,
    )
    param_names = list(trainable_params.keys())
    print(f"Trainable parameters: {len(param_names)} (took {time.time() - t0:.1f}s)")
    print_memory(device, "After functional conversion")

    def merged_params(trainable):
        return {**frozen_params, **trainable}

    # Hoist ``pad_token_id`` to a plain int: reading ``tokenizer.pad_token_id``
    # inside the per-example loss hits the tokenizer's C-level ``__getattr__``
    # every call, which Dynamo can't trace (torch.compile graph-breaks there).
    pad_token_id = tokenizer.pad_token_id

    def per_example_loss_fn(trainable, input_ids):
        # Mask pad positions to ``-100`` so training CE scores only real
        # tokens — same masking contract the eval path uses and the same
        # convention Trainer's ``DataCollatorForLanguageModeling`` applies.
        # Without this, the manual DP-FTRL loop trains on unmasked labels
        # while Trainer trains on masked labels, producing systematically
        # different ``train/loss`` curves under identical DP math. ``vmap``-safe.
        labels = torch.where(input_ids == pad_token_id, -100, input_ids)
        output = fmodel(merged_params(trainable), input_ids, labels=labels)
        return output.loss

    # Verified canary scoring: loss_scores builds the canary loader
    # internally and binds each score to its canary identifier.
    def score_canaries(params, reference_scores=None):
        return auditing.loss_scores(
            per_example_loss_fn,
            params,
            batch_argnums=(1,),
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

    def eval_loss(trainable):
        """Token-weighted CE over the eval set (pad tokens masked to ``-100``).

        Returns ``float('nan')`` when the eval set has zero scoring tokens
        (empty ``--num-eval-samples`` or all examples shorter than 2 tokens).
        """
        with torch.no_grad():
            total_loss, total_tokens = 0.0, 0
            for (input_ids,) in eval_loader:
                labels = input_ids.clone()
                labels[labels == pad_token_id] = -100
                output = fmodel(merged_params(trainable), input_ids, labels=labels)
                num_tokens = int((labels[..., 1:] != -100).sum().item())
                if num_tokens == 0:
                    continue
                total_loss += output.loss.item() * num_tokens
                total_tokens += num_tokens
            if total_tokens == 0:
                return float("nan")
            return total_loss / total_tokens

    if args.per_group_clipping:
        clip_norm = per_group(
            trainable_params,
            patterns=args.per_group_clipping,
            fallback=args.per_group_clipping_fallback,
        )
        print("\nPer-group clipping norms:")
        for gname, val in clip_norm.values.items():
            count = sum(1 for g in clip_norm.groups.values() if g == gname)
            print(f"  {gname}: {val:.3f} ({count} params)")
        print(f"  Effective (L2 of group bounds): {clip_norm.effective:.3f}")
    else:
        clip_norm = args.clipping_norm

    if args.clipping_mode == "auto":
        grad_fn, clip_state = auto_clipped_grad(
            per_example_loss_fn,
            argnums=0,
            batch_argnums=(1,),
            R=clip_norm,
            gamma=args.auto_gamma,
            normalize_by=args.batch_size,
            microbatch_size=args.microbatch_size,
            return_aux=True,
            second_moment=args.second_moment,
        )
    else:
        grad_fn, clip_state = clipped_grad(
            per_example_loss_fn,
            argnums=0,
            batch_argnums=(1,),
            clipping_norm=clip_norm,
            normalize_by=args.batch_size,
            microbatch_size=args.microbatch_size,
            return_aux=True,
            second_moment=args.second_moment,
        )

    # Caller-applied torch.compile of the whole DP grad transform (opt-in).
    if args.torch_compile:
        grad_fn = _compile_grad_fn(
            grad_fn,
            backend=args.torch_compile_backend,
            mode=args.torch_compile_mode,
        )

    zeta = (
        clip_norm.effective / args.batch_size
        if isinstance(clip_norm, PerGroup)
        else float(clip_norm) / args.batch_size
    )

    # --- LR schedule ---
    lr_schedule = make_lr_schedule(
        args.learning_rate,
        total_steps,
        kind=args.lr_schedule,
        min_ratio=args.lr_min_ratio,
        warmup_steps=args.lr_warmup_steps,
    )

    # ``band_mf`` / ``blt`` consume ``lr_schedule`` in their MF workload;
    # ``identity`` / ``none`` have no MF correlation at all and don't care;
    # ``bsr`` / ``bisr`` / ``lambda_cgd`` have correlation that ignores LR
    # shaping — using a non-constant LR there silently degrades utility
    # (privacy still holds).  Warn for that latter set only.
    _LR_SCHEDULE_OBLIVIOUS_BUT_CORRELATED = frozenset({"bsr", "bisr", "lambda_cgd"})
    schedule_active = args.lr_schedule != "none" or args.lr_warmup_steps > 0
    if schedule_active and args.mechanism in _LR_SCHEDULE_OBLIVIOUS_BUT_CORRELATED:
        print(
            f"\nWARNING: --lr-schedule={args.lr_schedule!r} / "
            f"--lr-warmup-steps={args.lr_warmup_steps} requested with "
            f"--mechanism {args.mechanism!r}, whose noise correlation "
            f"does not model an LR schedule.  The optimizer-side schedule "
            f"is still applied, but the MF noise stays tuned for the "
            f"constant-LR workload.  Privacy holds; utility may be "
            f"suboptimal.  Switch to --mechanism band_mf|blt for "
            f"schedule-aware noise."
        )

    if schedule_active:
        print(
            f"\nLR schedule: {args.lr_schedule} (peak={args.learning_rate}, "
            f"min={args.learning_rate * args.lr_min_ratio:g}, "
            f"warmup={args.lr_warmup_steps}, total={total_steps})"
        )
    else:
        print(f"\nLR schedule: constant {args.learning_rate} (no warmup)")
    print(f"  Peak LR: {args.learning_rate}")
    print(f"  Total steps: {total_steps}")
    if stop_at_step < total_steps:
        print(f"  Early stop at step: {stop_at_step} (--max-steps)")

    # --- Privacy calibration (single-shot) ---
    if args.target_delta is None:
        args.target_delta = 1.0 / (global_train_size**1.1)

    # Build the accounting mechanism.
    #
    # Two orthogonal toggles:
    #   ``is_adam_family``: optimizer has a first-moment EMA at β₁
    #     (``adamw``, ``ademamix``, ``lion``).  Drives MF workload
    #     momentum.
    #   ``use_second_moment`` (= ``args.second_moment``): switch from
    #     single-stream MF noise to private first+second moment noise.
    #     Requires the optimizer to consume ``SecondMomentNoiseOutput``.
    # "Adam-family" here = optimizer has a first-moment EMA at β₁; drives
    # the MF workload momentum.  ``adafactor`` / ``rmsprop`` / ``adagrad``
    # have only a second-moment accumulator and run on raw gradients in
    # the first-moment slot, so the MF workload momentum stays at 0
    # (configurable via ``--momentum``).
    is_adam_family = args.optimizer in ("adam", "adamw", "ademamix", "lion")
    use_second_moment = args.second_moment

    # Cross-flag validation: warn-and-disable on mismatch instead of raising.
    # The ``--second-moment`` knob is auxiliary (it enables a paired-stream
    # release), so silently degrading to single-stream noise on an
    # incompatible optimizer / mechanism beats failing the run outright.
    _SECOND_MOMENT_OPTIMIZERS = frozenset({"adamw", "ademamix"})
    if use_second_moment and args.optimizer not in _SECOND_MOMENT_OPTIMIZERS:
        print(
            f"\nWARNING: --second-moment requires an Adam-family optimizer "
            f"that consumes ``SecondMomentNoiseOutput`` "
            f"({sorted(_SECOND_MOMENT_OPTIMIZERS)}); got --optimizer "
            f"{args.optimizer!r}.  Disabling --second-moment for this run."
        )
        use_second_moment = False
    if use_second_moment and args.mechanism in ("identity", "none"):
        print(
            f"\nWARNING: --second-moment requires a correlated MF mechanism "
            f"to share the squared-stream Mahalanobis budget; got "
            f"--mechanism {args.mechanism!r}.  Disabling --second-moment "
            f"for this run."
        )
        use_second_moment = False

    def _workload_momentum() -> float:
        """Workload momentum for the primary (first moment) strategy."""
        return args.beta1 if is_adam_family else args.momentum

    def _make_strategy(momentum_override=None, lr_sched=None):
        """Build a strategy recipe for the selected mechanism.

        Strategies are recipes — horizon (``n_steps``), ``min_sep`` and
        ``max_participations`` are owned by the wrapping amplifier and
        the noise factory, not the strategy.  Only ``band_mf`` and
        ``blt`` model an LR schedule in their workload; for the others
        the ``lr_sched`` arg is ignored (the user is warned at startup).
        """
        mom = (
            momentum_override if momentum_override is not None else _workload_momentum()
        )
        if args.mechanism == "band_mf":
            return band_mf_strategy(
                bands=args.bands, momentum=mom, lr_schedule=lr_sched
            )
        elif args.mechanism == "blt":
            return blt_strategy(
                max_buffers=args.max_buffers, momentum=mom, lr_schedule=lr_sched
            )
        elif args.mechanism == "lambda_cgd":
            return lambda_cgd_strategy(lambda_=args.lambda_)
        elif args.mechanism == "bisr":
            return bisr_strategy(bandwidth=args.bisr_bandwidth, momentum=mom)
        elif args.mechanism == "bsr":
            return bsr_strategy(
                bandwidth=args.bsr_bandwidth, alpha=args.bsr_alpha, beta=mom
            )
        elif args.mechanism == "identity":
            return identity_strategy()
        else:
            return None

    strategy = _make_strategy(lr_sched=lr_schedule)

    # No paired-stream wrap on the accountant: the joint Mahalanobis
    # allocation in :func:`mf_gaussian_noise` makes the paired-release PLD
    # identical to the first-moment-only release at the same noise
    # multiplier.

    acc.set_discretization(
        seed=args.seed,
        mc_resolution=args.mc_resolution,
        mc_failure_probability=args.mc_failure_probability,
    )

    if args.mechanism == "band_mf" and strategy is not None:

        def acct_mechanism(nm):
            mechanism = dpftrl_acc.mf_gaussian(nm, strategy)
            if args.band_mf_sampling == "poisson":
                return dpftrl_acc.poisson(
                    mechanism,
                    sample_rate=sampling_prob,
                    n_steps=total_steps,
                    truncated_batch_size=args.truncated_batch_size,
                    dataset_size=(
                        global_train_size
                        if args.truncated_batch_size is not None
                        else None
                    ),
                )
            return dpftrl_acc.b_min_sep(
                mechanism,
                n_steps=total_steps,
                p0=p0,
            )
    elif (
        (args.mechanism == "blt" and strategy is not None)
        or (args.mechanism == "lambda_cgd" and strategy is not None)
        or (args.mechanism == "bisr" and strategy is not None)
        or (args.mechanism == "bsr" and strategy is not None)
    ):

        def acct_mechanism(nm):
            return dpftrl_acc.balls_in_bins(
                dpftrl_acc.mf_gaussian(nm, strategy),
                num_bins=expected_steps_per_epoch,
                n_steps=total_steps,
            )
    elif args.mechanism == "identity":

        def acct_mechanism(nm):
            return dpftrl_acc.poisson(
                dpftrl_acc.mf_gaussian(nm, identity_strategy()),
                sample_rate=sample_rate,
                n_steps=total_steps,
                truncated_batch_size=args.truncated_batch_size,
                dataset_size=(
                    global_train_size if args.truncated_batch_size is not None else None
                ),
            )
    elif args.mechanism == "none":

        def acct_mechanism(nm):
            return acc.nonprivate()
    else:
        raise ValueError(f"Unknown mechanism: {args.mechanism}")

    if args.mechanism == "none":
        noise_multiplier = 0.0
        print("\nNon-DP mode: noise_multiplier=0 (no privacy)")
    elif args.noise_multiplier is not None:
        noise_multiplier = args.noise_multiplier
        print(f"\nFixed noise multiplier: {noise_multiplier:.4f}")
    else:
        print(f"\nCalibrating noise (mechanism={args.mechanism})...")
        print(f"  Target: ε={args.target_epsilon}, δ={args.target_delta:.2e}")
        t0 = time.time()
        cal_min = args.calibration_min
        cal_max = args.calibration_max
        calibration = cal.calibrate(
            cal.epsilon_budget(args.target_epsilon, delta=args.target_delta),
            acct_mechanism,
            param_min=cal_min,
            param_max=cal_max,
            tolerance=args.calibration_tolerance,
        )
        noise_multiplier = calibration.param
        print(
            f"  Calibrated in {time.time() - t0:.1f}s: σ={noise_multiplier:.4f} (ε≈{calibration.achieved:.3f})"
        )

    if use_wandb:
        wandb.config.update(
            {
                "noise_multiplier": noise_multiplier,
                "target_delta": args.target_delta,
                "total_steps": total_steps,
            },
            allow_val_change=True,
        )

    # --- Create MF noise function + optimizer ---
    base_noise_stddev = noise_multiplier * zeta

    if args.mechanism == "none":
        print("\nNon-DP mode: using identity noise with stddev=0...")
    elif args.mechanism == "identity":
        print("\nCreating identity noise (i.i.d. Gaussian, DP-SGD baseline)...")
    elif use_second_moment:
        print(
            f"\nCreating private second-moment noise (β₁={args.beta1}, β₂={args.beta2})..."
        )
    else:
        print(f"\nCreating MF noise (β={args.momentum})...")
    t0 = time.time()

    # Participation context for the noise side: pull straight off the
    # wrapping amplifier so the streaming matrix tracks the calibrated PLD
    # (the :class:`opaque.dpftrl.accounting.amplification.types.MfAmplification`
    # Protocol guarantees every amplifier exposes ``n_steps`` / ``min_sep``
    # / ``max_participations``, including the degenerate-limit values for
    # bare-Poisson Identity).
    if args.mechanism == "none":
        noise_n_steps = total_steps
        noise_min_sep = 1
        noise_max_part = total_steps
    else:
        _amp = acct_mechanism(noise_multiplier)
        noise_n_steps = _amp.n_steps
        noise_min_sep = _amp.min_sep
        noise_max_part = _amp.max_participations

    if use_second_moment and args.mechanism not in ("identity", "none"):
        second_strategy = _make_strategy(
            momentum_override=args.beta2, lr_sched=lr_schedule
        )
        noise_fn, noise_state = mf_gaussian_noise(
            trainable_params,
            strategy,
            n_steps=noise_n_steps,
            min_sep=noise_min_sep,
            max_participations=noise_max_part,
            noise_multiplier=noise_multiplier,
            key=key(args.seed),
            second_moment_strategy=second_strategy,
        )
    elif args.mechanism in ("identity", "none"):
        noise_fn, noise_state = mf_gaussian_noise(
            trainable_params,
            identity_strategy(),
            n_steps=noise_n_steps,
            noise_multiplier=noise_multiplier,
            key=key(args.seed),
        )
    else:
        noise_fn, noise_state = mf_gaussian_noise(
            trainable_params,
            strategy,
            n_steps=noise_n_steps,
            min_sep=noise_min_sep,
            max_participations=noise_max_part,
            noise_multiplier=noise_multiplier,
            key=key(args.seed),
        )
    print(f"  Noise function created in {time.time() - t0:.1f}s")

    lr_callable = lr_schedule

    # ``noise_bias_correction`` is unsound under correlated MF noise: the
    # optimizer's BC reads ``NoisedPytree.noise_stddev`` (the base σ fed to
    # the streaming matrix) and treats it as the per-coordinate per-step
    # variance, but realized per-step variance is ``base_σ² · ‖row_t(C^-1)‖²``
    # — which varies by t for any non-Identity strategy.  Force the BC-aware
    # constructors to disable BC; the math becomes a no-op for the BC path.
    if args.optimizer == "sgd":
        from opaque.optimizers import sgd

        optimizer = sgd(
            lr=lr_callable,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adam":
        from opaque.optimizers import adam

        optimizer = adam(
            lr=lr_callable,
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
            noise_bias_correction=args.noise_bias_correction,
        )
    elif args.optimizer == "adamw":
        from opaque.optimizers import adamw

        # ``--second-moment`` drives the noise side; the same optimizer
        # consumes ``SecondMomentNoiseOutput`` when the noise output carries it.
        optimizer = adamw(
            lr=lr_callable,
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
            noise_bias_correction=args.noise_bias_correction,
        )
    elif args.optimizer == "ademamix":
        from opaque.optimizers import ademamix

        # β₃ and α default to the paper values (0.9999, 5.0).  Expose
        # CLI knobs for them once a real user case appears.
        optimizer = ademamix(
            lr=lr_callable,
            betas=(args.beta1, args.beta2, 0.9999),
            alpha=5.0,
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
            noise_bias_correction=args.noise_bias_correction,
        )
    elif args.optimizer == "lion":
        from opaque.optimizers import lion

        optimizer = lion(
            lr=lr_callable,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adafactor":
        from opaque.optimizers import adafactor

        optimizer = adafactor(
            lr=lr_callable,
            weight_decay=args.weight_decay,
            noise_bias_correction=args.noise_bias_correction,
        )
    elif args.optimizer == "rmsprop":
        from opaque.optimizers import rmsprop

        optimizer = rmsprop(
            lr=lr_callable,
            weight_decay=args.weight_decay,
            noise_bias_correction=args.noise_bias_correction,
        )
    elif args.optimizer == "adagrad":
        from opaque.optimizers import adagrad

        optimizer = adagrad(
            lr=lr_callable,
            weight_decay=args.weight_decay,
            noise_bias_correction=args.noise_bias_correction,
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    opt_state = optimizer.init(trainable_params)

    # --- Diagnostic: compute what identity baseline σ would be ---
    identity_sigma = None
    if args.mechanism not in ("identity", "none") and args.noise_multiplier is None:
        try:

            def identity_acct(nm):
                return dpftrl_acc.poisson(
                    dpftrl_acc.mf_gaussian(nm, identity_strategy()),
                    sample_rate=sample_rate,
                    n_steps=total_steps,
                )

            identity_cal = cal.calibrate(
                cal.epsilon_budget(args.target_epsilon, delta=args.target_delta),
                identity_acct,
                param_min=args.calibration_min,
                param_max=args.calibration_max,
                tolerance=args.calibration_tolerance,
            )
            identity_sigma = identity_cal.param
        except Exception:
            pass  # Non-critical diagnostic

    print("\nDP-FTRL setup:")
    print(f"  Mechanism: {args.mechanism}")
    second_moment_note = " + private second moment" if use_second_moment else ""
    if is_adam_family:
        print(
            f"  Optimizer: {args.optimizer}{second_moment_note} "
            f"(β₁={args.beta1}, β₂={args.beta2}, "
            f"ε={args.adam_eps}, weight_decay={args.weight_decay})"
        )
        if use_second_moment:
            print(
                f"  Workload: EMA β₁={args.beta1} (1st moment), β₂={args.beta2} (2nd moment)"
            )
        else:
            print(f"  Workload: EMA β₁={args.beta1} (single-stream MF)")
        if args.mechanism == "bsr":
            print(
                f"  BSR workload (α={args.bsr_alpha}, β=β₁={args.beta1}): "
                "noise strategy uses paper (α,β); independent of optimizer --weight-decay; require α>β."
            )
    else:
        # sgd / lion (lion technically has β₁ but no second moment; treat like
        # the SGD-style printout since neither consumes a squared-gradient stream).
        print(
            f"  Optimizer: {args.optimizer} (β={args.momentum}, "
            f"weight_decay={args.weight_decay})"
        )
        print(
            f"  Workload: momentum-SGD (β={args.momentum})"
            f"{' [prefix-sum]' if args.momentum == 1.0 else ''}"
        )
        if args.mechanism == "bsr":
            print(
                f"  BSR workload (α={args.bsr_alpha}, β={args.momentum}): "
                "paper (α,β) for noise; optimizer --weight-decay is separate."
            )
            print(
                "  Note: BSR coefficients assume constant LR in the paper; "
                "this script still uses the LR schedule only in the optimizer."
            )
    if isinstance(clip_norm, PerGroup):
        print("  Clipping norm: per-group (fixed patterns)")
        print(
            f"  Sensitivity (effective ζ): {zeta:.6f} (= {clip_norm.effective:.3f} / batch)"
        )
    else:
        print(f"  Clipping norm: {clip_norm} (fixed)")
        print(f"  Sensitivity: {zeta:.6f} (= {clip_norm} / {args.batch_size})")
    print(f"  Noise multiplier (σ): {noise_multiplier:.4f}")
    if use_second_moment and args.mechanism not in ("identity", "none"):
        # Per-stream σ is allocated by the dispatcher
        # (sensitivity-proportional Mahalanobis on the encoded streams) and
        # surfaces at runtime via ``noise_output.{noisy_grads,noisy_squared_grads}.noise_stddev``.
        print("  Paired-stream allocation: sensitivity-proportional Mahalanobis")
    else:
        print(
            f"  Base noise stddev: {base_noise_stddev:.6f} (= {noise_multiplier:.4f} × {zeta:.6f})"
        )
    if identity_sigma is not None:
        ratio = noise_multiplier / identity_sigma
        print(f"  Identity baseline σ: {identity_sigma:.4f} (ratio: {ratio:.2f}×)")
        print(
            f"  → MF needs {ratio:.2f}× more noise to hit ε={args.target_epsilon}; "
            f"correlated structure must compensate"
        )
    print(f"  Microbatch size: {args.microbatch_size}")
    if args.mechanism == "band_mf":
        print(f"  Bands: {args.bands}")
    elif args.mechanism == "blt":
        print(f"  Max buffers: {args.max_buffers}")
        print(f"  Min separation: {expected_steps_per_epoch}")
        print(f"  Max participations: {args.num_epochs}")
    elif args.mechanism == "lambda_cgd":
        print(f"  λ (lambda): {args.lambda_}")
        print("  Bandwidth: 2 (bidiagonal inverse)")
        print("  Column normalization: enabled (Appendix A, exact BnB)")
    elif args.mechanism == "bisr":
        print(f"  BISR bandwidth: {args.bisr_bandwidth}")
        print("  Column normalization: enabled (Appendix A, exact BnB)")
    elif args.mechanism == "bsr":
        bsr_beta = args.beta1 if is_adam_family else args.momentum
        print(
            f"  BSR: bandwidth={args.bsr_bandwidth}, workload (α={args.bsr_alpha}, β={bsr_beta})"
        )

    # ===================================================================
    # Training loop
    # ===================================================================
    print("\n" + "=" * 80)
    print("Starting DP-FTRL training...")
    print("=" * 80)

    losses = []
    clip_rates = []
    global_step = 0

    reset_peak_memory(device)
    print_memory(device, "Before training")

    # DP-FTRL accounting covers the complete declared horizon and is charged
    # once. This remains conservative when --max-steps truncates execution.
    accounting = Accountant()
    accounting |= acct_mechanism(noise_multiplier)

    # Compute reference (untrained) scores for auditing before any training.
    # Paper Algorithm 3: score = loss(w0, x) − loss(wℓ, x), so we need w0 losses.
    if args.audit and audit_cf is not None:
        print("\nComputing reference scores on untrained model...")
        audit_ref_scores = score_canaries(trainable_params)
        print(
            f"  Reference scores: mean={audit_ref_scores.scores.mean():.4f}, std={audit_ref_scores.scores.std():.4f}"
        )

    # Step-0 eval — baseline before any training step.  Logs the calibrated
    # values that downstream per-step metrics also report so the dashboard
    # has continuous lines (no broken first-point).
    initial_eval_loss = eval_loss(trainable_params)
    initial_epsilon = accounting.epsilon_at(args.target_delta)
    initial_clipping_norm = (
        clip_norm.effective if isinstance(clip_norm, PerGroup) else float(clip_norm)
    )
    initial_noise_std = noise_multiplier * (
        clip_norm.effective / args.batch_size
        if isinstance(clip_norm, PerGroup)
        else float(clip_norm) / args.batch_size
    )
    print(f"  → Step 0 eval: loss={initial_eval_loss:.4f}, ε={initial_epsilon:.3f}")
    if use_wandb:
        wandb.log(
            {
                "eval/loss": initial_eval_loss,
                "privacy/epsilon": initial_epsilon,
                "train/lr": float(lr_schedule(0)),
                "train/clipping_norm": initial_clipping_norm,
                "train/noise_std": initial_noise_std,
            },
            step=0,
        )

    # One DataLoader over the full ``total_steps`` stream; "epoch" is now
    # only a derived logging label.  Keep ``num_workers=0``: worker
    # prefetch would advance the sampler cursor ahead of executed steps.
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate,
    )

    for batch in train_loader:
        if global_step >= stop_at_step:
            if args.max_steps is not None:
                print(f"\nReached --max-steps={args.max_steps}, stopping.")
            break
        if global_step % expected_steps_per_epoch == 0:
            print(
                f"\nEpoch {global_step // expected_steps_per_epoch + 1}"
                f"/{args.num_epochs}"
            )
            print("-" * 80)

        (input_ids,) = batch
        batch_size = len(input_ids)

        lr_t = float(lr_schedule(global_step))

        with tracker.train(batch_size=batch_size) as sp:
            with offload_ctx:
                (grads, aux), clip_state = grad_fn(
                    trainable_params,
                    input_ids,
                    state=clip_state,
                )

            # ``grads`` is a ``SecondMomentClippingOutput`` when
            # ``--second-moment`` is on (clipped_grad produced both
            # streams per-example), or a single ``ClippedPytree``
            # otherwise — the noise function dispatches polymorphically.
            if is_ddp:
                clip_state, aux = sync(clip_state, aux)
                if isinstance(grads, SecondMomentClippingOutput):
                    sum_gradients_(grads.grads)
                    sum_gradients_(grads.squared_grads)
                else:
                    sum_gradients_(grads)
            sp.mark("clip")

            noisy_grads, noise_state = noise_fn(grads, noise_state)
            # All ranks generate identical noise from the same seed
            # (no rank-fold in the noise key) so the per-rank
            # ``noisy_grads`` already agree.  ``sync(noise_state)``
            # is a cheap cross-rank consistency check on the
            # internal step counter and latched sensitivity bound —
            # see :mod:`opaque.dpftrl.noise._distributed`.
            if is_ddp and not isinstance(noisy_grads, SecondMomentNoiseOutput):
                noise_state = sync(noise_state)
            if isinstance(noisy_grads, SecondMomentNoiseOutput):
                step_noise_stddev = noisy_grads.noisy_grads.noise_stddev
            else:
                step_noise_stddev = noisy_grads.noise_stddev
            sp.mark("noise")

            updates, opt_state = optimizer.update(
                noisy_grads,
                opt_state,
                params=trainable_params,
            )
            trainable_params = torchopt.apply_updates(trainable_params, updates)
            sp.mark("optimizer")

        if batch_size == 0:
            global_step += 1
            continue

        # --- Step metrics ---
        avg_loss = aux.loss_values.mean().item()
        clip_rate = aux.clipping_rate
        mean_grad_norm = aux.grad_norms.mean().item()
        losses.append(avg_loss)
        clip_rates.append(clip_rate)
        global_step += 1

        # --- Logging ---
        if global_step % args.log_steps == 0:
            if use_wandb:
                wb_metrics = {
                    "train/loss": avg_loss,
                    "train/batch_size": batch_size,
                    "train/clipping_norm": (
                        clip_norm.effective
                        if isinstance(clip_norm, PerGroup)
                        else clip_norm
                    ),
                    "train/clip_rate": clip_rate,
                    "train/grad_norm_mean": mean_grad_norm,
                    "train/clipped_grad_norm_mean": (
                        aux.clipped_grad_norms.mean().item()
                        if getattr(aux, "clipped_grad_norms", None) is not None
                        else 0.0
                    ),
                    "train/noise_std": (
                        step_noise_stddev.effective
                        if isinstance(step_noise_stddev, PerGroup)
                        else step_noise_stddev
                    ),
                    "train/lr": lr_t,
                    **tracker.train.last.to_dict(prefix="train/"),
                }
                if (
                    isinstance(clip_norm, PerGroup)
                    and getattr(aux, "group_norms", None) is not None
                ):
                    for gname in clip_norm.values:
                        gn_bound = clip_norm.values[gname]
                        wb_metrics[f"group/clipping_norm/{gname}"] = gn_bound
                        gnorms = aux.group_norms[gname]
                        wb_metrics[f"group/grad_norm/{gname}"] = gnorms.mean().item()
                        gn_clipped = float((gnorms > gn_bound).sum().item())
                        wb_metrics[f"group/clip_rate/{gname}"] = gn_clipped / max(
                            1.0, float(batch_size)
                        )
                        if isinstance(step_noise_stddev, PerGroup):
                            wb_metrics[f"group/noise_std/{gname}"] = (
                                step_noise_stddev.values[gname]
                            )
                wandb.log(wb_metrics, step=global_step)

            last = tracker.train.last
            _e, _s = divmod(global_step - 1, expected_steps_per_epoch)
            print(
                f"Step {global_step:4d} [E{_e + 1} S{_s + 1:3d}/{expected_steps_per_epoch:3d}] | "
                f"BS: {batch_size} | Loss: {avg_loss:.4f} | "
                f"Clip: {clip_rate:.1%} | GradNorm: {mean_grad_norm:.3f} | "
                f"LR: {lr_t:.2e} | "
                f"Time: {last.step_time_sec:.2f}s | Mem: {last.memory_peak_gb:.1f}GB"
            )

        # --- Eval ---
        if global_step % args.eval_steps == 0:
            current_eval_loss = eval_loss(trainable_params)
            epsilon = accounting.epsilon_at(args.target_delta)
            eval_msg = f"  → Eval: loss={current_eval_loss:.4f}, ε={epsilon:.3f}"
            metrics: dict[str, float] = {
                "eval/loss": current_eval_loss,
                "privacy/epsilon": epsilon,
            }
            if args.audit and audit_cf is not None:
                audit_estimate = run_audit(trainable_params)
                if audit_estimate is not None:
                    audit_eps = _audit_method(audit_estimate).epsilon
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

    # ===================================================================
    # Final summary
    # ===================================================================
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset} ({global_train_size} train samples)")
    print(f"\nMechanism: {args.mechanism}")
    second_moment_suffix = " + private second moment" if use_second_moment else ""
    if is_adam_family:
        print(
            f"Optimizer: {args.optimizer}{second_moment_suffix} "
            f"(β₁={args.beta1}, β₂={args.beta2})"
        )
    else:
        print(f"Optimizer: {args.optimizer} (β={args.momentum})")
    print("\nTraining results:")
    print(f"  Total steps: {global_step}")
    if losses:
        print(f"  Initial loss: {losses[0]:.4f}")
        print(f"  Final loss: {losses[-1]:.4f}")
        print(f"  Loss reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
    if clip_rates:
        print("\nClipping:")
        if isinstance(clip_norm, PerGroup):
            print(f"  Per-group clip (effective): {clip_norm.effective:.3f}")
        else:
            print(f"  Fixed norm: {clip_norm}")
        print(f"  Average clip rate: {sum(clip_rates) / len(clip_rates):.2%}")

    # Single-shot accounting
    if args.mechanism == "none":
        print("\nPrivacy: Non-DP baseline (no privacy guarantee)")
    else:
        final_epsilon = acct_mechanism(noise_multiplier).epsilon_at(args.target_delta)
        print("\nPrivacy (single-shot):")
        print(f"  Target: ε={args.target_epsilon:.3f}, δ={args.target_delta:.2e}")
        print(f"  Noise multiplier: {noise_multiplier:.4f}")
        print(f"  Final ε (theoretical): {final_epsilon:.4f}")
        # Evaluations report this same declared full-horizon bound.

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
    exit(main())
