# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""End-to-end DP-SGD LoRA training example for DPO (Direct Preference Optimization).

This is the DPO sibling of ``examples/train_dpsgd.py`` (the comprehensive
DP-SGD baseline) and ``examples/train_sft.py``. It ports the full production-style
DP-SGD scaffolding from ``train_dpsgd.py`` — clipping + noise + accounting +
calibration + auditing + LoRA + LR schedules + distributed/Poisson sampling +
W&B — and swaps in the DPO-specific loss, data, and reference machinery from
``opaque-alignment``:

  * ``preference_collator(pad_token_id, max_length)`` builds the batch with
    *six* mandatory tensors — chosen/rejected ``input_ids``, ``attention_mask``,
    ``completion_mask`` (each ``(B, L)``) — plus the two precomputed reference
    logp columns ``ref_chosen_logps`` / ``ref_rejected_logps`` (each ``(B,)``).
  * ``compute_ref_logprobs_for_dataset(...)`` precomputes the frozen reference
    model's per-example logps ONCE, outside the vmap, and caches them to a
    content-addressed ``.safetensors`` cache (so the expensive ref forward runs
    at most once).
    The reference is the LoRA base model: ``null_ref_context(model)`` disables
    the adapter during the precompute so the un-adapted base weights serve as
    the reference.
  * The per-example loss runs TWO forwards (chosen + rejected), turns each into a
    completion logp via ``sequence_logp``, subtracts the precomputed ref logps to
    form per-example log-ratios, and dispatches through ``_DPO_LOSSES[loss_type]``.
    Each loss output for example *i* depends only on example *i*'s data, so
    per-example sensitivity stays ``O(C)`` after clipping.

The reference-free methods (``simpo``/``cpo``/``orpo``) score the policy
log-prob directly, so they skip the reference precompute and the two ref-logp
tensors entirely: their per-example loss takes only the six preference tensors
(chosen/rejected ids, attention masks, completion masks) and runs the same
per-example vmap DP-SGD path with ``_BATCH_ARGNUMS_REF_FREE``.

Eval reports *reward metrics* on held-out preference pairs (chosen/rejected
reward means, accuracy, margin via ``reward_metrics``), NOT perplexity — DPO has
no token-level CE eval objective.

The mechanism is the caller's choice: swap the ``opaque.dpsgd``
imports below for ``opaque.dpftrl`` to run DP-FTRL instead. The loss closure
does not change.

----------------------------------------------------------------------------
SMOKE MODE (``--smoke``)
----------------------------------------------------------------------------
``--smoke`` runs the **full per-example vmap DP-SGD path** on a tiny,
randomly-initialized LlamaForCausalLM (no network, no HF download) over a small
synthetic preference dataset (~8 examples). It precomputes reference logps
(using the model itself as the reference for the smoke), then executes 2 real
DP-SGD steps and prints the DPO loss each step. The ref-logp cache is written to
a per-run temporary directory (no network, no shared state).

``_run_smoke`` has a fallback for environments where ``vmap(grad(...))`` over
the patched model fails on CPU: a single non-vmap chosen+rejected forward +
``_DPO_LOSSES["sigmoid"]`` to validate the loss wiring. The script never exits
non-zero in smoke mode.

USAGE:

  # Smoke test (CPU, ~seconds, no network)
  python examples/train_dpo.py --smoke

  # Quick test preset (SmolLM2-135M-Instruct + code-security DPO)
  python examples/train_dpo.py --preset smoke

  # Full production training on Qwen2.5-Coder-7B + code-security DPO at ε=8
  python examples/train_dpo.py --preset qwen-7b-codesec

  # 4-GPU distributed run with torchrun
  torchrun --nproc_per_node=4 examples/train_dpo.py --preset qwen-7b-codesec

  # Or customize individual parameters:
  python examples/train_dpo.py \\
    --model-name "HuggingFaceTB/SmolLM2-135M-Instruct" \\
    --dataset "CyberNative/Code_Vulnerability_Security_DPO" \\
    --loss-type sigmoid --beta 0.1 \\
    --num-train-samples 5000 --num-eval-samples 500 \\
    --num-epochs 1 --batch-size 16 --eval-steps 50 \\
    --target-epsilon 8.0 --learning-rate 5e-5 \\
    --lora-r 16 --lora-alpha 32 --max-length 1024 \\
    --lora-modules q_proj k_proj v_proj o_proj \\
    --audit --audit-canaries 500 --no-wandb
"""

from __future__ import annotations

# E402: ``apply_runtime_patches()`` must run before transformers/opaque submodules
# are imported, so the remaining imports intentionally follow that call.
# ruff: noqa: E402

import argparse
import contextlib
import hashlib
import importlib.util
import os
import sys
import tempfile
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

# DPO-specific machinery from opaque-alignment.
from opaque.alignment.dpo.collator import preference_collator
from opaque.alignment.dpo.data import extract_prompt
from opaque.alignment.dpo.loss import (
    apo_down_loss,
    apo_zero_loss,
    bco_loss,
    chosen_nll_loss,
    discopop_loss,
    exo_loss,
    hinge_loss,
    ipo_loss,
    mpo_combine,
    nca_loss,
    odds_ratio_loss,
    robust_loss,
    sequence_logp,
    sigmoid_loss,
    simpo_loss,
    sppo_loss,
)
from opaque.alignment.dpo.metric import reward_metrics
from opaque.alignment.metric import entropy_from_logits, mean_token_accuracy
from opaque.alignment.dpo.reference import (
    compute_ref_logprobs_for_dataset,
    null_ref_context,
)

# DP-FTRL mechanism swap: replace the two ``opaque.dpsgd`` noise/sampling imports
# above with their DP-FTRL counterparts (e.g. ``opaque.dpftrl.noise.band_mf_noise``)
# and feed the same ``ClippedPytree`` from ``clipped_grad`` below. The loss is
# mechanism-agnostic.

# Maps the CLI ``--loss-type`` string to a loss function. Keys are opaque's
# ``loss_type`` names (matching ``trl._dpo_trainer._DPO_HEADS``) so a string
# copies cleanly into the class-based ``DPOConfig``. ``chosen_nll`` is the
# chosen-completion NLL regulariser (TRL calls it ``sft``); ``sigmoid_norm``
# shares ``sigmoid``'s loss fn with normalization applied to the log-ratio.
_DPO_LOSSES = {
    "sigmoid": sigmoid_loss,
    "sigmoid_norm": sigmoid_loss,
    "hinge": hinge_loss,
    "robust": robust_loss,
    "ipo": ipo_loss,
    "discopop": discopop_loss,
    "chosen_nll": chosen_nll_loss,
    "apo_zero": apo_zero_loss,
    "apo_down": apo_down_loss,
    "exo_pair": exo_loss,
    "nca_pair": nca_loss,
    "bco_pair": bco_loss,
    "sppo_hard": sppo_loss,
    # ``cpo`` / ``orpo`` are composites special-cased in
    # ``_make_reference_free_loss``; the ``None`` entries keep them selectable
    # ``--loss-type`` values without a direct loss fn.
    "simpo": simpo_loss,
    "cpo": None,
    "orpo": None,
}

# Reference-free methods score the policy log-prob directly: no frozen reference,
# no ref-logp tensors. Dispatched through ``_make_reference_free_loss`` (not
# ``_DPO_LOSSES[...]``).
_REFERENCE_FREE = {"simpo", "cpo", "orpo"}

# vmap batch axis over the 8 per-example loss args (after params at argnums=0):
# chosen/rejected ids, attention masks, completion masks, plus the two ref logps.
_BATCH_ARGNUMS = (1, 2, 3, 4, 5, 6, 7, 8)

# Reference-free methods take only the six preference tensors (no ref logps),
# so the vmap batch axis spans indices 1..6.
_BATCH_ARGNUMS_REF_FREE = (1, 2, 3, 4, 5, 6)


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

    if dtype_name not in {"float16", "bfloat16"}:
        return "partial", f"dtype={dtype_name} (fused CE requires fp16/bf16)"

    return "enabled", "explicit kernel patches applied"


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
    """Stream only required rows, then materialize to in-memory static Dataset.

    Unlike the causal-LM variant, this does not validate a single text field —
    preference datasets carry ``chosen`` / ``rejected`` columns whose schema is
    checked downstream by ``_tokenize_preference_example``.
    """
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


def _normalize_preference_row(row):
    """Map dataset-specific schemas onto ``{prompt, chosen, rejected}``.

    CyberNative/Code_Vulnerability_Security_DPO carries ``(system, question,
    chosen, rejected)``; collapse ``(system, question)`` into a single string
    prompt (a base model has no chat template). Rows already carrying a
    ``prompt`` (or only ``chosen``/``rejected``) pass through unchanged.
    """
    if "prompt" in row or "question" not in row:
        return row
    system = (row.get("system") or "").strip()
    question = row["question"]
    prompt = f"{system}\n\n{question}" if system else question
    return {**row, "prompt": prompt}


def _tokenize_preference_example(example, tokenizer, max_length):
    """Tokenize a single DPO preference example into model-ready token ids.

    Expects ``example`` to have ``"prompt"``, ``"chosen"``, and ``"rejected"``
    keys (run ``extract_prompt`` first if the prompt is implicit).  Both
    ``chosen`` and ``rejected`` may be:

    - A ``list`` of chat messages (``{"role": ..., "content": ...}`` dicts),
      in which case the tokenizer's chat template is applied.
    - A plain string, tokenized directly.

    The ``chosen_completion_mask`` / ``rejected_completion_mask`` tensors are
    ``0`` over prompt tokens and ``1`` over response (completion) tokens.
    Sequences are truncated to ``max_length`` from the right.

    Returns a dict with keys:
    - ``chosen_input_ids``: ``list[int]``
    - ``rejected_input_ids``: ``list[int]``
    - ``chosen_completion_mask``: ``list[int]``  (0=prompt, 1=completion)
    - ``rejected_completion_mask``: ``list[int]``
    """
    prompt = example.get("prompt", [])
    chosen = example["chosen"]
    rejected = example["rejected"]

    def _apply_template(messages):
        """Apply chat template if messages is a list, otherwise tokenize string."""
        if isinstance(messages, list):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )
        # Plain string: tokenize directly (no special tokens added here).
        return tokenizer.encode(messages, add_special_tokens=False)

    # Encode the prompt alone to find the prompt boundary.
    if isinstance(prompt, list) and prompt:
        prompt_ids = tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,  # opens the assistant turn
        )
    elif isinstance(prompt, str) and prompt:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    else:
        prompt_ids = []

    # Encode full chosen / rejected sequences (prompt + completion).
    if isinstance(chosen, list):
        # chosen/rejected are message lists (completions only, prompt is separate)
        full_chosen = prompt + (chosen if isinstance(chosen, list) else [])
        full_rejected = prompt + (rejected if isinstance(rejected, list) else [])
        chosen_ids = _apply_template(full_chosen)
        rejected_ids = _apply_template(full_rejected)
    else:
        # chosen/rejected are plain strings
        chosen_ids = (
            prompt_ids + tokenizer.encode(chosen, add_special_tokens=False)
            if prompt_ids
            else tokenizer.encode(chosen, add_special_tokens=True)
        )
        rejected_ids = (
            prompt_ids + tokenizer.encode(rejected, add_special_tokens=False)
            if prompt_ids
            else tokenizer.encode(rejected, add_special_tokens=True)
        )

    prompt_len = len(prompt_ids)

    # Build completion masks: 0 over prompt tokens, 1 over completion tokens.
    chosen_cmask = [0] * min(prompt_len, len(chosen_ids)) + [1] * max(
        0, len(chosen_ids) - prompt_len
    )
    rejected_cmask = [0] * min(prompt_len, len(rejected_ids)) + [1] * max(
        0, len(rejected_ids) - prompt_len
    )

    # Truncate to max_length (keep-start).
    chosen_ids = chosen_ids[:max_length]
    rejected_ids = rejected_ids[:max_length]
    chosen_cmask = chosen_cmask[:max_length]
    rejected_cmask = rejected_cmask[:max_length]

    return {
        "chosen_input_ids": chosen_ids,
        "rejected_input_ids": rejected_ids,
        "chosen_completion_mask": chosen_cmask,
        "rejected_completion_mask": rejected_cmask,
    }


def _make_per_example_loss(fmodel, frozen, *, loss_type, beta):
    """Build the DPO per-example loss closure (TWO forwards: chosen + rejected).

    The returned callable has signature::

        per_example_loss(
            trainable_params,
            chosen_ids, chosen_mask, chosen_cmask,
            rejected_ids, rejected_mask, rejected_cmask,
            ref_chosen_logps, ref_rejected_logps,
        ) -> per-example scalar loss

    which is exactly ``argnums=0`` (trainable params) + the 8 per-example args
    in ``_BATCH_ARGNUMS``. ``frozen`` first in the merge so trainable params win
    on key collision. Each output depends only on this example's data.
    """

    def per_example_loss(
        trainable_params,
        chosen_ids,
        chosen_mask,
        chosen_cmask,
        rejected_ids,
        rejected_mask,
        rejected_cmask,
        ref_chosen_logps,
        ref_rejected_logps,
    ):
        merged = {**frozen, **trainable_params}
        chosen_out = fmodel(merged, input_ids=chosen_ids, attention_mask=chosen_mask)
        rejected_out = fmodel(
            merged, input_ids=rejected_ids, attention_mask=rejected_mask
        )
        chosen_logp = sequence_logp(chosen_out.logits, chosen_ids, chosen_cmask)
        rejected_logp = sequence_logp(rejected_out.logits, rejected_ids, rejected_cmask)
        # Log-ratios = policy logp - precomputed reference logp (per example).
        return _DPO_LOSSES[loss_type](
            chosen_logp - ref_chosen_logps,
            rejected_logp - ref_rejected_logps,
            beta=beta,
        )

    return per_example_loss


def _make_reference_free_loss(
    fmodel,
    frozen,
    *,
    loss_type,
    beta,
    simpo_gamma=1.0,
    cpo_alpha=1.0,
    orpo_lambda=1.0,
):
    """Build a reference-free per-example loss closure (TWO forwards, no ref).

    The returned callable has signature::

        per_example_loss(
            trainable_params,
            chosen_ids, chosen_mask, chosen_cmask,
            rejected_ids, rejected_mask, rejected_cmask,
        ) -> per-example scalar loss

    which is ``argnums=0`` (trainable params) + the 6 per-example preference
    args in ``_BATCH_ARGNUMS_REF_FREE``. No reference logps appear because SimPO,
    CPO, and ORPO score the policy log-prob directly. ``frozen`` first in the
    merge so trainable params win on key collision. Each output depends only on
    this example's data, so per-example sensitivity stays ``O(C)`` after
    clipping.

    SimPO and ORPO use length-normalized completion log-probs; CPO uses the raw
    (un-normalized) chosen/rejected log-probs as the preference signal.
    """

    def per_example_loss(
        trainable_params,
        chosen_ids,
        chosen_mask,
        chosen_cmask,
        rejected_ids,
        rejected_mask,
        rejected_cmask,
    ):
        merged = {**frozen, **trainable_params}
        chosen_out = fmodel(merged, input_ids=chosen_ids, attention_mask=chosen_mask)
        rejected_out = fmodel(
            merged, input_ids=rejected_ids, attention_mask=rejected_mask
        )
        if loss_type == "cpo":
            # CPO scores the raw (NOT length-normalized) completion log-probs:
            # a sigmoid preference term blended with a chosen-NLL regulariser.
            chosen_logp = sequence_logp(chosen_out.logits, chosen_ids, chosen_cmask)
            rejected_logp = sequence_logp(
                rejected_out.logits, rejected_ids, rejected_cmask
            )
            return mpo_combine(
                {
                    "pref": sigmoid_loss(chosen_logp, rejected_logp, beta=beta),
                    "nll": chosen_nll_loss(chosen_logp),
                },
                {"pref": 1.0, "nll": cpo_alpha},
            )

        # SimPO and ORPO both score length-normalized completion log-probs.
        chosen_logp = sequence_logp(
            chosen_out.logits, chosen_ids, chosen_cmask, length_normalized=True
        )
        rejected_logp = sequence_logp(
            rejected_out.logits, rejected_ids, rejected_cmask, length_normalized=True
        )
        if loss_type == "simpo":
            return simpo_loss(chosen_logp, rejected_logp, beta=beta, gamma=simpo_gamma)
        # ORPO: odds-ratio preference term blended with a chosen-NLL regulariser.
        return mpo_combine(
            {
                "or": odds_ratio_loss(chosen_logp, rejected_logp),
                "nll": chosen_nll_loss(chosen_logp),
            },
            {"or": 1.0, "nll": orpo_lambda},
        )

    return per_example_loss


def _make_ref_callable(model, device=None):
    """Wrap a model into a ``ref`` callable for compute_ref_logprobs_for_dataset.

    Returns ``ref(batch) -> {"ref_chosen_logps": (B,), "ref_rejected_logps": (B,)}``
    computed via ``sequence_logp`` under ``torch.no_grad()`` (contract:
    ``ref`` is a plain ``dict[str, Tensor] -> dict[str, Tensor]`` callable, which
    keeps the precompute helper mechanism- and model-agnostic).

    The precompute helper collates on CPU; this callable moves each input to the
    model's ``device`` before the forward and returns the logps on CPU so they
    serialize back into the dataset cleanly.
    """
    dev = device if device is not None else next(model.parameters()).device

    def ref(batch):
        with torch.no_grad():
            chosen_ids = batch["chosen_input_ids"].to(dev)
            rejected_ids = batch["rejected_input_ids"].to(dev)
            chosen_out = model(
                input_ids=chosen_ids,
                attention_mask=batch["chosen_attention_mask"].to(dev),
            )
            rejected_out = model(
                input_ids=rejected_ids,
                attention_mask=batch["rejected_attention_mask"].to(dev),
            )
            chosen_logp = sequence_logp(
                chosen_out.logits,
                chosen_ids,
                batch["chosen_completion_mask"].to(dev),
            )
            rejected_logp = sequence_logp(
                rejected_out.logits,
                rejected_ids,
                batch["rejected_completion_mask"].to(dev),
            )
        return {
            "ref_chosen_logps": chosen_logp.cpu(),
            "ref_rejected_logps": rejected_logp.cpu(),
        }

    return ref


def _base_model_state_digest(model) -> str:
    """Hash the effective adapter-disabled reference weights."""
    hasher = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if ".lora_" in name or ".modules_to_save." in name:
            continue
        value = tensor.detach().cpu().contiguous()
        hasher.update(name.encode("utf-8"))
        hasher.update(str(value.dtype).encode("ascii"))
        hasher.update(str(tuple(value.shape)).encode("ascii"))
        start = value.storage_offset() * value.element_size()
        end = start + value.numel() * value.element_size()
        hasher.update(bytes(value.untyped_storage()[start:end]))
    return hasher.hexdigest()


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
        description="End-to-end DP-SGD LoRA Direct Preference Optimization (DPO)"
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny CPU smoke test (random model, synthetic prefs, 2 steps). "
        "Bypasses all other configuration.",
    )

    # Preset configurations
    parser.add_argument(
        "--preset",
        type=str,
        choices=[
            "smoke",
            "qwen-7b-codesec",
            "mellum-codesec",
            "mellum2-codesec",
        ],
        default=None,
        help="Optional preset that fills in any unset arguments. Omit it to "
        "configure the run directly (at least --model-name and --dataset), or "
        "use --smoke for the hermetic no-network test. "
        "smoke=quick test SmolLM2-135M-Instruct + code-security DPO at ε=8, "
        "qwen-7b-codesec=Qwen2.5-Coder-7B + code-security DPO at ε=8 with adafactor @ 5e-5, "
        "mellum-codesec=Mellum-4b dense + code-security DPO at ε=8, "
        "mellum2-codesec=Mellum2-12B-A2.5B MoE + code-security DPO at ε=8.",
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
        help="HuggingFace preference dataset name (must have chosen/rejected "
        "columns; required unless a --preset sets it)",
    )
    data_group.add_argument(
        "--dataset-subset",
        "--dataset-name",
        dest="dataset_subset",
        type=str,
        default=None,
        help="Optional dataset subset (HF load_dataset 'name' argument).",
    )
    data_group.add_argument(
        "--dataset-split", type=str, default="train", help="Dataset split for training"
    )
    data_group.add_argument(
        "--num-train-samples",
        type=int,
        default=5000,
        help="Number of training preference pairs (default: 5000)",
    )
    data_group.add_argument(
        "--num-eval-samples",
        type=int,
        default=2000,
        help="Held-out preference-pair count for periodic reward-metric eval. "
        "rewards/accuracies is a binary signal so wants a larger eval set than "
        "scalar loss; 2000 pairs ≈ 0.7%% std-err on the accuracy estimate.",
    )
    data_group.add_argument(
        "--max-length",
        "--max-seq-len",
        dest="max_length",
        type=int,
        default=1024,
        help="Maximum sequence length for the preference collator",
    )

    dpo_group = parser.add_argument_group("dpo", "DPO loss and reference settings")
    dpo_group.add_argument(
        "--loss-type",
        type=str,
        choices=sorted(set(_DPO_LOSSES) | _REFERENCE_FREE),
        default="sigmoid",
        help="DPO loss variant (direct functions from opaque.alignment.dpo). "
        "Reference-based heads score policy log-ratios against a frozen "
        "reference; the reference-free methods simpo/cpo/orpo score the "
        "policy log-prob directly and skip the reference precompute.",
    )
    dpo_group.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO temperature beta (reference-deviation strength).",
    )
    dpo_group.add_argument(
        "--simpo-gamma",
        type=float,
        default=1.0,
        help="SimPO target reward margin γ (reference-free --loss-type simpo).",
    )
    dpo_group.add_argument(
        "--cpo-alpha",
        type=float,
        default=1.0,
        help="CPO chosen-NLL regulariser weight α (reference-free --loss-type cpo).",
    )
    dpo_group.add_argument(
        "--orpo-lambda",
        type=float,
        default=1.0,
        help="ORPO chosen-NLL regulariser weight λ (reference-free --loss-type orpo).",
    )
    dpo_group.add_argument(
        "--ref-cache-dir",
        type=str,
        default=None,
        help=(
            "Directory for the reference-logp cache (the content-addressed "
            "safetensors archive written by compute_ref_logprobs_for_dataset). "
            "Default: ``~/.cache/opaque/ref_logps`` — a persistent path so "
            "re-runs against the same (model, dataset, sample count) hit the "
            "cache. Pass an absolute path to override, or pass an explicit "
            "tempdir-style path for ephemeral caching. Ignored for "
            "reference-free --loss-type (simpo/cpo/orpo)."
        ),
    )
    dpo_group.add_argument(
        "--log-completion-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Log the logits-consuming completion telemetry (entropy, "
            "mean_token_accuracy, logits/*) alongside rewards/* and logps/*. "
            "Mirrors DPOConfig.log_completion_metrics; --no-log-completion-metrics "
            "skips those metrics so the eval path stays logits-light."
        ),
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
        "--num-epochs", type=int, default=1, help="Number of epochs"
    )
    train_group.add_argument(
        "--learning-rate", type=float, default=5.0e-5, help="Learning rate"
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
        default="adamw",
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
        help="Log eval reward metrics and privacy every N steps",
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
        help="Recompute activations in backward instead of storing them "
        "(trades compute for memory). Off by default; enable only when a config "
        "would otherwise run out of memory.",
    )
    train_group.add_argument(
        "--activation-offloading",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Offload saved tensors to CPU via save_on_cpu (works with or without checkpointing)",
    )

    lora_group = parser.add_argument_group("lora", "LoRA adapter settings")
    lora_group.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    lora_group.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
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
    tracking_group.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=None,
        help="W&B run tags (space-separated list); forwarded to wandb.init(tags=...)",
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

    # Code models train on code-security preference pairs, not general chat.
    # The loader collapses (system, question) -> prompt.
    _CODESEC = "CyberNative/Code_Vulnerability_Security_DPO"
    if args.preset == "smoke":
        _set("model_name", "HuggingFaceTB/SmolLM2-135M-Instruct")
        _set("dataset", _CODESEC)
        _set("num_train_samples", 256)
        _set("num_eval_samples", 64)
        _set("num_epochs", 1)
        _set("batch_size", 16)
        _set("microbatch_size", 4)
        _set("log_steps", 5)
        _set("eval_steps", 5)
        _set("target_epsilon", 8.0)
        _set("learning_rate", 5e-5)
        _set("loss_type", "sigmoid")
        _set("beta", 0.1)
        _set("lora_r", 16)
        _set("lora_alpha", 32)
        _set("max_length", 512)
        _set("lora_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
        _set("dtype", "bfloat16")
        _set("audit", False)
    elif args.preset == "qwen-7b-codesec":
        # Qwen2.5-Coder-7B + code-security DPO at ε=8.
        _set("model_name", "Qwen/Qwen2.5-Coder-7B-Instruct")
        _set("dataset", _CODESEC)
        _set("num_train_samples", 4000)
        _set("num_eval_samples", 500)
        _set("num_epochs", 2)
        _set("batch_size", 128)
        _set("microbatch_size", 8)
        _set("log_steps", 2)
        _set("eval_steps", 25)
        _set("target_epsilon", 8.0)
        _set("learning_rate", 5e-5)
        _set("optimizer", "adafactor")
        _set("loss_type", "sigmoid")
        _set("beta", 0.1)
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
    elif args.preset == "mellum2-codesec":
        # Mellum2-12B-A2.5B (MoE) DP-DPO at ε=8. LoRA targets the attention
        # projections only; the routed experts are frozen stacked nn.Parameter
        # weights, so the fused-MoE backward skips their per-sample gradients and
        # the policy fits microbatch=16 without gradient checkpointing.
        _set("model_name", "JetBrains/Mellum2-12B-A2.5B-Base")
        _set("dataset", _CODESEC)
        _set("num_train_samples", 4000)
        _set("num_eval_samples", 500)
        _set("num_epochs", 2)
        _set("batch_size", 128)
        _set("microbatch_size", 16)
        _set("log_steps", 2)
        _set("eval_steps", 25)
        _set("target_epsilon", 8.0)
        _set("learning_rate", 1e-4)
        _set("optimizer", "adafactor")
        _set("loss_type", "sigmoid")
        _set("beta", 0.3)
        _set("clipping_norm", 2.0)
        _set("lora_r", 16)
        _set("lora_alpha", 32)
        _set("max_length", 1024)
        _set("lora_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
        _set("dtype", "bfloat16")
    elif args.preset == "mellum-codesec":
        # Mellum-4b (dense Llama) DP-DPO at ε=8. Dense MLP, so LoRA targets the
        # gate/up/down projections alongside the attention projections.
        _set("model_name", "JetBrains/Mellum-4b-base")
        _set("dataset", _CODESEC)
        _set("num_train_samples", 4000)
        _set("num_eval_samples", 500)
        _set("num_epochs", 2)
        _set("batch_size", 128)
        _set("microbatch_size", 16)
        _set("log_steps", 2)
        _set("eval_steps", 25)
        _set("target_epsilon", 8.0)
        _set("learning_rate", 1e-4)
        _set("optimizer", "adafactor")
        _set("loss_type", "sigmoid")
        _set("beta", 0.3)
        _set("clipping_norm", 2.0)
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

    if not args.smoke:
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
    """Tiny CPU smoke test: random Llama, synthetic prefs, 2 real DP-SGD steps.

    Builds a tiny randomly-initialized LlamaForCausalLM (no network) and a small
    synthetic preference dataset, then runs the full per-example vmap DP-SGD path
    for 2 steps, printing the loss each step. It respects ``--loss-type``: the
    reference-based path precomputes reference logps (using the model itself as
    the reference) and dispatches an 8-tuple batch, while the reference-free path
    (simpo/cpo/orpo) skips the precompute and dispatches the six preference
    tensors. On a genuine vmap failure it falls back to a single non-vmap
    chosen+rejected forward + loss so the smoke still exits 0 (documented in the
    module header).
    """
    from transformers import LlamaConfig

    device = torch.device("cpu")
    torch.manual_seed(args.seed)

    print("=" * 72)
    print("train_dpo.py --smoke  (tiny random Llama, synthetic prefs, CPU)")
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
    apply_model_patches(model)  # vmap-safety wrappers (eager attn / batchify)
    model.eval()
    model.to(device)

    pad_token_id = 0
    max_length = 16
    batch_size = 4
    beta = args.beta
    loss_type = args.loss_type

    # --- Tiny synthetic preference dataset: 8 examples (no network) ---
    # Each example shares a prompt prefix between chosen/rejected; the completion
    # mask marks the differing response span, mirroring real DPO preprocessing.
    rng = torch.Generator().manual_seed(args.seed)

    def _make_example():
        prompt_len = int(torch.randint(2, 5, (1,), generator=rng))
        prompt = torch.randint(
            1, config.vocab_size, (prompt_len,), generator=rng
        ).tolist()
        chosen_resp_len = int(
            torch.randint(3, max_length - prompt_len, (1,), generator=rng)
        )
        rejected_resp_len = int(
            torch.randint(3, max_length - prompt_len, (1,), generator=rng)
        )
        chosen_resp = torch.randint(
            1, config.vocab_size, (chosen_resp_len,), generator=rng
        ).tolist()
        rejected_resp = torch.randint(
            1, config.vocab_size, (rejected_resp_len,), generator=rng
        ).tolist()
        chosen_ids = prompt + chosen_resp
        rejected_ids = prompt + rejected_resp
        # Completion mask: 0 over the prompt, 1 over the response.
        chosen_cmask = [0] * prompt_len + [1] * chosen_resp_len
        rejected_cmask = [0] * prompt_len + [1] * rejected_resp_len
        return {
            "chosen_input_ids": chosen_ids,
            "rejected_input_ids": rejected_ids,
            "chosen_completion_mask": chosen_cmask,
            "rejected_completion_mask": rejected_cmask,
        }

    rows = [_make_example() for _ in range(8)]
    dataset = Dataset.from_list(rows)

    collate = preference_collator(pad_token_id, max_length)

    reference_free = loss_type in _REFERENCE_FREE

    # --- Precompute reference logps ONCE, outside vmap, to a tmp cache dir ---
    # Reference-based heads need frozen reference logps (here the model itself is
    # the reference); reference-free methods skip the precompute. The per-run temp
    # cache keeps the smoke hermetic (no network, no shared state).
    if reference_free:
        print("\nReference-free loss (no reference precompute).")
        rows = list(dataset)
    else:
        print("\nPrecomputing reference logps (model-as-ref, tmp cache)...")
        with tempfile.TemporaryDirectory(prefix="opaque_dpo_smoke_") as cache_dir:
            dataset = compute_ref_logprobs_for_dataset(
                dataset,
                _make_ref_callable(model),
                collator=collate,
                output_columns=("ref_chosen_logps", "ref_rejected_logps"),
                batch_size=batch_size,
                cache_identity={
                    "kind": "dpo-smoke",
                    "reference": {
                        "state_sha256": _base_model_state_digest(model),
                    },
                },
                cache_dir=cache_dir,
            )
        rows = list(dataset)  # each row now carries the ref_*_logps columns
        print(
            f"  ref columns added: ref_chosen_logps[0]={rows[0]['ref_chosen_logps']:.4f}, "
            f"ref_rejected_logps[0]={rows[0]['ref_rejected_logps']:.4f}"
        )

    # --- Functional conversion (everything trainable on this tiny model) ---
    fmodel, trainable, frozen = make_functional(
        model, disable_autograd_tracking=True, partition_trainable=True
    )
    print(f"Trainable param tensors: {len(trainable)} | frozen: {len(frozen)}")

    if reference_free:
        per_example_loss = _make_reference_free_loss(
            fmodel,
            frozen,
            loss_type=loss_type,
            beta=beta,
            simpo_gamma=args.simpo_gamma,
            cpo_alpha=args.cpo_alpha,
            orpo_lambda=args.orpo_lambda,
        )
        batch_argnums = _BATCH_ARGNUMS_REF_FREE
    else:
        per_example_loss = _make_per_example_loss(
            fmodel, frozen, loss_type=loss_type, beta=beta
        )
        batch_argnums = _BATCH_ARGNUMS

    # --- Try the full per-example vmap DP-SGD path; fall back if it breaks ---
    try:
        from opaque.optimizers import adamw

        grad_fn, clip_state = clipped_grad(
            per_example_loss,
            argnums=0,
            batch_argnums=batch_argnums,
            clipping_norm=args.clipping_norm,
            normalize_by=batch_size,
            return_aux=True,
        )
        noise_fn, noise_state = gaussian_noise(
            noise_multiplier=args.noise_multiplier or 0.8, key=key(args.seed)
        )
        base_opt = adamw(lr=args.learning_rate)
        opt_state = base_opt.init(trainable)

        # Two DP-SGD steps over Poisson-sampled batches.
        sampler = PoissonSampler(
            rows,
            sample_rate=batch_size / len(rows),
            n_steps=2,
            key=fold_in(key(args.seed), 0, 0),
        )
        print(
            f"\nRunning 2 DP-SGD steps (full per-example vmap path, loss={loss_type})..."
        )
        step = 0
        for indices in sampler:
            batch = _collate_to_device(
                collate,
                [rows[i] for i in indices],
                device,
                reference_free=reference_free,
            )
            (grads, aux), clip_state = grad_fn(trainable, *batch, state=clip_state)
            noisy_grads, noise_state = noise_fn(grads, noise_state)
            updates, opt_state = base_opt.update(
                noisy_grads, opt_state, params=trainable
            )
            trainable = torchopt.apply_updates(trainable, updates)
            step += 1
            print(
                f"  step {step}/2 | bs={batch[0].shape[0]} | "
                f"dpo_loss={aux.loss_values.mean().item():.4f}"
            )

        print("\nSmoke OK: full per-example DP-SGD vmap path completed 2 steps.")
        return 0

    except Exception as exc:  # pragma: no cover - defensive fallback path
        # Fallback when vmap(grad(...)) over the patched model fails on this host:
        # validate loss wiring with one non-vmap forward + DPO loss, still exit 0.
        print(f"\nNote: full vmap DP-SGD path raised: {type(exc).__name__}: {exc}")
        print(
            "Falling back to a single non-vmap chosen+rejected forward + DPO "
            "loss to validate the loss wiring."
        )
        batch = _collate_to_device(
            collate, rows[:batch_size], device, reference_free=reference_free
        )
        with torch.no_grad():
            loss = per_example_loss(trainable, *batch)
        print(f"  non-vmap batch dpo_loss (per-example mean): {loss.mean().item():.4f}")
        print("\nSmoke OK (fallback path): DPO loss wiring validated.")
        return 0


def _collate_to_device(collate, examples, device, *, reference_free=False):
    """Collate raw preference rows into the per-example batch tuple on ``device``.

    Reference-based mode returns the 8-tuple in ``_BATCH_ARGNUMS`` order: chosen
    (ids, mask, cmask), rejected (ids, mask, cmask), ref (chosen, rejected).
    Reference-free mode (SimPO/CPO/ORPO) returns only the six preference tensors
    in ``_BATCH_ARGNUMS_REF_FREE`` order and does not require the ref columns —
    the precompute is skipped for those methods.
    """
    b = collate(examples)
    six = (
        b["chosen_input_ids"].to(device),
        b["chosen_attention_mask"].to(device),
        b["chosen_completion_mask"].to(device),
        b["rejected_input_ids"].to(device),
        b["rejected_attention_mask"].to(device),
        b["rejected_completion_mask"].to(device),
    )
    if reference_free:
        return six
    return six + (
        b["ref_chosen_logps"].to(device),
        b["ref_rejected_logps"].to(device),
    )


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
        print("DP-SGD LoRA Direct Preference Optimization (DPO)")
        print("=" * 80)

    # Initialize wandb (enabled by default, offline if no credentials)
    use_wandb = (not args.no_wandb) and is_main_process
    if use_wandb:
        # Generate default run name from key parameters if not specified
        if args.wandb_run_name is None:
            model_short = args.model_name.split("/")[-1]
            run_name = f"dpo_{model_short}_n{args.num_train_samples}_e{args.num_epochs}_b{args.batch_size}_eps{args.target_epsilon}_lr{args.learning_rate}"
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
            tags=args.tags,
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
    if tokenizer.chat_template is None:
        # Base models (e.g. JetBrains/Mellum-4b-base) ship no chat template;
        # _tokenize_preference_example calls apply_chat_template when a row's
        # chosen/rejected is a list-of-message dicts. Install a minimal ChatML
        # so those datasets still work without an instruct variant.
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        )

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
    # DPO consumes ``out.logits`` (not the model's ``.loss``), so we do NOT
    # opt into the fused linear+CE kernel here — the per-example loss needs
    # the full logits tensor to compute ``sequence_logp`` over completions.
    apply_model_patches(model)
    model.print_trainable_parameters()
    print_memory(device, "After LoRA")

    # Load and prepare dataset
    print(f"\nLoading dataset: {args.dataset}...")
    if args.dataset_subset:
        print(f"  Subset: {args.dataset_subset}")
    print(f"  Split: {args.dataset_split}")

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
        f"\nPreparing {args.num_eval_samples} eval + {args.num_train_samples} train pairs..."
    )
    eval_raw = dataset.take(args.num_eval_samples)
    train_raw = dataset.skip(args.num_eval_samples).take(args.num_train_samples)

    # Tokenize each split into preference token ids + completion masks.
    print(f"\nTokenizing preference pairs (max_length={args.max_length})...")

    def _tokenize_split(rows_iter, desc):
        out = []
        for raw_row in rows_iter:
            row = extract_prompt(_normalize_preference_row(raw_row))
            try:
                tok = _tokenize_preference_example(row, tokenizer, args.max_length)
            except Exception:
                continue
            # Skip degenerate examples: both sides must have completion tokens.
            if sum(tok["chosen_completion_mask"]) == 0:
                continue
            if sum(tok["rejected_completion_mask"]) == 0:
                continue
            out.append(tok)
        print(f"  {desc}: {len(out)} usable preference pairs")
        return Dataset.from_list(out)

    eval_dataset = _tokenize_split(eval_raw, "Tokenizing eval")
    train_dataset = _tokenize_split(train_raw, "Tokenizing train")

    if len(train_dataset) == 0:
        raise SystemExit(
            f"No usable preference examples found in '{args.dataset}'. "
            "Check that the dataset has 'chosen' and 'rejected' columns."
        )
    print(
        f"Prepared datasets: {len(train_dataset)} train pairs, {len(eval_dataset)} eval pairs"
    )

    # This flag selects the per-example batch shape, loss closure, and
    # batch_argnums everywhere below (reference-free skips the ref-logp tensors).
    reference_free = args.loss_type in _REFERENCE_FREE
    batch_argnums = _BATCH_ARGNUMS_REF_FREE if reference_free else _BATCH_ARGNUMS

    # Preference collator: 6 mandatory tensors, plus the 2 ref-logp columns in
    # the reference-based path once attached.
    collate_raw = preference_collator(tokenizer.pad_token_id, args.max_length)

    def collate(examples):
        return _collate_to_device(
            collate_raw, examples, device, reference_free=reference_free
        )

    # --- Precompute reference logps (LoRA base model as frozen reference) -----
    # ``null_ref_context(model)`` disables the LoRA adapter so the un-adapted base
    # weights serve as the reference policy (the canonical LoRA-DPO reference). The
    # ref forward runs at most once, cached to a content-addressed safetensors file.
    # Reference-free methods skip this entirely.
    if reference_free:
        print("\nReference-free loss selected (skipping reference precompute).")
    else:
        # Default --ref-cache-dir to ~/.cache/opaque/ref_logps so repeat runs over
        # the same (model, dataset, sample count) hit the cache.
        if args.ref_cache_dir is None:
            ref_cache_dir = os.path.expanduser("~/.cache/opaque/ref_logps")
        else:
            ref_cache_dir = os.path.expanduser(args.ref_cache_dir)
        print(
            "\nPrecomputing reference logps (LoRA base as ref, "
            f"cached to {ref_cache_dir})..."
        )
        ref_callable = _make_ref_callable(model, device=device)
        ref_cache_identity = {
            "kind": "dpo-reference-logprobs",
            "reference": {
                "adapter_mode": "disabled",
                "state_sha256": _base_model_state_digest(model),
            },
        }
        with null_ref_context(model):
            train_dataset = compute_ref_logprobs_for_dataset(
                train_dataset,
                ref_callable,
                collator=collate_raw,
                output_columns=("ref_chosen_logps", "ref_rejected_logps"),
                batch_size=args.eval_batch_size,
                cache_identity=ref_cache_identity,
                cache_dir=ref_cache_dir,
            )
            eval_dataset = compute_ref_logprobs_for_dataset(
                eval_dataset,
                ref_callable,
                collator=collate_raw,
                output_columns=("ref_chosen_logps", "ref_rejected_logps"),
                batch_size=args.eval_batch_size,
                cache_identity=ref_cache_identity,
                cache_dir=ref_cache_dir,
            )
        print(
            f"  ref_chosen_logps[0]={train_dataset[0]['ref_chosen_logps']:.4f}, "
            f"ref_rejected_logps[0]={train_dataset[0]['ref_rejected_logps']:.4f}"
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

    # Training uses Poisson sampling: each example independently sampled with
    # probability sample_rate each step. In parallel Poisson mode each rank samples
    # independently from the full dataset, so divide by world_size to keep the
    # global expected batch size = args.batch_size.
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

    # Per-example DPO loss closure (TWO forwards: chosen + rejected). Output for
    # example i depends only on example i's data, so per-example sensitivity is O(C).
    if reference_free:
        per_example_loss_fn = _make_reference_free_loss(
            fmodel,
            frozen_params,
            loss_type=args.loss_type,
            beta=args.beta,
            simpo_gamma=args.simpo_gamma,
            cpo_alpha=args.cpo_alpha,
            orpo_lambda=args.orpo_lambda,
        )
    else:
        per_example_loss_fn = _make_per_example_loss(
            fmodel, frozen_params, loss_type=args.loss_type, beta=args.beta
        )

    # Verified canary scoring: the per-example DPO loss is the membership
    # signal; loss_scores builds the canary loader internally and binds
    # each score to its canary identifier.
    def score_canaries(params, reference_scores=None):
        return auditing.loss_scores(
            per_example_loss_fn,
            params,
            batch_argnums=batch_argnums,
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

    def _masked_mean_logit(logits, completion_mask):
        """Mean logit over (shifted) completion positions — TRL ``logits/*``.

        Mirrors ``DPOTrainer._masked_mean_logit``: a masked weighted mean over
        the shifted completion positions (no boolean indexing). ``logits`` is
        ``(B, T, V)`` and ``completion_mask`` is ``(B, T)``; returns ``(B,)``.
        """
        shifted = logits[..., :-1, :]
        mask = (completion_mask[..., 1:] != 0).to(shifted.dtype)
        pos_mean = shifted.mean(dim=-1)
        return (pos_mean * mask).sum(-1) / mask.sum(-1).clamp(min=1)

    def eval_reward_metrics(trainable):
        """Full DPO telemetry over the held-out preference eval set.

        For DPO, eval = chosen/rejected reward means, accuracy, and margin on
        held-out preference pairs (NOT perplexity). Mirrors the class-based
        ``DPOTrainer._reward_aux`` logged set: ``rewards/*`` and the summed
        policy ``logps/*`` are always reported; the logits-consuming
        diagnostics (``logits/*``, ``entropy``, ``mean_token_accuracy``) are
        gated on ``--log-completion-metrics``. Returns a dict of floats; an
        empty eval set yields ``nan`` rewards.

        Reference-based heads report the policy-vs-reference log-ratios as the
        per-example rewards. Reference-free methods have no reference, so the
        reward is the policy completion log-prob directly (length-normalized for
        simpo/orpo to match their scoring, raw for cpo). The ``logps/*``
        telemetry is always the summed (un-normalized) sequence log-prob, as in
        the trainer.
        """
        length_normalized = args.loss_type in {"simpo", "orpo"}
        log_metrics = args.log_completion_metrics
        with torch.no_grad():
            merged = {**frozen_params, **trainable}
            chosen_lrs = []
            rejected_lrs = []
            chosen_logps = []
            rejected_logps = []
            logits_chosen = []
            logits_rejected = []
            entropies = []
            accuracies = []
            for batch in eval_loader:
                if reference_free:
                    (
                        chosen_ids,
                        chosen_mask,
                        chosen_cmask,
                        rejected_ids,
                        rejected_mask,
                        rejected_cmask,
                    ) = batch
                    ref_chosen_lp = 0.0
                    ref_rejected_lp = 0.0
                else:
                    (
                        chosen_ids,
                        chosen_mask,
                        chosen_cmask,
                        rejected_ids,
                        rejected_mask,
                        rejected_cmask,
                        ref_chosen_lp,
                        ref_rejected_lp,
                    ) = batch
                c_out = fmodel(merged, input_ids=chosen_ids, attention_mask=chosen_mask)
                r_out = fmodel(
                    merged, input_ids=rejected_ids, attention_mask=rejected_mask
                )
                c_logp = sequence_logp(
                    c_out.logits,
                    chosen_ids,
                    chosen_cmask,
                    length_normalized=length_normalized,
                )
                r_logp = sequence_logp(
                    r_out.logits,
                    rejected_ids,
                    rejected_cmask,
                    length_normalized=length_normalized,
                )
                chosen_lrs.append(c_logp - ref_chosen_lp)
                rejected_lrs.append(r_logp - ref_rejected_lp)
                # logps/* is always the summed (un-normalized) sequence logp.
                chosen_logps.append(
                    sequence_logp(c_out.logits, chosen_ids, chosen_cmask)
                )
                rejected_logps.append(
                    sequence_logp(r_out.logits, rejected_ids, rejected_cmask)
                )
                if log_metrics:
                    logits_chosen.append(_masked_mean_logit(c_out.logits, chosen_cmask))
                    logits_rejected.append(
                        _masked_mean_logit(r_out.logits, rejected_cmask)
                    )
                    # entropy_from_logits / mean_token_accuracy shift internally,
                    # so pass FULL-length logits + completion mask.
                    entropies.append(
                        0.5
                        * (
                            entropy_from_logits(c_out.logits, chosen_cmask)
                            + entropy_from_logits(r_out.logits, rejected_cmask)
                        )
                    )
                    accuracies.append(
                        mean_token_accuracy(c_out.logits, chosen_ids, chosen_cmask)
                    )
            if not chosen_lrs:
                return {
                    "rewards/chosen": float("nan"),
                    "rewards/rejected": float("nan"),
                    "rewards/accuracies": float("nan"),
                    "rewards/margins": float("nan"),
                }
            chosen_lr = torch.cat(chosen_lrs)
            rejected_lr = torch.cat(rejected_lrs)
            m = reward_metrics(chosen_lr, rejected_lr, beta=args.beta)
            result = {k: v.item() for k, v in m.items()}
            result["logps/chosen"] = torch.cat(chosen_logps).mean().item()
            result["logps/rejected"] = torch.cat(rejected_logps).mean().item()
            if log_metrics:
                result["logits/chosen"] = torch.cat(logits_chosen).mean().item()
                result["logits/rejected"] = torch.cat(logits_rejected).mean().item()
                # entropy / accuracy are per-batch scalars; mean over batches.
                result["entropy"] = torch.stack(entropies).mean().item()
                result["mean_token_accuracy"] = torch.stack(accuracies).mean().item()
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
    print("\nSetting up DP-SGD DPO training...")
    print(f"  Loss: {args.loss_type} (beta={args.beta})")
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
            batch_argnums=batch_argnums,
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
            batch_argnums=batch_argnums,
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
            batch_argnums=batch_argnums,
            clipping_norm=clip_norm,
            normalize_by=args.batch_size,
            microbatch_size=args.microbatch_size,
            return_aux=True,
        )

    # Calibrate noise multiplier from target privacy budget.
    total_steps = args.num_epochs * expected_steps_per_epoch

    # Compute delta from training set size: δ = 1/n^1.1 (keeps δ below 1/n).
    if args.target_delta is None:
        args.target_delta = 1.0 / (global_train_size**1.1)
    if use_wandb:
        wandb.config.update({"target_delta": args.target_delta}, allow_val_change=True)

    # Noise injection chain: base mechanism → adaclip (optional) → amplification.
    # Bounded Gaussian noise (Chen and Hale, 2024) confines per-coordinate support
    # but accounts as ordinary Gaussian at training scale (ℓ₂-ball clip, not a
    # product of intervals), so accounting uses dpsgd_acc.gaussian() either way.
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

    # Build LR schedule using opaque.scheduling primitives. Shares ``total_steps``
    # with the privacy calibration above so the schedule and accounting agree on
    # run length; ``--max-steps`` only truncates training (the schedule still
    # spans the full planned epoch count).
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
        # Inverse-sqrt timescale defaults to warmup when set, otherwise the full
        # run (a gentle ~1/sqrt(2) decay rather than an aggressive 1/sqrt(t)).
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

    # Setup optimizer. Noise metadata travels with ``NoisedPytree`` updates, so
    # construction needs no precomputed stddev; ``--noise-bias-correction`` only
    # gates whether the optimizer's DP-aware path consumes it (ignored for
    # optimizers without a BC path, e.g. sgd/lion).
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
        # Pass ``bound`` unconditionally: at ``noise_multiplier=0`` the bounded
        # path still clamps the input to the interval, keeping the mechanism
        # consistent with the user's chosen flag.
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

    # Step-0 eval: log baseline reward metrics before any training
    initial_metrics = eval_reward_metrics(trainable_params)
    initial_epsilon = accounting.epsilon_at(args.target_delta)
    initial_noise_std = _noise_stddev(initial_bound, noise_multiplier)
    print(
        f"  → Step 0 eval: acc={initial_metrics['rewards/accuracies']:.3f}, "
        f"margin={initial_metrics['rewards/margins']:.4f}, ε={initial_epsilon:.3f}"
    )
    if use_wandb:
        # Match the schema used at every later eval_steps boundary so W&B sees a
        # single dense family of eval metrics rather than two sparse families.
        wandb.log(
            {
                "eval/chosen_reward": initial_metrics["rewards/chosen"],
                "eval/rejected_reward": initial_metrics["rewards/rejected"],
                "eval/accuracy": initial_metrics["rewards/accuracies"],
                "eval/margin": initial_metrics["rewards/margins"],
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
                eval_m = eval_reward_metrics(trainable_params)
                # Cache PLD before eval so it serves as opaque boundary
                accounting = acc.cached(accounting)
                epsilon = accounting.epsilon_at(args.target_delta)

                metrics = {
                    "eval/chosen_reward": eval_m["rewards/chosen"],
                    "eval/rejected_reward": eval_m["rewards/rejected"],
                    "eval/accuracy": eval_m["rewards/accuracies"],
                    "eval/margin": eval_m["rewards/margins"],
                    "privacy/epsilon": epsilon,
                }
                eval_msg = (
                    f"  → Eval: acc={eval_m['rewards/accuracies']:.3f}, "
                    f"margin={eval_m['rewards/margins']:.4f}, "
                    f"chosen_r={eval_m['rewards/chosen']:.4f}, "
                    f"rejected_r={eval_m['rewards/rejected']:.4f}, ε={epsilon:.3f}"
                )

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
    print(f"Dataset: {args.dataset} ({global_train_size} train pairs)")
    print(f"Loss: {args.loss_type} (beta={args.beta})")
    print("\nTraining results:")
    print(f"  Total steps: {global_step}")
    if losses:
        print(f"  Initial loss: {losses[0]:.4f}")
        print(f"  Final loss: {losses[-1]:.4f}")
        if losses[0] != 0:
            print(
                f"  Loss reduction: {((losses[0] - losses[-1]) / abs(losses[0]) * 100):.1f}%"
            )

    final_metrics = eval_reward_metrics(trainable_params)
    print("\nFinal eval reward metrics:")
    print(f"  Chosen reward:   {final_metrics['rewards/chosen']:.4f}")
    print(f"  Rejected reward: {final_metrics['rewards/rejected']:.4f}")
    print(f"  Accuracy:        {final_metrics['rewards/accuracies']:.3f}")
    print(f"  Margin:          {final_metrics['rewards/margins']:.4f}")

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
