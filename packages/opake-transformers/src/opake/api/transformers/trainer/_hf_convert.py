"""HF ``transformers.TrainingArguments`` → opake manifest + converter.

Classifies every field on ``transformers.TrainingArguments`` into exactly one
bucket, driven through the shared engine in :mod:`._convert`:

- **DIRECT** — field name and semantics match opake; copy as-is.
- **RENAME** — HF name differs from opake; rename, preserve value.
- **TRANSFORM** — multi-field derivation (e.g. ``per_device_train_batch_size``
  + ``gradient_accumulation_steps`` collapse into the opake logical Poisson
  batch; ``max_grad_norm`` → ``clipping_norm``; Liger → perf kernels).
- **REJECT_IF_SET** — field exists in HF but has no opake equivalent; if set
  to anything non-default, raise ``ValueError`` with a per-field rationale +
  suggested alternative.
- **DROP_WITH_WARN** — field exists in HF but is irrelevant on the opake path;
  silently drop, emit a ``RuntimeWarning`` if non-default.

The partition is enforced by ``test_compat_manifest_exhaustive.py``: every
field on ``transformers.TrainingArguments`` appears in exactly one bucket.
Drift in upstream HF fails that test.

The public entry point is :meth:`TrainingArguments.from_hf`; this module holds
the machinery it delegates to.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any

from opake.exceptions import ConfigurationError, InputTypeError

from ._convert import (
    _apply_manifest,
    _get_dataclass_field_values,
    _normalize_dp_overrides,
    _reject_if_truthy,
)

if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger(__name__)


# ===========================================================================
# HF manifest — every ``transformers.TrainingArguments`` field, one bucket.
# ===========================================================================

# ---------------------------------------------------------------------------
# DIRECT — name and semantics match opake ``TrainingArguments``.
# ---------------------------------------------------------------------------
HF_DIRECT_FIELDS: frozenset[str] = frozenset(
    {
        # Output / scope
        "output_dir",
        "logging_dir",
        # Batch sizes
        "per_device_eval_batch_size",
        "eval_accumulation_steps",
        "eval_delay",
        # Optimizer / LR schedule
        "learning_rate",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "lr_scheduler_kwargs",
        # Training duration
        "num_train_epochs",
        "max_steps",
        # Logging
        "log_level",
        "log_level_replica",
        "log_on_each_node",
        "logging_strategy",
        "logging_first_step",
        "logging_steps",
        # Saving
        "save_strategy",
        "save_steps",
        "save_total_limit",
        "save_on_each_node",
        "save_only_model",
        "restore_callback_states_from_checkpoint",
        # Reproducibility
        "seed",
        "data_seed",
        "full_determinism",
        # Precision
        "use_cpu",
        "bf16",
        "bf16_full_eval",
        "tf32",
        # Distributed
        "local_rank",
        "ddp_backend",
        "ddp_timeout",
        # Evaluation
        "eval_strategy",
        "eval_steps",
        "eval_on_start",
        "eval_do_concat_batches",
        "prediction_loss_only",
        "include_for_metrics",
        "average_tokens_across_devices",
        "metric_for_best_model",
        "greater_is_better",
        "load_best_model_at_end",
        "ignore_data_skip",
        # DataLoader
        "dataloader_num_workers",
        "dataloader_persistent_workers",
        "dataloader_pin_memory",
        "dataloader_prefetch_factor",
        "dataloader_multiprocessing_context",
        "dataloader_in_order",
        "dataloader_drop_last",
        "remove_unused_columns",
        "torch_empty_cache_steps",
        # Labels
        "label_names",
        "label_smoothing_factor",
        # Reporting
        "report_to",
        "disable_tqdm",
        "run_name",
        "project",
        # Trackio (HF's W&B-style tracker) config — surfaced through the
        # ``report_to=['trackio']`` callback, which opake inherits via HF
        # Trainer's reporting machinery.  No DP-specific handling needed.
        "trackio_space_id",
        "trackio_bucket_id",
        "trackio_static_space_id",
        # Preemption: HF's JITCheckpointCallback installs a SIGTERM handler
        # that calls ``trainer._save_checkpoint(model, trial)``; opake's
        # override matches that signature and routes through the active
        # training context, so the DP accountant + sampler RNG land in the
        # snapshot intact.  No transform needed.
        "enable_jit_checkpoint",
        # Optimizer kwargs string / dict
        "optim_args",
        # Hub
        "push_to_hub",
        "hub_model_id",
        "hub_token",
        "hub_private_repo",
        "hub_revision",
        # Compile
        "torch_compile",
        "torch_compile_backend",
        "torch_compile_mode",
        # Misc
        "gradient_checkpointing",
        "gradient_checkpointing_kwargs",
        "skip_memory_metrics",
        "include_num_input_tokens_seen",
        "debug",
        "resume_from_checkpoint",
    }
)


# ---------------------------------------------------------------------------
# RENAME — HF name → opake name; value preserved.
# ---------------------------------------------------------------------------
HF_RENAME_MAP: dict[str, str] = {
    # Opake dropped the ``_type`` suffix on the scheduler kind.
    "lr_scheduler_type": "lr_scheduler",
}


# ---------------------------------------------------------------------------
# TRANSFORM — multi-field derivations.
# ---------------------------------------------------------------------------
def _batch_collapse(hf: dict[str, Any]) -> dict[str, Any]:
    """Collapse HF ``(per_device, grad_accum, auto_find_batch_size)`` into the
    opake logical Poisson batch + vmap chunk.

    HF's effective batch (``per_device × grad_accum``) IS the privacy-relevant
    unit, so it becomes opake's logical ``per_device_train_batch_size``;
    dropping ``grad_accum`` would under-account the amplification and emit a
    too-optimistic ε. The HF per-device size becomes the opake vmap chunk
    (``microbatch_size``), and HF's ``auto_find_batch_size`` maps to opake's
    ``auto_find_microbatch_size`` (shrinks the vmap chunk on OOM, not the
    logical batch — privacy-neutral).
    """
    per_device = int(hf.get("per_device_train_batch_size", 8))
    grad_accum = int(hf.get("gradient_accumulation_steps", 1) or 1)
    out: dict[str, Any] = {
        "per_device_train_batch_size": per_device * grad_accum,
        "auto_find_microbatch_size": bool(hf.get("auto_find_batch_size", False)),
    }
    if grad_accum > 1:
        # Only set the vmap chunk when grad_accum > 1; at grad_accum=1 the
        # logical batch == per_device, so leave microbatch_size at its
        # default (None → vmap over the full batch).
        out["microbatch_size"] = per_device
    return out


_HF_PAGED_OPTIMS = frozenset(
    {
        "adamw_apex_fused",
        "adamw_bnb_8bit",
        "adamw_8bit",
        "paged_adamw_8bit",
        "paged_adamw_32bit",
        "lion_8bit",
        "paged_lion_8bit",
        "paged_lion_32bit",
        "rmsprop_bnb",
        "rmsprop_bnb_8bit",
        "rmsprop_bnb_32bit",
    }
)


def _optim_collapse(hf: dict[str, Any]) -> dict[str, Any]:
    """Collapse HF optim aliases to opake's ``adamw`` (+ raise on paged variants).

    HF carries half a dozen ``adamw_*`` aliases (``adamw_torch``,
    ``adamw_hf``, ``adamw_torch_fused``) that all resolve to AdamW. Opake
    has a single ``adamw`` name. Quantized/Apex-fused variants
    (``paged_adamw_*``, ``adamw_8bit``, …) are not in opake-engine's
    torchopt path and raise.

    Other optimizer families (``sgd``, ``adafactor``, ``lion``, ``rmsprop``,
    …) pass through unchanged — opake's ``_resolve_optimizer_name`` will
    handle name resolution at trainer-construction time.
    """
    optim_value = hf.get("optim")
    if optim_value is None:
        return {}
    optim_str = str(optim_value).lower()
    # OptimizerNames enums stringify to e.g. ``"OptimizerNames.ADAMW_TORCH"``;
    # use the ``.value`` if present for a clean string.
    optim_str = getattr(optim_value, "value", optim_str)
    optim_str = str(optim_str).lower()

    if optim_str in _HF_PAGED_OPTIMS:
        raise ConfigurationError(
            *(
                f"hf_training_arguments.optim={optim_value!r}: Quantized / "
                f"Apex-fused optimizers are not in opake-engine's torchopt "
                f"path. Use ``optim='adamw'`` (opake's functional AdamW has no "
                f"fused CUDA kernel).",
            )
        )
    if optim_str in {"adamw_torch", "adamw_hf"}:
        return {"optim": "adamw"}
    # ``adamw_torch_fused`` translates to plain ``adamw``: fused is a kernel /
    # execution choice, not different optimizer math, and opake's functional
    # torchopt AdamW has no fused kernel (forwarding ``fused=True`` raised
    # ``TypeError`` at optimizer-build time).  Warn rather than silently
    # rewrite or fail — the update math is preserved, the kernel request is
    # not (#389).
    if optim_str == "adamw_torch_fused":
        warnings.warn(
            "opake: hf_training_arguments.optim='adamw_torch_fused' — the "
            "functional DP path has no fused AdamW kernel; using 'adamw' "
            "(identical update math, unfused execution).",
            RuntimeWarning,
            stacklevel=4,
        )
        return {"optim": "adamw"}
    # Pass through other optimizer names verbatim (as string, not enum).
    return {"optim": optim_str}


def _warmup_collapse(hf: dict[str, Any]) -> dict[str, Any]:
    """Collapse HF's warmup knobs onto opake's single ``warmup_steps``.

    Transformers 4.x carried an integer ``warmup_steps`` beside a
    fractional ``warmup_ratio`` and resolved the pair as "steps wins when
    non-zero".  5.x dropped the ratio and taught ``warmup_steps`` to read a
    value below 1 as a fraction of the total instead.  Opake follows 5.x,
    so both forms pass straight through: a 4.x ratio is already the
    fractional encoding and needs no arithmetic.
    """
    steps = hf.get("warmup_steps") or 0
    if steps:
        return {"warmup_steps": steps}
    return {"warmup_steps": hf.get("warmup_ratio") or 0}


def _max_grad_norm_to_clipping(hf: dict[str, Any]) -> dict[str, Any]:
    """Loosely map HF's global grad-norm clip → opake ``clipping_norm``.

    Not semantically identical (HF clips the aggregate gradient; opake
    clips per example for DP), but it's the closest knob and a sensible
    default. A ``clipping_norm`` DP override wins (applied after the manifest).
    """
    v = hf.get("max_grad_norm")
    return {"clipping_norm": float(v)} if v is not None else {}


def _liger_to_perf_kernels(hf: dict[str, Any]) -> dict[str, Any]:
    """Map HF Liger fused kernels → opake performance kernels."""
    if not hf.get("use_liger_kernel"):
        return {}
    out: dict[str, Any] = {"use_performance_kernels": True}
    cfg = hf.get("liger_kernel_config")
    if cfg:
        out["performance_kernels_config"] = dict(cfg)
    return out


HF_TRANSFORM_MAP: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "per_device_train_batch_size": _batch_collapse,
    "gradient_accumulation_steps": _batch_collapse,
    "auto_find_batch_size": _batch_collapse,
    "optim": _optim_collapse,
    "warmup_steps": _warmup_collapse,
    "warmup_ratio": _warmup_collapse,
    "max_grad_norm": _max_grad_norm_to_clipping,
    "use_liger_kernel": _liger_to_perf_kernels,
    "liger_kernel_config": _liger_to_perf_kernels,
}


# ---------------------------------------------------------------------------
# Fields the manifest classifies that exist only in some supported
# transformers releases. The declared range spans a major boundary, so the
# manifest covers the union of those surfaces.
# ---------------------------------------------------------------------------
HF_LEGACY_FIELDS: frozenset[str] = frozenset(
    {
        # Removed in 5.x, which folded the fraction into ``warmup_steps``.
        "warmup_ratio",
        # Added after the 5.0 compatibility floor.
        "dataloader_in_order",
        "dataloader_multiprocessing_context",
        "ddp_static_graph",
        "group_by_length",
        "logging_dir",
        "trackio_bucket_id",
        "trackio_static_space_id",
        "train_sampling_strategy",
    }
)


# ---------------------------------------------------------------------------
# REJECT_IF_SET — value must be the dataclass default to skip rejection.
# Callable returns an error message string (raised) or None (benign value).
# ---------------------------------------------------------------------------
HF_REJECTED_FIELDS: dict[str, Callable[[Any], str | None]] = {
    # Half-precision: opake is bf16-only.
    "fp16": _reject_if_truthy(
        "DP-SGD requires deterministic numerics; opake is bf16-only. "
        "Use ``bf16=True`` instead."
    ),
    "fp16_full_eval": _reject_if_truthy(
        "Use ``bf16_full_eval=True``; opake does not support fp16."
    ),
    # Distributed / sharding integrations not on the per-example DP path.
    "fsdp": _reject_if_truthy(
        "FSDP is not supported on the per-example DP-SGD path. Use the "
        "standard DDP backend via ``ddp_backend='nccl'``."
    ),
    "fsdp_config": _reject_if_truthy(
        "FSDP config is not supported; opake uses per-example DDP only."
    ),
    "deepspeed": _reject_if_truthy(
        "DeepSpeed is not supported on the per-example DP-SGD path."
    ),
    # NEFTune embedding noise would interact with the privacy accountant.
    "neftune_noise_alpha": _reject_if_truthy(
        "NEFTune embedding noise is not wired through the DP-SGD path "
        "(it would interact with the privacy accountant)."
    ),
    "group_by_length": _reject_if_truthy(
        "Length-grouped batching is incompatible with Poisson subsampling. "
        "Use the opake sampling_mode options instead."
    ),
    # Hub auto-push.
    "hub_always_push": _reject_if_truthy(
        "Per-checkpoint auto-push is not supported; use ``push_to_hub=True`` "
        "for the end-of-training push."
    ),
    # ``adafactor``, ``max_grad_norm``, ``use_liger_kernel`` /
    # ``liger_kernel_config``, and ``auto_find_batch_size`` are remapped in
    # HF_TRANSFORM_MAP, not rejected here.
}


# ---------------------------------------------------------------------------
# DROP_WITH_WARN — field exists in HF but is irrelevant on opake; drop with
# a warning when non-default.
# ---------------------------------------------------------------------------
HF_DROP_FIELDS: dict[str, str] = {
    # HF deprecated runtime-gating booleans; opake uses explicit
    # ``trainer.train()`` / ``trainer.evaluate()`` calls.
    "do_train": "HF deprecated runtime-gating booleans; call trainer.train() directly.",
    "do_eval": "HF deprecated runtime-gating booleans; call trainer.evaluate() directly.",
    "do_predict": "HF deprecated runtime-gating booleans; call trainer.predict() directly.",
    # Length-based grouping is incompatible with Poisson sampling.
    "length_column_name": (
        "Length-grouped batching is incompatible with Poisson subsampling "
        "and is not used by opake."
    ),
    "train_sampling_strategy": (
        "Opake uses ``sampling_mode`` (poisson / b_min_sep / balls_in_bins) "
        "to pair the train-time sampler with the privacy mechanism for DP "
        "guarantees; HF's 'random' / 'sequential' / 'group_by_length' don't "
        "map onto that namespace."
    ),
    # HF stats output paths opake does not honor.
    "logging_nan_inf_filter": "Opake's logging path computes NaN/Inf filtering separately.",
    # DDP knobs opake doesn't expose.
    "ddp_find_unused_parameters": "Opake's per-example DDP path doesn't use this knob.",
    "ddp_bucket_cap_mb": "Opake's per-example DDP path doesn't use bucket sizing.",
    "ddp_broadcast_buffers": "Opake's per-example DDP path manages buffer sync internally.",
    "ddp_static_graph": (
        "Opake doesn't wrap the model in ``DistributedDataParallel``; it "
        "manages cross-rank sync via direct ``torch.distributed.all_reduce`` "
        "calls, so PyTorch's DDP static-graph optimization has no effect here."
    ),
    "use_cache": (
        "Opake's per-example vmap+grad path can't carry the stateful "
        "``past_key_values`` cache, and training rarely needs it anyway. "
        "Opake forces ``model.config.use_cache = False`` regardless."
    ),
    # New HF features outside the opake path.
    "parallelism_config": "HF parallelism config is not used by opake.",
    "optim_target_modules": "Galore-target-modules selection is not on the opake optim path.",
    "batch_eval_metrics": "Streaming-eval metric callback is not used by opake.",
    "eval_use_gather_object": "HF Accelerate-only eval gather; not used by opake.",
    # Hub settings.
    "hub_strategy": (
        "Opake only supports end-of-training push via ``push_to_hub=True``; "
        "any HF ``hub_strategy`` setting is ignored."
    ),
    # Accelerate-driven config: opake manages distribution natively.
    "accelerator_config": (
        "Accelerate-driven config is not used; opake uses ``ddp_backend`` directly."
    ),
}


# ---------------------------------------------------------------------------
# Public API: convert an HF TrainingArguments instance to an opake dict.
# ---------------------------------------------------------------------------
def _convert_hf_training_arguments(
    hf_args: Any,
    *,
    strict: bool = True,
    **dp_overrides: Any,
) -> dict[str, Any]:
    """Translate an HF ``TrainingArguments`` instance into opake-side kwargs.

    The DP-override kwargs are merged onto the converted dict last (they
    take precedence over both HF values and HF-derived defaults).
    """
    # Local import so this module stays importable when HF is absent.
    try:
        from transformers import TrainingArguments as HFTrainingArguments
    except ImportError as e:
        raise ImportError(  # noqa: TRY003 - preserve standard Python error contract
            "_convert_hf_training_arguments requires the ``transformers`` "
            "package. Install it with ``pip install transformers``."
        ) from e

    if not isinstance(hf_args, HFTrainingArguments):
        raise InputTypeError(
            *(
                f"Expected ``transformers.TrainingArguments`` instance, got "
                f"{type(hf_args).__name__}. To convert a dict, build a "
                "``TrainingArguments(**your_dict)`` first.",
            )
        )

    source_values = _get_dataclass_field_values(hf_args)
    # HF's ``__post_init__`` rewrites several field defaults at runtime, so
    # ``dataclasses.fields()`` defaults don't match what an unconfigured user
    # sees. Build a baseline instance and use its field values as the
    # "user didn't set this" defaults.
    import tempfile

    baseline_output_dir = source_values.get("output_dir") or tempfile.mkdtemp(
        prefix="opake_baseline_"
    )
    baseline_kwargs = {"output_dir": baseline_output_dir}
    if source_values.get("use_cpu"):
        baseline_kwargs["use_cpu"] = True
    baseline = type(hf_args)(**baseline_kwargs)
    source_defaults = _get_dataclass_field_values(baseline)

    # Wrap REJECT and TRANSFORM signatures into the dispatcher contract.
    converted = _apply_manifest(
        source_values=source_values,
        source_defaults=source_defaults,
        direct=HF_DIRECT_FIELDS,
        rename=HF_RENAME_MAP,
        transform=HF_TRANSFORM_MAP,
        reject=HF_REJECTED_FIELDS,
        drop=HF_DROP_FIELDS,
        source_label="hf_training_arguments",
        strict=strict,
    )

    # Performance kernels default ON in opake but OFF in HF/TRL; default them
    # OFF here to match upstream (the Liger transform sets True when Liger was
    # on; a name override below can force either way).
    converted.setdefault("use_performance_kernels", False)

    # Overrides win over every converted/derived value (the privacy knobs plus
    # any opake field overridden by name, e.g. ``use_performance_kernels=True``
    # even though HF had no such field).
    overrides = _normalize_dp_overrides(dp_overrides)
    converted.update(overrides)

    log.info(
        "converted hf_training_arguments → %d opake fields (%d overridden by name)",
        len(converted),
        len(overrides),
    )
    return converted
