# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
#
# Trainer control-flow and public method surface adapted from Hugging Face
# Transformers Trainer (Apache-2.0; https://github.com/huggingface/transformers),
# then reworked around Opaque's functional per-example DP training flow.
# See ../../../../../NOTICE in this package for the full attribution.
"""DP-SGD Trainer for HuggingFace models.

Provides :class:`Trainer` — a differentially private, HF-Trainer-parity
trainer built on Opaque primitives.  See the class docstring for the
public method layout.

The trainer is shape-agnostic: any HF-style ``data_collator`` whose output
the model's forward accepts will work.  Domain-specific training that
builds on this trainer (SFT / DPO / KTO) should subclass it and override
:meth:`Trainer.compute_per_example_loss` — the single DP-correct
extension point that both training (vmap → grad → clip → noise) and
eval (vmap when ``include_for_metrics=['loss']``) route through.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import inspect
import json
import logging
import math
import os
import shutil
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torchopt
from datasets import Dataset
from torch import Tensor
from torch.utils.data import DataLoader

import opaque.accounting as acc
import opaque.dpsgd.accounting as dpsgd_acc
from opaque.accounting import Accountant
from opaque.accounting import calibration as cal
from opaque.accounting.types import DpHorizonProcess
from opaque.api.engine.clipping import clipped_grad
from opaque.api.engine.device import (
    device_capabilities,
    sdpa_autocast_under_vmap_broken,
)
from opaque.api.transformers._rng import IGNORE_DATA_SKIP_STREAM_FOLD
from opaque.dpftrl.noise import mf_gaussian_noise
from opaque.dpsgd.clipping import adaptive_clipped_grad, auto_clipped_grad
from opaque.dpsgd.noise import gaussian_noise
from opaque.exceptions import (
    CheckpointError,
    ConfigurationError,
    InputTypeError,
    OperationError,
)
from opaque.functional import make_functional
from opaque.optimizers.types import ScheduleFreeState
from opaque.profiling import PerfTracker, perf_tracker
from opaque.random import key, split
from opaque.serialization import (
    from_state_dict as opaque_from_state_dict,
)
from opaque.serialization import (
    state_dict as opaque_state_dict,
)
from opaque.types import PerGroup
from transformers import (
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    SequenceFeatureExtractor,
    enable_full_determinism,
    set_seed,
)
from transformers.data.data_collator import default_data_collator
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer_callback import TrainerCallback, TrainerControl
from transformers.trainer_utils import (
    RemoveColumnsCollator,
    TrainerMemoryTracker,
    seed_worker,
    speed_metrics,
)
from transformers.utils import find_labels

if TYPE_CHECKING:
    from opaque.random.types import RngKey
from . import _checkpoint as ckpt
from . import _distributed, _dpftrl, _eval, _hub
from ._callback import (
    BestModelSaveCallback,
    build_callback_handler,
    is_metric_improved,
    resolve_eval_metric,
)
from ._eval import EvalPrediction
from ._precision import eval_dtype
from ._scheduler import build_lr_schedule
from ._state import TrainerState
from ._training_arguments import TrainingArguments, _default_privacy_delta
from .types import EvaluationResult, TrainOutput

__all__ = [
    "Trainer",
    "EvaluationResult",
    "TrainOutput",
    "TrainingArguments",
]

log = logging.getLogger(__name__)
_ARG_DRIFT_ABSOLUTE_TOLERANCE = 1e-12
_ARG_DRIFT_RELATIVE_TOLERANCE = 1e-6
_IGNORE_INDEX = -100


def _callback_matches(
    candidate: type[TrainerCallback] | TrainerCallback,
    target: type[TrainerCallback] | TrainerCallback,
) -> bool:
    """Whether ``candidate`` should be removed when removing ``target``.

    Class targets match by ``isinstance`` (so passing the class removes the
    first instance of that class); instance targets match by identity.
    """
    if isinstance(target, type):
        if isinstance(candidate, type):
            return candidate is target
        return isinstance(candidate, target)
    return candidate is target


def _is_peft_model(model: Any) -> bool:
    """Whether ``model`` is a PEFT-wrapped model.

    Local reimplementation of HF's private ``transformers.trainer._is_peft_model``
    so we don't import private API. Returns ``False`` when ``peft`` is not installed.
    """
    try:
        from peft import PeftMixedModel, PeftModel
    except ImportError:
        return False
    return isinstance(model, (PeftModel, PeftMixedModel))


def _is_causal_lm_model(model: Any) -> bool:
    """Match Transformers' label-smoothing dispatch for causal language models."""
    if _is_peft_model(model):
        model = model.base_model.model
    model_name = (
        model._get_name() if hasattr(model, "_get_name") else type(model).__name__
    )
    return model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values()


def _disable_tokenizers_parallelism_before_fork() -> None:
    """Avoid HuggingFace tokenizers fork warnings from reporting integrations."""
    if "TOKENIZERS_PARALLELISM" in os.environ:
        return
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    log.debug(
        "Set TOKENIZERS_PARALLELISM=false before training; set it explicitly "
        "to override this Trainer default."
    )


def _effective(value: Any) -> float:
    """Extract scalar from float or PerGroup for logging."""
    return value.effective if hasattr(value, "effective") else float(value)


def _drift_differs(saved: Any, current: Any) -> bool:
    """Equality test used by ``_warn_on_arg_drift``.

    Floats compare with a relative tolerance of 1e-6 (silences spurious
    drift on benign rounding); everything else compares with ``!=``.
    """
    if isinstance(saved, float) and isinstance(current, float):
        return (
            abs(saved - current) / max(abs(saved), _ARG_DRIFT_ABSOLUTE_TOLERANCE)
            > _ARG_DRIFT_RELATIVE_TOLERANCE
        )
    return saved != current


def _resolve_drift_disposition(
    field_metadata: Mapping[str, Any], saved_mechanism: str
) -> str:
    """Resolve a field's ``drift`` disposition for the saved mechanism.

    The metadata value is either a string ("dp_relevant" / "shape" /
    "intentional_extend") or a dict for per-mechanism overrides
    (``{"gaussian": "intentional_extend", "default": "dp_relevant"}``).
    """
    drift = field_metadata.get("drift", "dp_relevant")
    if isinstance(drift, dict):
        return drift.get(saved_mechanism, drift.get("default", "dp_relevant"))
    return drift


def _compile_strict_chunk(
    fn: Callable, *, backend: str | Callable, mode: str
) -> Callable:
    """Compile one tensor-only gradient chunk as a strict dynamic graph."""
    return torch.compile(
        fn,
        backend=backend,
        mode=mode,
        fullgraph=True,
        dynamic=True,
    )


@dataclasses.dataclass
class _TrainingContext:
    """Mutable state carried through the training loop."""

    fmodel: Callable
    trainable_params: dict[str, Tensor]
    frozen_params: dict[str, Tensor]
    grad_fn: Callable
    clip_state: Any
    noise_fn: Callable
    noise_state: Any
    noise_multiplier: float
    opt: Any
    opt_state: Any
    lr_schedule: Callable[[int], float]
    accounting: Accountant
    mechanism: Callable
    # Cached process reused across independent step compositions. Whole-horizon
    # mechanisms are installed in ``accounting`` once and leave this as None.
    step_process: Any | None
    target_delta: float
    sample_rate: float
    calibration_source: str
    expected_steps_per_epoch: int
    total_steps: int
    num_epochs: int
    collate_fn: Callable
    batch_keys: tuple[str, ...] = ()
    offload_ctx: Any = dataclasses.field(default_factory=contextlib.nullcontext)
    opt_name: str = "adamw"
    current_sampler: Any = None
    # Checkpoint cursor for a distinct ignored-state Poisson stream.
    sampler_restart_step: int | None = None
    # ``global_step`` value corresponding to ``current_sampler.consumed == 0``
    # for this run — 0 unless a prior ``ignore_data_skip`` resume rebased the
    # sampler stream. Set once in ``_inner_training_loop`` before iteration.
    sampler_step_offset: int = 0
    save_steps_resolved: int = 0
    # Configured clip threshold (scalar or PerGroup).  Adaptive mode
    # overrides this each step via ``clip_state.clipping_norm``; fixed
    # mode reads the configured value directly because ``FixedClipState``
    # is a marker without per-state fields.
    clip_norm: Any = None
    mechanism_kind: str = "gaussian"
    is_horizon_process: bool = False
    horizon_process: DpHorizonProcess | None = None
    mf: _dpftrl.MFContext | None = None
    # Predicted stop-at-ε crossing step (absolute), or ``None`` when the
    # target is unreachable, unset, or the process is whole-horizon.
    # See :func:`predict_stop_step`.
    stop_at_step: int | None = None


def _initialize_accounting(
    process: Any,
) -> tuple[Accountant, Any | None, DpHorizonProcess | None]:
    """Build accounting state for a complete horizon or an independent step."""
    if isinstance(process, DpHorizonProcess):
        return Accountant(prefix=acc.cached(process)), None, process
    return Accountant(), acc.cached(process), None


def _account_independent_step(ctx: _TrainingContext) -> None:
    """Compose one step unless the context already holds a complete horizon."""
    if ctx.step_process is not None:
        ctx.accounting |= ctx.step_process


def predict_stop_step(
    prefix_process: Any,
    step_process: Any,
    *,
    target_epsilon: float,
    delta: float,
    k0: int,
    horizon: int,
) -> int | None:
    """Predict the first accounted step where ε reaches ``target_epsilon``.

    The accountant after ``k`` accrued steps is exactly
    ``prefix ∘ step_process^(k - k0)`` — the per-step ``|=`` fold and the
    ``Repeated`` (``*``) operator produce the same process — so for a
    deterministic, monotone ε(k) the crossing step can be binary-searched
    once at setup (~log2(horizon) ``epsilon_at`` probes) instead of being
    measured during the training loop.

    Returns the absolute step count in ``(k0, horizon]``, or ``None`` when
    the target is not reached within the horizon or the ε sequence fails
    the boundary monotonicity check (callers fall back to the log-boundary
    stop check).
    """
    remaining = horizon - k0
    if remaining < 1:
        return None

    cache: dict[int, float] = {}

    def eps(j: int) -> float:
        if j not in cache:
            proc = (
                step_process * j
                if prefix_process is None
                else prefix_process | (step_process * j)
            )
            cache[j] = proc.epsilon_at(delta)
        return cache[j]

    if eps(remaining) < target_epsilon:
        return None
    lo, hi = 1, remaining
    while lo < hi:
        mid = (lo + hi) // 2
        if eps(mid) >= target_epsilon:
            hi = mid
        else:
            lo = mid + 1
    # Guard against numerical non-monotonicity: the crossing must be a
    # genuine boundary (ε(M-1) < target <= ε(M)); both probes are cached
    # from the search, so the check is free.
    if eps(lo) < target_epsilon or (lo > 1 and eps(lo - 1) >= target_epsilon):
        log.warning(
            "stop-at-ε prediction failed the monotone-boundary check near "
            "step %d; falling back to the log-boundary check.",
            k0 + lo,
        )
        return None
    return k0 + lo


@contextlib.contextmanager
def _deep_json_recursion(limit: int = 30_000):
    """Temporarily raise the recursion limit around stdlib json of deep
    accountant wire dicts.

    The DpProcess codec walks composition spines iteratively, but stdlib
    ``json``'s C encoder/scanner still recurse once per nesting level and
    hit the default limit near ~1000 — an accountant with a thousand-plus
    heterogeneous accounted steps could otherwise not be checkpointed or
    resumed.  Bounded and restored on exit; 30k covers >25k-step spines.

    Do NOT raise the cap: the recursion limit is what keeps the json C
    scanner inside the native stack (empirically a hard segfault near
    depth 100k under limit 150k), and on Python 3.13+ the C recursion is
    additionally capped by ``Py_C_RECURSION_LIMIT`` regardless of this
    limit.  ``sys.setrecursionlimit`` is process-global; the save/load
    paths are single-threaded per process (DDP is per-process), so the
    temporary bump cannot race another thread's limit.
    """
    old = sys.getrecursionlimit()
    if limit > old:
        sys.setrecursionlimit(limit)
    try:
        yield
    finally:
        sys.setrecursionlimit(old)


class Trainer:
    """Differentially private trainer for HuggingFace models.

    Method decomposition mirrors HF Trainer:

    - ``train()`` → ``_setup_training()`` + ``_inner_training_loop()``
    - ``training_step()`` — clip → noise → optimize (fused, unlike HF)
    - ``evaluate()`` / ``predict()`` — both return :class:`EvaluationResult`;
      shared pipeline via ``_run_evaluation_loop``
    - ``compute_per_example_loss()`` — DP-correct override hook; the
      single extension point for SFT / DPO / KTO subclasses
    - ``create_optimizer()`` — functional optimizer (torchopt)
    - ``get_train_dataloader()`` — PoissonSampler
    - ``get_eval_dataloader()`` — standard DataLoader
    - ``log()`` — append to state + fire callbacks
    """

    #: eval_loss aggregation: ``True`` → weight per-example / batch losses by
    #: their real (non ``-100``) token counts, reconstructing the corpus
    #: per-token mean (HF CE parity; valid when the per-example loss is a
    #: token-mean CE).  Subclasses whose eval loss is a per-example objective
    #: in its own right (e.g. SFT ``dft``) set this ``False`` → plain
    #: per-example mean, matching the training objective's fixed,
    #: per-example-equal weighting (no data-dependent token-count divisor).
    _eval_token_weighted_loss: bool = True

    def __init__(
        self,
        model: PreTrainedModel | None = None,
        args: TrainingArguments | None = None,
        data_collator: Callable | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        processing_class: PreTrainedTokenizerBase | None = None,
        compute_loss_func: Callable | None = None,
        compute_metrics: Callable | None = None,
        callbacks: list[Any] | None = None,
        optimizers: tuple[Any | None, Any | None] = (None, None),
        optimizer_cls_and_kwargs: tuple[Any, dict[str, Any]] | None = None,
        preprocess_logits_for_metrics: Callable | None = None,
    ) -> None:
        if args is None:
            args = TrainingArguments(output_dir="tmp_trainer")
        # HF parity (``Trainer.__init__``): seed Python / NumPy / torch
        # global RNGs from ``args.seed`` so non-DP randomness (model-
        # head init when missing weights are randomised, dataset
        # shuffling without an explicit seed, user-supplied
        # ``compute_metrics`` calling ``torch.randn``, ...) is
        # reproducible run-to-run.  The DP RNG chain seeds itself from
        # ``key(args.seed)`` independently — this call is purely for
        # the non-DP surface.
        if args.full_determinism:
            enable_full_determinism(args.seed)
            log.warning(
                "full_determinism=True: enabled deterministic algorithms; this can reduce throughput."
            )
        else:
            set_seed(args.seed)
        if model is None:
            raise OperationError(*("`Trainer` requires a `model` argument",))
        self._functional_optimizer_factory: (
            tuple[Callable[..., Any], dict[str, Any]] | None
        ) = None
        self._functional_optimizer_name: str | None = None
        if any(item is not None for item in optimizers):
            raise ConfigurationError(
                *(
                    "Passing `optimizers` is not supported by Trainer: the DP path "
                    "uses a functional torchopt optimizer built after per-example "
                    "gradient clipping/noising is configured.",
                )
            )
        if optimizer_cls_and_kwargs is not None:
            from ._optim import (
                validate_functional_optimizer_cls_and_kwargs,
            )

            self._functional_optimizer_factory = (
                validate_functional_optimizer_cls_and_kwargs(optimizer_cls_and_kwargs)
            )
            _fn = self._functional_optimizer_factory[0]
            self._functional_optimizer_name = getattr(
                _fn, "__name__", type(_fn).__name__
            )
        self._model = model
        # Computed once: PEFT detection appears at multiple resume /
        # restore sites and the result depends only on the model class,
        # which doesn't change after construction.
        self._is_peft: bool = _is_peft_model(model)
        self._is_causal_lm: bool = _is_causal_lm_model(model)
        self.args = args
        if args.torch_compile and bool(
            getattr(model, "is_gradient_checkpointing", False)
        ):
            raise ConfigurationError(
                *(
                    "torch_compile=True is incompatible with a model that already "
                    "has gradient checkpointing enabled: checkpointed functional "
                    "transforms use saved-tensor hooks that AOTAutograd cannot safely "
                    "compose with torch.compile(vmap(grad(...))). Disable gradient "
                    "checkpointing on the model or disable torch compilation.",
                )
            )
        self._processing_class = processing_class
        self._base_callbacks: list[Any] = list(callbacks) if callbacks else []
        self.is_in_train = False
        # HF parity: when no collator is given, use
        # ``DataCollatorWithPadding`` for tokenizer / sequence-feature
        # processors and ``default_data_collator`` otherwise.  The
        # DataLoader collator stays CPU-only; tensors are moved to the
        # trainer device in ``_prepare_input`` on the main process.
        default_collator = (
            DataCollatorWithPadding(processing_class)
            if processing_class is not None
            and isinstance(
                processing_class, (PreTrainedTokenizerBase, SequenceFeatureExtractor)
            )
            else default_data_collator
        )
        self._data_collator = data_collator or default_collator
        self._compute_metrics = compute_metrics
        self._compute_loss_func = compute_loss_func
        self._preprocess_logits = preprocess_logits_for_metrics
        self._warned_per_example_logits_only = False
        self._warned_eval_on_train = False
        # Lazily-built per-example eval-loss closure (vmap'd).  Populated
        # by ``_get_eval_per_example_loss_fn`` on first use; reset to
        # ``None`` here so model rebinding can invalidate the cache.
        self._eval_per_example_loss_fn: Callable | None = None
        self._eval_per_example_loss_fn_model: Any = None
        self._eval_per_example_loss_only_fn: Callable | None = None
        self._eval_per_example_loss_only_fn_model: Any = None
        # Per-batch eval telemetry channel: a subclass ``prediction_step`` sets
        # this to a per-example dict that ``evaluation_loop`` collects + means
        # into the eval metrics (symmetric with the train-step aux channel).
        self._pending_eval_aux: dict[str, Tensor] | None = None

        # Default label_names so the eval loop can identify label tensors in
        # the batch dict.  HF parity: inspect the model's forward signature
        # for parameters whose name contains "label" — for
        # ``*ForQuestionAnswering`` models additionally pick up
        # ``start_positions`` / ``end_positions``.  Walk through the PEFT
        # wrapper to the base model so the inspected signature is the one
        # that actually consumes the labels.  Snapshot to a private
        # attribute so the user-supplied ``args`` is never mutated.
        if args.label_names is not None:
            self._label_names: list[str] = list(args.label_names)
        else:
            inspected = model
            if self._is_peft:
                if hasattr(model, "get_base_model"):
                    inspected = model.get_base_model()
                else:
                    inspected = model.base_model.model
            discovered = find_labels(inspected.__class__)
            self._label_names = list(discovered) if discovered else ["labels"]

        # HF parity: the trainer does not tokenise.  The user supplies
        # already-prepared datasets and a matching ``data_collator``.
        self._train_dataset = train_dataset
        self._eval_dataset = eval_dataset
        if args.eval_strategy != "no" and eval_dataset is None:
            raise CheckpointError(
                *(
                    f"You have set `args.eval_strategy` to {args.eval_strategy} but "
                    "didn't pass an `eval_dataset` to `Trainer`. Either set "
                    "`eval_strategy='no'` or pass an eval_dataset.",
                )
            )

        # Resolve device via ``args.device`` (forwards to
        # :meth:`TrainingArguments._setup_devices`, which honors
        # ``use_cpu`` / ``use_mps_device`` / ``no_cuda`` and bypasses
        # Accelerate).  HF parity: the trainer's device source of truth
        # is the args, not whatever device the user happened to leave
        # the model on.  The model is moved to ``self._device`` so the
        # batches we forward (also moved to the same device by the
        # collate-fn wrapper) line up.
        self._device = self.args.device
        _SUPPORTED_DEVICE_TYPES = {"cpu", "cuda", "mps"}
        if self._device.type not in _SUPPORTED_DEVICE_TYPES:
            raise ConfigurationError(
                *(
                    f"Trainer only supports cpu, cuda, and mps devices; "
                    f"got device={self._device!r}. "
                    f"Other backends (xpu, npu, mlu, musa, hpu, ...) are not supported.",
                )
            )
        # resolve rank/world topology immediately after the device
        # pick so every subsequent setup site (sampler, checkpoint, hub,
        # logging) can read the same snapshot.  Single-process is the trivial
        # case (rank=0, world=1, is_distributed=False).
        self._ddp = _distributed.resolve_ddp_state(self._device, self.args)
        _distributed.validate_ddp_backend(self.args, self._ddp)
        # Apply per-rank logging verbosity now that rank/world is known
        # (HF parity: main process uses ``log_level``, replicas use
        # ``log_level_replica``).
        _distributed.apply_logging(self.args, self._ddp)
        # HF parity (``Trainer._wrap_model``): place the model on the
        # resolved device.  ``model.to`` is a no-op when the model is
        # already there, so this is safe for callers who pre-placed.
        self._model.to(self._device)

        # Explicit patch sites (no import-time mutation of HF globals):
        # 1) global runtime compat (masking / collator / checkpoint hooks)
        # 2) ``apply_model_patches(..., compat=use_compat_patches, performance=True, kernels=use_performance_kernels)``
        from opaque.patches import apply_runtime_patches

        apply_runtime_patches(compat=True)
        self._apply_opaque_model_patches()

        # Compute precision: bf16 autocast for training, full-cast only for
        # the bf16_full_eval scope.  See _setup_precision.
        self._amp_dtype: torch.dtype | None = None
        self._setup_precision()

        # Functional state (populated by _setup_training, used by evaluate)
        self._ctx: _TrainingContext | None = None
        # Privacy accountant lives at the trainer level so ``save_model()``
        # can write ``accountant.json`` after training finishes.  The
        # ``_setup_training`` finally block copies the live accountant
        # off the per-run context into this slot; checkpoint loads
        # restore directly into it.
        self._accountant: Accountant | None = None
        self._train_dataloader: DataLoader | None = None
        self._eval_dataloader: DataLoader | None = None
        self._signature_columns: list[str] | None = None
        self._signature_columns_unavailable = False

        # All cross-field validation (save/eval strategy invariants,
        # load_best_model_at_end requirements, …) lives in
        # ``TrainingArguments.__post_init__``.  The trainer reads the
        # validated ``args`` directly — no defensive snapshots.

        # Callback state.  ``state.{logging,eval,save}_steps`` are
        # resolved from ``args`` via ``state.compute_steps`` so
        # fractional values (e.g. ``logging_steps=0.5``) and ``None``
        # are handled exactly as HF does — and so callbacks running at
        # ``on_init_end`` see the post-resolution cadence rather than
        # zeros from a premature ``int(...)`` truncation.
        self.state = TrainerState()
        self.state.max_steps = self._predict_total_steps()
        self.state.compute_steps(args)
        self._stamp_ddp_flags(self.state)
        # Smoothed-loss bookkeeping (HF parity, trainer-internal).
        # ``_tr_loss`` accumulates per-step losses on device; ``_total_loss_scalar``
        # is the running total across all logging windows; ``_globalstep_last_logged``
        # is the step at which we last drained ``_tr_loss``.  Reset to
        # ``self.state.global_step`` on resume so the first post-resume
        # log row averages over the right window.  The tensor stays on
        # device so the DDP gather of ``tr_loss`` needs no
        # extra device migration.
        self._tr_loss: Tensor = torch.tensor(0.0, device=self._device)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged: int = 0
        # Token-count bookkeeping.
        # Populated during the training loop when ``include_num_input_tokens_seen``
        # is set; used by ``log()`` for live tokens/sec and by the training summary.
        self._train_start_time: float | None = None
        self._control = TrainerControl()
        # Memory tracker (HF parity: ``TrainerMemoryTracker`` handles
        # ``skip_memory_metrics``, psutil availability, CPU + GPU tracking).
        self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
        self._memory_tracker.start()
        # Opaque step-level performance tracker: per-step wall-clock,
        # throughput, peak memory, and sub-step marks for clip / noise /
        # optimizer.  Sibling to ``_memory_tracker`` (which handles HF's
        # before/after-train memory metrics); the two surfaces are
        # complementary — HF's ``speed_metrics`` for run-level wall-clock
        # stays, the tracker adds per-step + post-warmup steady-state.
        self._perf_tracker: PerfTracker = perf_tracker(self._device)
        self._callback_handler = build_callback_handler(
            args=args,
            model=self._model,
            processing_class=processing_class,
            callbacks=self._base_callbacks,
        )
        # ``DefaultFlowCallback`` doesn't recognise ``save_strategy="best"``;
        # auto-inject the matching callback so user callbacks aren't required
        # to know about this gap.  Also inject it for ``load_best_model_at_end``
        # (with a real save cadence): otherwise the best metric can land on an
        # eval-only step with no checkpoint folder, leaving best unloadable
        # (issue #386).  Gated on the trainer-side snapshot so the
        # demoted-to-``"no"`` case (output_dir is None) doesn't install it.
        if self.args.save_strategy == "best" or (
            self.args.load_best_model_at_end and self.args.save_strategy != "no"
        ):
            self._callback_handler.add_callback(BestModelSaveCallback())
        if args.debug and "underflow_overflow" in str(args.debug):
            from transformers.debug_utils import DebugUnderflowOverflow

            # Keep a strong reference: the debug object owns module hooks.
            self._debug_underflow_overflow = DebugUnderflowOverflow(self._model)

        # Warn (don't raise) when output_dir is non-empty and overwrite is off
        # — user might be intending to resume.
        self._warn_if_existing_output_dir()

        # Fire HF's ``on_init_end`` once construction is complete so callbacks
        # that pre-flight invariants (e.g. ``EarlyStoppingCallback``) can run.
        self._control = self._callback_handler.on_init_end(
            args, self.state, self._control
        )
        # HF parity: stop the init-phase memory snapshot here so training
        # (which re-starts the tracker) gets a clean baseline.
        self._memory_tracker.stop_and_update_metrics()

        # Hub publishing: create/validate the repo up front when push_to_hub
        # is set so a misconfigured token/repo fails fast at construction
        # rather than after a full training run.  push_to_hub() also lazily
        # inits if called directly without this flag.
        self.hub_model_id: str | None = args.hub_model_id
        if args.push_to_hub:
            _hub.init_hf_repo(self)

    # ------------------------------------------------------------------
    # Public properties (TrainerProtocol)
    # ------------------------------------------------------------------

    @property
    def model(self) -> PreTrainedModel:
        return self._model

    @model.setter
    def model(self, value: PreTrainedModel) -> None:
        self._model = value
        if hasattr(self, "_callback_handler"):
            self._callback_handler.model = value

    @property
    def processing_class(self) -> PreTrainedTokenizerBase | None:
        return self._processing_class

    @property
    def data_collator(self) -> Callable:
        return self._data_collator

    @property
    def compute_metrics(self) -> Callable | None:
        return self._compute_metrics

    @property
    def preprocess_logits_for_metrics(self) -> Callable | None:
        return self._preprocess_logits

    @property
    def label_names(self) -> list[str]:
        return self._label_names

    @property
    def callback_handler(self) -> Any:
        return self._callback_handler

    @property
    def control(self) -> TrainerControl:
        return self._control

    @control.setter
    def control(self, value: TrainerControl) -> None:
        self._control = value

    @property
    def train_dataset(self) -> Any:
        """Public alias for ``_train_dataset`` (required by ``TrainingSummary.from_trainer``)."""
        return self._train_dataset

    @property
    def eval_dataset(self) -> Any:
        """Public alias for ``_eval_dataset`` (required by ``TrainingSummary.from_trainer``)."""
        return self._eval_dataset

    # ------------------------------------------------------------------
    # Public Trainer contract helpers
    # ------------------------------------------------------------------

    def add_callback(self, callback: type[TrainerCallback] | TrainerCallback) -> None:
        """Add a callback to the trainer's live callback handler.

        Also appended to ``self._base_callbacks`` so subsequent state
        resets (e.g. ``_reset_state_for_batch_size_retry``) rebuild the
        handler with this callback included.
        """
        self._callback_handler.add_callback(callback)
        self._base_callbacks.append(callback)

    def remove_callback(
        self,
        callback: type[TrainerCallback] | TrainerCallback,
    ) -> None:
        """Remove the first matching callback from the active handler."""
        self._callback_handler.remove_callback(callback)
        self._base_callbacks = [
            cb for cb in self._base_callbacks if not _callback_matches(cb, callback)
        ]

    def pop_callback(
        self,
        callback: type[TrainerCallback] | TrainerCallback,
    ) -> TrainerCallback | None:
        """Pop and return the first matching callback from the active handler."""
        popped = self._callback_handler.pop_callback(callback)
        self._base_callbacks = [
            cb for cb in self._base_callbacks if not _callback_matches(cb, callback)
        ]
        return popped

    def is_local_process_zero(self) -> bool:
        """Whether this is the local-rank-0 process (HF parity)."""
        return self._ddp.is_local_zero

    def is_world_process_zero(self) -> bool:
        """Whether this is the world-rank-0 process (HF parity)."""
        return self._ddp.is_world_zero

    def _stamp_ddp_flags(self, state: TrainerState) -> None:
        """Stamp per-rank ``is_*_process_zero`` flags onto ``state``.

        The flags are per-rank metadata (not part of the durable
        checkpoint contract), so any newly-constructed or
        freshly-deserialized ``TrainerState`` needs them set.
        """
        state.is_world_process_zero = self._ddp.is_world_zero
        state.is_local_process_zero = self._ddp.is_local_zero

    def _reset_state_for_new_run(self) -> None:
        self.state = TrainerState()
        self.state.max_steps = self._predict_total_steps()
        self.state.compute_steps(self.args)
        self._stamp_ddp_flags(self.state)
        self._control = TrainerControl()
        self._tr_loss = torch.tensor(0.0, device=self._device)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = 0
        self._train_start_time = None
        self._ctx = None
        self._train_dataloader = None
        self._eval_dataloader = None

    def _apply_opaque_model_patches(self) -> None:
        """``apply_model_patches(model, compat=use_compat_patches, performance=True, kernels=use_performance_kernels)``.

        ``use_compat_patches`` (default ``True``) gates vmap-safety
        patches: ``eager_attention``, ``batchify``, vmap-safe masking /
        collator / checkpoint hooks.  ``use_performance_kernels`` (default
        ``False``) gates the CUDA + Triton kernel group (``rope``,
        ``rms_norm``, ``activation``, ``cross_entropy``).  The
        ``performance`` bucket — ``kv_cache`` and the conditional fused-CE
        wrapper — is always enabled here because ``DynamicCache`` allocation leaks vmap refs
        and inflates training memory regardless of host;
        ``performance_kernels_config={"kv_cache": False}`` opts out.

        An unregistered Hugging Face model family raises a configuration
        error because its vmap-safety patches cannot be applied. Set
        ``use_compat_patches=False`` only when supplying a compatible
        custom model. Non-HF ``nn.Module`` models log a warning and remain
        supported.
        """
        self._fused_forward_uses_marker = False
        try:
            from opaque.patches import apply_model_patches
        except ImportError:
            log.debug("opaque.patches unavailable; skipping model patches.")
            return

        kwargs = dict(self.args.performance_kernels_config or {})
        apply_model_patches(
            self._model,
            compat=bool(self.args.use_compat_patches),
            performance=True,
            kernels=bool(self.args.use_performance_kernels),
            **kwargs,
        )

        def accepts_marker(module: Any, *, allow_var_kwargs: bool) -> bool:
            try:
                parameters = inspect.signature(module.forward).parameters.values()
            except (TypeError, ValueError):
                return False
            return any(
                parameter.name == "loss_only"
                or (
                    allow_var_kwargs and parameter.kind is inspect.Parameter.VAR_KEYWORD
                )
                for parameter in parameters
            )

        # A PEFT wrapper commonly accepts **kwargs while only its nested causal-LM
        # module carries the actual fused forward. Require both facts so an
        # unsupported custom model never receives an unknown marker.
        self._fused_forward_uses_marker = bool(
            kwargs.get("fused_linear_cross_entropy") is not False
            and accepts_marker(self._model, allow_var_kwargs=True)
            and any(
                accepts_marker(module, allow_var_kwargs=False)
                for module in self._model.modules()
            )
        )

    def _setup_precision(self) -> None:
        """Resolve compute precision (TF32, bf16 autocast).

        ``bf16=True`` enables autocast on the loss closure (forward); it
        does NOT cast the model.  Full-cast is reserved for
        ``bf16_full_eval`` (eval scope only — see :mod:`._precision`).
        bf16 is the only mixed-precision mode: it has the dynamic range
        that fp16's dynamic loss scaler exists to compensate for, so no
        scaler is needed (and fp16 training is intentionally unsupported).

        Sets:
            self._device — already resolved by caller.
            self._train_dtype — dtype the model parameters are stored in.
                Stays at whatever the caller pre-placed; autocast does
                NOT change it.
            self._amp_dtype — None | torch.bfloat16.  Driven into
                ``torch.autocast(device_type, dtype=self._amp_dtype)``
                inside the loss closure when set.
        """
        a = self.args
        # ``tf32`` is a single global flag flip.  HF semantics: ``None`` =
        # leave alone; explicit ``True``/``False`` flips both flags.  No
        # restore on shutdown (matches HF).
        if a.tf32 is not None and self._device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = bool(a.tf32)
            torch.backends.cudnn.allow_tf32 = bool(a.tf32)

        self._train_dtype = next(self._model.parameters()).dtype
        self._amp_dtype = torch.bfloat16 if a.bf16 else None
        self._workaround_mps_bf16_sdpa()

    def _workaround_mps_bf16_sdpa(self) -> None:
        """Force eager attention for bf16 DP on MPS (PyTorch SDPA/autocast bug).

        On MPS, ``torch.autocast`` does not cast ``scaled_dot_product_attention``
        under functorch ``vmap(grad)`` (it does on CPU/CUDA), so the bf16 DP step
        crashes in attention.  Eager attention sidesteps it **losslessly** —
        under ``vmap`` SDPA already decomposes to the same matmuls.  Probe-gated
        (:func:`opaque.device.sdpa_autocast_under_vmap_broken`) so this auto-drops
        on a torch that fixes the upstream bug.  See the minimal repro in that probe.
        """
        if (
            self._amp_dtype is None
            or self._device.type != "mps"
            or not sdpa_autocast_under_vmap_broken(self._device.type)
        ):
            return
        model = self._model
        cfg = getattr(model, "config", None)
        if getattr(cfg, "_attn_implementation", None) == "eager":
            return
        setter = getattr(model, "set_attn_implementation", None)
        if not callable(setter):
            log.warning(
                "bf16 DP on MPS hits a PyTorch bug (autocast doesn't cast SDPA "
                "under vmap(grad) on MPS) and %s exposes no "
                "set_attn_implementation; load the model with "
                "attn_implementation='eager' to run bf16 on MPS.",
                type(model).__name__,
            )
            return
        try:
            setter("eager")
        except Exception as e:
            log.warning(
                "bf16 DP on MPS: could not switch %s to eager attention "
                "(%s: %s); load it with attn_implementation='eager'.",
                type(model).__name__,
                type(e).__name__,
                e,
            )
            return
        log.warning(
            "bf16 DP on MPS: switched the model to eager attention.  PyTorch's "
            "autocast does not cast scaled_dot_product_attention under vmap(grad) "
            "on MPS (works on CPU/CUDA), which crashes the bf16 DP step; eager "
            "attention avoids it losslessly (SDPA decomposes to the same matmuls "
            "under vmap).  Pass attn_implementation='eager' to silence this, or "
            "use fp32 / CUDA.",
        )

    def _effective_output_dir(self) -> str | None:
        return self.args.output_dir

    # ------------------------------------------------------------------
    # train() → _setup_training() → _inner_training_loop()
    # ------------------------------------------------------------------

    def train(
        self,
        resume_from_checkpoint: str | bool | os.PathLike[str] | None = None,
        ignore_keys_for_eval: list[str] | None = None,
    ) -> TrainOutput:
        """Run the full DP-SGD training loop.

        Args:
            resume_from_checkpoint: ``None`` falls back to
                ``args.resume_from_checkpoint``. ``True`` auto-detects the latest
                ``checkpoint-*`` under ``args.output_dir``. A string or
                ``PathLike`` is treated as the concrete checkpoint directory.
            ignore_keys_for_eval: Model-output keys to omit while evaluating
                during training.

        Resume semantics under DP differ from HF's batch-replay model:

        - **Sampler resume restores the cursor, not the data order.**
          HF's ``Trainer`` rebuilds the dataloader and skips
          ``global_step`` batches one by one to recover the exact data
          order; we instead restore the sampler's ``consumed`` cursor.
          For the Poisson sampler this is done by advancing its NumPy
          generator forward by ``consumed`` discarded sampling steps
          (an ``O(consumed)`` RNG replay — cheap relative to training,
          but *not* an ``O(1)`` jump), so the continuation is a
          deterministic resume of the saved stream.  **Privacy budget is
          unchanged** — every iteration still consumes one
          Poisson-amplified Gaussian step and the accountant composes the
          same number of mechanisms — and the resumed subsample sequence
          from iteration N onward matches a continuous run from the same
          seed.  DP-valid either way.
        - **``ignore_data_skip=True``** skips sampler-state restore.
          Poisson resumes use a distinct stream; participation samplers
          require the saved cursor.
        - **Accountant on resume** preserves the mechanism lifecycle:
          independent releases load the saved ``Accountant`` as a prefix and
          calibrate remaining steps against it; whole-horizon mechanisms retain
          their single declared process. For independent releases, changing
          ``privacy_noise_multiplier`` / ``privacy_target_epsilon`` warns and
          composes the requested continuation onto the saved prefix. Whole-horizon
          mechanisms reject privacy-relevant drift because their saved process
          and schedule define one indivisible release.
        - **Stop-at-epsilon is enforced after every accounted step.** With a fixed
          positive noise multiplier and target epsilon, the crossing step is
          predicted before the loop and checked independently of logging cadence.
          A resumed run whose saved prefix has already spent the budget exits
          before fetching another batch.
        """
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None
        _disable_tokenizers_parallelism_before_fork()

        self.is_in_train = True
        try:
            return self._train_dispatch(resume_from_checkpoint, ignore_keys_for_eval)
        finally:
            self.is_in_train = False

    def _train_dispatch(
        self,
        resume_from_checkpoint: str | bool | os.PathLike[str] | None,
        ignore_keys_for_eval: list[str] | None,
    ) -> TrainOutput:
        """Inner dispatch."""
        if self._train_dataset is None:
            raise ConfigurationError(*("Trainer.train() requires a train_dataset.",))

        # ``microbatch_size`` controls the vmap chunk and defaults to
        # ``per_device_train_batch_size`` (one chunk per rank).
        # ``auto_find_microbatch_size`` then halves from this value on
        # CUDA-OOM.
        effective_microbatch_size = max(
            1,
            int(
                self.args.microbatch_size
                if self.args.microbatch_size is not None
                else self.args.per_device_train_batch_size
            ),
        )

        if not self.args.auto_find_microbatch_size:
            self.state.converged_microbatch_size = effective_microbatch_size
            return self._train_once(
                resume_from_checkpoint=resume_from_checkpoint,
                microbatch_size_override=effective_microbatch_size,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )

        initial_microbatch_size = effective_microbatch_size
        current_microbatch_size = initial_microbatch_size
        state_snapshot = TrainerState.from_json(self.state.to_json())
        model_snapshot = {
            k: v.detach().to("cpu").clone() for k, v in self._model.state_dict().items()
        }
        # ``load_state_dict`` restores tensor VALUES but not the
        # ``requires_grad`` partition.  An OOM raised mid-attempt (e.g.
        # inside the vmapped functional forward) can leave the trainable
        # (LoRA) params frozen; without re-asserting the flags the next
        # attempt's ``make_functional`` captures a different trainable set
        # and the post-training ``_restore_params`` guard fails with
        # "keys do not match the model's current requires_grad set".
        requires_grad_snapshot = {
            name: p.requires_grad for name, p in self._model.named_parameters()
        }
        rng_snapshot = ckpt.snapshot_rng_state()

        # Under DDP the OOM retry decision MUST be a cluster-wide collective.
        # Each rank's OOM is triggered by per-rank memory fragmentation at a
        # non-deterministic physical step, so if ranks halved the microbatch
        # independently they would fall out of step-lockstep: one rank restarts
        # at microbatch=4 step 0 while a sibling is still on microbatch=8 step 3.
        # DP-SGD's per-step collectives (``sum_gradients_`` AllReduce + the
        # ``ClippedPytree.max_norm`` cross-rank equality assert) then meet at
        # mismatched logical steps and raise ``max_norm mismatch across ranks``.
        # Fix: after every attempt, ``_cluster_needs_step_down`` all-reduces a
        # MAX of each rank's "needs to step down" flag — if ANY rank OOMs, EVERY
        # rank steps down together and restarts in lockstep. The returned run is
        # the first attempt at which no rank OOMs, so all ranks ran it at an
        # identical microbatch and stayed synchronised end-to-end.
        while True:
            # Stamp before the attempt so a successful run's logs carry it.
            self.state.converged_microbatch_size = current_microbatch_size
            local_oom = False
            local_oom_error: BaseException | None = None
            result = None
            try:
                result = self._train_once(
                    resume_from_checkpoint=resume_from_checkpoint,
                    microbatch_size_override=current_microbatch_size,
                    ignore_keys_for_eval=ignore_keys_for_eval,
                )
            except RuntimeError as err:
                if not self._is_retryable_oom(err):
                    raise
                local_oom = True
                local_oom_error = err

            # Cluster-wide retry decision: a rank that succeeded must still
            # step down (and discard ``result``) if any sibling OOM'd, so the
            # whole cluster re-runs the next attempt in lockstep.
            if self._cluster_needs_step_down(local_oom):
                if current_microbatch_size <= 1:
                    # Propagate the original OOM so callers see the actionable
                    # signal. Fall back to a synthetic message only when this
                    # rank didn't OOM locally (sibling-OOM-at-floor case).
                    if local_oom_error is not None:
                        raise local_oom_error
                    raise OperationError(
                        *(
                            "auto_find_microbatch_size exhausted: a sibling rank "
                            "still OOMs at microbatch_size=1. Reduce "
                            "per_device_train_batch_size (the logical Poisson batch) "
                            "or the model/sequence length.",
                        )
                    )
                next_microbatch_size = max(1, current_microbatch_size // 2)
                if next_microbatch_size == current_microbatch_size:
                    raise OperationError(
                        *(
                            "auto_find_microbatch_size cannot step down below "
                            f"microbatch_size={current_microbatch_size}.",
                        )
                    )
                log.warning(
                    "auto_find_microbatch_size: cluster OOM at microbatch_size=%d "
                    "(local_oom=%s), retrying all ranks with microbatch_size=%d",
                    current_microbatch_size,
                    local_oom,
                    next_microbatch_size,
                )
                self._model.load_state_dict(model_snapshot, strict=False)
                # Re-assert the trainable/frozen partition the OOM may have
                # clobbered, so the next attempt rebuilds the same
                # functional param set (see requires_grad_snapshot above).
                for name, p in self._model.named_parameters():
                    if name in requires_grad_snapshot:
                        p.requires_grad_(requires_grad_snapshot[name])
                ckpt.restore_rng_state(rng_snapshot)
                self._reset_state_for_batch_size_retry(state_snapshot)
                self._empty_device_cache_for_retry()
                current_microbatch_size = next_microbatch_size
                continue

            # No rank OOM'd at this size — every rank ran the identical
            # microbatch in lockstep, so ``result`` is valid on all ranks.
            assert result is not None  # local_oom is False here on every rank
            if current_microbatch_size != initial_microbatch_size:
                log.info(
                    "auto_find_microbatch_size: converged at "
                    "microbatch_size=%d (started at %d)",
                    current_microbatch_size,
                    initial_microbatch_size,
                )
            return result

    def _train_once(
        self,
        *,
        resume_from_checkpoint: str | bool | os.PathLike[str] | None,
        microbatch_size_override: int | None,
        ignore_keys_for_eval: list[str] | None,
    ) -> TrainOutput:
        if resume_from_checkpoint is None:
            resume_from_checkpoint = self.args.resume_from_checkpoint
        resume_path = self._resolve_resume_path(resume_from_checkpoint)

        # Pre-load weights so make_functional starts from the saved values.
        prefix_accountant: Accountant | None = None
        runtime_payload: ckpt.RuntimeCheckpoint | None = None
        trainer_state_json: dict[str, Any] | None = None
        if resume_path is not None:
            self._load_model_weights(resume_path)
            runtime_payload, prefix_accountant = self._read_runtime_for_resume(
                resume_path
            )
            self._validate_horizon_resume_calibration(runtime_payload)
            trainer_state_json = self._read_trainer_state(resume_path)
            if trainer_state_json is not None:
                self.state = TrainerState.from_json(trainer_state_json)
                self._stamp_ddp_flags(self.state)
                # Re-bind callback handler to the new state object.
                self._callback_handler.state = self.state

        ctx = self._setup_training(
            prefix_accountant=prefix_accountant,
            resume_runtime=runtime_payload,
            global_step_already_done=(
                self.state.global_step if resume_path is not None else 0
            ),
            microbatch_size_override=microbatch_size_override,
            sampler_restart_step=(
                self.state.global_step
                if resume_path is not None and self.args.ignore_data_skip
                else None
            ),
        )
        self._ctx = ctx

        if resume_path is not None:
            # ``_read_runtime_for_resume`` guarantees a complete payload
            # (dp_state + optimizer + accountant) or raises — weights-only
            # exports are rejected there — so resume always restores the full
            # DP runtime, never a partial one.
            self._apply_runtime_state(
                ctx, runtime_payload, prefix_accountant, resume_path
            )
            self._warn_on_arg_drift(runtime_payload)
            self._load_rng_state(resume_path)
            self._load_callback_states()
            # Stop-at-ε on resume: if the restored accountant already
            # exceeds target, skip the training loop.
            a = self.args
            if (
                not ctx.is_horizon_process
                and a.privacy_noise_multiplier is not None
                and a.privacy_noise_multiplier > 0
                and a.privacy_target_epsilon is not None
            ):
                ctx.accounting = acc.cached(ctx.accounting)
                resumed_eps = ctx.accounting.epsilon_at(ctx.target_delta)
                if resumed_eps >= a.privacy_target_epsilon:
                    self.state.privacy_target_epsilon_reached = True
                    log.info(
                        "stop-at-ε hit on resume: ε=%g >= target=%g; "
                        "skipping training loop",
                        resumed_eps,
                        a.privacy_target_epsilon,
                    )
                    return TrainOutput(
                        self.state.global_step,
                        0.0,
                        {
                            "privacy_epsilon": resumed_eps,
                            "privacy_delta": ctx.target_delta,
                        },
                    )

        # Predict the stop-at-ε step once at setup (accounting is
        # deterministic): the crossing step is binary-searchable up front, so
        # the in-loop check becomes a free integer comparison (#392). This is
        # only defined for independently composed step mechanisms.
        a = self.args
        if (
            not ctx.is_horizon_process
            and a.privacy_noise_multiplier is not None
            and a.privacy_noise_multiplier > 0
            and a.privacy_target_epsilon is not None
            and a.sampling_mode not in ("b_min_sep", "balls_in_bins")
            and self.state.global_step < ctx.total_steps
        ):
            ctx.stop_at_step = predict_stop_step(
                ctx.accounting.process,
                ctx.step_process,
                target_epsilon=a.privacy_target_epsilon,
                delta=ctx.target_delta,
                k0=self.state.global_step,
                horizon=ctx.total_steps,
            )
            if ctx.stop_at_step is not None:
                log.info(
                    "stop-at-ε: will stop after step %d of %d (target ε=%g at δ=%.2e)",
                    ctx.stop_at_step,
                    ctx.total_steps,
                    a.privacy_target_epsilon,
                    ctx.target_delta,
                )

        saved_sampler_state = (
            runtime_payload.sampler_state if runtime_payload is not None else None
        )
        # Older snapshots may include prefetched, untrained draws.
        # Normalize before deserialization replays the saved RNG stream.
        if (
            not a.ignore_data_skip
            and saved_sampler_state is not None
            and "consumed" in saved_sampler_state
            and int(saved_sampler_state["consumed"]) > self.state.global_step
        ):
            log.warning(
                "Clamping checkpoint sampler consumed=%s to global_step=%d "
                "before restoring the sampler.",
                saved_sampler_state["consumed"],
                self.state.global_step,
            )
            saved_sampler_state = {
                **saved_sampler_state,
                "consumed": self.state.global_step,
            }

        try:
            return self._inner_training_loop(
                ctx,
                resume_path=resume_path,
                saved_sampler_state=saved_sampler_state,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )
        finally:
            self._restore_params(self._published_params(ctx))
            # Promote accountant to trainer-level so save_model() can write
            # ``accountant.json`` after train() returns.
            self._accountant = ctx.accounting
            self._ctx = None
            self._train_dataloader = None
            self._eval_dataloader = None

    def _is_retryable_oom(self, err: RuntimeError) -> bool:
        """``True`` for an OOM that justifies a microbatch retry.

        ``torch.OutOfMemoryError`` (torch >= 2.4) is the typed exception
        emitted by both CUDA and MPS backends; ``isinstance`` is robust
        across torch versions and avoids matching unrelated runtime
        errors that happen to mention "out of memory" in the message.
        """
        return isinstance(err, torch.OutOfMemoryError)

    def _synchronize_grad_failure(self, error: Exception | None) -> None:
        """Raise rank-symmetrically before gradient collectives after a failure."""
        if not self._ddp.is_distributed:
            if error is not None:
                raise error
            return

        local_oom = isinstance(error, RuntimeError) and self._is_retryable_oom(error)
        flags = torch.tensor(
            [
                1.0 if local_oom else 0.0,
                1.0 if error is not None and not local_oom else 0.0,
            ],
            device=self._device,
        )
        torch.distributed.all_reduce(flags, op=torch.distributed.ReduceOp.MAX)
        if flags[1].item() > 0.0:
            if error is not None and not local_oom:
                raise error
            raise RuntimeError(  # noqa: TRY003 - preserve rank symmetry
                "collective gradient computation failed on a sibling rank"
            )
        if flags[0].item() > 0.0:
            raise torch.OutOfMemoryError(  # noqa: TRY003 - preserve PyTorch OOM type
                "collective microbatch retry (a rank OOM'd in grad_fn; "
                "whole cluster steps down to a smaller microbatch)."
            )

    def _cluster_needs_step_down(self, local_oom: bool) -> bool:
        """Whether any rank OOM'd this attempt (cluster-wide MAX all-reduce).

        The OOM-retry decision must be collective: if one rank steps the batch
        down and a sibling doesn't, they desync on the next collective. Returns
        ``local_oom`` unchanged when not distributed.
        """
        if not self._ddp.is_distributed:
            return local_oom
        flag = torch.tensor([1.0 if local_oom else 0.0], device=self._device)
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
        return bool(flag.item() > 0.0)

    def _reset_state_for_batch_size_retry(self, snapshot: TrainerState) -> None:
        self.state = TrainerState.from_json(snapshot.to_json())
        self._stamp_ddp_flags(self.state)
        self._callback_handler.state = self.state
        self._control = TrainerControl()
        self._ctx = None
        self._train_dataloader = None
        self._eval_dataloader = None
        self._tr_loss = torch.tensor(0.0, device=self._device)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = int(self.state.global_step)

    def _empty_device_cache_for_retry(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        elif self._device.type == "mps":
            torch.mps.empty_cache()

    def _set_resolved_privacy_args(
        self,
        *,
        target_delta: float,
        noise_multiplier: float,
        calibration_source: str,
        sample_rate: float,
        expected_batch_size: int,
        total_steps: int,
    ) -> None:
        """Expose run-resolved privacy constants for reporting callbacks."""
        self.state.privacy_resolved_delta = float(target_delta)
        self.state.privacy_resolved_noise_multiplier = float(noise_multiplier)
        self.state.privacy_calibration_source = calibration_source
        self.state.privacy_sample_rate = float(sample_rate)
        self.state.privacy_expected_batch_size = int(expected_batch_size)
        self.state.privacy_total_steps = int(total_steps)

    def _setup_training(
        self,
        *,
        prefix_accountant: Accountant | None = None,
        resume_runtime: ckpt.RuntimeCheckpoint | None = None,
        global_step_already_done: int = 0,
        microbatch_size_override: int | None = None,
        sampler_restart_step: int | None = None,
    ) -> _TrainingContext:
        """Functional conversion, clipping, calibration, optimizer.

        On resume, independent mechanisms calibrate remaining steps with the
        saved accountant composed on the left. Whole-horizon mechanisms
        recalibrate their complete declared process without composing a prefix.
        """
        a = self.args
        # --- Gradient checkpointing ---
        if a.gradient_checkpointing:
            gc_kwargs = a.gradient_checkpointing_kwargs or {"use_reentrant": False}
            self._model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gc_kwargs,
            )
            log.info("Gradient checkpointing: enabled")

        # --- CPU offload context ---
        offload_ctx: Any = contextlib.nullcontext()
        if a.activation_offloading:
            # ``pin_memory=False`` is forced: ``cpu_offload`` exists to
            # extend batches past the GPU ceiling, and pinning host RAM
            # would re-cap that expansion at the host limit (host-OOM is
            # an uncatchable SIGKILL that ``auto_find_microbatch_size``
            # cannot recover from).  Pageable host RAM is slower per
            # transfer but the OS can swap.
            offload_ctx = torch.autograd.graph.save_on_cpu(pin_memory=False)
            log.info("CPU offload: enabled")

        # --- Functional conversion ---
        log.info("Converting model to functional form...")
        fmodel, trainable_params, frozen_params = make_functional(
            self._model,
            disable_autograd_tracking=True,
            partition_trainable=True,
        )
        log.info("Trainable parameters: %d tensors", len(trainable_params))

        # --- Discover batch shape via a one-example collator dry run ---
        # The collator's tensor-valued output keys define the per-
        # example loss surface.  Non-tensor metadata (string IDs, raw
        # text) is allowed in the batch but excluded from the vmapped
        # loss closure.
        batch_keys = self._discover_batch_keys()

        # --- Build the per-example loss closure ---
        # Default behaviour: forward through ``fmodel`` and read
        # ``output["loss"]``, which transparently inherits HF's
        # ``LOSS_MAPPING`` dispatch (causal-LM gets ``ForCausalLMLoss``,
        # classification gets ``ForSequenceClassificationLoss``, …).
        # Subclasses override :meth:`compute_per_example_loss` for
        # domain-specific losses; ``_build_per_example_loss`` adapts that
        # method to the functional per-example calling convention.
        # Subclasses that override ``compute_per_example_loss_and_metrics`` emit
        # per-example telemetry; the loss closure then returns ``(loss, aux)`` and
        # the grad fn is built with ``has_aux=True``. Detected by override (no
        # flag); ``False`` keeps the standard path bit-identical.
        wants_metrics = self._overrides_metrics_seam()
        per_example_loss_fn, batch_argnums = self._build_per_example_loss(
            fmodel,
            frozen_params,
            batch_keys,
            with_metrics=wants_metrics,
        )

        # --- Sampling & step calculations ---
        # Logical batch (Poisson round size) drives DP accounting; physical
        # batch (per-device size) is the vmap chunk fed into clipping.
        expected_batch_size = a.train_batch_size
        microbatch_size = (
            int(microbatch_size_override)
            if microbatch_size_override is not None
            else a.per_device_train_batch_size
        )

        if self._train_dataset is None:
            raise ConfigurationError(*("Trainer.train() requires a train_dataset.",))
        dataset_size = self._effective_train_dataset_size()
        if dataset_size <= 0:
            raise ConfigurationError(
                *(
                    "Trainer requires a non-empty train_dataset: DP-SGD needs "
                    "at least one example to build the per-example loss surface "
                    "and calibrate Poisson sampling.",
                )
            )
        # Rank-local sample rate.  The user's ``expected_batch_size`` is the
        # *global* (cluster-wide) expected Poisson round size.  Under DDP we
        # shard the dataset (``opaque.distributed.local_shard``) and run the
        # Poisson sampler on each shard with the same epoch-folded key, so
        # the *local* rate equals the *global* rate by construction.  The
        # accountant uses regular ``acc.poisson`` over the global rate.
        #
        # ``dataset_size`` is ``len(train_dataset)`` (or the trimmed
        # multiple of ``world_size`` under DDP; see
        # :meth:`_effective_train_dataset_size`). Computing ``sample_rate``
        # from this denominator means the accountant calibrates noise for
        # exactly the ``q`` the sampler will use — there is no "actual q vs
        # accounted q" drift.
        sample_rate = expected_batch_size / dataset_size
        if sample_rate > 1.0:
            raise ConfigurationError(
                *(
                    "Trainer requires expected_batch_size <= len(train_dataset) "
                    "for Poisson sampling; got expected_batch_size="
                    f"{expected_batch_size} and len(train_dataset)={dataset_size}.",
                )
            )
        expected_steps_per_epoch, total_steps, num_epochs = self._steps_breakdown(
            dataset_size
        )

        self.state.max_steps = total_steps
        # HF-parity bookkeeping for ``trainer_state.json``.
        self.state.train_batch_size = int(microbatch_size)
        self.state.num_train_epochs = float(num_epochs)
        # Resolve fractional ``logging_steps`` / ``eval_steps`` /
        # ``save_steps`` to absolute step counts on ``state`` so
        # ``DefaultFlowCallback`` can read them.  Already called in
        # ``__init__`` with the same ``total_steps``, but repeated here
        # so subclasses overriding ``_predict_total_steps`` to no-op
        # still get a populated cadence by ``train()`` time.
        self.state.compute_steps(a)

        # Resolve save_steps from a fraction now that total_steps is known.
        save_steps_resolved = self._resolve_save_steps_int(total_steps)
        self.state.save_steps = save_steps_resolved

        # --- Clipping norm (scalar ``clipping_norm`` or per-group dict) ---
        mgn = a.clipping_norm
        if isinstance(mgn, dict):
            from opaque.api.engine.clipping import per_group as per_group_clipper

            fb = float(mgn["fallback"])
            patterns = {k: float(v) for k, v in mgn.items() if k != "fallback"}
            if not patterns:
                clip_norm: Any = fb
            else:
                clip_norm = per_group_clipper(
                    trainable_params,
                    fallback=fb,
                    **patterns,
                )
                log.info("Per-group clipping: %d groups", len(clip_norm.values))
        else:
            clip_norm = float(mgn)

        # AdaClip releases both a noisy clipping-rate estimate and noisy
        # gradients.  They must use independent streams for the composed
        # mechanism; reusing the root key makes both step-t streams identical.
        # Keep non-adaptive seeding unchanged for reproducibility.
        quantile_noise_key = gradient_noise_key = key(a.seed)
        if a.clipping_mode == "adaptive":
            quantile_noise_key, gradient_noise_key = split(gradient_noise_key)

        # --- Clipping ---
        grad_fn, clip_state = self._create_grad_fn(
            per_example_loss_fn,
            batch_argnums,
            a,
            clip_norm,
            expected_batch_size,
            microbatch_size,
            quantile_noise_key=quantile_noise_key,
            has_aux=wants_metrics,
        )

        # --- LR schedule ---
        # Built early so MF strategies (BandMF / BLT) can consume it for
        # workload-aware Toeplitz tuning.  The schedule only depends on
        # ``total_steps``, so moving it ahead of ``_build_mechanism`` is
        # a no-op for DP-SGD.
        lr_schedule = self.create_scheduler(num_training_steps=total_steps)

        # --- MF strategy (DP-FTRL only) ---
        mechanism_kind = a.privacy_noise_mechanism
        mf: _dpftrl.MFContext | None = None
        if mechanism_kind != "gaussian":
            mf_strategy = _dpftrl.build_strategy(
                mechanism_kind,
                (
                    a.privacy_noise_mechanism_kwargs
                    if isinstance(a.privacy_noise_mechanism_kwargs, dict)
                    else None
                ),
                lr_schedule=lr_schedule,
            )
            sk = a.sampling_kwargs if isinstance(a.sampling_kwargs, dict) else {}
            tb_raw = sk.get("truncated_batch_size")
            mf_amplifier_factory = _dpftrl.build_amplifier_factory(
                sampling_mode=a.sampling_mode,
                strategy=mf_strategy,
                sample_rate=sample_rate,
                n_steps=total_steps,
                num_bins=expected_steps_per_epoch,
                dataset_size=dataset_size,
                truncated_batch_size=int(tb_raw) if tb_raw is not None else None,
                world_size=self._ddp.world_size,
            )
            mf = _dpftrl.MFContext(
                strategy=mf_strategy, amplifier_factory=mf_amplifier_factory
            )

        # --- Privacy calibration ---
        target_delta = (
            a.privacy_target_delta
            if a.privacy_target_delta is not None
            else _default_privacy_delta(dataset_size)
        )
        mechanism = self._build_mechanism(
            a,
            expected_batch_size,
            sample_rate,
            clip_norm,
            dataset_size,
            n_steps=total_steps,
            num_bins=expected_steps_per_epoch,
            mf_amplifier_factory=mf.amplifier_factory if mf is not None else None,
        )
        noise_multiplier = self._calibrate_noise(
            a,
            mechanism,
            total_steps,
            target_delta,
            prefix_accountant=prefix_accountant,
            resume_horizon_noise_multiplier=(
                resume_runtime.noise_multiplier
                if resume_runtime is not None and resume_runtime.is_horizon_process
                else None
            ),
            global_step_already_done=global_step_already_done,
        )
        calibration_source = (
            "fixed" if a.privacy_noise_multiplier is not None else "calibrated"
        )
        self._set_resolved_privacy_args(
            target_delta=target_delta,
            noise_multiplier=noise_multiplier,
            calibration_source=calibration_source,
            sample_rate=sample_rate,
            expected_batch_size=expected_batch_size,
            total_steps=total_steps,
        )
        _sk = a.sampling_kwargs if isinstance(a.sampling_kwargs, dict) else {}
        _trunc_cap = _sk.get("truncated_batch_size")
        log.info(
            "Resolved privacy config: delta=%.2e, noise_multiplier=%.4f (%s), "
            "sample_rate=%.6f, total_steps=%d, truncated_batch_size=%s",
            target_delta,
            noise_multiplier,
            calibration_source,
            sample_rate,
            total_steps,
            # None ⇒ unbounded Poisson PLD; int ⇒ truncated_poisson_gaussian_pld.
            int(_trunc_cap) if _trunc_cap is not None else None,
        )

        # --- Optimizer ---
        opt, opt_state = self.create_optimizer(
            trainable_params,
            lr_schedule,
            clip_state,
            noise_multiplier,
        )
        # --- Noise ---
        # Sensitivity flows through the ``ClippedPytree`` returned by
        # ``clipped_grad`` (its ``.max_norm`` field).  ``noise_fn`` reads
        # that wrapper at call time and emits a ``NoisedPytree`` whose
        # ``.noise_stddev`` carries the realized σ for downstream
        # consumers (e.g. opaque optimizers' DP bias correction).
        if mechanism_kind == "gaussian":
            _gn_extra: dict[str, Any] = {
                _k: _v
                for _k, _v in (
                    a.privacy_noise_mechanism_kwargs.items()
                    if isinstance(a.privacy_noise_mechanism_kwargs, dict)
                    else ()
                )
                if _k == "compute_dtype"
            }
            noise_fn, noise_state = gaussian_noise(
                noise_multiplier=noise_multiplier,
                key=gradient_noise_key,
                **_gn_extra,
            )
        else:
            # DP-FTRL: pull the participation context (``n_steps`` /
            # ``min_sep`` / ``max_participations``) off the raw amplifier so
            # the streaming noise matrix tracks the calibrated PLD exactly.
            assert mf is not None
            _amp = mf.amplifier_factory(noise_multiplier)
            if int(_amp.n_steps) != total_steps:
                raise OperationError(
                    *(
                        "DP-FTRL amplifier horizon does not match the training "
                        f"horizon: {_amp.n_steps} != {total_steps}.",
                    )
                )
            noise_fn, noise_state = mf_gaussian_noise(
                trainable_params,
                mf.strategy,
                n_steps=int(_amp.n_steps),
                min_sep=int(_amp.min_sep),
                max_participations=int(_amp.max_participations),
                noise_multiplier=noise_multiplier,
                key=key(a.seed),
            )

        # --- Collate ---
        # Same wrapper used by the eval dataloader so train and eval
        # share key validation + device move (no asymmetric crash modes).
        collate_fn = self._resolve_collate_fn()

        accounting, step_process, horizon_process = _initialize_accounting(
            mechanism(noise_multiplier)
        )

        return _TrainingContext(
            fmodel=fmodel,
            trainable_params=trainable_params,
            frozen_params=frozen_params,
            grad_fn=grad_fn,
            clip_state=clip_state,
            noise_fn=noise_fn,
            noise_state=noise_state,
            noise_multiplier=noise_multiplier,
            opt=opt,
            opt_state=opt_state,
            lr_schedule=lr_schedule,
            accounting=accounting,
            mechanism=mechanism,
            step_process=step_process,
            target_delta=target_delta,
            sample_rate=sample_rate,
            calibration_source=calibration_source,
            expected_steps_per_epoch=expected_steps_per_epoch,
            total_steps=total_steps,
            num_epochs=num_epochs,
            collate_fn=collate_fn,
            batch_keys=batch_keys,
            offload_ctx=offload_ctx,
            opt_name=(
                self._functional_optimizer_name
                if self._functional_optimizer_factory is not None
                else a.optim
            ),
            sampler_restart_step=sampler_restart_step,
            save_steps_resolved=save_steps_resolved,
            clip_norm=clip_norm,
            mechanism_kind=mechanism_kind,
            is_horizon_process=horizon_process is not None,
            horizon_process=horizon_process,
            mf=mf,
        )

    def _inner_training_loop(
        self,
        ctx: _TrainingContext,
        *,
        resume_path: str | None = None,
        saved_sampler_state: dict[str, Any] | None = None,
        ignore_keys_for_eval: list[str] | None = None,
    ) -> TrainOutput:
        """Epoch/step loop with Poisson sampling."""
        a = self.args

        self._control = self._callback_handler.on_train_begin(
            self.args, self.state, self._control
        )

        # Emit setup-time constants once now so they land in W&B summary
        # (via _PRIVACY_SUMMARY_KEYS) for cross-run comparison while the run
        # is still live, instead of waiting for end-of-training.
        setup_constants: dict[str, Any] = {}
        if self.state.privacy_calibration_source is not None:
            setup_constants["privacy_calibration_source"] = (
                self.state.privacy_calibration_source
            )
        if self.state.privacy_calibration_noise_multiplier is not None:
            setup_constants["privacy_calibration_noise_multiplier"] = (
                self.state.privacy_calibration_noise_multiplier
            )
        if self.state.privacy_calibration_achieved_epsilon is not None:
            setup_constants["privacy_calibration_achieved_epsilon"] = (
                self.state.privacy_calibration_achieved_epsilon
            )
        if self.state.privacy_calibration_converged is not None:
            setup_constants["privacy_calibration_converged"] = (
                self.state.privacy_calibration_converged
            )
        if self.state.converged_microbatch_size is not None:
            setup_constants["converged_microbatch_size"] = (
                self.state.converged_microbatch_size
            )
        if setup_constants:
            self.log(setup_constants)

        log.info(
            "Starting DP-SGD training: %d epochs, ~%d steps/epoch, %d total",
            ctx.num_epochs,
            ctx.expected_steps_per_epoch,
            ctx.total_steps,
        )

        # On resume, pick up from the saved global_step/epoch.
        global_step = self.state.global_step if resume_path is not None else 0
        last_loss = 0.0
        last_step_result: dict[str, Any] = {}
        # HF parity: derive ``start_epoch`` from ``global_step``, not
        # ``state.epoch``.  ``state.epoch`` reaches an integer value at
        # epoch boundaries (e.g. step N of a 2-step epoch sets
        # ``state.epoch=1.0``); rounding it down would skip the epoch
        # we *just* finished.  ``global_step // steps_per_epoch`` is
        # what HF's ``Trainer._inner_training_loop`` uses
        # (``epochs_trained``).
        start_epoch = (
            global_step // max(1, ctx.expected_steps_per_epoch)
            if resume_path is not None
            else 0
        )
        # HF parity: anchor the smoothing window at the resume / start
        # boundary so the first post-resume log row averages over the
        # post-resume window only.
        self._globalstep_last_logged = global_step
        self._tr_loss, self._total_loss_scalar = (
            torch.tensor(0.0, device=self._device),
            0.0,
        )
        self._train_start_time = time.time()
        self._memory_tracker.start()
        if resume_path is not None:
            log.info(
                "Resuming from %s: epoch=%d, global_step=%d",
                resume_path,
                start_epoch,
                global_step,
            )

        # ``eval_on_start``: fires once before the inner loop runs.  HF
        # parity (``trainer.py`` ``_inner_training_loop``): the flag is
        # honored regardless of resume — users opting in want a baseline
        # eval at the *start* of each ``train()`` call, not only on
        # fresh runs.  Disable ``eval_on_start`` if you don't want it
        # on resume.
        if a.eval_on_start:
            self.evaluate(ignore_keys=ignore_keys_for_eval)

        # Build the train loader ONCE: a single
        # ``PoissonSampler(n_steps=total_steps)`` drives every
        # epoch; the outer loop's role is purely callback synthesis
        # (``on_epoch_begin`` / ``on_epoch_end``) and per-epoch break
        # handling.  Resume restores the sampler's ``consumed`` cursor
        # via the opaque.serialization registry; the restored sampler
        # is installed on ``ctx.current_sampler`` *before* loader
        # construction so ``DataLoader`` binds to it (the
        # ``batch_sampler`` attribute is immutable post-init).
        if (
            resume_path is not None
            and saved_sampler_state is not None
            and not a.ignore_data_skip
        ):
            from opaque.serialization import from_state_dict

            # Need a template sampler whose ``data_source`` matches the
            # saved length so ``from_state_dict`` can validate.  Build
            # one (without caching the loader yet), then replace it
            # with the restored cursor before the actual loader binds.
            if ctx.current_sampler is None:
                self._train_dataloader = None
                self.get_train_dataloader()  # populates ctx.current_sampler
                self._train_dataloader = None  # drop the cached loader
            ctx.current_sampler = from_state_dict(
                ctx.current_sampler, saved_sampler_state
            )

        train_loader = self.get_train_dataloader()
        # Snapshot the ``global_step`` at which ``consumed`` is 0, before
        # ``iter()`` can prefetch — lets a later checkpoint rebase the live
        # cursor back to an absolute step count (see ``ignore_data_skip``).
        ctx.sampler_step_offset = (
            global_step - ctx.current_sampler.consumed
            if ctx.current_sampler is not None
            else 0
        )
        train_loader_iter = iter(train_loader)

        for epoch in range(start_epoch, ctx.num_epochs):
            self.state.epoch = float(epoch)
            self._control = self._callback_handler.on_epoch_begin(
                self.args, self.state, self._control
            )
            if self._control.should_training_stop:
                break

            for step_idx in range(ctx.expected_steps_per_epoch):
                try:
                    batch = next(train_loader_iter)
                except StopIteration:
                    # Sampler exhausted before this epoch's quota — happens
                    # when ``total_steps`` doesn't divide evenly into the
                    # ``num_epochs × expected_steps_per_epoch`` budget, or
                    # on resume past the recorded ``total_steps``.
                    break
                batch = self._prepare_input(batch)
                batch_size = _eval.find_batch_size(batch) or 0

                self._control = self._callback_handler.on_step_begin(
                    self.args, self.state, self._control
                )

                # Independent mechanisms compose here; horizon contexts carry
                # no step process because their complete run was attached once.
                _account_independent_step(ctx)

                # Training step: clip → noise → optimize.  DP-SGD has no
                # substep concept; each iteration is a full optimizer step
                # over one Poisson-sampled logical batch.  ``on_substep_end``
                # is therefore not fired.
                step_result = self.training_step(self._model, batch)

                global_step += 1
                self.state.global_step = global_step
                # HF parity: ``state.epoch`` is fractional during the inner
                # loop — ``epoch + (step + 1) / steps_per_epoch``.  This is
                # what ``ProgressCallback``, W&B / TensorBoard timeline, and
                # any callback-driven cadence (``eval_delay``) read.
                self.state.epoch = float(epoch) + (
                    (step_idx + 1) / max(1, ctx.expected_steps_per_epoch)
                )

                # Empty Poisson round: no loss / tokens to accumulate.  The
                # optimizer still applied a pure-noise update and
                # ``global_step`` already advanced, so the log/save/eval gate
                # below must still run — a save or eval boundary can land
                # exactly on an empty round.  ``step_result`` carries only
                # ``{loss: 0, batch_size: 0}`` here, which the gate reads via
                # ``.get`` defaults (the logged loss is the windowed average,
                # unaffected by this step).
                if batch_size != 0:
                    last_loss = step_result["loss"]
                    last_step_result = step_result
                    # Loss accumulator stays on device for DDP gather.  A NaN
                    # reaching here reflects a genuine forward / loss-math
                    # divergence — propagate it through the running average so
                    # the user sees the honest signal instead of a
                    # smoothed-over fake curve.
                    tr_loss_step = torch.tensor(float(last_loss), device=self._device)
                    self._tr_loss = torch.add(self._tr_loss, tr_loss_step)
                # Token counting.
                if batch_size != 0 and a.include_num_input_tokens_seen != "no":
                    main_input_name = getattr(
                        self._model, "main_input_name", "input_ids"
                    )
                    if main_input_name in batch:
                        if a.include_num_input_tokens_seen == "non_padding":
                            if "attention_mask" in batch:
                                n_tokens = int(batch["attention_mask"].sum().item())
                            elif (
                                self._processing_class is not None
                                and hasattr(self._processing_class, "pad_token_id")
                                and self._processing_class.pad_token_id is not None
                            ):
                                n_tokens = int(
                                    (
                                        batch[main_input_name]
                                        != self._processing_class.pad_token_id
                                    )
                                    .sum()
                                    .item()
                                )
                            else:
                                log.warning(
                                    "include_num_input_tokens_seen='non_padding': "
                                    "no attention_mask and no pad_token_id on "
                                    "processing_class — falling back to all tokens."
                                )
                                n_tokens = batch[main_input_name].numel()
                        else:  # "all"
                            n_tokens = batch[main_input_name].numel()
                        # ``average_tokens_across_devices=True``
                        # (HF parity) sums the per-rank token count into the
                        # cluster-wide total so ``num_input_tokens_seen`` and
                        # the live tokens/sec rate reflect the whole DDP
                        # batch.  The flag default is True in HF; we respect
                        # whatever the user set on ``args``.
                        if self._ddp.is_distributed and getattr(
                            a, "average_tokens_across_devices", True
                        ):
                            from opaque.api.engine.distributed._state import (
                                reduce_scalar,
                            )

                            n_tokens = int(
                                reduce_scalar(
                                    n_tokens,
                                    op="sum",
                                    device=self._device,
                                )
                            )
                        self.state.num_input_tokens_seen += n_tokens
                # ``DefaultFlowCallback.on_step_end`` populates the
                # ``should_log/save/evaluate/training_stop`` flags from
                # ``state.{logging,eval,save}_steps``; user callbacks
                # registered after it can override.
                self._control = self._callback_handler.on_step_end(
                    self.args, self.state, self._control
                )
                self._maybe_log_save_evaluate(
                    ctx,
                    step_result,
                    global_step,
                    ignore_keys_for_eval=ignore_keys_for_eval,
                )

                # Stop-at-ε: free integer comparison against the predicted
                # crossing step — after the accounted step and its
                # log/save/eval gate, before the next batch fetch, so the
                # resumable sampler never advances past ``global_step`` and a
                # target reached exactly on the final step still sets the
                # flag (checked before the ``total_steps`` ceiling) (#392).
                if ctx.stop_at_step is not None and global_step >= ctx.stop_at_step:
                    self.state.privacy_target_epsilon_reached = True
                    self._control.should_training_stop = True
                    log.info(
                        "stop-at-ε: predicted budget boundary reached at step %d",
                        global_step,
                    )
                    break

                if self._control.should_training_stop:
                    break
                if self._control.should_epoch_stop:
                    break
                if 0 < a.max_steps <= global_step:
                    break
                # Hard ceiling at the calibrated horizon.  Noise was
                # calibrated for exactly ``ctx.total_steps`` composed
                # mechanisms; without this guard an epoch-driven resume
                # (``max_steps`` unset) with ``ignore_data_skip=True`` re-runs
                # the partial epoch from step 0 and overruns ``total_steps``,
                # spending more privacy budget than calibrated.
                if global_step >= ctx.total_steps:
                    break

            # Update state.epoch first so log_history rows are tagged correctly.
            self.state.epoch = float(epoch + 1)
            # ``DefaultFlowCallback.on_epoch_end`` populates
            # ``should_evaluate`` / ``should_save`` / ``should_log`` for
            # epoch-strategy cadence; ``_maybe_log_save_evaluate`` then acts
            # on the flags exactly as the step-end path does, so the
            # epoch-boundary log / eval / save sequence shares one call site.
            self._control = self._callback_handler.on_epoch_end(
                self.args, self.state, self._control
            )
            self._maybe_log_save_evaluate(
                ctx,
                last_step_result,
                global_step,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )

            if self._control.should_training_stop:
                break
            if 0 < a.max_steps <= global_step:
                break
            if global_step >= ctx.total_steps:  # calibrated-horizon ceiling
                break

        # Final save — parity with HF: when saving is enabled, the last step always
        # produces (or refreshes) a checkpoint, even if it doesn't align with save_steps.
        self._maybe_final_save(ctx, global_step)

        # Best-model rewind happens after the final save so the checkpoint dir exists.
        if a.load_best_model_at_end:
            self._load_best_model(ctx)

        # Final metrics
        final_epsilon = ctx.accounting.epsilon_at(ctx.target_delta)
        # HF parity: add any remaining tr_loss to the total before computing avg.
        # This ensures that even if logging_steps didn't align perfectly with the
        # final step, the total training loss includes all steps.
        self._total_loss_scalar += self._tr_loss.item()
        effective_global_step = max(global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step
        train_start = self._train_start_time or time.time()
        metrics: dict[str, Any] = speed_metrics(
            "train",
            train_start,
            num_samples=len(self._train_dataset) * ctx.num_epochs,
            num_steps=global_step,
            num_tokens=(
                self.state.num_input_tokens_seen
                if a.include_tokens_per_second
                else None
            ),
        )
        metrics.update(
            {
                "train_loss": train_loss,
                "train_steps": global_step,
                "privacy_epsilon": final_epsilon,
                "privacy_delta": ctx.target_delta,
                "privacy_noise_multiplier": ctx.noise_multiplier,
            }
        )
        if a.include_num_input_tokens_seen != "no":
            metrics["num_input_tokens_seen"] = self.state.num_input_tokens_seen
        self._memory_tracker.stop_and_update_metrics(metrics)

        log.info(
            "Training complete: %d steps, final loss=%.4f, epsilon=%.3f",
            global_step,
            train_loss,
            final_epsilon,
        )
        self.log(metrics, start_time=self._train_start_time)
        self._refresh_final_checkpoint_state(global_step)

        self._control = self._callback_handler.on_train_end(
            self.args, self.state, self._control
        )

        # Publish the finished model once, at the end of training (the only
        # auto-push point — no per-checkpoint uploads).
        if self.args.push_to_hub:
            _hub.push_to_hub(self, commit_message="End of training")

        return TrainOutput(global_step, train_loss, metrics)

    # ------------------------------------------------------------------
    # training_step() — single DP-SGD step
    # ------------------------------------------------------------------

    def training_step(
        self,
        model: Any,
        inputs: dict[str, Tensor],
    ) -> dict[str, Any]:
        """One DP-SGD step: clipped grad → noise → optimizer update.

        HF-shaped signature; the ``model`` argument is accepted for
        subclass-override compatibility but the actual forward goes
        through the functional fmodel held on ``self._ctx`` (the model
        is mutated in place by the trainer's training-context wiring,
        so passing it through is redundant).
        """
        ctx = self._ctx
        if ctx is None:
            raise OperationError(
                *(
                    "training_step called outside an active training run; "
                    "Trainer's functional context is not initialised.",
                )
            )
        inputs = self._prepare_input(inputs)
        # Subclass hook: augment the batch with tensors computed *outside* vmap
        # (e.g. TR-DPO's per-step reference logps). Default is a no-op. Any keys
        # it adds must already be present in ``ctx.batch_keys`` (discovered at
        # setup), so subclasses seed placeholder columns at construction and the
        # hook overwrites their values here.
        inputs = self._augment_inputs(inputs)
        # Positional batch tensors in the order discovered at
        # ``_setup_training`` time.  Matches the ``batch_argnums`` the
        # loss builder published, so ``vmap`` batches correctly.
        try:
            batch_args = tuple(inputs[k] for k in ctx.batch_keys)
        except KeyError as missing:
            raise InputTypeError(
                *(
                    f"data_collator output is missing required key {missing!s}; "
                    f"expected keys {list(ctx.batch_keys)!r} (discovered at "
                    f"_setup_training time from a dry run on one example).",
                )
            ) from None
        # Tracked separately for batch-size accounting; the first
        # tensor's leading dim is what HF's ``find_batch_size`` would
        # return.  ``clipped_grad`` short-circuits on empty batches
        # internally (returning zero grads + empty aux), and the DDP
        # collectives below run unchanged on a zero-grad pytree — every
        # rank issues a SUM AllReduce on identical-zero tensors, so the
        # cluster stays in lockstep even when individual ranks see empty
        # Poisson rounds.  Privacy budget is consumed for every step
        # regardless of realized batch size (Poisson accounting is
        # data-independent).
        leading = batch_args[0]
        step_batch_size = int(leading.shape[0])
        # Per-step perf tracker covers clip → DDP sync → noise → optimizer;
        # post-step metric bookkeeping below stays outside the scope.
        # ``sp.mark`` records the elapsed time since the previous mark.
        with self._perf_tracker.train(batch_size=step_batch_size) as sp:
            # Clipped gradients (with optional CPU offload). Any rank-local
            # failure must become a collective event before the gradient
            # AllReduce below. This includes lazy strict-compilation failures:
            # a rank with an empty Poisson draw skips the compiled kernel while
            # a non-empty sibling may fail during its first compilation.
            local_grad_error: Exception | None = None
            grads = aux = None
            try:
                # autocast wraps the *outer* grad_fn (vmap(grad)+clip) call —
                # the placement that actually casts on MPS (see _autocast_ctx).
                with ctx.offload_ctx, self._autocast_ctx():
                    (grads, aux), ctx.clip_state = ctx.grad_fn(
                        ctx.trainable_params,
                        *batch_args,
                        state=ctx.clip_state,
                    )
            except Exception as grad_error:
                local_grad_error = grad_error
            self._synchronize_grad_failure(local_grad_error)

            # DDP collectives between clipping and noise.
            # 1. ``sum_gradients_`` — AllReduce SUM the clipped per-example sum;
            #    after this every rank holds the cluster-wide gradient sum.
            # 2. ``sync(clip_state, aux)`` — under fixed clipping ``clip_state``
            #    sync asserts ``clipping_norm`` matches across ranks; under
            #    adaptive clipping it aggregates the clipped-count and
            #    recomputes the norm.  ``sync(aux)`` gathers the per-example
            #    tensor fields (``grad_norms`` / ``loss_values`` / …) and
            #    weights scalar fields (``clipping_rate``) across ranks so
            #    metrics reported below reflect the cluster-wide batch.
            # Noise can then be added independently on every rank (shared key
            # ⇒ identical noise) and the optimizer update is a pure function of
            # an identical input on every rank, so parameters stay in lockstep.
            if self._ddp.is_distributed:
                from opaque.distributed import sum_gradients_
                from opaque.distributed import sync as _opaque_sync

                sum_gradients_(grads)
                ctx.clip_state, aux = _opaque_sync(ctx.clip_state, aux)
            sp.mark("clip")

            # Noise injection — ``grads`` is a ``ClippedPytree`` whose
            # ``.max_norm`` carries the per-step sensitivity; ``noise_fn``
            # reads it directly and returns a ``NoisedPytree``.  Adaptive
            # clipping flows through unchanged because the wrapper updates
            # ``max_norm`` per call.
            noisy_grads, ctx.noise_state = ctx.noise_fn(grads, ctx.noise_state)
            sp.mark("noise")

            # HF parity: empty device cache *after* the forward/backward pass
            # (activations are freed) but *before* the optimizer update.
            # Uses global_step (pre-increment) to match HF's cadence.
            self._maybe_empty_device_cache()

            # Pre-optimizer hook fires *after* clipping+noise but *before* the
            # optimizer update.  ``grads`` exposes the clipped-and-noised
            # gradients keyed by parameter name so callbacks can compute group
            # norms without touching ``param.grad`` (which doesn't exist in the
            # functional path).
            # ``call_event`` rather than the per-hook method so we can forward
            # DP-specific kwargs (``grads``, ``trainable_params``) — HF's
            # ``CallbackHandler.on_pre_optimizer_step`` has a fixed signature.
            self._control = self._callback_handler.call_event(
                "on_pre_optimizer_step",
                self.args,
                self.state,
                self._control,
                grads=noisy_grads,
                trainable_params=ctx.trainable_params,
            )

            # Optimizer step — DP-aware optimizers read σ directly off the
            # ``NoisedPytree`` when ``noise_bias_correction=True`` was set at
            # construction (via ``optim_args``); no per-step kwargs are accepted.
            updates, ctx.opt_state = ctx.opt.update(
                noisy_grads,
                ctx.opt_state,
                params=ctx.trainable_params,
            )
            ctx.trainable_params = torchopt.apply_updates(ctx.trainable_params, updates)
            sp.mark("optimizer")

            # Post-optimizer hook: surface the post-update parameters so
            # callbacks tracking weight-update norms can snapshot them.
            self._control = self._callback_handler.call_event(
                "on_optimizer_step",
                self.args,
                self.state,
                self._control,
                trainable_params=ctx.trainable_params,
            )

        # After ``sync(aux)`` in distributed mode, ``aux.batch_size`` is
        # the cluster-wide realized batch size (sum across ranks); on a
        # single process it equals the local batch.  Use it as the truth
        # for the empty-step gate so a rank with local_bs=0 still reports
        # the cluster-wide loss when other ranks contributed examples.
        batch_size = int(getattr(aux, "batch_size", 0) or 0)
        if batch_size == 0:
            return {"loss": 0.0, "batch_size": 0}

        # Noise σ travels on the ``NoisedPytree`` wrapper; ``_effective``
        # handles both scalar and ``PerGroup`` shapes.  ``grads.max_norm``
        # is the *realized* per-step clipping threshold: ``adaptive_clipped_grad``
        # updates it geometrically via ``_next_clipping_norm`` each step, and
        # ``FixedClipState`` leaves it equal to the configured ``ctx.clip_norm``.
        # Read it off the ``ClippedPytree`` rather than ``ctx.clip_norm`` —
        # under adaptive mode ``AdaptiveClipState`` carries
        # ``_current_clipping_norm`` / ``_next_clipping_norm``, not
        # ``clipping_norm``.
        noise_std = noisy_grads.noise_stddev
        clipping_norm = grads.max_norm
        metrics: dict[str, Any] = {
            "loss": aux.loss_values.mean().item(),
            "batch_size": batch_size,
            "grad_norm": aux.grad_norms.mean().item(),
            "clip_rate": aux.clipping_rate,
            "clipping_norm": _effective(clipping_norm),
            "noise_std": _effective(noise_std),
        }
        if aux.clipped_grad_norms is not None and aux.clipped_grad_norms.numel() > 0:
            metrics["clipped_grad_norm"] = aux.clipped_grad_norms.mean().item()

        if aux.group_norms is not None and hasattr(clipping_norm, "values"):
            group_noise_std = noise_std if hasattr(noise_std, "values") else None
            group_metrics: dict[str, dict[str, float]] = {}
            for group_name, group_norms in aux.group_norms.items():
                if group_norms.numel() == 0:
                    continue
                group_bound = float(clipping_norm.values[group_name])
                group_values = {
                    "grad_norm": group_norms.mean().item(),
                    "clip_rate": float((group_norms > group_bound).sum().item())
                    / max(1.0, float(batch_size)),
                    "clipping_norm": group_bound,
                }
                if group_noise_std is not None:
                    group_values["noise_std"] = float(
                        group_noise_std.values[group_name]
                    )
                group_metrics[group_name] = group_values
            if group_metrics:
                metrics["group_metrics"] = group_metrics
                # Per-group aggregate: mean engagement + worst group, rather
                # than the engine's worst-group-only ``clipping_rate``.
                _rates = [g["clip_rate"] for g in group_metrics.values()]
                if _rates:
                    metrics["clip_rate"] = sum(_rates) / len(_rates)
                    metrics["clip_rate_max"] = max(_rates)

        # Per-example training telemetry from the loss closure (e.g. DPO
        # rewards). ``aux.loss_aux`` is a dict of per-example tensors, already
        # summed/gathered across ranks by ``sync(aux)``; mean each into a scalar.
        # Same un-noised diagnostic posture as the logged ``loss`` mean above.
        loss_aux = getattr(aux, "loss_aux", None)
        if loss_aux:
            metrics["loss_aux"] = {
                name: value.float().mean().item() for name, value in loss_aux.items()
            }

        return metrics

    # ------------------------------------------------------------------
    # evaluate() — functional forward, no param restoration
    # ------------------------------------------------------------------

    def _augment_inputs(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Hook to augment a prepared batch before the per-example vmap.

        Runs once per step in :meth:`training_step`, *outside* ``vmap`` and on
        the trainer device. The default is a no-op. Subclasses use it to
        overwrite placeholder batch tensors with quantities that must be
        recomputed each step from an evolving non-``vmap`` artefact — e.g.
        TR-DPO's reference log-probs from an EMA reference model. Keys it writes
        must already exist in ``ctx.batch_keys`` (seed them at construction).
        """
        return inputs

    def compute_per_example_loss(
        self,
        fmodel: Callable[..., Any],
        params: dict[str, Tensor],
        inputs: dict[str, Tensor],
        *,
        return_logits: bool = False,
    ) -> Tensor | tuple[Tensor, Any]:
        """Compute one example's loss; vmap-batched by the caller.

        This is the unified DP-correct override hook.  The trainer wraps
        it with ``vmap`` for training (then ``grad`` → clip → noise) and
        for per-example eval (when ``'loss' in include_for_metrics``).
        Subclasses (SFT, DPO, KTO, …) override this method to compute
        domain-specific per-example losses — the same override point
        covers training and eval semantics by construction.

        Default behavior: forward through ``fmodel(params, **inputs)``
        and read ``output["loss"]`` (HF's per-model ``LOSS_MAPPING``
        dispatch is inherited automatically — causal-LM, classification,
        seq2seq, etc.).  When a ``compute_loss_func`` was supplied at
        construction it is called as
        ``compute_loss_func(outputs, labels) -> scalar`` instead — a
        no-subclass escape hatch for one-off custom losses.

        ``args.label_smoothing_factor > 0`` is honored by both the
        Opaque CE kernels (which accept ``label_smoothing`` natively —
        the trainer pushes it through as a ``loss_kwarg``) and by a
        trainer-side ``cross_entropy(..., label_smoothing=...)`` rebuild
        that overrides the model's loss whenever logits are exposed.
        The rebuild covers the no-patches case (HF's native
        ``ForCausalLMLoss`` silently drops the kwarg); for the
        opt-in ``fused_linear_cross_entropy`` path (``logits=None``),
        the kernel applies smoothing inside the fused kernel.
        Subclasses overriding this method are responsible for
        smoothing semantics themselves.

        Args:
            fmodel: Functional model from
                :func:`opaque.functional.make_functional` (called as
                ``fmodel(params, **inputs)``).
            params: All model parameters merged
                (``frozen | trainable``).  Under vmap, ``trainable`` is
                replicated per example and ``frozen`` is broadcast.
            inputs: One example's input dict (under vmap; the caller
                stripped the leading batch dim before invoking this
                method).
            return_logits: When ``True``, also return the model's
                ``logits`` tensor — used by the per-example eval path
                so a single forward yields both per-example losses and
                predictions.

        Returns:
            Scalar ``loss`` (or ``(loss, logits)`` when
            ``return_logits=True``).
        """
        smoothing = float(self.args.label_smoothing_factor)
        # Push smoothing through to the loss function as a kwarg so the
        # Opaque CE kernels (both non-fused and fused-linear) apply it
        # natively.  Vmap-safe — the value is a scalar constant, not a
        # batched tensor.  HF model forwards have ``**kwargs`` that
        # propagate to ``loss_function``; HF's native CE silently drops
        # the kwarg but the trainer-side rebuild below corrects that.
        if smoothing > 0.0 and self._compute_loss_func is None:
            inputs = {**inputs, "label_smoothing": smoothing}
        if (
            not return_logits
            and self._compute_loss_func is None
            and self._fused_forward_uses_marker
        ):
            inputs = {**inputs, "loss_only": True}

        output = fmodel(params, **inputs)
        # Output is required to be dict-like (``ModelOutput`` /
        # ``Mapping``).  ``prediction_step`` enforces this contract at
        # the eval boundary; the training path lands here through
        # ``_build_per_example_loss`` which calls ``fmodel`` (and HF
        # families uniformly emit ``ModelOutput``).  A non-Mapping
        # surfaces as ``AttributeError`` on ``.get`` below — louder than
        # a silent fallback would be.
        output_logits = output.get("logits")

        if self._compute_loss_func is not None:
            labels = next((inputs[k] for k in self._label_names if k in inputs), None)
            loss = self._compute_loss_func(output, labels)
        else:
            loss = output.get("loss")
        if loss is None:
            raise OperationError(
                *(
                    "Trainer.compute_per_example_loss: model forward returned no "
                    "`loss` field.  Pass `compute_loss_func=` for a custom loss, "
                    "or override `compute_per_example_loss` in a subclass.",
                )
            )

        # Trainer-side rebuild: when logits are exposed, rewrite the
        # loss with ``F.cross_entropy(..., label_smoothing=...)`` so the
        # no-patches case (HF's native CE drops the kwarg) is also
        # honored.  Idempotent w.r.t. the kernel's smoothed loss: both
        # paths converge to the same math when ``label_smoothing > 0``.
        if (
            self._compute_loss_func is None
            and smoothing > 0.0
            and output_logits is not None
        ):
            label_key = next((k for k in self._label_names if k in inputs), None)
            labels_tensor = (
                inputs.get(label_key) if label_key is not None else inputs.get("labels")
            )
            if labels_tensor is not None:
                if (
                    self._is_causal_lm
                    and output_logits.ndim >= 2  # noqa: PLR2004
                    and output_logits.shape[:-1] == labels_tensor.shape
                    and output_logits.shape[-2] > 1
                ):
                    smooth_logits = output_logits[..., :-1, :].contiguous()
                    smooth_labels = labels_tensor[..., 1:].contiguous()
                else:
                    smooth_logits = output_logits
                    smooth_labels = labels_tensor
                loss = torch.nn.functional.cross_entropy(
                    smooth_logits.view(-1, smooth_logits.size(-1)),
                    smooth_labels.view(-1),
                    ignore_index=_IGNORE_INDEX,
                    label_smoothing=smoothing,
                )

        if return_logits:
            return loss, output_logits
        return loss

    def compute_per_example_loss_and_metrics(
        self,
        fmodel: Callable[..., Any],
        params: dict[str, Tensor],
        inputs: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Per-example ``(loss, telemetry)`` seam — the rich training/eval hook.

        Returns one example's loss **and** a dict of per-example telemetry
        tensors (e.g. DPO ``rewards/*``). The harness carries the telemetry
        through the clipped-grad ``loss_aux`` channel (DDP-summed by
        ``sync(aux)``), means it per logged step, and aggregates it in the eval
        loop — so a subclass that overrides this gets ``rewards/*`` logged in
        both train and eval from a single forward, with no extra wiring.

        The default has no extra telemetry: it delegates to
        :meth:`compute_per_example_loss`, so trainers whose per-example loss
        emits no metrics (SFT / causal-LM) override only that simpler hook and
        are unaffected by this seam. Overriding *this* method auto-enables the
        aux path (no flag): the trainer detects the override and threads
        ``has_aux`` accordingly.
        """
        return self.compute_per_example_loss(fmodel, params, inputs), {}

    def _overrides_metrics_seam(self) -> bool:
        """Whether a subclass overrides :meth:`compute_per_example_loss_and_metrics`."""
        return (
            type(self).compute_per_example_loss_and_metrics
            is not Trainer.compute_per_example_loss_and_metrics
        )

    def prediction_step(
        self,
        model: Any,
        inputs: dict[str, Tensor],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[Tensor | None, Tensor | tuple[Tensor, ...] | None, Any | None]:
        """Run one eval batch and return ``(loss, logits, labels)``.

        HF-shaped signature; the ``model`` argument is accepted for
        subclass-override compatibility.  The actual forward goes
        through ``self._ctx.fmodel`` mid-training (functional path) or
        ``self._model`` post-training (``nn.Module`` path) — both
        produce identically-shaped tensors so the functional/``nn.Module``
        choice is downstream-transparent.

        Two prediction shapes are possible, selected by
        ``include_for_metrics``:

        - **Standard path** (default): ``logits`` is a ``tuple`` of every
          output field surviving the ``ignore_keys + ["loss"]`` filter,
          collapsed to a bare tensor when length 1 — so multi-output
          models (seq2seq / vision) expose every auxiliary tensor to
          ``compute_metrics``.
        - **Per-example-loss path** (``"loss" in include_for_metrics``):
          the vmap'd closure returns real per-example losses plus the
          model's ``logits`` tensor *only*.  Auxiliary outputs are not
          collected on this path, so ``predictions`` is logits-only even
          for multi-output models.  A one-time warning is emitted; if you
          need full multi-output predictions, drop ``"loss"`` from
          ``include_for_metrics`` and use the standard path.

        ``inputs`` is forwarded to the model via ``**inputs`` after
        popping label tensors named in ``self._label_names`` (default:
        ``["labels"]``).  Any column the model ``forward`` accepts
        (``decoder_input_ids``, ``pixel_values``, ``token_type_ids``, …)
        is forwarded unchanged.

        When ``prediction_loss_only`` is ``True``, ``logits`` and ``labels``
        are returned as ``None`` and ``self._preprocess_logits`` is not
        called — the loop never materializes prediction tensors.

        ``ignore_keys`` filters keys out of ``ModelOutput`` containers
        before logits are extracted.  Defaults to ``[]``; the kv_cache
        patch (always-on under Trainer) already prevents
        ``past_key_values`` from landing in outputs, so there's nothing
        to filter by default.

        ``logits`` is a ``tuple`` of every output field *not* in
        ``ignore_keys + ["loss"]``, collapsed to a bare tensor when the
        tuple has length 1.  This exposes auxiliary outputs
        (``hidden_states``, ``attentions``,
        ``encoder_last_hidden_state``, …) to ``compute_metrics`` for
        seq2seq / encoder-decoder / vision models.

        The model ``forward`` is required to return a dict-like
        ``ModelOutput`` (or any ``Mapping`` of name → tensor).  Bare
        tuples, plain dataclasses, and bare tensors are rejected with
        :class:`TypeError`; wrap your forward to return a dict if you
        need a custom output shape.
        """
        label_keys = list(self._label_names) if self._label_names else []
        has_labels = bool(label_keys) and all(
            inputs.get(k) is not None for k in label_keys
        )

        inputs = self._prepare_input(inputs)
        # Split the batch into ``label_kwargs`` and model inputs.  Labels are
        # captured before the forward because user loss functions may pop
        # or otherwise consume them.
        model_inputs: dict[str, Any] = dict(inputs)
        labels_kwargs: dict[str, Tensor] = {}
        for label_key in label_keys:
            if label_key in model_inputs:
                labels_kwargs[label_key] = model_inputs.pop(label_key)
        if has_labels:
            label_values = tuple(labels_kwargs[k] for k in label_keys)
            labs: Any | None = _eval.nested_detach(
                label_values[0] if len(label_values) == 1 else label_values
            )
        else:
            labs = None

        # ``'loss' in include_for_metrics`` opts into real per-example
        # losses via the vmap'd eval closure. Requires labels
        # (loss-without-labels falls through to the standard path). In
        # loss-only mode the separate scalar closure avoids returning logits.
        use_per_example_loss = (
            "loss" in (self.args.include_for_metrics or []) and has_labels
        )

        if use_per_example_loss:
            vmapped_fn, _batch_argnums, batch_keys = self._get_eval_per_example_loss_fn(
                return_logits=not prediction_loss_only
            )
            if self._ctx is not None:
                trainable = self._ctx.trainable_params
            else:
                trainable = {
                    name: p
                    for name, p in self._model.named_parameters()
                    if p.requires_grad
                }
            # ``batch_keys`` / ``batch_argnums`` were discovered from the
            # *train* collator and baked into the vmap'd closure.  If the eval
            # collator emits a different key set, ``inputs.get(k)`` is ``None``
            # and vmap fails with an opaque error.  Validate up front and
            # raise a clear, actionable message instead.
            missing = [k for k in batch_keys if inputs.get(k) is None]
            if missing:
                raise ConfigurationError(
                    *(
                        "Per-example eval (include_for_metrics=['loss']) expects the "
                        f"eval batch to carry the train-discovered keys {list(batch_keys)!r}, "
                        f"but {missing!r} are absent (or None).  The eval collator/"
                        "dataset differs from the training one; align them, or drop "
                        "'loss' from include_for_metrics to use the standard eval path.",
                    )
                )
            batch_args = tuple(inputs.get(k) for k in batch_keys)
            with torch.no_grad():
                was_training = self._model.training
                if was_training:
                    self._model.eval()
                try:
                    output = vmapped_fn(trainable, *batch_args)
                finally:
                    if was_training:
                        self._model.train()
            if prediction_loss_only:
                return output.detach(), None, None
            loss, logits_tensor = output
            loss = loss.detach()
            # The per-example path collects only the model's ``logits``
            # tensor (not the full ``ignore_keys``-filtered output tuple the
            # standard path builds).  Surface that once so multi-output
            # users aren't silently handed logits-only predictions, and
            # honour an explicit ``logits`` entry in ``ignore_keys``.
            if not getattr(self, "_warned_per_example_logits_only", False):
                log.warning(
                    "include_for_metrics=['loss'] uses the per-example eval "
                    "path, which returns logits-only predictions (auxiliary "
                    "model outputs are not collected).  Drop 'loss' from "
                    "include_for_metrics for the full multi-output prediction "
                    "tuple."
                )
                self._warned_per_example_logits_only = True
            preds = (
                None
                if ignore_keys and "logits" in ignore_keys
                else (logits_tensor.detach() if logits_tensor is not None else None)
            )
            return loss, preds, labs

        # Batched forward: the default reduced eval path avoids the per-example
        # vmap while still using the configured training objective below.
        forward_inputs = {**model_inputs, **labels_kwargs}
        smoothing = float(self.args.label_smoothing_factor)
        if (
            smoothing > 0.0
            and self._compute_loss_func is None
            and self._fused_forward_uses_marker
        ):
            forward_inputs["label_smoothing"] = smoothing
        if (
            prediction_loss_only
            and has_labels
            and self._compute_loss_func is None
            and self._fused_forward_uses_marker
        ):
            forward_inputs["loss_only"] = True
        with torch.no_grad():
            was_training = self._model.training
            if was_training:
                self._model.eval()
            try:
                if self._ctx is not None:
                    merged = {**self._ctx.frozen_params, **self._ctx.trainable_params}
                    output = self._ctx.fmodel(merged, **forward_inputs)
                else:
                    output = model(**forward_inputs)
            finally:
                if was_training:
                    self._model.train()
            if has_labels and isinstance(output, Mapping):
                if self._compute_loss_func is not None:
                    custom_labels = next(
                        (labels_kwargs[k] for k in label_keys if k in labels_kwargs),
                        None,
                    )
                    loss = self._compute_loss_func(output, custom_labels)
                else:
                    loss = output.get("loss")
                    output_logits = output.get("logits")
                    if smoothing > 0.0 and output_logits is not None:
                        label_key = next(
                            (k for k in label_keys if k in labels_kwargs),
                            None,
                        )
                        labels_tensor = (
                            labels_kwargs.get(label_key)
                            if label_key is not None
                            else labels_kwargs.get("labels")
                        )
                        if labels_tensor is not None:
                            if (
                                self._is_causal_lm
                                and output_logits.ndim >= 3  # noqa: PLR2004
                                and output_logits.shape[:-1] == labels_tensor.shape
                                and output_logits.shape[-2] > 1
                            ):
                                smooth_logits = output_logits[..., :-1, :].contiguous()
                                smooth_labels = labels_tensor[..., 1:].contiguous()
                            else:
                                smooth_logits = output_logits
                                smooth_labels = labels_tensor
                            loss = torch.nn.functional.cross_entropy(
                                smooth_logits.view(-1, smooth_logits.size(-1)),
                                smooth_labels.view(-1),
                                ignore_index=_IGNORE_INDEX,
                                label_smoothing=smoothing,
                            )
                if loss is not None:
                    loss = loss.detach().mean()
            else:
                loss = None
        if prediction_loss_only:
            return loss, None, None

        if not isinstance(output, Mapping):
            raise InputTypeError(
                *(
                    "Trainer requires model.forward to return a dict-like "
                    "ModelOutput (or Mapping). "
                    f"Got {type(output).__name__}; wrap forward to return a dict.",
                )
            )

        # Collect every output field that survives the ``ignore_keys +
        # ["loss"]`` filter into a tuple, preserving model-defined order.
        # Non-tensor values are dropped (eval pipeline consumes tensor
        # predictions only).
        ignore_keys = list(ignore_keys) if ignore_keys is not None else []
        skip = set(ignore_keys)
        if has_labels:
            skip.add("loss")
        logits_tuple: tuple[Tensor, ...] = tuple(
            v for k, v in output.items() if k not in skip and isinstance(v, Tensor)
        )

        # Collapse a length-1 tuple to a bare tensor so single-output
        # models keep the simple ``compute_metrics(EvalPrediction(
        # predictions=tensor, ...))`` contract.  Length-0 → ``None``.
        if len(logits_tuple) == 0:
            logits: Tensor | tuple[Tensor, ...] | None = None
        elif len(logits_tuple) == 1:
            logits = logits_tuple[0]
        else:
            logits = logits_tuple

        return loss, logits, labs

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        *,
        description: str = "Evaluation",
        prediction_loss_only: bool | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> EvaluationResult:
        """Drive a full eval pass and return an :class:`EvaluationResult`.

        The loop produces ``(loss, logits, labels)`` per batch via
        :meth:`prediction_step` and feeds them to a
        :class:`~opaque.api.transformers.trainer._eval._PredictionAccumulator`;
        ``compute_metrics`` is invoked once at the end with a single
        :class:`EvalPrediction`.

        Optional ``EvalPrediction`` fields are opt-in via
        ``args.include_for_metrics`` (the HF-canonical knob):

        - ``'inputs'`` — populates ``EvalPrediction.inputs`` with the
          model's primary input column (sniffed from
          ``model.main_input_name``; ``"input_ids"`` for text models,
          ``"pixel_values"`` for vision, etc.).
        - ``'loss'`` — populates ``EvalPrediction.losses`` with **real
          per-example losses** computed via the vmap'd eval closure
          (:meth:`prediction_step` switches to a per-example forward
          when this is requested).

        ``prediction_loss_only`` (when not ``None``) overrides
        ``args.prediction_loss_only`` — used by HF when callers want a
        loss-only pass even though metrics are configured.

        ``description`` is logged at entry alongside ``Num examples`` and
        ``Batch size`` (HF parity).  Returned ``metrics`` contain only
        model/metric outputs; ``evaluate`` and ``predict`` add throughput
        and memory metrics around this raw loop, matching HF's layering.
        """
        a = self.args
        ploss_only = (
            bool(prediction_loss_only)
            if prediction_loss_only is not None
            else bool(a.prediction_loss_only)
        )

        include_for = set(a.include_for_metrics or [])
        include_inputs = "inputs" in include_for
        include_losses = "loss" in include_for

        # HF parity: ``inputs`` exposed to ``compute_metrics`` carries only
        # the model's *primary* input column, not the entire batch dict.
        main_input_name = getattr(self._model, "main_input_name", "input_ids")

        # HF-parity entry log so train-time eval, final eval, and predict
        # show up distinctly in train logs.
        try:
            num_examples = len(getattr(dataloader, "dataset", []) or [])
        except TypeError:
            num_examples = 0
        per_device_eval_bs = int(a.per_device_eval_batch_size or 0)
        log.info("***** Running %s *****", description)
        if num_examples:
            log.info("  Num examples = %d", num_examples)
        if per_device_eval_bs:
            log.info("  Batch size = %d", per_device_eval_bs)

        telemetry = _eval._EvaluationTelemetry(self._device)
        accumulator = _eval._PredictionAccumulator(
            prediction_loss_only=ploss_only,
            eval_accumulation_steps=a.eval_accumulation_steps,
            eval_do_concat_batches=bool(a.eval_do_concat_batches),
            include_inputs=include_inputs,
            include_losses=include_losses,
            telemetry=telemetry,
        )
        if a.eval_accumulation_steps:
            log.info(
                "Eval CPU offload engaged: flushing every %d batches",
                a.eval_accumulation_steps,
            )

        total_loss = 0.0
        loss_samples = 0
        total_samples = 0
        # Symmetric per-example eval telemetry: a subclass ``prediction_step``
        # populates ``self._pending_eval_aux`` with the same per-example dict the
        # training aux channel carries (e.g. DPO ``rewards/*``); collect + mean it
        # into the eval metrics, mirroring the train-step aux logging.
        eval_aux_chunks: dict[str, list[Tensor]] = {}
        # Under DDP a retryable OOM anywhere in the per-batch eval body must
        # become a *collective* event before the end-of-loop
        # ``reduce_scalar`` / gather: if the OOM'ing rank skipped those ops
        # while siblings issued them, the process group would deadlock.
        # Mirror the training-step guard — catch a retryable OOM, break out
        # of the batch loop, then all-reduce a MAX flag so EVERY rank raises
        # before collectives.
        local_oom = False

        with self._perf_tracker.eval(batch_size=num_examples):
            for batch in dataloader:
                bs = _eval.find_batch_size(batch) or 0
                if bs == 0:
                    continue
                self._pending_eval_aux = None
                try:
                    with telemetry.model():
                        loss, logits, labels = self.prediction_step(
                            self._model,
                            batch,
                            prediction_loss_only=ploss_only,
                            ignore_keys=ignore_keys,
                        )
                except RuntimeError as err:
                    if not (self._ddp.is_distributed and self._is_retryable_oom(err)):
                        raise
                    local_oom = True
                    self._pending_eval_aux = None
                    break
                try:
                    step_aux = self._pending_eval_aux
                    self._pending_eval_aux = None
                    if step_aux:
                        for name, value in step_aux.items():
                            eval_aux_chunks.setdefault(name, []).append(value.detach())

                    # Per-batch progress hook (HF parity); progress callbacks rely
                    # on this firing once per eval batch.
                    self._control = self._callback_handler.on_prediction_step(
                        self.args,
                        self.state,
                        self._control,
                    )

                    # ``loss`` is scalar (default forward) or 1-D per-example
                    # (when ``'loss' in include_for_metrics`` triggers the
                    # vmap'd eval closure).  The model's per-example CE is already
                    # the mean over real (non-``-100``) tokens, so:
                    #   - scalar branch: ``loss.item() * real_tokens_in_batch`` is
                    #     the total CE; dividing the running sum by the running
                    #     ``loss_samples`` count gives per-real-token mean CE.
                    #   - 1-D branch: ``loss[i] * real_tokens_in_example[i]`` is
                    #     example i's total CE; summing then dividing by the total
                    #     real-token count gives the same per-token mean.
                    # When labels aren't exposed (rare), or the trainer opted out
                    # of token weighting (``_eval_token_weighted_loss=False``),
                    # fall back to the plain per-example mean.
                    if loss is not None:
                        if (
                            labels is not None
                            and self._eval_token_weighted_loss
                            and self._is_causal_lm
                        ):
                            # HF's ForCausalLMLoss scores ``labels[..., 1:]`` (drops
                            # position 0 via the internal shift); the per-token-mean
                            # weighting denominator must match that count.
                            shifted = labels[..., 1:]
                            token_mask = shifted != _IGNORE_INDEX
                            if loss.ndim > 0:
                                # per-example: weight each by its real-token count
                                per_example_real = token_mask.sum(
                                    dim=tuple(range(1, shifted.ndim))
                                ).to(loss.dtype)
                                total_loss += float(
                                    (loss * per_example_real).sum().item()
                                )
                                loss_samples += int(per_example_real.sum().item())
                            else:
                                # scalar: weight by real-token count in the whole batch
                                real_tokens = int(token_mask.sum().item())
                                total_loss += float(loss.item()) * real_tokens
                                loss_samples += real_tokens
                        else:
                            # Non-causal objectives and trainers that opt out of
                            # token weighting use the plain per-example mean.
                            total_loss += (
                                float(loss.sum().item())
                                if loss.ndim > 0
                                else float(loss.item()) * bs
                            )
                            loss_samples += bs
                    total_samples += bs

                    if logits is not None and self._preprocess_logits is not None:
                        logits_for_hook: Tensor | tuple[Tensor, ...]
                        logits_for_hook = (
                            logits[0]
                            if isinstance(logits, tuple) and len(logits) == 1
                            else logits
                        )
                        logits = self._preprocess_logits(logits_for_hook, labels)

                    main_input = batch.get(main_input_name) if include_inputs else None
                    accumulator.add(
                        loss=loss,
                        logits=logits,
                        labels=labels,
                        inputs=main_input,
                        batch_size=bs,
                    )
                except RuntimeError as err:
                    if not (self._ddp.is_distributed and self._is_retryable_oom(err)):
                        raise
                    local_oom = True
                    self._pending_eval_aux = None
                    break

        # Cluster-wide OOM check before end-of-loop collectives. Ranks that
        # finished their shard wait here for siblings still iterating; a
        # mid-loop OOM on any rank then raises on every rank so nobody enters
        # ``reduce_scalar`` / gather. Matches the training-step guard.
        if self._ddp.is_distributed and self._cluster_needs_step_down(local_oom):
            raise torch.OutOfMemoryError(  # noqa: TRY003 - preserve PyTorch OOM type
                "collective eval batch retry (a rank OOM'd during eval batch "
                "processing; "
                "whole cluster steps down to a smaller "
                "per_device_eval_batch_size)."
            )

        # ----- Finalize metrics -----
        # under DDP each rank evaluated a disjoint shard of the
        # eval dataset.  Reduce the per-rank loss totals to a cluster-wide
        # mean before reporting; the prediction accumulator's tensors are
        # gathered inside ``finalize()`` below.
        if self._ddp.is_distributed:
            from opaque.api.engine.distributed._state import reduce_scalar

            with telemetry.phase("gather"):
                total_loss = reduce_scalar(
                    float(total_loss), op="sum", device=self._device
                )
                loss_samples = int(
                    reduce_scalar(loss_samples, op="sum", device=self._device)
                )
                total_samples = int(
                    reduce_scalar(total_samples, op="sum", device=self._device)
                )
        metrics: dict[str, Any] = {}
        if loss_samples > 0:
            metrics["loss"] = total_loss / loss_samples

        # Per-example eval telemetry (e.g. DPO ``rewards/*``): every rank
        # enters one optional-pytree gather, including ranks with empty shards.
        # Bare keys here; ``with_metric_prefix`` below namespaces them as
        # ``{prefix}_<key>``.
        if self._ddp.is_distributed:
            from opaque.api.engine.distributed._state import gather_pytree

            local_eval_aux = (
                {name: torch.cat(chunks) for name, chunks in eval_aux_chunks.items()}
                if eval_aux_chunks
                else None
            )
            with telemetry.phase("gather"):
                gathered_eval_aux = gather_pytree(local_eval_aux)
        else:
            gathered_eval_aux = {
                name: torch.cat(chunks) for name, chunks in eval_aux_chunks.items()
            }
        if gathered_eval_aux:
            for name, gathered in gathered_eval_aux.items():
                if gathered.numel() > 0:
                    metrics[name] = float(gathered.float().mean().item())

        # HF parity: under DDP each rank's dataloader sees a per-rank
        # shard (``local_shard``) so ``len(dataset)`` reports per-rank
        # count, not cluster-wide.  Use the AllReduce'd ``total_samples``
        # directly so truncation after gather doesn't slice the
        # cluster-wide tensor back down to one shard.
        if self._ddp.is_distributed:
            num_samples = total_samples
        else:
            num_samples = _eval.resolve_eval_num_samples(
                dataloader,
                observed=total_samples,
            )

        predictions, label_ids, inputs_arr, losses_tensor = accumulator.finalize(
            num_samples=num_samples,
            gather=self._ddp.is_distributed,
        )

        if total_samples > 0 and (
            self._compute_metrics is not None
            and not ploss_only
            and predictions is not None
        ):
            ep = EvalPrediction(
                predictions=predictions,
                label_ids=label_ids,
                inputs=inputs_arr,
                losses=losses_tensor,
            )
            with telemetry.phase("metric"):
                user_metrics = self._compute_metrics(ep)
            if user_metrics:
                metrics.update(user_metrics)
        # Empty-dataset path: HF silently skips ``compute_metrics``;
        # we mirror that — ``metrics`` carries only ``loss`` (absent
        # when total_samples == 0).  Caller-level evaluate/predict
        # wrappers add throughput metrics.

        metrics.update(telemetry.to_dict())

        # The eval performance stage now spans the complete prediction loop,
        # rather than synchronizing and retaining only the last batch record.
        # Caller-level ``speed_metrics`` still owns aggregate throughput keys.
        if self._perf_tracker.eval.last is not None:
            metrics.update(self._perf_tracker.eval.last.to_dict())

        # HF parity: scalarize numpy / tensor scalars before serialization.
        metrics = _eval.denumpify_detensorize(metrics)
        prefixed = _eval.with_metric_prefix(metrics, metric_key_prefix)

        return EvaluationResult(
            predictions=predictions,
            label_ids=label_ids,
            metrics=prefixed,
            num_samples=num_samples,
        )

    def evaluate(
        self,
        eval_dataset: Dataset | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """Evaluate by functional forward pass.

        During training, uses the functional model with current
        ``trainable_params``. After training (no active context),
        falls back to the ``nn.Module``.

        Installs the cached-accountant barrier on the active training
        context (when present), appends the metrics to
        ``state.log_history`` via :meth:`log`, fires the ``on_evaluate``
        callback, and updates ``state.best_metric`` / ``best_global_step``.
        Direct user calls behave identically to the eval calls the
        training loop makes.

        Returns a metrics dict with ``{prefix}_loss`` and any keys
        produced by a user-supplied ``compute_metrics`` (auto-prefixed
        with ``metric_key_prefix`` where missing — HF parity).

        Multi-dataset eval (HF parity): pass a ``Mapping`` of
        ``name -> dataset`` (or set ``eval_dataset`` to one at
        construction) to evaluate each split independently with metrics
        namespaced as ``{prefix}_{name}_*`` and merged into one dict;
        each sub-evaluation logs and fires ``on_evaluate`` on its own.
        """
        dataset = eval_dataset if eval_dataset is not None else self._eval_dataset
        if dataset is None:
            raise ConfigurationError(*("Trainer.evaluate() requires an eval_dataset.",))

        # Multi-dataset eval: recurse per split with a namespaced prefix and
        # merge (mirrors transformers.Trainer.evaluate).
        if isinstance(dataset, Mapping):
            merged: dict[str, float] = {}
            for name, sub_dataset in dataset.items():
                merged.update(
                    self.evaluate(
                        eval_dataset=sub_dataset,
                        ignore_keys=ignore_keys,
                        metric_key_prefix=f"{metric_key_prefix}_{name}",
                    )
                )
            return merged

        # DP footgun: evaluating on the (private) training set consumes no
        # privacy budget, but the reported metrics are computed on private
        # data and are NOT covered by the DP guarantee — publishing them
        # leaks.  Warn once so "DP end-to-end" isn't silently assumed.
        if (
            self._train_dataset is not None
            and dataset is self._train_dataset
            and not getattr(self, "_warned_eval_on_train", False)
        ):
            log.warning(
                "Evaluating on the training dataset: eval consumes no privacy "
                "budget, but the resulting metrics are computed on private "
                "data and carry NO differential-privacy guarantee.  Do not "
                "publish them as DP-protected."
            )
            self._warned_eval_on_train = True

        with self._use_published_functional_params():
            result = self._run_evaluation_loop(
                dataset,
                prediction_loss_only=True if self._compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        self._after_evaluate(result.metrics)
        return result.metrics

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "test",
    ) -> EvaluationResult:
        """Run prediction loop and return predictions + labels + metrics."""
        with self._use_published_functional_params():
            result = self._run_evaluation_loop(
                test_dataset,
                prediction_loss_only=None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
                description="Prediction",
            )
        self._control = self._callback_handler.on_predict(
            self.args,
            self.state,
            self._control,
            metrics=result.metrics,
        )
        return result

    def log_metrics(self, split: str, metrics: dict[str, float]) -> None:
        """Log metrics in HF's public helper shape."""
        if not _distributed.should_log(self.args, self._ddp):
            return
        log.info("***** %s metrics *****", split)
        for key_name in sorted(metrics):
            log.info("  %s = %s", key_name, metrics[key_name])

    def save_metrics(
        self,
        split: str,
        metrics: dict[str, float],
        combined: bool = True,
    ) -> None:
        """Save split metrics as JSON files under the effective output directory."""
        if not _distributed.should_save(self.args, self._ddp):
            return
        output_dir = self._effective_output_dir()
        if output_dir is None:
            raise ConfigurationError(
                *("save_metrics requires args.output_dir to be set",)
            )
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = Path(output_dir) / f"{split}_results.json"
        with path.open("w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True, default=str)
        if combined:
            all_path = Path(output_dir) / "all_results.json"
            if all_path.exists():
                with all_path.open() as f:
                    all_metrics = json.load(f)
            else:
                all_metrics = {}
            all_metrics.update(metrics)
            with all_path.open("w") as f:
                json.dump(all_metrics, f, indent=2, sort_keys=True, default=str)

    def save_state(self) -> None:
        """Save ``trainer_state.json`` under the effective output directory."""
        if not _distributed.should_save(self.args, self._ddp):
            return
        output_dir = self._effective_output_dir()
        if output_dir is None:
            raise ConfigurationError(
                *("save_state requires args.output_dir to be set",)
            )
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self._save_trainer_state(output_dir)

    # ------------------------------------------------------------------
    # Hub publishing (orthogonal to DP — publish the finished model)
    # ------------------------------------------------------------------

    def init_hf_repo(self, token: str | None = None) -> None:
        """Create (or validate) the HF Hub repo and populate ``self.hub_model_id``.

        Mirrors ``Trainer.init_hf_repo``.
        """
        _hub.init_hf_repo(self, token=token)

    def push_to_hub(
        self,
        commit_message: str | None = "End of training",
        blocking: bool = True,
        token: str | None = None,
        revision: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Upload the model to the HF Hub.

        Mirrors ``Trainer.push_to_hub``.  Restores in-memory params, writes the
        model card (with the Opaque DP ε/δ section), then uploads
        ``args.output_dir`` via ``huggingface_hub.upload_folder``.  Synchronous
        by default; there is no in-training auto-push.
        """
        return _hub.push_to_hub(
            self,
            commit_message=commit_message,
            blocking=blocking,
            token=token,
            revision=revision,
            **kwargs,
        )

    def create_model_card(
        self,
        language: str | None = None,
        license: str | None = None,
        tags: str | list[str] | None = None,
        model_name: str | None = None,
        finetuned_from: str | None = None,
        tasks: str | list[str] | None = None,
        dataset_tags: str | list[str] | None = None,
        dataset: str | list[str] | None = None,
        dataset_args: str | list[str] | None = None,
    ) -> None:
        """Write ``README.md`` to ``args.output_dir`` with HF model card + DP section.

        Mirrors ``Trainer.create_model_card`` with Opaque-specific additions:
        ``differential-privacy`` + ``opaque`` tags, and a ``## Privacy budget``
        section listing ε, δ, noise multiplier, and clipping norm.
        """
        _hub.create_model_card(
            self,
            language=language,
            license=license,
            tags=tags,
            model_name=model_name,
            finetuned_from=finetuned_from,
            tasks=tasks,
            dataset_tags=dataset_tags,
            dataset=dataset,
            dataset_args=dataset_args,
        )

    def _run_evaluation_loop(
        self,
        dataset: Dataset,
        *,
        prediction_loss_only: bool | None,
        ignore_keys: list[str] | None,
        metric_key_prefix: str,
        description: str = "Evaluation",
    ) -> EvaluationResult:
        """Drive a single :meth:`evaluation_loop` and return its result.

        Pure forward + accumulator; no callback / log side effects.  Shared by
        :meth:`evaluate` and :meth:`predict`.  When ``auto_find_microbatch_size``
        is set, an eval CUDA-OOM lowers ``args.per_device_eval_batch_size`` (and
        clears the cached eval dataloader) before retrying.
        """
        # HF parity: start/stop memory tracker around the eval loop so
        # ``skip_memory_metrics=False`` captures eval-phase memory usage.
        self._memory_tracker.start()
        # ``eval_dtype`` casts ``self._model`` in place.  During an active
        # training run the eval forward goes through the functional
        # ``ctx.fmodel`` + detached param dicts, which the cast does NOT
        # reach — so bf16_full_eval is a no-op mid-training (it only takes
        # effect for the post-training nn.Module path).  Surface that once
        # instead of letting it pass silently.
        if (
            self._ctx is not None
            and self.args.bf16_full_eval
            and not getattr(self, "_warned_full_eval_functional", False)
        ):
            log.warning(
                "bf16_full_eval does not apply to in-training evaluation: the "
                "functional eval forward runs at the training dtype.  Full-cast "
                "eval takes effect only for evaluation after train() returns "
                "(the nn.Module path)."
            )
            self._warned_full_eval_functional = True

        # ``auto_find_microbatch_size`` also guards eval: on CUDA-OOM, halve
        # ``per_device_eval_batch_size`` and retry. Eval has no gradient
        # accumulation, so the eval batch is a pure throughput knob — shrinking
        # it leaves the metrics unchanged. ``evaluation_loop`` turns a per-rank
        # OOM into a cluster-wide raise before its end-of-loop collectives (so
        # DDP doesn't deadlock); the step-down here then rebuilds loaders and
        # retries in lockstep.
        while True:
            # Time only the attempt that succeeds — a failed OOM attempt below
            # restarts the clock so eval throughput isn't under-reported.
            start_time = time.time()
            loader = self.get_eval_dataloader(dataset)
            self._callback_handler.eval_dataloader = loader
            local_oom = False
            local_oom_error: BaseException | None = None
            result = None
            try:
                with eval_dtype(self._model, self.args, self._train_dtype):
                    result = self.evaluation_loop(
                        loader,
                        description=description,
                        prediction_loss_only=prediction_loss_only,
                        ignore_keys=ignore_keys,
                        metric_key_prefix=metric_key_prefix,
                    )
            except RuntimeError as err:
                if not (
                    self.args.auto_find_microbatch_size and self._is_retryable_oom(err)
                ):
                    raise
                local_oom = True
                local_oom_error = err

            if not self._cluster_needs_step_down(local_oom):
                assert result is not None  # no rank OOM'd at this batch size
                break

            self._empty_device_cache_for_retry()
            current = max(1, int(self.args.per_device_eval_batch_size))
            if current <= 1:
                if local_oom_error is not None:
                    raise local_oom_error
                raise OperationError(
                    *(
                        "auto_find_microbatch_size: eval OOMs at "
                        "per_device_eval_batch_size=1. Reduce the eval sequence "
                        "length or the model size.",
                    )
                )
            reduced = max(1, current // 2)
            log.warning(
                "auto_find_microbatch_size: eval OOM at "
                "per_device_eval_batch_size=%d, retrying at %d.",
                current,
                reduced,
            )
            self.args.per_device_eval_batch_size = reduced
            # Drop the cached loader so the next attempt rebuilds it smaller.
            self._eval_dataloader = None

        total_batch_size = max(1, int(self.args.per_device_eval_batch_size))
        result.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=result.num_samples,
                num_steps=math.ceil(result.num_samples / total_batch_size),
            )
        )
        self._memory_tracker.stop_and_update_metrics(result.metrics)
        return result

    def _after_evaluate(self, metrics: dict[str, float]) -> None:
        """Apply the post-eval side effects HF parity requires.

        - Wraps the active accountant in :func:`acc.cached` so subsequent
          ε queries reuse the PLD up to this point.
        - Appends the metrics to ``state.log_history`` via :meth:`log`.
        - Fires ``on_evaluate``.
        Best-model tracking and metric-driven LR schedules are handled by
        the training loop after train-triggered evaluation, matching HF.
        """
        ctx = self._ctx
        if ctx is not None:
            ctx.accounting = acc.cached(ctx.accounting)
        self.log(dict(metrics))
        self._control = self._callback_handler.on_evaluate(
            self.args,
            self.state,
            self._control,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _pin_memory_enabled(self) -> bool:
        """Resolve effective ``pin_memory`` for DataLoaders.

        Pinned host memory is a CUDA-only optimization (it accelerates host
        → device async copies via ``cudaMemcpyAsync``); enabling it on CPU is
        a no-op and on MPS triggers a noisy PyTorch warning on every loader
        construction.  Mirror HF's behavior: only honor the user setting when
        the trainer is bound to a CUDA device.
        """
        return bool(self.args.dataloader_pin_memory) and self._device.type == "cuda"

    def _resolve_collate_fn(self, base_collator: Callable | None = None) -> Callable:
        """Resolve the collate function for both train and eval loaders.

        Wraps the HF collator with :func:`opaque.functional.empty_collate` so
        Poisson subsampling can yield an empty index list without crashing
        collators that assume ``examples[0]`` exists.

        The collator is deliberately CPU-only so DataLoader workers never touch
        CUDA/MPS; tensors move to ``self._device`` in ``_prepare_input`` on the
        main process.
        """
        from opaque.functional import empty_collate

        base = self._data_collator if base_collator is None else base_collator
        return empty_collate(base)

    def _maybe_prime_collate(self, collate_fn: Callable, dataset: Any) -> Callable:
        """Warm ``empty_collate`` with one real example so the first Poisson
        batch can be empty without needing ``examples[0]`` for a template.

        The wrapped ``empty_collate`` is stateful — calling it once with a
        real row caches the template tensor's shape so a later empty batch
        can be synthesized.  Returns the collator unmodified (priming is a
        side effect on the closure state).

        Falls back to unprimed when probing the dataset isn't possible
        (streaming iterables have no ``__len__`` / indexing; custom
        collators may reject a single-row list).  Each ``except`` is
        narrowed to the exact failure mode it's guarding against so that
        unrelated exceptions still surface.
        """
        if dataset is None:
            return collate_fn
        try:
            n = len(dataset)  # type: ignore[arg-type]
        except TypeError:
            # Streaming / iterable dataset without ``__len__``.
            return collate_fn
        if n == 0:
            return collate_fn
        try:
            row = dataset[0]  # type: ignore[index]
        except (TypeError, IndexError, KeyError):
            # Non-subscriptable dataset, or row lookup failed.
            return collate_fn
        try:
            collate_fn([row])
        except (TypeError, ValueError, KeyError):
            # Collator rejected the single-row list shape (rare —
            # default collators accept lists of any length including 1).
            return collate_fn
        return collate_fn

    def _prepare_input(self, value: Any) -> Any:
        """Recursively move tensor inputs to the trainer device.

        Mirrors HF's main-process input preparation while keeping
        DataLoader workers CPU-only.
        """
        if isinstance(value, Tensor):
            return value.to(self._device)
        if isinstance(value, Mapping):
            return {k: self._prepare_input(v) for k, v in value.items()}
        if isinstance(value, tuple):
            return tuple(self._prepare_input(v) for v in value)
        if isinstance(value, list):
            return [self._prepare_input(v) for v in value]
        return value

    def _set_signature_columns_if_needed(self) -> None:
        if self._signature_columns is not None or self._signature_columns_unavailable:
            return
        model_to_inspect = self._model
        if self._is_peft:
            if hasattr(self._model, "get_base_model"):
                model_to_inspect = self._model.get_base_model()
            else:
                model_to_inspect = self._model.base_model.model
        signature = inspect.signature(model_to_inspect.forward)
        signature_columns = list(signature.parameters.keys())
        signature_columns += list({"label", "label_ids", *self._label_names})

        # Opaque's import-time model patches may wrap ``forward`` into a generic
        # ``(*args, **kwargs)`` callable, which makes HF-style column pruning unsafe.
        label_columns = {"label", "label_ids", *self._label_names}
        semantic_columns = [
            name
            for name in signature_columns
            if name not in {"args", "kwargs", "self", *label_columns}
        ]
        if len(semantic_columns) == 0:
            self._signature_columns_unavailable = True
            self._signature_columns = []
            return

        self._signature_columns = signature_columns

    def _remove_unused_columns(
        self,
        dataset: Dataset,
        description: str | None = None,
    ) -> Dataset:
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        if self._signature_columns_unavailable:
            return dataset
        signature_columns = self._signature_columns or []

        ignored_columns = [
            k for k in dataset.column_names if k not in signature_columns
        ]
        if ignored_columns:
            dset_description = (
                "" if description is None else f"in the {description} set"
            )
            log.info(
                "The following columns %s don't have a corresponding argument in "
                "`%s.forward` and have been ignored: %s. If %s are not expected by "
                "`%s.forward`, you can safely ignore this message.",
                dset_description,
                self._model.__class__.__name__,
                ", ".join(ignored_columns),
                ", ".join(ignored_columns),
                self._model.__class__.__name__,
            )

        columns = [k for k in signature_columns if k in dataset.column_names]
        if not columns:
            raise ConfigurationError(
                *(
                    "No columns in the dataset match the model's forward method signature: "
                    f"({', '.join(signature_columns)}). The following columns have "
                    f"been ignored: [{', '.join(ignored_columns)}]. Please check the "
                    "dataset and model. You may need to set `remove_unused_columns=False` "
                    "in TrainingArguments.",
                )
            )
        return dataset.remove_columns(ignored_columns)

    def _get_collator_with_removed_columns(
        self,
        data_collator: Callable,
        description: str | None = None,
    ) -> Callable:
        if not self.args.remove_unused_columns:
            return data_collator
        self._set_signature_columns_if_needed()
        if self._signature_columns_unavailable:
            return data_collator
        return RemoveColumnsCollator(
            data_collator=data_collator,
            signature_columns=self._signature_columns or [],
            logger=log,
            description=description,
            model_name=self._model.__class__.__name__,
        )

    def _prepare_dataset_and_collator(
        self,
        dataset: Any,
        *,
        description: str,
        collate_fn: Callable,
    ) -> tuple[Any, Callable]:
        if isinstance(dataset, Dataset):
            return self._remove_unused_columns(
                dataset, description=description
            ), collate_fn
        return dataset, self._get_collator_with_removed_columns(
            collate_fn,
            description=description,
        )

    def _build_per_example_loss(
        self,
        fmodel: Callable[..., Any],
        frozen_params: dict[str, Tensor],
        batch_keys: tuple[str, ...],
        *,
        return_logits: bool = False,
        with_metrics: bool = False,
    ) -> tuple[Callable[..., Any], tuple[int, ...]]:
        """Wrap the per-example loss hook for ``vmap(grad(...))``.

        Bridges the user-facing override hooks (``compute_per_example_loss`` or,
        when ``with_metrics``, the richer ``compute_per_example_loss_and_metrics``)
        to ``clipped_grad``'s positional contract
        ``(trainable_params, *batch_args) -> scalar_loss``. The training-loop
        concerns — bf16 autocast and chunk-level ``torch.compile`` — wrap around
        the resulting gradient transform so subclasses need not implement them.

        Args:
            fmodel: Functional model from
                :func:`opaque.functional.make_functional`.
            frozen_params: Non-trainable parameters merged with
                ``trainable_params`` at every forward.
            batch_keys: Ordered tuple of tensor keys the collator emits
                (discovered via :meth:`_discover_batch_keys`).
            return_logits: When ``True``, the closure returns
                ``(loss, logits)`` instead of just ``loss``.
            with_metrics: When ``True``, the closure returns ``(loss, aux_dict)``
                via :meth:`compute_per_example_loss_and_metrics`; the caller
                pairs this with ``_create_grad_fn(..., has_aux=True)`` so
                ``clipped_grad`` forwards ``aux_dict`` into
                ``ClippedGradAux.loss_aux``.

        Returns:
            ``(per_example_loss_fn, batch_argnums)``.
        """
        keys = batch_keys

        def _call(merged: dict[str, Tensor], inputs: dict[str, Tensor]) -> Any:
            if with_metrics:
                return self.compute_per_example_loss_and_metrics(fmodel, merged, inputs)
            return self.compute_per_example_loss(
                fmodel, merged, inputs, return_logits=return_logits
            )

        def per_example_loss(
            trainable: dict[str, Tensor],
            *batch_args: Tensor,
        ) -> Any:
            # bf16 autocast (when enabled) is applied by the *caller* around the
            # ``grad_fn`` call — the PyTorch idiom ``with autocast(): grad_fn(...)``
            # — NOT here.  Entering autocast inside the function functorch
            # differentiates is silently ignored on MPS (the AutocastMPS key
            # isn't threaded through the ``grad`` transform), so it must wrap the
            # outer ``vmap(grad)`` execution.  See ``_autocast_ctx``.
            merged = {**frozen_params, **trainable}
            inputs = dict(zip(keys, batch_args, strict=True))
            return _call(merged, inputs)

        # ``torch.compile`` is injected into the tensor-only microbatch kernel
        # in ``_create_grad_fn`` (``vmap(grad(loss))`` + clip + reduction), NOT
        # applied to this inner loss. Compiling the loss and then applying
        # ``vmap(grad)`` outside is the unsupported ``grad(compiled_fn)`` pattern
        # — dynamo raises "Unsupported functorch tracing attempt" and silently
        # falls back to eager, so it bought nothing (verified: 1.05x vs 2.0x).
        # The loss therefore stays eager here; the whole transform is compiled
        # one level up, where functorch lives *inside* the compiled region.
        return per_example_loss, tuple(range(1, 1 + len(keys)))

    def _get_eval_per_example_loss_fn(
        self,
        *,
        return_logits: bool = True,
    ) -> tuple[Callable[..., Any], tuple[int, ...], tuple[str, ...]]:
        """Return a vmap'd per-example eval closure (cached) plus its batch keys.

        Used by :meth:`prediction_step` when the caller needs real per-example
        losses. With ``return_logits=True`` the closure returns
        ``(per_example_loss, logits)`` for one example; ``vmap`` over the batch
        produces a 1-D loss tensor plus batched logits in a single forward pass.
        ``return_logits=False`` builds a separate loss-only closure so callers
        that discard predictions do not retain a logits result.

        Cached per model identity — invalidated when ``self._model``
        rebinds (the cache key is the live ``self._model`` reference).
        During an active training run the trainer already has
        ``ctx.fmodel`` / ``ctx.frozen_params`` populated; outside
        training we run ``make_functional(self._model)`` once on first
        use and reuse the result.
        """
        cached_fn = (
            self._eval_per_example_loss_fn
            if return_logits
            else self._eval_per_example_loss_only_fn
        )
        cached_model = (
            self._eval_per_example_loss_fn_model
            if return_logits
            else self._eval_per_example_loss_only_fn_model
        )
        if cached_fn is not None and cached_model is self._model:
            fn, batch_argnums, batch_keys = cached_fn
            return fn, batch_argnums, batch_keys

        ctx = self._ctx
        if ctx is not None:
            fmodel = ctx.fmodel
            frozen_params = ctx.frozen_params
            batch_keys = ctx.batch_keys
        else:
            from opaque.functional import make_functional

            fmodel, _trainable, frozen_params = make_functional(
                self._model, partition_trainable=True
            )
            batch_keys = self._discover_batch_keys()
        fn, batch_argnums = self._build_per_example_loss(
            fmodel, frozen_params, batch_keys, return_logits=return_logits
        )
        vmapped = torch.vmap(fn, in_dims=(None,) + (0,) * len(batch_argnums))
        if return_logits:
            self._eval_per_example_loss_fn = (vmapped, batch_argnums, batch_keys)
            self._eval_per_example_loss_fn_model = self._model
        else:
            self._eval_per_example_loss_only_fn = (vmapped, batch_argnums, batch_keys)
            self._eval_per_example_loss_only_fn_model = self._model
        return vmapped, batch_argnums, batch_keys

    def _discover_batch_keys(self) -> tuple[str, ...]:
        """Discover the ordered tuple of tensor keys the collator emits.

        Runs the resolved collator on a single example and returns the
        keys whose values are ``torch.Tensor``.  This drives both the
        ``batch_argnums`` returned by :meth:`_build_per_example_loss`
        and the positional batch-arg ordering in :meth:`training_step`.
        Non-tensor metadata is excluded — the per-example loss closure
        operates only on tensors.
        """
        if len(self._train_dataset) == 0:
            raise ConfigurationError(
                *("Cannot discover batch keys: train_dataset is empty.",)
            )
        prepared_dataset, prepared_collator = self._prepare_dataset_and_collator(
            self._train_dataset,
            description="training",
            collate_fn=self._data_collator,
        )
        sample = prepared_collator([prepared_dataset[0]])
        if not isinstance(sample, Mapping):
            raise InputTypeError(
                *(f"data_collator must return a mapping; got {type(sample).__name__}.",)
            )
        keys = tuple(k for k, v in sample.items() if isinstance(v, Tensor))
        if not keys:
            raise InputTypeError(
                *(
                    "data_collator produced no tensor outputs on a one-example dry "
                    f"run; got keys={list(sample.keys())!r}.  The DP path requires "
                    "at least one batched tensor.",
                )
            )
        return keys

    def get_train_dataloader(self) -> DataLoader:
        """Train DataLoader.

        During training (``self._ctx`` populated) returns a
        single-pass :class:`PoissonSampler`-backed DataLoader bounded
        by ``ctx.total_steps`` (one sampler instance for the whole
        run; the outer epoch loop just synthesises boundaries for the
        HF callback surface).  Outside training (``ctx is None``,
        inspection mode) returns a standard DataLoader.

        Override in a subclass to plug in a custom sampler — DP
        correctness depends on the sampler producing each example with
        independent probability ``ctx.sample_rate``, so any override
        must preserve that invariant.
        """
        a = self.args
        ctx = self._ctx
        worker_init = self._dataloader_worker_init_fn()
        if ctx is None:
            dataset, base_collator = self._prepare_dataset_and_collator(
                self._train_dataset,
                description="training",
                collate_fn=self._data_collator,
            )
            return DataLoader(
                dataset,
                batch_size=a.per_device_train_batch_size,
                shuffle=False,
                collate_fn=self._maybe_prime_collate(
                    self._resolve_collate_fn(base_collator), dataset
                ),
                num_workers=a.dataloader_num_workers,
                pin_memory=self._pin_memory_enabled(),
                worker_init_fn=worker_init,
                multiprocessing_context=self._dataloader_multiprocessing_context(),
                in_order=a.dataloader_in_order,
            )
        if self._train_dataloader is not None:
            self._callback_handler.train_dataloader = self._train_dataloader
            return self._train_dataloader

        dataset, base_collator = self._prepare_dataset_and_collator(
            self._train_dataset,
            description="training",
            collate_fn=self._data_collator,
        )
        collate_fn = self._resolve_collate_fn(base_collator)
        collate_fn = self._maybe_prime_collate(collate_fn, dataset)

        # Under DDP each rank operates on a disjoint shard of the dataset
        # (``opaque.distributed.local_shard``) and runs the Poisson sampler
        # over its shard's *local* positions.  The sampler key is folded by
        # rank (below) so each rank draws an **independent** Bernoulli(q)
        # mask: with a shared key every rank would select the *same* local
        # offsets, perfectly co-including the records that happen to share a
        # local index across shards — not the i.i.d. global Poisson draw the
        # design intends (the per-record marginal stays Bernoulli(q) either
        # way, so the privacy accounting is unaffected; this is a sampling
        # *diversity* fix). ``build_sampler`` binds each mode's rate off the
        # same values ``_setup_training`` calibrated the accountant with
        # (``ctx.sample_rate`` for plain Poisson; the built amplifier's
        # per-band rate for ``cyclic_poisson``), computed from the same
        # denominator used here (see :meth:`_effective_train_dataset_size`),
        # so runtime and accountant cannot drift apart.
        #
        # Resume caveat (multi-GPU only): the sampler snapshot is
        # self-contained (carries its own key) and is written once on rank
        # 0, so resuming a DDP run currently restores rank 0's per-rank key
        # on every rank, re-introducing the cross-rank correlation after the
        # resume point.  Fully fixing that needs per-rank sampler snapshots;
        # tracked for the multi-GPU work and validated there.
        if self._ddp.world_size > 1:
            from opaque.distributed import local_shard

            world_size = self._ddp.world_size
            # Trims to a multiple of ``world_size`` when needed (see
            # :meth:`_effective_train_dataset_size`), guaranteeing every
            # rank's ``local_shard`` is the same length as the
            # denominator used above.
            effective_n = self._effective_train_dataset_size()
            if effective_n < len(dataset):
                from torch.utils.data import Subset

                log.warning(
                    "Train dataset has %d example(s), not evenly divisible by "
                    "world_size=%d; dropping %d tail example(s) to %d. The "
                    "accounting sample-rate denominator is the trimmed "
                    "length, not len(train_dataset).",
                    len(dataset),
                    world_size,
                    len(dataset) - effective_n,
                    effective_n,
                )
                dataset = Subset(dataset, range(effective_n))
            dataset = local_shard(
                dataset,
                rank=self._ddp.rank,
                world_size=world_size,
            )

        # ONE sampler for the whole run.  Resume installs
        # ``ctx.current_sampler`` from a registry-deserialised snapshot
        # before calling here, so the loader picks up the right cursor;
        # otherwise build a fresh sampler bound to the resolved
        # ``sampling_mode``.  Five modes are reachable through
        # ``TrainingArguments`` (validated by ``_ALLOWED_SAMPLERS``):
        # ``poisson`` (DP-SGD + ``mf_identity``) or its explicit
        # ``k_out_of_t`` alternative (gaussian), ``cyclic_poisson`` (the
        # ``mf_band`` default) or its explicit ``b_min_sep`` alternative, and
        # ``balls_in_bins`` (other MF mechanisms).  ``build_sampler`` also
        # constructs ``sequential`` for subclasses that call it directly, but
        # that is not exposed as a config ``sampling_mode`` value (no
        # matching accountant amplifier) and the config layer rejects it.
        # The sampler iterates end-to-end without per-epoch
        # re-instantiation; the outer epoch loop is purely a synthetic
        # boundary layer for HF callbacks.
        if ctx.current_sampler is None:
            from opaque.random import fold_in

            sampler_key = key(a.data_seed if a.data_seed is not None else a.seed)
            if ctx.sampler_restart_step is not None:
                # Restart ignored Poisson state on a cursor-derived stream so
                # the post-resume steps do not replay the Bernoulli draws the
                # discarded prefix already spent.
                sampler_key = fold_in(
                    sampler_key,
                    IGNORE_DATA_SKIP_STREAM_FOLD,
                    ctx.sampler_restart_step,
                )
            # Per-rank independent sampling: fold the rank into the key so
            # each shard draws a distinct Bernoulli(q) mask (see the block
            # comment above).  No-op at world_size == 1, preserving the
            # single-process seeding bit-for-bit.
            if self._ddp.world_size > 1:
                sampler_key = fold_in(sampler_key, self._ddp.rank)
            ctx.current_sampler = _dpftrl.build_sampler(
                sampling_mode=a.sampling_mode,
                dataset=dataset,
                sample_rate=ctx.sample_rate,
                n_steps=ctx.total_steps,
                key=sampler_key,
                sampling_kwargs=(
                    a.sampling_kwargs if isinstance(a.sampling_kwargs, dict) else None
                ),
                mf=ctx.mf,
                noise_multiplier=ctx.noise_multiplier,
                num_bins=ctx.expected_steps_per_epoch,
                expected_batch_size=int(a.train_batch_size),
            )
        sampler = ctx.current_sampler

        kwargs: dict[str, Any] = {
            "batch_sampler": sampler,
            "collate_fn": collate_fn,
            "num_workers": a.dataloader_num_workers,
            "pin_memory": self._pin_memory_enabled(),
            "worker_init_fn": worker_init,
            "multiprocessing_context": self._dataloader_multiprocessing_context(),
            "in_order": a.dataloader_in_order,
        }
        if a.dataloader_num_workers > 0:
            kwargs["persistent_workers"] = a.dataloader_persistent_workers
            if a.dataloader_prefetch_factor is not None:
                kwargs["prefetch_factor"] = a.dataloader_prefetch_factor

        self._train_dataloader = DataLoader(dataset, **kwargs)
        self._callback_handler.train_dataloader = self._train_dataloader
        return self._train_dataloader

    def get_eval_dataloader(self, eval_dataset: Dataset | None = None) -> DataLoader:
        """Standard DataLoader for evaluation.

        under DDP, the eval dataset is sharded into a contiguous
        per-rank slice via ``opaque.distributed.local_shard`` so each rank
        evaluates a disjoint subset.  Per-batch losses / predictions are
        reduced / gathered in :meth:`evaluation_loop` before metrics are
        computed.
        """
        raw_dataset = eval_dataset if eval_dataset is not None else self._eval_dataset
        if eval_dataset is None and self._eval_dataloader is not None:
            return self._eval_dataloader

        dataset, base_collator = self._prepare_dataset_and_collator(
            raw_dataset,
            description="evaluation",
            collate_fn=self._data_collator,
        )

        if self._ddp.world_size > 1:
            from opaque.distributed import local_shard

            # ``eval_do_concat_batches=False`` returns per-batch *lists*; the
            # gather then runs one collective per list element, and ranks with
            # different batch counts issue a different number of collectives →
            # hang.  Reject it under DDP rather than deadlock.
            if not self.args.eval_do_concat_batches:
                raise ConfigurationError(
                    *(
                        "eval_do_concat_batches=False is not supported under "
                        "distributed evaluation (world_size>1): the per-batch list "
                        "gather issues a data-dependent number of collectives and "
                        "can deadlock.  Set eval_do_concat_batches=True for DDP eval.",
                    )
                )
            dataset = local_shard(
                dataset,
                rank=self._ddp.rank,
                world_size=self._ddp.world_size,
            )

        eval_collate = self._resolve_collate_fn(base_collator)
        eval_collate = self._maybe_prime_collate(eval_collate, dataset)
        kwargs: dict[str, Any] = {
            "batch_size": self.args.per_device_eval_batch_size,
            "shuffle": False,
            "collate_fn": eval_collate,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self._pin_memory_enabled(),
            "drop_last": self.args.dataloader_drop_last,
            "worker_init_fn": self._dataloader_worker_init_fn(),
            "multiprocessing_context": self._dataloader_multiprocessing_context(),
            "in_order": self.args.dataloader_in_order,
        }
        if self.args.dataloader_num_workers > 0:
            kwargs["persistent_workers"] = self.args.dataloader_persistent_workers
            if self.args.dataloader_prefetch_factor is not None:
                kwargs["prefetch_factor"] = self.args.dataloader_prefetch_factor

        loader = DataLoader(dataset, **kwargs)
        if eval_dataset is None and self.args.dataloader_persistent_workers:
            self._eval_dataloader = loader
        return loader

    def _dataloader_multiprocessing_context(self) -> str | None:
        """Resolve DataLoader worker context with HF's MPS-safe default."""
        context = self.args.dataloader_multiprocessing_context
        if context is not None:
            return context
        if self._device.type == "mps" and self.args.dataloader_num_workers > 1:
            return "fork"
        return None

    def _dataloader_worker_init_fn(self) -> Callable[[int], None] | None:
        """Build a worker-init callable that re-seeds each worker process.

        HF parity (``Trainer._get_dataloader``): we forward each worker
        a deterministic seed derived from ``args.seed`` so that
        non-DP randomness inside a custom ``data_collator`` (or any
        torch / numpy / random call inside ``__getitem__``) is
        reproducible run-to-run.  Returns ``None`` when single-process
        loading is in use — :class:`torch.utils.data.DataLoader`
        ignores ``worker_init_fn`` in that case anyway, but spelling
        ``None`` keeps the spawn path explicit.

        DP correctness is unaffected: the per-step DP RNG chain
        (``key(args.seed)`` folded with ``state.epoch`` /
        ``iter_count``) is independent of ``torch`` / NumPy / Python's
        global generators.
        """
        a = self.args
        if a.dataloader_num_workers <= 0:
            return None
        # ``seed_worker(worker_id, num_workers, rank)`` consumes
        # ``torch.initial_seed()`` inside the worker process — which
        # itself is set by PyTorch from the base RNG seeded via
        # ``set_seed(args.seed)`` at trainer construction.
        return functools.partial(
            seed_worker,
            num_workers=a.dataloader_num_workers,
            rank=self._ddp.rank,
        )

    def _maybe_empty_device_cache(self) -> None:
        steps = self.args.torch_empty_cache_steps
        if steps is None or steps <= 0:
            return
        if self.state.global_step % steps != 0:
            return
        device_type = self._device.type
        if device_type == "cuda":
            torch.cuda.empty_cache()
        elif device_type == "mps":
            torch.mps.empty_cache()
        # cpu: no-op (nothing to evict)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def create_optimizer(
        self,
        trainable_params: dict[str, Tensor],
        lr_schedule: Callable[[int], float],
        clip_state: Any = None,
        noise_multiplier: float = 0.0,
    ) -> tuple[Any, Any]:
        """Create functional optimizer and initial state.

        Dispatches via :func:`opaque.api.transformers.trainer._optim.build_optimizer`,
        which resolves ``args.optim`` (canonical opaque name or HF alias)
        and forwards HF-canonical fields (``learning_rate``,
        ``weight_decay``, ``adam_beta1``/``adam_beta2``, ``adam_epsilon``)
        to the underlying opaque factory.  Opaque-only knobs (for example
        ``noise_bias_correction``, ``decoupled_weight_decay``,
        ``update_rms_clip``) are supplied through ``args.optim_args``,
        which ``TrainingArguments.__post_init__`` has already normalized
        from any of {Mapping, JSON object string, HF "key=value,..."
        string, None} to ``dict[str, Any] | None``.  Opaque factories
        raise ``TypeError`` on unknown keys, so typos surface immediately.

        ``clip_state`` and ``noise_multiplier`` are accepted for
        signature stability with subclasses; the wrapper-pytree noise
        flow makes them irrelevant at construction time (sensitivity
        flows through ``ClippedPytree.max_norm`` at every step).
        """
        from ._optim import build_optimizer

        del clip_state, noise_multiplier  # see docstring
        a = self.args
        extra = dict(a.optim_args or {})
        fac = self._functional_optimizer_factory
        if fac is not None:
            factory, init_kw = fac
            merged = {**dict(init_kw), **extra}
            merged.pop("lr", None)
            opt = factory(lr=lr_schedule, **merged)
            return opt, opt.init(trainable_params)
        opt = build_optimizer(a, lr_schedule, extra_kwargs=extra)
        return opt, opt.init(trainable_params)

    def create_scheduler(self, num_training_steps: int) -> Callable[[int], float]:
        """Build the LR schedule for the run.

        Dispatches via :func:`opaque.api.transformers.trainer._scheduler.build_lr_schedule`,
        which reads ``args.lr_scheduler``, ``args.lr_scheduler_kwargs``,
        and ``args.warmup_steps``.  Override in a subclass to supply a
        different schedule.
        """
        return build_lr_schedule(self.args, num_training_steps)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(self, logs: dict[str, Any], start_time: float | None = None) -> None:
        """Record metrics and fire ``on_log`` callback.

        HF parity:
        - ``start_time`` is the training-loop start timestamp; when
          provided and ``include_num_input_tokens_seen`` is set, the
          live tokens/sec rate is injected into the log row (mirrors
          HF 5.1's ``Trainer.log`` signature).
        - ``num_input_tokens_seen`` is appended to every log row when
          tracking is enabled — not just training-step rows.
        """
        if not _distributed.should_log(self.args, self._ddp):
            return
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen != "no":
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen
            if start_time is not None:
                logs.update(
                    speed_metrics(
                        "train",
                        start_time,
                        num_tokens=self.state.num_input_tokens_seen,
                    )
                )
        output = {**logs, "step": self.state.global_step}
        self.state.log_history.append(output)
        self._control = self._callback_handler.on_log(
            self.args, self.state, self._control, logs=output
        )

    def _maybe_log_save_evaluate(
        self,
        ctx: _TrainingContext,
        step_result: dict[str, Any],
        global_step: int,
        *,
        ignore_keys_for_eval: list[str] | None = None,
    ) -> None:
        """Act on ``control.should_log/evaluate/save`` populated by
        :class:`DefaultFlowCallback.on_step_end` (and possibly overridden
        by user callbacks)."""
        ctrl = self._control

        if ctrl.should_log:
            # Re-wrap the accountant in ``acc.cached`` at each log boundary
            # so subsequent ``epsilon_at`` queries within this window are
            # amortized.  Mirrors ``_after_evaluate`` and the manual loop.
            ctx.accounting = acc.cached(ctx.accounting)
            epsilon = ctx.accounting.epsilon_at(ctx.target_delta)
            # Stop-at-ε (fallback): owns the stop only when no crossing step
            # was predicted (``ctx.stop_at_step is None`` — Monte-Carlo
            # accountants, unreachable target); otherwise the in-loop integer
            # check is the single stop owner and this block only computed ε
            # for the log line.  Fixed-NM path only; the calibrated NM was
            # sized to hit target_epsilon at max_steps, so stopping earlier
            # would mean we over-noised the run.
            a = self.args
            if (
                ctx.stop_at_step is None
                and not ctx.is_horizon_process
                and a.privacy_noise_multiplier is not None
                and a.privacy_noise_multiplier > 0
                and a.privacy_target_epsilon is not None
                and epsilon >= a.privacy_target_epsilon
            ):
                # Horizon configurations cannot reach this fallback: combining
                # fixed noise with a target is rejected during validation.
                self.state.privacy_target_epsilon_reached = True
                self._control.should_training_stop = True
                log.info(
                    "stop-at-ε hit: ε=%g >= target=%g at step %d",
                    epsilon,
                    a.privacy_target_epsilon,
                    global_step,
                )
            # HF parity: ``loss`` is the *average* per-step loss across the
            # window since the last log boundary, not the per-step
            # instantaneous value.  Smooths out per-step variance that
            # would otherwise dominate the displayed curve.
            window = max(1, global_step - self._globalstep_last_logged)
            tr_loss_scalar = self._tr_loss.item()
            smoothed_loss = tr_loss_scalar / window
            # HF parity: accumulate into _total_loss_scalar, then reset tr_loss.
            self._total_loss_scalar += tr_loss_scalar
            self._tr_loss = torch.zeros_like(self._tr_loss)
            self._globalstep_last_logged = global_step
            # HF parity: log the LR that was just applied to the optimizer
            # update we performed for ``global_step``.  Inside torchopt the
            # schedule's step counter is incremented on every ``update`` so
            # the LR consumed by the update at iteration N was
            # ``schedule(N - 1)``; ``global_step`` has already been bumped
            # to N by the time we log here.
            #
            # ``ctrl.should_log`` is only set after at least one optimizer
            # update has fired (the only setter is
            # :class:`DefaultFlowCallback.on_step_end`, which runs *after*
            # ``global_step`` is bumped to ``>=1``).  Assert the invariant
            # here so a future caller that flips ``should_log`` from
            # ``on_train_begin`` (where ``global_step == 0``) gets a
            # loud failure instead of a silent ``schedule(-1)`` lookup.
            assert global_step >= 1, (
                f"_maybe_log_save_evaluate fired with global_step="
                f"{global_step}; expected >=1 (DefaultFlowCallback only "
                f"sets should_log post-step)."
            )
            logs = {
                "loss": smoothed_loss,
                "batch_size": step_result.get("batch_size", 0),
                "grad_norm": step_result.get("grad_norm", 0.0),
                "learning_rate": ctx.lr_schedule(global_step - 1),
                "privacy_epsilon": epsilon,
                "privacy_delta": ctx.target_delta,
                "privacy_clip_rate": step_result.get("clip_rate", 0.0),
                "privacy_clipping_norm": step_result.get("clipping_norm", 0.0),
                "privacy_noise_std": step_result.get("noise_std", 0.0),
                "privacy_noise_multiplier": ctx.noise_multiplier,
            }
            if "clip_rate_max" in step_result:
                logs["privacy_clip_rate_max"] = step_result["clip_rate_max"]
            if "clipped_grad_norm" in step_result:
                logs["privacy_clipped_grad_norm_mean"] = step_result[
                    "clipped_grad_norm"
                ]
            for group_name, group_values in step_result.get(
                "group_metrics", {}
            ).items():
                for metric_name, value in group_values.items():
                    logs[f"privacy_group_{group_name}_{metric_name}"] = value
            # Subclass training telemetry (e.g. DPO ``rewards/*``) surfaced by
            # ``training_step`` from the clipped-grad ``loss_aux`` channel.
            logs.update(step_result.get("loss_aux", {}))
            # Opaque per-step performance metrics (step_time_sec,
            # samples_per_second, memory_*, clip_sec / noise_sec /
            # optimizer_sec from ``sp.mark(...)``).  Bare keys; the
            # reporting-callback rewriter wraps them under ``train/``.
            if self._perf_tracker.train.last is not None:
                logs.update(self._perf_tracker.train.last.to_dict())
            self.log(logs, start_time=self._train_start_time)
            ctrl = self._control
            ctrl.should_log = False

        if ctrl.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            # ``evaluate()`` fires ``on_evaluate`` from inside
            # ``_after_evaluate``; :class:`BestModelSaveCallback` (auto-
            # injected when ``save_strategy="best"``) may have set
            # ``should_save`` there.  Refresh the local handle accordingly.
            ctrl = self._control
            is_new_best_metric = self._update_best_metric(metrics, global_step)
            if is_new_best_metric and self.args.load_best_model_at_end:
                # The regular save cadence can be less frequent than
                # evaluation, so materialize the parameters just evaluated.
                ctrl.should_save = True
            ctrl.should_evaluate = False

        if ctrl.should_save:
            self._save_checkpoint()
            ctrl.should_save = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _autocast_ctx(self) -> contextlib.AbstractContextManager:
        """Mixed-precision context to wrap the DP ``grad_fn`` call in.

        bf16 autocast must wrap the *outer* ``vmap(grad)`` call — the PyTorch
        idiom ``with autocast(): grad_fn(...)`` — not sit inside the grad'd
        loss.  An autocast entered inside the function functorch differentiates
        is silently ignored on MPS (the ``AutocastMPS`` key isn't threaded
        through the ``grad`` transform), so the cast must happen here, around the
        whole transform.  Returns a ``nullcontext`` when bf16 is off.
        """
        if self._amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self._device.type, dtype=self._amp_dtype)

    def _grad_compiler(self) -> Callable | None:
        """Return a strict compiler for the tensor-only gradient chunk.

        The clipping factories keep variable-size microbatch orchestration,
        diagnostics, and state updates eager.  The injected compiler sees only
        ``vmap(grad_and_value)`` plus per-example clipping and reduction, with a
        symbolic leading chunk dimension. This avoids specializing on realized
        Poisson or remainder batch sizes; PyTorch may retain a size-one variant.
        """
        a = self.args
        if not a.torch_compile:
            return None
        caps = device_capabilities(self._device)
        if not caps.supports_compile:
            log.warning(
                "torch_compile=True but device %r has no supported torch.compile "
                "backend; running the DP grad step eager.",
                self._device.type,
            )
            return None
        backend = a.torch_compile_backend or caps.recommended_compile_backend
        mode = a.torch_compile_mode or "default"
        log.info(
            "torch.compile enabled on strict DP gradient chunks: backend=%s mode=%s "
            "device=%s (inductor → Triton on CUDA, Metal on MPS).",
            backend,
            mode,
            self._device.type,
        )
        return lambda fn: _compile_strict_chunk(
            fn,
            backend=backend,
            mode=mode,
        )

    def _create_grad_fn(
        self,
        loss_fn: Callable[..., Any],
        batch_argnums: tuple[int, ...],
        a: TrainingArguments,
        clip_norm: Any,
        expected_batch_size: int,
        microbatch_size: int,
        *,
        quantile_noise_key: RngKey,
        has_aux: bool = False,
    ) -> tuple[Callable[..., Any], Any]:
        """Create the clipped gradient function based on clipping mode.

        ``loss_fn`` stays eager as a Python callable.  When compilation is
        enabled, the clipping factory compiles its tensor-only per-microbatch
        ``vmap(grad)+clip+reduce`` kernel and keeps orchestration/state eager.

        When ``has_aux`` is set, ``loss_fn`` returns ``(loss, aux_dict)`` and the
        per-example ``aux_dict`` is forwarded into ``ClippedGradAux.loss_aux``.
        """
        ca = a.clipping_kwargs
        target_quantile = float(ca.get("target_quantile", 0.5))
        clipping_norm_max = float(ca.get("clipping_norm_max", 10.0))
        auto_gamma = float(ca.get("gamma", 0.01))
        compiler = self._grad_compiler()

        if a.clipping_mode == "adaptive":
            grad_fn, state = adaptive_clipped_grad(
                loss_fn,
                argnums=0,
                has_aux=has_aux,
                batch_argnums=batch_argnums,
                initial_clipping_norm=clip_norm,
                target_quantile=target_quantile,
                clipping_norm_max=clipping_norm_max,
                microbatch_size=microbatch_size,
                return_aux=True,
                key=quantile_noise_key,
                normalize_by=expected_batch_size,
                _chunk_compiler=compiler,
            )
        elif a.clipping_mode == "auto":
            grad_fn, state = auto_clipped_grad(
                loss_fn,
                argnums=0,
                has_aux=has_aux,
                batch_argnums=batch_argnums,
                R=clip_norm,
                gamma=auto_gamma,
                normalize_by=expected_batch_size,
                microbatch_size=microbatch_size,
                return_aux=True,
                _chunk_compiler=compiler,
            )
        else:
            grad_fn, state = clipped_grad(
                loss_fn,
                argnums=0,
                has_aux=has_aux,
                batch_argnums=batch_argnums,
                clipping_norm=clip_norm,
                normalize_by=expected_batch_size,
                microbatch_size=microbatch_size,
                return_aux=True,
                _chunk_compiler=compiler,
            )
        return grad_fn, state

    def _build_mechanism(
        self,
        a: TrainingArguments,
        expected_batch_size: int,
        sample_rate: float,
        clip_norm: Any,
        dataset_size: int,
        *,
        n_steps: int,
        num_bins: int,
        mf_amplifier_factory: Callable[[float], Any] | None = None,
    ) -> Callable[..., Any]:
        """Build the privacy accounting mechanism chain.

        DP-SGD Poisson mechanisms return independently composable steps.
        DP-SGD k-out-of-t and all DP-FTRL mechanisms return complete
        whole-horizon processes and must not be composed per step.
        """
        if a.privacy_noise_mechanism != "gaussian":
            if mf_amplifier_factory is None:
                raise OperationError(
                    *(
                        "_build_mechanism reached the DP-FTRL branch without an "
                        "amplifier factory; _setup_training should populate it.",
                    )
                )
            return mf_amplifier_factory

        num_groups = clip_norm.num_groups if isinstance(clip_norm, PerGroup) else 1

        base = dpsgd_acc.gaussian
        if a.clipping_mode == "adaptive":
            _base = base

            def base(nm, _b=_base):
                return dpsgd_acc.adaclip(
                    _b(nm),
                    expected_batch_size=expected_batch_size,
                    num_groups=num_groups,
                )

        # Non-private substitution: at noise_multiplier == 0 the inner element
        # becomes ``acc.nonprivate()``, which composes through the poisson /
        # adaclip wrappers below to yield ``epsilon == inf`` — exactly the
        # documented ``noise_multiplier=0`` usage, no special-casing needed.
        _dp_element = base

        def _unamplified(nm, _b=_dp_element):
            return acc.nonprivate() if nm == 0.0 else _b(nm)

        sk = a.sampling_kwargs if isinstance(a.sampling_kwargs, dict) else {}
        tb_raw = sk.get("truncated_batch_size")
        tb_cap = int(tb_raw) if tb_raw is not None else None

        if a.sampling_mode == "k_out_of_t":
            k_raw = sk.get("k")
            allocation = sk.get("allocation")
            if k_raw is None or allocation is None:
                raise ConfigurationError(
                    *(
                        "sampling_mode='k_out_of_t' requires sampling_kwargs with "
                        "'k' and 'allocation'.",
                    )
                )
            if allocation not in ("block", "total"):
                raise ConfigurationError(
                    *(
                        "sampling_kwargs['allocation'] must be 'block' or 'total', "
                        f"got {allocation!r}.",
                    )
                )

            def mechanism(
                nm,
                _u=_unamplified,
                _k=int(k_raw),
                _t=n_steps,
                _allocation=allocation,
            ):
                return dpsgd_acc.k_out_of_t(
                    _u(nm),
                    k=_k,
                    t=_t,
                    allocation=_allocation,
                )

        elif tb_cap is not None:

            def mechanism(
                nm,
                _u=_unamplified,
                _cap=tb_cap,
                _n=dataset_size,
                _rate=sample_rate,
            ):
                return dpsgd_acc.poisson(
                    _u(nm),
                    _rate,
                    truncated_batch_size=_cap,
                    dataset_size=_n,
                )
        else:

            def mechanism(nm, _u=_unamplified, _rate=sample_rate):
                return dpsgd_acc.poisson(_u(nm), sample_rate=_rate)

        return mechanism

    def _calibrate_noise(
        self,
        a,
        mechanism,
        total_steps,
        target_delta,
        *,
        prefix_accountant: Accountant | None = None,
        resume_horizon_noise_multiplier: float | None = None,
        global_step_already_done: int = 0,
    ):
        """Calibrate or return fixed noise multiplier.

        With a saved accountant, independent mechanisms calibrate over the
        remaining steps with the saved process composed on the left.
        Whole-horizon mechanisms calibrate the complete declared process.
        """
        if resume_horizon_noise_multiplier is not None:
            if a.privacy_noise_multiplier is not None and _drift_differs(
                float(a.privacy_noise_multiplier),
                resume_horizon_noise_multiplier,
            ):
                raise CheckpointError(
                    *(
                        "Whole-horizon resume forbids privacy_noise_multiplier "
                        f"drift: saved={resume_horizon_noise_multiplier!r}, "
                        f"current={a.privacy_noise_multiplier!r}. Restart from "
                        "scratch to use a different fixed multiplier.",
                    )
                )
            log.info(
                "Restoring whole-horizon noise multiplier: %.4f",
                resume_horizon_noise_multiplier,
            )
            return resume_horizon_noise_multiplier
        if a.privacy_noise_multiplier is not None:
            log.info("Using fixed noise multiplier: %.4f", a.privacy_noise_multiplier)
            return a.privacy_noise_multiplier

        if prefix_accountant is None or global_step_already_done == 0:
            log.info(
                "Calibrating privacy (target eps=%.2f, delta=%.2e)...",
                a.privacy_target_epsilon,
                target_delta,
            )

            def objective(nm, _mechanism=mechanism, _steps=total_steps):
                process = _mechanism(nm)
                return (
                    process
                    if isinstance(process, DpHorizonProcess)
                    else process * _steps
                )
        else:
            remaining_steps = max(1, total_steps - global_step_already_done)
            log.info(
                "Recalibrating remaining %d/%d steps over saved accountant "
                "(target eps=%.2f, delta=%.2e)...",
                remaining_steps,
                total_steps,
                a.privacy_target_epsilon,
                target_delta,
            )
            prefix_process = prefix_accountant.process

            def objective(nm, _prefix=prefix_process, _rem=remaining_steps):
                process = mechanism(nm)
                return (
                    process
                    if isinstance(process, DpHorizonProcess)
                    else _prefix | (process * _rem)
                )

        ecal = a.noise_calibration_kwargs
        param_min = float(ecal["min"])
        param_max = float(ecal["max"])
        result = cal.calibrate(
            cal.epsilon_budget(a.privacy_target_epsilon, delta=target_delta),
            objective,
            param_min=param_min,
            param_max=param_max,
            tolerance=float(ecal["tolerance"]),
        )
        log.info(
            "Calibrated: noise_multiplier=%.4f, achieved eps=%.3f (converged=%s)",
            result.param,
            result.achieved,
            result.converged,
        )
        self.state.privacy_calibration_noise_multiplier = float(result.param)
        self.state.privacy_calibration_achieved_epsilon = float(result.achieved)
        self.state.privacy_calibration_converged = bool(result.converged)
        return result.param

    def _restore_params(self, trainable_params: dict[str, Tensor]) -> None:
        """Load trained parameters back into the nn.Module.

        Validates that every ``trainable_params`` key exists in the model
        (catching typo'd names in subclass overrides before the strict
        load) and overwrites those entries in the model's state dict.

        Keys come from ``trainable_params`` (the set that was actually
        trained) and are validated against the model's ``state_dict`` rather
        than re-derived from the live module's ``requires_grad`` flags, which
        are not a reliable record of what was trained under the functional
        training path.
        """
        model_keys = set(self._model.state_dict())
        unexpected = set(trainable_params) - model_keys
        if unexpected:
            raise ConfigurationError(
                *(
                    "Trainer._restore_params: trainable_params contains keys not "
                    f"present in the model: {sorted(unexpected)}",
                )
            )
        state_dict = self._model.state_dict()
        for name, tensor in trainable_params.items():
            state_dict[name] = tensor.detach()
        self._model.load_state_dict(state_dict, strict=True)

    @staticmethod
    def _published_params(ctx: _TrainingContext) -> dict[str, Tensor]:
        """Return the parameters users should evaluate and export."""
        if isinstance(ctx.opt_state, ScheduleFreeState):
            return ctx.opt_state.x
        return ctx.trainable_params

    @contextlib.contextmanager
    def _use_published_functional_params(self) -> Iterator[None]:
        """Temporarily expose published parameters to the functional eval path."""
        ctx = self._ctx
        if ctx is None or not isinstance(ctx.opt_state, ScheduleFreeState):
            yield
            return
        training_params = ctx.trainable_params
        ctx.trainable_params = ctx.opt_state.x
        try:
            yield
        finally:
            ctx.trainable_params = training_params

    @contextlib.contextmanager
    def _use_published_model_params(self) -> Iterator[None]:
        """Temporarily expose published parameters on the live module."""
        ctx = self._ctx
        if ctx is None or not isinstance(ctx.opt_state, ScheduleFreeState):
            yield
            return
        self._restore_params(ctx.opt_state.x)
        try:
            yield
        finally:
            self._restore_params(ctx.trainable_params)

    # ------------------------------------------------------------------
    # Save / checkpoint
    # ------------------------------------------------------------------

    def _effective_train_dataset_size(self) -> int:
        """Length of ``self._train_dataset`` for DDP sharding and accounting.

        Single source of truth for the training-time dataset size, and for
        the accounting sample-rate denominator. Under DDP, every rank needs
        an identical-length shard (fixed-order samplers need this to keep
        batch counts in sync across ranks), so when ``len(train_dataset)``
        is not an exact multiple of ``world_size`` the
        ``len(train_dataset) % world_size`` tail example(s) are dropped and
        the trimmed length becomes the accounting sample-rate denominator.

        Raises:
            ConfigurationError: If the (possibly trimmed) train dataset is
                empty.
        """
        if self._train_dataset is None:
            return 0
        n = len(self._train_dataset)
        world_size = self._ddp.world_size
        if world_size <= 1:
            return n
        if n % world_size != 0:
            n -= n % world_size
        if n == 0:
            raise ConfigurationError(
                *(
                    "Train dataset is empty; every rank requires at least "
                    f"one example after sharding across world_size="
                    f"{world_size} ranks.",
                )
            )
        return n

    def _steps_breakdown(
        self,
        dataset_size: int,
    ) -> tuple[int, int, int]:
        """Compute ``(steps_per_epoch, total_steps, num_epochs)``.

        Single source of truth for the step math; called from
        ``__init__`` (to populate ``state.max_steps`` /
        ``state.{logging,eval,save}_steps`` before ``on_init_end``)
        and again from ``_setup_training`` (where ``state`` is fully
        repopulated for the actual run).

        Subclasses that resize the train dataset between construction
        and ``train()`` may override ``_predict_total_steps`` to defer
        cadence resolution.
        """
        a = self.args
        sample_rate = a.train_batch_size / max(1, dataset_size)
        steps_per_epoch = math.ceil(1.0 / sample_rate)
        if a.max_steps > 0:
            total = a.max_steps
            num_epochs = math.ceil(total / max(1, steps_per_epoch))
        else:
            total = math.ceil(a.num_train_epochs * steps_per_epoch)
            # The epoch loop must be integral, but its final iteration may be
            # cut short once the fractional-epoch step horizon is reached.
            num_epochs = math.ceil(a.num_train_epochs)
        return steps_per_epoch, total, num_epochs

    def _predict_total_steps(self) -> int:
        """Predict ``total_steps`` for ``state.compute_steps`` at init time.

        Returns the same value as the ``total_steps`` field of
        ``_steps_breakdown(self._effective_train_dataset_size())`` — the
        denominator the actual training run will see, so ``state.max_steps``
        matches the cadence ``_setup_training`` produces.
        Override in subclasses where the dataset isn't sized at construction
        time (e.g. streaming datasets) — return ``0`` to signal "unknown" and
        ``state.compute_steps`` will leave ``state.{logging,eval,save}_steps``
        at HF defaults until ``_setup_training`` recomputes.
        """
        try:
            n = self._effective_train_dataset_size()
        except TypeError:
            return 0
        return self._steps_breakdown(n)[1]

    def _warn_if_existing_output_dir(self) -> None:
        a = self.args
        output_dir = self._effective_output_dir()
        if output_dir is None or a.overwrite_output_dir:
            return
        if not Path(output_dir).is_dir():
            return
        existing = ckpt.list_checkpoints(output_dir)
        if existing:
            log.warning(
                "Output directory %s already contains %d checkpoint(s) and "
                "overwrite_output_dir=False. Pass resume_from_checkpoint to continue, "
                "or set overwrite_output_dir=True to ignore them.",
                output_dir,
                len(existing),
            )

    def _resolve_save_steps_int(self, total_steps: int) -> int:
        """Resolve ``save_steps`` to an integer.

        ``save_steps`` is a fraction of ``total_steps`` when 0 < value < 1
        (HF parity); otherwise treated as an absolute integer step count.
        """
        v = self.args.save_steps
        if v is None:
            return 0
        if 0 < v < 1:
            return max(1, round(total_steps * float(v)))
        return max(1, int(v))

    def _update_best_metric(
        self,
        eval_metrics: dict[str, Any],
        global_step: int,
    ) -> bool:
        """Update ``state.best_*`` and report whether the metric improved.

        ``BestModelSaveCallback`` independently decides whether to set
        ``control.should_save`` for ``save_strategy='best'`` (it runs at
        ``on_evaluate``, before this method updates ``state.best_metric``);
        both use :func:`is_metric_improved` against the same operands so
        the two decisions can't drift.
        """
        a = self.args
        if a.metric_for_best_model is None:
            return False
        resolved = resolve_eval_metric(eval_metrics, a.metric_for_best_model)
        if resolved is None:
            key = a.metric_for_best_model
            if not key.startswith("eval_"):
                key = f"eval_{key}"
            raise ConfigurationError(
                *(
                    f"The `metric_for_best_model` training argument is set to {key!r}, "
                    "which is not found in the evaluation metrics. The available "
                    f"evaluation metrics are: {list(eval_metrics.keys())}. Consider "
                    "changing `metric_for_best_model`.",
                )
            )
        _, value = resolved
        if not is_metric_improved(
            value, self.state.best_metric, self.args.greater_is_better
        ):
            return False
        self.state.best_metric = value
        if self.args.save_strategy in {"steps", "epoch", "best"}:
            self.state.best_global_step = global_step
        return True

    def _load_best_model(self, ctx: _TrainingContext) -> None:
        """Restore best-checkpoint weights into the underlying ``nn.Module``.

        Ordering contract (HF parity):

        1. Read the saved state dict from ``state.best_model_checkpoint``.
        2. Mutate ``self._model`` via ``load_state_dict(...)`` — the
           caller verified the model's parameter set matches what was
           saved.
        3. Rebuild ``ctx.trainable_params`` from the freshly mutated
           module so the functional path picks up the loaded weights.

        Mutating the module first means a ``save_model()`` call between
        ``_load_best_model`` and the ``train()`` ``finally`` block sees
        the loaded weights even before ``_restore_params`` runs.

        Raises ``RuntimeError`` when ``load_best_model_at_end=True`` was
        requested but the best-checkpoint contract can't be honored —
        no best checkpoint was recorded (eval never improved) or the
        recorded directory has no weights file.  Soft-failing here
        leaves the user with the last-trained weights silently masquerading
        as "best", which is exactly what the flag is meant to prevent.
        """
        ckpt_dir = self.state.best_model_checkpoint
        if ckpt_dir is None:
            raise OperationError(
                *(
                    "load_best_model_at_end=True but no best checkpoint was recorded "
                    "during training (eval never improved on metric_for_best_model="
                    f"{self.args.metric_for_best_model!r}).  Either disable "
                    "load_best_model_at_end, or verify the eval/metric configuration "
                    "produces at least one improving step.",
                )
            )
        log.info("Loading best model from %s", ckpt_dir)
        new_state, mutated = self._read_weights_file(ckpt_dir)
        if not new_state and not mutated:
            raise OperationError(
                *(
                    f"load_best_model_at_end=True: best checkpoint recorded at "
                    f"{ckpt_dir!r} but no weights file (model.safetensors / "
                    "pytorch_model.bin / sharded index) was found there.  The "
                    "directory may have been pruned or moved between save and "
                    "end-of-train load.",
                )
            )

        # Mutate the underlying module so ``save_model()`` and any callback
        # firing on ``on_train_end`` observe the best weights immediately.
        # Sharded loads have already mutated the model in place — skip
        # the redundant ``load_state_dict`` call.  PEFT detection drives
        # ``strict``: adapter dirs ship only adapter weights, full-model
        # dirs ship the full state dict.
        if not mutated:
            strict = not self._is_peft
            self._model.load_state_dict(new_state, strict=strict)

        # Rebuild the functional view from the (now-mutated) module.  Keep
        # ``ctx.trainable_params`` keyed by the same names as before — i.e.
        # every parameter that was trainable at the start of the run.
        for name, param in self._model.named_parameters():
            if name in ctx.trainable_params:
                ctx.trainable_params[name] = param.detach().to(self._device)
        if isinstance(ctx.opt_state, ScheduleFreeState):
            ctx.opt_state = dataclasses.replace(
                ctx.opt_state,
                x={
                    name: tensor.detach().clone()
                    for name, tensor in ctx.trainable_params.items()
                },
            )

    def _read_weights_file(
        self,
        ckpt_dir: str,
    ) -> tuple[dict[str, torch.Tensor], bool]:
        """Load weights from a checkpoint directory.

        Supports four on-disk shapes:

        - Single-file safetensors: ``model.safetensors`` (full model) or
          ``adapter_model.safetensors`` (PEFT adapter).
        - Single-file pickle: ``pytorch_model.bin`` /
          ``adapter_model.bin``.
        - Sharded safetensors with index ``model.safetensors.index.json``.
        - Sharded pickle with index ``pytorch_model.bin.index.json``.

        Returns ``(state_dict, model_already_mutated)``.  In the
        single-file cases ``state_dict`` carries the loaded tensors and
        ``model_already_mutated`` is ``False`` (the caller must call
        ``self._model.load_state_dict(state_dict, ...)`` itself).  In
        the sharded case ``load_sharded_checkpoint`` has already
        mutated ``self._model`` in place, so the state dict comes back
        empty and ``model_already_mutated`` is ``True`` to signal "no
        further ``load_state_dict`` needed".  Returns ``({}, False)``
        when no checkpoint files were found.
        """
        from safetensors.torch import load_file as load_safetensors

        from transformers.utils import (
            SAFE_WEIGHTS_INDEX_NAME,
            WEIGHTS_INDEX_NAME,
        )

        # ``load_sharded_checkpoint`` relocated across our supported range:
        # ``transformers.modeling_utils`` in v4, ``transformers.trainer_utils``
        # in v5.  Import from wherever it lives rather than pinning a module.
        try:
            from transformers.modeling_utils import load_sharded_checkpoint
        except ImportError:  # transformers >= 5
            from transformers.trainer_utils import load_sharded_checkpoint

        # Single-file shapes win over sharded indices (HF parity:
        # ``save_pretrained`` writes a single file when the model fits
        # under ``max_shard_size``).
        candidates = [
            Path(ckpt_dir) / ckpt.SAFE_WEIGHTS_NAME,
            Path(ckpt_dir) / "adapter_model.safetensors",
            Path(ckpt_dir) / ckpt.WEIGHTS_NAME,
            Path(ckpt_dir) / "adapter_model.bin",
        ]
        for path in candidates:
            if not path.exists():
                continue
            if path.suffix == ".safetensors":
                return load_safetensors(str(path), device=str(self._device)), False
            # ``weights_only=False``: ``pytorch_model.bin`` is a pickled
            # state-dict that may carry ``torch.dtype`` / ``torch.device``
            # markers (HF historically stamps these into checkpoints) —
            # PyTorch 2.6's safe-load default rejects them.  Pinning the
            # explicit ``False`` keeps the behaviour we tested against.
            return (
                torch.load(str(path), map_location=self._device, weights_only=False),
                False,
            )

        # Sharded checkpoints: ``load_sharded_checkpoint`` mutates the
        # model in place.  ``strict=False`` mirrors the PEFT-friendly
        # single-file path so partial-key checkpoints still load.
        sharded_indices = (
            Path(ckpt_dir) / SAFE_WEIGHTS_INDEX_NAME,
            Path(ckpt_dir) / WEIGHTS_INDEX_NAME,
        )
        if any(p.exists() for p in sharded_indices):
            load_sharded_checkpoint(
                self._model,
                ckpt_dir,
                strict=False,
                prefer_safe=True,
            )
            return {}, True
        return {}, False

    def save_model(
        self,
        output_dir: str | None = None,
        _internal_call: bool = False,
    ) -> None:
        """Restore in-memory params into the model and call ``model.save_pretrained``.

        Mirrors ``Trainer.save_model``; safe to call after training finishes.

        Args:
            output_dir: Directory to save to.  Defaults to ``args.output_dir``.
            _internal_call: Set by :meth:`push_to_hub`'s already rank-gated
                internal save, to skip both push recursion and the barrier
                below (no other rank calls ``save_model`` on that path).
        """
        a = self.args
        target = output_dir or self._effective_output_dir()
        if target is None:
            raise ConfigurationError(
                *("save_model requires output_dir (arg or args.output_dir)",)
            )
        with self._use_published_model_params():
            if self._ctx is not None and not isinstance(
                self._ctx.opt_state, ScheduleFreeState
            ):
                self._restore_params(self._ctx.trainable_params)
            if _distributed.should_save(a, self._ddp):
                Path(target).mkdir(parents=True, exist_ok=True)
                self._save_model_artifacts(target)
                self._save_training_args(target)
                # Privacy provenance travels with every saved model.
                self.save_accountant(target)
        if not _internal_call:
            # Barrier so non-saving ranks don't proceed before the save lands.
            _distributed.barrier(self._ddp)
        # A direct user save with push_to_hub=True also publishes (HF parity).
        # ``_internal_call`` short-circuits the push triggered from within
        # ``push_to_hub`` itself.
        if a.push_to_hub and not _internal_call:
            _hub.push_to_hub(self, commit_message="Model save", revision=a.hub_revision)

    def save_accountant(self, output_dir: str | None = None) -> str | None:
        """Write the privacy accountant to ``accountant.json``; return its path.

        The privacy provenance of a run without the model weights beside it.
        :meth:`save_model` calls this, so a caller that wants only the
        accounting no longer has to re-save the whole model to harvest it.

        Uses the live accountant off the active training context when
        training is mid-flight (most up to date), otherwise the
        trainer-level slot that :meth:`train` populates when the inner loop
        exits — so this is accurate both from inside a callback and after
        ``train()`` returns.

        Args:
            output_dir: Directory to write into, created if missing.
                Defaults to ``args.output_dir``.

        Returns:
            The path written, or ``None`` when there was nothing to write:
            either no training has run yet (no accountant exists), or this
            rank is not the saving rank under ``args.should_save``.

        Note:
            Rank-gated but not a collective — it performs no barrier, so
            ranks are not synchronised on return. Call it on every rank and
            add your own barrier if a later step depends on the file being
            on disk.
        """
        target = output_dir or self._effective_output_dir()
        if target is None:
            raise ConfigurationError(
                *("save_accountant requires output_dir (arg or args.output_dir)",)
            )
        accountant = self._ctx.accounting if self._ctx is not None else self._accountant
        if accountant is None:
            log.info(
                "save_accountant called before any training run; "
                "no accountant to serialise."
            )
            return None
        if not _distributed.should_save(self.args, self._ddp):
            return None
        Path(target).mkdir(parents=True, exist_ok=True)
        self._save_accountant(target, accountant)
        return str(Path(target) / ckpt.DP_ACCOUNTANT_NAME)

    def _save_checkpoint(self, model: Any = None, trial: Any = None) -> str:
        """Write a complete ``checkpoint-<step>`` directory; returns its path.

        Under DDP, the rank-0 process writes shared artefacts (model weights,
        trainer state, training args, accountant, optimizer, DP runtime),
        every rank writes its own RNG snapshot (per-rank file so each rank
        can resume its own non-DP RNG), and each stage ends in a
        failure-propagating :func:`_distributed.checkpoint_barrier`: if any
        rank's segment of a stage raises, every rank raises instead of
        hanging at the barrier.

        Signature mirrors HF ``Trainer._save_checkpoint(model, trial)`` so
        HF-side callbacks that invoke it directly — notably
        :class:`transformers.trainer_jit_checkpoint.JITCheckpointCallback`
        on SIGTERM under ``enable_jit_checkpoint=True`` — compose without
        an adapter. ``model`` and ``trial`` are accepted and ignored: opaque
        tracks the live model and HP-search trial on ``self`` already.
        DP-aware state (accountant, sampler RNG, optimizer) is pulled from
        :attr:`self._ctx` (the active training context); a call outside an
        active ``train()`` invocation raises.
        """
        del model, trial  # HF parity; opaque uses ``self._ctx`` / ``self._model``.
        ctx = self._ctx
        if ctx is None:
            raise OperationError(
                *(
                    "Trainer._save_checkpoint called with no active training "
                    "context. Checkpoints carry DP accountant + sampler RNG + "
                    "optimizer state, which only exist while ``train()`` is "
                    "running.",
                )
            )
        step = int(self.state.global_step)
        a = self.args
        output_dir = self._effective_output_dir()
        if output_dir is None:
            raise ConfigurationError(
                *("Saving checkpoints requires args.output_dir to be set",)
            )
        ckpt_dir = str(Path(output_dir) / f"{ckpt.PREFIX_CHECKPOINT_DIR}-{step}")
        # Atomic publish: write everything into a sibling ``*.tmp`` staging
        # directory, then ``os.replace`` it onto the final ``checkpoint-N``
        # name only once all artefacts (and every rank's RNG snapshot) have
        # landed.  A crash mid-write leaves a ``checkpoint-N.tmp`` dir, which
        # the ``^checkpoint-(\d+)$`` discovery regex ignores — so resume and
        # rotation never select a half-written checkpoint (which, missing
        # ``dp_state.pt``, would otherwise route into the save_only_model
        # noise-reuse path).  Same-directory rename ⇒ atomic on POSIX.
        staging_dir = ckpt_dir + ".tmp"
        # Keep every rank's live module synchronized with the functional
        # training iterate; the saving rank swaps to published weights only
        # while serializing the model artifact below.
        self._restore_params(ctx.trainable_params)
        stage_error: Exception | None = None
        if _distributed.should_save(a, self._ddp):
            try:
                if Path(staging_dir).is_dir():
                    shutil.rmtree(staging_dir)  # stale leftover from a prior crash
                Path(staging_dir).mkdir(parents=True, exist_ok=True)
                with self._use_published_model_params():
                    if not isinstance(ctx.opt_state, ScheduleFreeState):
                        self._restore_params(ctx.trainable_params)
                    self._save_model_artifacts(staging_dir)

                # Register ``best_model_checkpoint`` by *looking up* the folder
                # named ``checkpoint-{best_global_step}``, rather than only when
                # the best step is this save's step.  The best-metric flow
                # materializes intermediate evaluation improvements immediately;
                # this fallback also preserves an existing best directory when a
                # later regular save writes its own trainer state.
                # Resolve *before* writing ``trainer_state.json`` so the file
                # lands once with the final ``best_model_checkpoint`` populated.
                # The path always uses the *final* ``checkpoint-N`` name (not the
                # staging dir), since that's what exists after the rename below.
                if self.state.best_global_step is not None:
                    if self.state.best_global_step == step:
                        # This very checkpoint is the best — point at its final
                        # name (it materialises at the rename).
                        self.state.best_model_checkpoint = ckpt_dir
                    else:
                        best_dir = str(
                            Path(output_dir)
                            / f"{ckpt.PREFIX_CHECKPOINT_DIR}-{self.state.best_global_step}"
                        )
                        if Path(best_dir).is_dir():
                            self.state.best_model_checkpoint = best_dir
                        else:
                            log.debug(
                                "best_global_step=%d but no checkpoint-%d/ folder "
                                "exists (best step fell into a non-saved bucket); "
                                "leaving best_model_checkpoint unset",
                                self.state.best_global_step,
                                self.state.best_global_step,
                            )

                self._save_trainer_state(staging_dir)
                self._save_training_args(staging_dir)
                self._save_accountant(staging_dir, ctx.accounting)
                if not a.save_only_model:
                    self._save_optimizer(staging_dir, ctx)
                    self._save_dp_runtime(staging_dir, ctx)
            except Exception as exc:
                stage_error = exc

        # Per-rank RNG snapshot — every rank, after rank-0 has created the
        # staging directory.
        _distributed.checkpoint_barrier(self._ddp, stage_error)

        stage_error = None
        if not a.save_only_model:
            try:
                self._save_rng_state(staging_dir)
            except Exception as exc:
                stage_error = exc
        # All ranks have finished writing into the staging dir; publish it.
        _distributed.checkpoint_barrier(self._ddp, stage_error)

        stage_error = None
        if _distributed.should_save(a, self._ddp):
            try:
                if Path(ckpt_dir).is_dir():
                    # Defensive: the only callers target a fresh step, but
                    # never let a stale dir block the atomic rename.
                    shutil.rmtree(ckpt_dir)
                Path(staging_dir).replace(ckpt_dir)
                # Rotation honours ``save_total_limit`` and protects best when set.
                ckpt.rotate_checkpoints(
                    output_dir,
                    save_total_limit=a.save_total_limit,
                    best_model_checkpoint=self.state.best_model_checkpoint,
                )
                self._control.should_save = False
                log.info("Saved checkpoint to %s", ckpt_dir)
                # Notify callbacks that a checkpoint was just written.
                self._control = self._callback_handler.on_save(
                    self.args, self.state, self._control
                )
            except Exception as exc:
                stage_error = exc

        # Final barrier so all ranks see post-save state consistently before
        # any continues into the next training step / eval / rotation.
        _distributed.checkpoint_barrier(self._ddp, stage_error)
        return ckpt_dir

    def _save_model_artifacts(self, output_dir: str) -> None:
        """Save model weights/config plus processing class using HF-compatible names."""
        if hasattr(self._model, "save_pretrained"):
            self._model.save_pretrained(
                output_dir,
                safe_serialization=self.args.save_safetensors,
            )
        else:
            state_dict = {
                name: tensor.detach().cpu()
                for name, tensor in self._model.state_dict().items()
            }
            if self.args.save_safetensors:
                from safetensors.torch import save_file as save_safetensors

                save_safetensors(
                    state_dict,
                    str(Path(output_dir) / ckpt.SAFE_WEIGHTS_NAME),
                    metadata={"format": "pt"},
                )
            else:
                torch.save(state_dict, str(Path(output_dir) / ckpt.WEIGHTS_NAME))
        if self._processing_class is not None:
            self._processing_class.save_pretrained(output_dir)

    def _save_optimizer(self, ckpt_dir: str, ctx: _TrainingContext) -> None:
        torch.save(
            opaque_state_dict(ctx.opt_state),
            str(Path(ckpt_dir) / ckpt.DP_OPTIMIZER_NAME),
        )

    def _save_dp_runtime(self, ckpt_dir: str, ctx: _TrainingContext) -> None:
        # ``state_dict`` from the opaque.serialization registry — each
        # sampler family (Poisson here, dp-ftrl variants in subclasses)
        # registers its own serializer pair at module-import time.
        from opaque.serialization import state_dict as opaque_state_dict

        sampler_state = (
            opaque_state_dict(ctx.current_sampler)
            if ctx.current_sampler is not None
            else None
        )
        # Worker prefetch can draw sampler indices ahead of what the trainer
        # has consumed; clamp to ``global_step - sampler_step_offset`` (the
        # offset rebases across an ``ignore_data_skip`` resume) so restore
        # never replays more than was actually trained on.
        if sampler_state is not None and "consumed" in sampler_state:
            trainer_consumed = self.state.global_step - ctx.sampler_step_offset
            sampler_state = {
                **sampler_state,
                "consumed": min(int(sampler_state["consumed"]), int(trainer_consumed)),
            }

        if ctx.mf is not None:
            _amp = ctx.mf.amplifier_factory(ctx.noise_multiplier)
            mf_n_steps: int | None = int(_amp.n_steps)
            mf_min_sep: int | None = int(_amp.min_sep)
            mf_max_participations: int | None = int(_amp.max_participations)
        else:
            mf_n_steps = mf_min_sep = mf_max_participations = None

        a = self.args
        ckpt.save_dp_runtime_state(
            str(Path(ckpt_dir) / ckpt.DP_STATE_NAME),
            clip_state=ctx.clip_state,
            noise_state=ctx.noise_state,
            sampler_state=sampler_state,
            sample_rate=ctx.sample_rate,
            target_delta=ctx.target_delta,
            noise_multiplier=ctx.noise_multiplier,
            expected_steps_per_epoch=ctx.expected_steps_per_epoch,
            expected_batch_size=int(a.train_batch_size),
            total_steps=ctx.total_steps,
            mechanism_kind=ctx.mechanism_kind,
            is_horizon_process=ctx.is_horizon_process,
            calibration_source=ctx.calibration_source,
            target_epsilon=a.privacy_target_epsilon,
            horizon_process_state=(
                opaque_state_dict(ctx.horizon_process)
                if ctx.horizon_process is not None
                else None
            ),
            mf_n_steps=mf_n_steps,
            mf_min_sep=mf_min_sep,
            mf_max_participations=mf_max_participations,
            lr_scheduler=a.lr_scheduler,
            learning_rate=a.learning_rate,
            warmup_steps=a.warmup_steps,
            lr_scheduler_kwargs=(
                a.lr_scheduler_kwargs
                if isinstance(a.lr_scheduler_kwargs, dict)
                else None
            ),
        )

    def _save_accountant(self, ckpt_dir: str, accountant: Accountant) -> None:
        path = Path(ckpt_dir) / ckpt.DP_ACCOUNTANT_NAME
        # Compact JSON (no indent): pretty-printing writes O(n^2)
        # indentation over a deep composition spine — ~800 MB at 10k
        # heterogeneous steps vs ~3 MB compact.
        with path.open("w") as f, _deep_json_recursion():
            json.dump(opaque_state_dict(accountant), f)

    def _save_rng_state(self, ckpt_dir: str) -> None:
        """Snapshot this rank's Python/NumPy/torch RNG state to disk.

        Each rank writes its own ``rng_state.pth`` (single-process) or
        ``rng_state_{rank}.pth`` (multi-rank) — every rank carries an
        independent non-DP RNG that affects collator stochasticity, model
        init, eval shuffling, etc.  The DP RNG chain is keyed off
        ``args.seed`` and folded by epoch / iter_count, so it is
        deterministic and does not need a per-rank file.
        """
        torch.save(
            ckpt.snapshot_rng_state(),
            ckpt.rng_state_path(
                ckpt_dir,
                rank=self._ddp.rank,
                world_size=self._ddp.world_size,
            ),
        )

    def _save_trainer_state(self, ckpt_dir: str) -> None:
        """Serialize :class:`TrainerState` to ``trainer_state.json``.

        Round-trips through ``TrainerState.to_json`` / ``from_json``.
        Callback state is collected via HF's ``ExportableState`` protocol
        (modern shape ``{"args": {...}, "attributes": {...}}`` produced
        by ``cb.state()``); callbacks that don't implement ``state()`` are
        skipped.
        """
        cb_states: dict[str, Any] = {}
        for cb in self._callback_handler.callbacks:
            state_fn = getattr(cb, "state", None)
            if callable(state_fn):
                cb_states[type(cb).__name__] = state_fn()
        self.state.stateful_callbacks = cb_states

        payload = self.state.to_json()
        path = Path(ckpt_dir) / ckpt.TRAINER_STATE_NAME
        with path.open("w") as f:
            json.dump(payload, f, indent=2, default=str)

    def _save_training_args(self, ckpt_dir: str) -> None:
        # Filename matches HF's ``TRAINING_ARGS_NAME``; HF tooling that
        # ``torch.load(.../training_args.bin)`` accepts the bundled
        # ``TrainingArguments`` because the dataclass is a strict
        # superset of ``TrainingArguments``.
        torch.save(self.args, str(Path(ckpt_dir) / ckpt.TRAINING_ARGS_NAME))

    def _maybe_final_save(self, ctx: _TrainingContext, global_step: int) -> None:
        """Always emit a final checkpoint when saving is enabled (HF parity).

        Skipped if a checkpoint at this exact step already exists (e.g. an
        epoch-strategy save just fired and we're now at end-of-training).
        """
        if self.args.save_strategy == "no":
            return
        output_dir = self._effective_output_dir()
        if output_dir is None:
            return
        target = str(Path(output_dir) / f"{ckpt.PREFIX_CHECKPOINT_DIR}-{global_step}")
        if Path(target).is_dir():
            return
        self._save_checkpoint()

    def _refresh_final_checkpoint_state(self, global_step: int) -> None:
        """Refresh final checkpoint metadata after final logs update callbacks."""
        if self.args.save_strategy == "no":
            return
        output_dir = self._effective_output_dir()
        if output_dir is None:
            return
        target = str(Path(output_dir) / f"{ckpt.PREFIX_CHECKPOINT_DIR}-{global_step}")
        if Path(target).is_dir():
            self._save_trainer_state(target)

    # ------------------------------------------------------------------
    # Resume / load
    # ------------------------------------------------------------------

    def _resolve_resume_path(
        self, value: str | bool | os.PathLike[str] | None
    ) -> str | None:
        """Resolve ``resume_from_checkpoint`` to a concrete directory or ``None``.

        ``True`` is **tolerant**: if a checkpoint exists under
        ``args.output_dir`` it is auto-selected; if no checkpoints exist
        the call falls through to ``None`` (fresh run) with an info-level
        log.  This lets scripts pass ``resume_from_checkpoint=True``
        unconditionally — "resume if you can, else start fresh" — without
        having to probe the filesystem first.  ``output_dir is None``
        still raises since that's a config error, not a missing artefact.
        """
        if value is None or value is False:
            return None
        if value is True:
            output_dir = self._effective_output_dir()
            if output_dir is None:
                raise ConfigurationError(
                    *("resume_from_checkpoint=True requires args.output_dir to be set",)
                )
            found = ckpt.get_last_checkpoint(output_dir)
            if found is None:
                log.info(
                    "resume_from_checkpoint=True but no checkpoints found under "
                    "%s; starting a fresh run.",
                    output_dir,
                )
                return None
            return found
        if isinstance(value, os.PathLike):
            value = os.fspath(value)
        if not isinstance(value, str):
            raise InputTypeError(
                *(
                    "resume_from_checkpoint must be str | bool | PathLike | None, "
                    f"got {type(value).__name__}",
                )
            )
        if not Path(value).is_dir():
            raise CheckpointError(
                *(f"resume_from_checkpoint directory does not exist: {value}",)
            )
        return value

    def _load_model_weights(self, ckpt_dir: str) -> None:
        """Load saved weights into ``self._model`` so make_functional starts from them."""
        new_state, mutated = self._read_weights_file(ckpt_dir)
        if not new_state and not mutated:
            log.warning("No weights file in %s; model untouched", ckpt_dir)
            return
        # Sharded loads already mutated ``self._model`` in place — skip
        # the redundant ``load_state_dict`` call.  ``strict`` follows
        # PEFT detection: adapter checkpoints store only adapter
        # parameters (subset → ``strict=False``); full-model checkpoints
        # surface mismatched keys as errors (``strict=True``).
        if not mutated:
            strict = not self._is_peft
            self._model.load_state_dict(new_state, strict=strict)

    def _read_runtime_for_resume(
        self, ckpt_dir: str
    ) -> tuple[ckpt.RuntimeCheckpoint, Accountant]:
        """Load a *complete* DP checkpoint for resume.

        A resumable DP checkpoint must carry the full runtime needed to
        continue a privacy-accounted process: ``dp_state.pt`` (clip /
        noise / sampler state), ``dp_optimizer.pt`` (optimizer state), and
        ``accountant.json`` (privacy provenance).  A checkpoint missing
        any of these is a **weights-only export** — e.g. one written with
        ``save_only_model=True``, an HF checkpoint, or a plain pretrained
        model — and is *not resumable*: continuing a DP run from it would
        rebuild the noise stream from scratch and/or discard the spent
        budget.

        To start a *fresh* DP run from such weights, load them at
        construction instead (``model=AutoModel.from_pretrained(...)``).
        The new run begins with a zero accountant, which is correct only
        when the prior training had no DP cost (e.g. public-data warmup);
        for a prior DP run whose accountant was lost, neither resume nor a
        fresh run is sound — restore ``accountant.json`` from the source
        of truth.

        Raises:
            CheckpointError: If any required DP runtime file is absent.
        """
        required = (
            ckpt.DP_STATE_NAME,
            ckpt.DP_OPTIMIZER_NAME,
            ckpt.DP_ACCOUNTANT_NAME,
        )
        missing = [name for name in required if not (Path(ckpt_dir) / name).exists()]
        if missing:
            raise CheckpointError(
                *(
                    f"Cannot resume training from {ckpt_dir}: missing DP runtime "
                    f"file(s) {missing}.  This is a weights-only export (e.g. "
                    "save_only_model=True, an HF checkpoint, or a pretrained "
                    "model), not a resumable DP checkpoint.  To start a fresh DP "
                    "run from these weights, load them at construction "
                    "(model=AutoModel.from_pretrained(...)) — the run begins with "
                    "a zero privacy accountant, sound only when the prior training "
                    "had no DP cost.  resume_from_checkpoint requires a complete DP "
                    "checkpoint produced by this trainer (save_only_model=False).",
                )
            )

        runtime_payload = ckpt.load_dp_runtime_state(
            str(Path(ckpt_dir) / ckpt.DP_STATE_NAME)
        )
        with (
            (Path(ckpt_dir) / ckpt.DP_ACCOUNTANT_NAME).open() as f,
            _deep_json_recursion(),
        ):
            accountant = opaque_from_state_dict(Accountant(), json.load(f))
        return runtime_payload, accountant

    def _read_trainer_state(self, ckpt_dir: str) -> dict[str, Any] | None:
        """Read ``trainer_state.json`` from a checkpoint directory."""
        path = Path(ckpt_dir) / ckpt.TRAINER_STATE_NAME
        if not path.exists():
            return None
        with path.open() as f:
            return json.load(f)

    def _apply_runtime_state(
        self,
        ctx: _TrainingContext,
        runtime: ckpt.RuntimeCheckpoint,
        accountant: Accountant | None,
        ckpt_dir: str,
    ) -> None:
        """Overwrite ctx fields with values restored from a checkpoint."""
        ctx.clip_state = opaque_from_state_dict(ctx.clip_state, runtime.clip_state)
        ctx.noise_state = opaque_from_state_dict(ctx.noise_state, runtime.noise_state)

        opt_path = Path(ckpt_dir) / ckpt.DP_OPTIMIZER_NAME
        if opt_path.exists():
            # Load flat serialisation on CPU; tensors move with ``opt.update``.
            opt_sd = torch.load(
                str(opt_path),
                map_location="cpu",
                weights_only=False,
            )
            ctx.opt_state = opaque_from_state_dict(ctx.opt_state, opt_sd)
            if isinstance(ctx.opt_state, ScheduleFreeState):
                ctx.trainable_params = {
                    name: (1.0 - ctx.opt_state.beta) * ctx.opt_state.z[name]
                    + ctx.opt_state.beta * ctx.opt_state.x[name]
                    for name in ctx.trainable_params
                }

        if accountant is not None:
            ctx.accounting = accountant

    def _load_rng_state(self, ckpt_dir: str) -> None:
        """Restore this rank's RNG snapshot from a checkpoint.

        Reads ``rng_state.pth`` (single-process) or ``rng_state_{rank}.pth``
        (multi-rank) — the world-aware helper picks the right path so the
        call site doesn't have to branch.
        """
        path = ckpt.rng_state_path(
            ckpt_dir,
            rank=self._ddp.rank,
            world_size=self._ddp.world_size,
        )
        if not Path(path).exists():
            return
        # ``weights_only=False``: the snapshot bundles
        # ``random.getstate()`` (a Python tuple) and NumPy's RNG state
        # alongside the torch tensors — the safe-load path can't
        # reconstruct either.  PyTorch 2.6+ flips the default to
        # ``True``; keeping the explicit ``False`` pins the contract.
        snap = torch.load(path, map_location="cpu", weights_only=False)
        ckpt.restore_rng_state(snap)

    def _load_callback_states(self) -> None:
        """Restore each callback's state when ``restore_callback_states_from_checkpoint`` is set.

        Reads from ``self.state.stateful_callbacks`` — the dataclass field
        populated by ``TrainerState.from_json`` during resume.  The
        payload uses HF's ``ExportableState`` shape ``{"args": {...},
        "attributes": {...}}``; saved attributes are set back onto the
        live callback instance so its identity is preserved
        (``EarlyStoppingCallback`` parity).

        Drift between the saved set and the live set is logged at
        ``info`` so a user who removed / added a callback between runs
        sees it explicitly instead of silently inheriting the previous
        run's callback state (or not).
        """
        if not self.args.restore_callback_states_from_checkpoint:
            return
        cb_states = self.state.stateful_callbacks or {}
        live_names = {type(cb).__name__ for cb in self._callback_handler.callbacks}
        saved_names = set(cb_states)

        missing_in_live = saved_names - live_names
        missing_in_saved = live_names - saved_names
        if missing_in_live:
            log.info(
                "Resume callback drift: %d saved callback(s) not present in "
                "live trainer (state will not be restored): %s",
                len(missing_in_live),
                sorted(missing_in_live),
            )
        if missing_in_saved:
            log.info(
                "Resume callback drift: %d live callback(s) not present in "
                "saved state (will start with fresh state): %s",
                len(missing_in_saved),
                sorted(missing_in_saved),
            )

        for cb in self._callback_handler.callbacks:
            name = type(cb).__name__
            if name not in cb_states:
                continue
            payload = cb_states[name]
            attrs = payload.get("attributes") or {} if isinstance(payload, dict) else {}
            for attr_key, value in attrs.items():
                setattr(cb, attr_key, value)

    def _validate_horizon_resume_calibration(
        self, runtime: ckpt.RuntimeCheckpoint
    ) -> None:
        """Reject calibration changes before restoring a horizon multiplier."""
        if not runtime.is_horizon_process:
            return

        current_source = (
            "fixed" if self.args.privacy_noise_multiplier is not None else "calibrated"
        )
        if runtime.calibration_source != current_source:
            raise CheckpointError(
                *(
                    "Whole-horizon resume forbids calibration mode drift: "
                    f"saved={runtime.calibration_source!r}, "
                    f"current={current_source!r}. Restart from scratch to change "
                    "between fixed and calibrated noise.",
                )
            )
        if current_source == "calibrated" and _drift_differs(
            runtime.target_epsilon,
            self.args.privacy_target_epsilon,
        ):
            raise CheckpointError(
                *(
                    "Whole-horizon resume forbids privacy_target_epsilon drift: "
                    f"saved={runtime.target_epsilon!r}, "
                    f"current={self.args.privacy_target_epsilon!r}. Restart from "
                    "scratch to calibrate against a different privacy budget.",
                )
            )

    def _warn_on_arg_drift(self, runtime: ckpt.RuntimeCheckpoint) -> None:
        """Surface drift between the saved checkpoint and current ``args``.

        Iterates over every ``RuntimeCheckpoint`` field tagged
        ``compare_on_resume=True``, compares saved vs current, and acts
        on the per-field ``drift`` disposition (see
        :class:`~opaque.api.transformers.trainer._checkpoint.RuntimeCheckpoint`
        for the vocabulary):

        - ``"dp_relevant"`` + independent-step DP-SGD: ``log.warning``.
        - ``"dp_relevant"`` + a whole-horizon process or DP-FTRL:
          ``raise ValueError`` — accounting uses the original schedule.
        - ``"shape"``: warn — trajectory differs but privacy is intact.
        - ``"intentional_extend"``: silent — normal user action (e.g.,
          extending ``total_steps`` under independent-step DP-SGD).

        Dict-valued dispositions resolve via the saved ``mechanism_kind``:
        the matching key wins; ``"default"`` is the fallback.
        """
        a = self.args
        ctx = self._ctx
        current_by_name = self._current_values_for_drift(a, ctx)
        saved_mechanism = runtime.mechanism_kind
        horizon_resume = runtime.is_horizon_process or (
            ctx is not None and ctx.is_horizon_process
        )

        for f in dataclasses.fields(runtime):
            if not f.metadata.get("compare_on_resume"):
                continue
            saved = getattr(runtime, f.name)
            current = current_by_name.get(f.name)
            if saved is None or current is None:
                continue
            if f.name == "horizon_process_state":
                saved = _dpftrl.resume_process_state(saved)
                current = _dpftrl.resume_process_state(current)
            if not _drift_differs(saved, current):
                continue

            disposition = _resolve_drift_disposition(f.metadata, saved_mechanism)
            if (
                f.name == "total_steps"
                and disposition == "intentional_extend"
                and horizon_resume
            ):
                # Gaussian k-out-of-t shares the "gaussian" mechanism label but
                # is still defined for one fixed horizon, unlike Poisson DP-SGD.
                disposition = "dp_relevant"
            if disposition == "intentional_extend":
                continue
            if disposition == "dp_relevant":
                if saved_mechanism != "gaussian" or horizon_resume:
                    raise CheckpointError(
                        *(
                            f"Whole-horizon resume forbids drift on {f.name!r}: "
                            f"saved={saved!r}, current={current!r}. The "
                            "accounting process is defined for the original "
                            "declared horizon; restart from scratch with the new arg.",
                        )
                    )
                log.warning(
                    "Resume arg drift on %s (dp_relevant, DP-SGD): "
                    "saved=%r, current=%r.",
                    f.name,
                    saved,
                    current,
                )
            elif disposition == "shape":
                log.warning(
                    "Resume arg drift on %s (shape): saved=%r, current=%r — "
                    "trajectory will differ from phase-1; using current args.",
                    f.name,
                    saved,
                    current,
                )
            else:
                log.warning(
                    "Resume arg drift on %s (%s): saved=%r, current=%r",
                    f.name,
                    disposition,
                    saved,
                    current,
                )

    def _current_values_for_drift(
        self, a, ctx: _TrainingContext | None
    ) -> dict[str, Any]:
        """Resolve current (post-setup) values for the drift-checked fields.

        Splits the lookup off so :meth:`_warn_on_arg_drift` stays a pure
        iteration over field metadata — no per-field if/else.  Returns
        ``None`` for any field whose live value can't be determined yet
        (e.g. ``sample_rate`` before ``ctx`` exists).
        """
        # ``sample_rate`` fallback (when ``ctx`` isn't built yet) must
        # mirror ``_setup_training``'s computation — same numerator
        # (``expected_batch_size``, which is ``world_size *
        # per_device_train_batch_size``) and same denominator (the
        # dataset size from ``_effective_train_dataset_size``, trimmed to
        # a multiple of ``world_size`` under DDP when needed).  Returns
        # ``None`` (skips the check) when the dataset isn't available.
        if ctx is not None:
            fallback_rate: float | None = ctx.sample_rate
        elif self._train_dataset is None:
            fallback_rate = None
        else:
            fallback_rate = a.train_batch_size / max(
                1, self._effective_train_dataset_size()
            )
        return {
            "sample_rate": fallback_rate,
            "target_delta": (
                ctx.target_delta if ctx is not None else a.privacy_target_delta
            ),
            # In *calibrated* mode the noise multiplier is recomputed over the
            # remaining steps and so legitimately differs from the saved one;
            # comparing them would fire a spurious "drift" warning on every
            # resume and erode trust in the genuinely-meaningful drift signals.
            # Return ``None`` (the drift loop skips ``None``) unless the user
            # pinned a fixed multiplier, where a mismatch is real drift.
            "noise_multiplier": (
                None
                if (ctx is not None and ctx.calibration_source == "calibrated")
                else (
                    ctx.noise_multiplier
                    if ctx is not None
                    else a.privacy_noise_multiplier
                )
            ),
            "total_steps": (
                ctx.total_steps if ctx is not None else self._predict_total_steps()
            ),
            "expected_batch_size": int(a.train_batch_size),
            "expected_steps_per_epoch": (
                ctx.expected_steps_per_epoch if ctx is not None else None
            ),
            "mechanism_kind": (
                ctx.mechanism_kind if ctx is not None else a.privacy_noise_mechanism
            ),
            "is_horizon_process": (ctx.is_horizon_process if ctx is not None else None),
            "horizon_process_state": (
                opaque_state_dict(ctx.horizon_process)
                if ctx is not None and ctx.horizon_process is not None
                else None
            ),
            # MF strategy params: live values are derived inside MFContext
            # construction and surfaced via ctx.mf; before ctx exists we
            # can't compare, so return None to skip the check.
            "mf_n_steps": (
                int(ctx.mf.amplifier_factory(ctx.noise_multiplier).n_steps)
                if ctx is not None and ctx.mf is not None
                else None
            ),
            "mf_min_sep": (
                int(ctx.mf.amplifier_factory(ctx.noise_multiplier).min_sep)
                if ctx is not None and ctx.mf is not None
                else None
            ),
            "mf_max_participations": (
                int(ctx.mf.amplifier_factory(ctx.noise_multiplier).max_participations)
                if ctx is not None and ctx.mf is not None
                else None
            ),
            # LR-schedule shape (privacy-neutral; warn-only drift).
            "lr_scheduler": a.lr_scheduler,
            "learning_rate": a.learning_rate,
            "warmup_steps": a.warmup_steps,
            "lr_scheduler_kwargs": (
                a.lr_scheduler_kwargs
                if isinstance(a.lr_scheduler_kwargs, dict)
                else None
            ),
        }
