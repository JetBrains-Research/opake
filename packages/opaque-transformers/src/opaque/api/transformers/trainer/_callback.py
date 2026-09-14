"""Callback handler construction for Trainer.

Trainer uses HF's :class:`transformers.trainer_callback.CallbackHandler`
and :class:`~transformers.trainer_callback.DefaultFlowCallback` directly.
This module just bundles the construction pattern HF's ``Trainer``
applies internally:

1. Prepend ``DEFAULT_CALLBACKS = [DefaultFlowCallback]``.
2. Append reporting integration callbacks from ``args.report_to``
   (``WandbCallback``, ``TensorBoardCallback``, etc.) — HF's
   :func:`transformers.integrations.get_reporting_integration_callbacks`
   returns the right set for each backend string.
3. Append user-supplied callbacks.
4. Construct the handler.
5. Register a progress callback (``ProgressCallback`` / ``PrinterCallback``)
   based on ``args.disable_tqdm``.

The handler stores ``optimizer=None`` and ``lr_scheduler=None`` because the
functional optimizer + ``Callable[[int], float]`` schedule have no
``torch.optim`` instance HF callbacks can read; callbacks that introspect
those slots see ``None``.

DP semantic note: ``on_substep_end`` is part of HF's hook surface but is
**not fired** by Trainer.  Each iteration of DP-SGD is one atomic
optimizer step (the Poisson round); there is no substep concept.  HF
callbacks that override ``on_substep_end`` will simply not be invoked.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
)

if TYPE_CHECKING:
    # ``TYPE_CHECKING`` import so the annotation on ``build_callback_handler``
    # types as our standalone ``TrainingArguments`` without taking a runtime
    # dependency from ``_callback`` on ``_config`` (today there is no cycle;
    # this is defensive for future evolution).
    from ._training_arguments import TrainingArguments


__all__ = [
    "DEFAULT_CALLBACKS",
    "BestModelSaveCallback",
    "build_callback_handler",
    "is_metric_improved",
    "resolve_eval_metric",
    "rewrite_logs_for_reporting",
    "wrap_reporting_callback_class",
]


def resolve_eval_metric(
    metrics: dict[str, Any] | None,
    metric_for_best_model: str | None,
) -> tuple[str, float] | None:
    """Look up ``metric_for_best_model`` in ``metrics`` with HF's ``eval_`` auto-prefix.

    Returns ``(resolved_key, value)`` if found, else ``None``.  Matches the
    key-resolution rule used by HF (and historically by
    :meth:`Trainer._update_best_metric`): try the raw key first; if absent
    and the key doesn't already start with ``"eval_"``, try the prefixed
    form.  Returns ``None`` when ``metric_for_best_model`` is ``None``, the
    metric dict is empty, or the key isn't present under either spelling —
    callers decide whether that's a no-op (callbacks) or an error (trainer).
    """
    if metric_for_best_model is None or not metrics:
        return None
    key = metric_for_best_model
    if key not in metrics and not key.startswith("eval_"):
        key = f"eval_{key}"
    if key not in metrics:
        return None
    return key, float(metrics[key])


def is_metric_improved(
    value: float,
    best_metric: float | None,
    greater_is_better: bool | None,
) -> bool:
    """``True`` if ``value`` is a strict improvement over ``best_metric``.

    The first call (``best_metric is None``) always counts as an
    improvement, mirroring HF.  ``greater_is_better=None`` is conservative:
    treated as ``False`` (lower is better) to match HF's
    ``metric_for_best_model='loss'`` default.
    """
    if best_metric is None:
        return True
    if greater_is_better:
        return value > best_metric
    return value < best_metric


log = logging.getLogger(__name__)


DEFAULT_CALLBACKS: list[type[TrainerCallback]] = [DefaultFlowCallback]


_GROUP_PRIVACY_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("_clipping_norm", "clipping_norm"),
    ("_grad_norm", "grad_norm"),
    ("_clip_rate", "clip_rate"),
    ("_noise_std", "noise_std"),
)
# Keys recorded in the W&B run summary in their rewritten ``privacy/*`` form,
# in addition to whatever per-step logging happens below. ``epsilon`` /
# ``delta`` / ``noise_multiplier`` stay in the per-step stream too — at one
# value per run they render as comparison stacks in W&B's multi-run view.
_PRIVACY_SUMMARY_KEYS = {
    "privacy_epsilon",
    "privacy_delta",
    "privacy_noise_multiplier",
    "privacy_calibration_source",
    "privacy_calibration_noise_multiplier",
    "privacy_calibration_achieved_epsilon",
    "privacy_calibration_converged",
    "privacy_noise_std",
    "train_converged_microbatch_size",
}
# Subset of summary keys dropped from the timeline stream because their value
# is fixed for the run. Epsilon remains: it accumulates for independent
# mechanisms and is the declared full-run bound for horizon mechanisms.
_PRIVACY_STREAM_SKIP_KEYS = {
    "privacy_calibration_source",
    "privacy_calibration_noise_multiplier",
    "privacy_calibration_achieved_epsilon",
    "privacy_calibration_converged",
    "privacy_noise_std",
    "train_converged_microbatch_size",
}
_WRAPPED_CALLBACK_CLASSES: dict[type[Any], type[Any]] = {}


def _rewrite_privacy_group_key(key: str) -> str | None:
    group_and_metric = key.removeprefix("privacy_group_")
    for suffix, metric_name in _GROUP_PRIVACY_SUFFIXES:
        if group_and_metric.endswith(suffix):
            group_name = group_and_metric[: -len(suffix)]
            if group_name:
                return f"privacy/group_{group_name}/{metric_name}"
    return None


def _rewrite_privacy_key(key: str) -> str | None:
    if key.startswith("privacy_group_"):
        group_key = _rewrite_privacy_group_key(key)
        if group_key is not None:
            return group_key
    if key.startswith("privacy_"):
        return "privacy/" + key.removeprefix("privacy_")
    return None


def rewrite_logs_for_reporting(logs: dict[str, Any] | None) -> dict[str, Any]:
    """Rewrite HF raw metric keys for hierarchy-aware reporting backends."""
    rewritten: dict[str, Any] = {}
    for key, value in (logs or {}).items():
        privacy_key = _rewrite_privacy_key(key)
        if privacy_key is not None:
            rewritten[privacy_key] = value
        elif key.startswith("eval_"):
            rewritten["eval/" + key.removeprefix("eval_")] = value
        elif key.startswith("test_"):
            rewritten["test/" + key.removeprefix("test_")] = value
        else:
            rewritten["train/" + key] = value
    return rewritten


def wrap_reporting_callback_class(callback_cls: type[Any]) -> type[Any]:
    """Wrap HF reporting callbacks whose log rewrite needs privacy keys."""
    if callback_cls in _WRAPPED_CALLBACK_CLASSES:
        return _WRAPPED_CALLBACK_CLASSES[callback_cls]

    class_name = getattr(callback_cls, "__name__", "")
    if class_name == "WandbCallback":

        class OpaqueWandbCallback(callback_cls):  # type: ignore[misc, valid-type]
            def on_log(self, args, state, control, model=None, logs=None, **kwargs):
                single_value_scalars = {
                    "train_runtime",
                    "train_samples_per_second",
                    "train_steps_per_second",
                    "train_loss",
                    "total_flos",
                }

                if self._wandb is None:
                    return
                if not self._initialized:
                    self.setup(args, state, model)
                if state.is_world_process_zero:
                    raw_logs = dict(logs or {})
                    for key, value in raw_logs.items():
                        if key in single_value_scalars:
                            self._wandb.run.summary[key] = value
                        if key in _PRIVACY_SUMMARY_KEYS:
                            # Privacy keys reach summary in their rewritten
                            # ``privacy/*`` form; non-privacy summary keys
                            # (e.g. ``train_converged_microbatch_size``) get
                            # the same ``train/*`` rewrite as the per-step
                            # stream so dashboards see one consistent name.
                            summary_key = _rewrite_privacy_key(key)
                            if summary_key is None:
                                summary_key = "train/" + key
                            self._wandb.run.summary[summary_key] = value
                    non_scalar_logs = {
                        key: value
                        for key, value in raw_logs.items()
                        if key not in single_value_scalars
                        and key not in _PRIVACY_STREAM_SKIP_KEYS
                    }
                    self._wandb.log(
                        {
                            **rewrite_logs_for_reporting(non_scalar_logs),
                            "train/global_step": state.global_step,
                        }
                    )

        OpaqueWandbCallback.__name__ = "OpaqueWandbCallback"
        wrapped = OpaqueWandbCallback
    elif class_name == "TensorBoardCallback":

        class OpaqueTensorBoardCallback(callback_cls):  # type: ignore[misc, valid-type]
            def on_log(self, args, state, control, logs=None, **kwargs):
                if not state.is_world_process_zero:
                    return

                if self.tb_writer is None:
                    self._init_summary_writer(args)

                if self.tb_writer is not None:
                    raw = {
                        key: value
                        for key, value in (logs or {}).items()
                        if key not in _PRIVACY_STREAM_SKIP_KEYS
                    }
                    for key, value in rewrite_logs_for_reporting(raw).items():
                        if isinstance(value, (int, float)):
                            self.tb_writer.add_scalar(key, value, state.global_step)
                        elif isinstance(value, str):
                            self.tb_writer.add_text(key, value, state.global_step)
                        else:
                            log.warning(
                                "Trainer is attempting to log a value of %r of "
                                "type %s for key %r as a scalar. This invocation "
                                "of TensorBoard's writer.add_scalar() is incorrect "
                                "so we dropped this attribute.",
                                value,
                                type(value),
                                key,
                            )

        OpaqueTensorBoardCallback.__name__ = "OpaqueTensorBoardCallback"
        wrapped = OpaqueTensorBoardCallback
    else:
        wrapped = callback_cls

    _WRAPPED_CALLBACK_CLASSES[callback_cls] = wrapped
    return wrapped


def build_callback_handler(
    args: TrainingArguments,
    model: Any,
    processing_class: Any,
    callbacks: list[TrainerCallback] | None = None,
) -> CallbackHandler:
    """Construct an HF ``CallbackHandler`` populated like ``Trainer.__init__``.

    Mirrors :class:`transformers.Trainer`'s init pattern:

    - ``DEFAULT_CALLBACKS`` (currently just :class:`DefaultFlowCallback`)
      come first so default cadence runs before user logic.
    - Reporting integration callbacks (W&B, TensorBoard, MLflow, …) are
      registered next, derived from ``args.report_to`` via HF's
      :func:`~transformers.integrations.get_reporting_integration_callbacks`.
    - User-supplied callbacks come after.
    - A progress callback is registered after construction:
      :class:`PrinterCallback` when ``args.disable_tqdm`` else
      :class:`ProgressCallback`.

    The handler stores ``optimizer=None`` and ``lr_scheduler=None`` since
    the DP path uses a functional optimizer + callable schedule.
    """
    reporting_callbacks = [
        wrap_reporting_callback_class(callback_cls)
        for callback_cls in get_reporting_integration_callbacks(args.report_to)
    ]
    user_callbacks = list(callbacks) if callbacks else []
    callback_list: list[Any] = [
        *DEFAULT_CALLBACKS,
        *reporting_callbacks,
        *user_callbacks,
    ]
    handler = CallbackHandler(
        callback_list,
        model,
        processing_class,
        None,  # optimizer
        None,  # lr_scheduler
    )
    handler.add_callback(PrinterCallback() if args.disable_tqdm else ProgressCallback())
    return handler


class BestModelSaveCallback(TrainerCallback):
    """Sets ``control.should_save`` when an evaluation improves ``best_metric``.

    HF's :class:`DefaultFlowCallback` recognises ``save_strategy="steps"``
    and ``"epoch"`` but not ``"best"``.  This callback fills that gap.
    Auto-injected by :class:`Trainer` when ``save_strategy == "best"`` or
    ``load_best_model_at_end=True`` (so best always lands on a saved step).

    Timing
    ------
    ``on_evaluate`` fires inside :meth:`Trainer._after_evaluate`, *before*
    :meth:`Trainer._update_best_metric` runs in the caller.  So when this
    callback fires, ``state.best_metric`` still holds the *previous* best —
    the same precondition :class:`~transformers.EarlyStoppingCallback` relies
    on.  We therefore compute the improvement comparison ourselves rather
    than reading post-update state.  The trainer's separate
    :meth:`~Trainer._update_best_metric` call records the new best under
    the canonical attribute names; the two stay in sync because both use
    :func:`is_metric_improved` against the same operands.
    """

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Force a save whenever an eval improves best, for both
        # ``save_strategy="best"`` and ``load_best_model_at_end`` with a
        # steps/epoch save cadence coarser than eval — otherwise
        # ``best_global_step`` can land on an eval-only step that has no
        # checkpoint folder (issue #386).
        if args.save_strategy != "best" and not args.load_best_model_at_end:
            return
        resolved = resolve_eval_metric(metrics, args.metric_for_best_model)
        if resolved is None:
            return
        _, value = resolved
        if is_metric_improved(value, state.best_metric, args.greater_is_better):
            control.should_save = True
