"""DP trainer state — standalone dataclass for :class:`DPTrainer`.

Mirrors HF :class:`transformers.TrainerState`'s field *names* for the
fields we use (so HF reporting callbacks duck-typing
``state.global_step``, ``state.epoch``, ``state.log_history`` keep
working unchanged) without inheriting from it. The field surface is
the subset we actually use, plus DPTrainer-specific privacy bookkeeping
and an explicit serialisation ``version``.

Dropped vs HF:

- ``total_flos`` — FLOP counter never written by the DP path.

HPO parity: HPO itself is not supported by DPTrainer (no
``hyperparameter_search`` entry point), but ``is_hyper_param_search``,
``trial_name``, and ``trial_params`` are kept on the state because HF's
own reporting callbacks (WandB, TensorBoard, ClearML, Neptune, ...)
read them via attribute access in their ``on_train_begin`` / logging
paths regardless of whether HPO is active. Defaults
(``False`` / ``None`` / ``None``) route every HPO-guarded branch into
its no-HPO arm — i.e. callbacks behave as in a normal non-HPO HF run.

Schema divergence: ``trainer_state.json`` written by DPTrainer may
include privacy telemetry keys in ``log_history``. HF's strict
:meth:`transformers.TrainerState.load_from_json` would reject; ours
filters unknown top-level keys (forward compat).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

__all__ = ["DPTrainerState"]


@dataclasses.dataclass
class DPTrainerState:
    """Trainer state for DPTrainer (standalone; not a ``TrainerState`` subclass).

    HF callbacks (TensorBoard, WandB, EarlyStoppingCallback, …) read
    ``state.global_step``, ``state.epoch``, ``state.log_history`` etc.
    via attribute access — not ``isinstance(state, TrainerState)`` —
    so duck-typing parity is preserved by keeping the field names.
    """

    # HF-named fields we use (parity for callbacks)
    epoch: float = 0.0
    global_step: int = 0
    max_steps: int = 0
    # Resolved cadence in absolute steps.  ``compute_steps`` pre-resolves
    # fractional values (e.g. ``logging_steps=0.5``) and ``None`` to ints
    # before storing here; the on-state representation is uniformly
    # ``int`` for all three.
    logging_steps: int = 500
    eval_steps: int | None = 500
    save_steps: int = 500
    train_batch_size: int | None = None
    num_train_epochs: int = 0
    num_input_tokens_seen: int = 0
    log_history: list[dict[str, float]] = dataclasses.field(default_factory=list)
    best_metric: float | None = None
    best_global_step: int | None = None
    best_model_checkpoint: str | None = None
    is_local_process_zero: bool = True
    is_world_process_zero: bool = True
    stateful_callbacks: dict[str, Any] = dataclasses.field(default_factory=dict)

    # HPO duck-typing parity: HPO is not supported on DPTrainer, but HF's
    # reporting callbacks read these attributes unconditionally on every
    # run (see module docstring).  Defaults route into the no-HPO arm.
    is_hyper_param_search: bool = False
    trial_name: str | None = None
    trial_params: dict[str, Any] | None = None

    # DPTrainer-specific privacy bookkeeping
    privacy_resolved_delta: float | None = None
    privacy_resolved_noise_multiplier: float | None = None
    privacy_calibration_source: str | None = None
    privacy_calibration_noise_multiplier: float | None = None
    privacy_calibration_achieved_epsilon: float | None = None
    privacy_calibration_converged: bool | None = None
    privacy_sample_rate: float | None = None
    privacy_expected_batch_size: int | None = None
    privacy_total_steps: int | None = None
    # True if training was halted early because ε reached privacy_target_epsilon.
    privacy_target_epsilon_reached: bool = False
    # Physical microbatch trained at (post auto-find-microbatch-size OOM search).
    converged_microbatch_size: int | None = None

    def compute_steps(self, args: Any) -> None:
        """Resolve absolute step counts for logging/eval/save.

        Reads ``self.max_steps`` (set by the caller before invocation).
        Fractional values < 1 are interpreted as fractions and rounded
        up; ``None`` is left alone.
        """
        for kind, num_steps in (
            ("logging", args.logging_steps),
            ("eval", args.eval_steps),
            ("save", args.save_steps),
        ):
            if num_steps is None:
                continue
            if num_steps < 1:
                num_steps = math.ceil(self.max_steps * num_steps)
            # Coerce to ``int`` so the resolved cadence on state is
            # uniformly typed (the dataclass declares int fields; user
            # ``args`` may carry the value as float).
            setattr(self, f"{kind}_steps", int(num_steps))

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-compatible dict for ``trainer_state.json``."""
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> DPTrainerState:
        """Reconstruct from a dict loaded from ``trainer_state.json``.

        Unknown keys are filtered (forward-compat with newer writers).
        """
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
