"""HuggingFace-style LR scheduler dispatch for Trainer.

Translates :attr:`TrainingArguments.lr_scheduler` plus
``lr_scheduler_kwargs`` into a ``Callable[[int], float]`` built from
:mod:`opaque.scheduling` primitives.

Supported types: ``constant``, ``constant_with_warmup``, ``linear``,
``cosine``, ``polynomial``, ``inverse_sqrt``, ``cosine_with_restarts``,
``cosine_with_min_lr``, ``cosine_warmup_with_min_lr``,
``warmup_stable_decay``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from opaque.exceptions import ConfigurationError
from opaque.scheduling import (
    constant_schedule,
    cosine_schedule,
    inverse_sqrt_schedule,
    linear_schedule,
    one_minus_sqrt_schedule,
    polynomial_schedule,
    with_restarts,
    with_warmup,
)
from opaque.scheduling.types import CosineSchedule
from transformers.trainer_utils import SchedulerType

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "build_lr_schedule",
    "get_warmup_steps",
]


_DEFERRED: set[str] = set()

_WSD_KWARGS = frozenset(
    {
        "num_decay_steps",
        "num_stable_steps",
        "warmup_type",
        "decay_type",
        "min_lr_ratio",
        "num_cycles",
    }
)

_ALLOWED_KWARGS: dict[str, frozenset[str]] = {
    "constant": frozenset(),
    "constant_with_warmup": frozenset(),
    "linear": frozenset(),
    "cosine": frozenset({"num_cycles"}),
    "polynomial": frozenset({"lr_end", "power"}),
    "inverse_sqrt": frozenset({"timescale"}),
    "cosine_with_restarts": frozenset({"num_cycles"}),
    "cosine_with_min_lr": frozenset({"num_cycles", "min_lr", "min_lr_rate"}),
    "cosine_warmup_with_min_lr": frozenset(
        {"num_cycles", "min_lr", "min_lr_rate", "warmup_lr_rate"}
    ),
    "warmup_stable_decay": _WSD_KWARGS,
}


def get_warmup_steps(
    num_training_steps: int,
    warmup_steps: int | float,
) -> int:
    """Resolve ``warmup_steps`` to an absolute step count.

    A value ``>= 1`` is already a step count and is truncated to an int; a
    value in ``(0, 1)`` is a fraction of ``num_training_steps``. Mirrors
    HF's ``TrainingArguments.get_warmup_steps``.
    """
    if warmup_steps >= 1:
        return int(warmup_steps)
    return math.ceil(num_training_steps * warmup_steps)


def _validate_kwargs(name: str, kwargs: dict[str, Any]) -> None:
    allowed = _ALLOWED_KWARGS[name]
    extra = set(kwargs) - allowed
    if extra:
        raise ConfigurationError(
            *(
                f"lr_scheduler_kwargs contains unsupported keys for "
                f"lr_scheduler={name!r}: {sorted(extra)}. "
                f"Allowed: {sorted(allowed) or '<none>'}.",
            )
        )


def build_lr_schedule(
    args: Any,
    num_training_steps: int,
) -> Callable[[int], float]:
    """Build a step -> learning-rate callable from HF-style training arguments.

    Reads ``args.lr_scheduler``, ``args.warmup_steps``,
    ``args.learning_rate``, and ``args.lr_scheduler_kwargs``.  Returns a
    callable suitable for passing as the ``lr`` argument of any torchopt
    optimizer factory.

    ``args.lr_scheduler`` may also be a :data:`Schedule` recipe;
    in that case the recipe is returned as-is and the HF-name dispatch
    is bypassed entirely.  Warmup / kwargs args are HF-name-dispatch-
    only and must be unset for the recipe path.
    """
    raw = args.lr_scheduler
    # User-supplied Schedule recipe path.  ``str`` / ``SchedulerType``
    # are excluded explicitly because both are technically callable
    # (SchedulerType is an Enum subclass; the class itself is callable
    # via ``SchedulerType(x)``), but instances of the enum aren't.
    if not isinstance(raw, (str, SchedulerType)) and callable(raw):
        if args.warmup_steps:
            raise ConfigurationError(
                *(
                    "warmup_steps is incompatible with a user-supplied "
                    "lr_scheduler Schedule; compose with_warmup(schedule, ...) "
                    "into the recipe yourself.",
                )
            )
        if args.lr_scheduler_kwargs:
            raise ConfigurationError(
                *(
                    "lr_scheduler_kwargs is incompatible with a user-"
                    "supplied lr_scheduler Schedule; configure the "
                    "recipe via its constructor instead.",
                )
            )
        return raw

    name = raw
    if hasattr(name, "value"):
        name = name.value
    base_lr = args.learning_rate
    # Tolerate ``None`` for callers that pre-date the
    # ``field(default_factory=dict)`` migration (HF parity:
    # ``TrainingArguments`` accepts JSON-string here too, but we don't —
    # see ``TrainingArguments`` field docstring).
    kwargs = dict(args.lr_scheduler_kwargs or {})

    if name in _DEFERRED:
        raise NotImplementedError(
            f"lr_scheduler={name!r} is a recognized HuggingFace scheduler "
            f"that Trainer doesn't implement yet. If you need it, please open "
            f"an issue at https://github.com/JetBrains-Research/opaque/issues. "
            f"Currently supported: {sorted(_ALLOWED_KWARGS)}."
        )
    if name not in _ALLOWED_KWARGS:
        raise ConfigurationError(
            *(f"Unknown lr_scheduler={name!r}. Supported: {sorted(_ALLOWED_KWARGS)}.",)
        )

    _validate_kwargs(name, kwargs)

    W = get_warmup_steps(num_training_steps, args.warmup_steps)
    decay_steps = max(1, num_training_steps - W)

    if name == "constant":
        return constant_schedule(base_lr)
    if name == "constant_with_warmup":
        return (
            with_warmup(base_lr, transition_steps=W)
            if W > 0
            else constant_schedule(base_lr)
        )
    if name == "warmup_stable_decay":
        return _build_wsd(base_lr, W, num_training_steps, kwargs)

    warmup_init = 0.0  # warmup ramp starts at 0 unless overridden below.

    if name == "linear":
        decay = linear_schedule(base_lr, 0.0, decay_steps, transition_begin=W)
    elif name == "cosine":
        decay = cosine_schedule(
            base_lr,
            0.0,
            decay_steps,
            transition_begin=W,
            num_cycles=kwargs.get("num_cycles", 0.5),
        )
    elif name == "polynomial":
        lr_end = float(kwargs.get("lr_end", 1e-7))
        if lr_end >= base_lr:
            raise ConfigurationError(
                *(f"lr_end ({lr_end}) must be smaller than initial lr ({base_lr})",)
            )
        decay = polynomial_schedule(
            base_lr,
            lr_end,
            kwargs.get("power", 1.0),
            decay_steps,
            transition_begin=W,
        )
    elif name == "inverse_sqrt":
        # Explicit ``None`` check (not truthiness): a user-supplied
        # ``timescale=0`` is invalid but should not silently fall through to
        # the warmup-steps default.
        ts = kwargs.get("timescale")
        timescale = ts if ts is not None else (W or 10_000)
        decay = inverse_sqrt_schedule(
            base_lr, transition_steps=timescale, transition_begin=W
        )
    elif name == "cosine_with_restarts":
        cycles = int(kwargs.get("num_cycles", 1))
        # Real-valued cycle length: ``with_restarts`` places restart
        # boundaries at ``k * decay_steps / cycles`` and the inner half-cosine
        # must span the *same* fractional length, else (for non-divisible
        # ``decay_steps``) the cosine bottoms out early and the cycle shapes
        # drift from HF's fractional-progress formula.  ``cosine_schedule``'s
        # factory truncates ``transition_steps`` to int, so build the inner
        # ``CosineSchedule`` directly to preserve the fractional span.
        cycle_length = decay_steps / cycles
        inner = CosineSchedule(
            init_value=float(base_lr),
            end_value=0.0,
            transition_steps=cycle_length,
            num_cycles=0.5,
        )
        decay = with_restarts(
            inner, transition_steps=decay_steps, num_cycles=cycles, transition_begin=W
        )
    elif name == "cosine_with_min_lr":
        end_value = _resolve_min_lr(base_lr, kwargs)
        decay = cosine_schedule(
            base_lr,
            end_value,
            decay_steps,
            transition_begin=W,
            num_cycles=kwargs.get("num_cycles", 0.5),
        )
    elif name == "cosine_warmup_with_min_lr":
        # Warmup-ramp divergence vs HF.  HF's
        # ``_get_cosine_with_min_lr_schedule_with_warmup_lr_rate_lambda``
        # (``transformers/optimization.py``) computes the warmup factor
        # as ``r + (1 - r) * step / (W - 1)`` (linear ramp anchored at
        # exact endpoints ``r`` at step 0 and ``1.0`` at step ``W - 1``).
        # We compose ``with_warmup`` instead, whose factor is
        # ``r + (1 - r) * progress`` with ``progress = step / W``.  Both
        # ramps agree at step 0 (``r``) and at the post-warmup boundary
        # (``decay(W) == base_lr``), but the interior values differ
        # whenever ``warmup_lr_rate > 0``.  Strict numerical parity with
        # HF's ``cosine_warmup_with_min_lr`` would require an
        # ``(W - 1)``-anchored ramp primitive that doesn't currently
        # exist in :mod:`opaque.scheduling`; the current behavior is
        # accepted as a documented divergence.
        end_value = _resolve_min_lr(base_lr, kwargs)
        decay = cosine_schedule(
            base_lr,
            end_value,
            decay_steps,
            transition_begin=W,
            num_cycles=kwargs.get("num_cycles", 0.5),
        )
        warmup_init = float(kwargs.get("warmup_lr_rate") or 0.0)
    else:  # pragma: no cover — guarded above.
        raise ConfigurationError(
            *(f"Unsupported warmup scheduler configuration: {name}",)
        )

    if W == 0:
        return decay
    return with_warmup(decay, transition_steps=W, ramp="linear", init_value=warmup_init)


def _resolve_min_lr(base_lr: float, kwargs: dict[str, Any]) -> float:
    """Translate HF's mutually-exclusive ``min_lr`` / ``min_lr_rate`` to an
    absolute end value for our cosine_schedule."""
    min_lr = kwargs.get("min_lr")
    min_lr_rate = kwargs.get("min_lr_rate")
    if min_lr is not None and min_lr_rate is not None:
        raise ConfigurationError(*("Set only one of min_lr or min_lr_rate.",))
    if min_lr is not None:
        return float(min_lr)
    if min_lr_rate is not None:
        return float(min_lr_rate) * base_lr
    raise ConfigurationError(
        *(
            "lr_scheduler_kwargs must include exactly one of {'min_lr': ...} "
            "or {'min_lr_rate': ...}.",
        )
    )


_WSD_RAMP_TYPES = {"linear", "cosine", "1-sqrt"}
_WSD_DECAY_TYPES = {"linear", "cosine", "1-sqrt"}


def _build_wsd(
    base_lr: float,
    W: int,
    num_training_steps: int,
    kwargs: dict[str, Any],
) -> Callable[[int], float]:
    """Build HF's warmup_stable_decay (WSD) schedule.

    Three phases, each modulated by ``min_lr_ratio`` so the schedule's
    floor is ``min_lr_ratio * base_lr``:

    1. warmup: ramp from floor to ``base_lr`` over ``W`` steps, shape
       chosen by ``warmup_type``.
    2. stable: ``base_lr`` for ``num_stable_steps``.
    3. decay: ``base_lr`` → floor over ``num_decay_steps``, shape chosen
       by ``decay_type``.  After decay, holds at floor.

    ``warmup_type`` and ``decay_type`` accept ``"linear"``, ``"cosine"``,
    or ``"1-sqrt"``.
    """
    warmup_type = kwargs.get("warmup_type", "linear")
    if warmup_type not in _WSD_RAMP_TYPES:
        raise ConfigurationError(
            *(
                f"warmup_stable_decay warmup_type must be one of "
                f"{sorted(_WSD_RAMP_TYPES)}; got {warmup_type!r}.",
            )
        )
    decay_type = kwargs.get("decay_type", "cosine")
    if decay_type not in _WSD_DECAY_TYPES:
        raise ConfigurationError(
            *(
                f"warmup_stable_decay decay_type must be one of "
                f"{sorted(_WSD_DECAY_TYPES)}; got {decay_type!r}.",
            )
        )

    if "num_decay_steps" not in kwargs:
        raise ConfigurationError(
            *(
                "warmup_stable_decay requires lr_scheduler_kwargs={'num_decay_steps': ...}.",
            )
        )
    D = int(kwargs["num_decay_steps"])
    S = int(kwargs.get("num_stable_steps", num_training_steps - W - D))
    min_lr_ratio = float(kwargs.get("min_lr_ratio", 0.0))
    num_cycles = float(kwargs.get("num_cycles", 0.5))

    floor = min_lr_ratio * base_lr

    # Decay curve: base_lr at step (W+S), floor at step (W+S+D), held at floor after.
    if decay_type == "linear":
        decay = linear_schedule(
            base_lr, floor, transition_steps=D, transition_begin=W + S
        )
    elif decay_type == "cosine":
        decay = cosine_schedule(
            base_lr,
            floor,
            transition_steps=D,
            transition_begin=W + S,
            num_cycles=num_cycles,
        )
    else:  # "1-sqrt"
        decay = one_minus_sqrt_schedule(
            base_lr,
            floor,
            transition_steps=D,
            transition_begin=W + S,
        )

    # Cosine post-decay continues oscillating; linear/1-sqrt clip naturally.
    # Clamp at floor for the trailing region.
    end_of_decay = W + S + D

    def core(step: int) -> float:
        if step >= end_of_decay:
            return floor
        return decay(step)

    if W <= 0:
        return core

    return with_warmup(
        core,
        transition_steps=W,
        ramp=warmup_type,
        init_value=min_lr_ratio,
    )
