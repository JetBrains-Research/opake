"""Public type definitions for :mod:`opake.optimizers`.

Re-exports the per-optimizer state dataclasses for type annotations. The
functional factories (``adam``, ``adamw``, ``lion``, …) live in
:mod:`opake.optimizers`.  Checkpointing uses :mod:`opake.serialization`
(``state_dict`` / ``from_state_dict``).
"""

from __future__ import annotations

from opake.api.optimizers._adadelta import AdadeltaState
from opake.api.optimizers._adafactor import AdafactorState
from opake.api.optimizers._adagrad import AdagradState
from opake.api.optimizers._adam import AdamState
from opake.api.optimizers._ademamix import AdEMAMixState
from opake.api.optimizers._lion import LionState
from opake.api.optimizers._radam import RAdamState
from opake.api.optimizers._rmsprop import RMSpropState
from opake.api.optimizers._schedule_free import ScheduleFreeState

__all__ = [
    "AdEMAMixState",
    "AdadeltaState",
    "AdafactorState",
    "AdagradState",
    "AdamState",
    "LionState",
    "RAdamState",
    "RMSpropState",
    "ScheduleFreeState",
]
