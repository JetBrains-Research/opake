"""TRL-style class trainers for :class:`Trainer` (implementation package).

``SFTTrainer`` / ``DPOTrainer`` are thin :class:`~opaque.api.transformers.trainer.Trainer`
subclasses that wire the merged ``opaque-alignment`` primitives through
TRL-shaped methods. Import the stable façade from :mod:`opaque.transformers.trl`.

See ``docs/development/sft-dpo-trainers-plan.md`` for the design.
"""

from __future__ import annotations

from ._dpo_config import DPOConfig
from ._dpo_trainer import DPOTrainer
from ._sft_config import SFTConfig
from ._sft_trainer import SFTTrainer

__all__ = [
    "DPOConfig",
    "DPOTrainer",
    "SFTConfig",
    "SFTTrainer",
]
