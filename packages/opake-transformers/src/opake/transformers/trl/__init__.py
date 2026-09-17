"""TRL-style DP trainers — public façade.

Implementation lives in :mod:`opake.api.transformers.trl`.

``SFTTrainer`` / ``DPOTrainer`` mirror ``trl.SFTTrainer`` / ``trl.DPOTrainer``
in structure and method names (iteration 1), built on Opake's per-example DP
:class:`~opake.transformers.trainer.DPTrainer` and consuming the
``opake.alignment`` primitives.
"""

from __future__ import annotations

from opake.api.transformers.trl import (
    DPOConfig,
    DPOTrainer,
    SFTConfig,
    SFTTrainer,
)

__all__ = [
    "DPOConfig",
    "DPOTrainer",
    "SFTConfig",
    "SFTTrainer",
]
