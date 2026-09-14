"""TRL-style trainers — public façade.

Implementation lives in :mod:`opaque.api.transformers.trl`.

``SFTTrainer`` / ``DPOTrainer`` mirror ``trl.SFTTrainer`` / ``trl.DPOTrainer``
in structure and method names (iteration 1), built on Opaque's per-example DP
:class:`~opaque.transformers.trainer.Trainer` and consuming the
``opaque.alignment`` primitives.
"""

from __future__ import annotations

from opaque.api.transformers.trl import (
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
