"""DP Trainer public façade.

Implementation lives in :mod:`opake.api.transformers.trainer`.
"""

from __future__ import annotations

from opake.api.transformers.trainer import (
    Trainer,
    TrainingArguments,
)

from . import types

__all__ = [
    "Trainer",
    "TrainingArguments",
    "types",
]
