"""Trainer public façade.

Implementation lives in :mod:`opaque.api.transformers.trainer`.
"""

from __future__ import annotations

from opaque.api.transformers.trainer import (
    Trainer,
    TrainingArguments,
)

from . import types

__all__ = [
    "Trainer",
    "TrainingArguments",
    "types",
]
