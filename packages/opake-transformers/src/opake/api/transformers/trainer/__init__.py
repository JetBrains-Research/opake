"""Trainer implementation package (``opake.api.transformers.trainer``).

Import the stable façade from :mod:`opake.transformers.trainer` or the
symbols re-exported from :mod:`opake.transformers`.
"""

from __future__ import annotations

from . import types
from ._trainer import Trainer
from ._training_arguments import TrainingArguments

__all__ = [
    "Trainer",
    "TrainingArguments",
    "types",
]
