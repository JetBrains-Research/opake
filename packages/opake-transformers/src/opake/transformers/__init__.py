"""Hugging Face Trainer integration for Opake.

Small root over the trainer implementation package: the core
:class:`~opake.transformers.trainer.DPTrainer` primitives are re-exported here,
and the TRL-style trainers live under :mod:`opake.transformers.trl`.

Importing this module does **not** mutate Hugging Face globals — ``DPTrainer``
applies the runtime + per-model patches during construction. Scripts that use
HF primitives without ``DPTrainer`` should call
:func:`opake.patches.apply_runtime_patches` once for the global runtime shims.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from opake.api.transformers.trainer import (
    DPTrainer,
    TrainingArguments,
)

from . import trl

try:
    __version__ = _pkg_version("opake-transformers")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "DPTrainer",
    "TrainingArguments",
    "__version__",
    "trl",
]
