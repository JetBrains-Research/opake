"""Hugging Face Trainer integration for Opaque.

Small root over the trainer implementation package: the core
:class:`~opaque.transformers.trainer.Trainer` primitives are re-exported here,
and the TRL-style trainers live under :mod:`opaque.transformers.trl`.

Importing this module does **not** mutate Hugging Face globals — ``Trainer``
applies the runtime + per-model patches during construction. Scripts that use
HF primitives without ``Trainer`` should call
:func:`opaque.patches.apply_runtime_patches` once for the global runtime shims.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from opaque.api.transformers.trainer import (
    Trainer,
    TrainingArguments,
)

from . import trl

try:
    __version__ = _pkg_version("opaque-transformers")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "Trainer",
    "TrainingArguments",
    "__version__",
    "trl",
]
