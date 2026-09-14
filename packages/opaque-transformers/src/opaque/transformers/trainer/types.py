"""Public result types for :class:`Trainer` — façade over
:mod:`opaque.api.transformers.trainer.types`.
"""

from __future__ import annotations

from opaque.api.transformers.trainer.types import EvaluationResult, TrainOutput

__all__ = ["EvaluationResult", "TrainOutput"]
