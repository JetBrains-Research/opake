"""Public result types for :class:`Trainer`.

``EvaluationResult`` and ``TrainOutput`` are the return shapes of the trainer's
``evaluate`` / ``predict`` / ``evaluation_loop`` and ``train`` methods. They
live here (rather than next to the trainer internals) so callers can import the
types without pulling in the trainer module.
"""

from __future__ import annotations

import dataclasses
from typing import Any, NamedTuple

__all__ = ["EvaluationResult", "TrainOutput"]


@dataclasses.dataclass
class EvaluationResult:
    """Output of :meth:`Trainer.evaluation_loop` /
    :meth:`Trainer.evaluate` / :meth:`Trainer.predict`.

    Fields mirror HF's ``EvalLoopOutput`` (``predictions``, ``label_ids``,
    ``metrics``, ``num_samples``); ``predict`` returns the same shape
    rather than the separate ``PredictionOutput`` HF used historically.
    """

    predictions: Any | None
    label_ids: Any | None
    metrics: dict[str, float]
    num_samples: int


class TrainOutput(NamedTuple):
    """Return type of ``Trainer.train()``, mirroring HF's TrainOutput."""

    global_step: int
    training_loss: float
    metrics: dict[str, float]
