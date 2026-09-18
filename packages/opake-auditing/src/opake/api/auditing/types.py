"""Public type definitions for :mod:`opake.auditing`.

Re-exports the auditing result data classes for type annotations. The
functional surface (``coin_flip``, ``loss_scores``, ``one_run``) lives
in the package init.
"""

from __future__ import annotations

from opake.api.auditing._coin_flip import CanaryScores, CoinFlip
from opake.api.auditing.one_run._estimate import OneRunEstimate

__all__ = ["CanaryScores", "CoinFlip", "OneRunEstimate"]
