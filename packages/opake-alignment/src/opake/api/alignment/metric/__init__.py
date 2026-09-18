"""Shared alignment metrics impl — exact token-level telemetry.

These metrics are detached but un-noised; release is outside Opake's DP
accounting. Preference metrics live in
:mod:`opake.api.alignment.dpo.metric`.
"""

from opake.api.alignment.metric._token import entropy_from_logits, mean_token_accuracy

__all__ = ["entropy_from_logits", "mean_token_accuracy"]
