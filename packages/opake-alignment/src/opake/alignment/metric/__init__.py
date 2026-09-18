"""opake.alignment.metric — shared token-level metrics.

Detached, un-noised token telemetry; release is outside Opake's DP accounting.
Preference metrics live in :mod:`opake.alignment.dpo.metric`.
"""

from opake.api.alignment.metric import entropy_from_logits, mean_token_accuracy

__all__ = ["entropy_from_logits", "mean_token_accuracy"]
