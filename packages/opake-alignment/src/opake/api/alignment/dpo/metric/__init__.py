"""DPO metrics impl — preference reward telemetry.

``reward_metrics`` returns detached, un-noised values; release is outside
Opake's DP accounting. Token metrics live in
:mod:`opake.api.alignment.metric`.
"""

from opake.api.alignment.dpo.metric._reward import reward_metrics

__all__ = ["reward_metrics"]
