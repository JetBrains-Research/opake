"""opake.alignment.dpo.metric façade — preference reward telemetry.

General token-level metrics live in the shared impl
:mod:`opake.api.alignment.metric`.
"""

from opake.api.alignment.dpo.metric import reward_metrics

__all__ = ["reward_metrics"]
