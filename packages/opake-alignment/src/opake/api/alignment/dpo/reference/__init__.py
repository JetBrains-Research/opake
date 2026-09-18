"""Reference-model handling impl — precompute, adapter dispatch, EMA sync.

These run **outside vmap** (a forward pass over a dataset / PEFT adapter
toggles).
"""

from opake.api.alignment.dpo.reference._adapter import (
    null_ref_context,
    with_disabled_adapter,
)
from opake.api.alignment.dpo.reference._precompute import (
    compute_ref_logprobs_for_dataset,
)
from opake.api.alignment.dpo.reference._sync import ema_update_reference

__all__ = [
    "compute_ref_logprobs_for_dataset",
    "ema_update_reference",
    "null_ref_context",
    "with_disabled_adapter",
]
