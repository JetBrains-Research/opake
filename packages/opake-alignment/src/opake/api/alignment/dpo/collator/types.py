"""Input and output schemas for the DPO preference collator."""

from __future__ import annotations

from typing import TYPE_CHECKING, NotRequired, TypedDict

if TYPE_CHECKING:
    import torch

__all__ = ["PreferenceBatch", "PreferenceExample"]


class PreferenceExample(TypedDict):
    """One tokenized chosen/rejected preference pair."""

    chosen_input_ids: list[int]
    rejected_input_ids: list[int]
    chosen_completion_mask: list[int]
    rejected_completion_mask: list[int]
    ref_chosen_logps: NotRequired[float]
    ref_rejected_logps: NotRequired[float]


class PreferenceBatch(TypedDict):
    """Tensor batch returned by :func:`preference_collator`."""

    chosen_input_ids: torch.Tensor
    chosen_attention_mask: torch.Tensor
    chosen_completion_mask: torch.Tensor
    rejected_input_ids: torch.Tensor
    rejected_attention_mask: torch.Tensor
    rejected_completion_mask: torch.Tensor
    ref_chosen_logps: NotRequired[torch.Tensor]
    ref_rejected_logps: NotRequired[torch.Tensor]
