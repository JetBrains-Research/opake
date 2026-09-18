"""opake.alignment.sft.loss façade — re-exports the SFT loss functions.

``nll_loss`` / ``dft_loss`` (on logits) and their memory-efficient fused twins
``fused_nll_loss`` / ``fused_dft_loss`` (on hidden states + ``lm_head`` weight).
"""

from opake.api.alignment.sft.loss import (
    dft_loss,
    fused_dft_loss,
    fused_nll_loss,
    nll_loss,
)

__all__ = ["dft_loss", "fused_dft_loss", "fused_nll_loss", "nll_loss"]
