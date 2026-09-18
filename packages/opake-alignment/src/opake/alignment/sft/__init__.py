"""opake.alignment.sft — functional SFT primitives (method namespace).

Mirrors ``opake.dpsgd`` / ``opake.dpftrl``: the method's primitives live in
sub-concern subpackages, reached directly (e.g. ``sft.loss.nll_loss``,
``sft.collator.language_modeling_collator``):

- ``loss``     — ``nll_loss`` / ``dft_loss`` and their memory-efficient fused
  twins ``fused_nll_loss`` / ``fused_dft_loss``.
- ``collator`` — language-modeling (SFT) collator factory.
"""

from opake.alignment.sft import collator, loss

__all__ = ["collator", "loss"]
