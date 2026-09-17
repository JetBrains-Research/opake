"""opake.alignment.dpo.loss — the DPO loss-construction toolkit.

Everything needed to assemble a per-example DPO loss, exposed as direct functions
(mapping a config string to one of them is the caller's concern):

- Per-sequence log-probabilities: ``sequence_logp`` and its memory-efficient
  fused drop-in ``fused_sequence_logp`` (project hidden states through the
  ``lm_head`` without materialising logits); pass ``length_normalized=True`` for
  the per-token mean reward used by SimPO / ORPO.
- Per-pair heads on log-ratios: ``sigmoid_loss``, ``hinge_loss``, …. Used
  reference-free by passing the policy logp itself as the log-ratio (CPO is
  ``mpo_combine`` of a reference-free ``sigmoid_loss`` and ``chosen_nll_loss``).
- Reference-free heads on (length-normalized) log-probs: ``simpo_loss``
  (length-normalized sigmoid with a target margin) and ``odds_ratio_loss``
  (ORPO; takes log-probs directly, not log-ratios).
- Log-ratio combinators for composite objectives: ``f_divergence_remap`` /
  ``f_divergence_logits``, ``mpo_combine``, ``wpo_weights``, ``ld_dpo_split``.

A DPO ``per_example_loss`` is ``head(sequence_logp(...) - ref_logp, …)``; see
``examples/train_dpo.py``.

Naming is ``<method>[_<variant>]_loss``: the preference/pairwise context is
implied by this namespace, and a ``_<variant>`` qualifier appears only to
disambiguate co-existing variants of one method (``apo_zero`` / ``apo_down``).
``chosen_nll_loss`` is the chosen-completion NLL regulariser used in MPO/RPO/CPO
blends — not the SFT method's loss (that lives in ``opake.alignment.sft.loss``).
"""

from opake.api.alignment.dpo.loss import (
    apo_down_loss,
    apo_zero_loss,
    bco_loss,
    chosen_nll_loss,
    discopop_loss,
    exo_loss,
    f_divergence_logits,
    f_divergence_remap,
    hinge_loss,
    ipo_loss,
    ld_dpo_split,
    mpo_combine,
    nca_loss,
    odds_ratio_loss,
    robust_loss,
    sigmoid_loss,
    simpo_loss,
    sppo_loss,
    wpo_weights,
)
from opake.api.alignment.logprob import (
    fused_sequence_logp,
    sequence_logp,
)

__all__ = [
    # per-sequence log-probabilities
    "sequence_logp",
    "fused_sequence_logp",
    # per-pair heads (on log-ratios)
    "sigmoid_loss",
    "hinge_loss",
    "robust_loss",
    "ipo_loss",
    "discopop_loss",
    "chosen_nll_loss",
    "apo_zero_loss",
    "apo_down_loss",
    "exo_loss",
    "nca_loss",
    "bco_loss",
    "sppo_loss",
    # reference-free heads (on length-normalized log-probs)
    "simpo_loss",
    "odds_ratio_loss",
    # log-ratio combinators (composite objectives)
    "f_divergence_remap",
    "f_divergence_logits",
    "mpo_combine",
    "wpo_weights",
    "ld_dpo_split",
]
