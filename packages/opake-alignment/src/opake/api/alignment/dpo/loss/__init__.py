"""DPO loss family impl — per-pair scalar losses and log-ratio helpers.

All variants are pure, elementwise, per-example functions.

Direct functions only — there is no string registry / resolver / variant enum
(mirrors ``opake.api.alignment.sft.loss``). A name→function dispatch is the
caller's concern: a config-string consumer (trainer / CLI) builds its own small
mapping at the call site, as ``examples/train_dpo.py`` does. The
``f_divergence_*``, ``mpo_combine``, ``wpo_weights``, and ``ld_dpo_split``
helpers preprocess log-ratios before a variant.
"""

from opake.api.alignment.dpo.loss._apo import apo_down_loss, apo_zero_loss
from opake.api.alignment.dpo.loss._bco import bco_loss
from opake.api.alignment.dpo.loss._chosen_nll import chosen_nll_loss
from opake.api.alignment.dpo.loss._discopop import discopop_loss
from opake.api.alignment.dpo.loss._exo import exo_loss
from opake.api.alignment.dpo.loss._f_divergence import (
    f_divergence_logits,
    f_divergence_remap,
)
from opake.api.alignment.dpo.loss._hinge import hinge_loss
from opake.api.alignment.dpo.loss._ipo import ipo_loss
from opake.api.alignment.dpo.loss._ld_dpo import ld_dpo_split
from opake.api.alignment.dpo.loss._mpo import mpo_combine
from opake.api.alignment.dpo.loss._nca import nca_loss
from opake.api.alignment.dpo.loss._orpo import odds_ratio_loss
from opake.api.alignment.dpo.loss._robust import robust_loss
from opake.api.alignment.dpo.loss._sigmoid import sigmoid_loss
from opake.api.alignment.dpo.loss._simpo import simpo_loss
from opake.api.alignment.dpo.loss._sppo import sppo_loss
from opake.api.alignment.dpo.loss._wpo import wpo_weights

__all__ = [
    # variants
    "sigmoid_loss",
    "hinge_loss",
    "robust_loss",
    "ipo_loss",
    "simpo_loss",
    "discopop_loss",
    "chosen_nll_loss",
    "apo_zero_loss",
    "apo_down_loss",
    "exo_loss",
    "nca_loss",
    "bco_loss",
    "sppo_loss",
    "odds_ratio_loss",
    # helpers
    "f_divergence_remap",
    "f_divergence_logits",
    "mpo_combine",
    "wpo_weights",
    "ld_dpo_split",
]
