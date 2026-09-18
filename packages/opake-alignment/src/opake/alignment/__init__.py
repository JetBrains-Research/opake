"""opake.alignment — functional primitives for DP-safe preference learning.

Method-first layout, mirroring ``opake.dpsgd`` / ``opake.dpftrl``: each
method owns its primitives under its own namespace —

- :mod:`opake.alignment.dpo` — DPO loss family (per-sequence logp, per-pair
  heads, log-ratio combinators), preference collator, reference-model helpers,
  reward telemetry, and preference prompt extraction.
- :mod:`opake.alignment.sft` — NLL / DFT losses and the language-modeling
  collator.
- :mod:`opake.alignment.data` — shared, method-agnostic chat-template data
  prep: install a training chat template, then tokenize chat turns into
  ``input_ids`` + a ``completion_mask`` for completion-only loss.
- :mod:`opake.alignment.metric` — shared, detached token-level metrics
  (next-token accuracy, prediction entropy) for eval logging.

Other shared, lower-level primitives (logprob) are internal impl under
``opake.api.alignment.*`` and are surfaced through the method that consumes
them (e.g. ``sequence_logp`` via :mod:`opake.alignment.dpo`), following the
shared-impl re-import pattern of ``opake.dpsgd.clipping``.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from opake.alignment import data, dpo, metric, sft

try:
    __version__ = _pkg_version("opake-alignment")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__", "data", "dpo", "metric", "sft"]
