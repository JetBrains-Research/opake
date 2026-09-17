"""opake-alignment implementation namespace: SFT method.

Method-first layout (mirrors ``opake.dpsgd`` / ``opake.dpftrl``): the SFT
method owns its loss math under ``sft/loss`` (and, as it lands, its collator
under ``sft/collator``). Shared primitives (logprob, metric, data) are
reimported from ``opake.api.alignment.*`` at their use sites.
"""

__all__: list[str] = []
