"""DP-FTRL accounting mechanism factories impl (matrix factorization).

A single ``mf_gaussian(nm, strategy)`` factory builds the accounting
mechanism for any MF strategy — the strategy carries the structural data
(sensitivity, Gram, coefficients) and the amplifications dispatch on
``type(strategy)`` at PLD time.
"""

from opake.api.accounting.dpftrl.mechanisms._mf_gaussian import mf_gaussian

__all__ = ["mf_gaussian"]
