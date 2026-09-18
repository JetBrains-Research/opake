"""DP-FTRL accounting mechanism factories façade (matrix factorization).

Single factory :func:`mf_gaussian(nm, strategy)` builds the accounting
mechanism for every MF strategy from :mod:`opake.dpftrl.noise`.
"""

from opake.api.accounting.dpftrl.mechanisms import mf_gaussian

__all__ = ["mf_gaussian"]
