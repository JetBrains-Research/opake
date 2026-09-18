"""DP-SGD samplers façade — Poisson and k-out-of-t allocation.

:class:`PoissonSampler` covers plain Poisson subsampling (default). Pass
``truncated_batch_size`` to cap batch size (truncated Poisson; weaker
privacy than plain Poisson at the same rate unless noise is recalibrated —
use matching accounting).

:class:`KOutOfTSampler` supports block and total allocation. Block allocation
draws one participation in each of ``k`` nearly equal blocks. Total allocation
chooses ``k`` steps uniformly from the horizon. Both pair with
:func:`opake.dpsgd.accounting.k_out_of_t`; total allocation currently uses the
block result as a conservative privacy bound.
"""

from opake.api.dpsgd.sampling import (
    KOutOfTSampler,
    PoissonSampler,
)

__all__ = ["KOutOfTSampler", "PoissonSampler"]
