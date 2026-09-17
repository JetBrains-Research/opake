"""Empirical privacy auditing façade — canary-based with pluggable attacks.

Quick start::

    import opake.auditing as auditing
    from opake.random import key

    cf = auditing.coin_flip(dataset, num_canaries=1000, key=key(42))
    train_data = dataset.select(cf.train_indices(len(dataset)))

    # ... DP-SGD training loop ...

    scores = auditing.loss_scores(loss_fn, params,
                                   batch_argnums=(1,),
                                   coin_flip=cf, dataset=dataset,
                                   batch_size=32, collate_fn=canary_collate)
    estimate = auditing.one_run(scores, coin_flip=cf)
    print(f"ε (empirical): {estimate.eps_delta().epsilon_at(delta=1e-5):.4f}")

Scoring with ``coin_flip=`` + ``dataset=`` pairs every score with the
dataset index of its canary; ``one_run`` joins scores to coin-flip labels
by those identifiers, so however the scoring pipeline orders its batches,
the audit cannot silently misalign.  A ``collate_fn`` that reorders
examples *within* a batch is the one exception — identifiers are attached
before collation, so keep it order-preserving.

Result data classes (``CanaryScores``, ``CoinFlip``, ``OneRunEstimate``)
live in :mod:`opake.auditing.types`.

References:
    - Xiang et al. (2025), https://arxiv.org/abs/2509.08704
    - Steinke, Nasr, Jagielski (2023), https://arxiv.org/abs/2305.08846
    - Carlini et al. (2022), https://arxiv.org/abs/2112.03570
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from opake.api.auditing import (
    canary_scores,
    coin_flip,
    gradient_scores,
    loss_scores,
    one_run,
)

try:
    __version__ = _pkg_version("opake-auditing")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "canary_scores",
    "coin_flip",
    "gradient_scores",
    "loss_scores",
    "one_run",
]
