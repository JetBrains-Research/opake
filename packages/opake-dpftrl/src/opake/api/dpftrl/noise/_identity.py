"""Identity MF strategy — uncorrelated noise baseline (DP-SGD via MF API).

Use ``mf_gaussian_noise(template, identity_strategy(), ...)`` for standard DP-SGD
with independent noise at each step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from opake.api.dpftrl.noise._strategy_codec import register_strategy
from opake.api.dpftrl.noise._streaming_matrix import StreamingMatrix, identity


@register_strategy
@dataclass(frozen=True, slots=True)
class IdentityStrategy:
    """Identity (uncorrelated) MF strategy — independent noise per step.

    Encoder ``C = I``: sensitivity ≡ 1 for any horizon; streaming matrix
    is the identity; gram matrix isn't read (the BnB Identity path uses
    a dedicated native primitive that exploits ``G = num_epochs · I_b``).
    """

    def coefficients(self, *, n_steps: int, **_) -> torch.Tensor:
        # First column of the n × n identity matrix.
        c = torch.zeros(n_steps, dtype=torch.float64)
        c[0] = 1.0
        return c

    def gram_matrix(self, **_) -> tuple[float, ...]:
        # The BnB Identity path never materialises the gram: it is exactly
        # ``num_epochs · I_b``, so the dominating pair collapses onto random
        # allocation and goes through the analytic transform instead.
        raise NotImplementedError(
            "IdentityStrategy does not materialize a gram matrix; BnB Identity "
            "uses ``random_allocation_gaussian_pld`` directly."
        )

    def streaming_matrix(self, **_) -> StreamingMatrix:
        return identity()

    def sensitivity(self, **_) -> float:
        return 1.0


def identity_strategy() -> IdentityStrategy:
    """Create an identity (DP-SGD) noise strategy."""
    return IdentityStrategy()


__all__ = ["IdentityStrategy", "identity_strategy"]
