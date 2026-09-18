"""Lion optimizer (Chen et al., 2023).

Sign-of-momentum update::

    c_t = β₁ m_{t-1} + (1 − β₁) g_t          # interpolate
    update = sign(c_t)                        # sign step (no second moment)
    m_t = β₂ m_{t-1} + (1 − β₂) g_t           # standard EMA for next step

Reference:
    Chen et al., "Symbolic Discovery of Optimization Algorithms",
    arXiv:2302.06675.

DP notes.  Lion has no second moment, so neither the φ-EMA bias
correction (DP-AdamW-BC) nor private second-moment substitution apply.
The update accepts noised gradients unchanged; ``sign()`` produces a
direction-only step.  No DP-aware mode is provided.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from opake.exceptions import ConfigurationError

try:
    from torchopt.base import GradientTransformation
except ImportError as exc:
    raise ImportError(  # noqa: TRY003 - preserve standard Python error contract
        "torchopt is required for opake.optimizers. "
        "Install it with: pip install 'torchopt>=0.7.3'"
    ) from exc

from opake.api.optimizers._chain import make_optimizer_chain
from opake.pytree import tree_map

if TYPE_CHECKING:
    from opake.types import TensorPytree

_LR = float | Callable[[int], float]


@dataclasses.dataclass(frozen=True)
class LionState:
    """State for Lion's single momentum buffer."""

    m: TensorPytree
    step: int


def _scale_by_lion(b1: float, b2: float) -> GradientTransformation:
    """Sign-of-momentum-blended-gradient update with delayed-EMA momentum."""

    def init_fn(params: Any) -> LionState:
        return LionState(m=tree_map(torch.zeros_like, params), step=0)

    def update_fn(
        updates: Any,
        state: LionState,
        *,
        params: Any = None,
        inplace: bool = False,
    ) -> tuple[Any, LionState]:
        # Direction: sign of (b1 * m + (1-b1) * g).
        direction = tree_map(
            lambda m, g: torch.sign(b1 * m + (1 - b1) * g),
            state.m,
            updates,
        )
        # Momentum update for next step uses β₂.
        new_m = tree_map(lambda m, g: b2 * m + (1 - b2) * g, state.m, updates)
        return direction, LionState(m=new_m, step=state.step + 1)

    return GradientTransformation(init_fn, update_fn)


def lion(
    lr: _LR = 1e-4,
    betas: tuple[float, float] = (0.9, 0.99),
    weight_decay: float = 0.0,
    *,
    decoupled_weight_decay: bool = True,
) -> GradientTransformation:
    """Create a Lion optimizer.

    Args:
        lr: Learning rate (Lion typically wants ~3-10× smaller than AdamW).
        betas: ``(β₁, β₂)`` — β₁ for the direction blend, β₂ for the
            momentum buffer.  Defaults match the paper.
        weight_decay: Weight-decay coefficient; usually larger than for
            AdamW (Lion's effective LR per coordinate is the constant 1).
        decoupled_weight_decay: Same semantics as
            :func:`opake.optimizers._adamw`.

    Returns:
        A ``torchopt.base.GradientTransformation``.
    """
    if len(betas) != 2:  # noqa: PLR2004 - Lion exposes the documented beta pair
        raise ConfigurationError(
            *(f"betas must contain exactly two values, got {betas}",)
        )
    b1, b2 = betas
    if not 0 <= b1 < 1:
        raise ConfigurationError(*(f"beta_1 must satisfy 0 <= beta_1 < 1, got {b1}",))
    if not 0 <= b2 < 1:
        raise ConfigurationError(*(f"beta_2 must satisfy 0 <= beta_2 < 1, got {b2}",))
    if weight_decay < 0:
        raise ConfigurationError(
            *(f"weight_decay must be non-negative, got {weight_decay}",)
        )
    return make_optimizer_chain(
        _scale_by_lion(b1, b2),
        lr=lr,
        weight_decay=weight_decay,
        decoupled_weight_decay=decoupled_weight_decay,
    )


__all__ = ["LionState", "lion"]
