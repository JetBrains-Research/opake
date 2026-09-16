r"""Internal MSE-optimal per-group noise allocation.

This module is **not** a supported public API (leading underscore).  Callers
inside the monorepo use it to share per-group Gaussian allocation without
creating package dependency cycles.

The exposed :func:`per_group_noise_stddev` helper implements the single-stream
MSE-optimal allocation.
"""

from __future__ import annotations

import math

from opaque.api.engine.types import PerGroup
from opaque.exceptions import ConfigurationError, InputTypeError


def per_group_noise_stddev(max_norm: PerGroup, noise_multiplier: float) -> PerGroup:
    r"""Compute MSE-optimal per-group noise standard deviations.

    Given per-group contribution bounds :math:`B_1, \dots, B_K`, returns
    per-group noise standard deviations:

    .. math::

        \sigma_i = \text{nm} \cdot
            \sqrt{B_i \cdot \sum_j B_j}

    This allocation minimizes the total noise MSE
    :math:`\sum_i d_i \sigma_i^2` (for equal group dimensions) among all
    allocations satisfying the Mahalanobis privacy constraint
    :math:`\sum_i (C_i/n)^2 / \sigma_i^2 \le 1/\text{nm}^2`.

    Privacy accounting is ``gaussian(nm)`` — identical to isotropic noise,
    with no composition penalty regardless of the number of groups.

    Args:
        max_norm: Per-group contribution bounds, typically
            ``clipped_grads.max_norm`` from per-group clipping.
        noise_multiplier: The noise multiplier used for privacy
            accounting via ``gaussian(nm)``.

    Returns:
        :class:`~opaque.types.PerGroup` with per-group noise standard
        deviations.

    Raises:
        TypeError: If ``max_norm`` is not ``PerGroup``.
        ValueError: If ``noise_multiplier`` is negative or any group bound
            is negative.
    """
    if not isinstance(max_norm, PerGroup):
        raise InputTypeError(
            *(
                "per_group_noise_stddev requires a PerGroup max_norm, "
                f"got {type(max_norm).__name__}.",
            )
        )
    if noise_multiplier < 0:
        raise ConfigurationError(
            *(f"noise_multiplier must be non-negative, got {noise_multiplier}",)
        )
    for group_name, value in max_norm.values.items():
        if value < 0:
            raise ConfigurationError(
                *(
                    "per-group bounds must be non-negative, "
                    f"got {value} for group '{group_name}'.",
                )
            )
    if noise_multiplier == 0.0:
        # Non-private run: zero noise regardless of the (possibly infinite,
        # i.e. clipping-disabled) per-group bounds.  Short-circuit before the
        # ``0 * sqrt(c * sum_c)`` product, which would be NaN when any bound
        # is +inf.
        return PerGroup(max_norm.groups, dict.fromkeys(max_norm.values, 0.0))
    sum_c = sum(max_norm.values.values())
    return PerGroup(
        max_norm.groups,
        {
            k: noise_multiplier * math.sqrt(c * sum_c)
            for k, c in max_norm.values.items()
        },
    )


__all__ = ["per_group_noise_stddev"]
