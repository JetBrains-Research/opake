"""MoE router-load release transformation for privacy accounting.

The clipped gradient and the batch router load of one step (see
:func:`opaque.dpsgd.clipping.moe_clipped_grad`) are one Gaussian on their
concatenation with whitened sensitivity ``1/nm² + ratio/nm²``, i.e. a
Gaussian at the joint multiplier ``nm / sqrt(1 + ratio)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from opaque.api.accounting.core import _native
from opaque.api.accounting.core._base import DpProcess, Pld
from opaque.api.accounting.core._pld_cache import pld_cache
from opaque.api.accounting.core.mechanisms._nonprivate import NonPrivate
from opaque.api.accounting.dpsgd.mechanisms._gaussian import Gaussian
from opaque.exceptions import ConfigurationError, InputTypeError

#: Mechanism types accepted as MoeAux inner.
_Inner = Gaussian | NonPrivate

DEFAULT_RATIO = 0.02


@dataclass(frozen=True, slots=True)
class MoeAux(DpProcess):
    """MoE router-load release transformation.

    Wraps an ``inner`` Gaussian mechanism and adds the cost of releasing the
    batch router load alongside the gradient with ``ratio`` of the whitened
    sensitivity, so the joint step is a Gaussian at
    ``inner.noise_multiplier / sqrt(1 + ratio)``.
    """

    inner: _Inner
    ratio: float = DEFAULT_RATIO

    def __post_init__(self) -> None:
        # Validated here so direct construction and deserialization cannot
        # price the load release as free.
        if not isinstance(self.inner, (Gaussian, NonPrivate)):
            raise InputTypeError(
                *(
                    "MoeAux requires a Gaussian or NonPrivate inner mechanism, "
                    f"got {type(self.inner).__name__}.",
                )
            )
        if not isinstance(self.ratio, (int, float)) or isinstance(self.ratio, bool):
            raise InputTypeError(*(f"ratio must be a number, got {self.ratio!r}",))
        if not math.isfinite(self.ratio) or self.ratio <= 0:
            raise ConfigurationError(
                *(f"ratio must be a positive finite number, got {self.ratio}",)
            )
        object.__setattr__(self, "ratio", float(self.ratio))

    @property
    def effective_noise_multiplier(self) -> float:
        """Joint multiplier ``nm / sqrt(1 + ratio)`` of the gradient-plus-load release.

        Returns ``0.0`` for a :class:`NonPrivate` inner (no noise).
        """
        match self.inner:
            case NonPrivate() | Gaussian(noise_multiplier=0):
                return 0.0
            case Gaussian(noise_multiplier=nm):
                return nm / math.sqrt(1.0 + self.ratio)

    @pld_cache(maxsize=8)
    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
        max_conv_grid: int | None = None,
        seed: int | None = None,
        mc_resolution: float | None = None,
        mc_failure_probability: float | None = None,
    ) -> Pld:
        from opaque.api.accounting.core.discretization import get_discretization

        config = get_discretization(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
            max_conv_grid=max_conv_grid,
            seed=seed,
            mc_resolution=mc_resolution,
            mc_failure_probability=mc_failure_probability,
        )
        native_cfg = config.to_native()
        effective = self.effective_noise_multiplier
        if effective == 0.0:
            return _native.non_private_pld(native_cfg)
        return _native.gaussian_pld(effective, native_cfg)


def moe_aux(inner: _Inner, *, ratio: float = DEFAULT_RATIO) -> MoeAux:
    """Account for the privacy cost of the MoE router-load release.

    Wraps an ``inner`` Gaussian and prices the load released by
    :func:`opaque.dpsgd.clipping.moe_clipped_grad` with the same ``ratio``:
    one Gaussian at ``inner.noise_multiplier / sqrt(1 + ratio)``, so at a
    fixed budget the gradient noise grows by ``sqrt(1 + ratio)``.

    Args:
        inner: ``gaussian(noise_multiplier)`` with the multiplier handed to
            ``gaussian_noise``.
        ratio: Share of the whitened sensitivity given to the load release,
            the same value given to ``moe_clipped_grad`` (default 0.02).

    Example::

        step = dpsgd_acc.poisson(
            dpsgd_acc.moe_aux(dpsgd_acc.gaussian(1.1), ratio=0.02),
            sample_rate=0.01,
        )
    """
    return MoeAux(inner=inner, ratio=ratio)
