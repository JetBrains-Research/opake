"""MoE router-load release transformation for privacy accounting.

Prices the joint release of :func:`opaque.dpsgd.clipping.moe_clipped_grad`:
the clipped gradient and the batch router load of one step are one Gaussian
mechanism on their concatenation, and whitening each half by its own noise
scale shows the joint per-record sensitivity to be ``1/nm² + ratio/nm²``
(the load half is allocated ``ratio`` of the gradient half's whitened
sensitivity).  The step is therefore a Gaussian mechanism at the joint
multiplier ``nm / sqrt(1 + ratio)``, which is what the amplifiers and the
PLD below evaluate.  The same ``ratio`` given to the clipper sets its load
noise; see :mod:`opaque.api.engine.clipping._moe`.
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
        # Validate here (not only in ``moe_aux()``) so direct construction and
        # deserialization -- the generic DpProcess codec rebuilds with
        # ``cls(**kwargs)`` -- cannot produce an instance that prices the load
        # release as free.
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

    Wraps an ``inner`` Gaussian mechanism and prices the batch router load
    released by :func:`opaque.dpsgd.clipping.moe_clipped_grad` with the same
    ``ratio``: the joint release is one Gaussian at multiplier
    ``inner.noise_multiplier / sqrt(1 + ratio)``, so at a fixed privacy budget
    the gradient noise is inflated by exactly ``sqrt(1 + ratio)`` (about one
    percent at the default).

    Args:
        inner: Base mechanism, ``gaussian(noise_multiplier)`` with the same
            multiplier handed to ``gaussian_noise``.
        ratio: Share of the whitened sensitivity given to the load release,
            the same value given to ``moe_clipped_grad`` (default 0.02).

    Returns:
        A :class:`MoeAux` process with an
        :attr:`~MoeAux.effective_noise_multiplier` property.

    Example::

        step = dpsgd_acc.poisson(
            dpsgd_acc.moe_aux(dpsgd_acc.gaussian(1.1), ratio=0.02),
            sample_rate=0.01,
        )
    """
    match inner:
        case Gaussian() | NonPrivate():
            pass
        case _:
            raise InputTypeError(
                *(
                    "moe_aux() requires a Gaussian or NonPrivate inner mechanism, "
                    f"got {type(inner).__name__}.",
                )
            )
    return MoeAux(inner=inner, ratio=ratio)
