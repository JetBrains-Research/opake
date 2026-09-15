"""Effective Gaussian multiplier of the DP-SGD mechanism family.

The subsampling amplifiers evaluate a Gaussian PLD at one multiplier; the
transformations riding a side release (:class:`AdaClip`, :class:`MoeAux`)
carry it as ``effective_noise_multiplier``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from opaque.api.accounting.core.mechanisms._nonprivate import NonPrivate
from opaque.api.accounting.dpsgd.mechanisms._adaclip import AdaClip
from opaque.api.accounting.dpsgd.mechanisms._gaussian import Gaussian
from opaque.api.accounting.dpsgd.mechanisms._moe import MoeAux

if TYPE_CHECKING:
    from opaque.api.accounting.core._base import DpProcess

#: Mechanism types the Gaussian-family amplifiers accept.
GaussianFamily = Gaussian | AdaClip | MoeAux | NonPrivate
GAUSSIAN_FAMILY = (Gaussian, AdaClip, MoeAux, NonPrivate)


def effective_gaussian_noise_multiplier(process: DpProcess) -> float | None:
    """Multiplier of the Gaussian a family member reduces to, or ``None``.

    ``0.0`` means non-private (a :class:`NonPrivate` inner or a zero
    multiplier); ``None`` means ``process`` is not in the family.
    """
    match process:
        case NonPrivate():
            return 0.0
        case Gaussian(noise_multiplier=nm):
            return nm
        case AdaClip() | MoeAux():
            return process.effective_noise_multiplier
        case _:
            return None
