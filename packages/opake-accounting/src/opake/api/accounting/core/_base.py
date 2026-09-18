# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
#
# DP process algebra and privacy-metric surface adapted in part from
# Google DP Accounting (Apache-2.0;
# https://github.com/google/differential-privacy/tree/main/python/dp_accounting),
# then reworked for Opake's PLD-native process composition model.
# See ../../../../../NOTICE in this package for the full attribution.
"""DpProcess: stateless composable DP process algebra.

Mechanism constructors store parameters as frozen-dataclass fields.
The PLD is computed on demand via the ``pld()`` method, which is automatically
cached by its resolved discretization configuration and process-free mechanism
inputs. Use
:func:`~opake.accounting.composition._cached` to increase cache size or as
a merge barrier.

Composition Optimizations
~~~~~~~~~~~~~~~~~~~~~~~~~

At construction time, the composition operators detect patterns that can
be reduced to cheaper operations using structural equality (``==``):

**Identity elision** (in ``__or__``):

- ``Identity | a``  →  ``a``
- ``a | Identity``  →  ``a``

**Flatten nested repeats** (in ``__mul__``):

- ``(step * n) * m``  →  ``Repeated(step, n * m)``
  Avoids nested ``self_compose`` calls (2 FFTs instead of 4).

**Merge same-leaf compose** (in ``__or__``):

- ``a | a``          →  ``a * 2``       (self_compose, 2 FFTs not 4)
- ``a * n | a``      →  ``a * (n + 1)``
- ``a | a * n``      →  ``a * (n + 1)``
- ``a * n | a * m``  →  ``a * (n + m)`` (2 FFTs not 6)

**Right-spine merge** (in ``__or__``):

- ``(X | a * n) | a * m``  →  ``X | a * (n + m)``
"""

from __future__ import annotations

import dataclasses
import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeAlias

from opake.exceptions import ConfigurationError

from . import _native

if TYPE_CHECKING:
    from collections.abc import Hashable

Pld: TypeAlias = _native.Pld

__all__ = ["DpProcess", "Pld"]

# Global registry of DpProcess subclasses for polymorphic deserialization.
# Automatically populated when each subclass is defined via __init_subclass__.
_PROCESS_REGISTRY: dict[str, type[DpProcess]] = {}


def _freeze_cache_key(value: object) -> Hashable:
    """Return a process-free, hashable representation of a dataclass value."""
    if isinstance(value, DpProcess):
        return value._pld_cache_key()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (
            type(value),
            tuple(
                _freeze_cache_key(getattr(value, field.name))
                for field in dataclasses.fields(value)
            ),
        )
    if isinstance(value, tuple):
        return tuple(_freeze_cache_key(item) for item in value)
    if isinstance(value, list):
        return tuple(_freeze_cache_key(item) for item in value)
    if isinstance(value, dict):
        return frozenset(
            (
                _freeze_cache_key(key),
                _freeze_cache_key(item),
            )
            for key, item in value.items()
        )
    if isinstance(value, set):
        return frozenset(_freeze_cache_key(item) for item in value)
    return value


def _register_dp_process_with_serialization(cls) -> None:
    """Hook each concrete process into :mod:`opake.serialization`."""
    from opake.api.accounting.core._process_codec import (
        _generic_from_state_dict,
        _generic_state_dict,
    )
    from opake.serialization import register_serializer

    # Named (not lambda) pair: the codec's iterative loader identity-checks
    # the registered load fn against ``_generic_from_state_dict`` to decide
    # whether a class has a CUSTOM serializer that must fire on load.
    register_serializer(cls, _generic_state_dict, _generic_from_state_dict)


class DpProcess(ABC):
    """A differential privacy process backed by a PLD.

    Abstract base class for all mechanisms.  Subclasses implement
    :meth:`pld` to compute the Privacy Loss Distribution on demand.
    Results are automatically cached by the resolved discretization
    configuration and process-free mechanism inputs.

    Supports:

    - **Privacy metrics**: epsilon_at(), delta_at(), advantage(), beta_at(), risk_at()
    - **Composition**: ``a | b`` (heterogeneous), ``a * k`` (homogeneous)

    Example::

        import opake.accounting as acc

        step = acc.poisson(acc.gaussian(1.1), 0.01)
        training = step * 1000
        eps = training.epsilon_at(1e-5)
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Auto-register concrete dataclass subclasses in the global registry.

        Skips abstract intermediates (``DpHorizonProcess`` etc.) that aren't
        dataclasses — they have no fields to serialize and can't be
        instantiated anyway.
        """
        super().__init_subclass__(**kwargs)
        if not dataclasses.is_dataclass(cls):
            return
        _PROCESS_REGISTRY[cls.__name__] = cls
        _register_dp_process_with_serialization(cls)

    @abstractmethod
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
        """Compute the Privacy Loss Distribution.

        Results are automatically cached by the resolved discretization
        configuration and process-free mechanism inputs. Use
        :func:`~opake.accounting.composition._cached` to increase cache size
        or as a merge barrier.

        Query-time overrides are broadcast to every node of a composed
        process, so a single ``seed`` is shared by all Monte-Carlo nodes —
        the same semantics as the global ``set_discretization`` config.

        Args:
            discretization: Grid spacing for PLD (query-time override).
            log_x_mass_truncation_bound: Log tail mass cutoff in x-space (query-time override).
            max_grid_size: Maximum grid size before coarsening (query-time override).
            max_conv_grid: Maximum convolution grid size for random-allocation PLD (query-time override).
            seed: RNG seed for Monte Carlo reproducibility
                (query-time override; ignored by analytic PLDs).
            mc_resolution: Maximum unresolved Monte Carlo mass.
            mc_failure_probability: Failure probability of the simultaneous
                Monte Carlo confidence band.
        """
        ...

    def epsilon_at(
        self,
        delta: float,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
        max_conv_grid: int | None = None,
        seed: int | None = None,
        mc_resolution: float | None = None,
        mc_failure_probability: float | None = None,
    ) -> float:
        """Smallest ε achieving (ε, δ)-DP.

        Args:
            delta: Failure probability.
            discretization: Grid spacing (query-time override).
            log_x_mass_truncation_bound: Log tail mass cutoff in x-space (query-time override).
            max_grid_size: Maximum grid size before coarsening (query-time override).
            max_conv_grid: Maximum convolution grid size for random-allocation PLD (query-time override).
            seed: RNG seed for Monte Carlo reproducibility
                (query-time override; ignored by analytic PLDs).
            mc_resolution: Maximum unresolved Monte Carlo mass.
            mc_failure_probability: Failure probability of the simultaneous
                Monte Carlo confidence band.
        """
        from .discretization import get_discretization

        configured_resolution = get_discretization(
            mc_resolution=mc_resolution
        ).mc_resolution
        effective_resolution = (
            min(configured_resolution, delta / 2.0)
            if delta > 0.0
            else configured_resolution
        )
        return self.pld(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
            max_conv_grid=max_conv_grid,
            seed=seed,
            mc_resolution=effective_resolution,
            mc_failure_probability=mc_failure_probability,
        ).epsilon_at(delta)

    def delta_at(
        self,
        epsilon: float,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
        max_conv_grid: int | None = None,
        seed: int | None = None,
        mc_resolution: float | None = None,
        mc_failure_probability: float | None = None,
    ) -> float:
        """Smallest δ achieving (ε, δ)-DP.

        Args:
            epsilon: Privacy budget.
            discretization: Grid spacing (query-time override).
            log_x_mass_truncation_bound: Log tail mass cutoff in x-space (query-time override).
            max_grid_size: Maximum grid size before coarsening (query-time override).
            max_conv_grid: Maximum convolution grid size for random-allocation PLD (query-time override).
            seed: RNG seed for Monte Carlo reproducibility
                (query-time override; ignored by analytic PLDs).
            mc_resolution: Maximum unresolved Monte Carlo mass.
            mc_failure_probability: Failure probability of the simultaneous
                Monte Carlo confidence band.
        """
        return self.pld(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
            max_conv_grid=max_conv_grid,
            seed=seed,
            mc_resolution=mc_resolution,
            mc_failure_probability=mc_failure_probability,
        ).delta_at(epsilon)

    def advantage(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
        max_conv_grid: int | None = None,
        seed: int | None = None,
        mc_resolution: float | None = None,
        mc_failure_probability: float | None = None,
    ) -> float:
        """Total-variation advantage (f-DP).

        Args:
            discretization: Grid spacing (query-time override).
            log_x_mass_truncation_bound: Log tail mass cutoff in x-space (query-time override).
            max_grid_size: Maximum grid size before coarsening (query-time override).
            max_conv_grid: Maximum convolution grid size for random-allocation PLD (query-time override).
            seed: RNG seed for Monte Carlo reproducibility
                (query-time override; ignored by analytic PLDs).
            mc_resolution: Maximum unresolved Monte Carlo mass.
            mc_failure_probability: Failure probability of the simultaneous
                Monte Carlo confidence band.
        """
        return self.pld(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
            max_conv_grid=max_conv_grid,
            seed=seed,
            mc_resolution=mc_resolution,
            mc_failure_probability=mc_failure_probability,
        ).advantage()

    def beta_at(
        self,
        alpha: float,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
        max_conv_grid: int | None = None,
        seed: int | None = None,
        mc_resolution: float | None = None,
        mc_failure_probability: float | None = None,
    ) -> float:
        """Type-II error at given Type-I error α.

        Args:
            alpha: Type-I error rate.
            discretization: Grid spacing (query-time override).
            log_x_mass_truncation_bound: Log tail mass cutoff in x-space (query-time override).
            max_grid_size: Maximum grid size before coarsening (query-time override).
            max_conv_grid: Maximum convolution grid size for random-allocation PLD (query-time override).
            seed: RNG seed for Monte Carlo reproducibility
                (query-time override; ignored by analytic PLDs).
            mc_resolution: Maximum unresolved Monte Carlo mass.
            mc_failure_probability: Failure probability of the simultaneous
                Monte Carlo confidence band.
        """
        return self.pld(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
            max_conv_grid=max_conv_grid,
            seed=seed,
            mc_resolution=mc_resolution,
            mc_failure_probability=mc_failure_probability,
        ).beta_at(alpha)

    def risk_at(
        self,
        prior: float,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
        max_conv_grid: int | None = None,
        seed: int | None = None,
        mc_resolution: float | None = None,
        mc_failure_probability: float | None = None,
    ) -> float:
        """Bayes risk under optimal adversary.

        Args:
            prior: Prior probability.
            discretization: Grid spacing (query-time override).
            log_x_mass_truncation_bound: Log tail mass cutoff in x-space (query-time override).
            max_grid_size: Maximum grid size before coarsening (query-time override).
            max_conv_grid: Maximum convolution grid size for random-allocation PLD (query-time override).
            seed: RNG seed for Monte Carlo reproducibility
                (query-time override; ignored by analytic PLDs).
            mc_resolution: Maximum unresolved Monte Carlo mass.
            mc_failure_probability: Failure probability of the simultaneous
                Monte Carlo confidence band.
        """
        return self.pld(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
            max_conv_grid=max_conv_grid,
            seed=seed,
            mc_resolution=mc_resolution,
            mc_failure_probability=mc_failure_probability,
        ).risk_at(prior)

    # -- Composition operators -----------------------------------------------

    def _leaf_and_count(self) -> tuple[DpProcess, int]:
        """Return ``(leaf, count)`` for merge optimization.

        Plain processes return ``(self, 1)``.
        Overridden by :class:`Repeated` to return ``(inner, count)``.
        """
        return (self, 1)

    def _pld_cache_key(self) -> Hashable:
        """Return every process-free input that can affect the requested PLD.

        Ordinary frozen process dataclasses use a process-free structural
        key. Processes that contain callables or defer mechanism resolution
        override this with an equally complete key of their resolved inputs.
        """
        return (
            type(self),
            tuple(
                _freeze_cache_key(getattr(self, field.name))
                for field in dataclasses.fields(self)
            ),
        )

    def _repeated_pld_cache_key(self, count: int) -> Hashable:
        """Return the inner cache key for ``repeated_pld(count)``.

        The enclosing :class:`Repeated` key records ``count``. The default
        implementation composes the ordinary PLD; count-sensitive overrides
        return the identity of their repeated computation.
        """
        del count
        return self._pld_cache_key()

    def _has_same_repeated_pld(self, other: DpProcess, count: int) -> bool:
        """Whether two leaves have the same ``count``-fold PLD computation."""
        if self is other:
            return True

        from opake.api.accounting.core.composition.types import Repeated

        return (
            Repeated(self, count)._pld_cache_key()
            == Repeated(other, count)._pld_cache_key()
        )

    def repeated_pld(
        self,
        count: int,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
        max_conv_grid: int | None = None,
        seed: int | None = None,
        mc_resolution: float | None = None,
        mc_failure_probability: float | None = None,
    ) -> Pld:
        """PLD for ``count`` independent applications of this process.

        Default: ``self.pld(...).self_compose(count)``.  Subclasses whose
        K-fold behaviour is *not* the K-fold composition of a single-shot
        PLD (e.g. DP-FTRL per-step adapters, where K steps' joint PLD
        depends on the deployed N-step strategy's correlation matrix)
        override this to compute the true K-step PLD directly.
        """
        from .discretization import get_discretization

        configured_resolution = get_discretization(
            mc_resolution=mc_resolution
        ).mc_resolution
        per_release_resolution = -math.expm1(math.log1p(-configured_resolution) / count)
        return self.pld(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
            max_conv_grid=max_conv_grid,
            seed=seed,
            mc_resolution=per_release_resolution,
            mc_failure_probability=mc_failure_probability,
        ).self_compose(count)

    def __mul__(self, count: int) -> DpProcess:
        """Repeat this process *count* times: ``step * 1000``.

        Flattens nested repeats: ``(step * n) * m`` → ``step * (n * m)``.

        Args:
            count: Number of repetitions. Must be positive.
                Use ``count=0`` is not allowed (use :func:`identity` instead).

        Raises:
            ValueError: If count < 1.
        """
        if count < 1:
            raise ConfigurationError(
                *(
                    f"Repeat count must be >= 1, got {count}. "
                    "Use identity() for zero privacy loss.",
                )
            )

        from opake.api.accounting.core.composition.types import Repeated

        leaf, existing = self._leaf_and_count()
        return Repeated(leaf, existing * count)

    def __rmul__(self, count: int) -> DpProcess:
        """Support ``1000 * step`` syntax."""
        return self.__mul__(count)

    def __or__(self, other: DpProcess) -> DpProcess:
        """Compose with another process: ``a | b``.

        Applies identity elision, direct merge, and right-spine merge
        when both leaves resolve to the same repeated PLD computation.
        """
        from opake.api.accounting.core.composition.types import Composed, Repeated
        from opake.api.accounting.core.mechanisms.types import Identity

        # Identity elision
        match (self, other):
            case (Identity(), _):
                return other
            case (_, Identity()):
                return self

        # Direct merge: both sides share the same leaf
        left_leaf, left_count = self._leaf_and_count()
        right_leaf, right_count = other._leaf_and_count()
        merged_count = left_count + right_count
        if left_leaf._has_same_repeated_pld(right_leaf, merged_count):
            return Repeated(left_leaf, merged_count)

        # Right-spine merge: (X | a*n) | a*m  →  X | a*(n+m)
        match self:
            case Composed():
                r_leaf, r_count = self.right._leaf_and_count()
                merged_count = r_count + right_count
                if r_leaf._has_same_repeated_pld(right_leaf, merged_count):
                    merged = Repeated(r_leaf, merged_count)
                    return Composed(self.left, merged)

        return Composed(self, other)
