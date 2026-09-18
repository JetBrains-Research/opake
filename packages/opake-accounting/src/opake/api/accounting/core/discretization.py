"""Private module for PLD discretization configuration management."""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from . import _native

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True, slots=True)
class DiscretizationConfig:
    """Discretization configuration for PLD computation.

    Args:
        discretization: Grid spacing for the PLD PMF. Smaller = tighter, slower.
        log_x_mass_truncation_bound: Log tail mass cutoff in x-space. Tails below exp(bound) are truncated.
        max_grid_size: Maximum grid bins before automatic coarsening.
        seed: RNG seed for reproducibility across worker counts within a fixed
            Opake and dependency build. Exact streams may change across releases.
        max_conv_grid: Maximum convolution grid size for random-allocation PLD transform.
        mc_resolution: Maximum unresolved Monte Carlo mass, in delta units.
        mc_failure_probability: Failure probability of the simultaneous Monte
            Carlo confidence band.
    """

    discretization: float = 1e-4
    log_x_mass_truncation_bound: float = -50.0
    max_grid_size: int = 10_000_000
    tail_mass_truncation: float = 1e-15
    seed: int = 42
    max_conv_grid: int = 32_768
    mc_resolution: float = 1e-5
    mc_failure_probability: float = 1e-6

    def to_native(self) -> _native.DiscretizationConfig:
        """Convert to Rust DiscretizationConfig for FFI calls."""
        return _native.DiscretizationConfig(
            self.discretization,
            self.log_x_mass_truncation_bound,
            self.max_grid_size,
            self.tail_mass_truncation,
            self.seed,
            self.max_conv_grid,
            self.mc_resolution,
            self.mc_failure_probability,
        )

    @property
    def resolved_num_mc_samples(self) -> int:
        """Sample count required by the configured simultaneous confidence band."""
        return self.to_native().resolved_num_mc_samples

    def warn_if_large_mc(self) -> None:
        """Warn when the derived Monte Carlo work is unusually large."""
        required = self.resolved_num_mc_samples
        if required > _MC_SAMPLE_WARNING_THRESHOLD:
            warnings.warn(
                f"Monte Carlo accounting requires {required:,} samples per "
                f"adjacency direction for mc_resolution={self.mc_resolution} "
                f"and mc_failure_probability={self.mc_failure_probability}.",
                RuntimeWarning,
                stacklevel=3,
            )


__all__ = [
    "DiscretizationConfig",
    "get_discretization",
    "set_discretization",
]

# Module-level discretization default
_default_config: DiscretizationConfig | None = None
_active_config: ContextVar[DiscretizationConfig | None] = ContextVar(
    "opake_accounting_active_discretization",
    default=None,
)
_MC_SAMPLE_WARNING_THRESHOLD = 50_000_000


@contextmanager
def _use_discretization(config: DiscretizationConfig) -> Iterator[None]:
    """Make a resolved config available to nested PLD computations."""
    token = _active_config.set(config)
    try:
        yield
    finally:
        _active_config.reset(token)


def set_discretization(
    *,
    discretization: float | None = None,
    log_x_mass_truncation_bound: float | None = None,
    max_grid_size: int | None = None,
    tail_mass_truncation: float | None = None,
    seed: int | None = None,
    max_conv_grid: int | None = None,
    mc_resolution: float | None = None,
    mc_failure_probability: float | None = None,
) -> None:
    """Update module-level default discretization parameters.

    This is a partial update: only the named parameters change, and every
    omitted parameter keeps its current default (mirroring how
    ``get_discretization`` resolves overrides). A bare call is a no-op.
    To return to the library defaults, read them with
    ``get_discretization()`` before changing them and pass them back
    explicitly.

    These defaults are used when query parameters are not provided.
    Existing process objects resolve these defaults at every PLD cache lookup:
    changing the defaults computes a distinct PLD, while restoring an equal
    configuration reuses its prior bounded cache entry.
    Until the first call, the defaults match Google's dp_accounting
    high-precision settings (see ``DiscretizationConfig``).

    Args:
        discretization: Grid spacing for PLD PMF. Smaller = more precise, larger = faster.
            Error scales as O(disc^2). Current default: 1e-4.
        log_x_mass_truncation_bound: Tails with log-probability below this bound in x-space
            are truncated. Current default: -50 (matching Google).
        max_grid_size: If grid exceeds this many bins, coarsen discretization
            automatically. Current default: 10,000,000.
        tail_mass_truncation: Chernoff tail budget during composition (current default 1e-15).
        seed: RNG seed for Monte Carlo reproducibility. Current default: 42.
        max_conv_grid: Maximum convolution grid size before the native
            accountant uses its bounded-memory composition path.
        mc_resolution: Maximum unresolved Monte Carlo mass. Current default: 1e-5.
        mc_failure_probability: Failure probability of the simultaneous Monte
            Carlo confidence band. Current default: 1e-6.

    Example::

        # Use coarser discretization for faster computation
        acc.set_discretization(discretization=1e-3)

        # Tighten Monte Carlo bounds; discretization stays at 1e-3
        acc.set_discretization(mc_resolution=1e-8, mc_failure_probability=1e-10)
    """
    if all(
        v is None
        for v in (
            discretization,
            log_x_mass_truncation_bound,
            max_grid_size,
            tail_mass_truncation,
            seed,
            max_conv_grid,
            mc_resolution,
            mc_failure_probability,
        )
    ):
        return

    global _default_config
    base = _default_config if _default_config is not None else DiscretizationConfig()
    overrides = {
        name: value
        for name, value in {
            "discretization": discretization,
            "log_x_mass_truncation_bound": log_x_mass_truncation_bound,
            "max_grid_size": max_grid_size,
            "tail_mass_truncation": tail_mass_truncation,
            "seed": seed,
            "max_conv_grid": max_conv_grid,
            "mc_resolution": mc_resolution,
            "mc_failure_probability": mc_failure_probability,
        }.items()
        if value is not None
    }
    _default_config = replace(base, **overrides)


def get_discretization(
    *,
    discretization: float | None = None,
    log_x_mass_truncation_bound: float | None = None,
    max_grid_size: int | None = None,
    tail_mass_truncation: float | None = None,
    seed: int | None = None,
    max_conv_grid: int | None = None,
    mc_resolution: float | None = None,
    mc_failure_probability: float | None = None,
) -> DiscretizationConfig:
    """Get discretization config with hierarchical resolution.

    Resolution priority: query param > global default > library default.
    Always returns a concrete config (never None).

    When called with no arguments, returns the current global default
    (or library default if none set). When called with arguments,
    returns a config with those overrides applied.

    Args:
        discretization: Grid spacing (query-time override).
        log_x_mass_truncation_bound: Log tail mass cutoff in x-space (query-time override).
        max_grid_size: Maximum grid size before coarsening (query-time override).
        tail_mass_truncation: Composition tail budget (query-time override).
        seed: RNG seed for Monte Carlo (query-time override).
        max_conv_grid: Maximum convolution grid size (query-time override).
        mc_resolution: Maximum unresolved Monte Carlo mass (query-time override).
        mc_failure_probability: Confidence-band failure probability
            (query-time override).

    Returns:
        Resolved DiscretizationConfig (always concrete, never None).

    Example::

        # Get current global default (or library default)
        cfg = acc.get_discretization()

        # Get config with query-time overrides
        cfg = acc.get_discretization(discretization=1e-3, log_x_mass_truncation_bound=-40.0)
    """
    # Start with global default or library default
    base = _active_config.get()
    if base is None:
        base = (
            _default_config if _default_config is not None else DiscretizationConfig()
        )

    # If no overrides, return base as-is
    if (
        discretization is None
        and log_x_mass_truncation_bound is None
        and max_grid_size is None
        and tail_mass_truncation is None
        and seed is None
        and max_conv_grid is None
        and mc_resolution is None
        and mc_failure_probability is None
    ):
        return base

    # Apply query-time overrides
    return DiscretizationConfig(
        discretization=discretization
        if discretization is not None
        else base.discretization,
        log_x_mass_truncation_bound=(
            log_x_mass_truncation_bound
            if log_x_mass_truncation_bound is not None
            else base.log_x_mass_truncation_bound
        ),
        max_grid_size=max_grid_size
        if max_grid_size is not None
        else base.max_grid_size,
        tail_mass_truncation=tail_mass_truncation
        if tail_mass_truncation is not None
        else base.tail_mass_truncation,
        seed=seed if seed is not None else base.seed,
        max_conv_grid=max_conv_grid
        if max_conv_grid is not None
        else base.max_conv_grid,
        mc_resolution=mc_resolution
        if mc_resolution is not None
        else base.mc_resolution,
        mc_failure_probability=(
            mc_failure_probability
            if mc_failure_probability is not None
            else base.mc_failure_probability
        ),
    )
