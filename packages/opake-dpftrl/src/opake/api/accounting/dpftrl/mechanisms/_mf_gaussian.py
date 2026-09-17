"""MF Gaussian mechanism — one mechanism class for every MF strategy.

The privacy of a matrix-factorization Gaussian release reduces to a
single Gaussian mechanism with effective noise multiplier
``noise_multiplier / strategy.sensitivity(n_steps, min_sep,
max_participations)``, regardless of which encoder ``C`` the training
side used.

The accounting amplifications (Poisson, BMinSep, BallsInBins) read
``inner.noise_multiplier`` and ``inner.strategy`` from the wrapped
:class:`MfGaussian` and supply their *own*
``(n_steps, min_sep, max_participations)`` at PLD time — they do not
read the fields stored on :class:`MfGaussian` itself.  Those fields are
only consulted when :meth:`MfGaussian.pld` is called bare
(unamplified), where an explicit horizon is required. With ``n_steps=None``
the object is an unresolved recipe for an amplifier, not a bare accountant.

Serialization: a custom serializer pair is registered here that emits
``{"type": "MfGaussian", "noise_multiplier": ..., "strategy":
{"type": "<StrategyName>", ...}, "n_steps": ..., "min_sep": ...,
"max_participations": ...}``.  The strategy sub-dict is produced and
consumed by :mod:`opake.api.dpftrl.noise._strategy_codec`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from opake.api.accounting.core import _native
from opake.api.accounting.core._base import DpProcess, Pld
from opake.api.accounting.core._pld_cache import pld_cache
from opake.api.accounting.core.discretization import get_discretization
from opake.api.dpftrl.noise._schedule_fingerprint import strategy_cache_key
from opake.exceptions import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from opake.api.dpftrl.noise.types import MfStrategy


@dataclass(frozen=True, slots=True)
class MfGaussian(DpProcess):
    """MF Gaussian mechanism — ``noise_multiplier`` + recipe ``strategy``.

    Bare-use (no amplification) requires ``n_steps`` (and optionally
    ``min_sep`` and ``max_participations``) so the strategy can resolve
    its sensitivity.  Wrapped in an amplifier these fields are ignored;
    the amplifier supplies its own participation context at PLD time.
    """

    noise_multiplier: float
    strategy: MfStrategy
    n_steps: int | None = None
    min_sep: int = 1
    max_participations: int | None = None

    def __post_init__(self) -> None:
        if self.noise_multiplier < 0:
            raise ConfigurationError(
                *(
                    f"noise_multiplier must be non-negative, got {self.noise_multiplier}",
                )
            )
        if self.n_steps is not None and self.n_steps < 1:
            raise ConfigurationError(*(f"n_steps must be >= 1, got {self.n_steps}",))
        if self.min_sep < 1:
            raise ConfigurationError(*(f"min_sep must be >= 1, got {self.min_sep}",))
        if self.max_participations is not None and self.max_participations < 1:
            raise ConfigurationError(
                *(
                    f"max_participations must be >= 1 or None, got {self.max_participations}",
                )
            )

    @property
    def _bare_n_steps(self) -> int:
        if self.n_steps is None:
            raise ConfigurationError(
                *(
                    "Bare MfGaussian accounting requires explicit n_steps. "
                    "Supply the release horizon, or wrap the recipe in a "
                    "DP-FTRL amplifier.",
                )
            )
        return self.n_steps

    @property
    def _effective_max_participations(self) -> int:
        return (
            self.max_participations
            if self.max_participations is not None
            else self._bare_n_steps
        )

    def _pld_cache_key(self) -> tuple[object, ...]:
        n_steps = self._bare_n_steps
        return (
            "MfGaussian",
            self.noise_multiplier,
            n_steps,
            self.min_sep,
            self.max_participations,
            strategy_cache_key(self.strategy, n_steps),
        )

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
        n_steps = self._bare_n_steps
        config = get_discretization(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
            max_conv_grid=max_conv_grid,
            seed=seed,
            mc_resolution=mc_resolution,
            mc_failure_probability=mc_failure_probability,
        )
        if self.noise_multiplier == 0:
            return _native.non_private_pld(config.to_native())
        sens = self.strategy.sensitivity(
            n_steps=n_steps,
            min_sep=self.min_sep,
            max_participations=self._effective_max_participations,
        )
        return _native.mf_gaussian_pld(
            self.noise_multiplier,
            sens,
            config.to_native(),
        )


def mf_gaussian(
    noise_multiplier: float,
    strategy: MfStrategy,
    *,
    n_steps: int | None = None,
    min_sep: int = 1,
    max_participations: int | None = None,
) -> MfGaussian:
    """MF Gaussian mechanism — noise multiplier + strategy recipe.

    Without ``n_steps``, constructs a recipe for an amplification factory;
    bare PLD or epsilon evaluation raises. With an explicit horizon,
    models a Gaussian release with effective noise
    multiplier ``noise_multiplier / strategy.sensitivity(n_steps, ...)``.
    Wrap in an amplification factory (``poisson``, ``b_min_sep``,
    ``balls_in_bins``) for the per-amplification PLD — those amplifiers
    supply their own ``n_steps``/``min_sep``/``max_participations`` and
    ignore the values passed here.

    Args:
        noise_multiplier: Raw noise standard deviation σ (>= 0).
        strategy: One of the strategy dataclasses from
            :mod:`opake.dpftrl.noise`.
        n_steps: Horizon for bare-use sensitivity evaluation. ``None`` (default)
            leaves the recipe unresolved for an amplifier to supply its horizon.
        min_sep: Bare-use min separation between participations (default 1).
        max_participations: Bare-use max participations per example
            (``None`` ⇒ ``n_steps``).

    Returns:
        An :class:`MfGaussian` process.
    """
    nm = float(noise_multiplier)
    # sigma >= 0 is validated in ``MfGaussian.__post_init__``.
    return MfGaussian(
        noise_multiplier=nm,
        strategy=strategy,
        n_steps=n_steps,
        min_sep=min_sep,
        max_participations=max_participations,
    )


# --- Custom serialization ---------------------------------------------------
#
# MfGaussian holds a ``strategy`` field whose value is one of the strategy
# dataclasses from opake.dpftrl.noise.  The generic DpProcess codec only
# knows how to emit primitives, containers, and nested DpProcess values;
# it would silently drop the strategy.  Register a custom serializer pair
# that delegates strategy (de)serialization to the strategy codec, which
# owns the strategy class name registry.


def _serialize_mf_gaussian(p: MfGaussian) -> dict[str, Any]:
    from opake.api.dpftrl.noise._strategy_codec import serialize_strategy

    return {
        "type": "MfGaussian",
        "noise_multiplier": p.noise_multiplier,
        "strategy": serialize_strategy(p.strategy),
        "n_steps": p.n_steps,
        "min_sep": p.min_sep,
        "max_participations": p.max_participations,
    }


def _load_mf_gaussian(_template: Any, sd: Mapping[str, Any]) -> MfGaussian:
    from opake.api.dpftrl.noise._strategy_codec import deserialize_strategy

    sd = dict(sd)
    sd.pop("type", None)
    return MfGaussian(
        noise_multiplier=sd["noise_multiplier"],
        strategy=deserialize_strategy(dict(sd["strategy"])),
        n_steps=sd.get("n_steps"),
        min_sep=sd.get("min_sep", 1),
        max_participations=sd.get("max_participations"),
    )


def _register_mf_gaussian_serializer() -> None:
    from opake.serialization import register_serializer

    register_serializer(MfGaussian, _serialize_mf_gaussian, _load_mf_gaussian)


_register_mf_gaussian_serializer()
