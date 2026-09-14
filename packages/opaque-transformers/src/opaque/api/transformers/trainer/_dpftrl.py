"""DP-FTRL integration helpers for :class:`Trainer`.

Pure factories: MF strategy construction, accountant amplifier
construction, sampler dispatch.  No state lives here — the trainer
owns the lifecycle.  The trainer calls these in :meth:`_setup_training`
(strategy + amplifier) and :meth:`get_train_dataloader` (sampler) when
:attr:`TrainingArguments.privacy_noise_mechanism` starts with ``"mf_"``.

The MF amplifier returned by :func:`build_amplifier_factory` is the
whole-process accountant (a :class:`DpHorizonProcess`); the trainer queries
``(n_steps, min_sep, max_participations)`` off it to build the matching
:func:`opaque.dpftrl.noise.mf_gaussian_noise` and accounts that process once.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from opaque.dpftrl import (
    BallsInBinsSampler,
    BMinSepSampler,
    CyclicPoissonSampler,
    SequentialBatchSampler,
    band_mf_strategy,
    bisr_strategy,
    blt_strategy,
    bsr_strategy,
    identity_strategy,
    lambda_cgd_strategy,
)
from opaque.dpftrl.accounting import (
    b_min_sep as _ftrl_b_min_sep,
)
from opaque.dpftrl.accounting import (
    balls_in_bins as _ftrl_balls_in_bins,
)
from opaque.dpftrl.accounting import mf_gaussian
from opaque.dpftrl.accounting import (
    poisson as _ftrl_poisson,
)
from opaque.dpftrl.noise.types import BandMfStrategy
from opaque.dpsgd.sampling import (
    KOutOfTSampler,
    PoissonSampler,
)
from opaque.exceptions import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Callable

    from opaque.random.types import RngKey


@dataclasses.dataclass(frozen=True)
class MFContext:
    """DP-FTRL provenance carried through the training loop.

    ``strategy`` is the built BandMF / BLT recipe; ``amplifier_factory``
    produces the raw DpHorizonProcess for a calibrated multiplier so callers
    (noise construction, sampler construction, checkpoint save) can read
    ``(n_steps, min_sep, max_participations, sampling_prob)`` off it on
    demand — the recipe + the amplifier are the single source of truth
    for everything privacy-derived; nothing downstream should re-parse
    the user's mechanism kwargs.
    """

    strategy: Any
    amplifier_factory: Callable[[float], Any]


# Strategy factory dispatch — keyed by ``privacy_noise_mechanism``.
_STRATEGY_FACTORIES: dict[str, Callable[..., Any]] = {
    "mf_band": band_mf_strategy,
    "mf_blt": blt_strategy,
    "mf_bisr": bisr_strategy,
    "mf_bsr": bsr_strategy,
    "mf_lambda_cgd": lambda_cgd_strategy,
    "mf_identity": identity_strategy,
}

# Strategies that consume the optimizer LR schedule for workload-aware
# Toeplitz coefficient tuning.  Other strategies ignore the LR schedule.
_LR_SCHEDULED_STRATEGIES: frozenset[str] = frozenset({"mf_band", "mf_blt"})


def resume_process_state(state: dict[str, Any]) -> dict[str, Any]:
    """Compare unspecified inner MF horizons with legacy one-step defaults."""
    inner = state.get("inner")
    if (
        state.get("type") in {"CyclicPoisson", "BMinSep", "BallsInBins"}
        and isinstance(inner, dict)
        and inner.get("type") == "MfGaussian"
        and inner.get("n_steps", 1) is None
    ):
        return {**state, "inner": {**inner, "n_steps": 1}}
    return state


def build_strategy(
    mechanism: str,
    kwargs: dict[str, Any] | None,
    lr_schedule: Any = None,
) -> Any:
    """Build the MF strategy recipe for ``mechanism``.

    ``kwargs`` are forwarded verbatim to the strategy factory; the
    trainer pre-merges its per-mechanism defaults in
    ``TrainingArguments.__post_init__``.

    For BandMF / BLT, the optimizer ``lr_schedule`` is auto-injected
    when the user did not already supply one in ``kwargs`` — those
    strategies tune their Toeplitz coefficients against the schedule
    for tighter privacy at the realised workload.  Schedule recipes
    from :mod:`opaque.scheduling` round-trip cleanly through the
    accountant; raw lambdas do not (a clear error from the strategy
    codec surfaces at serialization time).
    """
    factory = _STRATEGY_FACTORIES[mechanism]
    extra = dict(kwargs) if kwargs else {}
    if (
        mechanism in _LR_SCHEDULED_STRATEGIES
        and lr_schedule is not None
        and "lr_schedule" not in extra
    ):
        extra["lr_schedule"] = lr_schedule
    return factory(**extra)


def _cyclic_poisson_group_rate(
    sample_rate: float, bands: int, dataset_size: int, world_size: int = 1
) -> float:
    """Per-active-group conditional rate (Choquette-Choo et al. 2023,
    Algorithm 2 / Theorem 4).

    Under DDP each rank ``EQUAL_SPLIT``-partitions its own local shard
    (``dataset_size // world_size`` examples) into ``bands`` groups,
    truncating to ``floor(local_size / bands)``; the rate must target that
    *local* group size, not the global ``dataset_size // bands``, or the
    realised batch drifts from ``expected_batch_size``. ``world_size=1``
    is the single-process case.
    """
    bands = int(bands)
    local_size = int(dataset_size) // int(world_size)
    group_size = local_size // bands
    if group_size < 1:
        raise ConfigurationError(
            *(
                "sampling_mode='cyclic_poisson' requires (dataset_size // "
                f"world_size) // bands >= 1; got dataset_size={dataset_size}, "
                f"world_size={world_size}, bands={bands}.",
            )
        )
    group_rate = float(sample_rate) * local_size / group_size
    if group_rate > 1.0:
        raise ConfigurationError(
            *(
                "sampling_mode='cyclic_poisson' requires the per-band "
                f"conditional rate ({group_rate!r}) to be <= 1. Reduce "
                "expected_batch_size or decrease bands.",
            )
        )
    return group_rate


def build_amplifier_factory(
    *,
    sampling_mode: str,
    strategy: Any,
    sample_rate: float,
    n_steps: int,
    num_bins: int,
    dataset_size: int,
    truncated_batch_size: int | None,
    world_size: int = 1,
) -> Callable[[float], Any]:
    """Return ``nm → DpHorizonProcess`` — the *raw* amplifier instance.

    The trainer holds the result on its training context and queries it
    with the calibrated noise multiplier at noise-function construction
    time to read off ``(n_steps, min_sep, max_participations)``.
    """
    if sampling_mode == "poisson":
        if isinstance(strategy, BandMfStrategy):
            raise ConfigurationError(
                *(
                    "sampling_mode='poisson' does not realise the grouped, "
                    "rotating-active-group participation pattern BandMF's "
                    "cyclic-Poisson accounting assumes; use "
                    "sampling_mode='cyclic_poisson' or 'b_min_sep' for "
                    "privacy_noise_mechanism='mf_band' instead.",
                )
            )

        def amp(
            nm: float,
            _s: Any = strategy,
            _sr: float = sample_rate,
            _ns: int = n_steps,
            _tb: int | None = truncated_batch_size,
            _ds: int = dataset_size,
        ) -> Any:
            return _ftrl_poisson(
                mf_gaussian(nm, _s),
                sample_rate=_sr,
                n_steps=_ns,
                truncated_batch_size=_tb,
                dataset_size=_ds if _tb is not None else None,
            )

    elif sampling_mode == "b_min_sep":

        def amp(
            nm: float,
            _s: Any = strategy,
            _ns: int = n_steps,
            _p0: float = sample_rate,
        ) -> Any:
            return _ftrl_b_min_sep(mf_gaussian(nm, _s), n_steps=_ns, p0=_p0)

    elif sampling_mode == "cyclic_poisson":
        if not isinstance(strategy, BandMfStrategy):
            raise ConfigurationError(
                *(
                    "sampling_mode='cyclic_poisson' requires a BandMF "
                    f"strategy; got {type(strategy).__name__}.",
                )
            )
        if truncated_batch_size is not None:
            raise ConfigurationError(
                *(
                    "sampling_mode='cyclic_poisson' does not support "
                    "sampling_kwargs['truncated_batch_size'] — the BandMF "
                    "cyclic-Poisson accountant only supports truncation for "
                    "IdentityStrategy (mf_identity).",
                )
            )
        group_rate = _cyclic_poisson_group_rate(
            sample_rate, strategy.bands, dataset_size, world_size
        )

        def amp(
            nm: float,
            _s: Any = strategy,
            _gr: float = group_rate,
            _ns: int = n_steps,
        ) -> Any:
            return _ftrl_poisson(mf_gaussian(nm, _s), sample_rate=_gr, n_steps=_ns)

    elif sampling_mode == "balls_in_bins":

        def amp(
            nm: float,
            _s: Any = strategy,
            _nb: int = num_bins,
            _ns: int = n_steps,
        ) -> Any:
            return _ftrl_balls_in_bins(mf_gaussian(nm, _s), num_bins=_nb, n_steps=_ns)

    else:
        raise ConfigurationError(
            *(
                f"sampling_mode={sampling_mode!r} has no DP-FTRL amplifier "
                "configured.  Valid: 'poisson', 'b_min_sep', "
                "'cyclic_poisson', 'balls_in_bins'.",
            )
        )
    return amp


def build_sampler(
    *,
    sampling_mode: str,
    dataset: Any,
    sample_rate: float,
    n_steps: int,
    key: RngKey,
    sampling_kwargs: dict[str, Any] | None,
    mf: MFContext | None,
    noise_multiplier: float | None,
    num_bins: int,
    expected_batch_size: int,
) -> Any:
    """Construct the Opaque sampler matching ``sampling_mode``.

    Privacy-derived sampler parameters (``bands``, paper-``p``) are read
    off the built ``mf`` recipe / amplifier — never off ``sampling_kwargs``
    or ``mechanism_kwargs`` — so the runtime sampler cannot desync from
    the accountant. ``cyclic_poisson`` reads its per-band rate straight
    off ``mf.amplifier_factory(noise_multiplier).sample_rate``, the same
    value :func:`build_amplifier_factory` calibrated the accountant with,
    so there is a single source of truth under DDP.
    """
    sk = dict(sampling_kwargs) if sampling_kwargs else {}
    if sampling_mode == "poisson":
        if mf is not None and isinstance(mf.strategy, BandMfStrategy):
            raise ConfigurationError(
                *(
                    "sampling_mode='poisson' does not realise the grouped, "
                    "rotating-active-group participation pattern BandMF's "
                    "cyclic-Poisson accounting assumes; use "
                    "sampling_mode='cyclic_poisson' or 'b_min_sep' for "
                    "privacy_noise_mechanism='mf_band' instead.",
                )
            )
        tb_raw = sk.get("truncated_batch_size")
        truncated_batch_size = int(tb_raw) if tb_raw is not None else None
        return PoissonSampler(
            dataset,
            sample_rate=sample_rate,
            n_steps=n_steps,
            truncated_batch_size=truncated_batch_size,
            key=key,
        )
    if sampling_mode == "k_out_of_t":
        k_raw = sk.get("k")
        allocation = sk.get("allocation")
        if k_raw is None or allocation is None:
            raise ConfigurationError(
                *(
                    "sampling_mode='k_out_of_t' requires sampling_kwargs with "
                    "'k' and 'allocation'.",
                )
            )
        if allocation not in ("block", "total"):
            raise ConfigurationError(
                *(
                    "sampling_kwargs['allocation'] must be 'block' or 'total', got "
                    f"{allocation!r}.",
                )
            )
        return KOutOfTSampler(
            dataset,
            k=int(k_raw),
            t=n_steps,
            allocation=allocation,
            key=key,
        )
    if sampling_mode == "b_min_sep":
        if mf is None or noise_multiplier is None:
            raise ConfigurationError(
                *(
                    "sampling_mode='b_min_sep' requires a built MFContext and a "
                    "calibrated noise_multiplier; got mf=None or "
                    "noise_multiplier=None.",
                )
            )
        amp = mf.amplifier_factory(noise_multiplier)
        return BMinSepSampler(
            dataset,
            bands=int(mf.strategy.bands),
            sampling_prob=float(amp.sampling_prob),
            n_steps=n_steps,
            key=key,
        )
    if sampling_mode == "balls_in_bins":
        return BallsInBinsSampler(
            dataset,
            num_bins=num_bins,
            n_steps=n_steps,
            key=key,
        )
    if sampling_mode == "cyclic_poisson":
        if mf is None or noise_multiplier is None:
            raise ConfigurationError(
                *(
                    "sampling_mode='cyclic_poisson' requires a built MFContext "
                    "and a calibrated noise_multiplier; got mf=None or "
                    "noise_multiplier=None.",
                )
            )
        if not isinstance(mf.strategy, BandMfStrategy):
            raise ConfigurationError(
                *(
                    "sampling_mode='cyclic_poisson' requires a BandMF "
                    f"strategy; got {type(mf.strategy).__name__}.",
                )
            )
        if sk:
            raise ConfigurationError(
                *(
                    "sampling_mode='cyclic_poisson' does not accept "
                    f"sampling_kwargs; got {sorted(sk)!r}. bands and the "
                    "per-band rate are derived from the MFContext, not from "
                    "sampling_kwargs.",
                )
            )
        amp = mf.amplifier_factory(noise_multiplier)
        return CyclicPoissonSampler(
            dataset,
            sample_rate=float(amp.sample_rate),
            bands=int(mf.strategy.bands),
            n_steps=n_steps,
            key=key,
        )
    if sampling_mode == "sequential":
        return SequentialBatchSampler(dataset, batch_size=expected_batch_size)
    raise ConfigurationError(*(f"Unknown sampling_mode {sampling_mode!r}",))


__all__ = [
    "MFContext",
    "build_amplifier_factory",
    "build_sampler",
    "build_strategy",
]
