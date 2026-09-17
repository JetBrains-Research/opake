"""DP-FTRL-specific accounting factories façade.

**DP-FTRL amplifiers describe whole training processes.** Supply the full
``n_steps`` to the amplifier, leaving its inner ``mf_gaussian`` recipe's
horizon unspecified. For bare accounting, supply the horizon and runtime
participation bounds directly to ``mf_gaussian``; a horizonless recipe
cannot compute a PLD or epsilon on its own.

The accounting mechanism is :class:`MfGaussian` — a thin wrapper over
``(noise_multiplier, strategy)``.  The strategy (from
:mod:`opake.dpftrl.noise`) carries the structural decomposition once;
amplifications dispatch on its type at PLD time.

Mechanism factory (in :mod:`opake.dpftrl.accounting.mechanisms`):

- :func:`mf_gaussian` — single-argument-pair MF Gaussian factory:
  ``mf_gaussian(nm, strategy)``.

Amplification (in :mod:`opake.dpftrl.accounting.amplification`):

- :func:`poisson` — Poisson subsampling.  Accepts MfGaussian wrapping a
  ``BandMfStrategy`` (bands read from ``len(coefficients)``) or
  ``IdentityStrategy`` (bands ≡ 1).  Required keyword: ``n_steps``.
- :func:`b_min_sep` — warm-start b-min-sep Monte Carlo PLD for
  ``BandMfStrategy``.  Required keywords: ``n_steps``, ``p0``.
- :func:`balls_in_bins` — total privacy cost under fixed-partition
  Balls-in-Bins sampling.  Required keywords: ``num_bins``, ``n_steps``
  (must be a positive multiple of ``num_bins``).

Cross-cutting primitives (composition, calibration) live at
:mod:`opake.accounting`.

The amplification dataclass is named ``CyclicPoisson`` (rather than ``Poisson``)
to avoid a class-name collision with
:class:`opake.dpsgd.accounting.amplification.Poisson` in the serialization
registry.  The user-facing factory is still :func:`poisson`.

**Do not confuse** with :func:`~opake.accounting.identity` — that object
is the **composition algebra** identity (approximately ε=0), not MF
``identity_strategy``.

Example::

    import opake.accounting as acc
    import opake.dpftrl.accounting as ftrl_acc
    from opake.dpftrl.noise import band_mf_strategy, blt_strategy

    band_s = band_mf_strategy(bands=10)
    process = ftrl_acc.poisson(
        ftrl_acc.mf_gaussian(1.0, band_s), sample_rate=0.01, n_steps=1000,
    )
    eps = process.epsilon_at(1e-5)
"""

from opake.api.accounting.dpftrl import (
    b_min_sep,
    balls_in_bins,
    mf_gaussian,
    poisson,
)

__all__ = [
    "b_min_sep",
    "balls_in_bins",
    "mf_gaussian",
    "poisson",
]
