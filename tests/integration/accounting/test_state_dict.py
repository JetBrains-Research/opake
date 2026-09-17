"""Tests for DpProcess nested state_dict serialization (opake.serialization)."""

import math
from typing import cast

import opake.accounting as acc
import opake.dpftrl.accounting as ftrl_acc
import opake.dpsgd.accounting as dpsgd_acc
from opake.accounting import Accountant
from opake.api.accounting.dpftrl.amplification._b_min_sep import BMinSep
from opake.api.accounting.dpftrl.amplification._poisson import CyclicPoisson
from opake.dpftrl.accounting.types import MfGaussian
from opake.dpftrl.noise import band_mf_strategy
from opake.dpftrl.noise.types import BandMfStrategy
from opake.dpsgd.accounting.types import KOutOfT
from opake.serialization import from_state_dict, state_dict

# Template type selects the registered handler; PLD decode uses the root
# dict's ``type`` field, so any concrete :class:`~opake.accounting._base.DpProcess`
# instance is sufficient (``identity()`` is a stable choice).
_PROCESS_TEMPLATE = acc.identity()


def test_gaussian_state_dict_structure():
    proc = dpsgd_acc.gaussian(1.1)
    state = cast("dict[str, object]", state_dict(proc))
    assert state["type"] == "Gaussian"
    assert state["noise_multiplier"] == 1.1


def test_eps_delta_state_dict_structure():
    proc = acc.eps_delta(1.0, 1e-5)
    state = cast("dict[str, object]", state_dict(proc))
    assert state["type"] == "EpsDelta"
    assert state["epsilon"] == 1.0
    assert state["delta"] == 1e-5


def test_identity_state_dict_structure():
    proc = acc.identity()
    state = cast("dict[str, object]", state_dict(proc))
    assert state["type"] == "Identity"


def test_poisson_state_dict_structure():
    proc = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.8), 0.01)
    state = cast("dict", state_dict(proc))
    assert state["type"] == "Poisson"
    assert state["sample_rate"] == 0.01
    assert state["inner"]["type"] == "Gaussian"


def test_truncated_poisson_state_dict_structure():
    proc = dpsgd_acc.poisson(
        dpsgd_acc.gaussian(0.8),
        0.01,
        truncated_batch_size=128,
        dataset_size=10_000,
    )
    state = cast("dict", state_dict(proc))
    assert state["type"] == "Poisson"
    assert state["truncated_batch_size"] == 128
    assert state["dataset_size"] == 10_000
    assert state["inner"]["type"] == "Gaussian"


def test_k_out_of_t_state_dict_structure():
    proc = dpsgd_acc.k_out_of_t(dpsgd_acc.gaussian(0.8), k=4, t=64, allocation="block")
    state = cast("dict", state_dict(proc))
    assert state["type"] == "KOutOfT"
    assert state["k"] == 4
    assert state["n_steps"] == 64
    assert state["allocation"] == "block"
    assert state["inner"]["type"] == "Gaussian"


def test_k_out_of_t_round_trip():
    proc = dpsgd_acc.k_out_of_t(dpsgd_acc.gaussian(0.8), k=4, t=64, allocation="block")
    restored = from_state_dict(_PROCESS_TEMPLATE, state_dict(proc))
    assert isinstance(restored, KOutOfT)
    assert restored == proc


def test_total_k_out_of_t_round_trip_preserves_conservative_mode():
    proc = dpsgd_acc.k_out_of_t(dpsgd_acc.gaussian(0.8), k=4, t=64, allocation="total")
    restored = from_state_dict(_PROCESS_TEMPLATE, state_dict(proc))

    assert isinstance(restored, KOutOfT)
    assert restored == proc
    assert restored.allocation == "total"


def test_parallel_poisson_state_dict_structure():
    proc = dpsgd_acc.parallel_poisson(
        dpsgd_acc.gaussian(0.8), sample_rate=0.01, num_workers=4
    )
    state = cast("dict", state_dict(proc))
    assert state["type"] == "ParallelPoisson"
    assert state["num_workers"] == 4
    assert state["inner"]["type"] == "Poisson"
    assert state["inner"]["inner"]["type"] == "Gaussian"


def test_adaclip_state_dict_structure():
    proc = dpsgd_acc.adaclip(dpsgd_acc.gaussian(0.8), expected_batch_size=1000)
    state = cast("dict", state_dict(proc))
    assert state["type"] == "AdaClip"
    assert state["fraction_noise_std"] == 0.05
    assert state["expected_batch_size"] == 1000
    assert state["inner"]["type"] == "Gaussian"


def test_composed_state_dict_structure():
    left = dpsgd_acc.gaussian(0.8)
    right = acc.eps_delta(1.0, 1e-5)
    proc = left | right
    state = cast("dict", state_dict(proc))
    assert state["type"] == "Composed"
    assert state["left"]["type"] == "Gaussian"
    assert state["right"]["type"] == "EpsDelta"


def test_repeated_state_dict_structure():
    proc = dpsgd_acc.gaussian(0.8) * 3
    state = cast("dict", state_dict(proc))
    assert state["type"] == "Repeated"
    assert state["count"] == 3
    assert state["inner"]["type"] == "Gaussian"


def test_cached_state_dict_structure():
    proc = acc.cached(dpsgd_acc.gaussian(0.8))
    state = cast("dict", state_dict(proc))
    assert state["type"] == "CachedProcess"
    assert state["inner"]["type"] == "Gaussian"


def test_ftrl_poisson_state_dict_structure():
    strategy = band_mf_strategy(bands=2)
    proc = ftrl_acc.poisson(
        ftrl_acc.mf_gaussian(1.0, strategy),
        sample_rate=0.01,
        n_steps=200,
    )
    state = cast("dict", state_dict(proc))
    assert state["type"] == "CyclicPoisson"
    assert state["sample_rate"] == 0.01
    assert state["n_steps"] == 200
    assert state["inner"]["type"] == "MfGaussian"
    assert state["inner"]["noise_multiplier"] == 1.0
    assert state["inner"]["strategy"]["type"] == "BandMfStrategy"
    assert state["inner"]["strategy"]["bands"] == 2
    # Strategy is a pure recipe — derived data (coefficients, gram,
    # streaming) is regenerated on demand and not on the wire.
    assert "_coefficients" not in state["inner"]["strategy"]
    assert "sensitivity" not in state["inner"]["strategy"]


def test_b_min_sep_round_trip():
    proc = ftrl_acc.b_min_sep(
        ftrl_acc.mf_gaussian(1.0, band_mf_strategy(bands=2)),
        n_steps=100,
        p0=0.02,
    )
    state = state_dict(proc)
    restored = from_state_dict(_PROCESS_TEMPLATE, state)
    assert isinstance(restored, BMinSep)
    assert restored == proc


def test_ftrl_poisson_round_trip():
    proc = ftrl_acc.poisson(
        ftrl_acc.mf_gaussian(1.0, band_mf_strategy(bands=2)),
        sample_rate=0.01,
        n_steps=200,
    )
    state = state_dict(proc)
    restored = from_state_dict(_PROCESS_TEMPLATE, state)
    assert isinstance(restored, CyclicPoisson)
    assert isinstance(restored.inner, MfGaussian)
    assert isinstance(restored.inner.strategy, BandMfStrategy)
    assert restored == proc


def test_accountant_state_dict_roundtrip():
    acct = Accountant()
    step = dpsgd_acc.poisson(
        dpsgd_acc.adaclip(dpsgd_acc.gaussian(0.8), expected_batch_size=1000), 0.01
    )
    acct = acct | step
    state = state_dict(acct)
    restored = from_state_dict(Accountant(), state)
    eps = restored.epsilon_at(1e-5)
    assert math.isfinite(eps)
    assert eps > 0
