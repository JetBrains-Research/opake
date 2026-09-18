"""``calibrate`` clears registered :class:`NativeCache` instances in ``finally``."""

from __future__ import annotations

import pytest

from opake.api.accounting.core._native_cache import (
    NativeCache,
    _clear_all_native_caches,
)
from opake.api.accounting.core.calibration import calibrate, epsilon_budget
from opake.api.accounting.core.mechanisms import eps_delta
from opake.exceptions import CalibrationError


@pytest.fixture(autouse=True)
def _reset_native_caches_after() -> None:
    yield
    _clear_all_native_caches()


def test_calibrate_finally_clears_native_cache() -> None:
    destruct_calls = [0]

    def destruct(_h: int) -> None:
        destruct_calls[0] += 1

    cache = NativeCache(
        name="cal_probe",
        max_bytes_env=None,
        default_max_bytes=10**9,
        max_entries=10,
        nbytes_estimate=lambda _k: 1,
        destructor=destruct,
    )
    budget = epsilon_budget(3.0, delta=1e-5)

    def process(nm: float):
        cache.get_or_create(("probe", nm), lambda: 100 + int(nm * 1000) % 10000)
        return eps_delta(5.0 - float(nm), 1e-5)

    calibrate(budget, process, 0.5, 3.0, tolerance=0.05, max_iterations=30)
    assert len(cache) == 0
    assert destruct_calls[0] >= 1


def test_calibrate_finally_clears_on_process_exception() -> None:
    destruct_calls = [0]

    def destruct(_h: int) -> None:
        destruct_calls[0] += 1

    cache = NativeCache(
        name="cal_fail",
        max_bytes_env=None,
        default_max_bytes=10**9,
        max_entries=10,
        nbytes_estimate=lambda _k: 1,
        destructor=destruct,
    )
    budget = epsilon_budget(3.0, delta=1e-5)
    n_calls = [0]

    def process(nm: float):
        cache.get_or_create(("fixed",), lambda: 1)
        n_calls[0] += 1
        if n_calls[0] >= 3:
            raise RuntimeError("mid-search")
        return eps_delta(5.0 - float(nm), 1e-5)

    with pytest.raises(RuntimeError, match="mid-search"):
        calibrate(budget, process, 0.5, 3.0, tolerance=0.05, max_iterations=30)

    assert len(cache) == 0
    assert destruct_calls[0] >= 1


def test_calibrate_finally_clears_on_non_convergence() -> None:
    destruct_calls = [0]

    def destruct(_h: int) -> None:
        destruct_calls[0] += 1

    cache = NativeCache(
        name="cal_non_converging",
        max_bytes_env=None,
        default_max_bytes=10**9,
        max_entries=10,
        nbytes_estimate=lambda _k: 1,
        destructor=destruct,
    )
    budget = epsilon_budget(3.0, delta=1e-5)

    def process(nm: float):
        cache.get_or_create(("probe", nm), lambda: 100 + int(nm * 1000) % 10000)
        epsilon = 4.0 if nm < 1.0 else 2.0
        return eps_delta(epsilon, 1e-5)

    with pytest.raises(CalibrationError, match="did not converge"):
        calibrate(budget, process, 0.5, 3.0, max_iterations=4)

    assert len(cache) == 0
    assert destruct_calls[0] >= 1
