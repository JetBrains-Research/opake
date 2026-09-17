"""Tests for learning-rate schedules on matrix-factorization strategies."""

import pytest
import torch

from opake.api.dpftrl.noise._band_mf import _band_mf_coefficients_cached
from opake.dpftrl.noise import band_mf_strategy, blt_strategy
from opake.exceptions import CheckpointError
from opake.scheduling import linear_schedule
from opake.serialization import state_dict

_PART = {"n_steps": 64, "min_sep": 16, "max_participations": 4}


class TestBandMfLrScheduleCallable:
    def test_constant_callable_matches_no_schedule(self):
        s_none = band_mf_strategy(bands=4)
        s_const = band_mf_strategy(bands=4, lr_schedule=lambda _t: 1.0)
        torch.testing.assert_close(
            s_const.coefficients(n_steps=_PART["n_steps"]),
            s_none.coefficients(n_steps=_PART["n_steps"]),
        )

    def test_non_constant_schedule_changes_coefficients(self):
        s_const = band_mf_strategy(bands=4, lr_schedule=lambda _t: 1.0)
        s_ramp = band_mf_strategy(bands=4, lr_schedule=lambda t: 1.0 + 0.01 * t)
        coefs_const = s_const.coefficients(n_steps=_PART["n_steps"])
        coefs_ramp = s_ramp.coefficients(n_steps=_PART["n_steps"])
        assert not torch.allclose(coefs_const, coefs_ramp)

    def test_cache_hits_on_functionally_equal_callables(self):
        _band_mf_coefficients_cached.cache_clear()
        s1 = band_mf_strategy(bands=4, lr_schedule=lambda t: 1.0 + t)
        s2 = band_mf_strategy(bands=4, lr_schedule=lambda t: float(1.0 + t))
        s1.coefficients(n_steps=_PART["n_steps"])
        before_hits = _band_mf_coefficients_cached.cache_info().hits
        s2.coefficients(n_steps=_PART["n_steps"])
        after_hits = _band_mf_coefficients_cached.cache_info().hits
        assert after_hits == before_hits + 1

    def test_warmup_decay_schedule_runs(self):
        sched = linear_schedule(
            init_value=1.0, end_value=0.1, transition_steps=_PART["n_steps"]
        )
        s = band_mf_strategy(bands=4, lr_schedule=sched)
        coefs = s.coefficients(n_steps=_PART["n_steps"])
        assert coefs.shape[0] == 4  # ``bands``
        assert coefs[0] > 0

    def test_callable_schedule_not_serialisable(self):
        s = band_mf_strategy(bands=4, lr_schedule=lambda _t: 0.1)
        with pytest.raises(CheckpointError, match="callable strategy field"):
            state_dict(s)

    def test_none_schedule_serialises(self):
        sd = state_dict(band_mf_strategy(bands=4))
        assert sd["lr_schedule"] is None


class TestBltLrScheduleCallable:
    def test_constant_callable_matches_no_schedule(self):
        s_none = blt_strategy(max_buffers=3)
        s_const = blt_strategy(max_buffers=3, lr_schedule=lambda _t: 1.0)
        torch.testing.assert_close(
            s_const.coefficients(**_PART), s_none.coefficients(**_PART)
        )

    def test_callable_schedule_not_serialisable(self):
        s = blt_strategy(max_buffers=3, lr_schedule=lambda _t: 0.1)
        with pytest.raises(CheckpointError, match="callable strategy field"):
            state_dict(s)
