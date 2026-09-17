"""Tests for BltStrategy factory and accounting equivalence."""

import warnings

import pytest
import torch

import opake.dpftrl.accounting as ftrl_acc
from opake.api.accounting.core import _native
from opake.api.accounting.dpftrl.amplification import _balls_in_bins
from opake.api.dpftrl.noise._blt import BltStrategy, blt_strategy
from opake.api.dpftrl.noise._toeplitz import (
    materialize_lower_triangular,
    minsep_sensitivity_squared,
)

_PART = {"n_steps": 100, "min_sep": 25, "max_participations": 4}
_SMALL_PART = {"n_steps": 20, "min_sep": 5, "max_participations": 4}


class TestBltStrategy:
    def test_returns_correct_type(self):
        assert isinstance(blt_strategy(momentum=0.95), BltStrategy)

    def test_sensitivity_positive(self):
        s = blt_strategy(momentum=0.95)
        assert s.sensitivity(**_PART) > 0

    def test_gram_matrix_present(self):
        s = blt_strategy(momentum=0.95)
        gram = s.gram_matrix(**_PART)
        assert gram is not None
        assert len(gram) == 25 * 25

    def test_gram_matrix_matches_deployed_unnormalized_encoder(self):
        s = blt_strategy(max_buffers=2, momentum=0.917)
        coefs = s.coefficients(**_SMALL_PART).tolist()
        raw = _native.toeplitz_gram_matrix(
            coefs,
            _SMALL_PART["n_steps"],
            _SMALL_PART["min_sep"],
            _SMALL_PART["max_participations"],
            False,
        )
        normalized = _native.toeplitz_gram_matrix(
            coefs,
            _SMALL_PART["n_steps"],
            _SMALL_PART["min_sep"],
            _SMALL_PART["max_participations"],
            True,
        )

        gram = s.gram_matrix(**_SMALL_PART)

        assert gram == pytest.approx(raw)
        assert max(abs(a - b) for a, b in zip(raw, normalized, strict=True)) > 1e-3

    def test_coefficients_length(self):
        s = blt_strategy(momentum=0.95)
        assert s.coefficients(**_PART).shape[0] == 100

    def test_streaming_matrix_present(self):
        s = blt_strategy(momentum=0.95)
        assert s.streaming_matrix(**_PART) is not None

    def test_matches_old_sensitivity(self):
        assert blt_strategy(momentum=0.95).sensitivity(**_PART) > 0

    def test_single_participation(self):
        s = blt_strategy()
        assert s.sensitivity(n_steps=50, min_sep=1, max_participations=1) > 0

    @pytest.mark.parametrize("momentum", [0.0, 0.5])
    def test_nondefault_momentum_stays_in_minsep_domain(self, momentum):
        part = {"n_steps": 20, "min_sep": 1, "max_participations": 20}
        strategy = blt_strategy(max_buffers=3, momentum=momentum)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            coefs = strategy.coefficients(**part)

        assert torch.all(coefs >= 0)
        assert torch.all(coefs[:-1] >= coefs[1:])
        reported = strategy.sensitivity(**part)

        dense = materialize_lower_triangular(coefs, part["n_steps"])
        all_steps = torch.ones(part["n_steps"], dtype=coefs.dtype)
        aligned_sensitivity = torch.linalg.vector_norm(dense @ all_steps)
        assert reported == pytest.approx(float(aligned_sensitivity))

        expected = minsep_sensitivity_squared(
            coefs,
            min_sep=part["min_sep"],
            max_participations=part["max_participations"],
        ).sqrt()
        assert reported == pytest.approx(float(expected))


class TestBltPld:
    delta = 1e-5

    def test_blt_pld(self):
        s = blt_strategy(momentum=0.95)
        eps = ftrl_acc.mf_gaussian(1.0, s, **_PART).epsilon_at(self.delta)
        assert eps > 0

    def test_blt_bnb(self):
        s = blt_strategy(momentum=0.95)
        eps = ftrl_acc.balls_in_bins(
            ftrl_acc.mf_gaussian(1.0, s),
            num_bins=25,
            n_steps=100,
        ).epsilon_at(
            1e-2,
            mc_resolution=5e-3,
            mc_failure_probability=1e-2,
        )
        assert eps > 0

    def test_blt_bnb_prefix_charges_full_deployed_encoder(self, monkeypatch):
        s = blt_strategy(max_buffers=2, momentum=0.913)
        proc = ftrl_acc.balls_in_bins(
            ftrl_acc.mf_gaussian(1.0, s),
            num_bins=5,
            n_steps=20,
        )
        actual_native = _balls_in_bins._native
        captured: dict[str, tuple[float, ...]] = {}
        sentinel = object()

        class CapturingNative:
            def __getattr__(self, name):
                return getattr(actual_native, name)

            def bnb_mc_pld(self, gram, *_):
                captured["gram"] = tuple(gram)
                return sentinel

            def bnb_pld_from_transcript_handle(self, _handle, gram, *_):
                captured["gram"] = tuple(gram)
                return sentinel

        monkeypatch.setattr(_balls_in_bins, "_native", CapturingNative())

        assert proc.pld() is sentinel

        coefs = s.coefficients(
            n_steps=proc.n_steps,
            min_sep=proc.min_sep,
            max_participations=proc.max_participations,
        ).tolist()
        raw = actual_native.toeplitz_gram_matrix(coefs, 20, 5, 4, False)
        normalized = actual_native.toeplitz_gram_matrix(coefs, 20, 5, 4, True)
        assert captured["gram"] == pytest.approx(raw)
        assert max(abs(a - b) for a, b in zip(raw, normalized, strict=True)) > 1e-3
