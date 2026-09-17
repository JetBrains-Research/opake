"""Tests for BsrStrategy factory and accounting."""

import pytest

import opake.dpftrl.accounting as ftrl_acc
from opake.api.dpftrl.noise._bsr import BsrStrategy, bsr_strategy

_PART = {"n_steps": 100, "min_sep": 25, "max_participations": 4}


def _bsr(**overrides):
    kwargs = {"bandwidth": 4, "alpha": 1.0, "beta": 0.5}
    kwargs.update(overrides)
    return bsr_strategy(**kwargs)


class TestBsrStrategy:
    def test_returns_correct_type(self):
        assert isinstance(_bsr(), BsrStrategy)

    def test_sensitivity_positive(self):
        assert _bsr().sensitivity(**_PART) > 0

    def test_gram_matrix_present(self):
        gram = _bsr().gram_matrix(**_PART)
        assert gram is not None
        assert len(gram) == 25 * 25

    def test_streaming_matrix_present(self):
        assert _bsr().streaming_matrix(**_PART) is not None

    def test_coefficients_non_increasing_first_band(self):
        s = _bsr(bandwidth=6, beta=0.9)
        head = s.coefficients(n_steps=50, min_sep=10, max_participations=3).tolist()[:6]
        for i in range(1, len(head)):
            assert head[i] <= head[i - 1] + 1e-9

    def test_rejects_alpha_le_beta(self):
        with pytest.raises(ValueError, match="α > β"):
            bsr_strategy(bandwidth=4, alpha=0.9, beta=0.95)

    def test_rejects_momentum_one(self):
        with pytest.raises(ValueError, match=r"β in \[0, 1\)"):
            bsr_strategy(bandwidth=4, alpha=1.0, beta=1.0)

    def test_rejects_unknown_normalized_kwarg(self):
        with pytest.raises(TypeError, match="normalized"):
            bsr_strategy(bandwidth=4, alpha=1.0, beta=0.5, normalized=True)


class TestBsrPld:
    delta = 1e-5

    def test_bsr_pld(self):
        eps = ftrl_acc.mf_gaussian(1.0, _bsr(), **_PART).epsilon_at(self.delta)
        assert eps > 0

    def test_bsr_bnb(self):
        eps = ftrl_acc.balls_in_bins(
            ftrl_acc.mf_gaussian(1.0, _bsr()),
            num_bins=25,
            n_steps=100,
        ).epsilon_at(
            1e-2,
            mc_resolution=5e-3,
            mc_failure_probability=1e-2,
        )
        assert eps > 0
