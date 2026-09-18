"""Tests for opake.accounting.mechanisms — EpsDelta, Identity (shared); Gaussian via dpsgd."""

import math
from dataclasses import FrozenInstanceError

import pytest

import opake.accounting as acc
import opake.dpsgd.accounting as dpsgd_acc
from opake.api.accounting.core._base import DpProcess
from opake.api.accounting.core.mechanisms.types import EpsDelta, Identity
from opake.dpsgd.accounting.mechanisms.types import Gaussian

# ── Mechanism dataclass tests ────────────────────────────────────────


class TestGaussianDataclass:
    """Gaussian frozen dataclass."""

    def test_fields(self):
        g = Gaussian(1.1)
        assert g.noise_multiplier == pytest.approx(1.1)

    def test_frozen(self):
        g = Gaussian(1.1)
        with pytest.raises(FrozenInstanceError):
            g.noise_multiplier = 2.0  # type: ignore[misc]

    def test_is_dp_process(self):
        assert isinstance(Gaussian(1.0), DpProcess)

    def test_equality(self):
        assert Gaussian(1.0) == Gaussian(1.0)
        assert Gaussian(1.0) != Gaussian(1.1)

    def test_pld_with_query_config(self):
        """Config is now query-time - test pld() with different discretization."""
        g = Gaussian(1.0)
        # Both should compute successfully with different query configs
        pld1 = g.pld(discretization=1e-3)
        pld2 = g.pld(discretization=1e-4)
        eps1 = pld1.epsilon_at(1e-5)
        eps2 = pld2.epsilon_at(1e-5)
        assert math.isfinite(eps1)
        assert eps1 > 0
        assert math.isfinite(eps2)
        assert eps2 > 0

    def test_pld_returns_valid(self):
        pld = Gaussian(0.8).pld()
        eps = pld.epsilon_at(1e-5)
        assert math.isfinite(eps)
        assert eps > 0


class TestEpsDeltaDataclass:
    """EpsDelta frozen dataclass."""

    def test_fields(self):
        e = EpsDelta(1.0, 1e-5)
        assert e.epsilon == pytest.approx(1.0)
        assert e.delta == pytest.approx(1e-5)

    def test_frozen(self):
        e = EpsDelta(1.0, 1e-5)
        with pytest.raises(FrozenInstanceError):
            e.epsilon = 2.0  # type: ignore[misc]

    def test_is_dp_process(self):
        assert isinstance(EpsDelta(1.0, 1e-5), DpProcess)

    def test_equality(self):
        assert EpsDelta(1.0, 1e-5) == EpsDelta(1.0, 1e-5)
        assert EpsDelta(1.0, 1e-5) != EpsDelta(2.0, 1e-5)

    def test_pld_returns_valid(self):
        pld = EpsDelta(1.0, 1e-5).pld()
        d = pld.delta_at(1.0)
        assert math.isfinite(d)


# ── Constructor function tests ───────────────────────────────────────


class TestGaussianConstructor:
    """dpsgd_acc.gaussian() returns Gaussian with correct config."""

    def test_returns_gaussian(self):
        g = dpsgd_acc.gaussian(1.1)
        assert isinstance(g, Gaussian)
        assert g.noise_multiplier == pytest.approx(1.1)

    def test_default_config_none(self):
        g = dpsgd_acc.gaussian(1.1)
        # config is None or module default, either way pld works
        eps = g.epsilon_at(1e-5)
        assert math.isfinite(eps)
        assert eps > 0


class TestEpsDeltaConstructor:
    """acc.eps_delta() returns EpsDelta."""

    def test_returns_eps_delta(self):
        e = acc.eps_delta(1.0, 1e-5)
        assert isinstance(e, EpsDelta)
        assert e.epsilon == pytest.approx(1.0)
        assert e.delta == pytest.approx(1e-5)

    def test_pure_dp(self):
        """Default delta=0 → pure ε-DP."""
        e = acc.eps_delta(1.0)
        assert e.delta == pytest.approx(0.0)


class TestIdentityConstructor:
    """acc.identity() returns Identity."""

    def test_returns_identity(self):
        i = acc.identity()
        assert isinstance(i, Identity)

    def test_zero_epsilon(self):
        i = acc.identity()
        eps = i.epsilon_at(1e-5)
        assert eps == pytest.approx(0.0, abs=1e-10)


class TestIdentityDataclass:
    """Identity node — zero privacy loss."""

    def test_is_dp_process(self):
        assert isinstance(Identity(), DpProcess)

    def test_is_dataclass(self):
        """Identity is a dataclass."""
        i = Identity()
        import dataclasses

        assert dataclasses.is_dataclass(i)

    def test_zero_epsilon(self):
        eps = Identity().epsilon_at(1e-5)
        assert eps == pytest.approx(0.0, abs=1e-10)

    def test_zero_advantage(self):
        adv = Identity().advantage()
        assert adv == pytest.approx(0.0, abs=1e-10)

    def test_equality(self):
        assert Identity() == Identity()
