"""Bounds in ``__post_init__`` must fire for direct construction and
deserialization, not only through the factories."""

import pytest

import opake.accounting as acc
from opake.api.accounting.core.composition._repeated import Repeated
from opake.api.accounting.core.mechanisms._eps_delta import EpsDelta
from opake.exceptions import ConfigurationError
from opake.serialization import from_state_dict


class TestEpsDelta:
    def test_accepts_documented_bounds(self):
        assert EpsDelta(0.0, 0.0).epsilon == 0.0
        assert EpsDelta(1.0, 1.0).delta == 1.0

    @pytest.mark.parametrize("epsilon", [-1.0, -1e-12])
    def test_rejects_negative_epsilon(self, epsilon):
        with pytest.raises(ConfigurationError, match="epsilon must be non-negative"):
            EpsDelta(epsilon, 1e-5)

    @pytest.mark.parametrize("delta", [-1e-9, 1.0 + 1e-9, 2.0])
    def test_rejects_delta_outside_unit_interval(self, delta):
        with pytest.raises(ConfigurationError, match=r"delta must be in \[0, 1\]"):
            EpsDelta(1.0, delta)

    def test_factory_rejects_invalid_parameters(self):
        with pytest.raises(ValueError, match="epsilon must be non-negative"):
            acc.eps_delta(-1.0, 1e-5)
        with pytest.raises(ValueError, match=r"delta must be in \[0, 1\]"):
            acc.eps_delta(1.0, 1.5)


class TestRepeated:
    @pytest.mark.parametrize("count", [0, -3])
    def test_rejects_count_below_one(self, count):
        with pytest.raises(ConfigurationError, match=r"Repeat count must be >= 1"):
            Repeated(acc.eps_delta(1.0, 1e-5), count)

    @pytest.mark.parametrize("count", [0, -1])
    def test_factory_rejects_count_below_one(self, count):
        with pytest.raises(ValueError, match=r"Repeat count must be >= 1"):
            acc.repeat(acc.eps_delta(1.0, 1e-5), count=count)


class TestCodecValidation:
    """Tampered checkpoint dicts must fail on load, not pass silently."""

    @pytest.mark.parametrize(
        ("state", "match"),
        [
            (
                {"type": "EpsDelta", "epsilon": -1.0, "delta": 1e-5},
                "epsilon must be non-negative",
            ),
            (
                {"type": "EpsDelta", "epsilon": 1.0, "delta": 1.5},
                r"delta must be in \[0, 1\]",
            ),
            (
                {
                    "type": "Repeated",
                    "inner": {"type": "EpsDelta", "epsilon": 1.0, "delta": 1e-5},
                    "count": 0,
                },
                r"Repeat count must be >= 1",
            ),
        ],
    )
    def test_from_state_dict_rejects_invalid(self, state, match):
        with pytest.raises(ValueError, match=match):
            from_state_dict(acc.identity(), state)
