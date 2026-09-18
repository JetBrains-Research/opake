"""Tests for cached composition."""

from unittest.mock import patch

import opake.accounting as acc


def test_cached_process_preserves_repeated_pld_override():
    count = 3
    expected_epsilon = 2.0
    query = {
        "discretization": 0.1,
        "log_x_mass_truncation_bound": -20.0,
        "max_grid_size": 1_000_000,
        "max_conv_grid": 1_000_000,
        "seed": 7,
        "mc_resolution": 0.01,
        "mc_failure_probability": 1e-6,
    }
    process = acc.eps_delta(1.0)
    repeated_pld = acc.eps_delta(expected_epsilon).pld()

    with patch.object(
        type(process), "repeated_pld", return_value=repeated_pld
    ) as override:
        actual = acc.cached(process).repeated_pld(count, **query).epsilon_at(0.0)

    assert actual == expected_epsilon
    override.assert_called_once_with(count, **query)
