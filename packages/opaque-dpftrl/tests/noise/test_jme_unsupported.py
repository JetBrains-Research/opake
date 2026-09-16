"""DP-FTRL must not expose the unproved repeated-row JME construction."""

from __future__ import annotations

import pytest
import torch

from opaque.dpftrl.noise import identity_strategy, mf_gaussian_noise
from opaque.random import key


def test_mf_noise_rejects_legacy_second_moment_strategy_argument():
    with pytest.raises(TypeError, match="second_moment_strategy"):
        mf_gaussian_noise(
            torch.zeros(2),
            identity_strategy(),
            n_steps=2,
            noise_multiplier=1.0,
            key=key(0),
            second_moment_strategy=identity_strategy(),  # type: ignore[call-arg]
        )
