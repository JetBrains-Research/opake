"""Tests for the optimizer chain composer."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("torchopt")

from torchopt.base import GradientTransformation

from opake.api.optimizers._chain import make_optimizer_chain
from opake.exceptions import ConfigurationError
from opake.types import SecondMomentNoiseOutput, noised


def test_generic_kwargs_do_not_claim_second_moment_support():
    def init_fn(params):
        return ()

    def update_fn(updates, state, *, params=None, inplace=False, **kwargs):
        raise AssertionError("moment scaler must not run")

    scaler = GradientTransformation(init_fn, update_fn)
    optimizer = make_optimizer_chain(
        scaler,
        lr=0.1,
        weight_decay=0.0,
        optimizer_name="generic",
    )
    params = {"weight": torch.ones(2)}
    output = SecondMomentNoiseOutput(
        noised(params, max_norm=1.0, noise_stddev=0.1),
        noised(params, max_norm=1.0, noise_stddev=0.1),
    )

    with pytest.raises(
        ConfigurationError,
        match=(
            r'Optimizer "generic" cannot consume SecondMomentNoiseOutput.*'
            r"adadelta, adam, adamw, ademamix, radam, rmsprop"
        ),
    ):
        optimizer.update(output, optimizer.init(params), params=params)
