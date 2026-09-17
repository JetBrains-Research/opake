"""Shared update-RMS-clipping behavior across optimizer factories."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

pytest.importorskip("torchopt")

from opake.optimizers import (
    adadelta,
    adafactor,
    adam,
    adamw,
    ademamix,
    radam,
    rmsprop,
)

_Factory = Callable[[float], Any]


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(
            lambda threshold: adam(lr=1.0, weight_decay=0.0, update_rms_clip=threshold),
            id="adam",
        ),
        pytest.param(
            lambda threshold: adamw(
                lr=1.0, weight_decay=0.0, update_rms_clip=threshold
            ),
            id="adamw",
        ),
        pytest.param(
            lambda threshold: ademamix(
                lr=1.0, weight_decay=0.0, update_rms_clip=threshold
            ),
            id="ademamix",
        ),
        pytest.param(
            lambda threshold: adadelta(
                lr=1.0, weight_decay=0.0, update_rms_clip=threshold
            ),
            id="adadelta",
        ),
        pytest.param(
            lambda threshold: adafactor(
                lr=1.0, eps_root=1.0, weight_decay=0.0, update_rms_clip=threshold
            ),
            id="adafactor",
        ),
        pytest.param(
            lambda threshold: radam(
                lr=1.0, weight_decay=0.0, update_rms_clip=threshold
            ),
            id="radam",
        ),
        pytest.param(
            lambda threshold: rmsprop(
                lr=1.0, weight_decay=0.0, update_rms_clip=threshold
            ),
            id="rmsprop",
        ),
    ],
)
def test_update_rms_clip_uses_one_global_scale(factory: _Factory):
    """A low-RMS leaf is scaled when a higher-RMS leaf activates the global clip."""
    params = {
        "large": torch.zeros(4, 4),
        "small": torch.zeros(4),
    }
    grads = {
        "large": torch.ones(4, 4),
        "small": torch.tensor([10.0, 0.0, 0.0, 0.0]),
    }
    no_clip = factory(1e9)
    unclipped, _ = no_clip.update(grads, no_clip.init(params), params=params)

    global_rms = torch.sqrt(
        sum(update.pow(2).sum() for update in unclipped.values())
        / sum(update.numel() for update in unclipped.values())
    )
    leaf_rms = {name: update.pow(2).mean().sqrt() for name, update in unclipped.items()}
    low_rms = min(leaf_rms.values())
    assert low_rms < global_rms
    threshold = ((low_rms + global_rms) / 2).item()
    expected_scale = (global_rms / threshold).item()

    clipped = factory(threshold)
    updates, _ = clipped.update(grads, clipped.init(params), params=params)

    assert low_rms < threshold < global_rms
    for name in params:
        torch.testing.assert_close(
            updates[name],
            unclipped[name] / expected_scale,
            atol=1e-6,
            rtol=0,
        )
