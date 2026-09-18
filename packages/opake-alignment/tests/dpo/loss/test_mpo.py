# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit + vmap-safety tests for the MPO combinator helper (plan §7.1, §3.4).

Covers :func:`mpo_combine` — the TRL ``loss_type=list`` weighted blend (MPO). A
weighted-sum reference case, the subset-of-keys behaviour, and the missing-key
``KeyError`` guard.

Imports target the concrete impl paths because the public façade is wired by a
later unit (γ.W).
"""

from __future__ import annotations

import pytest
import torch
from torch.func import grad, vmap

from opake.api.alignment.dpo.loss._mpo import mpo_combine
from opake.exceptions import ConfigurationError

# ---------------------------------------------------------------------------
# mpo_combine — weighted blend (TRL loss_type=list)
# ---------------------------------------------------------------------------


def test_mpo_combine_weighted_sum() -> None:
    """{sigmoid, sft} with weights {1.0, 0.5} → t1 + 0.5 * t2."""
    t1 = torch.tensor([1.0, 2.0, 3.0])
    t2 = torch.tensor([10.0, 20.0, 30.0])
    out = mpo_combine({"sigmoid": t1, "sft": t2}, {"sigmoid": 1.0, "sft": 0.5})
    assert torch.allclose(out, t1 + 0.5 * t2)


def test_mpo_combine_subset_weights_ok() -> None:
    """weights may select a strict subset of the available losses."""
    t1 = torch.tensor([1.0, 2.0])
    t2 = torch.tensor([4.0, 8.0])
    out = mpo_combine({"sigmoid": t1, "sft": t2}, {"sigmoid": 2.0})
    assert torch.allclose(out, 2.0 * t1)


def test_mpo_combine_single_term() -> None:
    """A single-term blend returns the scaled term."""
    t1 = torch.tensor([1.0, 2.0, 3.0])
    out = mpo_combine({"sigmoid": t1}, {"sigmoid": 0.25})
    assert torch.allclose(out, 0.25 * t1)


def test_mpo_combine_broadcasts() -> None:
    """Loss tensors that broadcast together are combined per broadcast rules."""
    scalar = torch.tensor(1.0)
    vec = torch.tensor([1.0, 2.0, 3.0])
    out = mpo_combine({"a": scalar, "b": vec}, {"a": 2.0, "b": 1.0})
    assert torch.allclose(out, 2.0 * scalar + vec)
    assert out.shape == (3,)


def test_mpo_combine_missing_key_raises() -> None:
    """A weight key absent from losses raises ConfigurationError."""
    t1 = torch.tensor([1.0, 2.0])
    with pytest.raises(ConfigurationError):
        mpo_combine({"sigmoid": t1}, {"sigmoid": 1.0, "sft": 0.5})


def test_mpo_combine_vmap_grad_finite() -> None:
    """vmap(grad(mpo_combine-sum)) over (B,) inputs yields finite grads (§3.4)."""
    a = torch.tensor([0.5, -1.0, 2.0, 0.0])
    b = torch.tensor([1.0, 0.0, -2.0, 3.0])

    def per_example(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return mpo_combine({"a": x, "b": y}, {"a": 1.0, "b": 0.5}).sum()

    g = vmap(grad(per_example))(a, b)
    assert g.shape == (4,)
    assert torch.isfinite(g).all()
    # grad w.r.t. the "a" term is its weight (1.0).
    assert torch.allclose(g, torch.ones(4))
