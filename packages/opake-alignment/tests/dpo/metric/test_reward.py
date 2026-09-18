"""Unit tests for the DPO reward metrics (§7.9).

Hand-computed reference cases for :func:`reward_metrics` — chosen/rejected
means, accuracy fraction, and margin — on tiny per-example tensors. The metric
is telemetry (§3.3 telemetry rule), so each test also asserts the returned
tensors are detached (``requires_grad is False``) and scalar.
"""

from __future__ import annotations

import pytest
import torch

from opake.api.alignment.dpo.metric._reward import reward_metrics


def test_reward_metrics_reference() -> None:
    chosen = torch.tensor([1.0, 2.0, 0.5])
    rejected = torch.tensor([0.0, 3.0, 0.5])
    beta = 0.5
    out = reward_metrics(chosen, rejected, beta=beta)

    # rewards/chosen = beta * mean(chosen) = 0.5 * (3.5 / 3)
    assert out["rewards/chosen"].item() == pytest.approx(0.5 * (3.5 / 3))
    # rewards/rejected = beta * mean(rejected) = 0.5 * (3.5 / 3)
    assert out["rewards/rejected"].item() == pytest.approx(0.5 * (3.5 / 3))
    # accuracies = fraction where chosen > rejected: [T, F, F] -> 1/3
    assert out["rewards/accuracies"].item() == pytest.approx(1.0 / 3.0)
    # margins = beta * mean(chosen - rejected) = 0.5 * mean([1, -1, 0]) = 0
    assert out["rewards/margins"].item() == pytest.approx(0.0)


def test_reward_metrics_all_chosen_win() -> None:
    chosen = torch.tensor([2.0, 5.0])
    rejected = torch.tensor([1.0, 4.0])
    out = reward_metrics(chosen, rejected, beta=1.0)
    assert out["rewards/accuracies"].item() == pytest.approx(1.0)
    # margins = 1.0 * mean([1, 1]) = 1.0
    assert out["rewards/margins"].item() == pytest.approx(1.0)


def test_reward_metrics_detached_and_scalar() -> None:
    chosen = torch.tensor([1.0, 2.0], requires_grad=True)
    rejected = torch.tensor([0.0, 1.0], requires_grad=True)
    out = reward_metrics(chosen, rejected, beta=0.3)
    for key, value in out.items():
        assert value.requires_grad is False, key
        assert value.ndim == 0, key
