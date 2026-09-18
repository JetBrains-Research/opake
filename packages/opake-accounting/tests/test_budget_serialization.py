"""Budget checkpoint serialization through the public accountant API."""

from __future__ import annotations

import json
import logging

import pytest

import opake.accounting as acc
from opake.exceptions import CheckpointError
from opake.serialization import from_state_dict, state_dict


class _CustomBudget:
    """Non-dataclass implementation of the public Budget protocol."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.value = threshold
        self.name = "custom"
        self.decreasing = True

    def evaluate(self, process: object) -> float:
        return 0.0


class _UnregisteredBudget(_CustomBudget):
    """Separate protocol implementation that deliberately has no codec."""


def test_registered_non_dataclass_budget_round_trips() -> None:
    acc.register_budget_serializer(
        _CustomBudget,
        lambda budget: {"threshold": budget.threshold},
        lambda state: _CustomBudget(float(state["threshold"])),
    )
    accountant = acc.Accountant(budget=_CustomBudget(2.5))

    checkpoint = json.loads(json.dumps(state_dict(accountant)))
    restored = from_state_dict(acc.Accountant(), checkpoint)

    assert state_dict(restored) == checkpoint


@pytest.mark.parametrize(
    "budget",
    [
        acc.epsilon_budget(3.0, delta=1e-5),
        acc.delta_budget(1e-5, epsilon=3.0),
        acc.advantage_budget(0.1),
        acc.beta_budget(0.05, alpha=0.01),
        acc.risk_budget(0.1, prior=0.5),
    ],
)
def test_builtin_budget_round_trips(budget: object) -> None:
    checkpoint = state_dict(acc.Accountant(budget=budget))
    restored = from_state_dict(acc.Accountant(), checkpoint)

    assert state_dict(restored) == checkpoint


def test_unregistered_budget_reports_registration_requirement() -> None:
    with pytest.raises(CheckpointError, match="register_budget_serializer"):
        state_dict(acc.Accountant(budget=_UnregisteredBudget(2.5)))


def test_unknown_budget_checkpoint_type_reports_registration_requirement() -> None:
    serialized = state_dict(acc.Accountant())
    serialized["budget"] = {"type": "example.UnregisteredBudget"}

    with pytest.raises(CheckpointError, match="no budget serializer is registered"):
        from_state_dict(acc.Accountant(), serialized)


def test_template_budget_is_kept_when_checkpoint_has_none() -> None:
    budget = acc.epsilon_budget(0.1, delta=1e-5)
    checkpoint = state_dict(acc.Accountant() | acc.eps_delta(0.5, 1e-5))

    restored = from_state_dict(acc.Accountant(budget=budget), checkpoint)

    assert restored._budget is budget
    assert restored.budget_exceeded


def test_checkpoint_budget_override_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    saved = acc.epsilon_budget(2.0, delta=1e-5)
    template = acc.epsilon_budget(0.1, delta=1e-5)
    checkpoint = state_dict(acc.Accountant(budget=saved))

    with caplog.at_level(
        logging.WARNING, logger="opake.api.accounting.core._accountant"
    ):
        restored = from_state_dict(acc.Accountant(budget=template), checkpoint)

    assert restored._budget == saved
    assert "template budget is discarded" in caplog.text
