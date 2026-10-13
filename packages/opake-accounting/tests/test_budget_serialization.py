"""Budget checkpoint serialization through the public accountant API."""

from __future__ import annotations

import json

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


def test_budget_accessor() -> None:
    budget = acc.epsilon_budget(1.0, delta=1e-5)

    assert acc.Accountant().budget is None
    assert acc.Accountant(budget=budget).budget is budget


def test_registered_non_dataclass_budget_round_trips() -> None:
    acc.register_budget_serializer(
        _CustomBudget,
        lambda budget: {"threshold": budget.threshold},
        lambda state: _CustomBudget(float(state["threshold"])),
    )
    accountant = acc.Accountant(budget=_CustomBudget(2.5))

    checkpoint = json.loads(json.dumps(state_dict(accountant)))
    restored = from_state_dict(acc.Accountant(budget=_CustomBudget(2.5)), checkpoint)

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
    template = acc.Accountant(budget=acc.epsilon_budget(1.0, delta=1e-5))

    with pytest.raises(CheckpointError, match="no budget serializer is registered"):
        from_state_dict(template, serialized)


def test_template_budget_is_kept_when_checkpoint_has_none() -> None:
    budget = acc.epsilon_budget(0.1, delta=1e-5)
    checkpoint = state_dict(acc.Accountant() | acc.eps_delta(0.5, 1e-5))

    restored = from_state_dict(acc.Accountant(budget=budget), checkpoint)

    assert restored.budget is budget
    assert restored.budget_exceeded


def test_matching_checkpoint_and_template_budgets_restore() -> None:
    saved = acc.epsilon_budget(2.0, delta=1e-5)
    template = acc.epsilon_budget(2.0, delta=1e-5)
    checkpoint = state_dict(acc.Accountant(budget=saved))

    restored = from_state_dict(acc.Accountant(budget=template), checkpoint)

    assert restored.budget == saved


@pytest.mark.parametrize(
    ("saved_epsilon", "template_epsilon"),
    [(2.0, 0.1), (0.1, 2.0)],
)
def test_conflicting_checkpoint_and_template_budgets_raise(
    saved_epsilon: float, template_epsilon: float
) -> None:
    saved = acc.epsilon_budget(saved_epsilon, delta=1e-5)
    template = acc.epsilon_budget(template_epsilon, delta=1e-5)
    checkpoint = state_dict(acc.Accountant(budget=saved))

    with pytest.raises(CheckpointError, match=r"checkpoint budget.*template budget"):
        from_state_dict(acc.Accountant(budget=template), checkpoint)
