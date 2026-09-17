# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
#
# Accountant structure and public privacy-metric API adapted in part from
# Google DP Accounting (Apache-2.0;
# https://github.com/google/differential-privacy/tree/main/python/dp_accounting),
# then reworked for Opake's immutable accountant model.
# See ../../../../../NOTICE in this package for the full attribution.
"""Functional DP process accountant for training loops.

An Accountant tracks the accumulated privacy loss from composed DP processes.
It provides a functional API: composing a new process returns a fresh Accountant.

Merge optimization is handled entirely by :meth:`DpProcess.__or__`:
identical steps are collapsed using structural equality (``==``), so
``acct | step`` in a loop produces ``Repeated(step, n)`` — one
``self_compose(n)`` (2 FFTs) instead of *n* heterogeneous composes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from opake.api.accounting.core._budgets import (
    Budget,
    budget_from_state_dict,
    budget_state_dict,
)
from opake.api.accounting.core._process_codec import _load_dp_process
from opake.api.accounting.core.mechanisms.types import Identity

if TYPE_CHECKING:
    from opake.api.accounting.core._base import DpProcess

__all__ = ["Accountant"]

logger = logging.getLogger(__name__)


class Accountant:
    """Functional DP process accountant for training loops.

    Tracks accumulated privacy by composing processes over time.
    Provides a functional API: composing returns a new Accountant instead of mutating.

    Example::

        import opake.accounting as acc
        from opake.api.accounting.core._accountant import Accountant

        acct = Accountant()
        step = acc.poisson(acc.gaussian(1.1), 0.01)

        for i in range(num_steps):
            acct = acct | step

            if i % 100 == 0:
                eps = acct.epsilon_at(1e-5)
                print(f"Step {i}: ε={eps:.2f}")

    With an optional privacy budget::

        from opake.accounting import calibration as cal

        budget = cal.epsilon_budget(3.0, delta=1e-5)
        acct = Accountant(budget=budget)
        step = acc.poisson(acc.gaussian(1.1), 0.01)

        for i in range(num_steps):
            acct = acct | step

            if acct.budget_exceeded:
                print("Privacy budget exhausted!")
                break

    Seeding with a previously executed process (sequential composition
    across runs, e.g. SFT followed by DPO on the same dataset)::

        import json

        from opake.serialization import from_state_dict

        with open("sft_checkpoint/accountant.json") as f:
            sft = from_state_dict(Accountant(), json.load(f))

        acct = Accountant(budget=budget, prefix=sft.process)

        for i in range(num_dpo_steps):
            acct = acct | dpo_step   # composes on top of the SFT prefix
    """

    def __init__(
        self, budget: Budget | None = None, prefix: DpProcess | None = None
    ) -> None:
        """Initialize an Accountant.

        Args:
            budget: Optional privacy budget. If provided, enables
                budget_exceeded checks. Should be a Budget from the
                budgets module (e.g., epsilon_budget(3.0, delta=1e-5)).
            prefix: Optional already-executed process to seed the
                accountant with, instead of the default zero-cost
                :class:`Identity`. Subsequent compositions and all
                metrics (``epsilon_at``, ``budget_exceeded``, ...)
                account for the prefix.
        """
        self.process: DpProcess = Identity() if prefix is None else prefix
        self._budget: Budget | None = budget

    def __or__(self, process: DpProcess) -> Accountant:
        """Compose a new process onto this accountant.

        Returns a new Accountant with the composed process.  The original
        accountant is not modified.

        Merge optimization is automatic: ``DpProcess.__or__`` uses
        structural equality to collapse identical steps into a single
        :class:`~opake.accounting.composition.Repeated` node.

        Args:
            process: DpProcess to compose (e.g., from poisson(), gaussian(), etc.)

        Returns:
            New Accountant with composed process.

        Example::

            acct = Accountant()
            step = poisson(gaussian(1.1), 0.01)
            acct = acct | step  # One step
            acct = acct | step  # Collapsed into Repeated(step, 2)
        """
        return Accountant(budget=self._budget, prefix=self.process | process)

    def epsilon_at(self, delta: float) -> float:
        """Get epsilon for a target delta.

        Computes (ε, δ)-DP parameter epsilon at the given delta.

        Args:
            delta: Target failure probability. Typically 1e-5 or 1e-6.

        Returns:
            Privacy budget epsilon. Lower is more private.

        Example::

            acct = Accountant()
            step = poisson(gaussian(1.1), 0.01)
            for i in range(1000):
                acct = acct | step

            eps = acct.epsilon_at(1e-5)
            print(f"Privacy: (ε={eps:.2f}, δ=1e-5)")
        """
        return self.process.epsilon_at(delta)

    def delta_at(self, epsilon: float) -> float:
        """Get delta for a target epsilon.

        Computes (ε, δ)-DP parameter delta at the given epsilon.

        Args:
            epsilon: Privacy budget.

        Returns:
            Failure probability delta. Lower is better.
        """
        return self.process.delta_at(epsilon)

    def advantage(self) -> float:
        """Get f-DP advantage (total-variation privacy).

        Represents the maximum probability of distinguishing neighboring
        datasets. Lower is more private (0 = perfectly private).

        Returns:
            Advantage in [0, 1].
        """
        return self.process.advantage()

    def beta_at(self, alpha: float) -> float:
        """Get Type-II error rate (hypothesis testing interpretation).

        Args:
            alpha: Type-I error rate (false positive). Must be in [0, 1].

        Returns:
            Type-II error rate (false negative) in [0, 1].
            Higher is more private (attacker makes more mistakes).
        """
        return self.process.beta_at(alpha)

    def risk_at(self, prior: float) -> float:
        """Get Bayes risk under an optimal adversary.

        Args:
            prior: Prior probability that data came from D (vs D').
                Typically 0.5 for balanced prior.

        Returns:
            Bayes risk in [0, 0.5]. Higher is more private.
        """
        return self.process.risk_at(prior)

    @property
    def budget_exceeded(self) -> bool:
        """Check if accumulated privacy violates the budget.

        Returns False if no budget was specified. Otherwise, evaluates the
        target metric on the accumulated process and checks if it violates
        the budget bound.

        Returns:
            True if privacy budget is violated, False otherwise.
        """
        if self._budget is None:
            return False

        achieved = self._budget.evaluate(self.process)
        if self._budget.decreasing:
            return achieved > self._budget.value
        return achieved < self._budget.value


def _accountant_state_dict(acct: Accountant) -> dict[str, Any]:
    from opake.serialization import state_dict as opake_state_dict

    out: dict[str, Any] = {"process": dict(opake_state_dict(acct.process))}
    if acct._budget is not None:
        out["budget"] = budget_state_dict(acct._budget)
    return out


def _accountant_from_state_dict(
    template: Accountant, state: dict[str, Any]
) -> Accountant:
    """Restore the saved budget, or retain the template's when none was saved."""
    budget = template._budget
    if state.get("budget") is not None:
        budget = budget_from_state_dict(dict(state["budget"]))
        if template._budget is not None and budget != template._budget:
            logger.warning(
                "Restoring Accountant: checkpoint budget %r overrides the "
                "template's budget %r; the template budget is discarded.",
                budget,
                template._budget,
            )
    return Accountant(budget=budget, prefix=_load_dp_process(dict(state["process"])))


def _register_accountant_serialization() -> None:
    from opake.serialization import register_serializer

    register_serializer(
        Accountant,
        _accountant_state_dict,
        lambda template, sd: _accountant_from_state_dict(template, dict(sd)),
    )
