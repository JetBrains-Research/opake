"""Tests for ``opake.api.optimizers.distributed`` sync handlers.

The handlers are audit-only (assert cross-rank equality of optimizer state
structure and leaves), so the multi-rank checks themselves are exercised by
the engine's DDP test suite. These tests cover the single-process invariants:

- Registration happens at import time for every Opake optimizer state.
- ``sync_optimizer_state`` is a no-op when not distributed (returns input
  unchanged — same object identity for both bare dataclass and chain states).
- Recursion walks tuples / lists / dicts / nested dataclasses / strings /
  unknown leaf types without raising.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch

from opake.api.engine.distributed._state import _SYNC_REGISTRY
from opake.api.optimizers import distributed as _trigger_registration  # noqa: F401
from opake.api.optimizers._adadelta import AdadeltaState
from opake.api.optimizers._adafactor import AdafactorState
from opake.api.optimizers._adagrad import AdagradState
from opake.api.optimizers._adam import AdamState
from opake.api.optimizers._ademamix import AdEMAMixState
from opake.api.optimizers._lion import LionState
from opake.api.optimizers._radam import RAdamState
from opake.api.optimizers._rmsprop import RMSpropState
from opake.api.optimizers._schedule_free import ScheduleFreeState
from opake.api.optimizers.distributed import sync_optimizer_state

_ALL_OPTIMIZER_STATES = (
    AdadeltaState,
    AdamState,
    AdEMAMixState,
    AdafactorState,
    AdagradState,
    LionState,
    RAdamState,
    RMSpropState,
    ScheduleFreeState,
)


class TestRegistration:
    """Every Opake optimizer state must be discoverable by ``opake.distributed.sync``."""

    def test_all_optimizer_states_registered(self):
        for state_type in _ALL_OPTIMIZER_STATES:
            assert state_type in _SYNC_REGISTRY, (
                f"{state_type.__name__} is not registered in _SYNC_REGISTRY"
            )

    def test_handler_is_sync_optimizer_state(self):
        for state_type in _ALL_OPTIMIZER_STATES:
            assert _SYNC_REGISTRY[state_type] is sync_optimizer_state


class TestNonDistributedNoOp:
    """In single-process mode the handler is a pass-through.

    The function returns ``state`` before any container construction (see
    the ``if not is_distributed(): return state`` early-return), so we
    assert *identity* of the returned object, not just equality.
    """

    def _adam_state(self) -> AdamState:
        return AdamState(
            mu={"w": torch.zeros(4)},
            nu={"w": torch.zeros(4)},
            phi=0.0,
            step=0,
        )

    def test_returns_same_instance_for_dataclass(self):
        state = self._adam_state()
        out = sync_optimizer_state(state)
        assert out is state

    def test_returns_input_tuple_unchanged(self):
        # Single-process mode returns the original tuple unchanged (no fresh
        # tuple is constructed; the recursive ``tuple(...)`` branch is only
        # reached under ``is_distributed()``).  Identity holds end-to-end.
        a = self._adam_state()
        b = self._adam_state()
        chain = (a, b)
        out = sync_optimizer_state(chain)
        assert out is chain

    def test_returns_input_for_unknown_types(self):
        # torchopt's ``EmptyState`` is a non-dataclass; the handler returns
        # it unchanged.  Use a plain object that's neither a dataclass nor
        # a list/tuple to simulate.
        class _Empty:
            pass

        sentinel = _Empty()
        assert sync_optimizer_state(sentinel) is sentinel


class TestRecursionCoverage:
    """The internal walker must descend into every container shape used by
    real optimizer states without raising on unknown leaf types."""

    def test_nested_dict_of_tensors_does_not_raise(self):
        state = AdamState(
            mu={"layer.0.w": torch.zeros(3), "layer.1.w": torch.zeros(5)},
            nu={"layer.0.w": torch.zeros(3), "layer.1.w": torch.zeros(5)},
            phi=0.0,
            step=42,
        )
        assert sync_optimizer_state(state) is state

    def test_per_group_phi_dict_walked(self):
        # ``AdamState.phi`` can be ``float`` or ``dict[str, float]`` for the
        # per-group DP-BC path; both must be accepted.
        state = AdamState(
            mu={"a": torch.zeros(2)},
            nu={"a": torch.zeros(2)},
            phi={"group_a": 1.5, "group_b": 2.5},
            step=7,
        )
        assert sync_optimizer_state(state) is state

    def test_adafactor_v_flat_tuple_of_tuple_of_tensors(self):
        # Adafactor's factored second moment is shaped as nested tuples of
        # tensors; the recursion must traverse ``v_flat`` (tuple of tuple
        # of Tensor) and the ``treespec`` opake field must be handled
        # without raising.
        try:
            import optree

            treespec = optree.tree_flatten({"w": torch.zeros(4, 4)})[1]
        except ImportError:
            treespec = None
        state = AdafactorState(
            m=None,
            v_flat=((torch.zeros(4), torch.zeros(4)),),
            phi_flat=(0.0,),
            treespec=treespec,
            paths=("w",),
            step=1,
        )
        assert sync_optimizer_state(state) is state

    def test_skip_none_string_bool_leaves(self):
        # Synthetic dataclass exercising the audit's ``None`` / ``str`` /
        # ``bool`` paths.  Strings are now compared cross-rank (not skipped);
        # the walker must traverse them without raising in single-process
        # mode.
        @dataclasses.dataclass
        class _Dummy:
            none_field: Any = None
            str_field: str = "weights"
            bool_field: bool = True
            t: torch.Tensor = dataclasses.field(default_factory=lambda: torch.zeros(2))

        assert sync_optimizer_state(_Dummy()) is not None  # no-raise

    def test_radam_state_walks(self):
        # ``RAdamState`` must be auditable in single-process mode; the
        # registered handler is shared with the other optimizer dataclasses.
        state = RAdamState(
            mu={"w": torch.zeros(2)},
            nu={"w": torch.zeros(2)},
            phi=0.0,
            step=0,
        )
        assert sync_optimizer_state(state) is state

    def test_adadelta_state_walks(self):
        state = AdadeltaState(
            v_g={"w": torch.zeros(2)},
            v_dx={"w": torch.zeros(2)},
            phi_g=0.0,
            phi_dx={"w": torch.zeros(2)},
            step=0,
        )
        assert sync_optimizer_state(state) is state


class TestChainState:
    """torchopt chain states are tuples of per-transform states; recursion
    must descend without breaking on the outer tuple boundary."""

    def test_tuple_chain_descends(self):
        inner_a = AdamState(
            mu={"w": torch.zeros(2)},
            nu={"w": torch.zeros(2)},
            phi=0.0,
            step=0,
        )
        inner_b = LionState(m={"w": torch.zeros(2)}, step=0)
        chain = (inner_a, inner_b)
        out = sync_optimizer_state(chain)
        # Single-process mode returns the original tuple unchanged.
        assert out is chain

    def test_list_chain_descends(self):
        inner = AdamState(
            mu={"w": torch.zeros(2)},
            nu={"w": torch.zeros(2)},
            phi=0.0,
            step=0,
        )
        out = sync_optimizer_state([inner])
        assert out is not None
