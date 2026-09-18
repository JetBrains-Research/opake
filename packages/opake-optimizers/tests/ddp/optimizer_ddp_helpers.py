"""CPU/Gloo workers for functional optimizer distributed-state tests."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.distributed as dist
from opake_test_support import (
    cleanup_process_group as _cleanup_gloo,
)
from opake_test_support import (
    setup_gloo as _setup_gloo,
)
from opake_test_support import (
    spawn as _spawn,
)

_spawn_gloo = _spawn


def _worker_optimizer_state_audit_gloo(rank: int, world_size: int, port: int) -> None:
    from opake.api.optimizers._adam import AdamState
    from opake.api.optimizers._lion import LionState
    from opake.api.optimizers.distributed import sync_optimizer_state
    from opake.distributed import sync

    _setup_gloo(rank, world_size, port)
    try:
        adam = AdamState(
            mu={"weight": torch.tensor([1.0, -2.0])},
            nu={"weight": torch.tensor([3.0, 4.0])},
            phi={"weight": 0.25},
            step=7,
        )
        lion = LionState(m={"weight": torch.tensor([0.5, -0.5])}, step=7)

        assert sync(adam) is adam
        audited_chain = sync_optimizer_state((adam, lion))
        assert audited_chain == (adam, lion)

        mismatched = replace(adam, phi={"weight": float(rank)})
        with pytest.raises(RuntimeError, match=r"AdamState\.phi"):
            sync(mismatched)

        token = torch.tensor([1.0])
        dist.all_reduce(token, op=dist.ReduceOp.SUM)
        assert token.item() == float(world_size)
    finally:
        _cleanup_gloo()


def _worker_optimizer_state_micro_drift_gloo(
    rank: int, world_size: int, port: int
) -> None:
    """Sub-1e-5 optimizer-state drift must not slip past the audit.

    The fingerprints are float64 statistics of float32 state, so comparing
    them through a float32 collective at ``rtol=1e-5`` hides exactly the
    cross-rank divergence the audit exists to find.
    """
    from opake.api.optimizers._adam import AdamState
    from opake.api.optimizers.distributed import sync_optimizer_state

    _setup_gloo(rank, world_size, port)
    try:
        base = torch.arange(1000, dtype=torch.float32) * 0.001 + 1.0

        # Identical state on every rank passes, so exact comparison does not
        # turn legitimate float32 state into a false positive.
        clean = AdamState(
            mu={"weight": base.clone()},
            nu={"weight": base.clone() * 2.0},
            phi={"weight": 0.25},
            step=7,
        )
        assert sync_optimizer_state(clean) is clean

        # A uniform 1e-6 relative drift on a 1000-element float32 tensor.
        drifted_tensor = base * (1.0 + 1e-6) if rank == 1 else base.clone()
        with pytest.raises(RuntimeError, match=r"AdamState\.mu\['weight'\]\.sum"):
            sync_optimizer_state(replace(clean, mu={"weight": drifted_tensor}))

        # The same 1e-6 relative drift on a plain float leaf.
        drifted_scalar = 0.25 * (1.0 + 1e-6) if rank == 1 else 0.25
        with pytest.raises(RuntimeError, match=r"AdamState\.phi\['weight'\]"):
            sync_optimizer_state(replace(clean, phi={"weight": drifted_scalar}))

        # The process group survived both aborted audits: every rank raised at
        # the same collective, so no rank is left a reduction behind.
        token = torch.tensor([1.0])
        dist.all_reduce(token, op=dist.ReduceOp.SUM)
        assert token.item() == float(world_size)
    finally:
        _cleanup_gloo()
