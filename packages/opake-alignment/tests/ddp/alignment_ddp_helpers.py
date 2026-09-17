"""Gloo CPU workers for the sharded reference-logprob precompute."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
from opake_test_support import (
    cleanup_process_group as _cleanup_ddp,
)
from opake_test_support import (
    setup_gloo as _setup_gloo_base,
)
from opake_test_support import (
    spawn as _spawn,
)

from opake.exceptions import OperationError

_spawn_gloo = _spawn

# Short timeout: a mismatched collective sequence must fail promptly rather than
# hang the CI job.
_PG_TIMEOUT = timedelta(seconds=120)
_COLUMNS = ("ref_chosen_logps", "ref_rejected_logps")


def _setup_gloo(rank: int, world_size: int, port: int) -> None:
    """Initialize a CPU process group for the precompute regressions."""
    _setup_gloo_base(rank, world_size, port, timeout=_PG_TIMEOUT)


def _make_dataset(n_rows: int, start: int = 0) -> Any:
    from datasets import Dataset

    indices = list(range(start, start + n_rows))
    return Dataset.from_dict(
        {"idx": indices, "prompt": [f"p{index}" for index in indices]}
    )


def _collate(rows: list[dict]) -> dict[str, torch.Tensor]:
    return {"idx": torch.tensor([int(row["idx"]) for row in rows], dtype=torch.long)}


def _expected_columns(n_rows: int, start: int = 0) -> dict[str, list[float]]:
    indices = range(start, start + n_rows)
    return {
        "ref_chosen_logps": [float(index) for index in indices],
        "ref_rejected_logps": [-2.0 * index for index in indices],
    }


class _CountingRef:
    """Reference callable recording which examples this rank actually scored."""

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        self.dtype = dtype
        self.examples_seen: list[int] = []

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        idx = batch["idx"]
        self.examples_seen.extend(int(value) for value in idx)
        values = idx.to(self.dtype)
        return {"ref_chosen_logps": values, "ref_rejected_logps": -2.0 * values}


def _assert_columns(result: Any, n_rows: int, start: int = 0) -> None:
    assert len(result) == n_rows
    for name, values in _expected_columns(n_rows, start).items():
        assert result[name] == values


def _assert_no_desync(world_size: int) -> None:
    """A follow-up collective proves that every rank completed the prior path."""
    token = torch.tensor([float(dist.get_rank() + 1)])
    dist.all_reduce(token, op=dist.ReduceOp.SUM)
    assert abs(token.item() - sum(range(1, world_size + 1))) < 1e-5


def _assert_sharded_output(
    rank: int,
    world_size: int,
    cache_root: str,
    *,
    n_rows: int,
    start: int,
    dtype: torch.dtype,
    identity: str,
) -> None:
    from opake.alignment.dpo.reference import compute_ref_logprobs_for_dataset
    from opake.distributed import local_shard

    dataset = _make_dataset(n_rows, start)
    ref = _CountingRef(dtype=dtype)
    result = compute_ref_logprobs_for_dataset(
        dataset,
        ref,
        _collate,
        _COLUMNS,
        cache_identity={"kind": identity},
        batch_size=2,
        cache_dir=str(Path(cache_root) / identity),
        use_cache=False,
    )

    own_shard = local_shard(
        range(start, start + n_rows), rank=rank, world_size=world_size
    )
    assert ref.examples_seen == list(own_shard)
    _assert_columns(result, n_rows, start)
    _assert_no_desync(world_size)


def _assert_shared_cache_hit(rank: int, world_size: int, cache_root: str) -> None:
    from opake.alignment.dpo.reference import compute_ref_logprobs_for_dataset

    dataset = _make_dataset(6)
    call = {
        "cache_identity": {"kind": "shared-cache"},
        "batch_size": 2,
        "cache_dir": str(Path(cache_root) / "shared"),
        "use_cache": True,
    }
    compute_ref_logprobs_for_dataset(
        dataset, _CountingRef(), _collate, _COLUMNS, **call
    )

    ref = _CountingRef()
    result = compute_ref_logprobs_for_dataset(dataset, ref, _collate, _COLUMNS, **call)
    assert ref.examples_seen == []
    _assert_columns(result, 6)
    _assert_no_desync(world_size)


def _assert_divergent_cache_recomputes(
    rank: int,
    world_size: int,
    cache_root: str,
    *,
    shard: bool | None,
) -> None:
    from opake.alignment.dpo.reference import compute_ref_logprobs_for_dataset
    from opake.distributed import local_shard

    n_rows = 6
    cache_kind = "unsharded" if shard is False else "sharded"
    dataset = _make_dataset(n_rows)
    cache_dir = Path(cache_root) / cache_kind / f"rank{rank}"
    call = {
        "cache_identity": {"kind": f"divergent-cache-{cache_kind}"},
        "batch_size": 2,
        "cache_dir": str(cache_dir),
        "use_cache": True,
        "shard": shard,
    }
    compute_ref_logprobs_for_dataset(
        dataset, _CountingRef(), _collate, _COLUMNS, **call
    )
    assert any(cache_dir.glob("*.safetensors")) == (rank == 0)

    ref = _CountingRef()
    result = compute_ref_logprobs_for_dataset(dataset, ref, _collate, _COLUMNS, **call)
    expected_seen = (
        list(range(n_rows))
        if shard is False
        else list(local_shard(range(n_rows), rank=rank, world_size=world_size))
    )
    assert ref.examples_seen == expected_seen
    _assert_columns(result, n_rows)
    _assert_no_desync(world_size)


def _assert_divergent_use_cache(rank: int, world_size: int, cache_root: str) -> None:
    from opake.alignment.dpo.reference import compute_ref_logprobs_for_dataset

    dataset = _make_dataset(6)
    ref = _CountingRef()
    result = compute_ref_logprobs_for_dataset(
        dataset,
        ref,
        _collate,
        _COLUMNS,
        cache_identity={"kind": "divergent-use-cache"},
        batch_size=2,
        cache_dir=str(Path(cache_root) / "use-cache"),
        use_cache=(rank == 0),
    )
    _assert_columns(result, 6)
    _assert_no_desync(world_size)


def _assert_dataset_size_mismatch(rank: int, world_size: int) -> None:
    from opake.alignment.dpo.reference import compute_ref_logprobs_for_dataset

    with pytest.raises(RuntimeError, match="dataset size mismatch across ranks"):
        compute_ref_logprobs_for_dataset(
            _make_dataset(4 if rank == 0 else 3),
            _CountingRef(),
            _collate,
            _COLUMNS,
            cache_identity={"kind": "size-mismatch"},
            batch_size=2,
            use_cache=False,
        )
    _assert_no_desync(world_size)


def _assert_dataset_fingerprint_mismatch(rank: int, world_size: int) -> None:
    from opake.alignment.dpo.reference import compute_ref_logprobs_for_dataset

    with pytest.raises(
        OperationError, match="dataset fingerprint mismatch across ranks"
    ):
        compute_ref_logprobs_for_dataset(
            _make_dataset(4, start=rank * 10),
            _CountingRef(),
            _collate,
            _COLUMNS,
            cache_identity={"kind": "fingerprint-mismatch"},
            batch_size=2,
            use_cache=False,
        )
    _assert_no_desync(world_size)


def _worker_precompute_contract(
    rank: int, world_size: int, port: int, cache_root: str
) -> None:
    """Run all cross-rank precompute scenarios within one process group."""
    _setup_gloo(rank, world_size, port)
    try:
        _assert_sharded_output(
            rank,
            world_size,
            cache_root,
            n_rows=7,
            start=0,
            dtype=torch.float32,
            identity="sharded-remainder",
        )
        _assert_sharded_output(
            rank,
            world_size,
            cache_root,
            n_rows=2,
            start=7,
            dtype=torch.bfloat16,
            identity="empty-shard-bf16",
        )
        _assert_shared_cache_hit(rank, world_size, cache_root)
        _assert_divergent_cache_recomputes(rank, world_size, cache_root, shard=None)
        _assert_divergent_cache_recomputes(rank, world_size, cache_root, shard=False)
        _assert_divergent_use_cache(rank, world_size, cache_root)
        _assert_dataset_size_mismatch(rank, world_size)
        _assert_dataset_fingerprint_mismatch(rank, world_size)
    finally:
        _cleanup_ddp()
