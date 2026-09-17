"""Block and total k-out-of-t sampler contracts."""

from __future__ import annotations

from collections import Counter
from itertools import chain

import pytest

from opake.dpsgd.sampling import KOutOfTSampler
from opake.exceptions import ConfigurationError
from opake.random import key
from opake.serialization import from_state_dict, state_dict


def test_total_allocation_selects_each_record_exactly_k_times():
    sampler = KOutOfTSampler(
        list(range(100)),
        k=3,
        t=10,
        allocation="total",
        key=key(7),
    )
    counts = Counter(index for batch in sampler for index in batch)
    assert set(counts.values()) == {3}


def test_block_allocation_partitions_every_block():
    sampler = KOutOfTSampler(
        list(range(100)),
        k=4,
        t=19,
        allocation="block",
        key=key(7),
    )
    batches = list(sampler)
    offset = 0

    assert sampler.block_sizes == (4, 5, 5, 5)
    for block_size in sampler.block_sizes:
        assert sorted(chain(*batches[offset : offset + block_size])) == list(range(100))
        offset += block_size


def test_block_allocation_draws_each_block_independently():
    sampler = KOutOfTSampler(
        list(range(100)),
        k=4,
        t=20,
        allocation="block",
        key=key(7),
    )
    batches = list(sampler)
    epochs = [batches[offset : offset + 5] for offset in range(0, 20, 5)]

    assert any(epoch != epochs[0] for epoch in epochs[1:])


def test_allocation_modes_use_distinct_stream_domains():
    kwargs = {"data_source": list(range(100)), "k": 4, "t": 20, "key": key(7)}

    block = list(KOutOfTSampler(allocation="block", **kwargs))
    total = list(KOutOfTSampler(allocation="total", **kwargs))

    assert block != total


@pytest.mark.parametrize("allocation", ["block", "total"])
@pytest.mark.parametrize(
    ("k", "t", "consumed"), [(2, 8, 0), (2, 8, 5), (2, 8, 8), (4, 19, 7)]
)
def test_stream_is_reproducible_and_resumable(allocation: str, k, t, consumed):
    def make(seed: int):
        return KOutOfTSampler(
            list(range(40)),
            k=k,
            t=t,
            allocation=allocation,  # type: ignore[arg-type]
            key=key(seed),
        )

    expected = list(make(3))
    partial = make(3)
    iterator = iter(partial)
    for _ in range(consumed):
        next(iterator)
    restored = from_state_dict(make(99), state_dict(partial))

    assert restored.consumed == consumed
    assert list(restored) == expected[consumed:]


def test_expected_batch_size_and_block_sizes():
    block = KOutOfTSampler(
        list(range(100)),
        k=4,
        t=19,
        allocation="block",
        key=key(0),
    )
    total = KOutOfTSampler(
        list(range(100)),
        k=4,
        t=19,
        allocation="total",
        key=key(0),
    )

    assert block.expected_batch_size == pytest.approx(100 * 4 / 19)
    assert total.expected_batch_size == pytest.approx(100 * 4 / 19)
    assert block.block_sizes == (4, 5, 5, 5)
    assert total.block_sizes is None


def test_validation():
    with pytest.raises(ValueError, match="k"):
        KOutOfTSampler([1], k=3, t=2, allocation="total", key=key(0))
    with pytest.raises(ValueError, match="t"):
        KOutOfTSampler([1], k=1, t=0, allocation="total", key=key(0))
    with pytest.raises(ValueError, match="allocation"):
        KOutOfTSampler([1], k=1, t=1, allocation="bad", key=key(0))  # type: ignore[arg-type]


@pytest.mark.parametrize("allocation", ["block", "total"])
@pytest.mark.parametrize("field", ["k", "t", "allocation", "num_samples"])
def test_restore_rejects_changed_schedule(allocation, field):
    original = KOutOfTSampler(range(40), k=2, t=8, allocation=allocation, key=key(3))
    next(iter(original))
    snapshot = state_dict(original)
    changed = {
        "k": 3 if field == "k" else 2,
        "t": 9 if field == "t" else 8,
        "allocation": (
            {"block": "total", "total": "block"}[allocation]
            if field == "allocation"
            else allocation
        ),
    }
    template = KOutOfTSampler(
        range(41 if field == "num_samples" else 40), key=key(99), **changed
    )

    with pytest.raises(ConfigurationError, match=field):
        from_state_dict(template, snapshot)

    assert state_dict(original) == snapshot
    assert template.consumed == 0


@pytest.mark.parametrize("field", ["k", "t"])
@pytest.mark.parametrize("value", [True, 1.0, 1.9, "1"])
def test_restore_rejects_noninteger_schedule_fields(field, value):
    template = KOutOfTSampler(range(4), k=1, t=1, allocation="block", key=key(0))
    snapshot = state_dict(template)
    snapshot[field] = value

    with pytest.raises(ConfigurationError, match=field):
        from_state_dict(template, snapshot)
