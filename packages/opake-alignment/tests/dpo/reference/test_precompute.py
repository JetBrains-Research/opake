"""Tests for :func:`compute_ref_logprobs_for_dataset` (plan §7.8, §11.9).

Covers the cache round-trip contract (§11.9): the first call runs the
reference forward (call counter advances) and attaches the requested columns;
a second call on the *same* dataset + ``cache_identity`` + ``output_columns`` is a
cache HIT that skips ``ref`` entirely (a fresh counter stays at 0) and returns
identical column values. Also covers fingerprint sensitivity to ``cache_identity``,
column presence / length / values,
and import from the implementation paths.
"""

from __future__ import annotations

import stat

import datasets
import pytest
import torch

from opake.api.alignment.dpo.reference import _precompute
from opake.api.alignment.dpo.reference._precompute import (
    compute_ref_logprobs_for_dataset,
)

OUTPUT_COLUMNS = ("ref_chosen_logps", "ref_rejected_logps")


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_dataset() -> datasets.Dataset:
    """A tiny preference-style dataset keyed by a per-row integer index."""
    return datasets.Dataset.from_dict(
        {
            "idx": [0, 1, 2, 3, 4],
            "prompt": ["p0", "p1", "p2", "p3", "p4"],
        }
    )


def _collator(rows: list[dict]) -> dict[str, torch.Tensor]:
    """Batch the raw rows into a dict carrying the per-row index tensor."""
    return {"idx": torch.tensor([r["idx"] for r in rows], dtype=torch.long)}


class _CountingRef:
    """A deterministic ``ref`` callable that records how often it is invoked.

    Per the precompute contract, ``ref(batch) -> dict[str, (B,) tensor]``. The
    returned logps are a fixed function of the row index so the test can assert
    exact values, and ``calls`` lets the cache-HIT test prove ``ref`` was never
    invoked on the second pass.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.examples_seen = 0

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.calls += 1
        idx = batch["idx"].to(torch.float32)
        self.examples_seen += int(idx.numel())
        # Deterministic per-example values derived from the row index.
        return {
            "ref_chosen_logps": idx * 1.0,
            "ref_rejected_logps": idx * -2.0,
        }


def _expected(name: str, indices: list[int]) -> list[float]:
    """Mirror ``_CountingRef`` so tests assert against hand-computed values."""
    if name == "ref_chosen_logps":
        return [float(i) * 1.0 for i in indices]
    if name == "ref_rejected_logps":
        return [float(i) * -2.0 for i in indices]
    raise KeyError(name)


# ---------------------------------------------------------------------------
# Cache round-trip (§11.9)
# ---------------------------------------------------------------------------


def test_cache_round_trip_skips_ref_on_hit(tmp_path) -> None:
    """First call computes; second call (fresh ref) is a HIT and skips ref."""
    dataset = _make_dataset()
    cache_dir = str(tmp_path / "cache")
    cache_identity = {"kind": "dpo", "model": {"id": "model-v1"}}

    ref1 = _CountingRef()
    result1 = compute_ref_logprobs_for_dataset(
        dataset,
        ref1,
        _collator,
        OUTPUT_COLUMNS,
        batch_size=2,
        cache_identity=cache_identity,
        cache_dir=cache_dir,
    )
    # MISS: ref ran over every example.
    assert ref1.calls > 0, "first call should invoke ref (cache miss)"
    assert ref1.examples_seen == len(dataset)
    for name in OUTPUT_COLUMNS:
        assert name in result1.column_names

    # Second call: a FRESH ref whose counter starts at 0. Same dataset, key,
    # and output columns => cache HIT => ref must NOT be called.
    ref2 = _CountingRef()
    result2 = compute_ref_logprobs_for_dataset(
        dataset,
        ref2,
        _collator,
        OUTPUT_COLUMNS,
        batch_size=2,
        cache_identity={"model": {"id": "model-v1"}, "kind": "dpo"},
        cache_dir=cache_dir,
    )
    assert ref2.calls == 0, "second call should be a cache HIT and skip ref"

    # Columns identical between the two calls.
    for name in OUTPUT_COLUMNS:
        assert result1[name] == result2[name]


def test_different_cache_identity_recomputes(tmp_path) -> None:
    """A nested identity change is a MISS: ref runs under a new fingerprint."""
    dataset = _make_dataset()
    cache_dir = str(tmp_path / "cache")

    ref_a = _CountingRef()
    compute_ref_logprobs_for_dataset(
        dataset,
        ref_a,
        _collator,
        OUTPUT_COLUMNS,
        batch_size=2,
        cache_identity={"kind": "dpo", "model": {"id": "model-A"}},
        cache_dir=cache_dir,
    )
    assert ref_a.calls > 0

    # Nested identity change => distinct fingerprint => recompute.
    ref_b = _CountingRef()
    compute_ref_logprobs_for_dataset(
        dataset,
        ref_b,
        _collator,
        OUTPUT_COLUMNS,
        batch_size=2,
        cache_identity={"kind": "dpo", "model": {"id": "model-B"}},
        cache_dir=cache_dir,
    )
    assert ref_b.calls > 0, "different cache_identity should force recompute (MISS)"


def test_different_output_columns_recomputes(tmp_path) -> None:
    """output_columns participate in the fingerprint: a different set => MISS."""
    dataset = _make_dataset()
    cache_dir = str(tmp_path / "cache")
    cache_identity = {"kind": "dpo", "model": {"id": "model-v1"}}

    ref_full = _CountingRef()
    compute_ref_logprobs_for_dataset(
        dataset,
        ref_full,
        _collator,
        OUTPUT_COLUMNS,
        batch_size=2,
        cache_identity=cache_identity,
        cache_dir=cache_dir,
    )
    assert ref_full.calls > 0

    # Same dataset + key but only one requested column => different fingerprint.
    ref_single = _CountingRef()
    compute_ref_logprobs_for_dataset(
        dataset,
        ref_single,
        _collator,
        ("ref_chosen_logps",),
        batch_size=2,
        cache_identity=cache_identity,
        cache_dir=cache_dir,
    )
    assert ref_single.calls > 0, "different output_columns should force a MISS"


# ---------------------------------------------------------------------------
# Output columns: presence, length, values
# ---------------------------------------------------------------------------


def test_output_columns_present_with_correct_length_and_values(tmp_path) -> None:
    """Added columns exist, have len(dataset), and match the deterministic ref."""
    dataset = _make_dataset()
    cache_dir = str(tmp_path / "cache")
    indices = list(dataset["idx"])

    result = compute_ref_logprobs_for_dataset(
        dataset,
        _CountingRef(),
        _collator,
        OUTPUT_COLUMNS,
        batch_size=2,
        cache_identity={"purpose": "values"},
        cache_dir=cache_dir,
    )

    for name in OUTPUT_COLUMNS:
        assert name in result.column_names
        assert len(result[name]) == len(dataset)
        got = [float(v) for v in result[name]]
        assert got == pytest.approx(_expected(name, indices))


def test_cached_values_match_computed_values(tmp_path) -> None:
    """Values restored from the safetensors cache equal the freshly computed ones."""
    dataset = _make_dataset()
    cache_dir = str(tmp_path / "cache")
    indices = list(dataset["idx"])

    compute_ref_logprobs_for_dataset(
        dataset,
        _CountingRef(),
        _collator,
        OUTPUT_COLUMNS,
        batch_size=3,
        cache_identity={"purpose": "round-trip"},
        cache_dir=cache_dir,
    )
    # Second call: HIT, values come from the cache file.
    cached_result = compute_ref_logprobs_for_dataset(
        dataset,
        _CountingRef(),
        _collator,
        OUTPUT_COLUMNS,
        batch_size=3,
        cache_identity={"purpose": "round-trip"},
        cache_dir=cache_dir,
    )
    for name in OUTPUT_COLUMNS:
        got = [float(v) for v in cached_result[name]]
        assert got == pytest.approx(_expected(name, indices))


def test_bfloat16_ref_logps_round_trip(tmp_path) -> None:
    """A bf16 ``ref`` callable must not crash the cache writer (regression).

    The original ``.npz`` cache called ``tensor.numpy()`` on the ref-logp
    tensors before saving, which raises ``TypeError: Got unsupported ScalarType
    BFloat16`` because numpy cannot serialize bf16. The current safetensors
    writer round-trips bf16 natively; this test pins the contract.
    """
    dataset = _make_dataset()
    cache_dir = str(tmp_path / "cache_bf16")

    class _Bf16Ref:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, batch):
            self.calls += 1
            idx = batch["idx"].to(torch.bfloat16)
            return {
                "ref_chosen_logps": idx * 1.0,
                "ref_rejected_logps": idx * -2.0,
            }

    result = compute_ref_logprobs_for_dataset(
        dataset,
        _Bf16Ref(),
        _collator,
        OUTPUT_COLUMNS,
        batch_size=2,
        cache_identity={"purpose": "bf16-round-trip"},
        cache_dir=cache_dir,
    )

    indices = list(dataset["idx"])
    for name in OUTPUT_COLUMNS:
        got = [float(v) for v in result[name]]
        # bf16 has ~3 decimal digits; tolerate that on the HF-column readback.
        assert got == pytest.approx(_expected(name, indices), abs=1e-2)


def test_missing_cache_dir_is_created(tmp_path) -> None:
    """A new cache directory and archive are owner-only."""
    dataset = _make_dataset()
    cache_dir = tmp_path / "nested" / "does_not_exist_yet"
    assert not cache_dir.exists()

    compute_ref_logprobs_for_dataset(
        dataset,
        _CountingRef(),
        _collator,
        OUTPUT_COLUMNS,
        batch_size=2,
        cache_identity={"purpose": "mkdir"},
        cache_dir=str(cache_dir),
    )
    assert cache_dir.exists(), "cache_dir should be created on first write"
    cache_files = list(cache_dir.glob("*.safetensors"))
    assert cache_files, "expected a .safetensors cache file to be written"
    assert stat.S_IMODE(cache_dir.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in cache_files)


def test_cache_hit_restores_private_permissions(tmp_path) -> None:
    """Existing cache artifacts are made owner-only before a hit is loaded."""
    dataset = _make_dataset()
    cache_dir = tmp_path / "cache"
    cache_identity = {"purpose": "permissions"}

    compute_ref_logprobs_for_dataset(
        dataset,
        _CountingRef(),
        _collator,
        OUTPUT_COLUMNS,
        batch_size=2,
        cache_identity=cache_identity,
        cache_dir=str(cache_dir),
    )
    cache_file = next(cache_dir.glob("*.safetensors"))
    cache_dir.chmod(0o755)
    cache_file.chmod(0o644)

    ref = _CountingRef()
    compute_ref_logprobs_for_dataset(
        dataset,
        ref,
        _collator,
        OUTPUT_COLUMNS,
        batch_size=2,
        cache_identity=cache_identity,
        cache_dir=str(cache_dir),
    )

    assert ref.calls == 0
    assert stat.S_IMODE(cache_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(cache_file.stat().st_mode) == 0o600


def test_cache_permission_error_explains_remediation(tmp_path, monkeypatch) -> None:
    """Permission failures identify the safe cache configuration options."""

    def raise_permission_error(_path, _mode):
        raise PermissionError("denied")

    monkeypatch.setattr(_precompute.Path, "chmod", raise_permission_error)

    with pytest.raises(PermissionError, match="pass a private writable cache_dir"):
        _precompute._secure_cache_path(str(tmp_path / "cache" / "logps.safetensors"))


def test_disabled_cache_does_not_create_artifacts(tmp_path) -> None:
    """Non-reusable reference values are computed without touching disk."""
    dataset = _make_dataset()
    cache_dir = tmp_path / "cache"
    ref = _CountingRef()

    result = compute_ref_logprobs_for_dataset(
        dataset,
        ref,
        _collator,
        OUTPUT_COLUMNS,
        batch_size=2,
        cache_identity={"purpose": "non-persistent"},
        cache_dir=str(cache_dir),
        use_cache=False,
    )

    assert ref.examples_seen == len(dataset)
    assert not cache_dir.exists()
    for name in OUTPUT_COLUMNS:
        assert result[name] == pytest.approx(_expected(name, list(dataset["idx"])))


def test_unsupported_cache_identity_value_raises(tmp_path) -> None:
    dataset = _make_dataset()

    with pytest.raises(TypeError, match="unsupported value"):
        compute_ref_logprobs_for_dataset(
            dataset,
            _CountingRef(),
            _collator,
            OUTPUT_COLUMNS,
            cache_identity={"model": object()},
            cache_dir=str(tmp_path),
        )


# ---------------------------------------------------------------------------
# Stale-cache and per-example contract
# ---------------------------------------------------------------------------


def test_cache_with_wrong_row_count_is_recomputed(tmp_path) -> None:
    """An archive whose length disagrees with the dataset is a miss, not a crash."""
    dataset = _make_dataset()
    cache_dir = tmp_path / "cache"
    cache_identity = {"kind": "dpo"}

    compute_ref_logprobs_for_dataset(
        dataset,
        _CountingRef(),
        _collator,
        OUTPUT_COLUMNS,
        batch_size=2,
        cache_identity=cache_identity,
        cache_dir=str(cache_dir),
    )

    archive = next(cache_dir.glob("*.safetensors"))
    doubled = {
        name: torch.tensor(_expected(name, list(dataset["idx"]) * 2))
        for name in OUTPUT_COLUMNS
    }
    _precompute._save_cache(str(archive), doubled)

    ref = _CountingRef()
    result = compute_ref_logprobs_for_dataset(
        dataset,
        ref,
        _collator,
        OUTPUT_COLUMNS,
        batch_size=2,
        cache_identity=cache_identity,
        cache_dir=str(cache_dir),
    )

    assert ref.examples_seen == len(dataset)
    for name in OUTPUT_COLUMNS:
        assert result[name] == pytest.approx(_expected(name, list(dataset["idx"])))


def test_ref_returning_wrong_number_of_values_raises(tmp_path) -> None:
    """A ``ref`` that drops examples fails loudly instead of poisoning the cache."""

    def _short_ref(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        idx = batch["idx"].to(torch.float32)[:1]
        return {"ref_chosen_logps": idx, "ref_rejected_logps": idx}

    with pytest.raises(RuntimeError, match="did not return one value per row"):
        compute_ref_logprobs_for_dataset(
            _make_dataset(),
            _short_ref,
            _collator,
            OUTPUT_COLUMNS,
            batch_size=2,
            cache_identity={"kind": "short-ref"},
            cache_dir=str(tmp_path / "cache"),
        )


def test_shard_true_without_process_group_raises(tmp_path) -> None:
    """Asking for sharding that cannot happen fails rather than silently not."""
    with pytest.raises(RuntimeError, match="requires an initialised process group"):
        compute_ref_logprobs_for_dataset(
            _make_dataset(),
            _CountingRef(),
            _collator,
            OUTPUT_COLUMNS,
            batch_size=2,
            cache_identity={"kind": "shard-no-group"},
            cache_dir=str(tmp_path / "cache"),
            shard=True,
        )


@pytest.mark.parametrize("shard", [None, False])
def test_single_process_result_is_independent_of_shard(tmp_path, shard) -> None:
    """Without a process group both settings score the whole dataset."""
    dataset = _make_dataset()
    ref = _CountingRef()
    result = compute_ref_logprobs_for_dataset(
        dataset,
        ref,
        _collator,
        OUTPUT_COLUMNS,
        batch_size=2,
        cache_identity={"kind": f"shard-{shard}"},
        cache_dir=str(tmp_path / "cache"),
        shard=shard,
    )

    assert ref.examples_seen == len(dataset)
    for name in OUTPUT_COLUMNS:
        assert result[name] == pytest.approx(_expected(name, list(dataset["idx"])))
