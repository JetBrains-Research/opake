"""Cached, model-agnostic reference-logprob precomputation.

:func:`compute_ref_logprobs_for_dataset` runs a single one-shot forward pass
over a :class:`datasets.Dataset` to materialise per-example reference logprobs
(e.g. ``ref_chosen_logps`` / ``ref_rejected_logps`` for DPO) and attaches them
to the dataset as new columns. When enabled, results are persisted to a content-addressed ``.safetensors``
cache so the (expensive) reference forward runs at most once per
``(dataset, cache_identity, output_columns)`` triple. Cache archives contain
private per-example values, so cache directories and files are owner-only.
Callers remove a selected cache directory when its contents are no longer
needed.

**Outside vmap only.** This helper iterates a ``DataLoader``, runs a forward
pass under ``torch.no_grad()``, gathers across ranks, and writes a file. It must
be called *before* the per-example gradient loop, never inside ``vmap``/``grad``.

**Mechanism-agnostic by construction.** ``ref`` is a plain callable
``dict[str, Tensor] -> dict[str, Tensor]``. A :class:`PreTrainedModel` is wrapped
into such a callable by the caller (the trainer / example), which keeps this
module independent of any particular model class and trivially unit-testable
with a synthetic ``ref``.

**Cache fingerprint.** The cache filename is the SHA-256 of a versioned,
canonical encoding of the prepared dataset fingerprint, caller-supplied
``cache_identity``, and requested output columns. In the DPO trainer the
identity is the effective reference-model state, matching TRL's cache model.
Unsupported or non-deterministic values are rejected instead of falling back to
``repr``.

**On-disk format.** ``safetensors`` is the project standard (also used by
:class:`opaque.transformers.trainer.Trainer` for model weights). Native dtype
round-trip means a bf16 reference forward stores bf16 on disk with no precision
loss and no implicit conversion. The HF ``Dataset.add_column`` boundary still
demotes to a Python ``float`` list because PyArrow has no bf16 column type, but
that demotion is now explicit at the storage boundary, not hidden inside the
cache writer.

**Cross-rank handling.** When sharding is active each rank runs the reference
over its own :func:`local_shard` and the per-column shards are concatenated back
in rank order through one object collective. ``local_shard`` hands out
contiguous slices, so that reproduces the original dataset order. Shards may be
uneven (the last rank takes the remainder) and a rank whose shard is empty
contributes no rows while still joining the collective.

Every rank issues the same collectives regardless of its arguments: the cache
decision is reduced before it is acted on, so a group cannot split into ranks
that return early and ranks that wait in the gather. The group reuses the cache
only when every rank holds the archive, which means the node-local default
``cache_dir`` yields no reuse across nodes — point it at shared storage.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import load_file as _safetensors_load
from safetensors.torch import save_file as _safetensors_save
from torch.utils.data import DataLoader

from opaque.api.engine.distributed._state import (
    assert_scalar_equal,
    assert_string_equal,
    gather_pytree,
    reduce_scalar,
)
from opaque.distributed import (
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    local_shard,
    wait_for_everyone,
)
from opaque.exceptions import ConfigurationError, InputTypeError, OperationError
from opaque.serialization import from_state_dict, state_dict

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["compute_ref_logprobs_for_dataset"]


# Default cache root, lazily created when caching is enabled. A subdirectory of
# the system temp dir keeps the cache off the dataset's own storage and lets
# the OS reclaim it.
_DEFAULT_CACHE_SUBDIR = "opaque_ref_cache"
_CACHE_EXT = ".safetensors"
_CACHE_FINGERPRINT_VERSION = 2


def _canonical_identity_value(value: Any, path: str = "cache_identity") -> Any:
    """Validate and normalize a cache identity into canonical JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigurationError(*(f"{path} contains a non-finite float",))
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise InputTypeError(
                    *(f"{path} mapping keys must be strings, got {type(key)!r}",)
                )
            normalized[key] = _canonical_identity_value(item, f"{path}.{key}")
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _canonical_identity_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise InputTypeError(
        *(
            f"{path} contains unsupported value {type(value)!r}; use only JSON-like "
            "scalars, string-keyed mappings, and sequences",
        )
    )


def _cache_fingerprint(
    dataset: Any,
    cache_identity: Any,
    output_columns: Sequence[str],
) -> str:
    """SHA-256 over canonical dataset, reference, and output identities.

    ``datasets.Dataset._fingerprint`` is required because an object ``repr`` can
    include transient process identity and silently defeat cache correctness.
    """
    dataset_id = _dataset_fingerprint(dataset)
    if dataset_id is None:
        raise ConfigurationError(
            *(
                "dataset must expose a deterministic `_fingerprint` for reference caching",
            )
        )
    payload = {
        "version": _CACHE_FINGERPRINT_VERSION,
        "dataset": str(dataset_id),
        "identity": _canonical_identity_value(cache_identity),
        "output_columns": list(output_columns),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _dataset_fingerprint(dataset: Any) -> str | None:
    """Return the dataset's deterministic identity when it is available."""
    dataset_id = getattr(dataset, "_fingerprint", None)
    return None if dataset_id is None else str(dataset_id)


def _cache_path(
    cache_dir: str | None,
    fingerprint: str,
) -> str:
    """Resolve the ``.safetensors`` path, defaulting to ``<tmp>/opaque_ref_cache``."""
    if cache_dir is None:
        cache_dir = str(Path(tempfile.gettempdir()) / _DEFAULT_CACHE_SUBDIR)
    return str(Path(cache_dir) / f"{fingerprint}{_CACHE_EXT}")


def _load_cache(
    path: str,
    output_columns: Sequence[str],
    *,
    expected_rows: int,
) -> dict[str, torch.Tensor] | None:
    """Load ``output_columns`` from a ``.safetensors`` cache, or ``None`` on miss.

    A miss is any of: file absent, a requested column absent from the archive,
    a column whose length does not match ``expected_rows``, or a load error
    (all treated as a stale/corrupt cache that should be recomputed rather than
    crash the run).
    """
    if not Path(path).exists():
        return None
    try:
        flat = _safetensors_load(path)
    except Exception:
        return None
    if any(name not in flat for name in output_columns):
        return None
    if any(flat[name].shape[:1] != (expected_rows,) for name in output_columns):
        return None
    # Reconstruct the column dict via the serialization template so the load
    # path mirrors the save path (state_dict round-trip). The template
    # shape is taken from the archived tensors themselves (the per-example
    # count is not known statically), so the round-trip validates dtype +
    # structure without an artificial length assumption.
    template = {name: torch.empty_like(flat[name]) for name in output_columns}
    try:
        restored = from_state_dict(template, flat)
    except Exception:
        return None
    if any(name not in restored for name in output_columns):
        return None
    return {name: restored[name] for name in output_columns}


def _secure_cache_path(path: str) -> None:
    """Create the cache root and restrict any existing archive to its owner."""
    cache_dir = Path(path).parent
    try:
        cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        cache_dir.chmod(0o700)
        cache_file = Path(path)
        if cache_file.exists():
            cache_file.chmod(0o600)
    except PermissionError as error:
        raise PermissionError(  # noqa: TRY003 - preserve standard Python error contract
            f"cannot secure reference-logprob cache directory {cache_dir}; "
            "pass a private writable cache_dir or set use_cache=False"
        ) from error


def _save_cache(
    path: str,
    columns: dict[str, torch.Tensor],
) -> None:
    """Flatten ``columns`` through ``state_dict`` and write the safetensors cache.

    The parent directory and archive are owner-only. ``state_dict`` flattens
    the column dict into a flat ``str -> torch.Tensor`` mapping; the values are
    written directly via :func:`safetensors.torch.save_file` with no dtype
    conversion (native bf16/fp16/fp32 round-trip).
    """
    _secure_cache_path(path)
    flat = state_dict(columns)
    tensors: dict[str, torch.Tensor] = {}
    for key, value in flat.items():
        if isinstance(value, torch.Tensor):
            tensors[key] = value.detach().cpu().contiguous()
        else:
            tensors[key] = torch.as_tensor(value).cpu().contiguous()
    _safetensors_save(tensors, path)
    Path(path).chmod(0o600)


def _resolve_sharding(shard: bool | None) -> bool:
    """Decide whether this call shards, and reject an impossible request."""
    if shard is None:
        return is_distributed()
    if shard and not is_distributed():
        raise OperationError(
            *(
                "shard=True requires an initialised process group; call "
                "torch.distributed.init_process_group first, or pass shard=None "
                "to shard only when one is live.",
            )
        )
    return shard


def _cache_hit_on_every_rank(local_hit: bool) -> bool:
    """Reduce a rank-local cache hit to a decision the whole group shares.

    The default cache directory is node-local, so on a multi-node run some
    ranks can hit while others miss.
    """
    if not is_distributed():
        return local_hit
    return bool(reduce_scalar(int(local_hit), op="min"))


def _reference_forward(
    dataset: Any,
    ref: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
    collator: Callable[[list[dict]], dict[str, torch.Tensor]],
    columns: tuple[str, ...],
    *,
    batch_size: int,
    rank: int,
    world_size: int,
) -> dict[str, torch.Tensor] | None:
    """Run ``ref`` over this rank's shard, one ``(n_local,)`` tensor per column.

    Returns ``None`` when the local shard is empty, so the caller hands the
    gather an absent payload rather than a dtype of its own choosing.
    """
    shard = (
        local_shard(dataset, rank=rank, world_size=world_size)
        if world_size > 1
        else dataset
    )
    loader = DataLoader(shard, batch_size=batch_size, collate_fn=collator)
    parts: dict[str, list[torch.Tensor]] = {name: [] for name in columns}
    with torch.no_grad():
        for batch in loader:
            out = ref(batch)
            for name in columns:
                value = out[name]
                if not isinstance(value, torch.Tensor):
                    value = torch.as_tensor(value)
                parts[name].append(value.detach())
    if all(not part for part in parts.values()):
        return None
    return {name: torch.cat(parts[name], dim=0) for name in columns}


def _gather_columns(
    local: dict[str, torch.Tensor] | None,
    columns: tuple[str, ...],
    *,
    expected_rows: int,
    sharded: bool,
) -> dict[str, torch.Tensor]:
    """Concatenate per-rank column shards back into dataset order.

    One object collective covers every column, and ``gather_pytree`` orders the
    contributions by rank; combined with the contiguous slices ``local_shard``
    hands out, that reproduces the original dataset order.
    """
    gathered = gather_pytree(local, dim=0) if sharded else local
    if gathered is None:
        # No rank produced any rows, so there is no dtype to preserve.
        gathered = {name: torch.empty((0,), dtype=torch.float32) for name in columns}
    for name in columns:
        rows = int(gathered[name].shape[0])
        if rows != expected_rows:
            raise OperationError(
                *(
                    f"reference column {name!r} yielded {rows} values for a dataset "
                    f"of {expected_rows} examples; either `ref` did not return one "
                    "value per row, or the ranks disagree on the dataset",
                )
            )
    return gathered


def compute_ref_logprobs_for_dataset(
    dataset: Any,
    ref: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
    collator: Callable[[list[dict]], dict[str, torch.Tensor]],
    output_columns: Sequence[str],
    *,
    cache_identity: Any,
    batch_size: int = 8,
    cache_dir: str | None = None,
    use_cache: bool = True,
    shard: bool | None = None,
) -> Any:
    """Compute per-example reference logprobs, optionally using a cache.

    Runs a single pass over ``dataset`` calling ``ref(batch)`` for each
    collated batch, collecting one ``(B,)`` tensor per name in
    ``output_columns``, and attaches the concatenated per-example results to
    ``dataset`` as new columns. The forward runs under ``torch.no_grad()``.
    When ``use_cache`` is true, results are cached to a content-addressed
    ``.safetensors`` file keyed on ``(dataset._fingerprint, cache_identity,
    output_columns)``. The cache directory and archive are owner-only because
    they contain private per-example values. The caller is responsible for
    removing selected cache directories when their contents are no longer
    needed.

    Under a live process group each rank runs the reference over its own shard
    and the results are gathered back into dataset order, so the returned
    dataset is the same on every rank and matches a single-process run. Pass
    the whole dataset on every rank; shard for training afterwards.

    **Outside vmap only** — see the module docstring.

    Args:
        dataset: A :class:`datasets.Dataset`, identical on every rank. Iterated
            via a ``DataLoader`` with ``collator`` — over this rank's shard when
            a process group is live — and returned with ``output_columns`` added
            through :meth:`datasets.Dataset.add_column`.
        ref: A callable mapping a collated batch ``dict[str, Tensor]`` to a dict
            containing one ``(B,)`` tensor per name in ``output_columns``. A
            :class:`PreTrainedModel` is wrapped into such a callable by the
            caller, keeping this mechanism-agnostic.
        collator: ``Callable[[list[dict]], dict[str, Tensor]]`` producing a
            batched dict from a list of raw rows (the same collator later used
            for training).
        output_columns: The keys ``ref`` returns, e.g.
            ``("ref_chosen_logps", "ref_rejected_logps")``. Also folded into the
            cache fingerprint.
        cache_identity: Deterministic JSON-like identity for ``ref``. The DPO
            trainer supplies its effective reference-model state; dataset
            preparation changes are represented by ``dataset._fingerprint``.
            Mapping order does not affect the fingerprint. Unsupported values
            raise ``TypeError``.
        batch_size: ``DataLoader`` batch size for the forward pass. Default 8.
        cache_dir: Directory for the ``.safetensors`` cache. Defaults to
            ``<tempdir>/opaque_ref_cache`` (created on first miss).
        use_cache: Whether to read or write the persistent cache. Set to
            ``False`` when results cannot be reused, such as TR-DPO seed
            values that are overwritten before each training and evaluation
            step. Must agree across ranks.
        shard: Whether each rank scores only its own slice of ``dataset``.
            ``None`` (default) shards when a process group is live. ``False``
            makes every rank score the whole dataset. ``True`` requires an
            initialised process group and raises without one. Must agree
            across ranks.

    Returns:
        ``dataset`` with one new column per name in ``output_columns``, each of
        length ``len(dataset)``.
    """
    columns = tuple(output_columns)
    n_examples = len(dataset)
    sharded = _resolve_sharding(shard)
    if sharded:
        # Sharding is only well defined when every rank holds the same dataset.
        assert_scalar_equal(n_examples, name="reference precompute dataset size")
        dataset_id = _dataset_fingerprint(dataset)
        assert_string_equal(dataset_id, name="reference precompute dataset fingerprint")
        if dataset_id is None:
            raise ConfigurationError(
                *(
                    "dataset must expose a deterministic `_fingerprint` for "
                    "distributed reference precomputation",
                )
            )

    path: str | None = None
    cached: dict[str, torch.Tensor] | None = None
    if use_cache:
        fingerprint = _cache_fingerprint(dataset, cache_identity, columns)
        path = _cache_path(cache_dir, fingerprint)
        _secure_cache_path(path)
        cached = _load_cache(path, columns, expected_rows=n_examples)

    # Reduced unconditionally: the collective sequence must not depend on the
    # rank-local ``use_cache``.
    if _cache_hit_on_every_rank(cached is not None):
        # HIT on every rank: attach cached columns WITHOUT calling ``ref``.
        # PyArrow has no bf16 column type, so demote to a Python ``float``
        # list at the storage boundary (explicit, single conversion).
        assert cached is not None
        result = dataset
        for name in columns:
            result = result.add_column(name, cached[name].float().tolist())
        return result

    # MISS: run the reference forward once over this rank's shard.
    local = _reference_forward(
        dataset,
        ref,
        collator,
        columns,
        batch_size=batch_size,
        rank=get_rank() if sharded else 0,
        world_size=get_world_size() if sharded else 1,
    )
    gathered = _gather_columns(
        local, columns, expected_rows=n_examples, sharded=sharded
    )

    # Only the main process writes the cache; all ranks sync before returning.
    if use_cache and is_main_process():
        cpu_columns = {
            name: gathered[name].detach().cpu().contiguous() for name in columns
        }
        assert path is not None
        _save_cache(path, cpu_columns)
    wait_for_everyone()

    result = dataset
    for name in columns:
        # Same explicit fp32 demotion at the HF/PyArrow boundary as on the
        # HIT path above. Native dtype is preserved on disk; only the column
        # exposed to ``datasets`` is downcast.
        values = gathered[name].detach().cpu().float().tolist()
        result = result.add_column(name, values)
    return result
