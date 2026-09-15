"""Distributed synchronization helpers for core clipping components.

Implements all-reduce/all-gather patterns for the algorithm-agnostic
fixed clipping state and clipping auxiliary outputs. DP-SGD-specific
adaptive clipping sync lives in :mod:`opaque.dpsgd.clipping.distributed`
and self-registers via :func:`opaque.distributed.register_sync_type`.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch
import torch.distributed as dist

from opaque.api.engine.distributed import is_distributed
from opaque.api.engine.distributed._state import (
    get_world_size,
    reduce_scalar,
    register_sync_type,
)
from opaque.api.engine.pytree import (
    tree_flatten,
    tree_leaves,
    tree_map,
    tree_structure,
    tree_unflatten,
)
from opaque.exceptions import InputTypeError, OperationError

from ._auto import AutoClipState
from ._clipped_fun import ClippedFunAux, FixedClipState
from ._clipped_grad import ClippedGradAux

_MARKER_CLIP_STATES = (FixedClipState, AutoClipState)

__all__ = [
    "sync_aux",
    "sync_clip_state",
    "sync_clipped_fun_aux",
    "sync_clipped_grad_aux",
]


def sync_clip_state(
    state: FixedClipState | AutoClipState,
) -> FixedClipState | AutoClipState:
    """Synchronize a marker clipping state (fixed or AUTO-S).

    Both carry no fields — the threshold travels in
    ``ClippedPytree.max_norm`` and is fixed at construction — so there is
    nothing to reduce and no collective to issue.  Adaptive clipping does
    drift across steps and registers its own handler in
    :mod:`opaque.dpsgd.clipping.distributed`.
    """
    if not is_distributed():
        return state
    if not isinstance(state, _MARKER_CLIP_STATES):
        raise InputTypeError(
            *(
                "Expected a marker clip state "
                f"({' or '.join(t.__name__ for t in _MARKER_CLIP_STATES)}), "
                f"got {type(state).__name__}",
            )
        )

    return state


# Each supported aux family has a complete field schema.  Do not infer this
# from a subclass's dataclass fields: an unregistered extension could otherwise
# make only one rank issue an extra gather.
_AUX_FIELD_SCHEMAS = {
    ClippedFunAux: (
        "values",
        "norms",
        "clipped_norms",
        "value_aux",
        "clipping_rate",
        "batch_size",
        "group_norms",
    ),
    ClippedGradAux: (
        "loss_values",
        "grad_norms",
        "clipped_grad_norms",
        "loss_aux",
        "clipping_rate",
        "batch_size",
        "group_norms",
    ),
}
_SCALAR_AUX_FIELDS = frozenset({"clipping_rate", "batch_size"})


def _split_aux_fields(
    aux: ClippedFunAux | ClippedGradAux,
    schema_type: type[ClippedFunAux] | type[ClippedGradAux],
) -> tuple[dict[str, object], dict[str, object]]:
    """Split a supported auxiliary output using its declared field schema."""
    schema = _AUX_FIELD_SCHEMAS[schema_type]
    actual = tuple(
        field.name
        for field in dataclasses.fields(aux)
        if not field.name.startswith("_")
    )
    if actual != schema:
        unexpected = sorted(set(actual) - set(schema))
        missing = sorted(set(schema) - set(actual))
        details = []
        if unexpected:
            details.append(f"unexpected fields: {unexpected}")
        if missing:
            details.append(f"missing fields: {missing}")
        raise InputTypeError(
            *(
                f"{type(aux).__name__} does not match the {schema_type.__name__} "
                f"synchronization schema ({'; '.join(details)}).",
            )
        )

    tensor_fields = {
        name: getattr(aux, name) for name in schema if name not in _SCALAR_AUX_FIELDS
    }
    scalar_fields = {
        name: getattr(aux, name) for name in schema if name in _SCALAR_AUX_FIELDS
    }
    return tensor_fields, scalar_fields


def _cpu_payload(value: Any) -> Any:
    """Detach tensor leaves to CPU for ``all_gather_object`` pickling."""
    return tree_map(
        lambda leaf: leaf.detach().cpu() if isinstance(leaf, torch.Tensor) else leaf,
        value,
    )


def _infer_device(value: Any) -> torch.device:
    leaves = tree_leaves(value)
    return leaves[0].device if leaves else torch.device("cpu")


def _infer_device_from_fields(tensor_fields: dict[str, object]) -> torch.device:
    """Prefer any local tensor device so empty optional fields stay on-device."""
    for value in tensor_fields.values():
        leaves = tree_leaves(value)
        if leaves:
            return leaves[0].device
    return torch.device("cpu")


def _merge_gathered_values(values: list[Any], device: torch.device) -> Any:
    """Concatenate per-rank aux pytrees; ``None`` ranks contribute nothing.

    Non-``None`` payloads must share an optree structure (same group keys /
    nesting). Empty batches send ``None`` for optional fields such as
    ``group_norms``; that is filtered before flatten/cat/unflatten so the
    collective schedule stays fixed while the merge stays structure-typed.

    Nested ``None`` placeholders live in the treedef rather than the leaf list,
    so unflatten restores them. Every actual leaf must be a tensor: keeping one
    rank's non-tensor leaf would silently discard the other ranks' values.
    """
    present = [v for v in values if v is not None]
    if not present:
        return None

    treedef = tree_structure(present[0])
    for payload in present[1:]:
        other = tree_structure(payload)
        if other != treedef:
            raise InputTypeError(
                *(
                    "Distributed aux gather requires matching pytree structures "
                    f"across non-empty ranks; got {treedef} vs {other}",
                )
            )

    leaf_lists = [tree_flatten(payload)[0] for payload in present]
    if not leaf_lists[0]:
        # Structures made purely of None placeholders carry no leaves.
        return present[0]

    merged_leaves: list[Any] = []
    for i in range(len(leaf_lists[0])):
        column = [leaves[i] for leaves in leaf_lists]
        if not all(isinstance(leaf, torch.Tensor) for leaf in column):
            raise InputTypeError(
                *(
                    "Distributed aux gathering supports tensor leaves only; got "
                    f"{[type(leaf).__name__ for leaf in column]}. Nested None is "
                    "preserved structurally and does not need to be a leaf.",
                )
            )
        merged_leaves.append(torch.cat([leaf.to(device) for leaf in column], dim=0))
    return tree_unflatten(treedef, merged_leaves)


def _gather_aux_fields(tensor_fields: dict[str, object]) -> dict[str, object]:
    """Gather schema tensor fields with one collective per field.

    Field-level ``all_gather_object`` keeps the collective count identical when
    one rank has ``group_norms=None`` (empty batch) and another has a per-group
    dict. Leaf-wise ``tree_map`` cannot bridge that structure mismatch; once
    payloads are gathered, non-``None`` values are merged with
    :func:`~opaque.pytree.tree_flatten` / cat /
    :func:`~opaque.pytree.tree_unflatten`.
    """
    gathered: dict[str, object] = {}
    default_device = _infer_device_from_fields(tensor_fields)
    # Callers derive this declaration order from the registered aux schema.
    for name, local in tensor_fields.items():
        payloads = [None] * get_world_size()
        dist.all_gather_object(payloads, _cpu_payload(local))
        device = _infer_device(local) if tree_leaves(local) else default_device
        gathered[name] = _merge_gathered_values(payloads, device)
    return gathered


def _sync_clipping_rate(
    clipping_rate: float | None,
    norms: torch.Tensor | None,
) -> float | None:
    """Compute global clipping rate as a size-weighted average across ranks.

    Always issues the same two ``reduce_scalar`` collectives. An empty local
    batch contributes weight 0; branching on ``local_n > 0`` previously made
    empty ranks issue one all-reduce while non-empty ranks issued two, which
    desynchronized the process group after a single empty Poisson draw.
    """
    local_presence = int(clipping_rate is not None)
    min_presence = reduce_scalar(local_presence, op="min")
    max_presence = reduce_scalar(local_presence, op="max")
    if min_presence != max_presence:
        raise OperationError(
            *("Clipped auxiliary clipping_rate presence mismatch across ranks.",)
        )
    if not local_presence:
        return None

    local_n = float(norms.numel()) if isinstance(norms, torch.Tensor) else 0.0
    local_rate = float(clipping_rate)
    global_weighted_sum = reduce_scalar(local_rate * local_n, op="sum")
    global_total = reduce_scalar(local_n, op="sum")
    if global_total <= 0.0:
        return 0.0
    return global_weighted_sum / global_total


def _sync_batch_size(batch_size: int) -> int:
    """Sum batch_size across ranks."""
    return int(reduce_scalar(batch_size, op="sum"))


def sync_clipped_fun_aux(aux: ClippedFunAux) -> ClippedFunAux:
    """Synchronize ``ClippedFunAux`` by gathering per-example tensor fields."""
    if not is_distributed():
        return aux

    tensor_fields, scalar_fields = _split_aux_fields(aux, ClippedFunAux)
    gathered = _gather_aux_fields(tensor_fields)

    scalar_fields["clipping_rate"] = _sync_clipping_rate(aux.clipping_rate, aux.norms)
    scalar_fields["batch_size"] = _sync_batch_size(aux.batch_size)

    return type(aux)(**{**gathered, **scalar_fields})


def sync_clipped_grad_aux(aux: ClippedGradAux) -> ClippedGradAux:
    """Synchronize ``ClippedGradAux`` across distributed ranks.

    Handles any subclass of :class:`ClippedGradAux` (e.g. the DP-SGD
    ``AdaptiveClippedGradAux``) — the constructor of ``type(aux)`` is
    used to rebuild the synced instance.
    """
    if not is_distributed():
        return aux

    tensor_fields, scalar_fields = _split_aux_fields(aux, ClippedGradAux)
    gathered = _gather_aux_fields(tensor_fields)

    scalar_fields["clipping_rate"] = _sync_clipping_rate(
        aux.clipping_rate, aux.grad_norms
    )
    scalar_fields["batch_size"] = _sync_batch_size(aux.batch_size)

    return type(aux)(**{**gathered, **scalar_fields})


def sync_aux(
    aux: ClippedFunAux | ClippedGradAux,
) -> ClippedFunAux | ClippedGradAux:
    """Synchronize clipping auxiliary outputs across distributed ranks."""
    if isinstance(aux, ClippedGradAux):
        return sync_clipped_grad_aux(aux)
    if isinstance(aux, ClippedFunAux):
        return sync_clipped_fun_aux(aux)
    raise InputTypeError(*(f"Unsupported aux type for sync_aux: {type(aux)}",))


register_sync_type(FixedClipState, sync_clip_state)
register_sync_type(AutoClipState, sync_clip_state)
register_sync_type(ClippedFunAux, sync_clipped_fun_aux)
register_sync_type(ClippedGradAux, sync_clipped_grad_aux)
