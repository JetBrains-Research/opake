"""State synchronization, gathering, and type-dispatched sync for DP training.

This module exposes:

- scalar/tensor helpers: ``reduce_scalar``, ``gather_tensors``, ``gather_pytree``
- sanity checks: ``assert_scalar_equal``, ``assert_string_equal``, ``assert_pytree_equal``
- dataclass field sync: ``sync_object``
- type registry + dispatcher: ``register_sync_type``, ``sync``

The dispatcher is how user code synchronizes algorithm-specific states
(clipping, profiler, etc.) without having to know about each subsystem's
sync function: each subsystem registers its types on import.
"""

from __future__ import annotations

import contextlib
import inspect
from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from opake.api.engine.pytree import (
    tree_flatten,
    tree_leaves,
    tree_map,
    tree_structure,
    tree_unflatten,
)
from opake.exceptions import ConfigurationError, InputTypeError, OperationError

from .collectives import all_reduce_, get_world_size, is_distributed

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


def assert_scalar_equal(
    value: float | int,
    *,
    name: str,
    atol: float = 1e-8,
    rtol: float = 1e-5,
    compute_dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> None:
    """Raise if ``value`` is not equal across ranks within tolerance.

    Integer values are compared exactly through the integer reduction path.
    Floating-point values use ``compute_dtype`` (float32 by default).
    """
    if not is_distributed():
        return

    min_value = reduce_scalar(
        value, op="min", compute_dtype=compute_dtype, device=device
    )
    max_value = reduce_scalar(
        value, op="max", compute_dtype=compute_dtype, device=device
    )
    if isinstance(value, int) and not isinstance(value, bool):
        equal = min_value == max_value
    else:
        dtype = compute_dtype or torch.float32
        equal = torch.isclose(
            torch.tensor(min_value, dtype=dtype),
            torch.tensor(max_value, dtype=dtype),
            atol=atol,
            rtol=rtol,
        )
    if not equal:
        raise OperationError(
            *(f"{name} mismatch across ranks: min={min_value}, max={max_value}.",)
        )


def assert_string_equal(value: str | None, *, name: str) -> None:
    """Raise if an optional string is not exactly equal across ranks."""
    if not is_distributed():
        return

    values: list[str | None] = [None] * get_world_size()
    dist.all_gather_object(values, value)
    if any(other != values[0] for other in values[1:]):
        raise OperationError(*(f"{name} mismatch across ranks.",))


def assert_pytree_equal(
    pytree: Any,
    *,
    name: str = "pytree",
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> None:
    """Fingerprint-check a pytree for equality across ranks.

    Uses a cheap scalar (sum of elements) rather than transferring the whole
    tree. Useful for debugging divergence of params/grads.
    """
    if not is_distributed():
        return

    total = 0.0

    def _accumulate(leaf: Any) -> Any:
        nonlocal total
        if isinstance(leaf, torch.Tensor):
            total += leaf.detach().double().sum().item()
        return leaf

    tree_map(_accumulate, pytree)
    assert_scalar_equal(
        total,
        name=name,
        atol=atol,
        rtol=rtol,
        compute_dtype=torch.float64,
    )


def reduce_scalar(
    value: float | int,
    op: str = "mean",
    device: torch.device | None = None,
    *,
    compute_dtype: torch.dtype | None = None,
) -> float | int:
    """All-reduce a Python scalar.

    For NCCL backends, a CUDA device is required. If ``device`` is None a
    sensible one is inferred (current CUDA device for NCCL, else CPU).

    Integer values use an exact ``torch.int64`` collective. Floating-point
    values use ``compute_dtype`` when supplied and float32 otherwise.
    Integer means are computed from an exact integer sum and returned as a
    Python float.
    """
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise InputTypeError(*(f"value must be a float or int, got {type(value)}",))
    if compute_dtype is not None and not torch.is_floating_point(
        torch.empty((), dtype=compute_dtype)
    ):
        raise InputTypeError(
            *(
                f"compute_dtype must be a real floating-point dtype, got {compute_dtype!r}.",
            )
        )
    if isinstance(value, int) and compute_dtype is not None:
        raise InputTypeError(
            *("compute_dtype is only supported for floating-point values.",)
        )
    if not is_distributed():
        if isinstance(value, int) and op == "mean":
            return float(value)
        return value

    if device is None:
        backend = (
            dist.get_backend()
            if dist.is_available() and dist.is_initialized()
            else None
        )
        if backend == "nccl":
            if not torch.cuda.is_available():
                raise OperationError(
                    *(
                        "Distributed backend is 'nccl' but CUDA is not available; "
                        "provide an explicit `device` to `reduce_scalar` or initialize "
                        "with a CUDA-capable process.",
                    )
                )
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            device = torch.device("cpu")

    is_integer = isinstance(value, int)
    tensor = torch.tensor(
        value,
        dtype=torch.int64 if is_integer else compute_dtype or torch.float32,
        device=device,
    )
    if is_integer and op == "mean":
        all_reduce_(tensor, op="sum")
        return tensor.item() / get_world_size()
    all_reduce_(tensor, op=op)
    return tensor.item()


def _cpu_payload(value: Any) -> Any:
    return tree_map(
        lambda leaf: leaf.detach().cpu() if isinstance(leaf, torch.Tensor) else leaf,
        value,
    )


def _validate_gathered_tensor_column(
    tensors: list[torch.Tensor],
    *,
    ranks: list[int],
    leaf_index: int,
    dim: int,
) -> int:
    reference = tensors[0]
    if reference.ndim == 0:
        raise ConfigurationError(
            *(
                "Distributed tensor gathering cannot concatenate scalar tensor "
                f"leaf {leaf_index}; provide at least one dimension.",
            )
        )
    normalized_dim = dim if dim >= 0 else reference.ndim + dim
    if normalized_dim < 0 or normalized_dim >= reference.ndim:
        raise ConfigurationError(
            *(
                f"Gather dimension {dim} is out of range for tensor leaf {leaf_index} "
                f"with {reference.ndim} dimensions.",
            )
        )

    reference_rank = ranks[0]
    for rank, tensor in zip(ranks[1:], tensors[1:], strict=True):
        if tensor.dtype != reference.dtype:
            raise InputTypeError(
                *(
                    "Distributed tensor gathering requires matching dtypes; "
                    f"leaf {leaf_index} has {reference.dtype} on rank "
                    f"{reference_rank} and "
                    f"{tensor.dtype} on rank {rank}.",
                )
            )
        if tensor.ndim != reference.ndim:
            raise ConfigurationError(
                *(
                    "Distributed tensor gathering requires matching tensor ranks; "
                    f"leaf {leaf_index} has {reference.ndim} dimensions on rank "
                    f"{reference_rank} "
                    f"and {tensor.ndim} on rank {rank}.",
                )
            )
        for axis, (expected, actual) in enumerate(
            zip(reference.shape, tensor.shape, strict=True)
        ):
            if axis != normalized_dim and actual != expected:
                raise ConfigurationError(
                    *(
                        "Distributed tensor gathering requires matching "
                        "non-concatenated dimensions; "
                        f"leaf {leaf_index}, axis {axis} has size {expected} on "
                        f"rank {reference_rank} and {actual} on rank {rank}.",
                    )
                )
    return normalized_dim


def _merge_gathered_pytrees(
    values: list[Any],
    *,
    device: torch.device,
    dim: int,
) -> Any:
    """Merge rank-ordered optional tensor pytrees after one object collective."""
    present = [(rank, value) for rank, value in enumerate(values) if value is not None]
    if not present:
        return None

    first_rank, first = present[0]
    treedef = tree_structure(first)
    for rank, payload in present[1:]:
        other = tree_structure(payload)
        if other != treedef:
            raise InputTypeError(
                *(
                    "Distributed tensor gathering requires matching pytree "
                    "structures across non-empty ranks; "
                    f"rank {first_rank} has {treedef} and rank {rank} has {other}.",
                )
            )

    leaf_lists = [tree_flatten(payload)[0] for _, payload in present]
    if not leaf_lists[0]:
        return first

    merged_leaves: list[torch.Tensor] = []
    for leaf_index, column in enumerate(zip(*leaf_lists, strict=True)):
        if not all(isinstance(leaf, torch.Tensor) for leaf in column):
            raise InputTypeError(
                *(
                    "Distributed tensor gathering supports tensor leaves only; "
                    f"leaf {leaf_index} has types "
                    f"{[type(leaf).__name__ for leaf in column]}.",
                )
            )
        tensors = list(column)
        normalized_dim = _validate_gathered_tensor_column(
            tensors,
            ranks=[rank for rank, _ in present],
            leaf_index=leaf_index,
            dim=dim,
        )
        merged_leaves.append(
            torch.cat(
                [tensor.to(device) for tensor in tensors],
                dim=normalized_dim,
            )
        )
    return tree_unflatten(treedef, merged_leaves)


def gather_tensors(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Gather compatible variable-size tensors and concatenate along ``dim``."""
    if not is_distributed():
        return tensor

    gathered: list[Any] = [None] * get_world_size()
    dist.all_gather_object(gathered, tensor.detach().cpu())
    return _merge_gathered_pytrees(gathered, device=tensor.device, dim=dim)


def gather_pytree(pytree: Any, dim: int = 0) -> Any:
    """Gather an optional tensor pytree through one symmetric collective.

    Non-``None`` ranks must provide matching pytree structures and compatible
    tensor leaves. Rank-local ``None`` payloads contribute no rows while still
    participating in the collective.
    """
    if not is_distributed():
        return pytree

    local_leaves = tree_leaves(pytree)
    device = local_leaves[0].device if local_leaves else torch.device("cpu")
    gathered: list[Any] = [None] * get_world_size()
    dist.all_gather_object(gathered, _cpu_payload(pytree))
    return _merge_gathered_pytrees(gathered, device=device, dim=dim)


_VALID_FIELD_OPS = frozenset(
    {
        "sum",
        "mean",
        "max",
        "min",
        "product",
        "assert_equal",
        "assert_optional_equal",
        "local",
    }
)


def _validate_field_op(field_name: str, field_op: Any) -> None:
    """Reject an unusable ``field_ops`` entry before any collective is issued.

    Callables are checked against the one supported signature,
    ``fn(value, device)``.  Introspection deliberately does not follow
    ``functools.wraps``, so the arity checked is the one that will actually be
    called; a callable that exposes no signature at all is left to the call.
    """
    if isinstance(field_op, str):
        if field_op not in _VALID_FIELD_OPS:
            raise ConfigurationError(
                *(
                    f"field_ops[{field_name!r}] has unsupported operation {field_op!r}. "
                    f"Expected one of {sorted(_VALID_FIELD_OPS)} or a callable.",
                )
            )
        return
    if not callable(field_op):
        raise ConfigurationError(
            *(
                f"field_ops[{field_name!r}] must be an operation name or a callable, "
                f"got {type(field_op).__name__}.",
            )
        )
    try:
        signature = inspect.signature(field_op, follow_wrapped=False)
    except (TypeError, ValueError):
        return
    try:
        signature.bind(None, None)
    except TypeError as error:
        raise ConfigurationError(
            *(
                f"field_ops[{field_name!r}] must be callable as fn(value, device), "
                f"but its signature is {signature}: {error}.",
            )
        ) from error


def sync_object(
    state: Any,
    field_ops: Mapping[
        str, str | Callable[[Any, torch.device | None], float | int | None]
    ],
    device: torch.device | None = None,
) -> Any:
    """All-reduce scalar fields of a dataclass, returning a new instance.

    ``field_ops`` maps field name to a reduction op string
    (``"sum" | "mean" | "max" | "min" | "product" | "assert_equal" |
    "assert_optional_equal" | "local"``) or to a callable
    ``fn(value, device) -> float | int | None``.
    Every dataclass field must be present in this mapping. ``"local"`` marks a
    field whose rank-local value is intentionally carried through unchanged.

    **Callable semantics:** a callable must accept exactly ``fn(value, device)``
    and is invoked once, on the raw field value (any type).  A real ``float`` or
    ``int`` return (not ``bool``) replaces the field in the returned dataclass;
    a ``None`` return makes the callable **assertion only** — no field update.
    Exceptions propagate unchanged, so a callable that issues collectives issues
    them exactly once.

    The complete schema — operation names and callable signatures alike — is
    validated before any distributed collective begins, and operations execute
    in dataclass declaration order rather than mapping insertion order. This
    prevents rank-local values from changing the collective schedule.
    """
    if not is_dataclass(state):
        raise InputTypeError(*(f"state must be a dataclass, got {type(state)}",))

    dataclass_fields = fields(state)
    state_fields = {field.name for field in dataclass_fields}
    field_op_names = set(field_ops)
    unknown_fields = field_op_names - state_fields
    missing_fields = state_fields - field_op_names
    if unknown_fields or missing_fields:
        details = []
        if unknown_fields:
            details.append(f"unknown fields: {sorted(unknown_fields)}")
        if missing_fields:
            details.append(f"missing fields: {sorted(missing_fields)}")
        raise ConfigurationError(
            *(
                "field_ops must define every dataclass field; "
                + "; ".join(details)
                + ".",
            )
        )

    for field in dataclass_fields:
        _validate_field_op(field.name, field_ops[field.name])

    if not is_distributed():
        return state

    updates: dict[str, Any] = {}
    for field in dataclass_fields:
        field_name = field.name
        field_op = field_ops[field_name]
        value = getattr(state, field_name)
        if field_op == "local":
            continue
        if callable(field_op):
            result = field_op(value, device)
            if isinstance(result, bool) or not isinstance(result, (float, int)):
                continue
            if isinstance(value, bool):
                raise InputTypeError(
                    *(
                        f"Callable field_ops[{field_name!r}] cannot update a bool field; "
                        "use 'local' or return None for assertion-only behavior.",
                    )
                )
            synced = int(result) if isinstance(value, int) else float(result)
            updates[field_name] = synced
            continue
        if field_op == "assert_equal":
            if not isinstance(value, (float, int)) or isinstance(value, bool):
                raise InputTypeError(
                    *(
                        f"field_ops[{field_name!r}]='assert_equal' requires a float or int, "
                        f"got {type(value).__name__}.",
                    )
                )
            assert_scalar_equal(
                value,
                name=f"{type(state).__name__}.{field_name}",
                atol=0.0,
                rtol=0.0,
                compute_dtype=torch.float64 if isinstance(value, float) else None,
            )
            continue
        if field_op == "assert_optional_equal":
            is_present = int(value is not None)
            present_min = reduce_scalar(is_present, op="min", device=device)
            present_max = reduce_scalar(is_present, op="max", device=device)
            if present_min != present_max:
                raise OperationError(
                    *(
                        f"{type(state).__name__}.{field_name} presence mismatch across ranks.",
                    )
                )
            if not is_present:
                continue
            if not isinstance(value, (float, int)) or isinstance(value, bool):
                raise InputTypeError(
                    *(
                        f"field_ops[{field_name!r}]='assert_optional_equal' requires "
                        f"a float, int, or None, got {type(value).__name__}.",
                    )
                )
            assert_scalar_equal(
                value,
                name=f"{type(state).__name__}.{field_name}",
                atol=0.0,
                rtol=0.0,
                compute_dtype=torch.float64 if isinstance(value, float) else None,
            )
            continue
        if not isinstance(value, (float, int)) or isinstance(value, bool):
            raise InputTypeError(
                *(
                    f"field_ops[{field_name!r}]={field_op!r} requires a float or int, "
                    f"got {type(value).__name__}.",
                )
            )
        synced = reduce_scalar(value, op=field_op, device=device)
        if isinstance(value, int):
            synced = int(synced)
        updates[field_name] = synced

    if updates:
        state = type(state)(
            **{**{f.name: getattr(state, f.name) for f in fields(state)}, **updates}
        )
    return state


# ---------------------------------------------------------------------------
# Type-based sync dispatcher
# ---------------------------------------------------------------------------

_SYNC_REGISTRY: dict[type, Callable[[Any], Any]] = {}


def register_sync_type(state_type: type, sync_fn: Callable[[Any], Any]) -> None:
    """Register a sync function for ``state_type``.

    Subsystems (clipping, profiling, noise) call this on import to make their
    state types discoverable by :func:`sync`.  Subclasses inherit the
    registration through :func:`sync`'s MRO walk, so registering a base type
    covers its specialisations unless one of them registers its own handler.
    """
    _SYNC_REGISTRY[state_type] = sync_fn


def _resolve_sync_fn(state_type: type) -> Callable[[Any], Any] | None:
    """Return the handler for ``state_type`` or its nearest registered base.

    Exact type first, then ``__mro__`` order, matching the ``isinstance``
    semantics the underlying sync helpers already implement (for example
    ``sync_clipped_grad_aux`` rebuilds via ``type(aux)`` and so handles any
    ``ClippedGradAux`` subclass).
    """
    fn = _SYNC_REGISTRY.get(state_type)
    if fn is not None:
        return fn
    for base in state_type.__mro__[1:]:
        fn = _SYNC_REGISTRY.get(base)
        if fn is not None:
            return fn
    return None


def _ensure_builtin_sync_types_loaded() -> None:
    """Import internal registrations the first time a dispatch misses.

    Clipping registers itself synchronously; the performance profiler is
    a soft dependency — missing it must not break ``sync()``.  Profiling
    registers ``PerfState`` at the impl path; importing through the
    façade would require an extra side-effect import there, so we target
    the impl module directly.
    """
    import opake.api.engine.clipping._distributed

    with contextlib.suppress(ImportError):
        import opake.api.engine.profiling._distributed  # noqa: F401


def sync(*states: Any) -> Any:
    """Synchronize one or more registered state/aux objects across ranks.

    Dispatches on object type — exact match first, then ``__mro__`` — to
    whichever function was registered via :func:`register_sync_type`.  An
    unregistered type raises ``TypeError``; nothing is passed through
    unsynchronized.

    Returns:
        A single synchronized object if one argument was passed, otherwise a
        tuple matching the input order.
    """

    def _sync_one(single: Any) -> Any:
        state_type = type(single)
        sync_fn = _resolve_sync_fn(state_type)
        if sync_fn is None:
            _ensure_builtin_sync_types_loaded()
            sync_fn = _resolve_sync_fn(state_type)
        if sync_fn is not None:
            return sync_fn(single)
        raise InputTypeError(
            *(
                f"No sync function registered for {state_type.__name__}. "
                f"Registered types: {[t.__name__ for t in _SYNC_REGISTRY]}",
            )
        )

    if len(states) == 1:
        return _sync_one(states[0])
    if len(states) > 1:
        return tuple(_sync_one(s) for s in states)
    return ()


__all__ = [
    "assert_pytree_equal",
    "assert_scalar_equal",
    "assert_string_equal",
    "gather_pytree",
    "gather_tensors",
    "reduce_scalar",
    "register_sync_type",
    "sync",
    "sync_object",
]
