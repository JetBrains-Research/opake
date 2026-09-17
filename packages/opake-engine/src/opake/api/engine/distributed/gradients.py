"""Gradient / pytree reduction helpers for distributed DP training.

Provides ``reduce_pytree(_)`` for generic per-leaf all-reduce and
``sum_gradients(_)`` as a thin alias scoped to clipped gradients.

``ClippedPytree`` and ``NoisedPytree`` support the linear reductions whose DP
metadata semantics are determined: ``sum`` and ``mean``.  Summing disjoint local
clipped queries preserves the per-record max_norm; averaging divides it by world
size.  Summing independent Gaussian-noised local queries scales ``noise_stddev``
by ``sqrt(world_size)``; averaging scales it by ``1 / sqrt(world_size)``.

Paired-stream wrappers (:class:`~opake.types.SecondMomentClippingOutput`,
:class:`~opake.types.SecondMomentNoiseOutput`) recurse into both child
wrappers so both streams participate in the collective.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, TypeGuard

import torch
import torch.distributed as dist

from opake.api.engine.pytree import tree_map
from opake.api.engine.types import (
    ClippedPytree,
    NoisedPytree,
    PerGroup,
    SecondMomentClippingOutput,
    SecondMomentNoiseOutput,
)
from opake.exceptions import ConfigurationError, InputTypeError, OperationError

from ._state import assert_scalar_equal
from .collectives import all_reduce_, get_world_size, is_distributed


def _is_noised(pytree: Any) -> TypeGuard[NoisedPytree]:
    return isinstance(pytree, NoisedPytree)


_WRAPPER_REDUCTION_OPS = {"sum", "mean"}


def _assert_object_equal(value: Any, *, name: str) -> None:
    if not is_distributed():
        return
    gathered = [None] * get_world_size()
    dist.all_gather_object(gathered, value)
    mismatched = [idx for idx, other in enumerate(gathered) if other != value]
    if mismatched:
        raise OperationError(
            *(f"{name} mismatch across ranks: mismatched ranks={mismatched}.",)
        )


def _assert_public_metadata_equal(value: Any, *, name: str) -> None:
    if not is_distributed():
        return
    metadata_kind = (
        "per_group"
        if isinstance(value, PerGroup)
        else "none"
        if value is None
        else "scalar"
    )
    _assert_object_equal(metadata_kind, name=f"{name}.kind")
    if isinstance(value, PerGroup):
        _assert_object_equal(dict(value.groups), name=f"{name}.groups")
        _assert_object_equal(set(value.values), name=f"{name}.values.keys")
        for group_name in sorted(value.values):
            assert_scalar_equal(
                value.values[group_name],
                name=f"{name}.values[{group_name!r}]",
            )
        return
    if value is None:
        _assert_object_equal(value, name=name)
        return
    groups = getattr(value, "groups", None)
    values = getattr(value, "values", None)
    if isinstance(groups, dict) and isinstance(values, dict):
        _assert_object_equal(groups, name=f"{name}.groups")
        _assert_object_equal(set(values), name=f"{name}.values.keys")
        for group_name in sorted(values):
            assert_scalar_equal(
                values[group_name],
                name=f"{name}.values[{group_name!r}]",
            )
        return
    assert_scalar_equal(value, name=name)


def _assert_wrapper_reduction_supported(pytree: ClippedPytree, op: str) -> None:
    if op not in _WRAPPER_REDUCTION_OPS:
        raise InputTypeError(
            *(
                f"{type(pytree).__name__} distributed reduction only supports "
                "op='sum' or op='mean'. "
                "Use `.pytree` and reconstruct with an explicit max_norm for other reductions.",
            )
        )


def _metadata_world_size() -> int:
    return get_world_size() if is_distributed() else 1


def _scale_public_metadata(value: Any, factor: float) -> Any:
    if value is None:
        return None
    return factor * value


def _reduced_metadata(pytree: ClippedPytree, op: str, world_size: int) -> ClippedPytree:
    if op == "sum":
        max_norm = pytree.max_norm
        if _is_noised(pytree):
            noise_stddev = _scale_public_metadata(
                pytree.noise_stddev,
                math.sqrt(float(world_size)),
            )
            return replace(pytree, max_norm=max_norm, noise_stddev=noise_stddev)
        return replace(pytree, max_norm=max_norm)

    if op == "mean":
        max_norm = _scale_public_metadata(pytree.max_norm, 1.0 / float(world_size))
        if _is_noised(pytree):
            noise_stddev = _scale_public_metadata(
                pytree.noise_stddev,
                1.0 / math.sqrt(float(world_size)),
            )
            return replace(pytree, max_norm=max_norm, noise_stddev=noise_stddev)
        return replace(pytree, max_norm=max_norm)

    raise ConfigurationError(*(f"Unsupported wrapper reduction op: {op}",))


def _in_place_wrapper_metadata_changes(pytree: ClippedPytree, op: str) -> bool:
    world_size = _metadata_world_size()
    if world_size == 1:
        return False
    if op == "mean":
        return True
    return _is_noised(pytree) and pytree.noise_stddev is not None


def reduce_pytree_(pytree: Any, op: str = "sum") -> None:
    """All-reduce tensor leaves in place.

    In-place wrapper reductions are accepted only when the wrapper metadata
    stays unchanged.  Use :func:`reduce_pytree` for reductions such as noised
    ``sum`` or clipped/noised ``mean`` that need updated metadata.
    """
    if isinstance(pytree, SecondMomentClippingOutput):
        reduce_pytree_(pytree.grads, op=op)
        reduce_pytree_(pytree.squared_grads, op=op)
        return

    if isinstance(pytree, SecondMomentNoiseOutput):
        reduce_pytree_(pytree.noisy_grads, op=op)
        reduce_pytree_(pytree.noisy_squared_grads, op=op)
        return

    if isinstance(pytree, ClippedPytree):
        _assert_wrapper_reduction_supported(pytree, op)
        if _in_place_wrapper_metadata_changes(pytree, op):
            raise InputTypeError(
                *(
                    f"In-place {type(pytree).__name__} reduction would change metadata; "
                    "use reduce_pytree() instead.",
                )
            )
        _assert_public_metadata_equal(pytree.max_norm, name="ClippedPytree.max_norm")
        if _is_noised(pytree):
            _assert_public_metadata_equal(
                pytree.noise_stddev,
                name="NoisedPytree.noise_stddev",
            )
        reduce_pytree_(pytree.pytree, op=op)
        return

    if not is_distributed():
        return

    def _reduce(leaf: Any) -> Any:
        if isinstance(leaf, torch.Tensor):
            all_reduce_(leaf, op=op)
            return leaf
        raise InputTypeError(
            *(
                f"reduce_pytree_ expects tensor leaves after wrapper dispatch; "
                f"got {type(leaf).__name__}. Unwrap paired/custom containers "
                f"explicitly or register a reduction branch.",
            )
        )

    tree_map(_reduce, pytree)


def reduce_pytree(pytree: Any, op: str = "sum") -> Any:
    """Return a pytree with each tensor leaf reduced; input unchanged.

    When passed ``ClippedPytree`` or ``NoisedPytree``, preserves and updates the
    wrapper metadata for supported ``sum`` and ``mean`` reductions.  Paired
    second-moment wrappers recurse into both child streams.
    """
    if isinstance(pytree, SecondMomentClippingOutput):
        return SecondMomentClippingOutput(
            reduce_pytree(pytree.grads, op=op),
            reduce_pytree(pytree.squared_grads, op=op),
        )

    if isinstance(pytree, SecondMomentNoiseOutput):
        return SecondMomentNoiseOutput(
            reduce_pytree(pytree.noisy_grads, op=op),
            reduce_pytree(pytree.noisy_squared_grads, op=op),
        )

    if isinstance(pytree, ClippedPytree):
        _assert_wrapper_reduction_supported(pytree, op)
        _assert_public_metadata_equal(pytree.max_norm, name="ClippedPytree.max_norm")
        if _is_noised(pytree):
            _assert_public_metadata_equal(
                pytree.noise_stddev,
                name="NoisedPytree.noise_stddev",
            )
        reduced = pytree.clone()
        reduce_pytree_(reduced.pytree, op=op)
        return _reduced_metadata(reduced, op, _metadata_world_size())

    def _clone(leaf: Any) -> Any:
        if isinstance(leaf, torch.Tensor):
            return leaf.clone()
        raise InputTypeError(
            *(
                f"reduce_pytree expects tensor leaves after wrapper dispatch; "
                f"got {type(leaf).__name__}. Unwrap paired/custom containers "
                f"explicitly or register a reduction branch.",
            )
        )

    reduced = tree_map(_clone, pytree)
    reduce_pytree_(reduced, op=op)
    return reduced


def sum_gradients_(gradients: Any) -> None:
    """DP-specific alias for ``reduce_pytree_(op="sum")``."""
    reduce_pytree_(gradients, op="sum")


def sum_gradients(gradients: Any) -> Any:
    """DP-specific alias for ``reduce_pytree(op="sum")``."""
    return reduce_pytree(gradients, op="sum")


__all__ = [
    "reduce_pytree",
    "reduce_pytree_",
    "sum_gradients",
    "sum_gradients_",
]
