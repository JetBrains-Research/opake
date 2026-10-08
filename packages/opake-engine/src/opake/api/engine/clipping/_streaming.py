"""Leaf-streamed built-in clipping of already-batched function values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch.func import vmap

from opake.api.engine.clipping._pytree import (
    _accumulate_group_value,
    _group_real_numel,
    _leaf_sq_sum,
    _norm_roundoff,
    _norms_for_scaling,
    _real_dtype,
    _reduction_terms,
    _resolve_compute_dtype_for_reduction,
    _scale_tensor,
    _sq_accum_dtype,
    _tensor_path_leaves,
    _tree_real_numel,
    _validate_per_group_paths,
)
from opake.api.engine.pytree import tree_flatten, tree_unflatten
from opake.api.engine.types import PerGroup

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class _BuiltinScale:
    kind: Literal["fixed", "auto_s"]
    gamma: float = 0.0


_FIXED_SCALE = _BuiltinScale(kind="fixed")


def _stream_clip_and_sum(
    values: Any,
    clipping_norm: torch.Tensor | PerGroup,
    *,
    scale: _BuiltinScale,
    batch_size: int,
    reduce_leaf: Callable,
    compute_dtype: torch.dtype | None,
    second_moment: bool,
    return_aux: bool,
    return_stats: bool,
) -> tuple[Any, Any, Any, Any, Any]:
    """Keep raw values, but only one sanitized/scaled leaf at a time.

    Each leaf still uses the per-example reduction and scaling primitives from
    ``clip_pytree`` under vmap. This preserves the reduction order, roundoff
    bound and subnormal safeguard without materializing a clipped batch tree.
    The second pass reduces only after the *global* sanitized norm is known.
    """
    paths = None
    if isinstance(clipping_norm, PerGroup):
        paths, leaves, treedef = _tensor_path_leaves(values)
        _validate_per_group_paths(paths, clipping_norm)
    else:
        leaves, treedef = tree_flatten(values)
    acc_dtype = _resolve_compute_dtype_for_reduction(leaves, compute_dtype)
    sq_dtype = _sq_accum_dtype(leaves)
    example_leaves = [leaf[0] for leaf in leaves]
    roundoff = _norm_roundoff(
        acc_dtype, sq_dtype, len(leaves), _reduction_terms(example_leaves)
    )
    n_components = _tree_real_numel(example_leaves)
    group_sizes: dict[str, int] = {}
    if isinstance(clipping_norm, PerGroup):
        assert paths is not None
        group_sizes = _group_real_numel(paths, example_leaves, clipping_norm)

    sq_norm = None
    group_sq_norms: dict[str, torch.Tensor] = {}
    for index, leaf in enumerate(leaves):
        sanitized = torch.nan_to_num(leaf, nan=0.0, posinf=0.0, neginf=0.0)
        sq = vmap(lambda x: _leaf_sq_sum(x, acc_dtype, sq_dtype))(sanitized)
        sq_norm = sq if sq_norm is None else sq_norm + sq
        if paths is not None:
            group_name = clipping_norm.groups[paths[index]]
            _accumulate_group_value(group_sq_norms, group_name, sq)
        del sanitized, sq
    if sq_norm is None:
        sq_norm = torch.zeros(batch_size, dtype=sq_dtype)
    norm, scaling_norm = _norms_for_scaling(sq_norm, acc_dtype, n_components)

    def scale_ratio(bound_value, value_norm):
        bound = torch.clamp(
            torch.as_tensor(
                bound_value, dtype=value_norm.dtype, device=value_norm.device
            ),
            min=0.0,
        )
        if scale.kind == "auto_s":
            gamma = torch.as_tensor(
                scale.gamma, dtype=value_norm.dtype, device=value_norm.device
            )
            return bound / (value_norm + gamma)
        return bound / value_norm

    if isinstance(clipping_norm, PerGroup):
        group_norms: dict[str, torch.Tensor] = {}
        group_scaling_norms: dict[str, torch.Tensor] = {}
        for name, sq in group_sq_norms.items():
            measured, guarded = _norms_for_scaling(sq, acc_dtype, group_sizes[name])
            group_norms[name] = measured
            group_scaling_norms[name] = guarded
        group_ratios = {
            name: scale_ratio(clipping_norm.values[name], group_scaling_norms[name])
            for name in group_norms
        }
        ratio = None
    else:
        group_norms = None
        group_ratios = None
        ratio = scale_ratio(clipping_norm, scaling_norm)
    clamp_to_one = scale.kind != "auto_s"

    # Match global_norm's diagnostic precision, including mixed complex/real
    # trees. Its flat reductions deliberately differ from the clipping guard.
    diagnostic_dtype = acc_dtype
    if compute_dtype is None:
        complex_dtype = None
        for leaf in leaves:
            if leaf.is_complex():
                complex_dtype = (
                    leaf.dtype
                    if complex_dtype is None
                    else torch.promote_types(complex_dtype, leaf.dtype)
                )
        if complex_dtype is not None:
            diagnostic_dtype = torch.promote_types(
                torch.float32, _real_dtype(complex_dtype)
            )
    post_sq = torch.zeros_like(norm, dtype=diagnostic_dtype) if return_aux else None

    def add_stored_sq(total, stored):
        if stored.is_complex():
            real = stored.real.to(diagnostic_dtype)
            imag = stored.imag.to(diagnostic_dtype)
            return (
                total
                + (real * real).sum(dtype=diagnostic_dtype)
                + (imag * imag).sum(dtype=diagnostic_dtype)
            )
        x = stored.to(diagnostic_dtype)
        return total + (x * x).sum(dtype=diagnostic_dtype)

    reduced = []
    markers = []
    squared_reduced = []
    squared_markers = []
    for index, leaf in enumerate(leaves):
        leaf_ratio = (
            group_ratios[clipping_norm.groups[paths[index]]]
            if group_ratios is not None
            else ratio
        )
        sanitized = torch.nan_to_num(leaf, nan=0.0, posinf=0.0, neginf=0.0)
        stored = vmap(
            lambda x, r: _scale_tensor(
                x, r, acc_dtype, roundoff, clamp_to_one=clamp_to_one
            )
        )(sanitized, leaf_ratio)
        del sanitized
        if return_aux:
            post_sq = vmap(add_stored_sq)(post_sq, stored)
        reduced.append(reduce_leaf(stored))
        markers.append(leaf.new_zeros(()))
        if second_moment:
            squared = stored.square()
            squared_reduced.append(reduce_leaf(squared))
            squared_markers.append(squared.new_zeros(()))
            del squared
        del stored

    diagnostics = {}
    if return_aux or return_stats:
        diagnostics["norms"] = norm.to(acc_dtype).detach()
        if group_norms is not None:
            diagnostics["group_norms"] = {
                name: group_norm.to(acc_dtype).detach()
                for name, group_norm in group_norms.items()
            }
    if return_aux:
        diagnostics["clipped_norms"] = torch.sqrt(post_sq).detach()
    return (
        tree_unflatten(treedef, reduced),
        tree_unflatten(treedef, markers),
        tree_unflatten(treedef, squared_reduced) if second_moment else (),
        tree_unflatten(treedef, squared_markers) if second_moment else (),
        diagnostics if return_aux or return_stats else (),
    )
