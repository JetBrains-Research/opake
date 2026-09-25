"""Leaf-streamed fixed global clipping of already-batched function values."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.func import vmap

from opake.api.engine.clipping._pytree import (
    _leaf_sq_sum,
    _norm_roundoff,
    _real_dtype,
    _reduction_terms,
    _resolve_compute_dtype_for_reduction,
    _scale_tensor,
    _sq_accum_dtype,
)
from opake.api.engine.pytree import tree_flatten, tree_unflatten

if TYPE_CHECKING:
    from collections.abc import Callable


def _stream_clip_and_sum(
    values: Any,
    clipping_norm: torch.Tensor,
    *,
    batch_size: int,
    reduce_leaf: Callable,
    compute_dtype: torch.dtype | None,
    return_aux: bool,
    return_stats: bool,
) -> tuple[Any, Any, Any]:
    """Keep raw values, but only one sanitized/scaled leaf at a time.

    Each leaf still uses the per-example reduction and scaling primitives from
    ``clip_pytree`` under vmap. This preserves the reduction order, roundoff
    bound and subnormal safeguard without materializing a clipped batch tree.
    The second pass reduces only after the *global* sanitized norm is known.
    """
    leaves, treedef = tree_flatten(values)
    acc_dtype = _resolve_compute_dtype_for_reduction(leaves, compute_dtype)
    sq_dtype = _sq_accum_dtype(leaves)
    roundoff = _norm_roundoff(
        acc_dtype, sq_dtype, len(leaves), _reduction_terms([leaf[0] for leaf in leaves])
    )

    sq_norm = None
    for leaf in leaves:
        sanitized = torch.nan_to_num(leaf, nan=0.0, posinf=0.0, neginf=0.0)
        sq = vmap(lambda x: _leaf_sq_sum(x, acc_dtype, sq_dtype))(sanitized)
        sq_norm = sq if sq_norm is None else sq_norm + sq
        del sanitized, sq
    if sq_norm is None:
        sq_norm = torch.zeros(batch_size, dtype=sq_dtype)
    norm = torch.sqrt(sq_norm)
    bound = torch.clamp(clipping_norm.to(dtype=norm.dtype, device=norm.device), min=0.0)
    ratio = bound / norm

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
    for leaf in leaves:
        sanitized = torch.nan_to_num(leaf, nan=0.0, posinf=0.0, neginf=0.0)
        stored = vmap(
            lambda x, r: _scale_tensor(x, r, acc_dtype, roundoff, clamp_to_one=True)
        )(sanitized, ratio)
        del sanitized
        if return_aux:
            post_sq = vmap(add_stored_sq)(post_sq, stored)
        reduced.append(reduce_leaf(stored))
        markers.append(leaf.new_zeros(()))
        del stored

    diagnostics = {}
    if return_aux or return_stats:
        diagnostics["norms"] = norm.to(acc_dtype).detach()
    if return_aux:
        diagnostics["clipped_norms"] = torch.sqrt(post_sq).detach()
    return (
        tree_unflatten(treedef, reduced),
        tree_unflatten(treedef, markers),
        diagnostics if return_aux or return_stats else (),
    )
