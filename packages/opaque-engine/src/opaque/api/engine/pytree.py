"""PyTree utilities for working with nested parameter structures.

This module provides utilities for working with PyTrees, i.e. nested Python
containers (dicts, lists, tuples, …) whose leaves are tensors. We implement a
thin wrapper on top of `optree` to provide a stable, fast pytree API that does
not depend on PyTorch's private modules.

Key functions:
- `tree_leaves(tree)`: return the list of tensor leaves in a PyTree.
- `tree_map(fn, *trees)`: apply a function to corresponding leaves of one or
  more PyTrees and rebuild the structure.
- `global_norm(tree)`: compute the global L2 norm across all tensor leaves:
  sqrt(sum(||leaf||^2)).

Notes
- Only tensor leaves are considered for `tree_leaves` and `global_norm`.
- `tree_map` applies `fn` to all leaves, passing through non-tensor leaves
  unchanged unless `fn` handles them explicitly.
- Supports mixed dtypes and complex numbers, returning complex norm for complex tensors.
- Handles empty trees gracefully, returning 0.0 for empty PyTrees.
- Supports PyTree parameters with nested dictionaries of tensors.
- Handles microbatching for memory efficiency in gradient clipping.
"""

from collections.abc import Callable
from typing import Any

import optree as _ot
import torch

from opaque.exceptions import ConfigurationError, InputTypeError

_RAW_TENSOR_PYTREE_ERROR = (
    "Tensor-leaf operations require raw tensor pytrees. Pass a single-stream "
    "wrapper's `.pytree`; for paired output, select a stream and pass that "
    "stream's `.pytree`."
)

ParamPath = tuple[str | int, ...]
"""Optree leaf path: nested dict keys (``str``) and sequence indices (``int``).

Flat ``named_parameters`` dicts use a one-segment path, e.g.
``("layers.0.weight",)``.  Nested trees use multi-segment paths, e.g.
``("layers", 0, "weight")``.  The two never collide.

A bare leaf pytree (e.g. a single ``Tensor``) uses the empty path ``()``,
matching :func:`optree.tree_flatten_with_path`.
"""


def param_path(path: tuple[Any, ...] | list[Any] | str) -> ParamPath:
    """Normalize an optree path (or flat string key) to :data:`ParamPath`.

    The empty path ``()`` is valid and denotes a root leaf (a pytree that
    is itself a single tensor).
    """
    if isinstance(path, str):
        return (path,)
    out: list[str | int] = []
    for part in path:
        if isinstance(part, (str, int)):
            out.append(part)
        else:
            raise InputTypeError(
                *(
                    f"ParamPath components must be str or int; got {type(part).__name__}",
                )
            )
    return tuple(out)


def param_path_display(path: ParamPath) -> str:
    """Dotted display form for pattern matching and diagnostics."""
    return ".".join(str(p) for p in path)


def tree_flatten_with_paths(
    tree: Any,
) -> tuple[list[ParamPath], list[Any], Any]:
    """Flatten ``tree`` to ``(paths, leaves, treedef)`` with :data:`ParamPath` keys.

    Wraps :func:`optree.tree_flatten_with_path` so PerGroup consumers share one
    path convention.
    """
    raw_paths, leaves, treedef = _ot.tree_flatten_with_path(tree)
    paths = [param_path(p) for p in raw_paths]
    return paths, list(leaves), treedef


def tree_flatten(tree: Any) -> tuple[list[Any], Any]:
    """Flatten ``tree`` to ``(leaves, treedef)``.

    Thin wrapper around :func:`optree.tree_flatten`.  Prefer
    :func:`tree_flatten_with_paths` when callers need leaf identities.
    """
    leaves, treedef = _ot.tree_flatten(tree)
    return list(leaves), treedef


def tree_unflatten(treedef: Any, leaves: list[Any]) -> Any:
    """Rebuild a PyTree from ``treedef`` and ``leaves``.

    Thin wrapper around :func:`optree.tree_unflatten`.
    """
    return _ot.tree_unflatten(treedef, leaves)


def tree_structure(tree: Any) -> Any:
    """Return the optree structure of ``tree`` (no leaves).

    Thin wrapper around :func:`optree.tree_structure`.  Useful for asserting
    that independently gathered payloads share a layout before unflattening.
    """
    return _ot.tree_structure(tree)


def tree_leaves(tree: Any) -> list[torch.Tensor]:
    """Extract all leaf tensors from a PyTree.

    Args:
        tree: Nested structure (dict/list/tuple/…) containing tensors as leaves.

    Returns:
        List of all tensor leaves in the tree (non-tensor leaves are ignored).

    Raises:
        InputTypeError: If ``tree`` contains a DP wrapper.

    Example:
        >>> tree = {'a': torch.tensor([1, 2]), 'b': {'c': torch.tensor([3])}}
        >>> leaves = tree_leaves(tree)
        >>> len(leaves)
        2
    """
    flat, _ = _ot.tree_flatten(tree)
    tensor_leaves = [leaf for leaf in flat if isinstance(leaf, torch.Tensor)]
    if len(tensor_leaves) == len(flat):
        return tensor_leaves

    # Local import avoids the types -> pytree cycle.
    from opaque.api.engine.types import ClippedPytree

    if any(isinstance(leaf, ClippedPytree) for leaf in flat):
        raise InputTypeError(_RAW_TENSOR_PYTREE_ERROR)
    return tensor_leaves


def tree_map(fn: Callable[..., Any], *trees: Any) -> Any:
    """Apply function to all leaves of one or more PyTrees.

    Args:
        fn: Function to apply to each leaf (or set of leaves from multiple trees).
        *trees: One or more PyTrees with matching structure.

    Returns:
        PyTree with same structure as inputs, with `fn` applied to leaves.

    Example:
        >>> tree = {'a': torch.tensor([1.0, 2.0]), 'b': torch.tensor([3.0])}
        >>> doubled = tree_map(lambda x: x * 2, tree)
        >>> doubled['a']
        tensor([2., 4.])
    """
    return _ot.tree_map(fn, *trees)


def tree_map_with_path(
    fn: Callable[[ParamPath, Any], Any],
    tree: Any,
) -> Any:
    """Apply function to leaves with their :data:`ParamPath` in the tree.

    Paths and leaf order match :func:`tree_flatten_with_paths` (optree
    traversal, including sorted dict keys), so results align with
    :class:`~opaque.types.PerGroup` keys.

    Args:
        fn: Function that takes ``(path, leaf)`` where ``path`` is a
            :data:`ParamPath`.
        tree: PyTree to traverse

    Returns:
        PyTree with same structure, with fn applied to (path, leaf)

    Example:
        >>> tree = {'layer1': {'weight': torch.ones(2)}, 'layer2': {'bias': torch.zeros(3)}}
        >>> def print_shapes(path, leaf):
        ...     print(f"{path}: {leaf.shape}")
        ...     return leaf
        >>> tree_map_with_path(print_shapes, tree)
        ('layer1', 'weight'): torch.Size([2])
        ('layer2', 'bias'): torch.Size([3])
    """
    paths, leaves, treedef = tree_flatten_with_paths(tree)
    out = [fn(path, leaf) for path, leaf in zip(paths, leaves, strict=True)]
    return _ot.tree_unflatten(treedef, out)


def partition(
    predicate: Callable[[tuple[Any, ...], Any], bool],
    tree: Any,
) -> tuple[Any, Any]:
    """Partition PyTree into two trees based on predicate.

    This is the key function for LoRA-style training: split parameters into
    trainable (e.g., LoRA adapters) and frozen (e.g., pretrained backbone).

    Args:
        predicate: Function(path, value) -> bool. Returns True for first tree.
        tree: PyTree to partition

    Returns:
        Tuple of (matching_tree, non_matching_tree) with same structure as input.
        Branches where all leaves are filtered out are omitted.

    Example:
        >>> params = {
        ...     'encoder': {
        ...         'weight': torch.randn(10, 5),
        ...         'lora_a': torch.randn(10, 2),
        ...         'lora_b': torch.randn(2, 5),
        ...     }
        ... }
        >>> def is_lora(path, value):
        ...     return 'lora' in str(path)
        >>> trainable, frozen = partition(is_lora, params)
        >>> 'lora_a' in trainable['encoder']  # True
        >>> 'weight' in frozen['encoder']      # True
        >>> 'weight' in trainable['encoder']   # False

    References:
        Inspired by Haiku's hk.data_structures.partition():
        https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/data_structures.py
    """

    def _partition_subtree(path: tuple, subtree: Any) -> tuple[Any, Any]:
        if isinstance(subtree, dict):
            true_dict = {}
            false_dict = {}
            for k, v in subtree.items():
                true_part, false_part = _partition_subtree((*path, k), v)
                if true_part is not None:
                    true_dict[k] = true_part
                if false_part is not None:
                    false_dict[k] = false_part
            return (
                true_dict or None,
                false_dict or None,
            )
        elif isinstance(subtree, (list, tuple)):
            true_list = []
            false_list = []
            for i, v in enumerate(subtree):
                true_part, false_part = _partition_subtree((*path, i), v)
                true_list.append(true_part)
                false_list.append(false_part)
            # Keep same type (list vs tuple)
            any_true = any(x is not None for x in true_list)
            any_false = any(x is not None for x in false_list)
            return (
                type(subtree)(true_list) if any_true else None,
                type(subtree)(false_list) if any_false else None,
            )
        else:
            # Leaf node - apply predicate
            if predicate(path, subtree):
                return subtree, None
            else:
                return None, subtree

    true_tree, false_tree = _partition_subtree((), tree)
    # Return empty dicts instead of None for empty trees
    if true_tree is None:
        true_tree = {} if isinstance(tree, dict) else type(tree)()
    if false_tree is None:
        false_tree = {} if isinstance(tree, dict) else type(tree)()
    return true_tree, false_tree


def merge(*trees: Any) -> Any:
    """Merge multiple PyTrees into one.

    Later trees override earlier ones for overlapping keys.

    Args:
        *trees: PyTrees to merge (must have compatible structures)

    Returns:
        Merged PyTree

    Example:
        >>> trainable = {'encoder': {'lora_a': torch.ones(2)}}
        >>> frozen = {'encoder': {'weight': torch.zeros(3)}}
        >>> merged = merge(frozen, trainable)
        >>> merged
        {'encoder': {'weight': ..., 'lora_a': ...}}

    References:
        Inspired by Haiku's hk.data_structures.merge():
        https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/data_structures.py
    """
    if not trees:
        return {}

    # Filter out None/empty trees
    trees = [t for t in trees if t is not None]
    if not trees:
        return {}

    # Base case: single tree
    if len(trees) == 1:
        return trees[0]

    # Get first tree as reference
    result = trees[0]

    # Merge remaining trees
    for tree in trees[1:]:
        result = _merge_two(result, tree)

    return result


def _merge_two(tree1: Any, tree2: Any) -> Any:
    """Merge two PyTrees (tree2 overrides tree1)."""
    # If either is None, return the other
    if tree1 is None:
        return tree2
    if tree2 is None:
        return tree1

    # If both are dicts, merge recursively
    if isinstance(tree1, dict) and isinstance(tree2, dict):
        result = dict(tree1)  # Start with tree1
        for key in tree2:
            if key in result:
                # Recursively merge
                result[key] = _merge_two(result[key], tree2[key])
            else:
                # Add new key
                result[key] = tree2[key]
        return result

    # If both are lists/tuples, merge element-wise
    if isinstance(tree1, (list, tuple)) and isinstance(tree2, (list, tuple)):
        # Must have same length
        if len(tree1) != len(tree2):
            raise ConfigurationError(
                *(
                    f"Cannot merge sequences of different lengths: {len(tree1)} vs {len(tree2)}",
                )
            )
        merged = [_merge_two(a, b) for a, b in zip(tree1, tree2, strict=True)]
        return type(tree1)(merged)

    # Otherwise, tree2 overwrites tree1
    return tree2


def global_norm(
    tree: Any,
    *,
    compute_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Compute global L2 norm across all tensors in a PyTree.

    The global norm is the square root of the sum of squared norms of all
    leaf tensors:
        global_norm = sqrt(sum(||leaf||^2 for leaf in tree))

    Args:
        tree: PyTree of tensors (e.g., parameters or gradients)
        compute_dtype: Internal accumulation dtype.  ``None`` (default)
            promotes low-precision floats (fp16/bf16) to float32 for
            numerical stability and otherwise uses the input float dtype
            promoted to at least float32.  Pass an explicit dtype (e.g.
            ``torch.float64``) to force a specific accumulation precision.

    Returns:
        Scalar tensor containing the global L2 norm on the device of the first
        tensor leaf (or CPU if the tree is empty).  Output dtype matches
        the resolved compute dtype.

    Raises:
        InputTypeError: If ``tree`` contains a DP wrapper or ``compute_dtype``
            is not a real floating-point dtype.

    Example:
        >>> tree = {'w': torch.tensor([3.0, 4.0]), 'b': torch.tensor([0.0, 12.0])}
        >>> norm = global_norm(tree)
        >>> # norm = sqrt(3^2 + 4^2 + 0^2 + 12^2) = sqrt(169) = 13
        >>> torch.isclose(norm, torch.tensor(13.0))
        True

    References:
        This function is commonly used in gradient clipping for deep learning.
        See: Pascanu et al. 2013, "On the difficulty of training RNNs"
    """
    from opaque.api.engine.types import (
        ClippedPytree,
        NoisedPytree,
        SecondMomentClippingOutput,
        SecondMomentNoiseOutput,
    )

    if isinstance(
        tree,
        (
            ClippedPytree,
            NoisedPytree,
            SecondMomentClippingOutput,
            SecondMomentNoiseOutput,
        ),
    ):
        raise InputTypeError(_RAW_TENSOR_PYTREE_ERROR)

    if compute_dtype is not None and not torch.is_floating_point(
        torch.empty((), dtype=compute_dtype)
    ):
        raise InputTypeError(
            *(
                f"compute_dtype must be a real floating-point dtype, got "
                f"{compute_dtype!r}.  Integer/bool/complex compute dtypes can "
                f"silently corrupt the L2-norm reduction (the squared sum is "
                f"non-negative real and the final sqrt assumes a real "
                f"accumulator).",
            )
        )

    leaves = tree_leaves(tree)
    if not leaves:
        return torch.tensor(0.0, dtype=compute_dtype or torch.float32)

    if compute_dtype is not None:
        real_acc_dtype = compute_dtype
    else:
        # Auto-promote: at least float32, but match user's intent if they
        # supplied higher-precision inputs.
        dtypes = [t.dtype for t in leaves]
        float_dtypes = [
            dt for dt in dtypes if torch.is_floating_point(torch.empty((), dtype=dt))
        ]
        complex_dtypes = [
            dt for dt in dtypes if torch.is_complex(torch.empty((), dtype=dt))
        ]

        if complex_dtypes:
            acc_dtype = torch.complex64
            for dt in complex_dtypes:
                acc_dtype = torch.promote_types(acc_dtype, dt)
            # For complex, accumulate real magnitudes in the corresponding real dtype
            real_acc_dtype = torch.promote_types(
                torch.float32, torch.tensor(0, dtype=acc_dtype).real.dtype
            )
        elif float_dtypes:
            real_acc_dtype = torch.float32
            for dt in float_dtypes:
                real_acc_dtype = torch.promote_types(real_acc_dtype, dt)
        else:
            # All integer/bool → accumulate in float32
            real_acc_dtype = torch.float32

    device = leaves[0].device
    total = torch.zeros((), device=device, dtype=real_acc_dtype)

    for leaf in leaves:
        if torch.is_complex(leaf):
            # ||z||^2 = (real^2 + imag^2)
            real = leaf.real.to(real_acc_dtype)
            imag = leaf.imag.to(real_acc_dtype)
            total = (
                total
                + (real * real).sum(dtype=real_acc_dtype)
                + (imag * imag).sum(dtype=real_acc_dtype)
            )
        else:
            x = leaf.to(real_acc_dtype)
            total = total + (x * x).sum(dtype=real_acc_dtype)

    return torch.sqrt(total)


__all__ = [
    "ParamPath",
    "global_norm",
    "merge",
    "param_path",
    "param_path_display",
    "partition",
    "tree_flatten",
    "tree_flatten_with_paths",
    "tree_leaves",
    "tree_map",
    "tree_map_with_path",
    "tree_structure",
    "tree_unflatten",
]
