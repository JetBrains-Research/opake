"""Build a :class:`opake.types.PerGroup` from params + substring patterns::

from opake.dpsgd.clipping import per_group
pg = per_group(params, self_attn=1.0, mlp=2.0)
pg = per_group(params, q_proj=1.0, fallback=0.5)  # catch-all
"""

from __future__ import annotations

from typing import Any

import torch

from opake.api.engine.pytree import (
    ParamPath,
    param_path_display,
    tree_flatten_with_paths,
)
from opake.api.engine.types import PerGroup as _PerGroup
from opake.exceptions import ConfigurationError, InputTypeError


def _tensor_paths(params: Any) -> list[ParamPath]:
    """Leaf :data:`~opake.pytree.ParamPath`s for tensor leaves in ``params``."""
    paths, leaves, _ = tree_flatten_with_paths(params)
    out: list[ParamPath] = []
    for path, leaf in zip(paths, leaves, strict=True):
        if isinstance(leaf, torch.Tensor):
            out.append(path)
        elif leaf is None:
            continue
        else:
            raise InputTypeError(
                *(
                    "per_group expects a PyTree of tensors; "
                    f"non-tensor leaf at path {path!r}: {type(leaf).__name__}",
                )
            )
    if not out:
        raise ConfigurationError(*("per_group requires at least one tensor leaf.",))
    return out


def per_group(
    params: Any,
    /,
    patterns: dict[str, float] | None = None,
    *,
    fallback: float | None = None,
    allow_unused_patterns: bool = False,
    **kwargs: float,
) -> _PerGroup:
    """Construct PerGroup from parameter leaf paths and substring patterns.

    Each pattern is a substring matched against the dotted display form of
    each leaf :data:`~opake.pytree.ParamPath` (see
    :func:`~opake.pytree.param_path_display`).  Leaves whose display path
    contains the substring are assigned to that group.  Every leaf must
    match exactly one pattern (error on 0 or 2+).

    ``params`` may be any tensor pytree: flat ``named_parameters`` dicts
    (paths are one-segment ``(name,)``), nested dicts, lists, or tuples.
    The returned :class:`~opake.types.PerGroup` is compiled against those
    optree paths so clip / noise / optimizers look up the same identity.

    The ``patterns`` dict arg merges with ``**kwargs`` to support keys
    containing dots (which cannot be used as keyword arguments)::

        per_group(params, **{"layers.0": 0.5, "layers.1": 1.0})
        # equivalent to:
        per_group(params, patterns={"layers.0": 0.5, "layers.1": 1.0})

    The ``fallback`` parameter assigns a value to any parameter that does
    not match an explicit pattern.  Without ``fallback``, unmatched
    parameters raise ``ValueError``.

    By default, every explicit pattern must match at least one parameter.
    Set ``allow_unused_patterns=True`` when intentionally reusing a
    configuration across models with different parameter names.

    Args:
        params: Parameter pytree (flat or nested).  Leaf paths are matched
            against patterns via their dotted display form.
        patterns: Optional dict of ``{pattern: value}`` pairs.  Merged with
            ``**kwargs``.
        fallback: Optional value for parameters that don't match any
            explicit pattern.  Unmatched params are assigned to a group
            named ``"fallback"``.
        allow_unused_patterns: Whether to allow explicit patterns that do
            not match any parameter. Defaults to ``False``.
        **kwargs: Pattern-value pairs where the kwarg name is the substring
            pattern and the value is the per-group value (e.g. clipping norm).

    Returns:
        :class:`~opake.types.PerGroup` with pre-resolved
        path-to-group assignments.

    See also:
        :mod:`opake.serialization` — persist or restore a ``PerGroup`` with
        ``state_dict`` / ``from_state_dict``.  Call :func:`per_group` again
        when parameter keys or grouping patterns change.

    Raises:
        TypeError: If ``params`` has a non-tensor leaf.
        ValueError: If no patterns are provided, if a parameter matches zero
            patterns (and no ``fallback`` is given), if a parameter
            matches multiple patterns, if an explicit pattern matches no
            parameter (unless ``allow_unused_patterns=True``), or if any
            value is not positive.

    Examples:
        >>> per_group(params, self_attn=1.0, mlp=2.0)
        >>> per_group(params, self_attn=1.0, fallback=0.5)  # catch-all
        >>> per_group(params, q_proj=0.5, k_proj=0.5, v_proj=0.5, o_proj=0.8,
        ...           gate_proj=1.0, up_proj=1.0, down_proj=1.0)
        >>> per_group(params, **{f'layers.{i}': norms[i] for i in range(32)})
    """
    all_patterns: dict[str, float] = {}
    if patterns is not None:
        all_patterns.update(patterns)
    all_patterns.update(kwargs)

    if not all_patterns and fallback is None:
        raise ConfigurationError(*("At least one pattern must be provided.",))

    for pat, val in all_patterns.items():
        if val <= 0:
            raise ConfigurationError(
                *(f"Per-group value must be positive, got {val} for pattern '{pat}'.",)
            )

    if fallback is not None and fallback <= 0:
        raise ConfigurationError(
            *(f"Fallback value must be positive, got {fallback}.",)
        )

    param_paths = _tensor_paths(params)

    groups: dict[ParamPath, str] = {}
    matched_patterns: set[str] = set()
    for path in param_paths:
        display = param_path_display(path)
        matches = [pat for pat in all_patterns if pat in display]
        if len(matches) == 0:
            if fallback is not None:
                groups[path] = "fallback"
                continue
            raise ConfigurationError(
                *(
                    f"Parameter path {path!r} (display {display!r}) did not match "
                    f"any pattern. Available patterns: {list(all_patterns.keys())}. "
                    f"Use fallback=<value> to catch unmatched parameters.",
                )
            )
        if len(matches) > 1:
            raise ConfigurationError(
                *(
                    f"Parameter path {path!r} (display {display!r}) matched "
                    f"multiple patterns: {matches}. "
                    f"Each parameter must match exactly one pattern.",
                )
            )
        groups[path] = matches[0]
        matched_patterns.add(matches[0])

    unused_patterns = [
        pattern for pattern in all_patterns if pattern not in matched_patterns
    ]
    if unused_patterns and not allow_unused_patterns:
        sample_paths = [param_path_display(path) for path in param_paths[:3]]
        raise ConfigurationError(
            *(
                f"Patterns did not match any parameter: {unused_patterns}. "
                f"Sample parameter paths: {sample_paths}. "
                "Use allow_unused_patterns=True when intentionally sharing "
                "patterns across architectures.",
            )
        )

    used_groups = set(groups.values())
    values = {
        pat: float(val) for pat, val in all_patterns.items() if pat in used_groups
    }
    if fallback is not None and "fallback" in used_groups:
        values["fallback"] = float(fallback)
    return _PerGroup(groups=groups, values=values)


__all__ = ["per_group"]
