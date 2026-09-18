"""DP noise-variance bias correction (φ-EMA).

Shared primitive for any second-moment optimizer (Adam, AdEMAMix,
Adafactor, …) that wants to subtract the noise variance from its
``v̂_t`` estimate under DP-SGD.

Reference:
    Chooi et al., "DP-AdamW: Investigating Decoupled Weight Decay and
    Bias Correction in Private Deep Learning", arXiv:2511.07843.

Math.  Let ``Φ_t = σ_t²`` be the per-step noise *variance*.  The
``φ`` EMA tracks the same β₂-weighted average as the second-moment
EMA::

    φ_t = β₂ φ_{t-1} + (1 − β₂) Φ_t

After bias correction by ``1 − β₂^t``, ``φ̂_t = φ_t / (1 − β₂^t)`` is
an unbiased estimate of the noise-variance contribution to ``v̂_t``::

    v̂_corrected = v̂_t − φ̂_t   where that is positive
                  v̂_t         elsewhere

Coordinates where the noise estimate has overtaken the second moment are
left *uncorrected* rather than clamped to a small positive floor.  Their
signal-variance estimate is non-positive and therefore meaningless;
flooring would divide by ``≈ ε`` and amplify pure noise by ~1/ε, whereas
falling back to ``v̂_t`` degrades them to the non-private behavior, which
is bounded.  The additive ``ε`` in ``√v̂_corrected + ε`` is what keeps the
denominator away from zero.

The separate ``bc_floor`` clamp seen in some optimizers belongs to the
*private second-moment* branch (``noisy_squared_grads``), where ``v̂``
comes from an externally privatised ``g²`` stream and can be genuinely
negative — not to this φ-EMA path.

This module is mechanism-agnostic.  ``noise_stddev`` is a number (or a
``PerGroup`` of numbers) supplied by the caller; the noise *generation*
lives elsewhere (``opake.dpsgd.noise``, ``opake.dpftrl.noise``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opake.api.engine.pytree import ParamPath, tree_flatten_with_paths
from opake.exceptions import ConfigurationError, InputTypeError
from opake.types import PerGroup

if TYPE_CHECKING:
    from collections.abc import Iterator


def map_leaves_with_path(
    fn: Any,
    tree: Any,
    *others: Any,
) -> Any:
    """Apply ``fn(path, leaf, *other_leaves)`` and rebuild ``tree``'s structure.

    All trees must share the same :data:`~opake.pytree.ParamPath` sequence
    (same leaf order and structure).  Matching leaf *counts* alone is not
    enough — e.g. a length-2 list vs a 2-key dict would otherwise silently
    mis-align leaves.
    """
    import optree

    paths, leaves, treedef = tree_flatten_with_paths(tree)
    other_flat = [tree_flatten_with_paths(t) for t in others]
    for i, (other_paths, other_leaves, _) in enumerate(other_flat):
        if other_paths != paths:
            raise ConfigurationError(
                *(
                    f"pytree ParamPath mismatch for argument {i}: "
                    f"primary paths {paths!r}, got {other_paths!r}.",
                )
            )
        if len(other_leaves) != len(leaves):
            raise ConfigurationError(
                *(
                    f"pytree leaf count mismatch: primary has {len(leaves)}, "
                    f"argument {i} has {len(other_leaves)}",
                )
            )
    out_leaves = []
    for j, path in enumerate(paths):
        args = [leaves[j], *[flat[1][j] for flat in other_flat]]
        out_leaves.append(fn(path, *args))
    return optree.tree_unflatten(treedef, out_leaves)


def walk_param_leaves(tree: Any) -> Iterator[tuple[ParamPath, Any]]:
    """Yield ``(ParamPath, leaf)`` pairs for every leaf in ``tree``.

    Uses :func:`~opake.pytree.tree_flatten_with_paths` so paths match
    :class:`~opake.types.PerGroup.groups` keys (flat ``named_parameters``
    are one-segment paths; nested trees are multi-segment).
    """
    paths, leaves, _ = tree_flatten_with_paths(tree)
    yield from zip(paths, leaves, strict=True)


def init_per_group_phi(params: Any) -> dict[ParamPath, float]:
    """Initial φ-EMA dict: ``0.0`` per leaf path of ``params``."""
    return {path: 0.0 for path, _ in walk_param_leaves(params)}


def resolve_noise_variance(
    noise_stddev: float | PerGroup,
    path: ParamPath | str | None = None,
) -> float:
    """Square a (possibly per-group) noise stddev to get its variance.

    When ``noise_stddev`` is a :class:`PerGroup`, ``path`` selects the
    leaf's group; the per-path value is squared.  When it's a plain
    float, ``path`` is ignored.
    """
    if isinstance(noise_stddev, PerGroup):
        if path is None:
            raise ConfigurationError(
                *("resolve_noise_variance requires `path` for PerGroup noise_stddev",)
            )
        return float(noise_stddev.for_path(path)) ** 2
    return float(noise_stddev) ** 2


def is_per_group(noise_stddev: float | PerGroup) -> bool:
    """Return ``True`` iff ``noise_stddev`` is a :class:`PerGroup`."""
    return isinstance(noise_stddev, PerGroup)


def update_phi_ema(
    phi: float | dict[ParamPath, float],
    new_variance: float | dict[ParamPath, float],
    b2: float,
) -> float | dict[ParamPath, float]:
    """Advance the noise-variance EMA by one step::

        φ_t = β₂ φ_{t-1} + (1 − β₂) Φ_t

    Both arguments are scalars in the homogeneous case; in the per-group
    case both are path-keyed dicts.  Mixed shapes raise.
    """
    if isinstance(phi, dict):
        if not isinstance(new_variance, dict):
            raise InputTypeError(
                *(
                    "phi is per-group dict but new_variance is scalar; "
                    "either both must be per-group or both must be scalar.",
                )
            )
        return {k: b2 * phi[k] + (1 - b2) * new_variance[k] for k in phi}
    return b2 * phi + (1 - b2) * float(new_variance)


# Back-compat alias used by a few call sites / older comments.
walk_dict_leaves = walk_param_leaves


__all__ = [
    "init_per_group_phi",
    "is_per_group",
    "map_leaves_with_path",
    "resolve_noise_variance",
    "update_phi_ema",
    "walk_dict_leaves",
    "walk_param_leaves",
]
