"""Shared chain composer for functional optimizers.

Builds a ``torchopt`` ``GradientTransformation`` chain on top of a
moment-scaling primitive (``scale_by_adam`` for AdamW, ``sign-of-momentum``
for Lion, …).  The composer decides where weight decay attaches (decoupled
post-moment vs. L2 pre-moment), whether to clip the update by its RMS
(StableAdamW), and applies the (negative) learning rate at the end.

The composer understands the public clipped/noised metadata wrappers and routes
their DP metadata into the moment scaler when the scaler accepts it.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from opake.exceptions import InputTypeError

try:
    import torchopt
    from torchopt.alias.utils import scale_by_neg_lr
    from torchopt.base import GradientTransformation
except ImportError as exc:
    raise ImportError(  # noqa: TRY003 - preserve standard Python error contract
        "torchopt is required for opake.optimizers. "
        "Install it with: pip install 'torchopt>=0.7.3'"
    ) from exc

import torch

from opake.pytree import tree_map
from opake.types import ClippedPytree, NoisedPytree, SecondMomentNoiseOutput

_LR = float | Callable[[int], float]


def _rms_clip_transform(threshold: float) -> GradientTransformation:
    """Per-update RMS clip used by StableAdamW.

    Computes ``rms = sqrt(mean(update**2))`` over the entire update pytree
    (concatenated leaves) and divides the update by ``max(1, rms / threshold)``.
    No state.
    """

    def init_fn(params: Any) -> tuple:
        del params
        return ()

    def update_fn(
        updates: Any,
        state: tuple,
        *,
        params: Any = None,
        inplace: bool = False,
    ) -> tuple[Any, tuple]:
        return _clip_by_global_rms(updates, threshold), state

    return GradientTransformation(init_fn, update_fn)


def _clip_by_global_rms(updates: Any, threshold: float) -> Any:
    """Apply one parameter-count-weighted RMS scale to all tensor leaves."""
    # Global RMS across all leaves (param-count-weighted mean of squares).
    sq_sum: torch.Tensor | None = None
    count = 0
    for leaf in _iter_leaves(updates):
        accumulator_dtype = (
            torch.float32 if leaf.device.type == "mps" else torch.float64
        )
        leaf_sq_sum = leaf.detach().to(accumulator_dtype).pow(2).sum()
        sq_sum = leaf_sq_sum if sq_sum is None else sq_sum + leaf_sq_sum
        count += leaf.numel()
    if sq_sum is None:
        return updates
    rms = (sq_sum / count).sqrt()
    scale = torch.clamp(rms / threshold, min=1.0)
    return tree_map(lambda u: u / scale.to(dtype=u.dtype), updates)


def _iter_leaves(tree: Any):
    """Yield tensor leaves of a (possibly nested) pytree."""
    if isinstance(tree, torch.Tensor):
        yield tree
        return
    if isinstance(tree, dict):
        for v in tree.values():
            yield from _iter_leaves(v)
        return
    if isinstance(tree, (list, tuple)):
        for v in tree:
            yield from _iter_leaves(v)
        return
    # Non-tensor leaves are silently ignored — same convention as
    # ``opake.pytree.tree_leaves``.


def make_optimizer_chain(
    moment_scaler: GradientTransformation,
    lr: _LR,
    weight_decay: float,
    *,
    decoupled_weight_decay: bool = True,
    update_rms_clip: float | None = None,
) -> GradientTransformation:
    """Compose a moment-scaling primitive into a full optimizer.

    Layout (decoupled / AdamW style)::

        moment_scaler -> [rms_clip] -> add_decayed_weights -> scale_by_neg_lr

    Layout (L2 / Adam style)::

        add_decayed_weights -> moment_scaler -> [rms_clip] -> scale_by_neg_lr

    The L2 form folds ``wd * params`` into the gradient *before* moment
    scaling, so weight decay enters the EMAs.  The decoupled form leaves
    moment scaling on raw gradients and applies weight decay to the
    update post-moment-scaling but pre-LR-scaling — this is the standard
    AdamW recipe (Loshchilov & Hutter).

    ``update_rms_clip`` (StableAdamW): when set, divides the update by
    ``max(1, rms / threshold)`` after moment scaling and before WD/LR, where
    ``rms`` is computed model-wide across all tensor leaves. The clip applies
    only to the moment-scaled portion of the update, not to the weight-decay
    term.

    The returned ``GradientTransformation`` extracts DP metadata from
    ``NoisedPytree`` / ``SecondMomentNoiseOutput`` updates and threads
    it into the moment scaler internally — there is no public per-step
    metadata kwarg.
    """
    moment_update_params = inspect.signature(moment_scaler.update).parameters
    accepts_noise_stddev = "noise_stddev" in moment_update_params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in moment_update_params.values()
    )
    accepts_second_moment = "noisy_squared_grads" in moment_update_params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in moment_update_params.values()
    )

    def _route_noisy_pytree(updates: Any) -> tuple[Any, dict[str, Any]]:
        if isinstance(updates, NoisedPytree):
            if not accepts_noise_stddev:
                return updates.pytree, {}
            return updates.pytree, {"noise_stddev": updates.noise_stddev}
        if isinstance(updates, ClippedPytree):
            raise InputTypeError(
                *(
                    "optimizer.update() received ClippedPytree updates that have not "
                    "passed through a noise mechanism. Pass NoisedPytree outputs from "
                    "a DP mechanism, or unwrap `.pytree` explicitly for non-private use.",
                )
            )
        return updates, {}

    def _unwrap_second_moment_value(value: Any, *, name: str) -> Any:
        if isinstance(value, NoisedPytree):
            return value.pytree
        if isinstance(value, ClippedPytree):
            raise InputTypeError(
                *(
                    f"SecondMomentNoiseOutput.{name} is a ClippedPytree that has not "
                    "passed through a noise mechanism.",
                )
            )
        return value

    def _route_second_moment_output(updates: Any) -> tuple[Any, dict[str, Any]]:
        if not isinstance(updates, SecondMomentNoiseOutput):
            return _route_noisy_pytree(updates)
        if not accepts_second_moment:
            return _route_noisy_pytree(updates.noisy_grads)
        return (
            _unwrap_second_moment_value(updates.noisy_grads, name="noisy_grads"),
            {
                "noisy_squared_grads": _unwrap_second_moment_value(
                    updates.noisy_squared_grads, name="noisy_squared_grads"
                )
            },
        )

    wd = torchopt.transform.add_decayed_weights(weight_decay=weight_decay)
    neg_lr = scale_by_neg_lr(lr)
    clip = (
        _rms_clip_transform(float(update_rms_clip))
        if update_rms_clip is not None
        else None
    )

    if decoupled_weight_decay:

        def init_fn(params: Any) -> tuple:
            return (
                moment_scaler.init(params),
                clip.init(params) if clip is not None else (),
                wd.init(params),
                neg_lr.init(params),
            )

        def update_fn(
            updates: Any,
            state: tuple,
            *,
            params: Any = None,
            inplace: bool = False,
        ) -> tuple[Any, tuple]:
            s_mom, s_clip, s_wd, s_lr = state
            updates, routed = _route_second_moment_output(updates)
            updates, s_mom = moment_scaler.update(
                updates, s_mom, params=params, inplace=inplace, **routed
            )
            if clip is not None:
                updates, s_clip = clip.update(
                    updates, s_clip, params=params, inplace=inplace
                )
            updates, s_wd = wd.update(updates, s_wd, params=params, inplace=inplace)
            updates, s_lr = neg_lr.update(updates, s_lr, inplace=inplace)
            return updates, (s_mom, s_clip, s_wd, s_lr)

    else:
        # L2 form: wd is added to gradient before the moment scaler.
        def init_fn(params: Any) -> tuple:
            return (
                wd.init(params),
                moment_scaler.init(params),
                clip.init(params) if clip is not None else (),
                neg_lr.init(params),
            )

        def update_fn(  # type: ignore[misc]
            updates: Any,
            state: tuple,
            *,
            params: Any = None,
            inplace: bool = False,
        ) -> tuple[Any, tuple]:
            s_wd, s_mom, s_clip, s_lr = state
            updates, routed = _route_second_moment_output(updates)
            updates, s_wd = wd.update(updates, s_wd, params=params, inplace=inplace)
            updates, s_mom = moment_scaler.update(
                updates, s_mom, params=params, inplace=inplace, **routed
            )
            if clip is not None:
                updates, s_clip = clip.update(
                    updates, s_clip, params=params, inplace=inplace
                )
            updates, s_lr = neg_lr.update(updates, s_lr, inplace=inplace)
            return updates, (s_wd, s_mom, s_clip, s_lr)

    return GradientTransformation(init_fn, update_fn)


__all__ = ["make_optimizer_chain"]
