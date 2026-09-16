"""Gaussian noise generation for differential privacy.

This module provides a higher-order function for adding calibrated Gaussian
noise to clipped DP query values, optionally bounded to a closed interval
following the *bounded Gaussian mechanism* of Chen and Hale, "The Bounded
Gaussian Mechanism for Differential Privacy," J. Privacy and Confidentiality,
14(1), 2024 (https://arxiv.org/abs/2211.17230).

The API returns ``(noise_fn, state)`` where state is always immutable:

    >>> from opaque.random import key
    >>> from opaque.types import clipped
    >>> from opaque.dpsgd.noise import gaussian_noise
    >>> noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(42))
    >>> noisy_grads, state = noise_fn(clipped(grads, max_norm=1.0), state)

The constructor takes a noise multiplier, not a raw standard deviation.
Per-step sensitivity flows through the input ``ClippedPytree.max_norm`` metadata,
and the returned ``NoisedPytree`` carries the realized ``noise_stddev`` for
downstream optimizers.

Pass ``bound=B`` (or ``bound=(low, high)``) to confine the per-coordinate
output to ``[-B, B]`` (or ``[low, high]``); the noise is sampled from a
Gaussian renormalized over that interval via the inverse-CDF method.  At
training scale the per-coordinate analysis of the paper's bounded mechanism
does not apply. Bounded mode is experimental; the standard ``(ε, δ)``-Gaussian
accountant does not cover it.

The noise function is **purely local** — it uses exactly the key you provide.
For synchronized noise in distributed training, pass the same key on every rank.
For independent noise, derive a per-rank key with ``fold_in(key, rank)``.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import torch
from torch.autograd.profiler import record_function

from opaque.api.engine.noise_allocation import per_group_noise_stddev
from opaque.exceptions import ConfigurationError, InputTypeError
from opaque.pytree import tree_map
from opaque.random import fold_in as rng_fold_in
from opaque.random import generator_from_key
from opaque.random.types import RngKey
from opaque.types import (
    ClippedPytree,
    NoisedPytree,
    NoiseState,
    PerGroup,
)

if TYPE_CHECKING:
    from opaque.api.dpsgd.noise._types import GaussianNoiseFn

# Root of every key this mechanism derives. Distinct registered components use
# separate domains; instances of this component need distinct caller keys when
# their streams must differ.
GAUSSIAN_STREAM_FOLD = "opaque.dpsgd.gaussian"

_SQRT2 = math.sqrt(2.0)


@dataclasses.dataclass(frozen=True)
class GaussianNoiseState(NoiseState):
    """Immutable state for Gaussian noise generation.

    Holds an immutable RNG key for deterministic per-step derivation.
    Noise for step ``t`` is generated from
    ``fold_in(_rng_key, GAUSSIAN_STREAM_FOLD, t)``.

    Attributes:
        _step_counter: Number of noise_fn calls made.
        _rng_key: Immutable RNG key for deterministic derivation.
    """

    _step_counter: int
    _rng_key: RngKey


def _validate_noise_stddev(noise_stddev: float | PerGroup) -> None:
    """Validate that realized noise standard deviation is non-negative."""
    if isinstance(noise_stddev, PerGroup):
        for gname, value in noise_stddev.values.items():
            if value < 0:
                raise ConfigurationError(
                    *(
                        "noise standard deviation must be non-negative for all groups, "
                        f"got {value} for group '{gname}'",
                    )
                )
    else:
        if noise_stddev < 0:
            raise ConfigurationError(
                *(f"noise standard deviation must be non-negative, got {noise_stddev}",)
            )


def _resolve_noise_multiplier(noise_multiplier: float | None) -> float:
    if noise_multiplier is None:
        raise ConfigurationError(*("gaussian_noise() requires noise_multiplier.",))
    multiplier = float(noise_multiplier)
    if multiplier < 0:
        raise ConfigurationError(
            *(f"noise_multiplier must be non-negative, got {noise_multiplier}",)
        )
    return multiplier


def _resolve_bound(
    bound: float | tuple[float, float] | list[float] | None,
) -> tuple[float, float] | None:
    """Normalize the user-facing ``bound`` argument to ``(low, high)`` or None.

    Accepts ``None`` (unbounded), a positive scalar ``B`` interpreted as the
    symmetric interval ``[-B, B]``, or a 2-tuple/2-list ``(low, high)`` with
    ``low < high``.  Bounds must straddle zero — the support has to contain
    the unbiased mechanism centre — so ``low <= 0 <= high`` is required.
    """
    if bound is None:
        return None
    if isinstance(bound, (tuple, list)):
        if len(bound) != 2:  # noqa: PLR2004 - bound is the documented (low, high) pair
            raise ConfigurationError(
                *(
                    f"bound must be a 2-tuple (low, high) when given a sequence, "
                    f"got length {len(bound)}",
                )
            )
        low, high = float(bound[0]), float(bound[1])
    else:
        b = float(bound)
        if b <= 0:
            raise ConfigurationError(
                *(f"scalar bound must be positive (interpreted as [-B, B]), got {b}",)
            )
        low, high = -b, b
    if not low < high:
        raise ConfigurationError(
            *(f"bound must satisfy low < high, got ({low}, {high})",)
        )
    if not (low <= 0.0 <= high):
        raise ConfigurationError(
            *(
                "bound must straddle zero (low <= 0 <= high) so the support "
                f"contains the unbiased mechanism centre, got ({low}, {high})",
            )
        )
    return low, high


def gaussian_noise(
    *,
    noise_multiplier: float,
    key: RngKey,
    bound: float | tuple[float, float] | list[float] | None = None,
    compute_dtype: torch.dtype = torch.float32,
) -> tuple[
    GaussianNoiseFn,
    GaussianNoiseState,
]:
    """Create a Gaussian noise function with immutable state.

    Returns ``(noise_fn, state)`` where ``noise_fn`` adds calibrated Gaussian
    noise to :class:`opaque.types.ClippedPytree` inputs and returns updated
    state.  The realized standard deviation is
    ``noise_multiplier * clipped.max_norm``.  The output is a
    :class:`opaque.types.NoisedPytree` carrying that realized
    ``noise_stddev`` metadata.

    When ``bound`` is set, the output is sampled from a Gaussian renormalized
    over the per-coordinate interval and clamped to it — the *bounded
    Gaussian mechanism* of Chen and Hale (2024).

    The noise function uses exactly the ``key`` you provide — no auto-detection
    of distributed state. For synchronized noise in DDP, pass the same key on
    every rank. For independent noise, derive a per-rank key::

        from opaque.random import key, fold_in
        my_key = fold_in(key(42), rank)  # different noise per rank
        noise_fn, state = gaussian_noise(noise_multiplier=1.1, key=my_key)

    Args:
        noise_multiplier: Gaussian noise multiplier.  The realized standard
            deviation is ``noise_multiplier * clipped.max_norm``.
        key: Explicit RNG key for deterministic, functional randomness.
            Same key on all ranks → same noise (synchronized).
            ``fold_in(key, rank)`` → independent noise per rank.
        bound: Optional per-coordinate output bound.  ``None`` (default) ⇒
            unbounded standard Gaussian.  A positive scalar ``B`` ⇒ symmetric
            interval ``[-B, B]``.  A ``(low, high)`` tuple/list ⇒ asymmetric
            interval; must satisfy ``low <= 0 <= high``.  Units are absolute
            (same scale as the gradient / clip norm), not multiples of σ.
        compute_dtype: Internal inverse-CDF dtype. Defaults to ``torch.float32``;
            finite precision discretizes and bounds the representable tails.
            Output is cast back to the input dtype.

    Returns:
        A tuple ``(noise_fn, state)`` where:

        - ``noise_fn(clipped_grads, state) -> (noisy_grads, new_state)``
        - ``state`` is a :class:`GaussianNoiseState`

    Example:
        >>> import torch
        >>> from opaque.types import clipped
        >>> from opaque.dpsgd.noise import gaussian_noise
        >>> from opaque.random import key
        >>>
        >>> noise_fn, state = gaussian_noise(noise_multiplier=1.1, key=key(42))
        >>> grads = torch.zeros(10)
        >>> noisy_grads, state = noise_fn(clipped(grads, max_norm=1.0), state)

    Example (bounded output — symmetric ``[-3, 3]``):
        >>> noise_fn, state = gaussian_noise(
        ...     noise_multiplier=1.0, bound=3.0, key=key(42),
        ... )

    Example (bounded output — asymmetric ``[0, 10]``):
        >>> noise_fn, state = gaussian_noise(
        ...     noise_multiplier=1.0, bound=(0.0, 10.0), key=key(42),
        ... )

    References:
        Bo Chen and Matthew Hale, "The Bounded Gaussian Mechanism for
        Differential Privacy," J. Privacy and Confidentiality, 14(1), 2024.
        https://arxiv.org/abs/2211.17230
    """
    resolved_noise_multiplier = _resolve_noise_multiplier(noise_multiplier)
    resolved_bound = _resolve_bound(bound)

    if not isinstance(key, RngKey):
        raise InputTypeError(*(f"key must be RngKey, got {type(key)}",))

    state = GaussianNoiseState(
        _step_counter=0,
        _rng_key=key,
    )

    def _sample(
        center: torch.Tensor, std: float, generator: torch.Generator
    ) -> torch.Tensor:
        """Sample N(center, std²) by inverse CDF, optionally truncated.

        ``compute_dtype`` upcast/downcast preserves the type-stable boundary;
        uniforms are drawn on the CPU generator and moved to the input device.
        """
        in_dtype = center.dtype
        device = center.device

        if std == 0:
            if resolved_bound is None:
                return center
            low, high = resolved_bound
            return torch.clamp(center, min=low, max=high)

        u = torch.rand(center.shape, dtype=compute_dtype, generator=generator).to(
            device=device
        )

        center_c = center.to(compute_dtype) if in_dtype != compute_dtype else center

        # Clamp ``u`` away from 0 and 1 so ``2u-1`` never reaches ±1 and
        # ``erfinv`` can't return ±inf.  ``finfo.tiny`` (denormal min) is
        # below ``compute_dtype`` machine eps, so ``1 - tiny`` rounds back to
        # ``1.0`` and the upper clamp would be a no-op — use ``finfo.eps``.
        eps = torch.finfo(compute_dtype).eps
        if resolved_bound is None:
            # Unbounded N(0, 1) via inverse CDF: μ + σ √2 erfinv(2u - 1)
            u = torch.clamp(u, min=eps, max=1.0 - eps)
            sample = center_c + std * _SQRT2 * torch.erfinv(2.0 * u - 1.0)
        else:
            low, high = resolved_bound
            z_low = (low - center_c) / std
            z_high = (high - center_c) / std
            alpha = 0.5 * (1.0 + torch.erf(z_low / _SQRT2))
            beta = 0.5 * (1.0 + torch.erf(z_high / _SQRT2))
            u = alpha + u * (beta - alpha)
            u = torch.clamp(u, min=eps, max=1.0 - eps)
            sample = center_c + std * _SQRT2 * torch.erfinv(2.0 * u - 1.0)
            sample = torch.clamp(sample, min=low, max=high)

        return sample.to(dtype=in_dtype) if in_dtype != compute_dtype else sample

    def _add_noise_tree(grads, effective_stddev, generator):
        _validate_noise_stddev(effective_stddev)

        # Per-group σ: look up each leaf by optree ParamPath.
        if isinstance(effective_stddev, PerGroup):
            import optree

            from opaque.api.engine.pytree import tree_flatten_with_paths

            paths, leaves, treedef = tree_flatten_with_paths(grads)
            noised_leaves = []
            for path, tensor in zip(paths, leaves, strict=True):
                if not isinstance(tensor, torch.Tensor):
                    raise InputTypeError(
                        *(
                            "gaussian_noise with PerGroup stddev expects tensor "
                            f"leaves; got {type(tensor).__name__} at path {path!r}.",
                        )
                    )
                group_std = effective_stddev.for_path(path)
                noised_leaves.append(_sample(tensor, group_std, generator))
            return optree.tree_unflatten(treedef, noised_leaves)

        return tree_map(lambda t: _sample(t, effective_stddev, generator), grads)

    def _clipped_stddev(clipped: ClippedPytree) -> float | PerGroup:
        if isinstance(clipped.max_norm, PerGroup):
            return per_group_noise_stddev(clipped.max_norm, resolved_noise_multiplier)
        # Non-private run: force std = 0 before the ``nm * max_norm`` product so
        # a disabled clip (max_norm = +inf) gives ``0`` rather than ``0 * inf =
        # NaN`` (the ``std == 0`` short-circuit in ``_sample`` then fires).
        effective = (
            0.0
            if resolved_noise_multiplier == 0.0
            else resolved_noise_multiplier * clipped.max_norm
        )
        _validate_noise_stddev(effective)
        return effective

    def noise_fn(grads, st):
        """Add Gaussian noise to a clipped pytree."""
        with record_function("opaque::gaussian_noise"):
            return _noise_fn_impl(grads, st)

    def _noise_fn_impl(grads, st):
        next_state = GaussianNoiseState(
            _step_counter=st._step_counter + 1,
            _rng_key=st._rng_key,
        )

        if isinstance(grads, NoisedPytree):
            raise InputTypeError(
                *(
                    "gaussian_noise expects ClippedPytree inputs, not NoisedPytree "
                    "values that have already passed through a noise mechanism.",
                )
            )

        if not isinstance(grads, ClippedPytree):
            raise InputTypeError(
                *(
                    "gaussian_noise expects ClippedPytree inputs. Wrap manual "
                    "values with opaque.types.clipped(...).",
                )
            )

        effective_stddev = _clipped_stddev(grads)
        step_key = rng_fold_in(st._rng_key, GAUSSIAN_STREAM_FOLD, st._step_counter)
        noisy_tree = _add_noise_tree(
            grads.pytree,
            effective_stddev,
            generator_from_key(step_key),
        )
        return (
            NoisedPytree(
                pytree=noisy_tree,
                max_norm=grads.max_norm,
                noise_stddev=effective_stddev,
            ),
            next_state,
        )

    return noise_fn, state


__all__ = ["GaussianNoiseState", "gaussian_noise"]
