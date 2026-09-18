"""Gaussian noise generation for differential privacy.

This module provides a higher-order function for adding calibrated Gaussian
noise to clipped DP query values.

The API returns ``(noise_fn, state)`` where state is always immutable:

    >>> from opake.random import key
    >>> from opake.types import clipped
    >>> from opake.dpsgd.noise import gaussian_noise
    >>> noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(42))
    >>> noisy_grads, state = noise_fn(clipped(grads, max_norm=1.0), state)

The constructor takes a noise multiplier, not a raw standard deviation.
Per-step sensitivity flows through the input ``ClippedPytree.max_norm`` metadata,
and the returned ``NoisedPytree`` carries the realized ``noise_stddev`` for
downstream optimizers.

The noise function is **purely local** — it uses exactly the key you provide.
For synchronized noise in distributed training, pass the same key on every rank.
For independent noise, derive a per-rank key with ``fold_in(key, rank)``.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch
from torch.autograd.profiler import record_function

from opake.api.engine.noise_allocation import (
    PAIRED_FIRST_STREAM_FOLD,
    PAIRED_SECOND_STREAM_FOLD,
    per_group_noise_stddev,
    resolve_paired_clipped,
)
from opake.exceptions import ConfigurationError, InputTypeError
from opake.pytree import tree_map
from opake.random import fold_in as rng_fold_in
from opake.random import generator_from_key
from opake.random.types import RngKey
from opake.types import (
    ClippedPytree,
    NoisedPytree,
    NoiseState,
    PerGroup,
    SecondMomentClippingOutput,
    SecondMomentNoiseOutput,
)

if TYPE_CHECKING:
    from opake.api.dpsgd.noise._types import GaussianNoiseFn

# Root of every key this mechanism derives. Distinct registered components use
# separate domains; instances of this component need distinct caller keys when
# their streams must differ.
GAUSSIAN_STREAM_FOLD = "opake.dpsgd.gaussian"

_SUPPORTED_COMPUTE_DTYPES = frozenset(
    {torch.float16, torch.bfloat16, torch.float32, torch.float64}
)


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


def _validate_compute_dtype(compute_dtype: torch.dtype) -> None:
    if not isinstance(compute_dtype, torch.dtype):
        raise InputTypeError(
            *(f"compute_dtype must be torch.dtype, got {type(compute_dtype).__name__}",)
        )
    if compute_dtype not in _SUPPORTED_COMPUTE_DTYPES:
        raise ConfigurationError(
            *(
                "compute_dtype must be one of torch.float16, torch.bfloat16, "
                f"torch.float32, or torch.float64; got {compute_dtype}.",
            )
        )


def gaussian_noise(
    *,
    noise_multiplier: float,
    key: RngKey,
    compute_dtype: torch.dtype = torch.float32,
) -> tuple[
    GaussianNoiseFn,
    GaussianNoiseState,
]:
    """Create a Gaussian noise function with immutable state.

    Returns ``(noise_fn, state)`` where ``noise_fn`` adds calibrated Gaussian
    noise to :class:`opake.types.ClippedPytree` inputs and returns updated
    state.  The realized standard deviation is
    ``noise_multiplier * clipped.max_norm``.  The output is a
    :class:`opake.types.NoisedPytree` carrying that realized
    ``noise_stddev`` metadata.

    The noise function uses exactly the ``key`` you provide — no auto-detection
    of distributed state. For synchronized noise in DDP, pass the same key on
    every rank. For independent noise, derive a per-rank key::

        from opake.random import key, fold_in
        my_key = fold_in(key(42), rank)  # different noise per rank
        noise_fn, state = gaussian_noise(noise_multiplier=1.1, key=my_key)

    Args:
        noise_multiplier: Gaussian noise multiplier.  The realized standard
            deviation is ``noise_multiplier * clipped.max_norm``.
        key: Explicit RNG key for deterministic, functional randomness.
            Same key on all ranks → same noise (synchronized).
            ``fold_in(key, rank)`` → independent noise per rank.
        compute_dtype: One of ``torch.float16``, ``torch.bfloat16``,
            ``torch.float32``, or ``torch.float64`` for sampling and addition.
            Defaults to ``torch.float32``. Output is cast back to the input dtype.

    Returns:
        A tuple ``(noise_fn, state)`` where:

        - ``noise_fn(clipped_grads, state) -> (noisy_grads, new_state)``
        - ``state`` is a :class:`GaussianNoiseState`

    Example:
        >>> import torch
        >>> from opake.types import clipped
        >>> from opake.dpsgd.noise import gaussian_noise
        >>> from opake.random import key
        >>>
        >>> noise_fn, state = gaussian_noise(noise_multiplier=1.1, key=key(42))
        >>> grads = torch.zeros(10)
        >>> noisy_grads, state = noise_fn(clipped(grads, max_norm=1.0), state)
    """
    resolved_noise_multiplier = _resolve_noise_multiplier(noise_multiplier)
    _validate_compute_dtype(compute_dtype)

    if not isinstance(key, RngKey):
        raise InputTypeError(*(f"key must be RngKey, got {type(key)}",))

    state = GaussianNoiseState(
        _step_counter=0,
        _rng_key=key,
    )

    def _sample(
        center: torch.Tensor, std: float, generator: torch.Generator
    ) -> torch.Tensor:
        """Sample unbounded N(center, std²) with the supplied CPU generator."""
        if std == 0:
            return center

        in_dtype = center.dtype
        noise = torch.randn(
            center.shape,
            dtype=compute_dtype,
            generator=generator,
        ).to(device=center.device)
        center_c = center.to(compute_dtype) if in_dtype != compute_dtype else center
        sample = center_c + std * noise

        return sample.to(dtype=in_dtype) if in_dtype != compute_dtype else sample

    def _add_noise_tree(grads, effective_stddev, generator):
        _validate_noise_stddev(effective_stddev)

        # Per-group σ: look up each leaf by optree ParamPath.
        if isinstance(effective_stddev, PerGroup):
            import optree

            from opake.api.engine.pytree import tree_flatten_with_paths

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

    def _add_paired(
        clipped_input: SecondMomentClippingOutput, st: GaussianNoiseState
    ) -> tuple[SecondMomentNoiseOutput, GaussianNoiseState]:
        first_clipped, second_clipped, first_stddev, second_stddev = (
            resolve_paired_clipped(
                clipped_input,
                noise_multiplier=resolved_noise_multiplier,
            )
        )
        # Two independent noise streams, both beneath this mechanism's root
        # so they cannot collide with each other or with the single-stream
        # derivation (``fold_in(_rng_key, GAUSSIAN_STREAM_FOLD, step)``).
        first_step_key = rng_fold_in(
            st._rng_key,
            GAUSSIAN_STREAM_FOLD,
            PAIRED_FIRST_STREAM_FOLD,
            st._step_counter,
        )
        second_step_key = rng_fold_in(
            st._rng_key,
            GAUSSIAN_STREAM_FOLD,
            PAIRED_SECOND_STREAM_FOLD,
            st._step_counter,
        )
        noisy_grads = _add_noise_tree(
            first_clipped.pytree,
            first_stddev,
            generator_from_key(first_step_key),
        )
        noisy_squared = _add_noise_tree(
            second_clipped.pytree,
            second_stddev,
            generator_from_key(second_step_key),
        )
        next_state = GaussianNoiseState(
            _step_counter=st._step_counter + 1,
            _rng_key=st._rng_key,
        )
        return (
            SecondMomentNoiseOutput(
                NoisedPytree(
                    pytree=noisy_grads,
                    max_norm=first_clipped.max_norm,
                    noise_stddev=first_stddev,
                ),
                NoisedPytree(
                    pytree=noisy_squared,
                    max_norm=second_clipped.max_norm,
                    noise_stddev=second_stddev,
                ),
            ),
            next_state,
        )

    def noise_fn(grads, st):
        """Add Gaussian noise to a clipped pytree (or paired stream)."""
        with record_function("opake::gaussian_noise"):
            return _noise_fn_impl(grads, st)

    def _noise_fn_impl(grads, st):
        if isinstance(grads, SecondMomentClippingOutput):
            return _add_paired(grads, st)

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
                    "values with opake.types.clipped(...).",
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
