"""Paired-stream MF noise (first moment + private second moment).

The runtime per-stream stddev allocation lives in
``opake.api.engine.noise_allocation.paired_noise_stddevs``; this module
carries the joint state and the paired-stream factory that splits the
two strategies into independent noise streams sharing a common joint
Mahalanobis budget.
"""

from __future__ import annotations

import dataclasses
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import torch

from opake.api.engine.noise_allocation import paired_noise_stddevs
from opake.exceptions import InputTypeError
from opake.random import fold_in as rng_fold_in
from opake.types import (
    NoisedPytree,
    NoiseState,
    SecondMomentClippingOutput,
    SecondMomentNoiseOutput,
)

from ._engine import MFNoiseState, _expect_clipped, _validate_constant_max_norm

if TYPE_CHECKING:
    from collections.abc import Callable

    from opake.random.types import RngKey

    from .types import MfStrategy

# Integers `0` and `1` would alias `split(key, 2)`. See ``docs/reference/rng.md``.
SECOND_MOMENT_FIRST_STREAM_FOLD = "opake.dpftrl.second_moment.first"
SECOND_MOMENT_SECOND_STREAM_FOLD = "opake.dpftrl.second_moment.second"


@dataclasses.dataclass(frozen=True)
class SecondMomentMFNoiseState(NoiseState):
    """Internal state for MF noise with private second moments."""

    _first_state: MFNoiseState
    _second_state: MFNoiseState

    @property
    def _step_counter(self) -> int:  # type: ignore[override]
        return self._first_state._step_counter

    @property
    def _rng_key(self) -> RngKey:  # type: ignore[override]
        return self._first_state._rng_key


def make_second_moment_mf_noise(
    grad_template: Any,
    first_strategy: MfStrategy,
    second_strategy: MfStrategy,
    *,
    n_steps: int,
    min_sep: int,
    max_participations: int | None,
    noise_multiplier: float,
    key: RngKey,
    compute_dtype: torch.dtype = torch.float32,
) -> tuple[
    Callable[
        [Any, SecondMomentMFNoiseState],
        tuple[SecondMomentNoiseOutput, SecondMomentMFNoiseState],
    ],
    SecondMomentMFNoiseState,
]:
    """Build the paired-stream noise function.

    Both strategies are built at the same participation context; the
    joint Mahalanobis budget is calibrated as ``(c1/nm)²`` so the joint
    PLD matches the single-stream MF Gaussian accountant at
    ``(nm, c1)`` where ``c1`` is the first strategy's max column norm.
    """
    # Imported here to avoid a top-level cycle: distributed → engine → ...
    from ._distributed import mf_per_group_sync_fingerprint_for_latch
    from ._mf_gaussian_noise import _make_raw_mf_noise

    first_fn, first_state, first_row_l2_at = _make_raw_mf_noise(
        grad_template,
        first_strategy,
        n_steps=n_steps,
        min_sep=min_sep,
        max_participations=max_participations,
        key=rng_fold_in(key, SECOND_MOMENT_FIRST_STREAM_FOLD),
        compute_dtype=compute_dtype,
    )
    second_fn, second_state, second_row_l2_at = _make_raw_mf_noise(
        grad_template,
        second_strategy,
        n_steps=n_steps,
        min_sep=min_sep,
        max_participations=max_participations,
        key=rng_fold_in(key, SECOND_MOMENT_SECOND_STREAM_FOLD),
        compute_dtype=compute_dtype,
    )

    init_state = SecondMomentMFNoiseState(
        _first_state=first_state,
        _second_state=second_state,
    )

    def noise_fn(
        clipped_input: Any,
        st: SecondMomentMFNoiseState,
    ) -> tuple[SecondMomentNoiseOutput, SecondMomentMFNoiseState]:
        if not isinstance(clipped_input, SecondMomentClippingOutput):
            raise InputTypeError(
                *(
                    "mf_gaussian_noise was constructed with `second_moment_strategy` "
                    "and expects SecondMomentClippingOutput inputs (paired-stream).  "
                    "Build the paired form upstream via "
                    "`clipped_grad(..., second_moment=True)`, or rebuild the noise "
                    "function without `second_moment_strategy` for single-stream mode.",
                )
            )
        first_clipped = _expect_clipped(clipped_input.grads, op="mf_gaussian_noise")
        second_clipped = _expect_clipped(
            clipped_input.squared_grads, op="mf_gaussian_noise (squared stream)"
        )
        max_norm = _validate_constant_max_norm(
            first_clipped, st._first_state._first_max_norm, op="mf_gaussian_noise"
        )
        squared_max_norm = _validate_constant_max_norm(
            second_clipped,
            st._second_state._first_max_norm,
            op="mf_gaussian_noise (squared stream)",
        )
        # Max column norm = single-participation sensitivity at this horizon.
        # Reach through the polymorphic sensitivity surface so each strategy
        # uses its own closed-form (Identity = 1, BLT = ‖C‖_{1→2}, etc.).
        c1 = first_strategy.sensitivity(
            n_steps=n_steps, min_sep=n_steps, max_participations=1
        )
        c2 = second_strategy.sensitivity(
            n_steps=n_steps, min_sep=n_steps, max_participations=1
        )
        first_stddev, second_stddev = paired_noise_stddevs(
            noise_multiplier / c1,
            first=max_norm * c1,
            second=squared_max_norm * c2,
        )
        noisy_grads, new_first = first_fn(
            first_clipped.pytree,
            st._first_state,
            stddev=first_stddev,
        )
        noisy_squared, new_second = second_fn(
            second_clipped.pytree,
            st._second_state,
            stddev=second_stddev,
        )
        # Realized per-step σ on each stream = base σ · ‖row_t(C^-1)‖.
        # The streams share the same step counter (advanced inside each
        # ``*_fn`` above); read off the pre-increment step.
        step = st._first_state._step_counter
        first_realized_stddev = first_stddev * first_row_l2_at(step)
        second_realized_stddev = second_stddev * second_row_l2_at(step)
        sync_fp_first = mf_per_group_sync_fingerprint_for_latch(
            st._first_state, max_norm
        )
        sync_fp_second = mf_per_group_sync_fingerprint_for_latch(
            st._second_state, squared_max_norm
        )
        return (
            SecondMomentNoiseOutput(
                NoisedPytree(
                    pytree=noisy_grads,
                    max_norm=first_clipped.max_norm,
                    noise_stddev=first_realized_stddev,
                ),
                NoisedPytree(
                    pytree=noisy_squared,
                    max_norm=second_clipped.max_norm,
                    noise_stddev=second_realized_stddev,
                ),
            ),
            SecondMomentMFNoiseState(
                _first_state=replace(
                    new_first,
                    _first_max_norm=max_norm,
                    _first_max_norm_sync_fingerprint=sync_fp_first,
                ),
                _second_state=replace(
                    new_second,
                    _first_max_norm=squared_max_norm,
                    _first_max_norm_sync_fingerprint=sync_fp_second,
                ),
            ),
        )

    return noise_fn, init_state


__all__ = ["SecondMomentMFNoiseState", "make_second_moment_mf_noise"]
