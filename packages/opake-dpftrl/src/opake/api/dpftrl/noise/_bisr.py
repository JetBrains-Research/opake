"""BISR strategy — Banded Inverse Square Root MF mechanism.

BISR (Kalinin et al., ICLR 2026) generalises lambda-CGD to arbitrary
bandwidth p.  The inverse strategy matrix :math:`C^{-1}` is banded
Toeplitz with p coefficients.

References:
    - Kalinin, McKenna, Upadhyay, and Lampert (2026), "Back to Square
      Roots: An Optimal Bound on the Matrix Factorization Error for Multi-Epoch
      Differentially Private SGD", https://arxiv.org/abs/2505.12128
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import torch

from opake.api.dpftrl.noise._strategy_codec import register_strategy
from opake.exceptions import CheckpointError, ConfigurationError, InputTypeError
from opake.pytree import tree_flatten_with_paths, tree_map
from opake.serialization import from_state_dict, register_serializer, state_dict

from ._streaming_matrix import StreamingMatrix
from ._toeplitz import inverse_as_streaming_matrix

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from opake.api.engine.scheduling.types import Schedule
    from opake.random.types import RngKey

    from ._engine import MFNoiseState

_MIN_BANDWIDTH = 2
_MIN_LEADING_COEFFICIENT_MAGNITUDE = 1e-30
_BISR_STREAMING_STATE_VERSION = 1
_BISR_EXECUTION_IDENTITY_VERSION = 1


@dataclass(frozen=True, slots=True)
class _BisrExecutionIdentity:
    """Immutable parameters that determine one bounded BISR execution."""

    version: int
    inverse_coefficients: tuple[float, ...]
    normalized: bool
    n_steps: int
    compute_dtype: str


@dataclass(frozen=True, slots=True)
class _BisrStreamingState:
    """Versioned bounded state for direct BISR inverse convolution."""

    step: int
    history: tuple[Any, ...]
    execution_identity: _BisrExecutionIdentity


@dataclass(frozen=True, slots=True)
class _BisrStreamingPayload:
    """Structural payload delegated to Opake's generic serializer."""

    step: int
    history: tuple[Any, ...]
    execution_identity: _BisrExecutionIdentity


def _save_bisr_streaming_state(value: _BisrStreamingState) -> dict[str, Any]:
    payload = state_dict(
        _BisrStreamingPayload(
            step=value.step,
            history=value.history,
            execution_identity=value.execution_identity,
        )
    )
    return {"layout_version": _BISR_STREAMING_STATE_VERSION, **payload}


def _structure_only_payload_fields(
    payload: _BisrStreamingPayload,
) -> set[str]:
    """Return serializer paths without cloning model-sized tensor leaves."""
    structure_only_history = tree_map(lambda _value: 0, payload.history)
    structure_only = _BisrStreamingPayload(
        step=payload.step,
        history=structure_only_history,
        execution_identity=payload.execution_identity,
    )
    return set(state_dict(structure_only))


def _validate_saved_execution_identity(
    expected: _BisrExecutionIdentity,
    saved: Mapping[str, Any],
) -> None:
    """Reject a checkpoint configured for a different BISR execution."""
    prefix = "execution_identity."
    actual_fields = {
        name[len(prefix) :]: value
        for name, value in saved.items()
        if name.startswith(prefix)
    }
    expected_fields = state_dict(expected)
    mismatched = sorted(
        name
        for name, expected_value in expected_fields.items()
        if name not in actual_fields
        or type(actual_fields[name]) is not type(expected_value)
        or actual_fields[name] != expected_value
    )
    mismatched.extend(sorted(set(actual_fields) - set(expected_fields)))
    if mismatched:
        raise CheckpointError(
            *(
                "BISR checkpoint execution identity does not match the configured "
                f"runtime; mismatched fields={mismatched}. Rebuild the mechanism "
                "with the checkpoint's strategy, horizon, and compute dtype.",
            )
        )


def _load_bisr_streaming_state(
    template: _BisrStreamingState,
    saved: Mapping[str, Any],
) -> _BisrStreamingState:
    if not saved:
        return template

    version = saved.get("layout_version")
    if version is None:
        raise CheckpointError(
            *(
                "Cannot restore a legacy BISR dense-history checkpoint into "
                "the bounded streaming state. Resume it with the Opake "
                "version that created it.",
            )
        )
    if type(version) is not int or version != _BISR_STREAMING_STATE_VERSION:
        raise CheckpointError(
            *(
                f"Unsupported BISR streaming-state version {version!r}; "
                f"expected {_BISR_STREAMING_STATE_VERSION}.",
            )
        )

    payload_template = _BisrStreamingPayload(
        step=template.step,
        history=template.history,
        execution_identity=template.execution_identity,
    )
    # Check immutable execution parameters before inspecting/restoring history.
    _validate_saved_execution_identity(template.execution_identity, saved)

    expected_fields = _structure_only_payload_fields(payload_template)
    actual_fields = set(saved) - {"layout_version"}
    if actual_fields != expected_fields:
        missing = sorted(expected_fields - actual_fields)
        unexpected = sorted(actual_fields - expected_fields)
        raise CheckpointError(
            *(
                "BISR streaming-state fields do not match the current layout: "
                f"missing={missing}, unexpected={unexpected}.",
            )
        )

    restored = from_state_dict(
        payload_template,
        {name: value for name, value in saved.items() if name != "layout_version"},
    )
    if (
        type(restored.step) is not int
        or restored.step < 0
        or restored.step > template.execution_identity.n_steps
    ):
        raise CheckpointError(
            *(
                "BISR streaming-state step must be an int within the configured "
                f"horizon [0, {template.execution_identity.n_steps}], got "
                f"{restored.step!r}.",
            )
        )
    expected_history = len(template.execution_identity.inverse_coefficients) - 1
    if len(restored.history) != expected_history:
        raise CheckpointError(
            *(
                "BISR streaming history length does not match the configured "
                f"bandwidth: got {len(restored.history)}, expected "
                f"{expected_history}.",
            )
        )
    return _BisrStreamingState(
        step=restored.step,
        history=restored.history,
        execution_identity=restored.execution_identity,
    )


register_serializer(
    _BisrStreamingState,
    _save_bisr_streaming_state,
    _load_bisr_streaming_state,
)


def _native():
    from opake.api.accounting.core import _native as _n

    return _n


@lru_cache(maxsize=256)
def _bisr_gram_matrix_cached(
    inv: tuple[float, ...],
    normalized: bool,
    n_steps: int,
    min_sep: int,
    max_participations: int | None,
) -> tuple[float, ...]:
    """Gram sequence for BISR; cached across repeated σ / PLD probes."""
    return tuple(
        _native().bisr_gram_matrix(
            list(inv), n_steps, min_sep, max_participations, normalized
        )
    )


@lru_cache(maxsize=32)
def _bisr_inverse_coefficients_cached(bandwidth: int, beta: float) -> tuple[float, ...]:
    """Compute the alpha=1 BISR coefficients (Lemma 1, arXiv:2505.12128).

    For alpha=1: c_k = sum_{j=0}^{k} r_j * beta^j * r_{k-j}.
    The lemma's 0 <= beta < alpha <= 1 hypothesis therefore becomes
    0 <= beta < 1, enforced by :class:`BisrStrategy`.
    where r_0 = 1, r_j = ((j - 3/2) / j) * r_{j-1}.
    """
    r_tilde = [0.0] * bandwidth
    r_tilde[0] = 1.0
    for j in range(1, bandwidth):
        r_tilde[j] = ((j - 1.5) / j) * r_tilde[j - 1]

    if beta == 0.0:
        return tuple(r_tilde)

    coefs = [0.0] * bandwidth
    for k in range(bandwidth):
        s = 0.0
        for j in range(k + 1):
            s += r_tilde[j] * (beta**j) * r_tilde[k - j]
        coefs[k] = s
    return tuple(coefs)


def _zero_tree_in_dtype(tree: Any, compute_dtype: torch.dtype) -> Any:
    def make_zero(value: Any) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise InputTypeError(
                *(f"BISR noise expects tensor leaves; got {type(value).__name__}.",)
            )
        return torch.zeros_like(value, dtype=compute_dtype)

    return tree_map(make_zero, tree)


def _scale_tree(tree: Any, scale: float) -> Any:
    return tree_map(lambda value: value * scale, tree)


def _add_scaled_tree(left: Any, right: Any, scale: float) -> Any:
    return tree_map(lambda x, y: x + scale * y, left, right)


def _validate_history_tree(
    current_iid: Any,
    previous_iid: Any,
    *,
    history_index: int,
) -> None:
    """Reject direct state reuse that could broadcast or alter tree semantics."""
    current_paths, current_leaves, current_spec = tree_flatten_with_paths(current_iid)
    previous_paths, previous_leaves, previous_spec = tree_flatten_with_paths(
        previous_iid
    )
    if current_spec != previous_spec or current_paths != previous_paths:
        raise CheckpointError(
            *(
                "BISR streaming history PyTree structure does not match the "
                f"current noise tree at history index {history_index}.",
            )
        )
    for path, current, previous in zip(
        current_paths, current_leaves, previous_leaves, strict=True
    ):
        if not isinstance(current, torch.Tensor) or not isinstance(
            previous, torch.Tensor
        ):
            raise CheckpointError(
                *(
                    "BISR streaming history must contain tensor leaves; got "
                    f"current={type(current).__name__}, "
                    f"history={type(previous).__name__} at path {path!r}.",
                )
            )
        if (
            current.shape != previous.shape
            or current.dtype != previous.dtype
            or current.device != previous.device
        ):
            raise CheckpointError(
                *(
                    "BISR streaming history leaf does not match the current noise "
                    f"tree at path {path!r}: current(shape={tuple(current.shape)}, "
                    f"dtype={current.dtype}, device={current.device}), "
                    f"history(shape={tuple(previous.shape)}, "
                    f"dtype={previous.dtype}, device={previous.device}).",
                )
            )


def _direct_bisr_streaming_matrix(
    inverse_coefficients: tuple[float, ...],
    column_scales: tuple[float, ...],
    execution_identity: _BisrExecutionIdentity,
    *,
    compute_dtype: torch.dtype,
) -> StreamingMatrix[_BisrStreamingState]:
    """Apply a banded ``C^{-1}`` directly over a bounded iid-noise history."""
    n_steps = len(column_scales)
    window = len(inverse_coefficients) - 1

    def init_multiply(grad_template: Any) -> _BisrStreamingState:
        return _BisrStreamingState(
            step=0,
            history=tuple(
                _zero_tree_in_dtype(grad_template, compute_dtype) for _ in range(window)
            ),
            execution_identity=execution_identity,
        )

    def multiply_next(
        iid_noise: Any,
        state: _BisrStreamingState,
    ) -> tuple[Any, _BisrStreamingState]:
        if not isinstance(state, _BisrStreamingState):
            raise CheckpointError(
                *(
                    "BISR noise state does not use the bounded streaming layout. "
                    "Resume legacy dense-history checkpoints with the Opake "
                    "version that created them.",
                )
            )
        if (
            type(state.execution_identity) is not _BisrExecutionIdentity
            or state.execution_identity != execution_identity
        ):
            raise CheckpointError(
                *(
                    "BISR noise state execution identity does not match the "
                    "configured strategy, horizon, or compute dtype.",
                )
            )
        if state.step < 0 or state.step >= n_steps:
            raise ConfigurationError(
                *(
                    f"BISR streaming step {state.step} is outside the calibrated "
                    f"horizon [0, {n_steps}).",
                )
            )
        if len(state.history) != window:
            raise CheckpointError(
                *(
                    "BISR streaming history length does not match the configured "
                    f"bandwidth: got {len(state.history)}, expected {window}.",
                )
            )

        count = min(state.step, window)
        column_scale = column_scales[state.step]
        effective_taps = tuple(
            column_scale * tap for tap in inverse_coefficients[: count + 1]
        )
        correlated = _scale_tree(iid_noise, effective_taps[0])
        for history_index, (tap, previous_iid) in enumerate(
            zip(
                effective_taps[1:],
                state.history[:count],
                strict=True,
            ),
            start=1,
        ):
            _validate_history_tree(
                iid_noise,
                previous_iid,
                history_index=history_index,
            )
            correlated = _add_scaled_tree(correlated, previous_iid, tap)

        next_history = (iid_noise, *state.history[: window - 1]) if window else ()
        return correlated, _BisrStreamingState(
            step=state.step + 1,
            history=next_history,
            execution_identity=execution_identity,
        )

    return StreamingMatrix(init_multiply, multiply_next)


def _raise_unrepresentable_effective_coefficient(
    *,
    tap_index: int,
    step: int,
    value: float,
    compute_dtype: torch.dtype,
) -> None:
    raise ConfigurationError(
        *(
            "BISR effective runtime coefficient is not representable in the "
            f"configured compute dtype: tap={tap_index}, step={step}, "
            f"value={value!r}, compute_dtype={compute_dtype}.",
        )
    )


def _validate_effective_runtime_coefficients(
    inverse_coefficients: tuple[float, ...],
    column_scales: tuple[float, ...],
    compute_dtype: torch.dtype,
) -> None:
    """Fail before release if any active effective tap leaves the dtype range."""
    supported_compute_dtypes = {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }
    if compute_dtype not in supported_compute_dtypes:
        raise ConfigurationError(
            *(
                "BISR compute_dtype must be a supported real floating-point "
                f"dtype, got {compute_dtype}.",
            )
        )
    try:
        torch.finfo(compute_dtype)
    except (TypeError, RuntimeError) as exc:
        raise ConfigurationError(
            *(
                "BISR compute_dtype must be a supported floating-point dtype, "
                f"got {compute_dtype}.",
            )
        ) from exc

    last_step = len(column_scales) - 1
    for tap_index, tap in enumerate(inverse_coefficients):
        if tap == 0.0:
            continue
        # Prefix norms only decrease after reversal. For a tap that first
        # becomes active at ``tap_index``, these endpoints therefore cover its
        # largest and smallest effective magnitudes over the whole horizon.
        steps = (tap_index,) if tap_index == last_step else (tap_index, last_step)
        for step in steps:
            effective = column_scales[step] * tap
            if not math.isfinite(effective) or effective == 0.0:
                _raise_unrepresentable_effective_coefficient(
                    tap_index=tap_index,
                    step=step,
                    value=effective,
                    compute_dtype=compute_dtype,
                )
            try:
                cast = torch.tensor(effective, dtype=compute_dtype, device="cpu")
                cast_is_finite = bool(torch.isfinite(cast).item())
                cast_is_zero = cast.item() == 0
            except (TypeError, RuntimeError, ValueError) as exc:
                raise ConfigurationError(
                    *(
                        "BISR could not represent an effective runtime coefficient "
                        f"in compute_dtype={compute_dtype}.",
                    )
                ) from exc
            if not cast_is_finite or cast_is_zero:
                _raise_unrepresentable_effective_coefficient(
                    tap_index=tap_index,
                    step=step,
                    value=effective,
                    compute_dtype=compute_dtype,
                )


def _bisr_runtime_parameters(
    strategy: BisrStrategy,
    n_steps: int,
    compute_dtype: torch.dtype,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    inverse = tuple(float(value) for value in strategy._inv_coefs()[:n_steps])
    if strategy.normalized:
        strategy_coefficients = tuple(
            float(value)
            for value in _native().bisr_strategy_coefficients(
                list(strategy._inv_coefs()), n_steps
            )
        )
        prefix_norms: list[float] = []
        prefix_norm = 0.0
        for index, coefficient in enumerate(strategy_coefficients):
            prefix_norm = math.hypot(prefix_norm, coefficient)
            if not math.isfinite(prefix_norm) or prefix_norm <= 0.0:
                raise ConfigurationError(
                    *(
                        "BISR normalized column norm is invalid at forward "
                        f"coefficient {index}: {prefix_norm!r}.",
                    )
                )
            prefix_norms.append(prefix_norm)
        column_scales = tuple(reversed(prefix_norms))
    else:
        column_scales = (1.0,) * n_steps

    _validate_effective_runtime_coefficients(
        inverse,
        column_scales,
        compute_dtype,
    )

    row_l2_values: list[float] = []
    inverse_norm_scale = 0.0
    inverse_norm_sum_squares = 1.0
    for step in range(n_steps):
        if step < len(inverse):
            value = abs(inverse[step])
            if value != 0.0:
                if inverse_norm_scale < value:
                    ratio = inverse_norm_scale / value
                    inverse_norm_sum_squares = 1.0 + (
                        inverse_norm_sum_squares * ratio * ratio
                    )
                    inverse_norm_scale = value
                else:
                    ratio = value / inverse_norm_scale
                    inverse_norm_sum_squares += ratio * ratio
        row_l2 = (
            column_scales[step]
            * inverse_norm_scale
            * math.sqrt(inverse_norm_sum_squares)
        )
        if not math.isfinite(row_l2) or row_l2 <= 0.0:
            raise ConfigurationError(
                *(f"BISR effective row L2 norm is invalid at step {step}: {row_l2!r}.",)
            )
        row_l2_values.append(row_l2)
    return inverse, column_scales, tuple(row_l2_values)


def _make_bisr_noise(
    grad_template: Any,
    strategy: BisrStrategy,
    *,
    n_steps: int,
    key: RngKey,
    compute_dtype: torch.dtype = torch.float32,
) -> tuple[
    Callable[..., tuple[Any, MFNoiseState]],
    MFNoiseState,
    Callable[[int], float],
]:
    from ._engine import (
        _check_mf_horizon,
        _matrix_factorization_noise,
        _require_positive_int_horizon,
    )

    n_steps = _require_positive_int_horizon(n_steps)
    inverse, column_scales, row_l2 = _bisr_runtime_parameters(
        strategy,
        n_steps,
        compute_dtype,
    )
    execution_identity = _BisrExecutionIdentity(
        version=_BISR_EXECUTION_IDENTITY_VERSION,
        inverse_coefficients=inverse,
        normalized=bool(strategy.normalized),
        n_steps=n_steps,
        compute_dtype=str(compute_dtype),
    )
    streaming = _direct_bisr_streaming_matrix(
        inverse,
        column_scales,
        execution_identity,
        compute_dtype=compute_dtype,
    )
    raw_noise_fn, state = _matrix_factorization_noise(
        grad_template,
        streaming,
        key=key,
        compute_dtype=compute_dtype,
        n_steps=n_steps,
    )

    def noise_fn(clipped_grads: Any, st: MFNoiseState, *, stddev: Any):
        inner = st._inner_state
        if not isinstance(inner, _BisrStreamingState):
            raise CheckpointError(
                *(
                    "BISR noise state does not use the bounded streaming layout. "
                    "Resume legacy dense-history checkpoints with the Opake "
                    "version that created them.",
                )
            )
        if inner.step != st._step_counter:
            raise CheckpointError(
                *(
                    "BISR inner and outer step counters disagree: "
                    f"inner={inner.step}, outer={st._step_counter}. The checkpoint "
                    "is incomplete or uses an incompatible state layout.",
                )
            )
        return raw_noise_fn(clipped_grads, st, stddev=stddev)

    def row_l2_at(step: int) -> float:
        _check_mf_horizon(step, n_steps)
        return row_l2[step]

    return noise_fn, state, row_l2_at


@register_strategy
@dataclass(frozen=True, slots=True)
class BisrStrategy:
    """BISR (Banded Inverse Square Root) strategy — recipe only.

    Carries the workload knobs and an optional explicit
    ``inv_coefficients`` override for :math:`C^{-1}`.  Derived
    quantities are computed on demand.
    """

    bandwidth: int
    normalized: bool = True
    momentum: float = 0.0
    # Compatibility tombstone for legacy state dictionaries. Non-None values
    # are rejected because optimizer LR schedules are not part of this encoder.
    lr_schedule: Schedule | None = field(default=None, compare=False)
    inv_coefficients: tuple[float, ...] | None = field(default=None)

    def __post_init__(self) -> None:
        if self.lr_schedule is not None:
            raise ConfigurationError(
                *(
                    "BisrStrategy does not support lr_schedule. Learning-rate "
                    "schedules are optimizer post-processing and cannot weight its "
                    "Balls-in-Bins privacy accounting. Remove lr_schedule from the "
                    "strategy, pass it only to the optimizer, and recalibrate privacy "
                    "and noise for any result previously computed with this option.",
                )
            )
        if self.bandwidth < _MIN_BANDWIDTH:
            raise ConfigurationError(
                *(f"bandwidth must be >= 2, got {self.bandwidth}",)
            )
        if not math.isfinite(self.momentum) or not 0.0 <= self.momentum < 1.0:
            raise ConfigurationError(
                *(f"momentum must be finite and in [0, 1), got {self.momentum}",)
            )
        if (
            self.inv_coefficients is not None
            and len(self.inv_coefficients) != self.bandwidth
        ):
            raise ConfigurationError(
                *(
                    f"inv_coefficients length ({len(self.inv_coefficients)}) must "
                    f"equal bandwidth ({self.bandwidth})",
                )
            )
        if self.inv_coefficients is not None:
            if not all(math.isfinite(float(coef)) for coef in self.inv_coefficients):
                raise ConfigurationError(
                    *("inv_coefficients must contain only finite values",)
                )
            if (
                abs(float(self.inv_coefficients[0]))
                < _MIN_LEADING_COEFFICIENT_MAGNITUDE
            ):
                raise ConfigurationError(
                    *(
                        "inv_coefficients[0] must have magnitude "
                        f">= {_MIN_LEADING_COEFFICIENT_MAGNITUDE:.0e}",
                    )
                )

    def _inv_coefs(self) -> tuple[float, ...]:
        if self.inv_coefficients is not None:
            return self.inv_coefficients
        return _bisr_inverse_coefficients_cached(self.bandwidth, self.momentum)

    def coefficients(self, *, n_steps: int, **_) -> torch.Tensor:
        return torch.tensor(
            _native().bisr_strategy_coefficients(list(self._inv_coefs()), n_steps),
            dtype=torch.float64,
        )

    def gram_matrix(
        self, *, n_steps: int, min_sep: int, max_participations: int | None
    ) -> tuple[float, ...]:
        return _bisr_gram_matrix_cached(
            self._inv_coefs(),
            self.normalized,
            n_steps,
            min_sep,
            max_participations,
        )

    def streaming_matrix(self, *, n_steps: int, **_) -> StreamingMatrix:
        inv = self._inv_coefs()
        return inverse_as_streaming_matrix(
            self.coefficients(n_steps=n_steps),
            column_normalize_for_n=n_steps if self.normalized else None,
            # The strategy coefficients are dense (length n_steps), but
            # C^{-1} is banded with exactly these coefficients — hand them
            # over so the closed-form row norms stay O(bandwidth * n)
            # instead of running the length-n inversion recurrence. Only the
            # first n_steps entries lie within the horizon; a longer hint
            # describes a matrix that does not exist at this size.
            inverse_coefficients=torch.tensor(inv[:n_steps], dtype=torch.float64),
        )

    def raw_noise_factory(
        self,
        grad_template: Any,
        *,
        n_steps: int,
        min_sep: int,
        max_participations: int | None,
        key: RngKey,
        compute_dtype: torch.dtype,
    ):
        del min_sep, max_participations
        return _make_bisr_noise(
            grad_template,
            self,
            n_steps=n_steps,
            key=key,
            compute_dtype=compute_dtype,
        )

    def sensitivity(
        self, *, n_steps: int, min_sep: int, max_participations: int | None
    ) -> float:
        inv = list(self._inv_coefs())
        if self.normalized:
            sens_sq = _native().bisr_normalized_sensitivity_squared(
                inv, n_steps, min_sep, max_participations
            )
        else:
            sens_sq = _native().bisr_sensitivity_squared(
                inv, n_steps, min_sep, max_participations
            )
        return float(sens_sq**0.5)


def bisr_strategy(
    *,
    bandwidth: int,
    normalized: bool = True,
    momentum: float = 0.0,
    lr_schedule: Schedule | None = None,
    inv_coefficients: Sequence[float] | None = None,
) -> BisrStrategy:
    """Create a BISR (Banded Inverse Square Root) strategy recipe.

    Args:
        bandwidth: BISR bandwidth p (>= 2).
        normalized: Use column-normalized matrix (default True).
        momentum: The paper's beta in the alpha=1 specialization. Must be in
            ``[0, 1)`` (default 0), matching Lemma 1's
            ``0 <= beta < alpha <= 1`` hypothesis.
        lr_schedule: Deprecated compatibility argument. Only ``None`` is
            accepted. Pass learning-rate schedules to the optimizer instead.
        inv_coefficients: Explicit :math:`C^{-1}` coefficients (default BISR optimal).

    Returns:
        A :class:`BisrStrategy` recipe.
    """
    return BisrStrategy(
        bandwidth=bandwidth,
        normalized=normalized,
        momentum=momentum,
        lr_schedule=lr_schedule,
        inv_coefficients=(
            tuple(inv_coefficients) if inv_coefficients is not None else None
        ),
    )


__all__ = ["BisrStrategy", "bisr_strategy"]
