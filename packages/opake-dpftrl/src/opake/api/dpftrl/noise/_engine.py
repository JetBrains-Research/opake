"""Matrix factorization-based noise addition for DP-FTRL.

Implements correlated noise mechanisms that add noise to gradients using
matrix factorization. Instead of adding independent Gaussian noise at
each step (standard DP-SGD), these mechanisms add correlated noise that
achieves better utility at the same privacy budget.

The user-facing entry point is ``mf_gaussian_noise`` (in ``opake.noise``).
Internally, strategy modules (``band_mf``, ``blt``, etc.)
call ``_matrix_factorization_noise`` which returns ``(noise_fn, state)``.

References:
    - Correlated noise mechanisms: https://arxiv.org/abs/2506.08201
    - BandMF: https://arxiv.org/abs/2306.08153
    - BLT mechanisms: https://arxiv.org/abs/2404.16706
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import torch

from opake.exceptions import ConfigurationError, InputTypeError
from opake.pytree import tree_map
from opake.random import fold_in as rng_fold_in
from opake.random import generator_from_key
from opake.types import NoiseState, PerGroup

from . import _streaming_matrix as streaming_matrix

if TYPE_CHECKING:
    from collections.abc import Callable

    from opake.random.types import RngKey

MF_GAUSSIAN_STREAM_FOLD = "opake.dpftrl.mf_gaussian"


def _mf_gaussian_column_key(
    rng_key: RngKey,
    column: int,
    *substream: int | str,
) -> RngKey:
    return rng_fold_in(
        rng_key,
        MF_GAUSSIAN_STREAM_FOLD,
        "mf_gaussian_column",
        column,
        *substream,
    )


@dataclasses.dataclass(frozen=True)
class MFNoiseState(NoiseState):
    """State for matrix factorization noise.

    Restoration is template-driven. A saved state must be loaded into a
    template built with the same strategy recipe, horizon, compute dtype, and
    parameter-tree structure. The trainer checkpoint records and checks that
    execution provenance before restoring this state; standalone callers are
    responsible for preserving the same construction inputs.

    Attributes:
        _inner_state: Internal state (streaming matrix state or step counter).
        _step_counter: Number of noise_fn calls made.
        _rng_key: Immutable RNG key for deterministic per-step derivation.
        _first_max_norm: ``ClippedPytree.max_norm`` from the first call (scalar
            or :class:`~opake.types.PerGroup`), latched by the dispatcher to
            enforce constant per-step sensitivity.  ``None`` until the first
            call.  MF privacy analyses assume the per-step sensitivity is
            constant across the sequence.  Both fixed clipping
            (:func:`opake.dpftrl.clipping.clipped_grad`) and AUTO-S clipping
            (:func:`opake.dpftrl.clipping.auto_clipped_grad`) satisfy this
            assumption — the per-record bound is fixed at construction and
            does not depend on data.  Adaptive clipping
            (:func:`opake.dpsgd.clipping.adaptive_clipped_grad`) breaks it,
            so the dispatcher rejects subsequent calls whose ``max_norm``
            differs.
        _first_max_norm_sync_fingerprint: Deterministic 64-bit fingerprint
            (computed at latch time) for cheap cross-rank checks in
            :func:`sync_mf_noise_state`, for both scalar and ``PerGroup`` norms.
            ``None`` before the first call.
    """

    _inner_state: Any
    _step_counter: int
    _rng_key: RngKey
    _first_max_norm: float | PerGroup | None = None
    _first_max_norm_sync_fingerprint: int | None = None


def _internal_compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """Use at least float32 for internal MF noise computations."""
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _iid_normal_noise(
    target_tree: Any,
    stddev: float | PerGroup,
    generator: torch.Generator | None = None,
    compute_dtype: torch.dtype = torch.float32,
) -> Any:
    """Generate IID normal noise matching the structure of ``target_tree``.

    ``compute_dtype`` is the dtype used for the underlying ``torch.randn``
    call.  Defaults to ``torch.float32`` to match :func:`opake.dpsgd.noise.gaussian_noise`
    — sampling Gaussians in bf16/fp16 has discretization issues that distort
    the noise distribution.  Type stability on the public boundary is
    preserved by the caller (which casts back to the input dtype).
    """

    def _randn_on_device(
        shape: tuple[int, ...],
        *,
        noise_dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        try:
            return torch.randn(
                shape, dtype=noise_dtype, device=device, generator=generator
            )
        except RuntimeError as exc:
            if "device type for generator" in str(exc):
                return torch.randn(shape, dtype=noise_dtype, generator=generator).to(
                    device=device
                )
            raise

    if isinstance(stddev, PerGroup):
        import optree

        from opake.api.engine.pytree import tree_flatten_with_paths

        paths, leaves, treedef = tree_flatten_with_paths(target_tree)
        out_leaves: list[Any] = []
        for path, tensor in zip(paths, leaves, strict=True):
            if not isinstance(tensor, torch.Tensor):
                raise InputTypeError(
                    *(
                        "PerGroup MF noise expects tensor leaves; "
                        f"got {type(tensor).__name__} at path {path!r}.",
                    )
                )
            leaf_std = stddev.for_path(path)
            noise = _randn_on_device(
                tensor.shape,
                noise_dtype=compute_dtype,
                device=tensor.device,
                generator=generator,
            )
            out_leaves.append(noise * leaf_std)
        return optree.tree_unflatten(treedef, out_leaves)

    def make_noise(t):
        noise = _randn_on_device(
            t.shape,
            noise_dtype=compute_dtype,
            device=t.device,
            generator=generator,
        )
        return noise * stddev

    return tree_map(make_noise, target_tree)


def _gaussian_linear_combination(
    matrix_row: torch.Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    key: RngKey,
    leaf_index: int,
) -> torch.Tensor:
    """Compute a linear combination of standard Gaussian random variables.

    Given coefficients [c0, c1, ..., c_{k-1}] and IID z_0, ..., z_{k-1}
    ~ N(0, I), returns sum_i c_i * z_i. Each ``z_i`` is derived from the
    mechanism key and column ``i`` so it is reused by every row that references
    that column.
    """
    result = torch.zeros(shape, dtype=dtype, device=device)
    for idx in torch.nonzero(matrix_row, as_tuple=False).flatten().tolist():
        coef = matrix_row[idx].to(dtype)
        generator = generator_from_key(_mf_gaussian_column_key(key, idx, leaf_index))
        try:
            noise = torch.randn(shape, dtype=dtype, device=device, generator=generator)
        except RuntimeError as exc:
            if "device type for generator" in str(exc):
                noise = torch.randn(shape, dtype=dtype, generator=generator).to(
                    device=device
                )
            else:
                raise
        result = result + coef * noise
    return result


def _check_mf_horizon(step: int, n_steps: int) -> None:
    """Raise if ``step`` is outside the calibrated MF horizon ``[0, n_steps)``.

    MF strategies (and their accountants) are built for a fixed horizon.
    Calling ``noise_fn`` at ``step >= n_steps`` is either a zero-noise
    release (normalized λ-CGD) or unaccounted correlated noise (streaming
    / dense engines).  Fail closed rather than silently continuing.
    """
    if step >= n_steps:
        raise ConfigurationError(
            *(
                f"MF noise step {step} is outside the calibrated horizon "
                f"[0, {n_steps}). Rebuild the noise mechanism with a larger "
                f"n_steps, or stop calling noise_fn after {n_steps} iterations.",
            )
        )


def _require_positive_int_horizon(n_steps: object) -> int:
    """Validate a calibrated MF horizon as a positive ``int``.

    Rejects bools and non-integers (``int(2.9)`` truncation would silently
    shrink the horizon).  Returns the validated value for callers to latch.
    """
    if isinstance(n_steps, bool) or not isinstance(n_steps, int):
        raise InputTypeError(
            *(f"n_steps must be an int, got {type(n_steps).__name__}",)
        )
    if n_steps < 1:
        raise ConfigurationError(*(f"n_steps must be >= 1, got {n_steps}",))
    return n_steps


def _matrix_factorization_noise(
    grad_template: Any,
    noising: torch.Tensor | streaming_matrix.StreamingMatrix,
    *,
    key: RngKey,
    compute_dtype: torch.dtype = torch.float32,
    n_steps: int | None = None,
) -> tuple[
    Callable[..., tuple[Any, MFNoiseState]],
    MFNoiseState,
]:
    """Internal: create ``(noise_fn, state)`` from a noising matrix.

    This is the engine that all strategy modules call. Users
    should use ``mf_gaussian_noise`` (or a strategy wrapper) instead.

    Args:
        grad_template: Pytree with the same structure/shapes as gradients.
        noising: Dense 2D tensor or ``StreamingMatrix`` representing C^{-1}.
        key: Pre-resolved base ``RngKey``.
        compute_dtype: Dtype used for the underlying ``torch.randn`` and
            linear-combination arithmetic.  Matches the
            :func:`opake.dpsgd.noise.gaussian_noise` convention.
        n_steps: Calibrated training horizon.  When provided, ``noise_fn``
            raises once ``step >= n_steps``.  For dense tensors, defaults
            to ``noising.shape[0]`` when omitted.  Streaming matrices have
            no intrinsic size — omit only for direct engine callers that
            intentionally leave the sequence unbounded; the public
            :func:`mf_gaussian_noise` factory always passes ``n_steps``.

    Returns:
        ``(noise_fn, state)`` where
        ``noise_fn(grads, state, *, stddev) -> (noised, state)``.  ``stddev``
        is the per-step standard deviation for the base IID noise (scalar or
        :class:`~opake.types.PerGroup` for per-parameter-group allocation);
        the dispatcher derives it from ``noise_multiplier`` and
        ``ClippedPytree.max_norm``.
    """
    if isinstance(noising, torch.Tensor):
        return _tensor_mf_noise(
            grad_template,
            noising,
            key=key,
            compute_dtype=compute_dtype,
            n_steps=n_steps,
        )
    elif isinstance(noising, streaming_matrix.StreamingMatrix):
        return _streaming_mf_noise(
            grad_template,
            noising,
            key=key,
            compute_dtype=compute_dtype,
            n_steps=n_steps,
        )
    else:
        raise InputTypeError(*(f"Unsupported noising type: {type(noising)}",))


def _tensor_mf_noise(
    grad_template: Any,
    noising: torch.Tensor,
    *,
    key: RngKey,
    compute_dtype: torch.dtype = torch.float32,
    n_steps: int | None = None,
) -> tuple[Callable, MFNoiseState]:
    """(noise_fn, state) from a 2D noising matrix C^{-1}."""
    if noising.ndim != 2:  # noqa: PLR2004 - noising is explicitly a matrix
        raise ConfigurationError(*(f"Expected 2D matrix, found shape {noising.shape}",))
    horizon = (
        noising.shape[0] if n_steps is None else _require_positive_int_horizon(n_steps)
    )
    if horizon > noising.shape[0]:
        raise ConfigurationError(
            *(f"n_steps ({horizon}) exceeds noising matrix rows ({noising.shape[0]}).",)
        )

    state = MFNoiseState(
        _inner_state=torch.tensor(0, dtype=torch.long),
        _step_counter=0,
        _rng_key=key,
    )

    def noise_fn(clipped_grads, st, *, stddev: float | PerGroup):
        index = st._inner_state
        step_index = int(index)
        _check_mf_horizon(step_index, horizon)
        matrix_row_base = noising[step_index]
        import optree

        from opake.api.engine.pytree import tree_flatten_with_paths

        paths, leaves, treedef = tree_flatten_with_paths(clipped_grads)

        if isinstance(stddev, PerGroup):

            def add_noise_at_path(
                path, grad_tensor: torch.Tensor, leaf_index: int
            ) -> torch.Tensor:
                eff = stddev.for_path(path)
                matrix_row = matrix_row_base * eff
                noise = _gaussian_linear_combination(
                    matrix_row,
                    grad_tensor.shape,
                    compute_dtype,
                    grad_tensor.device,
                    st._rng_key,
                    leaf_index,
                )
                return (grad_tensor + noise).to(grad_tensor.dtype)

            noisy_leaves = []
            for leaf_index, (path, v) in enumerate(zip(paths, leaves, strict=True)):
                if not isinstance(v, torch.Tensor):
                    raise InputTypeError(
                        *(
                            "PerGroup dense MF noise expects tensor leaves; "
                            f"got {type(v).__name__} at path {path!r}.",
                        )
                    )
                noisy_leaves.append(add_noise_at_path(path, v, leaf_index))
            noisy_grads = optree.tree_unflatten(treedef, noisy_leaves)
        else:
            matrix_row = matrix_row_base * float(stddev)

            noisy_leaves = []
            for leaf_index, (path, v) in enumerate(zip(paths, leaves, strict=True)):
                if not isinstance(v, torch.Tensor):
                    raise InputTypeError(
                        *(
                            "Dense MF noise expects tensor leaves; "
                            f"got {type(v).__name__} at path {path!r}.",
                        )
                    )
                noise = _gaussian_linear_combination(
                    matrix_row,
                    v.shape,
                    compute_dtype,
                    v.device,
                    st._rng_key,
                    leaf_index,
                )
                noisy_leaves.append((v + noise).to(v.dtype))
            noisy_grads = optree.tree_unflatten(treedef, noisy_leaves)
        new_state = MFNoiseState(
            _inner_state=index + 1,
            _step_counter=st._step_counter + 1,
            _rng_key=st._rng_key,
        )
        return noisy_grads, new_state

    return noise_fn, state


def _streaming_mf_noise(
    grad_template: Any,
    noising: streaming_matrix.StreamingMatrix,
    *,
    key: RngKey,
    compute_dtype: torch.dtype = torch.float32,
    n_steps: int | None = None,
) -> tuple[Callable, MFNoiseState]:
    """(noise_fn, state) from a streaming noising matrix C^{-1}.

    ``n_steps`` is the calibrated horizon.  When provided, ``noise_fn``
    raises once ``step >= n_steps`` so past-horizon correlated noise is
    never released unaccounted.  Direct engine callers that omit
    ``n_steps`` keep the previous unbounded behaviour (tests that drive
    a streaming matrix without a declared horizon).
    """
    horizon = None if n_steps is None else _require_positive_int_horizon(n_steps)
    streaming_state = noising.init_multiply(grad_template)
    state = MFNoiseState(
        _inner_state=streaming_state,
        _step_counter=0,
        _rng_key=key,
    )

    def noise_fn(clipped_grads, st, *, stddev: float | PerGroup):
        step = st._step_counter
        if horizon is not None:
            _check_mf_horizon(step, horizon)
        step_key = _mf_gaussian_column_key(st._rng_key, step)
        g = generator_from_key(step_key)
        s_state = st._inner_state

        iid_noise = _iid_normal_noise(
            clipped_grads,
            stddev,
            generator=g,
            compute_dtype=compute_dtype,
        )
        corr_noise, new_streaming_state = noising.multiply_next(iid_noise, s_state)
        noisy_grads = tree_map(
            lambda grad, n: (grad + n).to(grad.dtype),
            clipped_grads,
            corr_noise,
        )
        new_state = MFNoiseState(
            _inner_state=new_streaming_state,
            _step_counter=step + 1,
            _rng_key=st._rng_key,
        )
        return noisy_grads, new_state

    return noise_fn, state


# ---- Input validation helpers shared by mf_gaussian_noise + second-moment ----


def _resolve_noise_multiplier(noise_multiplier: float) -> float:
    multiplier = float(noise_multiplier)
    if multiplier < 0:
        raise ConfigurationError(
            *(f"noise_multiplier must be non-negative, got {noise_multiplier}",)
        )
    return multiplier


def _expect_clipped(value: Any, *, op: str):
    """Reject NoisedPytree or non-ClippedPytree inputs to the noise function."""
    from opake.types import ClippedPytree, NoisedPytree

    if isinstance(value, NoisedPytree):
        raise InputTypeError(
            *(
                f"{op} expects ClippedPytree inputs, not NoisedPytree values that "
                "have already passed through a noise mechanism.",
            )
        )
    if not isinstance(value, ClippedPytree):
        raise InputTypeError(
            *(
                f"{op} expects ClippedPytree inputs. Wrap manual values with "
                "opake.types.clipped(...).",
            )
        )
    return value


def _validate_constant_max_norm(
    grads,
    first_max_norm: float | PerGroup | None,
    *,
    op: str,
) -> float | PerGroup:
    """Latch the per-step max_norm and reject changes across calls.

    MF privacy analyses calibrate noise from a sensitivity that is constant
    across the sequence.  Fixed clipping
    (:func:`opake.dpftrl.clipping.clipped_grad`) and AUTO-S clipping
    (:func:`opake.dpftrl.clipping.auto_clipped_grad`) both produce a
    constant ``ClippedPytree.max_norm`` and pass this latch.  Adaptive
    clipping (:func:`opake.dpsgd.clipping.adaptive_clipped_grad`) varies
    its threshold across steps, which breaks the proof; the factory
    latches the first-call max_norm in the state and rejects any
    subsequent call whose max_norm differs.
    """
    max_norm = grads.max_norm
    if isinstance(max_norm, PerGroup):
        for group_name, value in max_norm.values.items():
            if value < 0:
                raise ConfigurationError(
                    *(
                        f"ClippedPytree max_norm must be non-negative for all groups, "
                        f"got {value} for group '{group_name}'.",
                    )
                )
    else:
        if float(max_norm) < 0:
            raise ConfigurationError(
                *(f"ClippedPytree max_norm must be non-negative, got {grads.max_norm}",)
            )
    if first_max_norm is not None and max_norm != first_max_norm:
        raise ConfigurationError(
            *(
                f"{op} saw a varying ClippedPytree.max_norm across calls "
                f"(first={first_max_norm}, now={max_norm}). MF privacy proofs "
                "assume a constant per-step sensitivity; this is satisfied by "
                "fixed and AUTO-S clipping but not by adaptive clipping, which "
                "is therefore unsupported with MF noise.",
            )
        )
    return max_norm


# ---- Distributed state validation ----
