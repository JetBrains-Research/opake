"""Library for working with lower-triangular Toeplitz matrices.

Toeplitz matrices are constant along diagonals. For DP matrix factorization,
we work with lower-triangular banded Toeplitz strategy matrices, which
enable efficient streaming noise generation with correlated noise.

References:
    - BandMF: https://arxiv.org/abs/2306.08153
    - Fichtenberger et al.: https://arxiv.org/abs/2202.11205
    - Scaling BandMF: https://arxiv.org/abs/2405.15913
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Callable
from typing import Any, Protocol, TypeAlias, TypeVar

import numpy as np
import torch
from scipy.linalg import toeplitz as scipy_toeplitz
from scipy.signal import lfilter as scipy_lfilter

from opake.exceptions import ConfigurationError, OperationError

from . import (
    _checks as checks,
)
from . import (
    _sensitivity as sensitivity,
)
from . import (
    _streaming_matrix as streaming_matrix,
)

# ---------------------------------------------------------------------------
# Optimization (L-BFGS wrapper, formerly optimization.py)
# ---------------------------------------------------------------------------

_ParamT = TypeVar("_ParamT")
_CallbackFnType: TypeAlias = Callable[["_OptimCallbackArgs"], bool | None]


@dataclasses.dataclass
class _OptimCallbackArgs:
    """Information passed to the callback on each optimization step."""

    step: int
    loss: torch.Tensor
    grad: Any | None
    params: Any
    state: Any


class _EarlyStopException(Exception):
    """Internal exception for early stopping in scipy optimization."""

    def __init__(self, params):
        self.params = params
        super().__init__("Early stop")


def _lbfgs_optimize(
    loss_fn: Callable,
    params: torch.Tensor,
    *,
    max_optimizer_steps: int = 250,
    grad: bool = False,
    callback: _CallbackFnType = lambda _: None,
    bounds: list[tuple[float | None, float | None]] | None = None,
) -> torch.Tensor:
    """Optimize a differentiable loss function using L-BFGS.

    Uses scipy's L-BFGS-B optimizer. Parameters are internally cast to
    float64 for numerical stability.
    """
    from scipy.optimize import minimize

    original_dtype = params.dtype
    params_np = params.detach().double().numpy().copy()

    step_counter = [0]

    def scipy_loss(x):
        x_tensor = torch.tensor(x, dtype=torch.float64, requires_grad=not grad)

        if grad:
            loss_val, grad_val = loss_fn(x_tensor)
            loss_np = float(loss_val.detach())
            if isinstance(grad_val, torch.Tensor):
                grad_np = grad_val.detach().numpy().copy()
            else:
                grad_np = grad_val
        else:
            loss_val = loss_fn(x_tensor)
            loss_val.backward()
            loss_np = float(loss_val.detach())
            grad_np = x_tensor.grad.numpy().copy()

        cb_result = callback(
            _OptimCallbackArgs(
                step=step_counter[0],
                loss=torch.tensor(loss_np),
                grad=torch.tensor(grad_np) if grad_np is not None else None,
                params=x_tensor.detach(),
                state=None,
            )
        )
        step_counter[0] += 1

        if cb_result:
            raise _EarlyStopException(x)

        return loss_np, grad_np.astype("float64")

    try:
        result = minimize(
            scipy_loss,
            params_np,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": max_optimizer_steps, "ftol": 1e-15, "gtol": 1e-10},
        )
        if not math.isfinite(float(result.fun)) or not bool(
            torch.isfinite(torch.as_tensor(result.x)).all()
        ):
            raise OperationError(
                *(f"L-BFGS produced a non-finite result: {result.message}",)
            )
        optimal_params = torch.tensor(result.x, dtype=original_dtype)
    except _EarlyStopException as e:
        optimal_params = torch.tensor(e.params, dtype=original_dtype)

    return optimal_params


# ---------------------------------------------------------------------------
# Toeplitz matrix library
# ---------------------------------------------------------------------------

__all__ = [
    "inverse_as_streaming_matrix",
    "materialize_lower_triangular",
    "max_error",
    "mean_error",
    "minsep_sensitivity_squared",
    "minsep_sensitivity_upper_bound",
    "optimal_max_error_strategy_coefs",
    "optimize",
    "per_query_error",
    "sensitivity_squared",
]


def _l2_norm_squared(x: torch.Tensor) -> torch.Tensor:
    return torch.dot(x, x)


def _reconcile(coef: torch.Tensor, n: int | None = None) -> tuple[torch.Tensor, int]:
    """Reconcile Toeplitz coefficients with matrix size."""
    n = n or len(coef)
    coef = torch.as_tensor(coef, dtype=torch.float64)[:n]
    return coef, n


def pad_coefs_to_n(coef: torch.Tensor, n: int | None = None) -> torch.Tensor:
    """Materialize length-n Toeplitz coefficients (zero-padded)."""
    coef, n = _reconcile(coef, n)
    result = torch.zeros(n, dtype=coef.dtype)
    result[: len(coef)] = coef
    return result


def inverse_as_streaming_matrix(
    coef: torch.Tensor,
    column_normalize_for_n: int | None = None,
    *,
    inverse_coefficients: torch.Tensor | None = None,
) -> streaming_matrix.StreamingMatrix:
    """Create C^{-1} as a StreamingMatrix.

    If ``column_normalize_for_n`` is None, returns C^{-1} for an arbitrarily
    large banded Toeplitz C. Otherwise, C is column-normalized for size n.

    This implements Algorithm 9 from https://arxiv.org/abs/2306.08153.

    The returned matrix carries a closed-form ``row_norms_squared``:
    C^{-1} is lower-triangular Toeplitz, so its squared row norms are
    ``cumsum(inverse_coef**2)`` (rescaled by the normalization diagonal
    when ``column_normalize_for_n`` is set).  That is O(bands * n) at
    query time instead of the O(bands * n^2) generic probing.

    Args:
        coef: Toeplitz coefficients of the strategy matrix.
        column_normalize_for_n: If set, column-normalize C for this size.
        inverse_coefficients: Optional known Toeplitz coefficients of the
            un-normalized C^{-1}, treated as zero past their length (e.g.
            BISR's banded inverse).  Skips the O(bands * n) inversion
            recurrence inside the closed-form row norms.  Validated at
            construction: raises ``ValueError`` unless
            ``toeplitz(coef) @ toeplitz(inverse_coefficients)`` is the
            identity.  That check spans as many terms as the longer of the
            two coefficient windows, so the hint is only trusted out to that
            horizon; row norms past it are computed by the inversion
            recurrence instead, because a hint that merely agrees with the
            inverse over the checked window may still have a nonzero tail
            beyond it.  Gradients do not flow through the hint (it is
            redundant with ``coef``); a grad-carrying hint falls back to
            the probing path like a grad-carrying ``coef`` does.

    Returns:
        A StreamingMatrix representing C^{-1}.

    Raises:
        ValueError: If ``inverse_coefficients`` does not invert ``coef``.
    """
    coef, _ = _reconcile(coef, column_normalize_for_n)
    bands = coef.shape[0]

    hint_requires_grad = False
    inv_hint: torch.Tensor | None = None
    hint_horizon = 0
    if inverse_coefficients is not None:
        hint_requires_grad = (
            isinstance(inverse_coefficients, torch.Tensor)
            and inverse_coefficients.requires_grad
        )
        inv_hint = (
            torch.as_tensor(inverse_coefficients).detach().cpu().to(torch.float64)
        )
        # Entries past both coefficient windows hold only the truncation
        # tail of a length-n coef (e.g. BISR's dense strategy recovery),
        # not an inconsistency inside the n x n matrix — ignore them.
        window = max(coef.shape[0], inv_hint.shape[0])
        hint_horizon = window
        product = np.convolve(coef.detach().cpu().numpy(), inv_hint.numpy())[:window]
        identity = np.zeros_like(product)
        identity[0] = 1.0
        if not np.allclose(product, identity, atol=1e-8):
            raise ConfigurationError(
                *(
                    "inverse_coefficients is not the Toeplitz inverse of coef: "
                    "max |toeplitz(coef) @ toeplitz(inverse_coefficients) - I| = "
                    f"{np.abs(product - identity).max():.3e}",
                )
            )

    def init(abstract_yi):
        dtype = abstract_yi.dtype
        if dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
        zero = torch.zeros_like(abstract_yi, dtype=dtype)
        return zero.unsqueeze(0).expand(bands - 1, *zero.shape).clone()

    def _next(yi, state):
        if bands == 1:
            coef0 = coef[0].to(device=state.device, dtype=state.dtype)
            return yi.to(state.dtype) / coef0, state
        coef_local = coef.to(device=state.device, dtype=state.dtype)
        inner = torch.tensordot(coef_local[1:], state, dims=1)
        xi = (yi.to(state.dtype) - inner) / coef_local[0]
        new_state = torch.roll(state, 1, dims=0)
        new_state[0] = xi
        return xi, new_state

    Cinv = streaming_matrix.StreamingMatrix.from_array_implementation(init, _next)

    col_norms: torch.Tensor | None = None
    if column_normalize_for_n is not None:
        full_coef = pad_coefs_to_n(coef, column_normalize_for_n)
        col_norms = torch.sqrt(torch.cumsum(full_coef**2, dim=0)).flip(0)
        Cinv = streaming_matrix.scale_rows_and_columns(Cinv, row_scale=col_norms)

    # ``fallback`` has no row_norms_squared_fn, so calling its
    # row_norms_squared runs the generic probing — no recursion.
    fallback = Cinv

    def _row_norms_squared(n: int) -> torch.Tensor:
        if coef.requires_grad or hint_requires_grad:
            # lfilter would silently detach the autograd graph; keep the
            # differentiable probing path for optimization use.
            return fallback.row_norms_squared(n)
        if n == 0:
            return torch.zeros(0, dtype=torch.float64, device="cpu")
        if inv_hint is not None and n <= hint_horizon:
            # Validation only checked the convolution through ``hint_horizon``,
            # so the hint reproduces the true inverse coefficients only that
            # far. Zero-padding it past that point would silently assume the
            # inverse terminates there and under-report the row norms.
            inv_coefs = pad_coefs_to_n(inv_hint, n)
        else:
            impulse = np.zeros(n)
            impulse[0] = 1.0
            inv_coefs = torch.from_numpy(
                scipy_lfilter([1.0], coef.detach().cpu().numpy(), impulse)
            )
        norms = torch.cumsum(inv_coefs.square(), dim=0)
        if col_norms is not None:
            # Past ``column_normalize_for_n`` the probing path repeats the
            # last diagonal entry (see ``diagonal``); mirror that here.
            idx = torch.arange(n, device="cpu").clamp(max=col_norms.numel() - 1)
            scale = col_norms.detach().cpu()[idx]
            norms = norms * scale.square()
        return norms

    return dataclasses.replace(Cinv, row_norms_squared_fn=_row_norms_squared)


def optimal_max_error_strategy_coefs(n: int) -> torch.Tensor:
    """Returns optimal Toeplitz strategy coefficients for max error.

    From Fichtenberger, Henzinger, and Upadhyay:
    https://arxiv.org/abs/2202.11205

    Args:
        n: Number of coefficients.

    Returns:
        Tensor of Toeplitz coefficients.
    """
    k = torch.arange(n, dtype=torch.float64)
    ratios = (2 * k - 1) / (2 * k)
    ratios[0] = 1.0
    return torch.cumprod(ratios, dim=0)


def optimal_max_error_noising_coefs(n: int) -> torch.Tensor:
    """Returns optimal Toeplitz noising coefficients for max error.

    Args:
        n: Number of coefficients.

    Returns:
        Coefficients of C^{-1}.
    """
    c = optimal_max_error_strategy_coefs(n)
    result = c.clone()
    result[1:n] = c[1:n] - c[: n - 1]
    return result


def materialize_lower_triangular(
    coef: torch.Tensor, n: int | None = None
) -> torch.Tensor:
    """Create a lower-triangular Toeplitz matrix from coefficients.

    Example: coef=[a,b,c], n=4 gives::

        [[a 0 0 0]
         [b a 0 0]
         [c b a 0]
         [0 c b a]]

    Args:
        coef: Nonzero coefficients of the first column.
        n: Optional matrix size (defaults to len(coef)).

    Returns:
        Lower-triangular Toeplitz matrix.
    """
    full_coef = pad_coefs_to_n(coef, n)
    n_actual = len(full_coef)
    original_device = full_coef.device
    col = full_coef.detach().cpu().numpy()
    row = torch.zeros(n_actual, dtype=full_coef.dtype).numpy()
    row[0] = col[0]
    toeplitz_np = scipy_toeplitz(col, row)
    return torch.from_numpy(toeplitz_np).to(
        device=original_device, dtype=full_coef.dtype
    )


def solve_banded(coef: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Solve T_{coef} x = rhs for banded Toeplitz T.

    Args:
        coef: Toeplitz coefficients.
        rhs: Right-hand side vector.

    Returns:
        Solution x.
    """
    return (
        streaming_matrix.multiply_array(
            inverse_as_streaming_matrix(coef),
            rhs.unsqueeze(1) if rhs.ndim == 1 else rhs,
        ).squeeze(-1)
        if rhs.ndim == 1
        else streaming_matrix.multiply_array(inverse_as_streaming_matrix(coef), rhs)
    )


def multiply(
    lhs_coef: torch.Tensor,
    rhs_coef: torch.Tensor,
    n: int | None = None,
    skip_checks: bool = False,
) -> torch.Tensor:
    """Multiply two lower-triangular Toeplitz matrices (via convolution).

    Args:
        lhs_coef: Coefficients of the left matrix.
        rhs_coef: Coefficients of the right matrix.
        n: Optional matrix size.
        skip_checks: Skip input validation.

    Returns:
        Coefficients of the product matrix.
    """
    if not skip_checks and n is None and len(lhs_coef) != len(rhs_coef):
        raise ConfigurationError(
            *(
                "If n is not specified, lhs_coef and rhs_coef must have "
                f"the same length, found {len(lhs_coef)} and {len(rhs_coef)}.",
            )
        )
    lhs_coef, n = _reconcile(lhs_coef, n)
    rhs_coef, _ = _reconcile(rhs_coef, n)

    # Differentiable convolution via FFT (supports autograd for L-BFGS
    # optimization of BLT/banded Toeplitz with workload_coef).
    full_len = len(lhs_coef) + len(rhs_coef) - 1
    fa = torch.fft.rfft(lhs_coef, n=full_len)
    fb = torch.fft.rfft(rhs_coef, n=full_len)
    conv = torch.fft.irfft(fa * fb, n=full_len)[:n]
    return conv


def inverse_coef(coef: torch.Tensor, n: int | None = None) -> torch.Tensor:
    """Find the inverse coefficients of a Toeplitz matrix.

    Args:
        coef: Toeplitz coefficients of C.
        n: Optional matrix size.

    Returns:
        Toeplitz coefficients of C^{-1}.
    """
    coef, n = _reconcile(coef, n)
    e0 = torch.zeros(n, dtype=coef.dtype)
    e0[0] = 1.0
    return solve_banded(coef, e0)


def sensitivity_squared(coef: torch.Tensor, n: int | None = None) -> torch.Tensor:
    """Sensitivity^2 under single participation."""
    coef, _ = _reconcile(coef, n)
    return _l2_norm_squared(coef)


def minsep_sensitivity_squared(
    strategy_coef: torch.Tensor,
    min_sep: int,
    max_participations: int | None = None,
    n: int | None = None,
) -> torch.Tensor:
    """Return exact min-separation sensitivity squared on the theorem domain.

    The closed form requires finite, non-negative, non-increasing Toeplitz
    coefficients.  Use :func:`minsep_sensitivity_upper_bound` for arbitrary
    signed or non-monotone coefficients.

    Reference: https://arxiv.org/abs/2405.13763, Theorem 2.

    Args:
        strategy_coef: Toeplitz coefficients of C.
        min_sep: Minimum separation between participations.
        max_participations: Maximum participations.
        n: Optional matrix size.

    Returns:
        The exact sensitivity squared.
    """
    coef, n = _reconcile(strategy_coef, n)

    if not torch.all(torch.isfinite(coef)):
        raise ConfigurationError(*("coef must contain only finite values",))
    if not torch.all(coef >= 0):
        raise ConfigurationError(
            *(f"coef must be non-negative, found min={coef.min().item()}",)
        )
    if len(coef) > 1:
        incr = coef[1:] - coef[:-1]
        max_incr = incr.max()
        if max_incr > 0:
            raise ConfigurationError(
                *(
                    f"coef must be non-increasing, found increase "
                    f"{max_incr.item()} at index {incr.argmax().item()}",
                )
            )
    if min_sep <= 0:
        raise ConfigurationError(*("min_sep must be positive",))
    if max_participations is not None and max_participations < 0:
        raise ConfigurationError(
            *(f"max_participations must be non-negative, found {max_participations}",)
        )

    k = sensitivity.minsep_true_max_participations(
        n=n, min_sep=min_sep, max_participations=max_participations
    )
    if k == 0:
        return coef.new_zeros(())
    if k == 1:
        return _l2_norm_squared(coef)

    padding = (min_sep - n) % min_sep
    full_coef = pad_coefs_to_n(coef, n + padding)
    vector = full_coef.reshape(-1, min_sep).cumsum(dim=0).flatten()
    if min_sep * k < len(vector):
        vector[min_sep * k :] = (
            vector[min_sep * k :] - vector[: len(vector) - min_sep * k]
        )
    return torch.dot(vector[:n], vector[:n])


def minsep_sensitivity_upper_bound(
    strategy_coef: torch.Tensor,
    min_sep: int,
    max_participations: int | None = None,
    n: int | None = None,
) -> torch.Tensor:
    """Return a safe min-separation sensitivity bound for any Toeplitz column.

    Replacing each coefficient by the reverse cumulative maximum of its
    absolute value gives the smallest pointwise non-negative, non-increasing
    majorant.  The triangle inequality reduces the arbitrary signed case to
    that majorant, where :func:`minsep_sensitivity_squared` is exact.
    """
    coef, n = _reconcile(strategy_coef, n)
    if not torch.all(torch.isfinite(coef)):
        raise ConfigurationError(*("coef must contain only finite values",))
    if min_sep <= 0:
        raise ConfigurationError(*("min_sep must be positive",))
    if max_participations is not None and max_participations < 0:
        raise ConfigurationError(
            *(f"max_participations must be non-negative, found {max_participations}",)
        )

    k = sensitivity.minsep_true_max_participations(
        n=n, min_sep=min_sep, max_participations=max_participations
    )
    if k == 0:
        return coef.new_zeros(())
    if k == 1:
        return _l2_norm_squared(coef)

    majorant = torch.cummax(coef.abs().flip(0), dim=0).values.flip(0)
    return minsep_sensitivity_squared(
        majorant,
        min_sep=min_sep,
        max_participations=max_participations,
        n=n,
    )


def per_query_error(
    *,
    strategy_coef: torch.Tensor | None = None,
    noising_coef: torch.Tensor | None = None,
    n: int | None = None,
    workload_coef: torch.Tensor | None = None,
    query_weights: torch.Tensor | None = None,
    skip_checks: bool = False,
) -> torch.Tensor:
    """Expected per-query squared error for a (banded) Toeplitz mechanism.

    Exactly one of ``strategy_coef`` and ``noising_coef`` must be provided.

    Args:
        strategy_coef: Toeplitz coefficients of the strategy matrix.
        noising_coef: Toeplitz coefficients of the noising matrix.
        n: Matrix size.
        workload_coef: Workload matrix coefficients (defaults to all ones).
        query_weights: Per-query workload row weights. Each returned squared
            error is multiplied by the corresponding weight squared.
        skip_checks: Skip input validation.

    Returns:
        Per-query expected squared error, tensor of length n.
    """
    if not skip_checks:
        checks.check_exactly_one(strategy_coef=strategy_coef, noising_coef=noising_coef)

    if strategy_coef is not None:
        strategy_coef, n = _reconcile(strategy_coef, n)
        if workload_coef is not None:
            workload_coef = pad_coefs_to_n(workload_coef, n)
        else:
            workload_coef = torch.ones(n, dtype=strategy_coef.dtype)
        B_coef = solve_banded(strategy_coef, workload_coef)
    else:
        assert noising_coef is not None
        noising_coef, n = _reconcile(noising_coef, n)
        if workload_coef is None:
            B_coef = torch.cumsum(noising_coef, dim=0)
        else:
            B_coef = multiply(workload_coef, noising_coef, n=n, skip_checks=skip_checks)

    error = torch.cumsum(B_coef**2, dim=0)
    if query_weights is None:
        return error

    query_weights = torch.as_tensor(
        query_weights, dtype=error.dtype, device=error.device
    )
    if query_weights.ndim != 1 or query_weights.shape[0] != n:
        raise ConfigurationError(
            *(
                f"query_weights must have shape ({n},), got {tuple(query_weights.shape)}",
            )
        )
    return error * query_weights.square()


def max_error(
    *,
    strategy_coef: torch.Tensor | None = None,
    noising_coef: torch.Tensor | None = None,
    n: int | None = None,
    workload_coef: torch.Tensor | None = None,
    query_weights: torch.Tensor | None = None,
    skip_checks: bool = False,
) -> torch.Tensor:
    """Max-over-iterations squared error for a Toeplitz mechanism."""
    return per_query_error(
        strategy_coef=strategy_coef,
        noising_coef=noising_coef,
        n=n,
        workload_coef=workload_coef,
        query_weights=query_weights,
        skip_checks=skip_checks,
    ).max()


def mean_error(
    *,
    strategy_coef: torch.Tensor | None = None,
    noising_coef: torch.Tensor | None = None,
    n: int | None = None,
    workload_coef: torch.Tensor | None = None,
    query_weights: torch.Tensor | None = None,
    skip_checks: bool = False,
) -> torch.Tensor:
    """Mean-over-iterations squared error for a Toeplitz mechanism."""
    return per_query_error(
        strategy_coef=strategy_coef,
        noising_coef=noising_coef,
        n=n,
        workload_coef=workload_coef,
        query_weights=query_weights,
        skip_checks=skip_checks,
    ).mean()


class ErrorOrLossFn(Protocol):
    """Protocol for error functions."""

    def __call__(
        self,
        *,
        strategy_coef: torch.Tensor,
        n: int | None = None,
        workload_coef: torch.Tensor | None = None,
        query_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


def loss(
    strategy_coef: torch.Tensor,
    n: int | None = None,
    error_fn: ErrorOrLossFn = mean_error,
    workload_coef: torch.Tensor | None = None,
    query_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Error of C on a workload under single participation.

    Returns error * sensitivity_squared.

    Args:
        strategy_coef: Toeplitz coefficients of C.
        n: Matrix size.
        error_fn: Error function (mean_error or max_error).
        workload_coef: Toeplitz coefficients of the workload matrix.
            Defaults to ``None`` (prefix-sum workload, i.e. all ones).
        query_weights: Per-query workload row weights.

    Returns:
        Total squared error times sensitivity.
    """
    strategy_coef, n = _reconcile(strategy_coef, n)
    error_kwargs: dict[str, torch.Tensor | int] = {
        "strategy_coef": strategy_coef,
        "n": n,
    }
    if workload_coef is not None:
        error_kwargs["workload_coef"] = workload_coef
    if query_weights is not None:
        error_kwargs["query_weights"] = query_weights
    error = error_fn(**error_kwargs)
    sens_sq = sensitivity_squared(strategy_coef, n)
    return error * sens_sq


mean_loss = functools.partial(loss, error_fn=mean_error)
max_loss = functools.partial(loss, error_fn=max_error)


def optimize(
    n: int,
    bands: int,
    strategy_coef: torch.Tensor | None = None,
    max_optimizer_steps: int = 250,
    loss_fn=mean_loss,
    workload_coef: torch.Tensor | None = None,
    query_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Optimize banded Toeplitz strategy for a given workload.

    The resulting strategy can be used for single- and multi-participation
    settings, as long as the minimum separation >= number of bands.

    Args:
        n: Number of iterations.
        bands: Number of bands in the Toeplitz matrix.
        strategy_coef: Optional initial coefficients.
        max_optimizer_steps: Maximum L-BFGS iterations.
        loss_fn: Loss function (default: mean_loss).
        workload_coef: Toeplitz coefficients of the workload matrix.
            Defaults to ``None`` (prefix-sum workload).  For momentum-SGD
            with coefficient β, pass ``[1, β, β², ...]``.
        query_weights: Per-query workload row weights, such as a
            learning-rate schedule materialized at each training step.

    Returns:
        Optimized coefficients with L2 norm 1.
    """
    loss_kwargs: dict[str, torch.Tensor | int] = {"n": n}
    if workload_coef is not None:
        loss_kwargs["workload_coef"] = workload_coef
    if query_weights is not None:
        loss_kwargs["query_weights"] = query_weights
    partial_loss = functools.partial(loss_fn, **loss_kwargs)

    if strategy_coef is None:
        strategy_coef = optimal_max_error_strategy_coefs(bands)
    if strategy_coef.shape[0] != bands:
        raise ConfigurationError(*(f"{strategy_coef.shape=} != {bands=}",))

    params = _lbfgs_optimize(
        partial_loss,
        strategy_coef,
        max_optimizer_steps=max_optimizer_steps,
    )
    return params / torch.linalg.norm(params)
