"""Buffered Linear Toeplitz (BLT) strategy matrices.

BLT matrices provide a memory-efficient parameterization of Toeplitz
matrices via a small number of buffers. This enables:
- O(num_buffers) memory for noise generation (vs O(n) for general Toeplitz)
- Closed-form sensitivity and error computation
- Efficient optimization via L-BFGS

References:
    - BLT: https://arxiv.org/abs/2404.16706
    - Multi-epoch BLT: https://arxiv.org/abs/2408.08868
    - Inversion theorem: https://arxiv.org/abs/2504.21413
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from opake.exceptions import ConfigurationError, OperationError

from . import (
    _sensitivity as sensitivity,
)
from . import (
    _streaming_matrix as streaming_matrix,
)
from . import (
    _toeplitz as toeplitz,
)
from ._engine import _internal_compute_dtype

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)
_MIN_BUFFER_DECAY_GAP = 1e-9
_NEAR_ZERO_OUTPUT_SCALE = 1e-8
_MAX_OPTIMIZATION_BUFFERS = 15

__all__ = [
    "BufferedToeplitz",
    "inverse",
    "inverse_as_streaming_matrix",
    "iteration_error",
    "materialize",
    "max_error",
    "optimize",
    "sensitivity_squared",
]


# ── Internal builder ────────────────────────────────────────────────────


@dataclasses.dataclass
class _StreamingMatrixBuilder:
    """Builder to convert a BLT to a StreamingMatrix.

    Attributes:
        buf_decay: Decay factors for each buffer, shape (num_buffers,).
        output_scale: Scale factors for output, shape (num_buffers,).
    """

    buf_decay: np.ndarray
    output_scale: np.ndarray

    @property
    def dtype(self) -> np.dtype:
        assert self.output_scale.dtype == self.buf_decay.dtype
        return self.output_scale.dtype

    def _init(self, abstract_value: torch.Tensor) -> torch.Tensor:
        num_buffers = self.buf_decay.shape[0]
        dtype = _internal_compute_dtype(abstract_value.dtype)
        zero = torch.zeros_like(abstract_value, dtype=dtype)
        return zero.unsqueeze(0).expand(num_buffers, *zero.shape).clone()

    def _read(self, state: torch.Tensor) -> torch.Tensor:
        output_scale = torch.tensor(
            self.output_scale,
            dtype=state.dtype,
            device=state.device,
        )
        return torch.tensordot(output_scale, state, dims=([0], [0]))

    def _update(
        self, state: torch.Tensor, next_rhs_value: torch.Tensor
    ) -> torch.Tensor:
        buf_decay = torch.tensor(
            self.buf_decay,
            dtype=state.dtype,
            device=state.device,
        )
        if len(buf_decay) == 0:
            return state
        bufs = torch.diag(buf_decay) @ state.reshape(len(buf_decay), -1)
        bufs = bufs.reshape(state.shape) + next_rhs_value
        return bufs

    def build(self) -> streaming_matrix.StreamingMatrix:
        """Returns a StreamingMatrix representing C.

        Implements Algorithm 2 of https://arxiv.org/abs/2408.08868
        """

        def multiply_next(xi, state):
            yi = xi + self._read(state)
            state = self._update(state, xi)
            return yi, state

        return streaming_matrix.StreamingMatrix.from_array_implementation(
            self._init, multiply_next
        )

    def build_inverse(self) -> streaming_matrix.StreamingMatrix:
        """Returns a StreamingMatrix representing C^{-1}.

        Implements Algorithm 3 of https://arxiv.org/abs/2408.08868
        """

        def inv_multiply_next(yi, state):
            xi = yi - self._read(state)
            state = self._update(state, xi)
            return xi, state

        return streaming_matrix.StreamingMatrix.from_array_implementation(
            self._init, inv_multiply_next
        )


# ── Data container ──────────────────────────────────────────────────────


@dataclasses.dataclass
class BufferedToeplitz:
    """A lower-triangular Toeplitz C parameterized as a BLT.

    If buf_decay = [d1, d2] and output_scale = [s1, s2], for n = 5::

        C = [[1    0   0   0   0 ]     t0 = 1
             [t1   1   0   0   0 ]     t1 = s1       + s2
             [t2   t1  1   0   0 ]     t2 = s1*d1    + s2*d2
             [t3   t2  t1  1   0 ]     t3 = s1*d1**2 + s2*d2**2
             [t4   t3  t2  t1  1 ]]    t4 = s1*d1**3 + s2*d2**3

    Attributes:
        buf_decay: Shape (nbuf,), diagonal elements.
        output_scale: Shape (nbuf,), output scaling factors.
    """

    buf_decay: torch.Tensor
    output_scale: torch.Tensor

    def validate(self) -> None:
        """Validate basic properties of the BLT parameters."""
        if not (self.buf_decay.ndim <= 1 and self.output_scale.ndim <= 1):
            raise ConfigurationError(
                *(
                    f"buf_decay and output_scale must be 0D or 1D, but: "
                    f"buf_decay.shape={self.buf_decay.shape}, "
                    f"output_scale.shape={self.output_scale.shape}",
                )
            )
        if self.buf_decay.shape != self.output_scale.shape:
            raise ConfigurationError(
                *(
                    f"buf_decay and output_scale must have same shape: "
                    f"{self.buf_decay.shape} != {self.output_scale.shape}",
                )
            )

    @classmethod
    def build(
        cls,
        buf_decay,
        output_scale,
        dtype=torch.float64,
    ) -> BufferedToeplitz:
        """Construct and canonicalize a BLT.

        Args:
            buf_decay: Decay parameters.
            output_scale: Scale parameters.
            dtype: Data type (float64 recommended for numerical stability).

        Returns:
            A canonicalized BufferedToeplitz.
        """
        blt = cls(
            buf_decay=torch.as_tensor(buf_decay, dtype=dtype),
            output_scale=torch.as_tensor(output_scale, dtype=dtype),
        )
        return canonicalize(blt)

    @classmethod
    def from_rational_approx_to_sqrt_x(
        cls,
        num_buffers: int,
        *,
        max_buf_decay: float = 1.0,
        max_pillutla_score: float | None = None,
        buf_decay_scale: float = 1.6,
        buf_decay_shift: int = -1,
    ) -> BufferedToeplitz:
        """Returns a BLT from a rational approximation of 1/sqrt(1-x).

        From Proposition 4.5 of https://arxiv.org/abs/2404.16706v2.

        Args:
            num_buffers: Number of buffers (degree of rational function).
            max_buf_decay: Maximum value for buf_decay.
            max_pillutla_score: If set, enforce this bound.
            buf_decay_scale: Factor scaling the dynamic range.
            buf_decay_shift: Shift for the counter k.

        Returns:
            A BufferedToeplitz initialization.
        """
        if num_buffers < 1:
            raise ConfigurationError(*("num_buffers must be >= 1.",))

        degree = num_buffers
        d1 = (degree + 1) // 2
        h = buf_decay_scale * np.pi / np.sqrt(2 * (d1 + 1))

        ks = np.arange(-d1 + 1, degree - d1 + 1, 1) + buf_decay_shift
        buf_decay = 1 / (1 + np.exp(2 * h * ks))
        output_scale = -(buf_decay**2) * np.exp(3 * h * ks)
        constant_term = np.sum(np.exp(h * ks) * buf_decay)
        output_scale /= constant_term

        # Build C^{-1}, then invert to get C
        inv_blt = cls.build(buf_decay=buf_decay, output_scale=output_scale)
        blt = inverse(inv_blt)
        buf_decay_t = blt.buf_decay.clone()
        output_scale_t = blt.output_scale.clone()

        # Scale buf_decay if needed
        largest = buf_decay_t[0]
        scale = min(1.0, max_buf_decay / float(largest))
        buf_decay_t = buf_decay_t * scale

        if max_pillutla_score is not None:
            score = float(torch.sum(output_scale_t / buf_decay_t))
            score_scale = min(1.0, max_pillutla_score / score)
            output_scale_t = output_scale_t * score_scale

        return cls.build(buf_decay=buf_decay_t, output_scale=output_scale_t)

    @property
    def dtype(self):
        return self.buf_decay.dtype

    @property
    def _num_buffers(self) -> int:
        return self.buf_decay.shape[0]

    def pillutla_score(self) -> float:
        """Returns the Pillutla Score of the BLT.

        From Theorem 1 of https://arxiv.org/abs/2504.21413.
        Score < 1 ensures well-behaved inverse.

        Returns:
            sum(output_scale / buf_decay).
        """
        return float(torch.sum(self.output_scale / self.buf_decay))

    def __repr__(self) -> str:
        return (
            f"BufferedToeplitz(buf_decay={self.buf_decay.tolist()}, "
            f"output_scale={self.output_scale.tolist()})"
        )


# ── Standalone functions (extracted from BufferedToeplitz methods) ───────


def canonicalize(blt: BufferedToeplitz) -> BufferedToeplitz:
    """Return a BLT with buf_decay in decreasing order.

    Args:
        blt: The BLT to canonicalize.

    Returns:
        A new BufferedToeplitz with sorted buf_decay.
    """
    blt.validate()
    idx = torch.argsort(blt.buf_decay, descending=True)
    return BufferedToeplitz(
        buf_decay=blt.buf_decay[idx],
        output_scale=blt.output_scale[idx],
    )


def toeplitz_coefs(blt: BufferedToeplitz, n: int) -> torch.Tensor:
    """Returns the Toeplitz coefficients for C.

    Args:
        blt: The BLT.
        n: Number of coefficients.

    Returns:
        Tensor of n Toeplitz coefficients.
    """
    if blt._num_buffers == 0:
        result = torch.zeros(n, dtype=blt.dtype)
        result[0] = 1.0
        return result
    powers = torch.arange(n - 1, dtype=blt.dtype)
    tmp = blt.buf_decay.unsqueeze(0) ** powers.unsqueeze(1) * blt.output_scale
    return torch.cat([torch.ones(1, dtype=blt.dtype), tmp.sum(dim=1)])


def materialize(blt: BufferedToeplitz, n: int) -> torch.Tensor:
    """Convert to dense n x n matrix.

    Args:
        blt: The BLT.
        n: Matrix dimension.

    Returns:
        Dense lower-triangular Toeplitz matrix.
    """
    return toeplitz.materialize_lower_triangular(toeplitz_coefs(blt, n))


def inverse(blt: BufferedToeplitz, skip_checks: bool = False) -> BufferedToeplitz:
    """Compute the BLT parameterization of C^{-1}.

    Implements the inverse computation from Lemma 5.2 of
    https://arxiv.org/abs/2404.16706.

    Args:
        blt: The BLT to invert.
        skip_checks: Skip validation checks.

    Returns:
        A BufferedToeplitz representing C^{-1}.
    """
    if blt._num_buffers == 0:
        return BufferedToeplitz.build(buf_decay=[], output_scale=[])

    if not skip_checks and len(blt.buf_decay) > 1:
        gap = min_buf_decay_gap(blt.buf_decay)
        if gap < _MIN_BUFFER_DECAY_GAP:
            raise ConfigurationError(
                *(
                    "Input BLT has buf_decay values too close: "
                    f"gap={float(gap)}, buf_decay={blt.buf_decay}",
                )
            )

    nbuf = len(blt.buf_decay)
    Theta = torch.diag(blt.buf_decay)
    omega = blt.output_scale
    alpha = torch.ones(nbuf, dtype=blt.dtype)

    Theta2 = Theta - torch.outer(omega, alpha)
    omega2 = -omega

    # Diagonalize Theta2
    evals = torch.linalg.eigvals(Theta2).real

    # Closed-form eigenvectors
    evecs = omega.unsqueeze(1) / (evals.unsqueeze(0) - blt.buf_decay.unsqueeze(1))
    einv = torch.linalg.inv(evecs)

    if not skip_checks:
        Theta2_diag = evecs @ torch.diag(evals) @ einv
        if not torch.allclose(Theta2_diag, Theta2, atol=1e-7):
            raise OperationError(
                *(
                    f"Error computing inverse: Theta2 mismatch.\n"
                    f"blt={blt}\nevecs={evecs}\nevals={evals}",
                )
            )

    omega3 = (einv @ omega2) * (evecs.T @ alpha)
    return BufferedToeplitz.build(
        buf_decay=evals,
        output_scale=omega3,
        dtype=blt.dtype,
    )


def _streaming_matrix_builder(blt: BufferedToeplitz) -> _StreamingMatrixBuilder:
    """Create a _StreamingMatrixBuilder from a BLT."""
    dtype = np.float64
    return _StreamingMatrixBuilder(
        output_scale=blt.output_scale.detach().numpy().astype(dtype),
        buf_decay=blt.buf_decay.detach().numpy().astype(dtype),
    )


def as_streaming_matrix(blt: BufferedToeplitz) -> streaming_matrix.StreamingMatrix:
    """Returns a StreamingMatrix representing C.

    Args:
        blt: The BLT.

    Returns:
        StreamingMatrix for C.
    """
    return _streaming_matrix_builder(blt).build()


def inverse_as_streaming_matrix(
    blt: BufferedToeplitz,
) -> streaming_matrix.StreamingMatrix:
    """Returns a StreamingMatrix representing C^{-1}.

    The result carries a closed-form ``row_norms_squared``: C^{-1} is
    lower-triangular Toeplitz, so its squared row norms are cumulative
    sums of its squared first-column coefficients, recovered with a
    single impulse pass through the same buffer recurrence the streaming
    inverse runs — O(num_buffers * n) instead of the
    O(num_buffers * n^2) generic probing, and numerically identical to
    it. (An equivalent rational-transfer-function filter is not: its
    monomial-basis polynomials are ill-conditioned for many
    near-one decays.)

    Args:
        blt: The BLT.

    Returns:
        StreamingMatrix for C^{-1}.
    """

    def _row_norms_squared(n: int) -> torch.Tensor:
        if n == 0:
            return torch.zeros(0, dtype=torch.float64, device="cpu")
        buf_decay = blt.buf_decay.detach().cpu().to(torch.float64).numpy()
        output_scale = blt.output_scale.detach().cpu().to(torch.float64).numpy()
        inv_coefs = np.zeros(n)
        inv_coefs[0] = 1.0
        state = np.ones_like(buf_decay)
        for t in range(1, n):
            value = -output_scale.dot(state)
            state = buf_decay * state + value
            inv_coefs[t] = value
        return torch.cumsum(torch.from_numpy(inv_coefs).square(), dim=0)

    return dataclasses.replace(
        _streaming_matrix_builder(blt).build_inverse(),
        row_norms_squared_fn=_row_norms_squared,
    )


# ── Helper functions (unchanged) ────────────────────────────────────────


def min_buf_decay_gap(buf_decay: torch.Tensor) -> torch.Tensor:
    """Returns the minimum gap between buf_decay parameters.

    Args:
        buf_decay: The buf_decay parameters.

    Returns:
        min_{i!=j} |theta[i] - theta[j]|
    """
    theta = torch.as_tensor(buf_decay)
    A = theta.unsqueeze(1) - theta.unsqueeze(0)
    A = A + torch.diag(torch.full_like(theta, float("inf")))
    return torch.min(torch.abs(A))


def _gt_zero_penalty(x: torch.Tensor) -> torch.Tensor:
    """Penalize values to enforce x > 0."""
    return -torch.log(x).sum()


def _lt_one_penalty(x: torch.Tensor) -> torch.Tensor:
    """Penalize values to enforce x < 1."""
    return -torch.log(1 - x).sum()


def _lt_penalty(x: torch.Tensor, upper_bound: float) -> torch.Tensor:
    """Penalize values to enforce x < upper_bound."""
    return -torch.log(upper_bound - x).sum()


def _lt_zero_penalty(x: torch.Tensor) -> torch.Tensor:
    """Penalize values to enforce x < 0."""
    return -torch.log(-x).sum()


def geometric_sum(
    a: torch.Tensor, r: torch.Tensor, num: float = float("inf")
) -> torch.Tensor:
    """Compute a + a*r + a*r^2 + ... + a*r^(num-1).

    Uses a quadratic Taylor approximation near r=1 for numerical stability.
    The threshold for switching to the Taylor series is chosen to minimise
    the error in the *gradient* w.r.t. ``r`` (not just the value), following
    the JAX-Privacy implementation.

    Args:
        a: Scale factor(s).
        r: Common ratio(s), requires |r| < 1.
        num: Number of terms (inf for infinite series).

    Returns:
        The sum.
    """
    if math.isinf(num):
        return a / (1 - r)

    n_val = torch.as_tensor(num, dtype=torch.float64)
    n = num

    # Adaptive threshold: calibrated to minimise gradient error.
    # Constants from regression on numerical experiments (see JAX-Privacy).
    _SLOPE = 0.53018965
    _INTERCEPT = 3.33503185
    pow_threshold = _INTERCEPT + _SLOPE * torch.log(n_val)
    threshold = 1 - 10 ** (-pow_threshold)

    use_direct = r < threshold

    # Direct computation (safe when r is not near 1)
    safe_r = torch.where(use_direct, r, torch.zeros_like(r))
    direct = a * (1 - safe_r**n) / (1 - safe_r)

    # Quadratic Taylor polynomial at r = 1 (from sympy)
    x0 = n - 1
    x1 = r - 1
    series = (1 / 6) * a * n * (x0 * x1**2 * (n - 2) + 3 * x0 * x1 + 6)

    return torch.where(use_direct, direct, series)


# ── BLT error and sensitivity (already standalone) ──────────────────────


def sensitivity_squared(blt: BufferedToeplitz, n: float) -> torch.Tensor:
    """Compute sensitivity^2 for a BLT strategy matrix C.

    See Lemma 5.3 of https://arxiv.org/abs/2404.16706.

    Args:
        blt: The BLT representing C.
        n: Number of iterations (can be inf).

    Returns:
        Maximum column norm squared of C.
    """
    if blt._num_buffers == 0:
        return torch.tensor(1.0, dtype=torch.float64)

    if torch.any(blt.buf_decay > 1):
        return torch.tensor(float("inf"), dtype=torch.float64)

    omega = blt.output_scale
    theta = blt.buf_decay
    num = n - 1

    omega_pairs = omega.unsqueeze(0) * omega.unsqueeze(1)
    theta_pairs = theta.unsqueeze(0) * theta.unsqueeze(1)
    geo_pairs = geometric_sum(omega_pairs, theta_pairs, num=num)
    return 1.0 + geo_pairs.sum()


def max_error(inv_blt: BufferedToeplitz, n: float) -> torch.Tensor:
    """Max squared error for any iteration 0, ..., n-1.

    Args:
        inv_blt: BLT representing C^{-1}.
        n: Number of iterations.

    Returns:
        Max squared error.
    """
    return iteration_error(inv_blt, n - 1)


def iteration_error(inv_blt: BufferedToeplitz, i: float) -> torch.Tensor:
    """Compute the squared error on iteration i.

    For BLT matrices, the max error through iteration i is achieved at
    iteration i itself.

    Implements Lemma 5.4 of https://arxiv.org/abs/2404.16706.

    Args:
        inv_blt: BLT representing C^{-1}.
        i: Iteration index (0-based).

    Returns:
        Squared error at iteration i.
    """
    if inv_blt._num_buffers == 0:
        return torch.tensor(float(i + 1), dtype=torch.float64)

    n = i + 1
    omega = inv_blt.output_scale
    theta = inv_blt.buf_decay

    # Vectorised over buffers using broadcasting
    s1 = robust_max_error_Gamma_j(omega, theta, n).sum()
    s2 = robust_max_error_Gamma_jk(
        omega,
        theta,
        omega.unsqueeze(1),
        theta.unsqueeze(1),
        n,
    ).sum()

    return n * (1 + 2 * s1 + s2)


def limit_max_error(inv_blt: BufferedToeplitz) -> torch.Tensor:
    """Closed-form max squared error per iteration as n -> infinity.

    This is the limit of ``max_error(inv_blt, n) / n`` as n grows.
    Uses direct arithmetic on the BLT parameters (no loop over n),
    making it O(num_buffers^2).

    Args:
        inv_blt: BLT representing C^{-1}.

    Returns:
        The asymptotic per-iteration max squared error.
    """
    if inv_blt._num_buffers == 0:
        return torch.tensor(1.0, dtype=torch.float64)

    omega = inv_blt.output_scale
    theta = inv_blt.buf_decay

    omega_pairs = omega.unsqueeze(0) * omega.unsqueeze(1)
    theta_complement_pairs = (1 - theta).unsqueeze(0) * (1 - theta).unsqueeze(1)
    cross_term_sum = (omega_pairs / theta_complement_pairs).sum()

    return 1 + 2 * (omega / (1 - theta)).sum() + cross_term_sum


def limit_max_loss(blt: BufferedToeplitz) -> torch.Tensor:
    """Closed-form loss (error * sensitivity^2) as n -> infinity.

    Composes ``limit_max_error`` with ``sensitivity_squared(blt, inf)``.

    Args:
        blt: BLT representing the strategy C.

    Returns:
        The asymptotic loss.
    """
    sens_sq = sensitivity_squared(blt, n=float("inf"))
    inv_blt = inverse(blt, skip_checks=True)
    return limit_max_error(inv_blt) * sens_sq


def _max_error_Gamma_j(omega, theta, n):
    """Direct computation of Gamma_j for max error."""
    return (omega / (1.0 - theta)) * (
        1 - geometric_sum(torch.ones_like(theta), theta, num=n) / n
    )


def _max_error_Gamma_j_series(omega, theta, n):
    """Taylor series approximation to _max_error_Gamma_j near theta=1.

    Auto-generated via sympy (see robust_max_error_for_blts.ipynb in
    JAX-Privacy).
    """
    x0 = theta - 1
    x1 = omega * (n - 2) * (n - 1)
    return (
        -omega * (1 / 2 - n / 2) + (1 / 24) * x0**2 * x1 * (n - 3) + (1 / 6) * x0 * x1
    )


def robust_max_error_Gamma_j(omega, theta, n):
    """Robustly compute Gamma_j, dispatching to Taylor series near theta=1.

    Uses empirically-calibrated thresholds from JAX-Privacy to decide
    when to switch from the direct formula to a Taylor series.
    """
    n_t = torch.as_tensor(n, dtype=torch.float64)
    _J_SLOPE = 0.43877484
    _J_INTERCEPT = 2.91215085
    power = _J_INTERCEPT + _J_SLOPE * torch.log(n_t)
    threshold = 1 - 10 ** (-power)

    use_direct = theta < threshold
    safe_theta = torch.where(use_direct, theta, torch.zeros_like(theta))
    v0 = _max_error_Gamma_j(omega, safe_theta, n)
    v1 = _max_error_Gamma_j_series(omega, theta, n)
    return torch.where(use_direct, v0, v1)


def _max_error_Gamma_jk(omega1, theta1, omega2, theta2, n):
    """Direct computation of cross term Gamma_jk for max error."""
    temp1 = omega1 * omega2 / ((1 - theta1) * (1 - theta2))
    gs1 = geometric_sum(torch.ones_like(theta1), theta1, num=n)
    gs2 = geometric_sum(torch.ones_like(theta2), theta2, num=n)
    gs12 = geometric_sum(torch.ones_like(theta1), theta1 * theta2, num=n)
    temp2 = (n - gs1 - gs2 + gs12) / n
    return temp1 * temp2


def _max_error_Gamma_jk_series_j(omega1, theta1, omega2, theta2, n):
    """Taylor series in theta1 near 1, theta2 handled directly.

    Auto-generated via sympy (see robust_max_error_for_blts.ipynb in
    JAX-Privacy).
    """
    # fmt: off
    x0 = theta2 - 1
    x1 = theta2 ** (n + 1)
    x2 = -x1
    x3 = 6 * x0
    x4 = theta2 ** n
    x5 = n - 1
    x6 = theta1 - 1
    x7 = theta2 ** (n + 2)
    return (-1 / 6 * omega1 * omega2 * (
        n * x0 ** 3 * (3 * n + x5 * x6 * (n - 2) - 3) + n * x3 * (x2 + x4) - x3 * (theta2 + x2)
        + 3 * x6 * (n * x0 * (-x1 * x5 + x4 * x5) - 2 * n * (x1 - x7)
                     + 2 * theta2 ** 2 - 2 * x7)) / (n * x0 ** 4))
    # fmt: on


def _max_error_Gamma_jk_series_jk(omega1, theta1, omega2, theta2, n):
    """Taylor series in both theta1 and theta2 near 1.

    Auto-generated via sympy (see robust_max_error_for_blts.ipynb in
    JAX-Privacy).
    """
    # fmt: off
    x0 = n ** 2
    x1 = 3 * n ** 3 + 9 * n - 10 * x0 - 2
    return ((1 / 24) * omega1 * omega2 * (-12 * n + 8 * x0 + x1 * (theta1 - 1)
                                           + x1 * (theta2 - 1) + 4))
    # fmt: on


def robust_max_error_Gamma_jk(omega1, theta1, omega2, theta2, n):
    """Robustly compute Gamma_jk, dispatching to Taylor series.

    Uses three-way dispatch based on how close theta1/theta2 are to 1:
    - Both far from 1: direct formula
    - theta1 near 1, theta2 far: Taylor series in theta1
    - Both near 1: Taylor series in both

    The series_j approximation requires theta1 >= theta2, so we sort.
    """
    n_t = torch.as_tensor(n, dtype=torch.float64)
    # Ensure theta1 >= theta2 (required by series_j)
    theta1, theta2 = torch.maximum(theta1, theta2), torch.minimum(theta1, theta2)

    _JK_SLOPE = 0.35321577
    _JK_INTERCEPT = 2.81518052
    power = _JK_INTERCEPT + _JK_SLOPE * torch.log(n_t)
    threshold = 1 - 10 ** (-power)

    v0_predicate = theta1 < threshold
    v1_predicate = theta2 < threshold

    # Avoid inf/nan in untaken branches
    safe_theta1 = torch.where(v0_predicate, theta1, torch.zeros_like(theta1))
    safe_theta2_v0 = torch.where(v0_predicate, theta2, torch.zeros_like(theta2))
    v0 = _max_error_Gamma_jk(omega1, safe_theta1, omega2, safe_theta2_v0, n)

    safe_theta2_v1 = torch.where(v1_predicate, theta2, torch.zeros_like(theta2))
    v1 = _max_error_Gamma_jk_series_j(omega1, theta1, omega2, safe_theta2_v1, n)
    v2 = _max_error_Gamma_jk_series_jk(omega1, theta1, omega2, theta2, n)

    return torch.where(
        v0_predicate,
        v0,
        torch.where(v1_predicate, v1, v2),
    )


# ── Loss function and optimization ─────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class LossFn:
    """Encapsulates the loss to be optimized for a BLT.

    Attributes:
        error_for_inv: Error function taking C^{-1} BLT.
        sensitivity_squared_fn: Sensitivity function taking C BLT.
        n: Number of iterations.
        min_sep: Minimum separation of participations.
        max_participations: Maximum participations.
        penalty_strength: Multiplier for constraint penalties.
    """

    error_for_inv: Callable[[BufferedToeplitz], float]
    sensitivity_squared_fn: Callable[[BufferedToeplitz], float]
    n: int
    min_sep: int
    max_participations: int
    penalty_strength: float = 1e-8
    max_second_coef: float = 1.0
    min_theta_gap: float = 1e-12

    @classmethod
    def build_closed_form_single_participation(cls, n: int, **kwargs) -> LossFn:
        """Construct LossFn for single participation max-error."""
        return cls(
            error_for_inv=functools.partial(max_error, n=n),
            sensitivity_squared_fn=functools.partial(sensitivity_squared, n=n),
            n=n,
            min_sep=1,
            max_participations=1,
            **kwargs,
        )

    @classmethod
    def build_min_sep(
        cls,
        n: int,
        error: str = "max",
        min_sep: int = 1,
        max_participations: int | None = None,
        workload_coef: torch.Tensor | None = None,
        query_weights: torch.Tensor | None = None,
        **kwargs,
    ) -> LossFn:
        """Construct LossFn for min-sep participation."""

        def mean_error_fn(inv_blt):
            return toeplitz.mean_error(
                noising_coef=toeplitz_coefs(inv_blt, n),
                workload_coef=workload_coef,
                query_weights=query_weights,
            )

        def max_error_fn(inv_blt):
            return toeplitz.max_error(
                noising_coef=toeplitz_coefs(inv_blt, n),
                workload_coef=workload_coef,
                query_weights=query_weights,
            )

        if error == "mean":
            error_fn = mean_error_fn
        elif error == "max":
            error_fn = max_error_fn
        else:
            raise ConfigurationError(*(f"Unknown error={error}",))

        def minsep_sens_sq(blt):
            return toeplitz.minsep_sensitivity_squared(
                strategy_coef=toeplitz_coefs(blt, n),
                min_sep=min_sep,
                max_participations=max_participations,
            )

        return cls(
            n=n,
            error_for_inv=error_fn,
            sensitivity_squared_fn=minsep_sens_sq,
            min_sep=min_sep,
            max_participations=sensitivity.minsep_true_max_participations(
                n=n,
                min_sep=min_sep,
                max_participations=max_participations,
            ),
            **kwargs,
        )

    def compute_penalties(
        self, blt: BufferedToeplitz, inv_blt: BufferedToeplitz
    ) -> torch.Tensor:
        """Compute log-barrier penalties enforcing the BLT feasible set.

        Penalties enforce:
        - buf_decay in (0, 1)
        - output_scale > 0
        - inv_buf_decay in (0, 1)
        - inv_output_scale < 0
        - buf_decay gap >= min_theta_gap
        - second Toeplitz coefficient <= max_second_coef
        - Pillutla score < 1

        Args:
            blt: The strategy BLT (C).
            inv_blt: The inverse BLT (C^{-1}).

        Returns:
            Scalar penalty value.
        """
        penalty = torch.tensor(0.0, dtype=torch.float64)

        if blt._num_buffers == 0:
            return penalty

        # buf_decay in (0, 1)
        penalty = penalty + _gt_zero_penalty(blt.buf_decay)
        penalty = penalty + _lt_one_penalty(blt.buf_decay)

        # output_scale > 0
        penalty = penalty + _gt_zero_penalty(blt.output_scale)

        # inv_buf_decay in (0, 1)
        penalty = penalty + _gt_zero_penalty(inv_blt.buf_decay)
        penalty = penalty + _lt_one_penalty(inv_blt.buf_decay)

        # inv_output_scale < 0
        penalty = penalty + _lt_zero_penalty(inv_blt.output_scale)

        # buf_decay gap separation
        if blt._num_buffers > 1:
            gap = min_buf_decay_gap(blt.buf_decay)
            penalty = penalty - torch.log(gap - self.min_theta_gap)

        # Second coefficient: sum(output_scale) <= max_second_coef
        second_coef = torch.sum(blt.output_scale)
        penalty = penalty + _lt_penalty(second_coef, self.max_second_coef)

        # Pillutla score < 1
        pillutla = torch.sum(blt.output_scale / blt.buf_decay)
        penalty = penalty + _lt_one_penalty(pillutla.unsqueeze(0))

        return penalty


def loss(
    loss_fn: LossFn, blt: BufferedToeplitz, skip_checks: bool = False
) -> torch.Tensor:
    """Returns the loss (not including penalties).

    Args:
        loss_fn: The loss function configuration.
        blt: The BLT to evaluate.
        skip_checks: Skip validation.

    Returns:
        error * sensitivity_squared.
    """
    try:
        inv_blt = inverse(blt, skip_checks=skip_checks)
    except (RuntimeError, ValueError):
        return torch.tensor(float("inf"), dtype=torch.float64)
    error = loss_fn.error_for_inv(inv_blt)
    sens_sq = loss_fn.sensitivity_squared_fn(blt)
    return error * sens_sq


def penalized_loss(
    loss_fn: LossFn, blt: BufferedToeplitz, inv_blt: BufferedToeplitz
) -> torch.Tensor:
    """Returns loss + penalty_strength * penalties.

    This is the objective function used during L-BFGS optimization.

    Args:
        loss_fn: The loss function configuration.
        blt: The strategy BLT (C).
        inv_blt: The inverse BLT (C^{-1}).

    Returns:
        loss + penalties.
    """
    error = loss_fn.error_for_inv(inv_blt)
    sens_sq = loss_fn.sensitivity_squared_fn(blt)
    loss_val = error * sens_sq
    penalties = loss_fn.compute_penalties(blt, inv_blt)
    return loss_val + loss_fn.penalty_strength * penalties


def blt_pair_from_theta_pair(
    theta: torch.Tensor, theta_hat: torch.Tensor
) -> tuple[BufferedToeplitz, BufferedToeplitz]:
    """Compute BLTs (C, C_inv) from theta and theta_hat.

    Implements Lemma 5.2 of https://arxiv.org/abs/2404.16706.

    Args:
        theta: Array of thetas for C.
        theta_hat: Array of thetas for C^{-1}.

    Returns:
        Tuple (C_blt, C_inv_blt).
    """
    theta = torch.as_tensor(theta, dtype=torch.float64)
    theta_hat = torch.as_tensor(theta_hat, dtype=torch.float64)

    def get_omega(th, th_hat):
        numerators = torch.prod(th.unsqueeze(1) - th_hat.unsqueeze(0), dim=1)
        A = th.unsqueeze(1) - th.unsqueeze(0)
        A = A + torch.diag(torch.ones_like(th))
        A[range(len(th)), range(len(th))] = 1.0
        denominators = torch.prod(A, dim=1)
        return numerators / denominators

    return (
        BufferedToeplitz.build(
            output_scale=get_omega(theta, theta_hat),
            buf_decay=theta,
        ),
        BufferedToeplitz.build(
            output_scale=get_omega(theta_hat, theta),
            buf_decay=theta_hat,
        ),
    )


def get_init_blt(
    num_buffers: int = 3,
    init_blt: BufferedToeplitz | None = None,
) -> BufferedToeplitz:
    """Returns an initial BLT for optimization.

    Args:
        num_buffers: Number of buffers.
        init_blt: Optional explicit initialization.

    Returns:
        A BufferedToeplitz for initialization.
    """
    if init_blt is None:
        if num_buffers == 0:
            return BufferedToeplitz.build(buf_decay=[], output_scale=[])
        return BufferedToeplitz.from_rational_approx_to_sqrt_x(
            num_buffers=num_buffers,
            max_buf_decay=1 - 1e-6,
            max_pillutla_score=1 - 1e-6,
        )

    if len(init_blt.buf_decay) != num_buffers:
        raise ConfigurationError(
            *(
                f"num_buffers={num_buffers} does not match "
                f"len(init_blt.buf_decay)={len(init_blt.buf_decay)}",
            )
        )
    return init_blt


@dataclasses.dataclass(frozen=True)
class Parameterization:
    """A reparameterization of a BufferedToeplitz for L-BFGS optimization.

    Provides the mapping between optimizable parameters and the BLT pair
    (strategy, noising) used for loss computation.

    Attributes:
        params_from_blt: Extracts optimizable parameters from a BLT.
        blt_and_inverse_from_params: Reconstructs (C, C^{-1}) from parameters.
    """

    params_from_blt: Callable[[BufferedToeplitz], torch.Tensor]
    blt_and_inverse_from_params: Callable[
        ..., tuple[BufferedToeplitz, BufferedToeplitz]
    ]

    @classmethod
    def buf_decay_pair(cls) -> Parameterization:
        """Parameterization over the pair (theta, theta_hat).

        Optimises the buf_decay arrays of C and C^{-1} jointly, which is
        more numerically stable than optimising the full BLT and avoids
        the SVD inside ``inverse()``.

        Returns:
            A Parameterization.
        """

        def params_from_blt(blt: BufferedToeplitz) -> torch.Tensor:
            inv_blt = inverse(blt)
            return torch.cat([blt.buf_decay, inv_blt.buf_decay])

        def blt_and_inverse_from_params(
            params: torch.Tensor,
        ) -> tuple[BufferedToeplitz, BufferedToeplitz]:
            half = len(params) // 2
            theta = params[:half]
            theta_hat = params[half:]
            return blt_pair_from_theta_pair(theta, theta_hat)

        return cls(
            params_from_blt=params_from_blt,
            blt_and_inverse_from_params=blt_and_inverse_from_params,
        )

    @classmethod
    def interlaced_decay_pair(cls) -> Parameterization:
        """Parameterize strategy and inverse with interlaced decays.

        Ratios in ``(0, 1)`` keep ``theta_0 > theta_hat_0 > ... > 0``.
        The BLT residue formula then gives positive strategy output scales and
        negative inverse output scales, so strategy coefficients are
        non-negative and non-increasing by construction.
        """

        def params_from_blt(blt: BufferedToeplitz) -> torch.Tensor:
            blt = canonicalize(blt)
            inv_blt = canonicalize(inverse(blt))
            decays = torch.stack((blt.buf_decay, inv_blt.buf_decay), dim=1).flatten()
            valid = (
                torch.all(decays > 0)
                & (decays[0] < 1)
                & torch.all(decays[:-1] > decays[1:])
            )
            if not bool(valid):
                raise ConfigurationError(
                    *("buffer decays must strictly interlace in (0, 1)",)
                )
            return torch.cat((decays[:1], decays[1:] / decays[:-1]))

        def blt_and_inverse_from_params(
            params: torch.Tensor,
        ) -> tuple[BufferedToeplitz, BufferedToeplitz]:
            decays = torch.cumprod(params, dim=0)
            return blt_pair_from_theta_pair(decays[::2], decays[1::2])

        return cls(
            params_from_blt=params_from_blt,
            blt_and_inverse_from_params=blt_and_inverse_from_params,
        )


def get_parameterized_loss(
    param: Parameterization, loss_fn: LossFn
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Returns a scalar loss function over the flat parameter vector.

    Args:
        param: The parameterization.
        loss_fn: The loss function.

    Returns:
        A callable mapping flat parameters to scalar loss.
    """
    return lambda params: penalized_loss(
        loss_fn, *param.blt_and_inverse_from_params(params)
    )


def optimize_loss(
    loss_fn: LossFn,
    num_buffers: int = 1,
    init_blt: BufferedToeplitz | None = None,
    parameterization: Parameterization | None = None,
    max_optimizer_steps: int = 600,
    **kwargs,
) -> tuple[BufferedToeplitz, torch.Tensor]:
    """Optimise a BLT for a fixed number of buffers using L-BFGS.

    This is the main workhorse: it initialises a BLT, converts it to an
    optimisable parameter vector via ``parameterization``, runs L-BFGS,
    and returns the resulting BLT together with its (unpenalised) loss.

    Args:
        loss_fn: The loss function to minimise.
        num_buffers: Number of buffers for the BLT.
        init_blt: Optional explicit initial BLT.
        parameterization: Reparameterization to use.  Defaults to
            ``Parameterization.interlaced_decay_pair()``.
        max_optimizer_steps: Maximum L-BFGS iterations.
        **kwargs: Additional keyword arguments forwarded to
            ``_lbfgs_optimize``.

    Returns:
        Tuple ``(blt, loss_val)`` where *blt* is the optimised BLT and
        *loss_val* is the unpenalised loss value.

    Raises:
        OperationError: If the optimizer leaves the sensitivity domain or
            produces a non-finite loss.
    """
    from ._toeplitz import _lbfgs_optimize

    if num_buffers == 0:
        blt = BufferedToeplitz.build(buf_decay=[], output_scale=[])
        return blt, loss(loss_fn, blt)

    if parameterization is None:
        parameterization = Parameterization.interlaced_decay_pair()

    blt = get_init_blt(num_buffers=num_buffers, init_blt=init_blt)
    params = parameterization.params_from_blt(blt)

    loss_fn_to_optimize = get_parameterized_loss(parameterization, loss_fn)

    # Interlaced decay ratios must stay in (0, 1).
    if "bounds" not in kwargs:
        eps = 1e-9
        kwargs["bounds"] = [(eps, 1.0 - eps)] * len(params)

    params = _lbfgs_optimize(
        loss_fn_to_optimize,
        params,
        max_optimizer_steps=max_optimizer_steps,
        **kwargs,
    )

    blt, _ = parameterization.blt_and_inverse_from_params(params)
    blt = canonicalize(blt)

    try:
        loss_val = loss(loss_fn, blt)
    except ConfigurationError as error:
        raise OperationError(
            *(f"Optimization produced BLT outside the sensitivity domain:\n{blt}",)
        ) from error
    if not torch.isfinite(loss_val):
        raise OperationError(
            *(f"Optimization produced BLT with non-finite loss {loss_val}:\n{blt}",)
        )

    if torch.any(torch.abs(blt.output_scale) < _NEAR_ZERO_OUTPUT_SCALE):
        logger.warning(
            "BLT has near-zero output_scale parameters, which "
            "means some buffers are ignored. Consider re-optimizing "
            "with a smaller number of buffers.\n%s",
            blt,
        )
    return blt, loss_val


def _optimize_increasing_nbuf(
    opt_blt_and_loss_fn: Callable[[int], tuple[BufferedToeplitz, float]],
    min_buffers: int = 0,
    max_buffers: int = 10,
    rtol: float = 1.02,
) -> BufferedToeplitz:
    """Increase num_buffers until improvement < rtol."""
    prev_blt, prev_loss = opt_blt_and_loss_fn(min_buffers)
    for nbuf in range(min_buffers + 1, max_buffers + 1):
        try:
            blt, loss_val = opt_blt_and_loss_fn(nbuf)
        except RuntimeError as err:
            logger.warning("Optimization failed for %d buffers: %s", nbuf, err)
            blt, loss_val = None, float("inf")

        if rtol * loss_val < prev_loss:
            prev_blt, prev_loss = blt, loss_val
        else:
            return prev_blt
    return prev_blt


def optimize(
    *,
    n: int,
    min_sep: int = 1,
    max_participations: int | None = 1,
    error: str = "max",
    min_buffers: int = 0,
    max_buffers: int = 10,
    rtol: float = 1.01,
    workload_coef: torch.Tensor | None = None,
    query_weights: torch.Tensor | None = None,
    **kwargs,
) -> BufferedToeplitz:
    """Compute an optimised BLT with dynamically-chosen num_buffers.

    Uses L-BFGS to refine BLT parameters at each buffer count, then
    increases the buffer count until improvement drops below ``rtol``.

    For single-participation max-error, uses closed-form sensitivity and
    error from https://arxiv.org/abs/2404.16706, so optimisation time is
    essentially independent of ``n``.  For multi-participation or mean
    error, materialises Toeplitz coefficients (benefits from GPU for
    large ``n``).

    Args:
        n: Number of iterations the mechanism is optimised for.
        min_sep: Minimum separation of participations.
        max_participations: Maximum participations.
        error: ``'max'`` or ``'mean'``.
        min_buffers: Minimum buffers to try (inclusive).
        max_buffers: Maximum buffers to try (inclusive).
        rtol: Relative tolerance for loss improvement.
        workload_coef: Toeplitz coefficients of the workload matrix.
            Defaults to ``None`` (prefix-sum workload).  For momentum-SGD
            with coefficient β, pass ``[1, β, β², ...]``.
        query_weights: Per-query workload row weights.
        **kwargs: Additional keyword arguments forwarded to
            ``optimize_loss`` / ``_lbfgs_optimize``.

    Returns:
        An optimised BLT.
    """
    if max_buffers > _MAX_OPTIMIZATION_BUFFERS:
        raise ConfigurationError(
            *("max_buffers > 15 will likely cause numerical issues.",)
        )

    k = sensitivity.minsep_true_max_participations(
        n=n, min_sep=min_sep, max_participations=max_participations
    )

    # Closed-form path only works for an unweighted prefix-sum workload.
    if k == 1 and error == "max" and workload_coef is None and query_weights is None:
        loss_fn = LossFn.build_closed_form_single_participation(n=n)
    else:
        loss_fn = LossFn.build_min_sep(
            n=n,
            error=error,
            min_sep=min_sep,
            max_participations=max_participations,
            workload_coef=workload_coef,
            query_weights=query_weights,
        )

    return _optimize_increasing_nbuf(
        opt_blt_and_loss_fn=lambda nbuf: optimize_loss(
            loss_fn=loss_fn,
            num_buffers=nbuf,
            parameterization=Parameterization.interlaced_decay_pair(),
            **kwargs,
        ),
        min_buffers=min_buffers,
        max_buffers=max_buffers,
        rtol=rtol,
    )
