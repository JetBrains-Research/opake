"""Definition of the StreamingMatrix interface for memory-efficient matrix ops.

A StreamingMatrix represents a lower-triangular matrix that can be applied
to vectors in a streaming fashion (one element at a time), enabling
memory-efficient computation for matrix factorization mechanisms.

Example:
    >>> import torch
    >>> A = prefix_sum()
    >>> x = torch.arange(1, 5, dtype=torch.float64)
    >>> result = multiply_array(A, x)
    >>> print(result)
    tensor([ 1.,  3.,  6., 10.], dtype=torch.float64)
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import optree as _ot
import torch

from opake.exceptions import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Callable

State = TypeVar("State")


@dataclasses.dataclass(frozen=True)
class StreamingMatrix(Generic[State]):
    """A linear mapping x -> A x for a lower-triangular (streaming) A matrix.

    Via the attributes ``init_multiply`` and ``multiply_next``, this class
    allows efficient computation of A @ x in streaming fashion.

    The design encodes the fact that (Ax)[i] depends only on x[i] and
    state from computing (Ax)[0], ..., (Ax)[i-1], equivalent to A being
    lower-triangular.

    Attributes:
        init_multiply: Returns initial state given the first input element
            or a template with the expected shape.
        multiply_next: Returns (next_output, updated_state) from
            (next_input, current_state).
        row_norms_squared_fn: Optional closed-form implementation of
            :meth:`row_norms_squared`.  Builders that know the matrix
            structure (e.g. Toeplitz) attach one to avoid the generic
            O(n^2) probing; it must return the same values the probing
            fallback would.  Compositions drop it (falling back to
            probing) rather than propagate a stale closed form.
    """

    init_multiply: Callable[[Any], State]
    multiply_next: Callable[[Any, State], tuple[Any, State]]
    row_norms_squared_fn: Callable[[int], torch.Tensor] | None = None

    @classmethod
    def from_array_implementation(
        cls,
        init_multiply_fn: Callable[[torch.Tensor], State],
        multiply_next_fn: Callable[[torch.Tensor, State], tuple[torch.Tensor, State]],
    ) -> StreamingMatrix:
        """Construct a StreamingMatrix from single-array init/next functions.

        The provided functions operate on individual tensors and will be
        automatically lifted to operate on PyTrees of tensors.

        Args:
            init_multiply_fn: Initialize state from a single tensor (or its
                shape/dtype info).
            multiply_next_fn: Process one tensor element, returns
                (output, new_state).

        Returns:
            A StreamingMatrix that operates over PyTrees of tensors.
        """

        def lifted_init(abstract_value):
            if isinstance(abstract_value, torch.Tensor):
                return init_multiply_fn(abstract_value)
            # For PyTree inputs, create per-leaf states
            flat, spec = _ot.tree_flatten(abstract_value)
            flat_states = [init_multiply_fn(v) for v in flat]
            return (spec, flat_states)

        def lifted_next(value, state):
            if isinstance(value, torch.Tensor):
                return multiply_next_fn(value, state)
            # For PyTree inputs, apply per-leaf and reassemble
            spec, flat_states = state
            flat_values = _ot.tree_leaves(value)
            flat_outputs = []
            new_flat_states = []
            for v, s in zip(flat_values, flat_states, strict=True):
                out, ns = multiply_next_fn(v, s)
                flat_outputs.append(out)
                new_flat_states.append(ns)
            return spec.unflatten(flat_outputs), (spec, new_flat_states)

        return cls(lifted_init, lifted_next)

    def materialize(self, n: int) -> torch.Tensor:
        """Materialize this streaming matrix as a dense n x n tensor.

        Primarily for debugging and testing.

        Args:
            n: Size of the square matrix to materialize.

        Returns:
            An n x n dense tensor representation.
        """
        eye = torch.eye(n, dtype=torch.float64)
        return multiply_array(self, eye)

    def row_norms_squared(self, n: int) -> torch.Tensor:
        """Compute the row-wise L2 squared norms of the matrix.

        Given a StreamingMatrix B = A @ C^{-1}, this computes the per-query
        expected squared error of the factorization.

        Args:
            n: Number of rows to compute norms for.

        Returns:
            A tensor of length n with row-wise L2^2 norms.
        """
        if self.row_norms_squared_fn is not None:
            return self.row_norms_squared_fn(n)
        zero = torch.zeros(n, dtype=torch.float64)
        state = self.init_multiply(zero)
        norms = torch.zeros(n, dtype=torch.float64)
        for i in range(n):
            ei = torch.zeros(n, dtype=torch.float64)
            ei[i] = 1.0
            row, state = self.multiply_next(ei, state)
            norms[i] = torch.dot(row, row)
        return norms

    def __matmul__(self, other):
        """Multiply by another StreamingMatrix or a tensor."""
        if isinstance(other, StreamingMatrix):
            return multiply_streaming_matrices(self, other)
        elif isinstance(other, torch.Tensor):
            return multiply_array(self, other)
        else:
            return NotImplemented

    def __mul__(self, other: float) -> StreamingMatrix:
        """Multiply by a scalar."""
        return multiply_streaming_matrices(
            self, diagonal(torch.tensor([other], dtype=torch.float64))
        )


def scale_rows_and_columns(
    matrix: StreamingMatrix,
    row_scale: torch.Tensor | None = None,
    col_scale: torch.Tensor | None = None,
) -> StreamingMatrix:
    """Return a new StreamingMatrix with scaled rows and/or columns.

    Args:
        matrix: The matrix to wrap.
        row_scale: Multipliers for rows (diag(row_scale) @ matrix).
        col_scale: Multipliers for columns (matrix @ diag(col_scale)).

    Returns:
        The wrapped StreamingMatrix.
    """
    result = matrix
    if row_scale is not None:
        result = multiply_streaming_matrices(diagonal(row_scale), result)
    if col_scale is not None:
        result = multiply_streaming_matrices(result, diagonal(col_scale))
    return result


def multiply_array(A: StreamingMatrix, x: torch.Tensor) -> torch.Tensor:
    """Compute the matrix-vector product A @ x in streaming fashion.

    Args:
        A: A StreamingMatrix.
        x: A 2D tensor where each row is an element of the sequence.

    Returns:
        The result of A @ x.
    """
    n = x.shape[0]
    state = A.init_multiply(x[0])
    results = []
    for i in range(n):
        result_slice, state = A.multiply_next(x[i], state)
        results.append(result_slice)
    return torch.stack(results)


def multiply_streaming_matrices(
    A: StreamingMatrix,
    B: StreamingMatrix,
) -> StreamingMatrix:
    """Compose two StreamingMatrices: returns A @ B.

    Args:
        A: Left-hand side matrix.
        B: Right-hand side matrix.

    Returns:
        A StreamingMatrix representing A @ B.
    """

    def init_multiply(abstract_value):
        return A.init_multiply(abstract_value), B.init_multiply(abstract_value)

    def multiply_next(value, state):
        A_state, B_state = state
        inner, B_state = B.multiply_next(value, B_state)
        outer, A_state = A.multiply_next(inner, A_state)
        return outer, (A_state, B_state)

    return StreamingMatrix(init_multiply, multiply_next)


def identity() -> StreamingMatrix:
    """Create an identity StreamingMatrix."""
    return StreamingMatrix(lambda _: (), lambda value, _: (value, ()))


def prefix_sum() -> StreamingMatrix:
    """Create a StreamingMatrix for the prefix sum (lower-triangular ones).

    Returns:
        A StreamingMatrix representing the n x n lower-triangular matrix
        of all ones (cumulative sum workload).
    """

    def init_multiply(abstract_value):
        return torch.zeros_like(abstract_value)

    def multiply_next(value, state):
        result = state + value
        return result, result

    return StreamingMatrix.from_array_implementation(init_multiply, multiply_next)


def diagonal(diag: torch.Tensor) -> StreamingMatrix:
    """Create a StreamingMatrix representing a diagonal matrix.

    The returned StreamingMatrix is infinitely large; diagonal elements
    are taken from ``diag`` up to row n = diag.size, and equal diag[-1]
    beyond that point.

    Args:
        diag: A 1D tensor of diagonal elements.

    Returns:
        A StreamingMatrix representing the diagonal matrix.
    """

    def init_fn(abstract_value):
        return torch.tensor(0, dtype=torch.long)

    def next_fn(value, i):
        idx = min(int(i.item()), len(diag) - 1)
        return value * diag[idx], i + 1

    return StreamingMatrix.from_array_implementation(init_fn, next_fn)


def momentum_sgd_matrix(
    momentum: float = 0,
    learning_rates: torch.Tensor | None = None,
) -> StreamingMatrix:
    """Create a StreamingMatrix representing the momentum SGD workload.

    Args:
        momentum: Momentum coefficient (0 = no momentum).
        learning_rates: Per-step learning rates. Defaults to all ones.

    Returns:
        A StreamingMatrix encoding the optimizer workload.
    """
    if learning_rates is None:
        lr_sched = torch.ones(1, dtype=torch.float64)
    else:
        lr_sched = learning_rates.to(torch.float64)

    if lr_sched.min() <= 0.0:
        raise ConfigurationError(
            *(f"Learning rates must be positive. Found {learning_rates}",)
        )

    def init_multiply(abstract_value):
        dtype = torch.promote_types(abstract_value.dtype, lr_sched.dtype)
        zero = torch.zeros_like(abstract_value, dtype=dtype)
        return (torch.tensor(0, dtype=torch.long), zero.clone(), zero.clone())

    def multiply_next(value, state):
        index, momentum_buf, result = state
        momentum_buf = momentum * momentum_buf + value
        idx = min(int(index.item()), len(lr_sched) - 1)
        result = result + lr_sched[idx] * momentum_buf
        return result, (index + 1, momentum_buf, result)

    return StreamingMatrix.from_array_implementation(init_multiply, multiply_next)
