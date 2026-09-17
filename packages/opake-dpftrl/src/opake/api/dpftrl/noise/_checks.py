"""Utilities for validating matrix factorization inputs.

Provides checks for matrices used in the factorization A = B @ C and X = C.T @ C,
ensuring they satisfy expected properties (lower-triangular, symmetric, finite, etc.).
"""

from __future__ import annotations

import torch

from opake.exceptions import ConfigurationError


def _pad(s: str) -> str:
    return s + " " if s else ""


def check_lower_triangular(M: torch.Tensor, name: str = "", **allclose_kwargs) -> None:
    """Check that M is lower-triangular."""
    if not torch.allclose(M, torch.tril(M), **allclose_kwargs):
        raise ConfigurationError(
            *(f"Matrix {_pad(name)}should be lower-triangular, found\n{M}",)
        )


def check_is_matrix(M: torch.Tensor, name: str = "") -> None:
    """Check that M is a 2D tensor."""
    if M.ndim != 2:  # noqa: PLR2004 - matrix rank is self-evident here
        raise ConfigurationError(
            *(f"Matrix {_pad(name)}has unexpected shape {M.shape}",)
        )


def check_square(M: torch.Tensor, name: str = "") -> None:
    """Check that M is a square matrix."""
    if M.ndim != 2 or M.shape[0] != M.shape[1]:  # noqa: PLR2004 - matrix rank
        raise ConfigurationError(
            *(f"Matrix {_pad(name)}should be square, found shape {M.shape}",)
        )


def check_finite(M: torch.Tensor, name: str = "") -> None:
    """Check that all elements of M are finite."""
    if not torch.all(torch.isfinite(M)):
        raise ConfigurationError(*(f"Matrix {_pad(name)}is not finite, found\n{M}",))


def check_symmetric(M: torch.Tensor, name: str = "", **allclose_kwargs) -> None:
    """Check that M is symmetric."""
    if not torch.allclose(M, M.T, **allclose_kwargs):
        raise ConfigurationError(
            *(f"Matrix {_pad(name)}should be symmetric, found\n{M}",)
        )


def check_exactly_one(**kwargs) -> str:
    """Check that exactly one keyword argument is not None.

    Returns the name of the provided argument. Raises ``ValueError`` if
    zero or more than one argument is not None.

    Example:
        >>> which = check_exactly_one(strategy_coef=coef, noising_coef=None)
        >>> assert which == "strategy_coef"
    """
    provided = [k for k, v in kwargs.items() if v is not None]
    if len(provided) != 1:
        names = ", ".join(kwargs.keys())
        raise ConfigurationError(
            *(
                f"Specify exactly one of: {names}. "
                f"Got: {', '.join(provided) if provided else 'none'}",
            )
        )
    return provided[0]


def check(
    *,
    A: torch.Tensor | None = None,
    B: torch.Tensor | None = None,
    C: torch.Tensor | None = None,
    X: torch.Tensor | None = None,
    **allclose_kwargs,
) -> None:
    """Apply checks to matrices A = B @ C and X = C.T @ C.

    Any subset of the matrices can be provided.

    Args:
        A: The workload matrix.
        B: The decoder matrix, such that A = B @ C.
        C: The encoder (strategy) matrix.
        X: Symmetric Gram matrix C.T @ C.
        **allclose_kwargs: kwargs to pass to torch.allclose.

    Raises:
        ValueError: if matrices do not satisfy expected properties.
    """
    not_none: dict[str, torch.Tensor] = {}
    n: int | None = None

    if A is not None:
        check_finite(A, "A")
        check_square(A, "A")
        check_lower_triangular(A, "A", **allclose_kwargs)
        not_none["A"] = A
        n = A.shape[0]

    if B is not None:
        check_finite(B, "B")
        check_is_matrix(B, "B")
        if B.shape[0] == B.shape[1]:
            check_lower_triangular(B, "B", **allclose_kwargs)
        not_none["B"] = B
        n = B.shape[0]

    if C is not None:
        check_finite(C, "C")
        check_is_matrix(C, "C")
        if C.shape[0] == C.shape[1]:
            check_lower_triangular(C, "C", **allclose_kwargs)
        not_none["C"] = C
        n = C.shape[1]

    if X is not None:
        check_finite(X, "X")
        check_square(X, "X")
        check_symmetric(X, "X", **allclose_kwargs)
        not_none["X"] = X
        n = X.shape[0]

    if B is not None and C is not None and B.shape[1] != C.shape[0]:
        raise ConfigurationError(
            *(
                "B and C shapes do not match. Expected "
                f"B.shape[1] == C.shape[0], but found "
                f"B.shape={B.shape} and C.shape={C.shape}",
            )
        )

    expected_shapes = {
        "A": ("n", "n"),
        "B": ("n", "k"),
        "C": ("k", "n"),
        "X": ("n", "n"),
    }
    correct_shapes = True
    for name_key, m in not_none.items():
        for i, letter in enumerate(expected_shapes[name_key]):
            if (letter == "n") and n is not None and (m.shape[i] != n):
                correct_shapes = False

    if not correct_shapes:
        shapes = {k: v.shape for k, v in not_none.items()}
        raise ConfigurationError(
            *(
                f"Expected matrix shapes to match {expected_shapes}, "
                f"but found shapes:\n{shapes}",
            )
        )
