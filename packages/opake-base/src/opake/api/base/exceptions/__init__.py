"""Shared exception types for Opake public APIs."""

from __future__ import annotations

from typing import NoReturn


class OpakeError(Exception):
    """Base class for failures reported by Opake public APIs."""

    @classmethod
    def raise_(
        cls,
        message: str,
        *,
        cause: BaseException | None = None,
        suppress_context: bool = False,
    ) -> NoReturn:
        """Raise this category with a diagnostic message."""
        if suppress_context:
            raise cls(message) from None
        if cause is not None:
            raise cls(message) from cause
        raise cls(message)


class ConfigurationError(OpakeError, ValueError):
    """Raised when an Opake API receives invalid or incompatible configuration."""


class InputTypeError(OpakeError, TypeError):
    """Raised when an Opake API receives an argument of an invalid type."""


class OperationError(OpakeError, RuntimeError):
    """Raised when an Opake operation cannot complete its requested work."""


class CalibrationError(ConfigurationError):
    """Raised when privacy calibration cannot validate or satisfy its search."""


class PrivacyBudgetError(ConfigurationError):
    """Raised when a privacy-budget definition or operation is invalid."""


class CheckpointError(OperationError):
    """Raised when an Opake checkpoint cannot be saved, restored, or resumed."""


__all__ = [
    "CalibrationError",
    "CheckpointError",
    "ConfigurationError",
    "InputTypeError",
    "OpakeError",
    "OperationError",
    "PrivacyBudgetError",
]
