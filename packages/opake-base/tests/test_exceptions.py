"""Tests for the public semantic exception taxonomy."""

from __future__ import annotations

import pytest

from opake.exceptions import (
    CalibrationError,
    CheckpointError,
    ConfigurationError,
    InputTypeError,
    OpakeError,
    OperationError,
    PrivacyBudgetError,
)


def test_exception_categories_have_familiar_python_bases() -> None:
    """Semantic categories remain catchable through their standard base types."""
    assert issubclass(ConfigurationError, (OpakeError, ValueError))
    assert issubclass(CalibrationError, ConfigurationError)
    assert issubclass(PrivacyBudgetError, ConfigurationError)
    assert issubclass(InputTypeError, (OpakeError, TypeError))
    assert issubclass(OperationError, (OpakeError, RuntimeError))
    assert issubclass(CheckpointError, OperationError)


@pytest.mark.parametrize(
    "error_type",
    [
        ConfigurationError,
        CalibrationError,
        PrivacyBudgetError,
        InputTypeError,
        OperationError,
        CheckpointError,
    ],
)
def test_exception_category_raises_with_diagnostic_message(
    error_type: type[OpakeError],
) -> None:
    """Category construction leaves the detailed failure diagnostics intact."""
    with pytest.raises(error_type, match="detailed diagnostic"):
        error_type.raise_("detailed diagnostic")
