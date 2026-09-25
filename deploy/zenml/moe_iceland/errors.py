"""Consistent fail-closed deployment errors."""

from typing import NoReturn


def refuse(message: str) -> NoReturn:
    """Reject an unsafe or incompatible deployment with an actionable explanation."""
    raise ValueError(message)
