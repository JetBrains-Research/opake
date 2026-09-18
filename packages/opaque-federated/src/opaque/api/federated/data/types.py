"""Opaque-owned symbolic population and cohort values."""

from __future__ import annotations

from dataclasses import dataclass, field

from opaque.exceptions import ConfigurationError, InputTypeError


@dataclass(frozen=True)
class Population:
    """A named, versioned IFED population specification."""

    name: str
    version: str = "*"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.startswith("/"):
            raise ConfigurationError(
                *(
                    "population name must be a string starting with '/', got "
                    f"{self.name!r}",
                )
            )
        if not isinstance(self.version, str):
            raise InputTypeError(
                *(
                    "population version must be a string, got "
                    f"{type(self.version).__name__}",
                )
            )
        if not self.version:
            raise ConfigurationError(*("population version must not be empty",))


@dataclass(frozen=True)
class Cohort:
    """One symbolic Opaque batch, resolved by executing its federated round."""

    round: int
    size: int
    separation: int = 0
    rounds: int | None = None
    population: Population | None = None
    origin: object | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.round < 0:
            raise ConfigurationError(*(f"round must be >= 0, got {self.round}",))
        if self.size < 1:
            raise ConfigurationError(*(f"size must be >= 1, got {self.size}",))
        if self.separation < 0:
            raise ConfigurationError(
                *(f"separation must be >= 0, got {self.separation}",)
            )
        if self.rounds is not None and self.rounds < 1:
            raise ConfigurationError(*(f"rounds must be >= 1, got {self.rounds}",))


__all__ = ["Cohort", "Population"]
