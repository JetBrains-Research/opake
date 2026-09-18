"""Population factory."""

from __future__ import annotations

from opaque.api.federated.data.types import Population


def population(name: str, *, version: str = "*") -> Population:
    """Create an inert population specification for a sampler and loader."""
    return Population(name=name, version=version)


__all__ = ["population"]
