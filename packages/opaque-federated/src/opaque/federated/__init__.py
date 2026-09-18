"""Federated DP primitives on IFED: cohort samplers, round loaders, the
per-client clipping strategy, and the clipped-gradient loop driver."""

from opaque.api.federated import (
    Cohort,
    DataLoader,
    MinSepSampler,
    Population,
    clipped_grad,
    clipped_sum,
    population,
)

__all__ = [
    "Cohort",
    "DataLoader",
    "MinSepSampler",
    "Population",
    "clipped_grad",
    "clipped_sum",
    "population",
]
