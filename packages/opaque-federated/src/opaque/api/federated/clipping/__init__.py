"""Federated per-client clipping."""

from opaque.api.federated.clipping._clipped_grad import clipped_grad
from opaque.api.federated.clipping._strategy import clipped_sum

__all__ = ["clipped_grad", "clipped_sum"]
