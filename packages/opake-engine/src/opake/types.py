"""Cross-package DP-flow types.

Single canonical home for the data types that connect clipping →
noise → optimizer:

- **Pytree wrappers**: ``ClippedPytree`` (post-clipping),
  ``NoisedPytree`` (post-noise), and the paired-stream outputs
  ``SecondMomentClippingOutput`` / ``SecondMomentNoiseOutput``.
- **Per-group container**: ``PerGroup`` — a dict-like that flows
  through the entire pipeline carrying per-parameter-group scalar
  values.
- **Abstract state bases**: ``ClipState`` and ``NoiseState`` —
  markers shared across DP-SGD and DP-FTRL implementations.
- **Aliases**: ``MaxNorm``, ``NoiseStddev`` — opaque-typed unions
  used in wrapper metadata fields.
- **Factories**: ``clipped()`` and ``noised()`` — manual wrapper
  constructors for callers that already produced privatised values.
"""

from opake.api.engine.types import (
    ClippedPytree,
    ClipState,
    MaxNorm,
    NoisedPytree,
    NoiseState,
    NoiseStddev,
    ParamPath,
    PerGroup,
    SecondMomentClippingOutput,
    SecondMomentNoiseOutput,
    TensorPytree,
    clipped,
    noised,
)

__all__ = [
    "ClipState",
    "ClippedPytree",
    "MaxNorm",
    "NoiseState",
    "NoiseStddev",
    "NoisedPytree",
    "ParamPath",
    "PerGroup",
    "SecondMomentClippingOutput",
    "SecondMomentNoiseOutput",
    "TensorPytree",
    "clipped",
    "noised",
]
