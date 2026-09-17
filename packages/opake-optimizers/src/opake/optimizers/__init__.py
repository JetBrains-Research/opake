"""Functional optimizers — torchopt-based factories with a DP-aware update surface.

Headline factories:

- :func:`adamw` — universal Adam / AdamW with DP bias correction.
- :func:`adam` — original Adam / L2 variant.
- :func:`sgd` — vanilla SGD; unbiased under additive DP noise.
- :func:`radam`, :func:`lion`, :func:`ademamix`, :func:`adafactor`,
  :func:`adagrad`, :func:`adadelta`, :func:`rmsprop`,
  :func:`schedule_free` — additional families, some via ``torchopt``
  re-export.

State dataclasses live in :mod:`opake.optimizers.types`.
"""

from opake.api.optimizers import (
    adadelta,
    adafactor,
    adagrad,
    adam,
    adamw,
    ademamix,
    lion,
    radam,
    rmsprop,
    schedule_free,
    sgd,
)
from opake.optimizers import types

__all__ = [
    "adadelta",
    "adafactor",
    "adagrad",
    "adam",
    "adamw",
    "ademamix",
    "lion",
    "radam",
    "rmsprop",
    "schedule_free",
    "sgd",
    "types",
]
