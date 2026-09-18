# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Numerically safe accumulation dtypes for alignment reductions."""

from __future__ import annotations

import torch


def _compute_dtype(tensor: torch.Tensor) -> torch.dtype:
    """Accumulate in at least fp32 without downcasting fp64 inputs."""
    return torch.promote_types(tensor.dtype, torch.float32)
