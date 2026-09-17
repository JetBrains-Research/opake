"""Tests for shared PerGroup bias-correction helpers."""

from __future__ import annotations

import pytest
import torch

from opake.api.optimizers._bias_correction import map_leaves_with_path


def test_map_leaves_with_path_aligned_trees():
    a = {"w": torch.tensor([1.0]), "b": torch.tensor([2.0])}
    b = {"w": torch.tensor([10.0]), "b": torch.tensor([20.0])}

    def add(path, x, y):
        return x + y

    out = map_leaves_with_path(add, a, b)
    torch.testing.assert_close(out["w"], torch.tensor([11.0]))
    torch.testing.assert_close(out["b"], torch.tensor([22.0]))


def test_map_leaves_with_path_rejects_mismatched_paths():
    """Same leaf count but different ParamPath sequences must raise."""
    primary = {"w": torch.tensor([1.0]), "b": torch.tensor([2.0])}
    # list of length 2 → paths (0,), (1,) — not ("w",), ("b",)
    other = [torch.tensor([10.0]), torch.tensor([20.0])]

    with pytest.raises(ValueError, match="ParamPath mismatch"):
        map_leaves_with_path(lambda path, x, y: x + y, primary, other)
