"""Round-trip tests for :mod:`opake.serialization` (PerGroup, NumPy leaves)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from opake.serialization import from_state_dict, state_dict
from opake.types import PerGroup


@dataclass(frozen=True)
class _NdBox:
    """Minimal dataclass with an ndarray leaf (auditing-style layout)."""

    a: np.ndarray


def test_pergroup_roundtrip() -> None:
    pg = PerGroup(
        groups={"p.0": "g1", "p.1": "g2"},
        values={"g1": 1.5, "g2": 2.5},
    )
    sd = state_dict(pg)
    template = PerGroup(
        groups={"p.0": "g1", "p.1": "g2"},
        values={"g1": 0.0, "g2": 0.0},
    )
    restored = from_state_dict(template, sd)
    assert restored == pg


def test_numpy_leaf_roundtrip() -> None:
    template = _NdBox(np.zeros((2, 3), dtype=np.float32))
    obj = _NdBox(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))
    sd = state_dict(obj)
    out = from_state_dict(template, sd)
    assert isinstance(out, _NdBox)
    assert out.a.shape == (2, 3)
    assert out.a.dtype == np.float32
    np.testing.assert_array_equal(out.a, obj.a)


def test_root_numpy_roundtrip() -> None:
    arr = np.arange(6, dtype=np.int64).reshape(2, 3)
    sd = state_dict(arr)
    out = from_state_dict(np.zeros((2, 3), dtype=np.int64), sd)
    np.testing.assert_array_equal(out, arr)
