"""Restore policy for array leaves: shape is checked, dtype and device follow.

Restore is template-driven, so a checkpoint value that does not fit its slot
must fail at load rather than broadcast into a silently different run. Tensor
and ndarray leaves apply that rule identically; dtype and device follow the
template so a CPU checkpoint resumes on the training device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

from opake.exceptions import CheckpointError
from opake.serialization import from_state_dict, state_dict


@dataclass
class _Buffers:
    """Inner state holding a tuple of array leaves."""

    values: tuple[Any, ...]


@dataclass
class _State:
    """Outer state: dataclass → mapping → dataclass → tuple → leaf."""

    inner: dict[str, _Buffers]
    step: int


def _nested(*leaves: Any) -> _State:
    return _State(inner={"noise": _Buffers(values=leaves)}, step=3)


@pytest.mark.parametrize("zeros", [torch.zeros, np.zeros], ids=["tensor", "ndarray"])
def test_nested_shape_mismatch_rejected(zeros: Any) -> None:
    """Both leaf kinds reject a mismatch with ``CheckpointError``, naming the key."""
    saved = state_dict(_nested(zeros(2), zeros(3)))

    with pytest.raises(
        CheckpointError, match=r"shape \(3,\); template expects \(4,\)"
    ) as excinfo:
        from_state_dict(_nested(zeros(2), zeros(4)), saved)

    assert "inner.noise.values[1]" in " ".join(excinfo.value.__notes__)


def test_parameter_shape_mismatch_rejected() -> None:
    """``nn.Parameter`` restores through the tensor rule, not around it."""
    saved = state_dict({"p": nn.Parameter(torch.ones(3))})

    with pytest.raises(CheckpointError, match="shape"):
        from_state_dict({"p": nn.Parameter(torch.zeros(4))}, saved)


def test_missing_leaf_keeps_template() -> None:
    """An absent key is forward compatibility, not a mismatch."""
    template = _nested(torch.zeros(2), torch.zeros(4))

    restored = from_state_dict(template, {})

    assert restored.inner["noise"].values[1] is template.inner["noise"].values[1]


def test_dtype_follows_the_template() -> None:
    """The template carries the live compute dtype; the saved value is cast."""
    saved = state_dict({"buf": torch.ones(3, dtype=torch.float64)})

    restored = from_state_dict({"buf": torch.zeros(3, dtype=torch.float32)}, saved)

    assert restored["buf"].dtype is torch.float32
    assert torch.equal(restored["buf"], torch.ones(3))


def test_device_follows_the_template(all_devices: torch.device) -> None:
    """A checkpoint read onto the CPU lands on the template's device."""
    saved = state_dict({"buf": torch.ones(3)})

    restored = from_state_dict({"buf": torch.zeros(3, device=all_devices)}, saved)

    assert restored["buf"].device.type == all_devices.type
