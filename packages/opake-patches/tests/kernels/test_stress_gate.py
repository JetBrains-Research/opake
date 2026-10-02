"""Tests for the kernel stress memory gate."""

from types import SimpleNamespace

import pytest

from . import conftest as kernel_conftest


class _Item:
    def __init__(self, *, stress: bool) -> None:
        self.stress = stress
        self.added_markers = []

    def get_closest_marker(self, name: str):
        if name == "kernel_stress" and self.stress:
            return object()
        return None

    def add_marker(self, marker) -> None:
        self.added_markers.append(marker)


@pytest.mark.parametrize(("memory_gib", "should_skip"), [(16, True), (24, False)])
def test_memory_gate_marks_only_stress_tests(monkeypatch, memory_gib, should_skip):
    monkeypatch.setattr(kernel_conftest.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        kernel_conftest.torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(total_memory=memory_gib * 1024**3),
    )
    ordinary = _Item(stress=False)
    stress = _Item(stress=True)

    kernel_conftest.pytest_collection_modifyitems(None, [ordinary, stress])

    assert ordinary.added_markers == []
    assert bool(stress.added_markers) is should_skip
