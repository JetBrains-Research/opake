"""Shared pytest configuration and fixtures for all Opake packages.

Pytest discovers conftest.py by walking up from each test file, so every
package in the workspace inherits these fixtures automatically.
"""

import importlib
import os
import sys
from pathlib import Path

import pytest
import torch

_TEST_SUPPORT = str(Path(__file__).resolve().parent / "tests" / "_support")
if _TEST_SUPPORT not in sys.path:
    sys.path.insert(0, _TEST_SUPPORT)
_previous_pythonpath = os.environ.get("PYTHONPATH")
os.environ["PYTHONPATH"] = _TEST_SUPPORT + (
    os.pathsep + _previous_pythonpath if _previous_pythonpath else ""
)

_test_support = importlib.import_module("opake_test_support")
get_default_device = _test_support.get_default_device
get_default_gpu_device = _test_support.get_default_gpu_device
_set_random_seed = _test_support.set_random_seed

__all__ = ["get_default_device", "get_default_gpu_device"]


# ---------------------------------------------------------------------------
# Auto-skip logic for marker-based gating
# ---------------------------------------------------------------------------


def pytest_runtest_setup(item):
    """Auto-skip tests based on the three orthogonal markers (cuda/mps/slow)."""
    if "cuda" in item.keywords and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    if "mps" in item.keywords and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def device():
    """Provide device for tests (CUDA > MPS > CPU in priority order)."""
    return get_default_device()


@pytest.fixture(params=["cpu", "cuda", "mps"])
def all_devices(request):
    """Parametrize tests over all available devices."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if request.param == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    return torch.device(request.param)


@pytest.fixture
def set_random_seed():
    """Fixture to set random seed for reproducibility."""
    return _set_random_seed


@pytest.fixture
def simple_pytree():
    """Provide a simple PyTree for testing."""
    return {
        "weight": torch.tensor([3.0, 4.0]),
        "bias": torch.tensor([0.0, 12.0]),
    }


@pytest.fixture
def nested_pytree():
    """Provide a nested PyTree for testing."""
    return {
        "layer1": {
            "weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "bias": torch.tensor([0.5, -0.5]),
        },
        "layer2": {"weight": torch.tensor([1.0]), "bias": torch.tensor([0.0])},
    }
