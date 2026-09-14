"""Conftest for opaque-transformers tests.

We install the global HF runtime compat shims (masking, collator, checkpoint
hooks) at module load so they match what
:class:`~opaque.transformers.trainer.Trainer` applies during ``__init__``.
Guards: missing sub-packages must not break collection.

Shared model fixtures and compatibility helpers live in
``tests/_support/opaque_test_support.py`` and
``opaque-transformers/tests/_hf_shared.py``. Session-scoped fixture wrappers
are re-exported here for convenience.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make sibling `_shared.py` importable without relying on package-relative
# imports (pytest runs under --import-mode=importlib; adding a `tests/__init__.py`
# at multiple package roots collides, so tests/* are treated as plain dirs).
sys.path.append(str(Path(__file__).parent))

from _hf_shared import MODEL_CONFIGS, STANDARD_LORA_CONFIG
from opaque_test_support import fast_mc_accounting

# Apply global runtime compat patches (same env semantics as
# Trainer.__init__) so test collection matches the trainer's runtime.
try:
    from opaque.patches import apply_runtime_patches

    apply_runtime_patches(compat=True)
except ImportError:
    pass


@pytest.fixture(scope="session")
def model_configs():
    """Provide model configurations to tests."""
    return MODEL_CONFIGS


@pytest.fixture(scope="session")
def standard_lora_config():
    """Provide standard LoRA configuration."""
    return STANDARD_LORA_CONFIG


@pytest.fixture(autouse=True)
def _fast_mc_accounting():
    """Keep trainer smoke tests below the production MC resolution cost."""
    with fast_mc_accounting():
        yield
