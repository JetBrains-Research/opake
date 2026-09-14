"""Conftest for Transformers / Trainer integration tests.

Re-exports shared Qwen2 fixtures used by compatibility tests.

Applies the opaque-patches runtime patches once at collection time so the
vmap-safety patches are in effect for the whole test session. Patching is
opt-in in production; tests opt in via this conftest.
"""

import pytest
from opaque_test_support import qwen2_config, qwen2_tokenizer  # noqa: F401


@pytest.fixture(scope="session", autouse=True)
def _apply_opaque_hf_patches():
    transformers = pytest.importorskip("transformers")  # noqa: F841
    from opaque.patches import apply_runtime_patches

    apply_runtime_patches()
    return
