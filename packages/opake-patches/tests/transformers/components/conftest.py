"""Conftest for HuggingFace compatibility tests.

Re-exports shared Qwen2 fixtures used by component tests.

Applies ``opake.patches.apply_runtime_patches()`` once at collection time so that
the runtime patches are in effect for the whole test session. Patching
is opt-in in production; tests opt in via this conftest.
"""

import pytest

from opake_test_support import qwen2_config, qwen2_tokenizer  # noqa: F401

pytest.importorskip("transformers")
pytest.importorskip("peft")


@pytest.fixture(scope="session", autouse=True)
def _apply_opake_hf_patches():
    transformers = pytest.importorskip("transformers")  # noqa: F841
    from opake.patches import apply_runtime_patches

    apply_runtime_patches()
    return
