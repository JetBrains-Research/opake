# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the qwen3_5_moe family (MoE).

Shares the qwen3_moe / qwen3_next patch path. Its tiny config is too fragile to
construct in transformers 5.11 (many undefaulted hybrid fields), so the forward
suite is skipped; the patch wiring is exercised by the sibling MoE families and
the family-coverage guard.
"""

import pytest

pytest.importorskip("transformers")


@pytest.mark.skip(
    reason="qwen3_5_moe tiny config not constructible in tf5.11; same patch path as qwen3_moe"
)
def test_qwen3_5_moe_forward():
    pass
