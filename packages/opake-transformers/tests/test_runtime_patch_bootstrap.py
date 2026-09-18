# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Explicit runtime patch bootstrap."""

from __future__ import annotations

import subprocess
import sys

import pytest


def _run(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.mark.slow
def test_subprocess_apply_runtime_patches_sets_state():
    out = _run(
        "from opake.patches import apply_runtime_patches, is_runtime_patched\n"
        "apply_runtime_patches(compat=True)\n"
        "print(is_runtime_patched())"
    )
    assert out == "True"
