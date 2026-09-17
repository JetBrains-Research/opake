# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Guard: every registered model family has a dedicated ``test_<family>.py``.

The repo convention is one test file per family. This fails if a family is
registered without a matching test file, so a new family can't be added without
test coverage.
"""

from pathlib import Path

import pytest

pytest.importorskip("transformers")

from opake.api.patches.transformers import supported_families


def test_every_registered_family_has_a_test_file():
    here = Path(__file__).resolve().parent
    test_files = {p.name[len("test_") : -len(".py")] for p in here.glob("test_*.py")}
    missing = sorted(set(supported_families()) - test_files)
    assert not missing, f"registered families without a test_<family>.py: {missing}"
