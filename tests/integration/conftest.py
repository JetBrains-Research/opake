"""Integration-test hooks.

``torch.multiprocessing.spawn`` sends the parent's ``sys.path`` to child
interpreters (see ``multiprocessing.spawn.get_preparation_data``). Pytest's
runtime path tweaks are not always enough for children to resolve
``tests.integration...`` when unpickling spawn targets, so we ensure the
repository root is on ``sys.path`` for any code running under
``tests/integration/``.

We also set ``PYTHONPATH`` for subprocess-based tooling that keys off the
environment.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from opake_test_support import fast_mc_accounting

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT_S = str(_REPO_ROOT)

if _REPO_ROOT_S not in sys.path:
    sys.path.insert(0, _REPO_ROOT_S)

_SEP = os.pathsep
_cur = os.environ.get("PYTHONPATH", "")
_parts = [p for p in _cur.split(_SEP) if p]
if _REPO_ROOT_S not in _parts:
    os.environ["PYTHONPATH"] = _REPO_ROOT_S if not _cur else _REPO_ROOT_S + _SEP + _cur


def pytest_configure(config) -> None:
    """Keep repo root on ``sys.path`` / ``PYTHONPATH`` if something strips them."""
    if _REPO_ROOT_S not in sys.path:
        sys.path.insert(0, _REPO_ROOT_S)
    cur = os.environ.get("PYTHONPATH", "")
    parts = [p for p in cur.split(_SEP) if p]
    if _REPO_ROOT_S not in parts:
        os.environ["PYTHONPATH"] = (
            _REPO_ROOT_S if not cur else _REPO_ROOT_S + _SEP + cur
        )


@pytest.fixture(autouse=True)
def _fast_mc_accounting():
    """Keep integration smoke tests below the production MC resolution cost."""
    with fast_mc_accounting():
        yield
