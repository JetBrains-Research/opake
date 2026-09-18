"""Make the DDP helper module importable from spawned worker processes."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_DDP = str(Path(__file__).resolve().parent / "ddp")
if _DDP not in sys.path:
    sys.path.insert(0, _DDP)
_prev_pp = os.environ.get("PYTHONPATH")
os.environ["PYTHONPATH"] = _DDP + (os.pathsep + _prev_pp if _prev_pp else "")
