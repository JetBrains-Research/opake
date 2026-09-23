"""Import setup for non-published example packages."""

import sys
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).parents[2] / "examples"
sys.path.insert(0, str(EXAMPLES_ROOT))
