"""Check that published theoretical epsilon values match an executed notebook."""

# ruff: noqa: INP001

import json
import re
import sys
from pathlib import Path

EPSILON = re.compile(
    r"(?:epsilon|ε)(?:\s+at\s+delta\s*=\s*[^:]+)?\s*[:=]\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def epsilon_values(path: Path) -> list[str]:
    """Extract displayed theoretical ε values from code-cell outputs."""
    notebook = json.loads(path.read_text())
    values = []
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            for line in "".join(output.get("text", [])).splitlines():
                if line.lstrip().startswith(("Audited ε", "delta at epsilon=")):
                    continue
                if "=>" in line:
                    continue
                values.extend(EPSILON.findall(line))
    return values


def main(saved: Path, executed: Path) -> None:
    """Reject any drift in displayed theoretical ε values."""
    expected = epsilon_values(saved)
    actual = epsilon_values(executed)
    if not expected or expected != actual:
        print(
            f"{saved.name}: committed ε {expected} differs from executed ε {actual}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(f"{saved.name}: {len(actual)} published ε values match")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
