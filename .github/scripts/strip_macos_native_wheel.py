#!/usr/bin/env python3
"""Strip local symbols from an opake-accounting macOS wheel and repack it."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

NATIVE_MODULE_PATH = "opake/api/accounting/core/opake_accounting.abi3.so"


def main() -> int:
    """Strip the native extension and regenerate the wheel's RECORD metadata."""
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path, help="macOS opake-accounting wheel")
    args = parser.parse_args()

    wheel = args.wheel.resolve()
    if not wheel.is_file():
        parser.error(f"wheel not found: {wheel}")

    with tempfile.TemporaryDirectory() as temporary_directory:
        unpacked = Path(temporary_directory) / "wheel"
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(unpacked)

        native_module = unpacked / NATIVE_MODULE_PATH
        if not native_module.is_file():
            parser.error(f"native module not found: {NATIVE_MODULE_PATH}")

        subprocess.run(["xcrun", "strip", "-x", native_module], check=True)
        wheel.unlink()
        subprocess.run(
            [
                sys.executable,
                "-m",
                "wheel",
                "pack",
                unpacked,
                "--dest-dir",
                wheel.parent,
            ],
            check=True,
        )

    rebuilt_wheel = wheel.parent / wheel.name
    if not rebuilt_wheel.is_file():
        parser.error(f"failed to repack wheel: {rebuilt_wheel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
