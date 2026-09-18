#!/usr/bin/env python3
"""Validate opake-accounting artifact packaging policy from built artifacts."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import struct
import sys
import tarfile
import tomllib
import zipfile
from fnmatch import fnmatch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

TRANSIENT_BYTECODE_PATTERNS = ("**/__pycache__/**", "**/*.pyc", "**/*.pyo")
NATIVE_LIBRARY_SUFFIXES = (".so", ".dylib", ".dll", ".pyd")
NATIVE_MODULE_PATH = "opake/api/accounting/core/opake_accounting.abi3.so"
MACOS_DEPLOYMENT_TARGET = 11.0
RECORD_COLUMN_COUNT = 3
MACHO_HEADER_SIZE = 8
ELF_HEADER_SIZE = 20
MACHO_64_MAGIC = 0xFEEDFACF
MACHO_CPU_TYPE_ARM64 = 0x0100000C


def _is_transient(path: str) -> bool:
    """Check if a path matches any of the transient bytecode patterns."""
    return any(fnmatch(path, pattern) for pattern in TRANSIENT_BYTECODE_PATTERNS)


def _check_pyproject_config() -> list[str]:
    """Verify pyproject.toml has correct artifact policy settings."""
    pyproject = REPO_ROOT / "packages" / "opake-accounting" / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    errors: list[str] = []
    maturin = data["tool"]["maturin"]

    # Verify exclusion patterns are configured and cover transient bytecode
    exclude_patterns = list(maturin.get("exclude", []))
    test_paths = [
        "src/pkg/__pycache__/module.cpython-312.pyc",
        "src/pkg/temp.pyc",
        "src/pkg/temp.pyo",
    ]
    errors.extend(
        f"exclude patterns don't cover: {test_path}"
        for test_path in test_paths
        if not any(fnmatch(test_path, pattern) for pattern in exclude_patterns)
    )

    # Verify macOS deployment target
    target = maturin.get("target", {}).get("aarch64-apple-darwin", {})
    deployment_target = float(target.get("macos-deployment-target", 0))
    if deployment_target < MACOS_DEPLOYMENT_TARGET:
        errors.append(f"macOS deployment target {deployment_target} is below 11.0")

    return errors


def _check_cargo_config() -> list[str]:
    """Verify Cargo.toml package.exclude has transient bytecode patterns."""
    cargo = REPO_ROOT / "packages" / "opake-accounting" / "Cargo.toml"
    data = tomllib.loads(cargo.read_text(encoding="utf-8"))

    errors: list[str] = []
    exclude_patterns = data.get("package", {}).get("exclude", [])

    has_pycache = any("__pycache__" in str(pattern) for pattern in exclude_patterns)
    has_pyc = any("*.pyc" in str(pattern) for pattern in exclude_patterns)
    if not (has_pycache and has_pyc):
        errors.append(
            "Cargo.toml package.exclude missing transient bytecode patterns (__pycache__ or *.pyc)"
        )

    return errors


def _validate_wheel_artifact(wheel_path: Path) -> list[str]:
    errors: list[str] = []
    try:
        with zipfile.ZipFile(wheel_path) as wheel:
            bad_entry = wheel.testzip()
            if bad_entry is not None:
                errors.append(
                    f"wheel {wheel_path.name} has a corrupt entry: {bad_entry}"
                )
                return errors

            entries = wheel.namelist()
            errors.extend(
                f"wheel {wheel_path.name} contains transient file: {entry}"
                for entry in entries
                if _is_transient(entry)
            )

            errors.extend(_validate_record(wheel_path, wheel, entries))
            errors.extend(_validate_native_module(wheel_path, wheel, entries))
    except (OSError, zipfile.BadZipFile) as error:
        errors.append(f"wheel {wheel_path.name} is not a valid ZIP archive: {error}")

    return errors


def _validate_record(
    wheel_path: Path, wheel: zipfile.ZipFile, entries: list[str]
) -> list[str]:
    """Validate that RECORD hashes and sizes describe the wheel's contents."""
    record_paths = [entry for entry in entries if entry.endswith(".dist-info/RECORD")]
    if len(record_paths) != 1:
        return [
            f"wheel {wheel_path.name} must contain exactly one .dist-info/RECORD, "
            f"found {len(record_paths)}"
        ]

    record_path = record_paths[0]
    try:
        rows = list(
            csv.reader(
                io.TextIOWrapper(wheel.open(record_path), encoding="utf-8", newline="")
            )
        )
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        return [f"wheel {wheel_path.name} has an unreadable RECORD: {error}"]

    errors: list[str] = []
    recorded_paths: set[str] = set()
    for row in rows:
        if len(row) != RECORD_COLUMN_COUNT:
            errors.append(
                f"wheel {wheel_path.name} has a malformed RECORD row: {row!r}"
            )
            continue

        path, encoded_hash, size = row
        if path in recorded_paths:
            errors.append(f"wheel {wheel_path.name} records {path} more than once")
            continue
        recorded_paths.add(path)

        if path not in entries:
            errors.append(
                f"wheel {wheel_path.name} RECORD references missing file: {path}"
            )
            continue
        if path == record_path:
            if encoded_hash or size:
                errors.append(
                    f"wheel {wheel_path.name} RECORD entry must not hash itself"
                )
            continue
        if not encoded_hash.startswith("sha256="):
            errors.append(
                f"wheel {wheel_path.name} RECORD lacks a SHA-256 hash for {path}"
            )
            continue

        contents = wheel.read(path)
        expected_hash = encoded_hash.removeprefix("sha256=")
        actual_hash = (
            base64.urlsafe_b64encode(hashlib.sha256(contents).digest())
            .rstrip(b"=")
            .decode()
        )
        if actual_hash != expected_hash:
            errors.append(f"wheel {wheel_path.name} RECORD hash mismatch for {path}")
        if size != str(len(contents)):
            errors.append(f"wheel {wheel_path.name} RECORD size mismatch for {path}")

    missing = set(entries) - recorded_paths
    if missing:
        errors.append(
            f"wheel {wheel_path.name} RECORD omits files: {', '.join(sorted(missing))}"
        )
    return errors


def _validate_native_module(
    wheel_path: Path, wheel: zipfile.ZipFile, entries: list[str]
) -> list[str]:
    """Check that the wheel holds one native module for its declared platform."""
    native_entries = [
        entry for entry in entries if entry.endswith(NATIVE_LIBRARY_SUFFIXES)
    ]
    if native_entries != [NATIVE_MODULE_PATH]:
        return [
            f"wheel {wheel_path.name} must contain only {NATIVE_MODULE_PATH} as a "
            f"native library, found {native_entries}"
        ]

    binary = wheel.read(NATIVE_MODULE_PATH)
    if "macosx_11_0_arm64" in wheel_path.name:
        if len(binary) < MACHO_HEADER_SIZE:
            return [f"wheel {wheel_path.name} native module is truncated"]
        magic, cpu_type = struct.unpack_from("<II", binary)
        if magic != MACHO_64_MAGIC or cpu_type != MACHO_CPU_TYPE_ARM64:
            return [
                f"wheel {wheel_path.name} native module is not a 64-bit arm64 Mach-O"
            ]
    elif "x86_64" in wheel_path.name and (
        "manylinux_2_28" in wheel_path.name or "linux_x86_64" in wheel_path.name
    ):
        return _validate_elf_architecture(
            wheel_path, binary, expected_machine=62, expected_architecture="x86_64"
        )
    elif "aarch64" in wheel_path.name and (
        "manylinux_2_28" in wheel_path.name or "linux_aarch64" in wheel_path.name
    ):
        return _validate_elf_architecture(
            wheel_path, binary, expected_machine=183, expected_architecture="aarch64"
        )
    else:
        return [f"wheel {wheel_path.name} has an unsupported platform tag"]

    return []


def _validate_elf_architecture(
    wheel_path: Path,
    binary: bytes,
    *,
    expected_machine: int,
    expected_architecture: str,
) -> list[str]:
    """Validate an ELF library's architecture from its fixed header."""
    if len(binary) < ELF_HEADER_SIZE:
        return [f"wheel {wheel_path.name} native module is truncated"]
    if binary[:4] != b"\x7fELF":
        return [f"wheel {wheel_path.name} native module is not an ELF binary"]
    byte_order = {1: "<", 2: ">"}.get(binary[5])
    if byte_order is None:
        return [f"wheel {wheel_path.name} native module has an invalid ELF byte order"]
    machine = struct.unpack_from(f"{byte_order}H", binary, 18)[0]
    if machine != expected_machine:
        return [
            f"wheel {wheel_path.name} native module is not {expected_architecture} ELF"
        ]
    return []


def _validate_sdist_artifact(sdist_path: Path) -> list[str]:
    errors: list[str] = []
    try:
        with tarfile.open(sdist_path, "r:gz") as sdist:
            entries = [member.name for member in sdist.getmembers() if member.isfile()]
    except (OSError, tarfile.TarError) as error:
        return [f"sdist {sdist_path.name} is not a valid gzip tar archive: {error}"]

    for entry in entries:
        if _is_transient(entry):
            errors.append(f"sdist {sdist_path.name} contains transient file: {entry}")
        if entry.endswith(NATIVE_LIBRARY_SUFFIXES):
            errors.append(f"sdist {sdist_path.name} contains native library: {entry}")

    return errors


def main() -> int:
    """Validate configured and built opake-accounting artifact policy."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wheel-dir",
        type=Path,
        help="Directory containing built wheel files.",
    )
    args = parser.parse_args()

    errors: list[str] = []

    # Check configuration files
    errors.extend(_check_pyproject_config())
    errors.extend(_check_cargo_config())

    # Check built artifacts if provided
    if args.wheel_dir:
        for wheel_path in sorted(args.wheel_dir.glob("*.whl")):
            errors.extend(_validate_wheel_artifact(wheel_path))

        for sdist_path in sorted(args.wheel_dir.glob("*.tar.gz")):
            errors.extend(_validate_sdist_artifact(sdist_path))

    if errors:
        print("Accounting artifact policy violations:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("Accounting artifact policy validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
