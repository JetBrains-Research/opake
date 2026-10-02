#!/usr/bin/env python3
"""Require built wheels for exactly the distributions declared by the workspace."""

from __future__ import annotations

import argparse
import sys
import tomllib
import zipfile
from email import message_from_bytes
from pathlib import Path

from packaging.utils import canonicalize_name

REPO_ROOT = Path(__file__).resolve().parents[2]


def _project_name(pyproject: Path) -> str:
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    try:
        name = data["project"]["name"]
    except KeyError as error:
        raise ValueError(f"{pyproject} is missing [project].name") from error
    if not isinstance(name, str) or not name:
        raise ValueError(f"{pyproject} has an invalid [project].name")
    return canonicalize_name(name)


def expected_distributions(root_pyproject: Path) -> set[str]:
    """Return canonical names for the root project and all workspace members."""
    root_pyproject = root_pyproject.resolve()
    root = root_pyproject.parent
    data = tomllib.loads(root_pyproject.read_text(encoding="utf-8"))
    workspace = data.get("tool", {}).get("uv", {}).get("workspace", {})
    member_patterns = workspace.get("members", [])
    exclude_patterns = workspace.get("exclude", [])
    if not isinstance(member_patterns, list) or not all(
        isinstance(pattern, str) for pattern in member_patterns
    ):
        raise ValueError("tool.uv.workspace.members must be a list of paths")
    if not isinstance(exclude_patterns, list) or not all(
        isinstance(pattern, str) for pattern in exclude_patterns
    ):
        raise ValueError("tool.uv.workspace.exclude must be a list of paths")

    member_dirs: set[Path] = set()
    for pattern in member_patterns:
        matches = [path for path in root.glob(pattern) if path.is_dir()]
        if not matches:
            raise ValueError(
                f"workspace member pattern matched no directories: {pattern}"
            )
        member_dirs.update(path.resolve() for path in matches)

    excluded_dirs = {
        path.resolve()
        for pattern in exclude_patterns
        for path in root.glob(pattern)
        if path.is_dir()
    }

    projects = [root_pyproject]
    for member_dir in sorted(member_dirs - excluded_dirs):
        pyproject = member_dir / "pyproject.toml"
        if not pyproject.is_file():
            raise ValueError(f"workspace member has no pyproject.toml: {member_dir}")
        projects.append(pyproject)

    by_name: dict[str, Path] = {}
    for pyproject in projects:
        name = _project_name(pyproject)
        previous = by_name.get(name)
        if previous is not None:
            raise ValueError(
                f"duplicate distribution name {name!r}: {previous} and {pyproject}"
            )
        by_name[name] = pyproject
    return set(by_name)


def _wheel_name(wheel_path: Path) -> str:
    with zipfile.ZipFile(wheel_path) as wheel:
        metadata_paths = [
            path for path in wheel.namelist() if path.endswith(".dist-info/METADATA")
        ]
        if len(metadata_paths) != 1:
            raise ValueError(
                f"{wheel_path} contains {len(metadata_paths)} METADATA files; expected one"
            )
        metadata = message_from_bytes(wheel.read(metadata_paths[0]))
    name = metadata.get("Name")
    if not name:
        raise ValueError(f"{wheel_path} is missing Name metadata")
    return canonicalize_name(name)


def built_distributions(wheel_dir: Path) -> set[str]:
    """Return canonical distribution names found in the wheel directory."""
    wheel_paths = sorted(wheel_dir.rglob("*.whl"))
    if not wheel_paths:
        raise ValueError(f"no wheel files found in {wheel_dir}")
    return {_wheel_name(path) for path in wheel_paths}


def inventory_errors(expected: set[str], actual: set[str]) -> list[str]:
    """Describe an inexact built-wheel inventory."""
    errors: list[str] = []
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        errors.append(f"missing distributions: {', '.join(missing)}")
    if unexpected:
        errors.append(f"unexpected distributions: {', '.join(unexpected)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wheel-dir",
        type=Path,
        required=True,
        help="Directory containing the complete built-wheel family.",
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=REPO_ROOT / "pyproject.toml",
        help="Root workspace pyproject.toml.",
    )
    args = parser.parse_args()

    try:
        expected = expected_distributions(args.pyproject)
        actual = built_distributions(args.wheel_dir)
    except (OSError, ValueError, tomllib.TOMLDecodeError, zipfile.BadZipFile) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    errors = inventory_errors(expected, actual)
    if errors:
        print(*errors, sep="\n", file=sys.stderr)
        return 1

    print(f"Validated complete wheel inventory for {len(actual)} distributions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
