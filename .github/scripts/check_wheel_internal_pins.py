#!/usr/bin/env python3
"""Validate synchronized opake wheel pins from already built wheel metadata."""

from __future__ import annotations

import argparse
import sys
import tomllib
import zipfile
from email import message_from_bytes
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version

REPO_ROOT = Path(__file__).resolve().parents[2]
SENTINEL_SPECIFIER = ">=0.0.0.dev0"


def _pyproject_paths() -> list[Path]:
    return [
        REPO_ROOT / "pyproject.toml",
        *sorted((REPO_ROOT / "packages").glob("*/pyproject.toml")),
    ]


def _dist_to_pyproject() -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in _pyproject_paths():
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        dist_name = data["project"]["name"]
        mapping[dist_name] = path
    return mapping


def _project_requirements(pyproject_path: Path) -> list[Requirement]:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data["project"]

    requirements: list[Requirement] = [
        Requirement(requirement)
        for requirement in project.get("dependencies", [])
        if requirement.startswith("opake-")
    ]
    for extra_name, extra_requirements in project.get(
        "optional-dependencies", {}
    ).items():
        requirements.extend(
            Requirement(f'{requirement}; extra == "{extra_name}"')
            for requirement in extra_requirements
            if requirement.startswith("opake-")
        )
    return requirements


def _requirement_key(
    requirement: Requirement,
) -> tuple[str, tuple[str, ...], str | None]:
    marker = None if requirement.marker is None else str(requirement.marker)
    return (requirement.name, tuple(sorted(requirement.extras)), marker)


def _expected_requirement_keys(
    source_requirements: list[Requirement],
    *,
    built_version: str,
) -> set[tuple[str, tuple[str, ...], str | None]]:
    expected: set[tuple[str, tuple[str, ...], str | None]] = set()
    for requirement in source_requirements:
        if not _is_rewriteable_specifier(
            requirement,
            built_version=built_version,
        ):
            raise ValueError(
                "expected source requirement to use the development sentinel "
                f"{SENTINEL_SPECIFIER!r} or a synchronized pin equivalent to "
                f"'=={built_version}', "
                f"got {str(requirement)!r}"
            )
        expected.add(_requirement_key(requirement))
    return expected


def _is_rewriteable_specifier(
    requirement: Requirement,
    *,
    built_version: str,
) -> bool:
    if str(requirement.specifier) == SENTINEL_SPECIFIER:
        return True

    return _is_exact_pin_to_version(
        requirement,
        version=built_version,
    )


def _is_exact_pin_to_version(
    requirement: Requirement,
    *,
    version: str,
) -> bool:
    specifiers = list(requirement.specifier)
    if len(specifiers) != 1 or specifiers[0].operator != "==":
        return False
    try:
        return Version(specifiers[0].version) == Version(version)
    except InvalidVersion:
        return False


def _metadata_requirements(
    wheel_path: Path,
) -> tuple[str, str, dict[tuple[str, tuple[str, ...], str | None], Requirement]]:
    with zipfile.ZipFile(wheel_path) as wheel:
        metadata_name = next(
            name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = message_from_bytes(wheel.read(metadata_name))

    dist_name = metadata["Name"]
    version = metadata["Version"]
    if dist_name is None or version is None:
        raise ValueError(f"{wheel_path} is missing Name/Version metadata")

    requirements: dict[tuple[str, tuple[str, ...], str | None], Requirement] = {}
    for raw_requirement in metadata.get_all("Requires-Dist", []):
        requirement = Requirement(raw_requirement)
        if requirement.name.startswith("opake-"):
            requirements[_requirement_key(requirement)] = requirement
    return dist_name, version, requirements


def _validate_wheel(
    wheel_path: Path,
    *,
    dist_to_pyproject: dict[str, Path],
) -> list[str]:
    dist_name, version, built_requirements = _metadata_requirements(wheel_path)
    pyproject_path = dist_to_pyproject.get(dist_name)
    if pyproject_path is None:
        raise ValueError(f"no pyproject.toml found for built wheel {dist_name!r}")

    source_requirements = _project_requirements(pyproject_path)
    expected_requirement_keys = _expected_requirement_keys(
        source_requirements,
        built_version=version,
    )

    errors: list[str] = []
    if built_requirements.keys() != expected_requirement_keys:
        missing = sorted(expected_requirement_keys - built_requirements.keys())
        unexpected = sorted(built_requirements.keys() - expected_requirement_keys)
        if missing:
            errors.append(f"{wheel_path.name}: missing opake requirements {missing}")
        if unexpected:
            errors.append(
                f"{wheel_path.name}: unexpected opake requirements {unexpected}"
            )
        return errors

    for key in expected_requirement_keys:
        built_requirement = built_requirements[key]
        if not _is_exact_pin_to_version(
            built_requirement,
            version=version,
        ):
            errors.append(
                f"{wheel_path.name}: expected {built_requirement.name} to pin "
                f"a version equivalent to =={version}, "
                f"got {built_requirement.specifier}"
            )

    return errors


def main() -> int:
    """Validate internal dependency pins for every wheel in the target directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wheel-dir",
        type=Path,
        required=True,
        help="Directory containing built wheel files to validate.",
    )
    args = parser.parse_args()

    wheel_paths = sorted(args.wheel_dir.glob("*.whl"))
    if not wheel_paths:
        raise SystemExit(f"ERROR: no wheel files found in {args.wheel_dir}")

    dist_to_pyproject = _dist_to_pyproject()
    errors: list[str] = []
    for wheel_path in wheel_paths:
        errors.extend(_validate_wheel(wheel_path, dist_to_pyproject=dist_to_pyproject))

    if errors:
        print(*errors, sep="\n", file=sys.stderr)
        return 1

    print(f"Validated internal opake pins for {len(wheel_paths)} wheels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
