#!/usr/bin/env bash
#
# set_build_versions.sh — write the current git-derived version into every
# build artifact that can't be handled by setuptools-scm natively:
#
#   1. `packages/opake-accounting/pyproject.toml` — maturin reads the wheel
#      version from `[project] version`; maturin doesn't know about
#      setuptools-scm.
#   2. `packages/opake-accounting/Cargo.toml` — the Rust crate version must
#      match the Python wheel version for the PyO3 extension.
#   3. Every Python package `pyproject.toml` that depends on another
#      `opake-*` wheel — these internal runtime dependencies must be pinned to
#      the synchronized build version in published metadata, both for the root
#      `opake` umbrella wheel and for cross-package dependencies between leaf
#      wheels.
#
# For Python sub-packages the script also exports
# `SETUPTOOLS_SCM_PRETEND_VERSION` via `$GITHUB_ENV` so every wheel carries
# the same version — setuptools-scm's own git-describe would otherwise pick
# up the dirty tree (this script mutates pyprojects/Cargo.toml before the
# build) and produce a drifted `.dev0+g<sha>.d<date>` string.
#
# Intended to run in CI as a preflight before `uv build`. On local dev,
# leave pyprojects / Cargo.toml untouched — uv workspace resolution and
# setuptools-scm fallbacks handle that case.
#
# Usage: bash .github/scripts/set_build_versions.sh [VERSION]
#
# If VERSION is omitted, derive it from `git describe --tags --match 'v*'`
# and normalize to PEP 440:
#
#   v0.2.0                     → 0.2.0
#   v0.2.0-dirty               → 0.2.0+d20260422
#   v0.2.0-5-g810b6b2          → 0.2.1.dev5+g810b6b2
#   v0.2.0-5-g810b6b2-dirty    → 0.2.1.dev5+g810b6b2.d20260422
#   v0.3.0.rc1                 → 0.3.0.rc1
#   v0.3.0.rc1-5-g810b6b2      → 0.3.0.rc1.dev5+g810b6b2
#   v0.2.0.post1               → 0.2.0.post1
#   v0.2.0.post1-5-g810b6b2    → 0.2.1.dev5+g810b6b2
#
# PEP 440 gotcha: `0.3.0.rc1.dev5 < 0.3.0.rc1`, so post-rc1 distance builds
# sort *before* rc1. Pins like `>=0.3.0.rc1` still resolve to rc1 or later;
# dev wheels between rc1 and rc2 only install if pinned explicitly.

set -euo pipefail

VERSION="${1:-}"
if [[ -z "$VERSION" ]]; then
  if ! RAW=$(git describe --tags --match 'v*' --dirty 2>/dev/null); then
    echo "ERROR: no v* tag found and no explicit VERSION given" >&2
    exit 1
  fi
  RAW="${RAW#v}"

  DIRTY_SUFFIX=""
  if [[ "$RAW" == *-dirty ]]; then
    DIRTY_SUFFIX=".d$(date -u +%Y%m%d)"
    RAW="${RAW%-dirty}"
  fi

  # Supported release-tag shapes (all PEP 440-parseable):
  #   * release               — `X.Y.Z`               (e.g. 0.2.0)
  #   * post-release          — `X.Y.Z.postN`         (e.g. 0.2.0.post1)
  #   * pre-release           — `X.Y.Z.alphaN`        (e.g. 0.3.0.alpha1)
  #   * pre-release           — `X.Y.Z.betaN`         (e.g. 0.3.0.beta1)
  #   * release candidate     — `X.Y.Z.rcN`           (e.g. 0.3.0.rc1)
  #
  # `.devN` tag shapes are rejected — `.dev` is reserved for workflow-
  # generated pre-release wheels (main-merge dev wheels, PR preview wheels).
  # Anything else (hyphen-form, `-test` markers) is rejected up front.
  SUPPORTED_TAG='^([0-9]+)\.([0-9]+)\.([0-9]+)(\.(post|alpha|beta|rc)[0-9]+)?$'

  # git-describe emits `<tag>-<distance>-g<sha>` when HEAD is past the tag.
  # Normalize to PEP 440, mirroring setuptools-scm's `guess-next-dev` scheme.
  if [[ "$RAW" =~ ^(.+)-([0-9]+)-g([0-9a-f]+)$ ]]; then
    TAG_PART="${BASH_REMATCH[1]}"
    DISTANCE="${BASH_REMATCH[2]}"
    SHA="${BASH_REMATCH[3]}"

    if [[ "$TAG_PART" =~ $SUPPORTED_TAG ]]; then
      MAJOR="${BASH_REMATCH[1]}"
      MINOR="${BASH_REMATCH[2]}"
      PATCH="${BASH_REMATCH[3]}"
      PRERELEASE="${BASH_REMATCH[4]}"
      PRERELEASE_KIND="${BASH_REMATCH[5]}"

      case "$PRERELEASE_KIND" in
        alpha|beta|rc)
          # Post-prerelease progress: X.Y.Z.<rc|alpha|beta>N.dev<D>+g<sha>.
          # (See header note on PEP 440 ordering.)
          VERSION="${MAJOR}.${MINOR}.${PATCH}${PRERELEASE}.dev${DISTANCE}+g${SHA}${DIRTY_SUFFIX}"
          ;;
        *)
          # Release or post-release tag: bump patch, use distance as dev N.
          VERSION="${MAJOR}.${MINOR}.$((PATCH + 1)).dev${DISTANCE}+g${SHA}${DIRTY_SUFFIX}"
          ;;
      esac
    else
      echo "ERROR: tag '$TAG_PART' is not a supported shape (X.Y.Z or X.Y.Z.{postN,alphaN,betaN,rcN})" >&2
      exit 1
    fi
  elif [[ "$RAW" =~ $SUPPORTED_TAG ]]; then
    # Clean checkout at the tag — no distance, no sha.
    VERSION="${RAW}${DIRTY_SUFFIX:++${DIRTY_SUFFIX#.}}"
  else
    echo "ERROR: tag '$RAW' is not a supported shape (X.Y.Z or X.Y.Z.{postN,alphaN,betaN,rcN})" >&2
    exit 1
  fi
fi

echo "opake build version → $VERSION"

# --- opake-accounting (maturin) --------------------------------------------
# Python wheel version lives in the package's pyproject.toml and accepts the
# full PEP 440 string. Cargo.toml demands SemVer, which rejects PEP 440 form
# (`.dev42`, `.rc1`, `.post1`). Transform PEP 440 → SemVer for the Rust side:
#
#   0.2.0                     → 0.2.0
#   0.2.1.dev5+g<sha>         → 0.2.1-dev.5+g<sha>
#   0.3.0.rc1                 → 0.3.0-rc.1
#   0.3.0.rc1.dev5+g<sha>     → 0.3.0-rc.1.dev.5+g<sha>
#   0.2.0.post1               → 0.2.0+post.1
#   0.2.0.post1+d20260422     → 0.2.0+post.1.d20260422           (merge into existing build metadata — SemVer forbids two `+`)
#   0.2.0.post1.dev5+g<sha>   → 0.2.0-post.1.dev.5+g<sha>        (ambiguous when combined; lean SemVer pre-release)
#   0.2.0+d20260422           → 0.2.0+d20260422
CARGO_VERSION=$(echo "$VERSION" | sed -E \
  -e 's/\.(alpha|beta|rc|post)([0-9]+)\.(dev)([0-9]+)/-\1.\2.\3.\4/' \
  -e 's/\.(alpha|beta|rc)([0-9]+)/-\1.\2/' \
  -e 's/\.post([0-9]+)\+([0-9A-Za-z.-]+)/+post.\1.\2/' \
  -e 's/\.post([0-9]+)/+post.\1/' \
  -e 's/\.dev([0-9]+)/-dev.\1/')

# Python wheel metadata
sed -i.bak -E "s%^version = \"[^\"]+\"%version = \"$VERSION\"%" \
  packages/opake-accounting/pyproject.toml
rm -f packages/opake-accounting/pyproject.toml.bak

# Rust crate version (workspace-wide, inherited by opake-accounting/Cargo.toml)
sed -i.bak -E "s%^version = \"[^\"]+\"%version = \"$CARGO_VERSION\"%" Cargo.toml
rm -f Cargo.toml.bak

# --- opake: pin internal wheel dependencies to the same version ------------
python3 - <<'PY' "$VERSION"
from __future__ import annotations

from pathlib import Path
import sys
import tomllib


VERSION = sys.argv[1]
SENTINEL_SUFFIX = ">=0.0.0.dev0"
PYPROJECTS = (
    Path("pyproject.toml"),
    *sorted(Path("packages").glob("*/pyproject.toml")),
)


def project_list(data: dict[str, object], path: tuple[str, ...]) -> list[str]:
    current: object = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            joined = ".".join(path)
            raise SystemExit(f"ERROR: pyproject.toml is missing [{joined}]")
        current = current[key]
    if not isinstance(current, list):
        joined = ".".join(path)
        raise SystemExit(f"ERROR: pyproject.toml entry {joined} is not a list")
    if any(not isinstance(item, str) for item in current):
        joined = ".".join(path)
        raise SystemExit(f"ERROR: pyproject.toml entry {joined} must contain only strings")
    return current


def internal_dependency_targets(
    pyproject_path: Path,
    data: dict[str, object],
) -> list[tuple[tuple[str, ...], str]]:
    project = data.get("project")
    if not isinstance(project, dict):
        raise SystemExit(f"ERROR: {pyproject_path} is missing [project]")

    targets: list[tuple[tuple[str, ...], str]] = []

    if "dependencies" in project:
        for requirement in project_list(data, ("project", "dependencies")):
            if requirement.startswith("opake-"):
                targets.append((("project", "dependencies"), requirement))

    optional = project.get("optional-dependencies", {})
    if optional and not isinstance(optional, dict):
        raise SystemExit(
            f"ERROR: {pyproject_path} entry project.optional-dependencies is not a table"
        )
    if isinstance(optional, dict):
        for extra_name in optional:
            path = ("project", "optional-dependencies", extra_name)
            for requirement in project_list(data, path):
                if requirement.startswith("opake-"):
                    targets.append((path, requirement))

    return targets


for pyproject_path in PYPROJECTS:
    text = pyproject_path.read_text(encoding="utf-8")
    data = tomllib.loads(text)
    pin_targets = internal_dependency_targets(pyproject_path, data)

    for path, sentinel in pin_targets:
        joined = ".".join(path)
        if not sentinel.endswith(SENTINEL_SUFFIX):
            raise SystemExit(
                "ERROR: expected rewriteable sentinel for internal dependency "
                f"{sentinel!r} in {pyproject_path}:{joined}"
            )
        count = project_list(data, path).count(sentinel)
        if count != 1:
            raise SystemExit(
                f"ERROR: expected {sentinel!r} exactly once in {pyproject_path}:{joined}, found {count}"
            )
        pinned = sentinel.removesuffix(SENTINEL_SUFFIX) + f"=={VERSION}"
        quoted_sentinel = f'"{sentinel}"'
        quoted_pinned = f'"{pinned}"'
        if quoted_sentinel not in text:
            raise SystemExit(
                "ERROR: internal dependency is present in parsed metadata but "
                f"missing from source text: {pyproject_path}:{sentinel!r}"
            )
        text = text.replace(quoted_sentinel, quoted_pinned, 1)

    updated = tomllib.loads(text)
    for path, sentinel in pin_targets:
        joined = ".".join(path)
        items = project_list(updated, path)
        pinned = sentinel.removesuffix(SENTINEL_SUFFIX) + f"=={VERSION}"
        count = items.count(pinned)
        if count != 1:
            raise SystemExit(
                "ERROR: expected pinned requirement "
                f"{pinned!r} exactly once in {pyproject_path}:{joined}, found {count}"
            )
        if sentinel in items:
            raise SystemExit(
                f"ERROR: unreplaced sentinel {sentinel!r} remains in {pyproject_path}:{joined}"
            )

    pyproject_path.write_text(text, encoding="utf-8")
PY

echo "Updated version pins:"
grep -E "^version = \"|opake-[a-z-]+" Cargo.toml pyproject.toml packages/*/pyproject.toml | sed 's|^|  |'

# --- export for downstream build steps --------------------------------------
# setuptools-scm would otherwise re-derive the version from the now-dirty
# tree and drift from what we've written into opake-accounting/pyproject.toml.
if [[ -n "${GITHUB_ENV:-}" ]]; then
  echo "SETUPTOOLS_SCM_PRETEND_VERSION=$VERSION" >> "$GITHUB_ENV"
fi
