#!/usr/bin/env bash
# Install the dependency resolution selected by an explicit Python test group.

set -euo pipefail

: "${DEPENDENCY_SELECTION:=locked}"

# Later commands use `uv run --no-sync` so this selected resolution stays installed.
case "$DEPENDENCY_SELECTION" in
  locked)
    uv sync --locked --group dev --all-packages --extra all
    ;;
  minimum)
    uv sync --upgrade --resolution lowest-direct \
      --group dev --all-packages --extra all
    ;;
  latest)
    uv sync --upgrade --resolution highest \
      --group dev --all-packages --extra all
    ;;
  *)
    echo "::error::Unknown dependency selection: $DEPENDENCY_SELECTION"
    exit 1
    ;;
esac
