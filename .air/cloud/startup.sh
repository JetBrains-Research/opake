#!/bin/sh
set -eu

REPO_ROOT=$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd)
CACHE_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/opake-air"
STATE_DIR="$CACHE_ROOT/startup"
SYNC_STAMP="$STATE_DIR/dev-sync.sha256"

mkdir -p "$STATE_DIR"

apt_get() {
    apt-get \
        -o "Acquire::http::Proxy=${HTTP_PROXY:-}" \
        -o "Acquire::https::Proxy=${HTTPS_PROXY:-${HTTP_PROXY:-}}" \
        "$@"
}

ensure_apt_https_sources() {
    sources_file="/etc/apt/sources.list.d/ubuntu.sources"

    if [ ! -f "$sources_file" ]; then
        return
    fi

    if [ "$(id -u)" -eq 0 ]; then
        sed -i \
            -e 's|http://archive.ubuntu.com/ubuntu/|https://archive.ubuntu.com/ubuntu/|g' \
            -e 's|http://security.ubuntu.com/ubuntu/|https://security.ubuntu.com/ubuntu/|g' \
            "$sources_file"
        return
    fi

    if command -v sudo >/dev/null 2>&1; then
        sudo sed -i \
            -e 's|http://archive.ubuntu.com/ubuntu/|https://archive.ubuntu.com/ubuntu/|g' \
            -e 's|http://security.ubuntu.com/ubuntu/|https://security.ubuntu.com/ubuntu/|g' \
            "$sources_file"
    fi
}

ensure_system_packages() {
    if command -v cc >/dev/null 2>&1 && command -v curl >/dev/null 2>&1; then
        return
    fi

    if ! command -v apt-get >/dev/null 2>&1; then
        echo "Required system packages are missing and no supported package manager is available." >&2
        exit 1
    fi

    ensure_apt_https_sources

    if [ "$(id -u)" -eq 0 ]; then
        apt_get update
        apt_get install -y --no-install-recommends \
            curl ca-certificates build-essential pkg-config git
        return
    fi

    if command -v sudo >/dev/null 2>&1; then
        sudo apt-get \
            -o "Acquire::http::Proxy=${HTTP_PROXY:-}" \
            -o "Acquire::https::Proxy=${HTTPS_PROXY:-${HTTP_PROXY:-}}" \
            update
        sudo apt-get \
            -o "Acquire::http::Proxy=${HTTP_PROXY:-}" \
            -o "Acquire::https::Proxy=${HTTPS_PROXY:-${HTTP_PROXY:-}}" \
            install -y --no-install-recommends \
            curl ca-certificates build-essential pkg-config git
        return
    fi

    echo "Required system packages are missing and neither root nor sudo is available." >&2
    exit 1
}

ensure_uv() {
    if ! command -v uv >/dev/null 2>&1; then
        curl -x "${HTTPS_PROXY:-${HTTP_PROXY:-}}" -LsSf https://astral.sh/uv/install.sh | sh
    fi
}

ensure_rust() {
    if ! command -v cargo >/dev/null 2>&1; then
        curl -x "${HTTPS_PROXY:-${HTTP_PROXY:-}}" --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    fi
}

persist_shell_env() {
    # Persist the expansion for future shells.
    # shellcheck disable=SC2016
    grep -q 'opake uv/cargo PATH' "$HOME/.bashrc" 2>/dev/null || \
        printf '\n# opake uv/cargo PATH\nexport PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"\n' >> "$HOME/.bashrc"
    grep -q 'PYTHONUNBUFFERED' "$HOME/.bashrc" 2>/dev/null || \
        echo 'export PYTHONUNBUFFERED=1' >> "$HOME/.bashrc"
}

sync_fingerprint() {
    cat \
        "$REPO_ROOT/pyproject.toml" \
        "$REPO_ROOT/uv.lock" \
        "$REPO_ROOT/Cargo.lock" \
        "$REPO_ROOT/.air/cloud/startup.sh" | sha256sum | awk '{print $1}'
}

warm_workspace() {
    current_fingerprint=$(sync_fingerprint)
    previous_fingerprint=""

    if [ -f "$SYNC_STAMP" ]; then
        previous_fingerprint=$(cat "$SYNC_STAMP")
    fi

    if [ ! -d "$REPO_ROOT/.venv" ] || [ "$current_fingerprint" != "$previous_fingerprint" ]; then
        cd "$REPO_ROOT"
        uv sync --group dev --all-packages --extra all
        cargo test --workspace --no-run
        printf '%s' "$current_fingerprint" > "$SYNC_STAMP"
    fi
}

healthcheck() {
    cd "$REPO_ROOT"

    uv run python - <<'PY'
import opake.accounting
import opake.auditing
import opake.distributed
import opake.dpftrl
import opake.dpsgd
import opake.functional
import opake.profiling
import opake.random
import opake.scheduling
import opake.serialization
PY

    uv run pytest \
        packages/opake-accounting/tests/test_smoke.py \
        tests/contracts/test_pep420_no_init.py \
        -m "not cuda and not mps and not slow" \
        -q
}

ensure_system_packages
ensure_uv
ensure_rust

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export PYTHONUNBUFFERED=1

persist_shell_env
warm_workspace
healthcheck
