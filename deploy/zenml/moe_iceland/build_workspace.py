"""Install the pinned workspace in the image builder, without resolving dependencies."""

import os
import subprocess
import sys
from pathlib import Path

from .source import package_paths, verify_source


def main() -> None:
    """Build wheels only inside the bounded image-builder source tree."""
    root = Path.cwd()
    verify_source(root)
    paths = package_paths(root)
    native = "packages/opaque-accounting"
    command = [
        "uv",
        "--no-config",
        "pip",
        "install",
        "--python",
        sys.executable,
        "--no-deps",
        "--no-build-isolation",
        "--no-sources",
    ]
    env = {**os.environ, "SETUPTOOLS_SCM_PRETEND_VERSION": "0.0.0.dev0"}
    subprocess.run(
        [*command, "--config-settings", "build-args=--locked", str(root / native)],
        env=env,
        check=True,
        timeout=600,
    )
    subprocess.run(
        [*command, *(str(root / path) for path in paths if path != native)],
        env=env,
        check=True,
        timeout=600,
    )
    subprocess.run(
        ["uv", "pip", "check", "--python", sys.executable], check=True, timeout=30
    )


if __name__ == "__main__":
    main()
