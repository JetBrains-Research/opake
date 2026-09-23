"""Command-line entry point for differentially private supervised fine-tuning."""

# ruff: noqa: INP001

from __future__ import annotations

import sys
from pathlib import Path

from opaque_examples.sft import parse_sft_config


def main(output_dir: Path | None = None) -> int:
    """Resolve the CLI configuration and execute the reusable SFT runner."""
    config = parse_sft_config(sys.argv[1:])
    destination = Path(config.output_dir) if output_dir is None else output_dir

    from opaque_examples.sft.runner import run_sft

    run_sft(config, destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
