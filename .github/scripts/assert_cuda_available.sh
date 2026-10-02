#!/usr/bin/env bash
# Fail a CUDA test lane without a usable, supported device. conftest.py skips
# every `cuda`-marked test when torch reports no device, so pytest can otherwise
# report green without validating anything.

set -euo pipefail

: "${PYTEST_MARKER:?PYTEST_MARKER must be set}"

# Drop `not cuda` first, so a bare `cuda` remains only in expressions that
# positively select CUDA tests; the word boundaries stop `cudagraphs` matching.
positive_marker_terms="${PYTEST_MARKER//not cuda/}"
if [[ ! "$positive_marker_terms" =~ (^|[^[:alnum:]_])cuda([^[:alnum:]_]|$) ]]; then
  echo "Marker expression '$PYTEST_MARKER' does not select CUDA tests; no CUDA preflight needed."
  exit 0
fi

echo "Marker expression '$PYTEST_MARKER' selects CUDA tests; asserting a usable CUDA device."

if uv run --no-sync python - <<'PY'
import sys

import torch


def fail(reason: str) -> None:
    print(
        f"{reason} (torch {torch.__version__}, torch.version.cuda={torch.version.cuda})",
        file=sys.stderr,
    )
    sys.exit(1)


if not torch.cuda.is_available():
    fail("torch reports no available CUDA device")

devices = torch.cuda.device_count()
if devices < 1:
    fail("torch reports CUDA as available but exposes no device")

# The CUDA lane includes Triton kernels, whose supported NVIDIA floor is CC 8.0.
minimum_compute_capability = (8, 0)
compute_capability = torch.cuda.get_device_capability(0)
found = ".".join(map(str, compute_capability))
if compute_capability < minimum_compute_capability:
    required = ".".join(map(str, minimum_compute_capability))
    fail(
        f"CUDA device 0 has compute capability {found}; "
        f"the Triton test suite requires at least {required}"
    )

# A driver that answers queries is not a runtime that computes, and a host
# update can break the second while the first still responds. Read a value back
# off the device instead of trusting `is_available()` on its own.
probe = torch.zeros(1, device="cuda")
probe += 1
if probe.item() != 1.0:
    fail("arithmetic on a CUDA tensor returned an unexpected value")

print(
    f"CUDA is usable: {devices} device(s), device 0 is "
    f"{torch.cuda.get_device_name(0)} "
    f"(compute capability {found}, torch {torch.__version__}, "
    f"CUDA {torch.version.cuda})"
)
PY
then
  exit 0
fi

echo "::error::CUDA preflight failed on a lane selecting '$PYTEST_MARKER'. The lane requires a usable, supported CUDA device, so it stops before pytest can skip the selection or fail inside a kernel."
exit 1
