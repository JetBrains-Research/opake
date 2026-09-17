"""Cross-device parity for DP-SGD Gaussian noise (privacy correctness).

The privacy guarantee on Apple Silicon (MPS) and CUDA rests on the noise being
drawn from the *same* key-determined random stream regardless of device.
opake samples the underlying standard-normal values on a CPU generator and
moves them to the compute device (see ``_gaussian._sample``:
``torch.randn(..., generator=cpu_gen).to(device)``), so the stream is
device-independent; only the on-device addition rounds differently between
backends.

These tests lock that in: for a fixed key, CPU and accelerator noise must agree
to floating-point-rounding tolerance (proving the same draws), and a *different*
key must diverge by O(noise scale) (proving the tolerance isn't vacuous).  A
regression that made noise device-dependent — e.g. sampling uniforms on the
device with a device generator — would break these and flag a privacy bug, not
merely a performance one.  Track-B performance work (kernels / compile / dtype)
touches the model forward-backward only, never this noise path; these tests are
the guard that keeps that boundary honest.
"""

import pytest
import torch

from opake.dpsgd.noise import gaussian_noise
from opake.random import key
from opake.types import clipped

_N = 50_000
_SIGMA = 1.0
# 1e-4 is a comfortable ceiling for CPU-vs-MPS addition rounding while still
# rejecting a different random stream, which would differ by O(_SIGMA) ~ 1.0.
_ROUNDING_ATOL = 1e-4


def _noise(device: str, seed: int, n: int = _N, sigma: float = _SIGMA) -> torch.Tensor:
    """Return the pure noise sample on ``device`` for ``key(seed)``.

    Input is zeros, so the noised output *is* the noise — directly comparable
    across devices.  Always returned on CPU for comparison.
    """
    noise_fn, state = gaussian_noise(noise_multiplier=sigma, key=key(seed))
    zeros = torch.zeros(n, device=device)
    out, _ = noise_fn(clipped(zeros, max_norm=1.0), state)
    return out.pytree.detach().to("cpu")


def _assert_same_stream(cpu: torch.Tensor, other: torch.Tensor) -> None:
    diff = (cpu - other).abs().max().item()
    # Same stream: element-wise agreement at fp-rounding scale ...
    assert diff < _ROUNDING_ATOL, (
        f"cross-device noise diverged by {diff:.2e} (>= {_ROUNDING_ATOL:.0e}) — "
        "the random stream is not device-independent; privacy accounting would "
        "no longer match the realized noise."
    )
    # ... and that scale is negligible relative to the noise itself.
    assert diff < 0.01 * cpu.std().item()


@pytest.mark.mps
def test_gaussian_noise_cpu_mps_parity() -> None:
    """Same key → same noise on CPU and MPS (up to addition rounding)."""
    _assert_same_stream(_noise("cpu", seed=123), _noise("mps", seed=123))


@pytest.mark.mps
def test_gaussian_noise_mps_different_key_diverges() -> None:
    """Guard against a vacuous tolerance: a different key must not match."""
    cpu = _noise("cpu", seed=123)
    mps_other = _noise("mps", seed=124)
    assert (cpu - mps_other).abs().max().item() > _ROUNDING_ATOL


@pytest.mark.cuda
def test_gaussian_noise_cpu_cuda_parity() -> None:
    """Same key → same noise on CPU and CUDA (up to addition rounding)."""
    _assert_same_stream(_noise("cpu", seed=123), _noise("cuda", seed=123))
