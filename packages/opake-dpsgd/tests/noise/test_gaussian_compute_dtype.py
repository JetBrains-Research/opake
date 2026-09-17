"""Tests for the ``compute_dtype`` parameter on clipped Gaussian noise."""

from __future__ import annotations

import pytest
import torch

from opake.dpsgd.noise import gaussian_noise
from opake.exceptions import ConfigurationError, InputTypeError
from opake.random import key
from opake.types import clipped


def test_default_compute_dtype_is_fp32():
    noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(0))
    grads = {"a": torch.zeros(1024, dtype=torch.bfloat16)}
    noised, _ = noise_fn(clipped(grads, max_norm=1.0), state)
    assert noised.pytree["a"].dtype == torch.bfloat16


def test_fp32_input_stays_fp32():
    noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(0))
    grads = {"a": torch.zeros(1024, dtype=torch.float32)}
    noised, _ = noise_fn(clipped(grads, max_norm=1.0), state)
    assert noised.pytree["a"].dtype == torch.float32


@torch.no_grad()
def test_observed_variance_matches_calibrated_stddev():
    noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(42))
    grads = {"a": torch.zeros(8192, dtype=torch.float32)}
    noised, _ = noise_fn(clipped(grads, max_norm=1.0), state)
    observed_std = noised.pytree["a"].float().std().item()
    assert abs(observed_std - 1.0) < 0.05, f"observed_std={observed_std}"


def test_bf16_observed_variance_matches_calibrated_stddev():
    noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(42))
    grads = {"a": torch.zeros(8192, dtype=torch.bfloat16)}
    noised, _ = noise_fn(clipped(grads, max_norm=1.0), state)
    observed_std = noised.pytree["a"].float().std().item()
    assert abs(observed_std - 1.0) < 0.05, f"observed_std={observed_std}"


@pytest.mark.parametrize(
    "compute_dtype",
    [torch.float16, torch.bfloat16, torch.float32, torch.float64],
)
def test_supported_compute_dtypes_run(compute_dtype):
    noise_fn, state = gaussian_noise(
        noise_multiplier=1.0,
        key=key(0),
        compute_dtype=compute_dtype,
    )
    grads = {"a": torch.zeros(64, dtype=torch.float64)}
    noised, _ = noise_fn(clipped(grads, max_norm=1.0), state)
    assert noised.pytree["a"].dtype == torch.float64


def test_bf16_default_compute_dtype_beats_native_bf16():
    noise_fn, state = gaussian_noise(noise_multiplier=2.0, key=key(7))
    grads = {"a": torch.zeros(16384, dtype=torch.bfloat16)}
    noised, _ = noise_fn(clipped(grads, max_norm=1.0), state)
    observed_std = noised.pytree["a"].float().std().item()
    assert abs(observed_std - 2.0) < 0.05


@pytest.mark.parametrize("dtype", [torch.int64, torch.complex64])
def test_non_floating_compute_dtype_is_rejected(dtype):
    with pytest.raises(ConfigurationError, match="compute_dtype"):
        gaussian_noise(noise_multiplier=1.0, key=key(0), compute_dtype=dtype)


def test_compute_dtype_requires_torch_dtype():
    with pytest.raises(InputTypeError, match="compute_dtype"):
        gaussian_noise(noise_multiplier=1.0, key=key(0), compute_dtype="float32")  # type: ignore[arg-type]


def test_zero_noise_multiplier_short_circuits():
    noise_fn, state = gaussian_noise(noise_multiplier=0.0, key=key(0))
    grads = {"a": torch.ones(8, dtype=torch.bfloat16)}
    noised, _ = noise_fn(clipped(grads, max_norm=1.0), state)
    assert torch.equal(noised.pytree["a"], grads["a"])
    assert noised.pytree["a"].dtype == torch.bfloat16
