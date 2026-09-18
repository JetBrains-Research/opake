"""Tests for ``mf_gaussian_noise``'s ``compute_dtype`` kwarg.

Pins down that the user-facing knob behaves the same way as the parallel
``opake.dpsgd.noise.gaussian_noise(..., compute_dtype=...)``:

- Default is ``torch.float32`` (sampling Gaussians in bf16/fp16 distorts
  the noise distribution; not safe under the standard analysis).
- The dtype is used **directly** for the internal ``torch.randn`` —
  there is no auto-promotion of the input tensor's dtype.
- The input pytree's dtype is preserved on output (type-stable boundary).
"""

from __future__ import annotations

import torch

from opake.api.dpftrl.noise import _engine as mf_engine
from opake.api.dpftrl.noise._identity import identity_strategy
from opake.dpftrl.noise import mf_gaussian_noise
from opake.random import key
from opake.types import clipped


def _template():
    return {"w": torch.zeros(8, dtype=torch.float32)}


def _build_noise(*, compute_dtype=None):
    """Construct the IdentityStrategy MF noise fn.

    ``compute_dtype=None`` means "let the factory default apply" — this is
    how we probe the documented default.  Anything else is forwarded as
    the explicit user value.
    """
    kwargs = {
        "n_steps": 4,
        "noise_multiplier": 1.0,
        "key": key(42),
    }
    if compute_dtype is not None:
        kwargs["compute_dtype"] = compute_dtype
    return mf_gaussian_noise(_template(), identity_strategy(), **kwargs)


def test_compute_dtype_default_matches_explicit_float32():
    """No-arg call is bit-identical to an explicit ``compute_dtype=torch.float32``
    call on the same key.  Pins the documented default down to the inner
    ``torch.randn`` precision (not just the output boundary cast)."""
    grads = clipped({"w": torch.zeros(64, dtype=torch.float32)}, max_norm=1.0)
    nf_default, st_default = _build_noise()
    nf_explicit, st_explicit = _build_noise(compute_dtype=torch.float32)
    out_default, _ = nf_default(grads, st_default)
    out_explicit, _ = nf_explicit(grads, st_explicit)
    assert torch.equal(out_default.pytree["w"], out_explicit.pytree["w"])


def test_compute_dtype_default_propagates_to_torch_randn(monkeypatch):
    """Direct check that the default reaches the inner ``torch.randn`` as fp32."""
    captured: list[torch.dtype] = []
    real_randn = torch.randn

    def _capture(*args, **kwargs):
        captured.append(kwargs.get("dtype", torch.get_default_dtype()))
        return real_randn(*args, **kwargs)

    monkeypatch.setattr(mf_engine.torch, "randn", _capture)

    grads = clipped({"w": torch.zeros(8, dtype=torch.float32)}, max_norm=1.0)
    nf, st = _build_noise()
    nf(grads, st)

    assert captured, "torch.randn was not invoked inside the noise step"
    assert all(d == torch.float32 for d in captured), (
        f"expected every inner torch.randn to use float32, got {captured}"
    )


def test_compute_dtype_preserves_input_dtype_on_output():
    """Bf16 input pytree stays bf16 on output even with fp32 internal compute."""
    nf, st = _build_noise(compute_dtype=torch.float32)
    grads = clipped({"w": torch.zeros(8, dtype=torch.bfloat16)}, max_norm=1.0)
    noisy, _ = nf(grads, st)
    assert noisy.pytree["w"].dtype == torch.bfloat16


def test_compute_dtype_float64_override_propagates_to_torch_randn(monkeypatch):
    """User-passed fp64 reaches the inner ``torch.randn``."""
    captured: list[torch.dtype] = []
    real_randn = torch.randn

    def _capture(*args, **kwargs):
        captured.append(kwargs.get("dtype", torch.get_default_dtype()))
        return real_randn(*args, **kwargs)

    monkeypatch.setattr(mf_engine.torch, "randn", _capture)

    grads = clipped({"w": torch.zeros(8, dtype=torch.float64)}, max_norm=1.0)
    nf, st = _build_noise(compute_dtype=torch.float64)
    noisy, _ = nf(grads, st)

    assert noisy.pytree["w"].dtype == torch.float64
    assert captured, "torch.randn was not invoked inside the noise step"
    assert all(d == torch.float64 for d in captured), (
        f"expected every inner torch.randn to use float64, got {captured}"
    )


def test_compute_dtype_drives_inner_sampling_dtype():
    """fp32 vs fp64 ``compute_dtype`` produce different noise realisations on
    the same key — confirms the kwarg actually drives the internal sampling
    dtype rather than being a no-op.  Same-key fp32 sampling rounds to the
    fp32 grid, fp64 sampling does not, so the post-boundary fp32 outputs
    diverge."""
    grads = clipped({"w": torch.zeros(64, dtype=torch.float32)}, max_norm=1.0)
    nf32, s32 = _build_noise(compute_dtype=torch.float32)
    nf64, s64 = _build_noise(compute_dtype=torch.float64)
    out32, _ = nf32(grads, s32)
    out64, _ = nf64(grads, s64)
    assert not torch.equal(out32.pytree["w"], out64.pytree["w"].to(torch.float32))
