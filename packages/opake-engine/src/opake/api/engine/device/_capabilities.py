# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Single source of truth for per-device hardware capabilities.

Opake historically scattered device feature checks across the codebase: the
kernel router asks "CUDA + Triton?", the trainer config asks HuggingFace's
``is_torch_bf16_gpu_available()``, and the examples hard-code per-device dtype
and ``torch.compile`` rules.  That drift is how macOS/MPS functionality became
"hiddenly unavailable" — bf16 silently downgraded to fp32, fused kernels
silently disabled, ``torch.compile`` silently mis-targeted.

:func:`device_capabilities` answers all of those questions in one place so call
sites query a capability instead of re-deriving it.  The underlying probes are
deliberately empirical (allocate-and-run) rather than version-sniffing — so they
track whatever the installed PyTorch actually supports on the host hardware — and
each is cached per device *type* (``functools.lru_cache``), so a repeated query
re-runs no probe even though :func:`device_capabilities` itself rebuilds the
lightweight record each call.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch

__all__ = [
    "DeviceCapabilities",
    "device_capabilities",
    "fused_kernels_available",
    "sdpa_autocast_under_vmap_broken",
]


@functools.cache
def _triton_importable() -> bool:
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


@functools.cache
def _probe_bf16(device_type: str) -> bool:
    """Empirically determine whether ``device_type`` can run a bf16 op.

    A probe (rather than a version table) keeps this honest across the matrix
    of PyTorch versions and Apple Silicon generations: recent torch on an
    M-series GPU runs bf16, older stacks reject it, and this returns whatever
    is actually true on the host.
    """
    try:
        if device_type == "cuda":
            return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        if device_type == "mps":
            if not torch.backends.mps.is_available():
                return False
            probe = torch.ones(2, dtype=torch.bfloat16, device="mps")
            return (probe + probe).sum().item() == 4.0  # noqa: PLR2004 - bf16 probe
        if device_type == "cpu":
            # bf16 on CPU is functional (used for full-cast / autocast), just
            # slow; opake's trainer gates it behind ``use_cpu=True``.
            return True
    except Exception:
        return False
    return False


def fused_kernels_available() -> bool:
    """Host-level check: a CUDA device *and* an importable Triton.

    This is the runtime the Triton fused kernels (rope / rms_norm / swiglu /
    geglu / cross-entropy / fused-linear-CE) need.  On MPS / CPU (or CUDA
    without Triton) it returns ``False`` and callers fall back to the
    pure-PyTorch eager implementations.
    """
    return torch.cuda.is_available() and _triton_importable()


@functools.cache
def sdpa_autocast_under_vmap_broken(device_type: str) -> bool:
    """Whether ``torch.autocast`` fails to cast SDPA under ``vmap(grad)`` here.

    PyTorch bug (MPS): wrapped functorch tensors lose the MPS dispatch key, so
    ``AutocastMPS`` is never re-derived and ``torch.autocast`` silently no-ops
    under ``vmap``/``grad``.  A bf16 DP step (RoPE leaves q/k fp32 while v is the
    bf16 v_proj output) then crashes in ``scaled_dot_product_attention`` with a
    query/value dtype mismatch.  It casts fine on CPU/CUDA, and on MPS *without*
    functorch.  Tracked upstream as pytorch/pytorch#187265 (fix #187282).

    Empirical (run the failing op once, cached) and matched to the live torch, so
    the eager-attention workaround auto-drops the moment a release fixes it.
    Returns ``False`` on non-MPS devices and when the op runs cleanly.
    """
    if device_type != "mps" or not torch.backends.mps.is_available():
        return False

    def _loss(scale, q, k, v):
        out = torch.nn.functional.scaled_dot_product_attention(q * scale, k, v)
        return out.float().sum()

    q = torch.randn(2, 1, 4, 8, device="mps")
    k = torch.randn(2, 1, 4, 8, device="mps")
    v = torch.randn(2, 1, 4, 8, device="mps", dtype=torch.bfloat16)
    scale = torch.tensor(1.0, device="mps")
    try:
        with torch.autocast(device_type="mps", dtype=torch.bfloat16):
            torch.vmap(torch.func.grad(_loss), in_dims=(None, 0, 0, 0))(scale, q, k, v)
    except RuntimeError:
        return True  # dtype-mismatch crash → bug present, workaround needed
    return False  # ran cleanly → a torch that fixed it


def _recommended_compile_backend(device_type: str) -> str | None:
    # inductor generates Triton on CUDA, Metal on MPS (PyTorch 2.5+), and
    # C++/OpenMP on CPU — the right default on every device opake targets.
    if device_type in ("cuda", "mps", "cpu"):
        return "inductor"
    return None


@functools.cache
def _peak_memory_trackable(device_type: str) -> bool:
    """Whether the backend exposes a cheap, resettable allocated-memory peak."""
    if device_type == "cuda":
        return True
    if device_type != "mps" or not torch.backends.mps.is_available():
        return False

    # The generic API may exist without MPS allocator-statistics support.
    probe = torch.empty(1, device="mps")
    del probe
    try:
        stats = torch.accelerator.memory.memory_stats()
    except RuntimeError:
        return False
    return "allocated_bytes.all.peak" in stats


@dataclass(frozen=True)
class DeviceCapabilities:
    """What a given device can do, resolved once at the call site.

    Attributes:
        device_type: ``"cuda"``, ``"mps"``, or ``"cpu"``.
        supports_bf16: bf16 tensors/ops run on this device.
        supports_compile: ``torch.compile`` targets this device.
        recommended_compile_backend: the backend to default to
            (``"inductor"``) or ``None`` when compile isn't recommended.
        supports_fused_kernels: the Triton fused-kernel runtime is available
            (CUDA + Triton); otherwise eager fallbacks are used.
        peak_memory_trackable: the device exposes a cheap, resettable
            allocated-memory peak that ``step_perf`` can zero each step.
        supports_pin_memory: pinned host memory accelerates H2D copies here
            (CUDA only; a no-op on CPU, a noisy warning on MPS).
    """

    device_type: str
    supports_bf16: bool
    supports_compile: bool
    recommended_compile_backend: str | None
    supports_fused_kernels: bool
    peak_memory_trackable: bool
    supports_pin_memory: bool

    @property
    def is_accelerator(self) -> bool:
        """True for GPU-class devices (CUDA, MPS); False for CPU."""
        return self.device_type in ("cuda", "mps")


def device_capabilities(device: torch.device | str) -> DeviceCapabilities:
    """Resolve the :class:`DeviceCapabilities` for ``device``.

    Args:
        device: a ``torch.device`` or device string (``"cuda"``, ``"mps"``,
            ``"cpu"``, ``"cuda:1"`` …); only the device *type* matters.

    Returns:
        The capability record for that device type.

    Example:
        >>> caps = device_capabilities("mps")
        >>> dtype = torch.bfloat16 if caps.supports_bf16 else torch.float32
    """
    if isinstance(device, str):
        device = torch.device(device)
    dt = device.type
    return DeviceCapabilities(
        device_type=dt,
        supports_bf16=_probe_bf16(dt),
        supports_compile=_recommended_compile_backend(dt) is not None,
        recommended_compile_backend=_recommended_compile_backend(dt),
        supports_fused_kernels=(dt == "cuda" and fused_kernels_available()),
        peak_memory_trackable=_peak_memory_trackable(dt),
        supports_pin_memory=(dt == "cuda"),
    )
