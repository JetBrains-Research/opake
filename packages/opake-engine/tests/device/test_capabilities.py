"""Tests for the per-device capability helper (single source of truth)."""

import pytest
import torch

from opake.api.engine.device import _capabilities
from opake.device import (
    DeviceCapabilities,
    device_capabilities,
    fused_kernels_available,
    sdpa_autocast_under_vmap_broken,
)


class TestDeviceCapabilities:
    def test_accepts_str_and_device(self):
        assert device_capabilities("cpu") == device_capabilities(torch.device("cpu"))

    def test_only_device_type_matters(self):
        # Index is irrelevant — capabilities are a property of the backend.
        assert device_capabilities("cuda") == device_capabilities("cuda:3")

    def test_cpu_capabilities(self):
        caps = device_capabilities("cpu")
        assert isinstance(caps, DeviceCapabilities)
        assert caps.device_type == "cpu"
        assert caps.supports_bf16 is True  # functional (slow) on CPU
        assert caps.supports_fused_kernels is False  # Triton needs CUDA
        assert caps.peak_memory_trackable is False
        assert caps.supports_pin_memory is False
        assert caps.is_accelerator is False

    def test_compile_backend_is_inductor_or_none(self):
        for dt in ("cpu", "cuda", "mps"):
            caps = device_capabilities(dt)
            if caps.supports_compile:
                assert caps.recommended_compile_backend == "inductor"
            else:
                assert caps.recommended_compile_backend is None

    def test_fused_kernels_requires_cuda(self):
        # On any non-CUDA host the fused-kernel runtime is unavailable, and the
        # mps/cpu capability records agree with the host-level helper.
        assert device_capabilities("mps").supports_fused_kernels is False
        assert device_capabilities("cpu").supports_fused_kernels is False
        if not torch.cuda.is_available():
            assert fused_kernels_available() is False

    @pytest.mark.mps
    def test_mps_is_first_class(self):
        caps = device_capabilities("mps")
        assert caps.device_type == "mps"
        assert caps.is_accelerator is True
        # The headline of the macOS work: bf16 + torch.compile are available.
        assert caps.supports_bf16 is True
        assert caps.supports_compile is True
        assert caps.recommended_compile_backend == "inductor"
        # Honest about what MPS still lacks.
        assert caps.supports_fused_kernels is False
        assert caps.supports_pin_memory is False

    def test_mps_peak_tracking_detects_generic_allocator_stats(self, monkeypatch):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: object())
        monkeypatch.setattr(
            torch.accelerator.memory,
            "memory_stats",
            lambda: {"allocated_bytes.all.peak": 0},
        )
        _capabilities._peak_memory_trackable.cache_clear()
        try:
            assert _capabilities._peak_memory_trackable("mps") is True
        finally:
            _capabilities._peak_memory_trackable.cache_clear()

    def test_mps_peak_tracking_handles_unsupported_allocator_stats(self, monkeypatch):
        def unsupported_memory_stats():
            raise RuntimeError("MPS allocator statistics are not implemented")

        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: object())
        monkeypatch.setattr(
            torch.accelerator.memory,
            "memory_stats",
            unsupported_memory_stats,
        )
        _capabilities._peak_memory_trackable.cache_clear()
        try:
            assert _capabilities._peak_memory_trackable("mps") is False
        finally:
            _capabilities._peak_memory_trackable.cache_clear()

    @pytest.mark.cuda
    def test_cuda_capabilities(self):
        caps = device_capabilities("cuda")
        assert caps.device_type == "cuda"
        assert caps.is_accelerator is True
        assert caps.peak_memory_trackable is True
        assert caps.supports_pin_memory is True


class TestSdpaAutocastUnderVmapBroken:
    def test_false_off_mps(self):
        # The bug (and its eager-attention workaround) is MPS-only; CPU/CUDA
        # always cast SDPA correctly under vmap(grad).
        assert sdpa_autocast_under_vmap_broken("cpu") is False
        assert sdpa_autocast_under_vmap_broken("cuda") is False

    @pytest.mark.mps
    def test_returns_bool_on_mps(self):
        # Empirical probe — True on a buggy torch, False once the upstream fix
        # (pytorch/pytorch#187282) lands; either way a plain bool.
        assert isinstance(sdpa_autocast_under_vmap_broken("mps"), bool)
