"""Optional fused clipping kernels remain decoupled from opake-patches."""

from __future__ import annotations

import pytest
import torch

from opake.api.engine.clipping import _clipped_fun as cf
from opake.api.engine.clipping._clipped_fun import clipped_fun
from opake.exceptions import ConfigurationError


def test_triton_requires_a_registered_kernel(monkeypatch):
    monkeypatch.setattr(cf, "fused_kernels_available", lambda: True)
    monkeypatch.setattr(cf, "get_fused_clip_backend", lambda: None)

    with pytest.raises(ConfigurationError, match="apply_runtime_patches"):
        clipped_fun(lambda values: values, clip_backend="triton")


def test_auto_falls_back_when_no_kernel_is_registered(monkeypatch):
    monkeypatch.setattr(cf, "fused_kernels_available", lambda: True)
    monkeypatch.setattr(cf, "get_fused_clip_backend", lambda: None)
    batch = torch.tensor([[3.0, 4.0], [0.6, 0.8]])

    torch_fn, torch_state = clipped_fun(lambda values: values, clip_backend="torch")
    auto_fn, auto_state = clipped_fun(lambda values: values, clip_backend="auto")
    expected, _ = torch_fn(batch, state=torch_state)
    actual, _ = auto_fn(batch, state=auto_state)

    torch.testing.assert_close(actual.pytree, expected.pytree)
