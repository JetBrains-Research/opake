"""Optional fused clipping kernels remain decoupled from opake-patches."""

from __future__ import annotations

import torch

from opake.api.engine.clipping import _clipped_fun as cf
from opake.api.engine.clipping._clipped_fun import clipped_fun


def test_clipping_falls_back_when_no_kernel_is_registered(monkeypatch):
    monkeypatch.setattr(cf, "fused_kernels_available", lambda: True)
    monkeypatch.setattr(cf, "get_fused_clip_backend", lambda: None)
    batch = torch.tensor([[3.0, 4.0], [0.6, 0.8]])

    clip_fn, state = clipped_fun(lambda values: values, clipping_norm=1.0)
    actual, _ = clip_fn(batch, state=state)

    torch.testing.assert_close(actual.pytree, torch.tensor([1.2, 1.6]))
