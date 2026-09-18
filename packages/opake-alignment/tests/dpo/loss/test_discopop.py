"""Tests for the DPO DiscoPOP blended loss variant."""

from __future__ import annotations

import math
from importlib import util
from pathlib import Path

import torch
import torch.nn.functional as F

_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src/opake/api/alignment/dpo/loss/_discopop.py"
)
_SPEC = util.spec_from_file_location(
    "opake_api_alignment_dpo_loss_discopop", _MODULE_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load DiscoPOP module from {_MODULE_PATH}")
_MODULE = util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
discopop_loss = _MODULE.discopop_loss


class TestDpoDiscopop:
    def test_discopop_delta_zero(self) -> None:
        chosen = torch.tensor(0.0)
        rejected = torch.tensor(0.0)
        out = discopop_loss(chosen, rejected, beta=1.0)
        logits = 0.0
        tau = 0.05
        gate = torch.sigmoid(torch.tensor(logits / tau)).item()
        logistic = -F.logsigmoid(torch.tensor(logits)).item()
        exp_comp = math.exp(-logits)
        expected = logistic * (1 - gate) + exp_comp * gate
        assert out.shape == ()
        assert torch.allclose(out, torch.tensor(expected), atol=1e-5)

    def test_discopop_large_positive_logits(self) -> None:
        chosen = torch.tensor(10.0)
        rejected = torch.tensor(0.0)
        out = discopop_loss(chosen, rejected, beta=1.0)
        assert out.shape == ()
        assert 0.0 <= out.item() < 1e-3

    def test_discopop_large_negative_logits_finite(self) -> None:
        chosen = torch.tensor(-50.0)
        rejected = torch.tensor(0.0)
        out = discopop_loss(chosen, rejected, beta=1.0)
        assert out.shape == ()
        assert torch.isfinite(out)
        assert out.item() > 0.0

    def test_discopop_batched_shape(self) -> None:
        chosen = torch.randn(5)
        rejected = torch.randn(5)
        out = discopop_loss(chosen, rejected, beta=0.5)
        assert out.shape == (5,)

    def test_discopop_per_example_matches_batched(self) -> None:
        torch.manual_seed(3)
        chosen = torch.randn(4)
        rejected = torch.randn(4)
        batched = discopop_loss(chosen, rejected, beta=0.5)
        per_example = torch.stack(
            [discopop_loss(chosen[i], rejected[i], beta=0.5) for i in range(4)]
        )
        assert torch.allclose(batched, per_example, atol=1e-6)

    def test_discopop_custom_tau(self) -> None:
        chosen = torch.tensor(1.0)
        rejected = torch.tensor(0.0)
        out_default = discopop_loss(chosen, rejected, beta=1.0, discopop_tau=0.05)
        out_tau1 = discopop_loss(chosen, rejected, beta=1.0, discopop_tau=1.0)
        assert not torch.allclose(out_default, out_tau1, atol=1e-4)
        assert torch.isfinite(out_tau1)

    def test_discopop_dtype_safe_clamp(self) -> None:
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            chosen = torch.tensor([-200.0, 0.0, 200.0], dtype=dtype)
            rejected = torch.zeros_like(chosen)
            out = discopop_loss(chosen, rejected, beta=1.0)
            assert out.dtype == dtype
            assert torch.isfinite(out).all()

    def test_discopop_gradient_is_finite_for_low_precision(self) -> None:
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            chosen = torch.tensor([-100.0, -2.0, 2.0], dtype=dtype, requires_grad=True)
            rejected = torch.zeros_like(chosen)
            loss = discopop_loss(chosen, rejected, beta=1.0).sum()
            loss.backward()
            assert chosen.grad is not None
            assert torch.isfinite(chosen.grad).all()

    def test_discopop_nan_injection_locality(self) -> None:
        chosen = torch.tensor([1.0, float("nan"), -0.5, 0.2])
        rejected = torch.tensor([0.0, 0.0, 0.0, 0.0])
        out = discopop_loss(chosen, rejected, beta=1.0)
        assert out.shape == (4,)
        assert torch.isnan(out[1])
        assert torch.isfinite(out[0])
        assert torch.isfinite(out[2])
        assert torch.isfinite(out[3])
