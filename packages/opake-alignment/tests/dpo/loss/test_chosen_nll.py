"""Tests for the DPO SFT regulariser (chosen-NLL) loss variant.

Covers work-unit γ.3 of the opake-alignment plan (§10, §11.2, §11.3).

Hand-computed reference cases plus a NaN-injection DP-purity contract test
(§11.3) for :func:`chosen_nll_loss` (the SFT regulariser ``-chosen_logp``).

Imports target concrete implementation paths because the public façade
`__init__.py` is wired in the separate γ.W wire-up unit.
"""

from __future__ import annotations

import torch

from opake.api.alignment.dpo.loss._chosen_nll import chosen_nll_loss

# ---------------------------------------------------------------------------
# chosen_nll_loss
# ---------------------------------------------------------------------------


class TestDpoSft:
    """Tests for the SFT regulariser: -chosen_logp."""

    def test_sft_nll_scalar(self) -> None:
        """Returns -chosen_logp for a 0-dim tensor."""
        chosen_logp = torch.tensor(-3.5)
        out = chosen_nll_loss(chosen_logp)
        assert out.shape == ()
        assert torch.allclose(out, torch.tensor(3.5), atol=1e-7)

    def test_sft_nll_batched(self) -> None:
        """Returns -chosen_logp element-wise for a (B,) tensor."""
        chosen_logp = torch.tensor([-1.0, -2.5, -0.3])
        out = chosen_nll_loss(chosen_logp)
        assert out.shape == (3,)
        assert torch.allclose(out, -chosen_logp, atol=1e-7)

    def test_sft_ignores_extra_positional_args(self) -> None:
        """Extra positional arguments (e.g. rejected_logratio) are silently ignored."""
        chosen_logp = torch.tensor(-2.0)
        rejected_logratio = torch.tensor(99.0)  # should be ignored
        out = chosen_nll_loss(chosen_logp, rejected_logratio)
        assert torch.allclose(out, torch.tensor(2.0), atol=1e-7)

    def test_sft_ignores_extra_keyword_args(self) -> None:
        """Extra keyword arguments (beta, label_smoothing, …) are silently ignored."""
        chosen_logp = torch.tensor(-4.0)
        out = chosen_nll_loss(
            chosen_logp, beta=0.1, label_smoothing=0.05, discopop_tau=0.05
        )
        assert torch.allclose(out, torch.tensor(4.0), atol=1e-7)

    def test_sft_ignores_both_extra_args(self) -> None:
        """Can be called with full (chosen_logp, rejected, *, beta, ...) signature."""
        chosen_logp = torch.tensor(-1.5)
        rejected = torch.tensor(0.5)
        out = chosen_nll_loss(chosen_logp, rejected, beta=0.2, label_smoothing=0.0)
        assert torch.allclose(out, torch.tensor(1.5), atol=1e-7)

    def test_sft_nan_propagates_only_to_nan_example(self) -> None:
        """NaN in chosen_logp propagates only to that output index."""
        chosen_logp = torch.tensor([-1.0, float("nan"), -2.0, -0.5])
        out = chosen_nll_loss(chosen_logp)
        assert torch.isnan(out[1])
        assert torch.isfinite(out[0])
        assert torch.isfinite(out[2])
        assert torch.isfinite(out[3])
