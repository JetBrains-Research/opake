# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Condition a transform's internal backward graph on its entry grad mode."""

from __future__ import annotations

from opake.api.patches.torch.functorch_transform import prev_grad_mode


def apply() -> None:
    """Install the grad-mode-conditioned first-order backward."""
    import torch._functorch.eager_transforms as eager

    orig = eager._autograd_grad

    # grad_and_value_impl's internal backward has no grad_outputs.
    def _autograd_grad(
        outputs, inputs, grad_outputs=None, retain_graph=False, create_graph=True
    ):
        if create_graph and grad_outputs is None:
            create_graph = prev_grad_mode()
        return orig(
            outputs,
            inputs,
            grad_outputs,
            retain_graph=retain_graph,
            create_graph=create_graph,
        )

    eager._autograd_grad = _autograd_grad
