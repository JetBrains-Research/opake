# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Attention-mask isolation for vmapped per-example gradients."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("transformers")
import torch

from opake.functional import make_functional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from _test_utils import build_moe_model


def _per_example_grads(model, input_ids, mask, labels):
    fmodel, trainable, frozen = make_functional(
        model, disable_autograd_tracking=True, partition_trainable=True
    )

    def loss_fn(tr, ids, m, lbl):
        return fmodel(
            {**frozen, **tr}, input_ids=ids, attention_mask=m, labels=lbl
        ).loss

    grad_fn = torch.vmap(torch.func.grad(loss_fn), in_dims=(None, 0, 0, 0))
    return grad_fn(trainable, input_ids, mask, labels)


def test_all_valid_example_gradient_is_invariant_to_a_padded_mate(device):
    """A row's gradient is independent of padding in another vmapped row."""
    torch.manual_seed(0)
    model, _ = build_moe_model(
        "mellum",
        device,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        num_hidden_layers=1,
    )
    model.train()
    seq_len = 10
    torch.manual_seed(1)
    input_ids = torch.randint(3, 128, (2, seq_len), device=device)
    labels = input_ids.clone()
    full = torch.ones(2, seq_len, dtype=torch.long, device=device)
    padded = full.clone()
    padded[1, -4:] = 0
    padded_labels = torch.where(padded.bool(), labels, torch.full_like(labels, -100))

    grads_full = _per_example_grads(model, input_ids, full, labels)
    grads_mixed = _per_example_grads(model, input_ids, padded, padded_labels)
    for name, g in grads_full.items():
        torch.testing.assert_close(
            grads_mixed[name][0], g[0], atol=1e-6, rtol=1e-5, msg=name
        )

    # Sanity: the padded mate's own gradient differs, so the comparison is live.
    assert any(
        not torch.allclose(grads_mixed[name][1], g[1]) for name, g in grads_full.items()
    )
