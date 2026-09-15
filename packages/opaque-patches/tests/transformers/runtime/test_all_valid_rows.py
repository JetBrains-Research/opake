# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""All-valid-rows policy for the vmap-safe causal-mask builder."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("transformers")
import torch

from opaque.api.patches.transformers.runtime.masking import vmap_create_causal_mask
from opaque.functional import make_functional
from opaque.patches import all_valid_rows, set_all_valid_rows

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from _test_utils import build_moe_model


@pytest.fixture(autouse=True)
def _reset_policy():
    set_all_valid_rows(None)
    yield
    set_all_valid_rows(None)


class _SdpaConfig:
    _attn_implementation = "sdpa"


def _masks_under_vmap(attention_masks):
    input_embeds = torch.randn(1, 8, 16)
    created = []
    parameter = torch.ones(1, requires_grad=True)

    def create_mask(param, attention_mask):
        created.append(
            vmap_create_causal_mask(
                config=_SdpaConfig(),
                input_embeds=input_embeds,
                attention_mask=attention_mask,
                cache_position=torch.arange(8),
                past_key_values=None,
            )
        )
        return param.square().sum() + attention_mask.sum() * 0

    torch.vmap(torch.func.grad(create_mask), in_dims=(None, 0))(
        parameter, attention_masks
    )
    return created


def test_policy_round_trip_and_validation():
    assert all_valid_rows() is None
    set_all_valid_rows(True)
    assert all_valid_rows() is True
    set_all_valid_rows(False)
    assert all_valid_rows() is False
    with pytest.raises(TypeError):
        set_all_valid_rows(1)


def test_all_valid_true_allows_fast_path_without_probing():
    padded = torch.ones(3, 8, dtype=torch.bool)
    padded[1, -1] = False
    set_all_valid_rows(True)
    assert _masks_under_vmap(padded) == [None]
    # Plain (non-vmap) call with a padded 2-D mask is not probed either.
    mask = vmap_create_causal_mask(
        config=_SdpaConfig(),
        input_embeds=torch.randn(2, 8, 16),
        attention_mask=padded[:2].long(),
        cache_position=torch.arange(8),
        past_key_values=None,
    )
    assert mask is None


def test_all_valid_true_keeps_a_higher_rank_mask():
    """The policy only replaces the padding-mask probe; a 4-D mask is kept."""
    set_all_valid_rows(True)
    explicit = torch.zeros(2, 1, 8, 8, dtype=torch.bool).tril()
    mask = vmap_create_causal_mask(
        config=_SdpaConfig(),
        input_embeds=torch.randn(2, 8, 16),
        attention_mask=explicit,
        cache_position=torch.arange(8),
        past_key_values=None,
    )
    assert mask is not None


def test_all_valid_false_materialises_regardless_of_content():
    all_valid = torch.ones(3, 8, dtype=torch.bool)
    set_all_valid_rows(False)
    created = _masks_under_vmap(all_valid)
    assert len(created) == 1
    assert created[0] is not None
    assert created[0].shape[-2:] == (8, 8)
    # ``None`` attention masks keep the fast path: nothing to materialise.
    mask = vmap_create_causal_mask(
        config=_SdpaConfig(),
        input_embeds=torch.randn(2, 8, 16),
        attention_mask=None,
        cache_position=torch.arange(8),
        past_key_values=None,
    )
    assert mask is None


def test_policy_none_keeps_batch_probe():
    all_valid = torch.ones(3, 8, dtype=torch.bool)
    padded = all_valid.clone()
    padded[2, -2:] = False
    assert _masks_under_vmap(all_valid) == [None]
    created = _masks_under_vmap(padded)
    assert len(created) == 1
    assert created[0] is not None


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


def test_all_valid_example_gradient_invariant_to_padded_mate(device):
    """Under ``all_valid_rows=False`` a row's gradient ignores its microbatch."""
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

    set_all_valid_rows(False)
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

    # And the fast path (all_valid_rows=True) computes the same per-example gradients
    # for all-valid rows, only through a different kernel.
    set_all_valid_rows(True)
    grads_fast = _per_example_grads(model, input_ids, full, labels)
    for name, g in grads_full.items():
        torch.testing.assert_close(grads_fast[name], g, atol=1e-5, rtol=1e-4, msg=name)


def test_forward_matches_between_policies_on_all_valid_rows(device):
    torch.manual_seed(0)
    model, _ = build_moe_model(
        "mellum",
        device,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        num_hidden_layers=1,
    )
    model.eval()
    input_ids = torch.randint(3, 128, (2, 10), device=device)
    mask = torch.ones_like(input_ids)
    outputs = {}
    for policy in (None, True, False):
        set_all_valid_rows(policy)
        with torch.no_grad():
            outputs[policy] = model(input_ids=input_ids, attention_mask=mask).logits
    torch.testing.assert_close(outputs[True], outputs[None], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(outputs[False], outputs[None], atol=1e-5, rtol=1e-4)
