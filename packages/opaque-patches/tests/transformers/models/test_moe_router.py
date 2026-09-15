# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""MoE router helpers: geometry, fp32 router, loss-only forward with router logits.

Tiny random-init Mellum 2.0 models (E = 8, k = 2, L = 2) exercise
``moe_geometry``, the opt-in fp32 router and the loss-only causal-LM forward's
``output_router_logits`` contract on every cross-entropy route.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")

from opaque.api.engine.clipping import clipped_grad
from opaque.api.patches.transformers.components.cross_entropy import (
    FUSED_LINEAR_CE_ATTR,
)
from opaque.api.patches.transformers.components.router import (
    has_fp32_router,
    install_fp32_router,
    remove_fp32_router,
)
from opaque.exceptions import ConfigurationError
from opaque.functional import make_functional
from opaque.patches import apply_model_patches
from opaque.patches.transformers import moe_geometry

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _test_utils import build_moe_model

E, K, L = 8, 2, 2
VOCAB = 128


def _tiny_mellum(device, **overrides):
    torch.manual_seed(0)
    config = {
        "num_experts": E,
        "num_experts_per_tok": K,
        "moe_intermediate_size": 32,
        "num_hidden_layers": L,
    }
    config.update(overrides)
    model, mod = build_moe_model("mellum", device, **config)
    model.train()
    return model, mod


@contextlib.contextmanager
def _restored_class_forwards(model):
    """Undo class-level ``forward`` replacements made while the block runs."""
    saved = {}
    for module in model.modules():
        cls = type(module)
        if cls not in saved:
            saved[cls] = cls.__dict__.get("forward")
    try:
        yield
    finally:
        for cls, forward in saved.items():
            if forward is None:
                if "forward" in cls.__dict__:
                    delattr(cls, "forward")
            else:
                cls.forward = forward


def _ragged_batch(device, lengths, seq_len):
    torch.manual_seed(1)
    batch = len(lengths)
    input_ids = torch.randint(3, VOCAB, (batch, seq_len), device=device)
    positions = torch.arange(seq_len, device=device)[None, :]
    mask = (positions < torch.as_tensor(lengths, device=device)[:, None]).long()
    input_ids = torch.where(mask.bool(), input_ids, torch.zeros_like(input_ids))
    labels = torch.where(mask.bool(), input_ids, torch.full_like(input_ids, -100))
    return input_ids, mask, labels


def _router_modules(model, mod):
    return [m for m in model.modules() if isinstance(m, mod.MellumTopKRouter)]


def _per_example_forward(model):
    fmodel, trainable, frozen = make_functional(
        model, disable_autograd_tracking=True, partition_trainable=True
    )

    def forward(tr, ids, m, lbl, **kw):
        return fmodel(
            {**frozen, **tr}, input_ids=ids, attention_mask=m, labels=lbl, **kw
        )

    return forward, trainable


# ---------------------------------------------------------------------------
# moe_geometry
# ---------------------------------------------------------------------------


class TestMoeGeometry:
    def test_reads_routers_and_unpacks(self, device):
        model, _ = _tiny_mellum(device)
        geometry = moe_geometry(model)
        assert geometry == {"top_k": K, "num_experts": E, "num_layers": L}

        def consumer(*, top_k, num_experts, num_layers):
            return top_k, num_experts, num_layers

        assert consumer(**geometry) == (K, E, L)

    def test_counts_only_routed_layers(self, device):
        model, _ = _tiny_mellum(device, num_hidden_layers=3)
        assert moe_geometry(model)["num_layers"] == 3

    def test_dense_model_is_rejected(self):
        dense = torch.nn.Sequential(torch.nn.Linear(4, 4))
        with pytest.raises(ConfigurationError, match="no top-k router"):
            moe_geometry(dense)

    def test_disagreeing_routers_are_rejected(self, device):
        model, mod = _tiny_mellum(device)
        _router_modules(model, mod)[0].top_k = K + 1
        with pytest.raises(ConfigurationError, match="disagree"):
            moe_geometry(model)


# ---------------------------------------------------------------------------
# fp32 router
# ---------------------------------------------------------------------------


class TestFp32Router:
    def test_dtypes_and_indices_match_unpatched_router(self, device):
        model, mod = _tiny_mellum(device)
        router = _router_modules(model, mod)[0]
        torch.manual_seed(2)
        hidden = torch.randn(9, model.config.hidden_size, device=device)
        with torch.no_grad():
            ref_logits, ref_scores, ref_indices = router(hidden)
        undo = install_fp32_router(model)
        assert has_fp32_router(model)
        with torch.no_grad():
            logits, scores, indices = router(hidden)
        assert logits.dtype == torch.float32
        assert scores.dtype == hidden.dtype
        assert torch.equal(indices, ref_indices)
        torch.testing.assert_close(logits, ref_logits)
        torch.testing.assert_close(scores, ref_scores)

        undo()
        assert not has_fp32_router(model)
        with torch.no_grad():
            back_logits, _, back_indices = router(hidden)
        assert torch.equal(back_logits, ref_logits)
        assert torch.equal(back_indices, ref_indices)
        # The class-level forward was never touched.
        assert "forward" not in router.__dict__
        assert mod.MellumTopKRouter.forward is type(router).forward

    def test_bf16_hidden_states_get_fp32_logits(self, device):
        model, mod = _tiny_mellum(device)
        model.to(torch.bfloat16)
        router = _router_modules(model, mod)[0]
        hidden = torch.randn(
            9, model.config.hidden_size, device=device, dtype=torch.bfloat16
        )
        with torch.no_grad():
            stock_logits, stock_scores, _ = router(hidden)
        assert stock_logits.dtype == torch.bfloat16
        install_fp32_router(model)
        with torch.no_grad():
            logits, scores, indices = router(hidden)
        assert logits.dtype == torch.float32
        assert scores.dtype == torch.bfloat16 == stock_scores.dtype
        assert indices.shape == (9, K)
        torch.testing.assert_close(
            scores.float().sum(-1), torch.ones(9, device=device), atol=2e-2, rtol=0
        )
        remove_fp32_router(model)

    def test_installed_through_apply_model_patches_and_removable(self, device):
        model, mod = _tiny_mellum(device)
        apply_model_patches(model, router_fp32=True)
        assert has_fp32_router(model)
        assert all("forward" in r.__dict__ for r in _router_modules(model, mod))
        # Idempotent: a second install keeps a single removable binding.
        apply_model_patches(model, router_fp32=True)
        apply_model_patches(model, router_fp32=False)
        assert not has_fp32_router(model)
        assert all("forward" not in r.__dict__ for r in _router_modules(model, mod))
        # Default: the router is left alone.
        other, _ = _tiny_mellum(device)
        assert not has_fp32_router(other)

    def test_model_without_a_matching_router_is_rejected(self, device):
        """A requested fp32 router never installs nothing silently."""
        dense = torch.nn.Sequential(torch.nn.Linear(4, 4)).to(device)
        with pytest.raises(ConfigurationError, match="nothing to install"):
            install_fp32_router(dense)
        model, _ = _tiny_mellum(device)

        class NotTheRouter(torch.nn.Module):
            pass

        with pytest.raises(ConfigurationError, match="NotTheRouter"):
            install_fp32_router(model, router_cls=NotTheRouter)
        assert not has_fp32_router(model)

    def test_vmap_and_per_example_gradients(self, device):
        model, _ = _tiny_mellum(device)
        install_fp32_router(model)
        input_ids, mask, labels = _ragged_batch(device, [12, 12, 9, 12], 12)

        fmodel, trainable, frozen = make_functional(
            model, disable_autograd_tracking=True, partition_trainable=True
        )

        def per_example_loss(tr, fr, ids, m, lbl):
            return fmodel({**fr, **tr}, ids, attention_mask=m, labels=lbl).loss

        with torch.no_grad():
            losses = torch.vmap(per_example_loss, in_dims=(None, None, 0, 0, 0))(
                trainable, frozen, input_ids, mask, labels
            )
        assert losses.shape == (4,)
        assert torch.isfinite(losses).all()

        grad_fn, state = clipped_grad(
            per_example_loss, argnums=0, batch_argnums=(2, 3, 4), clipping_norm=1.0
        )
        grads, _ = grad_fn(trainable, frozen, input_ids, mask, labels, state=state)
        router_grads = [g for n, g in grads.pytree.items() if ".mlp.gate." in n]
        assert router_grads
        assert all(torch.isfinite(g).all() for g in router_grads)
        assert any(g.abs().sum() > 0 for g in router_grads)


# ---------------------------------------------------------------------------
# Loss-only forward: router logits without the batch-coupled auxiliary loss
# ---------------------------------------------------------------------------


class TestLossOnlyForward:
    def test_router_logits_under_vmap_keep_chunked_loss(self, device):
        model, _ = _tiny_mellum(device)
        seq_len = 12
        input_ids, mask, labels = _ragged_batch(device, [seq_len, 8, seq_len], seq_len)
        with _restored_class_forwards(model):
            apply_model_patches(model)
            forward, trainable = _per_example_forward(model)

            def with_router(tr, ids, m, lbl):
                out = forward(
                    tr, ids, m, lbl, loss_only=True, output_router_logits=True
                )
                assert out.logits is None
                assert out.aux_loss is None
                assert len(out.router_logits) == L
                stacked = torch.stack([z.reshape(-1, E) for z in out.router_logits])
                return out.loss, stacked

            def loss_only(tr, ids, m, lbl):
                return forward(tr, ids, m, lbl, loss_only=True).loss

            def default(tr, ids, m, lbl):
                out = forward(tr, ids, m, lbl)
                assert out.logits is not None
                return out.loss

            with torch.no_grad():
                loss_r, logits = torch.vmap(with_router, in_dims=(None, 0, 0, 0))(
                    trainable, input_ids, mask, labels
                )
                loss_c = torch.vmap(loss_only, in_dims=(None, 0, 0, 0))(
                    trainable, input_ids, mask, labels
                )
                loss_d = torch.vmap(default, in_dims=(None, 0, 0, 0))(
                    trainable, input_ids, mask, labels
                )
            assert logits.shape == (3, L, seq_len, E)
            assert torch.equal(loss_r, loss_c)
            torch.testing.assert_close(loss_r, loss_d, atol=1e-5, rtol=1e-5)

    def test_without_loss_only_hf_contract_is_kept(self, device):
        model, _ = _tiny_mellum(device)
        input_ids, mask, labels = _ragged_batch(device, [12, 8], 12)
        with _restored_class_forwards(model):
            apply_model_patches(model)
            with torch.no_grad():
                hf = model(
                    input_ids=input_ids,
                    attention_mask=mask,
                    labels=labels,
                    output_router_logits=True,
                )
                ours = model(
                    input_ids=input_ids,
                    attention_mask=mask,
                    labels=labels,
                    loss_only=True,
                    output_router_logits=True,
                )
            assert hf.aux_loss is not None
            assert hf.logits is not None
            assert ours.aux_loss is None
            assert ours.logits is None
            assert len(ours.router_logits) == L
            # HF adds ``coef * aux`` to the loss; ours is the plain CE.
            expected_hf = ours.loss + model.config.router_aux_loss_coef * hf.aux_loss
            torch.testing.assert_close(hf.loss, expected_hf, atol=1e-5, rtol=1e-5)

    def test_config_default_requests_router_logits_in_loss_only(self, device):
        model, _ = _tiny_mellum(device)
        input_ids, mask, labels = _ragged_batch(device, [12, 8], 12)
        with _restored_class_forwards(model):
            apply_model_patches(model)
            model.config.output_router_logits = True
            with torch.no_grad():
                out = model(
                    input_ids=input_ids,
                    attention_mask=mask,
                    labels=labels,
                    loss_only=True,
                )
            assert out.aux_loss is None
            assert out.logits is None
            assert len(out.router_logits) == L

    def test_explicit_false_overrides_the_config_default(self, device):
        model, _ = _tiny_mellum(device)
        input_ids, mask, labels = _ragged_batch(device, [12, 8], 12)
        with _restored_class_forwards(model):
            apply_model_patches(model)
            model.config.output_router_logits = True
            with torch.no_grad():
                out = model(
                    input_ids=input_ids,
                    attention_mask=mask,
                    labels=labels,
                    loss_only=True,
                    output_router_logits=False,
                )
            assert out.logits is None
            assert out.router_logits is None

    def test_eager_route_with_performance_off(self, device):
        """The wrapper is a compat patch: installed with the fused routes off."""
        model, _ = _tiny_mellum(device)
        input_ids, mask, labels = _ragged_batch(device, [12, 8], 12)
        with _restored_class_forwards(model):
            apply_model_patches(model, performance=False)
            assert getattr(model, FUSED_LINEAR_CE_ATTR) is False
            with torch.no_grad():
                out = model(
                    input_ids=input_ids,
                    attention_mask=mask,
                    labels=labels,
                    loss_only=True,
                    output_router_logits=True,
                )
                ref = model(input_ids=input_ids, attention_mask=mask, labels=labels)
            # Eager route: the full logits come back, the loss is HF's own
            # cross-entropy, and no auxiliary term is added.
            assert out.logits is not None
            assert out.aux_loss is None
            assert len(out.router_logits) == L
            torch.testing.assert_close(out.loss, ref.loss)
            torch.testing.assert_close(out.logits, ref.logits)

    def test_route_policy_is_per_instance(self, device):
        """A later model is not bound to the first install's route policy."""
        first, _ = _tiny_mellum(device)
        second, _ = _tiny_mellum(device)
        input_ids, mask, labels = _ragged_batch(device, [12, 8], 12)
        with _restored_class_forwards(first):
            apply_model_patches(first, performance=False)
            apply_model_patches(second)
            with torch.no_grad():
                eager = first(
                    input_ids=input_ids,
                    attention_mask=mask,
                    labels=labels,
                    loss_only=True,
                )
                chunked = second(
                    input_ids=input_ids,
                    attention_mask=mask,
                    labels=labels,
                    loss_only=True,
                )
            assert eager.logits is not None
            assert chunked.logits is None
            torch.testing.assert_close(eager.loss, chunked.loss, atol=1e-5, rtol=1e-5)
