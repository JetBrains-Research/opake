# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for gradient checkpointing under vmap(grad(...)).

Verifies that checkpoint patches are applied and produce correct gradients
when combined with clipped_grad on HuggingFace models.
"""

import pytest
import torch
import torch.nn.functional as F
from torch.func import grad, vmap
from torch.utils.checkpoint import checkpoint

from opake.api.patches.torch.checkpoint import is_checkpoint_patched
from opake.patches import apply_runtime_patches

apply_runtime_patches(vmap_checkpointing=True)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Checkpoint patch tests require CUDA",
    ),
]


def _peak_memory(fn) -> int:
    """Peak CUDA allocation of a second, warmed-up call to ``fn``."""
    import gc

    fn()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    return torch.cuda.max_memory_allocated()


def _assert_savings(mem_plain: int, mem_ckpt: int) -> None:
    savings = (mem_plain - mem_ckpt) / mem_plain
    assert savings > 0.15, (
        f"Expected >15% memory savings from checkpoint, got {savings:.1%} "
        f"(plain={mem_plain / 1e6:.0f}MB, ckpt={mem_ckpt / 1e6:.0f}MB)"
    )


class TestCheckpointPatches:
    """Verify checkpoint patches are applied and functional."""

    def test_patches_applied(self):
        assert is_checkpoint_patched()

    def test_vmap_grad_with_checkpoint(self, device):
        """vmap(grad(f)) works when f uses checkpoint internally."""
        W = torch.randn(32, 32, device=device)

        def f_ckpt(x):
            h = checkpoint(lambda x: F.gelu(x @ W), x, use_reentrant=False)
            h = checkpoint(lambda x: F.gelu(x @ W), h, use_reentrant=False)
            return h.sum()

        def f_ref(x):
            return F.gelu(F.gelu(x @ W) @ W).sum()

        x = torch.randn(4, 32, device=device)
        g = vmap(grad(f_ckpt))(x)
        g_ref = vmap(grad(f_ref))(x)
        torch.testing.assert_close(g, g_ref, rtol=1e-4, atol=1e-5)

    def test_functional_call_param_context_protocol(self, device):
        """functional_call + checkpoint: recomputation sees the correct params.

        Verifies that checkpoint recomputation reads the functional_call
        parameters, which the patch re-binds for the backward recompute.
        """
        model = torch.nn.Linear(32, 32, bias=False).to(device)

        new_weight = torch.randn(32, 32, device=device)
        params = {"weight": new_weight}

        def f(x):
            h = checkpoint(
                lambda x: F.gelu(torch.func.functional_call(model, params, (x,))),
                x,
                use_reentrant=False,
            )
            return h.sum()

        x = torch.randn(4, 32, device=device)
        g = vmap(grad(f))(x)

        # Reference: no checkpoint
        def f_ref(x):
            return F.gelu(torch.func.functional_call(model, params, (x,))).sum()

        g_ref = vmap(grad(f_ref))(x)
        torch.testing.assert_close(g, g_ref, rtol=1e-4, atol=1e-5)

    def test_checkpoint_saves_memory(self, device):
        """Checkpoint with patches actually reduces peak GPU memory."""
        if device.type != "cuda":
            pytest.skip("Memory measurement requires CUDA")

        d, n_layers = 512, 8
        Ws = [torch.randn(d, d, device=device) for _ in range(n_layers)]

        def f_plain(x):
            for W in Ws:
                x = F.gelu(x @ W)
            return x.sum()

        def f_ckpt(x):
            for s in range(0, n_layers, 4):

                def block(x, s=s):
                    for j in range(s, min(s + 4, n_layers)):
                        x = F.gelu(x @ Ws[j])
                    return x

                x = checkpoint(block, x, use_reentrant=False)
            return x.sum()

        B, N = 4, 1024
        with torch.no_grad():
            mem_plain = _peak_memory(
                lambda: vmap(grad(f_plain))(torch.randn(B, N, d, device=device))
            )
            mem_ckpt = _peak_memory(
                lambda: vmap(grad(f_ckpt))(torch.randn(B, N, d, device=device))
            )

        _assert_savings(mem_plain, mem_ckpt)

    def test_clipped_grad_saves_memory(self, device):
        """clipped_grad preserves checkpoint memory savings."""
        if device.type != "cuda":
            pytest.skip("Memory measurement requires CUDA")

        from opake.api.engine.clipping import clipped_grad

        d, n_layers = 512, 8
        params = {
            "weights": [torch.randn(d, d, device=device) for _ in range(n_layers)]
        }

        def loss_plain(p, x):
            for weight in p["weights"]:
                x = F.gelu(x @ weight)
            return x.sum()

        def loss_ckpt(p, x):
            for s in range(0, n_layers, 4):

                def block(x, s=s):
                    for j in range(s, min(s + 4, n_layers)):
                        x = F.gelu(x @ p["weights"][j])
                    return x

                x = checkpoint(block, x, use_reentrant=False)
            return x.sum()

        grad_plain, state_plain = clipped_grad(loss_plain, clipping_norm=1.0)
        grad_ckpt, state_ckpt = clipped_grad(loss_ckpt, clipping_norm=1.0)

        B, N = 4, 1024
        mem_plain = _peak_memory(
            lambda: grad_plain(
                params, torch.randn(B, N, d, device=device), state=state_plain
            )
        )
        mem_ckpt = _peak_memory(
            lambda: grad_ckpt(
                params, torch.randn(B, N, d, device=device), state=state_ckpt
            )
        )

        _assert_savings(mem_plain, mem_ckpt)


@pytest.mark.cuda
class TestCheckpointWithClippedGrad:
    """Test checkpoint with the full clipped_grad pipeline."""

    @pytest.mark.slow
    def test_use_reentrant_override(self, device):
        """Patch overrides use_reentrant=True to False."""
        transformers = pytest.importorskip("transformers")

        config = transformers.AutoConfig.from_pretrained("Qwen/Qwen2-0.5B")
        config.num_hidden_layers = 1
        model = transformers.AutoModelForCausalLM.from_config(config).to(device)

        # Explicitly pass use_reentrant=True — patch should override.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": True}
        )

        # Check that the stored func actually uses False.
        import functools

        for module in model.modules():
            if hasattr(module, "_gradient_checkpointing_func"):
                func = module._gradient_checkpointing_func
                if isinstance(func, functools.partial):
                    assert func.keywords.get("use_reentrant") is False, (
                        "use_reentrant was not overridden to False"
                    )

    @pytest.mark.slow
    def test_clipped_grad_with_checkpoint_model(self, device):
        """clipped_grad works on a HF model with gradient_checkpointing_enable()."""
        transformers = pytest.importorskip("transformers")
        peft = pytest.importorskip("peft")

        from opake.api.engine.clipping import clipped_grad
        from opake.functional import make_functional
        from opake.patches import apply_model_patches

        config = transformers.AutoConfig.from_pretrained("Qwen/Qwen2-0.5B")
        config.num_hidden_layers = 2

        model = transformers.AutoModelForCausalLM.from_config(config)
        # Model-level patches (vmap-safe RoPE, masking, etc.) must be applied
        # to the base model BEFORE wrapping with PEFT.
        apply_model_patches(model)
        model = peft.get_peft_model(
            model,
            peft.LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.0,
            ),
        ).to(device)
        # No explicit use_reentrant=False needed — our patch forces it.
        model.gradient_checkpointing_enable()

        tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        texts = ["Hello world", "Test input", "Third sample", "Fourth one"]
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, max_length=16, truncation=True
        )
        input_ids = inputs["input_ids"].to(device)
        labels = input_ids.clone()

        fmodel, trainable, frozen = make_functional(
            model, disable_autograd_tracking=True, partition_trainable=True
        )

        def loss_fn(trainable_params, frozen_params, ids, lbls):
            all_params = {**frozen_params, **trainable_params}
            return fmodel(all_params, ids, labels=lbls).loss

        grad_fn, clip_state = clipped_grad(
            loss_fn, argnums=0, batch_argnums=(2, 3), clipping_norm=1.0
        )
        grads, _ = grad_fn(trainable, frozen, input_ids, labels, state=clip_state)

        assert len(grads.pytree) > 0
        assert all(v.shape == trainable[k].shape for k, v in grads.pytree.items())
        assert any(v.abs().max() > 0 for v in grads.pytree.values()), (
            "All gradients are zero"
        )

    @pytest.mark.slow
    def test_checkpoint_gradient_correctness(self, device):
        """Checkpointed gradients match non-checkpointed for same model."""
        transformers = pytest.importorskip("transformers")
        peft = pytest.importorskip("peft")

        from opake.api.engine.clipping import clipped_grad
        from opake.functional import make_functional
        from opake.patches import apply_model_patches

        config = transformers.AutoConfig.from_pretrained("Qwen/Qwen2-0.5B")
        config.num_hidden_layers = 2

        tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        texts = ["Hello world", "Test input"]
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, max_length=16, truncation=True
        )
        input_ids = inputs["input_ids"].to(device)
        labels = input_ids.clone()

        lora_config = peft.LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
        )

        # Without checkpoint
        torch.manual_seed(42)
        model_a = transformers.AutoModelForCausalLM.from_config(config)
        apply_model_patches(model_a)
        model_a = peft.get_peft_model(model_a, lora_config).to(device)
        fmodel_a, train_a, frozen_a = make_functional(
            model_a, disable_autograd_tracking=True, partition_trainable=True
        )

        def loss_a(tp, fp, ids, lbls):
            return fmodel_a({**fp, **tp}, ids, labels=lbls).loss

        grad_fn_a, state_a = clipped_grad(
            loss_a, argnums=0, batch_argnums=(2, 3), clipping_norm=float("inf")
        )
        grads_a, _ = grad_fn_a(train_a, frozen_a, input_ids, labels, state=state_a)

        # With checkpoint (same weights)
        torch.manual_seed(42)
        model_b = transformers.AutoModelForCausalLM.from_config(config)
        apply_model_patches(model_b)
        model_b = peft.get_peft_model(model_b, lora_config).to(device)
        # Bare call — patch forces use_reentrant=False automatically.
        model_b.gradient_checkpointing_enable()
        fmodel_b, train_b, frozen_b = make_functional(
            model_b, disable_autograd_tracking=True, partition_trainable=True
        )

        def loss_b(tp, fp, ids, lbls):
            return fmodel_b({**fp, **tp}, ids, labels=lbls).loss

        grad_fn_b, state_b = clipped_grad(
            loss_b, argnums=0, batch_argnums=(2, 3), clipping_norm=float("inf")
        )
        grads_b, _ = grad_fn_b(train_b, frozen_b, input_ids, labels, state=state_b)

        for k in grads_a.pytree:
            # Checkpoint recomputation under vmap can cause small numerical
            # differences (~0.5%) due to different execution contexts.
            torch.testing.assert_close(
                grads_a.pytree[k],
                grads_b.pytree[k],
                rtol=1e-2,
                atol=5e-3,
                msg=f"Gradient mismatch for {k}",
            )
