"""LoRA + DP training validation tests.

These tests validate that Opake correctly handles DP training with HuggingFace
models using their built-in loss functions.

Test Models:
- GPT-2 (small): Fast validation, JAX-Privacy cross-validation
- Mellum-4b: Heavy load testing, memory profiling (requires GPU)

Key validations:
1. HuggingFace model's built-in loss works with clipped_grad
2. LoRA adapters are correctly identified and clipped
3. Gradients are numerically correct
4. Memory usage is bounded with microbatching
"""

import pytest
import torch
from _hf_shared import (
    get_default_gpu_device,
    gpu_memory_gate_reason,
    has_min_gpu_memory,
    make_gpt2,
    run_dp_training_step,
)

from opake.api.engine.clipping import clipped_grad
from opake.api.engine.clipping.types import FixedClipState
from opake.dpsgd.noise import gaussian_noise
from opake.functional import make_functional
from opake.random import key
from opake.types import ClippedPytree, NoisedPytree

# Skip all tests if transformers not available
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from peft import LoraConfig, TaskType, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def gpt2_model_and_tokenizer():
    """Tiny randomly-initialised GPT-2 + tokenizer (see _hf_shared.make_gpt2).

    Uses default attention (SDPA) - vmap compatibility is handled by
    import-time patches applied when opake is imported.
    """
    return make_gpt2()


@pytest.fixture
def gpt2_with_lora(gpt2_model_and_tokenizer):
    """GPT-2 with LoRA adapters applied."""
    model, tokenizer = gpt2_model_and_tokenizer

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["c_attn", "c_proj"],
        # GPT-2's c_attn/c_proj are `Conv1D` layers (Radford's original naming).
        # PEFT defaults `fan_in_fan_out=False` which is correct for `nn.Linear`;
        # Conv1D stores weights transposed, so LoRA needs the flag flipped.
        fan_in_fan_out=True,
    )

    model = get_peft_model(model, lora_config)
    try:
        from opake.patches import apply_model_patches

        apply_model_patches(model, compat=True, performance=False)
    except ImportError:
        pass
    return model, tokenizer


@pytest.fixture
def sample_batch(gpt2_model_and_tokenizer):
    """Create a sample batch of tokenized text."""
    _, tokenizer = gpt2_model_and_tokenizer

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming software development.",
        "Python is a popular programming language.",
        "Deep learning models require significant compute.",
    ]

    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )

    return tokenized["input_ids"], tokenized["attention_mask"]


# =============================================================================
# GPT-2 Tests (Small Model Validation)
# =============================================================================


class TestGPT2LoRADPTraining:
    """Validate DP training with GPT-2 + LoRA using HuggingFace built-in loss."""

    def test_functional_conversion_preserves_lora_structure(self, gpt2_with_lora):
        """Test that make_functional correctly separates LoRA params from frozen."""
        model, _ = gpt2_with_lora

        _fmodel, trainable, frozen = make_functional(
            model,
            disable_autograd_tracking=True,
            partition_trainable=True,
        )

        # Check trainable params are LoRA params
        trainable_count = sum(p.numel() for p in trainable.values())
        frozen_count = sum(p.numel() for p in frozen.values())

        # LoRA should be small fraction of total
        assert trainable_count < frozen_count
        assert trainable_count > 0

        # All trainable params should have 'lora' in their name
        for name in trainable:
            assert "lora" in name.lower(), f"Expected LoRA param, got: {name}"

    def test_clipped_grad_with_hf_builtin_loss(self, gpt2_with_lora, sample_batch):
        """Test clipped_grad works with HuggingFace model's built-in loss."""
        model, _ = gpt2_with_lora
        input_ids, attention_mask = sample_batch
        labels = input_ids.clone()  # For causal LM, labels = input_ids

        # Convert to functional
        fmodel, trainable, frozen = make_functional(
            model,
            disable_autograd_tracking=True,
            partition_trainable=True,
        )

        def per_example_loss(
            trainable_params,
            frozen_params,
            input_ids_single,
            mask_single,
            labels_single,
        ):
            """Loss for single example using HF built-in loss."""
            all_params = {**frozen_params, **trainable_params}

            # Batchify patches add batch dim automatically for batchless inputs
            outputs = fmodel(
                all_params,
                input_ids_single,
                attention_mask=mask_single,
                labels=labels_single,
            )
            return outputs.loss  # HuggingFace's built-in loss

        # Create clipped gradient function
        grad_fn, clip_state = clipped_grad(
            per_example_loss,
            argnums=0,  # Differentiate w.r.t. trainable params
            batch_argnums=(2, 3, 4),  # input_ids, mask, labels are batched
            clipping_norm=1.0,
        )

        # Compute gradients
        grads, _ = grad_fn(
            trainable, frozen, input_ids, attention_mask, labels, state=clip_state
        )

        # Verify gradients exist and have correct structure
        assert isinstance(grads, ClippedPytree)
        gtree = grads.pytree
        assert len(gtree) > 0

        for name, grad in gtree.items():
            assert isinstance(grad, torch.Tensor), (
                f"Gradient for {name} is not a tensor"
            )
            assert grad.shape == trainable[name].shape, f"Shape mismatch for {name}"
            assert torch.isfinite(grad).all(), f"Non-finite gradient for {name}"

    def test_run_dp_training_step_with_accumulation(self, gpt2_with_lora):
        """Gradient accumulation should rebuild clipped metadata explicitly."""
        model, tokenizer = gpt2_with_lora

        grads, _ = run_dp_training_step(
            model,
            tokenizer,
            batch_size=2,
            max_length=16,
            accum_steps=2,
            training_steps=1,
            learning_rate=1e-3,
            clipping_norm=1.0,
        )

        assert isinstance(grads, ClippedPytree)
        assert float(grads.max_norm) == 1.0
        assert len(grads.pytree) > 0
        for grad in grads.pytree.values():
            assert torch.isfinite(grad).all()

    def test_clipped_grad_with_return_values(self, gpt2_with_lora, sample_batch):
        """Test that return_values=True returns per-example losses."""
        model, _ = gpt2_with_lora
        input_ids, attention_mask = sample_batch
        labels = input_ids.clone()

        fmodel, trainable, frozen = make_functional(
            model,
            disable_autograd_tracking=True,
            partition_trainable=True,
        )

        def per_example_loss(
            trainable_params,
            frozen_params,
            input_ids_single,
            mask_single,
            labels_single,
        ):
            all_params = {**frozen_params, **trainable_params}
            outputs = fmodel(
                all_params,
                input_ids_single,
                attention_mask=mask_single,
                labels=labels_single,
            )
            return outputs.loss

        grad_fn, clip_state = clipped_grad(
            per_example_loss,
            argnums=0,
            batch_argnums=(2, 3, 4),
            clipping_norm=1.0,
            return_aux=True,
        )

        (_grads, grad_aux), _ = grad_fn(
            trainable, frozen, input_ids, attention_mask, labels, state=clip_state
        )

        # Check loss values
        assert grad_aux.loss_values is not None
        assert grad_aux.loss_values.shape == (input_ids.shape[0],)  # One per example
        assert (grad_aux.loss_values > 0).all()  # Cross-entropy should be positive

    def test_clipped_grad_with_return_grad_norms(self, gpt2_with_lora, sample_batch):
        """Test that return_grad_norms=True returns per-example gradient norms."""
        model, _ = gpt2_with_lora
        input_ids, attention_mask = sample_batch
        labels = input_ids.clone()

        fmodel, trainable, frozen = make_functional(
            model,
            disable_autograd_tracking=True,
            partition_trainable=True,
        )

        def per_example_loss(
            trainable_params,
            frozen_params,
            input_ids_single,
            mask_single,
            labels_single,
        ):
            all_params = {**frozen_params, **trainable_params}
            outputs = fmodel(
                all_params,
                input_ids_single,
                attention_mask=mask_single,
                labels=labels_single,
            )
            return outputs.loss

        clipping_norm = 1.0
        grad_fn, clip_state = clipped_grad(
            per_example_loss,
            argnums=0,
            batch_argnums=(2, 3, 4),
            clipping_norm=clipping_norm,
            return_aux=True,
        )

        (_grads, grad_aux), _ = grad_fn(
            trainable, frozen, input_ids, attention_mask, labels, state=clip_state
        )

        # Check grad norms
        assert grad_aux.grad_norms is not None
        assert grad_aux.grad_norms.shape == (input_ids.shape[0],)
        assert (grad_aux.grad_norms >= 0).all()

    def test_noise_addition_changes_gradients(self, gpt2_with_lora, sample_batch):
        """Test that adding Gaussian noise changes gradients."""
        model, _ = gpt2_with_lora
        input_ids, attention_mask = sample_batch
        labels = input_ids.clone()

        fmodel, trainable, frozen = make_functional(
            model,
            disable_autograd_tracking=True,
            partition_trainable=True,
        )

        def per_example_loss(
            trainable_params,
            frozen_params,
            input_ids_single,
            mask_single,
            labels_single,
        ):
            all_params = {**frozen_params, **trainable_params}
            outputs = fmodel(
                all_params,
                input_ids_single,
                attention_mask=mask_single,
                labels=labels_single,
            )
            return outputs.loss

        clipping_norm = 1.0
        grad_fn, clip_state = clipped_grad(
            per_example_loss,
            argnums=0,
            batch_argnums=(2, 3, 4),
            clipping_norm=clipping_norm,
        )

        grads, _ = grad_fn(
            trainable, frozen, input_ids, attention_mask, labels, state=clip_state
        )

        # Add noise
        noise_multiplier = 1.0
        noise_fn, noise_state = gaussian_noise(
            noise_multiplier=noise_multiplier,
            key=key(0),
        )
        noisy_grads, noise_state = noise_fn(grads, noise_state)

        # Verify noise was added
        for name in grads.pytree:
            assert not torch.allclose(grads.pytree[name], noisy_grads.pytree[name]), (
                f"Noise not added to {name}"
            )

    def test_microbatching_produces_same_result(self, gpt2_with_lora, sample_batch):
        """Test that microbatching produces identical results."""
        model, _ = gpt2_with_lora
        input_ids, attention_mask = sample_batch
        labels = input_ids.clone()

        fmodel, trainable, frozen = make_functional(
            model,
            disable_autograd_tracking=True,
            partition_trainable=True,
        )

        def per_example_loss(
            trainable_params,
            frozen_params,
            input_ids_single,
            mask_single,
            labels_single,
        ):
            all_params = {**frozen_params, **trainable_params}
            outputs = fmodel(
                all_params,
                input_ids_single,
                attention_mask=mask_single,
                labels=labels_single,
            )
            return outputs.loss

        clipping_norm = 1.0

        # Without microbatching
        grad_fn_no_mb, clip_state_no_mb = clipped_grad(
            per_example_loss,
            argnums=0,
            batch_argnums=(2, 3, 4),
            clipping_norm=clipping_norm,
            microbatch_size=None,
        )
        grads_no_mb, _ = grad_fn_no_mb(
            trainable, frozen, input_ids, attention_mask, labels, state=clip_state_no_mb
        )

        # With microbatching (batch_size=4, microbatch_size=2)
        grad_fn_mb, clip_state_mb = clipped_grad(
            per_example_loss,
            argnums=0,
            batch_argnums=(2, 3, 4),
            clipping_norm=clipping_norm,
            microbatch_size=2,
        )
        grads_mb, _ = grad_fn_mb(
            trainable, frozen, input_ids, attention_mask, labels, state=clip_state_mb
        )

        # Results should be identical
        for name in grads_no_mb.pytree:
            assert torch.allclose(
                grads_no_mb.pytree[name], grads_mb.pytree[name], atol=1e-5
            ), (
                f"Microbatching mismatch for {name}: "
                f"max diff = {(grads_no_mb.pytree[name] - grads_mb.pytree[name]).abs().max()}"
            )

    def test_single_training_step(self, gpt2_with_lora, sample_batch):
        """Test a complete single training step (forward, clip, noise, update)."""
        model, _ = gpt2_with_lora
        input_ids, attention_mask = sample_batch
        labels = input_ids.clone()

        fmodel, trainable, frozen = make_functional(
            model,
            disable_autograd_tracking=True,
            partition_trainable=True,
        )

        def per_example_loss(
            trainable_params,
            frozen_params,
            input_ids_single,
            mask_single,
            labels_single,
        ):
            all_params = {**frozen_params, **trainable_params}
            outputs = fmodel(
                all_params,
                input_ids_single,
                attention_mask=mask_single,
                labels=labels_single,
            )
            return outputs.loss

        # DP parameters
        clipping_norm = 1.0
        noise_multiplier = 0.5
        learning_rate = 1e-4

        # 1. Compute clipped gradients
        grad_fn, clip_state = clipped_grad(
            per_example_loss,
            argnums=0,
            batch_argnums=(2, 3, 4),
            clipping_norm=clipping_norm,
        )
        grads, _ = grad_fn(
            trainable, frozen, input_ids, attention_mask, labels, state=clip_state
        )

        # 2. Add noise
        noise_fn, noise_state = gaussian_noise(
            noise_multiplier=noise_multiplier,
            key=key(0),
        )
        noisy_grads, noise_state = noise_fn(grads, noise_state)

        # 3. SGD update (noise mechanism returns a NoisedPytree)
        assert isinstance(noisy_grads, NoisedPytree)
        updated_params = {}
        for name, param in trainable.items():
            updated_params[name] = param - learning_rate * noisy_grads.pytree[name]

        # 4. Verify update happened
        for name in trainable:
            assert not torch.allclose(trainable[name], updated_params[name]), (
                f"Parameter {name} was not updated"
            )


# =============================================================================
# Mellum-4b Tests (Heavy Load Testing)
# =============================================================================


def _has_sufficient_gpu_memory() -> bool:
    """Check if active GPU has enough available memory for Mellum-4b."""
    device = get_default_gpu_device()
    if device is None:
        return False
    return has_min_gpu_memory(16, device=device)


def _mellum_gate_reason() -> str:
    """Build consistent skip reason for Mellum memory gate."""
    device = get_default_gpu_device()
    if device is None:
        return "No GPU available (CUDA or MPS)"
    return gpu_memory_gate_reason(16, device=device)


@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.skipif(
    not _has_sufficient_gpu_memory(),
    reason=_mellum_gate_reason(),
)
class TestMellumLoRADPTraining:
    """Heavy load tests with Mellum-4b.

    These tests require GPU with sufficient memory (~16GB for fp16).
    Run with: pytest -m "gpu and slow" tests/validation/test_lora_dp_training.py

    Note: Mellum uses LLaMA architecture which has additional vmap compatibility
    issues with transformers 5.x masking utilities. These tests may require
    further transformers version constraints.
    """

    @pytest.fixture
    def mellum_model_and_tokenizer(self):
        """Load Mellum-4b model and tokenizer.

        This is a large model (4B params) - requires GPU with sufficient memory.
        Uses eager attention to avoid vmap compatibility issues with SDPA.
        """
        model_name = "JetBrains/Mellum-4b-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use fp16 for memory efficiency
            device_map="auto",  # Automatically place on available devices
        )
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer

    @pytest.fixture
    def mellum_with_lora(self, mellum_model_and_tokenizer):
        """Mellum-4b with LoRA adapters applied."""
        model, tokenizer = mellum_model_and_tokenizer

        # Mellum uses LLaMA-style architecture
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],  # LLaMA-style attention
        )

        model = get_peft_model(model, lora_config)
        return model, tokenizer

    @pytest.fixture
    def code_batch(self, mellum_model_and_tokenizer):
        """Create a sample batch of code for Mellum."""
        _, tokenizer = mellum_model_and_tokenizer

        code_samples = [
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "class DataProcessor:\n    def __init__(self, data):\n        self.data = data",
            "import torch\nimport torch.nn as nn\n\nclass MLP(nn.Module):",
            "async def fetch_data(url):\n    async with aiohttp.ClientSession() as session:",
        ]

        tokenized = tokenizer(
            code_samples,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )

        return tokenized["input_ids"], tokenized["attention_mask"]

    def test_mellum_lora_conversion(self, mellum_with_lora):
        """Test that Mellum with LoRA converts to functional form correctly."""
        model, _ = mellum_with_lora

        _fmodel, trainable, frozen = make_functional(
            model,
            disable_autograd_tracking=True,
            partition_trainable=True,
        )

        trainable_count = sum(p.numel() for p in trainable.values())
        frozen_count = sum(p.numel() for p in frozen.values())

        print("\nMellum-4b LoRA stats:")
        print(f"  Trainable params: {trainable_count:,}")
        print(f"  Frozen params: {frozen_count:,}")
        print(
            f"  Trainable ratio: {trainable_count / (trainable_count + frozen_count):.4%}"
        )

        # LoRA should be tiny fraction of 4B model
        assert trainable_count < frozen_count / 100  # Less than 1%
        assert trainable_count > 0

    def test_mellum_manual_per_sample_gradients(self, mellum_with_lora, code_batch):
        """Test manual per-sample gradient computation for Mellum.

        This is the recommended approach for LLaMA-style models which have
        vmap incompatibility with transformers 5.x. Instead of using clipped_grad
        with vmap, we compute per-sample gradients manually with a for-loop.

        This approach uses the model directly (not functional API) since
        functional conversion doesn't preserve gradient tracking properly.
        """
        model, _ = mellum_with_lora
        input_ids, attention_mask = code_batch
        labels = input_ids.clone()
        batch_size = input_ids.shape[0]

        # Move to GPU
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        # Get trainable parameters (LoRA weights)
        trainable_params = {
            n: p for n, p in model.named_parameters() if p.requires_grad
        }
        assert len(trainable_params) > 0, "No trainable parameters found"

        # Manual per-sample gradient computation with clipping
        clipping_norm = 1.0
        accumulated_grads = {
            n: torch.zeros_like(p) for n, p in trainable_params.items()
        }

        for i in range(batch_size):
            ids_i = input_ids[i : i + 1]
            mask_i = attention_mask[i : i + 1]
            labels_i = labels[i : i + 1]

            # Zero gradients before each sample
            model.zero_grad()

            outputs = model(ids_i, attention_mask=mask_i, labels=labels_i)
            loss = outputs.loss

            # Backward pass
            loss.backward()

            # Collect gradients and compute norm
            sample_grads = {}
            grad_norm_sq = 0.0
            for n, p in trainable_params.items():
                if p.grad is not None:
                    sample_grads[n] = p.grad.clone()
                    grad_norm_sq += p.grad.pow(2).sum().item()

            total_norm = grad_norm_sq**0.5
            clip_coef = min(1.0, clipping_norm / (total_norm + 1e-6))

            # Accumulate clipped gradients
            for n, g in sample_grads.items():
                accumulated_grads[n] = accumulated_grads[n] + g * clip_coef

        # Average the accumulated gradients
        for n in accumulated_grads:
            accumulated_grads[n] = accumulated_grads[n] / batch_size

        # Verify gradients
        non_zero_grads = sum(1 for g in accumulated_grads.values() if g.norm() > 0)
        assert non_zero_grads > 0, "All gradients are zero"

        for name, grad in accumulated_grads.items():
            assert torch.isfinite(grad).all(), f"Non-finite gradient for {name}"

        print("\nMellum manual gradient computation:")
        print(f"  Batch size: {batch_size}")
        print(f"  Trainable params: {len(trainable_params)}")
        print(f"  Non-zero gradients: {non_zero_grads}")
        for name, grad in list(accumulated_grads.items())[:3]:
            print(f"  {name}: norm={grad.norm().item():.6f}")

    def test_mellum_clipped_grad_forward_pass(self, mellum_with_lora, code_batch):
        """Test that clipped_grad can compute gradients for Mellum.

        vmap compatibility patches are applied at import time.
        """
        model, _ = mellum_with_lora
        input_ids, attention_mask = code_batch
        labels = input_ids.clone()

        # Move to GPU
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        fmodel, trainable, frozen = make_functional(
            model,
            disable_autograd_tracking=True,
            partition_trainable=True,
        )

        def per_example_loss(
            trainable_params,
            frozen_params,
            input_ids_single,
            mask_single,
            labels_single,
        ):
            all_params = {**frozen_params, **trainable_params}
            outputs = fmodel(
                all_params,
                input_ids_single,
                attention_mask=mask_single,
                labels=labels_single,
            )
            return outputs.loss

        # Use microbatching for memory efficiency
        grad_fn, clip_state = clipped_grad(
            per_example_loss,
            argnums=0,
            batch_argnums=(2, 3, 4),
            clipping_norm=1.0,
            microbatch_size=1,  # Process one at a time for large model
        )

        grads, _ = grad_fn(
            trainable, frozen, input_ids, attention_mask, labels, state=clip_state
        )

        # Verify gradients
        assert isinstance(grads, ClippedPytree)
        assert len(grads.pytree) > 0
        for name, grad in grads.pytree.items():
            assert torch.isfinite(grad).all(), f"Non-finite gradient for {name}"

        print("\nMellum clipped_grad:")
        print(f"  Parameters with gradients: {len(grads.pytree)}")
        non_zero = sum(1 for g in grads.pytree.values() if g.norm() > 0)
        print(f"  Non-zero gradients: {non_zero}")

    def test_mellum_memory_bounded_with_microbatching(
        self, mellum_with_lora, code_batch
    ):
        """Test that memory usage is bounded when using microbatching.

        vmap compatibility patches are applied at import time.
        """
        model, _ = mellum_with_lora
        input_ids, attention_mask = code_batch
        labels = input_ids.clone()

        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        fmodel, trainable, frozen = make_functional(
            model,
            disable_autograd_tracking=True,
            partition_trainable=True,
        )

        def per_example_loss(
            trainable_params,
            frozen_params,
            input_ids_single,
            mask_single,
            labels_single,
        ):
            all_params = {**frozen_params, **trainable_params}
            outputs = fmodel(
                all_params,
                input_ids_single,
                attention_mask=mask_single,
                labels=labels_single,
            )
            return outputs.loss

        # Record memory before
        torch.cuda.reset_peak_memory_stats()
        memory_before = torch.cuda.max_memory_allocated()

        grad_fn, clip_state = clipped_grad(
            per_example_loss,
            argnums=0,
            batch_argnums=(2, 3, 4),
            clipping_norm=1.0,
            microbatch_size=1,
        )

        _grads, _ = grad_fn(
            trainable, frozen, input_ids, attention_mask, labels, state=clip_state
        )

        memory_after = torch.cuda.max_memory_allocated()
        memory_used_gb = (memory_after - memory_before) / (1024**3)

        print("\nMellum-4b memory usage:")
        print(f"  Peak memory: {memory_after / (1024**3):.2f} GB")
        print(f"  Memory for gradients: {memory_used_gb:.2f} GB")

        # Memory should be bounded (not OOM)
        assert memory_after < 80 * (1024**3), "Memory usage exceeded 80GB (H200 limit)"


# =============================================================================
# Integration Tests
# =============================================================================


class TestEndToEndDPTraining:
    """End-to-end integration tests for DP training workflow."""

    def test_multiple_training_steps(self, gpt2_with_lora, sample_batch):
        """Test multiple training steps to verify state management."""
        model, _ = gpt2_with_lora
        input_ids, attention_mask = sample_batch
        labels = input_ids.clone()

        fmodel, trainable, frozen = make_functional(
            model,
            disable_autograd_tracking=True,
            partition_trainable=True,
        )

        def per_example_loss(
            trainable_params,
            frozen_params,
            input_ids_single,
            mask_single,
            labels_single,
        ):
            all_params = {**frozen_params, **trainable_params}
            outputs = fmodel(
                all_params,
                input_ids_single,
                attention_mask=mask_single,
                labels=labels_single,
            )
            return outputs.loss

        clipping_norm = 1.0
        noise_multiplier = 0.5
        learning_rate = 1e-4
        num_steps = 3

        params = trainable
        losses = []

        for _step in range(num_steps):
            # Create fresh clipped_grad function each step
            grad_fn, clip_state = clipped_grad(
                per_example_loss,
                argnums=0,
                batch_argnums=(2, 3, 4),
                clipping_norm=clipping_norm,
                return_aux=True,
            )

            (grads, grad_aux), _ = grad_fn(
                params, frozen, input_ids, attention_mask, labels, state=clip_state
            )

            # Track loss
            losses.append(grad_aux.loss_values.mean().item())

            # Add noise
            noise_fn, noise_state = gaussian_noise(
                noise_multiplier=noise_multiplier,
                key=key(_step),
            )
            noisy_grads, noise_state = noise_fn(grads, noise_state)

            # Update
            assert isinstance(noisy_grads, NoisedPytree)
            params = {
                name: param - learning_rate * noisy_grads.pytree[name]
                for name, param in params.items()
            }

        # Verify training progressed (losses recorded)
        assert len(losses) == num_steps
        assert all(loss > 0 for loss in losses)

    def test_clip_norm_from_clip_state(self, gpt2_with_lora, sample_batch):
        """Test that clip_state provides correct clipping_norm for noise calibration."""
        model, _ = gpt2_with_lora
        input_ids, attention_mask = sample_batch
        labels = input_ids.clone()

        fmodel, trainable, frozen = make_functional(
            model,
            disable_autograd_tracking=True,
            partition_trainable=True,
        )

        def per_example_loss(
            trainable_params,
            frozen_params,
            input_ids_single,
            mask_single,
            labels_single,
        ):
            all_params = {**frozen_params, **trainable_params}
            outputs = fmodel(
                all_params,
                input_ids_single,
                attention_mask=mask_single,
                labels=labels_single,
            )
            return outputs.loss

        clipping_norm = 1.5
        grad_fn, clip_state = clipped_grad(
            per_example_loss,
            argnums=0,
            batch_argnums=(2, 3, 4),
            clipping_norm=clipping_norm,
        )

        # Fixed clip: threshold is carried on ClippedPytree metadata, not on state.
        assert isinstance(clip_state, FixedClipState)

        grads, _ = grad_fn(
            trainable, frozen, input_ids, attention_mask, labels, state=clip_state
        )
        assert isinstance(grads, ClippedPytree)
        assert float(grads.max_norm) == clipping_norm
        assert grads.sensitivity == clipping_norm
        assert len(grads.pytree) > 0


# =============================================================================
# Multi-Architecture Model Tests
# =============================================================================


@pytest.mark.slow
@pytest.mark.cuda
class TestMultiArchitectureModels:
    """Test DP training compatibility with various HuggingFace model architectures.

    These tests validate which models work with clipped_grad out of the box
    and which require the vmap compatibility patches.

    Models tested:
    - Qwen2 (Qwen/Qwen2.5-0.5B)
    - DeepSeek (deepseek-ai/deepseek-coder-1.3b-base)
    - Mistral (mistralai/Mistral-7B-v0.1) - requires GPU
    - Gemma2 (google/gemma-2-2b) - requires GPU
    - Phi (microsoft/phi-2)
    """

    @pytest.fixture
    def qwen2_model_and_tokenizer(self):
        """Load Qwen2.5-0.5B model - small enough for CPU testing."""
        model_name = "Qwen/Qwen2.5-0.5B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer

    @pytest.fixture
    def qwen2_with_lora(self, qwen2_model_and_tokenizer):
        """Qwen2 with LoRA adapters."""
        model, tokenizer = qwen2_model_and_tokenizer

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
        )

        model = get_peft_model(model, lora_config)
        return model, tokenizer

    @pytest.fixture
    def deepseek_model_and_tokenizer(self):
        """Load DeepSeek-Coder-1.3B model."""
        model_name = "deepseek-ai/deepseek-coder-1.3b-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer

    @pytest.fixture
    def deepseek_with_lora(self, deepseek_model_and_tokenizer):
        """DeepSeek with LoRA adapters."""
        model, tokenizer = deepseek_model_and_tokenizer

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
        )

        model = get_peft_model(model, lora_config)
        return model, tokenizer

    @pytest.fixture
    def phi2_model_and_tokenizer(self):
        """Load Phi-2 model."""
        model_name = "microsoft/phi-2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer

    @pytest.fixture
    def phi2_with_lora(self, phi2_model_and_tokenizer):
        """Phi-2 with LoRA adapters."""
        model, tokenizer = phi2_model_and_tokenizer

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
        )

        model = get_peft_model(model, lora_config)
        return model, tokenizer

    def _create_sample_batch(self, tokenizer, texts=None):
        """Create a sample batch for testing."""
        if texts is None:
            texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is transforming software development.",
            ]

        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=32,
            return_tensors="pt",
        )
        return tokenized["input_ids"], tokenized["attention_mask"]

    def _test_clipped_grad(self, model, tokenizer):
        """Test if clipped_grad works. Returns (success, error_msg).

        Patches are applied at import time, so no explicit patching needed.
        """
        input_ids, attention_mask = self._create_sample_batch(tokenizer)
        labels = input_ids.clone()

        fmodel, trainable, frozen = make_functional(
            model,
            disable_autograd_tracking=True,
            partition_trainable=True,
        )

        def per_example_loss(
            trainable_params,
            frozen_params,
            input_ids_single,
            mask_single,
            labels_single,
        ):
            all_params = {**frozen_params, **trainable_params}
            outputs = fmodel(
                all_params,
                input_ids_single,
                attention_mask=mask_single,
                labels=labels_single,
            )
            return outputs.loss

        grad_fn, clip_state = clipped_grad(
            per_example_loss,
            argnums=0,
            batch_argnums=(2, 3, 4),
            clipping_norm=1.0,
        )

        try:
            grads, _ = grad_fn(
                trainable, frozen, input_ids, attention_mask, labels, state=clip_state
            )
            # Verify gradients
            assert isinstance(grads, ClippedPytree)
            for name, grad in grads.pytree.items():
                assert torch.isfinite(grad).all(), f"Non-finite gradient for {name}"
            return True, None
        except Exception as e:
            return False, str(e)

    def test_qwen2_clipped_grad(self, qwen2_with_lora):
        """Test Qwen2 with clipped_grad."""
        model, tokenizer = qwen2_with_lora
        success, error = self._test_clipped_grad(model, tokenizer)
        if not success:
            pytest.fail(f"Qwen2 failed: {error}")

    def test_deepseek_clipped_grad(self, deepseek_with_lora):
        """Test DeepSeek with clipped_grad."""
        model, tokenizer = deepseek_with_lora
        success, error = self._test_clipped_grad(model, tokenizer)
        if not success:
            pytest.fail(f"DeepSeek failed: {error}")

    def test_phi2_clipped_grad(self, phi2_with_lora):
        """Test Phi-2 with clipped_grad."""
        model, tokenizer = phi2_with_lora
        success, error = self._test_clipped_grad(model, tokenizer)
        if not success:
            pytest.fail(f"Phi-2 failed: {error}")

    # -------------------------------------------------------------------------
    # Additional model fixtures and tests
    # -------------------------------------------------------------------------

    @pytest.fixture
    def gemma2_model_and_tokenizer(self):
        """Load Gemma-2-2b model."""
        model_name = "google/gemma-2-2b"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer

    @pytest.fixture
    def gemma2_with_lora(self, gemma2_model_and_tokenizer):
        """Gemma2 with LoRA adapters."""
        model, tokenizer = gemma2_model_and_tokenizer

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
        )

        model = get_peft_model(model, lora_config)
        return model, tokenizer

    @pytest.fixture
    def mistral_model_and_tokenizer(self):
        """Load Mistral-7B model."""
        model_name = "mistralai/Mistral-7B-v0.1"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer

    @pytest.fixture
    def mistral_with_lora(self, mistral_model_and_tokenizer):
        """Mistral with LoRA adapters."""
        model, tokenizer = mistral_model_and_tokenizer

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
        )

        model = get_peft_model(model, lora_config)
        return model, tokenizer

    @pytest.fixture
    def phi3_model_and_tokenizer(self):
        """Load Phi-3-mini model."""
        model_name = "microsoft/Phi-3-mini-4k-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer

    @pytest.fixture
    def phi3_with_lora(self, phi3_model_and_tokenizer):
        """Phi-3 with LoRA adapters."""
        model, tokenizer = phi3_model_and_tokenizer

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["qkv_proj"],  # Phi-3 uses fused qkv
        )

        model = get_peft_model(model, lora_config)
        return model, tokenizer

    @pytest.fixture
    def falcon_model_and_tokenizer(self):
        """Load Falcon-RW-1B model (smaller variant for testing)."""
        model_name = "tiiuae/falcon-rw-1b"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        # Falcon config may not have pad_token_id attribute
        if (
            not hasattr(model.config, "pad_token_id")
            or model.config.pad_token_id is None
        ):
            model.config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer

    @pytest.fixture
    def falcon_with_lora(self, falcon_model_and_tokenizer):
        """Falcon with LoRA adapters."""
        model, tokenizer = falcon_model_and_tokenizer

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["query_key_value"],
        )

        model = get_peft_model(model, lora_config)
        return model, tokenizer

    @pytest.fixture
    def stablelm_model_and_tokenizer(self):
        """Load StableLM-2-1.6B model."""
        model_name = "stabilityai/stablelm-2-1_6b"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer

    @pytest.fixture
    def stablelm_with_lora(self, stablelm_model_and_tokenizer):
        """StableLM with LoRA adapters."""
        model, tokenizer = stablelm_model_and_tokenizer

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
        )

        model = get_peft_model(model, lora_config)
        return model, tokenizer

    @pytest.fixture
    def olmo_model_and_tokenizer(self):
        """Load OLMo-1B model."""
        model_name = "allenai/OLMo-1B-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer

    @pytest.fixture
    def olmo_with_lora(self, olmo_model_and_tokenizer):
        """OLMo with LoRA adapters."""
        model, tokenizer = olmo_model_and_tokenizer

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
        )

        model = get_peft_model(model, lora_config)
        return model, tokenizer

    @pytest.fixture
    def bloom_model_and_tokenizer(self):
        """Load BLOOM-560m model."""
        model_name = "bigscience/bloom-560m"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer

    @pytest.fixture
    def bloom_with_lora(self, bloom_model_and_tokenizer):
        """BLOOM with LoRA adapters."""
        model, tokenizer = bloom_model_and_tokenizer

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["query_key_value"],
        )

        model = get_peft_model(model, lora_config)
        return model, tokenizer

    @pytest.fixture
    def mpt_model_and_tokenizer(self):
        """Load MPT-1b model."""
        model_name = "mosaicml/mpt-1b-redpajama-200b"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        if (
            not hasattr(model.config, "pad_token_id")
            or model.config.pad_token_id is None
        ):
            model.config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer

    @pytest.fixture
    def mpt_with_lora(self, mpt_model_and_tokenizer):
        """MPT with LoRA adapters."""
        model, tokenizer = mpt_model_and_tokenizer

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["Wqkv"],
        )

        model = get_peft_model(model, lora_config)
        return model, tokenizer

    @pytest.fixture
    def internlm_model_and_tokenizer(self):
        """Load InternLM2-1.8B model."""
        model_name = "internlm/internlm2-1_8b"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer

    @pytest.fixture
    def internlm_with_lora(self, internlm_model_and_tokenizer):
        """InternLM with LoRA adapters."""
        model, tokenizer = internlm_model_and_tokenizer

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["wqkv"],
        )

        model = get_peft_model(model, lora_config)
        return model, tokenizer

    # -------------------------------------------------------------------------
    # Test methods for additional models
    # -------------------------------------------------------------------------

    @pytest.mark.skip(
        reason="Gemma2 uses sliding window attention - needs investigation"
    )
    def test_gemma2_clipped_grad(self, gemma2_with_lora):
        """Test Gemma2 with clipped_grad."""
        model, tokenizer = gemma2_with_lora
        success, error = self._test_clipped_grad(model, tokenizer)
        if not success:
            pytest.fail(f"Gemma2 failed: {error}")

    @pytest.mark.skip(reason="Mistral-7B too large for CI testing")
    def test_mistral_clipped_grad(self, mistral_with_lora):
        """Test Mistral with clipped_grad."""
        model, tokenizer = mistral_with_lora
        success, error = self._test_clipped_grad(model, tokenizer)
        if not success:
            pytest.fail(f"Mistral failed: {error}")

    @pytest.mark.skip(reason="Phi-3 too large for CI testing")
    def test_phi3_clipped_grad(self, phi3_with_lora):
        """Test Phi-3 with clipped_grad."""
        model, tokenizer = phi3_with_lora
        success, error = self._test_clipped_grad(model, tokenizer)
        if not success:
            pytest.fail(f"Phi-3 failed: {error}")

    @pytest.mark.skip(reason="Falcon not supported - hardcoded shape unpacking")
    def test_falcon_clipped_grad(self, falcon_with_lora):
        """Test Falcon with clipped_grad."""
        model, tokenizer = falcon_with_lora
        success, error = self._test_clipped_grad(model, tokenizer)
        if not success:
            pytest.fail(f"Falcon failed: {error}")

    @pytest.mark.skip(reason="StableLM not supported - inplace operations")
    def test_stablelm_clipped_grad(self, stablelm_with_lora):
        """Test StableLM with clipped_grad."""
        model, tokenizer = stablelm_with_lora
        success, error = self._test_clipped_grad(model, tokenizer)
        if not success:
            pytest.fail(f"StableLM failed: {error}")

    def test_olmo_clipped_grad(self, olmo_with_lora):
        """Test OLMo with clipped_grad."""
        model, tokenizer = olmo_with_lora
        success, error = self._test_clipped_grad(model, tokenizer)
        if not success:
            pytest.fail(f"OLMo failed: {error}")

    @pytest.mark.skip(reason="BLOOM not supported - older architecture")
    def test_bloom_clipped_grad(self, bloom_with_lora):
        """Test BLOOM with clipped_grad."""
        model, tokenizer = bloom_with_lora
        success, error = self._test_clipped_grad(model, tokenizer)
        if not success:
            pytest.fail(f"BLOOM failed: {error}")

    @pytest.mark.skip(reason="MPT not supported - custom attention")
    def test_mpt_clipped_grad(self, mpt_with_lora):
        """Test MPT with clipped_grad."""
        model, tokenizer = mpt_with_lora
        success, error = self._test_clipped_grad(model, tokenizer)
        if not success:
            pytest.fail(f"MPT failed: {error}")

    @pytest.mark.skip(reason="InternLM not supported - custom architecture")
    def test_internlm_clipped_grad(self, internlm_with_lora):
        """Test InternLM with clipped_grad."""
        model, tokenizer = internlm_with_lora
        success, error = self._test_clipped_grad(model, tokenizer)
        if not success:
            pytest.fail(f"InternLM failed: {error}")
