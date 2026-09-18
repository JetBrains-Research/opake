import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from opake.patches import apply_model_patches

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Kernel patch compatibility tests require CUDA/Triton",
)

RTOL = 0.0001
ATOL = 0.0001


class TestBatchifyForward:
    """Test _batchify_forward unsqueeze/squeeze round-trip with a real model."""

    def test_batchify_1d_input_ids(self):
        """1D input_ids should be unsqueezed, output logits squeezed back."""

        config = AutoConfig.from_pretrained("openai-community/gpt2")
        config.num_hidden_layers = 1
        model = AutoModelForCausalLM.from_config(config)
        apply_model_patches(model, performance=False, compat=True)
        from opake.api.patches.transformers.components.batchify import (
            _batchify_forward,
        )

        model.forward = _batchify_forward(model.forward)
        seq_len = 8
        input_ids = torch.randint(0, config.vocab_size, (seq_len,))
        labels = input_ids.clone()
        outputs = model(input_ids=input_ids, labels=labels)
        assert outputs.logits.ndim == 2, (
            f"Expected 2D logits (seq, vocab), got shape {outputs.logits.shape}"
        )
        assert outputs.logits.shape[0] == seq_len
        assert outputs.loss.ndim == 0

    def test_batchify_2d_input_ids_is_noop(self):
        """2D input_ids (already batched) should pass through unchanged."""

        config = AutoConfig.from_pretrained("openai-community/gpt2")
        config.num_hidden_layers = 1
        model = AutoModelForCausalLM.from_config(config)
        apply_model_patches(model, performance=False, compat=True)
        from opake.api.patches.transformers.components.batchify import (
            _batchify_forward,
        )

        model.forward = _batchify_forward(model.forward)
        batch, seq_len = (3, 8)
        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
        labels = input_ids.clone()
        outputs = model(input_ids=input_ids, labels=labels)
        assert outputs.logits.ndim == 3
        assert outputs.logits.shape == (batch, seq_len, config.vocab_size)

    def test_batchify_positional_input_ids(self):
        """input_ids passed positionally should also be batchified."""

        config = AutoConfig.from_pretrained("openai-community/gpt2")
        config.num_hidden_layers = 1
        model = AutoModelForCausalLM.from_config(config)
        apply_model_patches(model, performance=False, compat=True)
        from opake.api.patches.transformers.components.batchify import (
            _batchify_forward,
        )

        model.forward = _batchify_forward(model.forward)
        seq_len = 8
        input_ids = torch.randint(0, config.vocab_size, (seq_len,))
        outputs = model(input_ids)
        assert outputs.logits.ndim == 2, (
            f"Expected 2D logits, got shape {outputs.logits.shape}"
        )
