import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from opake.api.engine.clipping import clipped_grad
from opake.functional import make_functional
from opake.patches import apply_model_patches, apply_runtime_patches

apply_runtime_patches()
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Kernel patch compatibility tests require CUDA/Triton",
)

RTOL = 0.0001
ATOL = 0.0001


@pytest.mark.cuda
class TestEndToEnd:
    """End-to-end test with full model + LoRA + clipped_grad."""

    @pytest.mark.slow
    def test_qwen2_lora_clipped_grad(self, device):
        """Full pipeline: Qwen2 + LoRA + clipped_grad with kernel patches."""
        config = AutoConfig.from_pretrained("Qwen/Qwen2-0.5B")
        config.num_hidden_layers = 2
        model = AutoModelForCausalLM.from_config(config)
        lora_config = LoraConfig(
            r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0
        )
        model = get_peft_model(model, lora_config).to(device)
        apply_model_patches(model, performance=False, compat=True, lora=True)
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        texts = ["Hello world test", "Another example", "Third sample", "Final one"]
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, max_length=16, truncation=True
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        labels = input_ids.clone()
        fmodel, trainable, frozen = make_functional(
            model, disable_autograd_tracking=True, partition_trainable=True
        )

        def per_example_loss(trainable_params, frozen_params, ids, mask, lbls):
            all_params = {**frozen_params, **trainable_params}
            outputs = fmodel(all_params, ids, attention_mask=mask, labels=lbls)
            return outputs.loss

        grad_fn, clip_state = clipped_grad(
            per_example_loss, argnums=0, batch_argnums=(2, 3, 4), clipping_norm=1.0
        )
        grads, _state = grad_fn(
            trainable, frozen, input_ids, attention_mask, labels, state=clip_state
        )
        assert grads is not None, "No gradients returned"
        assert len(grads.pytree) > 0, "Empty gradient dict"
        for name, g in grads.pytree.items():
            assert not torch.isnan(g).any(), f"NaN in grad for {name}"
