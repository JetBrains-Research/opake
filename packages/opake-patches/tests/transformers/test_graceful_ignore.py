import pytest
import torch.nn as nn

from opake.api.patches.transformers._router import apply_transformers_model_patches


class MockConfig:
    def __init__(self, model_type):
        self.model_type = model_type


class MockModel(nn.Module):
    def __init__(self, model_type):
        super().__init__()
        self.config = MockConfig(model_type)


def test_graceful_ignore_kwargs():
    """Verify that unsupported kwargs (like rope) are gracefully ignored by specialized functions."""

    # Mistral does not have a custom RoPE patch in opake
    # However, we can pass rope=True to apply_kernels without raising an error.
    mistral_model = MockModel("mistral")
    try:
        apply_transformers_model_patches(mistral_model, rope=True)
    except TypeError:
        pytest.fail("apply_kernels raised TypeError for unsupported kwarg on mistral")

    # Likewise, passing unknown kwargs should be gracefully ignored by **kwargs catch-all.
    llama_model = MockModel("llama")
    try:
        apply_transformers_model_patches(
            llama_model, some_random_flag=True, another_flag=False
        )
    except TypeError:
        pytest.fail("apply_kernels raised TypeError for unknown kwarg on llama")
