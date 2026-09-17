# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the disable_dropout compat patch."""

import pytest

pytest.importorskip("transformers")

import torch.nn as nn

from opake.api.patches.transformers.components.dropout import disable_dropout
from opake.exceptions import ConfigurationError


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(0.3)
        self.lin = nn.Linear(4, 4)
        # float dropout-rate attribute, as model attentions carry for SDPA
        self.attention_dropout = 0.25


def test_disable_dropout_zeros_modules_and_attrs():
    m = _Block()
    disable_dropout(m)
    assert m.drop.p == 0.0
    assert m.attention_dropout == 0.0


def test_disable_dropout_via_apply_model_patches():
    """apply_model_patches disables dropout by default (compat); a gpt2-style
    model with 0.1 dropout ends up dropout-free."""
    from transformers.models.gpt2.modeling_gpt2 import GPT2Config, GPT2LMHeadModel

    from opake.patches import apply_model_patches

    config = GPT2Config(vocab_size=64, n_positions=64, n_embd=32, n_layer=1, n_head=2)
    assert config.attn_pdrop == 0.1  # default
    model = GPT2LMHeadModel(config)
    apply_model_patches(model, eager_attention=True)
    assert all(mod.p == 0.0 for mod in model.modules() if isinstance(mod, nn.Dropout))


def test_disable_dropout_opt_out():
    """``dropout=False`` keeps the model's dropout."""
    from transformers.models.gpt2.modeling_gpt2 import GPT2Config, GPT2LMHeadModel

    from opake.patches import apply_model_patches

    config = GPT2Config(vocab_size=64, n_positions=64, n_embd=32, n_layer=1, n_head=2)
    model = GPT2LMHeadModel(config)
    apply_model_patches(model, eager_attention=True, dropout=False)
    assert any(mod.p > 0.0 for mod in model.modules() if isinstance(mod, nn.Dropout))


def test_disable_dropout_requires_registered_family_for_unknown_model():
    class _Unknown(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Config", (), {"model_type": "unknown"})()

    from opake.patches import apply_model_patches

    with pytest.raises(
        ConfigurationError, match="compatibility/dropout/batchify patches require"
    ):
        apply_model_patches(_Unknown(), dropout=True, compat=False, performance=False)


def test_batchify_requires_registered_family_for_unknown_model():
    class _Unknown(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Config", (), {"model_type": "unknown"})()

    from opake.patches import apply_model_patches

    with pytest.raises(
        ConfigurationError, match="compatibility/dropout/batchify patches require"
    ):
        apply_model_patches(_Unknown(), batchify=True, compat=False, performance=False)


def test_unknown_registered_family_requires_explicit_opt_out():
    class _Unknown(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Config", (), {"model_type": "unknown"})()

    from opake.patches import apply_model_patches

    apply_model_patches(_Unknown(), compat=False, performance=False)
