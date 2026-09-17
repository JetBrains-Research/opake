# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Disable dropout for DP-SGD training.

DP-SGD trains without dropout in practice (per-sample clipping + noise is the
regulariser), and dropout doesn't play well with the per-sample-gradient path:
SDPA's *fused* dropout is incompatible with ``vmap`` ("same randomness with a
batched tensor input"). This compat patch traverses the model once and zeros
every dropout source — the ``nn.Dropout`` modules and the float dropout-rate
attributes that forwards read directly (e.g. the ``dropout_p`` SDPA receives
from ``self.attention_dropout``).
"""

from __future__ import annotations

import torch.nn as nn

# Float dropout-rate attributes that model forwards read directly (not via an
# ``nn.Dropout`` module) — notably the value SDPA gets as ``dropout_p``.
_DROPOUT_ATTRS = (
    "attention_dropout",
    "dropout",
    "hidden_dropout",
    "ffn_dropout",
    "attn_pdrop",
    "resid_pdrop",
    "embd_pdrop",
    "attention_probs_dropout_prob",
    "hidden_dropout_prob",
)


def _zero_float_attrs(obj) -> None:
    for attr in _DROPOUT_ATTRS:
        val = getattr(obj, attr, None)
        # bool is a subclass of int but never a dropout rate; float-only guard.
        if isinstance(val, float) and val != 0.0:
            setattr(obj, attr, 0.0)


def disable_dropout(model) -> None:
    """Zero all dropout in ``model`` (``nn.Dropout`` modules + float rate attrs +
    the config), so training is dropout-free and vmap-safe under any attention
    backend."""
    modules = getattr(model, "modules", None)
    if not callable(modules):  # not an nn.Module (e.g. a routing-test stub)
        return
    for module in modules():
        if isinstance(module, nn.modules.dropout._DropoutNd):
            module.p = 0.0
        _zero_float_attrs(module)
    config = getattr(model, "config", None)
    if config is not None:
        _zero_float_attrs(config)
