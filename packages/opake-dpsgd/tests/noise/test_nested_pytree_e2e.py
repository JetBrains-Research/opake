"""Nested pytree end-to-end: per_group → clip → gaussian_noise → adamw BC."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("torchopt")

from opake.api.engine.clipping._per_group import per_group
from opake.api.engine.clipping.fun import clip_pytree
from opake.dpsgd.noise import gaussian_noise
from opake.optimizers import adamw
from opake.pytree import tree_flatten_with_paths
from opake.random import key
from opake.types import ClippedPytree


def test_nested_clip_noise_adamw_bc_e2e():
    params = {
        "layer1": {
            "attn": torch.randn(4, 3),
            "mlp": torch.randn(3),
        },
        "layer2": {
            "attn": torch.randn(2, 4),
            "mlp": torch.randn(2),
        },
    }
    grads = {
        "layer1": {
            "attn": torch.randn_like(params["layer1"]["attn"]),
            "mlp": torch.randn_like(params["layer1"]["mlp"]),
        },
        "layer2": {
            "attn": torch.randn_like(params["layer2"]["attn"]),
            "mlp": torch.randn_like(params["layer2"]["mlp"]),
        },
    }
    pg = per_group(params, attn=1.0, mlp=2.0)
    clipped_tree, _ = clip_pytree(grads, pg)
    assert clipped_tree["layer1"]["attn"].shape == (4, 3)

    noise_fn, noise_state = gaussian_noise(noise_multiplier=0.5, key=key(0))
    noised, _ = noise_fn(ClippedPytree(pytree=clipped_tree, max_norm=pg), noise_state)
    assert set(noised.pytree.keys()) == {"layer1", "layer2"}

    opt = adamw(lr=1e-3, noise_bias_correction=True)
    state = opt.init(params)
    paths, _, _ = tree_flatten_with_paths(params)
    assert set(state[0].phi) == set(paths)

    updates, new_state = opt.update(noised, state, params=params)
    assert updates["layer1"]["attn"].shape == (4, 3)
    assert updates["layer2"]["mlp"].shape == (2,)
    assert set(new_state[0].phi) == set(paths)
    assert any(v > 0 for v in new_state[0].phi.values())
