# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""CPU-runnable gradient correctness for checkpoint under vmap(grad(...)).

These run on every torch version in CI (unlike the CUDA-gated suite in
test_checkpoint.py), and exercise the functional_call + checkpoint parameter
re-bind that the patches provide on a pre-fix torch (and that torch handles
natively once the upstream fix lands). The functional_call case poisons the
module's installed parameters, so a stale read during recomputation diverges
hard from the no-checkpoint reference.
"""

from __future__ import annotations

import torch
from torch.func import functional_call, grad, vmap
from torch.utils.checkpoint import checkpoint

from opake.patches import apply_runtime_patches

apply_runtime_patches(vmap_checkpointing=True)


def test_closure_param_vmap_grad_checkpoint_matches(device):
    W = torch.randn(8, 8, device=device, dtype=torch.float32)

    def f(x, use_ckpt):
        block = lambda z: (z @ W).tanh()  # noqa: E731
        h = checkpoint(block, x, use_reentrant=False) if use_ckpt else block(x)
        return (h * h).sum()

    x = torch.randn(5, 8, device=device, dtype=torch.float32)
    got = vmap(grad(lambda z: f(z, True)))(x)
    ref = vmap(grad(lambda z: f(z, False)))(x)
    torch.testing.assert_close(got, ref)


def test_functional_call_checkpoint_per_sample_grads_match(device):
    # functional_call wraps the whole module; checkpoint lives inside its
    # forward, so recomputation runs after functional_call restored the
    # originals. Poison the installed params so any stale read diverges.
    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(8, 8, dtype=torch.float32)

        def forward(self, x):
            return self.lin(x).tanh()

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = Inner()
            self.head = torch.nn.Linear(8, 1, dtype=torch.float32)

        def forward(self, x, use_ckpt):
            if use_ckpt:
                x = checkpoint(self.inner, x, use_reentrant=False)
            else:
                x = self.inner(x)
            return self.head(x).sum()

    torch.manual_seed(0)
    net = Net().to(device=device)
    params = {k: v.detach().clone() for k, v in net.named_parameters()}
    with torch.no_grad():
        for p in net.parameters():
            p.add_(123.0)  # poison: a stale read gives wrong gradients

    def loss(p, xi, use_ckpt):
        return functional_call(net, p, (xi, use_ckpt))

    batch = torch.randn(4, 8, device=device, dtype=torch.float32)
    ref = vmap(grad(loss), in_dims=(None, 0, None))(params, batch, False)
    got = vmap(grad(loss), in_dims=(None, 0, None))(params, batch, True)
    for k in ref:
        torch.testing.assert_close(got[k], ref[k], msg=f"mismatch for {k}")


def test_tied_functional_call_checkpoint_per_sample_grads_match(device):
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = torch.nn.Linear(8, 8, bias=False)
            self.head = torch.nn.Linear(8, 8, bias=False)
            self.head.weight = self.inner.weight

        def forward(self, x, use_ckpt):
            if use_ckpt:
                x = checkpoint(self.inner, x, use_reentrant=False)
            else:
                x = self.inner(x)
            return self.head(x).sum()

    torch.manual_seed(0)
    net = Net().to(device=device)
    params = {k: v.detach().clone() for k, v in net.named_parameters()}
    with torch.no_grad():
        net.inner.weight.add_(123.0)

    def loss(p, xi, use_ckpt):
        return functional_call(net, p, (xi, use_ckpt))

    batch = torch.randn(4, 8, device=device, dtype=torch.float32)
    ref = vmap(grad(loss), in_dims=(None, 0, None))(params, batch, False)
    got = vmap(grad(loss), in_dims=(None, 0, None))(params, batch, True)
    for k in ref:
        torch.testing.assert_close(got[k], ref[k], msg=f"mismatch for {k}")
