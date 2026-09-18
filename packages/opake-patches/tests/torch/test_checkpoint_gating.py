# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Capability-gating logic for the checkpoint patches.

Verifies that ``apply_checkpoint_patch`` applies exactly the right set of patches
for each torch regime (detected via the two capability probes), without mutating
torch: the per-patch ``apply`` functions are replaced by spies. Runs on any torch,
CPU-only.
"""

from __future__ import annotations

import pytest

from opake.api.patches.torch import checkpoint as ck

# Every per-patch module the orchestrator may call.
_TORCH_CORE = [
    "saved_tensor_hooks_guard",
    "save_on_cpu",
    "noop_save_inputs",
    "create_graph",
    "reparametrize_recompute",
]
_ALL = [*_TORCH_CORE, "huggingface"]


@pytest.fixture
def applied(monkeypatch):
    """Spy on each patch's ``apply`` and reset the idempotency flag."""
    calls: list[str] = []
    for name in _ALL:
        module = getattr(ck, name)
        monkeypatch.setattr(module, "apply", lambda _n=name: calls.append(_n))
    monkeypatch.setattr(ck, "_is_checkpoint_patched", False)
    return calls


def _set_regime(monkeypatch, *, native: bool, guard_scoped: bool):
    monkeypatch.setattr(ck.native_support, "native_checkpoint_support", lambda: native)
    monkeypatch.setattr(
        ck.native_support, "saved_tensor_hooks_guard_scoped", lambda: guard_scoped
    )


def test_native_support_applies_only_huggingface(applied, monkeypatch):
    _set_regime(monkeypatch, native=True, guard_scoped=True)
    ck.apply_checkpoint_patch(vmap_checkpointing=True)
    assert applied == ["huggingface"]
    assert ck.is_checkpoint_patched()


def test_unsupported_old_or_new_arch_applies_full_backport(applied, monkeypatch):
    # No native fix and no guard scoping: the complete backport plus HF glue.
    _set_regime(monkeypatch, native=False, guard_scoped=False)
    ck.apply_checkpoint_patch(vmap_checkpointing=True)
    assert set(applied) == set(_ALL)
    assert ck.is_checkpoint_patched()


def test_guard_scoped_but_no_native_skips_guard_and_save_on_cpu(applied, monkeypatch):
    # PR-A-only: torch already scopes the guard and fixes save_on_cpu.
    _set_regime(monkeypatch, native=False, guard_scoped=True)
    ck.apply_checkpoint_patch(vmap_checkpointing=True)
    assert "saved_tensor_hooks_guard" not in applied
    assert "save_on_cpu" not in applied
    assert set(applied) == set(_ALL) - {"saved_tensor_hooks_guard", "save_on_cpu"}


def test_native_but_guard_unscoped_still_applies_guard_and_save_on_cpu(
    applied, monkeypatch
):
    # Mixed regime: the param-lifetime fix is present but the guard scoping is
    # not, so the guard/save_on_cpu backports must still apply (the probes are
    # independent) while the checkpoint-core backports are skipped.
    _set_regime(monkeypatch, native=True, guard_scoped=False)
    ck.apply_checkpoint_patch(vmap_checkpointing=True)
    assert set(applied) == {"saved_tensor_hooks_guard", "save_on_cpu", "huggingface"}
    assert ck.is_checkpoint_patched()


def test_disabled_applies_nothing(applied, monkeypatch):
    _set_regime(monkeypatch, native=False, guard_scoped=False)
    ck.apply_checkpoint_patch(vmap_checkpointing=False)
    assert applied == []
    assert not ck.is_checkpoint_patched()


def test_idempotent(applied, monkeypatch):
    _set_regime(monkeypatch, native=False, guard_scoped=False)
    ck.apply_checkpoint_patch(vmap_checkpointing=True)
    first = list(applied)
    ck.apply_checkpoint_patch(vmap_checkpointing=True)
    assert applied == first  # second call is a no-op


def test_create_graph_patch_accepts_create_graph_keyword():
    """functorch calls the patched helper with ``create_graph=`` as a kwarg.

    Renaming that parameter (e.g. ARG unused-arg underscore) breaks
    ``vmap(grad(...))`` with TypeError: unexpected keyword argument.
    """
    import torch
    import torch._functorch.eager_transforms as eager

    from opake.api.patches.torch.checkpoint import create_graph as cg

    # Apply on a fresh copy of the current helper so the test is hermetic.
    before = eager._autograd_grad
    try:
        cg.apply()
        x = torch.tensor(1.0, requires_grad=True)
        (y,) = eager._autograd_grad((x * 2,), (x,), create_graph=True)
        assert torch.allclose(y, torch.tensor(2.0))
    finally:
        eager._autograd_grad = before


def test_guard_backport_rejects_hooks_during_compile(monkeypatch):
    """The backport uses PyTorch's compiled-hook safety behavior."""
    import torch

    from opake.api.patches.torch.checkpoint import saved_tensor_hooks_guard

    def use_save_on_cpu():
        with torch.autograd.graph.save_on_cpu():
            return torch.tensor(1.0)

    guarded = saved_tensor_hooks_guard._disable_saved_tensor_hooks_for_higher_order(
        use_save_on_cpu
    )
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    with pytest.raises(RuntimeError, match="saved tensor hooks"):
        guarded()
