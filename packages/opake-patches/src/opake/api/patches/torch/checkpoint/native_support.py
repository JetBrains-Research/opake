# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Probes for which parts of the checkpoint/vmap composition the running PyTorch
supports natively, so each patch is applied only when its capability is missing.

Detection is by feature probe, not version number: it stays correct across
backports and the case where only one of the two upstream PRs has landed.
"""

from __future__ import annotations


def native_checkpoint_support() -> bool:
    """True when torch supports ``vmap(grad(checkpoint(...)))`` natively.

    Sentinel: the parameter-lifetime fix records active reparametrizations on a
    thread-local so checkpoint recomputation re-binds the functional parameters.
    It ships together with the create_graph conditioning, so this one symbol
    gates the whole checkpoint-side patch set.
    """
    try:
        from torch.nn.utils import stateless
    except Exception:  # pragma: no cover - torch layout moved
        return False
    return hasattr(stateless, "_active_reparametrizations")


def saved_tensor_hooks_guard_scoped() -> bool:
    """True when torch scopes the saved-tensor-hooks guard to higher-order only.

    Sentinel: the old blanket ``doesnt_support_saved_tensors_hooks`` was renamed
    to ``disable_saved_tensors_hooks_for_higher_order``. When present, a single
    first-order transform already permits saved-tensor hooks (so checkpoint and
    save_on_cpu work) while higher-order differentiation still raises.
    """
    try:
        from torch._functorch import vmap
    except Exception:  # pragma: no cover - torch layout moved
        return False
    return hasattr(vmap, "disable_saved_tensors_hooks_for_higher_order")
