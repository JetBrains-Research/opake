"""Eager storage lifetimes at native chunk and clipping boundaries."""

from __future__ import annotations

import weakref

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from opake.api.engine.clipping import auto_clipped_grad, clipped_grad
from opake.api.engine.clipping._clipped_fun import _RuntimeClipState, clipped_fun
from opake.pytree import tree_leaves
from opake.types import PerGroup


@pytest.mark.parametrize("second_moment", [False, True])
@pytest.mark.parametrize("diagnostics", ["none", "aux", "stats"])
def test_chunk_temporaries_die_before_next_kernel(second_moment, diagnostics):
    references = []
    live_at_entry = []

    def observe(kernel):
        def invoke(*args, **kwargs):
            live_at_entry.append(
                [tuple(ref() is not None for ref in group) for group in references]
            )
            outputs = kernel(*args, **kwargs)
            references.append(
                tuple(weakref.ref(leaf) for leaf in tree_leaves(outputs[:4]))
            )
            return outputs

        return invoke

    fn, state = clipped_fun(
        lambda x: x,
        microbatch_size=2,
        clipping_norm=1.0,
        second_moment=second_moment,
        return_aux=diagnostics == "aux",
        return_stats=diagnostics == "stats",
        _chunk_compiler=observe,
    )
    output, new_state = fn(torch.ones(6, 17), state=state)
    assert new_state is state
    assert len(live_at_entry) == 3
    # The first reduction belongs to the accumulator until the second addition.
    assert live_at_entry[1][0][0]
    # No obsolete reductions or dtype markers survive into the third kernel.
    assert not any(any(group) for group in live_at_entry[2]), live_at_entry
    if diagnostics == "aux":
        assert output[1].norms.shape == (6,)
        assert output[1].values.shape == (6, 17)
    elif diagnostics == "stats":
        assert output[1].batch_size == 6


class _SanitizedStorageTracker(TorchDispatchMode):
    """Track physical nan_to_num outputs, not vmap wrappers or strong refs."""

    def __init__(self, leaf_numel):
        super().__init__()
        self.leaf_numel = leaf_numel
        self.refs = []
        self.peak = 0
        self.calls = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if (
            func == torch.ops.aten.nan_to_num.default
            and result.numel() == self.leaf_numel
        ):
            self.calls += 1
            self.refs.append(weakref.ref(result))
            self.peak = max(self.peak, sum(ref() is not None for ref in self.refs))
        return result


class _SquaredStorageTracker(TorchDispatchMode):
    def __init__(self, leaf_numel):
        super().__init__()
        self.leaf_numel = leaf_numel
        self.refs = []
        self.peak = 0
        self.calls = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if (
            func == torch.ops.aten.pow.Tensor_Scalar
            and args[1] == 2
            and result.numel() == self.leaf_numel
        ):
            self.calls += 1
            self.refs.append(weakref.ref(result))
            self.peak = max(self.peak, sum(ref() is not None for ref in self.refs))
        return result


@pytest.mark.parametrize(
    "mode",
    ["fixed", "runtime", "auto", "grouped", "runtime_grouped", "auto_grouped"],
)
def test_clipped_grad_does_not_retain_complete_sanitized_tree(mode):
    def loss(params, x):
        return sum((param * x).sum() for param in params.values())

    params = {f"p{i}": torch.ones(1024) for i in range(8)}
    clipping_norm = 1.0
    if "grouped" in mode:
        clipping_norm = PerGroup(
            groups=dict.fromkeys(params, "all"), values={"all": 1.0}
        )
    if mode.startswith("auto"):
        fn, state = auto_clipped_grad(
            loss,
            R=clipping_norm,
            gamma=0.1,
            return_aux=True,
            microbatch_size=4,
        )
    else:
        fn, state = clipped_grad(
            loss,
            clipping_norm=clipping_norm,
            return_aux=True,
            microbatch_size=4,
        )
        if mode == "runtime":
            state = _RuntimeClipState(clipping_norm=0.75)
        elif mode == "runtime_grouped":
            state = _RuntimeClipState(clipping_norm=0.75 * clipping_norm)
    tracker = _SanitizedStorageTracker(4 * 1024)
    with tracker:
        (result, aux), _ = fn(params, torch.ones(4, 1024), state=state)
    leaves = tree_leaves(result.pytree)
    assert len(leaves) == 8
    assert all(torch.isfinite(leaf).all() for leaf in leaves)
    assert aux.grad_norms.numel() == 4
    assert tracker.calls >= 8  # A vacuous tracker must never pass this regression.
    assert tracker.peak <= 1, f"simultaneous sanitized leaves: {tracker.peak}"


@pytest.mark.parametrize(
    "mode",
    ["fixed", "runtime", "auto", "grouped", "runtime_grouped", "auto_grouped"],
)
def test_paired_clipped_grad_does_not_retain_complete_squared_tree(mode):
    def loss(params, x):
        return sum((param * x).sum() for param in params.values())

    params = {f"p{i}": torch.ones(1024) for i in range(8)}
    options = {"return_aux": True, "microbatch_size": 4, "second_moment": True}
    clipping_norm = 1.0
    if "grouped" in mode:
        clipping_norm = PerGroup(
            groups=dict.fromkeys(params, "all"), values={"all": 1.0}
        )
    if mode.startswith("auto"):
        fn, state = auto_clipped_grad(loss, R=clipping_norm, gamma=0.1, **options)
    else:
        fn, state = clipped_grad(loss, clipping_norm=clipping_norm, **options)
        if mode == "runtime":
            state = _RuntimeClipState(clipping_norm=0.75)
        elif mode == "runtime_grouped":
            state = _RuntimeClipState(clipping_norm=0.75 * clipping_norm)
    tracker = _SquaredStorageTracker(4 * 1024)
    with tracker:
        (result, aux), _ = fn(params, torch.ones(4, 1024), state=state)
    assert len(tree_leaves(result.grads.pytree)) == 8
    assert len(tree_leaves(result.squared_grads.pytree)) == 8
    assert aux.grad_norms.numel() == 4
    assert tracker.calls >= 8
    assert tracker.peak <= 1, f"simultaneous squared leaves: {tracker.peak}"
