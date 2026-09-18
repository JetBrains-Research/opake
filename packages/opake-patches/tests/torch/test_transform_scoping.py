# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for transform behavior after checkpoint patching."""

from __future__ import annotations

import pytest
import torch
from torch.autograd.graph import save_on_cpu
from torch.func import grad, hessian, jacrev, jvp, vjp, vmap
from torch.utils.checkpoint import checkpoint

from opake.api.engine.clipping import clipped_grad
from opake.api.patches.torch.checkpoint import native_support
from opake.patches import apply_runtime_patches

apply_runtime_patches(vmap_checkpointing=True)

SECOND_DERIVATIVE = 12.0

backport_only = pytest.mark.skipif(
    native_support.native_checkpoint_support(),
    reason="torch conditions create_graph natively; the backport is not applied",
)
torch_2_13_only = pytest.mark.skipif(
    tuple(map(int, torch.__version__.split("+", maxsplit=1)[0].split(".")[:2]))
    < (2, 13),
    reason="full-graph transform support requires PyTorch 2.13+",
)


def cube(x):
    return (x**3).sum()


@pytest.fixture
def create_graph_flags(monkeypatch):
    """Record the ``create_graph`` each functorch backward asks autograd for."""
    flags: list[bool] = []
    orig = torch.autograd.grad

    def spy(*args, create_graph=False, **kwargs):
        flags.append(create_graph)
        return orig(*args, create_graph=create_graph, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", spy)
    return flags


def test_grad_of_grad_matches_upstream():
    got = grad(grad(cube))(torch.tensor(2.0))
    assert got.item() == pytest.approx(SECOND_DERIVATIVE)


def test_jacrev_of_grad_matches_upstream():
    got = jacrev(grad(cube))(torch.tensor(2.0))
    assert got.item() == pytest.approx(SECOND_DERIVATIVE)


def test_vjp_of_grad_matches_upstream():
    _, backward = vjp(grad(cube), torch.tensor(2.0))
    assert backward(torch.tensor(1.0))[0].item() == pytest.approx(SECOND_DERIVATIVE)


def test_forward_over_reverse_matches_upstream():
    _, tangent = jvp(grad(cube), (torch.tensor(2.0),), (torch.tensor(1.0),))
    assert tangent.item() == pytest.approx(SECOND_DERIVATIVE)
    assert hessian(cube)(torch.tensor(2.0)).item() == pytest.approx(SECOND_DERIVATIVE)


def test_vmap_of_higher_order_matches_upstream():
    got = vmap(grad(grad(cube)))(torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(got, torch.tensor([12.0, 18.0]))


def test_autograd_differentiates_through_grad():
    x = torch.tensor(2.0, requires_grad=True)
    (second,) = torch.autograd.grad(grad(cube)(x), x)
    assert second.item() == pytest.approx(SECOND_DERIVATIVE)


def test_autograd_differentiates_through_vmap_grad():
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    (second,) = torch.autograd.grad(vmap(grad(cube))(x).sum(), x)
    torch.testing.assert_close(second, torch.tensor([12.0, 18.0]))


def test_autograd_reaches_parameters_captured_by_the_transform():
    weight = torch.nn.Parameter(torch.tensor(2.0))
    penalty = grad(lambda x: (x * x * weight).sum())(torch.tensor(3.0))
    (dweight,) = torch.autograd.grad(penalty, weight)
    assert dweight.item() == pytest.approx(6.0)


@backport_only
def test_no_grad_entry_skips_the_inner_graph(create_graph_flags):
    with torch.no_grad():
        vmap(grad(cube))(torch.tensor([2.0, 3.0]))
    assert create_graph_flags == [False]


@backport_only
def test_grad_mode_entry_keeps_the_inner_graph(create_graph_flags):
    vmap(grad(cube))(torch.tensor([2.0, 3.0]))
    assert create_graph_flags == [True]


@backport_only
def test_clipped_grad_skips_the_inner_graph(create_graph_flags):
    grad_fn, state = clipped_grad(lambda w, x: ((x - w) ** 2).sum(), clipping_norm=1.0)
    grad_fn(torch.tensor(3.0), torch.tensor([0.0, 7.0]), state=state)
    assert create_graph_flags == [False]


@backport_only
def test_clipped_grad_keeps_the_inner_graph_under_an_outer_transform(
    create_graph_flags,
):
    grad_fn, state = clipped_grad(
        lambda w, x: ((x - w) ** 2).sum(), clipping_norm=100.0
    )

    def clipped_sum(w):
        result, _ = grad_fn(w, torch.tensor([0.0, 7.0]), state=state)
        return result.pytree.sum()

    second = grad(clipped_sum)(torch.tensor(3.0))

    assert create_graph_flags == [True, True]
    assert second.item() == pytest.approx(4.0)


def test_first_order_allows_saved_tensor_hooks():
    with save_on_cpu():
        assert grad(cube)(torch.tensor(3.0)).item() == pytest.approx(27.0)


def test_higher_order_rejects_saved_tensor_hooks():
    with save_on_cpu(), pytest.raises(RuntimeError, match="saved tensor hooks"):
        jacrev(grad(cube))(torch.tensor(2.0))


@torch_2_13_only
def test_compiles_the_dp_transform_with_the_patches_applied():
    grad_fn, state = clipped_grad(lambda w, x: ((x - w) ** 2).sum(), clipping_norm=1.0)
    args = (torch.tensor(3.0), torch.tensor([0.0, 7.0]))

    eager, _ = grad_fn(*args, state=state)
    compiled_fn = torch.compile(grad_fn, backend="aot_eager", fullgraph=True)
    compiled, _ = compiled_fn(*args, state=state)

    torch.testing.assert_close(compiled.pytree, eager.pytree)


@torch_2_13_only
def test_compiled_transform_rejects_saved_tensor_hooks():
    # PyTorch 2.13 can abort Dynamo tracing before it restores this thread-local
    # state. Preserve the pre-test state so the expected error cannot poison
    # later eager tests in the same worker.
    previous_error = (
        torch._C._autograd._saved_tensors_hooks_get_disabled_error_message()
    )
    weight = torch.randn(4, 4)

    def f(x):
        h = checkpoint(lambda z: torch.tanh(z @ weight), x, use_reentrant=False)
        return h.sum()

    try:
        compiled_fn = torch.compile(vmap(grad(f)), backend="aot_eager", fullgraph=True)
        with pytest.raises(RuntimeError, match="saved tensor hooks"):
            compiled_fn(torch.randn(3, 4))
    finally:
        if previous_error is None:
            torch._C._autograd._saved_tensors_hooks_enable()
        else:
            torch._C._autograd._saved_tensors_hooks_disable(previous_error)
