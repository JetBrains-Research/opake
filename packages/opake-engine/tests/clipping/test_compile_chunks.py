"""Strict compilation coverage for the private clipping chunk kernel."""

from __future__ import annotations

import importlib

import pytest
import torch
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.testing import CompileCounterWithBackend

from opake.api.engine.clipping import auto_clipped_grad, clipped_grad
from opake.api.engine.clipping._clipped_fun import clipped_fun
from opake.pytree import tree_leaves
from opake.types import PerGroup

cf = importlib.import_module("opake.api.engine.clipping._clipped_fun")


class _InstrumentedAotEagerCompiler:
    def __init__(self) -> None:
        self.compile_calls = 0
        self.frames: list[torch.fx.GraphModule] = []
        self.chunk_shapes: list[tuple[tuple[torch.Size, ...], torch.Size]] = []

    def __call__(self, fn):
        self.compile_calls += 1

        def fw_compiler(gm, _example_inputs):
            self.frames.append(gm)
            return gm.forward

        compiled = torch.compile(
            fn,
            backend=aot_autograd(fw_compiler=fw_compiler),
            fullgraph=True,
            dynamic=True,
        )

        def invoke(*args, **kwargs):
            outputs = compiled(*args, **kwargs)
            reduced_shapes = tuple(leaf.shape for leaf in tree_leaves(outputs[0]))
            self.chunk_shapes.append((reduced_shapes, outputs[4]["norms"].shape))
            return outputs

        return invoke


def _loss_with_aux(params, x, y):
    prediction = x @ params
    return 0.5 * (prediction - y).square(), {"prediction": prediction}


def _assert_aux_close(actual, expected) -> None:
    torch.testing.assert_close(actual.loss_values, expected.loss_values)
    torch.testing.assert_close(actual.grad_norms, expected.grad_norms)
    torch.testing.assert_close(actual.clipped_grad_norms, expected.clipped_grad_norms)
    torch.testing.assert_close(
        actual.loss_aux["prediction"], expected.loss_aux["prediction"]
    )
    assert actual.clipping_rate == expected.clipping_rate
    assert actual.batch_size == expected.batch_size


def test_fixed_chunk_kernel_reuses_dynamic_graph_across_chunk_sizes_with_aux():
    """Dynamic strict graphs handle full/remainder chunks and match eager."""
    torch._dynamo.reset()
    compiler = _InstrumentedAotEagerCompiler()
    compiled_fn, compiled_state = clipped_grad(
        _loss_with_aux,
        argnums=0,
        has_aux=True,
        batch_argnums=(1, 2),
        clipping_norm=0.75,
        normalize_by=8.0,
        return_aux=True,
        microbatch_size=4,
        _chunk_compiler=compiler,
    )
    eager_fn, eager_state = clipped_grad(
        _loss_with_aux,
        argnums=0,
        has_aux=True,
        batch_argnums=(1, 2),
        clipping_norm=0.75,
        normalize_by=8.0,
        return_aux=True,
        microbatch_size=4,
    )

    generator = torch.Generator().manual_seed(91)
    params = torch.randn(5, generator=generator)
    for batch_size in (10, 7):
        x = torch.randn(batch_size, 5, generator=generator)
        y = torch.randn(batch_size, generator=generator)
        (compiled_grads, compiled_aux), _ = compiled_fn(
            params, x, y, state=compiled_state
        )
        (eager_grads, eager_aux), _ = eager_fn(params, x, y, state=eager_state)

        torch.testing.assert_close(compiled_grads.pytree, eager_grads.pytree)
        _assert_aux_close(compiled_aux, eager_aux)

    assert compiler.compile_calls == 1
    # Symbolic dimensions specialize at size one on supported PyTorch versions.
    assert 1 <= len(compiler.frames) <= 2
    assert {diagnostic_shape[0] for _, diagnostic_shape in compiler.chunk_shapes} == {
        2,
        3,
        4,
    }
    assert all(
        reduced_shapes == (params.shape,) for reduced_shapes, _ in compiler.chunk_shapes
    )


@pytest.mark.parametrize("grouped", [False, True])
def test_auto_clipped_grad_threads_and_reuses_chunk_compiler(monkeypatch, grouped):
    """AUTO-S streams through the compiled chunk seam without rebuilding it."""
    calls = 0
    streamed = 0
    stream_clip_and_sum = cf._stream_clip_and_sum

    def observed(*args, **kwargs):
        nonlocal streamed
        streamed += 1
        return stream_clip_and_sum(*args, **kwargs)

    monkeypatch.setattr(cf, "_stream_clip_and_sum", observed)

    def compiler(fn):
        nonlocal calls
        calls += 1
        return fn

    def loss(params, x):
        return sum((leaf * x).sum() for leaf in tree_leaves(params))

    params = {"a": torch.randn(2), "b": torch.randn(2)}
    bound = 0.5
    if grouped:
        bound = PerGroup(groups={"a": "a", "b": "b"}, values={"a": 0.5, "b": 0.25})
    grad_fn, state = auto_clipped_grad(
        loss,
        argnums=0,
        batch_argnums=1,
        R=bound,
        microbatch_size=3,
        _chunk_compiler=compiler,
    )
    for batch_size in (5, 8):
        grads, _ = grad_fn(params, torch.randn(batch_size, 2), state=state)
        assert len(tree_leaves(grads.pytree)) == 2

    assert calls == 1
    assert streamed == 5


def test_second_moment_stats_match_eager_with_strict_chunks():
    backend = CompileCounterWithBackend("aot_eager")

    def compiler(fn):
        return torch.compile(fn, backend=backend, fullgraph=True)

    kwargs = {
        "argnums": 0,
        "batch_argnums": (1, 2),
        "clipping_norm": 0.75,
        "microbatch_size": 3,
        "return_stats": True,
        "second_moment": True,
    }
    compiled_fn, compiled_state = clipped_grad(
        lambda params, x, y: 0.5 * (x @ params - y).square(),
        _chunk_compiler=compiler,
        **kwargs,
    )
    eager_fn, eager_state = clipped_grad(
        lambda params, x, y: 0.5 * (x @ params - y).square(),
        **kwargs,
    )
    params = torch.randn(5)
    x = torch.randn(8, 5)
    y = torch.randn(8)

    (compiled, compiled_stats), _ = compiled_fn(params, x, y, state=compiled_state)
    (eager, eager_stats), _ = eager_fn(params, x, y, state=eager_state)

    torch.testing.assert_close(compiled.grads.pytree, eager.grads.pytree)
    torch.testing.assert_close(
        compiled.squared_grads.pytree, eager.squared_grads.pytree
    )
    assert compiled_stats == eager_stats
    assert 1 <= backend.frame_count <= 2


def test_chunk_cache_is_keyed_by_positional_batching_structure():
    compile_calls = 0

    def compiler(fn):
        nonlocal compile_calls
        compile_calls += 1
        return fn

    fn, state = clipped_fun(
        lambda x, scale=1.0: x * scale,
        batch_argnums=0,
        clipping_norm=10.0,
        microbatch_size=2,
        _chunk_compiler=compiler,
    )
    values = torch.tensor([1.0, 2.0])

    first, _ = fn(values, state=state)
    second, _ = fn(values, 2.0, state=state)

    torch.testing.assert_close(first.pytree, torch.tensor(3.0))
    torch.testing.assert_close(second.pytree, torch.tensor(6.0))
    assert compile_calls == 2


def test_wrapped_function_runtime_threshold_named_kwarg_is_preserved():
    fn, state = clipped_fun(
        lambda x, *, _runtime_clipping_norm: x * _runtime_clipping_norm,
        batch_argnums=0,
        clipping_norm=10.0,
        microbatch_size=2,
    )

    result, _ = fn(
        torch.tensor([1.0, 2.0]),
        state=state,
        _runtime_clipping_norm=2.0,
    )

    torch.testing.assert_close(result.pytree, torch.tensor(6.0))
    assert result.max_norm == 10.0
