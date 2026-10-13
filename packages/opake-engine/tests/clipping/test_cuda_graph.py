"""CUDA-graph replay of the per-microbatch clipping kernel."""

from __future__ import annotations

import importlib

import pytest
import torch
import torch.nn.functional as F

from opake.api.engine.clipping import CudaGraphChunkCompiler, clipped_grad
from opake.api.engine.clipping._cuda_graph import (
    _clone_input,
    _collect_tensor_copies,
    _leaf_key,
)
from opake.api.engine.types import PerGroup
from opake.exceptions import ConfigurationError

cf = importlib.import_module("opake.api.engine.clipping._clipped_fun")

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

D, H, N_OUT = 16, 32, 4
BATCHES = [12, 10, 13, 12, 9, 12]  # partial microbatches of 2, 1 and 1 at size 4


def _params(device):
    gen = torch.Generator().manual_seed(0)
    return {
        "w1": (torch.randn(D, H, generator=gen) * 0.3).to(device),
        "b1": torch.zeros(H, device=device),
        "w2": (torch.randn(H, N_OUT, generator=gen) * 0.3).to(device),
        "b2": torch.zeros(N_OUT, device=device),
    }


def _data(device):
    gen = torch.Generator().manual_seed(1)
    return [
        (
            torch.randn(b, D, generator=gen).to(device),
            torch.randint(0, N_OUT, (b,), generator=gen).to(device),
        )
        for b in BATCHES
    ]


def _loss(p, x, y):
    h = torch.tanh(x @ p["w1"] + p["b1"])
    return F.cross_entropy(h @ p["w2"] + p["b2"], y)


def _train(compiler, device="cuda", clipping_norm=0.5):
    kwargs = {
        "argnums": 0,
        "batch_argnums": (1, 2),
        "microbatch_size": 4,
        "return_aux": True,
        "normalize_by": 12,
    }
    if compiler is not None:
        kwargs["_chunk_compiler"] = compiler
    fn, state = clipped_grad(_loss, clipping_norm=clipping_norm, **kwargs)
    params = _params(device)
    trace = []
    for x, y in _data(device):
        (grads, aux), state = fn(params, x, y, state=state)
        trace.append(
            (
                {k: v.clone() for k, v in grads.pytree.items()},
                aux.grad_norms.clone(),
                aux.clipped_grad_norms.clone(),
            )
        )
        params = {k: v - 0.5 * grads.pytree[k] for k, v in params.items()}
    return trace


@pytest.mark.cuda
@requires_cuda
def test_graph_replay_matches_eager_bitwise():
    eager = _train(None)
    compiler = CudaGraphChunkCompiler()
    graphed = _train(compiler)
    assert compiler.captures == 3  # chunk sizes 4, 2 and 1
    assert compiler.replays > 0
    for e, g in zip(eager, graphed, strict=True):
        for name in e[0]:
            assert torch.equal(e[0][name], g[0][name])
        assert torch.equal(e[1], g[1])
        assert torch.equal(e[2], g[2])


def test_per_group_tensor_values_contribute_to_graph_key():
    groups = {("weight",): "main"}
    first = PerGroup(groups=groups, values={"main": torch.ones((), device="cpu")})
    second = PerGroup(groups=groups, values={"main": torch.ones(2, device="cpu")})

    first_key = _leaf_key(first)
    second_key = _leaf_key(second)

    hash(first_key)
    assert first_key != second_key


def test_per_group_tensor_values_are_copied_into_static_inputs():
    groups = {("weight",): "main"}
    static = _clone_input(PerGroup(groups=groups, values={"main": torch.tensor(2.0)}))
    source = PerGroup(groups=groups, values={"main": torch.tensor(3.0)})
    static_tensors = []
    source_tensors = []

    _collect_tensor_copies(static, source, static_tensors, source_tensors)
    torch._foreach_copy_(static_tensors, source_tensors)

    assert static.values["main"].item() == 3.0


@pytest.mark.cuda
@requires_cuda
def test_per_group_tensor_values_are_copied_before_replay():
    compiler = CudaGraphChunkCompiler()
    graphed = compiler(lambda norm, x: norm.values["main"] * x)
    groups = {("weight",): "main"}
    x = torch.ones(3, device="cuda")

    first = PerGroup(groups=groups, values={"main": torch.tensor(2.0, device="cuda")})
    second = PerGroup(groups=groups, values={"main": torch.tensor(3.0, device="cuda")})
    torch.testing.assert_close(graphed(first, x), 2 * x)
    torch.testing.assert_close(graphed(second, x), 3 * x)

    assert compiler.captures == 1
    assert compiler.replays == 2


@pytest.mark.cuda
@requires_cuda
def test_per_group_clipping_graph_replay_matches_eager():
    clipping_norm = PerGroup(
        groups={
            ("w1",): "embedding",
            ("b1",): "embedding",
            ("w2",): "head",
            ("b2",): "head",
        },
        values={"embedding": 0.5, "head": 0.25},
    )
    eager = _train(None, clipping_norm=clipping_norm)
    compiler = CudaGraphChunkCompiler()
    graphed = _train(compiler, clipping_norm=clipping_norm)

    assert compiler.captures == 3
    assert compiler.replays > 0
    for e, g in zip(eager, graphed, strict=True):
        for name in e[0]:
            assert torch.equal(e[0][name], g[0][name])
        assert torch.equal(e[1], g[1])
        assert torch.equal(e[2], g[2])


@pytest.mark.cuda
@requires_cuda
def test_signatures_beyond_max_graphs_run_eagerly():
    eager = _train(None)
    compiler = CudaGraphChunkCompiler(max_graphs=1)
    graphed = _train(compiler)
    assert compiler.captures == 1
    for e, g in zip(eager, graphed, strict=True):
        for name in e[0]:
            assert torch.equal(e[0][name], g[0][name])


@pytest.mark.cuda
@requires_cuda
def test_unhashable_non_tensor_arguments_are_rejected():
    class Unhashable:
        __hash__ = None

    graphed = CudaGraphChunkCompiler()(lambda x, extra: x * 2)
    with pytest.raises(ConfigurationError, match="hashable"):
        graphed(torch.ones(2, device="cuda"), Unhashable())


def test_default_graph_cap_is_four():
    assert CudaGraphChunkCompiler().max_graphs == 4


@pytest.mark.parametrize("max_graphs", [0, -1, True, 1.5, "3"])
def test_max_graphs_requires_positive_integer(max_graphs):
    with pytest.raises(
        ConfigurationError, match="max_graphs must be a positive integer"
    ):
        CudaGraphChunkCompiler(max_graphs=max_graphs)


def test_inputs_without_cuda_tensors_run_eagerly():
    compiler = CudaGraphChunkCompiler()
    graphed = compiler(lambda x: x * 2)
    torch.testing.assert_close(graphed(torch.ones(3)), torch.full((3,), 2.0))
    assert compiler.captures == 0


def test_graph_compiler_uses_torch_clipping_fallback():
    assert (
        cf._resolve_stream_impl(
            clipping_norm=1.0,
            second_moment=False,
            compute_dtype=None,
            scale_fn=None,
            chunk_compiler=CudaGraphChunkCompiler(),
        )
        is None
    )
