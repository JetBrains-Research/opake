"""CUDA-graph replay of the per-microbatch clipping kernel."""

from __future__ import annotations

import importlib

import pytest
import torch
import torch.nn.functional as F

from opake.api.engine.clipping import CudaGraphChunkCompiler, clipped_grad
from opake.api.engine.device import fused_kernels_available
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


def _train(backend, compiler, device="cuda"):
    kwargs = {
        "argnums": 0,
        "batch_argnums": (1, 2),
        "microbatch_size": 4,
        "return_aux": True,
        "normalize_by": 12,
        "clip_backend": backend,
    }
    if compiler is not None:
        kwargs["_chunk_compiler"] = compiler
    fn, state = clipped_grad(_loss, clipping_norm=0.5, **kwargs)
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


def _backends():
    return ["torch", "triton"] if fused_kernels_available() else ["torch"]


@pytest.mark.cuda
@requires_cuda
@pytest.mark.parametrize("backend", _backends())
def test_graph_replay_matches_eager_bitwise(backend):
    eager = _train(backend, None)
    compiler = CudaGraphChunkCompiler()
    graphed = _train(backend, compiler)
    assert compiler.captures == 3  # chunk sizes 4, 2 and 1
    assert compiler.replays > 0
    for e, g in zip(eager, graphed, strict=True):
        for name in e[0]:
            assert torch.equal(e[0][name], g[0][name])
        assert torch.equal(e[1], g[1])
        assert torch.equal(e[2], g[2])


@pytest.mark.cuda
@requires_cuda
def test_signatures_beyond_max_graphs_run_eagerly():
    eager = _train("torch", None)
    compiler = CudaGraphChunkCompiler(max_graphs=1)
    graphed = _train("torch", compiler)
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


def test_graph_compiler_keeps_the_fused_backend(monkeypatch):
    monkeypatch.setattr(cf, "fused_kernels_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None)
    kwargs = {
        "clipping_norm": 1.0,
        "second_moment": False,
        "compute_dtype": None,
        "scale_fn": None,
    }
    graph = CudaGraphChunkCompiler()
    assert cf._resolve_stream_impl("auto", chunk_compiler=graph, **kwargs) is not None
    assert cf._resolve_stream_impl("auto", chunk_compiler=lambda f: f, **kwargs) is None
    with pytest.raises(ConfigurationError, match="compiled microbatch kernel"):
        cf._resolve_stream_impl("triton", chunk_compiler=lambda f: f, **kwargs)
