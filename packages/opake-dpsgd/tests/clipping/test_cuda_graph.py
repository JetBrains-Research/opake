"""CUDA-graph replay with adaptive clipping: the threshold changes every step."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from opake.api.engine.clipping import CudaGraphChunkCompiler
from opake.api.engine.device import fused_kernels_available
from opake.dpsgd.clipping import adaptive_clipped_grad
from opake.random import key

D, H, N_OUT = 16, 32, 4
BATCHES = [12, 10, 13, 12, 9, 12]


def _loss(p, x, y):
    h = torch.tanh(x @ p["w1"] + p["b1"])
    return F.cross_entropy(h @ p["w2"] + p["b2"], y)


def _train(backend, compiler):
    gen = torch.Generator().manual_seed(0)
    params = {
        "w1": (torch.randn(D, H, generator=gen) * 0.3).cuda(),
        "b1": torch.zeros(H, device="cuda"),
        "w2": (torch.randn(H, N_OUT, generator=gen) * 0.3).cuda(),
        "b2": torch.zeros(N_OUT, device="cuda"),
    }
    data = [
        (
            torch.randn(b, D, generator=gen).cuda(),
            torch.randint(0, N_OUT, (b,), generator=gen).cuda(),
        )
        for b in BATCHES
    ]
    kwargs = {"_chunk_compiler": compiler} if compiler is not None else {}
    fn, state = adaptive_clipped_grad(
        _loss,
        argnums=0,
        batch_argnums=(1, 2),
        initial_clipping_norm=0.5,
        target_quantile=0.5,
        key=key(3),
        microbatch_size=4,
        return_aux=True,
        normalize_by=12,
        clip_backend=backend,
        **kwargs,
    )
    trace = []
    for x, y in data:
        (grads, aux), state = fn(params, x, y, state=state)
        trace.append(
            (
                {k: v.clone() for k, v in grads.pytree.items()},
                aux.grad_norms.clone(),
                state._current_clipping_norm,
            )
        )
        params = {k: v - 0.5 * grads.pytree[k] for k, v in params.items()}
    return trace


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize(
    "backend", ["torch", "triton"] if fused_kernels_available() else ["torch"]
)
def test_adaptive_threshold_reaches_every_replay(backend):
    eager = _train(backend, None)
    graphed = _train(backend, CudaGraphChunkCompiler())
    assert len({step[2] for step in eager}) > 1
    for e, g in zip(eager, graphed, strict=True):
        for name in e[0]:
            assert torch.equal(e[0][name], g[0][name])
        assert torch.equal(e[1], g[1])
        assert e[2] == g[2]
