"""``torch.compile`` of a clip-grad transform must preserve DP-exact gradients.

``clipped_grad`` / ``auto_clipped_grad`` / ``adaptive_clipped_grad`` are
*constructors* that return a ``vmap(grad)+clip`` transform.  The caller may
``torch.compile`` that transform — the PyTorch idiom ``gf = torch.compile(gf)``
— which is the supported, fusing pattern (functorch *inside* ``torch.compile``).
Compiling the inner loss and applying ``vmap(grad)`` outside it instead raises
"Unsupported functorch tracing attempt" and silently no-ops, so the transform is
the right thing to compile.

These tests lock in that the compiled transform is numerically identical to the
eager one for every clip mode — the clip is the DP sensitivity bound, so a
miscompiled grad would be a privacy bug, not just a perf regression.  We use the
``aot_eager`` backend so the test exercises the dynamo+functorch composition (the
thing that must hold) without requiring an inductor C toolchain in CI.
"""

import pytest
import torch
from torch._dynamo.testing import CompileCounterWithBackend

from opake.api.engine.clipping import auto_clipped_grad, clipped_grad
from opake.dpsgd.clipping import adaptive_clipped_grad
from opake.random import key


def _loss(params, x, y):
    return ((x @ params - y) ** 2).mean()


def _data():
    torch.manual_seed(0)
    params = torch.randn(4, dtype=torch.float64)
    x = torch.randn(8, 4, dtype=torch.float64)
    y = torch.randn(8, dtype=torch.float64)
    return params, x, y


def _build(mode):
    common = {
        "argnums": 0,
        "batch_argnums": (1, 2),
        "normalize_by": 8,
        "return_aux": True,
    }
    if mode == "fixed":
        return clipped_grad(_loss, clipping_norm=1.0, **common)
    if mode == "auto":
        return auto_clipped_grad(_loss, R=1.0, **common)
    return adaptive_clipped_grad(_loss, initial_clipping_norm=1.0, key=key(0), **common)


def _grads(out):
    # out == ((ClippedPytree, aux), state); single-tensor param → .pytree is a tensor
    return out[0][0].pytree


@pytest.mark.parametrize("mode", ["fixed", "auto", "adaptive"])
@pytest.mark.slow
def test_compiled_transform_matches_eager(mode):
    params, x, y = _data()
    gf_eager, st_e = _build(mode)
    gf_comp, st_c = _build(mode)
    # Caller-applied compile (the idiom): compile the returned transform.
    # aot_eager routes through dynamo + functorch without an inductor toolchain.
    gf_comp = torch.compile(gf_comp, backend="aot_eager")

    out_e = gf_eager(params, x, y, state=st_e)
    out_c = gf_comp(params, x, y, state=st_c)

    grads_e, grads_c = _grads(out_e), _grads(out_c)
    assert torch.allclose(grads_e, grads_c, atol=1e-10, rtol=0), (
        f"{mode}: compiled grad differs from eager by "
        f"{(grads_e - grads_c).abs().max().item():.2e} — DP sensitivity not preserved"
    )


def test_adaptive_chunk_compiles_once_across_threshold_updates():
    params, x, y = _data()
    backend = CompileCounterWithBackend("aot_eager")

    def compiler(fn):
        return torch.compile(fn, backend=backend, fullgraph=True)

    common = {
        "argnums": 0,
        "batch_argnums": (1, 2),
        "normalize_by": 8,
        "return_aux": True,
        "microbatch_size": 4,
    }
    eager_fn, eager_state = adaptive_clipped_grad(
        _loss,
        initial_clipping_norm=1.0,
        key=key(0),
        **common,
    )
    compiled_fn, compiled_state = adaptive_clipped_grad(
        _loss,
        initial_clipping_norm=1.0,
        key=key(0),
        _chunk_compiler=compiler,
        **common,
    )

    for _ in range(3):
        eager_out, eager_state = eager_fn(params, x, y, state=eager_state)
        compiled_out, compiled_state = compiled_fn(params, x, y, state=compiled_state)
        torch.testing.assert_close(
            compiled_out[0].pytree, eager_out[0].pytree, rtol=0, atol=0
        )
        assert compiled_state == eager_state

    assert backend.frame_count == 1


@pytest.mark.parametrize("mode", ["fixed", "auto", "adaptive"])
def test_eager_transform_runs(mode):
    """The plain (uncompiled) transform produces a finite clipped gradient."""
    params, x, y = _data()
    gf, st = _build(mode)
    out = gf(params, x, y, state=st)
    assert torch.isfinite(_grads(out)).all()
