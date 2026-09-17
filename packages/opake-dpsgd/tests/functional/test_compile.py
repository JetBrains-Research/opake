"""``torch.compile`` over the functional DP step.

Two axes are validated here:

1. **Loss-closure compile under ``vmap(grad(...))``** — the permissive
   compatibility path. We compare each stage of the DP pipeline
   (per-example grad → clipped sum → noised → torchopt update) eager vs.
   compiled and assert pytree-equal-or-tolerated parity.

2. **Full DP-transform compile** — the supported full-graph path, with
   ``vmap(grad(...))`` inside the compiled transform.

3. **Forward-only compile** — what HuggingFace calls ``jit_mode_eval``.
   A compiled functional model in ``eval()`` matches eager forward.

Compile-the-model-first is **not** a supported pattern with
``torch.func.functional_call`` and is intentionally NOT tested here.
"""

from __future__ import annotations

import shutil

import pytest
import torch
import torch.nn as nn
import torchopt

from opake.api.engine.clipping import clipped_grad
from opake.dpsgd.noise import gaussian_noise
from opake.functional import make_functional
from opake.random import key


def _cpu_inductor_available() -> bool:
    """CPU Inductor codegens C++; skip parametrizations when no compiler is around."""
    return shutil.which("g++") is not None or shutil.which("clang++") is not None


torch_2_13_only = pytest.mark.skipif(
    tuple(map(int, torch.__version__.split("+", maxsplit=1)[0].split(".")[:2]))
    < (2, 13),
    reason="full-DP-transform compilation requires PyTorch 2.13+",
)


def _build_model_and_batch(seed: int = 0, in_features: int = 8, hidden: int = 16):
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.GELU(),
        nn.Linear(hidden, 4),
    )
    x = torch.randn(5, in_features)
    y = torch.randn(5, 4)
    return model, x, y


def _build_eval_model(seed: int = 0):
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.GELU(),
        nn.Linear(16, 4),
    )
    model.eval()
    return model


def _run_dp_step(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    compile_backend: str | None = None,
    compile_fullgraph: bool = False,
    clip_norm: float = 1.0,
    noise_stddev: float = 0.0,
    rng_seed: int = 42,
):
    """One DP step. If ``compile_backend`` is set, compile the loss closure."""
    fmodel, params = make_functional(model)

    def loss_fn(p, xi, yi):
        # Per-example: xi/yi are unbatched under vmap.
        pred = fmodel(p, xi.unsqueeze(0)).squeeze(0)
        return ((pred - yi) ** 2).mean()

    if compile_backend is not None:
        loss_fn = torch.compile(
            loss_fn, backend=compile_backend, fullgraph=compile_fullgraph
        )

    grad_fn, clip_state = clipped_grad(
        loss_fn,
        argnums=0,
        batch_argnums=(1, 2),
        clipping_norm=clip_norm,
    )
    grads, _ = grad_fn(params, x, y, state=clip_state)

    noise_fn, ns = gaussian_noise(noise_multiplier=noise_stddev, key=key(rng_seed))
    noised, _ = noise_fn(grads, ns)

    optimizer = torchopt.adamw(lr=1e-2)
    opt_state = optimizer.init(params)
    updates, _ = optimizer.update(noised.pytree, opt_state, params=params)
    new_params = torchopt.apply_updates(params, updates, inplace=False)
    return grads.pytree, noised.pytree, new_params


def _assert_pytree_close(a, b, *, rtol: float, atol: float):
    assert len(a) == len(b)
    for la, lb in zip(a, b, strict=True):
        torch.testing.assert_close(la, lb, rtol=rtol, atol=atol)


# ----------------------------------------------------------------------------
# Loss-closure compile: vmap(grad(compile(loss))) parity vs eager
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
@pytest.mark.slow
def test_compile_loss_closure_grad_parity(backend: str):
    model, x, y = _build_model_and_batch()
    eager_grads, _, _ = _run_dp_step(model, x, y, compile_backend=None)
    compiled_grads, _, _ = _run_dp_step(model, x, y, compile_backend=backend)
    _assert_pytree_close(eager_grads, compiled_grads, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_compile_loss_closure_noised_grad_parity(backend: str):
    """Adding deterministic Gaussian noise on top of compiled grads still matches."""
    model, x, y = _build_model_and_batch()
    _, eager_noisy, _ = _run_dp_step(
        model, x, y, compile_backend=None, noise_stddev=0.5
    )
    _, compiled_noisy, _ = _run_dp_step(
        model, x, y, compile_backend=backend, noise_stddev=0.5
    )
    _assert_pytree_close(eager_noisy, compiled_noisy, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_compile_loss_closure_optimizer_step_parity(backend: str):
    """torchopt.adamw update on compiled grads ≈ on eager grads."""
    model, x, y = _build_model_and_batch()
    _, _, eager_params = _run_dp_step(model, x, y, compile_backend=None)
    _, _, compiled_params = _run_dp_step(model, x, y, compile_backend=backend)
    _assert_pytree_close(eager_params, compiled_params, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("in_features", "hidden"),
    [(8, 16), (128, 64)],
    ids=["flat_reduction", "blocked_reduction"],
)
@torch_2_13_only
def test_compile_dp_transform_fullgraph_aot_eager(in_features: int, hidden: int):
    """fullgraph=True compiles the outer DP transform without graph breaks.

    The wide case gives the clipping norm a leaf past
    ``_BLOCKED_REDUCTION_MIN`` (8192 elements), so the two-stage reduction is
    traced as well.  Sizing its blocks from the leaf's own shape compiles at
    the narrow width and breaks the graph at the wide one.
    """
    model, x, y = _build_model_and_batch(in_features=in_features, hidden=hidden)
    eager_grads, _, _ = _run_dp_step(model, x, y, compile_backend=None)

    fmodel, params = make_functional(model)

    def loss_fn(p, xi, yi):
        pred = fmodel(p, xi.unsqueeze(0)).squeeze(0)
        return ((pred - yi) ** 2).mean()

    grad_fn, clip_state = clipped_grad(
        loss_fn,
        argnums=0,
        batch_argnums=(1, 2),
        clipping_norm=1.0,
    )
    compiled_grad_fn = torch.compile(grad_fn, backend="aot_eager", fullgraph=True)
    compiled_grads, _ = compiled_grad_fn(params, x, y, state=clip_state)

    _assert_pytree_close(eager_grads, compiled_grads.pytree, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
@pytest.mark.parametrize("backend", ["inductor"])
def test_compile_loss_closure_grad_parity_cuda(backend: str):
    model, x, y = _build_model_and_batch()
    model = model.cuda()
    x, y = x.cuda(), y.cuda()
    eager_grads, _, _ = _run_dp_step(model, x, y, compile_backend=None)
    compiled_grads, _, _ = _run_dp_step(model, x, y, compile_backend=backend)
    _assert_pytree_close(eager_grads, compiled_grads, rtol=1e-4, atol=1e-5)


# ----------------------------------------------------------------------------
# Forward-only compile (HF jit_mode_eval analog)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_compile_eval_forward_parity(backend: str):
    """Compiled fmodel(params, x) ≈ eager fmodel(params, x) in eval()."""
    if backend == "inductor" and not _cpu_inductor_available():
        pytest.skip("CPU Inductor requires a host C++ compiler")
    model = _build_eval_model()
    fmodel, params = make_functional(model)
    x = torch.randn(3, 8)

    with torch.no_grad():
        eager_out = fmodel(params, x)
        compiled_fmodel = torch.compile(fmodel, backend=backend, mode="default")
        compiled_out = compiled_fmodel(params, x)

    torch.testing.assert_close(eager_out, compiled_out, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
def test_compile_eval_reduce_overhead_cuda():
    """Inductor + reduce-overhead on CUDA matches eager forward."""
    model = _build_eval_model().cuda()
    fmodel, params = make_functional(model)
    x = torch.randn(3, 8, device="cuda")

    with torch.no_grad():
        eager_out = fmodel(params, x)
        compiled_fmodel = torch.compile(
            fmodel, backend="inductor", mode="reduce-overhead"
        )
        compiled_out = compiled_fmodel(params, x)
        torch.cuda.synchronize()

    torch.testing.assert_close(eager_out, compiled_out, rtol=1e-4, atol=1e-5)


def test_compile_eval_no_param_mutation():
    """Compiling the forward must not mutate the params tuple it receives."""
    model = _build_eval_model()
    fmodel, params = make_functional(model)
    x = torch.randn(3, 8)

    snapshot = tuple(p.detach().clone() for p in params)
    compiled_fmodel = torch.compile(fmodel, backend="aot_eager")
    with torch.no_grad():
        _ = compiled_fmodel(params, x)

    for before, after in zip(snapshot, params, strict=True):
        torch.testing.assert_close(before, after)
