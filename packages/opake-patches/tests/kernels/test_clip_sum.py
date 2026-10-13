"""Automatic fused clipping dispatch for ``clipped_fun`` / ``clipped_grad``."""

from __future__ import annotations

import importlib
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.func import functional_call

from opake.api.engine.clipping import clipped_grad
from opake.api.engine.clipping._clipped_fun import clipped_fun
from opake.api.engine.device import fused_kernels_available
from opake.patches import apply_runtime_patches
from opake.types import PerGroup

cf = importlib.import_module("opake.api.engine.clipping._clipped_fun")

requires_fused = pytest.mark.skipif(
    not fused_kernels_available(), reason="needs a CUDA device and Triton"
)


@pytest.fixture(scope="module", autouse=True)
def _register_fused_clip_backend():
    if fused_kernels_available():
        apply_runtime_patches()


FP32 = {"w": torch.float32, "b": torch.float32, "e": torch.float32}
BF16 = {"w": torch.bfloat16, "b": torch.bfloat16, "e": torch.bfloat16}
MIXED = {"w": torch.float32, "b": torch.bfloat16, "e": torch.float32}


def _clip(tree, **kwargs):
    fn, state = clipped_fun(lambda v: v, batch_argnums=0, **kwargs)
    out, _ = fn(tree, state=state)
    return out


def _clip_without_fused(monkeypatch, tree, **kwargs):
    with monkeypatch.context() as context:
        context.setattr(cf, "get_fused_clip_backend", lambda: None)
        return _clip(tree, **kwargs)


def _tree(batch, dtypes, seed, norms=(0.3, 2.0), device="cuda"):
    """Per-example tree whose global norms are uniform in ``norms`` (C=1)."""
    gen = torch.Generator().manual_seed(seed)
    shapes = {"w": (64, 48), "b": (48,), "e": (3, 5, 7)}
    raw = {
        k: torch.randn(batch, *s, generator=gen, dtype=torch.float64)
        for k, s in shapes.items()
    }
    total = torch.sqrt(sum((v.flatten(1) ** 2).sum(1) for v in raw.values()))
    target = torch.empty(batch, dtype=torch.float64).uniform_(*norms, generator=gen)
    scale = target / total
    return {
        k: (v * scale.view(-1, *[1] * (v.dim() - 1))).to(dtypes[k]).to(device)
        for k, v in raw.items()
    }


def _spy_fused(monkeypatch):
    module = importlib.import_module("opake.api.patches.kernels._clip_sum")
    calls = []
    real = module.fused_clip_sum

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "fused_clip_sum", spy)
    return calls


def _stored_norm(tree):
    return math.sqrt(sum(float((v.double().cpu() ** 2).sum()) for v in tree.values()))


# --- dispatch rules (any host) ----------------------------------------------


def _resolve_auto(**overrides):
    options = {
        "clipping_norm": 1.0,
        "second_moment": False,
        "compute_dtype": None,
        "scale_fn": None,
        "chunk_compiler": None,
    }
    options.update(overrides)
    return cf._resolve_stream_impl(**options)


@pytest.mark.parametrize(
    "overrides",
    [
        {"clipping_norm": PerGroup(groups={"a": "g"}, values={"g": 1.0})},
        {"second_moment": True},
        {"compute_dtype": torch.float64},
        {"scale_fn": lambda v: (v, None)},
        {"chunk_compiler": lambda f: f},
    ],
)
def test_auto_resolves_to_torch_for_unsupported_configurations(overrides, monkeypatch):
    monkeypatch.setattr(cf, "fused_kernels_available", lambda: True)
    monkeypatch.setattr(cf, "get_fused_clip_backend", lambda: object())
    assert _resolve_auto(**overrides) is None


def test_empty_batch_falls_back_from_fused_kernel(monkeypatch):
    pytest.importorskip("triton")
    module = importlib.import_module("opake.api.patches.kernels._clip_sum")

    class CudaTensor(torch.Tensor):
        @property
        def is_cuda(self):
            return True

    leaf = torch.empty((0, 3)).as_subclass(CudaTensor)
    expected = object()
    calls = []

    def fallback(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(module, "_stream_clip_and_sum", fallback)
    actual = module.fused_stream_clip_and_sum(
        leaf,
        1.0,
        scale=SimpleNamespace(kind="fixed"),
        batch_size=0,
        reduce_leaf=lambda value: value,
        compute_dtype=None,
        second_moment=False,
        return_aux=False,
        return_stats=False,
    )

    assert actual is expected
    assert len(calls) == 1


def test_auto_resolves_to_torch_on_rocm_builds(monkeypatch):
    monkeypatch.setattr(cf, "fused_kernels_available", lambda: True)
    monkeypatch.setattr(cf, "get_fused_clip_backend", lambda: object())
    monkeypatch.setattr(torch.version, "hip", "6.0")
    assert _resolve_auto() is None


def test_dispatch_without_fused_kernels_uses_torch(monkeypatch):
    monkeypatch.setattr(cf, "fused_kernels_available", lambda: False)
    batch = torch.tensor([[3.0, 4.0], [0.6, 0.8]])
    clip_fn, state = clipped_fun(lambda values: values, clipping_norm=1.0)
    actual, _ = clip_fn(batch, state=state)
    torch.testing.assert_close(actual.pytree, torch.tensor([1.2, 1.6]))


# --- fused kernels (CUDA + Triton) ------------------------------------------


@pytest.mark.cuda
@requires_fused
@pytest.mark.parametrize("dtypes", [FP32, BF16, MIXED], ids=["fp32", "bf16", "mixed"])
@pytest.mark.parametrize("return_aux", [False, True])
@pytest.mark.parametrize("microbatch_size", [None, 4])
def test_fused_matches_torch(dtypes, return_aux, microbatch_size, monkeypatch):
    tree = _tree(16, dtypes, seed=0)
    kwargs = {
        "clipping_norm": 1.0,
        "normalize_by": 16,
        "return_aux": return_aux,
        "microbatch_size": microbatch_size,
    }
    expected = _clip_without_fused(monkeypatch, tree, **kwargs)
    calls = _spy_fused(monkeypatch)
    actual = _clip(tree, **kwargs)
    assert calls, "the fused kernels did not run"
    if return_aux:
        (expected, expected_aux), (actual, actual_aux) = expected, actual
        torch.testing.assert_close(
            actual_aux.norms, expected_aux.norms, rtol=1e-6, atol=0
        )
        torch.testing.assert_close(
            actual_aux.clipped_norms, expected_aux.clipped_norms, rtol=1e-5, atol=1e-7
        )
    assert actual.max_norm == expected.max_norm
    for key in tree:
        assert actual.pytree[key].dtype == expected.pytree[key].dtype
        torch.testing.assert_close(actual.pytree[key], expected.pytree[key])


@pytest.mark.cuda
@requires_fused
def test_unsupported_calls_fall_back(monkeypatch):
    tree = {"w": torch.randn(8, 16, device="cuda", dtype=torch.float16)}
    expected = _clip_without_fused(monkeypatch, tree, clipping_norm=1.0)
    calls = _spy_fused(monkeypatch)
    actual = _clip(tree, clipping_norm=1.0)
    assert calls == []
    torch.testing.assert_close(actual.pytree["w"], expected.pytree["w"], rtol=0, atol=0)


@pytest.mark.cuda
@requires_fused
@pytest.mark.parametrize("dtypes", [FP32, BF16, MIXED], ids=["fp32", "bf16", "mixed"])
@pytest.mark.parametrize("ratio", [4.0, 1.0001, 0.99995])
def test_stored_values_respect_the_bound(dtypes, ratio):
    # With one example the batch sum is that example's stored clipped value,
    # so its norm (in float64) must not exceed C at all.
    for seed in range(10):
        tree = _tree(1, dtypes, seed, norms=(ratio, ratio))
        out = _clip(tree, clipping_norm=1.0)
        assert _stored_norm(out.pytree) <= 1.0


@pytest.mark.cuda
@requires_fused
@pytest.mark.parametrize(
    ("dtype", "mantissa"), [(torch.float32, 23), (torch.bfloat16, 7)]
)
def test_bound_holds_when_scaling_rounds_back(dtype, mantissa):
    # Every entry sits just above a power of two, where a scale closer to one
    # than half an ulp would round each product back to the input.
    n = 1024
    v = 2.0**-4 * (1 + 2.0**-mantissa)
    half_ulp = 2.0 ** -(mantissa + 1) / (1 + 2.0**-mantissa)
    signs = torch.where(
        torch.rand(n, generator=torch.Generator().manual_seed(0)) < 0.5, -1.0, 1.0
    )
    x = (signs * v).to(dtype).reshape(1, n).cuda()
    bound = v * math.sqrt(n) / (1 + 0.9 * half_ulp)
    out = _clip({"x": x}, clipping_norm=bound)
    assert _stored_norm(out.pytree) <= bound


@pytest.mark.cuda
@requires_fused
def test_fused_results_are_deterministic():
    tree = _tree(64, MIXED, seed=1)
    first = _clip(tree, clipping_norm=1.0)
    second = _clip(tree, clipping_norm=1.0)
    for key in tree:
        assert torch.equal(first.pytree[key], second.pytree[key])


@pytest.mark.cuda
@requires_fused
@pytest.mark.parametrize("microbatch_size", [None, 3])
def test_clipped_grad_automatic_dispatch_matches_torch(microbatch_size, monkeypatch):
    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.Linear(64, 8)
    ).cuda()
    params = {n: p.detach() for n, p in net.named_parameters()}
    x = torch.randn(12, 5, 32, device="cuda")
    y = torch.randn(12, 5, 8, device="cuda") * 3.0

    def loss_fn(p, xb, yb):
        return F.mse_loss(functional_call(net, p, (xb,)), yb)

    kwargs = {
        "batch_argnums": (1, 2),
        "clipping_norm": 0.5,
        "normalize_by": 12,
        "return_aux": True,
        "microbatch_size": microbatch_size,
    }
    with monkeypatch.context() as context:
        context.setattr(cf, "get_fused_clip_backend", lambda: None)
        torch_fn, torch_state = clipped_grad(loss_fn, **kwargs)
        expected, _ = torch_fn(params, x, y, state=torch_state)
    fn, state = clipped_grad(loss_fn, **kwargs)
    actual, _ = fn(params, x, y, state=state)
    (expected, expected_aux), (actual, actual_aux) = expected, actual
    for name in params:
        torch.testing.assert_close(actual.pytree[name], expected.pytree[name])
    torch.testing.assert_close(
        actual_aux.grad_norms, expected_aux.grad_norms, rtol=1e-6, atol=0
    )
    torch.testing.assert_close(
        actual_aux.clipped_grad_norms,
        expected_aux.clipped_grad_norms,
        rtol=1e-5,
        atol=1e-7,
    )


def _many_leaf_tree(batch, n_leaves, seed, *, mixed=False, noncontiguous=False):
    """``n_leaves`` tensors of varied sizes, some spanning several 2048-tiles."""
    gen = torch.Generator().manual_seed(seed)
    sizes = [(3, 5), (2049,), (64, 70), (1,), (7, 300), (4100,)]
    tree = {}
    for i in range(n_leaves):
        dtype = torch.bfloat16 if (mixed and i % 2) else torch.float32
        value = torch.randn(batch, *sizes[i % len(sizes)], generator=gen) * 0.05
        value = value.to(device="cuda", dtype=dtype)
        if noncontiguous and value.dim() == 3:
            value = value.transpose(1, 2).contiguous().transpose(1, 2)
        tree[f"p{i}"] = value
    return tree


@pytest.mark.cuda
@requires_fused
@pytest.mark.parametrize("mixed", [False, True], ids=["fp32", "mixed"])
def test_fused_many_tensors_match_torch(mixed, monkeypatch):
    tree = _many_leaf_tree(5, 23, seed=3, mixed=mixed, noncontiguous=True)
    assert any(not v.is_contiguous() for v in tree.values())
    kwargs = {"clipping_norm": 1.0, "return_aux": True}
    (expected, expected_aux) = _clip_without_fused(monkeypatch, tree, **kwargs)
    calls = _spy_fused(monkeypatch)
    (actual, actual_aux) = _clip(tree, **kwargs)
    assert calls, "the fused kernels did not run"
    torch.testing.assert_close(actual_aux.norms, expected_aux.norms, rtol=1e-6, atol=0)
    for key in tree:
        assert actual.pytree[key].shape == expected.pytree[key].shape
        torch.testing.assert_close(actual.pytree[key], expected.pytree[key])


@pytest.mark.cuda
@requires_fused
@pytest.mark.parametrize("mixed", [False, True], ids=["fp32", "mixed"])
def test_fused_many_tensors_respect_the_bound(mixed):
    tree = _many_leaf_tree(1, 23, seed=4, mixed=mixed)
    norm = _stored_norm(tree)
    tree = {k: (v.double() * (1.0001 / norm)).to(v.dtype) for k, v in tree.items()}
    out = _clip(tree, clipping_norm=1.0)
    assert _stored_norm(out.pytree) <= 1.0


def _cuda_kernels_per_call(fn, tree, state):
    fn(tree, state=state)  # warm-up: compile and cache the tile plan
    torch.cuda.synchronize()
    activity = [torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activity) as prof:
        fn(tree, state=state)
        torch.cuda.synchronize()
    cuda = torch.autograd.DeviceType.CUDA
    return sum(1 for e in prof.events() if e.device_type == cuda)


@pytest.mark.cuda
@requires_fused
@pytest.mark.parametrize("mixed", [False, True], ids=["fp32", "mixed"])
def test_fused_launches_do_not_grow_with_the_number_of_tensors(mixed):
    counts = []
    for n_leaves in (6, 60):
        fn, state = clipped_fun(
            lambda v: v,
            batch_argnums=0,
            clipping_norm=1.0,
            return_aux=True,
        )
        tree = _many_leaf_tree(4, n_leaves, seed=5, mixed=mixed)
        counts.append(_cuda_kernels_per_call(fn, tree, state))
    assert counts[0] == counts[1], counts


@pytest.mark.cuda
@requires_fused
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
def test_fused_falls_back_for_trees_across_devices(monkeypatch):
    tree = {
        "a": torch.randn(4, 8, device="cuda:0"),
        "b": torch.randn(4, 8, device="cuda:1"),
    }
    calls = _spy_fused(monkeypatch)
    expected = _clip_without_fused(monkeypatch, tree, clipping_norm=1.0)
    actual = _clip(tree, clipping_norm=1.0)
    assert calls == []
    for key in tree:
        torch.testing.assert_close(
            actual.pytree[key], expected.pytree[key], rtol=0, atol=0
        )


@pytest.mark.cuda
@requires_fused
def test_fused_tile_plan_cache_is_bounded():
    module = importlib.import_module("opake.api.patches.kernels._clip_sum")
    for width in range(1, module._MAX_PLANS + 6):
        _clip({"w": torch.randn(2, width, device="cuda")}, clipping_norm=1.0)
    assert len(module._PLANS) <= module._MAX_PLANS
