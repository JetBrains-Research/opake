"""Compare streamed reductions to the original vmap(clip_pytree) path."""

from __future__ import annotations

import importlib

import pytest
import torch
from torch.func import grad, vmap

from opake.api.engine.clipping import auto_clipped_grad, clipped_grad
from opake.api.engine.clipping._clipped_fun import _RuntimeClipState, clipped_fun
from opake.api.engine.clipping._pytree import clip_pytree
from opake.pytree import global_norm, tree_leaves, tree_map
from opake.types import ClippedPytree, PerGroup

cf = importlib.import_module("opake.api.engine.clipping._clipped_fun")


def _assert_tree_equal(actual, expected):
    actual_leaves = tree_leaves(actual)
    expected_leaves = tree_leaves(expected)
    assert actual_leaves
    assert len(actual_leaves) == len(expected_leaves)
    for a, b in zip(actual_leaves, expected_leaves, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)


def _reference(fn, args, state, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(cf, "_streaming_supported", lambda *args: False)
        return fn(*args, state=state)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("microbatch_size", [None, 1, 3, 4, 20])
@pytest.mark.parametrize("compute_dtype", [None, torch.float32, torch.float64])
def test_adversarial_values_match_original(
    dtype, microbatch_size, compute_dtype, monkeypatch
):
    info = torch.finfo(dtype)
    values = torch.tensor(
        [
            [0, 0, 0, 0],
            [0.1, -0.2, 0.05, 0.125],
            [0.5, 0.5, 0.5, 0.5],
            [3, -4, 12, 0],
            [float("nan"), float("inf"), -float("inf"), 1],
            [info.tiny / 2, -info.tiny / 4, info.tiny, 0],
            [info.max / 4, -info.max / 8, 0, 0],
        ],
        dtype=dtype,
    )
    # Shared, noncontiguous views exercise input ownership and repeated leaves.
    tree = {"a": values[:, :2], "nested": (values[:, 2:], values[:, :2])}
    before = tree_map(torch.clone, tree)
    fn, state = clipped_fun(
        lambda x: (
            x,
            {"values": torch.tensor(0.5), "value_aux": {"loss": torch.tensor(2.0)}},
        ),
        clipping_norm=1.0,
        has_aux=True,
        return_aux=True,
        microbatch_size=microbatch_size,
        compute_dtype=compute_dtype,
    )
    (expected, expected_aux), old_state = _reference(fn, (tree,), state, monkeypatch)
    (actual, actual_aux), new_state = fn(tree, state=state)
    assert isinstance(actual, ClippedPytree)
    assert new_state is old_state is state
    assert actual.max_norm == expected.max_norm
    _assert_tree_equal(actual.pytree, expected.pytree)
    assert all(torch.isfinite(leaf).all() for leaf in tree_leaves(actual.pytree))
    for field in ("values", "norms", "clipped_norms", "value_aux"):
        _assert_tree_equal(getattr(actual_aux, field), getattr(expected_aux, field))
    assert actual_aux.batch_size == expected_aux.batch_size == 7
    assert actual_aux.clipping_rate == expected_aux.clipping_rate
    assert actual_aux.group_norms is expected_aux.group_norms is None
    _assert_tree_equal(tree, before)

    # Compare individual stored contributions, not just a potentially cancelling sum.
    single, single_state = clipped_fun(
        lambda x: x,
        clipping_norm=1.0,
        return_aux=True,
        microbatch_size=1,
        compute_dtype=compute_dtype,
        dtype=dtype,
    )
    for index in range(7):
        example = tree_map(lambda x, i=index: x[i], tree)
        (stored, aux), _ = single(
            tree_map(lambda x: x.unsqueeze(0), example), state=single_state
        )
        reference, _ = clip_pytree(example, 1.0, compute_dtype=compute_dtype)
        _assert_tree_equal(stored.pytree, reference)
        # Strict sensitivity assertion, with no tolerance that relaxes the bound.
        assert global_norm(stored.pytree, compute_dtype=torch.float64) <= 1.0
        torch.testing.assert_close(
            aux.clipped_norms[0],
            global_norm(stored.pytree, compute_dtype=compute_dtype),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("diagnostics", ["none", "stats", "aux"])
@pytest.mark.parametrize("microbatch_size", [None, 1, 3, 4])
@pytest.mark.parametrize("output_dtype", [None, torch.float32, torch.float64])
def test_grad_loss_state_and_mixed_dtypes(
    diagnostics, microbatch_size, output_dtype, monkeypatch
):
    generator = torch.Generator().manual_seed(118)
    params = {
        "a": torch.randn(7, generator=generator, dtype=torch.float64),
        "b": torch.randn(7, generator=generator),
    }
    x = torch.randn(10, 7, generator=generator)

    def loss(p, x):
        value = sum((leaf * x).sum() for leaf in p.values()).square()
        return value, {"prediction": value.detach() + 1}

    fn, state = clipped_grad(
        loss,
        has_aux=True,
        clipping_norm=0.75,
        normalize_by=16,
        microbatch_size=microbatch_size,
        dtype=output_dtype,
        compute_dtype=torch.float32,
        return_aux=diagnostics == "aux",
        return_stats=diagnostics == "stats",
    )
    expected, old_state = _reference(fn, (params, x), state, monkeypatch)
    actual, new_state = fn(params, x, state=state)
    assert new_state is old_state is state
    if diagnostics != "none":
        actual, aux = actual
        expected, expected_aux = expected
        assert aux.clipping_rate == expected_aux.clipping_rate
        assert aux.batch_size == expected_aux.batch_size == 10
        if diagnostics == "aux":
            for field in (
                "loss_values",
                "loss_aux",
                "grad_norms",
                "clipped_grad_norms",
            ):
                _assert_tree_equal(getattr(aux, field), getattr(expected_aux, field))
        else:
            assert aux.num_clipped == expected_aux.num_clipped
    assert actual.max_norm == expected.max_norm == 0.75 / 16
    _assert_tree_equal(actual.pytree, expected.pytree)


@pytest.mark.parametrize(
    "kind", ["grouped", "second_moment", "custom", "adaptive", "auto"]
)
def test_unsupported_modes_keep_original_path(kind, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail(f"{kind} must not dispatch to streaming")

    monkeypatch.setattr(cf, "_stream_clip_and_sum", forbidden)
    options = {"return_aux": True, "microbatch_size": 2}
    if kind == "auto":
        fn, state = auto_clipped_grad(lambda p, x: (p * x).sum(), **options)
        (result, aux), _ = fn(torch.ones(3), torch.ones(5, 3), state=state)
        assert tree_leaves(result.pytree)
        assert aux.grad_norms.numel() == 5
        return
    if kind == "grouped":
        options["clipping_norm"] = PerGroup(groups={"w": "all"}, values={"all": 1.0})
    elif kind == "second_moment":
        options["second_moment"] = True
    elif kind == "custom":
        options["_scale_fn"] = lambda x: clip_pytree(x, 1.0)
    fn, state = clipped_fun(lambda x: x, **options)
    if kind == "adaptive":
        state = _RuntimeClipState(clipping_norm=0.75)
    (result, aux), returned = fn({"w": torch.ones(5, 3)}, state=state)
    assert returned is state
    assert aux.norms.numel() == 5
    assert tree_leaves(
        result.grads.pytree if kind == "second_moment" else result.pytree
    )


def test_nested_transforms_match_original(monkeypatch):
    fn, state = clipped_fun(lambda p, x: p * x, batch_argnums=1, microbatch_size=2)

    def outer(p, x):
        result, _ = fn(p, x, state=state)
        return result.pytree.sum()

    p = torch.tensor([0.25, -0.5])
    x = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2) / 4
    with monkeypatch.context() as patch:
        patch.setattr(cf, "_streaming_supported", lambda *args: False)
        expected = vmap(grad(outer), in_dims=(None, 0))(p, x)
    actual = vmap(grad(outer), in_dims=(None, 0))(p, x)
    _assert_tree_equal(actual, expected)


@pytest.mark.parametrize("bound", [float("inf"), 1e-35, 1.0])
@pytest.mark.parametrize("kind", ["complex", "integer"])
def test_complex_and_integer_function_values(bound, kind, monkeypatch):
    tree = {
        "a": torch.tensor([[1 + 2j, 3 + 4j], [0j, 1j]], dtype=torch.complex64),
        "b": torch.ones(2, 3, dtype=torch.float64),
    }
    if kind == "integer":
        tree = {
            "a": torch.tensor([[0, 1], [3, 4]]),
            "b": torch.ones(2, 3, dtype=torch.int32),
        }
    fn, state = clipped_fun(
        lambda x: x, clipping_norm=bound, return_aux=True, microbatch_size=1
    )
    (expected, expected_aux), _ = _reference(fn, (tree,), state, monkeypatch)
    (actual, aux), _ = fn(tree, state=state)
    _assert_tree_equal(actual.pytree, expected.pytree)
    _assert_tree_equal(aux.norms, expected_aux.norms)
    _assert_tree_equal(aux.clipped_norms, expected_aux.clipped_norms)


@pytest.mark.parametrize("microbatch_size", [None, 2])
def test_empty_value_tree_preserves_diagnostics(microbatch_size, monkeypatch):
    fn, state = clipped_fun(
        lambda x: {}, return_aux=True, microbatch_size=microbatch_size
    )
    (expected, expected_aux), _ = _reference(
        fn, (torch.ones(3, 2),), state, monkeypatch
    )
    (actual, aux), _ = fn(torch.ones(3, 2), state=state)
    assert actual.pytree == expected.pytree == {}
    assert aux.values == expected_aux.values == {}
    assert aux.batch_size == expected_aux.batch_size == 3
    _assert_tree_equal(aux.norms, expected_aux.norms)
    _assert_tree_equal(aux.clipped_norms, expected_aux.clipped_norms)


def test_large_leaves_preserve_blocked_norm_reduction(monkeypatch):
    generator = torch.Generator().manual_seed(772)
    x = {
        "a": torch.randn(5, 1025, generator=generator),
        "b": torch.randn(5, 2049, generator=generator),
    }
    fn, state = clipped_fun(
        lambda x: x, return_aux=True, microbatch_size=2, clipping_norm=1.0
    )
    (expected, expected_aux), _ = _reference(fn, (x,), state, monkeypatch)
    (actual, aux), _ = fn(x, state=state)
    _assert_tree_equal(actual.pytree, expected.pytree)
    _assert_tree_equal(aux.norms, expected_aux.norms)
    _assert_tree_equal(aux.clipped_norms, expected_aux.clipped_norms)


def test_compiled_chunk_matches_original_with_nonfinite_inputs(monkeypatch):
    def compiler(kernel):
        return torch.compile(kernel, fullgraph=True, dynamic=True, backend="aot_eager")

    def loss(p, x):
        return (p * x).sum()

    p = torch.ones(7)
    x = torch.arange(70, dtype=torch.float32).reshape(10, 7)
    x[0, 0] = float("nan")
    x[1, 1] = float("inf")
    fn, state = clipped_grad(
        loss,
        clipping_norm=1.0,
        return_aux=True,
        microbatch_size=4,
        _chunk_compiler=compiler,
    )
    (expected, expected_aux), _ = _reference(fn, (p, x), state, monkeypatch)
    (actual, aux), _ = fn(p, x, state=state)
    _assert_tree_equal(actual.pytree, expected.pytree)
    for field in ("loss_values", "grad_norms", "clipped_grad_norms"):
        _assert_tree_equal(getattr(aux, field), getattr(expected_aux, field))


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cuda", marks=pytest.mark.cuda),
        pytest.param("mps", marks=pytest.mark.mps),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_accelerator_matches_original(device, dtype, monkeypatch):
    generator = torch.Generator().manual_seed(923)
    x = {
        "a": torch.randn(13, 1025, generator=generator).to(device=device, dtype=dtype),
        "b": torch.randn(13, 17, generator=generator).to(device=device, dtype=dtype),
    }
    fn, state = clipped_fun(lambda x: x, return_aux=True, microbatch_size=4)
    (expected, expected_aux), _ = _reference(fn, (x,), state, monkeypatch)
    (actual, aux), _ = fn(x, state=state)
    _assert_tree_equal(actual.pytree, expected.pytree)
    _assert_tree_equal(aux.norms, expected_aux.norms)
    _assert_tree_equal(aux.clipped_norms, expected_aux.clipped_norms)
    for index in range(13):
        example = tree_map(lambda leaf, i=index: leaf[i : i + 1], x)
        (stored, _), _ = fn(example, state=state)
        cpu_values = tree_map(lambda leaf: leaf.cpu().to(torch.float64), stored.pytree)
        assert tree_leaves(cpu_values)
        assert global_norm(cpu_values) <= 1.0
