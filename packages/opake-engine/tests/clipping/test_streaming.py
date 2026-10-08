"""Compare streamed reductions to the original vmap(clip_pytree) path."""

from __future__ import annotations

import importlib

import pytest
import torch
from torch.func import grad, vmap

from opake.api.engine.clipping import auto_clipped_grad, clipped_grad
from opake.api.engine.clipping._clipped_fun import _RuntimeClipState, clipped_fun
from opake.api.engine.clipping._pytree import clip_pytree
from opake.api.engine.clipping.fun import auto_clipped_fun
from opake.exceptions import ConfigurationError
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


def _stable_norm(tensor):
    value = tensor.detach().cpu().double()
    scale = float(value.abs().max())
    if scale == 0.0:
        return 0.0
    return scale * float(((value / scale) ** 2).sum()) ** 0.5


def _reference(fn, args, state, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(cf, "_streaming_supported", lambda *args: False)
        return fn(*args, state=state)


@pytest.mark.parametrize(
    ("dtype", "microbatch_size", "compute_dtype"),
    [
        (torch.float16, None, None),
        (torch.float16, 1, torch.float32),
        (torch.bfloat16, 3, torch.float64),
        (torch.float32, 4, None),
        (torch.float64, 20, torch.float32),
    ],
)
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


@pytest.mark.parametrize("mode", ["fixed", "auto"])
@pytest.mark.parametrize("per_group", [False, True])
@pytest.mark.parametrize("microbatch_size", [None, 1])
def test_streaming_scaling_covers_square_underflow(mode, per_group, microbatch_size):
    tiny = torch.full((1024,), 1e-23)
    values = torch.stack((tiny, torch.zeros_like(tiny)))
    true_norm = _stable_norm(tiny)
    bound = 1.0 if mode == "auto" else 0.84 * true_norm
    configured = (
        PerGroup(groups={"w": "tiny"}, values={"tiny": bound}) if per_group else bound
    )
    options = {"microbatch_size": microbatch_size, "return_aux": True}
    if mode == "auto":
        fn, state = auto_clipped_fun(
            lambda x: {"w": x},
            R=configured,
            gamma=true_norm / 100.0,
            **options,
        )
    else:
        fn, state = clipped_fun(lambda x: {"w": x}, clipping_norm=configured, **options)

    (result, aux), _ = fn(values, state=state)

    assert torch.equal(aux.norms, torch.zeros(2))
    if per_group:
        assert torch.equal(aux.group_norms["tiny"], torch.zeros(2))
    assert _stable_norm(result.pytree["w"]) <= bound


@pytest.mark.parametrize("microbatch_size", [None, 2])
@pytest.mark.parametrize("return_stats", [False, True])
def test_discarded_non_tensor_aux_is_not_batched(
    microbatch_size, return_stats, monkeypatch
):
    def with_metadata(x):
        return x, "discarded metadata"

    fn, state = clipped_fun(
        with_metadata,
        has_aux=True,
        return_aux=False,
        return_stats=return_stats,
        microbatch_size=microbatch_size,
        clipping_norm=1.0,
    )
    x = torch.tensor([[0.25, 0.5], [3.0, 4.0], [-0.5, 0.0]])

    expected, old_state = _reference(fn, (x,), state, monkeypatch)
    actual, new_state = fn(x, state=state)

    assert new_state is old_state is state
    if return_stats:
        actual, actual_stats = actual
        expected, expected_stats = expected
        assert actual_stats == expected_stats
    _assert_tree_equal(actual.pytree, expected.pytree)


@pytest.mark.parametrize(
    ("diagnostics", "microbatch_size", "output_dtype"),
    [
        ("none", None, None),
        ("stats", 1, torch.float32),
        ("aux", 3, torch.float64),
        ("none", 4, torch.float64),
        ("stats", None, None),
        ("aux", 4, torch.float32),
    ],
)
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
    ("diagnostics", "microbatch_size"),
    [("none", None), ("stats", 2), ("aux", 2)],
)
def test_runtime_thresholds_stream_and_match_original(
    diagnostics, microbatch_size, monkeypatch
):
    values = torch.tensor(
        [
            [0.25, -0.5, 1.0],
            [float("nan"), float("inf"), -float("inf")],
            [3.0, 4.0, 0.0],
            [-0.125, 0.0, 0.5],
        ]
    )
    tree = {"a": values[:, :2], "b": values[:, 2:]}
    fn, _ = clipped_fun(
        lambda x: x,
        clipping_norm=9.0,
        normalize_by=4,
        microbatch_size=microbatch_size,
        return_aux=diagnostics == "aux",
        return_stats=diagnostics == "stats",
    )
    streamed_thresholds = []
    stream_clip_and_sum = cf._stream_clip_and_sum

    def observed(*args, **kwargs):
        streamed_thresholds.append(args[1].item())
        return stream_clip_and_sum(*args, **kwargs)

    monkeypatch.setattr(cf, "_stream_clip_and_sum", observed)
    for threshold in (0.75, 1.5):
        state = _RuntimeClipState(clipping_norm=threshold)
        expected, _ = _reference(fn, (tree,), state, monkeypatch)
        actual, returned = fn(tree, state=state)
        assert returned is state
        if diagnostics != "none":
            actual, aux = actual
            expected, expected_aux = expected
            assert aux.clipping_rate == expected_aux.clipping_rate
            assert aux.batch_size == expected_aux.batch_size == 4
            if diagnostics == "aux":
                for field in ("values", "norms", "clipped_norms"):
                    _assert_tree_equal(
                        getattr(aux, field), getattr(expected_aux, field)
                    )
            else:
                assert aux.num_clipped == expected_aux.num_clipped
        assert actual.max_norm == expected.max_norm == threshold / 4
        _assert_tree_equal(actual.pytree, expected.pytree)

    chunks_per_call = 1 if microbatch_size is None else 2
    assert streamed_thresholds == [0.75] * chunks_per_call + [1.5] * chunks_per_call


@pytest.mark.parametrize(
    ("diagnostics", "microbatch_size"),
    [("none", None), ("stats", 2), ("aux", 2)],
)
def test_global_auto_s_streams_and_matches_original(
    diagnostics, microbatch_size, monkeypatch
):
    params = {
        "a": torch.tensor([0.25, -0.5], dtype=torch.float64),
        "b": torch.tensor([1.0], dtype=torch.float32),
    }
    values = torch.tensor(
        [
            [0.0, 0.5, -1.0],
            [float("nan"), float("inf"), -float("inf")],
            [3.0, 4.0, 0.0],
            [-0.125, 0.0, 0.5],
        ]
    )

    def loss(p, x):
        return (p["a"] * x[:2]).sum() + (p["b"] * x[2:]).sum()

    fn, state = auto_clipped_grad(
        loss,
        R=0.75,
        gamma=0.2,
        normalize_by=4,
        microbatch_size=microbatch_size,
        return_aux=diagnostics == "aux",
        return_stats=diagnostics == "stats",
    )
    calls = 0
    stream_clip_and_sum = cf._stream_clip_and_sum

    def observed(*args, **kwargs):
        nonlocal calls
        calls += 1
        return stream_clip_and_sum(*args, **kwargs)

    expected, _ = _reference(fn, (params, values), state, monkeypatch)
    monkeypatch.setattr(cf, "_stream_clip_and_sum", observed)
    actual, returned = fn(params, values, state=state)

    assert returned is state
    if diagnostics != "none":
        actual, aux = actual
        expected, expected_aux = expected
        assert aux.clipping_rate == expected_aux.clipping_rate
        assert aux.batch_size == expected_aux.batch_size == 4
        if diagnostics == "aux":
            for field in (
                "loss_values",
                "grad_norms",
                "clipped_grad_norms",
            ):
                _assert_tree_equal(getattr(aux, field), getattr(expected_aux, field))
        else:
            assert aux.num_clipped == expected_aux.num_clipped
    assert actual.max_norm == expected.max_norm == 0.75 / 4
    _assert_tree_equal(actual.pytree, expected.pytree)
    assert calls == (1 if microbatch_size is None else 2)


@pytest.mark.parametrize("mode", ["fixed", "auto"])
@pytest.mark.parametrize("microbatch_size", [None, 1])
def test_float64_radius_preserves_advertised_bound(mode, microbatch_size):
    def loss(p, x):
        return (p * x).sum()

    options = {"clipping_norm": 0.1}
    factory = clipped_grad
    if mode == "auto":
        factory = auto_clipped_grad
        options = {"R": 0.1, "gamma": 0.01}
    fn, state = factory(loss, microbatch_size=microbatch_size, **options)

    params = torch.ones(1, dtype=torch.float64)
    values = torch.tensor([[1e8]], dtype=torch.float64)
    actual, _ = fn(params, values, state=state)

    assert actual.max_norm == 0.1
    assert global_norm(actual.pytree, compute_dtype=torch.float64) <= actual.max_norm


@pytest.mark.parametrize(
    ("mode", "microbatch_size", "diagnostics"),
    [
        ("fixed", None, "none"),
        ("fixed", 2, "aux"),
        ("runtime", None, "stats"),
        ("runtime", 2, "none"),
        ("auto", None, "aux"),
        ("auto", 2, "stats"),
    ],
)
def test_global_paired_moments_stream_and_match_original(
    mode, microbatch_size, diagnostics, monkeypatch
):
    params = {
        "a": torch.tensor([0.25, -0.5], dtype=torch.float64),
        "b": torch.tensor([1.0], dtype=torch.float32),
    }
    values = torch.tensor(
        [
            [0.0, 0.5, -1.0],
            [float("nan"), float("inf"), -float("inf")],
            [3.0, 4.0, 0.0],
            [-0.125, 0.0, 0.5],
        ]
    )

    def loss(p, x):
        return (p["a"] * x[:2]).sum() + (p["b"] * x[2:]).sum()

    options = {
        "normalize_by": 4,
        "microbatch_size": microbatch_size,
        "second_moment": True,
        "return_aux": diagnostics == "aux",
        "return_stats": diagnostics == "stats",
    }
    if mode == "auto":
        fn, state = auto_clipped_grad(loss, R=0.75, gamma=0.2, **options)
    else:
        fn, state = clipped_grad(loss, clipping_norm=0.75, **options)
        if mode == "runtime":
            state = _RuntimeClipState(clipping_norm=0.5)

    calls = 0
    stream_clip_and_sum = cf._stream_clip_and_sum

    def observed(*args, **kwargs):
        nonlocal calls
        calls += 1
        return stream_clip_and_sum(*args, **kwargs)

    expected, _ = _reference(fn, (params, values), state, monkeypatch)
    monkeypatch.setattr(cf, "_stream_clip_and_sum", observed)
    actual, returned = fn(params, values, state=state)

    assert returned is state
    if diagnostics != "none":
        actual, aux = actual
        expected, expected_aux = expected
        assert aux.clipping_rate == expected_aux.clipping_rate
        assert aux.batch_size == expected_aux.batch_size == 4
        if diagnostics == "aux":
            for field in (
                "loss_values",
                "grad_norms",
                "clipped_grad_norms",
            ):
                _assert_tree_equal(getattr(aux, field), getattr(expected_aux, field))
        else:
            assert aux.num_clipped == expected_aux.num_clipped
    assert actual.grads.max_norm == expected.grads.max_norm
    assert actual.squared_grads.max_norm == expected.squared_grads.max_norm
    _assert_tree_equal(actual.grads.pytree, expected.grads.pytree)
    _assert_tree_equal(actual.squared_grads.pytree, expected.squared_grads.pytree)
    assert calls == (1 if microbatch_size is None else 2)


@pytest.mark.parametrize(
    ("mode", "second_moment", "microbatch_size", "diagnostics"),
    [
        ("fixed", False, None, "none"),
        ("fixed", True, 2, "aux"),
        ("runtime", False, 2, "stats"),
        ("runtime", True, None, "aux"),
        ("auto", False, None, "aux"),
        ("auto", True, 2, "stats"),
    ],
)
def test_per_group_streams_and_matches_original(
    mode, second_moment, microbatch_size, diagnostics, monkeypatch
):
    values = torch.tensor(
        [
            [0.0, 0.5, -1.0, 2.0],
            [float("nan"), float("inf"), -float("inf"), 0.25],
            [3.0, 4.0, 0.0, -2.0],
            [-0.125, 0.0, 0.5, 0.75],
        ]
    )
    tree = {
        "a": values[:, :1].double(),
        "b": values[:, 1:3],
        "c": values[:, 3:].to(torch.bfloat16),
    }
    configured = PerGroup(
        groups={"a": "shared", "b": "shared", "c": "other"},
        values={"shared": 0.75, "other": 0.5},
    )
    params = {name: torch.ones_like(leaf[0]) for name, leaf in tree.items()}

    def loss(p, x):
        return sum((p[name] * leaf).sum() for name, leaf in x.items())

    options = {
        "normalize_by": 4,
        "microbatch_size": microbatch_size,
        "second_moment": second_moment,
        "return_aux": diagnostics == "aux",
        "return_stats": diagnostics == "stats",
    }
    if mode == "auto":
        fn, state = auto_clipped_grad(loss, R=configured, gamma=0.2, **options)
    else:
        fn, state = clipped_grad(loss, clipping_norm=configured, **options)
    current = configured
    if mode == "runtime":
        current = PerGroup(
            groups=configured.groups,
            values={"shared": 1.25, "other": 0.25},
        )
        state = _RuntimeClipState(clipping_norm=current)

    calls = 0
    stream_clip_and_sum = cf._stream_clip_and_sum

    def observed(*args, **kwargs):
        nonlocal calls
        calls += 1
        return stream_clip_and_sum(*args, **kwargs)

    expected, _ = _reference(fn, (params, tree), state, monkeypatch)
    monkeypatch.setattr(cf, "_stream_clip_and_sum", observed)
    actual, returned = fn(params, tree, state=state)

    assert returned is state
    if diagnostics != "none":
        actual, aux = actual
        expected, expected_aux = expected
        assert aux.clipping_rate == expected_aux.clipping_rate
        assert aux.batch_size == expected_aux.batch_size == 4
        if diagnostics == "aux":
            for field in (
                "loss_values",
                "grad_norms",
                "clipped_grad_norms",
                "group_norms",
            ):
                _assert_tree_equal(getattr(aux, field), getattr(expected_aux, field))
        else:
            assert aux.num_clipped == expected_aux.num_clipped
    if second_moment:
        assert actual.grads.max_norm == expected.grads.max_norm == current / 4
        assert (
            actual.squared_grads.max_norm
            == expected.squared_grads.max_norm
            == (current * current) / 4
        )
        _assert_tree_equal(actual.grads.pytree, expected.grads.pytree)
        _assert_tree_equal(actual.squared_grads.pytree, expected.squared_grads.pytree)
    else:
        assert actual.max_norm == expected.max_norm == current / 4
        _assert_tree_equal(actual.pytree, expected.pytree)
    assert calls == (1 if microbatch_size is None else 2)


def test_per_group_stream_preserves_path_validation(monkeypatch):
    clipping_norm = PerGroup(groups={"a": "all", "missing": "all"}, values={"all": 1.0})
    fn, state = clipped_fun(lambda x: x, clipping_norm=clipping_norm)
    tree = {"a": torch.ones(2, 3), "b": torch.ones(2, 3)}

    with pytest.raises(ConfigurationError) as reference_error:
        _reference(fn, (tree,), state, monkeypatch)
    with pytest.raises(type(reference_error.value)) as actual_error:
        fn(tree, state=state)
    assert str(actual_error.value) == str(reference_error.value)


@pytest.mark.parametrize(
    "kind",
    [
        "custom",
        "runtime_custom",
    ],
)
def test_unsupported_modes_keep_original_path(kind, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail(f"{kind} must not dispatch to streaming")

    monkeypatch.setattr(cf, "_stream_clip_and_sum", forbidden)
    options = {"return_aux": True, "microbatch_size": 2}
    if "custom" in kind:
        options["_scale_fn"] = lambda x: clip_pytree(x, 1.0)
    fn, state = clipped_fun(lambda x: x, **options)
    if kind.startswith("runtime_"):
        state = _RuntimeClipState(clipping_norm=0.75)
    (result, aux), returned = fn({"w": torch.ones(5, 3)}, state=state)
    assert returned is state
    assert aux.norms.numel() == 5
    assert tree_leaves(result.pytree)


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
        "a": torch.randn(5, 4097, generator=generator),
        "b": torch.randn(5, 8193, generator=generator),
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
