"""Numerical guarantees of clipping under microbatching and low precision.

Two properties, swept over tree shapes, norms and dtypes (issue #343):

- ``norm(clipped) <= clipping_norm`` holds on the values *as stored*.
- Microbatched accumulation runs in ``compute_dtype``.
"""

from __future__ import annotations

import pytest
import torch
from torch.func import grad

from opaque.api.engine.clipping import _pytree
from opaque.api.engine.clipping._clipped_fun import clipped_fun
from opaque.api.engine.clipping._pytree import auto_scale_pytree, clip_pytree
from opaque.pytree import tree_leaves, tree_map
from opaque.types import ClippedPytree, PerGroup

LOW_PRECISION_DTYPES = [torch.bfloat16, torch.float16]
ALL_DTYPES = [*LOW_PRECISION_DTYPES, torch.float32, torch.float64]

# Tree shapes: scalar, vector, matrix, nested, and a ragged multi-leaf tree.
TREE_SHAPES = {
    "scalar": lambda g, d: {"a": torch.randn((), generator=g).to(d)},
    "vector": lambda g, d: {"a": torch.randn(17, generator=g).to(d)},
    "matrix": lambda g, d: {"a": torch.randn(8, 12, generator=g).to(d)},
    "nested": lambda g, d: {
        "enc": {"w": torch.randn(5, 3, generator=g).to(d)},
        "dec": {"w": torch.randn(7, generator=g).to(d), "b": torch.zeros(2, dtype=d)},
    },
    "ragged": lambda g, d: {
        f"p{i}": torch.randn(i + 1, generator=g).to(d) for i in range(6)
    },
}

# Spans the regime where the scale is comfortably normal through the regime
# where it underflows into float16 subnormals (||x|| / C > ~1.6e4).
NORM_SCALES = [1e-3, 1.0, 1e2, 1e4, 1e6]


def _exact_norm(pytree) -> float:
    """L2 norm of the stored values, computed without further rounding."""
    total = torch.zeros((), dtype=torch.float64)
    for leaf in tree_leaves(pytree):
        total = total + (leaf.to(torch.float64) ** 2).sum()
    return float(torch.sqrt(total))


def _unwrap(value):
    assert isinstance(value, ClippedPytree)
    return value.pytree


# ----------------------------------------------------------------------------
# The bound holds in the stored dtype
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("shape", list(TREE_SHAPES))
def test_clip_pytree_bound_holds_in_stored_dtype(dtype, shape):
    """``norm(out) <= C`` must hold exactly, with no tolerance allowance."""
    generator = torch.Generator().manual_seed(0)
    for magnitude in NORM_SCALES:
        for clipping_norm in (0.01, 1.0, 7.5):
            pytree = TREE_SHAPES[shape](generator, dtype)
            pytree = tree_map(lambda leaf, m=magnitude: (leaf * m).to(dtype), pytree)
            clipped, _ = clip_pytree(pytree, clipping_norm=clipping_norm)
            assert _exact_norm(clipped) <= clipping_norm, (
                f"{dtype} {shape} magnitude={magnitude} C={clipping_norm}"
            )


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_auto_scale_pytree_bound_holds_in_stored_dtype(dtype):
    """AUTO-S has no ``min(1, .)``, so its bound leans entirely on the guard."""
    generator = torch.Generator().manual_seed(1)
    for magnitude in NORM_SCALES:
        for R in (0.01, 1.0, 7.5):
            pytree = {"w": (torch.randn(64, generator=generator) * magnitude).to(dtype)}
            scaled, _ = auto_scale_pytree(pytree, R=R, gamma=0.01)
            assert _exact_norm(scaled) <= R, f"{dtype} magnitude={magnitude} R={R}"


@pytest.mark.parametrize("dtype", LOW_PRECISION_DTYPES)
def test_per_group_clip_bound_holds_per_group(dtype):
    """Each group must satisfy its own bound in the stored dtype."""
    generator = torch.Generator().manual_seed(2)
    values = {"attn": 0.5, "mlp": 2.0}
    groups = PerGroup(
        groups={"attn.q": "attn", "attn.k": "attn", "mlp.w": "mlp"},
        values=values,
    )
    for magnitude in NORM_SCALES:
        pytree = {
            "attn.q": (torch.randn(11, generator=generator) * magnitude).to(dtype),
            "attn.k": (torch.randn(4, 3, generator=generator) * magnitude).to(dtype),
            "mlp.w": (torch.randn(9, generator=generator) * magnitude).to(dtype),
        }
        clipped, _ = clip_pytree(pytree, clipping_norm=groups)
        assert (
            _exact_norm({k: clipped[k] for k in ("attn.q", "attn.k")}) <= values["attn"]
        )
        assert _exact_norm({"mlp.w": clipped["mlp.w"]}) <= values["mlp"]


@pytest.mark.parametrize("dtype", LOW_PRECISION_DTYPES)
def test_mixed_dtype_tree_bound_holds(dtype):
    """A mixed-precision tree must still satisfy the bound as stored."""
    generator = torch.Generator().manual_seed(3)
    for magnitude in NORM_SCALES:
        pytree = {
            "low": (torch.randn(20, generator=generator) * magnitude).to(dtype),
            "high": (torch.randn(20, generator=generator) * magnitude).double(),
        }
        clipped, _ = clip_pytree(pytree, clipping_norm=1.0)
        assert _exact_norm(clipped) <= 1.0


@pytest.mark.parametrize("n_leaves", [1, 50, 200, 600])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_bound_holds_as_leaf_count_grows(n_leaves, dtype):
    """The guard must cover the norm reduction, whose error grows with leaves.

    A realistic model has hundreds of parameter tensors.
    """
    generator = torch.Generator().manual_seed(8)
    for trial in range(20):
        pytree = {
            f"p{i}": (torch.randn(64, generator=generator, dtype=torch.float64) * 10)
            .to(dtype)
            .clone()
            for i in range(n_leaves)
        }
        clipped, _ = clip_pytree(pytree, clipping_norm=1.0)
        assert _exact_norm(clipped) <= 1.0, f"{dtype} n_leaves={n_leaves} trial={trial}"


@pytest.mark.parametrize("low", LOW_PRECISION_DTYPES)
def test_low_precision_leaf_does_not_zero_its_neighbours(low):
    """The subnormal guard is per leaf, so one narrow leaf cannot zero the rest."""
    pytree = {
        "narrow": torch.tensor([1.0], dtype=low),
        "f32": torch.tensor([1000.0, 0.0], dtype=torch.float32),
        "f64": torch.tensor([1000.0, 0.0], dtype=torch.float64),
    }
    clipped, _ = clip_pytree(pytree, clipping_norm=1e-6)

    assert _exact_norm(clipped) <= 1e-6
    for name in ("f32", "f64"):
        assert clipped[name].abs().sum() > 0, f"{name} was zeroed by the {low} leaf"


# ----------------------------------------------------------------------------
# Documented edge cases stay exact
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_guard_does_not_perturb_unclipped_inputs(dtype):
    """The guard is applied before ``min(1, .)``, so passthrough stays exact."""
    pytree = {"w": torch.tensor([0.03, -0.04], dtype=dtype)}

    unclipped, _ = clip_pytree(pytree, clipping_norm=1000.0)
    assert torch.equal(unclipped["w"], pytree["w"])

    infinite, _ = clip_pytree(pytree, clipping_norm=float("inf"))
    assert torch.equal(infinite["w"], pytree["w"])

    zeros = {"w": torch.zeros(2, dtype=dtype)}
    zero_norm, _ = clip_pytree(zeros, clipping_norm=1.0)
    assert torch.equal(zero_norm["w"], zeros["w"])


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_zero_clipping_norm_returns_exact_zeros(dtype):
    """The absolute term must not push the scale negative at ``C=0``."""
    pytree = {"w": torch.tensor([3.0, 4.0], dtype=dtype)}
    clipped, _ = clip_pytree(pytree, clipping_norm=0.0)
    assert torch.equal(clipped["w"], torch.zeros(2, dtype=dtype))


def test_guard_is_negligible_at_float32():
    """The guard must not cost meaningful utility at full precision."""
    pytree = {"w": torch.tensor([300.0, 400.0])}
    clipped, _ = clip_pytree(pytree, clipping_norm=1.0)
    assert _exact_norm(clipped) == pytest.approx(1.0, rel=1e-6)


def test_integer_leaves_do_not_break_the_guard():
    """``torch.finfo`` is undefined on integer dtypes; they must be skipped."""
    pytree = {"w": torch.tensor([3.0, 4.0]), "step": torch.tensor([2, 5])}
    clipped, _ = clip_pytree(pytree, clipping_norm=1.0)
    assert torch.isfinite(clipped["w"]).all()


def test_clip_pytree_is_differentiable_through_the_guard():
    """The guard is a constant affine map, so gradients must still flow."""
    x = torch.tensor([3.0, 4.0], requires_grad=True)
    clipped, _ = clip_pytree({"w": x}, clipping_norm=1.0)
    clipped["w"].sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_complex_leaves_clip_by_their_magnitude(dtype):
    """``|z|^2 = re^2 + im^2``; a real accumulator would drop the imaginary part."""
    for leaf in (
        torch.tensor([3 + 4j], dtype=dtype),
        torch.tensor([0 + 5j], dtype=dtype),  # a real cast reads this as norm 0
    ):
        clipped, aux = clip_pytree({"w": leaf}, clipping_norm=1.0)
        assert float(aux.norm) == pytest.approx(5.0)
        assert torch.linalg.vector_norm(clipped["w"].to(torch.complex128)) <= 1.0


@pytest.mark.parametrize("bad_dtype", [torch.int32, torch.bool, torch.complex64])
def test_non_real_compute_dtype_is_rejected(bad_dtype):
    """A non-real accumulator corrupts the reduction; reject it at the boundary."""
    with pytest.raises(TypeError, match="real floating-point"):
        clip_pytree({"w": torch.tensor([3.0, 4.0])}, 1.0, compute_dtype=bad_dtype)


def _one_dominant_element(n, device, dtype=torch.float32):
    """A leaf whose reduction is worst-case: one large term, many small ones.

    Built on the CPU because MPS has no float64 to normalize in.
    """
    leaf = torch.full((n,), 1e-4, dtype=dtype)
    leaf[0] = 1.0
    leaf = leaf * (3.0 / torch.linalg.vector_norm(leaf.double())).to(dtype)
    return leaf.to(device)


def _norm_on_cpu(tensor):
    """Exact norm of a stored tensor; MPS cannot hold the float64 itself."""
    return torch.linalg.vector_norm(tensor.cpu().double())


def test_bound_holds_for_one_large_leaf(all_devices):
    """The error inside a single leaf's reduction grows with its element count.

    Runs on MPS in CI, where the accumulator is limited to float32 and this is
    the term that dominates.
    """
    leaf = _one_dominant_element(1 << 22, all_devices)
    clipped, _ = clip_pytree({"w": leaf}, clipping_norm=1.0)
    assert _norm_on_cpu(clipped["w"]) <= 1.0


@pytest.mark.parametrize("n", [1 << 14, 1 << 22])
def test_bound_holds_without_a_float64_accumulator(monkeypatch, n):
    """Pin the MPS configuration on every host: no float64 anywhere."""
    monkeypatch.setattr(_pytree, "_SQ_NORM_ACCUM_DTYPE", torch.float32)
    leaf = _one_dominant_element(n, torch.device("cpu"))
    clipped, _ = clip_pytree({"w": leaf}, clipping_norm=1.0)
    assert _norm_on_cpu(clipped["w"]) <= 1.0


@pytest.mark.parametrize("n", [1 << 12, 1 << 13, 1 << 20])
def test_blocked_reduction_traces_in_a_full_graph(n):
    """The two-stage reduction must survive ``fullgraph=True`` at any leaf size.

    ``clipped_grad`` is on the hot path, so a graph break here is a
    whole-pipeline regression rather than a local one.  Sizing the blocks from
    the leaf itself (``isqrt(numel)``) traces fine below
    ``_BLOCKED_REDUCTION_MIN`` and fails above it, on a symbolic shape.
    """
    torch._dynamo.reset()

    def reduce(leaf):
        leaves = [leaf]
        # The bound and the reduction it describes have to agree, so trace both.
        terms = _pytree._reduction_terms(leaves)
        return _pytree._sq_norm(leaves, torch.float32, torch.float32) + terms

    compiled = torch.compile(reduce, backend="aot_eager", fullgraph=True, dynamic=True)
    leaf = torch.randn(n)
    torch.testing.assert_close(compiled(leaf), reduce(leaf))


# ----------------------------------------------------------------------------
# Microbatch accumulation precision
# ----------------------------------------------------------------------------


def _loss_fn(params, x):
    return ((params * x) ** 2).sum()


def _run(microbatch_size, batch, params, **kwargs):
    grad_fn = grad(_loss_fn, argnums=0)
    clipped_fn, state = clipped_fun(
        lambda p, x: grad_fn(p, x),
        batch_argnums=1,
        clipping_norm=1.0,
        microbatch_size=microbatch_size,
        **kwargs,
    )
    result, _ = clipped_fn(params, batch, state=state)
    return _unwrap(result)


@pytest.mark.parametrize("dtype", LOW_PRECISION_DTYPES)
@pytest.mark.parametrize("microbatch_size", [1, 3, 8, 32])
def test_microbatching_matches_full_batch_in_low_precision(dtype, microbatch_size):
    """Splitting the batch must not degrade the sum (issue #343, AUD-081)."""
    generator = torch.Generator().manual_seed(4)
    params = torch.randn(16, generator=generator).to(dtype)
    batch = torch.randn(64, 16, generator=generator).to(dtype)

    full = _run(None, batch, params).to(torch.float64)
    micro = _run(microbatch_size, batch, params).to(torch.float64)

    # Both land in the same output dtype, so they can differ by at most the
    # rounding of that final cast.
    denom = torch.linalg.vector_norm(full).clamp(min=1e-12)
    relative = (torch.linalg.vector_norm(micro - full) / denom).item()
    assert relative < 1e-2 * torch.finfo(dtype).eps


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("microbatch_size", [1, 8])
def test_microbatching_is_unchanged_at_full_precision(dtype, microbatch_size):
    """fp32/fp64 need no promotion, so the arithmetic must be bit-identical."""
    generator = torch.Generator().manual_seed(5)
    params = torch.randn(8, generator=generator).to(dtype)
    batch = torch.randn(32, 8, generator=generator).to(dtype)

    reference = None
    for accumulate_in in (None, dtype):
        result = _run(microbatch_size, batch, params, compute_dtype=accumulate_in)
        if reference is None:
            reference = result
        assert torch.equal(result, reference)


def test_microbatch_accumulates_at_the_requested_output_precision():
    """A wider ``dtype=`` must not be degraded by a narrower ``compute_dtype``."""
    generator = torch.Generator().manual_seed(11)
    params = torch.randn(64, generator=generator)
    batch = torch.randn(4096, 64, generator=generator)

    exact = _run(None, batch.double(), params.double(), dtype=torch.float64).double()
    full = _run(None, batch, params, dtype=torch.float64).double()
    micro = _run(8, batch, params, dtype=torch.float64).double()

    denom = torch.linalg.vector_norm(exact).clamp(min=1e-12)
    err_full = (torch.linalg.vector_norm(full - exact) / denom).item()
    err_micro = (torch.linalg.vector_norm(micro - exact) / denom).item()

    # Microbatching may not be materially worse than not microbatching.
    assert err_micro <= max(10 * err_full, 1e-12)


@pytest.mark.parametrize("dtype", LOW_PRECISION_DTYPES)
@pytest.mark.parametrize("output_dtype", [None, torch.float32])
def test_microbatch_output_dtype_is_preserved(dtype, output_dtype):
    """Deferring the cast must not change the caller-visible dtype."""
    generator = torch.Generator().manual_seed(6)
    params = torch.randn(8, generator=generator).to(dtype)
    batch = torch.randn(12, 8, generator=generator).to(dtype)

    result = _run(4, batch, params, dtype=output_dtype)
    assert result.dtype == (dtype if output_dtype is None else output_dtype)
