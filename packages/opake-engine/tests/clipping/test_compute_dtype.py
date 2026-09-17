"""Tests for the ``compute_dtype`` parameter on clipping APIs.

Verifies the type-stable + auto-promote contract:
- ``compute_dtype=None`` (default): bf16/fp16 inputs are accumulated in fp32
  internally; output dtype matches input.
- ``compute_dtype=<dtype>`` (explicit): reductions run in that dtype.
- ``dtype`` (output) is independent: ``None`` matches input, explicit forces
  output dtype.
"""

from __future__ import annotations

import pytest
import torch
from torch.func import grad

from opake.api.engine.clipping._clipped_fun import clipped_fun
from opake.api.engine.clipping._pytree import clip_pytree
from opake.pytree import global_norm
from opake.types import ClippedPytree


def _unwrap_clipped(value):
    assert isinstance(value, ClippedPytree)
    return value.pytree


# ----------------------------------------------------------------------------
# global_norm
# ----------------------------------------------------------------------------


def test_global_norm_default_promotes_bf16_to_fp32():
    """bf16 input → fp32 norm by default (auto-promote)."""
    tree = {"a": torch.tensor([3.0, 4.0], dtype=torch.bfloat16)}
    n = global_norm(tree)
    assert n.dtype == torch.float32
    assert torch.isclose(n, torch.tensor(5.0))


def test_global_norm_explicit_fp64():
    """compute_dtype=fp64 forces fp64 even for fp32 inputs."""
    tree = {"a": torch.tensor([3.0, 4.0], dtype=torch.float32)}
    n = global_norm(tree, compute_dtype=torch.float64)
    assert n.dtype == torch.float64


def test_global_norm_explicit_bf16_honored():
    """compute_dtype=bf16 forces bf16 (caller asked for it)."""
    tree = {"a": torch.tensor([3.0, 4.0], dtype=torch.bfloat16)}
    n = global_norm(tree, compute_dtype=torch.bfloat16)
    assert n.dtype == torch.bfloat16


@pytest.mark.parametrize(
    "bad_dtype", [torch.int32, torch.int64, torch.bool, torch.complex64]
)
def test_global_norm_rejects_non_real_floating_compute_dtype(bad_dtype):
    """Integer / bool / complex compute_dtype is rejected at boundary."""
    tree = {"a": torch.tensor([3.0, 4.0])}
    with pytest.raises(TypeError, match="real floating-point"):
        global_norm(tree, compute_dtype=bad_dtype)


def test_global_norm_bf16_norm_unbiased_under_default():
    """Many small contributions: fp32 accumulation matches fp32 baseline."""
    torch.manual_seed(0)
    n_params = 8192
    raw = torch.randn(n_params, dtype=torch.float32) * 0.01
    tree_fp32 = {"a": raw}
    tree_bf16 = {"a": raw.to(torch.bfloat16)}
    n_fp32 = global_norm(tree_fp32)
    n_bf16_default = global_norm(tree_bf16)  # auto-promote → fp32
    n_bf16_native = global_norm(tree_bf16, compute_dtype=torch.bfloat16)
    # The auto-promoted version should be far closer to the fp32 baseline
    # than the native-bf16 reduction (which rounds away small contributions).
    assert (n_bf16_default - n_fp32).abs() < (n_bf16_native - n_fp32).abs()


# ----------------------------------------------------------------------------
# clip_pytree
# ----------------------------------------------------------------------------


def test_clip_pytree_default_output_matches_input_dtype():
    """Type-stable boundary: bf16 in → bf16 out by default."""
    tree = {"a": torch.tensor([6.0, 8.0], dtype=torch.bfloat16)}
    clipped, _aux = clip_pytree(tree, clipping_norm=5.0)
    assert clipped["a"].dtype == torch.bfloat16
    # ||(6, 8)|| = 10; scale = 5/10 = 0.5; clipped = (3, 4); ||clipped|| = 5
    expected = torch.tensor([3.0, 4.0], dtype=torch.bfloat16)
    assert torch.allclose(clipped["a"].float(), expected.float(), atol=0.1)


def test_clip_pytree_compute_dtype_propagates_to_norm():
    """compute_dtype controls the norm reduction precision."""
    tree = {"a": torch.tensor([3.0, 4.0], dtype=torch.float32)}
    _, aux_default = clip_pytree(tree, clipping_norm=10.0)
    _, aux_fp64 = clip_pytree(tree, clipping_norm=10.0, compute_dtype=torch.float64)
    assert aux_default.norm.dtype == torch.float32
    assert aux_fp64.norm.dtype == torch.float64


def test_clip_pytree_per_group_mixed_dtypes_promote_to_highest():
    """Per-group reduction must scan all leaves: fp32 + fp64 ⇒ fp64 accumulator.

    Regression for a Copilot-flagged bug where the per-group path picked
    the *first* leaf's dtype as the accumulator, silently downcasting
    higher-precision peers (e.g. fp64).
    """
    from opake.types import PerGroup

    pg = PerGroup(
        groups={"a": "g", "b": "g"},
        values={"g": 100.0},
    )
    # Mixed dtypes: float32 leaf and float64 leaf in the same group.
    tree = {
        "a": torch.tensor([3.0, 4.0], dtype=torch.float32),
        "b": torch.tensor([12.0], dtype=torch.float64),
    }
    _, aux = clip_pytree(tree, clipping_norm=pg)
    # The per-group norm must accumulate at fp64, not fp32.
    assert aux.group_norms["g"].dtype == torch.float64


# ----------------------------------------------------------------------------
# clipped_fun: dtype (output) vs compute_dtype (internal) split
# ----------------------------------------------------------------------------


def _loss_fn(params, x):
    """Simple per-example loss for vmap'd grad."""
    return ((params * x) ** 2).sum()


def test_clipped_fun_output_dtype_default_matches_input():
    """dtype=None → output dtype matches function output dtype."""

    grad_fn = grad(_loss_fn, argnums=0)

    def per_example(params, x):
        return grad_fn(params, x)

    clipped_fn, state = clipped_fun(per_example, batch_argnums=1, clipping_norm=1.0)
    params = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    x = torch.randn(4, 2, dtype=torch.bfloat16)
    result, _ = clipped_fn(params, x, state=state)
    result = _unwrap_clipped(result)
    assert result.dtype == torch.bfloat16


def test_clipped_fun_output_dtype_overridable():
    """dtype=fp32 → caller gets fp32 sum back regardless of input."""

    grad_fn = grad(_loss_fn, argnums=0)

    def per_example(params, x):
        return grad_fn(params, x)

    clipped_fn, state = clipped_fun(
        per_example,
        batch_argnums=1,
        clipping_norm=1.0,
        dtype=torch.float32,
    )
    params = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    x = torch.randn(4, 2, dtype=torch.bfloat16)
    result, _ = clipped_fn(params, x, state=state)
    result = _unwrap_clipped(result)
    assert result.dtype == torch.float32


def test_clipped_fun_compute_dtype_independent_of_output():
    """compute_dtype is internal-only; default dtype keeps output type-stable."""

    grad_fn = grad(_loss_fn, argnums=0)

    def per_example(params, x):
        return grad_fn(params, x)

    clipped_fn, state = clipped_fun(
        per_example,
        batch_argnums=1,
        clipping_norm=1.0,
        compute_dtype=torch.float32,  # accumulate in fp32
        # dtype left as None → output matches input (bf16)
    )
    params = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    x = torch.randn(4, 2, dtype=torch.bfloat16)
    result, _ = clipped_fn(params, x, state=state)
    result = _unwrap_clipped(result)
    assert result.dtype == torch.bfloat16


def test_clipped_fun_microbatch_honors_compute_dtype():
    """microbatch path also honors compute_dtype."""

    grad_fn = grad(_loss_fn, argnums=0)

    def per_example(params, x):
        return grad_fn(params, x)

    clipped_fn, state = clipped_fun(
        per_example,
        batch_argnums=1,
        clipping_norm=1.0,
        microbatch_size=2,
        compute_dtype=torch.float32,
    )
    params = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    x = torch.randn(6, 2, dtype=torch.bfloat16)
    result, _ = clipped_fn(params, x, state=state)
    result = _unwrap_clipped(result)
    # default dtype=None → output stays bf16; the fp32 compute happens internally
    assert result.dtype == torch.bfloat16


@pytest.mark.parametrize("compute_dtype", [None, torch.float32, torch.float64])
def test_clipped_fun_smoke_compute_dtype_variants(compute_dtype):
    """Each compute_dtype option produces a finite, sensible result."""

    grad_fn = grad(_loss_fn, argnums=0)

    def per_example(params, x):
        return grad_fn(params, x)

    clipped_fn, state = clipped_fun(
        per_example,
        batch_argnums=1,
        clipping_norm=1.0,
        compute_dtype=compute_dtype,
    )
    params = torch.tensor([1.0, 2.0])
    x = torch.randn(4, 2)
    result, _ = clipped_fn(params, x, state=state)
    result = _unwrap_clipped(result)
    assert torch.isfinite(result).all()
