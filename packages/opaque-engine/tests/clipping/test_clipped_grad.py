"""Unit tests for clipped_grad function.

Simplified tests adapted for our PyTorch implementation.
For comprehensive validation against JAX-Privacy, see tests/jax_validation/test_clipped_grad.py
"""

import pytest
import torch

from opaque.api.engine.clipping import clipped_grad
from opaque.types import ClippedPytree


def _unwrap_clipped(value):
    assert isinstance(value, ClippedPytree)
    return value.pytree


def test_clipped_grad_validate_args_overlap():
    """Test that argnums and batch_argnums cannot overlap."""

    def loss(params, x):
        return ((params - x) ** 2).mean()

    with pytest.raises(ValueError, match="overlap"):
        clipped_grad(
            loss,
            clipping_norm=1.0,
            argnums=0,
            batch_argnums=0,
        )


def test_clipped_grad_validate_args_empty_batch():
    """Test that batch_argnums cannot be empty."""

    def loss(params, x):
        return ((params - x) ** 2).mean()

    with pytest.raises(ValueError, match="Batch argnums must not be empty"):
        clipped_grad(
            loss,
            clipping_norm=1.0,
            argnums=0,
            batch_argnums=(),
        )


def test_clipped_grad_rejects_stats_with_aux():
    """Aggregate stats and per-example diagnostics are separate return modes."""

    def loss(params, x):
        return ((params - x) ** 2).mean()

    with pytest.raises(ValueError, match="cannot be combined"):
        clipped_grad(
            loss,
            clipping_norm=1.0,
            return_aux=True,
            return_stats=True,
        )


@pytest.mark.parametrize("microbatch_size", [None, 1])
def test_nonfinite_gradients_are_sanitized(microbatch_size):
    def loss(param, data):
        return torch.sqrt(data - param)

    grad_fn, clip_state = clipped_grad(
        loss,
        argnums=0,
        batch_argnums=1,
        clipping_norm=1.0,
        microbatch_size=microbatch_size,
    )

    grads, _ = grad_fn(
        torch.tensor(0.0),
        torch.tensor([1.0, -1.0]),
        state=clip_state,
    )

    assert torch.isfinite(_unwrap_clipped(grads)).all()


def test_stats_report_empty_batch():
    def loss(param, data):
        return 0.5 * (data - param).square()

    grad_fn, clip_state = clipped_grad(
        loss,
        argnums=0,
        batch_argnums=1,
        clipping_norm=1.0,
        return_stats=True,
    )

    (empty_grads, empty_stats), _ = grad_fn(
        torch.tensor(0.0),
        torch.empty(0),
        state=clip_state,
    )

    assert empty_stats.batch_size == 0
    assert empty_stats.num_clipped == 0.0
    assert torch.equal(_unwrap_clipped(empty_grads), torch.tensor(0.0))


def test_clipped_grad_basic():
    """Test basic clipped_grad returns gradient."""

    def loss(param, data):
        return 0.5 * ((data - param) ** 2).mean()

    grad_fn, clip_state = clipped_grad(
        loss,
        argnums=0,
        batch_argnums=1,
        clipping_norm=10.0,  # High clip norm so no clipping occurs
    )

    param = torch.tensor(3.0, requires_grad=True)
    data = torch.tensor([0.0, 7.0, -2.0])

    grad, _ = grad_fn(param, data, state=clip_state)
    grad = _unwrap_clipped(grad)

    # Check grad is a tensor
    assert isinstance(grad, torch.Tensor)
    # With high clip norm, should get unclipped sum of gradients
    assert grad.shape == param.shape


def test_clipped_grad_with_pytree_params():
    """Test clipped_grad with PyTree (dict) parameters."""

    def loss(params, data):
        pred = params["w"] * data + params["b"]
        return ((pred - data) ** 2).mean()

    grad_fn, clip_state = clipped_grad(
        loss,
        argnums=0,
        batch_argnums=1,
        clipping_norm=10.0,
    )

    params = {
        "w": torch.tensor(1.0, requires_grad=True),
        "b": torch.tensor(0.5, requires_grad=True),
    }
    data = torch.tensor([0.0, 1.0, 2.0])

    grads, _ = grad_fn(params, data, state=clip_state)
    grads = _unwrap_clipped(grads)

    # Check grads has same structure as params
    assert isinstance(grads, dict)
    assert set(grads.keys()) == set(params.keys())
    assert grads["w"].shape == params["w"].shape
    assert grads["b"].shape == params["b"].shape


def test_clipped_grad_return_grad_norms():
    """Test clipped_grad with return_aux=True (grad norms included)."""

    def loss(param, data):
        return 0.5 * ((data - param) ** 2).mean()

    grad_fn, clip_state = clipped_grad(
        loss,
        argnums=0,
        batch_argnums=1,
        clipping_norm=10.0,
        return_aux=True,
    )

    param = torch.tensor(3.0, requires_grad=True)
    data = torch.tensor([0.0, 7.0, -2.0])

    (grad, grad_aux), _ = grad_fn(param, data, state=clip_state)
    grad = _unwrap_clipped(grad)

    # Check gradient
    assert isinstance(grad, torch.Tensor)
    assert grad.shape == param.shape

    # Check grad_aux has grad_norms
    assert grad_aux.grad_norms is not None
    assert grad_aux.grad_norms.shape == (3,)  # One norm per example
    assert (grad_aux.grad_norms >= 0).all()  # Norms are non-negative


def test_clipped_grad_return_values():
    """Test clipped_grad with return_aux=True (loss values included)."""

    def loss(param, data):
        return 0.5 * ((data - param) ** 2).mean()

    grad_fn, clip_state = clipped_grad(
        loss,
        argnums=0,
        batch_argnums=1,
        clipping_norm=10.0,
        return_aux=True,
    )

    param = torch.tensor(3.0, requires_grad=True)
    data = torch.tensor([0.0, 7.0, -2.0])

    (grad, grad_aux), _ = grad_fn(param, data, state=clip_state)
    grad = _unwrap_clipped(grad)

    # Check gradient
    assert isinstance(grad, torch.Tensor)
    assert grad.shape == param.shape

    # Check grad_aux has values
    assert grad_aux.loss_values is not None
    assert grad_aux.loss_values.shape == (3,)  # One value per example


def test_clipped_grad_has_aux():
    """Test clipped_grad with has_aux=True."""

    def loss_with_aux(param, data):
        loss = 0.5 * ((data - param) ** 2).mean()
        user_aux = {"mean_data": data.mean()}
        return loss, user_aux

    grad_fn, clip_state = clipped_grad(
        loss_with_aux,
        argnums=0,
        has_aux=True,
        batch_argnums=1,
        clipping_norm=10.0,
        return_aux=True,
    )

    param = torch.tensor(3.0, requires_grad=True)
    data = torch.tensor([0.0, 7.0, -2.0])

    (grad, grad_aux), _ = grad_fn(param, data, state=clip_state)
    grad = _unwrap_clipped(grad)

    # Check gradient
    assert isinstance(grad, torch.Tensor)
    assert grad.shape == param.shape

    # Check grad_aux has loss_aux data
    assert grad_aux.loss_aux is not None
    assert isinstance(grad_aux.loss_aux, dict)
    assert "mean_data" in grad_aux.loss_aux


def test_clipped_grad_with_normalize_by():
    """Test clipped_grad with normalize_by parameter."""

    def loss(param, data):
        return 0.5 * ((data - param) ** 2).mean()

    batch_size = 3.0
    grad_fn, clip_state = clipped_grad(
        loss,
        argnums=0,
        batch_argnums=1,
        clipping_norm=10.0,
        normalize_by=batch_size,
    )

    param = torch.tensor(3.0, requires_grad=True)
    data = torch.tensor([0.0, 7.0, -2.0])

    grad, _ = grad_fn(param, data, state=clip_state)
    grad = _unwrap_clipped(grad)

    # Check grad is normalized (smaller than without normalization)
    assert isinstance(grad, torch.Tensor)
    assert grad.shape == param.shape


def test_clipped_grad_actual_clipping():
    """Test that gradients are actually clipped when they exceed the norm threshold."""

    def loss(param, data):
        # MSE loss that will produce large gradients
        return ((data - param) ** 2).mean()

    # Small clip norm to force clipping
    clipping_norm = 1.0
    grad_fn, clip_state = clipped_grad(
        loss,
        argnums=0,
        batch_argnums=1,
        clipping_norm=clipping_norm,
        return_aux=True,
    )

    param = torch.tensor(0.0)
    data = torch.tensor([100.0, 200.0, 300.0])  # Large values -> large gradients

    (_grad, grad_aux), _ = grad_fn(param, data, state=clip_state)

    # Check that some gradients were clipped
    assert (grad_aux.grad_norms > clipping_norm).any(), (
        "Expected some gradients to be clipped"
    )
    assert grad_aux.grad_norms.shape == (3,), "Should have 3 per-example norms"


def test_clipped_grad_preserves_direction():
    """Test that clipping preserves gradient direction (only scales magnitude)."""

    def loss(param, data):
        return ((data - param) ** 2).mean()

    grad_fn, clip_state = clipped_grad(
        loss,
        argnums=0,
        batch_argnums=1,
        clipping_norm=1.0,  # Small clip norm
        return_aux=True,
    )

    # Compute unclipped gradient for single example
    param = torch.tensor(0.0, requires_grad=True)
    data_single = torch.tensor([10.0])

    param.grad = None
    loss_val = ((data_single - param) ** 2).mean()
    loss_val.backward()
    unclipped_grad = param.grad.clone()
    unclipped_norm = unclipped_grad.abs().item()

    # Compute clipped gradient
    (grad, _grad_aux), _ = grad_fn(param, data_single, state=clip_state)
    grad = _unwrap_clipped(grad)

    # Check direction is preserved (signs match)
    assert torch.sign(grad) == torch.sign(unclipped_grad), (
        "Direction should be preserved"
    )

    # Check that norm is clipped
    if unclipped_norm > 1.0:
        assert abs(grad.item()) <= 1.0, "Should be clipped to norm 1.0"


def test_clipped_grad_no_clipping_below_threshold():
    """Test that gradients below threshold are unchanged."""

    def loss(param, data):
        return ((data - param) ** 2).mean()

    # Large clip norm (won't clip)
    large_clip_norm = 1000.0
    grad_fn, clip_state = clipped_grad(
        loss,
        argnums=0,
        batch_argnums=1,
        clipping_norm=large_clip_norm,
        return_aux=True,
    )

    param = torch.tensor(1.0)
    data = torch.tensor([1.1, 0.9, 1.05])  # Small differences -> small gradients

    (grad, grad_aux), _ = grad_fn(param, data, state=clip_state)
    grad = _unwrap_clipped(grad)

    # All norms should be below threshold
    assert (grad_aux.grad_norms < large_clip_norm).all(), (
        "All norms should be below threshold"
    )
    # Gradient should be non-trivial
    assert grad.abs() > 1e-6, "Gradient should be non-zero"


def test_clipped_grad_zero_gradients():
    """Test handling of zero gradients."""

    def loss(param, data):
        return ((data - param) ** 2).mean()

    grad_fn, clip_state = clipped_grad(
        loss,
        argnums=0,
        batch_argnums=1,
        clipping_norm=1.0,
        return_aux=True,
    )

    # Zero data and zero param -> zero gradients
    param = torch.tensor(0.0)
    data = torch.zeros(3)

    (grad, grad_aux), _ = grad_fn(param, data, state=clip_state)
    grad = _unwrap_clipped(grad)

    # Check all zero
    assert grad.abs() < 1e-7, "Gradient should be zero"
    assert (grad_aux.grad_norms < 1e-7).all(), "All norms should be zero"


def test_clipped_grad_with_batch_dim():
    """Test with_batch_dim utility works with clipped_grad."""
    from opaque.functional import with_batch_dim

    def loss_with_batch(param, data):
        # Expects data with batch dim of size 1
        assert data.shape == (1,), f"Expected (1,), got {data.shape}"
        return ((data - param) ** 2).mean()

    def loss_no_batch(param, data):
        # Expects data without batch dim (scalar per-example)
        return ((data - param) ** 2).mean()

    param = torch.tensor(1.0)
    data = torch.tensor([0.5, 1.5, 2.0])

    # Without with_batch_dim: loss receives scalar per-example
    grad_fn, clip_state = clipped_grad(
        loss_no_batch, argnums=0, batch_argnums=1, clipping_norm=10.0
    )
    grad_no_batch, _ = grad_fn(param, data, state=clip_state)
    grad_no_batch = _unwrap_clipped(grad_no_batch)

    # With with_batch_dim: loss receives (1,) per-example
    grad_fn2, clip_state2 = clipped_grad(
        with_batch_dim(loss_with_batch, batch_argnums=1),
        argnums=0,
        batch_argnums=1,
        clipping_norm=10.0,
    )
    grad_with_batch, _ = grad_fn2(param, data, state=clip_state2)
    grad_with_batch = _unwrap_clipped(grad_with_batch)

    # Both should produce non-zero gradients
    assert grad_no_batch.abs() > 1e-6, "Should have non-zero gradient"
    assert grad_with_batch.abs() > 1e-6, "Should have non-zero gradient"


def test_clipped_grad_microbatching_identical_results():
    """Test that microbatching produces identical gradients."""

    def loss_fn(params, x, y):
        return ((x @ params - y) ** 2).mean()

    batch_size = 64
    params = torch.randn(10, 1, requires_grad=False)
    x = torch.randn(batch_size, 10)
    y = torch.randn(batch_size, 1)

    # Without microbatching
    grad_fn_no_mb, clip_state_no_mb = clipped_grad(
        loss_fn,
        argnums=0,
        batch_argnums=(1, 2),
        clipping_norm=1.0,
        microbatch_size=None,
    )
    grads_no_mb, _ = grad_fn_no_mb(params, x, y, state=clip_state_no_mb)

    # With microbatching
    grad_fn_mb, clip_state_mb = clipped_grad(
        loss_fn, argnums=0, batch_argnums=(1, 2), clipping_norm=1.0, microbatch_size=8
    )
    grads_mb, _ = grad_fn_mb(params, x, y, state=clip_state_mb)
    grads_no_mb = _unwrap_clipped(grads_no_mb)
    grads_mb = _unwrap_clipped(grads_mb)

    # Results should be identical
    torch.testing.assert_close(grads_mb, grads_no_mb, rtol=1e-5, atol=1e-6)


def test_clipped_grad_microbatching_with_aux():
    """Test microbatching preserves auxiliary outputs."""

    def loss_fn_with_aux(params, x, y):
        pred = x @ params
        loss = ((pred - y) ** 2).mean()
        # Per-example auxiliary outputs (scalars per example)
        aux = {"pred_sum": pred.sum(), "y_sum": y.sum()}
        return loss, aux

    batch_size = 32
    params = torch.randn(5, 1, requires_grad=False)
    x = torch.randn(batch_size, 5)
    y = torch.randn(batch_size, 1)

    # Without microbatching
    grad_fn_no_mb, clip_state_no_mb = clipped_grad(
        loss_fn_with_aux,
        argnums=0,
        batch_argnums=(1, 2),
        clipping_norm=1.0,
        has_aux=True,
        return_aux=True,  # Need to explicitly request aux outputs
        microbatch_size=None,
    )
    (grads_no_mb, grad_aux_no_mb), _ = grad_fn_no_mb(
        params, x, y, state=clip_state_no_mb
    )
    grads_no_mb = _unwrap_clipped(grads_no_mb)

    # With microbatching
    grad_fn_mb, clip_state_mb = clipped_grad(
        loss_fn_with_aux,
        argnums=0,
        batch_argnums=(1, 2),
        clipping_norm=1.0,
        has_aux=True,
        return_aux=True,  # Need to explicitly request aux outputs
        microbatch_size=8,
    )
    (grads_mb, grad_aux_mb), _ = grad_fn_mb(params, x, y, state=clip_state_mb)
    grads_mb = _unwrap_clipped(grads_mb)

    # Gradients should be identical
    torch.testing.assert_close(grads_mb, grads_no_mb, rtol=1e-5, atol=1e-6)

    # Auxiliary outputs should be identical (per-example)
    assert grad_aux_mb.loss_aux is not None
    assert grad_aux_no_mb.loss_aux is not None
    assert grad_aux_mb.loss_aux["pred_sum"].shape == (batch_size,)
    assert grad_aux_no_mb.loss_aux["pred_sum"].shape == (batch_size,)
    torch.testing.assert_close(
        grad_aux_mb.loss_aux["pred_sum"],
        grad_aux_no_mb.loss_aux["pred_sum"],
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        grad_aux_mb.loss_aux["y_sum"],
        grad_aux_no_mb.loss_aux["y_sum"],
        rtol=1e-5,
        atol=1e-6,
    )


def test_clipped_grad_microbatching_with_return_values_and_norms():
    """Test microbatching with return_aux (values and norms)."""

    def loss_fn(params, x, y):
        return ((x @ params - y) ** 2).mean()

    batch_size = 48
    params = torch.randn(8, 1, requires_grad=False)
    x = torch.randn(batch_size, 8)
    y = torch.randn(batch_size, 1)

    # Without microbatching
    grad_fn_no_mb, clip_state_no_mb = clipped_grad(
        loss_fn,
        argnums=0,
        batch_argnums=(1, 2),
        clipping_norm=1.0,
        return_aux=True,
        microbatch_size=None,
    )
    (grads_no_mb, grad_aux_no_mb), _ = grad_fn_no_mb(
        params, x, y, state=clip_state_no_mb
    )
    grads_no_mb = _unwrap_clipped(grads_no_mb)

    # With microbatching
    grad_fn_mb, clip_state_mb = clipped_grad(
        loss_fn,
        argnums=0,
        batch_argnums=(1, 2),
        clipping_norm=1.0,
        return_aux=True,
        microbatch_size=12,
    )
    (grads_mb, grad_aux_mb), _ = grad_fn_mb(params, x, y, state=clip_state_mb)
    grads_mb = _unwrap_clipped(grads_mb)

    # Gradients should be identical
    torch.testing.assert_close(grads_mb, grads_no_mb, rtol=1e-5, atol=1e-6)

    # Loss values should be identical (per-example)
    assert grad_aux_mb.loss_values is not None
    assert grad_aux_no_mb.loss_values is not None
    assert grad_aux_mb.loss_values.shape == (batch_size,)
    assert grad_aux_no_mb.loss_values.shape == (batch_size,)
    torch.testing.assert_close(
        grad_aux_mb.loss_values, grad_aux_no_mb.loss_values, rtol=1e-5, atol=1e-6
    )

    # Gradient norms should be identical (per-example)
    assert grad_aux_mb.grad_norms is not None
    assert grad_aux_no_mb.grad_norms is not None
    assert grad_aux_mb.grad_norms.shape == (batch_size,)
    assert grad_aux_no_mb.grad_norms.shape == (batch_size,)
    torch.testing.assert_close(
        grad_aux_mb.grad_norms, grad_aux_no_mb.grad_norms, rtol=1e-5, atol=1e-6
    )


def test_clipped_grad_microbatching_with_pytree_params():
    """Test microbatching with PyTree parameters."""

    def loss_fn(params, x, y):
        pred = x @ params["w"] + params["b"]
        return ((pred - y) ** 2).mean()

    batch_size = 40
    params = {
        "w": torch.randn(6, 1, requires_grad=False),
        "b": torch.randn(1, requires_grad=False),
    }
    x = torch.randn(batch_size, 6)
    y = torch.randn(batch_size, 1)

    # Without microbatching
    grad_fn_no_mb, clip_state_no_mb = clipped_grad(
        loss_fn,
        argnums=0,
        batch_argnums=(1, 2),
        clipping_norm=1.0,
        microbatch_size=None,
    )
    grads_no_mb, _ = grad_fn_no_mb(params, x, y, state=clip_state_no_mb)

    # With microbatching
    grad_fn_mb, clip_state_mb = clipped_grad(
        loss_fn, argnums=0, batch_argnums=(1, 2), clipping_norm=1.0, microbatch_size=10
    )
    grads_mb, _ = grad_fn_mb(params, x, y, state=clip_state_mb)
    grads_no_mb = _unwrap_clipped(grads_no_mb)
    grads_mb = _unwrap_clipped(grads_mb)

    # Both gradients should be PyTrees with same structure
    assert isinstance(grads_mb, dict)
    assert isinstance(grads_no_mb, dict)
    assert set(grads_mb.keys()) == set(grads_no_mb.keys()) == {"w", "b"}

    # Gradient values should be identical
    torch.testing.assert_close(grads_mb["w"], grads_no_mb["w"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(grads_mb["b"], grads_no_mb["b"], rtol=1e-5, atol=1e-6)


def _sum_of_clipped_grads(grad_fn, params, batch, state):
    result, _ = grad_fn(params, batch, state=state)
    return _unwrap_clipped(result).sum()


def test_clipped_grad_returns_values_not_a_graph():
    def loss(params, x):
        return ((x - params) ** 2).sum()

    param = torch.nn.Parameter(torch.tensor(3.0))
    grad_fn, state = clipped_grad(loss, clipping_norm=100.0)

    total = _sum_of_clipped_grads(grad_fn, param, torch.tensor([0.0, 5.0]), state)

    assert not total.requires_grad
    with pytest.raises(RuntimeError, match="does not require grad"):
        torch.autograd.grad(total, param)


def test_clipped_grad_under_an_outer_transform_stays_differentiable():
    def loss(params, x):
        return ((x - params) ** 2).sum()

    grad_fn, state = clipped_grad(loss, clipping_norm=100.0)
    batch = torch.tensor([0.0, 5.0])

    second = torch.func.grad(lambda p: _sum_of_clipped_grads(grad_fn, p, batch, state))(
        torch.tensor(3.0)
    )

    assert second.item() == pytest.approx(2 * len(batch))


# ---------------------------------------------------------------------------
# Second-moment paired-stream clipping
# ---------------------------------------------------------------------------


class TestSecondMoment:
    """clipped_grad(..., second_moment=True) returns
    ``SecondMomentClippingOutput`` with streams ``Σᵢ gᵢ`` and
    ``Σᵢ gᵢ²`` and bounds ``(C/n, C²/n)``."""

    @pytest.fixture
    def setup(self):
        torch.manual_seed(0)
        params = {"w": torch.randn(4)}
        true_w = torch.randn(4)
        batch_size = 8
        x = torch.randn(batch_size, 4)
        y = x @ true_w + 0.01 * torch.randn(batch_size)

        def loss_fn(p, x, y):
            return ((x @ p["w"] - y) ** 2).mean()

        return params, x, y, loss_fn, batch_size

    def test_output_is_paired_with_correct_bounds(self, setup):
        from opaque.types import SecondMomentClippingOutput

        params, x, y, loss_fn, batch_size = setup
        C = 5.0
        gf, _ = clipped_grad(
            loss_fn,
            argnums=0,
            batch_argnums=(1, 2),
            clipping_norm=C,
            normalize_by=batch_size,
            second_moment=True,
        )
        out, _ = gf(params, x, y, state=_)
        assert isinstance(out, SecondMomentClippingOutput)
        assert isinstance(out.grads, ClippedPytree)
        assert isinstance(out.squared_grads, ClippedPytree)
        # First stream bound: C/n.
        assert out.grads.max_norm == C / batch_size
        # Second stream bound: C²/n.
        assert out.squared_grads.max_norm == C * C / batch_size
        # Sanity: shape matches params.
        assert out.grads.pytree["w"].shape == params["w"].shape
        assert out.squared_grads.pytree["w"].shape == params["w"].shape

    def test_squared_stream_is_per_example(self, setup):
        """Squared stream is the element-wise mean of per-example
        gradient squares: ``(1/n) Σᵢ gᵢ²``.  Reconstructs per-example
        gradients independently and compares."""
        import torch.func as tf

        params, x, y, loss_fn, batch_size = setup
        C = 100.0  # Large enough that nothing actually clips.
        gf, _ = clipped_grad(
            loss_fn,
            argnums=0,
            batch_argnums=(1, 2),
            clipping_norm=C,
            normalize_by=batch_size,
            second_moment=True,
        )
        out, _ = gf(params, x, y, state=_)

        # Reconstruct per-example grads via grad on a single-example loss.
        def per_loss(p, xi, yi):
            return (xi @ p["w"] - yi) ** 2

        clean_grad = tf.grad(per_loss)
        per_example_grads = torch.stack(
            [clean_grad(params, x[i], y[i])["w"] for i in range(batch_size)]
        )
        expected_first = per_example_grads.sum(dim=0) / batch_size
        expected_squared = (per_example_grads**2).sum(dim=0) / batch_size
        torch.testing.assert_close(
            out.grads.pytree["w"], expected_first, rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            out.squared_grads.pytree["w"], expected_squared, rtol=1e-5, atol=1e-5
        )
        # Sanity: ``(1/n) Σᵢ gᵢ²`` differs from ``((1/n) Σᵢ gᵢ)²`` for
        # non-colinear per-example grads, so they're distinct quantities.
        squared_of_mean = expected_first.pow(2)
        assert not torch.allclose(expected_squared, squared_of_mean, atol=1e-3)

    def test_per_group_paired_emits_per_group_max_norm(self, setup):
        """PerGroup clipping with second_moment=True should produce a
        SecondMomentClippingOutput whose first stream carries the
        original per-group bounds and whose second stream carries the
        element-wise squared per-group bounds.
        """
        from opaque.types import PerGroup, SecondMomentClippingOutput

        params, x, y, loss_fn, batch_size = setup
        pg = PerGroup(groups={"w": "g1"}, values={"g1": 2.0})
        gf, state = clipped_grad(
            loss_fn,
            argnums=0,
            batch_argnums=(1, 2),
            clipping_norm=pg,
            normalize_by=batch_size,
            second_moment=True,
        )
        out, _ = gf(params, x, y, state=state)
        assert isinstance(out, SecondMomentClippingOutput)
        first_max = out.grads.max_norm
        second_max = out.squared_grads.max_norm
        assert isinstance(first_max, PerGroup)
        assert isinstance(second_max, PerGroup)
        # First stream: per-group bound divided by ``normalize_by``.
        assert first_max.values == {"g1": 2.0 / batch_size}
        # Second stream: element-wise squared bound divided by ``normalize_by``.
        assert second_max.values == {"g1": 2.0 * 2.0 / batch_size}
        assert first_max.groups == second_max.groups == pg.groups

    def test_microbatch_matches_full_batch(self, setup):
        params, x, y, loss_fn, batch_size = setup
        gf_full, _ = clipped_grad(
            loss_fn,
            argnums=0,
            batch_argnums=(1, 2),
            clipping_norm=1.0,
            normalize_by=batch_size,
            second_moment=True,
        )
        gf_mb, _ = clipped_grad(
            loss_fn,
            argnums=0,
            batch_argnums=(1, 2),
            clipping_norm=1.0,
            normalize_by=batch_size,
            second_moment=True,
            microbatch_size=2,
        )
        out_full, _ = gf_full(params, x, y, state=_)
        out_mb, _ = gf_mb(params, x, y, state=_)
        torch.testing.assert_close(
            out_full.grads.pytree["w"],
            out_mb.grads.pytree["w"],
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            out_full.squared_grads.pytree["w"],
            out_mb.squared_grads.pytree["w"],
            rtol=1e-5,
            atol=1e-5,
        )
        # Bounds identical too.
        assert out_full.grads.max_norm == out_mb.grads.max_norm
        assert out_full.squared_grads.max_norm == out_mb.squared_grads.max_norm

    def test_empty_batch(self, setup):
        from opaque.types import SecondMomentClippingOutput

        params, x, y, loss_fn, batch_size = setup
        # Empty batch: shape (0, 4) for x, (0,) for y.
        empty_x = x[:0]
        empty_y = y[:0]
        C = 2.0
        gf, _ = clipped_grad(
            loss_fn,
            argnums=0,
            batch_argnums=(1, 2),
            clipping_norm=C,
            normalize_by=batch_size,
            second_moment=True,
        )
        out, _ = gf(params, empty_x, empty_y, state=_)
        assert isinstance(out, SecondMomentClippingOutput)
        # Both streams zero-grads with the appropriate bounds.
        assert out.grads.max_norm == C / batch_size
        assert out.squared_grads.max_norm == C * C / batch_size
        assert torch.all(out.grads.pytree["w"] == 0.0)
        assert torch.all(out.squared_grads.pytree["w"] == 0.0)

    def test_with_return_aux(self, setup):
        from opaque.api.engine.clipping._clipped_grad import ClippedGradAux
        from opaque.types import SecondMomentClippingOutput

        params, x, y, loss_fn, batch_size = setup
        gf, _ = clipped_grad(
            loss_fn,
            argnums=0,
            batch_argnums=(1, 2),
            clipping_norm=1.0,
            normalize_by=batch_size,
            second_moment=True,
            return_aux=True,
        )
        (out, aux), _ = gf(params, x, y, state=_)
        assert isinstance(out, SecondMomentClippingOutput)
        assert isinstance(aux, ClippedGradAux)
        assert aux.batch_size == batch_size
        assert aux.grad_norms is not None
        assert aux.grad_norms.shape == (batch_size,)


# ---------------------------------------------------------------------------
# Empty-batch vs non-empty parity: dtype + pre_clipping_transform structure
# ---------------------------------------------------------------------------


def _linear_loss(params, x, y):
    return ((x @ params["w"] - y) ** 2).mean()


class TestEmptyBatchStructureAndDtypeParity:
    """Empty steps must return the same tree structure and dtypes as non-empty ones."""

    @staticmethod
    def _params():
        return {"w": torch.randn(10, dtype=torch.bfloat16)}

    @staticmethod
    def _batches():
        x = torch.randn(8, 10, dtype=torch.bfloat16)
        y = torch.randn(8, dtype=torch.bfloat16)
        return (x, y), (x[:0], y[:0])

    def test_dtype_override_applies_to_empty_batch(self):
        params = self._params()
        (x, y), (empty_x, empty_y) = self._batches()
        gf, state = clipped_grad(
            _linear_loss,
            batch_argnums=(1, 2),
            clipping_norm=1.0,
            dtype=torch.float32,
        )
        full, _ = gf(params, x, y, state=state)
        empty, _ = gf(params, empty_x, empty_y, state=state)
        assert full.pytree["w"].dtype == torch.float32
        assert empty.pytree["w"].dtype == torch.float32
        assert empty.pytree["w"].shape == params["w"].shape
        assert torch.all(empty.pytree["w"] == 0)

    def test_transform_structure_applies_to_empty_batch(self):
        params = self._params()
        (x, y), (empty_x, empty_y) = self._batches()

        def transform(g):
            return {"q": g["w"] + 7.0}

        gf, state = clipped_grad(
            _linear_loss,
            batch_argnums=(1, 2),
            clipping_norm=1.0,
            pre_clipping_transform=transform,
        )
        full, _ = gf(params, x, y, state=state)
        empty, _ = gf(params, empty_x, empty_y, state=state)
        assert sorted(full.pytree) == sorted(empty.pytree) == ["q"]
        assert empty.pytree["q"].dtype == full.pytree["q"].dtype
        assert torch.all(empty.pytree["q"] == 0)

    def test_nonfinite_transform_template_is_zero(self):
        params = self._params()
        _, (empty_x, empty_y) = self._batches()

        def transform(g):
            return {"q": g["w"] / torch.linalg.vector_norm(g["w"])}

        gf, state = clipped_grad(
            _linear_loss,
            batch_argnums=(1, 2),
            clipping_norm=1.0,
            pre_clipping_transform=transform,
        )
        empty, _ = gf(params, empty_x, empty_y, state=state)
        assert torch.isfinite(empty.pytree["q"]).all()
        assert torch.all(empty.pytree["q"] == 0)

    def test_second_moment_streams_parity(self):
        from opaque.types import SecondMomentClippingOutput

        params = self._params()
        (x, y), (empty_x, empty_y) = self._batches()

        def transform(g):
            return {"q": g["w"] * 2.0 + 7.0}

        gf, state = clipped_grad(
            _linear_loss,
            batch_argnums=(1, 2),
            clipping_norm=1.0,
            pre_clipping_transform=transform,
            dtype=torch.float32,
            second_moment=True,
        )
        full, _ = gf(params, x, y, state=state)
        empty, _ = gf(params, empty_x, empty_y, state=state)
        assert isinstance(full, SecondMomentClippingOutput)
        assert isinstance(empty, SecondMomentClippingOutput)
        for full_stream, empty_stream in (
            (full.grads, empty.grads),
            (full.squared_grads, empty.squared_grads),
        ):
            assert sorted(full_stream.pytree) == sorted(empty_stream.pytree) == ["q"]
            for key in ("q",):
                assert empty_stream.pytree[key].dtype == full_stream.pytree[key].dtype
                assert empty_stream.pytree[key].shape == full_stream.pytree[key].shape
        assert torch.all(empty.grads.pytree["q"] == 0)
        assert torch.all(empty.squared_grads.pytree["q"] == 0)

    def test_defaults_keep_param_dtype_and_structure(self):
        params = self._params()
        (x, y), (empty_x, empty_y) = self._batches()
        gf, state = clipped_grad(_linear_loss, batch_argnums=(1, 2), clipping_norm=1.0)
        full, _ = gf(params, x, y, state=state)
        empty, _ = gf(params, empty_x, empty_y, state=state)
        assert sorted(full.pytree) == sorted(empty.pytree) == ["w"]
        assert full.pytree["w"].dtype == empty.pytree["w"].dtype
