"""Tests for functional_utils module."""

from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import grad, vmap

from opake.functional import make_functional, with_batch_dim


def test_make_functional_basic():
    """Test basic functionality of make_functional."""
    # Create a simple linear model
    model = nn.Linear(10, 1)

    # Convert to functional form
    fmodel, params = make_functional(model)

    # Check params is a tuple
    assert isinstance(params, tuple)
    assert len(params) == 2  # weight and bias

    # Check params have correct shapes
    assert params[0].shape == (1, 10)  # weight
    assert params[1].shape == (1,)  # bias

    # Test forward pass
    x = torch.randn(5, 10)
    output = fmodel(params, x)

    assert output.shape == (5, 1)
    assert output.dtype == torch.float32


def test_make_functional_matches_original():
    """Test that functional model matches original model output."""
    # Create model
    model = nn.Linear(10, 5)

    # Convert to functional
    fmodel, params = make_functional(model)

    # Test data
    x = torch.randn(3, 10)

    # Compare outputs
    with torch.no_grad():
        output_original = model(x)
        output_functional = fmodel(params, x)

    assert torch.allclose(output_original, output_functional, atol=1e-6)


def test_make_functional_with_mlp():
    """Test make_functional with a multi-layer network."""

    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 1)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x.squeeze(-1)

    model = SimpleMLP()
    fmodel, params = make_functional(model)

    # Should have 6 parameters (3 weights + 3 biases)
    assert len(params) == 6

    # Test forward pass
    x = torch.randn(4, 10)
    output = fmodel(params, x)
    assert output.shape == (4,)


def test_make_functional_with_grad():
    """Test that make_functional works with torch.func.grad."""
    # Simple model
    model = nn.Linear(5, 1)
    fmodel, params = make_functional(model)

    # Define loss function
    def loss_fn(p, x, y):
        pred = fmodel(p, x)
        return ((pred - y) ** 2).mean()

    # Test data
    x = torch.randn(3, 5)
    y = torch.randn(3, 1)

    # Compute gradients
    grads = grad(loss_fn)(params, x, y)

    # Check gradients
    assert isinstance(grads, tuple)
    assert len(grads) == 2
    assert grads[0].shape == params[0].shape  # weight grad
    assert grads[1].shape == params[1].shape  # bias grad


def test_make_functional_with_vmap():
    """Test that make_functional works with torch.func.vmap."""
    # Simple model
    model = nn.Linear(5, 1)
    fmodel, params = make_functional(model)

    # Per-example loss function
    def loss_single(p, x, y):
        pred = fmodel(p, x.unsqueeze(0))
        return ((pred - y) ** 2).mean()

    # Test data (batch)
    x_batch = torch.randn(8, 5)
    y_batch = torch.randn(8, 1)

    # Compute per-example gradients
    per_example_grads = vmap(grad(loss_single), in_dims=(None, 0, 0))(
        params, x_batch, y_batch
    )

    # Check structure: tuple of (batch_size, *param_shape)
    assert isinstance(per_example_grads, tuple)
    assert len(per_example_grads) == 2
    assert per_example_grads[0].shape == (8, 1, 5)  # weight grads
    assert per_example_grads[1].shape == (8, 1)  # bias grads


def test_make_functional_disable_autograd_tracking():
    """Test disable_autograd_tracking parameter."""
    model = nn.Linear(5, 1)

    # Without disable_autograd_tracking
    _fmodel1, params1 = make_functional(model, disable_autograd_tracking=False)
    assert all(p.requires_grad for p in params1)

    # With disable_autograd_tracking
    _fmodel2, params2 = make_functional(model, disable_autograd_tracking=True)
    assert not any(p.requires_grad for p in params2)


def test_make_functional_preserves_device(device):
    """Test that make_functional preserves device."""
    model = nn.Linear(5, 1).to(device)
    fmodel, params = make_functional(model)

    # Params should be on the target device
    assert all(p.device.type == device.type for p in params)

    # Forward pass should work on the target device
    x = torch.randn(3, 5, device=device)
    output = fmodel(params, x)
    assert output.device.type == device.type


def test_make_functional_with_kwargs():
    """Test that fmodel works with keyword arguments."""

    class ModelWithKwargs(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(5, 1)

        def forward(self, x, scale=1.0):
            return self.fc(x) * scale

    model = ModelWithKwargs()
    fmodel, params = make_functional(model)

    x = torch.randn(3, 5)

    # Test with default kwargs
    output1 = fmodel(params, x)
    assert output1.shape == (3, 1)

    # Test with custom kwargs
    output2 = fmodel(params, x, scale=2.0)
    assert torch.allclose(output2, output1 * 2.0, atol=1e-6)


def test_make_functional_parameter_independence():
    """Test that modifying params doesn't affect original model."""
    model = nn.Linear(5, 1)
    original_weight = model.weight.data.clone()

    _fmodel, params = make_functional(model)

    # Modify params
    params = tuple(p * 2 for p in params)

    # Original model should be unchanged
    assert torch.allclose(model.weight.data, original_weight)


def test_make_functional_with_sequential():
    """Test make_functional with nn.Sequential."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )

    fmodel, params = make_functional(model)

    # Should have 4 parameters (2 weights + 2 biases)
    assert len(params) == 4

    # Test forward pass
    x = torch.randn(4, 10)
    output = fmodel(params, x)
    assert output.shape == (4, 5)


# =============================================================================
# with_batch_dim tests
# =============================================================================


class TestWithBatchDimAlwaysMode:
    """Tests for min_ndim=None (always unsqueeze, no output squeeze)."""

    def test_always_unsqueezes_positional_args(self):
        def fn(x, y):
            return x, y

        wrapped = with_batch_dim(fn, batch_argnums=(0, 1))
        x = torch.randn(5)
        y = torch.randn(3)
        out_x, out_y = wrapped(x, y)
        assert out_x.shape == (1, 5)
        assert out_y.shape == (1, 3)

    def test_no_output_squeeze(self):
        def fn(x):
            return x

        wrapped = with_batch_dim(fn, batch_argnums=0)
        x = torch.randn(5)
        out = wrapped(x)
        # Output keeps the unsqueezed shape — no squeeze in always mode
        assert out.shape == (1, 5)

    def test_empty_batch_argnums_is_noop(self):
        def fn(x):
            return x

        wrapped = with_batch_dim(fn)
        x = torch.randn(5)
        out = wrapped(x)
        assert out.shape == (5,)


class TestWithBatchDimConditionalMode:
    """Tests for min_ndim=N (conditional unsqueeze + output squeeze)."""

    def test_unsqueezes_below_threshold(self):
        calls = []

        def fn(x=None):
            calls.append(x.shape)
            return x

        wrapped = with_batch_dim(fn, batch_kwargs=("x",), min_ndim=2)
        # 1D input → below threshold → unsqueeze + squeeze output
        x = torch.randn(5)
        out = wrapped(x=x)
        assert calls[-1] == (1, 5), "fn should see unsqueezed input"
        assert out.shape == (5,), "output should be squeezed back"

    def test_noop_at_threshold(self):
        calls = []

        def fn(x=None):
            calls.append(x.shape)
            return x

        wrapped = with_batch_dim(fn, batch_kwargs=("x",), min_ndim=2)
        # 2D input → at threshold → no unsqueeze, no output squeeze
        x = torch.randn(4, 5)
        out = wrapped(x=x)
        assert calls[-1] == (4, 5)
        assert out.shape == (4, 5)

    def test_dict_kwargs_per_threshold(self):
        calls = {}

        def fn(input_ids=None, inputs_embeds=None):
            calls["input_ids"] = input_ids.shape if input_ids is not None else None
            calls["inputs_embeds"] = (
                inputs_embeds.shape if inputs_embeds is not None else None
            )
            return input_ids

        wrapped = with_batch_dim(
            fn, batch_kwargs={"input_ids": 2, "inputs_embeds": 3}, min_ndim=2
        )

        ids = torch.randn(10)  # 1D < 2 → unsqueeze
        embeds = torch.randn(10, 64)  # 2D < 3 → unsqueeze
        out = wrapped(input_ids=ids, inputs_embeds=embeds)

        assert calls["input_ids"] == (1, 10)
        assert calls["inputs_embeds"] == (1, 10, 64)
        assert out.shape == (10,)  # squeezed back

    def test_squeeze_dict_output(self):
        def fn(x=None):
            return {"logits": torch.randn(1, 5, 100), "loss": torch.tensor(0.5)}

        wrapped = with_batch_dim(fn, batch_kwargs=("x",), min_ndim=2)
        result = wrapped(x=torch.randn(5))
        assert result["logits"].shape == (5, 100)
        assert result["loss"].shape == ()  # scalar untouched

    def test_squeeze_tuple_output(self):
        def fn(x=None):
            return (torch.randn(1, 5), torch.randn(1, 3))

        wrapped = with_batch_dim(fn, batch_kwargs=("x",), min_ndim=2)
        a, b = wrapped(x=torch.randn(5))
        assert a.shape == (5,)
        assert b.shape == (3,)

    def test_squeeze_namedtuple_output(self):
        Result = namedtuple("Result", ["logits", "loss"])

        def fn(x=None):
            return Result(logits=torch.randn(1, 5, 100), loss=torch.tensor(0.5))

        wrapped = with_batch_dim(fn, batch_kwargs=("x",), min_ndim=2)
        result = wrapped(x=torch.randn(5))
        assert isinstance(result, Result)
        assert result.logits.shape == (5, 100)
        assert result.loss.shape == ()

    def test_none_kwargs_skipped(self):
        def fn(x=None, y=None):
            return x

        wrapped = with_batch_dim(fn, batch_kwargs=("x", "y"), min_ndim=2)
        x = torch.randn(5)
        out = wrapped(x=x, y=None)
        assert out.shape == (5,)


class TestWithBatchDimDoubleWrapGuard:
    """Tests for the double-wrap prevention."""

    def test_double_wrap_returns_same(self):
        def fn(x):
            return x

        wrapped = with_batch_dim(fn, batch_argnums=0, min_ndim=2)
        double_wrapped = with_batch_dim(wrapped, batch_argnums=0, min_ndim=2)
        assert wrapped is double_wrapped

    def test_batchified_attribute_set(self):
        def fn(x):
            return x

        wrapped = with_batch_dim(fn, batch_argnums=0)
        assert wrapped._opake_batchified is True


class TestWithBatchDimPositionalNormalization:
    """Tests for signature-based positional arg → kwarg normalization."""

    def test_positional_batch_kwarg_unsqueezed(self):
        """batch_kwargs arg passed positionally should still be processed."""
        calls = []

        def fn(self, input_ids=None):
            calls.append(input_ids.shape)
            return input_ids

        wrapped = with_batch_dim(fn, batch_kwargs={"input_ids": 2}, min_ndim=2)
        sentinel = object()  # stand-in for self
        ids = torch.randn(5)  # 1D → should be unsqueezed
        out = wrapped(sentinel, ids)  # input_ids passed positionally
        assert calls[-1] == (1, 5), "fn should see unsqueezed input_ids"
        assert out.shape == (5,), "output should be squeezed back"

    def test_positional_mixed_with_keyword(self):
        """Some args positional, some keyword — all should be processed."""
        calls = {}

        def fn(self, input_ids=None, attention_mask=None, inputs_embeds=None):
            calls["input_ids"] = input_ids.shape if input_ids is not None else None
            calls["attention_mask"] = (
                attention_mask.shape if attention_mask is not None else None
            )
            return input_ids

        wrapped = with_batch_dim(
            fn,
            batch_kwargs={"input_ids": 2, "attention_mask": 2, "inputs_embeds": 3},
            min_ndim=2,
        )
        sentinel = object()
        ids = torch.randn(5)  # positional
        mask = torch.randn(5)  # keyword
        out = wrapped(sentinel, ids, attention_mask=mask)
        assert calls["input_ids"] == (1, 5)
        assert calls["attention_mask"] == (1, 5)
        assert out.shape == (5,)

    def test_batched_positional_is_noop(self):
        """Already-batched positional arg should not be unsqueezed."""
        calls = []

        def fn(self, input_ids=None):
            calls.append(input_ids.shape)
            return input_ids

        wrapped = with_batch_dim(fn, batch_kwargs={"input_ids": 2}, min_ndim=2)
        sentinel = object()
        ids = torch.randn(4, 5)  # 2D → at threshold → no-op
        out = wrapped(sentinel, ids)
        assert calls[-1] == (4, 5)
        assert out.shape == (4, 5)
