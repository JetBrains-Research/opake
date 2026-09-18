import pytest
import torch

from opake.types import ClippedPytree, NoisedPytree, PerGroup, clipped, noised


def test_functional_constructors_wrap_metadata(device):
    pytree = {"w": torch.ones(2, device=device)}

    clipped_value = clipped(pytree, max_norm=2.0)
    noisy_value = noised(pytree, max_norm=2.0, noise_stddev=0.5)

    assert isinstance(clipped_value, ClippedPytree)
    assert clipped_value.pytree is pytree
    assert clipped_value.max_norm == pytest.approx(2.0)
    assert isinstance(noisy_value, NoisedPytree)
    assert noisy_value.pytree is pytree
    assert noisy_value.max_norm == pytest.approx(2.0)
    assert noisy_value.noise_stddev == pytest.approx(0.5)


def test_scalar_multiplication_scales_tensor_leaves_and_bound(device):
    clipped_value = ClippedPytree(
        {"w": torch.tensor([1.0, -2.0], device=device), "meta": None},
        max_norm=3.0,
    )

    scaled = -2.0 * clipped_value

    torch.testing.assert_close(
        scaled.pytree["w"], torch.tensor([-2.0, 4.0], device=device)
    )
    assert scaled.pytree["meta"] is None
    assert scaled.max_norm == pytest.approx(6.0)
    assert scaled.sensitivity == pytest.approx(6.0)


def test_division_scales_per_group_bound(device):
    max_norm = PerGroup(
        groups={"layer.weight": "layer", "head.weight": "head"},
        values={"layer": 2.0, "head": 4.0},
    )
    clipped_value = ClippedPytree(
        {"layer.weight": torch.ones(2, device=device)},
        max_norm=max_norm,
    )

    scaled = clipped_value / 2.0

    torch.testing.assert_close(
        scaled.pytree["layer.weight"], torch.full((2,), 0.5, device=device)
    )
    assert isinstance(scaled.max_norm, PerGroup)
    assert scaled.max_norm.groups == max_norm.groups
    assert scaled.max_norm.values == {"layer": 1.0, "head": 2.0}
    assert scaled.sensitivity == pytest.approx((1.0**2 + 2.0**2) ** 0.5)


def test_noisy_pytree_scales_noise_stddev(device):
    noised = NoisedPytree(
        {"w": torch.tensor([1.0], device=device)},
        max_norm=2.0,
        noise_stddev=0.5,
    )

    scaled = noised * -3.0

    assert isinstance(scaled, NoisedPytree)
    torch.testing.assert_close(scaled.pytree["w"], torch.tensor([-3.0], device=device))
    assert scaled.max_norm == pytest.approx(6.0)
    assert scaled.noise_stddev == pytest.approx(1.5)


def test_clone_detach_to_preserve_metadata(device):
    tensor = torch.tensor([1.0], device=device, requires_grad=True)
    clipped_value = ClippedPytree({"w": tensor}, max_norm=1.0)

    cloned = clipped_value.clone()
    detached = clipped_value.detach()
    moved = clipped_value.to(dtype=torch.float32)

    assert cloned.max_norm == clipped_value.max_norm
    assert cloned.pytree["w"] is not tensor
    assert detached.max_norm == clipped_value.max_norm
    assert detached.pytree["w"].requires_grad is False
    assert moved.max_norm == clipped_value.max_norm
    assert moved.pytree["w"].dtype == torch.float32


def test_unsupported_operations_raise_helpful_error(device):
    clipped_value = ClippedPytree({"w": torch.ones(1, device=device)}, max_norm=1.0)

    with pytest.raises(TypeError, match="reconstruct the clipped value"):
        _ = clipped_value + clipped_value
    with pytest.raises(TypeError, match="reconstruct the clipped value"):
        _ = clipped_value + 1.0
    with pytest.raises(TypeError, match="reconstruct the clipped value"):
        _ = 1.0 / clipped_value
    with pytest.raises(TypeError, match="public real-number scalars"):
        _ = clipped_value * clipped_value
    with pytest.raises(TypeError, match="public real-number scalars"):
        _ = clipped_value * torch.tensor(2.0, device=device)
    with pytest.raises(TypeError, match="public real-number scalars"):
        _ = clipped_value * True
    with pytest.raises(ZeroDivisionError):
        _ = clipped_value / 0.0


# ----- PerGroup × PerGroup elementwise multiplication ---------------------


def test_per_group_elementwise_product():
    """``PerGroup * PerGroup`` multiplies per-group values element-wise.

    This is what enables ``clipping_norm * clipping_norm`` to produce
    the squared per-group sensitivities consumed by the paired second-
    moment release.
    """
    a = PerGroup(
        groups={"p1": "g1", "p2": "g2"},
        values={"g1": 2.0, "g2": 3.0},
    )
    b = PerGroup(
        groups={"p1": "g1", "p2": "g2"},
        values={"g1": 4.0, "g2": 5.0},
    )

    result = a * b

    assert isinstance(result, PerGroup)
    assert result.groups == a.groups
    assert result.values == {"g1": 8.0, "g2": 15.0}


def test_per_group_self_squared_for_second_moment_sensitivity():
    """``pg * pg`` yields per-group squared values — used to derive
    second-stream sensitivities from per-group clipping norms.
    """
    pg = PerGroup(
        groups={"layer.weight": "layer", "head.weight": "head"},
        values={"layer": 2.0, "head": 0.5},
    )

    squared = pg * pg

    assert squared.values == {"layer": 4.0, "head": 0.25}


def test_per_group_product_rejects_mismatched_groups_mapping():
    a = PerGroup(
        groups={"p1": "g1", "p2": "g2"},
        values={"g1": 2.0, "g2": 3.0},
    )
    b = PerGroup(
        groups={"q1": "g1", "q2": "g2"},  # different parameter names
        values={"g1": 4.0, "g2": 5.0},
    )
    with pytest.raises(ValueError, match="identical group mappings"):
        _ = a * b


def test_per_group_product_rejects_mismatched_group_sets():
    a = PerGroup(
        groups={"p1": "g1", "p2": "g2"},
        values={"g1": 2.0, "g2": 3.0},
    )
    b = PerGroup(
        groups={"p1": "g1", "p2": "g2"},
        values={"g1": 4.0, "g99": 5.0},  # different group key
    )
    with pytest.raises(ValueError, match="identical group sets"):
        _ = a * b


def test_per_group_scalar_multiplication_still_works():
    """Scalar multiplication continues to broadcast to every group."""
    pg = PerGroup(
        groups={"p1": "g1", "p2": "g2"},
        values={"g1": 2.0, "g2": 3.0},
    )

    scaled = pg * 0.5

    assert isinstance(scaled, PerGroup)
    assert scaled.values == {"g1": 1.0, "g2": 1.5}
