import math
from typing import Any, cast

import pytest
import torch

from opaque import pytree as pu
from opaque.exceptions import InputTypeError
from opaque.types import (
    SecondMomentClippingOutput,
    SecondMomentNoiseOutput,
    clipped,
    noised,
)


def _to_device(tree: Any, device: torch.device) -> Any:
    def move(x):
        return x.to(device) if isinstance(x, torch.Tensor) else x

    return pu.tree_map(move, tree)


def test_tree_leaves_collects_only_tensors_and_is_stable(device):
    tree = {
        "a": torch.tensor([1, 2], dtype=torch.float32),
        "b": {
            "x": torch.tensor([3.0], dtype=torch.float32),
            "y": "not-a-tensor",
            "z": 42,
        },
        "c": (torch.tensor([5.0, 6.0]), [torch.tensor(7.0)]),
    }
    tree = _to_device(tree, device)

    leaves = pu.tree_leaves(tree)
    assert all(isinstance(t, torch.Tensor) for t in leaves)
    # Expect exactly 5 tensor leaves: [1,2], [3], [5,6], 7.0 => 4? plus maybe something
    # Specifically: a(1x2), b.x(1), c[0](2), c[1][0](scalar) => 4 leaves
    assert len(leaves) == 4
    # Order isn't guaranteed across containers; check multiset of shapes
    shapes = sorted(tuple(t.shape) for t in leaves)
    assert shapes == sorted([(2,), (1,), (2,), ()])


def test_tree_map_applies_fn_and_preserves_structure(device):
    tree = {
        "w": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "b": torch.tensor([0.5, -0.5]),
        "meta": {"name": "layer"},
    }
    tree = _to_device(tree, device)

    doubled = pu.tree_map(lambda x: x * 2 if isinstance(x, torch.Tensor) else x, tree)
    assert isinstance(doubled, dict)
    assert doubled["w"].device.type == device.type
    torch.testing.assert_close(doubled["w"], tree["w"] * 2)
    torch.testing.assert_close(doubled["b"], tree["b"] * 2)
    assert doubled["meta"]["name"] == "layer"


def test_tree_map_multiple_trees_elementwise(device):
    t1 = {"x": torch.tensor([1.0, 2.0])}
    t2 = {"x": torch.tensor([3.0, 4.0])}
    t1 = _to_device(t1, device)
    t2 = _to_device(t2, device)

    summed = pu.tree_map(lambda a, b: a + b, t1, t2)
    torch.testing.assert_close(summed["x"], torch.tensor([4.0, 6.0], device=device))


def test_tree_flatten_unflatten_roundtrip(device):
    tree = {
        "attn": torch.tensor([1.0, 2.0]),
        "mlp": {"w": torch.tensor([3.0]), "b": torch.tensor([4.0])},
    }
    tree = _to_device(tree, device)
    leaves, treedef = pu.tree_flatten(tree)
    assert pu.tree_structure(tree) == treedef
    assert all(isinstance(leaf, torch.Tensor) for leaf in leaves)
    rebuilt = pu.tree_unflatten(treedef, [leaf * 2 for leaf in leaves])
    torch.testing.assert_close(rebuilt["attn"], tree["attn"] * 2)
    torch.testing.assert_close(rebuilt["mlp"]["w"], tree["mlp"]["w"] * 2)
    torch.testing.assert_close(rebuilt["mlp"]["b"], tree["mlp"]["b"] * 2)


@pytest.mark.parametrize(
    ("tree", "expected"),
    [
        ({}, 0.0),
        ({"x": torch.tensor([3.0, 4.0])}, 5.0),
        (
            {
                "a": torch.tensor([3.0, 4.0]),
                "b": {"c": torch.tensor([0.0, 12.0])},
            },
            13.0,
        ),
    ],
)
def test_global_norm_l2(tree, expected, device):
    tree = _to_device(tree, device)
    got = pu.global_norm(tree)
    assert isinstance(got, torch.Tensor)
    assert got.shape == ()
    # Empty trees return CPU tensor, non-empty should match input device type
    if tree:
        assert got.device.type == device.type
    else:
        assert got.device.type == "cpu"
    assert math.isclose(float(got), expected, rel_tol=0, abs_tol=1e-6)


def test_global_norm_mixed_dtypes_promotes_to_float(device):
    tree = {
        "i": torch.tensor([1, 2, 3], dtype=torch.int32),
        "f": torch.tensor([0.5, -0.5], dtype=torch.float32),
    }
    tree = _to_device(tree, device)
    got = pu.global_norm(tree)
    assert got.dtype.is_floating_point
    # Expected sqrt(1^2+2^2+3^2+0.5^2+(-0.5)^2) = sqrt(1+4+9+0.25+0.25)=sqrt(14.5)
    expected = (1 + 4 + 9 + 0.25 + 0.25) ** 0.5
    assert math.isclose(float(got), expected, rel_tol=0, abs_tol=1e-6)


def test_global_norm_complex_uses_squared_magnitude(device):
    z = torch.tensor([3 + 4j, 1 - 2j], dtype=torch.complex64)
    tree = _to_device({"z": z}, device)
    got = pu.global_norm(tree)
    # Norm = sqrt((3^2+4^2) + (1^2+2^2)) = sqrt(25 + 5) = sqrt(30)
    expected = 30**0.5
    assert math.isclose(float(got), expected, rel_tol=0, abs_tol=1e-6)


def _dp_wrapper_cases():
    payload = {"w": torch.tensor([3.0, 4.0])}
    clipped_value = clipped(payload, max_norm=1.0)
    noised_value = noised(payload, max_norm=1.0, noise_stddev=0.5)
    return [
        pytest.param(clipped_value, id="clipped"),
        pytest.param(noised_value, id="noised"),
        pytest.param(
            SecondMomentClippingOutput(
                grads=clipped_value,
                squared_grads=clipped_value,
            ),
            id="second_moment_clipped",
        ),
        pytest.param(
            SecondMomentNoiseOutput(
                noisy_grads=noised_value,
                noisy_squared_grads=noised_value,
            ),
            id="second_moment_noised",
        ),
    ]


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(pu.tree_leaves, id="tree_leaves"),
        pytest.param(pu.global_norm, id="global_norm"),
    ],
)
@pytest.mark.parametrize("tree", _dp_wrapper_cases())
def test_tensor_leaf_operations_reject_dp_pytree_wrappers(operation, tree):
    with pytest.raises(InputTypeError) as exc_info:
        operation(tree)

    message = str(exc_info.value)
    assert "raw tensor pytrees" in message
    assert "`.pytree`" in message
    if isinstance(tree, (SecondMomentClippingOutput, SecondMomentNoiseOutput)):
        assert "select a stream" in message


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(pu.tree_leaves, id="tree_leaves"),
        pytest.param(pu.global_norm, id="global_norm"),
    ],
)
def test_tensor_leaf_operations_reject_nested_dp_pytree_wrapper(operation):
    tree = {
        "wrapped": clipped({"w": torch.tensor([3.0, 4.0])}, max_norm=1.0),
        "plain": torch.tensor([12.0]),
    }

    with pytest.raises(InputTypeError, match="raw tensor pytrees"):
        operation(tree)


def test_global_norm_rejects_paired_output_with_raw_fields():
    tree = SecondMomentClippingOutput(
        grads=cast("Any", torch.tensor([3.0])),
        squared_grads=cast("Any", torch.tensor([4.0])),
    )

    with pytest.raises(InputTypeError, match="raw tensor pytrees"):
        pu.global_norm(tree)
