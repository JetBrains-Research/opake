"""Tests for PyTree partition and merge functions."""

import pytest
import torch

from opake.pytree import merge, partition, tree_map_with_path


class TestTreeMapWithPath:
    """Tests for tree_map_with_path function."""

    def test_simple_dict(self):
        """Test with simple dictionary."""
        tree = {"a": torch.tensor([1, 2]), "b": torch.tensor([3, 4, 5])}

        paths = []

        def collect_paths(path, leaf):
            paths.append(path)
            return leaf

        tree_map_with_path(collect_paths, tree)

        assert ("a",) in paths
        assert ("b",) in paths
        assert len(paths) == 2

    def test_nested_dict(self):
        """Test with nested dictionary."""
        tree = {
            "layer1": {"weight": torch.ones(2, 3), "bias": torch.zeros(3)},
            "layer2": {"weight": torch.ones(3, 4)},
        }

        paths = []

        def collect_paths(path, leaf):
            paths.append(path)
            return leaf

        tree_map_with_path(collect_paths, tree)

        assert ("layer1", "weight") in paths
        assert ("layer1", "bias") in paths
        assert ("layer2", "weight") in paths
        assert len(paths) == 3

    def test_transformation(self):
        """Test that transformation works."""
        tree = {"a": torch.ones(2), "b": torch.ones(3)}

        def add_path_length(path, leaf):
            return leaf + len(path)

        result = tree_map_with_path(add_path_length, tree)

        assert torch.allclose(result["a"], torch.tensor([2.0, 2.0]))
        assert torch.allclose(result["b"], torch.tensor([2.0, 2.0, 2.0]))

    def test_paths_match_tree_flatten_with_paths(self):
        """Hand-rolled walker and optree flatten share ParamPath keys."""
        from opake.pytree import tree_flatten_with_paths

        tree = {
            "layers": [
                {"weight": torch.ones(2), "bias": torch.zeros(2)},
                {"weight": torch.ones(3)},
            ]
        }
        walked = []
        tree_map_with_path(lambda p, leaf: walked.append(p) or leaf, tree)
        flat_paths, _, _ = tree_flatten_with_paths(tree)
        assert walked == flat_paths

    def test_leaf_tensor_root_path(self):
        """A bare Tensor is a valid pytree with empty ParamPath ``()``."""
        from opake.pytree import tree_flatten_with_paths

        leaf = torch.ones(3)
        paths, leaves, _ = tree_flatten_with_paths(leaf)
        assert paths == [()]
        assert len(leaves) == 1
        out = tree_map_with_path(lambda p, x: (p, x * 2), leaf)
        assert out[0] == ()
        torch.testing.assert_close(out[1], leaf * 2)


class TestPartition:
    """Tests for partition function."""

    def test_simple_dict(self):
        """Test partitioning simple dictionary."""
        tree = {
            "a": torch.tensor([1, 2]),
            "b": torch.tensor([3, 4]),
            "c": torch.tensor([5]),
        }

        def predicate(path, value):
            return path[0] in ["a", "c"]

        true_tree, false_tree = partition(predicate, tree)

        assert "a" in true_tree
        assert "c" in true_tree
        assert "b" not in true_tree

        assert "b" in false_tree
        assert "a" not in false_tree
        assert "c" not in false_tree

    def test_lora_pattern(self):
        """Test LoRA-style partitioning."""
        tree = {
            "encoder": {
                "weight": torch.randn(10, 5),
                "bias": torch.randn(5),
                "lora_a": torch.randn(10, 2),
                "lora_b": torch.randn(2, 5),
            },
            "decoder": {"weight": torch.randn(5, 3), "lora_a": torch.randn(5, 1)},
        }

        def is_lora(path, value):
            return "lora" in str(path)

        trainable, frozen = partition(is_lora, tree)

        # Check trainable has only LoRA params
        assert "lora_a" in trainable["encoder"]
        assert "lora_b" in trainable["encoder"]
        assert "weight" not in trainable["encoder"]
        assert "bias" not in trainable["encoder"]

        assert "lora_a" in trainable["decoder"]
        assert "weight" not in trainable["decoder"]

        # Check frozen has non-LoRA params
        assert "weight" in frozen["encoder"]
        assert "bias" in frozen["encoder"]
        assert "lora_a" not in frozen["encoder"]
        assert "lora_b" not in frozen["encoder"]

        assert "weight" in frozen["decoder"]
        assert "lora_a" not in frozen["decoder"]

    def test_nested_structure_preserved(self):
        """Test that nested structure is preserved."""
        tree = {
            "layer1": {"sublayer": {"weight": torch.ones(2)}, "bias": torch.zeros(2)},
            "layer2": {"weight": torch.ones(3)},
        }

        def predicate(path, value):
            return "weight" in path

        weights, biases = partition(predicate, tree)

        # Check structure
        assert "layer1" in weights
        assert "sublayer" in weights["layer1"]
        assert "weight" in weights["layer1"]["sublayer"]

        assert "layer2" in weights
        assert "weight" in weights["layer2"]

        assert "layer1" in biases
        assert "bias" in biases["layer1"]

    def test_all_match(self):
        """Test when all elements match predicate."""
        tree = {"a": torch.ones(2), "b": torch.ones(3)}

        def all_true(path, value):
            return True

        true_tree, false_tree = partition(all_true, tree)

        assert "a" in true_tree
        assert "b" in true_tree
        assert false_tree == {}

    def test_none_match(self):
        """Test when no elements match predicate."""
        tree = {"a": torch.ones(2), "b": torch.ones(3)}

        def all_false(path, value):
            return False

        true_tree, false_tree = partition(all_false, tree)

        assert true_tree == {}
        assert "a" in false_tree
        assert "b" in false_tree

    def test_with_lists(self):
        """Test partitioning with list structure."""
        tree = [torch.ones(2), torch.zeros(3), torch.ones(4)]

        def predicate(path, value):
            return path[0] in [0, 2]

        true_tree, false_tree = partition(predicate, tree)

        assert true_tree[0] is not None
        assert true_tree[1] is None
        assert true_tree[2] is not None

        assert false_tree[0] is None
        assert false_tree[1] is not None
        assert false_tree[2] is None

    def test_empty_tree(self):
        """Test with empty tree."""
        tree = {}

        def predicate(path, value):
            return True

        true_tree, false_tree = partition(predicate, tree)

        assert true_tree == {}
        assert false_tree == {}


class TestMerge:
    """Tests for merge function."""

    def test_simple_merge(self):
        """Test merging two simple dicts."""
        tree1 = {"a": torch.tensor([1, 2]), "b": torch.tensor([3])}
        tree2 = {"c": torch.tensor([4, 5])}

        result = merge(tree1, tree2)

        assert "a" in result
        assert "b" in result
        assert "c" in result
        assert torch.allclose(result["a"], tree1["a"])
        assert torch.allclose(result["c"], tree2["c"])

    def test_overlapping_keys(self):
        """Test that later trees override earlier ones."""
        tree1 = {"a": torch.tensor([1, 2]), "b": torch.tensor([3])}
        tree2 = {"a": torch.tensor([4, 5]), "c": torch.tensor([6])}

        result = merge(tree1, tree2)

        # tree2's 'a' should override tree1's 'a'
        assert torch.allclose(result["a"], tree2["a"])
        assert torch.allclose(result["b"], tree1["b"])
        assert torch.allclose(result["c"], tree2["c"])

    def test_nested_merge(self):
        """Test merging nested structures."""
        tree1 = {
            "encoder": {"weight": torch.ones(3)},
            "decoder": {"weight": torch.zeros(2)},
        }
        tree2 = {"encoder": {"bias": torch.ones(3)}}

        result = merge(tree1, tree2)

        # Should have both weight and bias in encoder
        assert "weight" in result["encoder"]
        assert "bias" in result["encoder"]
        assert "decoder" in result
        assert torch.allclose(result["encoder"]["weight"], tree1["encoder"]["weight"])
        assert torch.allclose(result["encoder"]["bias"], tree2["encoder"]["bias"])

    def test_lora_merge(self):
        """Test merging trainable and frozen params (LoRA use case)."""
        frozen = {
            "encoder": {"weight": torch.ones(10, 5), "bias": torch.zeros(5)},
            "decoder": {"weight": torch.ones(5, 3)},
        }

        trainable = {
            "encoder": {"lora_a": torch.ones(10, 2), "lora_b": torch.ones(2, 5)},
            "decoder": {"lora_a": torch.ones(5, 1)},
        }

        result = merge(frozen, trainable)

        # Should have all params
        assert "weight" in result["encoder"]
        assert "bias" in result["encoder"]
        assert "lora_a" in result["encoder"]
        assert "lora_b" in result["encoder"]

        assert "weight" in result["decoder"]
        assert "lora_a" in result["decoder"]

    def test_merge_multiple_trees(self):
        """Test merging more than two trees."""
        tree1 = {"a": torch.tensor([1])}
        tree2 = {"b": torch.tensor([2])}
        tree3 = {"c": torch.tensor([3])}

        result = merge(tree1, tree2, tree3)

        assert "a" in result
        assert "b" in result
        assert "c" in result

    def test_merge_with_none(self):
        """Test merging with None values."""
        tree1 = {"a": torch.tensor([1])}
        tree2 = None

        result = merge(tree1, tree2)

        assert "a" in result
        assert torch.allclose(result["a"], tree1["a"])

    def test_merge_empty(self):
        """Test merging empty trees."""
        result = merge({}, {})
        assert result == {}

    def test_merge_preserves_first_when_no_overlap(self):
        """Test that first tree is preserved when there's no overlap."""
        tree1 = {"a": torch.tensor([1, 2, 3])}
        tree2 = {"b": torch.tensor([4, 5])}

        result = merge(tree1, tree2)

        # Should be exact same tensor for non-overlapping keys
        assert result["a"] is tree1["a"]
        assert result["b"] is tree2["b"]


class TestPartitionMergeRoundtrip:
    """Tests for partition -> merge roundtrip."""

    def test_partition_merge_identity(self):
        """Test that partition + merge recovers original tree."""
        original = {
            "encoder": {
                "weight": torch.randn(10, 5),
                "lora_a": torch.randn(10, 2),
                "lora_b": torch.randn(2, 5),
            },
            "decoder": {"weight": torch.randn(5, 3)},
        }

        def is_lora(path, value):
            return "lora" in str(path)

        trainable, frozen = partition(is_lora, original)
        reconstructed = merge(frozen, trainable)

        # Check all keys present
        assert set(reconstructed.keys()) == set(original.keys())
        assert set(reconstructed["encoder"].keys()) == set(original["encoder"].keys())
        assert set(reconstructed["decoder"].keys()) == set(original["decoder"].keys())

        # Check all tensors are the same
        assert torch.allclose(
            reconstructed["encoder"]["weight"], original["encoder"]["weight"]
        )
        assert torch.allclose(
            reconstructed["encoder"]["lora_a"], original["encoder"]["lora_a"]
        )
        assert torch.allclose(
            reconstructed["encoder"]["lora_b"], original["encoder"]["lora_b"]
        )
        assert torch.allclose(
            reconstructed["decoder"]["weight"], original["decoder"]["weight"]
        )


class TestLoRAWorkflow:
    """Integration tests for LoRA workflow."""

    def test_typical_lora_workflow(self):
        """Test typical LoRA fine-tuning workflow."""
        # 1. Full model parameters
        model_params = {
            "encoder": {
                "weight": torch.randn(100, 50),
                "bias": torch.randn(50),
                "lora_a": torch.randn(100, 4),  # Low-rank adapter
                "lora_b": torch.randn(4, 50),  # Low-rank adapter
            },
            "decoder": {
                "weight": torch.randn(50, 10),
                "lora_a": torch.randn(50, 4),
                "lora_b": torch.randn(4, 10),
            },
        }

        # 2. Partition into trainable (LoRA) and frozen (pretrained)
        def is_lora(path, value):
            return "lora" in str(path)

        trainable_params, frozen_params = partition(is_lora, model_params)

        # 3. Verify partition
        # Trainable should only have LoRA params
        assert len(trainable_params["encoder"]) == 2  # lora_a, lora_b
        assert len(trainable_params["decoder"]) == 2

        # Frozen should only have pretrained params
        assert len(frozen_params["encoder"]) == 2  # weight, bias
        assert len(frozen_params["decoder"]) == 1  # weight

        # 4. Simulate training step (update only trainable)
        def update_params(params, lr=0.01):
            return {k: v - lr * torch.randn_like(v) for k, v in params.items()}

        # Update nested structure
        updated_trainable = {
            "encoder": update_params(trainable_params["encoder"]),
            "decoder": update_params(trainable_params["decoder"]),
        }

        # 5. Merge back for next forward pass
        updated_model = merge(frozen_params, updated_trainable)

        # 6. Verify merge
        assert set(updated_model.keys()) == set(model_params.keys())
        assert set(updated_model["encoder"].keys()) == set(
            model_params["encoder"].keys()
        )

        # Frozen params unchanged
        assert torch.allclose(
            updated_model["encoder"]["weight"], frozen_params["encoder"]["weight"]
        )
        assert torch.allclose(
            updated_model["encoder"]["bias"], frozen_params["encoder"]["bias"]
        )

        # Trainable params updated (different from original)
        assert not torch.allclose(
            updated_model["encoder"]["lora_a"], model_params["encoder"]["lora_a"]
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
