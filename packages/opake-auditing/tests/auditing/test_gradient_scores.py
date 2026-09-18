"""Tests for auditing.gradient_scores."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset

import opake.auditing as auditing
from opake.auditing import gradient_scores, one_run
from opake.auditing.types import CanaryScores
from opake.random import key


@pytest.fixture
def linear_setup():
    """Create a simple linear model, dataset, and loss function."""
    torch.manual_seed(42)
    input_dim = 10
    n_samples = 200

    X = torch.randn(n_samples, input_dim)
    true_w = torch.randn(input_dim)
    y = X @ true_w

    dataset = TensorDataset(X, y)
    params = true_w.clone()  # Perfect model

    def loss_fn(params, x, y):
        return F.mse_loss(x @ params, y, reduction="sum")

    return params, dataset, loss_fn


class TestGradientScores:
    """Tests for auditing.gradient_scores."""

    @pytest.mark.slow
    def test_basic_scoring(self, linear_setup):
        """Test that gradient_scores returns one score per canary."""
        params, dataset, loss_fn = linear_setup
        cf = auditing.coin_flip(dataset, num_canaries=50, key=key(42))
        scores = gradient_scores(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            coin_flip=cf,
            dataset=dataset,
        )
        assert isinstance(scores, CanaryScores)
        assert scores.scores.shape == (50,)
        assert len(scores) == 50

    def test_only_canaries_are_scored(self, linear_setup):
        """Only the partition's canaries are scored, not the whole dataset."""
        params, dataset, loss_fn = linear_setup
        cf = auditing.coin_flip(dataset, num_canaries=5, key=key(42))
        scores = gradient_scores(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            coin_flip=cf,
            dataset=dataset,
        )
        assert len(scores) == 5
        assert len(scores) < len(dataset)
        np.testing.assert_array_equal(scores.canary_indices, cf.canary_indices)

    def test_scores_are_non_positive(self, linear_setup):
        """Test that scores are <= 0 (negative squared norms)."""
        params, dataset, loss_fn = linear_setup
        cf = auditing.coin_flip(dataset, num_canaries=50, key=key(42))
        scores = gradient_scores(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            coin_flip=cf,
            dataset=dataset,
        )
        assert np.all(scores.scores <= 0)

    def test_perfect_model_has_near_zero_scores(self, linear_setup):
        """Perfect model: gradient norms near 0, so scores near 0."""
        params, dataset, loss_fn = linear_setup
        cf = auditing.coin_flip(dataset, num_canaries=50, key=key(42))
        scores = gradient_scores(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            coin_flip=cf,
            dataset=dataset,
        )
        # Perfect model has zero loss => zero gradient => scores near 0
        assert np.all(scores.scores > -1e-3)

    def test_scores_match_per_example_gradient_norms(self, linear_setup):
        """Each score is the negated squared gradient norm of its canary."""
        params, dataset, loss_fn = linear_setup
        random_params = torch.randn_like(params)
        cf = auditing.coin_flip(dataset, num_canaries=25, key=key(42))
        scores = gradient_scores(
            loss_fn,
            random_params,
            batch_argnums=(1, 2),
            coin_flip=cf,
            dataset=dataset,
        )

        X, y = dataset.tensors
        with torch.no_grad():
            # d/dw (x @ w - y)^2 = 2 (x @ w - y) x
            residual = X @ random_params - y
            grads = 2 * residual.unsqueeze(1) * X
            all_expected = -(grads.square().sum(dim=1))
        expected = all_expected.numpy()[scores.canary_indices]
        np.testing.assert_allclose(scores.scores, expected, rtol=1e-4)

    def test_random_model_has_larger_norms(self, linear_setup):
        """Random model has larger gradient norms than perfect model."""
        params, dataset, loss_fn = linear_setup
        cf = auditing.coin_flip(dataset, num_canaries=50, key=key(42))

        scores_perfect = gradient_scores(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            coin_flip=cf,
            dataset=dataset,
        )

        random_params = torch.randn_like(params) * 10
        scores_random = gradient_scores(
            loss_fn,
            random_params,
            batch_argnums=(1, 2),
            coin_flip=cf,
            dataset=dataset,
        )

        # Random model has larger gradient norms => more negative scores
        assert np.mean(scores_perfect.scores) > np.mean(scores_random.scores)

    def test_batch_size_does_not_affect_result(self, linear_setup):
        """Test that different batch sizes give same scores."""
        params, dataset, loss_fn = linear_setup
        cf = auditing.coin_flip(dataset, num_canaries=50, key=key(42))
        scores_32 = gradient_scores(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            coin_flip=cf,
            dataset=dataset,
            batch_size=32,
        )
        scores_128 = gradient_scores(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            coin_flip=cf,
            dataset=dataset,
            batch_size=128,
        )
        np.testing.assert_allclose(scores_32.scores, scores_128.scores, atol=1e-5)
        np.testing.assert_array_equal(
            scores_32.canary_indices, scores_128.canary_indices
        )

    def test_single_batch_argnum(self):
        """Test with a single batch argument."""
        torch.manual_seed(42)
        X = torch.randn(30, 5)
        params = torch.randn(5)
        dataset = TensorDataset(X)

        def loss_fn(params, x):
            return (x @ params).pow(2).sum()

        cf = auditing.coin_flip(dataset, num_canaries=15, key=key(1))
        scores = gradient_scores(
            loss_fn,
            params,
            batch_argnums=(1,),
            coin_flip=cf,
            dataset=dataset,
            batch_size=16,
        )
        assert scores.scores.shape == (15,)
        assert np.all(scores.scores <= 0)


class TestGradientScoresValidation:
    """Tests for input validation."""

    def test_scoring_with_train_subset_raises(self, linear_setup):
        params, dataset, loss_fn = linear_setup
        cf = auditing.coin_flip(dataset, num_canaries=50, key=key(42))

        with pytest.raises(ValueError, match="full concatenated dataset"):
            gradient_scores(
                loss_fn,
                params,
                batch_argnums=(1, 2),
                coin_flip=cf,
                dataset=cf.train_subset(dataset),
            )

    def test_scoring_with_mismatched_dataset_size_raises(self, linear_setup):
        params, dataset, loss_fn = linear_setup
        cf = auditing.coin_flip(dataset, num_canaries=50, key=key(42))
        mismatched = TensorDataset(*(tensor[:-1] for tensor in dataset.tensors))

        with pytest.raises(ValueError, match="dataset length"):
            gradient_scores(
                loss_fn,
                params,
                batch_argnums=(1, 2),
                coin_flip=cf,
                dataset=mismatched,
            )

    def test_batch_argnums_includes_zero_raises(self):
        """batch_argnums=(0,) must raise since position 0 is params."""
        torch.manual_seed(42)
        X = torch.randn(10, 4)
        params = torch.randn(4)
        dataset = TensorDataset(X)
        cf = auditing.coin_flip(dataset, num_canaries=10, key=key(0))

        def loss_fn(x, params):
            return (x @ params).pow(2).sum()

        with pytest.raises(ValueError, match="must not be in batch_argnums"):
            gradient_scores(
                loss_fn,
                params,
                batch_argnums=(0,),
                coin_flip=cf,
                dataset=dataset,
            )

    def test_empty_batch_argnums_raises(self):
        """Empty batch_argnums must raise."""
        torch.manual_seed(42)
        params = torch.randn(4)
        dataset = TensorDataset(torch.randn(10, 4))
        cf = auditing.coin_flip(dataset, num_canaries=10, key=key(0))

        def loss_fn(params, x):
            return (x @ params).pow(2).sum()

        with pytest.raises(ValueError, match="non-empty"):
            gradient_scores(
                loss_fn,
                params,
                batch_argnums=(),
                coin_flip=cf,
                dataset=dataset,
            )


class TestGradientScoresReference:
    """Tests for reference_scores subtraction."""

    @pytest.fixture
    def reference_setup(self):
        """Dataset, params, and loss function for reference calibration."""
        torch.manual_seed(42)
        dim = 8
        n = 50

        tokens = torch.randn(n, dim)
        dataset = TensorDataset(tokens)
        params = torch.randn(dim)
        ref_params = torch.randn(dim)

        def loss_fn(params, tokens):
            return F.mse_loss(tokens @ params, torch.zeros(1), reduction="sum")

        return params, ref_params, dataset, loss_fn

    def test_reference_scores_subtraction(self, reference_setup):
        """Test that reference_scores are subtracted from current scores."""
        params, ref_params, dataset, loss_fn = reference_setup
        cf = auditing.coin_flip(dataset, num_canaries=20, key=key(7))

        ref_scores = gradient_scores(
            loss_fn,
            ref_params,
            batch_argnums=(1,),
            coin_flip=cf,
            dataset=dataset,
        )
        scores = gradient_scores(
            loss_fn,
            params,
            batch_argnums=(1,),
            coin_flip=cf,
            dataset=dataset,
            reference_scores=ref_scores,
        )

        # Manually compute expected: raw - ref
        raw_scores = gradient_scores(
            loss_fn,
            params,
            batch_argnums=(1,),
            coin_flip=cf,
            dataset=dataset,
        )
        np.testing.assert_array_equal(
            raw_scores.canary_indices, ref_scores.canary_indices
        )
        expected = raw_scores.scores - ref_scores.scores
        np.testing.assert_allclose(scores.scores, expected, atol=1e-5)
        np.testing.assert_array_equal(scores.canary_indices, cf.canary_indices)

    def test_reference_not_covering_canaries_raises(self, reference_setup):
        """A reference missing some canaries raises ValueError."""
        params, _ref_params, dataset, loss_fn = reference_setup
        cf = auditing.coin_flip(dataset, num_canaries=10, key=key(7))
        partial_ref = CanaryScores(np.zeros(5), canary_indices=cf.canary_indices[:5])

        with pytest.raises(ValueError, match="do not cover"):
            gradient_scores(
                loss_fn,
                params,
                batch_argnums=(1,),
                coin_flip=cf,
                dataset=dataset,
                reference_scores=partial_ref,
            )

    def test_bare_reference_raises(self, reference_setup):
        """A bare ndarray reference carries no identifiers and is rejected."""
        params, _ref_params, dataset, loss_fn = reference_setup
        cf = auditing.coin_flip(dataset, num_canaries=10, key=key(7))

        with pytest.raises(TypeError, match="identifiers"):
            gradient_scores(
                loss_fn,
                params,
                batch_argnums=(1,),
                coin_flip=cf,
                dataset=dataset,
                reference_scores=np.zeros(10),
            )


class TestGradientScoresPyTree:
    """Tests for PyTree (dict) parameter support."""

    def test_dict_params(self):
        """Test gradient_scores with dict parameters (common for models)."""
        torch.manual_seed(42)
        n = 20
        X = torch.randn(n, 4)
        dataset = TensorDataset(X)
        params = {"w": torch.randn(4, 3), "b": torch.randn(3)}

        def loss_fn(params, x):
            return F.mse_loss(x @ params["w"] + params["b"], torch.zeros(3))

        cf = auditing.coin_flip(dataset, num_canaries=12, key=key(4))
        scores = gradient_scores(
            loss_fn,
            params,
            batch_argnums=(1,),
            coin_flip=cf,
            dataset=dataset,
            batch_size=8,
        )
        assert scores.scores.shape == (12,)
        assert np.all(scores.scores <= 0)


class TestEndToEnd:
    """Tests for end-to-end workflow: coin_flip -> gradient_scores -> one_run."""

    def test_full_workflow(self):
        """Test the full coin_flip -> gradient_scores -> one_run flow."""
        torch.manual_seed(42)
        input_dim = 10
        n_samples = 200

        X = torch.randn(n_samples, input_dim)
        true_w = torch.randn(input_dim)
        y = X @ true_w
        dataset = TensorDataset(X, y)

        # Train to convergence on a subset (members)
        params = true_w.clone()

        cf = auditing.coin_flip(dataset, num_canaries=50, key=key(42))

        def loss_fn(params, x, y):
            return F.mse_loss(x @ params, y, reduction="sum")

        scores = gradient_scores(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            coin_flip=cf,
            dataset=dataset,
            batch_size=32,
        )
        assert isinstance(scores, CanaryScores)
        np.testing.assert_array_equal(scores.canary_indices, cf.canary_indices)

        estimate = one_run(scores, coin_flip=cf)
        # Perfect model should give a valid estimate
        assert estimate.eps_delta().epsilon_at(significance=0.05, delta=1e-5) >= 0.0
