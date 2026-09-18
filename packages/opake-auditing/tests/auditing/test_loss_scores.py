"""Tests for auditing.loss_scores."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset

import opake.auditing as auditing
from opake.auditing import loss_scores, one_run
from opake.auditing.types import CanaryScores, OneRunEstimate
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


class TestLossScores:
    """Tests for auditing.loss_scores."""

    def test_scoring_with_train_subset_raises(self, linear_setup):
        params, dataset, loss_fn = linear_setup
        cf = auditing.coin_flip(dataset, num_canaries=50, key=key(42))

        with pytest.raises(ValueError, match="full concatenated dataset"):
            loss_scores(
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
            loss_scores(
                loss_fn,
                params,
                batch_argnums=(1, 2),
                coin_flip=cf,
                dataset=mismatched,
            )

    @pytest.mark.parametrize(
        ("batch_argnums", "message"),
        [
            ((2, 1), r"must be sorted, got \(2, 1\)"),
            ((1, 1), r"must be unique, got \(1, 1\)"),
            ((-1,), r"must be non-negative, got \(-1,\)"),
            ((2,), r"out of range.*got \(2,\)"),
        ],
    )
    def test_invalid_batch_argnums_raise(self, batch_argnums, message):
        """Invalid batch positions are rejected before scoring."""
        params = torch.tensor(1.0)
        dataset = TensorDataset(torch.tensor([1.0]))
        cf = auditing.coin_flip(dataset, num_canaries=1, key=key(0))

        def loss_fn(params, x):
            return (params - x).square()

        with pytest.raises(ValueError, match=message):
            loss_scores(
                loss_fn,
                params,
                batch_argnums=batch_argnums,
                coin_flip=cf,
                dataset=dataset,
            )

    def test_basic_scoring(self, linear_setup):
        """Test that loss_scores returns one score per canary."""
        params, dataset, loss_fn = linear_setup
        cf = auditing.coin_flip(dataset, num_canaries=50, key=key(42))
        scores = loss_scores(
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
        scores = loss_scores(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            coin_flip=cf,
            dataset=dataset,
        )
        assert len(scores) == 5
        assert len(scores) < len(dataset)

    def test_scores_are_negative_loss(self, linear_setup):
        """Test that scores are negated losses (higher = better fit)."""
        params, dataset, loss_fn = linear_setup
        cf = auditing.coin_flip(dataset, num_canaries=50, key=key(42))
        scores = loss_scores(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            coin_flip=cf,
            dataset=dataset,
        )
        # Perfect model: losses near 0, scores near 0 (but negative of near-0)
        assert np.all(scores.scores <= 1e-3)

    def test_scores_match_per_example_losses(self, linear_setup):
        """Each score is the negated loss of the canary it identifies."""
        params, dataset, loss_fn = linear_setup
        random_params = torch.randn_like(params)
        cf = auditing.coin_flip(dataset, num_canaries=25, key=key(42))
        scores = loss_scores(
            loss_fn,
            random_params,
            batch_argnums=(1, 2),
            coin_flip=cf,
            dataset=dataset,
        )

        X, y = dataset.tensors
        with torch.no_grad():
            all_losses = (X @ random_params - y).square()
        expected = -all_losses.numpy()[scores.canary_indices]
        np.testing.assert_allclose(scores.scores, expected, atol=1e-5)

    def test_trained_vs_untrained(self, linear_setup):
        """Test that trained model gives higher scores than random."""
        params, dataset, loss_fn = linear_setup
        random_params = torch.randn_like(params)
        cf = auditing.coin_flip(dataset, num_canaries=50, key=key(42))

        scores_trained = loss_scores(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            coin_flip=cf,
            dataset=dataset,
        )
        scores_random = loss_scores(
            loss_fn,
            random_params,
            batch_argnums=(1, 2),
            coin_flip=cf,
            dataset=dataset,
        )

        assert np.mean(scores_trained.scores) > np.mean(scores_random.scores)


class TestLossScoresSingleArg:
    """Tests for loss_scores with single batch arg (HF-like pattern)."""

    @pytest.fixture
    def single_arg_setup(self):
        """Create a setup where loss_fn takes (params, tokens)."""
        torch.manual_seed(42)
        n_samples = 100
        dim = 8

        tokens = torch.randn(n_samples, dim)
        dataset = TensorDataset(tokens)
        params = torch.randn(dim)

        def loss_fn(params, tokens):
            return F.mse_loss(tokens @ params, torch.zeros(1), reduction="sum")

        return params, dataset, loss_fn

    def test_single_batch_arg(self, single_arg_setup):
        """Test scoring with a single batched argument."""
        params, dataset, loss_fn = single_arg_setup
        cf = auditing.coin_flip(dataset, num_canaries=40, key=key(1))
        scores = loss_scores(
            loss_fn,
            params,
            batch_argnums=(1,),
            coin_flip=cf,
            dataset=dataset,
        )
        assert scores.scores.shape == (40,)


class TestLossScoresDictBatch:
    """Tests for loss_scores with dict-style batches (HuggingFace pattern)."""

    @pytest.fixture
    def dict_dataset_setup(self):
        """Create a setup with a dict-style dataset and collate_fn."""
        torch.manual_seed(42)
        n_samples = 50
        dim = 8

        tokens = torch.randn(n_samples, dim)

        class DictDataset:
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return {"input_ids": self.data[idx]}

        dataset = DictDataset(tokens)
        params = torch.randn(dim)

        def loss_fn(params, tokens):
            return F.mse_loss(tokens @ params, torch.zeros(1), reduction="sum")

        def collate_fn(batch):
            """Collate dict batches into tensor tuples (HF pattern)."""
            return (torch.stack([b["input_ids"] for b in batch]),)

        return params, dataset, loss_fn, collate_fn

    def test_dict_batch_with_collate(self, dict_dataset_setup):
        """Test dict-style batches with collate that returns tuples."""
        params, dataset, loss_fn, collate_fn = dict_dataset_setup
        cf = auditing.coin_flip(dataset, num_canaries=30, key=key(2))
        scores = loss_scores(
            loss_fn,
            params,
            batch_argnums=(1,),
            coin_flip=cf,
            dataset=dataset,
            batch_size=16,
            collate_fn=collate_fn,
        )
        assert scores.scores.shape == (30,)


class TestReferenceScores:
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

        ref_scores = loss_scores(
            loss_fn,
            ref_params,
            batch_argnums=(1,),
            coin_flip=cf,
            dataset=dataset,
        )
        scores = loss_scores(
            loss_fn,
            params,
            batch_argnums=(1,),
            coin_flip=cf,
            dataset=dataset,
            reference_scores=ref_scores,
        )

        # Manually compute expected: -loss(params) - (-loss(ref_params))
        raw_scores = loss_scores(
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
            loss_scores(
                loss_fn,
                params,
                batch_argnums=(1,),
                coin_flip=cf,
                dataset=dataset,
                reference_scores=partial_ref,
            )


class TestEndToEnd:
    """Tests for end-to-end workflow: coin_flip -> loss_scores -> one_run."""

    def test_full_workflow(self, linear_setup):
        """Test the full coin_flip -> loss_scores -> one_run flow."""
        params, dataset, loss_fn = linear_setup

        cf = auditing.coin_flip(dataset, num_canaries=50, key=key(42))
        scores = loss_scores(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            coin_flip=cf,
            dataset=dataset,
            batch_size=32,
        )
        assert isinstance(scores, CanaryScores)

        estimate = one_run(scores, coin_flip=cf)

        assert isinstance(estimate, OneRunEstimate)
        assert estimate.n_in + estimate.n_out == 50
        assert 0.0 <= estimate.attack_auc() <= 1.0
        assert estimate.eps_delta().epsilon_at(delta=0.0) >= 0.0
        assert estimate.attack_beta_at(alpha=0.1) >= 0.0

    def test_with_collate_fn(self):
        """Test workflow with custom collate_fn on the internal loader."""
        torch.manual_seed(42)
        n_samples = 50
        dim = 8

        tokens = torch.randn(n_samples, dim)

        class DictDataset:
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return {"input_ids": self.data[idx]}

        dataset = DictDataset(tokens)
        params = torch.randn(dim)

        def loss_fn(params, tokens):
            return F.mse_loss(tokens @ params, torch.zeros(1), reduction="sum")

        def collate_fn(batch):
            return (torch.stack([b["input_ids"] for b in batch]),)

        cf = auditing.coin_flip(dataset, num_canaries=20, key=key(42))
        scores = loss_scores(
            loss_fn,
            params,
            batch_argnums=(1,),
            coin_flip=cf,
            dataset=dataset,
            batch_size=16,
            collate_fn=collate_fn,
        )

        estimate = one_run(scores, coin_flip=cf)
        assert isinstance(estimate, OneRunEstimate)
        assert estimate.n_in + estimate.n_out == 20


class TestVerifiedScoring:
    """#371: verified scoring binds every score to its canary identifier."""

    def test_identifiers_match_partition(self, linear_setup):
        params, dataset, loss_fn = linear_setup
        cf = auditing.coin_flip(dataset, num_canaries=40, key=key(3))
        scores = auditing.loss_scores(
            loss_fn, params, batch_argnums=(1, 2), coin_flip=cf, dataset=dataset
        )
        assert isinstance(scores, CanaryScores)
        np.testing.assert_array_equal(scores.canary_indices, cf.canary_indices)
        est = one_run(scores, coin_flip=cf)
        np.testing.assert_array_equal(est.canary_indices, cf.canary_indices)

    def test_collate_dropping_rows_raises(self, linear_setup):
        params, dataset, loss_fn = linear_setup
        cf = auditing.coin_flip(dataset, num_canaries=40, key=key(3))

        def dropping_collate(batch):
            kept = batch[:-1]
            return (
                torch.stack([x for x, _ in kept]),
                torch.stack([y for _, y in kept]),
            )

        with pytest.raises(ValueError, match="exactly one row per example"):
            auditing.loss_scores(
                loss_fn,
                params,
                batch_argnums=(1, 2),
                coin_flip=cf,
                dataset=dataset,
                batch_size=8,
                collate_fn=dropping_collate,
            )

    def test_order_preserving_collate_is_unaffected(self, linear_setup):
        params, dataset, loss_fn = linear_setup
        cf = auditing.coin_flip(dataset, num_canaries=40, key=key(3))

        def explicit_collate(batch):
            return (
                torch.stack([x for x, _ in batch]),
                torch.stack([y for _, y in batch]),
            )

        default = auditing.loss_scores(
            loss_fn, params, batch_argnums=(1, 2), coin_flip=cf, dataset=dataset
        )
        explicit = auditing.loss_scores(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            coin_flip=cf,
            dataset=dataset,
            collate_fn=explicit_collate,
        )
        np.testing.assert_allclose(explicit.scores, default.scores, atol=1e-6)
        np.testing.assert_array_equal(explicit.canary_indices, default.canary_indices)

    def test_batch_size_does_not_affect_verified_scores(self, linear_setup):
        params, dataset, loss_fn = linear_setup
        cf = auditing.coin_flip(dataset, num_canaries=40, key=key(3))
        small = auditing.loss_scores(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            coin_flip=cf,
            dataset=dataset,
            batch_size=7,
        )
        large = auditing.loss_scores(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            coin_flip=cf,
            dataset=dataset,
            batch_size=64,
        )
        np.testing.assert_allclose(small.scores, large.scores, atol=1e-5)

    def test_verified_reference_aligns_by_identifier(self, linear_setup):
        params, dataset, loss_fn = linear_setup
        cf = auditing.coin_flip(dataset, num_canaries=40, key=key(3))
        ref = auditing.loss_scores(
            loss_fn,
            torch.randn_like(params),
            batch_argnums=(1, 2),
            coin_flip=cf,
            dataset=dataset,
        )
        perm = np.random.default_rng(0).permutation(40)
        ref_shuffled = CanaryScores(
            ref.scores[perm], canary_indices=ref.canary_indices[perm]
        )
        calibrated = auditing.loss_scores(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            coin_flip=cf,
            dataset=dataset,
            reference_scores=ref,
        )
        calibrated_shuffled_ref = auditing.loss_scores(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            coin_flip=cf,
            dataset=dataset,
            reference_scores=ref_shuffled,
        )
        np.testing.assert_allclose(
            calibrated.scores, calibrated_shuffled_ref.scores, atol=1e-6
        )


class TestScoringArgumentValidation:
    """Argument validation for verified scoring."""

    def test_missing_dataset_raises(self, linear_setup):
        params, dataset, loss_fn = linear_setup
        cf = auditing.coin_flip(dataset, num_canaries=10, key=key(0))
        with pytest.raises(TypeError, match="dataset"):
            loss_scores(loss_fn, params, batch_argnums=(1, 2), coin_flip=cf)

    def test_missing_coin_flip_raises(self, linear_setup):
        params, dataset, loss_fn = linear_setup
        with pytest.raises(TypeError, match="coin_flip"):
            loss_scores(loss_fn, params, batch_argnums=(1, 2), dataset=dataset)

    @pytest.mark.parametrize("bad_batch_size", [0, -1])
    def test_non_positive_batch_size_raises(self, linear_setup, bad_batch_size):
        params, dataset, loss_fn = linear_setup
        cf = auditing.coin_flip(dataset, num_canaries=10, key=key(0))
        with pytest.raises(ValueError, match="batch_size must be positive"):
            loss_scores(
                loss_fn,
                params,
                batch_argnums=(1, 2),
                coin_flip=cf,
                dataset=dataset,
                batch_size=bad_batch_size,
            )

    @pytest.mark.parametrize("bad_batch_size", [None, 8.0, True])
    def test_non_int_batch_size_raises(self, linear_setup, bad_batch_size):
        """A non-int batch_size must raise, not reach the loader.

        ``DataLoader(batch_size=None)`` disables automatic batching, which
        would silently score one example per "batch" instead of failing.
        """
        params, dataset, loss_fn = linear_setup
        cf = auditing.coin_flip(dataset, num_canaries=10, key=key(0))
        with pytest.raises(TypeError, match="batch_size must be an int"):
            loss_scores(
                loss_fn,
                params,
                batch_argnums=(1, 2),
                coin_flip=cf,
                dataset=dataset,
                batch_size=bad_batch_size,
            )

    def test_bare_reference_with_verified_scoring_raises(self, linear_setup):
        params, dataset, loss_fn = linear_setup
        cf = auditing.coin_flip(dataset, num_canaries=10, key=key(0))
        with pytest.raises(TypeError, match="identifiers"):
            loss_scores(
                loss_fn,
                params,
                batch_argnums=(1, 2),
                coin_flip=cf,
                dataset=dataset,
                reference_scores=np.zeros(10),
            )
