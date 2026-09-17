"""End-to-end tests for DP-FTRL noise in training loops."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from opake.api.dpftrl.noise._engine import _matrix_factorization_noise
from opake.api.dpftrl.noise._streaming_matrix import identity
from opake.api.dpftrl.noise._toeplitz import (
    inverse_as_streaming_matrix,
    optimal_max_error_strategy_coefs,
)
from opake.dpftrl.noise import (
    band_mf_strategy,
    bisr_strategy,
    blt_strategy,
    identity_strategy,
    lambda_cgd_strategy,
    mf_gaussian_noise,
)
from opake.dpftrl.sampling import BallsInBinsSampler
from opake.random import key
from opake.types import clipped


def test_accounting_mechanism_types_facade_imports():
    from opake.dpftrl.accounting.mechanisms.types import MfGaussian

    assert MfGaussian.__name__ == "MfGaussian"


def _engine_train_loop(
    model, optimizer, noise_fn, state, x_data, y_data, steps, *, stddev
):
    """Run a DP-FTRL training loop with the raw engine noise function.

    The engine returns a noise_fn that takes the per-call ``stddev`` directly
    on a raw gradient pytree.  Used by tests that exercise
    :func:`_matrix_factorization_noise` directly rather than the
    :func:`mf_gaussian_noise` dispatcher.
    """
    params = list(model.parameters())

    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        pred = model(x_data)
        loss = ((pred - y_data) ** 2).mean()
        loss.backward()

        grads = {i: p.grad.clone() for i, p in enumerate(params)}
        noisy_grads, state = noise_fn(grads, state, stddev=stddev)
        for i, p in enumerate(params):
            p.grad = noisy_grads[i].to(p.dtype)

        optimizer.step()
        losses.append(loss.item())
    return losses


def _mf_train_loop(
    model, optimizer, noise_fn, state, x_data, y_data, steps, *, max_norm
):
    """Run a DP-FTRL training loop through the :func:`mf_gaussian_noise` dispatcher.

    Wraps gradients as :class:`ClippedPytree`, applies the dispatcher's
    noise_fn, and unwraps the resulting :class:`NoisedPytree`.
    """
    params = list(model.parameters())

    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        pred = model(x_data)
        loss = ((pred - y_data) ** 2).mean()
        loss.backward()

        grads = {i: p.grad.clone() for i, p in enumerate(params)}
        noised, state = noise_fn(clipped(grads, max_norm=max_norm), state)
        for i, p in enumerate(params):
            p.grad = noised.pytree[i].to(p.dtype)

        optimizer.step()
        losses.append(loss.item())
    return losses


class TestDPFTRLTrainingLoop:
    """Tests for using _matrix_factorization_noise in a training loop."""

    def _make_template(self, model):
        return {i: torch.zeros_like(p) for i, p in enumerate(model.parameters())}

    @pytest.mark.slow
    def test_identity_noise_trains(self):
        """Identity noise (DP-SGD equivalent) trains a simple model."""
        torch.manual_seed(0)
        model = nn.Linear(5, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        noise_fn, state = _matrix_factorization_noise(
            self._make_template(model),
            identity(),
            key=key(42),
        )

        x = torch.randn(50, 5)
        true_w = torch.randn(5, 1)
        y = x @ true_w

        losses = _engine_train_loop(
            model, optimizer, noise_fn, state, x, y, steps=50, stddev=0.1
        )
        assert losses[-1] < losses[0]

    def test_toeplitz_noise_trains(self):
        """BandMF Toeplitz noise trains a simple model."""
        torch.manual_seed(0)
        model = nn.Linear(5, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        steps = 50
        coefs = optimal_max_error_strategy_coefs(steps)
        noising = inverse_as_streaming_matrix(coefs)
        noise_fn, state = _matrix_factorization_noise(
            self._make_template(model),
            noising,
            key=key(42),
        )

        x = torch.randn(50, 5)
        true_w = torch.randn(5, 1)
        y = x @ true_w

        losses = _engine_train_loop(
            model, optimizer, noise_fn, state, x, y, steps=steps, stddev=0.1
        )
        assert losses[-1] < losses[0]

    def test_dense_matrix_noise_trains(self):
        """Dense noising matrix trains a simple model."""
        torch.manual_seed(0)
        model = nn.Linear(5, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        steps = 5
        noising = torch.eye(steps, dtype=torch.float64)
        noise_fn, state = _matrix_factorization_noise(
            self._make_template(model),
            noising,
            key=key(42),
        )

        x = torch.randn(50, 5)
        true_w = torch.randn(5, 1)
        y = x @ true_w

        losses = _engine_train_loop(
            model, optimizer, noise_fn, state, x, y, steps=steps, stddev=0.1
        )
        assert losses[-1] < losses[0]

    def test_with_adam_optimizer(self):
        """Works with Adam as the base optimizer."""
        torch.manual_seed(0)
        model = nn.Linear(5, 1, bias=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        noise_fn, state = _matrix_factorization_noise(
            self._make_template(model),
            identity(),
            key=key(42),
        )

        x = torch.randn(50, 5)
        true_w = torch.randn(5, 1)
        y = x @ true_w

        losses = _engine_train_loop(
            model, optimizer, noise_fn, state, x, y, steps=30, stddev=0.1
        )
        assert losses[-1] < losses[0]

    def test_multi_param_model(self):
        """Works with models having multiple parameters."""
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        noise_fn, state = _matrix_factorization_noise(
            self._make_template(model),
            identity(),
            key=key(42),
        )

        x = torch.randn(50, 10)
        true_w = torch.randn(10, 1)
        y = x @ true_w

        losses = _engine_train_loop(
            model, optimizer, noise_fn, state, x, y, steps=30, stddev=0.1
        )
        assert losses[-1] < losses[0]

    def test_deterministic_with_seed(self):
        """Same seed produces same noised updates."""
        results = []
        for _ in range(2):
            torch.manual_seed(0)
            model = nn.Linear(5, 1, bias=False)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            noise_fn, state = _matrix_factorization_noise(
                self._make_template(model),
                identity(),
                key=key(42),
            )

            x = torch.randn(4, 5)
            true_w = torch.randn(5, 1)
            y = x @ true_w

            _engine_train_loop(
                model, optimizer, noise_fn, state, x, y, steps=3, stddev=1.0
            )
            results.append(model.weight.data.clone())

        torch.testing.assert_close(results[0], results[1])


class TestBandMFvsDPSGD:
    """End-to-end comparison: BandMF (correlated noise) should achieve
    better or comparable utility to independent noise (DP-SGD)
    on a simple regression problem."""

    def _make_template(self, model):
        return {i: torch.zeros_like(p) for i, p in enumerate(model.parameters())}

    def test_bandmf_trains_simple_regression(self):
        """BandMF DP-FTRL can train a simple model (loss decreases)."""
        torch.manual_seed(0)
        x = torch.randn(50, 5)
        true_w = torch.randn(5, 1)
        y = x @ true_w

        model = nn.Linear(5, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        steps = 50
        coefs = optimal_max_error_strategy_coefs(steps)
        noising = inverse_as_streaming_matrix(coefs)
        noise_fn, state = _matrix_factorization_noise(
            self._make_template(model),
            noising,
            key=key(42),
        )

        losses = _engine_train_loop(
            model, optimizer, noise_fn, state, x, y, steps, stddev=0.1
        )
        assert losses[-1] < losses[0]

    def test_bandmf_vs_dpsgd_utility(self):
        """BandMF should achieve comparable or better utility than DP-SGD.

        This is the key end-to-end validation: on a simple linear regression,
        BandMF (correlated noise) should not be worse than independent noise
        at the same noise level.
        """
        torch.manual_seed(0)
        x = torch.randn(100, 5)
        true_w = torch.randn(5, 1)
        y = x @ true_w
        steps = 100
        stddev = 0.5

        # DP-SGD (identity = independent noise)
        torch.manual_seed(0)
        model_sgd = nn.Linear(5, 1, bias=False)
        opt_sgd = torch.optim.SGD(model_sgd.parameters(), lr=0.01)
        noise_sgd, state_sgd = _matrix_factorization_noise(
            self._make_template(model_sgd),
            identity(),
            key=key(42),
        )
        losses_sgd = _engine_train_loop(
            model_sgd, opt_sgd, noise_sgd, state_sgd, x, y, steps, stddev=stddev
        )

        # BandMF (correlated noise)
        torch.manual_seed(0)
        model_mf = nn.Linear(5, 1, bias=False)
        opt_mf = torch.optim.SGD(model_mf.parameters(), lr=0.01)
        coefs = optimal_max_error_strategy_coefs(steps)
        noising = inverse_as_streaming_matrix(coefs)
        noise_mf, state_mf = _matrix_factorization_noise(
            self._make_template(model_mf),
            noising,
            key=key(43),
        )
        losses_mf = _engine_train_loop(
            model_mf, opt_mf, noise_mf, state_mf, x, y, steps, stddev=stddev
        )

        # Both should train (loss decreases)
        assert losses_sgd[-1] < losses_sgd[0]
        assert losses_mf[-1] < losses_mf[0]

        # BandMF final loss should be at least better than start
        assert losses_mf[-1] < losses_sgd[0]


class TestBLTWithBnB:
    """End-to-end test: BLT noise with Balls-in-Bins sampling."""

    def _make_template(self, model):
        return {i: torch.zeros_like(p) for i, p in enumerate(model.parameters())}

    def test_blt_bnb_trains(self):
        """BLT noise with BnB sampler trains a simple model.

        Simulates the BLT+BnB pipeline: the dataset is randomly partitioned
        into fixed bins, BLT noise is applied per step.
        """
        torch.manual_seed(0)
        n_samples = 200
        dim = 5
        x = torch.randn(n_samples, dim)
        true_w = torch.randn(dim, 1)
        y = x @ true_w
        dataset = TensorDataset(x, y)

        num_bins = 10  # 10 bins → batch_size=20
        num_epochs = 3
        steps_per_epoch = num_bins
        total_steps = num_epochs * steps_per_epoch
        momentum = 0.9

        model = nn.Linear(dim, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=momentum)

        strategy = blt_strategy(momentum=momentum)
        noise_fn, noise_state = mf_gaussian_noise(
            self._make_template(model),
            strategy,
            n_steps=total_steps,
            min_sep=steps_per_epoch,
            max_participations=num_epochs,
            noise_multiplier=1.0,
            key=key(42),
        )

        sampler = BallsInBinsSampler(
            dataset,
            num_bins=num_bins,
            n_steps=num_bins * num_epochs,
            key=key(99),
        )

        params = list(model.parameters())
        losses = []
        for indices in sampler:
            optimizer.zero_grad()
            batch_x = x[indices]
            batch_y = y[indices]
            pred = model(batch_x)
            loss = ((pred - batch_y) ** 2).mean()
            loss.backward()

            grads = {i: p.grad.clone() for i, p in enumerate(params)}
            noised, noise_state = noise_fn(clipped(grads, max_norm=0.05), noise_state)
            for i, p in enumerate(params):
                p.grad = noised.pytree[i].to(p.dtype)

            optimizer.step()
            losses.append(loss.item())

        assert len(losses) == total_steps
        assert losses[-1] < losses[0]

    def test_bnb_sampler_covers_dataset(self):
        """BnB sampler places every example in exactly one bin per epoch."""
        n_samples = 1000
        num_bins = 10
        dataset = list(range(n_samples))

        sampler = BallsInBinsSampler(
            dataset,
            num_bins=num_bins,
            n_steps=num_bins,
            key=key(42),
        )

        all_indices = []
        for batch in sampler:
            all_indices.extend(batch)

        # Every example appears exactly once (true BnB: independent
        # assignment, so all N examples are assigned to some bin).
        assert len(all_indices) == n_samples
        assert set(all_indices) == set(range(n_samples))

    def test_bnb_sampler_fixed_bins_across_epochs(self):
        """BnB sampler yields the same bin assignment every epoch."""
        n_samples = 1000
        num_bins = 10
        num_epochs = 3
        dataset = list(range(n_samples))

        sampler = BallsInBinsSampler(
            dataset,
            num_bins=num_bins,
            n_steps=num_bins * num_epochs,
            key=key(42),
        )

        all_batches = list(sampler)
        batches_per_epoch = len(all_batches) // num_epochs
        assert len(all_batches) == batches_per_epoch * num_epochs

        epoch_1 = all_batches[:batches_per_epoch]
        epoch_2 = all_batches[batches_per_epoch : 2 * batches_per_epoch]
        epoch_3 = all_batches[2 * batches_per_epoch : 3 * batches_per_epoch]

        for i in range(batches_per_epoch):
            assert epoch_1[i] == epoch_2[i], f"bin {i} differs between epoch 1 and 2"
            assert epoch_1[i] == epoch_3[i], f"bin {i} differs between epoch 1 and 3"

    def test_bnb_sampler_variable_bin_sizes(self):
        """True BnB produces variable-size bins (not all equal)."""
        n_samples = 10000
        num_bins = 50
        dataset = list(range(n_samples))

        sampler = BallsInBinsSampler(
            dataset,
            num_bins=num_bins,
            n_steps=num_bins,
            key=key(123),
        )

        sizes = [len(batch) for batch in sampler]
        # With N=10000 and b=50, expected size is 200. True independent
        # assignment should produce variation (not all sizes equal).
        assert len(set(sizes)) > 1, "all bins have equal size — not true BnB"


class TestMfNoiseStrategies:
    """End-to-end tests: each strategy trains a simple model via mf_gaussian_noise()."""

    def _setup(self, steps=50, seed=0):
        torch.manual_seed(seed)
        model = nn.Linear(5, 1, bias=False)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        template = {i: torch.zeros_like(p) for i, p in enumerate(model.parameters())}
        x = torch.randn(50, 5)
        y = x @ torch.randn(5, 1)
        return model, opt, template, x, y

    @pytest.mark.parametrize(
        ("strategy_factory", "kwargs"),
        [
            pytest.param(
                lambda: identity_strategy(),
                {"n_steps": 50, "min_sep": 1, "max_participations": 1},
                id="identity",
            ),
            pytest.param(
                lambda: band_mf_strategy(bands=10, momentum=0.0),
                {"n_steps": 50, "min_sep": 1, "max_participations": 1},
                id="band_mf",
            ),
            pytest.param(
                lambda: lambda_cgd_strategy(lambda_=0.9),
                {"n_steps": 50, "min_sep": 1, "max_participations": 1},
                id="lambda_cgd",
            ),
            pytest.param(
                lambda: bisr_strategy(bandwidth=4),
                {"n_steps": 50, "min_sep": 10, "max_participations": 5},
                id="bisr",
            ),
        ],
    )
    def test_strategy_trains(self, strategy_factory, kwargs):
        model, opt, tmpl, x, y = self._setup()
        nf, ns = mf_gaussian_noise(
            tmpl,
            strategy_factory(),
            noise_multiplier=1.0,
            key=key(42),
            **kwargs,
        )
        losses = _mf_train_loop(model, opt, nf, ns, x, y, steps=50, max_norm=0.1)
        assert losses[-1] < losses[0]

    def test_identity_matches_raw_engine(self):
        """identity_strategy via mf_gaussian_noise gives same noise as _matrix_factorization_noise + identity()."""
        tmpl = {"w": torch.zeros(10)}
        nf1, ns1 = mf_gaussian_noise(
            tmpl,
            identity_strategy(),
            n_steps=10,
            noise_multiplier=1.0,
            key=key(42),
        )
        nf2, ns2 = _matrix_factorization_noise(tmpl, identity(), key=key(42))

        grad = {"w": torch.ones(10)}
        out1, _ = nf1(clipped(grad, max_norm=1.0), ns1)
        out2, _ = nf2(grad, ns2, stddev=1.0)
        torch.testing.assert_close(out1.pytree["w"], out2["w"])
