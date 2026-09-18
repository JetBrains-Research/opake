"""Engine-only tests for RNG convenience helpers.

DP-SGD-aware integration tests (helpers used with ``gaussian_noise``)
live in ``packages/opake-dpsgd/tests/rng/test_rng_helpers_integration.py``;
opake-engine has no dependency on opake-dpsgd.
"""

import pytest
import torch

from opake.random import key, random_key, set_reproducible_pytorch_seed
from opake.random.types import RngKey


class TestRandomKey:
    """Test non-deterministic key generation for prototyping."""

    def test_returns_rng_key(self):
        """Should return an RngKey instance."""
        k = random_key()
        assert isinstance(k, RngKey)

    def test_different_on_each_call(self):
        """Should generate different keys on consecutive calls."""
        k1 = random_key()
        k2 = random_key()
        assert k1.seed != k2.seed

    def test_produces_working_generators(self):
        """Generated keys should work with generator_from_key."""
        from opake.random import generator_from_key

        k = random_key()
        gen = generator_from_key(k)
        assert isinstance(gen, torch.Generator)

        # Should produce different random values
        t1 = torch.randn(10, generator=gen)
        k2 = random_key()
        gen2 = generator_from_key(k2)
        t2 = torch.randn(10, generator=gen2)
        assert not torch.allclose(t1, t2)


class TestReproducibleSeed:
    """``set_reproducible_pytorch_seed`` seeds every available device RNG."""

    @pytest.fixture(autouse=True)
    def _restore_deterministic_algorithms(self):
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            yield
        finally:
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)

    def test_cpu_reproducible(self):
        set_reproducible_pytorch_seed(key(42))
        a = torch.randn(1000)
        set_reproducible_pytorch_seed(key(42))
        b = torch.randn(1000)
        assert torch.equal(a, b)

    @pytest.mark.mps
    def test_mps_reproducible(self):
        """The MPS generator must be seeded too, not just CPU/CUDA.

        Regression guard: model-side stochastic ops (dropout, weight init) on
        Apple Silicon are only reproducible if ``torch.mps.manual_seed`` runs.
        """
        set_reproducible_pytorch_seed(key(42))
        a = torch.randn(1000, device="mps").cpu()
        set_reproducible_pytorch_seed(key(42))
        b = torch.randn(1000, device="mps").cpu()
        assert torch.equal(a, b)

        set_reproducible_pytorch_seed(key(43))
        c = torch.randn(1000, device="mps").cpu()
        assert not torch.equal(a, c)
