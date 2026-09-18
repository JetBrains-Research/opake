"""End-to-end accounting tests for ``opake.dpsgd.accounting``."""

from __future__ import annotations


class TestEndToEndCalibration:
    """Constructed mechanisms compute valid PLDs through the new namespace."""

    def test_poisson_gaussian(self):
        import math

        import opake.dpsgd.accounting as dpsgd_acc

        proc = dpsgd_acc.poisson(dpsgd_acc.gaussian(1.1), 0.01) * 1000
        eps = proc.epsilon_at(1e-5)
        assert math.isfinite(eps)
        assert eps > 0

    def test_truncated_poisson_gaussian(self):
        import math

        import opake.dpsgd.accounting as dpsgd_acc

        proc = dpsgd_acc.poisson(
            dpsgd_acc.gaussian(0.8),
            0.01,
            truncated_batch_size=128,
            dataset_size=10_000,
        )
        eps = (proc * 100).epsilon_at(1e-5)
        assert math.isfinite(eps)
        assert eps > 0
