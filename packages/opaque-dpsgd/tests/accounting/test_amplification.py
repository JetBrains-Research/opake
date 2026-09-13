"""DP-SGD amplification — Poisson (optionally truncated), ParallelPoisson,
and KOutOfT."""

import math
from collections.abc import Callable
from dataclasses import FrozenInstanceError

import pytest

import opaque.accounting as acc
import opaque.dpsgd.accounting as dpsgd_acc
from opaque.api.accounting.core import _native
from opaque.api.accounting.core._base import DpProcess
from opaque.dpsgd.accounting.amplification.types import (
    KOutOfT,
    ParallelPoisson,
    Poisson,
)
from opaque.dpsgd.accounting.mechanisms.types import Gaussian
from opaque.exceptions import ConfigurationError
from opaque.serialization import from_state_dict, state_dict

# ── Amplification dataclass tests ────────────────────────────────────


class TestPoissonDataclass:
    """Poisson frozen dataclass (plain Poisson)."""

    def test_fields(self):
        g = Gaussian(0.8)
        p = Poisson(g, 0.01)
        assert p.inner is g
        assert p.sample_rate == pytest.approx(0.01)
        assert p.truncated_batch_size is None
        assert p.dataset_size is None

    def test_frozen(self):
        p = Poisson(Gaussian(0.8), 0.01)
        with pytest.raises(FrozenInstanceError):
            p.sample_rate = 0.1  # type: ignore[misc]

    def test_is_dp_process(self):
        assert isinstance(Poisson(Gaussian(0.8), 0.01), DpProcess)

    def test_equality(self):
        assert Poisson(Gaussian(0.8), 0.01) == Poisson(Gaussian(0.8), 0.01)
        assert Poisson(Gaussian(0.8), 0.01) != Poisson(Gaussian(0.8), 0.02)

    def test_pld_returns_valid(self):
        pld = Poisson(Gaussian(0.8), 0.01).pld()
        eps = pld.epsilon_at(1e-5)
        assert math.isfinite(eps)
        assert eps > 0

    @pytest.mark.parametrize("sample_rate", [0.0, -0.01, 1.01])
    def test_rejects_invalid_sample_rate(self, sample_rate):
        with pytest.raises(
            ConfigurationError, match=r"sample_rate must be in \(0, 1\]"
        ):
            Poisson(Gaussian(0.8), sample_rate)

    def test_accepts_full_participation_rate(self):
        assert Poisson(Gaussian(0.8), 1.0).sample_rate == pytest.approx(1.0)

    def test_full_participation_matches_gaussian(self):
        poisson_eps = Poisson(Gaussian(0.8), 1.0).epsilon_at(1e-5)
        gaussian_eps = Gaussian(0.8).epsilon_at(1e-5)
        assert poisson_eps == pytest.approx(gaussian_eps)

    def test_rejects_full_rate_with_truncation(self):
        with pytest.raises(ConfigurationError, match=r"sample_rate=1\.0"):
            Poisson(Gaussian(0.8), 1.0, truncated_batch_size=128, dataset_size=10_000)


class TestPoissonFullRateSerialization:
    """A Poisson process stored with sample_rate=1.0 still round-trips."""

    def test_state_dict_round_trip_at_q1(self):
        from opaque.serialization import from_state_dict, state_dict

        proc = Poisson(Gaussian(0.8), 1.0)
        restored = from_state_dict(Poisson(Gaussian(0.8), 0.5), state_dict(proc))
        assert isinstance(restored, Poisson)
        assert restored.sample_rate == pytest.approx(1.0)
        assert restored.epsilon_at(1e-5) == pytest.approx(
            Gaussian(0.8).epsilon_at(1e-5)
        )


class TestPoissonTruncatedDataclass:
    """Poisson with truncation switched on."""

    def test_fields(self):
        g = Gaussian(0.8)
        t = Poisson(g, 0.01, truncated_batch_size=128, dataset_size=10_000)
        assert t.inner is g
        assert t.sample_rate == pytest.approx(0.01)
        assert t.truncated_batch_size == 128
        assert t.dataset_size == 10_000

    def test_pld_returns_valid(self):
        pld = Poisson(
            Gaussian(0.8), 0.01, truncated_batch_size=128, dataset_size=10_000
        ).pld()
        eps = pld.epsilon_at(1e-5)
        assert math.isfinite(eps)
        assert eps > 0


class TestParallelPoissonDataclass:
    """ParallelPoisson frozen dataclass."""

    def test_fields(self):
        inner = Poisson(Gaussian(0.8), 0.01)
        a = ParallelPoisson(inner, 4)
        assert a.inner is inner
        assert a.num_workers == 4

    def test_frozen(self):
        a = ParallelPoisson(Poisson(Gaussian(0.8), 0.01), 4)
        with pytest.raises(FrozenInstanceError):
            a.num_workers = 8  # type: ignore[misc]

    def test_is_dp_process(self):
        assert isinstance(ParallelPoisson(Poisson(Gaussian(0.8), 0.01), 4), DpProcess)

    def test_pld_returns_valid(self):
        pld = ParallelPoisson(Poisson(Gaussian(0.8), 0.01), 4).pld()
        eps = pld.epsilon_at(1e-5)
        assert math.isfinite(eps)
        assert eps > 0

    @pytest.mark.parametrize("num_workers", [0, -1, 1.5, True])
    def test_rejects_invalid_num_workers(self, num_workers):
        with pytest.raises(
            ConfigurationError, match="num_workers must be a positive integer"
        ):
            ParallelPoisson(Poisson(Gaussian(0.8), 0.01), num_workers)  # type: ignore[arg-type]

    def test_rejects_truncated_poisson_inner(self):
        truncated = Poisson(
            Gaussian(0.8), 0.01, truncated_batch_size=128, dataset_size=10_000
        )
        with pytest.raises(ValueError, match="does not support truncated"):
            ParallelPoisson(truncated, 4)


# ── Constructor function tests ───────────────────────────────────────


class TestPoissonConstructor:
    """dpsgd_acc.poisson() validates inner type and returns Poisson."""

    def test_returns_poisson(self):
        p = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.8), 0.01)
        assert isinstance(p, Poisson)
        assert isinstance(p.inner, Gaussian)
        assert p.inner.noise_multiplier == pytest.approx(0.8)
        assert p.sample_rate == pytest.approx(0.01)
        assert p.truncated_batch_size is None
        assert p.dataset_size is None

    def test_accepts_generic_dp_process(self):
        step = dpsgd_acc.poisson(acc.eps_delta(0.3), 0.2)

        assert step.inner == acc.eps_delta(0.3)
        assert math.isfinite(step.epsilon_at(1e-5))

    def test_full_participation_matches_generic_inner(self):
        inner = acc.eps_delta(0.3)
        step = dpsgd_acc.poisson(inner, 1.0)

        assert step.epsilon_at(1e-5) == pytest.approx(inner.epsilon_at(1e-5))

    def test_serializes_generic_dp_process(self):
        step = dpsgd_acc.poisson(acc.eps_delta(0.3), 0.2)
        restored = from_state_dict(Poisson(Gaussian(1.0), 0.2), state_dict(step))

        assert restored == step
        assert restored.epsilon_at(1e-5) == pytest.approx(step.epsilon_at(1e-5))

    def test_rejects_non_process(self):
        with pytest.raises(TypeError, match="DpProcess"):
            dpsgd_acc.poisson("bad", 0.01)  # type: ignore[arg-type]

    @pytest.mark.parametrize("sample_rate", [0.0, -0.01, 1.01])
    def test_rejects_invalid_sample_rate(self, sample_rate):
        with pytest.raises(
            ConfigurationError, match=r"sample_rate must be in \(0, 1\]"
        ):
            dpsgd_acc.poisson(dpsgd_acc.gaussian(0.8), sample_rate)

    def test_accepts_full_participation_rate(self):
        step = dpsgd_acc.poisson(dpsgd_acc.gaussian(0.8), 1.0)
        assert step.sample_rate == pytest.approx(1.0)
        assert step.epsilon_at(1e-5) == pytest.approx(
            dpsgd_acc.gaussian(0.8).epsilon_at(1e-5)
        )

    def test_full_participation_adaclip_matches_gaussian(self):
        ac = dpsgd_acc.adaclip(dpsgd_acc.gaussian(0.8), expected_batch_size=1000)
        step = dpsgd_acc.poisson(ac, 1.0)
        assert step.epsilon_at(1e-5) == pytest.approx(
            dpsgd_acc.gaussian(ac.effective_noise_multiplier).epsilon_at(1e-5)
        )

    def test_accepts_adaclip(self):
        step = dpsgd_acc.poisson(
            dpsgd_acc.adaclip(dpsgd_acc.gaussian(0.8), expected_batch_size=1000), 0.01
        )
        eps = step.epsilon_at(1e-5)
        assert math.isfinite(eps)
        assert eps > 0

    def test_propagates_config(self):
        """Config is now query-time, so this test verifies pld() accepts discretization."""
        g = dpsgd_acc.gaussian(0.8)
        p = dpsgd_acc.poisson(g, 0.01)
        pld1 = p.pld(discretization=1e-3)
        pld2 = p.pld(discretization=1e-4)
        eps1 = pld1.epsilon_at(1e-5)
        eps2 = pld2.epsilon_at(1e-5)
        assert math.isfinite(eps1)
        assert eps1 > 0
        assert math.isfinite(eps2)
        assert eps2 > 0

    def test_gaussian_uses_generic_native_path(self):
        sigma, sample_rate, steps, delta = 0.6, 0.001, 1000, 1e-8
        assert not hasattr(_native, "poisson_gaussian_pld")
        process = dpsgd_acc.poisson(dpsgd_acc.gaussian(sigma), sample_rate) * steps
        assert math.isfinite(process.epsilon_at(delta))


class TestPoissonTruncatedConstructor:
    """dpsgd_acc.poisson(..., truncated_batch_size=..., dataset_size=...)."""

    def test_returns_truncated_poisson(self):
        t = dpsgd_acc.poisson(
            dpsgd_acc.gaussian(0.8),
            0.01,
            truncated_batch_size=128,
            dataset_size=10_000,
        )
        assert isinstance(t, Poisson)
        assert t.truncated_batch_size == 128
        assert t.dataset_size == 10_000

    def test_rejects_non_gaussian(self):
        with pytest.raises(TypeError, match="truncated Poisson"):
            dpsgd_acc.poisson(
                acc.eps_delta(1.0),
                0.01,
                truncated_batch_size=128,
                dataset_size=10_000,
            )  # type: ignore[arg-type]

    def test_accepts_adaclip(self):
        step = dpsgd_acc.poisson(
            dpsgd_acc.adaclip(dpsgd_acc.gaussian(0.8), expected_batch_size=1000),
            0.01,
            truncated_batch_size=128,
            dataset_size=10_000,
        )
        eps = step.epsilon_at(1e-5)
        assert math.isfinite(eps)
        assert eps > 0

    def test_requires_both_truncation_args(self):
        with pytest.raises(
            ConfigurationError, match="truncated_batch_size and dataset_size"
        ):
            dpsgd_acc.poisson(dpsgd_acc.gaussian(0.8), 0.01, truncated_batch_size=128)
        with pytest.raises(
            ConfigurationError, match="truncated_batch_size and dataset_size"
        ):
            dpsgd_acc.poisson(dpsgd_acc.gaussian(0.8), 0.01, dataset_size=10_000)

    @pytest.mark.parametrize(
        ("field", "kwargs"),
        [
            (
                "truncated_batch_size",
                {"truncated_batch_size": 128.0, "dataset_size": 10_000},
            ),
            ("dataset_size", {"truncated_batch_size": 128, "dataset_size": 10_000.0}),
        ],
    )
    def test_requires_integral_truncation_parameters(self, field, kwargs):
        with pytest.raises(TypeError, match=field):
            dpsgd_acc.poisson(dpsgd_acc.gaussian(0.8), 0.01, **kwargs)

    def test_rejects_full_rate_with_truncation(self):
        with pytest.raises(ConfigurationError, match=r"sample_rate=1\.0"):
            dpsgd_acc.poisson(
                dpsgd_acc.gaussian(0.8),
                1.0,
                truncated_batch_size=128,
                dataset_size=10_000,
            )


class TestParallelPoissonConstructor:
    """dpsgd_acc.parallel_poisson() takes (Gaussian, sample_rate, num_workers)."""

    def test_returns_parallel_poisson(self):
        a = dpsgd_acc.parallel_poisson(
            dpsgd_acc.gaussian(0.8), sample_rate=0.01, num_workers=4
        )
        assert isinstance(a, ParallelPoisson)
        assert a.num_workers == 4

    def test_rejects_non_gaussian(self):
        with pytest.raises(TypeError, match="Gaussian"):
            dpsgd_acc.parallel_poisson("bad", sample_rate=0.01, num_workers=4)  # type: ignore[arg-type]

    @pytest.mark.parametrize("sample_rate", [0.0, 1.0, -0.01, 1.01])
    def test_rejects_invalid_sample_rate(self, sample_rate):
        with pytest.raises(
            ConfigurationError, match=r"sample_rate must be in \(0, 1\)"
        ):
            dpsgd_acc.parallel_poisson(
                dpsgd_acc.gaussian(0.8), sample_rate=sample_rate, num_workers=4
            )

    @pytest.mark.parametrize("num_workers", [0, -1, 1.5, True])
    def test_rejects_invalid_num_workers(self, num_workers):
        with pytest.raises(
            ConfigurationError, match="num_workers must be a positive integer"
        ):
            dpsgd_acc.parallel_poisson(
                dpsgd_acc.gaussian(0.8), sample_rate=0.01, num_workers=num_workers
            )


# ── Parallel Poisson automatic truncation tests ──────────────────────


class TestParallelPoissonAutoTruncation:
    """Automatic truncation from query-time discretization settings."""

    @pytest.mark.slow
    def test_auto_respects_query_time_discretization_overrides(self):
        nm = 0.8
        q = 0.0032
        m = 8
        delta = 1e-8
        auto = dpsgd_acc.parallel_poisson(
            dpsgd_acc.gaussian(nm), sample_rate=q, num_workers=m
        )

        eps_tight = auto.epsilon_at(delta, log_x_mass_truncation_bound=-50.0)
        eps_loose = auto.epsilon_at(delta, log_x_mass_truncation_bound=-15.0)

        assert eps_loose >= eps_tight - 1e-10


# ── Block and total k-out-of-t ────────────────────────────────────────


class TestKOutOfTDataclass:
    """KOutOfT frozen dataclass."""

    def test_fields(self):
        g = Gaussian(0.8)
        r = KOutOfT(g, 4, 64, "block")
        assert r.inner is g
        assert r.k == 4
        assert r.t == 64
        assert r.n_steps == 64

    def test_frozen(self):
        r = KOutOfT(Gaussian(0.8), 4, 64, "block")
        with pytest.raises(FrozenInstanceError):
            r.k = 2  # type: ignore[misc]

    def test_is_dp_process(self):
        assert isinstance(KOutOfT(Gaussian(0.8), 4, 64, "block"), DpProcess)

    def test_equality(self):
        assert KOutOfT(Gaussian(0.8), 4, 64, "block") == KOutOfT(
            Gaussian(0.8), 4, 64, "block"
        )
        assert KOutOfT(Gaussian(0.8), 4, 64, "block") != KOutOfT(
            Gaussian(0.8), 4, 64, "total"
        )

    def test_block_sizes_cover_uneven_horizon(self):
        assert KOutOfT(Gaussian(0.8), 3, 10, "block").block_sizes == (3, 3, 4)

    def test_validates_on_direct_construction(self):
        """Deserialization calls ``cls(**kwargs)``, bypassing the factory, so
        the bound has to live in ``__post_init__``."""
        with pytest.raises(ValueError, match="allocation"):
            KOutOfT(Gaussian(0.8), 3, 64, "bad")  # type: ignore[arg-type]

    @pytest.mark.slow
    def test_pld_returns_valid(self):
        eps = KOutOfT(Gaussian(1.0), 4, 32, "block").pld().epsilon_at(1e-8)
        assert math.isfinite(eps)
        assert eps > 0


class TestKOutOfTConstructor:
    """dpsgd_acc.k_out_of_t() takes a declared allocation type."""

    def test_returns_k_out_of_t(self):
        r = dpsgd_acc.k_out_of_t(dpsgd_acc.gaussian(0.8), k=4, t=64, allocation="block")
        assert isinstance(r, KOutOfT)
        assert r.k == 4

    def test_k_and_t_are_keyword_only(self):
        with pytest.raises(TypeError):
            dpsgd_acc.k_out_of_t(dpsgd_acc.gaussian(0.8), 4, 64)  # type: ignore[misc]

    def test_rejects_non_gaussian(self):
        with pytest.raises(TypeError, match="Gaussian"):
            dpsgd_acc.k_out_of_t("bad", k=4, t=64, allocation="block")  # type: ignore[arg-type]

    def test_accepts_total_allocation_with_conservative_accounting(self):
        r = dpsgd_acc.k_out_of_t(dpsgd_acc.gaussian(0.8), k=4, t=64, allocation="total")
        assert r.allocation == "total"

    @pytest.mark.slow
    def test_accepts_adaclip(self):
        r = dpsgd_acc.k_out_of_t(
            dpsgd_acc.adaclip(dpsgd_acc.gaussian(1.0), expected_batch_size=256),
            k=4,
            t=32,
            allocation="block",
        )
        eps = r.epsilon_at(1e-8)
        assert math.isfinite(eps)
        assert eps > 0

    def test_nonprivate_inner_is_infinite(self):
        r = dpsgd_acc.k_out_of_t(acc.nonprivate(), k=4, t=32, allocation="block")
        assert math.isinf(r.epsilon_at(1e-8))

    def test_zero_noise_gaussian_is_infinite(self):
        """``Gaussian(0)`` short-circuits before reaching the native primitive,
        which requires ``σ > 0``."""
        r = dpsgd_acc.k_out_of_t(dpsgd_acc.gaussian(0.0), k=4, t=32, allocation="block")
        assert math.isinf(r.epsilon_at(1e-8))


class TestKOutOfTTightness:
    """The reason the process exists: it beats Poisson at the matched rate."""

    @pytest.mark.slow
    def test_below_poisson_at_matched_rate(self):
        b, sigma, delta = 16, 1.0, 1e-8
        ra = dpsgd_acc.k_out_of_t(
            dpsgd_acc.gaussian(sigma), k=1, t=b, allocation="block"
        )
        po = dpsgd_acc.poisson(dpsgd_acc.gaussian(sigma), 1.0 / b)
        assert ra.epsilon_at(delta) < (po * b).epsilon_at(delta)

    @pytest.mark.slow
    def test_monotone_in_num_bins(self):
        """More bins to hide among, less privacy loss per epoch."""
        sigma, delta = 1.0, 1e-8
        eps = [
            dpsgd_acc.k_out_of_t(
                dpsgd_acc.gaussian(sigma), k=1, t=b, allocation="block"
            ).epsilon_at(delta)
            for b in (4, 8, 16)
        ]
        assert eps[0] > eps[1] > eps[2]


# ── Deterministic regression vectors ──────────────────────────────────


def _adaclip() -> DpProcess:
    return dpsgd_acc.adaclip(
        dpsgd_acc.gaussian(1.1),
        expected_batch_size=250,
        num_groups=3,
    )


_DeterministicAmplificationFactory = Callable[[], DpProcess]


class TestDeterministicAmplificationVectors:
    """Committed ε values for deterministic amplification combinations."""

    @pytest.mark.slow
    @pytest.mark.parametrize(
        ("name", "factory", "delta", "expected"),
        [
            pytest.param(
                "poisson(adaclip(gaussian(1.1)), q=0.01) * 200",
                lambda: dpsgd_acc.poisson(_adaclip(), 0.01) * 200,
                1e-5,
                0.7256797187172928,
                id="poisson-adaclip",
            ),
            pytest.param(
                "poisson(gaussian(1.1), q=0.01, cap=64, n=50000) * 200",
                lambda: (
                    dpsgd_acc.poisson(
                        dpsgd_acc.gaussian(1.1),
                        0.01,
                        truncated_batch_size=64,
                        dataset_size=50_000,
                    )
                    * 200
                ),
                1e-5,
                1.8536078379130241,
                id="truncated-poisson-gaussian",
            ),
            pytest.param(
                "poisson(adaclip(gaussian(1.1)), q=0.01, cap=64, n=50000) * 200",
                lambda: (
                    dpsgd_acc.poisson(
                        _adaclip(),
                        0.01,
                        truncated_batch_size=64,
                        dataset_size=50_000,
                    )
                    * 200
                ),
                1e-5,
                1.8789749867731147,
                id="truncated-poisson-adaclip",
            ),
            pytest.param(
                "parallel_poisson(adaclip(gaussian(1.1)), q=0.01, workers=4) * 200",
                lambda: dpsgd_acc.parallel_poisson(_adaclip(), 0.01, 4) * 200,
                1e-5,
                3.3561060950994523,
                id="parallel-poisson-adaclip",
            ),
            pytest.param(
                "k_out_of_t(gaussian(1.0), k=2, t=16, block)",
                lambda: dpsgd_acc.k_out_of_t(
                    dpsgd_acc.gaussian(1.0),
                    k=2,
                    t=16,
                    allocation="block",
                ),
                1e-8,
                4.687320185091083,
                id="k-out-of-t-gaussian",
            ),
            pytest.param(
                "k_out_of_t(adaclip(gaussian(1.1)), k=2, t=16, block)",
                lambda: dpsgd_acc.k_out_of_t(_adaclip(), k=2, t=16, allocation="block"),
                1e-8,
                3.9650600884447935,
                id="k-out-of-t-adaclip",
            ),
        ],
    )
    def test_epsilon_matches_committed_vector(
        self,
        name: str,
        factory: _DeterministicAmplificationFactory,
        delta: float,
        expected: float,
    ):
        actual = factory().epsilon_at(delta)

        # The native PLD calculation differs by a few nanounits across CPU
        # architectures, so retain the tight relative check with a matching
        # absolute allowance.
        assert actual == pytest.approx(expected, rel=1e-9, abs=3e-9), (
            f"{name}, delta={delta}: epsilon drifted; "
            f"committed={expected:.17g}, observed={actual:.17g}"
        )
