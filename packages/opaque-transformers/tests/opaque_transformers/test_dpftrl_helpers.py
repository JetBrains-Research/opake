# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the DP-FTRL helper module (``_dpftrl``).

Covers strategy construction, amplifier wiring, per-step composition,
and sampler dispatch — the pure-function surface that the trainer
calls into.  End-to-end Trainer + DP-FTRL is covered by
``tests/validation/test_dp_ftrl_trainer.py``.
"""

from __future__ import annotations

import pytest
from torch.utils.data import Dataset

from opaque.api.accounting.core._horizon import DpHorizonProcess
from opaque.api.transformers.trainer import _dpftrl
from opaque.dpftrl import (
    BallsInBinsSampler,
    BMinSepSampler,
    CyclicPoissonSampler,
    SequentialBatchSampler,
)
from opaque.dpftrl.noise.types import (
    BandMfStrategy,
    BisrStrategy,
    BltStrategy,
    BsrStrategy,
    IdentityStrategy,
    LambdaCgdStrategy,
)
from opaque.dpsgd.sampling import KOutOfTSampler, PoissonSampler
from opaque.random import key
from opaque.serialization import state_dict

_STRATEGY_CASES = {
    "mf_band": ({"bands": 4}, BandMfStrategy),
    "mf_blt": ({"max_buffers": 4}, BltStrategy),
    "mf_bisr": ({"bandwidth": 4}, BisrStrategy),
    "mf_bsr": ({"bandwidth": 4, "alpha": 1.0, "beta": 0.9}, BsrStrategy),
    "mf_lambda_cgd": ({"lambda_": 0.5}, LambdaCgdStrategy),
    "mf_identity": ({}, IdentityStrategy),
}


class _ListDataset(Dataset):
    def __init__(self, n: int = 32) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int) -> int:
        return i


class TestBuildStrategyLrSchedule:
    """``lr_schedule`` auto-injection for BandMF / BLT only."""

    def test_band_mf_receives_lr_schedule(self):
        from opaque.scheduling import cosine_schedule

        sched = cosine_schedule(1e-3, 0.0, 1000)
        strategy = _dpftrl.build_strategy("mf_band", {"bands": 4}, lr_schedule=sched)
        assert isinstance(strategy, BandMfStrategy)
        assert strategy.lr_schedule is sched

    def test_blt_receives_lr_schedule(self):
        from opaque.scheduling import cosine_schedule

        sched = cosine_schedule(1e-3, 0.0, 1000)
        strategy = _dpftrl.build_strategy(
            "mf_blt", {"max_buffers": 4}, lr_schedule=sched
        )
        assert isinstance(strategy, BltStrategy)
        assert strategy.lr_schedule is sched

    def test_other_mechanisms_ignore_lr_schedule(self):
        # BiSR/BSR/lambda_cgd/identity don't accept an lr_schedule kwarg;
        # build_strategy must not pass one through (no TypeError from the
        # factory).
        from opaque.scheduling import cosine_schedule

        sched = cosine_schedule(1e-3, 0.0, 1000)
        for mech, kw in [
            ("mf_bisr", {"bandwidth": 4}),
            ("mf_bsr", {"bandwidth": 4, "alpha": 1.0, "beta": 0.9}),
            ("mf_lambda_cgd", {"lambda_": 0.5}),
            ("mf_identity", {}),
        ]:
            strategy = _dpftrl.build_strategy(mech, kw, lr_schedule=sched)
            assert not hasattr(strategy, "lr_schedule") or strategy.lr_schedule is None

    def test_user_kwarg_wins_over_auto_injection(self):
        from opaque.scheduling import constant_schedule, cosine_schedule

        live = cosine_schedule(1e-3, 0.0, 1000)
        user = constant_schedule(5e-4)
        strategy = _dpftrl.build_strategy(
            "mf_band",
            {"bands": 4, "lr_schedule": user},
            lr_schedule=live,
        )
        assert strategy.lr_schedule is user


class TestBuildStrategy:
    @pytest.mark.parametrize("mechanism", _STRATEGY_CASES)
    def test_builds_strategy(self, mechanism):
        kwargs, strategy_type = _STRATEGY_CASES[mechanism]
        strategy = _dpftrl.build_strategy(mechanism, kwargs)
        assert isinstance(strategy, strategy_type)

    def test_cases_cover_registered_strategies(self):
        assert _STRATEGY_CASES.keys() == _dpftrl._STRATEGY_FACTORIES.keys()

    @pytest.mark.parametrize("mechanism", _STRATEGY_CASES)
    def test_rejects_unknown_kwargs(self, mechanism):
        kwargs, _ = _STRATEGY_CASES[mechanism]
        unexpected_key = "__unexpected__"
        with pytest.raises(TypeError, match="unexpected keyword argument") as exc_info:
            _dpftrl.build_strategy(
                mechanism,
                {**kwargs, unexpected_key: 1},
            )
        assert unexpected_key in str(exc_info.value)

    def test_identity_accepts_none_kwargs(self):
        assert _dpftrl.build_strategy("mf_identity", None) == _dpftrl.build_strategy(
            "mf_identity", {}
        )


class TestBuildAmplifierFactory:
    @pytest.mark.parametrize(
        ("sampling_mode", "mechanism", "kwargs"),
        [
            ("poisson", "mf_identity", {}),
            ("b_min_sep", "mf_band", {"bands": 4}),
            ("cyclic_poisson", "mf_band", {"bands": 4}),
            ("balls_in_bins", "mf_blt", {"max_buffers": 4}),
        ],
    )
    def test_resume_comparison_only_normalizes_unused_inner_horizon(
        self, sampling_mode, mechanism, kwargs
    ):
        amp = _dpftrl.build_amplifier_factory(
            sampling_mode=sampling_mode,
            strategy=_dpftrl.build_strategy(mechanism, kwargs),
            sample_rate=0.05,
            n_steps=100,
            num_bins=10,
            dataset_size=1000,
            truncated_batch_size=None,
        )
        current = state_dict(amp(1.0))
        saved = {**current, "inner": {**current["inner"], "n_steps": 1}}
        assert _dpftrl.resume_process_state(current) == saved
        assert _dpftrl.resume_process_state(saved) == saved
        assert current["inner"]["n_steps"] is None
        assert _dpftrl.resume_process_state(current["inner"])["n_steps"] is None
        assert _dpftrl.resume_process_state({**current, "n_steps": 200}) != saved
        assert _dpftrl.resume_process_state(state_dict(amp(2.0))) != saved

    def test_identity_poisson(self):
        strategy = _dpftrl.build_strategy("mf_identity", {})
        amp = _dpftrl.build_amplifier_factory(
            sampling_mode="poisson",
            strategy=strategy,
            sample_rate=0.05,
            n_steps=100,
            num_bins=10,
            dataset_size=1000,
            truncated_batch_size=None,
        )
        proc = amp(1.0)
        assert isinstance(proc, DpHorizonProcess)
        assert proc.n_steps == 100

    @pytest.mark.parametrize("bands", [1, 4])
    def test_band_rejects_poisson_strategy_mismatch(self, bands):
        """A whole-dataset Poisson accountant does not realise BandMF's
        grouped, rotating-active-group participation pattern — reject it
        rather than silently mis-accounting (issue #776)."""
        strategy = _dpftrl.build_strategy("mf_band", {"bands": bands})
        with pytest.raises(ValueError, match="cyclic_poisson"):
            _dpftrl.build_amplifier_factory(
                sampling_mode="poisson",
                strategy=strategy,
                sample_rate=0.05,
                n_steps=100,
                num_bins=10,
                dataset_size=1000,
                truncated_batch_size=None,
            )

    def test_band_b_min_sep(self):
        strategy = _dpftrl.build_strategy("mf_band", {"bands": 4})
        amp = _dpftrl.build_amplifier_factory(
            sampling_mode="b_min_sep",
            strategy=strategy,
            sample_rate=0.05,
            n_steps=100,
            num_bins=10,
            dataset_size=1000,
            truncated_batch_size=None,
        )
        proc = amp(1.0)
        assert proc.n_steps == 100
        assert proc.min_sep == 4

    @pytest.mark.parametrize(
        ("bands", "world_size", "sample_rate", "expected_rate"),
        [
            (4, 1, 0.05, 0.2),  # exact division: floor(1000/4)=250
            (3, 1, 0.1, 100 / 333),  # single-process remainder: floor(1000/3)=333
            (3, 4, 0.1, 0.1 * 250 / 83),  # DDP: local=1000//4=250, floor(250/3)=83
        ],
    )
    def test_band_cyclic_poisson_group_rate(
        self, bands, world_size, sample_rate, expected_rate
    ):
        """Group rate targets ``floor((dataset_size // world_size) / bands)``
        — the local per-rank group size ``CyclicPoissonSampler`` actually
        realises under DDP, not the global ``floor(dataset_size / bands)``
        (Choquette-Choo et al. 2023, Algorithm 2 / Theorem 4)."""
        strategy = _dpftrl.build_strategy("mf_band", {"bands": bands})
        amp = _dpftrl.build_amplifier_factory(
            sampling_mode="cyclic_poisson",
            strategy=strategy,
            sample_rate=sample_rate,
            n_steps=100,
            num_bins=10,
            dataset_size=1000,
            truncated_batch_size=None,
            world_size=world_size,
        )
        proc = amp(1.0)
        assert proc.n_steps == 100
        assert proc.sample_rate == pytest.approx(expected_rate)

    def test_band_cyclic_poisson_rejects_undersized_dataset(self):
        strategy = _dpftrl.build_strategy("mf_band", {"bands": 4})
        with pytest.raises(ValueError, match="bands >= 1"):
            _dpftrl.build_amplifier_factory(
                sampling_mode="cyclic_poisson",
                strategy=strategy,
                sample_rate=0.5,
                n_steps=100,
                num_bins=10,
                dataset_size=3,  # 3 // 4 == 0
                truncated_batch_size=None,
            )

    def test_band_cyclic_poisson_rejects_rate_above_one(self):
        strategy = _dpftrl.build_strategy("mf_band", {"bands": 4})
        with pytest.raises(ValueError, match="<= 1"):
            _dpftrl.build_amplifier_factory(
                sampling_mode="cyclic_poisson",
                strategy=strategy,
                sample_rate=0.3,  # 0.3 * 4 == 1.2 > 1
                n_steps=100,
                num_bins=10,
                dataset_size=1000,
                truncated_batch_size=None,
            )

    def test_cyclic_poisson_rejects_non_band_strategy(self):
        strategy = _dpftrl.build_strategy("mf_identity", {})
        with pytest.raises(ValueError, match="BandMF"):
            _dpftrl.build_amplifier_factory(
                sampling_mode="cyclic_poisson",
                strategy=strategy,
                sample_rate=0.05,
                n_steps=100,
                num_bins=10,
                dataset_size=1000,
                truncated_batch_size=None,
            )

    def test_cyclic_poisson_rejects_truncated_batch_size(self):
        strategy = _dpftrl.build_strategy("mf_band", {"bands": 4})
        with pytest.raises(ValueError, match="truncated_batch_size"):
            _dpftrl.build_amplifier_factory(
                sampling_mode="cyclic_poisson",
                strategy=strategy,
                sample_rate=0.05,
                n_steps=100,
                num_bins=10,
                dataset_size=1000,
                truncated_batch_size=16,
            )

    def test_blt_balls_in_bins(self):
        strategy = _dpftrl.build_strategy("mf_blt", {"max_buffers": 4})
        amp = _dpftrl.build_amplifier_factory(
            sampling_mode="balls_in_bins",
            strategy=strategy,
            sample_rate=0.1,
            n_steps=100,
            num_bins=10,
            dataset_size=1000,
            truncated_batch_size=None,
        )
        proc = amp(1.0)
        assert proc.n_steps == 100

    def test_unknown_sampling_mode_raises(self):
        strategy = _dpftrl.build_strategy("mf_band", {"bands": 4})
        with pytest.raises(ValueError, match="sampling_mode"):
            _dpftrl.build_amplifier_factory(
                sampling_mode="bogus",
                strategy=strategy,
                sample_rate=0.05,
                n_steps=100,
                num_bins=10,
                dataset_size=1000,
                truncated_batch_size=None,
            )


def _mf_band_context(bands: int, sample_rate: float, n_steps: int) -> _dpftrl.MFContext:
    """Build an ``MFContext`` for BandMF tests — strategy + amplifier."""
    strategy = _dpftrl.build_strategy("mf_band", {"bands": bands})
    amplifier_factory = _dpftrl.build_amplifier_factory(
        sampling_mode="b_min_sep",
        strategy=strategy,
        sample_rate=sample_rate,
        n_steps=n_steps,
        num_bins=0,
        dataset_size=0,
        truncated_batch_size=None,
    )
    return _dpftrl.MFContext(strategy=strategy, amplifier_factory=amplifier_factory)


def _cyclic_mf_band_context(
    bands: int, sample_rate: float, n_steps: int, dataset_size: int, world_size: int = 1
) -> _dpftrl.MFContext:
    """Build an ``MFContext`` with a cyclic_poisson-flavored amplifier, so
    ``build_sampler`` can read a real ``.sample_rate`` off it."""
    strategy = _dpftrl.build_strategy("mf_band", {"bands": bands})
    amplifier_factory = _dpftrl.build_amplifier_factory(
        sampling_mode="cyclic_poisson",
        strategy=strategy,
        sample_rate=sample_rate,
        n_steps=n_steps,
        num_bins=0,
        dataset_size=dataset_size,
        truncated_batch_size=None,
        world_size=world_size,
    )
    return _dpftrl.MFContext(strategy=strategy, amplifier_factory=amplifier_factory)


def _mf_identity_context(sample_rate: float, n_steps: int) -> _dpftrl.MFContext:
    """Build an ``MFContext`` for IdentityStrategy tests (wrong-strategy cases)."""
    strategy = _dpftrl.build_strategy("mf_identity", {})
    amplifier_factory = _dpftrl.build_amplifier_factory(
        sampling_mode="poisson",
        strategy=strategy,
        sample_rate=sample_rate,
        n_steps=n_steps,
        num_bins=0,
        dataset_size=0,
        truncated_batch_size=None,
    )
    return _dpftrl.MFContext(strategy=strategy, amplifier_factory=amplifier_factory)


class TestBuildSampler:
    def test_poisson(self):
        dataset = _ListDataset(64)
        sampler = _dpftrl.build_sampler(
            sampling_mode="poisson",
            dataset=dataset,
            sample_rate=0.1,
            n_steps=8,
            key=key(0),
            sampling_kwargs=None,
            mf=None,
            noise_multiplier=None,
            num_bins=4,
            expected_batch_size=4,
        )
        assert isinstance(sampler, PoissonSampler)

    @pytest.mark.parametrize("bands", [1, 4])
    def test_poisson_rejects_band_mf_context(self, bands):
        """A whole-dataset Poisson sampler does not realise BandMF's grouped
        participation pattern — reject it rather than silently
        mis-accounting (issue #776)."""
        dataset = _ListDataset(64)
        mf = _mf_band_context(bands=bands, sample_rate=0.1, n_steps=8)
        with pytest.raises(ValueError, match="cyclic_poisson"):
            _dpftrl.build_sampler(
                sampling_mode="poisson",
                dataset=dataset,
                sample_rate=0.1,
                n_steps=8,
                key=key(0),
                sampling_kwargs=None,
                mf=mf,
                noise_multiplier=None,
                num_bins=4,
                expected_batch_size=4,
            )

    def test_block_k_out_of_t(self):
        dataset = _ListDataset(64)
        sampler = _dpftrl.build_sampler(
            sampling_mode="k_out_of_t",
            dataset=dataset,
            sample_rate=0.1,
            n_steps=8,
            key=key(0),
            sampling_kwargs={"k": 2, "allocation": "block"},
            mf=None,
            noise_multiplier=None,
            num_bins=4,
            expected_batch_size=4,
        )
        assert isinstance(sampler, KOutOfTSampler)
        assert sampler.k == 2
        assert sampler.allocation == "block"

    def test_total_k_out_of_t(self):
        dataset = _ListDataset(64)
        sampler = _dpftrl.build_sampler(
            sampling_mode="k_out_of_t",
            dataset=dataset,
            sample_rate=0.1,
            n_steps=8,
            key=key(0),
            sampling_kwargs={"k": 3, "allocation": "total"},
            mf=None,
            noise_multiplier=None,
            num_bins=4,
            expected_batch_size=4,
        )
        assert isinstance(sampler, KOutOfTSampler)
        assert sampler.allocation == "total"

    def test_b_min_sep_reads_bands_and_sampling_prob_from_amplifier(self):
        from opaque.api.accounting.dpftrl.amplification._b_min_sep import (
            participation_p_from_per_example_rate,
        )

        dataset = _ListDataset(64)
        p0, bands, n_steps = 0.05, 4, 8
        mf = _mf_band_context(bands=bands, sample_rate=p0, n_steps=n_steps)
        sampler = _dpftrl.build_sampler(
            sampling_mode="b_min_sep",
            dataset=dataset,
            sample_rate=p0,
            n_steps=n_steps,
            key=key(0),
            sampling_kwargs=None,
            mf=mf,
            noise_multiplier=1.0,
            num_bins=4,
            expected_batch_size=4,
        )
        assert isinstance(sampler, BMinSepSampler)
        assert sampler.bands == bands
        assert sampler.sampling_prob == participation_p_from_per_example_rate(p0, bands)

    def test_b_min_sep_without_mf_raises(self):
        dataset = _ListDataset(64)
        with pytest.raises(ValueError, match="requires a built MFContext"):
            _dpftrl.build_sampler(
                sampling_mode="b_min_sep",
                dataset=dataset,
                sample_rate=0.05,
                n_steps=8,
                key=key(0),
                sampling_kwargs=None,
                mf=None,
                noise_multiplier=1.0,
                num_bins=4,
                expected_batch_size=4,
            )

    def test_balls_in_bins(self):
        dataset = _ListDataset(64)
        sampler = _dpftrl.build_sampler(
            sampling_mode="balls_in_bins",
            dataset=dataset,
            sample_rate=0.1,
            n_steps=8,
            key=key(0),
            sampling_kwargs=None,
            mf=None,
            noise_multiplier=None,
            num_bins=4,
            expected_batch_size=4,
        )
        assert isinstance(sampler, BallsInBinsSampler)
        assert sampler.num_bins == 4

    def test_cyclic_poisson(self):
        """Sampler rate conversion matches
        ``TestBuildAmplifierFactory.test_band_cyclic_poisson_group_rate``."""
        dataset = _ListDataset(64)
        mf = _cyclic_mf_band_context(
            bands=4, sample_rate=0.1, n_steps=8, dataset_size=64
        )
        sampler = _dpftrl.build_sampler(
            sampling_mode="cyclic_poisson",
            dataset=dataset,
            sample_rate=0.1,
            n_steps=8,
            key=key(0),
            sampling_kwargs=None,
            mf=mf,
            noise_multiplier=1.0,
            num_bins=4,
            expected_batch_size=4,
        )
        assert isinstance(sampler, CyclicPoissonSampler)
        assert sampler.bands == 4
        assert sampler.sample_rate == pytest.approx(0.4)  # 0.1 * 4

    def test_cyclic_poisson_matches_accountant_under_ddp(self):
        """The runtime sampler must read the same group rate the
        accountant was calibrated with — including under DDP, where each
        rank's local group truncation would desync the two if the sampler
        recomputed the rate independently (issue #776 follow-up)."""
        mf = _cyclic_mf_band_context(
            bands=3, sample_rate=100 / 1000, n_steps=8, dataset_size=1000, world_size=4
        )
        local_shard = _ListDataset(250)  # one rank's local shard, world_size=4
        sampler = _dpftrl.build_sampler(
            sampling_mode="cyclic_poisson",
            dataset=local_shard,
            sample_rate=100 / 1000,
            n_steps=8,
            key=key(0),
            sampling_kwargs=None,
            mf=mf,
            noise_multiplier=1.0,
            num_bins=4,
            expected_batch_size=100,
        )
        accountant_rate = mf.amplifier_factory(1.0).sample_rate
        assert sampler.sample_rate == pytest.approx(accountant_rate)
        assert sampler.sample_rate == pytest.approx(100 / 1000 * 250 / 83)

    def test_cyclic_poisson_without_mf_raises(self):
        dataset = _ListDataset(64)
        with pytest.raises(ValueError, match="requires a built MFContext"):
            _dpftrl.build_sampler(
                sampling_mode="cyclic_poisson",
                dataset=dataset,
                sample_rate=0.1,
                n_steps=8,
                key=key(0),
                sampling_kwargs=None,
                mf=None,
                noise_multiplier=1.0,
                num_bins=4,
                expected_batch_size=4,
            )

    def test_cyclic_poisson_without_noise_multiplier_raises(self):
        dataset = _ListDataset(64)
        mf = _cyclic_mf_band_context(
            bands=4, sample_rate=0.1, n_steps=8, dataset_size=64
        )
        with pytest.raises(ValueError, match="requires a built MFContext"):
            _dpftrl.build_sampler(
                sampling_mode="cyclic_poisson",
                dataset=dataset,
                sample_rate=0.1,
                n_steps=8,
                key=key(0),
                sampling_kwargs=None,
                mf=mf,
                noise_multiplier=None,
                num_bins=4,
                expected_batch_size=4,
            )

    def test_cyclic_poisson_rejects_non_band_strategy(self):
        dataset = _ListDataset(64)
        mf = _mf_identity_context(sample_rate=0.1, n_steps=8)
        with pytest.raises(ValueError, match="BandMF"):
            _dpftrl.build_sampler(
                sampling_mode="cyclic_poisson",
                dataset=dataset,
                sample_rate=0.1,
                n_steps=8,
                key=key(0),
                sampling_kwargs=None,
                mf=mf,
                noise_multiplier=1.0,
                num_bins=4,
                expected_batch_size=4,
            )

    def test_cyclic_poisson_rejects_sampling_kwargs(self):
        """The BandMF accountant does not support truncation; fail closed
        instead of silently dropping ``truncated_batch_size`` (issue #776)."""
        dataset = _ListDataset(64)
        mf = _mf_band_context(bands=4, sample_rate=0.1, n_steps=8)
        with pytest.raises(ValueError, match="sampling_kwargs"):
            _dpftrl.build_sampler(
                sampling_mode="cyclic_poisson",
                dataset=dataset,
                sample_rate=0.1,
                n_steps=8,
                key=key(0),
                sampling_kwargs={"truncated_batch_size": 4},
                mf=mf,
                noise_multiplier=1.0,
                num_bins=4,
                expected_batch_size=4,
            )

    def test_sequential(self):
        dataset = _ListDataset(64)
        sampler = _dpftrl.build_sampler(
            sampling_mode="sequential",
            dataset=dataset,
            sample_rate=0.1,
            n_steps=8,
            key=key(0),
            sampling_kwargs=None,
            mf=None,
            noise_multiplier=None,
            num_bins=4,
            expected_batch_size=8,
        )
        assert isinstance(sampler, SequentialBatchSampler)

    def test_unknown_sampling_mode_raises(self):
        dataset = _ListDataset(64)
        with pytest.raises(ValueError, match="Unknown sampling_mode"):
            _dpftrl.build_sampler(
                sampling_mode="nope",
                dataset=dataset,
                sample_rate=0.1,
                n_steps=8,
                key=key(0),
                sampling_kwargs=None,
                mf=None,
                noise_multiplier=None,
                num_bins=4,
                expected_batch_size=4,
            )

    def test_poisson_honors_truncated_batch_size(self):
        dataset = _ListDataset(64)
        sampler = _dpftrl.build_sampler(
            sampling_mode="poisson",
            dataset=dataset,
            sample_rate=0.5,
            n_steps=4,
            key=key(0),
            sampling_kwargs={"truncated_batch_size": 8},
            mf=None,
            noise_multiplier=None,
            num_bins=4,
            expected_batch_size=4,
        )
        assert sampler.truncated_batch_size == 8
