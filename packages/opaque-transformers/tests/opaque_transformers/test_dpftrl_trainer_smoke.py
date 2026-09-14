# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Fast end-to-end smoke tests for Trainer + DP-FTRL.

Uses a tiny embedded LM (no HF model load) so the full
mechanism-dispatch surface — strategy construction, amplifier wiring,
noise-function lifecycle, sampler dispatch, and checkpoint+resume —
can be exercised in seconds.

The slower GPT-2-based parity is covered in
``tests/validation/test_dp_ftrl_trainer.py``; this file pins the
in-package CI signal so regressions surface immediately.
"""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from opaque.api.transformers.trainer import _dpftrl
from opaque.api.transformers.trainer._trainer import Trainer
from opaque.exceptions import CheckpointError, ConfigurationError
from opaque.transformers import TrainingArguments


class _TinyLM(torch.nn.Module):
    """8-dim causal-LM toy model — fast to train end-to-end on CPU."""

    def __init__(self, vocab: int = 32, dim: int = 8) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, dim)
        self.head = torch.nn.Linear(dim, vocab)

    def forward(self, input_ids: torch.Tensor, **_: object) -> dict[str, torch.Tensor]:
        hidden = self.embed(input_ids)
        logits = self.head(hidden)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), input_ids.reshape(-1)
        )
        return {"loss": loss, "logits": logits}


class _TinyDS(Dataset):
    def __init__(self, n: int = 64, seq: int = 8, vocab: int = 32) -> None:
        torch.manual_seed(0)
        self._data = torch.randint(0, vocab, (n, seq))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self._data[i]}


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {"input_ids": torch.stack([b["input_ids"] for b in batch])}


# Mellum-shaped defaults are tuned for real training; with a 16-step
# smoke run the strategy hyperparameters must be small enough to
# satisfy per-strategy validity constraints (e.g. BandMF requires
# ``bands <= n_steps``).  These keep the test surface fast while staying
# in legitimate mechanism territory.
_MF_TEST_KWARGS: dict[str, dict[str, object]] = {
    "mf_band": {"bands": 4},
    "mf_blt": {"max_buffers": 4},
    "mf_bisr": {"bandwidth": 4},
    "mf_bsr": {"bandwidth": 4, "alpha": 1.0, "beta": 0.9},
    "mf_lambda_cgd": {"lambda_": 0.5},
    "mf_identity": {},
}


def _args(
    *,
    output_dir: str,
    mechanism: str,
    max_steps: int,
    save_steps: int | None = None,
    noise_multiplier: float | None = 1.0,
    target_epsilon: float | None = None,
    clipping_norm: float | str = 1.0,
    sampling_mode: str = "auto",
    sampling_kwargs: dict[str, object] | None = None,
) -> TrainingArguments:
    kwargs = dict(_MF_TEST_KWARGS[mechanism]) if mechanism in _MF_TEST_KWARGS else {}
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        max_steps=max_steps,
        save_steps=save_steps,
        save_strategy="steps" if save_steps else "no",
        privacy_noise_mechanism=mechanism,
        privacy_noise_mechanism_kwargs=kwargs,
        sampling_mode=sampling_mode,
        sampling_kwargs=sampling_kwargs,
        privacy_noise_multiplier=noise_multiplier,
        privacy_target_epsilon=target_epsilon,
        clipping_norm=clipping_norm,
        learning_rate=1e-3,
        optim="sgd",
        report_to=[],
        eval_strategy="no",
        logging_strategy="no",
        disable_tqdm=True,
        use_cpu=True,
        seed=0,
        dataloader_num_workers=0,
    )


_MF_MECHANISMS = pytest.mark.parametrize(
    "mechanism",
    ["mf_identity", "mf_band", "mf_blt", "mf_bisr", "mf_bsr", "mf_lambda_cgd"],
)


class TestDpFtrlTrain:
    @_MF_MECHANISMS
    def test_trains_for_each_mechanism(self, tmp_path, mechanism):
        # mf_identity is the only mechanism whose tiny smoke-test budget
        # (max_steps=4) doesn't trip a per-strategy validity guard
        # (BandMF bands<=n_steps, BallsInBins n_steps%num_bins==0).
        # For the others we use 16 steps to satisfy num_bins=16 (from
        # dataset_size=64 / batch_size=4) and BandMF bands=4 <= 16.
        max_steps = 4 if mechanism == "mf_identity" else 16
        args = _args(
            output_dir=str(tmp_path / mechanism),
            mechanism=mechanism,
            max_steps=max_steps,
        )
        torch.manual_seed(0)
        trainer = Trainer(
            model=_TinyLM(),
            args=args,
            train_dataset=_TinyDS(),
            data_collator=_collate,
        )
        out = trainer.train()
        assert out.global_step == max_steps
        # Every MF mechanism reports the same privacy metric surface.
        assert "privacy_epsilon" in out.metrics
        assert out.metrics["privacy_noise_multiplier"] == pytest.approx(1.0)

    def test_step_logs_report_only_full_horizon_epsilon(self, tmp_path):
        args = _args(
            output_dir=str(tmp_path / "full-horizon-logs"),
            mechanism="mf_identity",
            max_steps=4,
        )
        args.logging_strategy = "steps"
        args.logging_steps = 1
        trainer = Trainer(
            model=_TinyLM(),
            args=args,
            train_dataset=_TinyDS(),
            data_collator=_collate,
        )

        out = trainer.train()
        logged = [
            row["privacy_epsilon"]
            for row in trainer.state.log_history
            if "privacy_epsilon" in row
        ]

        assert logged
        assert logged == pytest.approx([out.metrics["privacy_epsilon"]] * len(logged))


class TestDpFtrlSamplerDispatch:
    def test_band_poisson_mutation_rejected_before_calibration(
        self, tmp_path, monkeypatch
    ):
        args = _args(
            output_dir=str(tmp_path),
            mechanism="mf_band",
            max_steps=16,
            noise_multiplier=None,
            target_epsilon=8.0,
        )
        trainer = Trainer(
            model=_TinyLM(),
            args=args,
            train_dataset=_TinyDS(),
            data_collator=_collate,
        )
        args.sampling_mode = "poisson"

        def unexpected_calibration(*_args, **_kwargs):
            pytest.fail("invalid sampling must be rejected before calibration")

        monkeypatch.setattr(trainer, "_calibrate_noise", unexpected_calibration)
        with pytest.raises(ConfigurationError, match="sampling_mode='poisson'"):
            trainer.train()
        assert trainer.state.global_step == 0

    def test_band_poisson_callback_mutation_rejected_before_sampling(
        self, tmp_path, monkeypatch
    ):
        from transformers import TrainerCallback

        from opaque.api.transformers.trainer import _dpftrl

        args = _args(output_dir=str(tmp_path), mechanism="mf_band", max_steps=16)
        trainer = Trainer(
            model=_TinyLM(),
            args=args,
            train_dataset=_TinyDS(),
            data_collator=_collate,
        )

        class _ChangeSampler(TrainerCallback):
            def on_train_begin(self, args_, state_, control_, **_kw):
                args_.sampling_mode = "poisson"

        def unexpected_sampler(*_args, **_kwargs):
            pytest.fail("invalid sampling must be rejected before sampler construction")

        monkeypatch.setattr(_dpftrl, "PoissonSampler", unexpected_sampler)
        trainer.add_callback(_ChangeSampler())
        with pytest.raises(ConfigurationError, match="sampling_mode='poisson'"):
            trainer.train()
        assert trainer.state.global_step == 0

    @pytest.mark.parametrize(
        ("mechanism", "expected_sampler_module_name"),
        [
            ("mf_identity", "opaque.api.dpsgd.sampling._poisson"),
            ("mf_band", "opaque.api.dpftrl.sampling._poisson"),
            ("mf_blt", "opaque.api.dpftrl.sampling._balls_in_bins"),
            ("mf_bisr", "opaque.api.dpftrl.sampling._balls_in_bins"),
            ("mf_bsr", "opaque.api.dpftrl.sampling._balls_in_bins"),
            ("mf_lambda_cgd", "opaque.api.dpftrl.sampling._balls_in_bins"),
        ],
    )
    def test_sampler_dispatched_via_auto(
        self, tmp_path, mechanism, expected_sampler_module_name
    ):
        # Argument validation already pins ``sampling_mode``; this test
        # confirms the trainer's ``get_train_dataloader`` builds the
        # right concrete sampler class.  Snapshot the sampler with an
        # ``on_step_begin`` callback (fires after the dataloader and
        # sampler are constructed).
        max_steps = 4 if mechanism == "mf_identity" else 16
        args = _args(
            output_dir=str(tmp_path / mechanism),
            mechanism=mechanism,
            max_steps=max_steps,
        )
        torch.manual_seed(0)
        trainer = Trainer(
            model=_TinyLM(),
            args=args,
            train_dataset=_TinyDS(),
            data_collator=_collate,
        )
        captured: dict[str, type] = {}

        from transformers import TrainerCallback

        class _Capture(TrainerCallback):
            def on_step_begin(self, args_, state_, control_, **_kw):
                ctx = getattr(trainer, "_ctx", None)
                if ctx is not None and ctx.current_sampler is not None:
                    captured.setdefault("cls", type(ctx.current_sampler))

        trainer.add_callback(_Capture())
        trainer.train()
        assert "cls" in captured, "sampler was not captured at on_step_begin"
        assert captured["cls"].__module__ == expected_sampler_module_name

    def test_mf_band_sampler_and_accountant_share_conditional_rate(self, tmp_path):
        """End-to-end regression for issue #776: the runtime
        ``CyclicPoissonSampler`` and the ``CyclicPoisson`` accountant must
        use the same per-band conditional rate (``bands * sample_rate``)."""
        bands = 4
        args = _args(
            output_dir=str(tmp_path / "mf_band-rate-consistency"),
            mechanism="mf_band",
            max_steps=16,
        )
        torch.manual_seed(0)
        trainer = Trainer(
            model=_TinyLM(),
            args=args,
            train_dataset=_TinyDS(),
            data_collator=_collate,
        )
        captured: dict[str, object] = {}

        from transformers import TrainerCallback

        class _Capture(TrainerCallback):
            def on_step_begin(self, args_, state_, control_, **_kw):
                ctx = getattr(trainer, "_ctx", None)
                if ctx is not None and ctx.current_sampler is not None:
                    captured.setdefault("sampler", ctx.current_sampler)
                    captured.setdefault("ctx", ctx)

        trainer.add_callback(_Capture())
        trainer.train()
        assert "sampler" in captured, "sampler was not captured at on_step_begin"

        sampler = captured["sampler"]
        ctx = captured["ctx"]
        assert sampler.bands == bands

        accountant_process = ctx.mf.amplifier_factory(ctx.noise_multiplier)
        expected_group_rate = ctx.sample_rate * bands
        assert sampler.sample_rate == pytest.approx(expected_group_rate)
        assert accountant_process.sample_rate == pytest.approx(expected_group_rate)
        assert sampler.sample_rate == pytest.approx(accountant_process.sample_rate)


class TestTrainerAllocationModes:
    @pytest.mark.parametrize(
        ("mechanism", "sampling_mode", "max_steps", "allocation"),
        [
            ("gaussian", "k_out_of_t", 18, "block"),
            ("gaussian", "k_out_of_t", 10, "total"),
            ("mf_identity", "balls_in_bins", 16, None),
        ],
    )
    @pytest.mark.slow
    def test_trains_complete_schedule(
        self, tmp_path, mechanism, sampling_mode, max_steps, allocation
    ):
        kwargs = {}
        if sampling_mode == "k_out_of_t":
            kwargs["sampling_kwargs"] = {"k": 2, "allocation": allocation}
        args = _args(
            output_dir=str(tmp_path / f"{mechanism}-{sampling_mode}"),
            mechanism=mechanism,
            sampling_mode=sampling_mode,
            max_steps=max_steps,
            **kwargs,
        )
        trainer = Trainer(
            model=_TinyLM(),
            args=args,
            train_dataset=_TinyDS(),
            data_collator=_collate,
        )
        out = trainer.train()
        assert out.global_step == max_steps
        assert "privacy_epsilon" in out.metrics


class TestDpFtrlCheckpointRoundTrip:
    @pytest.mark.parametrize(
        ("mechanism", "max_steps"),
        [
            ("mf_identity", 8),
            ("mf_band", 16),
            ("mf_blt", 16),
            ("mf_bisr", 16),
            ("mf_bsr", 16),
            ("mf_lambda_cgd", 16),
        ],
    )
    @pytest.mark.parametrize("legacy_inner_horizon", [False, True])
    def test_resume_from_midtrain_checkpoint(
        self, tmp_path, monkeypatch, mechanism, max_steps, legacy_inner_horizon
    ):
        outdir = tmp_path / mechanism
        ds = _TinyDS()

        args1 = _args(
            output_dir=str(outdir),
            mechanism=mechanism,
            max_steps=max_steps,
            save_steps=4,
        )
        torch.manual_seed(0)
        trainer1 = Trainer(
            model=_TinyLM(),
            args=args1,
            train_dataset=ds,
            data_collator=_collate,
        )
        with monkeypatch.context() as patch:
            if legacy_inner_horizon:
                patch.setattr(
                    _dpftrl, "mf_gaussian", partial(_dpftrl.mf_gaussian, n_steps=1)
                )
            out1 = trainer1.train()
        assert out1.global_step == max_steps
        expected_params = {
            name: value.detach().clone()
            for name, value in trainer1.model.named_parameters()
        }

        mid_ckpt = outdir / f"checkpoint-{max_steps // 2}"
        assert mid_ckpt.is_dir()

        outdir2 = tmp_path / f"{mechanism}-resumed"
        args2 = _args(
            output_dir=str(outdir2),
            mechanism=mechanism,
            max_steps=max_steps,
            save_steps=4,
        )
        torch.manual_seed(123)
        trainer2 = Trainer(
            model=_TinyLM(),
            args=args2,
            train_dataset=ds,
            data_collator=_collate,
        )
        out2 = trainer2.train(resume_from_checkpoint=str(mid_ckpt))
        assert out2.global_step == max_steps

        actual_params = dict(trainer2.model.named_parameters())
        assert list(actual_params) == list(expected_params)
        for name, expected in expected_params.items():
            torch.testing.assert_close(
                actual_params[name].detach(),
                expected,
                rtol=0.0,
                atol=0.0,
                msg=name,
            )

        assert out1.metrics["privacy_epsilon"] == pytest.approx(
            out2.metrics["privacy_epsilon"], rel=1e-3
        )

    @pytest.mark.parametrize("drift_target", ["arguments", "sampler_state"])
    def test_k_out_of_t_resume_rejects_parameter_drift(
        self, tmp_path, monkeypatch, drift_target
    ):
        from opaque.api.transformers.trainer import _checkpoint as ckpt

        outdir = tmp_path / "k-out-of-t"
        ds = _TinyDS()
        trainer1 = Trainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(outdir),
                mechanism="gaussian",
                max_steps=4,
                save_steps=2,
                sampling_mode="k_out_of_t",
                sampling_kwargs={"k": 2, "allocation": "block"},
            ),
            train_dataset=ds,
            data_collator=_collate,
        )
        trainer1.train()

        checkpoint = outdir / "checkpoint-2"
        if drift_target == "sampler_state":
            # Keep the accountant and training arguments unchanged: restoring a
            # different sampling law must still fail before another update.
            runtime_path = checkpoint / ckpt.DP_STATE_NAME
            runtime = ckpt.load_dp_runtime_state(str(runtime_path))
            runtime.sampler_state["k"] = 1
            torch.save(runtime, runtime_path)

        trainer2 = Trainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(tmp_path / "k-out-of-t-resumed"),
                mechanism="gaussian",
                max_steps=4,
                sampling_mode="k_out_of_t",
                sampling_kwargs={
                    "k": 1 if drift_target == "arguments" else 2,
                    "allocation": "block",
                },
            ),
            train_dataset=ds,
            data_collator=_collate,
        )

        def unexpected_step(*_args, **_kwargs):
            pytest.fail("sampling-law drift must fail before the next training step")

        monkeypatch.setattr(trainer2, "training_step", unexpected_step)
        error, match = (
            (CheckpointError, "horizon_process_state")
            if drift_target == "arguments"
            else (ConfigurationError, "k mismatch")
        )
        with pytest.raises(error, match=match):
            trainer2.train(resume_from_checkpoint=str(checkpoint))

    def test_mf_resume_rejects_same_shape_strategy_drift(self, tmp_path):
        outdir = tmp_path / "mf-strategy-drift"
        ds = _TinyDS()
        trainer1 = Trainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(outdir),
                mechanism="mf_band",
                max_steps=16,
                save_steps=4,
            ),
            train_dataset=ds,
            data_collator=_collate,
        )
        trainer1.train()

        resumed_args = _args(
            output_dir=str(tmp_path / "mf-strategy-drift-resumed"),
            mechanism="mf_band",
            max_steps=16,
        )
        resumed_args.privacy_noise_mechanism_kwargs = {
            "bands": 4,
            "momentum": 0.5,
        }
        trainer2 = Trainer(
            model=_TinyLM(),
            args=resumed_args,
            train_dataset=ds,
            data_collator=_collate,
        )

        with pytest.raises(CheckpointError, match="horizon_process_state"):
            trainer2.train(resume_from_checkpoint=str(outdir / "checkpoint-4"))

    def test_calibrated_horizon_resume_restores_noise_multiplier(self, tmp_path):
        outdir = tmp_path / "calibrated"
        ds = _TinyDS()
        trainer1 = Trainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(outdir),
                mechanism="mf_identity",
                max_steps=4,
                save_steps=2,
                noise_multiplier=None,
                target_epsilon=5.0,
            ),
            train_dataset=ds,
            data_collator=_collate,
        )
        original = trainer1.train()

        trainer2 = Trainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(tmp_path / "calibrated-resumed"),
                mechanism="mf_identity",
                max_steps=4,
                noise_multiplier=None,
                target_epsilon=5.0,
            ),
            train_dataset=ds,
            data_collator=_collate,
        )
        resumed = trainer2.train(resume_from_checkpoint=str(outdir / "checkpoint-2"))

        assert resumed.metrics["privacy_noise_multiplier"] == pytest.approx(
            original.metrics["privacy_noise_multiplier"]
        )
        assert resumed.metrics["privacy_epsilon"] == pytest.approx(
            original.metrics["privacy_epsilon"]
        )

    def test_calibrated_horizon_resume_rejects_target_drift(self, tmp_path):
        outdir = tmp_path / "calibrated-target-drift"
        ds = _TinyDS()
        trainer1 = Trainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(outdir),
                mechanism="mf_identity",
                max_steps=4,
                save_steps=2,
                noise_multiplier=None,
                target_epsilon=5.0,
            ),
            train_dataset=ds,
            data_collator=_collate,
        )
        trainer1.train()

        trainer2 = Trainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(tmp_path / "calibrated-target-drift-resumed"),
                mechanism="mf_identity",
                max_steps=4,
                noise_multiplier=None,
                target_epsilon=20.0,
            ),
            train_dataset=ds,
            data_collator=_collate,
        )

        with pytest.raises(CheckpointError, match="privacy_target_epsilon drift"):
            trainer2.train(resume_from_checkpoint=str(outdir / "checkpoint-2"))

    def test_fixed_horizon_resume_rejects_calibrated_mode(self, tmp_path):
        outdir = tmp_path / "fixed-to-calibrated"
        ds = _TinyDS()
        trainer1 = Trainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(outdir),
                mechanism="mf_identity",
                max_steps=4,
                save_steps=2,
                noise_multiplier=0.0,
            ),
            train_dataset=ds,
            data_collator=_collate,
        )
        trainer1.train()

        trainer2 = Trainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(tmp_path / "fixed-to-calibrated-resumed"),
                mechanism="mf_identity",
                max_steps=4,
                noise_multiplier=None,
                target_epsilon=1.0,
            ),
            train_dataset=ds,
            data_collator=_collate,
        )

        with pytest.raises(CheckpointError, match="calibration mode drift"):
            trainer2.train(resume_from_checkpoint=str(outdir / "checkpoint-2"))

    def test_fixed_horizon_resume_rejects_noise_multiplier_drift(self, tmp_path):
        outdir = tmp_path / "fixed-noise"
        ds = _TinyDS()
        trainer1 = Trainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(outdir),
                mechanism="mf_identity",
                max_steps=4,
                save_steps=2,
                noise_multiplier=1.0,
            ),
            train_dataset=ds,
            data_collator=_collate,
        )
        trainer1.train()

        trainer2 = Trainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(tmp_path / "fixed-noise-resumed"),
                mechanism="mf_identity",
                max_steps=4,
                noise_multiplier=2.0,
            ),
            train_dataset=ds,
            data_collator=_collate,
        )

        with pytest.raises(CheckpointError, match="privacy_noise_multiplier drift"):
            trainer2.train(resume_from_checkpoint=str(outdir / "checkpoint-2"))


class TestDpFtrlLrScheduleIntegration:
    """The optimizer LR schedule auto-flows into BandMF / BLT strategies.

    Pre-refactor (opaque.scheduling as a closure DSL), the trainer
    silently dropped the schedule because the strategy codec rejected
    callable fields.  Post-refactor, schedules are frozen recipe
    dataclasses, round-trip through the strategy codec via tagged
    sub-dicts, and the trainer's ``_setup_training`` auto-injects the
    live LR schedule into BandMF / BLT.
    """

    def test_band_mf_strategy_receives_schedule(self, tmp_path):
        # The trainer's live LR schedule should appear on
        # ``ctx.mf.strategy.lr_schedule``.  Snapshot via an
        # ``on_step_begin`` callback (fires after ``_setup_training``
        # populates ``_ctx``).
        from transformers import TrainerCallback

        from opaque.scheduling.types import ConstantSchedule, CosineSchedule

        captured: dict[str, type] = {}
        sentinel_trainer: dict[str, object] = {}

        class _Snap(TrainerCallback):
            def on_step_begin(self, args_, state_, ctrl_, **_kw):
                trainer = sentinel_trainer["trainer"]
                mf = getattr(trainer._ctx, "mf", None)
                if mf is not None and "ls_cls" not in captured:
                    captured["ls_cls"] = type(mf.strategy.lr_schedule)

        for sched_type, expected_cls in (
            ("cosine", CosineSchedule),
            ("constant", ConstantSchedule),
        ):
            args = TrainingArguments(
                output_dir=str(tmp_path / sched_type),
                per_device_train_batch_size=4,
                max_steps=16,
                privacy_noise_mechanism="mf_band",
                privacy_noise_mechanism_kwargs={"bands": 4},
                privacy_noise_multiplier=1.0,
                clipping_norm=1.0,
                learning_rate=1e-3,
                optim="sgd",
                lr_scheduler=sched_type,
                report_to=[],
                save_strategy="no",
                eval_strategy="no",
                logging_strategy="no",
                disable_tqdm=True,
                use_cpu=True,
                seed=0,
            )
            torch.manual_seed(0)
            trainer = Trainer(
                model=_TinyLM(),
                args=args,
                train_dataset=_TinyDS(),
                data_collator=_collate,
            )
            sentinel_trainer["trainer"] = trainer
            captured.clear()
            trainer.add_callback(_Snap())
            trainer.train()
            assert captured.get("ls_cls") is expected_cls, (
                f"expected {expected_cls.__name__} for "
                f"lr_scheduler={sched_type!r}, got "
                f"{captured.get('ls_cls')}"
            )

    def test_resume_preserves_schedule_in_accountant(self, tmp_path):
        # Save mid-train, resume to completion, verify ε matches a
        # from-scratch run — the saved accountant.json must round-trip
        # the cosine schedule baked into the BandMfStrategy.
        outdir = tmp_path / "bandmf_cosine"
        kwargs = {
            "per_device_train_batch_size": 4,
            "max_steps": 16,
            "save_steps": 4,
            "privacy_noise_mechanism": "mf_band",
            "privacy_noise_mechanism_kwargs": {"bands": 4},
            "privacy_noise_multiplier": 1.0,
            "clipping_norm": 1.0,
            "learning_rate": 1e-3,
            "optim": "sgd",
            "lr_scheduler": "cosine",
            "report_to": [],
            "save_strategy": "steps",
            "eval_strategy": "no",
            "logging_strategy": "no",
            "disable_tqdm": True,
            "use_cpu": True,
            "seed": 0,
        }

        # From-scratch run.
        args = TrainingArguments(output_dir=str(outdir), **kwargs)
        torch.manual_seed(0)
        t1 = Trainer(
            model=_TinyLM(),
            args=args,
            train_dataset=_TinyDS(),
            data_collator=_collate,
        )
        out1 = t1.train()

        # Find a mid-train checkpoint.
        ckpts = sorted(
            [
                d.name
                for d in Path(outdir).iterdir()
                if d.name.startswith("checkpoint-")
            ],
            key=lambda d: int(d.split("-")[1]),
        )
        assert len(ckpts) >= 2
        mid = str(outdir / ckpts[-2])

        # Resume.
        outdir2 = tmp_path / "bandmf_cosine_resumed"
        args2 = TrainingArguments(output_dir=str(outdir2), **kwargs)
        torch.manual_seed(0)
        t2 = Trainer(
            model=_TinyLM(),
            args=args2,
            train_dataset=_TinyDS(),
            data_collator=_collate,
        )
        out2 = t2.train(resume_from_checkpoint=mid)

        # Identical ε is the strongest signal that the accountant's
        # internal schedule survived the disk round-trip.
        assert out1.metrics["privacy_epsilon"] == pytest.approx(
            out2.metrics["privacy_epsilon"], rel=1e-9
        )


class TestGaussianPathUnchanged:
    """Sanity: DP-SGD path still works (no regression from MF wiring)."""

    def test_gaussian_train(self, tmp_path):
        args = _args(
            output_dir=str(tmp_path / "gaussian"),
            mechanism="gaussian",
            max_steps=4,
        )
        torch.manual_seed(0)
        trainer = Trainer(
            model=_TinyLM(),
            args=args,
            train_dataset=_TinyDS(),
            data_collator=_collate,
        )
        out = trainer.train()
        assert out.global_step == 4
        assert "privacy_epsilon" in out.metrics


class TestNonPrivateZeroNoise:
    """``privacy_noise_multiplier=0`` → non-DP baseline.

    The chosen mechanism and sampler are kept intact; the accountant
    composes a non-private step so ε=∞ is reported, and zero noise is
    added (clipping still applies unless disabled).
    """

    @pytest.mark.parametrize(
        ("mechanism", "max_steps"),
        [("gaussian", 4), ("mf_identity", 4), ("mf_band", 16)],
    )
    def test_zero_noise_reports_inf_epsilon(self, tmp_path, mechanism, max_steps):
        args = _args(
            output_dir=str(tmp_path / mechanism),
            mechanism=mechanism,
            max_steps=max_steps,
            noise_multiplier=0.0,
        )
        torch.manual_seed(0)
        trainer = Trainer(
            model=_TinyLM(),
            args=args,
            train_dataset=_TinyDS(),
            data_collator=_collate,
        )
        out = trainer.train()

        # ε=∞ is the legal non-private output; the multiplier round-trips
        # as a fixed 0.0 with the "fixed" provenance tag.
        assert out.global_step > 0
        assert math.isinf(out.metrics["privacy_epsilon"])
        assert out.metrics["privacy_noise_multiplier"] == 0.0
        assert trainer.state.privacy_resolved_noise_multiplier == 0.0
        assert trainer.state.privacy_calibration_source == "fixed"
        assert math.isfinite(out.metrics["train_loss"])

    def test_zero_noise_is_deterministic(self, tmp_path):
        # σ=0 ⇒ no randomness from the noise stream: two runs at the same
        # seed must produce identical weights.
        def _run(tag):
            args = _args(
                output_dir=str(tmp_path / tag),
                mechanism="gaussian",
                max_steps=4,
                noise_multiplier=0.0,
            )
            torch.manual_seed(0)
            trainer = Trainer(
                model=_TinyLM(),
                args=args,
                train_dataset=_TinyDS(),
                data_collator=_collate,
            )
            trainer.train()
            return {n: p.detach().clone() for n, p in trainer.model.named_parameters()}

        a, b = _run("a"), _run("b")
        for n in a:
            assert torch.equal(a[n], b[n]), f"param {n} differs across σ=0 runs"

    def test_disabled_clipping_zero_noise_no_nan(self, tmp_path):
        # clipping_norm=math.inf with σ=0 is true non-private SGD;
        # the 0*inf NaN hazard in the noise std must be guarded.
        args = _args(
            output_dir=str(tmp_path / "noclip"),
            mechanism="gaussian",
            max_steps=4,
            noise_multiplier=0.0,
            clipping_norm=math.inf,
        )
        assert math.isinf(args.clipping_norm)
        torch.manual_seed(0)
        trainer = Trainer(
            model=_TinyLM(),
            args=args,
            train_dataset=_TinyDS(),
            data_collator=_collate,
        )
        out = trainer.train()
        assert math.isfinite(out.metrics["train_loss"])
        for n, p in trainer.model.named_parameters():
            assert not torch.isnan(p).any(), f"NaN in param {n}"
