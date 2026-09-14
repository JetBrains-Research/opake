"""Unit tests for ``trainer/_checkpoint.py`` helpers."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Mapping
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import opaque.api.transformers.trainer._checkpoint as ckpt
from opaque.api.engine.clipping.types import FixedClipState
from opaque.dpftrl.noise import (
    band_mf_strategy,
    bisr_strategy,
    blt_strategy,
    bsr_strategy,
    identity_strategy,
    lambda_cgd_strategy,
    mf_gaussian_noise,
)
from opaque.dpsgd.noise import gaussian_noise
from opaque.exceptions import CheckpointError
from opaque.random import key
from opaque.random.types import RngKey
from opaque.serialization import from_state_dict as opaque_from_state_dict
from opaque.types import (
    PerGroup,
    SecondMomentClippingOutput,
    SecondMomentNoiseOutput,
    clipped,
)


class TestParseCheckpointStep:
    def test_valid(self):
        assert ckpt.parse_checkpoint_step("checkpoint-500") == 500
        assert ckpt.parse_checkpoint_step("/tmp/output/checkpoint-12") == 12

    def test_invalid(self):
        assert ckpt.parse_checkpoint_step("model") is None
        assert ckpt.parse_checkpoint_step("checkpoint-abc") is None
        assert ckpt.parse_checkpoint_step("checkpoint-500-extra") is None


class TestListAndLastCheckpoint:
    def test_empty_dir_returns_empty_and_none(self, tmp_path):
        assert ckpt.list_checkpoints(str(tmp_path)) == []
        assert ckpt.get_last_checkpoint(str(tmp_path)) is None

    def test_nonexistent_dir(self, tmp_path):
        missing = str(tmp_path / "nope")
        assert ckpt.list_checkpoints(missing) == []
        assert ckpt.get_last_checkpoint(missing) is None

    def test_sorted_ascending(self, tmp_path):
        for step in (100, 10, 50):
            (tmp_path / f"checkpoint-{step}").mkdir()
        (tmp_path / "model").mkdir()  # ignored
        listing = ckpt.list_checkpoints(str(tmp_path))
        assert [Path(p).name for p in listing] == [
            "checkpoint-10",
            "checkpoint-50",
            "checkpoint-100",
        ]
        assert Path(ckpt.get_last_checkpoint(str(tmp_path))).name == "checkpoint-100"

    def test_files_with_checkpoint_prefix_ignored(self, tmp_path):
        (tmp_path / "checkpoint-99").touch()  # file, not dir
        assert ckpt.list_checkpoints(str(tmp_path)) == []


class TestRotateCheckpoints:
    def _make(self, path, step):
        d = path / f"checkpoint-{step}"
        d.mkdir()
        (d / "marker").touch()

    def test_no_op_when_limit_none(self, tmp_path):
        for s in (1, 2, 3):
            self._make(tmp_path, s)
        ckpt.rotate_checkpoints(str(tmp_path), save_total_limit=None)
        assert len(ckpt.list_checkpoints(str(tmp_path))) == 3

    def test_no_op_when_under_limit(self, tmp_path):
        for s in (1, 2):
            self._make(tmp_path, s)
        ckpt.rotate_checkpoints(str(tmp_path), save_total_limit=5)
        assert len(ckpt.list_checkpoints(str(tmp_path))) == 2

    def test_keeps_n_most_recent(self, tmp_path):
        for s in (1, 2, 3, 4, 5):
            self._make(tmp_path, s)
        ckpt.rotate_checkpoints(str(tmp_path), save_total_limit=2)
        names = [Path(p).name for p in ckpt.list_checkpoints(str(tmp_path))]
        assert names == ["checkpoint-4", "checkpoint-5"]

    def test_protects_best_when_outside_window(self, tmp_path):
        for s in (1, 2, 3, 4, 5):
            self._make(tmp_path, s)
        best = str(tmp_path / "checkpoint-1")
        ckpt.rotate_checkpoints(
            str(tmp_path), save_total_limit=2, best_model_checkpoint=best
        )
        names = sorted(Path(p).name for p in ckpt.list_checkpoints(str(tmp_path)))
        # Both most-recent (5) and best (1) survive; total kept = max(2, 2) = 2
        assert "checkpoint-1" in names
        assert "checkpoint-5" in names

    def test_save_total_limit_one_with_best(self, tmp_path):
        # HF parity: limit=1 with a different best keeps both.
        for s in (1, 2, 3):
            self._make(tmp_path, s)
        best = str(tmp_path / "checkpoint-2")
        ckpt.rotate_checkpoints(
            str(tmp_path), save_total_limit=1, best_model_checkpoint=best
        )
        names = sorted(Path(p).name for p in ckpt.list_checkpoints(str(tmp_path)))
        assert names == ["checkpoint-2", "checkpoint-3"]


class TestRngSnapshot:
    def test_roundtrip_python(self):
        import random as r

        r.seed(123)
        snap = ckpt.snapshot_rng_state()
        a = r.random()

        r.seed(999)  # disturb
        ckpt.restore_rng_state(snap)
        b = r.random()
        assert a == b

    def test_roundtrip_numpy(self):
        import numpy as np

        np.random.seed(7)
        snap = ckpt.snapshot_rng_state()
        a = np.random.rand(5)

        np.random.seed(0)
        ckpt.restore_rng_state(snap)
        b = np.random.rand(5)
        assert (a == b).all()

    def test_roundtrip_torch_cpu(self):
        torch.manual_seed(42)
        snap = ckpt.snapshot_rng_state()
        a = torch.rand(5)

        torch.manual_seed(0)
        ckpt.restore_rng_state(snap)
        b = torch.rand(5)
        assert torch.equal(a, b)


class TestDpRuntimeBundle:
    def test_roundtrip(self, tmp_path):
        clip = FixedClipState()
        _, noise = gaussian_noise(noise_multiplier=1.0, key=key(11))

        path = str(tmp_path / "dp_runtime.pt")
        ckpt.save_dp_runtime_state(
            path,
            clip_state=clip,
            noise_state=noise,
            sampler_state={
                "key_seed": 5,
                "key_impl": "opaque_threefry_like",
                "consumed": 2,
                "num_samples": 100,
                "sample_rate": 0.1,
                "n_steps": 10,
                "truncated_batch_size": None,
            },
            sample_rate=0.1,
            target_delta=1e-5,
            noise_multiplier=1.1,
            expected_steps_per_epoch=10,
            expected_batch_size=32,
            total_steps=30,
            is_horizon_process=True,
            calibration_source="calibrated",
            target_epsilon=5.0,
            horizon_process_state={"type": "ExampleHorizon", "n_steps": 30},
        )
        loaded = ckpt.load_dp_runtime_state(path)

        assert isinstance(loaded, ckpt.RuntimeCheckpoint)
        assert opaque_from_state_dict(clip, loaded.clip_state) == clip
        assert opaque_from_state_dict(noise, loaded.noise_state) == noise
        assert loaded.version == ckpt.DP_STATE_BUNDLE_VERSION
        assert loaded.sampler_state["consumed"] == 2
        assert loaded.sample_rate == pytest.approx(0.1)
        assert loaded.target_delta == pytest.approx(1e-5)
        assert loaded.noise_multiplier == pytest.approx(1.1)
        assert loaded.expected_steps_per_epoch == 10
        assert loaded.total_steps == 30
        assert loaded.is_horizon_process is True
        assert loaded.calibration_source == "calibrated"
        assert loaded.target_epsilon == pytest.approx(5.0)
        assert loaded.horizon_process_state == {
            "type": "ExampleHorizon",
            "n_steps": 30,
        }

    def test_unsupported_clip_state_type_raises(self, tmp_path):
        path = str(tmp_path / "dp.pt")
        _, noise = gaussian_noise(noise_multiplier=1.0, key=key(0))
        with pytest.raises(CheckpointError, match="clip_state must be a ClipState"):
            ckpt.save_dp_runtime_state(
                path,
                clip_state="not_a_clip_state",
                noise_state=noise,
                sampler_state=None,
                sample_rate=0.1,
                target_delta=1e-5,
                noise_multiplier=1.0,
                expected_steps_per_epoch=1,
                expected_batch_size=32,
                total_steps=1,
            )

    def test_unsupported_noise_state_type_raises(self, tmp_path):
        path = str(tmp_path / "dp.pt")
        clip = FixedClipState()
        with pytest.raises(CheckpointError, match="noise_state must be a NoiseState"):
            ckpt.save_dp_runtime_state(
                path,
                clip_state=clip,
                noise_state="not_noise",
                sampler_state=None,
                sample_rate=0.1,
                target_delta=1e-5,
                noise_multiplier=1.0,
                expected_steps_per_epoch=1,
                expected_batch_size=32,
                total_steps=1,
            )

    def test_rejects_pre_removal_bundle_version(self, tmp_path):
        path = str(tmp_path / "dp.pt")
        fake = ckpt.RuntimeCheckpoint(
            version=7,
            clip_state={},
            noise_state={},
            sampler_state=None,
            sample_rate=0.1,
            target_delta=1e-5,
            noise_multiplier=1.0,
            expected_steps_per_epoch=1,
            expected_batch_size=32,
            total_steps=1,
        )
        torch.save(fake, path)
        with pytest.raises(CheckpointError, match="unsupported dp_state"):
            ckpt.load_dp_runtime_state(path)


_NOISE_STEPS = 6
_NOISE_CHECKPOINT_STEP = 3


def _noise_tree() -> dict[str, Any]:
    return {
        "fallback": torch.zeros(3, dtype=torch.float64),
        "nested": {"head": torch.zeros((2, 2), dtype=torch.float64)},
    }


def _per_group_norm(*, fallback: float, head: float) -> PerGroup:
    return PerGroup(
        groups={
            ("fallback",): "fallback",
            ("nested", "head"): "head",
        },
        values={"fallback": fallback, "head": head},
    )


def _mf_noise_factory(strategy_factory: Callable[[], Any]) -> Callable[[RngKey], Any]:
    def make_noise(rng_key: RngKey):
        return mf_gaussian_noise(
            _noise_tree(),
            strategy_factory(),
            n_steps=_NOISE_STEPS,
            min_sep=_NOISE_STEPS,
            max_participations=1,
            noise_multiplier=1.0,
            key=rng_key,
            compute_dtype=torch.float64,
        )

    return make_noise


def _paired_noise_factory(rng_key: RngKey):
    return mf_gaussian_noise(
        _noise_tree(),
        band_mf_strategy(bands=3, momentum=0.9),
        n_steps=_NOISE_STEPS,
        min_sep=_NOISE_STEPS,
        max_participations=1,
        noise_multiplier=1.0,
        key=rng_key,
        compute_dtype=torch.float64,
        second_moment_strategy=lambda_cgd_strategy(lambda_=0.5),
    )


@dataclasses.dataclass(frozen=True)
class _NoiseContinuityCase:
    make_noise: Callable[[RngKey], Any]
    make_input: Callable[[], Any]
    make_poison_input: Callable[[], Any]
    mechanism_kind: str
    buffered: bool = False


def _scalar_input(max_norm: float):
    return clipped(_noise_tree(), max_norm=max_norm)


def _per_group_input(*, fallback: float, head: float):
    return clipped(
        _noise_tree(),
        max_norm=_per_group_norm(fallback=fallback, head=head),
    )


def _paired_input(*, max_norm: float, squared_max_norm: float):
    return SecondMomentClippingOutput(
        grads=_scalar_input(max_norm),
        squared_grads=_scalar_input(squared_max_norm),
    )


_NOISE_CASES = [
    pytest.param(
        _NoiseContinuityCase(
            make_noise=lambda rng_key: gaussian_noise(
                noise_multiplier=1.0,
                key=rng_key,
                compute_dtype=torch.float64,
            ),
            make_input=lambda: _scalar_input(1.0),
            make_poison_input=lambda: _scalar_input(7.0),
            mechanism_kind="gaussian",
        ),
        id="gaussian",
    ),
    pytest.param(
        _NoiseContinuityCase(
            make_noise=_mf_noise_factory(identity_strategy),
            make_input=lambda: _scalar_input(1.0),
            make_poison_input=lambda: _scalar_input(7.0),
            mechanism_kind="mf_identity",
        ),
        id="mf-identity",
    ),
    pytest.param(
        _NoiseContinuityCase(
            make_noise=_mf_noise_factory(
                lambda: band_mf_strategy(bands=3, momentum=0.9)
            ),
            make_input=lambda: _per_group_input(fallback=1.0, head=2.0),
            make_poison_input=lambda: _per_group_input(
                fallback=7.0,
                head=8.0,
            ),
            mechanism_kind="mf_band",
            buffered=True,
        ),
        id="mf-band-per-group",
    ),
    pytest.param(
        _NoiseContinuityCase(
            make_noise=_mf_noise_factory(
                lambda: blt_strategy(max_buffers=2, momentum=0.9)
            ),
            make_input=lambda: _scalar_input(1.0),
            make_poison_input=lambda: _scalar_input(7.0),
            mechanism_kind="mf_blt",
            buffered=True,
        ),
        id="mf-blt",
    ),
    pytest.param(
        _NoiseContinuityCase(
            make_noise=_mf_noise_factory(
                lambda: bisr_strategy(bandwidth=3, momentum=0.9)
            ),
            make_input=lambda: _scalar_input(1.0),
            make_poison_input=lambda: _scalar_input(7.0),
            mechanism_kind="mf_bisr",
            buffered=True,
        ),
        id="mf-bisr",
    ),
    pytest.param(
        _NoiseContinuityCase(
            make_noise=_mf_noise_factory(
                lambda: bsr_strategy(bandwidth=3, alpha=1.0, beta=0.9)
            ),
            make_input=lambda: _scalar_input(1.0),
            make_poison_input=lambda: _scalar_input(7.0),
            mechanism_kind="mf_bsr",
            buffered=True,
        ),
        id="mf-bsr",
    ),
    pytest.param(
        _NoiseContinuityCase(
            make_noise=_mf_noise_factory(lambda: lambda_cgd_strategy(lambda_=0.5)),
            make_input=lambda: _scalar_input(1.0),
            make_poison_input=lambda: _scalar_input(7.0),
            mechanism_kind="mf_lambda_cgd",
        ),
        id="mf-lambda-cgd",
    ),
    pytest.param(
        _NoiseContinuityCase(
            make_noise=_paired_noise_factory,
            make_input=lambda: _paired_input(max_norm=1.0, squared_max_norm=1.0),
            make_poison_input=lambda: _paired_input(
                max_norm=7.0,
                squared_max_norm=49.0,
            ),
            mechanism_kind="mf_band",
            buffered=True,
        ),
        id="mf-paired-band-lambda-cgd",
    ),
]


def _assert_nested_equal(actual: Any, expected: Any, path: str = "root") -> None:
    assert type(actual) is type(expected), path
    if isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype, path
        assert actual.device == expected.device, path
        assert torch.equal(actual, expected), path
        return
    if dataclasses.is_dataclass(expected):
        for field in dataclasses.fields(expected):
            _assert_nested_equal(
                getattr(actual, field.name),
                getattr(expected, field.name),
                f"{path}.{field.name}",
            )
        return
    if isinstance(expected, Mapping):
        assert set(actual) == set(expected), path
        for key_ in expected:
            _assert_nested_equal(actual[key_], expected[key_], f"{path}[{key_!r}]")
        return
    if isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), path
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _assert_nested_equal(actual_item, expected_item, f"{path}[{index}]")
        return
    assert actual == expected, path


def _nested_tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            yield from _nested_tensors(getattr(value, field.name))
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _nested_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _nested_tensors(child)


def _noise_streams(output: Any):
    if isinstance(output, SecondMomentNoiseOutput):
        return output.noisy_grads, output.noisy_squared_grads
    return (output,)


class TestNoiseStreamContinuity:
    """DP noise state and outputs continue exactly across a runtime checkpoint."""

    @pytest.mark.parametrize("case", _NOISE_CASES)
    def test_continues_after_resume(self, tmp_path, case):
        uninterrupted_fn, uninterrupted_state = case.make_noise(key(42))
        uninterrupted_outputs = []
        for _ in range(_NOISE_STEPS):
            output, uninterrupted_state = uninterrupted_fn(
                case.make_input(),
                uninterrupted_state,
            )
            uninterrupted_outputs.append(output)

        interrupted_fn, interrupted_state = case.make_noise(key(42))
        interrupted_outputs = []
        for _ in range(_NOISE_CHECKPOINT_STEP):
            output, interrupted_state = interrupted_fn(
                case.make_input(),
                interrupted_state,
            )
            interrupted_outputs.append(output)

        for actual, expected in zip(
            interrupted_outputs,
            uninterrupted_outputs[:_NOISE_CHECKPOINT_STEP],
            strict=True,
        ):
            _assert_nested_equal(actual, expected)
        if case.buffered:
            assert any(
                bool(torch.count_nonzero(value))
                for value in _nested_tensors(interrupted_state)
            )

        path = str(tmp_path / "dp_runtime.pt")
        is_mf = case.mechanism_kind.startswith("mf_")
        ckpt.save_dp_runtime_state(
            path,
            clip_state=FixedClipState(),
            noise_state=interrupted_state,
            sampler_state=None,
            sample_rate=0.1,
            target_delta=1e-5,
            noise_multiplier=1.0,
            expected_steps_per_epoch=1,
            expected_batch_size=1,
            total_steps=_NOISE_STEPS,
            mechanism_kind=case.mechanism_kind,
            mf_n_steps=_NOISE_STEPS if is_mf else None,
            mf_min_sep=_NOISE_STEPS if is_mf else None,
            mf_max_participations=1 if is_mf else None,
        )
        checkpoint = ckpt.load_dp_runtime_state(path)

        for restore_mode in ("fresh", "poisoned"):
            poisoned = restore_mode == "poisoned"
            restore_key = (
                RngKey(seed=999, impl="poison-template") if poisoned else key(42)
            )
            resumed_fn, state_template = case.make_noise(restore_key)
            if poisoned:
                _, state_template = resumed_fn(
                    case.make_poison_input(),
                    state_template,
                )
            resumed_state = opaque_from_state_dict(
                state_template,
                checkpoint.noise_state,
            )
            _assert_nested_equal(
                resumed_state,
                interrupted_state,
                path=f"{restore_mode}.restored_state",
            )

            resumed_outputs = list(interrupted_outputs)
            for _ in range(_NOISE_CHECKPOINT_STEP, _NOISE_STEPS):
                output, resumed_state = resumed_fn(
                    case.make_input(),
                    resumed_state,
                )
                resumed_outputs.append(output)

            for actual, expected in zip(
                resumed_outputs,
                uninterrupted_outputs,
                strict=True,
            ):
                _assert_nested_equal(actual, expected)
            _assert_nested_equal(resumed_state, uninterrupted_state)

            first_output = uninterrupted_outputs[0]
            resumed_output = resumed_outputs[_NOISE_CHECKPOINT_STEP]
            for resumed_stream, first_stream in zip(
                _noise_streams(resumed_output),
                _noise_streams(first_output),
                strict=True,
            ):
                for resumed_leaf, first_leaf in zip(
                    _nested_tensors(resumed_stream.pytree),
                    _nested_tensors(first_stream.pytree),
                    strict=True,
                ):
                    assert not torch.equal(resumed_leaf, first_leaf)


class TestRuntimeCheckpointDriftMetadata:
    """Per-field ``drift`` disposition on ``RuntimeCheckpoint`` is intact.

    The trainer's ``_warn_on_arg_drift`` reads these metadata keys to
    dispatch warn / raise / silent actions; if a field loses its
    ``drift`` tag, drift handling silently regresses to the default
    (``dp_relevant``), which is the safer side but hides the design
    intent.
    """

    def _field(self, name):
        for f in fields(ckpt.RuntimeCheckpoint):
            if f.name == name:
                return f
        raise AssertionError(f"no field {name!r} on RuntimeCheckpoint")

    @pytest.mark.parametrize(
        ("field_name", "expected"),
        [
            ("sample_rate", "dp_relevant"),
            ("target_delta", "dp_relevant"),
            ("noise_multiplier", "dp_relevant"),
            ("expected_steps_per_epoch", "dp_relevant"),
            ("expected_batch_size", "dp_relevant"),
            ("mechanism_kind", "dp_relevant"),
            ("is_horizon_process", "dp_relevant"),
            ("horizon_process_state", "dp_relevant"),
            ("mf_n_steps", "dp_relevant"),
            ("mf_min_sep", "dp_relevant"),
            ("mf_max_participations", "dp_relevant"),
            ("lr_scheduler", "shape"),
            ("learning_rate", "shape"),
            ("warmup_steps", "shape"),
            ("lr_scheduler_kwargs", "shape"),
        ],
    )
    def test_string_dispositions(self, field_name, expected):
        meta = self._field(field_name).metadata
        assert meta.get("compare_on_resume") is True, field_name
        assert meta.get("drift") == expected, field_name

    def test_total_steps_per_mechanism_override(self):
        """``total_steps`` is silent for DP-SGD (intentional extend), forbidden
        for DP-FTRL (MF strategy is shape-locked)."""
        meta = self._field("total_steps").metadata
        assert meta.get("compare_on_resume") is True
        drift = meta.get("drift")
        assert isinstance(drift, dict)
        assert drift.get("gaussian") == "intentional_extend"
        assert drift.get("default") == "dp_relevant"


class TestDriftDispositionResolution:
    """``_resolve_drift_disposition`` picks the right rule per mechanism."""

    def test_string_disposition_passthrough(self):
        from opaque.api.transformers.trainer._trainer import (
            _resolve_drift_disposition,
        )

        meta = {"drift": "shape"}
        assert _resolve_drift_disposition(meta, "gaussian") == "shape"
        assert _resolve_drift_disposition(meta, "mf_band") == "shape"

    def test_dict_disposition_per_mechanism(self):
        from opaque.api.transformers.trainer._trainer import (
            _resolve_drift_disposition,
        )

        meta = {
            "drift": {
                "gaussian": "intentional_extend",
                "default": "dp_relevant",
            }
        }
        assert _resolve_drift_disposition(meta, "gaussian") == "intentional_extend"
        assert _resolve_drift_disposition(meta, "mf_band") == "dp_relevant"
        assert _resolve_drift_disposition(meta, "mf_blt") == "dp_relevant"

    def test_default_disposition_when_missing(self):
        from opaque.api.transformers.trainer._trainer import (
            _resolve_drift_disposition,
        )

        # Missing ``drift`` key falls back to ``dp_relevant`` (safest).
        assert _resolve_drift_disposition({}, "gaussian") == "dp_relevant"


class TestResumeDrift:
    @pytest.fixture
    def drift_case(self, monkeypatch):
        from opaque.api.transformers.trainer._trainer import Trainer

        runtime = ckpt.RuntimeCheckpoint(
            version=ckpt.DP_STATE_BUNDLE_VERSION,
            clip_state={},
            noise_state={},
            sampler_state=None,
            sample_rate=0.125,
            target_delta=1e-5,
            noise_multiplier=1.0,
            expected_steps_per_epoch=8,
            expected_batch_size=8,
            total_steps=16,
        )
        current = {"expected_steps_per_epoch": 16}
        trainer = Trainer.__new__(Trainer)
        trainer.args = SimpleNamespace()
        trainer._ctx = SimpleNamespace(is_horizon_process=False)
        monkeypatch.setattr(trainer, "_current_values_for_drift", lambda *_: current)
        return trainer, runtime, current

    @pytest.mark.parametrize(
        ("mechanism", "saved_horizon", "current_horizon"),
        [
            ("gaussian", True, True),
            ("gaussian", True, False),
            ("gaussian", False, True),
            ("mf_blt", True, True),
        ],
    )
    def test_epoch_shape_drift_is_fatal_for_horizons(
        self, drift_case, mechanism, saved_horizon, current_horizon
    ):
        trainer, runtime, _ = drift_case
        runtime.mechanism_kind = mechanism
        runtime.is_horizon_process = saved_horizon
        trainer._ctx.is_horizon_process = current_horizon

        with pytest.raises(CheckpointError, match="expected_steps_per_epoch"):
            trainer._warn_on_arg_drift(runtime)

    def test_independent_drift_warning_reports_values(self, drift_case, caplog):
        trainer, runtime, _ = drift_case

        with caplog.at_level(logging.WARNING):
            trainer._warn_on_arg_drift(runtime)

        messages = [r.getMessage() for r in caplog.records]
        assert len(messages) == 1
        assert "expected_steps_per_epoch" in messages[0]
        assert "saved=8" in messages[0]
        assert "current=16" in messages[0]
        assert "still yields a correct" not in messages[0]
        assert "RDP" not in messages[0]

    def test_independent_run_can_extend(self, drift_case, caplog):
        trainer, runtime, current = drift_case
        current.clear()
        current["total_steps"] = 32

        with caplog.at_level(logging.WARNING):
            trainer._warn_on_arg_drift(runtime)

        assert not caplog.records
