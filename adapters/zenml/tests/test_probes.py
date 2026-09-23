"""Tests for the optional ZenML delivery steps and pipelines."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("zenml")

from adapters.zenml import pipelines, steps

if TYPE_CHECKING:
    from urllib.request import Request


class _SuccessfulResponse:
    def __enter__(self) -> _SuccessfulResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def getcode(self) -> int:
        return 200


def test_probe_report_is_sanitized_and_request_has_no_auth_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_name = "OPAQUE_TEST_HF_TOKEN"
    secret_value = "do-not-leak-this-token"
    captured_request: Request | None = None

    def fake_urlopen(request: Request, *, timeout: float) -> _SuccessfulResponse:
        nonlocal captured_request
        assert timeout == 10.0
        captured_request = request
        return _SuccessfulResponse()

    monkeypatch.setenv(secret_name, secret_value)
    monkeypatch.setattr(steps, "urlopen", fake_urlopen)
    monkeypatch.setattr(steps.torch.cuda, "is_available", lambda: False)

    artifact_path, report = steps.probe_environment.entrypoint(
        require_gpu=False,
        required_secret_env=[secret_name],
        check_huggingface=True,
    )
    try:
        serialized_report = json.dumps(report, sort_keys=True)
        assert report["required_environment"] == {secret_name: True}
        assert secret_value not in serialized_report
        assert captured_request is not None
        assert captured_request.full_url == steps.HUGGINGFACE_API_URL
        assert "Authorization" not in captured_request.headers
    finally:
        shutil.rmtree(artifact_path)


def test_probe_fails_when_required_environment_variable_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_name = "OPAQUE_TEST_MISSING_SECRET"
    monkeypatch.delenv(secret_name, raising=False)

    with pytest.raises(RuntimeError, match=secret_name):
        steps.probe_environment.entrypoint(
            require_gpu=False,
            required_secret_env=[secret_name],
            check_huggingface=False,
        )


def test_gpu_requirement_allocates_and_computes_on_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    devices: list[str | None] = []

    class FakeTensor:
        def square(self) -> FakeTensor:
            return self

        def sum(self) -> FakeTensor:
            return self

        def item(self) -> float:
            return 14.0

    def fake_tensor(_values: list[float], *, device: str | None = None) -> FakeTensor:
        devices.append(device)
        return FakeTensor()

    monkeypatch.setattr(steps.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(steps.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(steps.torch, "tensor", fake_tensor)

    artifact_path, report = steps.probe_environment.entrypoint(
        require_gpu=True,
        required_secret_env=[],
        check_huggingface=False,
    )
    try:
        assert devices == ["cuda"]
        assert report["cuda"]["probe_computation_succeeded"] is True
    finally:
        shutil.rmtree(artifact_path)


def test_cpu_artifact_checksum_verifies_and_tampering_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(steps.torch.cuda, "is_available", lambda: False)
    artifact_path, report = steps.probe_environment.entrypoint(
        require_gpu=False,
        required_secret_env=[],
        check_huggingface=False,
    )
    try:
        verification = steps.verify_probe_artifact.entrypoint(
            probe_artifact=artifact_path,
            expected_report=report,
        )
        assert verification["verified"] is True
        assert verification["report_matches"] is True

        report_path = Path(artifact_path) / steps.REPORT_FILENAME
        report_path.write_text("{}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="checksum mismatch"):
            steps.verify_probe_artifact.entrypoint(
                probe_artifact=artifact_path,
                expected_report=report,
            )
    finally:
        shutil.rmtree(artifact_path)


def test_steps_are_uncached_and_outputs_are_named() -> None:
    assert steps.probe_environment.configuration.enable_cache is False
    assert steps.verify_probe_artifact.configuration.enable_cache is False
    assert set(steps.probe_environment.entrypoint_definition.outputs) == {
        "probe_artifact",
        "probe_report",
    }
    assert set(steps.verify_probe_artifact.entrypoint_definition.outputs) == {
        "probe_verification_report"
    }


@dataclass(frozen=True)
class _FrozenRunSummary:
    global_step: int
    final_loss: float

    def to_dict(self) -> dict[str, object]:
        return {"global_step": self.global_step, "final_loss": self.final_loss}


def test_train_sft_step_delegates_and_keeps_only_bundle_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_commit_sha = "0123456789abcdef"
    job_config: dict[str, object] = {
        "model_name_or_path": "sensitive-model",
        "dataset_name": "owner/dataset",
    }
    resume_reference = "opaque-sft-checkpoint-run@7"
    workspace = tmp_path / "workspace"
    resume_directory = workspace / "resume-checkpoint"
    output_directory = workspace / "model-bundle"
    resolved_config = object()
    frozen_summary = _FrozenRunSummary(global_step=3, final_loss=1.25)
    calls: dict[str, object] = {}

    def fake_resolve(received_config: dict[str, object]) -> object:
        calls["resolve"] = received_config
        return resolved_config

    def fake_mkdtemp(*, prefix: str) -> str:
        assert prefix == "opaque-zenml-sft-"
        workspace.mkdir()
        return str(workspace)

    def fake_materialize(reference: str, destination: Path) -> Path:
        assert os.environ["OPAQUE_SOURCE_COMMIT_SHA"] == "previous-value"
        calls["materialize"] = (reference, destination)
        destination.mkdir()
        (destination / "checkpoint.bin").write_bytes(b"checkpoint")
        return destination

    class FakeCallback:
        def __init__(
            self, *, parent_run_id: str, resume_checkpoint: Path | None
        ) -> None:
            assert os.environ["OPAQUE_SOURCE_COMMIT_SHA"] == "previous-value"
            calls["callback_args"] = (parent_run_id, resume_checkpoint)
            calls["callback"] = self

    def fake_run(
        config: object,
        output_dir: Path,
        *,
        callbacks: tuple[object, ...],
        resume_from_checkpoint: Path | None,
        source_commit_sha: str,
    ) -> _FrozenRunSummary:
        assert os.environ["OPAQUE_SOURCE_COMMIT_SHA"] == source_commit_sha
        calls["run"] = (
            config,
            output_dir,
            callbacks,
            resume_from_checkpoint,
            source_commit_sha,
        )
        output_dir.mkdir()
        (output_dir / "model.bin").write_bytes(b"model")
        return frozen_summary

    runner_module = ModuleType("opaque_examples.sft.runner")
    runner_module.__dict__["run_sft"] = fake_run
    monkeypatch.setattr(steps.tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(steps, "resolve_sft_config", fake_resolve)
    monkeypatch.setattr(steps, "_load_sft_runner", lambda: runner_module)
    monkeypatch.setattr(steps, "_get_current_run_id", lambda: "zenml-run-id")
    monkeypatch.setattr(steps, "materialize_checkpoint", fake_materialize)
    monkeypatch.setattr(steps, "ZenMLCheckpointCallback", FakeCallback)
    monkeypatch.setenv("OPAQUE_SOURCE_COMMIT_SHA", "previous-value")
    environment_before = dict(os.environ)

    model_bundle, run_summary = steps.train_sft_step.entrypoint(
        config=job_config,
        source_commit_sha=source_commit_sha,
        resume_checkpoint=resume_reference,
    )
    try:
        assert calls["resolve"] == job_config
        assert calls["materialize"] == (resume_reference, resume_directory)
        assert calls["callback_args"] == ("zenml-run-id", resume_directory)
        assert calls["run"] == (
            resolved_config,
            output_directory,
            (calls["callback"],),
            resume_directory,
            source_commit_sha,
        )
        assert model_bundle == output_directory
        assert model_bundle.is_dir()
        assert (model_bundle / "model.bin").read_bytes() == b"model"
        assert run_summary == {"global_step": 3, "final_loss": 1.25}
        json.dumps(run_summary, allow_nan=False)
        assert not resume_directory.exists()
        assert set(workspace.iterdir()) == {model_bundle}
        assert dict(os.environ) == environment_before
    finally:
        shutil.rmtree(workspace)


def test_train_sft_step_restores_environment_and_cleans_workspace_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "failed-workspace"

    def fake_mkdtemp(*, prefix: str) -> str:
        assert prefix == "opaque-zenml-sft-"
        workspace.mkdir()
        return str(workspace)

    def fail_run(
        _config: object,
        output_dir: Path,
        *,
        callbacks: tuple[object, ...],
        resume_from_checkpoint: Path | None,
        source_commit_sha: str,
    ) -> None:
        assert callbacks
        assert resume_from_checkpoint is None
        assert source_commit_sha == "abcdef1"
        assert os.environ["OPAQUE_SOURCE_COMMIT_SHA"] == source_commit_sha
        output_dir.mkdir()
        raise RuntimeError("training failed")

    runner_module = ModuleType("opaque_examples.sft.runner")
    runner_module.__dict__["run_sft"] = fail_run
    monkeypatch.setattr(steps.tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(steps, "resolve_sft_config", lambda *_args: object())
    monkeypatch.setattr(steps, "_load_sft_runner", lambda: runner_module)
    monkeypatch.setattr(steps, "_get_current_run_id", lambda: "failed-run-id")
    monkeypatch.setattr(
        steps,
        "materialize_checkpoint",
        lambda *_args: pytest.fail("resume artifact must not be materialized"),
    )
    monkeypatch.setattr(
        steps,
        "ZenMLCheckpointCallback",
        lambda **_kwargs: object(),
    )
    monkeypatch.delenv("OPAQUE_SOURCE_COMMIT_SHA", raising=False)

    with pytest.raises(RuntimeError, match="training failed"):
        steps.train_sft_step.entrypoint(
            config={"model_name_or_path": "model", "dataset_name": "dataset"},
            source_commit_sha="abcdef1",
        )

    assert "OPAQUE_SOURCE_COMMIT_SHA" not in os.environ
    assert not workspace.exists()


def test_train_sft_step_requires_one_gpu_and_has_named_outputs() -> None:
    assert steps.train_sft_step.configuration.enable_cache is False
    resources = steps.train_sft_step.configuration.settings["resources"]
    assert resources.gpu_count == 1
    assert set(steps.train_sft_step.entrypoint_definition.outputs) == {
        "model_bundle",
        "run_summary",
    }


def test_sft_pipeline_wires_training_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[dict[str, object], str, str | None]] = []
    config: dict[str, object] = {"logical_batch_size": 4}

    def fake_train(
        *,
        config: dict[str, object],
        source_commit_sha: str,
        resume_checkpoint: str | None,
    ) -> None:
        calls.append((config, source_commit_sha, resume_checkpoint))

    monkeypatch.setattr(pipelines, "train_sft_step", fake_train)
    pipelines.sft_pipeline.entrypoint(
        config=config,
        source_commit_sha="abcdef",
        resume_checkpoint="checkpoint@3",
    )

    assert calls == [(config, "abcdef", "checkpoint@3")]
    assert pipelines.sft_pipeline.configuration.enable_cache is False
