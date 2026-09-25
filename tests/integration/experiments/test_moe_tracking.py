"""Public-metric filtering, offline SDK logging, and resumable log mirroring."""

import argparse
import json
from operator import attrgetter
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
from examples.moe_privacy.tracking import (
    ExperimentTracker,
    TrackingOptions,
    add_tracking_arguments,
    options_from_args,
    public_metrics,
)
from examples.moe_privacy.wandb_mirror import ArmMirror, campaign_progress, metric_rows

_PUBLIC_ENV = {
    "OPAQUE_ZENML_RUN_ID": "12345678-1234-4234-8234-123456789abc",
    "OPAQUE_ZENML_PROJECT_ID": "23456789-1234-4234-8234-123456789abc",
    "OPAQUE_ZENML_STACK_ID": "34567890-1234-4234-8234-123456789abc",
    "OPAQUE_DEPLOYMENT_SOURCE_SHA256": "a" * 64,
    "OPAQUE_DEPLOYMENT_IMAGE": "registry.example.org/team/moe@sha256:" + "b" * 64,
    "OPAQUE_DEPLOYMENT_RESOURCE_PROFILE": "one-80gb-gpu-v1",
}
_FAILURE_POINTS = [
    ("init", "init"),
    ("init", "define_metric"),
    ("init", "summary.__setitem__"),
    ("log_metrics", "log"),
    ("log_progress", "log"),
    ("log_progress", "summary.__setitem__"),
    ("complete", "config.update"),
    ("complete", "log"),
    ("complete", "summary.update"),
    ("complete", "summary.__setitem__"),
    ("finish", "summary.__setitem__"),
    ("finish", "finish"),
]


@pytest.fixture(autouse=True)
def clean_tracking_environment(monkeypatch):
    for key in (
        *_PUBLIC_ENV,
        "WANDB_MODE",
        "WANDB_BASE_URL",
        "WANDB_ENTITY",
        "WANDB_PROJECT",
        "WANDB_RUN_GROUP",
        "WANDB_NAME",
        "WANDB_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def sdk(monkeypatch):
    wandb = pytest.importorskip("wandb")
    run = MagicMock()
    run.id = "public01"
    run.url = "https://jetbrains.wandb.io/federated-compute/opaque/runs/public01"
    sdk = SimpleNamespace(
        init=Mock(return_value=run), Settings=wandb.Settings, errors=wandb.errors
    )

    def load(name):
        assert name == "wandb"
        return sdk

    monkeypatch.setattr("examples.moe_privacy.tracking.importlib.import_module", load)
    return sdk


def _summary():
    return {
        "status": "completed",
        "config": {"steps": 2, "model_id": "public/model", "noise_seed": 123456},
        "initial": {"step": 0, "eval_loss": 2.0},
        "final": {"step": 2, "eval_loss": 1.8},
        "test": {"eval_loss": 1.9},
        "privacy": {"epsilon": 7.9, "noise_seed": 123456},
    }


def _tracker(tmp_path, *, fail_open=True, mode="online", config=None):
    return ExperimentTracker(
        tmp_path,
        config=config if config is not None else {"steps": 2},
        arm="dp_aux",
        seed=0,
        experiment="synthetic_moe",
        options=TrackingOptions(mode=mode, fail_open=fail_open),
    )


def _sdk_method(sdk, method):
    return sdk.init if method == "init" else attrgetter(method)(sdk.init.return_value)


def _invoke(tracker, operation):
    if operation == "init":
        tracker.start()
    elif operation == "log_metrics":
        tracker.log_metrics({"step": 0, "eval_loss": 2.0})
    elif operation == "log_progress":
        tracker.log_progress(1, 2, 0.1)
    elif operation == "complete":
        tracker.complete(_summary())
    else:
        assert operation == "finish"
        tracker.__exit__(None, None, None)


def test_fail_open_cli_is_explicit_and_legacy_namespaces_remain_strict(monkeypatch):
    monkeypatch.setenv("WANDB_FAIL_OPEN", "true")
    parser = argparse.ArgumentParser()
    add_tracking_arguments(parser)
    args = parser.parse_args([])
    assert TrackingOptions().fail_open is False
    assert options_from_args(args).fail_open is False
    del args.wandb_fail_open
    assert options_from_args(args).fail_open is False
    args = parser.parse_args(["--wandb-mode", "online", "--wandb-fail-open"])
    assert options_from_args(args).mode == "online"
    assert options_from_args(args).fail_open is True


@pytest.mark.parametrize("fail_open", ["true", 1, None])
def test_fail_open_rejects_non_boolean_options(fail_open):
    with pytest.raises(TypeError, match="boolean"):
        TrackingOptions(fail_open=fail_open)


@pytest.mark.parametrize(("operation", "method"), _FAILURE_POINTS)
def test_online_fail_open_preserves_local_results(tmp_path, sdk, operation, method):
    summary = _summary()
    summary_bytes = json.dumps(summary).encode()
    metric_bytes = b'{"step": 2, "eval_loss": 1.8}\n'
    error = sdk.errors.CommError("credential=must-not-be-recorded " * 1000)
    with _tracker(tmp_path) as tracker:
        if operation == "init":
            _sdk_method(sdk, method).side_effect = error
        tracker.start()
        if operation not in {"init", "finish"}:
            _sdk_method(sdk, method).side_effect = error
            _invoke(tracker, operation)
        (tmp_path / "summary.json").write_bytes(summary_bytes)
        (tmp_path / "metrics.jsonl").write_bytes(metric_bytes)
        tracker.complete(summary)
        assert tracker.completed
        assert tracker.status == "completed"
        if operation == "finish":
            _sdk_method(sdk, method).side_effect = error
        else:
            assert (
                json.loads((tmp_path / "wandb_run.json").read_text())["upload_status"]
                == "failed"
            )
    (tmp_path / "adapter.safetensors").write_bytes(b"local export")
    assert (tmp_path / "summary.json").read_bytes() == summary_bytes
    assert (tmp_path / "metrics.jsonl").read_bytes() == metric_bytes
    assert json.dumps(summary).encode() == summary_bytes
    assert tracker.completed
    assert tracker.status == "completed"
    receipt = json.loads((tmp_path / "wandb_run.json").read_text())
    assert receipt["status"] == "completed"
    assert receipt["upload_status"] == "failed"
    failure = (tmp_path / "tracking_failure.json").read_text()
    assert json.loads(failure) == {
        "operation": operation,
        "exception_type": "CommError",
    }
    assert len(failure) < 256
    assert "credential" not in failure
    assert (tmp_path / "wandb").is_dir()
    run = sdk.init.return_value
    if method == "init":
        assert receipt["url"] is None
        run.finish.assert_not_called()
    else:
        run.finish.assert_called_once_with(exit_code=0)
    run.log_artifact.assert_not_called()
    run.save.assert_not_called()
    run.watch.assert_not_called()


@pytest.mark.parametrize(("operation", "method"), _FAILURE_POINTS)
@pytest.mark.parametrize(("mode", "fail_open"), [("online", False), ("offline", True)])
def test_outages_stay_strict_without_online_opt_in(
    tmp_path, sdk, operation, method, mode, fail_open
):
    tracker = _tracker(tmp_path, mode=mode, fail_open=fail_open)
    if operation != "init":
        tracker.start()
    if operation == "finish":
        tracker.complete(_summary())
    error = sdk.errors.CommError("network unavailable")
    _sdk_method(sdk, method).side_effect = error
    with pytest.raises(sdk.errors.CommError) as caught:
        _invoke(tracker, operation)
    assert caught.value is error
    assert not (tmp_path / "tracking_failure.json").exists()


@pytest.mark.parametrize(
    "operation", ["init", "log_metrics", "log_progress", "complete", "finish"]
)
@pytest.mark.parametrize(
    "error_type", [ValueError, TypeError, RuntimeError, PermissionError]
)
def test_fail_open_never_swallows_programming_or_local_io_errors(
    tmp_path, sdk, operation, error_type
):
    tracker = _tracker(tmp_path)
    if operation != "init":
        tracker.start()
    method = {"init": "init", "complete": "config.update", "finish": "finish"}.get(
        operation, "log"
    )
    error = error_type("not a communication error")
    _sdk_method(sdk, method).side_effect = error
    with pytest.raises(error_type) as caught:
        _invoke(tracker, operation)
    assert caught.value is error
    assert not (tmp_path / "tracking_failure.json").exists()


@pytest.mark.parametrize("error_name", ["Error", "UsageError"])
def test_fail_open_does_not_catch_generic_sdk_errors(tmp_path, sdk, error_name):
    error_type = getattr(sdk.errors, error_name)
    sdk.init.side_effect = error_type("invalid SDK configuration")
    with pytest.raises(error_type):
        _tracker(tmp_path).start()
    assert not (tmp_path / "tracking_failure.json").exists()


@pytest.mark.parametrize("error_type", [ConnectionError, BrokenPipeError, TimeoutError])
def test_online_transport_failures_are_fail_open(tmp_path, sdk, error_type):
    sdk.init.side_effect = error_type("private connection details")
    with _tracker(tmp_path) as tracker:
        tracker.start()
        tracker.complete(_summary())
    assert json.loads((tmp_path / "tracking_failure.json").read_text()) == {
        "operation": "init",
        "exception_type": error_type.__name__,
    }


@pytest.mark.parametrize("error_name", ["ConnectionError", "Timeout", "InvalidURL"])
def test_requests_transport_errors_do_not_include_invalid_urls(
    tmp_path, sdk, error_name
):
    import requests.exceptions

    error_type = getattr(requests.exceptions, error_name)
    sdk.init.side_effect = error_type("private transport details")
    tracker = _tracker(tmp_path)
    if error_name == "InvalidURL":
        with pytest.raises(error_type):
            tracker.start()
        assert not (tmp_path / "tracking_failure.json").exists()
    else:
        tracker.start()
        assert json.loads((tmp_path / "tracking_failure.json").read_text()) == {
            "operation": "init",
            "exception_type": error_name,
        }


@pytest.mark.parametrize("online", [False, True])
def test_auto_fail_open_requires_effective_online_mode(
    tmp_path, monkeypatch, sdk, online
):
    if online:
        monkeypatch.setenv("WANDB_API_KEY", "not-for-public-config")
    sdk.init.side_effect = sdk.errors.CommError("unavailable")
    tracker = _tracker(tmp_path, mode="auto")
    if online:
        tracker.start()
        assert (tmp_path / "tracking_failure.json").exists()
    else:
        with pytest.raises(sdk.errors.CommError):
            tracker.start()
        assert not (tmp_path / "tracking_failure.json").exists()
    assert sdk.init.call_args.kwargs["mode"] == ("online" if online else "offline")


def test_first_failure_is_bounded_and_does_not_retry_logs(tmp_path, sdk):
    with _tracker(tmp_path) as tracker:
        tracker.start()
        run = sdk.init.return_value
        run.log.side_effect = sdk.errors.CommError("secret first failure")
        run.finish.side_effect = TimeoutError("secret second failure")
        for step in range(100):
            tracker.start()
            tracker.log_metrics({"step": step, "eval_loss": 2.0})
            tracker.log_progress(step, 100, 0.1)
        tracker.complete(_summary())
    sdk.init.assert_called_once()
    run.log.assert_called_once()
    run.finish.assert_called_once_with(exit_code=0)
    assert json.loads((tmp_path / "tracking_failure.json").read_text()) == {
        "operation": "log_metrics",
        "exception_type": "CommError",
    }
    assert (
        json.loads((tmp_path / "wandb_run.json").read_text())["upload_status"]
        == "failed"
    )


@pytest.mark.parametrize("failed_init", [False, True])
def test_invalid_inputs_still_raise_after_outage(tmp_path, sdk, failed_init):
    tracker = _tracker(tmp_path)
    if failed_init:
        sdk.init.side_effect = sdk.errors.CommError("unavailable")
    tracker.start()
    if not failed_init:
        sdk.init.return_value.log.side_effect = sdk.errors.CommError("unavailable")
        tracker.log_metrics({"step": 0, "eval_loss": 2.0})
    for step in (-1, 0.5, True, None):
        with pytest.raises(ValueError, match="integer step"):
            tracker.log_metrics({"step": step, "eval_loss": 1.0})
    with pytest.raises(ValueError, match="optimizer progress"):
        tracker.log_progress(3, 2, 1.0)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        tracker.log_progress(1, 2, float("nan"))
    with pytest.raises(ValueError, match="completed experiment"):
        tracker.complete({"status": "failed"})
    summary = _summary()
    summary["final"]["step"] = -1
    with pytest.raises(ValueError, match="integer step"):
        tracker.complete(summary)
    summary = _summary()
    summary["config"]["model_id"] = object()
    with pytest.raises(TypeError):
        tracker.complete(summary)
    assert not tracker.completed


def test_invalid_config_is_not_hidden_by_init_outage(tmp_path, sdk):
    sdk.init.side_effect = sdk.errors.CommError("unavailable")
    with pytest.raises(TypeError):
        _tracker(tmp_path, config={"model_id": object()}).start()
    sdk.init.assert_not_called()
    assert not (tmp_path / "tracking_failure.json").exists()


def test_complete_validates_rows_before_sdk_communication(tmp_path, sdk):
    tracker = _tracker(tmp_path)
    tracker.start()
    sdk.init.return_value.config.update.side_effect = sdk.errors.CommError(
        "unavailable"
    )
    summary = _summary()
    summary["final"]["step"] = "not an integer"
    with pytest.raises(ValueError, match="integer step"):
        tracker.complete(summary)
    sdk.init.return_value.config.update.assert_not_called()
    assert not tracker.completed


def test_finish_outage_does_not_mask_training_exception(tmp_path, sdk):
    tracker = _tracker(tmp_path)

    def fail_training():
        with tracker:
            tracker.start()
            sdk.init.return_value.finish.side_effect = sdk.errors.CommError(
                "unavailable"
            )
            raise ValueError("training failed")

    with pytest.raises(ValueError, match="training failed"):
        fail_training()
    assert tracker.status == "failed"
    assert not tracker.completed
    receipt = json.loads((tmp_path / "wandb_run.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["upload_status"] == "failed"
    sdk.init.return_value.finish.assert_called_once_with(exit_code=1)


@pytest.mark.parametrize("mode", ["online", "offline"])
def test_training_completion_is_not_an_upload_receipt(tmp_path, sdk, mode):
    with _tracker(tmp_path, mode=mode) as tracker:
        tracker.start()
        tracker.complete(_summary())
        receipt = json.loads((tmp_path / "wandb_run.json").read_text())
        assert receipt["status"] == "completed"
        assert receipt["upload_status"] == (
            "pending" if mode == "online" else "offline"
        )
    receipt = json.loads((tmp_path / "wandb_run.json").read_text())
    assert receipt["upload_status"] == ("completed" if mode == "online" else "offline")
    assert set(receipt) == {
        "run_id",
        "url",
        "status",
        "entity",
        "project",
        "base_url",
        "upload_status",
    }
    assert not (tmp_path / "tracking_failure.json").exists()
    assert not any(key.startswith(("zenml_", "deployment_")) for key in tracker.config)


@pytest.mark.parametrize("failed_init", [False, True])
def test_validated_public_environment_links_config_and_receipt(
    tmp_path, monkeypatch, sdk, failed_init
):
    for key, value in _PUBLIC_ENV.items():
        monkeypatch.setenv(key, value)
    for key in (
        "WANDB_API_KEY",
        "OPAQUE_ZENML_SECRET",
        "OPAQUE_DEPLOYMENT_OTHER",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ):
        monkeypatch.setenv(key, "do-not-capture-this-secret")
    if failed_init:
        sdk.init.side_effect = sdk.errors.CommError("do-not-capture-this-secret")
    with _tracker(
        tmp_path, config={"zenml_run_id": "spoof", "noise_seed": 123456}
    ) as tracker:
        tracker.start()
        tracker.complete(_summary())
    expected = {
        key.removeprefix("OPAQUE_").lower(): value for key, value in _PUBLIC_ENV.items()
    }
    config = sdk.init.call_args.kwargs["config"]
    receipt = json.loads((tmp_path / "wandb_run.json").read_text())
    for key, value in expected.items():
        assert config[key] == value
        assert receipt[key] == value
    assert set(receipt) == {
        "run_id",
        "url",
        "status",
        "entity",
        "project",
        "base_url",
        "upload_status",
        *expected,
    }
    assert "do-not-capture-this-secret" not in json.dumps([config, receipt])
    assert "noise_seed" not in config
    assert sdk.init.call_args.kwargs["tags"] == ["moe", "dp_aux", "public-data"]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("OPAQUE_ZENML_RUN_ID", "secret-run-name"),
        ("OPAQUE_ZENML_PROJECT_ID", ""),
        ("OPAQUE_ZENML_STACK_ID", _PUBLIC_ENV["OPAQUE_ZENML_STACK_ID"] + "\nsecret"),
        ("OPAQUE_DEPLOYMENT_SOURCE_SHA256", "g" * 64),
        ("OPAQUE_DEPLOYMENT_SOURCE_SHA256", "a" * 63),
        ("OPAQUE_DEPLOYMENT_RESOURCE_PROFILE", "other-profile"),
        ("OPAQUE_DEPLOYMENT_IMAGE", "registry.example.org/team/moe:latest"),
        (
            "OPAQUE_DEPLOYMENT_IMAGE",
            "https://" + _PUBLIC_ENV["OPAQUE_DEPLOYMENT_IMAGE"],
        ),
        (
            "OPAQUE_DEPLOYMENT_IMAGE",
            "user:secret@" + _PUBLIC_ENV["OPAQUE_DEPLOYMENT_IMAGE"],
        ),
        (
            "OPAQUE_DEPLOYMENT_IMAGE",
            _PUBLIC_ENV["OPAQUE_DEPLOYMENT_IMAGE"] + "?token=secret",
        ),
        (
            "OPAQUE_DEPLOYMENT_IMAGE",
            "registry.example.org/team/moe:tag@sha256:" + "b" * 64,
        ),
        ("OPAQUE_DEPLOYMENT_IMAGE", "registry.example.org/../moe@sha256:" + "b" * 64),
        (
            "OPAQUE_DEPLOYMENT_IMAGE",
            "registry..example.org/team/moe@sha256:" + "b" * 64,
        ),
        (
            "OPAQUE_DEPLOYMENT_IMAGE",
            "registry.example.org:65536/moe@sha256:" + "b" * 64,
        ),
        ("OPAQUE_DEPLOYMENT_IMAGE", "registry.example.org:0/moe@sha256:" + "b" * 64),
        ("OPAQUE_DEPLOYMENT_IMAGE", "moe@sha256:" + "b" * 64),
        (
            "OPAQUE_DEPLOYMENT_IMAGE",
            "registry.example.org/" + "a" * 1000 + "@sha256:" + "b" * 64,
        ),
    ],
)
def test_invalid_link_environment_raises_without_disclosing_values(
    tmp_path, monkeypatch, sdk, key, value
):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match=key) as caught:
        _tracker(tmp_path).start()
    if value:
        assert value not in str(caught.value)
    sdk.init.assert_not_called()
    assert not (tmp_path / "wandb_run.json").exists()
    assert not (tmp_path / "tracking_failure.json").exists()


def test_digest_registry_port_and_uppercase_uuid_and_source_hash(
    tmp_path, monkeypatch, sdk
):
    values = {
        "OPAQUE_DEPLOYMENT_IMAGE": "localhost:5000/team/moe@sha256:" + "b" * 64,
        "OPAQUE_ZENML_RUN_ID": _PUBLIC_ENV["OPAQUE_ZENML_RUN_ID"].upper(),
        "OPAQUE_DEPLOYMENT_SOURCE_SHA256": "A" * 64,
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    with _tracker(tmp_path) as tracker:
        tracker.start()
        tracker.complete(_summary())
    receipt = json.loads((tmp_path / "wandb_run.json").read_text())
    for key, value in values.items():
        name = key.removeprefix("OPAQUE_").lower()
        assert receipt[name] == value
        assert tracker.config[name] == value
    assert "zenml_stack_id" not in receipt


def test_public_metrics_drop_private_and_nonfinite_fields():
    assert public_metrics(
        {
            "eval_loss": 1.5,
            "eval_load_cv": 0.3,
            "eval_layer_0_expert_1_share": 0.2,
            "loss": 2.0,
            "aux_loss": 7.0,
            "noise_seed": 314,
            "seed": 19,
            "eval_private_debug": 99.0,
            "eval_runtime": float("nan"),
            "prompt": "secret",
        }
    ) == {"eval/loss": 1.5, "eval/load_cv": 0.3, "eval/layer_0_expert_1_share": 0.2}


@pytest.mark.parametrize("mode", [None, "disabled"])
def test_disabled_tracker_never_imports_sdk_or_creates_output(
    tmp_path, monkeypatch, mode
):
    def fail_import(name):
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(
        "examples.moe_privacy.tracking.importlib.import_module", fail_import
    )
    path = tmp_path / "absent"
    with ExperimentTracker(
        path,
        config={},
        arm="dp",
        seed=0,
        experiment="synthetic_moe",
        options=TrackingOptions(mode=mode, fail_open=True)
        if mode is not None
        else None,
    ) as tracker:
        tracker.start()
        tracker.log_progress(1, 2, 0.5)
        tracker.log_metrics({"step": 0, "eval_loss": 1.5})
        tracker.complete({"status": "completed"})
    assert not path.exists()


@pytest.mark.parametrize("arm", ["trl_reference", "trl_reference_aux"])
def test_trl_controls_are_labeled_native_and_nonprivate(tmp_path, sdk, arm):
    tracker = ExperimentTracker(
        tmp_path,
        config={
            "router_aux_loss_coef": 0.2,
            "target_epsilon": 8,
            "load_noise_ratio": 1,
            "trainer_backend": "trl.SFTTrainer",
            "physical_batch_size": 4,
            "gradient_accumulation_steps": 8,
            "batch_clipping_norm": 1.0,
        },
        arm=arm,
        seed=0,
        experiment="native_trl_moe_sft",
        options=TrackingOptions(
            mode="online", fail_open=True, name=f"{arm}-coef0.2-ratio1-trl"
        ),
    )
    balanced = arm == "trl_reference_aux"
    assert tracker.config["private"] is False
    assert tracker.config["target_epsilon"] is None
    assert tracker.config["load_noise_ratio"] is None
    assert tracker.config["balancing_enabled"] is balanced
    assert tracker.config["balancing_kind"] == (
        "native_current_batch" if balanced else "off"
    )
    assert tracker.config["router_aux_loss_coef"] == (0.2 if balanced else 0.0)
    assert tracker.config["trainer_backend"] == "trl.SFTTrainer"
    assert tracker.config["nonprivate_control"] == "native_trl_batch_gradient_clipping"
    with tracker:
        tracker.start()
        tracker.complete({"status": "completed", "config": dict(tracker.config)})
    kwargs = sdk.init.call_args.kwargs
    assert kwargs["tags"] == ["moe", arm, "public-data"]
    assert kwargs["name"] == f"{arm}-coef0.2-ratio1-trl"
    actual = sdk.init.return_value.config.update.call_args.args[0]
    assert actual["target_epsilon"] is None
    assert actual["load_noise_ratio"] is None
    assert actual["router_aux_loss_coef"] == (0.2 if balanced else 0.0)


@pytest.mark.parametrize("fail_open", [False, True])
def test_explicit_missing_sdk_is_actionable_but_auto_warns(
    tmp_path, monkeypatch, fail_open
):
    def missing(name):
        raise ImportError(name)

    monkeypatch.setattr(
        "examples.moe_privacy.tracking.importlib.import_module", missing
    )
    tracker = ExperimentTracker(
        tmp_path,
        config={},
        arm="dp",
        seed=0,
        experiment="synthetic_moe",
        options=TrackingOptions(mode="online", fail_open=fail_open),
    )
    with pytest.raises(RuntimeError, match="optional SDK"):
        tracker.start()
    tracker.options = TrackingOptions(mode="auto", fail_open=fail_open)
    with pytest.warns(UserWarning, match="local JSON"):
        tracker.start()


def test_real_offline_sdk_records_public_history_and_summary(tmp_path):
    pytest.importorskip("wandb")
    config = {
        "name": "offline-test",
        "steps": 2,
        "target_epsilon": 8,
        "noise_seed": 123456,
        "WANDB_API_KEY": "do-not-upload",
    }
    with ExperimentTracker(
        tmp_path,
        config=config,
        arm="dp",
        seed=7,
        experiment="synthetic_moe",
        options=TrackingOptions(mode="offline"),
    ) as tracker:
        tracker.start()
        tracker.log_metrics({"step": 0, "eval_loss": 2.0, "loss": 17.0})
        tracker.log_progress(1, 2, 0.1)
        tracker.log_progress(2, 2, 0.2)
        tracker.complete(
            {
                "status": "completed",
                "config": {"model_id": "public/model", "noise_seed": 123456},
                "privacy": {
                    "private": True,
                    "epsilon": 7.9,
                    "delta": 1e-5,
                    "noise_seed": 123456,
                },
                "initial": {"step": 0, "eval_loss": 2.0},
                "final": {"step": 2, "eval_loss": 1.8},
            }
        )
        run = tracker.run
        assert run.config["model_seed"] == 7
        assert run.config["model_id"] == "public/model"
        assert "noise_seed" not in run.config
        assert "WANDB_API_KEY" not in run.config
        assert run.summary["privacy/epsilon"] == 7.9
        assert run.summary["final/eval/loss"] == 1.8
        with pytest.raises(KeyError):
            run.summary["privacy/noise_seed"]
        assert run.settings.console == "off"
        assert run.settings.disable_git
        assert run.settings.x_disable_stats
        assert run.settings.x_disable_meta
        assert run.settings.base_url == "https://jetbrains.wandb.io"
    receipt = json.loads((tmp_path / "wandb_run.json").read_text())
    assert receipt["status"] == "completed"
    assert receipt["upload_status"] == "offline"
    assert len(receipt["run_id"]) == 8
    assert (tmp_path / "wandb").is_dir()


def test_offline_exception_is_not_a_completed_experiment(tmp_path):
    pytest.importorskip("wandb")

    def fail_training():
        with ExperimentTracker(
            tmp_path,
            config={},
            arm="reference",
            seed=0,
            experiment="synthetic_moe",
            options=TrackingOptions(mode="offline"),
        ) as tracker:
            tracker.start()
            assert tracker.run.config["target_epsilon"] is None
            raise ValueError("training failed")

    with pytest.raises(ValueError, match="training failed"):
        fail_training()
    assert json.loads((tmp_path / "wandb_run.json").read_text())["status"] == "failed"


def test_partial_metric_write_is_retried(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step": 0, "eval_loss": 2}\n{"step": 1')
    assert metric_rows(path) == [{"step": 0, "eval_loss": 2}]
    path.write_text(path.read_text() + ', "eval_loss": 1}\n')
    assert len(metric_rows(path)) == 2


def test_serial_campaign_progress_is_assigned_to_correct_arm(tmp_path):
    path = tmp_path / "campaign.log"
    rows = [
        {"event": "optimizer_step", "step": 1, "steps": 2, "seconds": 1},
        {"event": "completed", "arm": "reference"},
        {"event": "optimizer_step", "step": 1, "steps": 2, "seconds": 2},
        {"event": "completed", "arm": "dp"},
        {"event": "optimizer_step", "step": 1, "steps": 2, "seconds": 3},
    ]
    path.write_text(
        "warning: not JSON\n" + "".join(json.dumps(row) + "\n" for row in rows)
    )
    progress = campaign_progress(path, ("reference", "dp", "dp_aux"))
    assert [progress[arm][0]["seconds"] for arm in progress] == [1, 2, 3]
    with pytest.raises(ValueError, match="completion order"):
        campaign_progress(path, ("dp", "reference", "dp_aux"))


class Recorder:
    """Capture mirror events without substituting any training or SDK behavior."""

    def __init__(self):
        self.config = {"arm": "dp"}
        self.events = []

    def start(self):
        self.events.append("start")

    def log_metrics(self, row):
        self.events.append(row)

    def log_progress(self, step, total, seconds):
        self.events.append((step, total, seconds))

    def complete(self, summary):
        self.events.append("complete")

    def __exit__(self, *args):
        self.events.append("finish")


def test_mirror_backfill_rewrite_and_restart_cursors(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step": 0, "eval_loss": 2}\n')
    recorder = Recorder()
    mirror = ArmMirror(tmp_path, recorder)
    progress = [{"step": 1, "steps": 2, "seconds": 1.0}]
    mirror.poll(progress)
    assert recorder.events == ["start", {"step": 0, "eval_loss": 2}, (1, 2, 1.0)]
    mirror.poll(progress)
    assert len(recorder.events) == 3
    path.write_text('{"step": 0, "eval_loss": 2, "eval_load_cv": 0.5}\n')
    mirror.poll(progress)
    assert recorder.events[-1] == {"step": 0, "eval_load_cv": 0.5}
    restarted = Recorder()
    mirror = ArmMirror(tmp_path, restarted)
    mirror.poll(progress)
    assert restarted.events == ["start"]
    (tmp_path / "summary.json").write_text('{"status":"completed","arm":"dp"}')
    mirror.poll(progress)
    assert restarted.events == ["start", "complete", "finish"]
    assert mirror.completed
    assert ArmMirror(tmp_path, Recorder()).completed


def test_mirror_waits_for_future_arm_without_creating_its_directory(tmp_path):
    directory = tmp_path / "future"
    recorder = Recorder()
    mirror = ArmMirror(directory, recorder)
    mirror.poll([])
    assert not directory.exists()
    assert not recorder.events
