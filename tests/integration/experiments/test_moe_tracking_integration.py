"""Public-only tracking hooks around the real prepared MoE runners."""

import json
import math
import sys
from types import SimpleNamespace

import pytest
import torch
from examples.moe_privacy import run, sft_run
from examples.moe_privacy.tracking import ExperimentTracker, TrackingOptions


class MetricsSpy:
    def __init__(self):
        self.metrics = []
        self.progress = []

    def log_metrics(self, row):
        self.metrics.append(dict(row))

    def log_progress(self, step, total, seconds):
        self.progress.append((step, total, seconds))


def test_callback_forwards_public_evaluation_and_progress_only(tmp_path, monkeypatch):
    clock = iter([10.0, 12.0, 20.0, 23.0])
    monkeypatch.setattr(run.time, "monotonic", lambda: next(clock))
    tracker = MetricsSpy()
    recorder = run.PublicMetricsRecorder(tmp_path / "metrics.jsonl", tracker=tracker)
    args = SimpleNamespace(max_steps=2, seed=101, data_seed=202)
    state = SimpleNamespace(global_step=0)
    recorder.on_evaluate(
        args,
        state,
        None,
        metrics={
            "eval_loss": 2.0,
            "eval_routing_entropy": 0.75,
            "loss": 99.0,
            "train_loss": 100.0,
            "aux_loss": 101.0,
        },
    )
    recorder.on_train_begin(args, state, None)
    state.global_step = 1
    recorder.on_step_end(args, state, None)
    recorder.on_log(args, state, None, logs={"loss": 99.0, "train_loss": 100.0})

    expected = {
        "step": 0,
        "elapsed_seconds": 2.0,
        "eval_loss": 2.0,
        "eval_routing_entropy": 0.75,
    }
    assert tracker.metrics == recorder.rows == [expected]
    assert json.loads(recorder.path.read_text()) == expected
    assert tracker.progress == [(1, 2, 3.0)]


def test_sft_stdout_progress_is_preserved(tmp_path, monkeypatch, capsys):
    clock = iter([10.0, 20.0, 20.5, 23.0, 23.5])
    monkeypatch.setattr(run.time, "monotonic", lambda: next(clock))
    tracker = MetricsSpy()
    recorder = sft_run.PublicMetricsRecorder(
        tmp_path / "metrics.jsonl", tracker=tracker
    )
    progress = sft_run.Progress()
    args = SimpleNamespace(max_steps=2)
    state = SimpleNamespace(global_step=1)
    for callback in (recorder, progress):
        callback.on_train_begin(args, state, None)
    for callback in (recorder, progress):
        callback.on_step_end(args, state, None)
    assert tracker.progress == [(1, 2, 3.0)]
    assert json.loads(capsys.readouterr().out) == {
        "event": "optimizer_step",
        "step": 1,
        "steps": 2,
        "seconds": 3.0,
    }


@pytest.mark.parametrize("runner", [run, sft_run], ids=["synthetic", "sft"])
@pytest.mark.parametrize("mode", ["online", "offline", "disabled"])
def test_cli_forwards_tracking_options(runner, mode, tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    output = tmp_path / "result"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            runner.__name__,
            "--config",
            str(config_path),
            "--arm",
            "dp_aux",
            "--seed",
            "7",
            "--output-dir",
            str(output),
            "--wandb-mode",
            mode,
            "--wandb-group",
            "matched-arms",
        ],
    )
    calls = []

    def stop_at_runner(config, arm, seed, output_dir, **kwargs):
        calls.append((config, arm, seed, output_dir, kwargs))
        raise RuntimeError("runner boundary reached")

    monkeypatch.setattr(runner, "run_experiment", stop_at_runner)
    with pytest.raises(RuntimeError, match="runner boundary reached"):
        runner.main()
    config, arm, seed, output_dir, kwargs = calls[0]
    assert config == (
        run.ExperimentConfig() if runner is run else sft_run.SFTExperimentConfig()
    )
    assert (arm, seed, output_dir) == ("dp_aux", 7, output)
    assert kwargs["tracking"].mode == mode
    assert kwargs["tracking"].group == "matched-arms"
    assert kwargs["device"] == ("cpu" if runner is run else "cuda")
    assert not output.exists()


@pytest.mark.parametrize("runner", [run, sft_run], ids=["synthetic", "sft"])
def test_existing_output_rejected_before_tracking_starts(runner, tmp_path, monkeypatch):
    device = "cpu" if runner is run else "cuda"
    config = run.ExperimentConfig() if runner is run else sft_run.SFTExperimentConfig()
    monkeypatch.setattr(runner, "resolve_device", lambda _: torch.device(device))

    def unexpected_start(self):
        pytest.fail("tracking must not start before the fresh-output check")

    monkeypatch.setattr(runner.ExperimentTracker, "start", unexpected_start)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner.run_experiment(
            config,
            "dp_aux",
            0,
            tmp_path,
            device=device,
            tracking=TrackingOptions(mode="online"),
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.slow
def test_real_two_update_run_with_tracking_disabled(tmp_path, monkeypatch):
    config = run.ExperimentConfig(
        name="tracking-test",
        train_sequences=16,
        validation_sequences=4,
        sequence_length=6,
        expected_batch_size=4,
        microbatch_size=2,
        steps=2,
        eval_every=1,
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=4,
        top_k=2,
        moe_intermediate_size=8,
    )
    output = tmp_path / "result"
    events = []

    class RecordingTracker(ExperimentTracker):
        def start(self):
            assert output.is_dir()
            assert list(output.iterdir()) == []
            events.append(("start", None))
            return super().start()

        def log_metrics(self, row):
            events.append(("metrics", dict(row)))
            return super().log_metrics(row)

        def log_progress(self, step, total, seconds):
            events.append(("progress", (step, total, seconds)))
            return super().log_progress(step, total, seconds)

        def complete(self, summary):
            events.append(("complete", summary))
            return super().complete(summary)

        def __exit__(self, exc_type, exc_value, traceback):
            events.append(("exit", exc_type))
            return super().__exit__(exc_type, exc_value, traceback)

    monkeypatch.setattr(run, "ExperimentTracker", RecordingTracker)
    seeds = iter([101, 202])
    monkeypatch.setattr(run.secrets, "randbits", lambda _: next(seeds))
    monkeypatch.setenv("WANDB_MODE", "online")
    summary = run.run_experiment(
        config, "dp_aux", 7, output, tracking=TrackingOptions(mode="disabled")
    )

    assert events[0] == ("start", None)
    assert events[-2:] == [("complete", summary), ("exit", None)]
    initial_routing = next(
        index
        for index, (event, row) in enumerate(events)
        if event == "metrics" and row["step"] == 0 and "eval_load_cv" in row
    )
    first_update = next(
        index for index, (event, _) in enumerate(events) if event == "progress"
    )
    assert initial_routing < first_update
    assert (
        events[initial_routing][1]["eval_load_cv"] == summary["initial"]["eval_load_cv"]
    )
    progress = [value for event, value in events if event == "progress"]
    assert [value[:2] for value in progress] == [(1, 2), (2, 2)]
    assert 0 <= progress[0][2] <= progress[1][2]
    for event, row in events:
        if event == "metrics":
            assert all(
                key in {"step", "elapsed_seconds"} or key.startswith("eval_")
                for key in row
            )
    assert summary["status"] == "completed"
    assert summary["privacy"]["steps"] == 2
    assert 0 < summary["privacy"]["epsilon"] <= config.target_epsilon
    assert math.isfinite(summary["final"]["eval_loss"])
    assert json.loads((output / "summary.json").read_text()) == summary
    rows = [
        json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()
    ]
    assert rows[0] == summary["initial"]
    assert rows[-1] == summary["final"]
    assert (output / "model" / "model.safetensors").is_file()
    assert not (output / "wandb_run.json").exists()
