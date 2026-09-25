"""Device selection and transfers without changing the experiment's statistics."""

import sys
from types import SimpleNamespace

import pytest
import torch
from examples.moe_privacy import run
from examples.moe_privacy.checks import run_checks
from examples.moe_privacy.tracking import TrackingOptions


@pytest.mark.parametrize("device", ["mps", "auto", "cuda:0"])
def test_model_rejects_unsupported_device(device):
    with pytest.raises(ValueError, match=r"cpu.*cuda"):
        run.make_model(run.ExperimentConfig(), 0, device=device)


@pytest.mark.parametrize("entrypoint", ["model", "arguments", "checks", "experiment"])
def test_unavailable_cuda_fails_before_work(entrypoint, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def unexpected_work(*args, **kwargs):
        pytest.fail("unavailable CUDA must fail before model construction or seeding")

    monkeypatch.setattr(run, "MellumForCausalLM", unexpected_work)
    monkeypatch.setattr(run.secrets, "randbits", unexpected_work)
    config = run.ExperimentConfig()
    output = tmp_path / "result"
    calls = {
        "model": lambda: run.make_model(config, 0, device="cuda"),
        "arguments": lambda: run.training_arguments(
            config, "dp_aux", output, device="cuda"
        ),
        "checks": lambda: run_checks(config, device="cuda"),
        "experiment": lambda: run.run_experiment(
            config, "dp_aux", 0, output, checks=True, device="cuda"
        ),
    }
    with pytest.raises(RuntimeError, match=r"CUDA.*unavailable.*--device cpu"):
        calls[entrypoint]()
    assert not output.exists()


def test_executor_selection_keeps_cpu_default_and_fp32(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    config = run.ExperimentConfig()
    cpu = run.training_arguments(config, "dp_aux", tmp_path)
    cuda = run.training_arguments(config, "dp_aux", tmp_path, device="cuda")
    assert cpu.use_cpu
    assert cpu.device == torch.device("cpu")
    assert not cuda.use_cpu
    assert cuda.device.type == "cuda"
    for args in (cpu, cuda):
        assert not args.bf16
        assert args.tf32 is False
        assert not args.use_performance_kernels
        assert args.optim == "sgd"


def test_executor_rejects_device_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        run.TrainingArguments, "device", property(lambda _: torch.device("cpu"))
    )
    with pytest.raises(RuntimeError, match=r"resolved.*cpu.*requested.*cuda"):
        run.training_arguments(
            run.ExperimentConfig(), "dp_aux", tmp_path, device="cuda"
        )


def test_default_and_explicit_cpu_initialization_match():
    config = run.ExperimentConfig(sequence_length=8)
    before = torch.random.get_rng_state()
    default = run.make_model(config, 7)
    explicit = run.make_model(config, 7, device="cpu")
    assert torch.equal(before, torch.random.get_rng_state())
    torch.testing.assert_close(
        default.state_dict(), explicit.state_dict(), rtol=0, atol=0
    )
    assert all(parameter.device.type == "cpu" for parameter in default.parameters())
    assert all(parameter.dtype == torch.float32 for parameter in default.parameters())
    assert default.config._attn_implementation == "eager"


def test_cpu_batch_transfer_preserves_data():
    config = run.ExperimentConfig(sequence_length=8)
    dataset = run.GrammarDataset(config, 2, config.validation_seed)
    batch = run.collate([dataset[0], dataset[1]])
    moved = run.move_batch(batch, torch.device("cpu"))
    assert moved is not batch
    torch.testing.assert_close(moved, batch, rtol=0, atol=0)
    assert all(value.device.type == "cpu" for value in moved.values())


def test_cpu_telemetry_does_not_call_cuda(tmp_path, monkeypatch):
    def unexpected_cuda(*args, **kwargs):
        pytest.fail("CPU execution must not initialize CUDA telemetry")

    for name in ("synchronize", "get_device_name", "max_memory_allocated"):
        monkeypatch.setattr(torch.cuda, name, unexpected_cuda)
    recorder = run.PublicMetricsRecorder(tmp_path / "metrics.jsonl")
    recorder.on_step_end(None, None, None)
    execution = run.execution_metadata(torch.device("cpu"))
    assert execution["device"] == "cpu"
    assert execution["device_name"]
    assert execution["cuda_version"] == torch.version.cuda
    assert execution["peak_cuda_memory_allocated_bytes"] is None


def test_cuda_telemetry_keeps_peak_across_counter_resets(tmp_path, monkeypatch):
    device = torch.device("cuda:0")
    synchronized = []
    peaks = iter([128, 512, 256, 64, 32])
    monkeypatch.setattr(torch.cuda, "synchronize", synchronized.append)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: next(peaks))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "test GPU")
    recorder = run.PublicMetricsRecorder(tmp_path / "metrics.jsonl", device)
    recorder.on_train_begin(None, None, None)
    recorder.on_step_end(None, None, None)
    recorder.on_evaluate(
        None, SimpleNamespace(global_step=1), None, metrics={"eval_loss": 1.0}
    )
    execution = run.execution_metadata(
        device,
        peak_cuda_memory_allocated_bytes=recorder.peak_cuda_memory_allocated_bytes,
    )
    assert execution["device"] == "cuda:0"
    assert execution["device_name"] == "test GPU"
    assert execution["peak_cuda_memory_allocated_bytes"] == 512
    assert synchronized == [device, device, device]


@pytest.mark.parametrize("device", [None, "cuda"], ids=["default_cpu", "explicit_cuda"])
def test_cli_passes_runtime_device_without_config_changes(
    device, tmp_path, monkeypatch
):
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    argv = [
        "run",
        "--config",
        str(config_path),
        "--arm",
        "dp_aux",
        "--output-dir",
        str(tmp_path / "result"),
        "--checks",
    ]
    if device is not None:
        argv.extend(["--device", device])
    monkeypatch.setattr(sys, "argv", argv)

    def stop_at_executor(config, arm, seed, output_dir, **kwargs):
        assert config == run.ExperimentConfig()
        assert kwargs == {
            "checks": True,
            "device": device or "cpu",
            "tracking": TrackingOptions(),
        }
        raise RuntimeError("executor boundary reached")

    monkeypatch.setattr(run, "run_experiment", stop_at_executor)
    with pytest.raises(RuntimeError, match="executor boundary reached"):
        run.main()


@pytest.mark.cuda
def test_cuda_uses_identical_cpu_initialized_weights_and_moves_batches():
    config = run.ExperimentConfig(sequence_length=8)
    cpu = run.make_model(config, 7)
    before = torch.cuda.get_rng_state()
    with torch.device("cuda"):
        cuda = run.make_model(config, 7, device="cuda")
    assert torch.equal(before, torch.cuda.get_rng_state())
    device = next(cuda.parameters()).device
    assert device.type == "cuda"
    assert all(parameter.device == device for parameter in cuda.parameters())
    assert all(buffer.device == device for buffer in cuda.buffers())
    assert all(parameter.dtype == torch.float32 for parameter in cuda.parameters())
    torch.testing.assert_close(
        {name: value.cpu() for name, value in cuda.state_dict().items()},
        cpu.state_dict(),
        rtol=0,
        atol=0,
    )
    dataset = run.GrammarDataset(config, 2, config.validation_seed)
    batch = run.collate([dataset[0], dataset[1]])
    moved = run.move_batch(batch, device)
    assert all(value.device == device for value in moved.values())
    assert all(value.device.type == "cpu" for value in batch.values())
    torch.testing.assert_close(
        {name: value.cpu() for name, value in moved.items()}, batch, rtol=0, atol=0
    )
    metrics = run.evaluate_routing(cuda, dataset, config)
    assert metrics["eval_max_expert_share"] > 0
    assert cuda.training
