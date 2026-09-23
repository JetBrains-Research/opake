from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from opaque_examples.sft import runner
from opaque_examples.sft.config import resolve_sft_config
from opaque_examples.sft.data import DatasetManifest, PreparedDataset

_MELLUM2_MAGICODER_SMOKE: dict[str, object] = {
    "model_name_or_path": "JetBrains/Mellum2-12B-A2.5B-Base",
    "model_revision": "main",
    "dataset_name": "ise-uiuc/Magicoder-OSS-Instruct-75K",
    "dataset_revision": "main",
    "prompt_field": "problem",
    "completion_field": "solution",
    "completion_only_loss": True,
    "packing": False,
    "loss_type": "chunked_nll",
    "max_length": 2048,
    "max_train_samples": 512,
    "max_eval_samples": 64,
    "eval_fraction": 0.01,
    "split_seed": 42,
    "num_train_epochs": 1.0,
    "max_steps": 2,
    "logical_batch_size": 128,
    "physical_microbatch_size": 2,
    "auto_microbatch_backoff": True,
    "target_epsilon": 8.0,
    "target_delta": None,
    "noise_multiplier": None,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "lora_dropout": 0.0,
    "bf16": True,
    "attn_implementation": "sdpa",
    "use_performance_kernels": False,
    "gradient_checkpointing": True,
    "chunked_nll": True,
    "save_steps": 1,
    "eval_steps": 1,
    "logging_steps": 1,
}


def _smoke_config():
    return resolve_sft_config(_MELLUM2_MAGICODER_SMOKE)


class _Factory:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def from_pretrained(self, *args: Any, **kwargs: Any) -> Any:
        self.events.append(self.name)
        self.calls.append((args, kwargs))
        if self.name == "tokenizer":
            return SimpleNamespace(pad_token_id=None, pad_token=None, eos_token="<eos>")
        return SimpleNamespace(name="model")


class _Constructor:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(**kwargs)


class _Wandb:
    def __init__(self, events: list[str]) -> None:
        self.run = SimpleNamespace(id="context-wandb-id", url="https://wandb/run/1")
        self.events = events
        self.finish_calls = 0

    def finish(self) -> None:
        self.events.append("wandb.finish")
        self.finish_calls += 1


class _BindingCallback:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.bound: dict[str, Any] | None = None
        self.checkpoint_artifact_ids = ("checkpoint-artifact-id",)

    def bind_sft_run(self, **metadata: Any) -> None:
        self.events.append("callback.bind")
        self.bound = metadata


class _FakeTrainer:
    instances: ClassVar[list[_FakeTrainer]] = []
    events: ClassVar[list[str]] = []
    fail_train = False

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.model = kwargs["model"]
        self.args = kwargs["args"]
        self.state = SimpleNamespace(
            global_step=7,
            privacy_resolved_delta=1e-5,
            privacy_resolved_noise_multiplier=1.75,
            privacy_calibration_converged=True,
            privacy_target_epsilon_reached=False,
            privacy_sample_rate=0.25,
            privacy_total_steps=7,
            converged_microbatch_size=1,
            log_history=[{"privacy_epsilon": 3.5}],
        )
        self.__class__.instances.append(self)
        self.__class__.events.append("trainer.init")

    def train(self, *, resume_from_checkpoint: Any) -> Any:
        self.__class__.events.append("trainer.train")
        self.resume_from_checkpoint = resume_from_checkpoint
        work_dir = Path(self.args.output_dir)
        (work_dir / "checkpoint-7").mkdir(parents=True)
        if self.__class__.fail_train:
            raise RuntimeError("training failed")
        return SimpleNamespace(
            global_step=7,
            training_loss=0.25,
            metrics={
                "train_loss": 0.25,
                "train_samples_per_second": 12.5,
                "privacy_epsilon": 3.5,
                "privacy_delta": 1e-5,
                "privacy_noise_multiplier": 1.75,
                "max_peak_memory_gb": 6.25,
            },
        )

    def evaluate(self) -> dict[str, float]:
        self.__class__.events.append("trainer.evaluate")
        return {"eval_loss": 0.5, "eval_samples_per_second": 20.0}

    def save_model(self, output_dir: str) -> None:
        self.__class__.events.append("trainer.save_model")
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "adapter_model.safetensors").write_text("trained", encoding="utf-8")
        (target / "training_args.bin").write_text("training", encoding="utf-8")
        (target / "accountant.json").write_text("accountant", encoding="utf-8")


@pytest.fixture(autouse=True)
def _reset_trainer() -> None:
    _FakeTrainer.instances = []
    _FakeTrainer.events = []
    _FakeTrainer.fail_train = False


def _prepared() -> PreparedDataset:
    manifest = DatasetManifest(
        source_id="ise-uiuc/Magicoder-OSS-Instruct-75K",
        source_revision="main",
        original_fingerprint="fingerprint",
        prompt_field="problem",
        completion_field="solution",
        split_seed=42,
        eval_fraction=0.01,
        train_cap=512,
        eval_cap=64,
        source_count=600,
        usable_count=600,
        rejected_count=0,
        split_train_count=594,
        split_eval_count=6,
        final_train_count=512,
        final_eval_count=6,
    )
    return PreparedDataset(
        train_dataset=range(512),
        eval_dataset=range(6),
        delta=1e-5,
        manifest=manifest,
    )


def _install_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    writer: Any,
    events: list[str],
) -> tuple[Any, _Factory, _Factory, _Constructor, _Constructor, _Wandb]:
    raw_dataset = object()
    tokenizer_factory = _Factory("tokenizer", events)
    model_factory = _Factory("model", events)
    lora_constructor = _Constructor()
    args_constructor = _Constructor()
    wandb = _Wandb(events)
    dataset_calls: list[dict[str, Any]] = []

    def load_dataset(**kwargs: Any) -> object:
        events.append("load_dataset")
        dataset_calls.append(kwargs)
        return raw_dataset

    def apply_runtime_patches() -> None:
        events.append("runtime_patch")

    def apply_model_patches(model: Any) -> None:
        assert model.name == "model"
        events.append("model_patch")

    dependencies = runner._RunnerDependencies(
        torch=SimpleNamespace(
            bfloat16="bf16",
            float32="float32",
            cuda=SimpleNamespace(is_available=lambda: True),
        ),
        load_dataset=load_dataset,
        auto_tokenizer=tokenizer_factory,
        auto_model_for_causal_lm=model_factory,
        lora_config=lora_constructor,
        sft_config=args_constructor,
        sft_trainer=_FakeTrainer,
        apply_runtime_patches=apply_runtime_patches,
        apply_model_patches=apply_model_patches,
        write_final_bundle=writer,
        wandb=wandb,
    )
    monkeypatch.setattr(runner, "_load_dependencies", lambda: dependencies)
    dependencies.dataset_calls = dataset_calls
    dependencies.raw_dataset = raw_dataset
    return (
        dependencies,
        tokenizer_factory,
        model_factory,
        lora_constructor,
        args_constructor,
        wandb,
    )


def test_run_sft_wires_training_extracts_summary_and_atomically_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "published"
    events = _FakeTrainer.events
    bundle_calls: list[tuple[Path, dict[str, Any]]] = []

    def write_final_bundle(bundle_dir: Path, **kwargs: Any) -> Path:
        events.append("write_bundle")
        assert not output_dir.exists()
        assert Path(kwargs["config"].output_dir) != bundle_dir
        bundle_calls.append((bundle_dir, kwargs))
        kwargs["model"].save_pretrained(bundle_dir)
        assert not (bundle_dir / "training_args.bin").exists()
        assert not (bundle_dir / "accountant.json").exists()
        return bundle_dir

    (
        dependencies,
        tokenizer_factory,
        model_factory,
        lora_constructor,
        args_constructor,
        wandb,
    ) = _install_dependencies(monkeypatch, writer=write_final_bundle, events=events)
    prepared = _prepared()
    prepare_calls: list[tuple[Any, dict[str, Any]]] = []

    def prepare(dataset: Any, **kwargs: Any) -> PreparedDataset:
        events.append("prepare_dataset")
        prepare_calls.append((dataset, kwargs))
        return prepared

    monkeypatch.setattr(runner, "prepare_magicoder_dataset", prepare)
    monkeypatch.setenv("OPAQUE_SOURCE_COMMIT_SHA", "a" * 40)
    monkeypatch.setenv("ZENML_RUN_ID", "zenml-id")
    monkeypatch.setenv("ZENML_RUN_URL", "https://zenml/run/1")
    callback = _BindingCallback(events)
    resume = tmp_path / "resume" / "checkpoint-4"

    summary = runner.run_sft(
        _smoke_config(),
        output_dir,
        callbacks=(callback,),
        resume_from_checkpoint=resume,
    )

    assert dependencies.dataset_calls == [
        {
            "path": "ise-uiuc/Magicoder-OSS-Instruct-75K",
            "name": None,
            "split": "train",
            "revision": "main",
            "streaming": False,
        }
    ]
    assert prepare_calls == [
        (
            dependencies.raw_dataset,
            {
                "source_id": "ise-uiuc/Magicoder-OSS-Instruct-75K",
                "source_revision": "main",
                "prompt_field": "problem",
                "completion_field": "solution",
                "eval_fraction": 0.01,
                "split_seed": 42,
                "train_cap": 512,
                "eval_cap": 64,
                "target_delta": None,
            },
        )
    ]
    assert tokenizer_factory.calls == [
        (("JetBrains/Mellum2-12B-A2.5B-Base",), {"revision": "main"})
    ]
    assert model_factory.calls == [
        (
            ("JetBrains/Mellum2-12B-A2.5B-Base",),
            {
                "revision": "main",
                "torch_dtype": "bf16",
                "attn_implementation": "sdpa",
            },
        )
    ]
    assert lora_constructor.calls == [
        {
            "r": 4,
            "lora_alpha": 8.0,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "lora_dropout": 0.0,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
    ]
    training_args = args_constructor.calls[0]
    assert training_args["completion_only_loss"] is True
    assert training_args["loss_type"] == "chunked_nll"
    assert training_args["per_device_train_batch_size"] == 128
    assert training_args["microbatch_size"] == 2
    assert training_args["auto_find_microbatch_size"] is True
    assert training_args["privacy_target_epsilon"] == 8.0
    assert training_args["privacy_target_delta"] == prepared.delta
    assert training_args["gradient_checkpointing"] is True
    assert training_args["skip_memory_metrics"] is True
    trainer = _FakeTrainer.instances[0]
    assert trainer.kwargs["callbacks"] == [callback]
    assert trainer.kwargs["train_dataset"] is prepared.train_dataset
    assert trainer.kwargs["eval_dataset"] is prepared.eval_dataset
    assert trainer.resume_from_checkpoint is resume
    assert callback.bound == {
        "config": _smoke_config(),
        "dataset_manifest": prepared.manifest,
        "resolved_delta": prepared.delta,
        "source_commit_sha": "a" * 40,
        "run_references": {
            "wandb": {
                "run_id": "context-wandb-id",
                "url": "https://wandb/run/1",
            },
            "zenml": {"run_id": "zenml-id", "url": "https://zenml/run/1"},
        },
    }
    assert events.index("prepare_dataset") < events.index("callback.bind")
    assert events.index("callback.bind") < events.index("trainer.init")
    assert summary.global_step == 7
    assert summary.train_count == 512
    assert summary.eval_count == 6
    assert summary.train_loss == 0.25
    assert summary.eval_loss == 0.5
    assert summary.final_epsilon == 3.5
    assert summary.resolved_delta == 1e-5
    assert summary.noise_multiplier == 1.75
    assert summary.converged_physical_microbatch_size == 1
    assert summary.throughput_samples_per_second == 12.5
    assert summary.peak_memory_gb == 6.25
    assert summary.checkpoint_artifact_ids == ("checkpoint-artifact-id",)
    assert summary.run_references == {
        "wandb": {
            "run_id": "context-wandb-id",
            "url": "https://wandb/run/1",
        },
        "zenml": {"run_id": "zenml-id", "url": "https://zenml/run/1"},
    }
    assert events.index("runtime_patch") < events.index("model_patch")
    assert events.index("trainer.train") < events.index("trainer.evaluate")
    assert events.index("trainer.evaluate") < events.index("write_bundle")
    assert events.index("write_bundle") < events.index("trainer.save_model")
    assert events[-1] == "wandb.finish"
    assert wandb.finish_calls == 1
    assert (output_dir / "adapter_model.safetensors").read_text(
        encoding="utf-8"
    ) == "trained"
    bundle_dir, bundle_kwargs = bundle_calls[0]
    assert bundle_dir != output_dir
    assert isinstance(bundle_kwargs["model"], runner._TrainerModelSaver)
    assert bundle_kwargs["dataset_manifest"] is prepared.manifest
    assert bundle_kwargs["summary"] is summary
    assert bundle_kwargs["train_metrics"]["privacy_epsilon"] == 3.5
    assert bundle_kwargs["eval_metrics"]["eval_loss"] == 0.5
    assert bundle_kwargs["privacy"]["epsilon"] == 3.5
    assert bundle_kwargs["source_commit_sha"] == "a" * 40
    assert bundle_kwargs["run_references"] == summary.run_references
    assert not Path(training_args["output_dir"]).exists()
    json.dumps(asdict(summary), allow_nan=False)
    assert summary.to_dict()["train_examples"] == 512
    assert summary.to_dict()["converged_microbatch_size"] == 1
    with pytest.raises(FrozenInstanceError):
        summary.global_step = 8  # type: ignore[misc]


@pytest.mark.parametrize("failure_stage", ["train", "bundle"])
def test_run_sft_cleans_temporary_state_and_finalizes_wandb_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    output_dir = tmp_path / "published"
    events = _FakeTrainer.events
    staging_paths: list[Path] = []

    def write_final_bundle(bundle_dir: Path, **_: Any) -> Path:
        staging_paths.append(bundle_dir)
        (bundle_dir / "partial").write_text("partial", encoding="utf-8")
        raise RuntimeError("bundle failed")

    _, _, _, _, args_constructor, wandb = _install_dependencies(
        monkeypatch, writer=write_final_bundle, events=events
    )
    monkeypatch.setattr(
        runner, "prepare_magicoder_dataset", lambda *_args, **_kwargs: _prepared()
    )
    _FakeTrainer.fail_train = failure_stage == "train"

    with pytest.raises(RuntimeError, match=f"{failure_stage}.*failed"):
        runner.run_sft(_smoke_config(), output_dir)

    assert not output_dir.exists()
    assert not Path(args_constructor.calls[0]["output_dir"]).exists()
    assert all(not path.exists() for path in staging_paths)
    assert wandb.finish_calls == 1


def test_existing_output_is_never_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "published"
    output_dir.mkdir()
    marker = output_dir / "keep"
    marker.write_text("original", encoding="utf-8")

    monkeypatch.setattr(
        runner,
        "_load_dependencies",
        lambda: pytest.fail("dependencies must not load for an occupied output"),
    )

    with pytest.raises(FileExistsError):
        runner.run_sft(_smoke_config(), output_dir)

    assert marker.read_text(encoding="utf-8") == "original"
