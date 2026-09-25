"""Real Mellum expert LoRA through Opaque's functional training path."""

import json

import pytest
import torch
from examples.moe_privacy.run import (
    ExperimentConfig,
    GrammarDataset,
    collate,
    epsilon_for_run,
    make_model,
)
from examples.moe_privacy.sft_adapters import (
    configure_expert_lora,
    export_trainable,
    load_trainable,
)
from safetensors.torch import load_file, save_file
from torch.func import grad, vmap
from torch.nn.utils import parametrize
from transformers.models.mellum.modeling_mellum import (
    MellumExperts,
    MellumForCausalLM,
    MellumTopKRouter,
)

from opaque.functional import make_functional
from opaque.patches.transformers import moe_geometry
from opaque.transformers import DPTrainer, TrainingArguments


@pytest.fixture
def config():
    return ExperimentConfig(
        name="sft-adapters-test",
        train_sequences=16,
        validation_sequences=4,
        sequence_length=6,
        expected_batch_size=4,
        microbatch_size=2,
        steps=2,
        eval_every=2,
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=4,
        top_k=2,
        moe_intermediate_size=8,
    )


@pytest.fixture
def adapted(config):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(7)
        return configure_expert_lora(make_model(config, 0))


@pytest.fixture
def batch(config):
    dataset = GrammarDataset(config, 3, config.validation_seed)
    batch = collate([dataset[index] for index in range(3)])
    batch["labels"][:, :2] = -100
    batch["input_ids"][2, -2:] = 0
    batch["attention_mask"][2, -2:] = 0
    batch["labels"][2, -2:] = -100
    return batch


def test_native_expert_adapters_do_not_apply_opaque_patches(config, monkeypatch):
    def unexpected_patch(*args, **kwargs):
        raise AssertionError("native adapters must not install Opaque patches")

    monkeypatch.setattr(
        "examples.moe_privacy.sft_adapters.apply_model_patches", unexpected_patch
    )
    model, metadata = configure_expert_lora(make_model(config, 0), apply_patches=False)
    assert metadata["router_count"] == config.num_layers
    assert metadata["expert_adapter_count"] == 4 * config.num_layers
    assert all(
        parameter.dtype == torch.float32
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_expert_adapter_geometry_and_trainable_tree(config, adapted):
    model, metadata = adapted
    assert isinstance(model, MellumForCausalLM)
    assert metadata["adapter_kind"] == "static_parametrization_expert_lora"
    assert json.loads(json.dumps(metadata)) == metadata
    assert metadata["router_count"] == config.num_layers
    assert metadata["expert_adapter_count"] == 4 * config.num_layers
    assert (
        metadata["geometry"]
        == moe_geometry(model)
        == {
            "num_layers": config.num_layers,
            "num_experts": config.num_experts,
            "top_k": config.top_k,
        }
    )
    assert metadata["target_parameters"] == sorted(
        f"model.layers.{layer}.mlp.experts.{parameter}"
        for layer in range(config.num_layers)
        for parameter in ("gate_up_proj", "down_proj")
    )
    assert model._expert_lora_config == {
        "rank": 4,
        "alpha": 8,
        "target_parameters": metadata["target_parameters"],
    }
    _, trainable, frozen = make_functional(model, partition_trainable=True)
    assert set(trainable) == set(
        metadata["expert_adapter_names"] + metadata["router_names"]
    )
    assert set(trainable).isdisjoint(frozen)
    assert set(trainable) | set(frozen) == dict(model.named_parameters()).keys()
    expected_count = (
        config.num_layers
        * config.num_experts
        * (
            4 * (2 * config.hidden_size + 3 * config.moe_intermediate_size)
            + config.hidden_size
        )
    )
    assert metadata["trainable_parameters"] == expected_count
    assert metadata["total_parameters"] == sum(p.numel() for p in model.parameters())
    for name, module in model.named_modules():
        if isinstance(module, MellumExperts):
            for target in ("gate_up_proj", "down_proj"):
                assert parametrize.is_parametrized(module, target)
                stack = module.parametrizations[target]
                assert len(stack) == 1
                assert not stack.original.requires_grad
                experts, out_features, in_features = stack.original.shape
                adapter = stack[0]
                assert adapter.lora_A.shape == (experts, 4, in_features)
                assert adapter.lora_B.shape == (experts, out_features, 4)
                assert adapter.lora_A.dtype == adapter.lora_B.dtype == torch.float32
                assert adapter.lora_A.abs().max() <= in_features**-0.5
                assert torch.count_nonzero(adapter.lora_B) == 0
                assert adapter.scaling == 2
                torch.testing.assert_close(
                    getattr(module, target), stack.original, atol=0, rtol=0
                )
        if isinstance(module, MellumTopKRouter):
            assert f"{name}.weight" in trainable
            assert module.weight.dtype == torch.float32
            assert tuple(module.weight.shape) == (
                config.num_experts,
                config.hidden_size,
            )


def test_expert_lora_eager_task_gradients(adapted, batch):
    model, metadata = adapted
    model(**batch, use_cache=False, router_aux_loss=False).loss.backward()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None, name
        else:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            if name in metadata["router_names"] or name.endswith(".lora_B"):
                assert parameter.grad.norm() > 1e-12, name


def test_bfloat16_backbone_keeps_actual_routers_fp32(config):
    base = make_model(config, 0).to(dtype=torch.bfloat16)
    model, metadata = configure_expert_lora(base)
    parameters = dict(model.named_parameters())
    for name in metadata["router_names"]:
        assert parameters[name].requires_grad
        assert parameters[name].dtype == torch.float32
    for parameter in parameters.values():
        if parameter.requires_grad:
            assert parameter.dtype == torch.float32
        else:
            assert parameter.dtype == torch.bfloat16


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rank": 0},
        {"rank": True},
        {"rank": 1.5},
        {"alpha": 0},
        {"alpha": float("inf")},
        {"alpha": float("nan")},
    ],
)
def test_invalid_adapter_config_does_not_freeze_the_model(config, kwargs):
    base = make_model(config, 0)
    with pytest.raises(ValueError, match=r"rank|alpha"):
        configure_expert_lora(base, **kwargs)
    assert all(parameter.requires_grad for parameter in base.parameters())


def test_cannot_configure_an_existing_adapter(adapted):
    model, _ = adapted
    with pytest.raises(ValueError, match="without adapters"):
        configure_expert_lora(model)


def _snapshot(model):
    return {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }


def _perturb_trainable(model):
    generator = torch.Generator().manual_seed(42)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(0.05 * torch.randn(parameter.shape, generator=generator))


def test_weights_only_export_restores_adapters_and_routers(
    config, adapted, batch, tmp_path
):
    model, metadata = adapted
    model.eval()
    _perturb_trainable(model)
    original = _snapshot(model)
    with torch.no_grad():
        expected = model(**batch, use_cache=False, output_router_logits=True)
    output = tmp_path / "export"
    export_trainable(model, output)
    assert {path.name for path in output.iterdir()} == {
        "trainable.safetensors",
        "adapter_spec.json",
    }
    tensors = load_file(output / "trainable.safetensors")
    assert tensors.keys() == set(
        metadata["expert_adapter_names"] + metadata["router_names"]
    )
    for name, value in tensors.items():
        torch.testing.assert_close(value, original[name], atol=0, rtol=0)
    spec = json.loads((output / "adapter_spec.json").read_text())
    assert spec["adapter_kind"] == "static_parametrization_expert_lora"
    assert spec["config"] == model._expert_lora_config
    assert spec["metadata"] == metadata
    assert spec["tensors"].keys() == tensors.keys()
    again = tmp_path / "again"
    export_trainable(model, again)
    for path in output.iterdir():
        assert (again / path.name).read_bytes() == path.read_bytes()

    restored, _ = configure_expert_lora(make_model(config, 0))
    restored.eval()
    before = _snapshot(restored)
    assert all(not torch.equal(before[name], tensors[name]) for name in tensors)
    references = dict(restored.named_parameters())
    load_trainable(restored, output)
    with torch.no_grad():
        actual = restored(**batch, use_cache=False, output_router_logits=True)
    torch.testing.assert_close(actual.logits, expected.logits, atol=0, rtol=0)
    torch.testing.assert_close(
        actual.router_logits, expected.router_logits, atol=0, rtol=0
    )
    for name, parameter in restored.named_parameters():
        assert parameter is references[name]
        target = tensors[name] if parameter.requires_grad else before[name]
        torch.testing.assert_close(parameter, target, atol=0, rtol=0)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, original[name], atol=0, rtol=0)


def test_export_refuses_existing_directories(adapted, tmp_path):
    model, _ = adapted
    output = tmp_path / "export"
    export_trainable(model, output)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(FileExistsError):
        export_trainable(model, output)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileExistsError):
        export_trainable(model, empty)
    assert not list(empty.iterdir())


@pytest.mark.parametrize("corruption", ["missing", "unexpected", "shape", "dtype"])
def test_load_validates_all_tensors_before_copying(
    config, adapted, tmp_path, corruption
):
    source, _ = adapted
    _perturb_trainable(source)
    output = tmp_path / "export"
    export_trainable(source, output)
    tensors = load_file(output / "trainable.safetensors")
    name = max(tensors)
    if corruption == "missing":
        del tensors[name]
    elif corruption == "unexpected":
        tensors["frozen.weight"] = torch.zeros(2, 2)
    elif corruption == "shape":
        tensors[name] = tensors[name][:-1].contiguous()
    else:
        tensors[name] = tensors[name].double()
    save_file(tensors, output / "trainable.safetensors")
    restored, _ = configure_expert_lora(make_model(config, 0))
    before = _snapshot(restored)
    with pytest.raises(ValueError, match=r"keys differ|shape mismatch|dtype mismatch"):
        load_trainable(restored, output)
    for name, parameter in restored.named_parameters():
        torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)


@pytest.mark.parametrize("section", ["config", "metadata"])
def test_load_rejects_mismatched_configuration(adapted, tmp_path, section):
    model, _ = adapted
    output = tmp_path / "export"
    export_trainable(model, output)
    path = output / "adapter_spec.json"
    saved = json.loads(path.read_text())
    saved[section]["unexpected"] = True
    path.write_text(json.dumps(saved))
    before = _snapshot(model)
    with pytest.raises(ValueError, match="does not match"):
        load_trainable(model, output)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)


def _parametrizations(model):
    return {
        name: module.parametrizations
        for name, module in model.named_modules()
        if parametrize.is_parametrized(module)
    }


def test_expert_lora_functional_vmap_matches_single_example_loop(adapted, batch):
    model, metadata = adapted
    model.train()
    original = dict(model.named_parameters())
    parametrizations_before = _parametrizations(model)
    assert len(parametrizations_before) == metadata["geometry"]["num_layers"]
    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in original.items()
        if not parameter.requires_grad
    }
    fmodel, trainable, frozen = make_functional(
        model, disable_autograd_tracking=True, partition_trainable=True
    )
    assert frozen.keys() == frozen_before.keys()

    def loss(parameters, row):
        return fmodel(
            parameters,
            **{name: value.unsqueeze(0) for name, value in row.items()},
            use_cache=False,
            router_aux_loss=False,
        ).loss

    for _ in range(2):
        actual = vmap(grad(loss), in_dims=(None, 0))(trainable, batch)
        rows = [
            grad(loss)(trainable, {name: value[index] for name, value in batch.items()})
            for index in range(batch["input_ids"].shape[0])
        ]
        assert actual.keys() == trainable.keys()
        for name, value in actual.items():
            assert torch.isfinite(value).all(), name
            torch.testing.assert_close(
                value,
                torch.stack([row[name] for row in rows]),
                atol=2e-6,
                rtol=2e-4,
                msg=name,
            )
            if name in metadata["router_names"] or name.endswith(".lora_B"):
                assert value.norm() > 1e-12, name
        current = dict(model.named_parameters())
        assert current.keys() == original.keys()
        for name, parameter in current.items():
            assert parameter is original[name]
            if name in frozen_before:
                torch.testing.assert_close(
                    parameter, frozen_before[name], atol=0, rtol=0
                )
        assert _parametrizations(model) == parametrizations_before
        assert moe_geometry(model) == metadata["geometry"]


def test_expert_lora_two_update_dp_aux_trainer(config, adapted, tmp_path):
    model, metadata = adapted
    parametrizations_before = _parametrizations(model)
    assert len(parametrizations_before) == metadata["geometry"]["num_layers"]
    original = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    args = TrainingArguments(
        output_dir=str(tmp_path / "trainer"),
        per_device_train_batch_size=config.expected_batch_size,
        microbatch_size=config.microbatch_size,
        max_steps=config.steps,
        learning_rate=0.1,
        optim="sgd",
        lr_scheduler="constant",
        weight_decay=0.0,
        clipping_mode="fixed",
        clipping_norm=config.clipping_norm,
        sampling_mode="poisson",
        privacy_noise_mechanism="gaussian",
        privacy_noise_multiplier=1.0,
        privacy_target_delta=config.delta,
        privacy_accounting=True,
        router_aux_loss_coef=config.router_aux_loss_coef,
        router_aux_kwargs={
            "ratio": config.load_noise_ratio,
            "max_tokens": config.sequence_length,
        },
        use_performance_kernels=False,
        performance_kernels_config={"router_fp32": True},
        use_cpu=True,
        bf16=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        eval_strategy="no",
        logging_strategy="no",
        save_strategy="no",
        report_to=[],
        disable_tqdm=True,
        seed=0,
        data_seed=1,
    )
    trainer = DPTrainer(
        model=model,
        args=args,
        train_dataset=GrammarDataset(config, config.train_sequences, config.data_seed),
        data_collator=collate,
    )
    assert trainer._moe_geometry == metadata["geometry"]
    result = trainer.train()
    assert result.global_step == config.steps == 2
    assert "MoeAux" in repr(trainer._accountant.process)
    epsilon = result.metrics["privacy_epsilon"]
    assert epsilon == pytest.approx(
        epsilon_for_run(config, "dp_aux", 1.0, config.steps), rel=1e-5
    )
    assert epsilon > epsilon_for_run(config, "dp", 1.0, config.steps)
    current = dict(model.named_parameters())
    assert current.keys() == original.keys()
    for name, parameter in current.items():
        if parameter.requires_grad:
            assert not torch.equal(parameter, original[name]), name
        else:
            torch.testing.assert_close(parameter, original[name], atol=0, rtol=0)
    assert _parametrizations(model) == parametrizations_before
