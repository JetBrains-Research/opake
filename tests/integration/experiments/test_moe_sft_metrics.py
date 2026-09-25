"""Public completion quality, dispatch balance, and frozen Mellum gradients."""

import json
import math

import pytest
import torch
import torch.nn.functional as F
from examples.moe_privacy import sft_metrics
from examples.moe_privacy.run import ExperimentConfig, make_model
from examples.moe_privacy.sft_adapters import configure_expert_lora
from examples.moe_privacy.sft_data import CompletionCollator
from examples.moe_privacy.sft_metrics import (
    _balancing_surrogate,
    _load_statistics,
    _noise_forecast,
    _public_load_reference,
    _routing_metrics,
    evaluate_public,
    probe_balancing_signal,
)
from transformers.models.mellum.modeling_mellum import MellumTopKRouter

from opaque.dpsgd.clipping import moe_clipped_grad
from opaque.random import key as rng_key


@pytest.fixture
def model():
    config = ExperimentConfig(
        name="public-sft-metrics-test",
        train_sequences=16,
        validation_sequences=4,
        sequence_length=6,
        expected_batch_size=4,
        microbatch_size=1,
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
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(7)
        adapted, _ = configure_expert_lora(make_model(config, 0), rank=2, alpha=4)
    return adapted


@pytest.fixture
def records():
    return [
        {"input_ids": [1, 3, 4, 5, 2], "completion_mask": [0, 0, 1, 1, 1]},
        {"input_ids": [1, 6, 2], "completion_mask": [0, 0, 1]},
        {"input_ids": [1, 7, 8, 9, 10, 2], "completion_mask": [0, 0, 0, 1, 1, 1]},
    ]


@pytest.fixture
def collator():
    return CompletionCollator(6, 0)


def _probe(model, dataset, collator, **overrides):
    kwargs = {
        "coefficient": 0.05,
        "mean_tokens": 6.0,
        "max_tokens": 6,
        "expected_batch_size": 4,
        "noise_multiplier": 1.2,
        "ratio": 0.02,
        "filter_beta": 0.99,
        "steps": 256,
    }
    return probe_balancing_signal(model, dataset, collator, **(kwargs | overrides))


def _router_parameters(model):
    return tuple(
        module.weight
        for module in model.modules()
        if isinstance(module, MellumTopKRouter)
    )


def _norm(gradients):
    return math.sqrt(sum(float(g.double().square().sum()) for g in gradients))


def _collapse(model):
    with torch.no_grad():
        model.get_input_embeddings().weight.copy_(
            torch.linspace(0.1, 1, model.config.hidden_size).expand_as(
                model.get_input_embeddings().weight
            )
        )
        for parameter in _router_parameters(model):
            parameter.copy_(
                torch.tensor([0.2, 0.1, -0.1, -0.2])[:, None].expand_as(parameter)
            )


def _manual_objectives(output, batch, mean_tokens):
    mask = batch["attention_mask"].bool()
    tokens = mask.sum(1)
    experts, top_k = 4, 2
    probs = torch.stack(
        [
            logits.reshape(*mask.shape, experts).float().softmax(-1)
            for logits in output.router_logits
        ]
    )
    assignments = F.one_hot(probs.topk(top_k, dim=-1).indices, experts).sum(-2)
    counts = (assignments * mask[None, ..., None]).sum(2).permute(1, 0, 2)
    mean_probs = (probs * mask[None, ..., None]).sum((0, 2)) / (
        len(probs) * tokens[:, None]
    )
    centered = (
        (counts.double() / tokens[:, None, None] - top_k / experts)
        * (tokens[:, None, None] / mean_tokens)
    ).mean((0, 1))
    centered = centered - centered.mean()
    f_tilde = (top_k / experts + centered).clamp(0, 1).float().detach()
    aux = (
        experts
        * tokens
        / mean_tokens
        * (mean_probs * (f_tilde - top_k / experts)).sum(-1)
    )
    true_load = counts.sum((0, 1)) / (len(probs) * tokens.sum())
    pooled_probs = (probs * mask[None, ..., None]).sum((0, 1, 2)) / (
        len(probs) * tokens.sum()
    )
    switch = experts * (true_load.detach() * pooled_probs).sum()
    labels = batch["labels"][:, 1:]
    nll = F.cross_entropy(
        output.logits[:, :-1].float().transpose(1, 2), labels, reduction="none"
    )
    task = nll.double().sum(1) / labels.ne(-100).sum(1)
    return task, aux, switch, f_tilde, counts


def test_analytical_uniform_and_collapsed_assignment_shares():
    counts = torch.tensor([[10, 10, 10, 10], [20, 20, 0, 0]])
    stats = _load_statistics(counts)
    torch.testing.assert_close(stats["cv"], torch.tensor([0.0, 1.0]).double())
    torch.testing.assert_close(stats["max_to_mean"], torch.tensor([1.0, 2.0]).double())
    torch.testing.assert_close(stats["entropy"], torch.tensor([1.0, 0.5]).double())
    torch.testing.assert_close(
        stats["effective_experts"], torch.tensor([4.0, 2.0]).double()
    )
    torch.testing.assert_close(
        stats["underused_fraction"], torch.tensor([0.0, 0.5]).double()
    )


def test_pooled_expert_indices_can_be_balanced_with_collapsed_layers():
    counts = torch.tensor([[10, 0], [0, 10]])
    result = _routing_metrics(counts, [_load_statistics(counts)])
    assert result["eval_pooled_load_cv"] == 0
    assert result["eval_pooled_max_to_mean_load"] == 1
    assert result["eval_pooled_effective_experts"] == pytest.approx(2)
    assert result["eval_pooled_routing_entropy"] == pytest.approx(1)
    assert result["eval_global_load_cv_mean"] == 1
    assert result["eval_global_max_to_mean_load"] == 2
    assert result["eval_global_effective_experts_mean"] == 1
    assert result["eval_global_routing_entropy_mean"] == 0


def test_global_aggregation_cannot_hide_bad_dispatch_batches():
    first = torch.tensor([[8, 8, 0, 0], [8, 8, 0, 0]])
    second = torch.tensor([[0, 0, 8, 8], [0, 0, 8, 8]])
    result = _routing_metrics(
        first + second, [_load_statistics(first), _load_statistics(second)]
    )
    assert result["eval_global_load_cv_mean"] == 0
    assert result["eval_global_max_to_mean_load"] == 1
    assert result["eval_batch_load_cv_mean"] == 1
    assert result["eval_batch_load_cv_p95"] == 1
    assert result["eval_batch_max_to_mean_load_mean"] == 2
    assert result["eval_batch_max_to_mean_load_p95"] == 2


def test_layer_summary_quantiles_and_low_load_fraction():
    counts = torch.tensor([[10, 10, 10, 10], [20, 20, 0, 0]])
    result = _routing_metrics(counts, [_load_statistics(counts)])
    assert result["eval_global_load_cv_mean"] == 0.5
    assert result["eval_global_load_cv_worst"] == 1
    assert result["eval_global_load_cv_p90"] == pytest.approx(0.9)
    assert result["eval_global_routing_entropy_mean"] == pytest.approx(0.75)
    assert result["eval_global_effective_experts_mean"] == pytest.approx(3)
    assert result["eval_global_underused_expert_fraction"] == 0.25
    assert result["eval_layer_1_max_to_mean_load"] == 2


def test_batch_quantiles_summarize_each_batch_and_preserve_its_worst_layer():
    counts = torch.tensor([[10, 10, 10, 10], [20, 20, 0, 0]])
    result = _routing_metrics(counts, [_load_statistics(counts)])
    assert result["eval_batch_load_cv_mean"] == 0.5
    assert result["eval_batch_load_cv_p95"] == 0.5
    assert result["eval_batch_max_to_mean_load_mean"] == 2
    assert result["eval_batch_max_to_mean_load_p95"] == 2


def test_mellum2_geometry_assignment_share_and_noise_bound():
    counts = torch.zeros(28, 64)
    counts[:, :8] = 10
    stats = _load_statistics(counts)
    torch.testing.assert_close(stats["max_to_mean"], torch.full((28,), 8.0).double())
    torch.testing.assert_close(
        stats["cv"], torch.full((28,), math.sqrt(7), dtype=torch.float64)
    )
    torch.testing.assert_close(
        stats["effective_experts"], torch.full((28,), 8.0).double()
    )
    forecast = _noise_forecast(
        {"num_layers": 28, "num_experts": 64, "top_k": 8},
        mean_tokens=512,
        max_tokens=1024,
        expected_batch_size=256,
        noise_multiplier=1,
        ratio=0.02,
        filter_beta=0.99,
        steps=256,
    )
    assert forecast["load_bound"] == 28
    assert forecast["predicted_load_noise_std"] == pytest.approx(
        28 / (256 * math.sqrt(0.02))
    )


@pytest.mark.parametrize(
    "counts",
    [
        torch.zeros(2, 4),
        torch.tensor([[1.0, -1.0]]),
        torch.tensor([[1.0, float("nan")]]),
    ],
)
def test_invalid_routing_counts_are_rejected(counts):
    with pytest.raises(ValueError, match="routing counts"):
        _load_statistics(counts)


def test_public_load_reference_uses_token_weight_projection_and_clamp():
    counts = torch.tensor([[[4, 4, 0, 0]], [[2, 2, 0, 0]]])
    tokens = torch.tensor([4, 2])
    f_tilde, centered, true_load = _public_load_reference(
        counts, tokens, top_k=2, mean_tokens=2
    )
    torch.testing.assert_close(
        centered, torch.tensor([0.75, 0.75, -0.75, -0.75]).double()
    )
    torch.testing.assert_close(f_tilde, torch.tensor([1.0, 1.0, 0.0, 0.0]))
    torch.testing.assert_close(true_load, torch.tensor([1.0, 1.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="assignment count"):
        _public_load_reference(counts, tokens + 1, top_k=2, mean_tokens=2)


def test_evaluation_matches_manual_completion_loss_and_counts(
    model, records, collator, tmp_path
):
    model.eval()
    batch = collator(records)
    with torch.no_grad():
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_router_logits=True,
            router_aux_loss=False,
            use_cache=False,
        )
        labels = batch["labels"][:, 1:]
        losses = F.cross_entropy(
            output.logits[:, :-1].float().transpose(1, 2), labels, reduction="none"
        )
        valid = labels.ne(-100)
        expected_nll = losses.double().sum() / valid.sum()
        expected_example_nll = (losses.double().sum(1) / valid.sum(1)).mean()
        expected_accuracy = (
            output.logits[:, :-1].argmax(-1).eq(labels) & valid
        ).sum() / valid.sum()
        counts = _manual_objectives(output, batch, mean_tokens=6)[-1]
        pooled_counts = counts.sum((0, 1)).double()
        shares = pooled_counts / pooled_counts.sum()
        positive_shares = shares[shares > 0]
        entropy = -(positive_shares * positive_shares.log()).sum()
    path = tmp_path / "public.jsonl"
    result = evaluate_public(model, records, collator, batch_size=2, output_path=path)
    assert result["eval_nll_token_mean"] == pytest.approx(float(expected_nll), abs=1e-6)
    assert result["eval_nll_example_mean"] == pytest.approx(
        float(expected_example_nll), abs=1e-6
    )
    assert result["eval_teacher_forced_token_accuracy"] == pytest.approx(
        float(expected_accuracy)
    )
    assert result["eval_supervised_tokens"] == 7
    assert result["eval_attended_tokens"] == 14
    assert result["eval_routed_assignments"] == 14 * 2 * 2
    assert result["eval_pooled_load_cv"] == pytest.approx(
        float(shares.std(correction=0) * 4)
    )
    assert result["eval_pooled_max_to_mean_load"] == pytest.approx(
        float(shares.max() * 4)
    )
    assert result["eval_pooled_effective_experts"] == pytest.approx(
        float(entropy.exp())
    )
    assert result["eval_pooled_routing_entropy"] == pytest.approx(
        float(entropy / math.log(4))
    )
    assert result["eval_records"] == 3
    assert result["eval_batches"] == 2
    assert result["eval_batch_size"] == 2
    assert result["eval_last_batch_size"] == 1
    assert result["eval_elapsed_seconds"] > 0
    assert result["eval_attended_tokens_per_second"] == pytest.approx(
        14 / result["eval_elapsed_seconds"]
    )
    assert all(
        key.startswith("eval_") and math.isfinite(value)
        for key, value in result.items()
    )
    assert all(type(value) is float for value in result.values())
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["record_index"] for row in rows] == [0, 1, 2]
    assert [row["supervised_tokens"] for row in rows] == [3, 1, 3]
    assert sum(row["nll_sum"] for row in rows) / 7 == pytest.approx(
        result["eval_nll_token_mean"]
    )
    assert set(rows[0]) == {
        "record_index",
        "nll_sum",
        "nll",
        "supervised_tokens",
        "attended_tokens",
        "teacher_forced_correct_tokens",
    }
    assert not model.training


def test_evaluation_one_forward_per_batch_and_restores_modes(model, records, collator):
    model.train()
    model.model.layers[0].eval()
    modes = [module.training for module in model.modules()]
    calls = []

    def observe(module, args, kwargs):
        assert not torch.is_grad_enabled()
        assert not torch.is_autocast_enabled("cpu")
        assert not module.training
        assert kwargs["output_router_logits"] is True
        assert kwargs["router_aux_loss"] is False
        assert kwargs["use_cache"] is False
        assert "labels" not in kwargs
        calls.append(kwargs["input_ids"].shape[0])

    handle = model.register_forward_pre_hook(observe, with_kwargs=True)
    try:
        evaluate_public(model, records, collator, batch_size=2)
    finally:
        handle.remove()
    assert calls == [2, 1]
    assert [module.training for module in model.modules()] == modes
    assert all(parameter.grad is None for parameter in model.parameters())


def test_quality_and_global_routing_are_batching_and_padding_invariant(
    model, records, collator
):
    singles = evaluate_public(model, records, collator, batch_size=1)
    together = evaluate_public(model, records, CompletionCollator(9, 0), batch_size=3)
    stable = {
        key
        for key in singles
        if key.startswith(("eval_global_", "eval_layer_", "eval_pooled_", "eval_nll_"))
    } | {
        "eval_records",
        "eval_supervised_tokens",
        "eval_attended_tokens",
        "eval_teacher_forced_token_accuracy",
        "eval_routed_assignments",
    }
    for key in stable:
        assert singles[key] == pytest.approx(together[key], abs=1e-6)


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_invalid_evaluation_batch_size(model, records, collator, batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        evaluate_public(model, records, collator, batch_size=batch_size)


def test_empty_dataset_and_empty_shifted_targets(model, records, collator):
    with pytest.raises(ValueError, match="nonempty"):
        evaluate_public(model, [], collator)

    def empty_labels(rows):
        batch = collator(rows)
        batch["labels"].fill_(-100)
        return batch

    with pytest.raises(ValueError, match="supervised"):
        evaluate_public(model, records, empty_labels)


@pytest.mark.parametrize(
    "defect",
    [
        "padding_label",
        "nonbinary_mask",
        "label_out_of_range",
        "record_count",
        "label_shape",
    ],
)
def test_malformed_collator_outputs(model, records, collator, defect):
    def damaged(rows):
        batch = collator(rows)
        if defect == "padding_label":
            batch["labels"][0, -1] = 0
        elif defect == "nonbinary_mask":
            batch["attention_mask"][0, 0] = 2
        elif defect == "label_out_of_range":
            batch["labels"][0, 2] = model.config.vocab_size
        elif defect == "record_count":
            batch = {name: tensor[:1] for name, tensor in batch.items()}
        else:
            batch["labels"] = batch["labels"][:, :-1]
        return batch

    with pytest.raises(ValueError, match=r"padding|binary|vocabulary|mismatch"):
        evaluate_public(model, records, damaged)


def test_nonsupervised_predictions_never_enter_quality_metrics(
    model, records, collator
):
    expected = evaluate_public(model, records, collator)
    batch = collator(records)
    predicted_targets = batch["labels"][:, 1:].ne(-100)

    def corrupt_ignored(module, args, output):
        output.logits[:, :-1][~predicted_targets] = float("nan")
        output.logits[:, -1] = float("nan")
        return output

    handle = model.register_forward_hook(corrupt_ignored)
    try:
        actual = evaluate_public(model, records, collator)
    finally:
        handle.remove()
    for name in (
        "eval_nll_token_mean",
        "eval_nll_example_mean",
        "eval_teacher_forced_token_accuracy",
    ):
        assert actual[name] == expected[name]


@pytest.mark.parametrize(
    "defect", ["nonfinite_logits", "missing_layer", "wrong_tokens", "nonfinite_router"]
)
def test_output_validation_restores_mode(model, records, collator, defect):
    model.train()

    def damage(module, args, output):
        if defect == "nonfinite_logits":
            output.logits.fill_(float("nan"))
        elif defect == "missing_layer":
            output.router_logits = output.router_logits[:-1]
        elif defect == "wrong_tokens":
            output.router_logits = (
                output.router_logits[0][:-1],
                *output.router_logits[1:],
            )
        else:
            output.router_logits[0].fill_(float("inf"))
        return output

    handle = model.register_forward_hook(damage)
    try:
        with pytest.raises(ValueError, match=r"finite|router|token"):
            evaluate_public(model, records, collator)
    finally:
        handle.remove()
    assert model.training


def test_uniform_surrogate_has_zero_gradient_but_collapsed_load_does_not(
    model, records, collator
):
    _collapse(model)
    batch = collator(records[:1])
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        output_router_logits=True,
        router_aux_loss=False,
        use_cache=False,
    )
    mask = batch["attention_mask"].reshape(-1).bool()
    probs = torch.stack(
        [z.softmax(-1)[mask].mean(0) for z in output.router_logits]
    ).mean(0)
    uniform = _balancing_surrogate(
        probs, torch.full((4,), 0.5), 5 / 6, num_experts=4, top_k=2
    )
    zero = torch.autograd.grad(uniform, _router_parameters(model), retain_graph=True)
    assert _norm(zero) == 0
    collapsed = _balancing_surrogate(
        probs, torch.tensor([1.0, 1.0, 0.0, 0.0]), 5 / 6, num_experts=4, top_k=2
    )
    gradients = torch.autograd.grad(collapsed, _router_parameters(model))
    assert _norm(gradients) > 0


def test_probe_coefficient_scaling_noise_formula_and_no_mutation(
    model, records, collator
):
    _collapse(model)
    model.train()
    model.model.layers[0].eval()
    modes = [module.training for module in model.modules()]
    before = {name: value.clone() for name, value in model.state_dict().items()}
    for parameter in _router_parameters(model):
        parameter.grad = torch.ones_like(parameter)
    gradients = [parameter.grad for parameter in model.parameters()]
    rng = torch.random.get_rng_state().clone()
    with torch.no_grad():
        result = _probe(model, records, collator)
    double = _probe(model, records, collator, coefficient=0.1)
    assert result["public_only"]
    assert not result["privacy_protected"]
    assert result["gradient_scope"] == "full_trainable"
    assert result["diagnostic_batch_size"] == 1
    assert result["records"] == 3
    assert result["first_private_step_aux_grad_norm"] == 0
    assert result["router_aux_grad_norm_unscaled"] > 0
    assert result["router_aux_grad_norm_at_coefficient"] == pytest.approx(
        0.05 * result["router_aux_grad_norm_unscaled"]
    )
    assert double["router_aux_grad_norm_at_coefficient"] == pytest.approx(
        2 * result["router_aux_grad_norm_at_coefficient"]
    )
    assert double["router_task_grad_norm"] == pytest.approx(
        result["router_task_grad_norm"]
    )
    assert -1 <= result["router_task_aux_cosine"] <= 1
    assert 0 < result["full_combined_unit_clip_scale_mean"] <= 1
    sigma = 1.2 / math.sqrt(0.02) * math.sqrt(2 * 2 * (1 - 2 / 4)) / 4
    pooled = sigma / math.sqrt(2) * math.sqrt(3 / 4)
    stderr = pooled * math.sqrt(
        (1 - 0.99) / (1 + 0.99) * (1 + 0.99**256) / (1 - 0.99**256)
    )
    assert result["predicted_load_noise_std"] == pytest.approx(sigma)
    assert result["predicted_pooled_release_noise_std"] == pytest.approx(pooled)
    assert result["predicted_ema_noise_stderr"] == pytest.approx(stderr)
    assert result["predicted_ema_noise_to_signal_ratio"] == pytest.approx(
        stderr / result["public_centered_load_rms"]
    )
    assert [module.training for module in model.modules()] == modes
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    for parameter, previous in zip(model.parameters(), gradients, strict=True):
        assert parameter.grad is previous
        if previous is not None:
            assert torch.all(previous == 1)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    json.dumps(result, allow_nan=False)


def test_probe_matches_independent_unclipped_public_batch_gradients(
    model, records, collator
):
    model.eval()
    batch = collator(records)
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        output_router_logits=True,
        router_aux_loss=False,
        use_cache=False,
    )
    task, aux, switch, f_tilde, _ = _manual_objectives(output, batch, 6.0)
    routers = _router_parameters(model)
    task_grad = torch.autograd.grad(task.mean(), routers, retain_graph=True)
    aux_grad = torch.autograd.grad(aux.mean(), routers, retain_graph=True)
    switch_grad = torch.autograd.grad(switch, routers)
    result = _probe(model, records, collator)
    assert result["task_nll_mean"] == pytest.approx(float(task.mean().detach()))
    assert result["public_switch_loss"] == pytest.approx(float(switch.detach()))
    assert result["public_f_tilde"] == pytest.approx(f_tilde.tolist(), abs=1e-7)
    for name, gradients in (
        ("router_task_grad_norm", task_grad),
        ("router_aux_grad_norm_unscaled", aux_grad),
        ("router_switch_grad_norm_unscaled", switch_grad),
    ):
        assert result[name] == pytest.approx(_norm(gradients), rel=2e-5, abs=1e-10)
    dot = sum(
        float((left.double() * right.double()).sum())
        for left, right in zip(task_grad, aux_grad, strict=True)
    )
    assert result["router_task_aux_cosine"] == pytest.approx(
        dot / (_norm(task_grad) * _norm(aux_grad)), abs=1e-5
    )


def test_full_trainable_unit_clip_reference_uses_combined_per_record_gradient(
    model, records, collator
):
    _collapse(model)
    model.eval()
    batch = collator(records[:1])
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        output_router_logits=True,
        router_aux_loss=False,
        use_cache=False,
    )
    task, aux, _, _, _ = _manual_objectives(output, batch, 6.0)
    trainable = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    task_grad = torch.autograd.grad(task.mean(), trainable, retain_graph=True)
    combined_grad = torch.autograd.grad(task.mean() + 50 * aux.mean(), trainable)
    result = _probe(model, records, collator, coefficient=50, records=1)
    assert result["clip_reference_norm"] == 1
    assert result["full_task_per_record_grad_norm_mean"] == pytest.approx(
        _norm(task_grad), rel=1e-5
    )
    assert result["full_combined_per_record_grad_norm_mean"] == pytest.approx(
        _norm(combined_grad), rel=1e-5
    )
    assert result["full_combined_unit_clip_scale_mean"] == pytest.approx(
        1 / _norm(combined_grad), rel=1e-5
    )
    assert result["full_combined_unit_clip_scale_mean"] < 1


def test_load_reference_and_noise_forecast_match_existing_mechanism(
    model, records, collator
):
    model.eval()
    batch = collator(records)
    with torch.no_grad():
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_router_logits=True,
            router_aux_loss=False,
            use_cache=False,
        )
        _, _, _, reference, counts = _manual_objectives(output, batch, 2.0)
    router_logits = torch.stack(
        [logits.reshape(len(records), 6, 4) for logits in output.router_logits], dim=1
    )
    payload = {"routers": router_logits, "attention_mask": batch["attention_mask"]}
    geometry = {"num_layers": 2, "num_experts": 4, "top_k": 2}

    def loss(parameters, example):
        return (
            parameters["dummy"].square().sum(),
            tuple(example["routers"].unbind(0)),
            example["attention_mask"],
        )

    def factory(noise_multiplier):
        return moe_clipped_grad(
            loss,
            clipping_norm=1,
            normalize_by=len(records),
            noise_multiplier=noise_multiplier,
            ratio=0.02,
            key=rng_key(0),
            **geometry,
            max_tokens=6,
            mean_tokens=2,
            alpha=0.05,
            filter_beta=0.99,
        )

    clip, state = factory(0)
    _, state = clip({"dummy": torch.zeros(1)}, payload, state=state)
    calculated, _, _ = _public_load_reference(
        counts, batch["attention_mask"].sum(1), top_k=2, mean_tokens=2
    )
    torch.testing.assert_close(state.f_tilde, reference, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(calculated, state.f_tilde, atol=1e-6, rtol=1e-6)
    noisy_clip, noisy_state = factory(1.2)
    for step in range(1, 4):
        _, noisy_state = noisy_clip(
            {"dummy": torch.zeros(1)}, payload, state=noisy_state
        )
        forecast = _noise_forecast(
            geometry,
            mean_tokens=2,
            max_tokens=6,
            expected_batch_size=len(records),
            noise_multiplier=1.2,
            ratio=0.02,
            filter_beta=0.99,
            steps=step,
        )
        assert forecast["predicted_load_noise_std"] == pytest.approx(
            noisy_state.load_noise_std
        )
        assert forecast["predicted_pooled_release_noise_std"] == pytest.approx(
            noisy_state.release_noise_std
        )
        assert forecast["predicted_ema_noise_stderr"] == pytest.approx(
            noisy_state.filtered_noise_std
        )


def test_uniform_public_load_reports_zero_auxiliary_and_undefined_signal_ratio(model):
    _collapse(model)
    with torch.no_grad():
        model.get_input_embeddings().weight[4].neg_()
    data = [
        {"input_ids": [token] * 4, "completion_mask": [0, 0, 1, 1]} for token in (3, 4)
    ]
    result = _probe(model, data, CompletionCollator(4, 0), max_tokens=4, mean_tokens=4)
    assert result["public_f_tilde"] == [0.5] * 4
    assert result["public_centered_load_rms"] == 0
    assert result["router_aux_grad_norm_unscaled"] == 0
    assert result["router_task_aux_cosine"] is None
    assert result["predicted_ema_noise_to_signal_ratio"] is None
    assert result["predicted_load_noise_to_signal_ratio"] is None
    json.dumps(result, allow_nan=False)


def test_router_only_budget_fallback_keeps_actual_router_gradients(
    model, records, collator, monkeypatch
):
    full = _probe(model, records, collator, records=1)
    monkeypatch.setattr(sft_metrics, "_FULL_GRADIENT_BUDGET_BYTES", 0)
    limited = _probe(model, records, collator, records=1)
    assert limited["gradient_scope"] == "router_only"
    assert not limited["full_trainable_gradients_measured"]
    assert "full_combined_unit_clip_scale_mean" not in limited
    assert "full_task_per_record_grad_norm_mean" not in limited
    for name in (
        "router_task_grad_norm",
        "router_aux_grad_norm_unscaled",
        "router_task_aux_cosine",
        "router_switch_grad_norm_unscaled",
    ):
        assert limited[name] == pytest.approx(full[name], rel=1e-6)


def test_probe_is_bounded_to_eight_public_records_and_single_record_forwards(
    model, records, collator
):
    calls = []

    def observe(module, args, kwargs):
        assert kwargs["input_ids"].shape[0] == 1
        assert not module.training
        assert kwargs["router_aux_loss"] is False
        assert kwargs["use_cache"] is False
        calls.append(torch.is_grad_enabled())

    handle = model.register_forward_pre_hook(observe, with_kwargs=True)
    try:
        result = _probe(model, records * 4, collator)
    finally:
        handle.remove()
    assert result["records"] == 8
    assert calls == [False] * 8 + [True] * 8


def test_zero_coefficient_zero_noise_and_one_release(model, records, collator):
    _collapse(model)
    result = _probe(
        model, records, collator, coefficient=0, noise_multiplier=0, steps=1, records=1
    )
    assert result["router_aux_grad_norm_unscaled"] > 0
    assert result["router_aux_grad_norm_at_coefficient"] == 0
    assert result["router_combined_grad_norm"] == result["router_task_grad_norm"]
    assert result["predicted_load_noise_std"] == 0
    assert result["predicted_ema_noise_stderr"] == 0
    noisy = _probe(model, records, collator, steps=1, records=1)
    assert noisy["predicted_ema_noise_stderr"] == pytest.approx(
        noisy["predicted_pooled_release_noise_std"]
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"coefficient": float("nan")},
        {"coefficient": -1},
        {"mean_tokens": 0},
        {"max_tokens": 0},
        {"expected_batch_size": 0},
        {"noise_multiplier": -1},
        {"ratio": 0},
        {"filter_beta": 1},
        {"steps": 0},
        {"records": 0},
    ],
)
def test_invalid_probe_configuration(model, records, collator, overrides):
    with pytest.raises(ValueError, match=next(iter(overrides))):
        _probe(model, records, collator, **overrides)


def test_probe_requires_trainable_routers_and_respects_public_token_bound(
    model, records, collator
):
    with pytest.raises(ValueError, match="max_tokens"):
        _probe(model, records, collator, max_tokens=4)
    model.requires_grad_(False)
    with pytest.raises(ValueError, match=r"trainable.*router|router.*trainable"):
        _probe(model, records, collator)


def test_probe_rejects_empty_data_and_inference_mode(model, records, collator):
    with pytest.raises(ValueError, match="nonempty"):
        _probe(model, [], collator)
    with torch.inference_mode(), pytest.raises(ValueError, match="inference_mode"):
        _probe(model, records, collator)


@pytest.mark.cuda
def test_cuda_evaluation_and_probe(model, records, collator):
    model = model.cuda()
    results = evaluate_public(model, records, collator, batch_size=2)
    assert results["eval_supervised_tokens"] == 7
    assert all(math.isfinite(value) for value in results.values())
    probe = _probe(model, records, collator, records=1)
    assert probe["router_aux_grad_norm_unscaled"] >= 0
    assert all(parameter.grad is None for parameter in model.parameters())
