"""Fail-closed, synthetic correctness gate on the same real Mellum MoE.

The public keys here are for deterministic diagnostics only. They are never used
by the training runner, which samples fresh, unpublished randomness.
"""

from __future__ import annotations

import math
from dataclasses import replace

import torch
import torch.nn.functional as F
from examples.moe_privacy.run import (
    ExperimentConfig,
    GrammarDataset,
    collate,
    execution_metadata,
    make_model,
    move_batch,
    resolve_device,
    synchronize_device,
)
from torch.func import grad, vmap

from opaque.dpsgd.clipping import moe_clipped_grad
from opaque.dpsgd.noise import gaussian_noise
from opaque.functional import make_functional
from opaque.random import key

_MIN_GRADIENT_NORM = 1e-12
_MIN_NOISE_STD_RATIO = 0.9
_MAX_NOISE_STD_RATIO = 1.1
_MAX_STANDARDIZED_NOISE_MEAN = 0.03


def _require(condition: bool, message: str) -> None:
    if not condition:
        detail = f"MoE correctness gate failed: {message}"
        raise RuntimeError(detail)


def _close_trees(actual, expected) -> float:
    _require(actual.keys() == expected.keys(), "parameter coverage differs")
    largest = 0.0
    for name, value in actual.items():
        torch.testing.assert_close(
            value, expected[name], atol=2e-6, rtol=2e-4, msg=name
        )
        largest = max(largest, float((value - expected[name]).abs().max().detach()))
    return largest


def run_checks(config: ExperimentConfig, seed: int = 0, *, device: str = "cpu") -> dict:
    """Check real-model gradients, clipping, load release, and noise coverage."""
    target_device = resolve_device(device)
    torch.set_num_threads(1)
    synchronize_device(target_device)
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)
    model = make_model(config, seed, device=device)
    model.train()
    fmodel, params, frozen = make_functional(
        model, disable_autograd_tracking=True, partition_trainable=True
    )
    _require(not frozen, "some model parameters are frozen")
    router_names = [
        f"{name}.weight"
        for name, module in model.named_modules()
        if type(module).__name__ == "MellumTopKRouter"
    ]
    expert_names = [name for name in params if ".experts." in name]
    _require(len(router_names) == config.num_layers, "missing trainable Mellum routers")
    _require(
        len(expert_names) == 2 * config.num_layers, "missing stacked expert matrices"
    )
    _require(
        all(name in params for name in router_names),
        "router omitted from parameter tree",
    )
    dataset = GrammarDataset(config, 4, config.validation_seed)
    batch = collate([dataset[index] for index in range(3)])
    batch["input_ids"][2, -2:] = 0
    batch["attention_mask"][2, -2:] = 0
    batch["labels"][2, -2:] = -100
    batch = move_batch(batch, target_device)
    lagged_load = torch.linspace(0.05, 0.95, config.num_experts).to(target_device)

    def loss_and_router(parameters, row):
        output = fmodel(
            parameters,
            **{name: value.unsqueeze(0) for name, value in row.items()},
            output_router_logits=True,
            router_aux_loss=False,
        )
        routers = tuple(
            logits.reshape(-1, config.num_experts) for logits in output.router_logits
        )
        return output.loss, routers, row["attention_mask"]

    def surrogate(routers, mask, load):
        valid = mask.float()
        count = valid.sum()
        probabilities = sum(
            (logits.float().softmax(-1) * valid[:, None]).sum(0) for logits in routers
        ) / (config.num_layers * count.clamp_min(1))
        return (
            config.num_experts
            * (count / config.sequence_length)
            * torch.dot(load - config.top_k / config.num_experts, probabilities)
        )

    def objective(parameters, row, load):
        loss, routers, mask = loss_and_router(parameters, row)
        return loss + config.router_aux_loss_coef * surrogate(routers, mask, load)

    per_example = vmap(grad(objective), in_dims=(None, 0, None))(
        params, batch, lagged_load
    )
    rows = [{name: value[index] for name, value in batch.items()} for index in range(3)]
    loop = [grad(objective)(params, row, lagged_load) for row in rows]
    loop_tree = {
        name: torch.stack([result[name] for result in loop]) for name in params
    }
    loop_error = _close_trees(per_example, loop_tree)
    task_gradient = grad(lambda parameters: loss_and_router(parameters, rows[0])[0])(
        params
    )
    router_task_norm = math.sqrt(
        sum(float(task_gradient[name].square().sum()) for name in router_names)
    )
    router_aux_norm = math.sqrt(
        sum(
            float((loop[0][name] - task_gradient[name]).square().sum())
            for name in router_names
        )
    )
    expert_task_norm = math.sqrt(
        sum(float(task_gradient[name].square().sum()) for name in expert_names)
    )
    _require(
        math.isfinite(router_task_norm) and router_task_norm > _MIN_GRADIENT_NORM,
        "router task gradient is zero/non-finite",
    )
    _require(
        math.isfinite(router_aux_norm) and router_aux_norm > _MIN_GRADIENT_NORM,
        "lagged balancing gradient is zero/non-finite",
    )
    _require(
        math.isfinite(expert_task_norm) and expert_task_norm > _MIN_GRADIENT_NORM,
        "expert task gradient is zero/non-finite",
    )

    def first_record_objective(parameters, inputs):
        output = fmodel(
            parameters, **inputs, output_router_logits=True, router_aux_loss=False
        )
        task = F.cross_entropy(output.logits[0, :-1], inputs["labels"][0, 1:])
        routers = tuple(
            logits.reshape(len(inputs["input_ids"]), -1, config.num_experts)[0]
            for logits in output.router_logits
        )
        return task + config.router_aux_loss_coef * surrogate(
            routers, inputs["attention_mask"][0], lagged_load
        )

    original = grad(first_record_objective)(params, batch)
    neighbor = {name: value.clone() for name, value in batch.items()}
    for name in neighbor:
        neighbor[name][1] = dataset[3][name].to(target_device)
    changed = grad(first_record_objective)(params, neighbor)
    neighbor_error = max(
        _close_trees(original, changed), _close_trees(original, loop[0])
    )
    padded = {name: value.clone() for name, value in rows[2].items()}
    padded["input_ids"][-2:] = padded["input_ids"].new_tensor([10, 23])
    padding_error = _close_trees(grad(objective)(params, padded, lagged_load), loop[2])

    norms = sum(
        value.reshape(3, -1).square().sum(1) for value in per_example.values()
    ).sqrt()
    bound = min(config.clipping_norm, float(norms.min()) / 2)
    _require(
        math.isfinite(bound) and bound > 0,
        "non-finite or zero full-vector gradient norm",
    )

    def clipper(microbatch_size=None):
        function, state = moe_clipped_grad(
            loss_and_router,
            clipping_norm=bound,
            normalize_by=config.expected_batch_size,
            batch_argnums=1,
            noise_multiplier=1.0,
            ratio=config.load_noise_ratio,
            key=key(9000),
            top_k=config.top_k,
            num_experts=config.num_experts,
            num_layers=config.num_layers,
            max_tokens=config.sequence_length,
            mean_tokens=config.sequence_length,
            alpha=config.router_aux_loss_coef,
            filter_beta=config.filter_beta,
            microbatch_size=microbatch_size,
        )
        # Opaque keeps its load filter on CPU and copies f_tilde to the parameters.
        return function, replace(
            state, f_tilde=lagged_load.to(state.f_tilde.device).clone()
        )

    full_fn, full_state = clipper()
    full, next_full = full_fn(params, batch, state=full_state)
    scales = (bound / norms.clamp_min(1e-12)).clamp_max(1)
    manual = {
        name: (value * scales.reshape(3, *([1] * (value.ndim - 1)))).sum(0)
        / config.expected_batch_size
        for name, value in per_example.items()
    }
    clipping_error = _close_trees(full.pytree, manual)
    micro_fn, micro_state = clipper(2)
    micro, next_micro = micro_fn(params, batch, state=micro_state)
    microbatch_error = _close_trees(full.pytree, micro.pytree)
    torch.testing.assert_close(
        next_full.f_tilde, next_micro.f_tilde, atol=2e-6, rtol=2e-4
    )
    _require(
        math.isclose(full.max_norm, bound / config.expected_batch_size),
        "incorrect gradient sensitivity",
    )
    expected_load_std = math.sqrt(
        config.top_k * config.num_layers * (1 - config.top_k / config.num_experts)
    )
    expected_load_std /= math.sqrt(config.load_noise_ratio) * config.expected_batch_size
    _require(
        math.isclose(full_state.load_noise_std, expected_load_std),
        "incorrect load sensitivity/noise",
    )

    noise_fn, noise_state = gaussian_noise(noise_multiplier=1.0, key=key(9001))
    noised, _ = noise_fn(full, noise_state)
    _require(noised.pytree.keys() == params.keys(), "noise omits trainable parameters")
    residual = torch.cat(
        [(noised.pytree[name] - full.pytree[name]).flatten() for name in params]
    )
    standardized = residual / (bound / config.expected_batch_size)
    noise_std_ratio = float(standardized.std())
    _require(
        _MIN_NOISE_STD_RATIO < noise_std_ratio < _MAX_NOISE_STD_RATIO,
        "gradient noise scale differs from accounted scale",
    )
    _require(
        abs(float(standardized.mean())) < _MAX_STANDARDIZED_NOISE_MEAN,
        "gradient noise is not centered",
    )

    # Equal logits force a fixed top-k subset, leaving E-k experts entirely unused.
    collapsed = {name: value.detach().clone() for name, value in params.items()}
    for name in router_names:
        collapsed[name].zero_()
    forced, _ = full_fn(collapsed, batch, state=full_state)
    forced_noised, _ = noise_fn(forced, noise_state)
    _, routers, _ = loss_and_router(collapsed, rows[0])
    unused_matrices = 0
    for layer, router_name in enumerate(router_names):
        selected = (
            routers[layer]
            .float()
            .softmax(-1)
            .topk(config.top_k, dim=-1)
            .indices.unique()
        )
        unused = sorted(set(range(config.num_experts)) - set(selected.tolist()))
        _require(
            len(unused) == config.num_experts - config.top_k,
            "failed to create unused experts",
        )
        prefix = router_name.removesuffix(".gate.weight") + ".experts."
        for name in expert_names:
            if not name.startswith(prefix):
                continue
            _require(
                not bool(forced.pytree[name][unused].count_nonzero()),
                "unused expert has a task gradient",
            )
            _require(
                bool(
                    (
                        forced_noised.pytree[name][unused].flatten(1).norm(dim=1) > 0
                    ).all()
                ),
                "unused expert was not noised",
            )
            unused_matrices += len(unused)
    _require(
        unused_matrices == 2 * config.num_layers * (config.num_experts - config.top_k),
        "unused-expert coverage is incomplete",
    )

    empty = {name: value[:0] for name, value in batch.items()}
    empty_clipped, empty_state = full_fn(params, empty, state=full_state)
    _require(
        all(not bool(value.count_nonzero()) for value in empty_clipped.pytree.values()),
        "empty batch produced a data gradient",
    )
    _require(
        empty_state.step == full_state.step + 1,
        "empty batch did not advance load release",
    )
    _require(
        not torch.equal(empty_state.f_tilde, full_state.f_tilde),
        "empty batch did not release noisy load",
    )
    empty_noised, _ = noise_fn(empty_clipped, noise_state)
    _require(
        all(bool(value.count_nonzero()) for value in empty_noised.pytree.values()),
        "empty batch skipped gradient noise",
    )
    _require(
        all(
            value.device == target_device
            for tree in (
                params,
                per_example,
                full.pytree,
                micro.pytree,
                noised.pytree,
                forced_noised.pytree,
                empty_clipped.pytree,
                empty_noised.pytree,
            )
            for value in tree.values()
        ),
        "model, gradients, clipping and noise must use the requested device",
    )

    return {
        "passed": True,
        "model": "MellumForCausalLM",
        "execution": execution_metadata(next(iter(params.values())).device),
        "parameter_tensors": len(params),
        "router_parameters": router_names,
        "expert_parameters": expert_names,
        "vmap_loop_max_abs_error": loop_error,
        "neighbor_independence_max_abs_error": neighbor_error,
        "padding_max_abs_error": padding_error,
        "global_clipping_max_abs_error": clipping_error,
        "microbatch_max_abs_error": microbatch_error,
        "router_task_gradient_norm": router_task_norm,
        "router_aux_gradient_norm": router_aux_norm,
        "expert_task_gradient_norm": expert_task_norm,
        "tested_clipping_norm": bound,
        "expected_gradient_noise_std": bound / config.expected_batch_size,
        "expected_load_noise_std": expected_load_std,
        "gradient_noise_std_ratio": noise_std_ratio,
        "unused_expert_matrices_noised": unused_matrices,
        "empty_batch_releases_verified": True,
        "scope": "synthetic implementation checks, not a formal differential privacy proof",
    }
