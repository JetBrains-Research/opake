"""Public-only completion evaluation and frozen-parameter balancing diagnostics.

Neither entry point protects its inputs with differential privacy. Pass only
explicitly public evaluation/diagnostic records, never private training records.
Token accuracy is teacher-forced, not program correctness. Routing skew describes
assignments, not achieved speedup or token dropping; Mellum does not drop tokens.
"""

from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers.models.mellum.modeling_mellum import MellumTopKRouter

from opaque.patches.transformers import moe_geometry

_LOSS_TOKEN_CHUNK = 128
_FULL_GRADIENT_BUDGET_BYTES = 256 * 1024**2
_MATRIX_NDIM = 2
_BATCHED_NDIM = 3
_MIN_SEQUENCE_LENGTH = 2
_MIN_EXPERTS = 2
_IGNORE_INDEX = -100
_UNDERUSED_UNIFORM_FRACTION = 0.1


def _positive_integer(value, name):
    if type(value) is not int or value <= 0:
        message = f"{name} must be a positive integer"
        raise ValueError(message)


def _finite_number(value, name, *, positive=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        bound = "positive" if positive else "nonnegative"
        message = f"{name} must be finite and {bound}"
        raise ValueError(message)


def _geometry_and_device(model):
    geometry = moe_geometry(model)
    if not 1 <= geometry["top_k"] < geometry["num_experts"]:
        message = "require 1 <= top_k < num_experts"
        raise ValueError(message)
    devices = {parameter.device for parameter in model.parameters()}
    if len(devices) != 1 or next(iter(devices)).type not in ("cpu", "cuda"):
        message = "public diagnostics require a single CPU or CUDA device"
        raise ValueError(message)
    return geometry, next(iter(devices))


@contextmanager
def _evaluation_mode(model):
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, training in modes:
            module.training = training


def _autocast(device):
    return torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    )


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _prepare_batch(rows, collator, device):
    batch = collator(rows)
    keys = ("input_ids", "attention_mask", "labels")
    if any(
        key not in batch or not isinstance(batch[key], torch.Tensor) for key in keys
    ):
        message = "collator must return input_ids, attention_mask and labels tensors"
        raise ValueError(message)
    batch = {key: batch[key].to(device) for key in keys}
    ids, attention, labels = (batch[key] for key in keys)
    if (
        ids.ndim != _MATRIX_NDIM
        or ids.shape[0] != len(rows)
        or ids.shape[1] < _MIN_SEQUENCE_LENGTH
        or attention.shape != ids.shape
        or labels.shape != ids.shape
    ):
        message = "collator record/token count mismatch"
        raise ValueError(message)
    if ids.dtype != torch.long or labels.dtype != torch.long:
        message = "input_ids and labels must be integer (torch.long) tensors"
        raise ValueError(message)
    if ((attention != 0) & (attention != 1)).any():
        message = "attention_mask must be finite and binary"
        raise ValueError(message)
    if ((labels < 0) & (labels != _IGNORE_INDEX)).any():
        message = "labels must be token IDs or -100"
        raise ValueError(message)
    if (labels.ne(_IGNORE_INDEX) & attention.eq(0)).any():
        message = "labels must mask padding with -100"
        raise ValueError(message)
    if (_target_mask(batch).sum(1) == 0).any():
        message = "every public record must contain shifted supervised tokens"
        raise ValueError(message)
    return batch


def _target_mask(batch):
    attention = batch["attention_mask"].bool()
    return (
        batch["labels"][:, 1:].ne(_IGNORE_INDEX) & attention[:, 1:] & attention[:, :-1]
    )


def _forward(model, batch):
    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        output_router_logits=True,
        router_aux_loss=False,
        use_cache=False,
        return_dict=True,
    )


def _completion_statistics(logits, batch):
    if logits.ndim != _BATCHED_NDIM or logits.shape[:2] != batch["labels"].shape:
        message = "model logits have a record/token count mismatch"
        raise ValueError(message)
    mask = _target_mask(batch)
    losses, correct = [], []
    for row in range(logits.shape[0]):
        pieces = []
        hits = torch.zeros((), dtype=torch.long, device=logits.device)
        for indices in mask[row].nonzero().flatten().split(_LOSS_TOKEN_CHUNK):
            scores = logits[row, indices].float()
            targets = batch["labels"][row, indices + 1]
            if not torch.isfinite(scores).all():
                message = "supervised logits must be finite"
                raise ValueError(message)
            if (targets >= logits.shape[-1]).any():
                message = "supervised label exceeds model vocabulary"
                raise ValueError(message)
            pieces.append(
                F.cross_entropy(scores, targets, reduction="none").double().sum()
            )
            hits = hits + scores.argmax(-1).eq(targets).sum()
        losses.append(torch.stack(pieces).sum())
        correct.append(hits)
    return torch.stack(losses), mask.sum(1), torch.stack(correct)


def _router_statistics(router_logits, attention_mask, geometry):
    layers, experts, top_k = (
        geometry[key] for key in ("num_layers", "num_experts", "top_k")
    )
    if not isinstance(router_logits, (tuple, list)) or len(router_logits) != layers:
        message = f"expected router logits for {layers} layers"
        raise ValueError(message)
    batch_size, length = attention_mask.shape
    mask = attention_mask.bool()
    tokens = mask.sum(1)
    if (tokens == 0).any():
        message = "router counts require attended tokens in every record"
        raise ValueError(message)
    row_ids = torch.arange(batch_size, device=mask.device)[:, None].expand_as(mask)[
        mask
    ]
    probability_sum = torch.zeros(batch_size, experts, device=mask.device)
    counts = []
    for logits in router_logits:
        if tuple(logits.shape) not in (
            (batch_size * length, experts),
            (batch_size, length, experts),
        ):
            message = "router token/expert count mismatch"
            raise ValueError(message)
        if not torch.isfinite(logits).all():
            message = "router logits must be finite"
            raise ValueError(message)
        probs = logits.reshape(batch_size, length, experts).float().softmax(-1)
        # Recover the executed top-k from fp32 probabilities, as Mellum does.
        indices = probs[mask].topk(top_k, dim=-1).indices
        counts.append(
            torch.bincount(
                (indices + row_ids[:, None] * experts).flatten(),
                minlength=batch_size * experts,
            ).reshape(batch_size, experts)
        )
        probability_sum = probability_sum + (probs * mask[..., None]).sum(1)
    counts = torch.stack(counts, dim=1)
    if not torch.equal(counts.sum(-1), (tokens[:, None] * top_k).expand(-1, layers)):
        message = "router assignment count mismatch"
        raise ValueError(message)
    return counts, probability_sum / (layers * tokens[:, None])


def _load_statistics(counts):
    loads = counts.detach().to(device="cpu", dtype=torch.float64)
    if (
        loads.ndim != _MATRIX_NDIM
        or loads.shape[0] == 0
        or loads.shape[1] < _MIN_EXPERTS
        or not torch.isfinite(loads).all()
        or (loads < 0).any()
        or (loads.sum(-1) <= 0).any()
    ):
        message = "routing counts must be finite, nonnegative and nonempty (L, E)"
        raise ValueError(message)
    shares = loads / loads.sum(-1, keepdim=True)
    experts = shares.shape[-1]
    entropy = -(shares * shares.clamp_min(torch.finfo(torch.float64).tiny).log()).sum(
        -1
    )
    return {
        "cv": shares.std(-1, correction=0) * experts,
        "max_to_mean": shares.max(-1).values * experts,
        "entropy": entropy / math.log(experts),
        "effective_experts": entropy.exp(),
        "underused_fraction": (shares * experts < _UNDERUSED_UNIFORM_FRACTION)
        .double()
        .mean(-1),
    }


def _routing_metrics(counts, batch_statistics):
    stats = _load_statistics(counts)
    pooled_stats = _load_statistics(counts.sum(0, keepdim=True))
    result = {
        "eval_global_load_cv_mean": float(stats["cv"].mean()),
        "eval_global_load_cv_worst": float(stats["cv"].max()),
        "eval_global_load_cv_p90": float(stats["cv"].quantile(0.9)),
        "eval_global_max_to_mean_load": float(stats["max_to_mean"].max()),
        "eval_global_routing_entropy_mean": float(stats["entropy"].mean()),
        "eval_global_effective_experts_mean": float(stats["effective_experts"].mean()),
        "eval_global_underused_expert_fraction": float(
            stats["underused_fraction"].mean()
        ),
        "eval_pooled_load_cv": float(pooled_stats["cv"][0]),
        "eval_pooled_max_to_mean_load": float(pooled_stats["max_to_mean"][0]),
        "eval_pooled_effective_experts": float(pooled_stats["effective_experts"][0]),
        "eval_pooled_routing_entropy": float(pooled_stats["entropy"][0]),
    }
    for layer in range(counts.shape[0]):
        for name, values in (
            ("load_cv", stats["cv"]),
            ("max_to_mean_load", stats["max_to_mean"]),
            ("routing_entropy", stats["entropy"]),
        ):
            result[f"eval_layer_{layer}_{name}"] = float(values[layer])
    for name, values in (
        ("load_cv", torch.stack([batch["cv"].mean() for batch in batch_statistics])),
        (
            "max_to_mean_load",
            torch.stack([batch["max_to_mean"].max() for batch in batch_statistics]),
        ),
    ):
        result[f"eval_batch_{name}_mean"] = float(values.mean())
        result[f"eval_batch_{name}_p95"] = float(values.quantile(0.95))
    return result


@torch.no_grad()
def evaluate_public(
    model, dataset, collator, *, batch_size=4, output_path: Path | None = None
) -> dict[str, float]:
    """Evaluate explicitly public records with one forward per batch.

    The collator's ``labels == -100`` identify prompts; padding is excluded by
    both labels and attention. NLL uses shifted completion labels, and accuracy
    is *teacher-forced*, not generation/code correctness. Routing includes all
    attended tokens (including prompts and final tokens), never padding.

    Global statistics first sum assignments over records separately per layer.
    CV mean/worst/p90 summarize those layers; max-to-mean is the worst layer's
    ``E * max(assignment_share)``, not token participation. Entropy is normalized
    by ``log(E)``; effective experts is ``exp(entropy)`` before normalization.
    Underused means below 0.1 times the uniform assignment target. Batch CV first
    averages layers; batch max-to-mean retains the worst layer. Their mean/p95
    weight each batch equally, including the last partial batch.

    Pooled statistics sum the global (L, E) counts across layers, then normalize
    assignment shares across E. They describe pooled expert-index balance across
    layers, not physical per-layer expert load or resource balance; collapsed
    layers can still have uniform pooled shares.

    Wall time/throughput include collation and metric reductions, not JSONL
    writing; they are evaluation cost, not a claimed routing speedup. CUDA work
    is synchronized, BF16 autocast is CUDA-only, and peak-memory counters are
    untouched. Optional JSONL contains ordinal record indices, losses and token
    counts only, no text/token IDs. Empty targets and invalid outputs raise.
    """
    _positive_integer(batch_size, "batch_size")
    if len(dataset) == 0:
        message = "public evaluation dataset must be nonempty"
        raise ValueError(message)
    geometry, device = _geometry_and_device(model)
    counts = torch.zeros(
        geometry["num_layers"], geometry["num_experts"], dtype=torch.long
    )
    batch_statistics, rows = [], []
    _synchronize(device)
    start_time = time.perf_counter()
    with _evaluation_mode(model):
        for start in range(0, len(dataset), batch_size):
            records = [
                dataset[index]
                for index in range(start, min(start + batch_size, len(dataset)))
            ]
            batch = _prepare_batch(records, collator, device)
            with _autocast(device):
                output = _forward(model, batch)
            nll, tokens, correct = _completion_statistics(output.logits, batch)
            routed, _ = _router_statistics(
                output.router_logits, batch["attention_mask"], geometry
            )
            batch_counts = routed.sum(0).cpu()
            counts = counts + batch_counts
            batch_statistics.append(_load_statistics(batch_counts))
            attended = batch["attention_mask"].sum(1).tolist()
            for offset, (loss, target_count, hits, token_count) in enumerate(
                zip(
                    nll.tolist(),
                    tokens.tolist(),
                    correct.tolist(),
                    attended,
                    strict=True,
                )
            ):
                rows.append(
                    {
                        "record_index": start + offset,
                        "nll_sum": loss,
                        "nll": loss / target_count,
                        "supervised_tokens": target_count,
                        "attended_tokens": int(token_count),
                        "teacher_forced_correct_tokens": hits,
                    }
                )
            del output
    result = _routing_metrics(counts, batch_statistics)
    _synchronize(device)
    elapsed = max(time.perf_counter() - start_time, 1e-12)
    attended_tokens = sum(row["attended_tokens"] for row in rows)
    supervised_tokens = sum(row["supervised_tokens"] for row in rows)
    result.update(
        {
            "eval_nll_sum": math.fsum(row["nll_sum"] for row in rows),
            "eval_nll_token_mean": math.fsum(row["nll_sum"] for row in rows)
            / supervised_tokens,
            "eval_nll_example_mean": math.fsum(row["nll"] for row in rows) / len(rows),
            "eval_teacher_forced_token_accuracy": sum(
                row["teacher_forced_correct_tokens"] for row in rows
            )
            / supervised_tokens,
            "eval_supervised_tokens": float(supervised_tokens),
            "eval_attended_tokens": float(attended_tokens),
            "eval_records": float(len(rows)),
            "eval_batches": float(len(batch_statistics)),
            "eval_batch_size": float(batch_size),
            "eval_last_batch_size": float(len(records)),
            "eval_router_layers": float(geometry["num_layers"]),
            "eval_experts": float(geometry["num_experts"]),
            "eval_top_k": float(geometry["top_k"]),
            "eval_routed_assignments": float(counts.sum()),
            "eval_elapsed_seconds": elapsed,
            "eval_attended_tokens_per_second": attended_tokens / elapsed,
            "eval_supervised_tokens_per_second": supervised_tokens / elapsed,
        }
    )
    if not all(math.isfinite(value) for value in result.values()):
        message = "public evaluation metrics must be finite"
        raise ValueError(message)
    if output_path is not None:
        with Path(output_path).open("w", encoding="utf-8") as stream:
            stream.writelines(json.dumps(row, allow_nan=False) + "\n" for row in rows)
    return result


def _public_load_reference(counts, tokens, *, top_k, mean_tokens):
    counts = counts.detach().to(device="cpu", dtype=torch.float64)
    tokens = tokens.detach().to(device="cpu", dtype=torch.float64)
    if (
        counts.ndim != _BATCHED_NDIM
        or min(counts.shape) < 1
        or tokens.shape != (counts.shape[0],)
        or not torch.isfinite(counts).all()
        or not torch.isfinite(tokens).all()
        or (counts < 0).any()
        or (tokens <= 0).any()
        or not torch.equal(
            counts.sum(-1), (tokens[:, None] * top_k).expand(counts.shape[:2])
        )
    ):
        message = "public router assignment count mismatch"
        raise ValueError(message)
    uniform = top_k / counts.shape[-1]
    weighted_centered = (counts - tokens[:, None, None] * uniform) / mean_tokens
    pooled = weighted_centered.mean(0).mean(0)
    centered = pooled - pooled.mean()
    f_tilde = (uniform + centered).clamp(0, 1).float()
    true_load = (counts.sum(0).mean(0) / tokens.sum()).float()
    return f_tilde, centered, true_load


def _balancing_surrogate(probs, f_tilde, token_weight, *, num_experts, top_k):
    centered = f_tilde.detach().to(probs) - top_k / num_experts
    return num_experts * token_weight * (centered * probs).sum()


def _noise_forecast(
    geometry,
    *,
    mean_tokens,
    max_tokens,
    expected_batch_size,
    noise_multiplier,
    ratio,
    filter_beta,
    steps,
):
    experts, top_k, layers = (
        geometry[key] for key in ("num_experts", "top_k", "num_layers")
    )
    bound = max_tokens / mean_tokens * math.sqrt(top_k * layers * (1 - top_k / experts))
    load_std = noise_multiplier / math.sqrt(ratio) * bound / expected_batch_size
    release_std = load_std * math.sqrt((experts - 1) / (experts * layers))
    decay = math.exp(steps * math.log(filter_beta))
    stationary = release_std * math.sqrt((1 - filter_beta) / (1 + filter_beta))
    stderr = stationary * math.sqrt(
        (1 + decay) / -math.expm1(steps * math.log(filter_beta))
    )
    result = {
        "load_bound": bound,
        "predicted_load_noise_std": load_std,
        "predicted_pooled_release_noise_std": release_std,
        "predicted_ema_noise_stderr": stderr,
        "predicted_stationary_ema_noise_stderr": stationary,
    }
    if not all(math.isfinite(value) for value in result.values()):
        message = "predicted noise scales must be finite"
        raise ValueError(message)
    return result


def _gradient_layout(model, geometry):
    trainable = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    routers = tuple(
        module.weight
        for module in model.modules()
        if isinstance(module, MellumTopKRouter)
    )
    if len(routers) != geometry["num_layers"] or not all(
        p.requires_grad for p in routers
    ):
        message = (
            "probe requires trainable Mellum routers before DPTrainer construction"
        )
        raise ValueError(message)
    trainable_bytes = sum(p.numel() * max(p.element_size(), 4) for p in trainable)
    router_bytes = sum(p.numel() * max(p.element_size(), 4) for p in routers)
    # Gradient vectors plus three CPU-fp64 router accumulators, not activations.
    estimated_bytes = 4 * trainable_bytes + 6 * router_bytes
    full = estimated_bytes <= _FULL_GRADIENT_BUDGET_BYTES
    parameters = trainable if full else routers
    positions = {id(parameter): index for index, parameter in enumerate(parameters)}
    return (
        parameters,
        [positions[id(parameter)] for parameter in routers],
        {
            "gradient_scope": "full_trainable" if full else "router_only",
            "full_trainable_gradients_measured": full,
            "trainable_parameters": sum(p.numel() for p in trainable),
            "router_parameters": sum(p.numel() for p in routers),
            "estimated_full_gradient_buffer_bytes": estimated_bytes,
            "full_gradient_budget_bytes": _FULL_GRADIENT_BUDGET_BYTES,
        },
    )


def _gradients(loss, parameters, *, retain_graph):
    if not loss.requires_grad:
        message = "probe needs eager gradients before DPTrainer construction"
        raise ValueError(message)
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=True
    )
    result = tuple(
        torch.zeros_like(p) if g is None else g.detach()
        for p, g in zip(parameters, gradients, strict=True)
    )
    if not all(torch.isfinite(gradient).all() for gradient in result):
        message = "public diagnostic gradients must be finite"
        raise ValueError(message)
    return result


def _dot(left, right):
    return math.fsum(
        float((a.double() * b.double()).sum()) for a, b in zip(left, right, strict=True)
    )


def _ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else None


def _gradient_comparison(task, aux, coefficient):
    task_squared, aux_squared, dot = _dot(task, task), _dot(aux, aux), _dot(task, aux)
    task_norm, aux_norm = math.sqrt(task_squared), math.sqrt(aux_squared)
    cosine = _ratio(dot, task_norm * aux_norm)
    return {
        "task_grad_norm": task_norm,
        "aux_grad_norm_unscaled": aux_norm,
        "aux_grad_norm_at_coefficient": coefficient * aux_norm,
        "aux_to_task_grad_ratio_at_coefficient": _ratio(
            coefficient * aux_norm, task_norm
        ),
        "task_aux_cosine": None if cosine is None else max(-1.0, min(1.0, cosine)),
        "combined_grad_norm": math.sqrt(
            max(
                0.0, task_squared + coefficient**2 * aux_squared + 2 * coefficient * dot
            )
        ),
    }


def _probe_gradients(
    model,
    rows,
    collator,
    device,
    geometry,
    parameters,
    router_indices,
    *,
    f_tilde,
    true_load,
    mean_tokens,
    mean_attended,
    coefficient,
    full,
):
    router_parameters = tuple(parameters[index] for index in router_indices)
    sums = [
        tuple(
            torch.zeros_like(p, device="cpu", dtype=torch.float64)
            for p in router_parameters
        )
        for _ in range(3)
    ]
    record_metrics = []
    experts, top_k = geometry["num_experts"], geometry["top_k"]
    for row in rows:
        batch = _prepare_batch([row], collator, device)
        with _autocast(device):
            output = _forward(model, batch)
        nll, targets, _ = _completion_statistics(output.logits, batch)
        _, probabilities = _router_statistics(
            output.router_logits, batch["attention_mask"], geometry
        )
        tokens = batch["attention_mask"].sum()
        task = nll[0] / targets[0]
        aux = _balancing_surrogate(
            probabilities[0],
            f_tilde,
            tokens / mean_tokens,
            num_experts=experts,
            top_k=top_k,
        )
        switch = (
            _balancing_surrogate(
                probabilities[0],
                true_load,
                tokens / mean_attended,
                num_experts=experts,
                top_k=top_k,
            )
            + top_k * tokens / mean_attended
        )
        task_grad = _gradients(task, parameters, retain_graph=True)
        aux_grad = _gradients(aux, parameters, retain_graph=True)
        switch_grad = _gradients(switch, router_parameters, retain_graph=False)
        router_task = tuple(task_grad[index] for index in router_indices)
        router_aux = tuple(aux_grad[index] for index in router_indices)
        router_stats = _gradient_comparison(router_task, router_aux, coefficient)
        record = {
            "task_nll_mean": float(task.detach()),
            "aux_surrogate_unscaled_mean": float(aux.detach()),
            "public_switch_loss": float(switch.detach()),
            "router_task_per_record_grad_norm_mean": router_stats["task_grad_norm"],
            "router_aux_per_record_grad_norm_unscaled_mean": router_stats[
                "aux_grad_norm_unscaled"
            ],
            "router_combined_per_record_grad_norm_mean": router_stats[
                "combined_grad_norm"
            ],
        }
        if full:
            full_stats = _gradient_comparison(task_grad, aux_grad, coefficient)
            for name, value in (
                ("task", full_stats["task_grad_norm"]),
                ("aux_unscaled", full_stats["aux_grad_norm_unscaled"]),
                ("combined", full_stats["combined_grad_norm"]),
            ):
                record[f"full_{name}_per_record_grad_norm_mean"] = value
            for name in ("task", "combined"):
                norm = full_stats[f"{name}_grad_norm"]
                record[f"full_{name}_unit_clip_scale_mean"] = (
                    min(1.0, 1 / norm) if norm > 0 else 1.0
                )
        record_metrics.append(record)
        for index, gradients in enumerate((router_task, router_aux, switch_grad)):
            sums[index] = tuple(
                total + gradient.to(device="cpu", dtype=torch.float64) / len(rows)
                for total, gradient in zip(sums[index], gradients, strict=True)
            )
        del output, task_grad, aux_grad, switch_grad, router_task, router_aux
        del nll, task, aux, switch, probabilities, gradients
    result = {
        key: math.fsum(record[key] for record in record_metrics) / len(rows)
        for key in record_metrics[0]
    }
    result.update(
        {
            f"router_{key}": value
            for key, value in _gradient_comparison(
                sums[0], sums[1], coefficient
            ).items()
        }
    )
    switch_stats = _gradient_comparison(sums[0], sums[2], coefficient)
    result.update(
        {
            "router_switch_grad_norm_unscaled": switch_stats["aux_grad_norm_unscaled"],
            "router_switch_grad_norm_at_coefficient": switch_stats[
                "aux_grad_norm_at_coefficient"
            ],
            "router_switch_to_task_grad_ratio_at_coefficient": switch_stats[
                "aux_to_task_grad_ratio_at_coefficient"
            ],
            "router_task_switch_cosine": switch_stats["task_aux_cosine"],
            "router_switch_combined_grad_norm": switch_stats["combined_grad_norm"],
            "router_aux_switch_cosine": _gradient_comparison(sums[1], sums[2], 1.0)[
                "task_aux_cosine"
            ],
            "aux_surrogate_at_coefficient_mean": coefficient
            * result["aux_surrogate_unscaled_mean"],
        }
    )
    return result


def probe_balancing_signal(
    model,
    dataset,
    collator,
    *,
    coefficient: float,
    mean_tokens: float,
    max_tokens: int,
    expected_batch_size: int,
    noise_multiplier: float,
    ratio: float,
    filter_beta: float,
    steps: int,
    records: int = 8,
) -> dict:
    """Measure an UNPROTECTED diagnostic on explicitly public records only.

    Call before DPTrainer construction, while router/adapter parameters still
    have eager gradients. At most ``records`` prefix records are used, always
    one at a time. A no-grad pilot precedes the gradient pass, with parameters
    frozen throughout both passes; no optimizer, clipping or noise is applied.
    Existing weights, .grad values, trainability and individual module modes are
    preserved. No randomness/seed state is generated or exported.

    The pilot estimates the expected-batch load mean by the public sample mean
    of ``(T_x / mean_tokens) * (h_x - k/E)``. This is not the undersized pilot sum
    divided by expected_batch_size. As in moe_clipped_grad, layer pooling and
    sum-zero projection precede ``clamp(k/E + centered, 0, 1)``. A bias-corrected
    EMA of this constant noiseless reference equals the reference at any positive
    step. Each auxiliary gradient uses exactly ``E * T_x/mean_tokens *
    <f_tilde - k/E, P_x>``. This is a frozen public reference, not an estimate of
    the realized lagged/private filter quality during training. In particular,
    the actual first private step uses uniform f_tilde and ZERO auxiliary gradient.

    Router norms/cosines compare means of per-record task/auxiliary gradients;
    separately named per-record norm means expose cancellation. ``switch`` keys
    use the true current public token/layer-pooled load and realized token total:
    the unclipped, nonprivate batch Switch utility reference, not the surrogate.
    Ratios/cosines with zero denominators are None, never misleading finite zeros.

    Full-trainable per-record norms are measured only below a conservative 256
    MiB gradient-buffer estimate (which excludes model/activation memory).
    Otherwise only routers are differentiated and full-tree clipping is NOT
    inferred. Unit clip scales are illustrative min(1, 1/norm), explicitly C=1;
    no training clipping norm is supplied to this API. Noise forecasts reproduce
    the constant-parameter release and bias-corrected EMA *pre-clamp* standard
    errors after ``steps`` releases, not empirical errors, privacy guarantees,
    or runtime utility claims. Raw per-layer and projected/pooled forecasts are
    compared against their corresponding public centered-load RMS values.
    """
    for name, value in (
        ("max_tokens", max_tokens),
        ("expected_batch_size", expected_batch_size),
        ("steps", steps),
        ("records", records),
    ):
        _positive_integer(value, name)
    for name, value in (
        ("coefficient", coefficient),
        ("noise_multiplier", noise_multiplier),
    ):
        _finite_number(value, name)
    for name, value in (
        ("mean_tokens", mean_tokens),
        ("ratio", ratio),
        ("filter_beta", filter_beta),
    ):
        _finite_number(value, name, positive=True)
    if filter_beta >= 1:
        message = "filter_beta must be in (0, 1)"
        raise ValueError(message)
    if len(dataset) == 0:
        message = "public diagnostic dataset must be nonempty"
        raise ValueError(message)
    if torch.is_inference_mode_enabled():
        message = "gradient probe cannot run in inference_mode"
        raise ValueError(message)
    geometry, device = _geometry_and_device(model)
    parameters, router_indices, layout = _gradient_layout(model, geometry)
    forecast = _noise_forecast(
        geometry,
        mean_tokens=mean_tokens,
        max_tokens=max_tokens,
        expected_batch_size=expected_batch_size,
        noise_multiplier=noise_multiplier,
        ratio=ratio,
        filter_beta=filter_beta,
        steps=steps,
    )
    rows = [dataset[index] for index in range(min(records, len(dataset)))]
    counts, tokens = [], []
    _synchronize(device)
    start_time = time.perf_counter()
    with _evaluation_mode(model), torch.enable_grad():
        with torch.no_grad():
            for row in rows:
                batch = _prepare_batch([row], collator, device)
                attended = batch["attention_mask"].sum(1)
                if (attended > max_tokens).any():
                    message = "public diagnostic record exceeds max_tokens"
                    raise ValueError(message)
                with _autocast(device):
                    output = _forward(model, batch)
                routed, _ = _router_statistics(
                    output.router_logits, batch["attention_mask"], geometry
                )
                counts.append(routed.cpu())
                tokens.append(attended.cpu())
                del output
        token_counts = torch.cat(tokens)
        pilot_counts = torch.cat(counts)
        f_tilde, centered, true_load = _public_load_reference(
            pilot_counts, token_counts, top_k=geometry["top_k"], mean_tokens=mean_tokens
        )
        mean_attended = float(token_counts.double().mean())
        gradients = _probe_gradients(
            model,
            rows,
            collator,
            device,
            geometry,
            parameters,
            router_indices,
            f_tilde=f_tilde,
            true_load=true_load,
            mean_tokens=mean_tokens,
            mean_attended=mean_attended,
            coefficient=coefficient,
            full=layout["full_trainable_gradients_measured"],
        )
    _synchronize(device)
    signal_rms = float(centered.square().mean().sqrt())
    per_layer_centered = (
        pilot_counts.double()
        - token_counts[:, None, None].double()
        * (geometry["top_k"] / geometry["num_experts"])
    ).mean(0) / mean_tokens
    per_layer_rms = float(per_layer_centered.square().mean().sqrt())
    result = {
        "public_only": True,
        "privacy_protected": False,
        "reference": "frozen_public_pilot_record_mean",
        "gradient_aggregation": "mean_of_public_per_record_gradients",
        "noise_interpretation": "predicted_constant_noise_pre_clamp_not_runtime_filter_error",
        "first_private_step_note": "uniform f_tilde=k/E gives zero auxiliary gradient",
        "first_private_step_aux_grad_norm": 0.0,
        "clip_reference_norm": 1.0,
        "records": len(rows),
        "diagnostic_batch_size": 1,
        "coefficient": coefficient,
        "mean_tokens": mean_tokens,
        "max_tokens": max_tokens,
        "expected_batch_size": expected_batch_size,
        "noise_multiplier": noise_multiplier,
        "ratio": ratio,
        "filter_beta": filter_beta,
        "steps": steps,
        **geometry,
        **layout,
        **forecast,
        **gradients,
        "public_f_tilde": f_tilde.tolist(),
        "public_mean_attended_tokens": mean_attended,
        "public_centered_load_rms": signal_rms,
        "public_per_layer_centered_load_rms": per_layer_rms,
        "public_clamped_centered_load_rms": float(
            (f_tilde.double() - geometry["top_k"] / geometry["num_experts"])
            .square()
            .mean()
            .sqrt()
        ),
        "predicted_ema_noise_to_signal_ratio": _ratio(
            forecast["predicted_ema_noise_stderr"], signal_rms
        ),
        "predicted_load_noise_to_signal_ratio": _ratio(
            forecast["predicted_load_noise_std"], per_layer_rms
        ),
        "predicted_pooled_release_noise_to_signal_ratio": _ratio(
            forecast["predicted_pooled_release_noise_std"], signal_rms
        ),
        "elapsed_seconds": time.perf_counter() - start_time,
    }
    json.dumps(result, allow_nan=False)
    return result


__all__ = ["evaluate_public", "probe_balancing_signal"]
