"""Greedy public code generation and Docker-only EvalPlus 0.3.1 evaluation.

Run ``python -m examples.moe_privacy.code_eval --help`` for the two commands.
Call ``generate_samples`` with an already loaded model/tokenizer to reuse GPU
memory; it never loads a second model. Supply the uncompressed, unmodified
official JSONL releases listed in BENCHMARKS. No downloads happen in evaluation.
After all training and adapter export, ``merge_for_evaluation(model)`` optionally
materializes static LoRA weights once, irreversibly freezing the inference model.

Prompts are used verbatim, without chat templates, examples, or whitespace
normalization. The fixed entry_point_top_level_v1 policy stops at the first
top-level definition, class, import, assertion, print, module guard, decorator,
or fence after the public target function. It waits for that target in either
the prompt (HumanEval) or completion (MBPP); it never consults expected answers
or tests. Markers inside strings, brackets, and nested code are ignored. Like
EvalPlus direct-completion stops, this can exclude helpers/imports emitted after
the target; no correctness-dependent exception or repair is made.
Only the prefix preceding that marker is graded, without sanitization. Emitted
text including the marker is preserved in raw_samples.jsonl; token counts,
latency, and token-limit flags describe actual generation, not the graded prefix.
Use identical budgets and this policy for all arms. A smoke limit selects the
first numeric task IDs and is never a full benchmark score.

Build Dockerfile.eval separately, then provide that trusted image by digest or
local reference (resolved to its inspected immutable ID, never pulled). Docker
can explicitly use ``--docker-sudo`` (``sudo -n docker``); this never grants the
container privileges or changes its unprivileged UID.
Docker is a host-isolation boundary, not a proof against malicious benchmark cheating
or kernel exploits; use a dedicated sandbox host. The execution worker exists
only inside that image, not in this host module. Ground-truth code is untrusted
too. Never invoke evalplus.evaluate on the host.

Verified interfaces: github.com/evalplus/evalplus/tree/v0.3.1, specifically
evaluate.py, data/{humaneval,mbpp}.py, provider/{utility,hf}.py, and eval/__init__.py.
The stopping markers follow its direct-completion conventions with an explicit
target guard and lexical nesting checks, not its chat prompt or sanitizer.
EvalPlus's timeout
status measures whole-task process timeouts; individual test timeouts can be
reported as fail. Wilson intervals describe the task pass fraction, not training
seed variability. A container-level timeout is an evaluation failure, not a score.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import stat
import subprocess
import time
import tokenize
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

EVALPLUS_VERSION = "0.3.1"
GENERATION_FORMAT_VERSION = 2
MODEL_ID = "JetBrains/Mellum2-12B-A2.5B-Base"
MODEL_REVISION = "271755e48ab6b2ed0ef224eaaabe2d25275fb8ee"
LORA_RANK, LORA_ALPHA = 4, 8
BENCHMARKS = {
    "humaneval": {"version": "v0.1.10", "count": 164, "prefix": "HumanEval"},
    "mbpp": {"version": "v0.2.0", "count": 378, "prefix": "Mbpp"},
}
IMAGE_LABELS = {
    "org.opaque.code-eval.protocol": "1",
    "org.opaque.code-eval.evalplus": EVALPLUS_VERSION,
}
IMAGE_ENTRYPOINT = ["python", "-I", "/opt/evaluate.py"]
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_CONTAINER_NAME = re.compile(r"opaque-code-eval-[0-9a-f]{32}\Z")
_COMPLETION_BOUNDARY = re.compile(
    r"(?:async[ \t]+def[ \t]+|def[ \t]+|class[ \t]+|import[ \t]+|from[ \t]+"
    r"|assert[ \t]+|print[ \t]*\(|if[ \t]+__name__[ \t]*(?:==|:)|@|```)"
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _md5(data: bytes) -> str:
    # EvalPlus identifies the exact uncompressed dataset bytes with MD5.
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        message = f"Cannot read JSON artifact {path}: {exc}"
        raise ValueError(message) from exc


def _jsonl(data: bytes) -> list[dict]:
    try:
        rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    except ValueError as exc:
        message = "Expected uncompressed UTF-8 JSONL"
        raise ValueError(message) from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        message = "JSONL must contain nonempty object records"
        raise ValueError(message)
    return rows


def _indexed(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for row in rows:
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or task_id in result:
            message = f"Missing or duplicate task_id: {task_id!r}"
            raise ValueError(message)
        result[task_id] = row
    return result


def load_benchmark(dataset: str, path: str | Path) -> tuple[dict, dict]:
    """Read public data as inert JSON, checking the entire declared task set."""
    if dataset not in BENCHMARKS:
        message = f"Unknown benchmark: {dataset}"
        raise ValueError(message)
    spec = BENCHMARKS[dataset]
    raw = Path(path).read_bytes()
    tasks = _indexed(_jsonl(raw))
    if len(tasks) != spec["count"]:
        message = f"{dataset} requires all {spec['count']} tasks, got {len(tasks)}"
        raise ValueError(message)
    for task_id, task in tasks.items():
        if not re.fullmatch(rf"{spec['prefix']}/(?:0|[1-9][0-9]*)", task_id):
            message = f"Invalid {dataset} task ID: {task_id}"
            raise ValueError(message)
        for field in ("prompt", "canonical_solution", "entry_point", "contract"):
            if not isinstance(task.get(field), str):
                message = f"{task_id}: missing string field {field}"
                raise ValueError(message)
        if not task["prompt"] or not task["entry_point"].isidentifier():
            message = f"{task_id}: invalid prompt or entry point"
            raise ValueError(message)
        for field in ("base_input", "plus_input"):
            # MBPP+ v0.2.0 stores Mbpp/793's zero extended tests as {}.
            if (
                dataset == "mbpp"
                and field == "plus_input"
                and isinstance(task.get(field), (list, dict))
                and not task[field]
            ):
                continue
            if not isinstance(task.get(field), list) or not task[field]:
                message = f"{task_id}: missing nonempty {field}"
                raise ValueError(message)
        atol = task.get("atol")
        if type(atol) not in (int, float) or not math.isfinite(atol) or atol < 0:
            message = f"{task_id}: invalid atol"
            raise ValueError(message)
    if dataset == "humaneval" and set(tasks) != {
        f"HumanEval/{i}" for i in range(spec["count"])
    }:
        message = "HumanEval IDs must cover exactly 0..163"
        raise ValueError(message)
    tasks = dict(sorted(tasks.items(), key=lambda item: int(item[0].split("/")[1])))
    release = "HumanEvalPlus" if dataset == "humaneval" else "MbppPlus"
    with_extended = sum(bool(task["plus_input"]) for task in tasks.values())
    return tasks, {
        "dataset": dataset,
        "version": spec["version"],
        "declared_count": spec["count"],
        "tasks_with_extended_tests": with_extended,
        "tasks_without_extended_tests": len(tasks) - with_extended,
        "sha256": _sha256(raw),
        "evalplus_md5": _md5(raw),
        "source_url": (
            f"https://github.com/evalplus/{release.lower()}_release/releases/"
            f"download/{spec['version']}/{release}.jsonl.gz"
        ),
    }


def generation_settings(max_new_tokens: int = 512) -> dict:
    """The public, identical base-completion protocol for every arm."""
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        message = "max_new_tokens must be a positive integer"
        raise ValueError(message)
    return {
        "max_new_tokens": max_new_tokens,
        "batch_size": 1,
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "use_cache": True,
        "prompt_format": "canonical_verbatim",
        "add_special_tokens": False,
        "stop": "function_boundary_or_eos_or_token_budget",
        "stop_policy": "entry_point_top_level_v1",
        "stop_pattern": _COMPLETION_BOUNDARY.pattern,
        "postprocessing": "prefix_before_boundary_no_sanitizer",
        "attn_implementation": "eager",
        "autocast": False,
        "router_fp32": True,
        "router_aux_loss": False,
    }


def _selected_ids(tasks: dict, smoke_limit: int | None) -> list[str]:
    if smoke_limit is None:
        return list(tasks)
    if type(smoke_limit) is not int or not 1 <= smoke_limit <= len(tasks):
        message = "smoke_limit must be between 1 and the full task count"
        raise ValueError(message)
    return list(tasks)[:smoke_limit]


def completion_boundary(prompt: str, completion: str, entry_point: str) -> dict | None:
    """Locate a fixed lexical boundary; never compile, execute, or repair Python.

    This recognizes the first public target definition before permitting a
    stop, including when MBPP supplies only a description rather than a header.
    Invalid or incomplete target code stays invalid: only a suffix is removed.
    """
    source = prompt + completion
    offsets, offset = [], 0
    for line in source.split("\n"):
        offsets.append(offset)
        offset += len(line) + 1
    target = re.compile(
        r"(?:async[ \t]+)?def[ \t]+" + re.escape(entry_point) + r"[ \t]*\("
    )
    seen_target = False
    indentation = brackets = 0
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.INDENT:
                indentation += 1
            elif token.type == tokenize.DEDENT:
                indentation -= 1
            elif indentation == brackets == token.start[1] == 0 and token.type in (
                tokenize.NAME,
                tokenize.OP,
                tokenize.ERRORTOKEN,
            ):
                start = offsets[token.start[0] - 1]
                if not seen_target and target.match(source, start):
                    seen_target = True
                elif seen_target and start >= len(prompt):
                    marker = _COMPLETION_BOUNDARY.match(source, start)
                    if marker:
                        return {
                            "completion_offset": start - len(prompt),
                            "marker": marker.group(),
                        }
            if token.type == tokenize.OP:
                if token.string in ("(", "[", "{"):
                    brackets += 1
                elif token.string in (")", "]", "}"):
                    brackets -= 1
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # Incremental generations routinely end inside a string or expression.
        # Preserve such text; syntax validity is checked only in the sandbox.
        return None
    return None


class _FunctionCompletionStop:
    """Batch-one callable for Transformers' StoppingCriteriaList, imported lazily."""

    def __init__(self, tokenizer, prompt: str, entry_point: str, prompt_tokens: int):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.entry_point = entry_point
        self.prompt_tokens = prompt_tokens
        self.boundary = None

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[0] != 1:
            message = "Function-completion stopping requires batch size one"
            raise ValueError(message)
        completion = self.tokenizer.decode(
            input_ids[0, self.prompt_tokens :].tolist(),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        self.boundary = completion_boundary(self.prompt, completion, self.entry_point)
        return self.boundary is not None


def decode_completion(
    tokenizer,
    token_ids: list[int],
    eos_ids: list[int],
    budget: int,
    *,
    boundary: dict | None = None,
):
    """Remove only the terminating EOS, preserving the generated Python text."""
    if not token_ids or len(token_ids) > budget:
        message = "Invalid generated token count"
        raise ValueError(message)
    stop = next((i for i, token in enumerate(token_ids) if token in eos_ids), None)
    if stop is None and len(token_ids) != budget and boundary is None:
        message = "Generation stopped before EOS, a function boundary, or the public token budget"
        raise ValueError(message)
    text = tokenizer.decode(
        token_ids if stop is None else token_ids[:stop],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return text, {
        "generated_tokens": len(token_ids) if stop is None else stop + 1,
        "token_limit_truncated": stop is None and len(token_ids) == budget,
        "stop_reason": (
            "function_boundary"
            if boundary is not None
            else "token_limit"
            if stop is None
            else "eos"
        ),
    }


def read_adapter_spec(export_dir: str | Path) -> dict:
    """Preflight a static r4/alpha8 export before loading the large backbone."""
    export_dir = Path(export_dir)
    spec = _read_json(export_dir / "adapter_spec.json")
    if not isinstance(spec, dict):
        message = "adapter_spec.json must be an object"
        raise ValueError(message)
    config = spec.get("config")
    if (
        type(spec.get("format_version")) is not int
        or spec["format_version"] != 1
        or spec.get("adapter_kind") != "static_parametrization_expert_lora"
        or not isinstance(config, dict)
        or type(config.get("rank")) is not int
        or config["rank"] != LORA_RANK
        or type(config.get("alpha")) not in (int, float)
        or config["alpha"] != LORA_ALPHA
        or not isinstance(spec.get("metadata"), dict)
        or not isinstance(spec.get("tensors"), dict)
        or not spec["tensors"]
    ):
        message = "Require a format-1 static expert-LoRA r4/alpha8 export"
        raise ValueError(message)
    for name, tensor in spec["tensors"].items():
        if (
            not isinstance(name, str)
            or not isinstance(tensor, dict)
            or tensor.get("dtype") != "torch.float32"
            or not isinstance(tensor.get("shape"), list)
            or not tensor["shape"]
            or any(type(size) is not int or size <= 0 for size in tensor["shape"])
        ):
            message = f"Malformed trainable tensor specification: {name}"
            raise ValueError(message)
    if not (export_dir / "trainable.safetensors").is_file():
        message = "Export is missing trainable.safetensors"
        raise ValueError(message)
    return spec


def restore_adapters(model, export_dir: str | Path):
    """Reconstruct and validate all saved adapter/router tensors with the helper."""
    from examples.moe_privacy.sft_adapters import (
        configure_expert_lora,
        load_trainable,
    )

    spec = read_adapter_spec(export_dir)
    model, metadata = configure_expert_lora(
        model, rank=spec["config"]["rank"], alpha=spec["config"]["alpha"]
    )
    if _json_bytes(spec["metadata"]) != _json_bytes(metadata):
        message = "Export metadata does not match the reconstructed model specification"
        raise ValueError(message)
    load_trainable(model, export_dir)
    return model, metadata


def _prepare_model(model) -> dict:
    import torch
    from torch.nn.utils import parametrize
    from transformers.models.mellum.modeling_mellum import MellumTopKRouter

    from opaque.patches import apply_model_patches, apply_runtime_patches

    if getattr(model.config, "model_type", None) != "mellum":
        message = "This evaluation protocol requires a Mellum model"
        raise ValueError(message)
    apply_runtime_patches(performance=False)
    apply_model_patches(
        model, performance=False, peft=False, router_fp32=True, router_aux_loss=False
    )
    model.set_attn_implementation("eager")
    routers = [m for m in model.modules() if isinstance(m, MellumTopKRouter)]
    if not routers:
        message = "Model has no Mellum routers"
        raise ValueError(message)
    for router in routers:
        router.to(dtype=torch.float32)
    config = getattr(model, "_expert_lora_config", None)
    merged = getattr(model, "_code_eval_merged", False)
    if merged and config is None:
        message = "Merged inference requires its original adapter configuration"
        raise ValueError(message)
    if config is not None:
        if (
            not isinstance(config, dict)
            or config.get("rank") != LORA_RANK
            or config.get("alpha") != LORA_ALPHA
        ):
            message = "Require static expert LoRA r4/alpha8 for all adapted arms"
            raise ValueError(message)
        expected = {
            f"{name}.weight"
            for name, module in model.named_modules()
            if isinstance(module, MellumTopKRouter)
        } | {
            f"{name}.parametrizations.{parameter}.0.{factor}"
            for target in config["target_parameters"]
            for name, parameter in [target.rsplit(".", 1)]
            for factor in ("lora_A", "lora_B")
        }
        trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
        if merged:
            if trainable or any(
                parametrize.is_parametrized(module) for module in model.modules()
            ):
                message = "Merged inference must stay frozen, without parametrizations"
                raise ValueError(message)
        elif set(trainable) != expected or any(
            p.dtype != torch.float32 for p in trainable.values()
        ):
            message = "Trainability must be exactly FP32 expert LoRA and routers"
            raise ValueError(message)
    return {
        "adapter_config": config,
        "router_count": len(routers),
        "merged_for_evaluation": merged,
    }


def merge_for_evaluation(model):
    """Merge static LoRA in place, once; return the frozen, eval-mode model.

    Irreversible: finish training and export adapters BEFORE calling this. Do not
    train or export adapters from the returned model. Only one expert tensor is
    materialized at a time, using the same non-autocast arithmetic as generation.
    """
    import torch
    from torch.nn.utils import parametrize

    info = _prepare_model(model)
    if info["adapter_config"] is None or info["merged_for_evaluation"]:
        message = "Require unmerged static expert LoRA; merging is inference-only"
        raise ValueError(message)
    targets = []
    for target in info["adapter_config"]["target_parameters"]:
        name, parameter = target.rsplit(".", 1)
        module = model.get_submodule(name)
        if not parametrize.is_parametrized(module, parameter):
            message = f"Missing static LoRA parametrization: {target}"
            raise ValueError(message)
        targets.append((module, parameter))
    device = model.get_input_embeddings().weight.device
    with torch.no_grad(), torch.autocast(device.type, enabled=False):
        for module, parameter in targets:
            parametrize.remove_parametrizations(
                module, parameter, leave_parametrized=True
            )
    model.requires_grad_(False)
    model.eval()
    model._code_eval_merged = True
    return model


def _sync(device):
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def generate_samples(
    model,
    tokenizer,
    *,
    dataset: str,
    dataset_path: str | Path,
    output_dir: str | Path,
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
    max_new_tokens: int = 512,
    smoke_limit: int | None = None,
    arm: str | None = None,
) -> dict:
    """Generate one greedy completion/task, with lexical stopping only.

    The output directory must not exist. ``samples.jsonl`` is canonical EvalPlus
    input; ``raw_samples.jsonl`` preserves the complete emitted text, including
    any stop marker. ``generation.json`` binds both hashes, prompts, and tasks
    to the public protocol. Failure leaves an incomplete directory, not a score.
    Eval mode and router-output config are restored; compatibility patches and
    FP32 router storage remain installed. No RNG is seeded, saved, or published.
    """
    import torch
    from transformers import GenerationConfig, StoppingCriteriaList

    if (
        not isinstance(model_id, str)
        or not model_id
        or not isinstance(model_revision, str)
        or not _REVISION.fullmatch(model_revision)
    ):
        message = "Specify model_id and an immutable 40-hex model_revision"
        raise ValueError(message)
    if arm is not None and (not isinstance(arm, str) or not arm.strip()):
        message = "arm must be a nonempty public string label"
        raise ValueError(message)
    resolved = getattr(model.config, "_commit_hash", None)
    if resolved is not None and resolved != model_revision:
        message = "Loaded model revision differs from the declared revision"
        raise ValueError(message)
    tasks, benchmark = load_benchmark(dataset, dataset_path)
    ids = _selected_ids(tasks, smoke_limit)
    settings = generation_settings(max_new_tokens)
    model_info = _prepare_model(model)
    arm = arm or ("base" if model_info["adapter_config"] is None else "adapter")
    device = model.get_input_embeddings().weight.device
    if device.type == "meta" or any(p.device != device for p in model.parameters()):
        message = "Generation requires a materialized, single-device model"
        raise ValueError(message)
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    if (
        not isinstance(eos, list)
        or not eos
        or any(type(token) is not int or token < 0 for token in eos)
    ):
        message = "A valid EOS token ID is required"
        raise ValueError(message)
    pad = tokenizer.pad_token_id
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        num_return_sequences=1,
        use_cache=True,
        eos_token_id=eos,
        pad_token_id=eos[0] if pad is None else pad,
        bos_token_id=tokenizer.bos_token_id,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    samples_path = output_dir / "samples.jsonl"
    raw_samples_path = output_dir / "raw_samples.jsonl"
    records = []
    was_training = model.training
    old_router_outputs = model.config.output_router_logits
    old_aux_coef = model.config.router_aux_loss_coef
    model.eval()
    model.config.output_router_logits = False
    model.config.router_aux_loss_coef = 0.0
    try:
        with (
            samples_path.open("xb") as output,
            raw_samples_path.open("xb") as raw_output,
            torch.inference_mode(),
            torch.autocast(device.type, enabled=False),
        ):
            for task_id in ids:
                prompt = tasks[task_id]["prompt"]
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=False,
                    return_token_type_ids=False,
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                prompt_tokens = inputs["input_ids"].shape[1]
                if (
                    prompt_tokens + max_new_tokens
                    > model.config.max_position_embeddings
                ):
                    message = f"{task_id}: prompt + budget exceeds model context"
                    raise ValueError(message)
                stopper = _FunctionCompletionStop(
                    tokenizer, prompt, tasks[task_id]["entry_point"], prompt_tokens
                )
                _sync(device)
                start = time.perf_counter()
                generated = model.generate(
                    **inputs,
                    generation_config=generation_config,
                    stopping_criteria=StoppingCriteriaList([stopper]),
                )
                _sync(device)
                elapsed = time.perf_counter() - start
                if generated.shape[0] != 1 or not torch.equal(
                    generated[0, :prompt_tokens], inputs["input_ids"][0]
                ):
                    message = "Generation must return one prompt-prefixed sequence"
                    raise ValueError(message)
                completion, stop = decode_completion(
                    tokenizer,
                    generated[0, prompt_tokens:].tolist(),
                    eos,
                    max_new_tokens,
                    boundary=stopper.boundary,
                )
                boundary = completion_boundary(
                    prompt, completion, tasks[task_id]["entry_point"]
                )
                if boundary != stopper.boundary:
                    message = "Final text differs from the observed stopping boundary"
                    raise ValueError(message)
                raw_solution = prompt + completion
                solution = prompt + (
                    completion[: boundary["completion_offset"]]
                    if boundary is not None
                    else completion
                )
                output.write(_json_bytes({"task_id": task_id, "solution": solution}))
                raw_output.write(
                    _json_bytes({"task_id": task_id, "solution": raw_solution})
                )
                output.flush()
                raw_output.flush()
                records.append(
                    {
                        "task_id": task_id,
                        "prompt_sha256": _sha256(prompt.encode()),
                        "solution_sha256": _sha256(solution.encode()),
                        "raw_solution_sha256": _sha256(raw_solution.encode()),
                        "completion_boundary": boundary,
                        "prompt_tokens": prompt_tokens,
                        "latency_seconds": elapsed,
                        **stop,
                    }
                )
    finally:
        model.config.output_router_logits = old_router_outputs
        model.config.router_aux_loss_coef = old_aux_coef
        model.train(was_training)
    manifest = {
        "format_version": GENERATION_FORMAT_VERSION,
        "evalplus_version": EVALPLUS_VERSION,
        "mode": "full" if smoke_limit is None else "smoke",
        "smoke_limit": smoke_limit,
        "benchmark": benchmark,
        "model": {"id": model_id, "revision": model_revision, "arm": arm, **model_info},
        "generation": {**settings, "eos_token_ids": eos},
        "task_ids": ids,
        "tasks": records,
        "samples_sha256": _sha256(samples_path.read_bytes()),
        "raw_samples_sha256": _sha256(raw_samples_path.read_bytes()),
    }
    (output_dir / "generation.json").write_bytes(_json_bytes(manifest))
    return manifest


def validate_generation(
    tasks: dict,
    benchmark: dict,
    samples: bytes,
    manifest: dict,
    *,
    raw_samples: bytes | None = None,
):
    """Reject incomplete, altered, mismatched, or ambiguously labeled artifacts."""
    try:
        ids = _selected_ids(tasks, manifest["smoke_limit"])
        mode = "full" if manifest["smoke_limit"] is None else "smoke"
        settings = dict(manifest["generation"])
        eos = settings.pop("eos_token_ids")
        expected_settings = generation_settings(settings["max_new_tokens"])
        if (
            manifest["format_version"] != GENERATION_FORMAT_VERSION
            or manifest["evalplus_version"] != EVALPLUS_VERSION
            or manifest["benchmark"] != benchmark
            or manifest["mode"] != mode
            or manifest["task_ids"] != ids
            or manifest["samples_sha256"] != _sha256(samples)
            or raw_samples is None
            or manifest["raw_samples_sha256"] != _sha256(raw_samples)
            or settings != expected_settings
            or not isinstance(eos, list)
            or not eos
            or any(type(token) is not int or token < 0 for token in eos)
            or not _REVISION.fullmatch(manifest["model"]["revision"])
        ):
            message = "Generation manifest differs from the public protocol/data"
            raise ValueError(message)
        solutions = _indexed(_jsonl(samples))
        raw_solutions = _indexed(_jsonl(raw_samples))
        records = _indexed(manifest["tasks"])
        if list(solutions) != ids or list(records) != ids or list(raw_solutions) != ids:
            message = "Samples and generation metadata must align with every task"
            raise ValueError(message)
        for task_id in ids:
            sample, record = solutions[task_id], records[task_id]
            solution = sample["solution"]
            prompt = tasks[task_id]["prompt"]
            raw_sample = raw_solutions[task_id]
            raw_solution = raw_sample["solution"]
            if (
                set(raw_sample) != {"task_id", "solution"}
                or not isinstance(raw_solution, str)
                or not raw_solution.startswith(prompt)
                or record["raw_solution_sha256"] != _sha256(raw_solution.encode())
            ):
                message = f"Invalid raw generation metadata for {task_id}"
                raise ValueError(message)
            boundary = completion_boundary(
                prompt, raw_solution[len(prompt) :], tasks[task_id]["entry_point"]
            )
            expected_solution = (
                raw_solution[: len(prompt) + boundary["completion_offset"]]
                if boundary is not None
                else raw_solution
            )
            count = record["generated_tokens"]
            truncated = record["token_limit_truncated"]
            reason = record["stop_reason"]
            if (
                set(sample) != {"task_id", "solution"}
                or not isinstance(solution, str)
                or not solution.startswith(prompt)
                or solution != expected_solution
                or record["completion_boundary"] != boundary
                or record["prompt_sha256"] != _sha256(prompt.encode())
                or record["solution_sha256"] != _sha256(solution.encode())
                or type(count) is not int
                or not 1 <= count <= settings["max_new_tokens"]
                or type(truncated) is not bool
                or reason
                != (
                    "function_boundary"
                    if boundary is not None
                    else "token_limit"
                    if truncated
                    else "eos"
                )
                or truncated
                != (count == settings["max_new_tokens"] and reason != "eos")
                or (truncated and count != settings["max_new_tokens"])
                or type(record["prompt_tokens"]) is not int
                or record["prompt_tokens"] <= 0
                or type(record["latency_seconds"]) not in (int, float)
                or not math.isfinite(record["latency_seconds"])
                or record["latency_seconds"] < 0
            ):
                message = f"Invalid generation metadata for {task_id}"
                raise ValueError(message)
    except (KeyError, TypeError, AttributeError) as exc:
        message = "Malformed generation manifest or samples"
        raise ValueError(message) from exc
    return solutions, records


def wilson_interval(passed: int, total: int) -> list[float]:
    """Two-sided 95% Wilson score interval for a task pass fraction."""
    if (
        type(passed) is not int
        or type(total) is not int
        or total <= 0
        or not 0 <= passed <= total
    ):
        message = "Require integer counts 0 <= passed <= total, total > 0"
        raise ValueError(message)
    z = 1.959963984540054
    fraction = passed / total
    denominator = 1 + z * z / total
    center = (fraction + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(fraction * (1 - fraction) / total + z * z / (4 * total**2))
    radius /= denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def summarize_results(
    results: dict, solutions: dict, records: dict, syntax: dict, *, dataset_hash: str
):
    """Parse the inspected v0.3.1 schema; no execution or inferred successes."""
    if not all(
        isinstance(value, dict) for value in (results, solutions, records, syntax)
    ):
        message = "Malformed EvalPlus results or task metadata"
        raise ValueError(message)
    ids = set(solutions)
    if not ids or set(records) != ids or set(syntax) != ids:
        message = "Result metadata task IDs do not align with samples"
        raise ValueError(message)
    if (
        results.get("hash") != dataset_hash
        or not isinstance(results.get("eval"), dict)
        or set(results["eval"]) != ids
    ):
        message = "EvalPlus results have a different dataset hash or task set"
        raise ValueError(message)
    outcomes = []
    for task_id, sample in solutions.items():
        rows = results["eval"][task_id]
        if (
            not isinstance(rows, list)
            or len(rows) != 1
            or not isinstance(rows[0], dict)
        ):
            message = "pass@1 requires exactly one result per task"
            raise ValueError(message)
        row = rows[0]
        base, plus = row.get("base_status"), row.get("plus_status")
        if (
            row.get("task_id") != task_id
            or row.get("solution") != sample["solution"]
            or base not in ("pass", "fail", "timeout")
            or plus not in ("pass", "fail", "timeout")
            or type(syntax[task_id]) is not bool
        ):
            message = f"Malformed or mismatched EvalPlus result: {task_id}"
            raise ValueError(message)
        outcomes.append(
            {
                **records[task_id],
                "base_status": base,
                "plus_status": plus,
                "base_pass": base == "pass",
                "plus_pass": base == plus == "pass",
                "syntax_parse_success": syntax[task_id],
                "timeout": "timeout" in (base, plus),
            }
        )
    total = len(outcomes)
    summary = {"task_count": total}
    for level in ("base", "plus"):
        passed = sum(row[f"{level}_pass"] for row in outcomes)
        summary[level] = {
            "passed": passed,
            "pass@1": passed / total,
            "wilson_95": wilson_interval(passed, total),
            "timeout_rate": sum(row[f"{level}_status"] == "timeout" for row in outcomes)
            / total,
        }
    summary.update(
        {
            "syntax_parse_success_rate": sum(
                row["syntax_parse_success"] for row in outcomes
            )
            / total,
            "timeout_rate": sum(row["timeout"] for row in outcomes) / total,
            "truncation_rate": sum(row["token_limit_truncated"] for row in outcomes)
            / total,
            "function_boundary_stop_rate": sum(
                row.get("stop_reason") == "function_boundary" for row in outcomes
            )
            / total,
            "generated_tokens": sum(row["generated_tokens"] for row in outcomes),
            "generation_seconds": sum(row["latency_seconds"] for row in outcomes),
            "timeout_definition": "EvalPlus whole-task process status; per-test timeouts may be fail",
        }
    )
    return summary, outcomes


@dataclass(frozen=True)
class SandboxLimits:
    """Fixed single-worker sandbox with bounded CPU, memory and host wall time."""

    cpus: int = 2
    memory_gb: int = 8
    timeout_seconds: int = 3600

    def __post_init__(self):
        for value, upper in (
            (self.cpus, 8),
            (self.memory_gb, 32),
            (self.timeout_seconds, 86400),
        ):
            if type(value) is not int or not 1 <= value <= upper:
                message = f"Sandbox limit must be an integer in [1, {upper}]"
                raise ValueError(message)


def _docker_prefix(docker_sudo: bool) -> list[str]:
    if type(docker_sudo) is not bool:
        message = "docker_sudo must be an explicit boolean"
        raise ValueError(message)
    return ["sudo", "-n", "docker"] if docker_sudo else ["docker"]


def inspect_image(image: str, *, docker_sudo: bool = False) -> str:
    """Resolve a local trusted image to its immutable ID, without pulling it."""
    if not image or image.startswith("-") or any(c.isspace() for c in image):
        message = "Provide a Docker image reference, digest, or immutable ID"
        raise ValueError(message)
    try:
        completed = subprocess.run(
            [
                *_docker_prefix(docker_sudo),
                "image",
                "inspect",
                "--format",
                "{{json .}}",
                image,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        info = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        message = "Docker/image unavailable: build Dockerfile.eval and supply --image; if required explicitly use --docker-sudo; no host fallback"
        raise RuntimeError(message) from exc
    if not isinstance(info, dict) or not isinstance(info.get("Config"), dict):
        message = "Malformed Docker image inspection metadata"
        raise ValueError(message)
    config = info.get("Config", {})
    image_id = info.get("Id", "")
    if (
        not isinstance(image_id, str)
        or not _IMAGE_ID.fullmatch(image_id)
        or not isinstance(config.get("Labels"), dict)
        or any(
            (config.get("Labels") or {}).get(k) != v for k, v in IMAGE_LABELS.items()
        )
        or config.get("Entrypoint") != IMAGE_ENTRYPOINT
        or config.get("Volumes")
    ):
        message = (
            "Image must be the pinned Dockerfile.eval image, without declared volumes"
        )
        raise ValueError(message)
    if image.startswith("sha256:") and image != image_id:
        message = "Inspected image ID differs from the requested ID"
        raise ValueError(message)
    if "@" in image and image not in (info.get("RepoDigests") or []):
        message = "Inspected image does not match the requested repository digest"
        raise ValueError(message)
    return image_id


def _mount_path(path: str | Path, *, directory: bool = False) -> Path:
    path = Path(path).resolve(strict=True)
    if any(c in str(path) for c in (",", "\n", "\r")):
        message = "Docker bind paths must not contain commas or newlines"
        raise ValueError(message)
    if not (path.is_dir() if directory else path.is_file()):
        message = f"Not a regular {'directory' if directory else 'file'}: {path}"
        raise ValueError(message)
    return path


def docker_command(
    *,
    image_id: str,
    name: str,
    dataset: str,
    dataset_path: Path,
    samples_path: Path,
    output_dir: Path,
    mode: str,
    limits: SandboxLimits,
    docker_sudo: bool = False,
) -> list[str]:
    """Build the sole execution boundary; never mount a parent directory."""
    if not _IMAGE_ID.fullmatch(image_id) or not _CONTAINER_NAME.fullmatch(name):
        message = "Require an inspected immutable image ID and fresh sandbox name"
        raise ValueError(message)
    if dataset not in BENCHMARKS or mode not in ("full", "smoke"):
        message = "Invalid dataset or evaluation mode"
        raise ValueError(message)
    data = _mount_path(dataset_path)
    samples = _mount_path(samples_path)
    output = _mount_path(output_dir, directory=True)
    return [
        *_docker_prefix(docker_sudo),
        "run",
        "--rm",
        "--pull",
        "never",
        "--name",
        name,
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        "65534:65534",
        "--cpus",
        str(limits.cpus),
        "--memory",
        f"{limits.memory_gb}g",
        "--memory-swap",
        f"{limits.memory_gb}g",
        "--pids-limit",
        "128",
        "--ipc",
        "private",
        "--shm-size",
        "64m",
        "--ulimit",
        "nofile=256:256",
        "--ulimit",
        "fsize=1073741824:1073741824",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=2g,mode=1777",
        "--workdir",
        "/output",
        "--mount",
        f"type=bind,source={data},target=/input/benchmark.jsonl,readonly",
        "--mount",
        f"type=bind,source={samples},target=/input/samples.jsonl,readonly",
        "--mount",
        f"type=bind,source={output},target=/output",
        image_id,
        dataset,
        mode,
    ]


def _run_container(
    command: list[str], output_dir: Path, timeout: int, *, docker_sudo: bool = False
) -> None:
    prefix = _docker_prefix(docker_sudo)
    if command[: len(prefix) + 1] != [*prefix, "run"]:
        message = "Sandbox launch and cleanup must use the same Docker prefix"
        raise ValueError(message)
    name = command[command.index("--name") + 1]
    if not _CONTAINER_NAME.fullmatch(name):
        message = "Refusing cleanup of a non-owned container name"
        raise ValueError(message)
    try:
        with (output_dir / "container.log").open("xb") as log:
            try:
                subprocess.run(
                    command,
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                )
            finally:
                # docker run's client can die while its container keeps running.
                cleanup = subprocess.run(
                    [*prefix, "rm", "-f", name],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if cleanup.returncode and "No such container" not in cleanup.stderr:
                    message = f"Could not remove owned sandbox {name}: {cleanup.stderr}"
                    raise RuntimeError(message)
    except subprocess.TimeoutExpired as exc:
        message = f"Evaluation timed out; owned sandbox {name} cleanup requested; no score produced"
        raise RuntimeError(message) from exc
    except (OSError, subprocess.CalledProcessError) as exc:
        message = f"Docker evaluation failed; inspect {output_dir / 'container.log'}; no host fallback"
        raise RuntimeError(message) from exc


def _read_container_json(path: Path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > 128 * 1024 * 1024:
        message = f"Unsafe or oversized container output: {path}"
        raise ValueError(message)
    return _read_json(path)


def evaluate_samples(
    *,
    dataset: str,
    dataset_path: str | Path,
    samples_dir: str | Path,
    output_dir: str | Path,
    image: str,
    limits: SandboxLimits | None = None,
    docker_sudo: bool = False,
) -> dict:
    """Evaluate a completed generation bundle strictly in a verified local image."""
    limits = limits or SandboxLimits()
    dataset_path, samples_dir = Path(dataset_path), Path(samples_dir)
    tasks, benchmark = load_benchmark(dataset, dataset_path)
    manifest = _read_json(samples_dir / "generation.json")
    samples_path = samples_dir / "samples.jsonl"
    solutions, records = validate_generation(
        tasks,
        benchmark,
        samples_path.read_bytes(),
        manifest,
        raw_samples=(samples_dir / "raw_samples.jsonl").read_bytes(),
    )
    image_id = inspect_image(image, docker_sudo=docker_sudo)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    # Only this new, public-results directory is writable by the sandbox UID.
    output_dir.chmod(0o777)
    name = "opaque-code-eval-" + uuid.uuid4().hex
    command = docker_command(
        image_id=image_id,
        name=name,
        dataset=dataset,
        dataset_path=dataset_path,
        samples_path=samples_path,
        output_dir=output_dir,
        mode=manifest["mode"],
        limits=limits,
        docker_sudo=docker_sudo,
    )
    _run_container(command, output_dir, limits.timeout_seconds, docker_sudo=docker_sudo)
    execution = _read_container_json(output_dir / "execution.json")
    raw = dataset_path.read_bytes()
    if manifest["mode"] == "smoke":
        raw = (
            b"\n".join(
                line
                for line in raw.splitlines()
                if line.strip() and json.loads(line)["task_id"] in solutions
            )
            + b"\n"
        )
    effective_hash = _md5(raw)
    if (
        not isinstance(execution, dict)
        or execution.get("evalplus_version") != EVALPLUS_VERSION
        or execution.get("source_sha256") != benchmark["sha256"]
        or execution.get("dataset_hash") != effective_hash
        or execution.get("mode") != manifest["mode"]
        or execution.get("task_ids") != manifest["task_ids"]
    ):
        message = "Container execution metadata differs from the requested evaluation"
        raise ValueError(message)
    summary, outcomes = summarize_results(
        _read_container_json(output_dir / "samples_eval_results.json"),
        solutions,
        records,
        execution.get("syntax"),
        dataset_hash=effective_hash,
    )
    summary.update(
        {
            "mode": manifest["mode"],
            "benchmark": benchmark,
            "model": manifest["model"],
            "generation": manifest["generation"],
            "evalplus_version": EVALPLUS_VERSION,
            "image_id": image_id,
            "docker_prefix": _docker_prefix(docker_sudo),
            "container_name": name,
            "limits": asdict(limits),
            "effective_dataset_md5": effective_hash,
            "samples_sha256": manifest["samples_sha256"],
            "raw_samples_sha256": manifest["raw_samples_sha256"],
            "graded_sample_kind": "function_completion_prefix_not_raw_generation",
        }
    )
    for filename, value in (("summary.json", summary), ("per_task.json", outcomes)):
        # Exclusive create: never follow a link left by untrusted code.
        with (output_dir / filename).open("xb") as output:
            output.write(_json_bytes(value))
    return summary


def main(argv: list[str] | None = None) -> None:
    """CLI with lazy ML dependencies and no host execution option."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser(
        "generate", help="Generate text only; no code execution"
    )
    evaluate = commands.add_parser(
        "evaluate", help="Run only in an inspected, isolated Docker image"
    )
    for subparser in (generate, evaluate):
        subparser.add_argument("--dataset", choices=BENCHMARKS, required=True)
        subparser.add_argument(
            "--dataset-path",
            type=Path,
            required=True,
            help="Full official uncompressed JSONL, not a mini/noextreme dataset",
        )
        subparser.add_argument(
            "--output-dir", type=Path, required=True, help="Must not exist"
        )
    generate.add_argument("--model-id", default=MODEL_ID)
    generate.add_argument("--model-revision", default=MODEL_REVISION)
    generate.add_argument("--adapters", type=Path)
    generate.add_argument(
        "--merge-for-evaluation",
        action="store_true",
        help="Irreversibly merge loaded adapters for frozen inference only",
    )
    generate.add_argument("--device", default="cuda")
    generate.add_argument(
        "--arm", help="Public arm label (defaults to base or adapter)"
    )
    generate.add_argument("--max-new-tokens", type=int, default=512)
    generate.add_argument(
        "--smoke-limit", type=int, help="First N tasks; explicitly labeled smoke"
    )
    generate.add_argument("--local-files-only", action="store_true")
    evaluate.add_argument("--samples-dir", type=Path, required=True)
    evaluate.add_argument(
        "--image",
        required=True,
        help="Trusted locally built Dockerfile.eval image (no automatic pulls)",
    )
    evaluate.add_argument("--timeout-seconds", type=int, default=3600)
    evaluate.add_argument("--cpus", type=int, default=2)
    evaluate.add_argument("--memory-gb", type=int, default=8)
    evaluate.add_argument(
        "--docker-sudo",
        action="store_true",
        help="Explicitly use sudo -n docker for inspection, launch, and cleanup",
    )
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        summary = evaluate_samples(
            dataset=args.dataset,
            dataset_path=args.dataset_path,
            samples_dir=args.samples_dir,
            output_dir=args.output_dir,
            image=args.image,
            limits=SandboxLimits(args.cpus, args.memory_gb, args.timeout_seconds),
            docker_sudo=args.docker_sudo,
        )
        print(json.dumps(summary, sort_keys=True, allow_nan=False))
        return
    if not _REVISION.fullmatch(args.model_revision):
        parser.error("--model-revision must be an immutable 40-hex revision")
    tasks, _ = load_benchmark(args.dataset, args.dataset_path)
    _selected_ids(tasks, args.smoke_limit)
    generation_settings(args.max_new_tokens)
    if args.output_dir.exists():
        parser.error("--output-dir must not exist")
    if args.merge_for_evaluation and not args.adapters:
        parser.error("--merge-for-evaluation requires --adapters")
    if args.adapters:
        read_adapter_spec(args.adapters)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        message = "Generation needs the parent's Mellum-compatible Torch/Transformers environment"
        raise RuntimeError(message) from exc
    kwargs = {
        "revision": args.model_revision,
        "trust_remote_code": False,
        "local_files_only": args.local_files_only,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.bfloat16, attn_implementation="eager", **kwargs
    ).to(args.device)
    if args.adapters:
        model, _ = restore_adapters(model, args.adapters)
    if args.merge_for_evaluation:
        merge_for_evaluation(model)
    generate_samples(
        model,
        tokenizer,
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        model_id=args.model_id,
        model_revision=args.model_revision,
        max_new_tokens=args.max_new_tokens,
        smoke_limit=args.smoke_limit,
        arm=args.arm,
    )


if __name__ == "__main__":
    main()
