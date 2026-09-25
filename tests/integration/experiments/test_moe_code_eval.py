"""Generation/protocol tests; no benchmark code or Docker is executed here.

Result fixtures exercise parsing/arithmetic only, not benchmark correctness.
Real execution must pass the separately run, bounded Docker sandbox gate.
"""

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from examples.moe_privacy import code_eval as evaluation
from examples.moe_privacy.sft_adapters import configure_expert_lora, export_trainable
from safetensors.torch import load_file, save_file
from scipy.stats import binomtest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from torch.nn.utils import parametrize
from transformers import (
    MellumConfig,
    MellumForCausalLM,
    PreTrainedTokenizerFast,
    StoppingCriteriaList,
)

IMAGE_ID = "sha256:" + "a" * 64
CONTAINER_NAME = "opaque-code-eval-" + "b" * 32


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def benchmark_file(tmp_path):
    path = tmp_path / "humaneval.jsonl"
    # Inert fixtures, not a replacement benchmark. The deliberately dangerous
    # canonical string must never be run while loading or generating prompts.
    write_jsonl(
        path,
        [
            {
                "task_id": f"HumanEval/{i}",
                "prompt": f'def solve_{i}(x):\n    """Return x."""\n',
                "canonical_solution": "    raise RuntimeError('never execute on host')\n",
                "entry_point": f"solve_{i}",
                "contract": "",
                "base_input": [[1]],
                "plus_input": [[2]],
                "atol": 0,
            }
            for i in reversed(range(164))
        ],
    )
    return path


@pytest.fixture
def tiny_model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        config = MellumConfig(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            head_dim=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
            num_experts=2,
            num_experts_per_tok=1,
            moe_intermediate_size=8,
            pad_token_id=1,
            bos_token_id=2,
            eos_token_id=3,
            output_router_logits=True,
            router_aux_loss_coef=0.25,
        )
        config._attn_implementation = "eager"
        return MellumForCausalLM(config)


@pytest.fixture
def tokenizer():
    backend = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "[PAD]": 1, "[BOS]": 2, "[EOS]": 3, "return": 4, "x": 5},
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )


def test_loads_full_tasks_in_numeric_order_without_executing(benchmark_file):
    tasks, metadata = evaluation.load_benchmark("humaneval", benchmark_file)
    assert list(tasks) == [f"HumanEval/{i}" for i in range(164)]
    assert metadata["declared_count"] == 164
    assert metadata["version"] == "v0.1.10"
    assert metadata["sha256"] == hashlib.sha256(benchmark_file.read_bytes()).hexdigest()
    assert metadata["tasks_with_extended_tests"] == 164
    assert metadata["tasks_without_extended_tests"] == 0


@pytest.mark.parametrize("problem", ["missing", "duplicate", "wrong_id", "schema"])
def test_rejects_incomplete_or_malformed_benchmark(benchmark_file, problem):
    rows = [json.loads(line) for line in benchmark_file.read_text().splitlines()]
    if problem == "missing":
        rows.pop()
    elif problem == "duplicate":
        rows[-1] = rows[0]
    elif problem == "wrong_id":
        rows[-1]["task_id"] = "HumanEval/999"
    else:
        rows[-1].pop("plus_input")
    write_jsonl(benchmark_file, rows)
    message = {
        "missing": "requires all",
        "duplicate": "duplicate task_id",
        "wrong_id": "IDs must cover",
        "schema": "missing nonempty",
    }[problem]
    with pytest.raises(ValueError, match=message):
        evaluation.load_benchmark("humaneval", benchmark_file)


def test_mbpp_requires_378_tasks_not_legacy_399(benchmark_file):
    row = json.loads(benchmark_file.read_text().splitlines()[0])
    rows = [{**row, "task_id": f"Mbpp/{i * 2}"} for i in range(378)]
    write_jsonl(benchmark_file, rows)
    tasks, metadata = evaluation.load_benchmark("mbpp", benchmark_file)
    assert len(tasks) == metadata["declared_count"] == 378
    assert metadata["version"] == "v0.2.0"
    rows.extend({**row, "task_id": f"Mbpp/{i}"} for i in range(1000, 1021))
    write_jsonl(benchmark_file, rows)
    with pytest.raises(ValueError, match="requires all 378"):
        evaluation.load_benchmark("mbpp", benchmark_file)


@pytest.fixture
def mbpp_empty_extended_file(benchmark_file):
    row = json.loads(benchmark_file.read_text().splitlines()[0])
    rows = [{**row, "task_id": f"Mbpp/{i}"} for i in range(377)]
    # Official MBPP+ v0.2.0 uses an empty object for this task's extra inputs.
    rows.append({**row, "task_id": "Mbpp/793", "plus_input": {}})
    write_jsonl(benchmark_file, rows)
    return benchmark_file


@pytest.mark.parametrize("empty_inputs", [{}, []])
def test_mbpp_accepts_empty_extended_inputs(mbpp_empty_extended_file, empty_inputs):
    path = mbpp_empty_extended_file
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["plus_input"] = empty_inputs
    write_jsonl(path, rows)
    raw = path.read_bytes()
    tasks, metadata = evaluation.load_benchmark("mbpp", path)
    assert len(tasks) == metadata["declared_count"] == 378
    assert tasks["Mbpp/793"] == rows[-1]
    assert metadata["tasks_with_extended_tests"] == 377
    assert metadata["tasks_without_extended_tests"] == 1
    assert metadata["sha256"] == hashlib.sha256(raw).hexdigest()
    assert path.read_bytes() == raw


@pytest.mark.parametrize("invalid", [None, False, 0, "", "[]", {"bad": []}, "missing"])
def test_mbpp_rejects_invalid_extended_inputs(mbpp_empty_extended_file, invalid):
    path = mbpp_empty_extended_file
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if invalid == "missing":
        rows[-1].pop("plus_input")
    else:
        rows[-1]["plus_input"] = invalid
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="plus_input"):
        evaluation.load_benchmark("mbpp", path)


@pytest.mark.parametrize("invalid", [None, [], {}, "", {"bad": []}])
def test_mbpp_still_requires_nonempty_base_inputs(mbpp_empty_extended_file, invalid):
    path = mbpp_empty_extended_file
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["base_input"] = invalid
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="base_input"):
        evaluation.load_benchmark("mbpp", path)


@pytest.mark.parametrize(
    ("ids", "budget", "expected", "truncated"),
    [
        ([4, 5, 3], 3, "return x", False),
        ([4, 5], 2, "return x", True),
        ([3], 4, "", False),
    ],
)
def test_eos_and_budget_without_function_boundary(
    tokenizer, ids, budget, expected, truncated
):
    text, record = evaluation.decode_completion(tokenizer, ids, [3], budget)
    assert text == expected
    assert record["generated_tokens"] == len(ids)
    assert record["token_limit_truncated"] is truncated


def test_no_silent_short_completion_or_special_token_stripping(tokenizer):
    with pytest.raises(ValueError, match="stopped before EOS"):
        evaluation.decode_completion(tokenizer, [4], [3], 2)
    text, _ = evaluation.decode_completion(tokenizer, [0, 3], [3], 2)
    assert text == "[UNK]"


def test_completion_boundary_preserves_complete_answer_before_incomplete_suffix():
    prompt = "def has_close_elements(numbers, threshold):\n"
    answer = (
        "    for i in range(len(numbers)):\n"
        "        for j in range(i + 1, len(numbers)):\n"
        "            if abs(numbers[i] - numbers[j]) < threshold:\n"
        "                return True\n"
        "    return False\n\n\n"
    )
    for suffix in ("def unrelated(", "import pytest\n\ndef test_incomplete():\n"):
        completion = answer + suffix
        boundary = evaluation.completion_boundary(
            prompt, completion, "has_close_elements"
        )
        assert completion[: boundary["completion_offset"]] == answer
        assert completion[boundary["completion_offset"] :] == suffix


@pytest.mark.parametrize(
    "suffix",
    [
        "def other(",
        "async def other(",
        "class Tests:",
        "import pytest",
        "from your_module import solve",
        "assert solve(1) == 1",
        "if __name__ == '__main__':",
        "print(solve(1))",
        "@decorator",
        "```",
    ],
)
def test_completion_boundary_uses_fixed_top_level_markers(suffix):
    answer = "    return x\n\n"
    boundary = evaluation.completion_boundary(
        "def solve(x):\n", answer + suffix, "solve"
    )
    assert boundary["completion_offset"] == len(answer)


def test_completion_boundary_waits_for_mbpp_target_not_first_generated_def():
    prompt = '"""Write a function to return x."""\n'
    answer = "import math\n\ndef helper(x):\n    return x\n\ndef solve(x):\n    return helper(x)\n\n"
    assert evaluation.completion_boundary(prompt, answer, "solve") is None
    boundary = evaluation.completion_boundary(prompt, answer + "def test_", "solve")
    assert boundary["completion_offset"] == len(answer)


def test_completion_boundary_does_not_fix_wrong_or_incomplete_targets():
    prompt = "def solve(x):\n"
    wrong = "    return -x\n\n"
    boundary = evaluation.completion_boundary(prompt, wrong + "def test_", "solve")
    assert (wrong + "def test_")[: boundary["completion_offset"]] == wrong
    for incomplete in ("    return (\n", "    if x:\n", "    return '''\n"):
        assert evaluation.completion_boundary(prompt, incomplete, "solve") is None
    assert (
        evaluation.completion_boundary(prompt, "    return (\ndef test_", "solve")
        is None
    )
    incomplete = "    if x:\n\n"
    boundary = evaluation.completion_boundary(prompt, incomplete + "def test_", "solve")
    assert (incomplete + "def test_")[: boundary["completion_offset"]] == incomplete


def test_completion_boundary_ignores_strings_nested_code_and_partial_keywords():
    prompt = "def solve(x):\n"
    answer = (
        "    text = '''\ndef not_a_definition(\nimport pytest\n'''\n"
        "    def nested():\n        return x\n"
        "    if x:\n        import math\n"
        "    # def test_not_a_boundary\n"
        "    return nested()\n\n"
        "# def comment_not_a_boundary\n"
    )
    assert evaluation.completion_boundary(prompt, answer, "solve") is None
    assert evaluation.completion_boundary(prompt, answer + "import", "solve") is None
    assert evaluation.completion_boundary(prompt, answer + "important", "solve") is None
    boundary = evaluation.completion_boundary(prompt, answer + "import ", "solve")
    assert boundary == {"completion_offset": len(answer), "marker": "import "}


def test_transformers_stopping_criteria_decodes_incrementally_and_preserves_raw():
    vocabulary = dict(enumerate(sorted(pre_tokenizers.ByteLevel.alphabet())))
    backend = Tokenizer(
        models.BPE({value: key for key, value in vocabulary.items()}, merges=[])
    )
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    prompt = "def solve(x):\n"
    answer = "    return x\n\n"
    suffix = "import pytest\n\ndef unfinished("
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    generated_ids = tokenizer.encode(answer + suffix, add_special_tokens=False)
    stopper = evaluation._FunctionCompletionStop(
        tokenizer, prompt, "solve", len(prompt_ids)
    )
    criteria = StoppingCriteriaList([stopper])
    for count in range(1, len(generated_ids) + 1):
        input_ids = torch.tensor([prompt_ids + generated_ids[:count]])
        if criteria(input_ids, scores=None).item():
            break
    else:
        pytest.fail("No function-boundary stop")
    raw, record = evaluation.decode_completion(
        tokenizer, generated_ids[:count], [999], 512, boundary=stopper.boundary
    )
    assert raw == answer + "import "
    assert raw[: stopper.boundary["completion_offset"]] == answer
    assert record == {
        "generated_tokens": count,
        "stop_reason": "function_boundary",
        "token_limit_truncated": False,
    }
    _, at_budget = evaluation.decode_completion(
        tokenizer, generated_ids[:count], [999], count, boundary=stopper.boundary
    )
    assert at_budget["token_limit_truncated"] is True
    assert at_budget["stop_reason"] == "function_boundary"
    with pytest.raises(ValueError, match="batch size one"):
        stopper(input_ids.repeat(2, 1), scores=None)


@pytest.mark.parametrize(
    "corruption",
    [
        None,
        "missing_raw",
        "raw_hash",
        "raw_task",
        "raw_record_hash",
        "prefix",
        "boundary",
        "reason",
        "truncation",
        "policy",
    ],
)
def test_raw_and_graded_generation_binding(benchmark_file, corruption):
    # Inert protocol-validation fixture, not an actual model generation or score.
    tasks, benchmark = evaluation.load_benchmark("humaneval", benchmark_file)
    prompt = tasks["HumanEval/0"]["prompt"]
    answer = "    return x\n\n"
    solution = prompt + answer
    raw_solution = solution + "import "
    sample = {"task_id": "HumanEval/0", "solution": solution}
    raw_sample = {**sample, "solution": raw_solution}
    samples, raw_samples = (
        evaluation._json_bytes(sample),
        evaluation._json_bytes(raw_sample),
    )
    record = {
        "task_id": "HumanEval/0",
        "prompt_sha256": evaluation._sha256(prompt.encode()),
        "solution_sha256": evaluation._sha256(solution.encode()),
        "raw_solution_sha256": evaluation._sha256(raw_solution.encode()),
        "completion_boundary": {"completion_offset": len(answer), "marker": "import "},
        "generated_tokens": 20,
        "prompt_tokens": 10,
        "latency_seconds": 1.0,
        "stop_reason": "function_boundary",
        "token_limit_truncated": False,
    }
    manifest = {
        "format_version": 2,
        "evalplus_version": evaluation.EVALPLUS_VERSION,
        "benchmark": benchmark,
        "mode": "smoke",
        "smoke_limit": 1,
        "task_ids": ["HumanEval/0"],
        "tasks": [record],
        "model": {"revision": evaluation.MODEL_REVISION},
        "generation": {**evaluation.generation_settings(), "eos_token_ids": [0]},
        "samples_sha256": evaluation._sha256(samples),
        "raw_samples_sha256": evaluation._sha256(raw_samples),
    }
    if corruption == "missing_raw":
        raw_samples = None
    elif corruption == "raw_hash":
        raw_samples += b"\n"
    elif corruption == "raw_task":
        raw_samples = evaluation._json_bytes({**raw_sample, "task_id": "HumanEval/1"})
        manifest["raw_samples_sha256"] = evaluation._sha256(raw_samples)
    elif corruption == "raw_record_hash":
        record["raw_solution_sha256"] = "wrong"
    elif corruption == "prefix":
        sample["solution"] = solution.replace("return x", "return -x")
        samples = evaluation._json_bytes(sample)
        manifest["samples_sha256"] = evaluation._sha256(samples)
        record["solution_sha256"] = evaluation._sha256(sample["solution"].encode())
    elif corruption == "boundary":
        record["completion_boundary"]["completion_offset"] -= 1
    elif corruption == "reason":
        record["stop_reason"] = "eos"
    elif corruption == "truncation":
        record["token_limit_truncated"] = True
    elif corruption == "policy":
        manifest["generation"]["stop_policy"] = "per_task_best"
    if corruption is not None:
        message = {
            "raw_task": "align with every task",
            "raw_record_hash": "raw generation metadata",
            "prefix": "generation metadata",
            "boundary": "generation metadata",
            "reason": "generation metadata",
            "truncation": "generation metadata",
        }.get(corruption, "manifest")
        with pytest.raises(ValueError, match=message):
            evaluation.validate_generation(
                tasks, benchmark, samples, manifest, raw_samples=raw_samples
            )
    else:
        solutions, records = evaluation.validate_generation(
            tasks, benchmark, samples, manifest, raw_samples=raw_samples
        )
        assert solutions == {"HumanEval/0": sample}
        assert records == {"HumanEval/0": record}


@pytest.mark.parametrize("adapted", [False, True])
def test_real_greedy_generation_reuses_model_and_preserves_state(
    tiny_model, tokenizer, benchmark_file, tmp_path, monkeypatch, adapted
):
    if adapted:
        configure_expert_lora(tiny_model)
    native_generate = tiny_model.generate
    observed = []

    def generate(**kwargs):
        assert "router_aux_loss" not in kwargs
        assert "output_router_logits" not in kwargs
        assert isinstance(kwargs["stopping_criteria"], StoppingCriteriaList)
        assert not torch.is_autocast_enabled("cpu")
        result = native_generate(**kwargs)
        observed.append(result[0, kwargs["input_ids"].shape[1] :].tolist())
        return result

    monkeypatch.setattr(tiny_model, "generate", generate)
    output = tmp_path / "generation"
    flags = {name: value.requires_grad for name, value in tiny_model.named_parameters()}
    rng = torch.get_rng_state().clone()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        manifest = evaluation.generate_samples(
            tiny_model,
            tokenizer,
            dataset="humaneval",
            dataset_path=benchmark_file,
            output_dir=output,
            max_new_tokens=3,
            smoke_limit=1,
        )
        assert torch.is_autocast_enabled("cpu")
    assert tiny_model.training
    assert tiny_model.config.output_router_logits
    assert tiny_model.config.router_aux_loss_coef == 0.25
    assert flags == {
        name: value.requires_grad for name, value in tiny_model.named_parameters()
    }
    assert torch.equal(rng, torch.get_rng_state())
    assert manifest["mode"] == "smoke"
    assert manifest["task_ids"] == ["HumanEval/0"]
    assert manifest["generation"]["do_sample"] is False
    assert manifest["model"]["arm"] == ("adapter" if adapted else "base")
    samples = (output / "samples.jsonl").read_bytes()
    raw_samples = (output / "raw_samples.jsonl").read_bytes()
    assert manifest["raw_samples_sha256"] == hashlib.sha256(raw_samples).hexdigest()
    tasks, metadata = evaluation.load_benchmark("humaneval", benchmark_file)
    solutions, records = evaluation.validate_generation(
        tasks, metadata, samples, manifest, raw_samples=raw_samples
    )
    assert solutions["HumanEval/0"]["solution"].startswith(
        tasks["HumanEval/0"]["prompt"]
    )
    tokens = observed[0]
    if tokenizer.eos_token_id in tokens:
        tokens = tokens[: tokens.index(tokenizer.eos_token_id)]
    completion = tokenizer.decode(
        tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    assert (
        solutions["HumanEval/0"]["solution"]
        == tasks["HumanEval/0"]["prompt"] + completion
    )
    assert 1 <= records["HumanEval/0"]["generated_tokens"] <= 3
    assert records["HumanEval/0"]["latency_seconds"] > 0
    repeated = tmp_path / "repeated"
    evaluation.generate_samples(
        tiny_model,
        tokenizer,
        dataset="humaneval",
        dataset_path=benchmark_file,
        output_dir=repeated,
        max_new_tokens=3,
        smoke_limit=1,
    )
    assert (repeated / "samples.jsonl").read_bytes() == samples
    for field, value in (
        ("mode", "full"),
        ("task_ids", ["HumanEval/1"]),
        ("samples_sha256", "bad"),
        ("generation", {}),
    ):
        corrupted = {**manifest, field: value}
        with pytest.raises(ValueError, match="manifest"):
            evaluation.validate_generation(
                tasks, metadata, samples, corrupted, raw_samples=raw_samples
            )
    corrupted = copy.deepcopy(manifest)
    corrupted["tasks"][0]["latency_seconds"] = float("nan")
    with pytest.raises(ValueError, match="metadata"):
        evaluation.validate_generation(
            tasks, metadata, samples, corrupted, raw_samples=raw_samples
        )
    missing_sample = evaluation._json_bytes(
        {"task_id": "HumanEval/1", "solution": "inert"}
    )
    corrupted = {
        **manifest,
        "samples_sha256": hashlib.sha256(missing_sample).hexdigest(),
    }
    with pytest.raises(ValueError, match="align with every task"):
        evaluation.validate_generation(
            tasks, metadata, missing_sample, corrupted, raw_samples=raw_samples
        )
    duplicated = samples + samples
    corrupted = {**manifest, "samples_sha256": hashlib.sha256(duplicated).hexdigest()}
    with pytest.raises(ValueError, match="duplicate task_id"):
        evaluation.validate_generation(
            tasks, metadata, duplicated, corrupted, raw_samples=raw_samples
        )


def test_full_selection_and_invalid_smoke_limits(benchmark_file):
    tasks, _ = evaluation.load_benchmark("humaneval", benchmark_file)
    assert evaluation._selected_ids(tasks, None) == list(tasks)
    assert evaluation._selected_ids(tasks, 2) == ["HumanEval/0", "HumanEval/1"]
    for limit in (0, -1, 165, True, 1.5):
        with pytest.raises(ValueError, match="smoke_limit"):
            evaluation._selected_ids(tasks, limit)


def test_context_overflow_fails_without_truncating_prompt(
    tiny_model, tokenizer, benchmark_file, tmp_path
):
    with pytest.raises(ValueError, match="exceeds model context"):
        evaluation.generate_samples(
            tiny_model,
            tokenizer,
            dataset="humaneval",
            dataset_path=benchmark_file,
            output_dir=tmp_path / "overflow",
            max_new_tokens=128,
            smoke_limit=1,
        )
    assert tiny_model.training
    assert not (tmp_path / "overflow" / "generation.json").exists()


def test_real_static_adapter_export_restoration(tiny_model, tmp_path):
    base = tiny_model.to(dtype=torch.bfloat16)
    fresh = copy.deepcopy(base)
    model, metadata = configure_expert_lora(base)
    with torch.no_grad():
        for i, parameter in enumerate(p for p in model.parameters() if p.requires_grad):
            parameter.fill_((i + 1) / 100)
    output = tmp_path / "adapters"
    export_trainable(model, output)
    restored, restored_metadata = evaluation.restore_adapters(fresh, output)
    assert restored_metadata == metadata
    assert set(dict(restored.named_parameters())) == set(dict(model.named_parameters()))
    for name, parameter in model.named_parameters():
        actual = dict(restored.named_parameters())[name]
        assert torch.equal(actual, parameter), name
        assert actual.requires_grad == parameter.requires_grad
        if actual.requires_grad:
            assert actual.dtype == torch.float32
    evaluation._prepare_model(restored)
    restored.get_input_embeddings().weight.requires_grad_(True)
    with pytest.raises(ValueError, match="Trainability"):
        evaluation._prepare_model(restored)


@pytest.mark.parametrize(
    "problem",
    [
        "kind",
        "rank",
        "missing_weights",
        "shape",
        "metadata",
        "metadata_type",
        "format_type",
        "tensor_dtype",
        "tensor_shape",
        "tensor_keys",
    ],
)
def test_malformed_export_rejected_by_preflight_or_real_loader(
    tiny_model, tmp_path, problem
):
    fresh = copy.deepcopy(tiny_model)
    model, _ = configure_expert_lora(tiny_model)
    output = tmp_path / "adapters"
    export_trainable(model, output)
    path = output / "adapter_spec.json"
    spec = json.loads(path.read_text())
    if problem == "kind":
        spec["adapter_kind"] = "peft"
    elif problem == "rank":
        spec["config"]["rank"] = True
    elif problem == "missing_weights":
        (output / "trainable.safetensors").unlink()
    elif problem == "shape":
        next(iter(spec["tensors"].values()))["shape"] = [1]
    elif problem == "metadata_type":
        spec["metadata"]["router_count"] = True
    elif problem == "format_type":
        spec["format_version"] = True
    elif problem.startswith("tensor_"):
        tensors = load_file(output / "trainable.safetensors")
        key = next(iter(tensors))
        if problem == "tensor_dtype":
            tensors[key] = tensors[key].half()
        elif problem == "tensor_shape":
            tensors[key] = tensors[key].reshape(-1)
        else:
            tensors.pop(key)
        save_file(tensors, output / "trainable.safetensors")
    else:
        spec["metadata"]["router_count"] = 99
    path.write_text(json.dumps(spec))
    message = {
        "kind": "static expert-LoRA",
        "rank": "static expert-LoRA",
        "missing_weights": "missing trainable",
        "shape": "specification",
        "metadata": "specification",
        "metadata_type": "specification",
        "format_type": "static expert-LoRA",
        "tensor_dtype": "dtype mismatch",
        "tensor_shape": "shape mismatch",
        "tensor_keys": "tensor keys differ",
    }[problem]
    with pytest.raises(ValueError, match=message):
        evaluation.restore_adapters(fresh, output)


@pytest.fixture
def result_fixture():
    # Explicit synthetic parser inputs, never represented as an actual evaluation.
    statuses = [
        ("pass", "pass"),
        ("pass", "fail"),
        ("timeout", "pass"),
        ("fail", "timeout"),
    ]
    solutions, records, syntax, results = (
        {},
        {},
        {},
        {"hash": "fixture-md5", "eval": {}},
    )
    for i, (base, plus) in enumerate(statuses):
        task_id = f"HumanEval/{i}"
        solutions[task_id] = {"task_id": task_id, "solution": "inert parser fixture"}
        records[task_id] = {
            "task_id": task_id,
            "token_limit_truncated": i == 0,
            "generated_tokens": 3,
            "latency_seconds": 0.5,
        }
        syntax[task_id] = i < 3
        results["eval"][task_id] = [
            {**solutions[task_id], "base_status": base, "plus_status": plus}
        ]
    return results, solutions, records, syntax


def test_pass_fraction_plus_requires_base_and_wilson_timeout_metrics(result_fixture):
    summary, outcomes = evaluation.summarize_results(
        *result_fixture, dataset_hash="fixture-md5"
    )
    assert summary["base"]["pass@1"] == 0.5
    assert summary["plus"]["pass@1"] == 0.25
    assert summary["base"]["wilson_95"] == pytest.approx([0.150038989, 0.849961011])
    assert summary["timeout_rate"] == 0.5
    assert summary["base"]["timeout_rate"] == summary["plus"]["timeout_rate"] == 0.25
    assert summary["truncation_rate"] == 0.25
    assert summary["syntax_parse_success_rate"] == 0.75
    assert summary["generated_tokens"] == 12
    assert summary["generation_seconds"] == 2
    assert outcomes[2]["plus_pass"] is False


@pytest.mark.parametrize(
    "problem",
    [
        "missing",
        "extra",
        "duplicate",
        "solution",
        "task_id",
        "hash",
        "status",
        "syntax",
    ],
)
def test_results_must_align_exactly_with_tasks_and_samples(result_fixture, problem):
    results, solutions, records, syntax = result_fixture
    row = results["eval"]["HumanEval/0"][0]
    if problem == "missing":
        results["eval"].pop("HumanEval/0")
    elif problem == "extra":
        results["eval"]["HumanEval/999"] = [row]
    elif problem == "duplicate":
        results["eval"]["HumanEval/0"].append(row)
    elif problem in ("solution", "task_id"):
        row[problem] = "wrong"
    elif problem == "hash":
        results["hash"] = "wrong"
    elif problem == "status":
        row["plus_status"] = None
    else:
        syntax["HumanEval/0"] = "true"
    message = "exactly one result" if problem == "duplicate" else "EvalPlus result"
    with pytest.raises(ValueError, match=message):
        evaluation.summarize_results(
            results, solutions, records, syntax, dataset_hash="fixture-md5"
        )


def test_wilson_boundaries_and_invalid_counts():
    for passed in (0, 1, 82, 163, 164):
        expected = binomtest(passed, 164).proportion_ci(method="wilson")
        assert evaluation.wilson_interval(passed, 164) == pytest.approx(expected)
    for counts in ((0, 0), (-1, 4), (5, 4), (True, 4)):
        with pytest.raises(ValueError, match="Require integer counts"):
            evaluation.wilson_interval(*counts)


@pytest.fixture(params=[False, True])
def docker_sudo(request):
    return request.param


@pytest.fixture
def docker_command(tmp_path, benchmark_file, docker_sudo):
    samples = tmp_path / "samples.jsonl"
    samples.write_text("inert: never passed to a real Docker process")
    output = tmp_path / "output"
    output.mkdir()
    return evaluation.docker_command(
        image_id=IMAGE_ID,
        name=CONTAINER_NAME,
        dataset="humaneval",
        dataset_path=benchmark_file,
        samples_path=samples,
        output_dir=output,
        mode="full",
        limits=evaluation.SandboxLimits(),
        docker_sudo=docker_sudo,
    ), output


def test_exact_docker_isolation_flags_and_mounts(docker_command, docker_sudo):
    command, output = docker_command
    prefix = ["sudo", "-n", "docker"] if docker_sudo else ["docker"]
    assert command[: len(prefix) + 1] == [*prefix, "run"]
    expected = {
        "--pull": "never",
        "--name": CONTAINER_NAME,
        "--network": "none",
        "--cap-drop": "ALL",
        "--security-opt": "no-new-privileges",
        "--user": "65534:65534",
        "--cpus": "2",
        "--memory": "8g",
        "--memory-swap": "8g",
        "--pids-limit": "128",
        "--ipc": "private",
        "--shm-size": "64m",
        "--tmpfs": "/tmp:rw,noexec,nosuid,nodev,size=2g,mode=1777",
        "--workdir": "/output",
    }
    for flag, value in expected.items():
        assert command.count(flag) == 1
        assert command[command.index(flag) + 1] == value
    assert "--read-only" in command
    assert "--rm" in command
    assert [command[i + 1] for i, part in enumerate(command) if part == "--ulimit"] == [
        "nofile=256:256",
        "fsize=1073741824:1073741824",
    ]
    mounts = [command[i + 1] for i, part in enumerate(command) if part == "--mount"]
    assert len(mounts) == 3
    assert mounts[0].endswith(",target=/input/benchmark.jsonl,readonly")
    assert mounts[1].endswith(",target=/input/samples.jsonl,readonly")
    assert mounts[2] == f"type=bind,source={output.resolve()},target=/output"
    assert command[-3:] == [IMAGE_ID, "humaneval", "full"]
    for flag in (
        "--gpus",
        "--privileged",
        "--env-file",
        "--env",
        "-v",
        "--volume",
        "--device",
        "--pid",
    ):
        assert flag not in command


def test_groundtruth_cache_file_limit_is_sufficient_and_bounded(
    docker_command, tmp_path
):
    command, _ = docker_command
    file_limit = next(arg for arg in command if arg.startswith("fsize="))
    soft, hard = file_limit.removeprefix("fsize=").split(":")
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import errno
import resource
import signal
import sys

soft, hard = map(int, sys.argv[2:])
resource.setrlimit(resource.RLIMIT_FSIZE, (soft, hard))
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
with open(sys.argv[1], "wb", buffering=0) as cache:
    cache.seek(128 * 1024 * 1024)
    cache.write(b"x")
    cache.seek(soft)
    try:
        cache.write(b"x")
    except OSError as exc:
        assert exc.errno == errno.EFBIG
    else:
        raise AssertionError("Sandbox file-size limit was not enforced")
""",
            str(tmp_path / "cache-probe"),
            soft,
            hard,
        ],
        check=True,
        timeout=10,
    )


@pytest.mark.parametrize("failure", ["timeout", "exit", "interrupt"])
def test_cleanup_removes_only_own_container(
    docker_command, monkeypatch, failure, docker_sudo
):
    command, output = docker_command
    prefix = ["sudo", "-n", "docker"] if docker_sudo else ["docker"]
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        if args == command:
            assert kwargs["timeout"] == 5
            if failure == "timeout":
                raise subprocess.TimeoutExpired(args, 5)
            if failure == "exit":
                raise subprocess.CalledProcessError(1, args)
            raise KeyboardInterrupt
        assert args == [*prefix, "rm", "-f", CONTAINER_NAME]
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(evaluation.subprocess, "run", run)
    expected = KeyboardInterrupt if failure == "interrupt" else RuntimeError
    with pytest.raises(expected):
        evaluation._run_container(command, output, 5, docker_sudo=docker_sudo)
    assert calls == [command, [*prefix, "rm", "-f", CONTAINER_NAME]]
    assert not (output / "summary.json").exists()


def test_image_inspection_pins_id_and_refuses_unverified_images(
    monkeypatch, docker_sudo
):
    prefix = ["sudo", "-n", "docker"] if docker_sudo else ["docker"]
    info = {
        "Id": IMAGE_ID,
        "Config": {
            "Labels": dict(evaluation.IMAGE_LABELS),
            "Entrypoint": evaluation.IMAGE_ENTRYPOINT,
        },
        "RepoDigests": ["trusted@sha256:" + "c" * 64],
    }

    def inspect(args, **kwargs):
        assert args[: len(prefix) + 4] == [
            *prefix,
            "image",
            "inspect",
            "--format",
            "{{json .}}",
        ]
        return subprocess.CompletedProcess(args, 0, json.dumps(info), "")

    monkeypatch.setattr(evaluation.subprocess, "run", inspect)
    assert (
        evaluation.inspect_image("trusted:local", docker_sudo=docker_sudo) == IMAGE_ID
    )
    assert (
        evaluation.inspect_image(info["RepoDigests"][0], docker_sudo=docker_sudo)
        == IMAGE_ID
    )
    with pytest.raises(ValueError, match="digest"):
        evaluation.inspect_image("other@sha256:" + "d" * 64, docker_sudo=docker_sudo)
    info["Config"]["Volumes"] = {"/surprise": {}}
    with pytest.raises(ValueError, match="volumes"):
        evaluation.inspect_image("trusted:local", docker_sudo=docker_sudo)
    info["Config"].pop("Volumes")
    info["Config"]["Labels"] = {}
    with pytest.raises(ValueError, match="pinned"):
        evaluation.inspect_image("trusted:local", docker_sudo=docker_sudo)


def test_missing_docker_has_no_host_fallback(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(evaluation.subprocess, "run", missing)
    with pytest.raises(RuntimeError, match="no host fallback"):
        evaluation.inspect_image(IMAGE_ID)


def test_rejects_mount_injection_and_output_symlinks(tmp_path):
    path = tmp_path / "bad,readonly"
    path.write_text("public")
    with pytest.raises(ValueError, match="commas"):
        evaluation._mount_path(path)
    link = tmp_path / "result.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="Unsafe"):
        evaluation._read_container_json(link)


def test_cli_help_does_not_need_site_packages():
    script = Path(evaluation.__file__).resolve()
    result = subprocess.run(
        [sys.executable, "-S", str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert "generate" in result.stdout
    assert "evaluate" in result.stdout


@pytest.mark.parametrize("results", [None, [], {"hash": "fixture-md5", "eval": None}])
def test_malformed_result_objects_fail_closed(result_fixture, results):
    _, solutions, records, syntax = result_fixture
    with pytest.raises(ValueError, match="EvalPlus results"):
        evaluation.summarize_results(
            results, solutions, records, syntax, dataset_hash="fixture-md5"
        )


@pytest.mark.parametrize(
    "info", [None, [], {"Config": None}, {"Id": None, "Config": {}}]
)
def test_malformed_image_metadata_fails_closed(monkeypatch, info):
    def inspect(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, json.dumps(info), "")

    monkeypatch.setattr(evaluation.subprocess, "run", inspect)
    with pytest.raises(ValueError, match=r"inspection metadata|pinned Dockerfile"):
        evaluation.inspect_image(IMAGE_ID)


@pytest.mark.parametrize(
    "limits",
    [{"cpus": 0}, {"cpus": True}, {"memory_gb": 33}, {"timeout_seconds": 86401}],
)
def test_sandbox_limits_are_finite_bounded_positive_integers(limits):
    with pytest.raises(ValueError, match="Sandbox limit"):
        evaluation.SandboxLimits(**limits)


def test_existing_output_is_rejected_before_container_start(
    tiny_model, tokenizer, benchmark_file, tmp_path, monkeypatch
):
    samples_dir = tmp_path / "generation"
    evaluation.generate_samples(
        tiny_model,
        tokenizer,
        dataset="humaneval",
        dataset_path=benchmark_file,
        output_dir=samples_dir,
        max_new_tokens=1,
        smoke_limit=1,
    )
    calls = []

    def inspect(args, **kwargs):
        calls.append(args)
        assert args[:3] == ["docker", "image", "inspect"]
        info = {
            "Id": IMAGE_ID,
            "Config": {
                "Labels": evaluation.IMAGE_LABELS,
                "Entrypoint": evaluation.IMAGE_ENTRYPOINT,
            },
        }
        return subprocess.CompletedProcess(args, 0, json.dumps(info), "")

    monkeypatch.setattr(evaluation.subprocess, "run", inspect)
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "unrelated"
    sentinel.write_text("untouched")
    with pytest.raises(FileExistsError):
        evaluation.evaluate_samples(
            dataset="humaneval",
            dataset_path=benchmark_file,
            samples_dir=samples_dir,
            output_dir=output,
            image=IMAGE_ID,
        )
    assert len(calls) == 1
    assert sentinel.read_text() == "untouched"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_merge_has_exact_forward_parity_and_records_inference_status(
    tiny_model, tokenizer, benchmark_file, tmp_path, dtype
):
    model, _ = configure_expert_lora(tiny_model.to(dtype=dtype))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("lora_B"):
                parameter.fill_(0.03)
    evaluation._prepare_model(model)
    model.eval()
    inputs = torch.tensor([[4, 5, 4]])
    with torch.no_grad():
        before = model(input_ids=inputs, use_cache=False).logits
        weights = {}
        for target in model._expert_lora_config["target_parameters"]:
            name, parameter = target.rsplit(".", 1)
            weights[target] = getattr(model.get_submodule(name), parameter).clone()
    rng = torch.get_rng_state().clone()
    assert evaluation.merge_for_evaluation(model) is model
    assert not model.training
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert not any(parametrize.is_parametrized(module) for module in model.modules())
    for target, expected in weights.items():
        name, parameter = target.rsplit(".", 1)
        assert torch.equal(getattr(model.get_submodule(name), parameter), expected)
    with torch.no_grad():
        after = model(input_ids=inputs, use_cache=False).logits
    assert torch.equal(before, after)
    assert torch.equal(torch.get_rng_state(), rng)
    manifest = evaluation.generate_samples(
        model,
        tokenizer,
        dataset="humaneval",
        dataset_path=benchmark_file,
        output_dir=tmp_path / "merged",
        max_new_tokens=2,
        smoke_limit=1,
    )
    assert manifest["model"]["merged_for_evaluation"] is True
    assert manifest["model"]["arm"] == "adapter"
    with pytest.raises(ValueError, match="unmerged"):
        evaluation.merge_for_evaluation(model)
    next(model.parameters()).requires_grad_(True)
    with pytest.raises(ValueError, match="stay frozen"):
        evaluation._prepare_model(model)


def test_merge_requires_static_adapters(tiny_model):
    with pytest.raises(ValueError, match="unmerged static expert LoRA"):
        evaluation.merge_for_evaluation(tiny_model)


def test_evaluate_uses_sudo_for_inspection_run_and_timeout_cleanup(
    tiny_model, tokenizer, benchmark_file, tmp_path, monkeypatch
):
    samples_dir = tmp_path / "generation"
    evaluation.generate_samples(
        tiny_model,
        tokenizer,
        dataset="humaneval",
        dataset_path=benchmark_file,
        output_dir=samples_dir,
        max_new_tokens=1,
        smoke_limit=2,
    )
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        assert args[:3] == ["sudo", "-n", "docker"]
        if args[3] == "image":
            info = {
                "Id": IMAGE_ID,
                "Config": {
                    "Labels": evaluation.IMAGE_LABELS,
                    "Entrypoint": evaluation.IMAGE_ENTRYPOINT,
                },
            }
            return subprocess.CompletedProcess(args, 0, json.dumps(info), "")
        if args[3] == "run":
            assert args[-1] == "smoke"
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])
        assert args[3:5] == ["rm", "-f"]
        assert args[-1] == calls[1][calls[1].index("--name") + 1]
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(evaluation.subprocess, "run", run)
    output = tmp_path / "failed_evaluation"
    with pytest.raises(RuntimeError, match="timed out"):
        evaluation.evaluate_samples(
            dataset="humaneval",
            dataset_path=benchmark_file,
            samples_dir=samples_dir,
            output_dir=output,
            image=IMAGE_ID,
            docker_sudo=True,
            limits=evaluation.SandboxLimits(timeout_seconds=5),
        )
    assert len(calls) == 3
    assert not (output / "summary.json").exists()
