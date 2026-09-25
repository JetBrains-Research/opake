"""Pinned public code instructions, with one prompt/answer pair per DP record."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch
from datasets import Dataset, load_dataset

MIN_SEQUENCE_LENGTH = 4


def prompt_key(prompt: str) -> str:
    """Identify duplicate public instructions independently of whitespace."""
    return hashlib.sha256(" ".join(prompt.split()).encode()).hexdigest()


def fingerprint(rows: Dataset) -> str:
    """Hash the actual, ordered tokenized records used by an arm."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True).encode() + b"\n")
    return digest.hexdigest()


def prepare_split(
    raw, tokenizer, *, count, max_length, seed, excluded=(), require_full_answers=False
):
    """Select public records reproducibly, excluding duplicate/held-out prompts."""
    if count <= 0 or max_length < MIN_SEQUENCE_LENGTH:
        message = "count must be positive and max_length must be at least 4"
        raise ValueError(message)
    seen = set(excluded)
    rows, keys = [], []
    rejected = {"duplicate_prompt": 0, "empty_text": 0, "no_answer_tokens": 0}
    truncated = 0
    for row in raw.shuffle(seed=seed):
        prompt, answer = row["prompt"].strip(), row["completion"].strip()
        if not prompt or not answer:
            rejected["empty_text"] += 1
            continue
        key = prompt_key(prompt)
        if key in seen:
            rejected["duplicate_prompt"] += 1
            continue
        formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
        prompt_ids = tokenizer(formatted, add_special_tokens=True)["input_ids"]
        answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
        if not answer_ids or len(prompt_ids) >= max_length - 1:
            rejected["no_answer_tokens"] += 1
            continue
        if tokenizer.eos_token_id is not None:
            answer_ids = [*answer_ids, tokenizer.eos_token_id]
        if require_full_answers and len(prompt_ids) + len(answer_ids) > max_length:
            rejected["overlength"] = rejected.get("overlength", 0) + 1
            continue
        ids = (prompt_ids + answer_ids)[:max_length]
        mask = ([0] * len(prompt_ids) + [1] * len(answer_ids))[:max_length]
        truncated += len(prompt_ids) + len(answer_ids) > max_length
        rows.append({"input_ids": ids, "completion_mask": mask})
        keys.append(key)
        seen.add(key)
        if len(rows) == count:
            break
    if len(rows) != count:
        message = f"only {len(rows)} usable public records; requested {count}"
        raise ValueError(message)
    dataset = Dataset.from_list(rows)
    return dataset, {
        "records": len(dataset),
        "sha256": fingerprint(dataset),
        "prompt_hashes": keys,
        "truncated_answers": truncated,
        "rejected_before_selection_completed": rejected,
        "completion_tokens": sum(sum(row["completion_mask"]) for row in rows),
        "attended_tokens": sum(len(row["input_ids"]) for row in rows),
    }


def prepare_data(config, tokenizer):
    """Load immutable public train/test splits; never pack unrelated records."""
    if getattr(config, "dataset_format", "code_alpaca") == "magicoder_oss":
        train, validation, _, _, metadata = prepare_partitions(config, tokenizer)
        return train, validation, metadata
    raw = load_dataset(config.dataset_id, revision=config.dataset_revision)
    train, train_info = prepare_split(
        raw["train"],
        tokenizer,
        count=config.train_sequences,
        max_length=config.sequence_length,
        seed=config.data_seed,
    )
    all_train_prompts = {prompt_key(row["prompt"]) for row in raw["train"]}
    validation, validation_info = prepare_split(
        raw["test"],
        tokenizer,
        count=config.validation_sequences,
        max_length=config.sequence_length,
        seed=config.validation_seed,
        excluded=all_train_prompts,
    )
    return (
        train,
        validation,
        {
            "kind": "public_code_instruction_completion",
            "dataset_id": config.dataset_id,
            "dataset_revision": config.dataset_revision,
            "train": train_info,
            "validation": validation_info,
            "train_sha256": train_info["sha256"],
            "validation_sha256": validation_info["sha256"],
            "packing": False,
            "held_out_from": "this fine-tuning experiment, not verified for pretraining",
        },
    )


def prepare_partitions(config, tokenizer):
    """Reserve test and diagnostic records before any training or model selection."""
    if config.dataset_format == "code_alpaca":
        if config.test_sequences or config.diagnostic_sequences:
            message = "additional partitions require the magicoder_oss format"
            raise ValueError(message)
        train, validation, metadata = prepare_data(config, tokenizer)
        empty = Dataset.from_list([])
        return train, validation, empty, empty, metadata
    raw = load_dataset(
        config.dataset_id, revision=config.dataset_revision, token=False
    )["train"]
    languages = {language.lower() for language in config.dataset_languages}
    formatted = []
    language_filtered = 0
    for row in raw:
        if languages and row["lang"].lower() not in languages:
            language_filtered += 1
            continue
        formatted.append({"prompt": row["problem"], "completion": row["solution"]})
    names = ("test", "validation", "diagnostic", "train")
    counts = [getattr(config, f"{name}_sequences") for name in names]
    selected, selection = prepare_split(
        Dataset.from_list(formatted),
        tokenizer,
        count=sum(counts),
        max_length=config.sequence_length,
        seed=config.data_seed,
        require_full_answers=config.require_full_answers,
    )
    partitions = {}
    metadata = {
        "kind": "public_code_instruction_completion",
        "dataset_id": config.dataset_id,
        "dataset_revision": config.dataset_revision,
        "languages": sorted(languages),
        "packing": False,
        "require_full_answers": config.require_full_answers,
        "held_out_from": "this fine-tuning experiment, not verified for pretraining",
        "selection": {
            "language_filtered": language_filtered,
            "overlength_filtered": selection["rejected_before_selection_completed"].get(
                "overlength", 0
            ),
            "rejected_before_selection_completed": selection[
                "rejected_before_selection_completed"
            ],
        },
    }
    start = 0
    for name, count in zip(names, counts, strict=True):
        part = selected.select(range(start, start + count))
        keys = selection["prompt_hashes"][start : start + count]
        exclusions = set(config.excluded_train_prompt_hashes)
        if name == "train" and exclusions.intersection(keys):
            part, train_info = prepare_split(
                Dataset.from_list(formatted),
                tokenizer,
                count=count,
                max_length=config.sequence_length,
                seed=config.data_seed,
                excluded=[*selection["prompt_hashes"][:start], *exclusions],
                require_full_answers=config.require_full_answers,
            )
            keys = train_info["prompt_hashes"]
        partitions[name] = part
        metadata[name] = {
            "records": count,
            "sha256": fingerprint(part),
            "prompt_hashes": keys,
            "truncated_answers": 0 if config.require_full_answers else None,
            "completion_tokens": sum(sum(row["completion_mask"]) for row in part),
            "attended_tokens": sum(len(row["input_ids"]) for row in part),
        }
        metadata[f"{name}_sha256"] = metadata[name]["sha256"]
        start += count
    metadata["excluded_train_prompt_hashes"] = sorted(
        config.excluded_train_prompt_hashes
    )
    return (
        *(partitions[name] for name in ("train", "validation", "test", "diagnostic")),
        metadata,
    )


@dataclass(frozen=True)
class CompletionCollator:
    """Pad to a public bound, masking prompts/padding only from the task loss."""

    max_length: int
    pad_token_id: int

    def __call__(self, rows):
        shape = (len(rows), self.max_length)
        ids = torch.full(shape, self.pad_token_id, dtype=torch.long)
        attention = torch.zeros(shape, dtype=torch.long)
        labels = torch.full(shape, -100, dtype=torch.long)
        for index, row in enumerate(rows):
            tokens, mask = row["input_ids"], row["completion_mask"]
            length = len(tokens)
            if not 1 < length <= self.max_length or len(mask) != length:
                message = "record/mask length must fit the public token bound"
                raise ValueError(message)
            if any(value not in (0, 1) for value in mask) or not any(mask[1:]):
                message = "every record must contain supervised answer tokens"
                raise ValueError(message)
            token_tensor = torch.tensor(tokens, dtype=torch.long)
            ids[index, :length] = token_tensor
            attention[index, :length] = 1
            labels[index, :length] = torch.where(
                torch.tensor(mask, dtype=torch.bool), token_tensor, -100
            )
        return {"input_ids": ids, "attention_mask": attention, "labels": labels}
