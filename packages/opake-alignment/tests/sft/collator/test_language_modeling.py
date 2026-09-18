"""Tests for the language-modeling collator factory (plan §7.6, §3.1, §11).

Covers:

- Padding and truncation correctness on a hand-built batch of 3 ragged examples.
- ``labels`` pad positions are ``-100``; ``completion_only_loss=True`` additionally
  masks non-completion positions to ``-100``.
- ``pad_to_multiple_of=8`` rounds ``L`` up to the next multiple of 8.
- Key-set stability (output always has exactly the required keys).
- Determinism: calling the collator twice on the same list yields
  ``torch.equal`` tensors.
- ``completion_mask`` output key present iff at least one input example carries
  ``"completion_mask"``.
- Adversarial edge cases: all-zero ``completion_mask`` row, examples longer than
  ``max_length``, and ``pad_to_multiple_of`` interaction with truncated length.

Import targets the concrete implementation path; the public façade is wired in
unit β.W.
"""

from __future__ import annotations

import torch

from opake.api.alignment.sft.collator._language_modeling import (
    language_modeling_collator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PAD = 0  # pad_token_id used throughout


def _make_collator(**kwargs):
    """Convenience wrapper with sensible defaults."""
    kwargs.setdefault("pad_token_id", _PAD)
    kwargs.setdefault("max_length", 16)
    return language_modeling_collator(**kwargs)


# ---------------------------------------------------------------------------
# 1. Basic padding / truncation on a hand-built batch of 3 ragged examples
# ---------------------------------------------------------------------------

_RAGGED_EXAMPLES = [
    {"input_ids": [1, 2, 3, 4, 5]},  # length 5
    {"input_ids": [10, 20]},  # length 2  ← shortest
    {"input_ids": [7, 8, 9, 11, 12, 13]},  # length 6  ← longest
]

# Expected L = max(5, 2, 6) = 6  (no pad_to_multiple_of, max_length=16)
_EXPECTED_L = 6
_EXPECTED_B = 3


def test_basic_shapes() -> None:
    """Output tensors have the correct batch and sequence dimensions."""
    collate = _make_collator()
    batch = collate(_RAGGED_EXAMPLES)

    assert batch["input_ids"].shape == (_EXPECTED_B, _EXPECTED_L)
    assert batch["attention_mask"].shape == (_EXPECTED_B, _EXPECTED_L)
    assert batch["labels"].shape == (_EXPECTED_B, _EXPECTED_L)


def test_basic_input_ids_content() -> None:
    """Real tokens are placed correctly; padding fills the remaining positions."""
    collate = _make_collator()
    batch = collate(_RAGGED_EXAMPLES)

    ids = batch["input_ids"]
    # Row 0: [1, 2, 3, 4, 5, PAD]
    assert ids[0, :5].tolist() == [1, 2, 3, 4, 5]
    assert ids[0, 5].item() == _PAD
    # Row 1: [10, 20, PAD, PAD, PAD, PAD]
    assert ids[1, :2].tolist() == [10, 20]
    assert ids[1, 2:].tolist() == [_PAD] * 4
    # Row 2: [7, 8, 9, 11, 12, 13]  (no padding needed)
    assert ids[2, :6].tolist() == [7, 8, 9, 11, 12, 13]


def test_basic_attention_mask_content() -> None:
    """Attention mask is 1 on real tokens and 0 on pad positions."""
    collate = _make_collator()
    batch = collate(_RAGGED_EXAMPLES)

    attn = batch["attention_mask"]
    assert attn[0].tolist() == [1, 1, 1, 1, 1, 0]
    assert attn[1].tolist() == [1, 1, 0, 0, 0, 0]
    assert attn[2].tolist() == [1, 1, 1, 1, 1, 1]


def test_basic_labels_content() -> None:
    """Labels copy input_ids; pad positions are -100."""
    collate = _make_collator()
    batch = collate(_RAGGED_EXAMPLES)

    labels = batch["labels"]
    # Row 0: tokens [1,2,3,4,5], pad → -100
    assert labels[0, :5].tolist() == [1, 2, 3, 4, 5]
    assert labels[0, 5].item() == -100
    # Row 1: tokens [10,20], pads → -100
    assert labels[1, :2].tolist() == [10, 20]
    assert labels[1, 2:].tolist() == [-100] * 4
    # Row 2: no padding
    assert labels[2, :6].tolist() == [7, 8, 9, 11, 12, 13]


def test_basic_no_completion_mask_key_when_absent() -> None:
    """``completion_mask`` key is absent when no example carries it."""
    collate = _make_collator()
    batch = collate(_RAGGED_EXAMPLES)
    assert "completion_mask" not in batch


# ---------------------------------------------------------------------------
# 3. completion_mask output key iff inputs carry it
# ---------------------------------------------------------------------------


def test_completion_mask_present_when_all_carry_it() -> None:
    """``completion_mask`` key present when every example supplies it."""
    examples = [
        {"input_ids": [1, 2, 3], "completion_mask": [0, 1, 1]},
        {"input_ids": [4, 5], "completion_mask": [0, 1]},
    ]
    collate = _make_collator()
    batch = collate(examples)
    assert "completion_mask" in batch
    assert batch["completion_mask"].shape == (2, 3)


def test_completion_mask_present_when_some_carry_it() -> None:
    """``completion_mask`` key present even if only *one* example carries it."""
    examples = [
        {"input_ids": [1, 2, 3], "completion_mask": [0, 1, 1]},
        {"input_ids": [4, 5]},  # no completion_mask
    ]
    collate = _make_collator()
    batch = collate(examples)
    assert "completion_mask" in batch


def test_completion_mask_absent_when_none_carry_it() -> None:
    """``completion_mask`` key absent when no example supplies it."""
    examples = [
        {"input_ids": [1, 2, 3]},
        {"input_ids": [4, 5]},
    ]
    collate = _make_collator()
    batch = collate(examples)
    assert "completion_mask" not in batch


def test_completion_mask_values_correctly_padded() -> None:
    """Completion mask is correctly placed; padding columns are 0."""
    examples = [
        {"input_ids": [1, 2, 3, 4], "completion_mask": [0, 0, 1, 1]},
        {"input_ids": [5, 6], "completion_mask": [0, 1]},
    ]
    collate = _make_collator()
    batch = collate(examples)
    cm = batch["completion_mask"]
    # Row 0: [0, 0, 1, 1]  — no padding needed
    assert cm[0].tolist() == [0, 0, 1, 1]
    # Row 1: [0, 1, 0, 0]  — padded with 0s
    assert cm[1].tolist() == [0, 1, 0, 0]


def test_completion_mask_missing_row_marks_real_tokens() -> None:
    """An example with no ``completion_mask`` marks all real tokens."""
    examples = [
        {"input_ids": [1, 2, 3], "completion_mask": [0, 1, 1]},
        {"input_ids": [4, 5]},
    ]
    collate = _make_collator()
    batch = collate(examples)
    cm = batch["completion_mask"]
    assert cm[1].tolist() == [1, 1, 0]


# ---------------------------------------------------------------------------
# 4. labels with completion_only_loss=True
# ---------------------------------------------------------------------------


def test_labels_completion_only_loss_masks_non_completion() -> None:
    """With ``completion_only_loss=True`` prompt tokens get -100 in labels."""
    examples = [
        {"input_ids": [1, 2, 3, 4], "completion_mask": [0, 0, 1, 1]},
    ]
    collate = language_modeling_collator(
        pad_token_id=_PAD, max_length=16, completion_only_loss=True
    )
    batch = collate(examples)
    labels = batch["labels"]
    # Positions 0,1 are non-completion → -100; positions 2,3 are completion → token ids
    assert labels[0].tolist() == [-100, -100, 3, 4]


def test_labels_completion_only_loss_no_mask_retains_all_real_tokens() -> None:
    """Without ``completion_mask``, completion-only mode leaves labels intact."""
    examples = [
        {"input_ids": [1, 2, 3]},
    ]
    collate = language_modeling_collator(
        pad_token_id=_PAD, max_length=16, completion_only_loss=True
    )
    batch = collate(examples)
    labels = batch["labels"]
    assert labels[0].tolist() == [1, 2, 3]


def test_labels_completion_only_loss_mixed_batch() -> None:
    """Mixed batch: one example has completion_mask, one does not."""
    examples = [
        {"input_ids": [1, 2, 3], "completion_mask": [0, 1, 1]},
        {"input_ids": [4, 5, 6]},
    ]
    collate = language_modeling_collator(
        pad_token_id=_PAD, max_length=16, completion_only_loss=True
    )
    batch = collate(examples)
    labels = batch["labels"]
    # Row 0: prompt (pos 0) → -100; completion (pos 1, 2) → token ids
    assert labels[0].tolist() == [-100, 2, 3]
    # Row 1 has no completion mask, so all real tokens remain supervised.
    assert labels[1].tolist() == [4, 5, 6]
    assert batch["completion_mask"][1].tolist() == [1, 1, 1]


def test_labels_no_completion_only_loss_retains_all_real_tokens() -> None:
    """Without ``completion_only_loss``, labels are not masked by completion_mask."""
    examples = [
        {"input_ids": [1, 2, 3, 4], "completion_mask": [0, 0, 1, 1]},
    ]
    collate = _make_collator(completion_only_loss=False)
    batch = collate(examples)
    # All real tokens preserved; pad position → -100 (but there's none here)
    assert batch["labels"][0].tolist() == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# 5. Truncation: examples longer than max_length
# ---------------------------------------------------------------------------


def test_truncation_keep_start() -> None:
    """Examples exceeding ``max_length`` are truncated from the right (keep-start)."""
    max_len = 4
    long_example = [{"input_ids": [1, 2, 3, 4, 5, 6, 7, 8]}]
    collate = language_modeling_collator(pad_token_id=_PAD, max_length=max_len)
    batch = collate(long_example)

    # L should equal max_length (the example was longer)
    assert batch["input_ids"].shape == (1, max_len)
    assert batch["input_ids"][0].tolist() == [1, 2, 3, 4]


def test_truncation_attention_mask_all_ones() -> None:
    """After truncation to exactly max_length, attention_mask is all 1s."""
    max_len = 3
    long_example = [{"input_ids": [10, 20, 30, 40, 50]}]
    collate = language_modeling_collator(pad_token_id=_PAD, max_length=max_len)
    batch = collate(long_example)
    assert batch["attention_mask"][0].tolist() == [1, 1, 1]


def test_truncation_labels_no_pad() -> None:
    """When a truncated example fills max_length, labels contain no -100 (no pads)."""
    max_len = 3
    long_example = [{"input_ids": [10, 20, 30, 40, 50]}]
    collate = language_modeling_collator(pad_token_id=_PAD, max_length=max_len)
    batch = collate(long_example)
    # No pad positions → labels should be [10, 20, 30]
    assert batch["labels"][0].tolist() == [10, 20, 30]


def test_truncation_completion_mask_truncated() -> None:
    """completion_mask is also truncated to max_length when the example is long."""
    max_len = 4
    examples = [
        {"input_ids": [1, 2, 3, 4, 5, 6], "completion_mask": [0, 0, 1, 1, 1, 1]},
    ]
    collate = language_modeling_collator(pad_token_id=_PAD, max_length=max_len)
    batch = collate(examples)
    # Truncated to first 4 tokens: completion_mask[:4] = [0, 0, 1, 1]
    assert batch["completion_mask"][0].tolist() == [0, 0, 1, 1]


def test_truncation_with_shorter_sibling() -> None:
    """In a mixed batch, L = min(max_length, max_example_length_after_truncation)."""
    max_len = 5
    examples = [
        {"input_ids": [1, 2, 3, 4, 5, 6, 7]},  # longer → truncated to 5
        {"input_ids": [10, 20]},  # shorter
    ]
    collate = language_modeling_collator(pad_token_id=_PAD, max_length=max_len)
    batch = collate(examples)
    # L should be max_len = 5 (the truncated longer example drives L)
    assert batch["input_ids"].shape == (2, max_len)
    assert batch["input_ids"][0].tolist() == [1, 2, 3, 4, 5]
    assert batch["input_ids"][1].tolist() == [10, 20, 0, 0, 0]


# ---------------------------------------------------------------------------
# 6. pad_to_multiple_of
# ---------------------------------------------------------------------------


def test_pad_to_multiple_of_8_rounds_up() -> None:
    """L is rounded up to the next multiple of 8."""
    # Longest example has 5 tokens → L would be 5 without rounding → rounds to 8.
    examples = [
        {"input_ids": [1, 2, 3, 4, 5]},
        {"input_ids": [10]},
    ]
    collate = language_modeling_collator(
        pad_token_id=_PAD, max_length=32, pad_to_multiple_of=8
    )
    batch = collate(examples)
    assert batch["input_ids"].shape[1] == 8
    assert batch["attention_mask"].shape[1] == 8
    assert batch["labels"].shape[1] == 8


def test_pad_to_multiple_of_already_multiple() -> None:
    """When max(example lengths) is already a multiple, no extra padding added."""
    examples = [
        {"input_ids": [1, 2, 3, 4, 5, 6, 7, 8]},  # length 8 — already a multiple
    ]
    collate = language_modeling_collator(
        pad_token_id=_PAD, max_length=32, pad_to_multiple_of=8
    )
    batch = collate(examples)
    assert batch["input_ids"].shape[1] == 8


def test_pad_to_multiple_of_with_truncation() -> None:
    """pad_to_multiple_of is applied after truncation.

    max_length=5, pad_to_multiple_of=8: longest example truncated to 5,
    then rounded up to 8.
    """
    examples = [
        {"input_ids": list(range(10))},  # length 10 → truncated to 5
        {"input_ids": [0, 1]},  # length 2
    ]
    collate = language_modeling_collator(
        pad_token_id=_PAD, max_length=5, pad_to_multiple_of=8
    )
    batch = collate(examples)
    # After truncation: max example length = 5; rounded up to multiple of 8 = 8.
    assert batch["input_ids"].shape[1] == 8
    # The truncated row: first 5 tokens correct, positions 5-7 are PAD.
    assert batch["input_ids"][0, :5].tolist() == list(range(5))
    assert batch["input_ids"][0, 5:].tolist() == [_PAD, _PAD, _PAD]


def test_pad_to_multiple_of_labels_respect_rounding() -> None:
    """Extra positions added by rounding are also set to -100 in labels."""
    examples = [{"input_ids": [1, 2, 3]}]
    collate = language_modeling_collator(
        pad_token_id=_PAD, max_length=32, pad_to_multiple_of=8
    )
    batch = collate(examples)
    labels = batch["labels"]
    # Positions 0-2 are real tokens; positions 3-7 are padding → -100.
    assert labels[0, :3].tolist() == [1, 2, 3]
    assert all(v == -100 for v in labels[0, 3:].tolist())


# ---------------------------------------------------------------------------
# 7. Key-set stability
# ---------------------------------------------------------------------------


def test_key_set_without_completion_mask() -> None:
    """Output keys are exactly {input_ids, attention_mask, labels}."""
    collate = _make_collator()
    batch = collate([{"input_ids": [1, 2, 3]}])
    assert set(batch.keys()) == {"input_ids", "attention_mask", "labels"}


def test_key_set_with_completion_mask() -> None:
    """Output keys are exactly {input_ids, attention_mask, labels, completion_mask}."""
    collate = _make_collator()
    batch = collate([{"input_ids": [1, 2, 3], "completion_mask": [0, 1, 1]}])
    assert set(batch.keys()) == {
        "input_ids",
        "attention_mask",
        "labels",
        "completion_mask",
    }


# ---------------------------------------------------------------------------
# 8. Determinism: same input → identical tensors
# ---------------------------------------------------------------------------


def test_determinism_simple_batch() -> None:
    """Calling the same collator twice on the same list yields torch.equal tensors."""
    collate = _make_collator()
    examples = _RAGGED_EXAMPLES
    batch_a = collate(examples)
    batch_b = collate(examples)
    for key in batch_a:
        assert torch.equal(batch_a[key], batch_b[key]), f"mismatch on key '{key}'"


def test_determinism_with_completion_mask() -> None:
    """Determinism holds when completion_mask is present."""
    collate = language_modeling_collator(
        pad_token_id=_PAD, max_length=16, completion_only_loss=True
    )
    examples = [
        {"input_ids": [1, 2, 3, 4], "completion_mask": [0, 0, 1, 1]},
        {"input_ids": [5, 6], "completion_mask": [0, 1]},
        {"input_ids": [7, 8, 9]},
    ]
    batch_a = collate(examples)
    batch_b = collate(examples)
    for key in batch_a:
        assert torch.equal(batch_a[key], batch_b[key]), f"mismatch on key '{key}'"


def test_determinism_with_pad_to_multiple_of() -> None:
    """Determinism holds with pad_to_multiple_of."""
    collate = language_modeling_collator(
        pad_token_id=_PAD, max_length=32, pad_to_multiple_of=8
    )
    examples = [
        {"input_ids": [1, 2, 3]},
        {"input_ids": [4, 5, 6, 7, 8, 9]},
    ]
    batch_a = collate(examples)
    batch_b = collate(examples)
    for key in batch_a:
        assert torch.equal(batch_a[key], batch_b[key]), f"mismatch on key '{key}'"


# ---------------------------------------------------------------------------
# 9. Adversarial edge cases
# ---------------------------------------------------------------------------


def test_all_zero_completion_mask_row() -> None:
    """An example with an all-zero completion_mask is valid; its completion row is 0.

    With ``completion_only_loss=True`` this means all real tokens in that row
    receive ``-100`` in ``labels``.
    """
    examples = [
        {"input_ids": [1, 2, 3, 4], "completion_mask": [1, 1, 1, 1]},
        {"input_ids": [5, 6, 7], "completion_mask": [0, 0, 0]},  # empty completion
    ]
    collate = language_modeling_collator(
        pad_token_id=_PAD, max_length=16, completion_only_loss=True
    )
    batch = collate(examples)
    # Row 0: all completion tokens → all labels are token ids
    assert batch["labels"][0].tolist() == [1, 2, 3, 4]
    # Row 1: no completion tokens → all real tokens are -100; pad (pos 3) → -100
    assert batch["labels"][1].tolist() == [-100, -100, -100, -100]
    # completion_mask row 1 is all-zero
    assert batch["completion_mask"][1].tolist() == [0, 0, 0, 0]


def test_single_token_example() -> None:
    """Single-token examples do not crash and produce correct shapes."""
    examples = [{"input_ids": [42]}]
    collate = _make_collator()
    batch = collate(examples)
    assert batch["input_ids"].shape == (1, 1)
    assert batch["input_ids"][0, 0].item() == 42
    assert batch["attention_mask"][0, 0].item() == 1
    assert batch["labels"][0, 0].item() == 42


def test_all_examples_same_length_no_padding() -> None:
    """When all examples have the same length, no padding is added."""
    examples = [
        {"input_ids": [1, 2, 3]},
        {"input_ids": [4, 5, 6]},
        {"input_ids": [7, 8, 9]},
    ]
    collate = _make_collator()
    batch = collate(examples)
    # No padding → attention_mask is all ones
    assert batch["attention_mask"].tolist() == [[1, 1, 1]] * 3


def test_max_length_exactly_equals_example_length() -> None:
    """When max_length equals the longest example, no truncation occurs."""
    examples = [{"input_ids": [1, 2, 3, 4, 5]}]
    collate = language_modeling_collator(pad_token_id=_PAD, max_length=5)
    batch = collate(examples)
    assert batch["input_ids"][0].tolist() == [1, 2, 3, 4, 5]
    assert batch["attention_mask"][0].tolist() == [1, 1, 1, 1, 1]


def test_completion_only_loss_without_any_completion_mask_in_batch() -> None:
    """completion_only_loss=True but no example carries completion_mask.

    Full-sequence labels are retained and no completion_mask key is emitted.
    """
    examples = [
        {"input_ids": [1, 2, 3]},
        {"input_ids": [4, 5]},
    ]
    collate = language_modeling_collator(
        pad_token_id=_PAD, max_length=16, completion_only_loss=True
    )
    batch = collate(examples)
    # No completion_mask key in output
    assert "completion_mask" not in batch
    assert batch["labels"][0].tolist() == [1, 2, 3]
    assert batch["labels"][1].tolist() == [4, 5, -100]


def test_dtype_is_long() -> None:
    """All output tensors have dtype torch.long."""
    examples = [
        {"input_ids": [1, 2, 3], "completion_mask": [0, 1, 1]},
    ]
    collate = _make_collator()
    batch = collate(examples)
    for key, tensor in batch.items():
        assert tensor.dtype == torch.long, f"expected long dtype for key '{key}'"


def test_large_batch_shapes() -> None:
    """Collating a larger batch (B=8) produces the expected shape."""
    torch.manual_seed(0)
    examples = [
        {"input_ids": list(range(i + 1, i + torch.randint(3, 12, ()).item() + 2))}
        for i in range(8)
    ]
    max_len = 16
    collate = language_modeling_collator(pad_token_id=_PAD, max_length=max_len)
    batch = collate(examples)
    B, L = batch["input_ids"].shape
    assert B == 8
    assert max_len >= L
    assert batch["attention_mask"].shape == (B, L)
    assert batch["labels"].shape == (B, L)
