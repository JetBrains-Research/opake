"""Unit tests for the preference collator (plan §7.6, gate data).

Tests the DPO ``(B, ...)`` per-side layout collator.  Covers:

- Correct right-padding on chosen and rejected sides *independently* (plan
  risk α1: chosen and rejected are separate ``(B, Lc)`` / ``(B, Lr)`` tensors
  — they are NOT concatenated into a ``(2B, L)`` tensor as TRL does).
- ``ref_*_logps`` optional keys present iff *all* inputs carry them; shape
  ``(B,)``.
- Determinism: same input → ``torch.equal`` output (called twice).
- Key-set stability: no unexpected keys appear or disappear across calls.
- Truncation to ``max_length`` (keep-start).
- ``pad_to_multiple_of`` rounding applied *independently* per side.
- Empty-batch edge case.

Import path targets the concrete implementation (public façade is wired in
β.W, not yet landed).
"""

from __future__ import annotations

import torch

from opake.api.alignment.dpo.collator._preference import preference_collator

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

PAD = 0  # pad_token_id used throughout


def _make_example(
    chosen_ids: list[int],
    rejected_ids: list[int],
    chosen_cmask: list[int],
    rejected_cmask: list[int],
    *,
    ref_chosen_logps: float | None = None,
    ref_rejected_logps: float | None = None,
) -> dict:
    """Construct a minimal preference example dict."""
    ex: dict = {
        "chosen_input_ids": chosen_ids,
        "rejected_input_ids": rejected_ids,
        "chosen_completion_mask": chosen_cmask,
        "rejected_completion_mask": rejected_cmask,
    }
    if ref_chosen_logps is not None:
        ex["ref_chosen_logps"] = ref_chosen_logps
    if ref_rejected_logps is not None:
        ex["ref_rejected_logps"] = ref_rejected_logps
    return ex


# ---------------------------------------------------------------------------
# Core padding tests
# ---------------------------------------------------------------------------


def test_padding_chosen_and_rejected_independently() -> None:
    """Chosen and rejected pads to their *own* per-side maxima (not a shared L).

    This is the central ``(B, ...)`` layout assertion (risk α1): ``Lc`` and
    ``Lr`` may differ.  If the collator accidentally concatenated the two sides,
    both dimensions would equal ``max(Lc, Lr)``.
    """
    # Chosen sequences: lengths 3, 5  → Lc = 5
    # Rejected sequences: lengths 2, 2 → Lr = 2
    batch = [
        _make_example(
            chosen_ids=[1, 2, 3],
            rejected_ids=[10, 11],
            chosen_cmask=[0, 1, 1],
            rejected_cmask=[0, 1],
        ),
        _make_example(
            chosen_ids=[1, 2, 3, 4, 5],
            rejected_ids=[20, 21],
            chosen_cmask=[0, 0, 1, 1, 1],
            rejected_cmask=[0, 1],
        ),
    ]
    collate = preference_collator(PAD, max_length=16)
    out = collate(batch)

    assert out["chosen_input_ids"].shape == (2, 5), "Lc should be 5"
    assert out["rejected_input_ids"].shape == (2, 2), "Lr should be 2"
    assert out["chosen_attention_mask"].shape == (2, 5)
    assert out["rejected_attention_mask"].shape == (2, 2)
    assert out["chosen_completion_mask"].shape == (2, 5)
    assert out["rejected_completion_mask"].shape == (2, 2)


def test_padding_values_chosen_side() -> None:
    """Pad positions on the chosen side are filled with ``pad_token_id``."""
    batch = [
        _make_example(
            chosen_ids=[1, 2, 3],
            rejected_ids=[10],
            chosen_cmask=[0, 1, 1],
            rejected_cmask=[1],
        ),
        _make_example(
            chosen_ids=[4, 5],
            rejected_ids=[20],
            chosen_cmask=[1, 1],
            rejected_cmask=[1],
        ),
    ]
    collate = preference_collator(PAD, max_length=16)
    out = collate(batch)

    # chosen side: Lc = 3
    # Row 0: [1, 2, 3] — no padding
    assert out["chosen_input_ids"][0].tolist() == [1, 2, 3]
    # Row 1: [4, 5, PAD] — one pad position
    assert out["chosen_input_ids"][1].tolist() == [4, 5, PAD]
    # Attention mask: 1 for real tokens, 0 for pad
    assert out["chosen_attention_mask"][0].tolist() == [1, 1, 1]
    assert out["chosen_attention_mask"][1].tolist() == [1, 1, 0]
    # Completion mask for row 1 is padded with 0
    assert out["chosen_completion_mask"][1, 2].item() == 0


def test_padding_values_rejected_side() -> None:
    """Pad positions on the rejected side are filled with ``pad_token_id``."""
    batch = [
        _make_example(
            chosen_ids=[1],
            rejected_ids=[10, 11, 12],
            chosen_cmask=[1],
            rejected_cmask=[0, 1, 1],
        ),
        _make_example(
            chosen_ids=[2],
            rejected_ids=[20],
            chosen_cmask=[1],
            rejected_cmask=[1],
        ),
    ]
    collate = preference_collator(PAD, max_length=16)
    out = collate(batch)

    # rejected side: Lr = 3
    assert out["rejected_input_ids"][0].tolist() == [10, 11, 12]
    assert out["rejected_input_ids"][1].tolist() == [20, PAD, PAD]
    assert out["rejected_attention_mask"][1].tolist() == [1, 0, 0]
    # Completion mask padded with 0
    assert out["rejected_completion_mask"][1, 1].item() == 0
    assert out["rejected_completion_mask"][1, 2].item() == 0


def test_three_pairs_differing_lengths() -> None:
    """B=3 batch: chosen and rejected sides pad independently to their maxima."""
    batch = [
        _make_example(
            chosen_ids=[1, 2],
            rejected_ids=[10, 11, 12, 13],
            chosen_cmask=[0, 1],
            rejected_cmask=[0, 0, 1, 1],
        ),
        _make_example(
            chosen_ids=[3, 4, 5, 6],
            rejected_ids=[20, 21],
            chosen_cmask=[0, 1, 1, 1],
            rejected_cmask=[0, 1],
        ),
        _make_example(
            chosen_ids=[7, 8, 9],
            rejected_ids=[30, 31, 32],
            chosen_cmask=[0, 0, 1],
            rejected_cmask=[0, 1, 1],
        ),
    ]
    collate = preference_collator(PAD, max_length=32)
    out = collate(batch)

    # Lc = max(2, 4, 3) = 4; Lr = max(4, 2, 3) = 4
    assert out["chosen_input_ids"].shape == (3, 4)
    assert out["rejected_input_ids"].shape == (3, 4)

    # Spot-check: row 0 chosen is padded from len=2 to Lc=4
    assert out["chosen_input_ids"][0, 2].item() == PAD
    assert out["chosen_input_ids"][0, 3].item() == PAD
    assert out["chosen_attention_mask"][0, 2].item() == 0

    # Spot-check: row 1 rejected is padded from len=2 to Lr=4
    assert out["rejected_input_ids"][1, 2].item() == PAD
    assert out["rejected_attention_mask"][1, 2].item() == 0
    assert out["rejected_completion_mask"][1, 2].item() == 0


# ---------------------------------------------------------------------------
# Optional ref_*_logps keys
# ---------------------------------------------------------------------------


def test_ref_logps_absent_when_not_in_inputs() -> None:
    """No ``ref_chosen_logps`` / ``ref_rejected_logps`` when inputs lack them."""
    batch = [
        _make_example([1, 2], [10], [0, 1], [1]),
        _make_example([3], [20, 21], [1], [0, 1]),
    ]
    collate = preference_collator(PAD, max_length=16)
    out = collate(batch)

    assert "ref_chosen_logps" not in out
    assert "ref_rejected_logps" not in out


def test_ref_logps_present_when_all_inputs_carry_them() -> None:
    """Both ``ref_*_logps`` appear (shape ``(B,)``) when all examples carry them."""
    batch = [
        _make_example(
            [1, 2],
            [10],
            [0, 1],
            [1],
            ref_chosen_logps=-1.5,
            ref_rejected_logps=-2.0,
        ),
        _make_example(
            [3],
            [20, 21],
            [1],
            [0, 1],
            ref_chosen_logps=-0.8,
            ref_rejected_logps=-3.1,
        ),
    ]
    collate = preference_collator(PAD, max_length=16)
    out = collate(batch)

    assert "ref_chosen_logps" in out
    assert "ref_rejected_logps" in out
    assert out["ref_chosen_logps"].shape == (2,)
    assert out["ref_rejected_logps"].shape == (2,)
    assert out["ref_chosen_logps"].dtype == torch.float32
    assert out["ref_rejected_logps"].dtype == torch.float32

    assert torch.allclose(
        out["ref_chosen_logps"],
        torch.tensor([-1.5, -0.8]),
        atol=1e-6,
    )
    assert torch.allclose(
        out["ref_rejected_logps"],
        torch.tensor([-2.0, -3.1]),
        atol=1e-6,
    )


def test_ref_chosen_logps_absent_when_only_some_inputs_carry_it() -> None:
    """``ref_chosen_logps`` omitted when only a *subset* of examples have it.

    Adversarial check: partial presence must NOT produce the key.
    """
    batch = [
        _make_example(
            [1, 2],
            [10],
            [0, 1],
            [1],
            ref_chosen_logps=-1.5,
            # ref_rejected_logps NOT provided
        ),
        _make_example(
            [3],
            [20],
            [1],
            [1],
            # ref_chosen_logps NOT provided
            ref_rejected_logps=-2.0,
        ),
    ]
    collate = preference_collator(PAD, max_length=16)
    out = collate(batch)

    # Neither key should appear: the all-or-nothing rule applies per key.
    assert "ref_chosen_logps" not in out
    assert "ref_rejected_logps" not in out


def test_ref_logps_absent_when_mixed_presence() -> None:
    """ref key absent when only *some* examples in the batch carry it."""
    batch = [
        _make_example(
            [1],
            [10],
            [1],
            [1],
            ref_chosen_logps=-1.0,
            ref_rejected_logps=-2.0,
        ),
        _make_example(
            [2],
            [20],
            [1],
            [1],
            # No ref keys for this example
        ),
    ]
    collate = preference_collator(PAD, max_length=16)
    out = collate(batch)

    assert "ref_chosen_logps" not in out
    assert "ref_rejected_logps" not in out


# ---------------------------------------------------------------------------
# Determinism and key-set stability
# ---------------------------------------------------------------------------


def _make_standard_batch() -> list[dict]:
    """A reusable 2-pair batch for determinism / key-set tests."""
    return [
        _make_example(
            chosen_ids=[1, 2, 3],
            rejected_ids=[10, 11],
            chosen_cmask=[0, 1, 1],
            rejected_cmask=[0, 1],
            ref_chosen_logps=-1.0,
            ref_rejected_logps=-2.0,
        ),
        _make_example(
            chosen_ids=[4, 5],
            rejected_ids=[20, 21, 22],
            chosen_cmask=[1, 1],
            rejected_cmask=[0, 1, 1],
            ref_chosen_logps=-0.5,
            ref_rejected_logps=-3.0,
        ),
    ]


def test_determinism_same_input_equal_output() -> None:
    """Calling the collator twice on the same input yields ``torch.equal`` tensors."""
    batch = _make_standard_batch()
    collate = preference_collator(PAD, max_length=16)

    out1 = collate(batch)
    out2 = collate(batch)

    for key in out1:
        assert torch.equal(out1[key], out2[key]), f"Mismatch on key '{key}'"


def test_key_set_stability() -> None:
    """The output key set is stable across calls and matches the expected schema."""
    batch = _make_standard_batch()
    collate = preference_collator(PAD, max_length=16)

    expected_keys = {
        "chosen_input_ids",
        "chosen_attention_mask",
        "chosen_completion_mask",
        "rejected_input_ids",
        "rejected_attention_mask",
        "rejected_completion_mask",
        "ref_chosen_logps",
        "ref_rejected_logps",
    }

    out1 = collate(batch)
    out2 = collate(batch)

    assert set(out1.keys()) == expected_keys
    assert set(out2.keys()) == expected_keys


def test_key_set_without_ref_logps() -> None:
    """Without ref logps, exactly the six mandatory keys are present."""
    batch = [
        _make_example([1, 2], [10], [0, 1], [1]),
        _make_example([3], [20, 21], [1], [0, 1]),
    ]
    collate = preference_collator(PAD, max_length=16)
    out = collate(batch)

    expected_keys = {
        "chosen_input_ids",
        "chosen_attention_mask",
        "chosen_completion_mask",
        "rejected_input_ids",
        "rejected_attention_mask",
        "rejected_completion_mask",
    }
    assert set(out.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def test_truncation_to_max_length_chosen() -> None:
    """Chosen sequences longer than ``max_length`` are truncated (keep-start)."""
    long_ids = list(range(1, 11))  # length 10
    batch = [
        _make_example(
            chosen_ids=long_ids,
            rejected_ids=[99],
            chosen_cmask=[0] * 5 + [1] * 5,
            rejected_cmask=[1],
        ),
    ]
    collate = preference_collator(PAD, max_length=6)
    out = collate(batch)

    assert out["chosen_input_ids"].shape == (1, 6)
    # Truncated to first 6 tokens: [1, 2, 3, 4, 5, 6]
    assert out["chosen_input_ids"][0].tolist() == [1, 2, 3, 4, 5, 6]
    # Completion mask also truncated: [0, 0, 0, 0, 0, 1]
    assert out["chosen_completion_mask"][0].tolist() == [0, 0, 0, 0, 0, 1]


def test_truncation_to_max_length_rejected() -> None:
    """Rejected sequences longer than ``max_length`` are truncated (keep-start)."""
    long_ids = list(range(100, 110))  # length 10
    batch = [
        _make_example(
            chosen_ids=[1],
            rejected_ids=long_ids,
            chosen_cmask=[1],
            rejected_cmask=[0] * 3 + [1] * 7,
        ),
    ]
    collate = preference_collator(PAD, max_length=4)
    out = collate(batch)

    assert out["rejected_input_ids"].shape == (1, 4)
    assert out["rejected_input_ids"][0].tolist() == [100, 101, 102, 103]
    assert out["rejected_completion_mask"][0].tolist() == [0, 0, 0, 1]


def test_truncation_shorter_than_max_no_extra_padding() -> None:
    """Sequences shorter than ``max_length`` are not padded beyond the batch max."""
    batch = [
        _make_example([1, 2], [10], [0, 1], [1]),
    ]
    collate = preference_collator(PAD, max_length=100)
    out = collate(batch)

    # Single example: Lc = 2, Lr = 1 — not padded to max_length=100.
    assert out["chosen_input_ids"].shape == (1, 2)
    assert out["rejected_input_ids"].shape == (1, 1)


# ---------------------------------------------------------------------------
# pad_to_multiple_of
# ---------------------------------------------------------------------------


def test_pad_to_multiple_of_chosen_independently() -> None:
    """``pad_to_multiple_of`` rounds chosen side up independently of rejected."""
    # Chosen batch max = 3 → round up to 4 (multiple of 4)
    # Rejected batch max = 5 → round up to 8 (multiple of 4)
    batch = [
        _make_example(
            chosen_ids=[1, 2, 3],
            rejected_ids=[10, 11, 12, 13, 14],
            chosen_cmask=[0, 1, 1],
            rejected_cmask=[0, 0, 1, 1, 1],
        ),
    ]
    collate = preference_collator(PAD, max_length=32, pad_to_multiple_of=4)
    out = collate(batch)

    assert out["chosen_input_ids"].shape[1] == 4  # ceil(3/4)*4
    assert out["rejected_input_ids"].shape[1] == 8  # ceil(5/4)*4


def test_pad_to_multiple_of_exact_multiple_unchanged() -> None:
    """When the batch max is already a multiple, shape stays unchanged."""
    # Chosen batch max = 4 (already a multiple of 4) → no extra pad
    batch = [
        _make_example(
            chosen_ids=[1, 2, 3, 4],
            rejected_ids=[10, 11],
            chosen_cmask=[0, 0, 1, 1],
            rejected_cmask=[0, 1],
        ),
    ]
    collate = preference_collator(PAD, max_length=32, pad_to_multiple_of=4)
    out = collate(batch)

    assert out["chosen_input_ids"].shape[1] == 4


def test_pad_to_multiple_of_none_no_rounding() -> None:
    """``pad_to_multiple_of=None`` leaves the batch max dimension unchanged."""
    batch = [
        _make_example(
            chosen_ids=[1, 2, 3],
            rejected_ids=[10, 11],
            chosen_cmask=[0, 1, 1],
            rejected_cmask=[0, 1],
        ),
    ]
    collate = preference_collator(PAD, max_length=32, pad_to_multiple_of=None)
    out = collate(batch)

    assert out["chosen_input_ids"].shape[1] == 3
    assert out["rejected_input_ids"].shape[1] == 2


def test_pad_to_multiple_of_completion_mask_zero_padded() -> None:
    """Completion-mask pad positions added by ``pad_to_multiple_of`` are 0."""
    # Chosen batch max = 3 → rounded to 8 (multiple of 8).
    # Positions 3..7 must be 0 in the completion mask.
    batch = [
        _make_example(
            chosen_ids=[1, 2, 3],
            rejected_ids=[10],
            chosen_cmask=[0, 1, 1],
            rejected_cmask=[1],
        ),
    ]
    collate = preference_collator(PAD, max_length=32, pad_to_multiple_of=8)
    out = collate(batch)

    assert out["chosen_input_ids"].shape[1] == 8
    # Original 3 completion-mask values, then 5 zeros
    assert out["chosen_completion_mask"][0].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]


# ---------------------------------------------------------------------------
# Chosen vs rejected independence — adversarial length mismatch checks
# ---------------------------------------------------------------------------


def test_chosen_longer_than_rejected() -> None:
    """Lc > Lr: the two sides have different final widths with no cross-contamination."""
    batch = [
        _make_example(
            chosen_ids=[1, 2, 3, 4, 5, 6],
            rejected_ids=[10, 11],
            chosen_cmask=[0, 0, 0, 1, 1, 1],
            rejected_cmask=[0, 1],
        ),
    ]
    collate = preference_collator(PAD, max_length=32)
    out = collate(batch)

    assert out["chosen_input_ids"].shape == (1, 6)
    assert out["rejected_input_ids"].shape == (1, 2)
    # These shapes must differ — no shared padding dimension.
    assert out["chosen_input_ids"].shape[1] != out["rejected_input_ids"].shape[1]


def test_rejected_longer_than_chosen() -> None:
    """Lr > Lc: symmetric variant of the length-mismatch test."""
    batch = [
        _make_example(
            chosen_ids=[1, 2],
            rejected_ids=[10, 11, 12, 13, 14, 15],
            chosen_cmask=[0, 1],
            rejected_cmask=[0, 0, 0, 1, 1, 1],
        ),
    ]
    collate = preference_collator(PAD, max_length=32)
    out = collate(batch)

    assert out["chosen_input_ids"].shape == (1, 2)
    assert out["rejected_input_ids"].shape == (1, 6)
    assert out["chosen_input_ids"].shape[1] != out["rejected_input_ids"].shape[1]


def test_lengths_vary_within_batch_both_sides() -> None:
    """Within-batch length variance on both sides is handled per-side, not globally."""
    # Chosen lengths: [1, 4] → Lc = 4
    # Rejected lengths: [3, 2] → Lr = 3
    batch = [
        _make_example(
            chosen_ids=[1],
            rejected_ids=[10, 11, 12],
            chosen_cmask=[1],
            rejected_cmask=[0, 1, 1],
        ),
        _make_example(
            chosen_ids=[2, 3, 4, 5],
            rejected_ids=[20, 21],
            chosen_cmask=[0, 1, 1, 1],
            rejected_cmask=[0, 1],
        ),
    ]
    collate = preference_collator(PAD, max_length=32)
    out = collate(batch)

    assert out["chosen_input_ids"].shape == (2, 4)
    assert out["rejected_input_ids"].shape == (2, 3)


# ---------------------------------------------------------------------------
# Attention mask correctness
# ---------------------------------------------------------------------------


def test_attention_mask_matches_non_pad_positions() -> None:
    """attention_mask is 1 exactly where input_ids != pad_token_id."""
    batch = [
        _make_example(
            chosen_ids=[5, 6, 7],
            rejected_ids=[50, 51],
            chosen_cmask=[0, 1, 1],
            rejected_cmask=[0, 1],
        ),
        _make_example(
            chosen_ids=[8],
            rejected_ids=[60, 61, 62],
            chosen_cmask=[1],
            rejected_cmask=[0, 1, 1],
        ),
    ]
    collate = preference_collator(PAD, max_length=32)
    out = collate(batch)

    # Chosen: non-pad positions should match attention mask = 1.
    for i in range(2):
        ids = out["chosen_input_ids"][i]
        mask = out["chosen_attention_mask"][i]
        assert torch.equal(mask, (ids != PAD).long())

    # Rejected: same check.
    for i in range(2):
        ids = out["rejected_input_ids"][i]
        mask = out["rejected_attention_mask"][i]
        assert torch.equal(mask, (ids != PAD).long())


# ---------------------------------------------------------------------------
# Completion mask correctness
# ---------------------------------------------------------------------------


def test_completion_mask_zero_at_pad_positions() -> None:
    """Completion mask is 0 at all padding positions."""
    batch = [
        _make_example(
            chosen_ids=[1, 2, 3],
            rejected_ids=[10, 11, 12, 13],
            chosen_cmask=[0, 1, 1],
            rejected_cmask=[0, 0, 1, 1],
        ),
        _make_example(
            chosen_ids=[4],
            rejected_ids=[20],
            chosen_cmask=[1],
            rejected_cmask=[1],
        ),
    ]
    collate = preference_collator(PAD, max_length=32)
    out = collate(batch)

    # Row 1, chosen: only position 0 is real; positions 1..2 are pad.
    assert out["chosen_completion_mask"][1, 1].item() == 0
    assert out["chosen_completion_mask"][1, 2].item() == 0

    # Row 1, rejected: only position 0 is real; positions 1..3 are pad.
    for j in range(1, 4):
        assert out["rejected_completion_mask"][1, j].item() == 0


# ---------------------------------------------------------------------------
# Output tensor dtypes
# ---------------------------------------------------------------------------


def test_output_tensor_dtypes() -> None:
    """All mandatory output tensors are ``torch.long`` (int64)."""
    batch = _make_standard_batch()
    collate = preference_collator(PAD, max_length=16)
    out = collate(batch)

    long_keys = [
        "chosen_input_ids",
        "chosen_attention_mask",
        "chosen_completion_mask",
        "rejected_input_ids",
        "rejected_attention_mask",
        "rejected_completion_mask",
    ]
    for key in long_keys:
        assert out[key].dtype == torch.long, f"Expected torch.long for '{key}'"

    assert out["ref_chosen_logps"].dtype == torch.float32
    assert out["ref_rejected_logps"].dtype == torch.float32
