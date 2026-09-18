"""Unit tests for :func:`extract_prompt` (plan §7.7, work-unit β.5).

Tests cover the full behavioural contract of the heuristic:

- Conversational pair with a shared 2-message prefix → prompt is that
  prefix, suffixes are the divergent tails.
- Token-list pair (lists of ints) → common prefix extraction.
- Idempotency: example already containing ``"prompt"`` returned unchanged.
- No common prefix (first element differs) → empty ``prompt``, full
  chosen/rejected as suffixes.
- Extra keys preserved through the transform.
- Adversarial cases: fully-identical chosen/rejected; one list shorter than
  the other; non-list ``chosen``/``rejected`` without a prompt key.
"""

from __future__ import annotations

import pytest

from opake.api.alignment.dpo.data._prompt import extract_prompt

# ---------------------------------------------------------------------------
# Conversational preference pair — 2-message shared prefix
# ---------------------------------------------------------------------------

SYSTEM_MSG = {"role": "system", "content": "You are helpful."}
USER_MSG = {"role": "user", "content": "What is 2 + 2?"}
CHOSEN_REPLY = {"role": "assistant", "content": "4"}
REJECTED_REPLY = {"role": "assistant", "content": "5"}


def test_conversational_pair_two_message_prefix() -> None:
    """Chosen and rejected share a 2-message prefix; both have 1-message suffix."""
    example = {
        "chosen": [SYSTEM_MSG, USER_MSG, CHOSEN_REPLY],
        "rejected": [SYSTEM_MSG, USER_MSG, REJECTED_REPLY],
    }
    result = extract_prompt(example)

    assert result["prompt"] == [SYSTEM_MSG, USER_MSG]
    assert result["chosen"] == [CHOSEN_REPLY]
    assert result["rejected"] == [REJECTED_REPLY]


def test_conversational_pair_one_message_prefix() -> None:
    """Only the first message is shared; divergence at the second message."""
    chosen_turn2 = {"role": "user", "content": "Follow-up A"}
    rejected_turn2 = {"role": "user", "content": "Follow-up B"}
    example = {
        "chosen": [USER_MSG, chosen_turn2, CHOSEN_REPLY],
        "rejected": [USER_MSG, rejected_turn2, REJECTED_REPLY],
    }
    result = extract_prompt(example)

    assert result["prompt"] == [USER_MSG]
    assert result["chosen"] == [chosen_turn2, CHOSEN_REPLY]
    assert result["rejected"] == [rejected_turn2, REJECTED_REPLY]


# ---------------------------------------------------------------------------
# Token-list pair (lists of ints)
# ---------------------------------------------------------------------------


def test_token_list_pair_common_prefix() -> None:
    """Prefix of integer token lists is extracted correctly."""
    example = {
        "chosen": [1, 2, 3, 10, 11],
        "rejected": [1, 2, 3, 20, 21],
    }
    result = extract_prompt(example)

    assert result["prompt"] == [1, 2, 3]
    assert result["chosen"] == [10, 11]
    assert result["rejected"] == [20, 21]


def test_token_list_no_common_prefix() -> None:
    """First element differs → empty prompt, full lists as suffixes."""
    example = {
        "chosen": [10, 1, 2],
        "rejected": [20, 1, 2],
    }
    result = extract_prompt(example)

    assert result["prompt"] == []
    assert result["chosen"] == [10, 1, 2]
    assert result["rejected"] == [20, 1, 2]


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_idempotent_when_prompt_present() -> None:
    """If the example already has ``"prompt"``, it is returned unchanged."""
    original = {
        "prompt": [USER_MSG],
        "chosen": [CHOSEN_REPLY],
        "rejected": [REJECTED_REPLY],
    }
    result = extract_prompt(original)
    # Must be the identical object (no copying).
    assert result is original


def test_idempotent_with_empty_prompt() -> None:
    """Even an empty ``"prompt"`` key makes the example return unchanged."""
    original = {
        "prompt": [],
        "chosen": [1, 2],
        "rejected": [1, 3],
    }
    result = extract_prompt(original)
    assert result is original


# ---------------------------------------------------------------------------
# Extra keys preserved
# ---------------------------------------------------------------------------


def test_extra_keys_preserved() -> None:
    """Arbitrary extra keys survive the transform without modification."""
    example = {
        "chosen": [1, 2, 10],
        "rejected": [1, 2, 20],
        "source": "dataset_v1",
        "weight": 0.9,
        "meta": {"lang": "en"},
    }
    result = extract_prompt(example)

    assert result["source"] == "dataset_v1"
    assert result["weight"] == pytest.approx(0.9)
    assert result["meta"] == {"lang": "en"}
    # Sanity-check the primary transform still happened.
    assert result["prompt"] == [1, 2]
    assert result["chosen"] == [10]
    assert result["rejected"] == [20]


# ---------------------------------------------------------------------------
# Adversarial: fully identical chosen and rejected
# ---------------------------------------------------------------------------


def test_fully_identical_chosen_rejected() -> None:
    """When chosen == rejected the entire list is the prompt; suffixes are empty."""
    example = {
        "chosen": [1, 2, 3],
        "rejected": [1, 2, 3],
    }
    result = extract_prompt(example)

    assert result["prompt"] == [1, 2, 3]
    assert result["chosen"] == []
    assert result["rejected"] == []


# ---------------------------------------------------------------------------
# Adversarial: one list shorter than the other
# ---------------------------------------------------------------------------


def test_chosen_shorter_than_rejected() -> None:
    """Common prefix stops at the end of the shorter list (zip semantics)."""
    example = {
        "chosen": [1, 2],
        "rejected": [1, 2, 3, 4],
    }
    result = extract_prompt(example)

    # zip stops at len(chosen) == 2; all elements equal → prefix is [1, 2].
    assert result["prompt"] == [1, 2]
    assert result["chosen"] == []
    assert result["rejected"] == [3, 4]


def test_rejected_shorter_than_chosen() -> None:
    """Common prefix stops at the end of the shorter list (zip semantics)."""
    example = {
        "chosen": [1, 2, 5, 6],
        "rejected": [1, 2],
    }
    result = extract_prompt(example)

    assert result["prompt"] == [1, 2]
    assert result["chosen"] == [5, 6]
    assert result["rejected"] == []


def test_chosen_shorter_with_mismatch_before_end() -> None:
    """Mismatch before the shorter list ends → prefix stops at mismatch."""
    example = {
        "chosen": [1, 99],
        "rejected": [1, 2, 3],
    }
    result = extract_prompt(example)

    assert result["prompt"] == [1]
    assert result["chosen"] == [99]
    assert result["rejected"] == [2, 3]


# ---------------------------------------------------------------------------
# Adversarial: non-list inputs without a prompt key
# ---------------------------------------------------------------------------


def test_unpaired_example_no_chosen_rejected_returned_unchanged() -> None:
    """Unpaired example (prompt + completion, no chosen/rejected) is unchanged."""
    original = {
        "prompt": "Who are you?",
        "completion": "I am an AI.",
    }
    result = extract_prompt(original)
    assert result is original


def test_missing_chosen_key_returned_unchanged() -> None:
    """Example with only 'rejected' (no 'chosen') is returned unchanged."""
    original = {"rejected": [1, 2, 3]}
    result = extract_prompt(original)
    assert result is original


def test_missing_rejected_key_returned_unchanged() -> None:
    """Example with only 'chosen' (no 'rejected') is returned unchanged."""
    original = {"chosen": [1, 2, 3]}
    result = extract_prompt(original)
    assert result is original


# ---------------------------------------------------------------------------
# Empty lists edge case
# ---------------------------------------------------------------------------


def test_both_empty_lists() -> None:
    """Both chosen and rejected are empty → prompt and suffixes are all empty."""
    example = {
        "chosen": [],
        "rejected": [],
    }
    result = extract_prompt(example)

    assert result["prompt"] == []
    assert result["chosen"] == []
    assert result["rejected"] == []


def test_one_empty_one_nonempty() -> None:
    """One empty list, one non-empty → prefix is empty (zip yields nothing)."""
    example = {
        "chosen": [],
        "rejected": [1, 2, 3],
    }
    result = extract_prompt(example)

    assert result["prompt"] == []
    assert result["chosen"] == []
    assert result["rejected"] == [1, 2, 3]
