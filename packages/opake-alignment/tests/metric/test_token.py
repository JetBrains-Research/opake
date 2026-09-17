"""Unit tests for the shared token-level metrics.

Hand-computed reference cases for the general LM telemetry primitives:

- :func:`entropy_from_logits` — uniform logits over ``V`` classes give
  entropy ``log(V)``; masking selects the contributing positions.
- :func:`mean_token_accuracy` — a hand-built shifted next-token case.

Every metric is telemetry, so each test also asserts the returned tensor is
detached (``requires_grad is False``) and that masked reductions do not divide
by zero on an all-masked input.
"""

from __future__ import annotations

import math

import pytest
import torch

from opake.api.alignment.metric._token import (
    entropy_from_logits,
    mean_token_accuracy,
)


# --------------------------------------------------------------------------- #
# entropy_from_logits
# --------------------------------------------------------------------------- #
def test_entropy_uniform_logits_equals_log_v() -> None:
    vocab = 7
    # Equal logits over V classes -> uniform softmax -> entropy = log(V).
    logits = torch.zeros(4, vocab)
    mask = torch.ones(4)
    ent = entropy_from_logits(logits, mask)
    assert ent.item() == pytest.approx(math.log(vocab))


def test_entropy_masked_positions_only() -> None:
    vocab = 5
    logits = torch.zeros(3, vocab)
    # One-hot-ish very peaked position should have ~0 entropy, but it is masked
    # out, so the mean must equal log(V) from the two uniform positions.
    logits[2] = torch.tensor([100.0, 0.0, 0.0, 0.0, 0.0])
    mask = torch.tensor([1.0, 1.0, 0.0])
    ent = entropy_from_logits(logits, mask)
    assert ent.item() == pytest.approx(math.log(vocab))


def test_entropy_all_masked_no_divide_by_zero() -> None:
    logits = torch.zeros(3, 6)
    mask = torch.zeros(3)
    ent = entropy_from_logits(logits, mask)
    assert ent.item() == pytest.approx(0.0)
    assert torch.isfinite(ent).item()


def test_entropy_detached() -> None:
    logits = torch.zeros(2, 4, requires_grad=True)
    mask = torch.ones(2)
    ent = entropy_from_logits(logits, mask)
    assert ent.requires_grad is False
    assert ent.ndim == 0


def test_entropy_shifts_like_accuracy() -> None:
    # Position 0 is peaked (~0 entropy), positions 1,2 are uniform (entropy
    # log V). The shift drops position 2's logits and uses mask[1:], so only
    # positions {0, 1} contribute: (0 + log V) / 2 = log(V) / 2.
    vocab = 4
    logits = torch.zeros(3, vocab)
    logits[0] = torch.tensor([100.0, 0.0, 0.0, 0.0])  # peaked -> ~0 entropy
    mask = torch.ones(3)
    ent = entropy_from_logits(logits, mask)
    assert ent.item() == pytest.approx(math.log(vocab) / 2.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# mean_token_accuracy
# --------------------------------------------------------------------------- #
def test_mean_token_accuracy_reference() -> None:
    # seq length 4, vocab 3. After shift, predictions at positions [0,1,2]
    # are compared against labels[1:] = labels[1], labels[2], labels[3].
    # Build logits so argmax along vocab is deterministic.
    logits = torch.tensor(
        [
            [
                [9.0, 0.0, 0.0],  # pos 0 -> argmax 0
                [0.0, 9.0, 0.0],  # pos 1 -> argmax 1
                [0.0, 0.0, 9.0],  # pos 2 -> argmax 2
                [9.0, 0.0, 0.0],  # pos 3 -> dropped by shift
            ]
        ]
    )
    labels = torch.tensor([[5, 0, 2, 2]])  # labels[1:] = [0, 2, 2]
    mask = torch.tensor([[1, 1, 1, 1]])
    # shifted preds = [0, 1, 2]; shifted labels = [0, 2, 2]; shifted mask = [1,1,1].
    # correct = [True, False, True] -> 2 / 3.
    acc = mean_token_accuracy(logits, labels, mask)
    assert acc.item() == pytest.approx(2.0 / 3.0)


def test_mean_token_accuracy_respects_mask() -> None:
    logits = torch.tensor(
        [
            [
                [9.0, 0.0],  # pos 0 -> argmax 0
                [9.0, 0.0],  # pos 1 -> argmax 0
                [0.0, 9.0],  # pos 2 -> argmax 1
            ]
        ]
    )
    labels = torch.tensor([[0, 0, 1]])  # labels[1:] = [0, 1]
    # shifted mask = mask[1:] = [1, 0]: only the first shifted position counts.
    mask = torch.tensor([[1, 1, 0]])
    # preds[0:2] = [0, 0]; labels[1:] = [0, 1]; only position 0 is unmasked and
    # correct -> 1 / 1 = 1.0.
    acc = mean_token_accuracy(logits, labels, mask)
    assert acc.item() == pytest.approx(1.0)


def test_mean_token_accuracy_all_masked_no_divide_by_zero() -> None:
    logits = torch.zeros(1, 4, 3)
    labels = torch.zeros(1, 4, dtype=torch.long)
    mask = torch.zeros(1, 4, dtype=torch.long)
    acc = mean_token_accuracy(logits, labels, mask)
    assert acc.item() == pytest.approx(0.0)
    assert torch.isfinite(acc).item()


def test_mean_token_accuracy_detached() -> None:
    logits = torch.zeros(1, 4, 3, requires_grad=True)
    labels = torch.zeros(1, 4, dtype=torch.long)
    mask = torch.ones(1, 4, dtype=torch.long)
    acc = mean_token_accuracy(logits, labels, mask)
    assert acc.requires_grad is False
    assert acc.ndim == 0
