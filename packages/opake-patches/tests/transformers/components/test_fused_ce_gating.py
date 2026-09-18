# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fused linear+CE eligibility (Phase 2 loss parity)."""

import torch

from opake.api.patches.transformers.components.cross_entropy import (
    _fused_linear_ce_loss_is_supported,
)


def test_fused_ce_allowed_full_sequence_no_extra_loss_kwargs():
    assert _fused_linear_ce_loss_is_supported(0, {}) is True
    assert _fused_linear_ce_loss_is_supported(0, {"ignore_index": -100}) is True


def test_fused_ce_blocked_logit_slice():
    assert _fused_linear_ce_loss_is_supported(1, {}) is False


def test_fused_ce_blocked_tensor_logit_slice():
    assert _fused_linear_ce_loss_is_supported(torch.tensor(0), {}) is False


def test_fused_ce_blocked_shift_labels():
    assert (
        _fused_linear_ce_loss_is_supported(0, {"shift_labels": torch.tensor([1])})
        is False
    )


def test_fused_ce_allows_nonzero_label_smoothing():
    assert _fused_linear_ce_loss_is_supported(0, {"label_smoothing": 0.1}) is True


def test_fused_ce_blocked_class_weight():
    w = torch.ones(5)
    assert _fused_linear_ce_loss_is_supported(0, {"weight": w}) is False


def test_fused_ce_blocked_ignore_index_tensor():
    assert (
        _fused_linear_ce_loss_is_supported(0, {"ignore_index": torch.tensor(-100)})
        is False
    )


def test_label_smoothing_zero_still_allowed():
    assert _fused_linear_ce_loss_is_supported(0, {"label_smoothing": 0.0}) is True
