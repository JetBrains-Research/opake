# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""``_microbatch_transform``: owner-supplied per-microbatch argument rewrite."""

from __future__ import annotations

import pytest
import torch

from opake.api.engine.clipping import clipped_grad
from opake.exceptions import ConfigurationError, OperationError


def _loss(w, x):
    # Uses only the first x.shape[-1] weights, so dropping all-zero trailing
    # feature columns leaves every per-example gradient unchanged.
    return (x @ w[: x.shape[-1]]) ** 2


def _drop_zero_columns(calls):
    def transform(args):
        w, x = args
        keep = int(torch.nonzero(x.abs().sum(0)).max()) + 1
        calls.append((x.shape[0], x.shape[1], keep))
        return (w, x[:, :keep])

    return transform


def _batch():
    torch.manual_seed(0)
    x = torch.zeros(5, 8)
    for r, length in enumerate([2, 3, 6, 4, 1]):
        x[r, :length] = torch.randn(length)
    return torch.randn(8), x


def test_output_preserving_transform_runs_per_microbatch_and_matches():
    w, x = _batch()
    base_fn, state = clipped_grad(_loss, clipping_norm=0.5, microbatch_size=2)
    base, _ = base_fn(w, x, state=state)
    calls: list[tuple[int, int, int]] = []
    fn, state = clipped_grad(
        _loss,
        clipping_norm=0.5,
        microbatch_size=2,
        _microbatch_transform=_drop_zero_columns(calls),
    )
    out, _ = fn(w, x, state=state)
    assert [c[0] for c in calls] == [2, 2, 1]
    assert [c[2] for c in calls] == [3, 6, 1]
    torch.testing.assert_close(out.pytree, base.pytree)
    assert out.max_norm == base.max_norm


def test_transform_must_keep_every_example():
    w, x = _batch()
    fn, state = clipped_grad(
        _loss,
        clipping_norm=0.5,
        microbatch_size=2,
        _microbatch_transform=lambda args: (args[0], args[1][:1]),
    )
    with pytest.raises(OperationError, match="keep every example"):
        fn(w, x, state=state)


def test_transform_requires_microbatches():
    with pytest.raises(ConfigurationError, match="_microbatch_transform requires"):
        clipped_grad(_loss, clipping_norm=0.5, _microbatch_transform=lambda a: a)
