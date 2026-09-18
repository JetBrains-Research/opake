# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for generation-cache handling."""

from types import SimpleNamespace

from opake.api.patches.transformers.components.kv_cache import _disable_kv_cache


def test_empty_legacy_cache_is_disabled_during_training():
    seen: dict[str, object] = {}

    def forward(_model, **kwargs):
        seen.update(kwargs)

    wrapped = _disable_kv_cache(forward)
    wrapped(SimpleNamespace(training=True), past_key_values=(), use_cache=True)

    assert seen["use_cache"] is False
