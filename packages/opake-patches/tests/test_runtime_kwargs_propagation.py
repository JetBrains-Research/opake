"""Regression: ``apply_model_patches`` must propagate ``performance`` /
``compat`` / per-flag kwargs to its auto-triggered ``apply_runtime_patches``
call.

Before the fix, the auto-call was ``apply_runtime_patches()`` with no kwargs,
so ``apply_model_patches(model, compat=False)`` silently enabled all runtime
compat patches (``vmap_masking``, ``empty_batches``, ``vmap_checkpointing``)
anyway — defeating the public-API contract that the group flags imply.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_runtime_flag():
    import opake.api.patches as patches_init

    saved = patches_init._runtime_patches_applied
    patches_init._runtime_patches_applied = False
    yield
    patches_init._runtime_patches_applied = saved


class _StubCfg:
    model_type = "definitely-not-registered-runtime-propagation-test"


class _StubModel:
    config = _StubCfg()


def test_apply_model_patches_propagates_compat_false_to_runtime(monkeypatch):
    """``compat=False`` on the model entrypoint must reach the runtime entrypoint."""
    import opake.api.patches as patches_init

    captured: dict = {}

    def fake_apply_runtime_patches(**kwargs):
        captured.update(kwargs)
        patches_init._runtime_patches_applied = True

    monkeypatch.setattr(
        patches_init, "apply_runtime_patches", fake_apply_runtime_patches
    )

    patches_init.apply_model_patches(_StubModel(), compat=False, peft=False)

    assert captured.get("compat") is False
    assert captured.get("performance") is True


def test_apply_model_patches_propagates_performance_false_to_runtime(monkeypatch):
    import opake.api.patches as patches_init

    captured: dict = {}

    def fake_apply_runtime_patches(**kwargs):
        captured.update(kwargs)
        patches_init._runtime_patches_applied = True

    monkeypatch.setattr(
        patches_init, "apply_runtime_patches", fake_apply_runtime_patches
    )

    patches_init.apply_model_patches(_StubModel(), performance=False, peft=False)

    assert captured.get("performance") is False
    assert captured.get("compat") is True


def test_apply_model_patches_propagates_explicit_runtime_kwargs(monkeypatch):
    """Per-flag overrides (e.g. ``vmap_checkpointing=False``) pass through too."""
    import opake.api.patches as patches_init

    captured: dict = {}

    def fake_apply_runtime_patches(**kwargs):
        captured.update(kwargs)
        patches_init._runtime_patches_applied = True

    monkeypatch.setattr(
        patches_init, "apply_runtime_patches", fake_apply_runtime_patches
    )

    patches_init.apply_model_patches(
        _StubModel(),
        peft=False,
        vmap_checkpointing=False,
        empty_batches=False,
    )

    assert captured.get("vmap_checkpointing") is False
    assert captured.get("empty_batches") is False


def test_apply_model_patches_compat_false_does_not_enable_runtime_patches(monkeypatch):
    """End-to-end: when compat=False, none of the three runtime sub-patches fire."""
    import opake.api.patches as patches_init
    import opake.api.patches.torch as torch_pkg
    from opake.api.patches.transformers.runtime import (
        collator as collator_runtime,
    )
    from opake.api.patches.transformers.runtime import (
        masking as masking_runtime,
    )

    calls: list[str] = []

    monkeypatch.setattr(
        masking_runtime,
        "apply_masking_patches",
        lambda **kw: calls.append("masking"),
    )
    monkeypatch.setattr(
        collator_runtime,
        "apply_collator_patches",
        lambda **kw: calls.append("collator"),
    )
    monkeypatch.setattr(
        torch_pkg,
        "apply_checkpoint_patch",
        lambda **kw: calls.append("checkpoint"),
    )

    patches_init.apply_model_patches(_StubModel(), compat=False, peft=False)

    assert calls == [], (
        f"compat=False should suppress all runtime compat patches; got {calls}"
    )


def test_apply_model_patches_default_compat_enables_runtime_patches(monkeypatch):
    """Sanity: default behavior (compat=True) still applies the runtime patches."""
    import opake.api.patches as patches_init
    import opake.api.patches.torch as torch_pkg
    from opake.api.patches.transformers.runtime import (
        collator as collator_runtime,
    )
    from opake.api.patches.transformers.runtime import (
        masking as masking_runtime,
    )

    calls: list[str] = []

    monkeypatch.setattr(
        masking_runtime,
        "apply_masking_patches",
        lambda **kw: calls.append("masking"),
    )
    monkeypatch.setattr(
        collator_runtime,
        "apply_collator_patches",
        lambda **kw: calls.append("collator"),
    )
    monkeypatch.setattr(
        torch_pkg,
        "apply_checkpoint_patch",
        lambda **kw: calls.append("checkpoint"),
    )

    patches_init.apply_model_patches(_StubModel(), peft=False)

    assert set(calls) == {"masking", "collator", "checkpoint"}


# Which patches apply for which torch regime is covered by
# tests/torch/test_checkpoint_gating.py.
