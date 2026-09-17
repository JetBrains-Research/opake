"""Unit tests for :func:`null_ref_context` and :func:`with_disabled_adapter`.

Covers the four §7.8 dispatch paths:

1. **LoRA-disable path** (§7.8 row 3): ``null_ref_context(peft_model)`` —
   adapter is disabled inside the block (forward matches base model); adapter
   is active again after the block.
2. **LoRA-with-ref-adapter path** (§7.8 row 2): ``null_ref_context(peft_model)``
   when a ``"ref"`` adapter exists — active adapter is ``"ref"`` inside the
   block; original active adapter is restored on exit.
3. **Separate-model path** (§7.8 row 1): ``null_ref_context(policy,
   ref_model=other_model)`` — no-op (doesn't raise, doesn't mutate), including
   when either model is PEFT-wrapped.
4. **Non-PEFT no-ref path** (§7.8 row 4): ``null_ref_context(plain_module)``
   — no-op.
5. ``with_disabled_adapter`` no-op on a plain ``nn.Linear``.

All tests run CPU-only and use the smallest possible PEFT models (a tiny
``nn.Linear``-backed module) to keep the suite fast.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

peft = pytest.importorskip("peft")

from peft import LoraConfig, get_peft_model  # noqa: E402

from opake.api.alignment.dpo.reference._adapter import (  # noqa: E402
    null_ref_context,
    with_disabled_adapter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TinyMLP(nn.Module):
    """Minimal nn.Module that PEFT can wrap via a single linear layer."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 8, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _make_peft_model(adapter_name: str = "default") -> peft.PeftModel:
    """Return a PEFT-wrapped TinyMLP with ``init_lora_weights=False``.

    ``init_lora_weights=False`` ensures the LoRA delta is non-zero from
    construction, so forward outputs differ between the LoRA-active and
    LoRA-disabled (base model) states — making the distinction testable.
    """
    base = _TinyMLP()
    lora_cfg = LoraConfig(
        target_modules=["linear"],
        r=2,
        lora_alpha=4,
        init_lora_weights=False,
    )
    return get_peft_model(base, lora_cfg, adapter_name=adapter_name)


# ---------------------------------------------------------------------------
# §7.8 row 3 — LoRA-disable path (no separate ref_model)
# ---------------------------------------------------------------------------


def test_lora_disable_path_output_matches_base() -> None:
    """Inside null_ref_context(peft_model) the forward equals the base model."""
    model = _make_peft_model()
    x = torch.randn(2, 8)

    # Baseline: with adapter active.
    out_peft = model(x)

    # Inside context: adapter disabled → base model forward.
    inside_outputs: list[torch.Tensor] = []
    with null_ref_context(model):
        inside_outputs.append(model(x))

    # After context: adapter active again.
    out_after = model(x)

    # Compute what pure base-model output looks like.
    with model.disable_adapter():
        out_base = model(x)

    # Inside the null_ref_context the output must match the base model.
    assert torch.allclose(inside_outputs[0], out_base), (
        f"Inside null_ref_context: expected base output {out_base}, "
        f"got {inside_outputs[0]}"
    )

    # The LoRA delta must be non-trivial so this test is meaningful.
    assert not torch.allclose(out_peft, out_base), (
        "LoRA delta is zero — test is vacuous (increase r or check PEFT init)"
    )

    # After the context the adapter must be re-enabled.
    assert torch.allclose(out_after, out_peft), (
        f"After null_ref_context: expected peft output {out_peft}, got {out_after}"
    )


def test_lora_disable_path_adapter_restored_on_exit() -> None:
    """Active adapter name is restored after null_ref_context (no ref_model)."""
    model = _make_peft_model()

    active_before = list(model.active_adapters)

    with null_ref_context(model):
        pass  # just enter and exit

    active_after = list(model.active_adapters)
    assert active_after == active_before, (
        f"Active adapter not restored: {active_before!r} → {active_after!r}"
    )


def test_lora_disable_path_adapter_restored_on_exception() -> None:
    """Active adapter is restored even when the block raises."""
    model = _make_peft_model()
    active_before = list(model.active_adapters)

    with pytest.raises(RuntimeError, match="test error"), null_ref_context(model):
        raise RuntimeError("test error")

    active_after = list(model.active_adapters)
    assert active_after == active_before, "Active adapter not restored after exception"


# ---------------------------------------------------------------------------
# §7.8 row 2 — LoRA-with-ref-adapter path
# ---------------------------------------------------------------------------


def test_lora_ref_adapter_path_activates_ref() -> None:
    """Inside null_ref_context the active adapter is 'ref'."""
    model = _make_peft_model()

    # Add a second adapter named 'ref'.
    ref_lora_cfg = LoraConfig(
        target_modules=["linear"],
        r=4,
        lora_alpha=8,
        init_lora_weights=False,
    )
    model.add_adapter("ref", ref_lora_cfg)

    # Start on 'default'.
    model.set_adapter("default")
    assert list(model.active_adapters) == ["default"]

    inside_adapters: list[list[str]] = []
    with null_ref_context(model):
        inside_adapters.append(list(model.active_adapters))

    # Inside the context the ref adapter must be active.
    assert inside_adapters[0] == ["ref"], (
        f"Expected ['ref'] inside null_ref_context, got {inside_adapters[0]}"
    )

    # After the context 'default' must be restored.
    assert list(model.active_adapters) == ["default"], (
        f"Expected ['default'] after null_ref_context, got {list(model.active_adapters)}"
    )


def test_lora_ref_adapter_path_restores_on_exception() -> None:
    """Original adapter is restored when block raises (LoRA-with-ref path)."""
    model = _make_peft_model()
    ref_lora_cfg = LoraConfig(
        target_modules=["linear"],
        r=2,
        lora_alpha=4,
        init_lora_weights=False,
    )
    model.add_adapter("ref", ref_lora_cfg)
    model.set_adapter("default")

    with pytest.raises(ValueError, match="boom"), null_ref_context(model):
        raise ValueError("boom")

    assert list(model.active_adapters) == ["default"], (
        "Active adapter not restored after exception (ref-adapter path)"
    )


def test_lora_ref_adapter_output_differs_from_default() -> None:
    """Forwards with 'default' and 'ref' adapters produce different results."""
    model = _make_peft_model()
    ref_lora_cfg = LoraConfig(
        target_modules=["linear"],
        r=4,
        lora_alpha=8,
        init_lora_weights=False,
    )
    model.add_adapter("ref", ref_lora_cfg)
    model.set_adapter("default")

    x = torch.randn(2, 8)
    out_default = model(x)

    with null_ref_context(model):
        out_ref = model(x)

    # The two adapters have different random weights → different outputs.
    assert not torch.allclose(out_default, out_ref), (
        "Expected different outputs for 'default' and 'ref' adapters"
    )


# ---------------------------------------------------------------------------
# §7.8 row 1 — separate-model path (no-op)
# ---------------------------------------------------------------------------


def test_separate_model_path_noop() -> None:
    """null_ref_context with a plain ref_model doesn't raise and doesn't mutate."""
    policy = _TinyMLP()
    ref = _TinyMLP()

    x = torch.randn(2, 8)
    out_before = policy(x).detach().clone()

    with null_ref_context(policy, ref_model=ref):
        out_inside = policy(x)

    out_after = policy(x)

    # Outputs must be identical (nothing changed).
    assert torch.allclose(out_inside, out_before), "Separate-model path mutated policy"
    assert torch.allclose(out_after, out_before), "Separate-model path mutated policy"


def test_separate_model_path_noop_with_peft_policy() -> None:
    """With a plain (non-PEFT) ref_model the PEFT policy is left untouched."""
    policy = _make_peft_model()
    ref = _TinyMLP()  # plain model — triggers the separate-model no-op path

    active_before = list(policy.active_adapters)

    with null_ref_context(policy, ref_model=ref):
        active_inside = list(policy.active_adapters)

    active_after = list(policy.active_adapters)

    assert active_inside == active_before, "Separate-model path mutated PEFT adapter"
    assert active_after == active_before, "Separate-model path mutated PEFT adapter"


def test_separate_peft_reference_path_leaves_policy_adapter_active() -> None:
    """A PEFT reference is still separate and must not switch policy adapters."""
    policy = _make_peft_model()
    reference = _make_peft_model()
    policy.add_adapter(
        "ref",
        LoraConfig(
            target_modules=["linear"],
            r=2,
            lora_alpha=4,
            init_lora_weights=False,
        ),
    )
    policy.set_adapter("default")
    x = torch.randn(2, 8)
    out_before = policy(x).detach().clone()

    with null_ref_context(policy, ref_model=reference):
        active_inside = list(policy.active_adapters)
        out_inside = policy(x)

    assert active_inside == ["default"]
    assert list(policy.active_adapters) == ["default"]
    assert torch.allclose(out_inside, out_before)


def test_shared_peft_reference_model_is_rejected_before_mutating_policy() -> None:
    """Passing the policy as its own explicit reference is ambiguous and invalid."""
    policy = _make_peft_model()
    policy.add_adapter(
        "ref",
        LoraConfig(
            target_modules=["linear"],
            r=2,
            lora_alpha=4,
            init_lora_weights=False,
        ),
    )
    policy.set_adapter("default")

    with (
        pytest.raises(ValueError, match="separate model"),
        null_ref_context(policy, ref_model=policy),
    ):
        pytest.fail("invalid shared reference entered the context")

    assert list(policy.active_adapters) == ["default"]


# ---------------------------------------------------------------------------
# §7.8 row 4 — non-PEFT, no ref (no-op)
# ---------------------------------------------------------------------------


def test_non_peft_no_ref_noop() -> None:
    """null_ref_context on a plain nn.Module with no ref is a no-op."""
    model = nn.Linear(4, 4)
    x = torch.randn(3, 4)
    out_before = model(x).detach().clone()

    with null_ref_context(model):
        out_inside = model(x)

    assert torch.allclose(out_inside, out_before), (
        "non-PEFT no-ref path unexpectedly changed model output"
    )


def test_non_peft_no_ref_no_attribute_error() -> None:
    """Plain nn.Module does not raise AttributeError inside null_ref_context."""
    model = nn.Linear(4, 4)
    entered = False
    with null_ref_context(model):
        entered = True
    assert entered


# ---------------------------------------------------------------------------
# with_disabled_adapter — plain module no-op
# ---------------------------------------------------------------------------


def test_with_disabled_adapter_noop_on_plain_linear() -> None:
    """with_disabled_adapter is a no-op on a plain nn.Linear."""
    layer = nn.Linear(4, 4)
    x = torch.randn(3, 4)
    out_before = layer(x).detach().clone()

    with with_disabled_adapter(layer):
        out_inside = layer(x)

    assert torch.allclose(out_inside, out_before), (
        "with_disabled_adapter mutated plain nn.Linear"
    )


def test_with_disabled_adapter_noop_on_plain_linear_no_raise() -> None:
    """with_disabled_adapter does not raise on a plain nn.Module."""
    layer = nn.Linear(4, 4)
    entered = False
    with with_disabled_adapter(layer):
        entered = True
    assert entered


def test_with_disabled_adapter_disables_on_peft_model() -> None:
    """with_disabled_adapter disables the LoRA adapter inside the block."""
    model = _make_peft_model()
    x = torch.randn(2, 8)

    out_peft = model(x)
    with model.disable_adapter():
        out_base_direct = model(x)

    with with_disabled_adapter(model):
        out_base_via_helper = model(x)

    # with_disabled_adapter must produce the same result as disable_adapter().
    assert torch.allclose(out_base_via_helper, out_base_direct), (
        "with_disabled_adapter did not disable the adapter"
    )

    # Verify LoRA delta is non-trivial (test is meaningful).
    assert not torch.allclose(out_peft, out_base_direct)

    # After the block the adapter is active again.
    out_after = model(x)
    assert torch.allclose(out_after, out_peft), (
        "with_disabled_adapter did not restore the adapter on exit"
    )
