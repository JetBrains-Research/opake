"""Reference-adapter context managers for PEFT-backed reference models.

Provides :func:`null_ref_context` and :func:`with_disabled_adapter` — the two
context managers that implement the four ref-model configurations for
PEFT-backed reference models.

Design is closely modelled on TRL ``utils.use_adapter`` (see
``trl/trainer/utils.py``) but expressed as pure context managers rather than
trainer methods, so they compose cleanly with functional training loops.

.. warning:: **Outside vmap only.**

    Both helpers mutate ``nn.Module`` adapter-selection state (calls to
    ``model.set_adapter`` / ``model.disable_adapter``).  PEFT adapter flags are
    ``nn.Module`` state, not explicit function arguments, so call these helpers
    in the *outer* loop (outside the ``vmap(grad(...))`` region) to switch which
    weights the forward pass uses, then vmap the pure forward inside the
    activated context.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from opake.exceptions import ConfigurationError

__all__ = ["null_ref_context", "with_disabled_adapter"]

if TYPE_CHECKING:
    from collections.abc import Generator


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_peft_model(model: Any) -> bool:
    """Return ``True`` when *model* is a PEFT-wrapped model.

    Tries ``peft.utils.other.is_peft_model`` first (canonical location as of
    PEFT ≥ 0.14); falls back to checking for a ``peft_config`` attribute when
    the import path differs across PEFT versions.
    """
    try:
        # Canonical: peft < 0.18 exposed is_peft_model in utils.other.
        from peft.utils.other import is_peft_model  # type: ignore[import-untyped]

        return bool(is_peft_model(model))
    except ImportError:
        pass

    try:
        # Fallback for versions that moved the helper or removed it.
        from peft import PeftModel  # type: ignore[import-untyped]

        return isinstance(model, PeftModel)
    except ImportError:
        pass

    # Last resort: duck-type on the presence of peft_config.
    return hasattr(model, "peft_config")


def _get_active_adapters(model: Any) -> list[str] | str | None:
    """Snapshot the currently active adapter(s), defensive across PEFT versions.

    Returns a *copy* (list or str) so the caller can restore it regardless of
    whether the attribute is mutated in-place by PEFT.
    """
    # PEFT ≥ 0.10 uses ``active_adapters`` (list[str]).
    if hasattr(model, "active_adapters"):
        val = model.active_adapters
        if isinstance(val, list):
            return list(val)  # copy
        return val  # str or other — return as-is

    # Older PEFT used ``active_adapter`` (str).
    if hasattr(model, "active_adapter"):
        return model.active_adapter

    return None


def _restore_adapters(model: Any, saved: list[str] | str | None) -> None:
    """Restore the adapter state captured by :func:`_get_active_adapters`."""
    if saved is None:
        return

    # Normalise to a list for uniform handling.
    adapter_names: list[str] = [saved] if isinstance(saved, str) else list(saved)
    if not adapter_names:
        return

    if hasattr(model, "set_adapter"):
        # Pass a single string when there is exactly one adapter (covers the
        # most common case and avoids list-vs-str inconsistencies in older PEFT).
        if len(adapter_names) == 1:
            model.set_adapter(adapter_names[0])
        else:
            model.set_adapter(adapter_names)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@contextmanager
def null_ref_context(
    model: Any,
    ref_model: Any | None = None,
) -> Generator[None, None, None]:
    """Context manager that turns *model* into its own reference inside the block.

    Dispatches per the ref-model design table:

    ========================  ================================================
    Configuration             Behaviour
    ========================  ================================================
    Separate model            *ref_model* is not ``None`` → **no-op**
                              (caller uses *ref_model* directly; no policy
                              adapter to toggle).
    LoRA with ``"ref"``       ``is_peft_model(model)`` and ``"ref" in
    adapter clone             model.peft_config`` → call
                 ``model.set_adapter("ref")`` on enter; restore
                              the previously-active adapter(s) on exit via a
                              ``finally`` block.
    LoRA without separate     ``is_peft_model(model)`` and *ref_model* is
    ref                       ``None`` → call ``model.disable_adapter()``
                 (base model serves as reference); re-enable on
                              exit via ``finally``.
    Explicit callable /       *model* is not a PEFT model and *ref_model* is
    non-PEFT no-ref           ``None`` → **no-op** (caller-supplied callable
                 or frozen base model, nothing to toggle).
    ========================  ================================================

    Args:
        model: The policy model.  May be a PEFT :class:`~peft.PeftModel` or an
            ordinary :class:`torch.nn.Module`.
        ref_model: Optional separate reference model. When not ``None``, the
            caller is responsible for forwarding through *ref_model* directly;
            this context manager is a no-op. It must not be ``model`` itself.

    .. warning:: **Outside vmap only**.

        This helper mutates ``nn.Module`` adapter-selection state. Call it in
        the outer loop; vmap the pure forward *inside* the activated context.

    Note:
        Modelled on TRL ``utils.use_adapter`` (``trl/trainer/utils.py``).
    """
    if ref_model is model:
        raise ConfigurationError(
            *(
                "ref_model must be a separate model; omit ref_model to use the "
                "policy's PEFT reference adapter or adapter-disabled base model.",
            )
        )

    # ── Row 1: separate ref_model ───────────────────────────────────────────
    if ref_model is not None:
        # Caller owns ref_model; nothing to toggle on the policy model.
        yield
        return

    # ── Remaining rows require model to be a PEFT model. ────────────────────
    if not _is_peft_model(model):
        # Row 4: non-PEFT model with no ref → no-op.
        yield
        return

    # ── Row 2: LoRA model with a "ref" adapter clone. ───────────────────────
    if "ref" in model.peft_config:
        saved = _get_active_adapters(model)
        model.set_adapter("ref")
        try:
            yield
        finally:
            _restore_adapters(model, saved)
        return

    # ── Row 3: LoRA model with no separate ref → disable adapter. ───────────
    # (ref_model is None here, established by the guard at Row 1.)
    with with_disabled_adapter(model):
        yield


@contextmanager
def with_disabled_adapter(model: Any) -> Generator[None, None, None]:
    """Thin context manager that disables adapters on *model* for the block.

    When *model* is a PEFT model, enters ``model.disable_adapter()`` so the
    base model weights serve as the forward pass.  The adapter is re-enabled on
    exit (``finally``).  When *model* is not a PEFT model, this is a **no-op**.

    Used internally by the LoRA-disable reference path in
    :func:`null_ref_context` and exposed publicly for callers
    that want to forward through the base model directly.

    Args:
        model: The model whose adapters should be disabled.  May be a PEFT
            :class:`~peft.PeftModel` or an ordinary :class:`torch.nn.Module`.

    .. warning:: **Outside vmap only**.

        This helper mutates ``nn.Module`` adapter-selection state. Call it in
        the outer loop; vmap the pure forward *inside* the activated context.

    Note:
        Modelled on TRL ``utils.use_adapter`` (``trl/trainer/utils.py``).
    """
    if not _is_peft_model(model):
        yield
        return

    with model.disable_adapter():
        yield
