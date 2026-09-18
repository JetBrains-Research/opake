"""Family-level patches for HuggingFace ``transformers.models.X.modeling_X``.

A "family" is a HF architecture (Llama, Gemma, Qwen, ...).  This module
owns:

- :func:`family_name` — detect a model's family from its instance.
- :func:`make_apply_family_patches` — factory that produces the per-family
  ``apply_X_family_patches`` function.

Family-level patches mutate **module-level attributes** of the family's
modeling module (``mod.repeat_kv``, ``mod.eager_attention_forward``,
``mod.apply_rotary_pos_emb``, etc.).  These affect every code path that
reaches the patched name within that family — once per process,
regardless of how many model *instances* exist.

``mod.ALL_ATTENTION_FUNCTIONS`` is shared by every HF modeling module, so it
is rebound to a family-private instance rather than written through.

Per-class patches (``LlamaMLP.forward``, ``LlamaRMSNorm.forward``, etc.)
live in :mod:`._factory` and run per-model-instance.

Idempotency: keyed on family name and enabled concern in
:data:`_PATCHED_FAMILIES`.  Calling ``apply_llama_family_patches()``
twice with the same enabled concerns in the same process is a no-op the
second time, while a later call can still enable a concern that was
previously disabled. Tests that need to re-apply can clear the cache via
:func:`_reset_patched_families`.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

from opake.api.patches.transformers.components.attention import (
    vmap_eager_attention_forward,
    vmap_repeat_kv,
    vmap_sdpa_attention_forward,
)
from opake.api.patches.transformers.components.masking import (
    apply_module_masking_patch,
)
from opake.api.patches.transformers.components.rope import _opake_apply_rotary_pos_emb

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from types import ModuleType

    from opake.api.patches.transformers.types import (
        FamilyPatchFn,
        ForwardFn,
        ModulePatcher,
    )

log = logging.getLogger(__name__)


_PATCHED_FAMILIES: dict[str, set[str]] = {}


def _reset_patched_families() -> None:
    """Clear the family-patch idempotency cache (test hook only)."""
    _PATCHED_FAMILIES.clear()


def _scoped_attention_functions(
    mod: ModuleType,
) -> MutableMapping[str, ForwardFn] | None:
    """Return ``mod``'s own ``ALL_ATTENTION_FUNCTIONS``, installing it if absent.

    Writing to the shared instance would reroute every family. Returns ``None``
    if the interface is missing or cannot be cloned, leaving it untouched.
    """
    shared = getattr(mod, "ALL_ATTENTION_FUNCTIONS", None)
    if shared is None:
        return None
    if getattr(shared, "_opake_scoped_to", None) == mod.__name__:
        return shared
    try:
        scoped = type(shared)()
    except TypeError:
        log.debug(
            "opake: cannot scope %s.ALL_ATTENTION_FUNCTIONS; "
            "leaving the shared attention registry untouched.",
            mod.__name__,
        )
        return None
    # Carry over local (module-scoped) overrides the shared instance already
    # holds; class-wide registrations remain reachable via its global mapping.
    local = getattr(shared, "_local_mapping", None)
    scoped_local = getattr(scoped, "_local_mapping", None)
    if isinstance(local, dict) and isinstance(scoped_local, dict):
        scoped_local.update(local)
    scoped._opake_scoped_to = mod.__name__
    mod.ALL_ATTENTION_FUNCTIONS = scoped
    return scoped


def family_name(model: object) -> str | None:
    """Return the opake-patches family name for an HF model instance.

    Tries ``model.config.model_type`` first (the HF-canonical signal),
    then falls back to the modeling module path
    (``transformers.models.X.modeling_X`` → ``X``).

    Returns ``None`` if the model is not a HuggingFace transformers model.
    """
    cfg = getattr(model, "config", None)
    if cfg is not None:
        kind = getattr(cfg, "model_type", None)
        if kind:
            return str(kind)
    module = type(model).__module__
    parts = module.split(".")
    if len(parts) >= 3 and parts[0] == "transformers" and parts[1] == "models":  # noqa: PLR2004 - HF module path layout
        return parts[2]
    return None


def make_apply_family_patches(
    *,
    family: str,
    module_path: str,
    repeat_kv_replacement: ForwardFn | None = vmap_repeat_kv,
    eager_attention_replacement: ForwardFn | None = vmap_eager_attention_forward,
    sdpa_attention_replacement: ForwardFn | None = vmap_sdpa_attention_forward,
    rope_replacement: ForwardFn | None = _opake_apply_rotary_pos_emb,
    masking_module_patcher: ModulePatcher | None = apply_module_masking_patch,
) -> FamilyPatchFn:
    """Build an ``apply_X_family_patches`` function for a given family.

    The returned function patches module-level attributes of
    ``module_path`` when ``eager_attention`` and/or ``rope`` are enabled.
    Idempotent — keyed on ``family``.

    Defaults to opake's vmap-safe implementations; pass overrides to
    plug in your own (e.g. for a non-HF-shaped architecture), or pass
    ``None`` to skip that particular concern.

    Args:
        family: Short family name (``"llama"``, ``"gemma"``, …); must
            match what :func:`family_name` returns for instances of this
            family.
        module_path: Dotted path to the modeling module.
        repeat_kv_replacement: Replacement for ``mod.repeat_kv``.
            Default: opake's vmap-safe ``vmap_repeat_kv``.  ``None``
            skips the patch.
        eager_attention_replacement: Replacement for
            ``mod.eager_attention_forward``.  Default: opake's
            ``vmap_eager_attention_forward``.  ``None`` skips.
        sdpa_attention_replacement: Replacement for the ``"sdpa"`` entry of
            an ``ALL_ATTENTION_FUNCTIONS`` interface bound in ``module_path``
            alone. Defaults to opake's replication-free GQA shim. ``None``
            leaves SDPA untouched for this family.
        rope_replacement: Replacement for ``mod.apply_rotary_pos_emb``.
            Default: opake's ``_opake_apply_rotary_pos_emb``.  ``None``
            skips.
        masking_module_patcher: Function called as ``f(mod)`` to apply
            module-scoped masking patches (re-imported ``create_causal_mask``
            etc.).  Default: opake's ``apply_module_masking_patch``.
            ``None`` skips.

    Returns:
        Callable with signature
        ``apply(*, performance=True, compat=True, kernels=None, **kwargs) -> None``.
        ``eager_attention`` defaults from ``compat``; ``rope`` (Triton
        kernel) defaults from ``kernels``, which itself defaults from
        ``performance`` when not passed explicitly.
    """

    def apply(
        *,
        performance: bool = True,
        compat: bool = True,
        kernels: bool | None = None,
        **kwargs: object,
    ) -> None:
        if kernels is None:
            kernels = performance
        eager_attention = kwargs.get("eager_attention", compat)
        # rope is a Triton-only kernel and its eager path isn't faithful for every
        # rotary variant (partial / interleaved) — so only install it where the
        # Triton runtime exists; off-CUDA the model keeps HF's own rope.
        from opake.api.engine.device import fused_kernels_available

        rope = kwargs.get("rope", kernels and fused_kernels_available())
        requested_concerns = set()
        if eager_attention:
            requested_concerns.add("eager_attention")
        if rope:
            requested_concerns.add("rope")
        if not requested_concerns:
            return

        patched_concerns = _PATCHED_FAMILIES.get(family, set())
        if requested_concerns <= patched_concerns:
            return

        try:
            mod = importlib.import_module(module_path)
        except ImportError:
            return

        patched_now: set[str] = set()

        if eager_attention and "eager_attention" not in patched_concerns:
            patched = False
            if repeat_kv_replacement is not None and hasattr(mod, "repeat_kv"):
                mod.repeat_kv = repeat_kv_replacement
                patched = True
            if eager_attention_replacement is not None and hasattr(
                mod, "eager_attention_forward"
            ):
                mod.eager_attention_forward = eager_attention_replacement
                patched = True
            if sdpa_attention_replacement is not None:
                # Scoped to this family's modeling module: the shared
                # process-global interface must keep the stock SDPA entry.
                scoped_attention = _scoped_attention_functions(mod)
                if scoped_attention is not None:
                    scoped_attention["sdpa"] = sdpa_attention_replacement
                    patched = True
            if masking_module_patcher is not None:
                masking_result = masking_module_patcher(mod)
                if masking_result is not False:
                    patched = True
            if patched:
                patched_now.add("eager_attention")
        if (
            rope
            and "rope" not in patched_concerns
            and rope_replacement is not None
            and hasattr(mod, "apply_rotary_pos_emb")
        ):
            mod.apply_rotary_pos_emb = rope_replacement
            patched_now.add("rope")

        if patched_now:
            _PATCHED_FAMILIES.setdefault(family, set()).update(patched_now)

    apply.__name__ = f"apply_{family}_family_patches"
    apply.__qualname__ = apply.__name__
    apply._opake_family = family  # type: ignore[attr-defined]
    return apply


__all__ = [
    "_reset_patched_families",
    "family_name",
    "make_apply_family_patches",
]
