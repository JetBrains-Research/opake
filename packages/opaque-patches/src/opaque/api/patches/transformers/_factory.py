"""Per-model patch factory.

:func:`make_apply_model_patches` produces the ``apply_X_patches``
function for a HuggingFace model family.  It composes:

1. An ``apply_X_family_patches`` function (built by
   :func:`opaque.patches.transformers._family.make_apply_family_patches`)
   that mutates the family's modeling module on first encounter
   (idempotent, lazy).
2. Per-class patches on the *current model instance*: MLP, RMSNorm,
   DecoderLayer (for fused-add variants), and ForCausalLM.

This collapses what was previously ~17 hand-rolled per-model files into
~17 small factory invocations.  Architectural choices (which MLP kind,
which RMSNorm casting mode, whether the family supports fused-add-RMS)
are encoded in the factory call's kwargs and closed over the returned
function — so e.g. Gemma's ``activation_kind="geglu_exact"`` cannot accidentally
dispatch to SwiGLU.
"""

from __future__ import annotations

import functools
import importlib
import logging
from typing import TYPE_CHECKING, Literal

from opaque.api.patches.transformers._router import (
    _has_kernel_runtime,
    _patch_forward,
)
from opaque.api.patches.transformers.components.batchify import apply_batchify_patch
from opaque.api.patches.transformers.components.cross_entropy import (
    FUSED_LINEAR_CE_ATTR,
    _fused_linear_ce_supports_class,
    _make_fused_ce_causal_lm_forward,
    apply_causal_lm_loss_function_patch,
)
from opaque.api.patches.transformers.components.fused_add_rms_norm import (
    _fused_add_rms_fac_gemma,
    _fused_add_rms_fac_granite,
    _fused_add_rms_fac_llama,
    _fused_add_rms_fac_phi3,
)
from opaque.api.patches.transformers.components.geglu import (
    _make_geglu_approx_mlp_forward,
    _make_geglu_exact_mlp_forward,
)
from opaque.api.patches.transformers.components.kv_cache import apply_kv_cache_patch
from opaque.api.patches.transformers.components.moe import (
    _make_moe_experts_forward,
)
from opaque.api.patches.transformers.components.rms_norm import (
    _rmsnorm_fac_gemma,
    _rmsnorm_fac_gemma2,
    _rmsnorm_fac_glm4,
    _rmsnorm_fac_llama,
    _rmsnorm_fac_olmo2,
)
from opaque.api.patches.transformers.components.router import (
    install_fp32_router,
    remove_fp32_router,
)
from opaque.api.patches.transformers.components.swiglu import (
    _make_phi3_mlp_forward,
    _make_swiglu_mlp_forward,
)
from opaque.exceptions import ConfigurationError

if TYPE_CHECKING:
    from opaque.api.patches.transformers.types import (
        FamilyPatchFn,
        ForwardFactory,
        ModelPatchFn,
    )

log = logging.getLogger(__name__)


# Dispatch tables — single source of truth for which factory function
# implements each kind.  Adding a new activation / RMSNorm variant:
# register here, then reference by string from per-model factory calls.
ActivationKind = Literal["swiglu", "phi3_swiglu", "geglu_exact", "geglu_approx"]
RmsNormKind = Literal["llama", "gemma", "gemma2", "olmo2", "glm4"]
FusedAddRmsKind = Literal["llama", "gemma", "phi3", "granite"]
MoeKind = Literal["swiglu"]

_ACTIVATION_FACTORIES = {
    "swiglu": _make_swiglu_mlp_forward,
    "phi3_swiglu": _make_phi3_mlp_forward,
    "geglu_exact": _make_geglu_exact_mlp_forward,
    "geglu_approx": _make_geglu_approx_mlp_forward,
}

# Stacked-weight MoE experts (HF v5 ``*Experts`` modules). The expert FFN is
# SwiGLU, so the single registered kind dispatches to the vmap/DP-safe MoE
# kernel; register more here for non-SwiGLU expert activations.
_MOE_FACTORIES = {
    "swiglu": _make_moe_experts_forward,
}

_RMSNORM_FACTORIES = {
    "llama": _rmsnorm_fac_llama,
    "gemma": _rmsnorm_fac_gemma,
    "gemma2": _rmsnorm_fac_gemma2,
    "olmo2": _rmsnorm_fac_olmo2,
    "glm4": _rmsnorm_fac_glm4,
}

_FUSED_ADD_RMS_FACTORIES = {
    "llama": _fused_add_rms_fac_llama,
    "gemma": _fused_add_rms_fac_gemma,
    "phi3": _fused_add_rms_fac_phi3,
    "granite": _fused_add_rms_fac_granite,
}


# ----------------------------------------------------------------------------
# Public registration helpers — let users plug their own kernel variants in.
# ----------------------------------------------------------------------------


def register_activation_kind(name: str, factory: ForwardFactory) -> None:
    """Register a custom gated-activation forward factory under ``name``.

    Args:
        name: Identifier used in :func:`make_apply_model_patches` as
            ``activation_kind=name``.
        factory: Callable taking the original module's bound ``forward``
            and returning a new forward.
    """
    _ACTIVATION_FACTORIES[name] = factory


def register_rms_norm_kind(name: str, factory: ForwardFactory) -> None:
    """Register a custom RMSNorm forward factory under ``name`` (read by
    ``make_apply_model_patches(rms_norm_kind=name)``)."""
    _RMSNORM_FACTORIES[name] = factory


def register_fused_add_rms_kind(name: str, factory: ForwardFactory) -> None:
    """Register a custom fused-add-RMSNorm DecoderLayer factory under
    ``name`` (read by
    ``make_apply_model_patches(fused_add_rms_kind=name)``)."""
    _FUSED_ADD_RMS_FACTORIES[name] = factory


def register_moe_kind(name: str, factory: ForwardFactory) -> None:
    """Register a custom MoE experts-forward factory under ``name`` (read by
    ``make_apply_model_patches(moe_kind=name)``)."""
    _MOE_FACTORIES[name] = factory


def _resolve(
    spec: str | ForwardFactory | None,
    registry: dict[str, ForwardFactory],
) -> ForwardFactory | None:
    """Convert a kind-spec to a factory callable.

    ``spec`` accepts: ``None`` (no patch), a registered string name,
    or a callable used directly.
    """
    if spec is None:
        return None
    if callable(spec):
        return spec
    if spec in registry:
        return registry[spec]
    raise ConfigurationError(
        *(
            f"Unknown kind {spec!r}; registered: {sorted(registry)}.  "
            "Pass a callable directly, or use ``register_*_kind`` to add a custom name.",
        )
    )


def _normalize_chunked_linear_cross_entropy(value: object) -> bool | int:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value < 0:
            ConfigurationError.raise_(
                "chunked_linear_cross_entropy must be a boolean or non-negative integer"
            )
        return False if value == 0 else value
    ConfigurationError.raise_(
        "chunked_linear_cross_entropy must be a boolean or non-negative integer"
    )


def make_apply_model_patches(
    *,
    family: str,
    family_apply: FamilyPatchFn,
    module_path: str,
    classes: dict[str, str],
    activation_kind: str | ForwardFactory | None = None,
    rms_norm_kind: str | ForwardFactory | None = None,
    fused_add_rms_kind: str | ForwardFactory | None = None,
    moe_kind: str | ForwardFactory | None = None,
    fused_linear_cross_entropy: bool = True,
    chunked_linear_cross_entropy: bool | int = False,
) -> ModelPatchFn:
    """Build an ``apply_X_patches`` function for a given family.

    The returned function applies family-runtime patches lazily on first
    call (via ``family_apply``) and patches per-class methods on the
    given model instance.  Architecture choices are closed over the
    returned function so e.g. an MLP kernel can't be cross-wired to the
    wrong activation flag.

    Args:
        family: Family name (matches :func:`family_name`).
        family_apply: The matching ``apply_X_family_patches`` produced
            by :func:`make_apply_family_patches`.  Called first inside
            the returned function (idempotent).
        module_path: Dotted path to the modeling module.
        classes: Mapping of role → HF class name.  Recognized roles:
            ``"mlp"``, ``"rms_norm"``, ``"decoder_layer"``,
            ``"causal_lm"``, ``"experts"``, ``"router"``.  Roles absent
            from the mapping are skipped (e.g. Cohere has no RMSNorm; omit the
            ``"rms_norm"`` entry). ``"router"`` names the top-k router module
            that ``router_fp32=True`` rebinds to fp32 logits.
        activation_kind: Which gated-activation forward factory to use.
            Either a registered string name (``"swiglu"``,
            ``"geglu_exact"``, …), a callable used directly, or ``None``
            (no activation patch).  The user-facing kwarg that gates this
            concern is always ``activation``; the family decides which
            concrete activation kernel is deployed.
        rms_norm_kind: Same shape — registered name, callable, or ``None``.
        fused_add_rms_kind: Same shape — for the DecoderLayer fused-add
            variant.
        moe_kind: Same shape — for the stacked-weight MoE experts module
            (``classes["experts"]``). Gated by the ``moe`` kwarg, which
            defaults from ``compat`` (it is a vmap-safety patch required for
            DP-SGD, not a CUDA kernel). ``opaque_moe`` transparently uses the
            sparse grouped-GEMM Triton kernel on CUDA bf16/fp16. ``None`` for
            dense models.
        fused_linear_cross_entropy: Whether the family supports the fused
            LM-head loss within the required backward-error envelope.
        chunked_linear_cross_entropy: Whether the family uses the portable
            chunked LM-head loss instead of the fused CUDA kernel. ``True``
            selects its default tile width, a positive integer sets the width,
            and ``False`` or ``0`` disables it.

    Returns:
        Callable with signature
        ``apply(model=None, *, performance=True, compat=True,
        kernels=None, **kwargs) -> None``.
        Per-concern kwargs default into the right group:
        ``rope``, ``rms_norm``, ``activation``, ``cross_entropy``,
        ``grouped_moe`` → ``kernels`` (itself defaulting to ``performance``
        when ``None``); ``kv_cache`` → ``performance``;
        ``eager_attention``, ``batchify``, ``moe`` → ``compat``.
        ``moe`` installs the vmap-safe experts forward (DP-SGD needs it);
        ``grouped_moe`` only chooses its grouped-GEMM fast path (kernel-fused
        Triton on CUDA / ``torch._grouped_mm`` on MPS-CPU) vs the dense compat
        path, so a dense run keeps a correct, vmap-safe MoE.
        ``loss_only_forward`` → ``compat`` (or the fused routes being on)
        installs the loss-only causal-LM forward; ``fused_linear_cross_entropy``
        inherits from ``performance`` and decides whether it takes the fused /
        chunked routes or the eager ``lm_head`` branch. ``router_fp32``
        (default ``False``) binds an fp32-logit forward on the family's
        router instances; ``False`` removes an earlier install.
    """
    activation_factory = _resolve(activation_kind, _ACTIVATION_FACTORIES)
    rms_norm_factory = _resolve(rms_norm_kind, _RMSNORM_FACTORIES)
    fused_add_rms_factory = _resolve(fused_add_rms_kind, _FUSED_ADD_RMS_FACTORIES)
    moe_factory = _resolve(moe_kind, _MOE_FACTORIES)
    chunked_linear_cross_entropy = _normalize_chunked_linear_cross_entropy(
        chunked_linear_cross_entropy
    )

    def apply(
        model: object | None = None,
        *,
        performance: bool = True,
        compat: bool = True,
        kernels: bool | None = None,
        **kwargs: object,
    ) -> None:
        # ``kernels`` requests accelerated kernels; each install below checks the
        # environment. ``triton_ok`` gates the Triton-only kernels (activation /
        # rms_norm / cross_entropy) so they're never installed off-CUDA — there
        # they'd only fall back to eager, and not always faithfully. The portable
        # grouped-GEMM MoE is NOT gated on it: it runs on MPS/CPU via
        # ``torch._grouped_mm`` (dense fallback where unavailable).
        if kernels is None:
            kernels = performance
        triton_ok = _has_kernel_runtime()
        # Lazily apply family-runtime patches.  Idempotent.
        family_apply(performance=performance, compat=compat, kernels=kernels, **kwargs)

        try:
            mod = importlib.import_module(module_path)
        except ImportError:
            return

        # Gated activation inside the model's MLP module (SwiGLU / GeGLU /
        # Phi3-style SwiGLU). Triton kernel — ``triton_ok`` gates the DEFAULT (so
        # ``kernels`` doesn't install it off-CUDA); an explicit ``activation=True``
        # is still honored.
        if activation_factory is not None and kwargs.get(
            "activation", kernels and triton_ok
        ):
            mlp_class = classes.get("mlp")
            if mlp_class is not None:
                _patch_forward(getattr(mod, mlp_class, None), activation_factory, model)

        # Stacked-weight MoE experts (HF v5 ``*Experts`` module). Two gates:
        #
        #   1. INSTALLING the replacement forward is a vmap-safety patch — HF's
        #      experts forward isn't vmap(grad)-able and DP-SGD breaks without it.
        #      So it lives in the ``compat`` bucket (``moe`` -> ``compat``), NOT
        #      the perf gate: the vmap-safe forward must be present even for a
        #      dense (grouped_moe=False) run, on any host.
        #   2. WHICH path that forward takes is a performance gate: ``grouped_moe``
        #      (-> ``kernels``) picks the grouped-GEMM fast path — kernel-fused
        #      Triton on CUDA bf16/fp16, ``torch._grouped_mm`` on MPS/CPU — while
        #      ``grouped_moe=False`` forces the dense, always-correct ``Opaque_MoE``.
        #      Both grouped paths are performance variations; only dense is compat.
        #
        # The ``moe`` gate is separate from ``activation`` since a model may have
        # both routed experts and dense MLP layers. Absent ``Experts`` class
        # (dense-only / pre-v5) no-ops below.
        if moe_factory is not None and kwargs.get("moe", compat):
            experts_class = classes.get("experts")
            if experts_class is not None:
                grouped_moe = kwargs.get("grouped_moe", kernels)
                _patch_forward(
                    getattr(mod, experts_class, None),
                    functools.partial(moe_factory, grouped=grouped_moe),
                    model,
                )

        # fp32-logit router (opt-in, instance-level, removable).
        router_fp32 = kwargs.get("router_fp32")
        if router_fp32 is not None and model is not None:
            router_class = classes.get("router")
            router_cls = getattr(mod, router_class, None) if router_class else None
            if router_fp32:
                install_fp32_router(model, router_cls=router_cls)
            else:
                remove_fp32_router(model)

        # RMSNorm (unified standalone + fused-add). Triton kernel — ``triton_ok``
        # gates the default; explicit ``rms_norm=True`` honored.
        rms_norm_on = kwargs.get("rms_norm", kernels and triton_ok)
        if rms_norm_on:
            if rms_norm_factory is not None:
                rms_norm_class = classes.get("rms_norm")
                if rms_norm_class is not None:
                    _patch_forward(
                        getattr(mod, rms_norm_class, None),
                        rms_norm_factory,
                        model,
                    )
            if fused_add_rms_factory is not None:
                decoder_class = classes.get("decoder_layer")
                if decoder_class is not None:
                    _patch_forward(
                        getattr(mod, decoder_class, None),
                        fused_add_rms_factory,
                        model,
                    )

        # Cross-entropy patches are split into two gates.
        #   - ``cross_entropy`` (defaults from ``kernels``): installs
        #     ``Opaque_CrossEntropyLoss`` via ``loss_function``. Operates
        #     on materialized logits; the model still returns them.
        #   - ``fused_linear_cross_entropy`` (defaults from ``performance``):
        #     installs a conditional forward. Marker-false calls delegate to the
        #     original model; loss-only calls skip ``lm_head`` materialization and
        #     return ``logits=None``.
        causal_lm_class = classes.get("causal_lm")
        causal_lm_obj = getattr(mod, causal_lm_class, None) if causal_lm_class else None
        if (
            kwargs.get("cross_entropy", kernels and triton_ok)
            and causal_lm_obj is not None
        ):
            apply_causal_lm_loss_function_patch(model, causal_lm_obj)
        chunked_linear_ce = kwargs.get(
            "chunked_linear_cross_entropy", chunked_linear_cross_entropy
        )
        if (
            chunked_linear_ce is True
            and isinstance(chunked_linear_cross_entropy, int)
            and not isinstance(chunked_linear_cross_entropy, bool)
        ):
            chunked_linear_ce = chunked_linear_cross_entropy
        else:
            chunked_linear_ce = _normalize_chunked_linear_cross_entropy(
                chunked_linear_ce
            )
        # Loss-only causal-LM forward (compat); which CE route it takes is the
        # performance decision, recorded per instance.
        enable_fused_linear_ce = kwargs.get("fused_linear_cross_entropy")
        if enable_fused_linear_ce is None:
            enable_fused_linear_ce = performance
        fused_routes = bool(
            (fused_linear_cross_entropy or chunked_linear_ce) and enable_fused_linear_ce
        )
        if (
            kwargs.get("loss_only_forward", compat or fused_routes)
            and causal_lm_obj is not None
            and _fused_linear_ce_supports_class(causal_lm_obj)
        ):
            linear_ce_factory = functools.partial(
                _make_fused_ce_causal_lm_forward, fused=fused_routes
            )
            if chunked_linear_ce and fused_routes:
                linear_ce_factory = functools.partial(
                    _make_fused_ce_causal_lm_forward,
                    force_chunked=chunked_linear_ce,
                    fused=True,
                )
            modules = getattr(model, "modules", None)
            if callable(modules):
                for module in modules():
                    if isinstance(module, causal_lm_obj):
                        setattr(module, FUSED_LINEAR_CE_ATTR, fused_routes)

            _patch_forward(
                causal_lm_obj,
                linear_ce_factory,
                model,
            )

        # vmap-safety patch on the causal-LM class.
        if kwargs.get("batchify", compat) and causal_lm_obj is not None:
            apply_batchify_patch(causal_lm_obj, model)
        # KV cache disabler: skips the DynamicCache allocation that HF
        # makes on every training forward, and breaks the cache's
        # circular references that vmap would leak. Pure Python — sits
        # in the ``performance`` bucket rather than ``kernels`` so it
        # still applies on CPU / MPS hosts.
        if kwargs.get("kv_cache", performance) and causal_lm_obj is not None:
            apply_kv_cache_patch(causal_lm_obj, model)

    apply.__name__ = f"apply_{family}_patches"
    apply.__qualname__ = apply.__name__
    apply._opaque_family = family  # type: ignore[attr-defined]
    return apply


__all__ = [
    "ActivationKind",
    "FusedAddRmsKind",
    "MoeKind",
    "RmsNormKind",
    "make_apply_model_patches",
    "register_activation_kind",
    "register_fused_add_rms_kind",
    "register_moe_kind",
    "register_rms_norm_kind",
]
