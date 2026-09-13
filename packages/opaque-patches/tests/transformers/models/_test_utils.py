# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for testing vmap and gradients on models."""

import sys
from contextlib import contextmanager

import torch

from opaque.api.engine.clipping import clipped_grad
from opaque.functional import make_functional


def get_tiny_config_kwargs():
    return {
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 128,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "rope_theta": 10000.0,
    }


def assert_forward_no_grad(model, device):
    model.eval()
    torch.manual_seed(0)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 10), device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    assert out.logits.shape == (2, 10, model.config.vocab_size)


def assert_forward_backward(model, device):
    model.train()
    torch.manual_seed(0)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 10), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
    loss.backward()
    # Assert some gradient exists
    assert any(p.grad is not None for p in model.parameters())


def assert_vmap_forward(model, device):
    model.eval()
    torch.manual_seed(0)
    batch, seq = 4, 8
    input_ids = torch.randint(0, model.config.vocab_size, (batch, seq), device=device)
    attention_mask = torch.ones(batch, seq, dtype=torch.long, device=device)

    def per_example(ids, mask):
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask)
        return out.logits.mean()

    vmapped = torch.vmap(per_example)(input_ids, attention_mask)
    assert vmapped.shape == (batch,)


def assert_vmap_grad(model, device, dtype=None):
    """Run the real DP-SGD ``clipped_grad`` (vmap(grad)) pipeline.

    ``dtype`` casts the model first — pass ``torch.bfloat16`` on CUDA to exercise
    the Triton kernel paths (e.g. the fused MoE kernel), which only engage for
    bf16/fp16 CUDA tensors; the default (fp32) runs the pure-torch fallbacks.
    """
    if dtype is not None:
        model = model.to(dtype)
    model.train()
    torch.manual_seed(0)
    batch, seq, vocab = 4, 12, model.config.vocab_size
    input_ids = torch.randint(0, vocab, (batch, seq), device=device)
    attention_mask = torch.ones(batch, seq, dtype=torch.long, device=device)
    labels = input_ids.clone()

    fmodel, trainable, frozen = make_functional(
        model, disable_autograd_tracking=True, partition_trainable=True
    )

    def per_example_loss(trainable_params, frozen_params, ids, mask, lbls):
        all_params = {**frozen_params, **trainable_params}
        outputs = fmodel(all_params, ids, attention_mask=mask, labels=lbls)
        return outputs.loss

    grad_fn, clip_state = clipped_grad(
        per_example_loss, argnums=0, batch_argnums=(2, 3, 4), clipping_norm=1.0
    )
    grads, _state = grad_fn(
        trainable, frozen, input_ids, attention_mask, labels, state=clip_state
    )
    assert len(grads.pytree) > 0
    assert all(torch.isfinite(g).all() for g in grads.pytree.values())


# ----------------------------------------------------------------------------
# MoE helpers — the stacked-weight ``*Experts`` families resolve their HF
# Config / ForCausalLM by ``model_type`` so per-family test files stay thin.
# ----------------------------------------------------------------------------


def build_moe_model(family, device, attn_impl="sdpa", **config_overrides):
    """Build + patch a tiny MoE model. Returns ``(model, modeling_module)``.

    Defaults to ``attn_impl="sdpa"`` — the transformers production default — so
    the suite exercises the SDPA path; pass ``attn_impl="eager"`` for the eager
    reference validation.

    The Opaque MoE patch targets the stacked-weight ``*Experts`` module
    (transformers v5+). On versions where the family still uses the old
    ``*SparseMoeBlock`` (a ModuleList of per-expert MLPs, no stacked experts),
    DP-SGD MoE isn't supported — skip rather than exercise the vmap-broken HF
    forward. Capability check (presence of a ``*Experts`` class), not a version
    number, so it tracks the API rather than the release.
    """
    import importlib

    import pytest

    from opaque.patches import apply_model_patches

    module_path = f"transformers.models.{family}.modeling_{family}"
    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError as error:
        if error.name in {f"transformers.models.{family}", module_path}:
            import transformers

            pytest.skip(
                f"{family} is unavailable in transformers {transformers.__version__}"
            )
        raise
    if not any(n.endswith("Experts") for n in dir(mod)):
        import transformers

        pytest.skip(
            f"{family}: stacked *Experts module absent in transformers "
            f"{transformers.__version__} (v5+ feature)"
        )
    cfg_mod = importlib.import_module(
        f"transformers.models.{family}.configuration_{family}"
    )
    config_cls = next(
        getattr(cfg_mod, n)
        for n in dir(cfg_mod)
        if n.endswith("Config") and "PreTrained" not in n
    )
    causal_lm_cls = next(getattr(mod, n) for n in dir(mod) if n.endswith("ForCausalLM"))
    kwargs = get_tiny_config_kwargs()
    kwargs.update(config_overrides)
    config = config_cls(**kwargs)
    config._attn_implementation = attn_impl
    try:
        model = causal_lm_cls(config).to(device)
    except ValueError as e:
        # Some architectures (gpt_oss, deepseek_v4) are eager-only — HF rejects
        # sdpa at init. Fall back to eager rather than fail the family.
        if attn_impl == "eager" or "scaled_dot_product" not in str(e):
            raise
        config._attn_implementation = "eager"
        model = causal_lm_cls(config).to(device)
    apply_model_patches(model, eager_attention=True)
    return model, mod


def experts_forward_patched(modeling_module):
    """True if the family's stacked ``*Experts`` forward is on the Opaque kernel."""
    cls = next(
        (
            getattr(modeling_module, n)
            for n in dir(modeling_module)
            if n.endswith("Experts")
        ),
        None,
    )
    return cls is not None and hasattr(cls.forward, "__opaque_patched__")


# ----------------------------------------------------------------------------
# Parity helpers — compare patched vs unpatched models
# ----------------------------------------------------------------------------

# Tolerances keyed by (dtype_str, has_softcapping).
# Softcapping families use slightly relaxed fp32 tolerances because the
# patched eager attention may accumulate a few extra rounding steps.
_FP32_TOLERANCES = {"rtol": 1e-5, "atol": 1e-6}
_SOFTCAP_FP32_TOLERANCES = {"rtol": 1e-4, "atol": 1e-5}
_BF16_TOLERANCES = {"rtol": 1e-2, "atol": 1e-2}


def _get_tolerances(dtype: torch.dtype, softcapping: bool = False) -> dict:
    """Return dtype-specific tolerances for allclose assertions."""
    if dtype == torch.bfloat16:
        return _BF16_TOLERANCES
    elif softcapping:
        return _SOFTCAP_FP32_TOLERANCES
    else:
        return _FP32_TOLERANCES


def _get_param_names(model) -> list[str]:
    """Return sorted list of fully-qualified parameter names."""
    return sorted(k for k, _ in model.named_parameters())


def _get_params_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return {name: param} for all parameters, sorted by name."""
    return dict(model.named_parameters())


def _collect_grads(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return {name: grad} for all parameters that have a gradient."""
    return {name: p.grad for name, p in model.named_parameters() if p.grad is not None}


def build_patched_model_pair(
    config_cls,
    model_cls,
    device,
    attn_impl: str = "sdpa",
    config_kwargs: dict | None = None,
):
    """Build two tiny models with identical weights.

    ``config_kwargs`` is the full config dict — defaults to
    ``get_tiny_config_kwargs()`` when ``None``.  Pass a custom dict (e.g. with
    ``num_key_value_heads`` removed) for families like GPT2 that lack GQA.

    Returns ``(upstream_model, patchable_model)``.
    """
    import copy

    if config_kwargs is None:
        config_kwargs = get_tiny_config_kwargs()
    config = config_cls(**config_kwargs)
    config._attn_implementation = attn_impl

    try:
        model = model_cls(config).to(device)
    except ValueError as e:
        if attn_impl == "eager" or "scaled_dot_product" not in str(e):
            raise
        config._attn_implementation = "eager"
        model = model_cls(config).to(device)

    return model, copy.deepcopy(model)


# Module attributes the family patches rebind; restored on context exit.
_FAMILY_MODULE_PATCH_NAMES = (
    "create_causal_mask",
    "create_sliding_window_causal_mask",
    "create_recurrent_attention_mask",
    "repeat_kv",
    "eager_attention_forward",
    "apply_rotary_pos_emb",
)


@contextmanager
def parity_model_patches(
    model: torch.nn.Module, apply_model_patches_kwargs: dict | None = None
):
    """Apply model patches without installing global runtime patches."""
    from opaque.api.patches.transformers._family import _reset_patched_families
    from opaque.api.patches.transformers._router import apply_transformers_model_patches

    original_forwards = {
        type(module): type(module).forward for module in model.modules()
    }
    modeling_module = sys.modules.get(type(model).__module__)
    module_originals = {
        name: getattr(modeling_module, name)
        for name in _FAMILY_MODULE_PATCH_NAMES
        if hasattr(modeling_module, name)
    }
    # ``ALL_ATTENTION_FUNCTIONS`` is a process-wide registry, so a family that
    # replaces its "sdpa" entry (gemma2) would otherwise reroute every family.
    # Save the entire object because patching may replace it with a scoped instance.
    original_attention_functions = getattr(
        modeling_module, "ALL_ATTENTION_FUNCTIONS", None
    )

    _reset_patched_families()
    try:
        apply_transformers_model_patches(
            model, **(apply_model_patches_kwargs or {"eager_attention": True})
        )
        yield
    finally:
        for module_cls, forward in original_forwards.items():
            module_cls.forward = forward
        for name, value in module_originals.items():
            setattr(modeling_module, name, value)
        if original_attention_functions is not None:
            modeling_module.ALL_ATTENTION_FUNCTIONS = original_attention_functions
        _reset_patched_families()


def build_runtime_patched_model_pair(
    config_cls,
    model_cls,
    device,
    attn_impl: str = "sdpa",
    config_kwargs: dict | None = None,
    apply_model_patches_kwargs: dict | None = None,
):
    """Build a pair sharing runtime shims required for upstream vmap support."""
    unpatched, patched = build_patched_model_pair(
        config_cls,
        model_cls,
        device,
        attn_impl=attn_impl,
        config_kwargs=config_kwargs,
    )
    from opaque.patches import apply_model_patches

    apply_model_patches(
        patched, **(apply_model_patches_kwargs or {"eager_attention": True})
    )
    return unpatched, patched


def _assert_tensors_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float,
    atol: float,
    label: str = "",
) -> None:
    """Assert two tensors are close; on failure, print per-element stats."""
    diff = (actual.float() - expected.float()).abs()
    abs_err = diff.max().item()
    expected_abs = expected.float().abs()
    mask = expected_abs >= 1e-6
    if mask.sum() > 0:
        rel_err = (diff[mask] / expected_abs[mask]).max().item()
    else:
        rel_err = float("nan")

    prefix = f"{label}: " if label else ""
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    except AssertionError:
        raise AssertionError(
            f"{prefix}{name} mismatch: abs_err={abs_err:.2e}, rel_err={rel_err:.2e}"
            f" (rtol={rtol:.0e}, atol={atol:.0e})"
        ) from None


def assert_parity_forward(
    model_cls,
    config_cls,
    device,
    attn_impl: str = "sdpa",
    config_kwargs: dict | None = None,
    softcapping: bool = False,
    label: str = "",
    apply_model_patches_kwargs: dict | None = None,
    dtype: torch.dtype | None = None,
    seq_len: int = 10,
    use_attention_mask: bool = True,
):
    """Compare forward logits between patched and unpatched models.

    Returns ``(logits_patched, logits_upstream)``.
    """
    torch.manual_seed(0)
    unpatched, patched = build_patched_model_pair(
        config_cls,
        model_cls,
        device,
        attn_impl=attn_impl,
        config_kwargs=config_kwargs,
    )
    if dtype is not None:
        unpatched = unpatched.to(dtype)
        patched = patched.to(dtype)

    unpatched.eval()
    patched.eval()
    vocab = unpatched.config.vocab_size
    input_ids = torch.randint(0, vocab, (2, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids) if use_attention_mask else None

    with torch.no_grad():
        logits_ref = unpatched(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits
        with parity_model_patches(patched, apply_model_patches_kwargs):
            logits_test = patched(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits

    assert logits_ref.shape == logits_test.shape, (
        f"{label} shape mismatch: {logits_ref.shape} vs {logits_test.shape}"
    )
    actual_dtype = logits_test.dtype
    tolerances = _get_tolerances(actual_dtype, softcapping)
    _assert_tensors_close(
        "forward logits",
        logits_test,
        logits_ref,
        rtol=tolerances["rtol"],
        atol=tolerances["atol"],
        label=label,
    )
    return logits_test, logits_ref


def assert_logits_differ(
    logits: torch.Tensor,
    other: torch.Tensor,
    softcapping: bool = False,
    label: str = "",
) -> None:
    """Assert two logit tensors differ by more than parity noise."""
    tolerances = _get_tolerances(logits.dtype, softcapping)
    prefix = f"{label}: " if label else ""
    max_diff = (logits.float() - other.float()).abs().max().item()
    assert not torch.allclose(
        logits, other, rtol=tolerances["rtol"], atol=tolerances["atol"]
    ), (
        f"{prefix}logits are unchanged (max diff {max_diff:.2e} within "
        f"rtol={tolerances['rtol']:.0e}, atol={tolerances['atol']:.0e}) — the "
        f"feature under test is not constraining the output, so the parity "
        f"assertion is vacuous"
    )


def assert_parity_grad(
    model_cls,
    config_cls,
    device,
    attn_impl: str = "sdpa",
    config_kwargs: dict | None = None,
    softcapping: bool = False,
    label: str = "",
    apply_model_patches_kwargs: dict | None = None,
    dtype: torch.dtype | None = None,
):
    """Compare per-parameter gradients between patched and unpatched models."""
    torch.manual_seed(0)
    unpatched, patched = build_patched_model_pair(
        config_cls,
        model_cls,
        device,
        attn_impl=attn_impl,
        config_kwargs=config_kwargs,
    )
    if dtype is not None:
        unpatched = unpatched.to(dtype)
        patched = patched.to(dtype)

    unpatched.train()
    patched.train()
    vocab = unpatched.config.vocab_size
    input_ids = torch.randint(0, vocab, (2, 10), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    # --- unpatched backward ---
    loss_ref = unpatched(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels
    ).loss
    loss_ref.backward()
    grads_ref = _collect_grads(unpatched)

    # --- patched backward ---
    with parity_model_patches(patched, apply_model_patches_kwargs):
        loss_test = patched(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        ).loss
        loss_test.backward()
        grads_test = _collect_grads(patched)

    actual_dtype = dtype or next(patched.parameters()).dtype
    tolerances = _get_tolerances(actual_dtype, softcapping)

    param_names_ref = set(grads_ref.keys())
    param_names_test = set(grads_test.keys())
    assert param_names_ref == param_names_test, (
        f"{label} gradient key mismatch: extra={param_names_test - param_names_ref}, "
        f"missing={param_names_ref - param_names_test}"
    )

    for name in sorted(grads_ref):
        _assert_tensors_close(
            f"backward grad {name}",
            grads_test[name],
            grads_ref[name],
            rtol=tolerances["rtol"],
            atol=tolerances["atol"],
            label=label,
        )


def assert_parity_vmap_grad(
    model_cls,
    config_cls,
    device,
    attn_impl: str = "sdpa",
    config_kwargs: dict | None = None,
    softcapping: bool = False,
    label: str = "",
    apply_model_patches_kwargs: dict | None = None,
    dtype: torch.dtype | None = None,
):
    """Compare vmap gradients against an upstream runtime-compatible reference."""
    torch.manual_seed(0)
    unpatched, patched = build_runtime_patched_model_pair(
        config_cls,
        model_cls,
        device,
        attn_impl=attn_impl,
        config_kwargs=config_kwargs,
        apply_model_patches_kwargs=apply_model_patches_kwargs,
    )
    if dtype is not None:
        unpatched = unpatched.to(dtype)
        patched = patched.to(dtype)

    unpatched.train()
    patched.train()
    batch, seq, vocab = 4, 12, unpatched.config.vocab_size
    input_ids = torch.randint(0, vocab, (batch, seq), device=device)
    attention_mask = torch.ones(batch, seq, dtype=torch.long, device=device)
    labels = input_ids.clone()

    def _vmap_grads(model):
        fmodel, trainable, frozen = make_functional(
            model, disable_autograd_tracking=True, partition_trainable=True
        )

        def per_example_loss(trainable_params, frozen_params, ids, mask, lbls):
            all_params = {**frozen_params, **trainable_params}
            outputs = fmodel(all_params, ids, attention_mask=mask, labels=lbls)
            return outputs.loss

        grad_fn, clip_state = clipped_grad(
            per_example_loss, argnums=0, batch_argnums=(2, 3, 4), clipping_norm=1.0
        )
        grads, _state = grad_fn(
            trainable, frozen, input_ids, attention_mask, labels, state=clip_state
        )
        return grads

    grads_ref = _vmap_grads(unpatched)
    grads_test = _vmap_grads(patched)

    actual_dtype = dtype or next(patched.parameters()).dtype
    tolerances = _get_tolerances(actual_dtype, softcapping)

    assert set(grads_ref.pytree.keys()) == set(grads_test.pytree.keys()), (
        f"{label} vmap_grad key mismatch"
    )
    for name in sorted(grads_ref.pytree):
        _assert_tensors_close(
            f"vmap_grad {name}",
            grads_test.pytree[name],
            grads_ref.pytree[name],
            rtol=tolerances["rtol"],
            atol=tolerances["atol"],
            label=label,
        )
