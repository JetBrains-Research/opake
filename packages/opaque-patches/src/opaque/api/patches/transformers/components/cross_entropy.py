# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""vmap-compatible causal-language-model cross-entropy replacements."""

from __future__ import annotations

import torch
import torch.nn as nn


def _pytorch_causal_lm_loss(
    logits,
    labels,
    vocab_size: int,
    ignore_index: int = -100,
    shift_labels=None,
    label_smoothing: float = 0.0,
    **kwargs,
) -> torch.Tensor:
    """Standard PyTorch cross-entropy loss for non-CUDA devices.

    Always reduces with mean-over-non-ignored-tokens (matches
    ``F.cross_entropy(..., reduction="mean", ignore_index=-100)``).
    Per-batch reductions that would couple per-example gradients are
    not supported — they'd break DP-SGD's per-example sensitivity bound.

    ``label_smoothing`` is a named parameter (not a kwarg) so the
    signature documents that opaque honors it.  HF's stock
    ``ForCausalLMLoss`` accepts it via ``**kwargs`` and silently drops
    it; HF Trainer rebuilds the loss separately.  Opaque honors it
    inline.
    """
    logits = logits.float()

    if shift_labels is None:
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    logits_flat = logits.view(-1, vocab_size)
    shift_labels_flat = shift_labels.view(-1)

    return nn.functional.cross_entropy(
        logits_flat,
        shift_labels_flat,
        ignore_index=ignore_index,
        label_smoothing=float(label_smoothing or 0.0),
    )


def _opaque_causal_lm_loss(
    logits,
    labels,
    vocab_size: int,
    ignore_index: int = -100,
    shift_labels=None,
    label_smoothing: float = 0.0,
    **kwargs,
) -> torch.Tensor:
    """CausalLM loss using Opaque cross-entropy Triton kernel.

    Supports all vocab sizes via chunked computation for vocab > 65536.
    Falls back to PyTorch cross-entropy on non-CUDA devices.  Always
    reduces with mean-over-non-ignored-tokens; per-batch reductions are
    not supported (would break DP-SGD per-example sensitivity).

    ``label_smoothing`` is a named parameter — the Triton kernel applies
    smoothing directly (matching ``F.cross_entropy(...,
    label_smoothing=...)``), and the CPU fallback passes it to
    ``F.cross_entropy``.
    """
    # Triton kernels require CUDA — fall back to standard CE on CPU/MPS
    if not logits.is_cuda:
        return _pytorch_causal_lm_loss(
            logits,
            labels,
            vocab_size,
            ignore_index,
            shift_labels,
            label_smoothing=label_smoothing,
            **kwargs,
        )

    from opaque.api.patches.kernels.cross_entropy import Opaque_CrossEntropyLoss

    logits = logits.float()

    if shift_labels is None:
        # Shift so that tokens < n predict n (same as HF ForCausalLMLoss)
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    logits_flat = logits.view(-1, vocab_size)
    shift_labels_flat = shift_labels.view(-1)
    shift_labels_flat = shift_labels_flat.to(logits_flat.device)

    losses, _ = Opaque_CrossEntropyLoss.apply(
        logits_flat, shift_labels_flat, 0, 0, float(label_smoothing or 0.0)
    )

    # Mask out ignored positions so they get zero upstream gradient
    mask = shift_labels_flat != ignore_index
    masked_losses = losses * mask.float()

    return masked_losses.sum() / mask.sum().clamp(min=1)


def apply_causal_lm_loss_function_patch(model, target_cls: type[nn.Module]) -> bool:
    """Attach Opaque's causal-LM loss to matching model instances."""
    if model is None:
        return False

    patched = False
    for module in model.modules():
        if type(module) is target_cls:
            module.loss_function = _opaque_causal_lm_loss
            patched = True
    return patched


# ForCausalLM classes eligible for fused linear + cross-entropy loss.
# All share identical structure: self.model(backbone) → self.lm_head → loss.
_FUSED_CE_CAUSAL_LM = [
    ("transformers.models.llama.modeling_llama", "LlamaForCausalLM"),
    ("transformers.models.mistral.modeling_mistral", "MistralForCausalLM"),
    ("transformers.models.ministral.modeling_ministral", "MinistralForCausalLM"),
    ("transformers.models.mellum.modeling_mellum", "MellumForCausalLM"),
    ("transformers.models.qwen2.modeling_qwen2", "Qwen2ForCausalLM"),
    ("transformers.models.qwen3.modeling_qwen3", "Qwen3ForCausalLM"),
    ("transformers.models.smollm3.modeling_smollm3", "SmolLM3ForCausalLM"),
    ("transformers.models.gemma.modeling_gemma", "GemmaForCausalLM"),
    ("transformers.models.gemma2.modeling_gemma2", "Gemma2ForCausalLM"),
    ("transformers.models.granite.modeling_granite", "GraniteForCausalLM"),
    ("transformers.models.cohere.modeling_cohere", "CohereForCausalLM"),
    ("transformers.models.cohere2.modeling_cohere2", "Cohere2ForCausalLM"),
    ("transformers.models.olmo2.modeling_olmo2", "Olmo2ForCausalLM"),
    ("transformers.models.olmo3.modeling_olmo3", "Olmo3ForCausalLM"),
    ("transformers.models.glm4.modeling_glm4", "Glm4ForCausalLM"),
    ("transformers.models.gemma3.modeling_gemma3", "Gemma3ForCausalLM"),
    ("transformers.models.exaone4.modeling_exaone4", "Exaone4ForCausalLM"),
]


def _fused_linear_ce_supports_class(target_cls: type[nn.Module]) -> bool:
    """Whether the generic fused forward matches this causal-LM structure."""
    identity = (target_cls.__module__, target_cls.__name__)
    return identity in _FUSED_CE_CAUSAL_LM or not target_cls.__module__.startswith(
        "transformers.models."
    )


def _fused_linear_ce_loss_is_supported(
    logits_to_keep: int | torch.Tensor,
    kwargs: dict,
) -> bool:
    """Return True only when fused linear+CE can match HF ``loss_function`` behavior.

    Fused path uses full-sequence hidden states and fixed label shift. Non-default
    ``logits_to_keep`` needs sliced logits (and label alignment) — defer to the
    original forward.
    """
    if torch.is_tensor(logits_to_keep):
        return False
    if not isinstance(logits_to_keep, int):
        return False
    if logits_to_keep != 0:
        return False
    if kwargs.get("shift_labels") is not None:
        return False
    if kwargs.get("weight") is not None:
        return False
    ii = kwargs.get("ignore_index", -100)
    return not torch.is_tensor(ii)


#: Instance attribute the family ``apply`` sets on a causal-LM module to say
#: whether the fused / chunked cross-entropy routes are enabled for it.  The
#: class-level wrapper is installed once per process, so the route policy of
#: a later model must not be frozen into that first closure.
FUSED_LINEAR_CE_ATTR = "_opaque_fused_linear_ce"


def _make_fused_ce_causal_lm_forward(
    original, *, force_chunked: bool | int = False, fused: bool = True
):
    """Loss-only ``ForCausalLM`` forward for per-example objectives.

    The wrapper is inert unless a call passes ``loss_only=True`` together
    with ``labels``; inference and logits-consuming labelled calls defer to
    ``original`` and keep the model-native contract.  A loss-only call
    computes the cross-entropy on one of three routes: the Triton fused
    linear + cross-entropy kernel (CUDA, bf16 / fp16), the portable chunked
    kernel (any other host, or ``force_chunked``), or the eager ``lm_head``
    plus ``loss_function`` branch when the fused routes are disabled
    (``fused=False``, or the instance attribute :data:`FUSED_LINEAR_CE_ATTR`
    set to ``False``) or when ``loss_function`` would need options the
    kernels do not support (non-zero ``logits_to_keep``, ``shift_labels``,
    class ``weight``, CUDA fp32 hidden states).  The fused routes return
    ``logits=None``; the eager route returns the full logits.

    ``output_router_logits=True`` (from the call or the config default) on a
    loss-only call asks a MoE backbone for its per-layer router logits and
    hands them back untouched in a ``MoeCausalLMOutputWithPast`` with
    ``aux_loss=None``: HF's batch-coupled load-balancing loss is neither
    computed nor added to ``loss``, because it has no per-example gradient
    and under ``vmap`` it would degenerate into the per-sequence objective.
    That HF contract is still reached without ``loss_only``, which defers to
    the original forward.
    """

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        logits_to_keep: int | torch.Tensor = 0,
        loss_only: bool = False,
        **kwargs,
    ):
        # The wrapper is inert unless this call explicitly permits a loss-only
        # result. Inference and logits-consuming labeled calls stay model-native.
        if labels is None or not loss_only:
            return original(
                self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

        # A MoE backbone records its per-layer router logits when asked.  On
        # this per-example path they are handed back untouched and HF's
        # batch-coupled load-balancing loss is never computed: it has no
        # per-example gradient, and under ``vmap`` it would be the
        # per-sequence objective.  The key is popped so it never reaches
        # ``loss_function``.
        output_router_logits = kwargs.pop("output_router_logits", None)
        if output_router_logits is None:
            output_router_logits = bool(
                getattr(self.config, "output_router_logits", False)
            )
        backbone_kwargs = kwargs
        if output_router_logits:
            backbone_kwargs = {**kwargs, "output_router_logits": True}

        # Resolve config defaults
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Call backbone
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **backbone_kwargs,
        )
        hidden_states = outputs[0]

        # CUDA + half precision routes to the Triton kernel; any other host
        # (MPS/CPU) routes to the pure-PyTorch chunked kernel, which streams the
        # log-sum-exp over bounded token/vocabulary tiles.  The wrapper is
        # installed with the compat bucket; whether these routes are enabled
        # is the performance decision the family ``apply`` records on the
        # instance (falling back to the install-time default).
        fused_enabled = bool(getattr(self, FUSED_LINEAR_CE_ATTR, fused))
        use_fused_ce = (
            loss_only
            and fused_enabled
            and _fused_linear_ce_loss_is_supported(logits_to_keep, kwargs)
            and (
                (
                    hidden_states.is_cuda
                    and hidden_states.dtype in (torch.bfloat16, torch.float16)
                )
                or not hidden_states.is_cuda
            )
        )

        if use_fused_ce:
            use_chunked_ce = not hidden_states.is_cuda or bool(force_chunked)
            if not use_chunked_ce:
                from opaque.api.patches.kernels.linear_cross_entropy import (
                    Opaque_LinearCrossEntropyLoss,
                )

                ce_loss_fn = Opaque_LinearCrossEntropyLoss.apply
            else:
                from opaque.api.patches.kernels._linear_ce_chunked import (
                    linear_nll_sum_chunked as ce_loss_fn,
                )

            weight = self.lm_head.weight

            # Keep family scaling scalar so a large transformed head is never
            # materialized. The kernels apply it to each logits tile and include
            # it in the hidden/head gradient chain rule.
            configured_logit_scale = getattr(self.config, "logit_scale", None)
            configured_logits_scaling = getattr(self.config, "logits_scaling", None)
            logit_scale = float(
                1.0 if configured_logit_scale is None else configured_logit_scale
            )
            logits_scaling = float(
                1.0 if configured_logits_scaling is None else configured_logits_scaling
            )
            logit_scale /= logits_scaling

            # Gemma2 softcapping: softcap * tanh(logits / softcap)
            softcap = getattr(self.config, "final_logit_softcapping", 0) or 0

            ignore_index = int(kwargs.get("ignore_index", -100))
            label_smoothing = float(kwargs.get("label_smoothing") or 0.0)

            ce_args = (
                hidden_states,
                weight,
                labels,
                ignore_index,
                softcap,
                label_smoothing,
                False,  # use_token_scaling: plain CE for the LM-head loss
                logit_scale,
            )
            if use_chunked_ce:
                chunk_vocab = (
                    force_chunked
                    if isinstance(force_chunked, int)
                    and not isinstance(force_chunked, bool)
                    else None
                )
                nll_sum = ce_loss_fn(*ce_args, chunk_vocab=chunk_vocab)
            else:
                nll_sum = ce_loss_fn(*ce_args)

            # Always reduce with mean-over-non-ignored-tokens; per-batch
            # reductions would break DP-SGD per-example sensitivity.
            shifted_labels = labels[..., 1:].contiguous().flatten()
            n_valid = (shifted_labels != ignore_index).sum().float().clamp(min=1)
            loss = nll_sum / n_valid

            logits = None
        else:
            # Match HF: slice hidden states like ``lm_head`` in modeling code.
            slice_indices = (
                slice(-logits_to_keep, None)
                if isinstance(logits_to_keep, int)
                else logits_to_keep
            )
            logits = self.lm_head(hidden_states[:, slice_indices, :])
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        if not return_dict:
            output = (logits, *outputs[1:])
            return (loss, *output) if loss is not None else output

        if hasattr(outputs, "router_logits"):
            from transformers.modeling_outputs import MoeCausalLMOutputWithPast

            # The batch-coupled auxiliary loss is never computed on this
            # per-example path (see the router-logits note above).
            return MoeCausalLMOutputWithPast(
                loss=loss,
                aux_loss=None,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                router_logits=outputs.router_logits,
            )

        from transformers.modeling_outputs import CausalLMOutputWithPast

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    return forward
