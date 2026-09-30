# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Model logit transforms relevant to raw-head alignment losses."""

from __future__ import annotations

_POST_HEAD_TRANSFORM_CAUSAL_LM = {
    ("transformers.models.cohere.modeling_cohere", "CohereForCausalLM"),
    ("transformers.models.cohere2.modeling_cohere2", "Cohere2ForCausalLM"),
    ("transformers.models.granite.modeling_granite", "GraniteForCausalLM"),
    ("transformers.models.gemma2.modeling_gemma2", "Gemma2ForCausalLM"),
    ("transformers.models.gemma3.modeling_gemma3", "Gemma3ForCausalLM"),
}


def _has_family_logit_transform(model: object) -> bool:
    """Whether raw lm-head projections can differ from model-native logits."""
    if hasattr(model, "peft_config"):
        model = getattr(getattr(model, "base_model", None), "model", None)
    if model is None:
        return False
    return any(
        (cls.__module__, cls.__name__) in _POST_HEAD_TRANSFORM_CAUSAL_LM
        for cls in type(model).__mro__
    )
