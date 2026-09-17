# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""RMSNorm replacements backed by Opake's vmap-safe kernels."""

from __future__ import annotations


def _make_rms_norm_forward(
    original, *, casting_mode: str, offset: float, in_place_bwd: bool
):
    """RMSNorm forward using Opake Triton kernel."""

    def forward(self, hidden_states):
        if not hidden_states.is_cuda:
            return original(self, hidden_states)
        from opake.api.patches.kernels.rms_norm import Opake_RMSNorm

        eps = getattr(self, "variance_epsilon", None) or getattr(self, "eps", 1e-6)
        normalized, _, _ = Opake_RMSNorm.apply(
            hidden_states,
            self.weight,
            float(eps),
            float(offset),
            casting_mode,
            in_place_bwd,
            None,
        )
        return normalized

    return forward


def _rmsnorm_fac_llama(orig):
    return _make_rms_norm_forward(
        orig, casting_mode="llama", offset=0.0, in_place_bwd=True
    )


def _rmsnorm_fac_gemma(orig):
    return _make_rms_norm_forward(
        orig, casting_mode="gemma", offset=1.0, in_place_bwd=True
    )


def _rmsnorm_fac_gemma2(orig):
    return _make_rms_norm_forward(
        orig, casting_mode="gemma", offset=1.0, in_place_bwd=False
    )


def _rmsnorm_fac_olmo2(orig):
    # OLMo2 follows the same numeric formula as Llama-style RMSNorm but uses
    # non in-place backward in Liger's model-specific patching.
    return _make_rms_norm_forward(
        orig, casting_mode="llama", offset=0.0, in_place_bwd=False
    )


def _rmsnorm_fac_glm4(orig):
    return _make_rms_norm_forward(
        orig, casting_mode="llama", offset=0.0, in_place_bwd=False
    )
