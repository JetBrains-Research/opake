# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""SwiGLU MLP replacements backed by Opake's vmap-safe kernels."""

from __future__ import annotations


def _make_swiglu_mlp_forward(original):
    """SwiGLU MLP forward using Opake Triton kernel."""

    def forward(self, x):
        if not x.is_cuda:
            return original(self, x)
        from opake.api.patches.kernels.swiglu import Opake_SwiGLU

        return self.down_proj(Opake_SwiGLU.apply(self.gate_proj(x), self.up_proj(x)))

    return forward


def _make_phi3_mlp_forward(original):
    """Phi3 MLP forward (combined gate_up_proj) using Opake Triton kernel."""

    def forward(self, hidden_states):
        if not hidden_states.is_cuda:
            return original(self, hidden_states)
        from opake.api.patches.kernels.swiglu import Opake_SwiGLU

        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return self.down_proj(Opake_SwiGLU.apply(gate, up))

    return forward
