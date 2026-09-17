# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""GeGLU MLP replacements backed by Opake's vmap-safe kernels."""

from __future__ import annotations


def _make_geglu_exact_mlp_forward(original):
    """Gemma MLP forward using Opake GeGLU exact kernel."""

    def forward(self, x):
        if not x.is_cuda:
            return original(self, x)
        from opake.api.patches.kernels.geglu import Opake_GeGLU_Exact

        return self.down_proj(
            Opake_GeGLU_Exact.apply(self.gate_proj(x), self.up_proj(x))
        )

    return forward


def _make_geglu_approx_mlp_forward(original):
    """Gemma2 MLP forward using Opake GeGLU approx kernel."""

    def forward(self, x):
        if not x.is_cuda:
            return original(self, x)
        from opake.api.patches.kernels.geglu import Opake_GeGLU_Approx

        return self.down_proj(
            Opake_GeGLU_Approx.apply(self.gate_proj(x), self.up_proj(x))
        )

    return forward
