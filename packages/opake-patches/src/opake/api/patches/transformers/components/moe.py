# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Forward patch for HF v5 stacked-weight MoE experts (``*Experts`` modules).

Swaps only ``forward(hidden_states, top_k_index, top_k_weights)`` onto the
vmap/DP-safe :func:`opake_moe`, leaving the router, aux loss, and
parameters untouched. Installing the replacement is a vmap-safety enabler
(DP-SGD needs it) and always applies — ``opake_moe`` runs on CPU/fp32 via a
pure-torch dense path, so there is no fallback to HF's (vmap-broken) forward.

*Which* internal path that forward takes is a separate performance gate:
``grouped`` (wired from the ``grouped_moe`` patch gate) picks the grouped-GEMM
fast path — kernel-fused Triton on CUDA bf16/fp16, ``torch._grouped_mm`` on
MPS/CPU — while ``grouped=False`` forces the dense ``Opake_MoE``. All paths are
numerically equivalent (see :func:`opake_moe`); only the dense one is the compat
fallback.
"""

from __future__ import annotations


def _make_moe_experts_forward(original, *, grouped=True):
    """Build a stacked-weight MoE experts ``forward`` using the Opake kernel.

    ``grouped`` (captured at patch time from the ``grouped_moe`` gate) selects the
    grouped-GEMM performance path vs the dense compat path inside
    :func:`opake_moe`.
    """

    def forward(self, hidden_states, top_k_index, top_k_weights):
        from opake.api.patches.kernels.moe import opake_moe

        # reshape (not view) keeps this vmap-traceable under an extra mapped dim.
        hidden_dim = self.gate_up_proj.shape[-1]
        orig_shape = hidden_states.shape
        x = hidden_states.reshape(-1, hidden_dim)
        idx = top_k_index.reshape(-1, top_k_index.shape[-1])
        weights = top_k_weights.reshape(-1, top_k_weights.shape[-1])

        out = opake_moe(
            x, self.gate_up_proj, self.down_proj, idx, weights, grouped=grouped
        )
        return out.reshape(orig_shape)

    return forward
