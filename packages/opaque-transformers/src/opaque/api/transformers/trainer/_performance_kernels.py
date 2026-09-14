"""``use_performance_kernels`` / ``performance_kernels_config`` integration.

Trainer drives opaque-patches via two split umbrellas:

* ``use_compat_patches`` → ``compat`` (vmap-safety: ``eager_attention``,
  ``batchify``, vmap-safe masking / collator / checkpoint hooks).
* ``use_performance_kernels`` → ``kernels`` (CUDA + Triton kernel group:
  ``rope``, ``rms_norm``, ``activation``, and ``cross_entropy``).
* ``fused_linear_cross_entropy`` inherits from the always-on ``performance``
  bucket; the trainer activates it only for calls known to be loss-only.

The ``performance`` bucket stays on by default regardless of
``use_performance_kernels``: ``kv_cache`` is a pure
Python patch that disables HF's ``DynamicCache`` allocation, which
otherwise leaks vmap references and inflates training memory.  Users who
want to keep the cache (e.g. for an HF model whose forward depends on
it) opt out explicitly via ``performance_kernels_config={"kv_cache":
False}``.

``performance_kernels_config`` is a flat ``dict[str, bool | int]`` forwarded
without key translation. Supported keys mirror the opaque-patches surface:
``rope``, ``rms_norm``, ``activation``, ``cross_entropy``,
``fused_linear_cross_entropy``, ``chunked_linear_cross_entropy``, ``kv_cache``,
``eager_attention``, ``batchify``. The chunked setting accepts a positive
maximum vocabulary tile width while token tiling remains automatic; ``False``
or ``0`` disables it.
"""

from __future__ import annotations

from typing import Any


def apply_performance_kernels_via_opaque_patches(
    model: Any,
    kernel_config: dict[str, Any] | None = None,
) -> None:
    """Apply opaque-patches kernels with a flat opaque-shaped config.

    Mutates ``model`` in place.  When ``kernel_config`` is ``None`` every
    supported kernel for the model family is enabled (full performance
    set) while compat wrappers stay on.

    **Trainer** applies the same stack internally — callers rarely
    need this function directly.
    """
    from opaque.patches import apply_model_patches

    apply_model_patches(
        model,
        performance=True,
        compat=True,
        kernels=True,
        **(kernel_config or {}),
    )


__all__ = [
    "apply_performance_kernels_via_opaque_patches",
]
