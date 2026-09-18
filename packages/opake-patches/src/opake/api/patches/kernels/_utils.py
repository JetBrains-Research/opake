# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
#
# General kernel utilities (the section above "Linear cross-entropy utilities")
# are extracted from the Unsloth project
# (Apache-2.0; https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/utils.py)
# and adapted for standalone use in opake kernels without the Unsloth runtime.
#
# Linear cross-entropy utilities (the section below) are ported from Apple's
# cut_cross_entropy project (Apache-2.0; https://github.com/apple/ml-cross-entropy).
#
# See ../../../../../NOTICE in this package for the full attribution.
"""Kernel utilities - self-contained, no unsloth runtime dependency.

Extracted from unsloth/kernels/utils.py for standalone use in opake kernels.
"""

from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from opake.exceptions import OperationError

# Constants
MAX_FUSED_SIZE: int = 65536
_IGNORE_INDEX = -100
_BLOCK_SIZE_FOR_32_WARPS = 32768
_BLOCK_SIZE_FOR_16_WARPS = 8192
_BLOCK_SIZE_FOR_8_WARPS = 2048
_MAX_BATCH_SIZE_BIN = 1024
_MIN_BATCH_SIZE_BIN = 128
_MAX_PER_ROW_KERNEL_BLOCK_SIZE = 256
_MIN_ROWS_FOR_BLOCK_KERNEL = 32768
next_power_of_2 = triton.next_power_of_2


def calculate_settings(n: int) -> tuple[int, int]:
    """Calculate block size and num_warps for Triton kernel launch.

    Args:
        n: Problem size (e.g., hidden dimension)

    Returns:
        Tuple of (BLOCK_SIZE, num_warps)
    """
    BLOCK_SIZE: int = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise OperationError(
            *(
                f"Cannot launch Triton kernel since n = {n} exceeds "
                f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}.",
            )
        )
    num_warps: int = 4
    if BLOCK_SIZE >= _BLOCK_SIZE_FOR_32_WARPS:
        num_warps = 32
    elif BLOCK_SIZE >= _BLOCK_SIZE_FOR_16_WARPS:
        num_warps = 16
    elif BLOCK_SIZE >= _BLOCK_SIZE_FOR_8_WARPS:
        num_warps = 8
    return BLOCK_SIZE, num_warps


# Device context manager
DEVICE_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0

if DEVICE_COUNT > 1:

    def torch_gpu_device(device):
        """Context manager for multi-GPU."""
        return torch.cuda.device(device)
else:

    def torch_gpu_device(device):
        """No-op context manager for single GPU."""
        return nullcontext()


# Triton version compatibility
def _get_triton_version():
    """Get triton version as tuple for comparison."""
    version_str = triton.__version__
    parts = version_str.split(".")
    return tuple(int(p) for p in parts[:3] if p.isdigit())


_TRITON_VERSION = _get_triton_version()

if _TRITON_VERSION >= (3, 0, 0):
    try:
        from triton.language.extra import libdevice

        triton_tanh = libdevice.tanh
    except ImportError:
        triton_tanh = tl.math.tanh
    triton_cast = tl.cast
else:
    triton_tanh = tl.math.tanh

    @triton.jit
    def triton_cast(x, dtype):
        return x.to(dtype)


# Int32 safety limits for large tensor indexing
NUM_INT32_ELEMENTS = 2**31
SAFE_INT32_BUFFER_MULTIPLIER = 4
BLOCK_SIZE_DEFAULT = 1024
INT32_SAFETY_BUFFER = (
    NUM_INT32_ELEMENTS - BLOCK_SIZE_DEFAULT * SAFE_INT32_BUFFER_MULTIPLIER
)


def needs_long_indexing(n_elements: int) -> bool:
    """Check if tensor requires int64 indexing."""
    return n_elements > INT32_SAFETY_BUFFER


# =============================================================================
# Linear cross-entropy utilities
# Ported from Apple's cut_cross_entropy project
# (Apache-2.0; https://github.com/apple/ml-cross-entropy)
# =============================================================================


def _build_flat_valids(
    targets: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor | None:
    """Build flat index tensor of valid (non-ignored) token positions.

    ``targets`` may be a shifted, non-contiguous view; indices are flattened
    in that view's logical order. Returns None if all tokens are valid.
    """
    valid_mask = targets != ignore_index
    valids = valid_mask.reshape(-1).nonzero().to(torch.int32)
    return valids.squeeze(1) if valids.numel() != valid_mask.numel() else None


def b_bin_fn(b: int) -> int:
    """Batch size binning for autotune key stability."""
    if b >= _MAX_BATCH_SIZE_BIN:
        return _MAX_BATCH_SIZE_BIN
    elif b <= _MIN_BATCH_SIZE_BIN:
        return _MIN_BATCH_SIZE_BIN
    else:
        return 512


def cast_to_dtype(dtype: torch.dtype, *tensors):
    """Cast floating tensors to ``dtype``, preserving all other arguments.

    Tensors already in ``dtype`` are returned unchanged, so callers can use this
    on the fast path without allocating a copy. ``None`` and non-floating tensors
    are passed through.
    """
    out = []
    for tensor in tensors:
        if (
            isinstance(tensor, torch.Tensor)
            and tensor.is_floating_point()
            and tensor.dtype != dtype
        ):
            out.append(tensor.to(dtype))
        else:
            out.append(tensor)
    return tuple(out)


def active_cuda_dtype(tensor: torch.Tensor) -> torch.dtype:
    """Return the active CUDA autocast dtype, or ``tensor.dtype`` otherwise."""
    if tensor.is_cuda and torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype("cuda")
    return tensor.dtype


def follow_autocast(*tensors):
    """Cast floating-point CUDA tensors to the active autocast dtype, if any.

    Public ``opake_*`` wrappers call this at the entry point so that a kernel
    used inside ``torch.autocast(device_type="cuda", dtype=...)`` runs in the
    autocast dtype end-to-end. Without this, the kernels are dtype-passthrough
    — not autocast-aware — producing a hybrid graph that defeats the user's
    autocast intent.

    Non-tensor arguments and integer tensors are returned unchanged. ``None``
    is passed through. The cast is a no-op when autocast is inactive or when
    a tensor already has the target dtype.
    """
    if not torch.is_autocast_enabled("cuda"):
        return tensors
    return cast_to_dtype(torch.get_autocast_dtype("cuda"), *tensors)


def ensure_cuda_tensors(*tensors: torch.Tensor, fn_name: str) -> None:
    """Validate that all tensors are CUDA tensors for Triton kernels.

    Args:
        *tensors: Input tensors to validate
        fn_name: Public API function name for error reporting

    Raises:
        RuntimeError: If any tensor is not on CUDA
    """
    for tensor in tensors:
        if not torch.is_tensor(tensor):
            continue
        if tensor.device.type != "cuda":
            raise OperationError(
                *(
                    f"{fn_name} requires CUDA tensors (Triton kernel backend); "
                    f"got device={tensor.device.type}. "
                    "Use non-kernel PyTorch path on MPS/CPU.",
                )
            )


# =============================================================================
# Triton JIT utilities for fused linear cross-entropy kernels
# =============================================================================

if _TRITON_VERSION >= (3, 0, 0):
    try:
        from triton.language.extra.libdevice import log1p as _tl_log1p
    except ImportError:
        _tl_log1p = tl.math.log1p
else:
    _tl_log1p = tl.math.log1p


@triton.jit
def tl_softcapping(v, softcap):
    return triton_tanh(v / softcap) * softcap


@triton.jit
def tl_softcapping_grad(dv, v, softcap):
    v = v / softcap
    return dv * (1 - v * v)


@triton.jit
def tl_logaddexp(a, b):
    minx = tl.minimum(a, b)
    mx = tl.maximum(a, b)
    return _tl_log1p(tl.exp(minx - mx)) + mx


@triton.jit
def tl_lock_add(ptrs, v, mask, lock_ptr):
    while tl.atomic_cas(lock_ptr, 0, 1) == 1:
        pass

    cur_v = tl.load(ptrs, mask=mask, other=0.0, eviction_policy="evict_last")
    new_v = v + cur_v
    tl.store(ptrs, new_v, mask=mask, eviction_policy="evict_last")

    tl.debug_barrier()
    tl.atomic_xchg(lock_ptr, 0)
