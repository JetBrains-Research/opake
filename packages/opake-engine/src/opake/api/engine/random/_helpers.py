"""Convenience helpers for common RNG patterns.

This module provides ergonomic wrappers around the core RNG primitives for
typical use cases in DP training:

- ``random_key()`` - Non-deterministic keys for prototyping
- ``set_reproducible_pytorch_seed()`` - Configure PyTorch/CUDNN reproducibility
"""

from __future__ import annotations

import contextlib
import os
import secrets

from ._engine import RngKey, key, split


def random_key() -> RngKey:
    """Create a non-deterministic key using system entropy.

    Useful for prototyping when reproducibility is not critical. For production
    training, prefer ``key()`` with an explicit seed and ``fold_in()`` for
    per-step / per-rank derivation.

    Returns:
        A randomly initialized RngKey.

    Example:
        >>> from opake.api.engine.random import random_key
        >>> from opake.dpsgd.noise import gaussian_noise
        >>> k = random_key()
        >>> noise_fn, state = gaussian_noise(noise_multiplier=1.1, key=k)
    """
    random_seed = secrets.randbits(64)
    return key(random_seed)


def set_reproducible_pytorch_seed(key_val: RngKey) -> None:
    """Set all PyTorch/CUDNN seeds from a single RngKey for reproducible training.

    Configures PyTorch and CUDNN for deterministic behavior by setting:

    - ``torch.manual_seed()`` for CPU operations
    - ``torch.cuda.manual_seed_all()`` for all CUDA GPU operations
    - ``torch.mps.manual_seed()`` for Apple Silicon (MPS) operations
    - ``torch.backends.cudnn`` flags for deterministic convolutions
    - Environment variables for cuBLAS library determinism

    This function has side effects (modifies global state). Call it once at
    the start of your training script before creating any tensors or models.

    Args:
        key_val: RngKey from opake.api.engine.random module (typically from ``key()``
            or ``random_key()``).

    Example:
        Setup framework reproducibility once at startup, then use
        ``fold_in()`` for per-step DP operations:

        >>> from opake.api.engine.random import key, fold_in, set_reproducible_pytorch_seed
        >>> from opake.dpsgd.noise import gaussian_noise
        >>>
        >>> # At start of training - configure all PyTorch/CUDNN RNG sources
        >>> set_reproducible_pytorch_seed(key(42))
        >>>
        >>> # Then use fold_in for per-step DP randomness
        >>> base = key(42)
        >>> for step in range(num_steps):
        ...     step_key = fold_in(base, step)
        ...     noise_fn, state = gaussian_noise(noise_multiplier=1.1, key=step_key)
        ...     # ... training step ...

    Note:
        Setting determinism flags has a performance cost (typically 10-30% slower).
        See PyTorch documentation on ``torch.use_deterministic_algorithms()``
        and ``torch.backends.cudnn`` for details.

    See Also:
        - ``key()``: Create RngKey from integer seed
        - ``fold_in()``: Deterministic key derivation
        - ``random_key()``: Non-deterministic key from system entropy
        - torch.manual_seed: PyTorch CPU RNG seed setter
        - torch.use_deterministic_algorithms: PyTorch determinism flag
    """
    # Import here to avoid requiring torch as hard dependency at module load
    import numpy as np
    import torch

    # Derive independent seeds for different RNG sources
    seeds = split(key_val, num=3)
    torch_cpu_seed = seeds[0].seed % (2**31 - 1)  # Keep within C++ int32 range
    torch_gpu_seed = seeds[1].seed % (2**31 - 1)
    numpy_seed = seeds[2].seed % (2**32)

    # Set CPU and GPU seeds
    torch.manual_seed(int(torch_cpu_seed))
    torch.cuda.manual_seed_all(int(torch_gpu_seed))
    # Apple Silicon (MPS) has a separate RNG that ``torch.cuda.manual_seed_all``
    # (a lazy no-op off CUDA) never touches; seed it explicitly so model-side
    # stochastic ops (dropout, weight init) are reproducible on MPS too.
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(int(torch_gpu_seed))

    # Set numpy seed if available
    with contextlib.suppress(ImportError, RuntimeError):
        np.random.seed(int(numpy_seed))

    # Configure CUDNN for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set PyTorch deterministic algorithms (PyTorch >= 1.11)
    # May reduce performance and may raise errors if operations lack deterministic implementations
    with contextlib.suppress(AttributeError, RuntimeError):
        torch.use_deterministic_algorithms(True)

    # Configure cuBLAS for deterministic behavior
    # Format: ":size" or ":size:seed" (we use ":16:8" - 16MB workspace with seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


__all__ = ["random_key", "set_reproducible_pytorch_seed"]
