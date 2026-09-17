"""Generic RNG engine with JAX-style key semantics.

This module provides explicit, immutable RNG state transitions inspired by JAX:

- ``key(seed)`` creates an initial key
- ``split(key, n)`` derives deterministic child keys
- ``fold_in(key, data)`` domain-separates by deterministic metadata

The engine is backend-agnostic and can be bridged to PyTorch via
``generator_from_key`` when APIs require ``torch.Generator``.
"""

from __future__ import annotations

import dataclasses
import hashlib

import torch

from opake.exceptions import ConfigurationError, InputTypeError

_MAX_TORCH_SEED = 2**63 - 1


def _to_uint64(value: int) -> int:
    return value & ((1 << 64) - 1)


def _stable_hash64(*parts: object) -> int:
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        if isinstance(part, int):
            h.update(b"i")
            h.update(part.to_bytes(16, byteorder="little", signed=True))
        else:
            h.update(b"s")
            h.update(str(part).encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest(), byteorder="little", signed=False)


@dataclasses.dataclass(frozen=True)
class RngKey:
    """Immutable RNG key.

    Attributes:
        seed: Canonical seed value used as key material.
        impl: Logical implementation identifier.
    """

    seed: int
    impl: str = "opake_threefry_like"


def key(seed: int) -> RngKey:
    """Create a PRNG key from an integer seed."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise InputTypeError(*(f"seed must be int, got {type(seed)}",))
    return RngKey(seed=_to_uint64(seed))


def fold_in(rng_key: RngKey, *data: int | str) -> RngKey:
    """Fold one or more values into a key to deterministically derive a new key.

    Accepts a variable number of int/str arguments. Each value is folded
    sequentially, so ``fold_in(k, a, b)`` equals ``fold_in(fold_in(k, a), b)``.

    Integers and strings are hashed down disjoint paths: ``fold_in(k, 1)`` and
    ``fold_in(k, "1")`` are different keys, and no sequence of integer folds
    can reach a key derived through a string fold.  That separation is what
    makes the domain convention work, so the two kinds are used for different
    jobs:

    - **Integers are the caller's.** Steps, ranks, epochs, leaf and group
      indices, and every key :func:`split` hands back are integer folds of a
      key you already hold.
    - **Strings root a mechanism.** A mechanism that draws randomness folds one
      unique string tag into the key it was given, once, and derives everything
      else beneath that tag.

    Skipping the string root is the failure this convention exists to prevent:
    ``fold_in(key, step)`` is the derivation *every* mechanism writes, so two
    mechanisms handed the same base key draw byte-identical noise, and nothing
    — not a test, an error, or an accountant — reports it.  See
    ``docs/reference/rng.md`` for the convention and the tags Opake's own
    mechanisms already occupy.

    Args:
        rng_key: Base key.
        *data: One or more int or str values to fold in sequentially.

    Returns:
        A new RngKey derived from the base key and all folded values.

    Raises:
        TypeError: If ``rng_key`` is not an :class:`RngKey` or any value is not
            an int (excluding ``bool``) or str.
        ValueError: If no data values are provided.

    Example:
        >>> k = key(42)
        >>> stream = fold_in(k, "mylab.rare_events")  # root your mechanism
        >>> fold_in(stream, step)                     # then step, rank, leaf...
        >>> fold_in(stream, step, rank)               # equals fold_in twice
    """
    if not isinstance(rng_key, RngKey):
        raise InputTypeError(*(f"rng_key must be RngKey, got {type(rng_key)}",))
    if not data:
        raise ConfigurationError(*("fold_in requires at least one data argument",))
    result = rng_key
    for d in data:
        if not isinstance(d, (int, str)) or isinstance(d, bool):
            raise InputTypeError(*(f"data must be int or str, got {type(d)}",))
        mixed = _stable_hash64(result.seed, d)
        result = RngKey(seed=mixed, impl=result.impl)
    return result


def split(rng_key: RngKey, num: int = 2) -> tuple[RngKey, ...]:
    """Split a key into ``num`` deterministic child keys.

    Defined as ``fold_in(rng_key, i) for i in range(num)``, so the children are
    exactly the integer folds of ``rng_key`` — which is why a mechanism roots
    itself with a string tag rather than a small integer: an integer root would
    hand out the same keys ``split`` does.  See :func:`fold_in`.
    """
    if not isinstance(num, int) or isinstance(num, bool):
        raise InputTypeError(*(f"num must be int, got {type(num)}",))
    if num < 1:
        raise ConfigurationError(*(f"num must be >= 1, got {num}",))
    return tuple(fold_in(rng_key, i) for i in range(num))


def generator_from_key(rng_key: RngKey) -> torch.Generator:
    """Create a deterministic ``torch.Generator`` from a key."""
    seed = rng_key.seed % _MAX_TORCH_SEED
    return torch.Generator().manual_seed(seed)


__all__ = ["RngKey", "fold_in", "generator_from_key", "key", "split"]
