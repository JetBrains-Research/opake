"""Regression tests for mechanism RNG domain separation."""

from __future__ import annotations

import torch

from opake.api.dpsgd.clipping._adaptive import ADAPTIVE_CLIPPING_STREAM_FOLD
from opake.api.dpsgd.noise._gaussian import GAUSSIAN_STREAM_FOLD
from opake.dpsgd.noise import gaussian_noise
from opake.random import fold_in, generator_from_key, key, split
from opake.types import clipped


def _draw(rng_key, size: int = 6) -> torch.Tensor:
    return torch.randn(size, generator=generator_from_key(rng_key))


def _mechanism_draw(base, *, root: str, step: int, size: int = 6) -> torch.Tensor:
    """A mechanism written the way the docs say to write one."""
    return _draw(fold_in(base, root, step), size=size)


class TestMechanismRoots:
    def test_distinct_roots_give_distinct_streams(self):
        base = key(1234)
        first = _mechanism_draw(base, root="mylab.rare_events", step=3)
        second = _mechanism_draw(base, root="otherlab.canary_noise", step=3)
        assert not torch.equal(first, second)

    def test_unrooted_draws_collide(self):
        """Why the root is not optional — the failure mode, pinned.

        ``fold_in(key, step)`` is the derivation every mechanism reaches for
        first.  Two mechanisms that stop there share a stream exactly.
        """
        base = key(1234)
        mine = _draw(fold_in(base, 3))
        theirs = _draw(fold_in(base, 3))
        assert torch.equal(mine, theirs)

    def test_researcher_root_misses_every_shipped_mechanism(self):
        """A new mechanism cannot land on a stream Opake already draws from."""
        base = key(1234)
        mine = _mechanism_draw(base, root="mylab.rare_events", step=0)
        for shipped in (GAUSSIAN_STREAM_FOLD, ADAPTIVE_CLIPPING_STREAM_FOLD):
            assert not torch.equal(mine, _mechanism_draw(base, root=shipped, step=0))

    def test_sampled_integer_derivations_miss_the_gaussian_stream(self):
        base = key(1234)
        reachable = {child.seed for child in split(base, 16)}
        reachable |= {fold_in(base, index).seed for index in range(256)}
        assert fold_in(base, GAUSSIAN_STREAM_FOLD).seed not in reachable
        assert fold_in(base, ADAPTIVE_CLIPPING_STREAM_FOLD).seed not in reachable


class TestSharedKeyAcrossShippedMechanisms:
    def test_gaussian_noise_does_not_collide_with_a_researcher_mechanism(self):
        """The end-to-end shape of the bug: one key, two mechanisms."""
        base = key(1234)
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=base)
        noised, _ = noise_fn(clipped({"a": torch.zeros(6)}, max_norm=1.0), state)

        mine = _mechanism_draw(base, root="mylab.rare_events", step=0)
        assert not torch.equal(noised.pytree["a"], mine)

        # ...and specifically not the unrooted derivation either, which is what
        # gaussian_noise itself used to draw from.
        assert not torch.equal(noised.pytree["a"], _draw(fold_in(base, 0)))
