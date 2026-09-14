"""Composed/Repeated DpProcess registration must fire at import time.

``_PROCESS_REGISTRY`` is filled by the dataclass ``__init_subclass__``
hook on :class:`DpProcess` subclasses; deserialization
(:func:`opaque.api.accounting.core._process_codec._load_dp_process`)
looks types up by name from that registry.  If the composition module
imports :class:`Composed` / :class:`Repeated` lazily, plain
``import opaque.accounting`` leaves the registry incomplete and
round-tripping a tree containing those nodes fails with
``ValueError: Unknown nested DpProcess type 'Composed' in field 'inner'``.

Tests use only opaque-accounting-owned process types (EpsDelta,
Composed, Repeated, CachedProcess) so they don't require opaque-dpsgd
or opaque-dpftrl.
"""

from __future__ import annotations

import json

import opaque.accounting as acc
from opaque.api.accounting.core._base import _PROCESS_REGISTRY
from opaque.serialization import from_state_dict, state_dict


def test_process_registry_contains_composition_nodes() -> None:
    """Plain ``import opaque.accounting`` must register composition nodes."""
    assert "Composed" in _PROCESS_REGISTRY, (
        "Composed missing from _PROCESS_REGISTRY. Check the import "
        "side-effect in opaque/api/accounting/core/composition/__init__.py."
    )
    assert "Repeated" in _PROCESS_REGISTRY, (
        "Repeated missing from _PROCESS_REGISTRY. Check the import "
        "side-effect in opaque/api/accounting/core/composition/__init__.py."
    )
    assert "CachedProcess" in _PROCESS_REGISTRY


def test_cached_composed_repeated_roundtrips_via_json() -> None:
    """``cached(compose(repeat(EpsDelta, 3), EpsDelta))`` must round-trip.

    Mirrors the structural pattern Trainer writes into its accountant
    checkpoint: a cached opaque-boundary at the top, a Composed node
    splicing the prior phase's repeated step, and a Repeated wrapper
    for the homogeneous repetition.  This is the exact tree shape that
    raised on resume before the registry fix landed.
    """
    leaf_a = acc.eps_delta(0.5, 1e-6)
    leaf_b = acc.eps_delta(0.1, 1e-7)

    # Composed(Repeated(leaf_a, 3), leaf_b)  =  (leaf_a * 3) | leaf_b
    tree = acc.cached(acc.compose(acc.repeat(leaf_a, 3), leaf_b))

    sd = dict(state_dict(tree))
    # JSON round-trip exercises the codec's full path; the codec emits
    # only Python primitives and plain containers, so JSON is lossless.
    blob = json.dumps(sd)
    restored_sd = json.loads(blob)

    # Template-driven restore: from_state_dict overwrites a freshly-built
    # tree of the same shape.
    template = acc.cached(acc.compose(acc.repeat(leaf_a, 3), leaf_b))
    restored = from_state_dict(template, restored_sd)

    assert restored == tree, (
        "Tree did not round-trip through state_dict / from_state_dict; "
        "structural equality should hold for the (cached, composed, "
        "repeated, EpsDelta) tree."
    )
