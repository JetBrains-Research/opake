"""Deep DpProcess composition chains serialize, restore, and compare iteratively.

Regression tests for issue #334: heterogeneous accounted steps that never
merge grow one ``Composed`` node per step, and the recursive codec /
dataclass ``__eq__`` / ``__repr__`` overflowed the interpreter stack at a
few hundred steps — making long-running accountants un-checkpointable.
"""

from __future__ import annotations

import dataclasses
import gc
import tracemalloc

import pytest

import opake.accounting as acc
from opake.api.accounting.core._accountant import Accountant
from opake.api.accounting.core._process_codec import (
    _load_dp_process,
    _serialize_dp_process,
)
from opake.api.accounting.core.composition.types import (
    CachedProcess,
    Composed,
    Repeated,
)
from opake.exceptions import CheckpointError
from opake.serialization import from_state_dict, state_dict

DEPTH = 10_000


def _hetero_chain(depth: int):
    """Left-skewed Composed chain: heterogeneous steps that never merge."""
    p = acc.eps_delta(0.01, 1e-9)
    for i in range(1, depth):
        p = Composed(p, acc.eps_delta(0.01 + (i % 7) * 1e-6, 1e-9))
    return p


def test_depth_10k_round_trips_and_compares_equal():
    p = _hetero_chain(DEPTH)
    sd = state_dict(Accountant(prefix=p))  # no RecursionError
    restored = from_state_dict(Accountant(), sd)  # no RecursionError
    assert restored.process == p  # iterative eq, distinct objects
    assert hash(restored.process) == hash(p)
    assert repr(p)  # iterative repr


def test_deep_equality_detects_leaf_difference():
    a = _hetero_chain(DEPTH)
    b = Composed(_hetero_chain(DEPTH - 1), acc.eps_delta(0.5, 1e-9))
    assert a != b


def test_deep_wrappers_mixed():
    p = _hetero_chain(2_000)
    p = Repeated(CachedProcess(p), 3)
    sd = _serialize_dp_process(p)
    assert _load_dp_process(sd) == p


def test_wire_format_matches_recursive_reference_on_small_trees():
    def reference(p):
        """The pre-fix recursive codec, inlined as an oracle."""
        out = {"type": p.__class__.__name__}
        for f in dataclasses.fields(p):
            v = getattr(p, f.name)
            if isinstance(
                v, (Composed, Repeated, CachedProcess)
            ) or dataclasses.is_dataclass(v):
                out[f.name] = reference(v)
            elif isinstance(v, (int, float, bool, str, type(None), tuple, list)):
                out[f.name] = v
        return out

    small = [
        acc.eps_delta(0.1, 1e-9),
        Composed(acc.eps_delta(0.1, 1e-9), acc.eps_delta(0.2, 1e-9)),
        Repeated(acc.eps_delta(0.1, 1e-9), 5),
        CachedProcess(
            Composed(
                acc.eps_delta(0.1, 1e-9),
                Repeated(acc.eps_delta(0.2, 1e-9), 3),
            )
        ),
    ]
    for p in small:
        got = _serialize_dp_process(p)
        assert got == reference(p)  # wire format identical
        assert list(got) == list(reference(p))  # key order too
        rt = _load_dp_process(got)
        assert rt == p
        assert hash(rt) == hash(p)


def test_equality_matches_dataclass_semantics_on_small_trees():
    a = Composed(acc.eps_delta(0.1, 1e-9), acc.eps_delta(0.2, 1e-9))
    b = Composed(acc.eps_delta(0.1, 1e-9), acc.eps_delta(0.2, 1e-9))
    c = Composed(acc.eps_delta(0.1, 1e-9), acc.eps_delta(0.3, 1e-9))
    assert a == b
    assert a != c
    assert Repeated(acc.eps_delta(0.1, 1e-9), 2) != Repeated(
        acc.eps_delta(0.1, 1e-9), 3
    )
    assert a != acc.eps_delta(0.1, 1e-9)  # cross-class
    assert (a == object()) is False  # NotImplemented fallback


class TestCustomSerializerDispatch:
    """#334 review F1: wrapper children with CUSTOM serializers must load
    through them — the generic leaf parser cannot read their wire format."""

    @staticmethod
    def _make_custom_leaf():
        from dataclasses import dataclass

        from opake.api.accounting.core._base import DpProcess
        from opake.serialization import register_serializer

        # ``slots=True`` matters: it makes the decorator create a NEW
        # class, re-firing ``DpProcess.__init_subclass__`` with
        # ``__dataclass_fields__`` present — the production
        # registration route into ``_PROCESS_REGISTRY``.
        @dataclass(frozen=True, slots=True)
        class _CustomLeaf(DpProcess):
            payload: float

            def pld(self, **_kwargs):  # pragma: no cover - never evaluated
                raise NotImplementedError

        def _save(obj):
            # Deliberately NOT the generic field layout: the generic leaf
            # parser would reject the "blob" key.
            return {"type": "_CustomLeaf", "blob": f"v:{obj.payload}"}

        def _load(_template, sd):
            return _CustomLeaf(payload=float(sd["blob"].split(":", 1)[1]))

        register_serializer(_CustomLeaf, _save, _load)
        return _CustomLeaf

    def test_custom_child_of_each_wrapper_round_trips(self):
        leaf_cls = self._make_custom_leaf()
        for p in (
            Repeated(leaf_cls(1.5), 3),
            Composed(acc.eps_delta(0.1, 1e-9), leaf_cls(2.5)),
            CachedProcess(leaf_cls(3.5)),
        ):
            sd = _serialize_dp_process(p)
            restored = _load_dp_process(sd)
            assert restored == p, type(p).__name__


class TestAliasedSubDicts:
    """#334 review F2: the same dict object as two children must load."""

    def test_aliased_leaf_child(self):
        d = _serialize_dp_process(acc.eps_delta(0.1, 1e-9))
        sd = {"type": "Composed", "left": d, "right": d}
        restored = _load_dp_process(sd)
        assert restored == Composed(acc.eps_delta(0.1, 1e-9), acc.eps_delta(0.1, 1e-9))

    def test_aliased_wrapper_child(self):
        w = _serialize_dp_process(Repeated(acc.eps_delta(0.2, 1e-9), 2))
        sd = {"type": "Composed", "left": w, "right": w}
        restored = _load_dp_process(sd)
        inner = Repeated(acc.eps_delta(0.2, 1e-9), 2)
        assert restored == Composed(inner, inner)


class TestWrapperLoadErrors:
    """#334 review F5: wrapper-path error behavior is explicit."""

    def test_missing_type_raises_with_clear_message(self):
        with pytest.raises(CheckpointError, match="missing required field 'type'"):
            _load_dp_process({})

    def test_missing_child_field_raises(self):
        left = _serialize_dp_process(acc.eps_delta(0.1, 1e-9))
        with pytest.raises(CheckpointError, match="missing required field 'right'"):
            _load_dp_process({"type": "Composed", "left": left})

    def test_non_dict_child_raises_with_clear_message(self):
        with pytest.raises(
            CheckpointError, match="must be a serialized DpProcess dict"
        ):
            _load_dp_process({"type": "Composed", "left": 5, "right": 6})

    def test_extra_keys_raise(self):
        w = _serialize_dp_process(Repeated(acc.eps_delta(0.1, 1e-9), 2))
        w["stray"] = 1
        with pytest.raises(CheckpointError, match="unexpected keys"):
            _load_dp_process(w)


class TestDeepRepr:
    """#334 review F3: repr survives both spines and the cached-alternation."""

    def test_right_skewed_composed(self):
        p = acc.eps_delta(0.01, 1e-9)
        for _ in range(5_000):
            p = Composed(acc.eps_delta(0.01, 1e-9), p)  # grow the RIGHT spine
        assert repr(p).startswith("Composed(left=")

    def test_cached_composed_alternation(self):
        # The documented incremental idiom ``acct = cached(acct)`` per round.
        p = acc.eps_delta(0.01, 1e-9)
        for _ in range(3_000):
            p = CachedProcess(Composed(p, acc.eps_delta(0.01, 1e-9)))
        assert repr(p).startswith("CachedProcess(inner=")

    def test_repr_memory_proportional_to_output(self):
        """#334 review: a skewed spine must not retain a string per node
        (the memoized version peaked ~3 GB for a ~600 KB result)."""
        p = _hetero_chain(DEPTH)
        repr(_hetero_chain(2))  # warm lazy imports outside the traced window
        gc.collect()
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            s = repr(p)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        # Measured ~3x output; the quadratic version was ~5000x.
        assert peak < 20 * len(s)

    def test_shared_child_matches_dataclass_format(self):
        x = acc.eps_delta(0.1, 1e-9)
        p = Composed(x, x)
        assert repr(p) == f"Composed(left={x!r}, right={x!r})"
        r = Repeated(x, 4)
        assert repr(r) == f"Repeated(inner={x!r}, count=4)"
        c = CachedProcess(x)
        assert repr(c) == f"CachedProcess(inner={x!r})"
