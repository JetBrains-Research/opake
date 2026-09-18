"""Registry prefix joining / _subdict behaviour (sequence paths, empty keys)."""

from __future__ import annotations

from dataclasses import dataclass

from opake.serialization import (
    from_state_dict,
    register_serializer,
    state_dict,
)


@dataclass
class _RegNode:
    """Registered leaf whose flat keys use bracket and dotted relatives."""

    tag: str


def _reg_save(_o: _RegNode) -> dict[str, object]:
    return {"": "root", "[0]": 1, "inner.x": 2}


def _reg_load(_t: _RegNode, sd: dict[str, object]) -> _RegNode:
    assert sd[""] == "root"
    assert sd["[0]"] == 1
    assert sd["inner.x"] == 2
    return _RegNode(tag="ok")


@dataclass
class _Wrap:
    cell: _RegNode


@dataclass
class _WrapList:
    items: list[_RegNode]


def test_registered_paths_join_like_structural() -> None:
    register_serializer(_RegNode, _reg_save, _reg_load)
    w = _Wrap(cell=_RegNode(tag="a"))
    flat = state_dict(w)
    assert flat["cell"] == "root"
    assert flat["cell[0]"] == 1
    assert flat["cell.inner.x"] == 2
    out = from_state_dict(_Wrap(cell=_RegNode(tag="z")), flat)
    assert out.cell.tag == "ok"


def test_registered_under_list_prefix() -> None:
    wl = _WrapList(items=[_RegNode(tag="b")])
    flat = state_dict(wl)
    assert flat["items[0]"] == "root"
    assert flat["items[0][0]"] == 1
    assert flat["items[0].inner.x"] == 2
    out = from_state_dict(_WrapList(items=[_RegNode(tag="z")]), flat)
    assert len(out.items) == 1
    assert out.items[0].tag == "ok"
