"""Fail-closed dispatch: MRO resolution + raise on unrecognised leaves.

These exercise the base wheel in isolation (no torch): the dispatcher must
resolve a subclass to its registered base handler, must raise rather than
silently drop a leaf no handler claims, and must honour the
``register_template_restored`` escape hatch for inert leaves.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from opake.serialization import (
    from_state_dict,
    register_serializer,
    register_template_restored,
    resolve_serializer,
    state_dict,
)


class _Base:
    def __init__(self, x: int) -> None:
        self.x = x


class _Sub(_Base):
    """Subclass with no handler of its own."""


def _base_save(obj: _Base) -> dict[str, object]:
    return {"": obj.x}


def _base_load(_template: _Base, sd: dict[str, object]) -> _Base:
    return _Base(int(sd[""]))


@dataclass
class _Box:
    cell: object


def test_subclass_resolves_to_base_handler() -> None:
    register_serializer(_Base, _base_save, _base_load)

    resolved = resolve_serializer(_Sub)
    assert resolved is not None

    flat = state_dict(_Box(cell=_Sub(7)))
    assert flat["cell"] == 7

    out = from_state_dict(_Box(cell=_Base(0)), flat)
    assert isinstance(out.cell, _Base)
    assert out.cell.x == 7


class _Opake:
    """A leaf no handler claims and no container matches."""


def test_save_raises_on_unrecognized_leaf() -> None:
    with pytest.raises(TypeError, match="Cannot serialize"):
        state_dict(_Box(cell=_Opake()))


def test_load_raises_on_unrecognized_leaf() -> None:
    with pytest.raises(TypeError, match="Cannot restore"):
        from_state_dict(_Box(cell=_Opake()), {})


def test_error_names_the_type_and_path() -> None:
    with pytest.raises(TypeError) as excinfo:
        state_dict({"deep": {"leaf": _Opake()}})
    message = str(excinfo.value)
    assert "_Opake" in message
    assert "deep.leaf" in message


def test_template_restored_leaf_is_inert_not_an_error() -> None:
    register_template_restored(_Opake)
    try:
        sentinel = _Opake()
        flat = state_dict(_Box(cell=sentinel))
        assert flat == {}

        template_leaf = _Opake()
        out = from_state_dict(_Box(cell=template_leaf), flat)
        assert out.cell is template_leaf
    finally:
        from opake.api.base.serialization._registry import _REGISTRY

        _REGISTRY.pop(_Opake, None)
