"""Structural serialization for generic mappings."""

from types import MappingProxyType

from opake.serialization import from_state_dict, state_dict


def test_mapping_proxy_roundtrip() -> None:
    template = MappingProxyType({"group": 0.0})

    serialized = state_dict(MappingProxyType({"group": 1.5}))
    restored = from_state_dict(template, serialized)

    assert serialized == {"group": 1.5}
    assert restored == {"group": 1.5}
