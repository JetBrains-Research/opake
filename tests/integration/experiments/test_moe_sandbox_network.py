"""Network preflight rejects active, addressed, routed and unknown interfaces."""

import pytest
from examples.moe_privacy.sandbox_network import _KERNEL_TUNNELS, validate_snapshot


def snapshot():
    return {
        "lo": {"flags": 0x49, "ipv4": "127.0.0.1"},
        **{name: {"flags": 0x80, "ipv4": None} for name in _KERNEL_TUNNELS},
    }


def test_accepts_only_inert_kernel_tunnels_or_loopback():
    validate_snapshot(snapshot(), [], [], ["kernel unreachable route lo"])
    validate_snapshot({"lo": snapshot()["lo"]}, [], [], [])


@pytest.mark.parametrize("name", sorted(_KERNEL_TUNNELS))
@pytest.mark.parametrize(
    ("field", "value"), [("flags", 0x1), ("flags", 0x40), ("ipv4", "10.0.0.2")]
)
def test_every_active_or_addressed_tunnel_is_rejected(name, field, value):
    interfaces = snapshot()
    interfaces[name][field] = value
    with pytest.raises(RuntimeError, match="active or has"):
        validate_snapshot(interfaces, [], [], [])


@pytest.mark.parametrize("field", ["ipv6_addresses", "ipv4_routes", "ipv6_routes"])
def test_non_loopback_routes_or_ipv6_addresses_rejected(field):
    arguments = {"ipv6_addresses": [], "ipv4_routes": [], "ipv6_routes": []}
    arguments[field] = ["gre0 route gre0"]
    with pytest.raises(RuntimeError, match="forbidden"):
        validate_snapshot(snapshot(), **arguments)


def test_unknown_interface_rejected_even_when_down_and_unaddressed():
    interfaces = snapshot()
    interfaces["eth0"] = {"flags": 0, "ipv4": None}
    with pytest.raises(RuntimeError, match="unknown interface"):
        validate_snapshot(interfaces, [], [], [])
