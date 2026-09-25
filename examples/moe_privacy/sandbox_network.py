"""Fail-closed checks for a network-none Linux evaluator namespace."""

import errno
import fcntl
import socket
import struct
from pathlib import Path

_KERNEL_TUNNELS = {
    "tunl0",
    "gre0",
    "gretap0",
    "erspan0",
    "ip_vti0",
    "ip6_vti0",
    "sit0",
    "ip6tnl0",
    "ip6gre0",
}
_ACTIVE_FLAGS = 0x1 | 0x40  # IFF_UP | IFF_RUNNING
_SIOCGIFADDR = 0x8915


def validate_snapshot(
    interfaces: dict,
    ipv6_addresses: list[str],
    ipv4_routes: list[str],
    ipv6_routes: list[str],
) -> None:
    """Allow loopback and only inactive, unaddressed kernel-default tunnel devices."""
    if "lo" not in interfaces or set(interfaces) - ({"lo"} | _KERNEL_TUNNELS):
        message = "Network-none namespace contains an unknown interface"
        raise RuntimeError(message)
    for name, details in interfaces.items():
        if name != "lo" and (
            details["flags"] & _ACTIVE_FLAGS or details["ipv4"] is not None
        ):
            message = "A non-loopback interface is active or has an IPv4 address"
            raise RuntimeError(message)
    for rows in (ipv6_addresses, ipv6_routes):
        if any(not row.split() or row.split()[-1] != "lo" for row in rows):
            message = "Non-loopback IPv6 addressing/routing is forbidden"
            raise RuntimeError(message)
    if any(not row.split() or row.split()[0] != "lo" for row in ipv4_routes):
        message = "Non-loopback IPv4 routing is forbidden"
        raise RuntimeError(message)


def assert_network_disabled() -> None:
    """Inspect kernel namespace state; interface names alone do not prove isolation."""
    if not Path("/.dockerenv").is_file():
        message = "This network check is only for the isolated Docker evaluator"
        raise RuntimeError(message)
    names = {name for _, name in socket.if_nameindex()}
    sysfs = Path("/sys/class/net")
    if names != {path.name for path in sysfs.iterdir() if path.is_dir()}:
        message = "Socket/sysfs network namespace views disagree"
        raise RuntimeError(message)
    interfaces = {}
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as control:
        for name in sorted(names):
            address = None
            try:
                value = fcntl.ioctl(
                    control.fileno(), _SIOCGIFADDR, struct.pack("256s", name.encode())
                )
                address = socket.inet_ntoa(value[20:24])
            except OSError as error:
                if error.errno != errno.EADDRNOTAVAIL:
                    raise
            interfaces[name] = {
                "flags": int((sysfs / name / "flags").read_text(), 16),
                "ipv4": address,
            }
    validate_snapshot(
        interfaces,
        Path("/proc/net/if_inet6").read_text().splitlines(),
        Path("/proc/net/route").read_text().splitlines()[1:],
        Path("/proc/net/ipv6_route").read_text().splitlines(),
    )
