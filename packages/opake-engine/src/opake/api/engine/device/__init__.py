# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-device capability resolution (single source of truth).

See :mod:`opake.api.engine.device._capabilities` for the probe details.
"""

from opake.api.engine.device._capabilities import (
    DeviceCapabilities,
    device_capabilities,
    fused_kernels_available,
    sdpa_autocast_under_vmap_broken,
)

__all__ = [
    "DeviceCapabilities",
    "device_capabilities",
    "fused_kernels_available",
    "sdpa_autocast_under_vmap_broken",
]
