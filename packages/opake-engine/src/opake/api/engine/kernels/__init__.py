# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Private Triton kernels used by the engine.

Modules here import Triton at import time, so engine code loads them lazily and
only after :func:`opake.api.engine.device.fused_kernels_available` reports a CUDA
device with an importable Triton. Nothing in this package is public API.
"""
