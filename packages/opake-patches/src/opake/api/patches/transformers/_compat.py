# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""transformers major-version markers — use only where v4/v5 genuinely fork and
can't be handled universally (prefer ``getattr``/``hasattr`` guards elsewhere)."""

from __future__ import annotations

try:
    import transformers
    from packaging.version import parse as _parse

    _MAJOR = _parse(transformers.__version__).major
except Exception:  # transformers absent — patches no-op anyway
    _MAJOR = 0

IS_TRANSFORMERS_V5 = _MAJOR >= 5  # noqa: PLR2004 - Transformers v5 API boundary
IS_TRANSFORMERS_V4 = _MAJOR == 4  # noqa: PLR2004 - Transformers v4 API boundary


__all__ = ["IS_TRANSFORMERS_V4", "IS_TRANSFORMERS_V5"]
