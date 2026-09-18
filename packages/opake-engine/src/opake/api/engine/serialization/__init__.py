"""Torch / numpy serialization handlers registered against the base registry.

Importing this module is enough to register the ``torch.Tensor`` and
``numpy.ndarray`` exact-type handlers with
``opake.api.base.serialization``. Engine's ``__init__.py`` triggers
this on engine load, so every consumer that imports anything from
engine (which is anyone using torch in opake) gets tensor / ndarray
serialization for free.
"""

from __future__ import annotations

from opake.api.engine.serialization import _structural  # noqa: F401

__all__: list[str] = []
