"""Engine bootstrap.

Importing any module under ``opake.api.engine`` (or any of the engine
façades — ``opake.types``, ``opake.pytree``, …) triggers this
``__init__`` first, which performs the side-effect imports that wire
engine handlers into the foundation registries:

- ``serialization`` registers ``torch.Tensor`` / ``numpy.ndarray`` as
  exact-type handlers with the ``opake.api.base.serialization``
  registry. Without this import, ``state_dict(tensor)`` would skip
  tensors as opake leaves.
"""

from __future__ import annotations

import opake.api.engine.serialization  # noqa: F401  (registers tensor/ndarray handlers)
