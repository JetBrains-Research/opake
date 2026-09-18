# opake-base

Pure-Python foundation for the Opake library. Ships the serialization
registry and dispatcher that every other `opake-*` wheel registers
handlers against:

- `opake.api.base.serialization._registry` — `register_serializer`,
  `lookup_serializer`.
- `opake.api.base.serialization._dispatch` — `state_dict`,
  `from_state_dict`.
- `opake.api.base.serialization._structural` — generic Python container
  walker (dataclass, NamedTuple, tuple, list, mappings, primitives). Torch
  tensors and NumPy arrays are registered as exact-type handlers from
  `opake-engine` and `opake-accounting` independently.

`opake-base` has **no third-party dependencies** — only the Python
standard library. This is what lets `opake-accounting` ship as a torch-free
standalone wheel: it depends only on `opake-base`.

The user-facing API lives at the `opake.serialization` façade (shipped by
this wheel).
