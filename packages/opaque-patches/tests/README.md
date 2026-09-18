# `opaque-patches` tests

`torch/`, `peft/` and `transformers/` mirror the patch targets, so they
deliberately carry **no** `__init__.py`. As namespace portions they lose to the
installed regular packages of the same name, which keeps `import torch` working
when a tool (e.g. a PyCharm run configuration) puts this test root on
`PYTHONPATH`. Adding `__init__.py` back would shadow the real packages.

Module names stay unique via `--import-mode=importlib` in the root
`pyproject.toml`.
