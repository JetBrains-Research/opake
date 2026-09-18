# `opake_transformers` integration tests

Trainer and HF-compat tests for `opake-transformers` live here. The directory
is not named `transformers` to avoid shadowing the installed package under
pytest’s importlib mode.

Implementation code under test is in `opake.api.transformers.trainer`;
public imports should use `opake.transformers` or
`opake.transformers.trainer`.
