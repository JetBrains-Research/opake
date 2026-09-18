# Installation

> **Work in progress:** Opake is research software under active development.
> Its differential-privacy mechanisms, accounting, and privacy guarantees are
> still being validated and may change. Do not rely on it for production or
> compliance-sensitive privacy guarantees without independent validation for
> your use case.

## Requirements

- Python 3.11 through 3.13
- PyTorch 2.9 or later

## From JetBrains Packages

Install `opake` as the single public package entry point:

```bash
pip install opake \
  --index-url https://packages.jetbrains.team/pypi/p/fed/python/simple/
```

Or with `uv`:

```bash
uv add opake \
  --index https://packages.jetbrains.team/pypi/p/fed/python/simple/
```

### Extras

```bash
pip install "opake[auditing]"      # + opake-auditing (empirical privacy auditing)
pip install "opake[dpftrl]"        # + opake-dpftrl (correlated-noise mechanisms)
pip install "opake[transformers]"  # + opake-transformers + opake-patches[transformers]
pip install "opake[all]"           # everything above
```

## From Source

Clone the repository when developing Opake, inspecting its implementation, or
running its test suite. For ordinary use, prefer the published package above.

```bash
# Clone the repository
git clone https://github.com/JetBrains-Research/opake.git
cd opake

# Install with uv (recommended)
uv sync --group dev --all-packages --extra all
```

## Development Installation

```bash
# Install dev dependencies for all packages
uv sync --group dev --all-packages --extra all

# Verify installation (non-GPU tests)
uv run pytest -m "not cuda and not mps and not slow"
```

## Optional Dependencies

### Documentation

To build documentation locally:

```bash
uv sync --group docs
uv run mkdocs serve
```

Visit <http://localhost:8000> to view the docs.

## Verify Installation

```python
from importlib.metadata import version

print("opake-base version:", version("opake-base"))
print("opake-engine version:", version("opake-engine"))
print("opake-dpsgd version:", version("opake-dpsgd"))
```

`opake` itself is a [PEP 420] namespace with no top-level Python code, so
query the installed distributions individually.

## PyCharm

In PyCharm, select the `uv` interpreter for the project where you ran
`uv add`. Code completion and Quick Documentation follow Opake's public façade
imports, such as `opake.dpsgd.clipping`, `opake.dpsgd.noise`, and
`opake.accounting`; avoid copying `opake.api.*` paths from implementation
tracebacks into application code. Clone Opake and use an editable workspace
only when you need to debug or change its implementation.

[PEP 420]: https://peps.python.org/pep-0420/
