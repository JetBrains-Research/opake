# Installation

> **Work in progress:** Opake is research software under active development.
> Its differential-privacy mechanisms, accounting, and privacy guarantees are
> still being validated and may change. Do not rely on it for production or
> compliance-sensitive privacy guarantees without independent validation for
> your use case.

## Requirements

- Python 3.11 through 3.13
- PyTorch 2.9 or later

## From PyPI

Install `opake` as the single public package entry point:

```bash
pip install opake
```

Or with `uv`:

```bash
uv add opake
```

The umbrella `opake` distribution pulls in a curated bundle of sub-packages.
Each sub-package is also published separately (`opake-base`, `opake-engine`,
`opake-optimizers`, `opake-accounting`, `opake-dpsgd`, `opake-dpftrl`,
`opake-auditing`, `opake-patches`, `opake-transformers`, `opake-alignment`),
so you can install an individual distribution when you want a narrower
dependency footprint — for example `pip install opake-accounting` for the
torch-free PLD accounting alone. Prefer `opake` unless you have a specific
reason to pick sub-packages apart; the umbrella keeps the versions of the
sub-packages it bundles consistent.

### Extras

```bash
pip install "opake[auditing]"      # + opake-auditing (empirical privacy auditing)
pip install "opake[dpftrl]"        # + opake-dpftrl (correlated-noise mechanisms)
pip install "opake[alignment]"     # + opake-alignment (DP-safe SFT / DPO primitives)
pip install "opake[transformers]"  # + opake-transformers + opake-patches[transformers]
pip install "opake[trl]"           # + trl (config converters only; trainers are native)
pip install "opake[all]"           # everything above
```

## From JetBrains Packages

Every release published to PyPI is mirrored to the JetBrains Packages index.
The index additionally carries `.dev` wheels built from `main`, which are not
published to PyPI:

```bash
pip install opake \
  --index-url https://packages.jetbrains.team/pypi/p/fed/python/simple/
```

Or with `uv`:

```bash
uv add opake \
  --index https://packages.jetbrains.team/pypi/p/fed/python/simple/
```

To install an unreleased `.dev` wheel by hand, pin the exact version so the
index is consulted (`pip install opake==0.16.1.dev3 --index-url ...`).

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
