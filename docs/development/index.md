# Development

This guide is for contributors working from a GitHub fork or a local clone of
Opake. It summarizes the maintained workflow in
[CONTRIBUTING.md](https://github.com/JetBrains-Research/opake/blob/main/CONTRIBUTING.md);
consult that file for the full testing, documentation, and pull-request conventions.

## Get the source

Fork [JetBrains-Research/opake](https://github.com/JetBrains-Research/opake),
then clone your fork and add the upstream repository if you use it to keep your
branch current:

```bash
git clone https://github.com/<your-account>/opake.git
cd opake
git remote add upstream https://github.com/JetBrains-Research/opake.git
```

Opake requires Python 3.11 through 3.13, Rust stable, and
[uv](https://docs.astral.sh/uv/). Create the complete contributor environment:

```bash
uv sync --group dev --all-packages --extra all
```

Add only the dependency groups required for the work at hand:

```bash
uv sync --group docs --all-packages
uv sync --group examples --all-packages --extra all
```

## Navigate the workspace

Opake is a `uv` workspace of independently installable packages that
contribute to the shared `opake` and `opake.api` PEP 420 namespaces. Package
implementations live under `packages/opake-*/src/opake/api/`; the matching
`src/opake/` directories are public façades that re-export the supported user
API.

The accounting package includes the Rust/PyO3 extension. Run its Cargo tests
from the repository root along with the relevant Python tests when changing the
native boundary.

## Validate a change

Run the smallest relevant package test first, then the normal contributor
checks before opening a pull request:

```bash
uv run pytest -m "not cuda and not mps and not slow"
uv run ruff check packages/
uv run ruff format --check packages/
uv run --group docs mkdocs build --strict
cargo test --workspace
```

CUDA, MPS, and slow tests have separate pytest markers; hardware-specific tests
skip when their backend is unavailable. See the
[test-marker reference](https://github.com/JetBrains-Research/opake/blob/main/CONTRIBUTING.md#test-markers-and-filtering)
for the available selections.

## Submit a pull request

Use a focused branch from your fork, add regression coverage for behavior
changes, and ensure public API additions include type annotations and
Google-style docstrings. The pull-request title follows Conventional Commits,
for example `fix(accounting): handle empty compositions`; its body should
briefly explain the problem and solution. See the
[pull-request checklist](https://github.com/JetBrains-Research/opake/blob/main/CONTRIBUTING.md#pull-request-process)
for the complete checklist.
