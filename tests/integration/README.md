# tests/integration/

End-to-end tests that exercise multiple Opake wheels together. Run them with
the full workspace install: `uv sync --all-packages --extra all`.

## Layout

```
tests/integration/
├── README.md
├── accounting/              — cross-stack PLD / serialization accountants
├── noise/                   — MF vs DP-SGD Gaussian noise comparisons
├── sampling/                — DP-SGD vs DP-FTRL sampling-scheme separation
└── transformers/            — HF-anchored LoRA pipelines (synthetic + Hub)
    ├── test_dpsgd_pipeline.py
    ├── test_dpftrl_pipeline.py
    ├── test_ddp_pipeline.py
    ├── test_dpsgd_short_run_parity.py   — slow + cuda (1 vs 2 GPU)
    └── test_dpftrl_short_run_parity.py
```

## What lives here vs. in a wheel's `tests/`

A test belongs in `packages/<wheel>/tests/` when every package it imports is
in `<wheel>`'s transitive dependency cone. That covers the vast majority of
tests.

A test belongs here when it imports across two or more wheels with
**mutual non-dependency**, or when it exercises the end-to-end DP
pipeline (clipping + noise + optimizer + patches all in one).

Applying normal framework patches alone does not make a test a patches test;
it remains a DP-pipeline test.

## Markers

- **No marker**: synthetic-config integration tests run in the PR gate
  (CPU + MPS).
- **`slow`**: integration tests that download a model from HF Hub on
  first run; excluded from the PR gate, run on push to main.
- **`cuda`**: distributed / multi-GPU tests; auto-skip on hosts without
  CUDA. Wheel-local distributed suites under `packages/*/tests/ddp/`
  use this marker.

## Discovery

Pytest picks these up automatically: the root `pyproject.toml`
`testpaths` is `["packages", "tests"]`. Run them along with the rest of
the suite or in isolation:

```bash
uv run pytest tests/integration/                         # everything not slow / cuda
uv run pytest tests/integration/ -m slow                 # slow tests (HF downloads)
uv run pytest tests/integration/ -m cuda                 # CUDA tests (multi-GPU DDP)
uv run pytest tests/integration/accounting/              # accounting cross-stack
uv run pytest tests/integration/noise/                   # noise cross-stack
uv run pytest tests/integration/transformers/            # HF pipeline + short runs
```
