# AGENTS.md

Agent briefing for the **Opake** monorepo — a functional DP-SGD / DP-FTRL
library for PyTorch. See `README.md` and `CONTRIBUTING.md` for user docs.

## Project snapshot

- **Language**: Python 3.11+ (< 3.14) + Rust stable (≥ 1.83)
- **Package manager**: `uv`
- **Hardware**: GPU for training runs; CPU/MPS for most tests
- **Testing**: `pytest` (Python, ~1200 tests) + `cargo test` (Rust)

Opake provides composable primitives for differentially private model
training in PyTorch. Built on `torch.func` (vmap, grad), every component
uses explicit state — no hooks, no subclassing, no hidden mutation.

## Packages (post-split layout)

Every sub-package lives under `opake.*` / `opake.api.*` as PEP 420
implicit namespaces. Implementation lives at `opake.api.<contrib>.*`;
users import the same surface via thin re-export façades at
`opake.<concern>` / `opake.<stack>.<concern>`. Multiple wheels
contribute to the `opake/` and `opake/api/` namespaces — neither root
ships an `__init__.py`.

| Distribution | Import roots | Purpose | Depends on |
| --- | --- | --- | --- |
| `opake` | — | umbrella pin for the default bundle | sub-wheels |
| `opake-base` | `opake.api.base.serialization`; façade `opake.serialization` | Pure-Python serialization registry + dispatcher (the seam for `state_dict` / `from_state_dict`); no torch / numpy / optree | stdlib only |
| `opake-engine` | `opake.api.engine.{types,pytree,random,serialization,distributed,noise_allocation,clipping,functional,scheduling,profiling}`; façades `opake.types`, `opake.pytree`, `opake.random`, `opake.distributed`, `opake.functional`, `opake.scheduling`, `opake.profiling` | Torch substrate: pytree wrappers (`ClippedPytree`, `NoisedPytree`, `PerGroup`), `RngKey`, fixed + AUTO-S clipping, schedules + warmup, DDP plumbing, profiler, structural state-dict for tensors/ndarrays/dataclasses, per-group / paired noise stddev math | `opake-base`, torch, numpy, optree |
| `opake-optimizers` | `opake.api.optimizers`; façade `opake.optimizers` | Torchopt-based functional optimizer chain (DP-aware AdamW-BC and friends) | `opake-engine`, torchopt |
| `opake-accounting` | `opake.api.accounting.core` (+ Rust ext); façade `opake.accounting` | PLD privacy accounting (PyO3 extension at `opake.api.accounting.core.opake_accounting`, aliased as `_native`); torch-free | `opake-base` |
| `opake-dpsgd` | `opake.api.dpsgd.*`, `opake.api.accounting.dpsgd.*`; façade `opake.dpsgd` | Gaussian / per-group noise, adaptive clipping, Poisson + truncated-Poisson samplers, DP-SGD-specific accounting factories | `opake-engine`, `opake-accounting` |
| `opake-dpftrl` | `opake.api.dpftrl.*`, `opake.api.accounting.dpftrl.*`; façade `opake.dpftrl` | MF mechanisms (BLT, BSR, BiSR, band-MF, λ-CGD), private second moments, Poisson + b-min-sep + balls-in-bins + sequential samplers, DP-FTRL-specific accounting factories | `opake-engine`, `opake-accounting` |
| `opake-auditing` | `opake.api.auditing.*`; façade `opake.auditing` | Empirical privacy auditing (one-run, coin-flip, loss attacks) | `opake-engine`, `opake-accounting` |
| `opake-patches` | `opake.api.patches.*`; façade `opake.patches` | Torch checkpoint patches + HF Transformers compat (vmap-safe attention, KV cache) + fused Triton kernels (SwiGLU, GeGLU, RoPE, fused CE, LoRA) | `opake-engine` |
| `opake-transformers` | `opake.api.transformers.*`; façade `opake.transformers` | HF trainer + integration | `opake-engine`, `opake-patches`, transformers, peft |

Sub-packages are independently installable; `pip install opake-dpsgd`
pulls only `opake-engine`, `opake-accounting`, and their transitive
deps. `pip install opake-accounting` alone is **torch-free** (only
`opake-base` + the Rust extension).

## Architecture contracts

`.junie/architecture-contracts.md` is the single source of truth for
package boundaries, public API architecture, test placement, artifact
guarantees, and advisory API-design rules. Read it before planning, implementing,
or reviewing a change that affects those areas. Do not reproduce its full rule
set in agent instructions or source-tree inventory tests.

`.junie/differential-privacy-review.md` is the review protocol for
privacy-sensitive and mathematical changes. Read it when work affects privacy
mechanisms, sensitivity, clipping, noise, randomness, sampling, amplification,
composition, accounting, matrix strategies, distributed equivalence,
serialization of privacy state, auditing, or a mathematical privacy claim. Use
its literature map to verify theorem-dependent claims against primary sources.

For code review, also read and follow `.junie/review-guidelines.md`.


## Pull requests

The repo squash-merges. The PR title becomes the commit subject; the
PR body becomes the commit body (repo-level squash setting =
`PR_TITLE` + `PR_BODY`). Both feed `git-cliff` when release preparation builds
the draft Release body from an exact tag-to-candidate range.

**Title** — Conventional Commits form `<type>(scope): <imperative subject>`:

- Types `git-cliff` categorizes (see [cliff.toml](cliff.toml)):
  `feat` / `add` → Added, `fix` → Fixed,
  `refactor` / `change` / `perf` / `deps` → Changed, `docs` → Documentation,
  `test` → Tests, `ci` / `build` → CI/CD, `delete` → Removed,
  `chore` / `style` → skipped.
- Scope is optional but encouraged — e.g., `fix(accounting): …`.
- Breaking change: append `!` (`feat(dpsgd)!: …`) or include a
  `BREAKING CHANGE:` footer in the body.
- Subject starts with a lowercase letter and reads as an imperative
  (`add`, `fix`, `remove`) — not past-tense.
- The PR-gate workflow runs `amannn/action-semantic-pull-request@v6`
  and fails the check if the title doesn't parse.

**Body** — short prose:

- 2–4 sentences of "why" + what the change does. This text lands in
  `git log` and feeds the AI summary for every release line containing the
  commit.
- Keep it readable for a future spelunker; avoid checklist-only bodies.

**Gate** — on every push the PR workflow runs Linux amd64, dependency-boundary,
macOS arm64, Linux arm64, and CUDA validation, plus Rust tests, the docs build, title
validation, and autoformat checks. Preview wheels
(`0.X.Y.devN+pr.<num>.g<sha>`) build alongside and appear as
downloadable workflow artifacts on the run page (14-day retention).

## Key commands

```bash
uv sync --group dev --all-packages --extra all     # test suite: pytest, ruff, scipy + all package extras
uv sync --group examples --all-packages --extra all  # examples and all package extras
uv run pytest -m "not cuda and not mps and not slow and not distributed"  # PR-equivalent suite
uv run pytest -m "slow"                           # slow tests (run on push to main)
uv run ruff check packages/                      # lint
uv run ruff format --check packages/             # format check
cargo test --workspace                           # Rust tests
cargo test --workspace --lib -- --ignored        # Rust slow tests
```

Per-package tests:

```bash
uv run pytest packages/opake-base/tests/
uv run pytest packages/opake-engine/tests/
uv run pytest packages/opake-optimizers/tests/
uv run pytest packages/opake-dpsgd/tests/
uv run pytest packages/opake-dpftrl/tests/
uv run pytest packages/opake-auditing/tests/
uv run pytest packages/opake-patches/tests/
uv run pytest packages/opake-transformers/tests/
uv run pytest packages/opake-accounting/tests/  # smoke; PLD factory tests live under dpsgd/dpftrl
```

## Installation matrix

```bash
pip install opake-base                  # serialization registry only (stdlib-only, torch-free)
pip install opake-engine                # torch substrate (types, pytree, clipping, distributed, ...)
pip install opake-optimizers            # torchopt-based functional optimizers
pip install opake-accounting            # PLD accounting (torch-free standalone)
pip install opake-dpsgd                 # DP-SGD mechanisms
pip install opake-dpsgd[optimizers]     # DP-SGD + opake-optimizers
pip install opake-dpftrl                # MF (DP-FTRL) mechanisms
pip install opake-patches               # PyTorch checkpoint + HF compat patches
pip install opake-patches[transformers] # + HF Transformers + PEFT extras
pip install opake-transformers          # HF trainer integration
pip install "opake[all]"                # everything
```

### Dependency groups

The root `pyproject.toml` keeps three dev-facing dependency groups:

- `dev` — pytest, pytest-cov, ruff, scipy (statistical tests).
- `examples` — torchopt, datasets, wandb, and everything `examples/` scripts need.
- `docs` — mkdocs stack.

Everything else lives in the relevant package's
`[project.optional-dependencies]`:

| Extra | Pulls in |
| --- | --- |
| `opake-patches[transformers]` | `transformers`, `peft` |
| `opake-dpsgd[optimizers]` | `opake-optimizers` (torchopt-based functional optimizers) |
| `opake-dpftrl[optimizers]` | `opake-optimizers` |
| `opake-accounting[cross-validation]` | `dp-accounting`, `riskcal` |
| `opake[all]` | everything |

## Patching model (on-import)

`opake.patches` exposes explicit entry points. `opake.transformers`
does not patch Hugging Face globals at import time; `DPTrainer`
applies runtime and model patches during construction, and non-trainer
flows should call `opake.patches.apply_runtime_patches()` once plus
`opake.patches.apply_model_patches(model)` for each model instance.
There is no top-level `opake.patch_all()`.

Patch submodules:

- `opake.patches.torch` — gradient-checkpointing for `torch.utils.checkpoint`.
- `opake.patches.kernels` — fused Triton kernels (SwiGLU, GeGLU, RoPE,
  fused CE, LoRA).
- `opake.patches.transformers` — HF Transformers model patches
  (vmap-safe attention, KV cache, per-model component replacements).
- `opake.patches.peft` — PEFT/LoRA patches (vmap-safe linear, MLP, QKV).
- `opake.transformers` — compatibility-only runtime (Poisson-collator
  compat, trainer integration).

## Key architectural notes

### Kernel pattern (`opake.performance.kernels`)

Triton kernels use a two-level `autograd.Function` for `vmap(grad())`
support: `Opake_Foo` main entry + `_FooBackward` with their own `vmap()`
methods. New-style API (`setup_context()`); **not** compatible with
`@torch.amp.custom_fwd`/`@custom_bwd` (PyTorch #132388). Forward runs
under caller's autocast, backward has autocast OFF.

### Accounting native module

- Rust crate name: `opake_accounting` (Cargo `[lib].name`, valid Rust
  identifier; used by doctests via `use opake_accounting::...`).
- PyO3 `#[pymodule]` function: `opake_accounting` → compiled artifact is
  `opake/accounting/opake_accounting.abi3.so`.
- maturin `module-name = "opake.accounting.opake_accounting"`,
  `python-packages = ["opake.accounting"]`.
- The Python facade at `opake.accounting/__init__.py` does
  `from . import opake_accounting as _native`; all submodules continue to
  use `_native` as the private-impl alias. No top-level `opake_accounting`
  Python module exists anywhere.

### Partition policy

`opake-engine` holds algorithm-agnostic torch-using primitives.
Anything that only one algorithm would construct (DP-SGD adaptive
clipping, truncated Poisson; MF b-min-sep / cyclic / balls-in-bins /
sequential sampling, BLT/BSR/BiSR/band-MF/λ-CGD noise, private
second-moment streams) lives with that algorithm.

AUTO-S clipping (`auto_clipped_grad`) lives in `opake-engine` because
its per-record sensitivity bound is constant and data-independent
(`sup_g ‖R · g / (‖g‖ + γ)‖ ≤ R`), making it compatible with both
DP-SGD's Gaussian mechanism and DP-FTRL's matrix-factorization
mechanisms — exactly like fixed clipping. Adaptive clipping is the only
clipping rule whose threshold drifts across steps, so it is the only one
that violates the constant per-step sensitivity assumption MF privacy
proofs require, and it correctly stays in `opake.dpsgd.clipping`.

### Test design

Do not add tests whose only purpose is pinning prose in documentation,
READMEs, or docstrings to verbatim strings or required words. Test behavior or
stable machine-readable structure instead. A docs-only clarification may have
no dedicated regression test when neither is available.

### Test markers

Four orthogonal markers, declared in the root `pyproject.toml`:

- `cuda` — test needs CUDA; auto-skipped on non-CUDA hosts.
- `mps` — test needs Apple Metal; auto-skipped on non-MPS hosts.
- `slow` — test takes >5 s on CPU; excluded from PR CI (`and not slow`)
  and run on pushes to `main` (the CI job strips the `and not slow`
  clause conditionally).
- `distributed` — test launches multiple CPU/Gloo ranks; selected by the
  dedicated Linux distributed lane rather than general platform lanes.

Rust tests above five seconds use `#[ignore = "slow"]`. PR CI runs the default
unit/doc-test set; main and release additionally run the ignored library tests.

Gated HuggingFace models use `@requires_hf_auth` imported from the shared
`tests/_support/opake_test_support.py` module. It is a
`skipif(not has_hf_token())` mark, not a pytest marker. Set `HF_TOKEN`
(or `HUGGINGFACEHUB_API_TOKEN` / `HUGGINGFACE_TOKEN`) to run them.

CI lane marker expressions:

- PR Linux amd64 (locked): `-m "not cuda and not mps and not slow and not distributed"`.
- PR Linux amd64 (distributed): `-m "distributed and not cuda"`.
- PR Linux amd64 dependency boundaries (Python 3.11/3.13):
  `-m "not cuda and not mps and not slow and not distributed"`.
- PR macOS arm64: `-m "not cuda and not slow and not distributed"`.
- PR Linux arm64: `-m "not cuda and not mps and not slow and not distributed"`.
- PR CUDA locked (self-hosted): `-m "cuda and not slow"`.
- PR CUDA dependency boundaries (self-hosted, Python 3.11/3.13):
  `-m "cuda and not slow"`.
- Main Linux amd64 (locked): `-m "not cuda and not mps and not distributed"`.
- Main Linux amd64 (distributed): `-m "distributed and not cuda"`.
- Main Linux amd64 dependency boundaries (Python 3.11/3.13):
  `-m "not cuda and not mps and not slow and not distributed"`.
- Main macOS arm64: `-m "not cuda and not distributed"`.
- Main Linux arm64: `-m "not cuda and not mps and not distributed"`.
- Main CUDA locked (self-hosted): `-m "cuda"`.
- Main CUDA dependency boundaries (self-hosted, Python 3.11/3.13):
  `-m "cuda and not slow"`.
- Dependency selection uses the committed lock or uv's `lowest-direct` /
  `highest` strategies. Main platform lanes retain slow-test coverage.
  Every selected test, dependency resolution, and workflow failure blocks its
  caller.

### Supported HF model families

LLaMA / Mistral / Ministral / Qwen2 / Qwen3 / SmolLM3 / OLMo2 / OLMo3 /
GLM4 / Phi-3 / Gemma / Gemma2 / Gemma3 (text) / Granite / Cohere / Cohere2 /
Exaone4 / DeepSeek (inherits LLaMA). Text-first; see
`docs/user-guide/huggingface.md`. Nemotron is deferred (no
`eager_attention_forward` and a non-gated `NemotronMLP` in 4.57.1).

## Non-obvious notes

- `uv sync` triggers a full Rust build of `opake-accounting` via maturin
  (first run ~30s; cached afterwards).
- Pure library — no application server or database; testing is entirely
  `pytest` + `cargo test`.
- CUDA/MPS tests auto-skip when the accelerator is unavailable (marker-
  driven). HuggingFace compat tests also skip via `pytest.importorskip()`
  when `transformers` / `peft` aren't installed.
- CI guardrail: a single shell step in `.github/workflows/ci.yml`
  enforces that no sub-package ships `src/opake/__init__.py` (the
  PEP 420 invariant).

## Training examples

The `examples/` scripts are optional integration examples. Install the
`examples` dependency group before using them, inspect their command-line help,
and choose a compatible model and dataset available in your environment.
Configure any experiment tracking service through its own documented settings;
the repository does not require a particular provider.

## Documentation

- User-facing: `docs/` (MkDocs, Material theme).
  - End-to-end guides: `docs/user-guide/{dp-sgd,dp-ftrl}.md`.
  - Concept reference (per-topic): `docs/user-guide/{clipping,noise,
    sampling,accounting,distributed,...}.md`.
  - API reference (public façades): `docs/reference/`.
  - Mechanism reference (split per stack): `docs/mechanisms/{dp-sgd,
    dp-ftrl}/`.
  - Tutorials: `docs/tutorials/*.ipynb`.
- This file (`AGENTS.md`) is agent-oriented; users should read
  `docs/index.md` or `README.md`.
- Keep user-facing docs and code comments diary-free: describe the
  current API and behavior, not the development history (file moves,
  package regroups, removed dependencies, planned-but-unimplemented
  features). Migration narrative belongs in PR bodies and the
  changelog. Forward-references to features that don't yet exist in
  the codebase don't belong anywhere.

## Cursor Cloud specific instructions

This is a pure library — no application server, database, or external service
is needed. The development loop is entirely `uv sync` + `pytest` + `cargo test`.

### Environment prerequisites

- **Python 3.12** (system default on the VM) satisfies the `>=3.11,<3.14` constraint.
- **Rust stable** (≥ 1.83) is pre-installed for the `opake-accounting` PyO3 build.
- **uv** must be on `PATH` (`$HOME/.local/bin`). Install via
  `curl -LsSf https://astral.sh/uv/install.sh | sh` if missing.

### Running services

There are no long-running services. See the **Key commands** section above for
the canonical lint / test / Rust-test commands.

### Non-obvious gotchas

- The first `uv sync` triggers a full Rust/maturin build of `opake-accounting`
  (~30 s cold, cached afterwards). Subsequent syncs are fast (~seconds).
- The namespace is PEP 420 — there is **no** `opake.core` import path.
  Public primitives live at `opake.{types,pytree,random,distributed,
  functional,scheduling,profiling,serialization,optimizers}` (provided by
  `opake-base` + `opake-engine` + `opake-optimizers`); stack code
  imports clipping via `opake.dpsgd.clipping` / `opake.dpftrl.clipping`.
- `gaussian_noise` returns `(noise_fn, state)` and the inner `noise_fn` signature
  is `noise_fn(clipped_pytree, state) -> (noised_pytree, new_state)` (positional args).
- `clipped_grad` returns `(clip_fn, clip_state)` and `clip_fn` is called as
  `clip_fn(params, batch, state=clip_state) -> (ClippedPytree, new_state)`.
- `opake.accounting` is the cross-cutting surface (composition, calibration,
  generic mechanisms, native Rust extension). Algorithm-specific factories
  (`gaussian`, `poisson`, `adaclip`, etc.) live in `opake.dpsgd.accounting`;
  MF-specific ones (`band_mf`, `blt`, `bisr`, etc.) live in
  `opake.dpftrl.accounting`.
- CUDA/MPS tests auto-skip; no special handling needed on CPU-only VMs.
- Running the `examples/` training scripts requires the `examples` dependency
  group (`uv sync --group examples --all-packages --extra all`).
- The example scripts download models and datasets from the Hugging Face Hub.
  Two constraints apply with the pinned `transformers` / `huggingface_hub`
  versions: the model must belong to a supported family (listed above), as
  unsupported architectures such as GPT-2 fail inside the opake patches; and
  datasets must be referenced by their namespaced Hub id (`owner/name`), since
  the legacy single-name ids are no longer accepted.

### PR workflow

The PR title **must** follow Conventional Commits: `<type>(scope): <imperative subject>`.
The PR-gate workflow (`action-semantic-pull-request`) rejects titles that don't
parse. Accepted types: `feat`/`add`, `fix`, `refactor`/`change`/`perf`/`deps`,
`docs`, `test`, `ci`/`build`, `delete`, `chore`/`style`. Append `!` for breaking changes.
Subject starts lowercase and reads as an imperative (`add`, `fix`, `remove`).
See the **Pull requests** section above for full details.

1. Push changes and create/update the PR (with a valid Conventional Commits title).
2. Wait ~5 minutes for GitHub Copilot review comments to appear.
3. Read the Copilot comments — address the ones that make sense (fix the
   code or docs), ignore the ones that don't.
4. Reply inline to each comment explaining what you did (or why you
   disagree). Leave comments **unresolved** — the author resolves them.
5. Wait for CI/CD checks to complete. If any fail, read the logs
   (`gh run view --log`), fix the issue, push again, and repeat from step 2.
