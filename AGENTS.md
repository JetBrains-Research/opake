# AGENTS.md

Agent briefing for the **Opaque** monorepo — a functional DP-SGD / DP-FTRL
library for PyTorch. See `README.md` and `CONTRIBUTING.md` for user docs.

## Project snapshot

- **Language**: Python 3.11+ (< 3.14) + Rust stable (≥ 1.83)
- **Package manager**: `uv`
- **Hardware**: GPU for training runs; CPU/MPS for most tests
- **Testing**: `pytest` (Python, ~1200 tests) + `cargo test` (Rust)

Opaque provides composable primitives for differentially private model
training in PyTorch. Built on `torch.func` (vmap, grad), every component
uses explicit state — no hooks, no subclassing, no hidden mutation.

## Packages (post-split layout)

Every sub-package lives under `opaque.*` / `opaque.api.*` as PEP 420
implicit namespaces. Implementation lives at `opaque.api.<contrib>.*`;
users import the same surface via thin re-export façades at
`opaque.<concern>` / `opaque.<stack>.<concern>`. Multiple wheels
contribute to the `opaque/` and `opaque/api/` namespaces — neither root
ships an `__init__.py`.

| Distribution | Import roots | Purpose | Depends on |
| --- | --- | --- | --- |
| `opaque` | — | umbrella pin for the default bundle | sub-wheels |
| `opaque-base` | `opaque.api.base.serialization`; façade `opaque.serialization` | Pure-Python serialization registry + dispatcher (the seam for `state_dict` / `from_state_dict`); no torch / numpy / optree | stdlib only |
| `opaque-engine` | `opaque.api.engine.{types,pytree,random,serialization,distributed,noise_allocation,clipping,functional,scheduling,profiling}`; façades `opaque.types`, `opaque.pytree`, `opaque.random`, `opaque.distributed`, `opaque.functional`, `opaque.scheduling`, `opaque.profiling` | Torch substrate: pytree wrappers (`ClippedPytree`, `NoisedPytree`, `PerGroup`), `RngKey`, fixed + AUTO-S clipping, schedules + warmup, DDP plumbing, profiler, structural state-dict for tensors/ndarrays/dataclasses, per-group / paired noise stddev math | `opaque-base`, torch, numpy, optree |
| `opaque-optimizers` | `opaque.api.optimizers`; façade `opaque.optimizers` | Torchopt-based functional optimizer chain (DP-aware AdamW-BC and friends) | `opaque-engine`, torchopt |
| `opaque-accounting` | `opaque.api.accounting.core` (+ Rust ext); façade `opaque.accounting` | PLD privacy accounting (PyO3 extension at `opaque.api.accounting.core.opaque_accounting`, aliased as `_native`); torch-free | `opaque-base` |
| `opaque-dpsgd` | `opaque.api.dpsgd.*`, `opaque.api.accounting.dpsgd.*`; façade `opaque.dpsgd` | Gaussian / per-group noise, adaptive clipping, Poisson + truncated-Poisson samplers, DP-SGD-specific accounting factories | `opaque-engine`, `opaque-accounting` |
| `opaque-dpftrl` | `opaque.api.dpftrl.*`, `opaque.api.accounting.dpftrl.*`; façade `opaque.dpftrl` | MF mechanisms (BLT, BSR, BiSR, band-MF, λ-CGD), private second moments, Poisson + b-min-sep + balls-in-bins + sequential samplers, DP-FTRL-specific accounting factories | `opaque-engine`, `opaque-accounting` |
| `opaque-auditing` | `opaque.api.auditing.*`; façade `opaque.auditing` | Empirical privacy auditing (one-run, coin-flip, loss attacks) | `opaque-engine`, `opaque-accounting` |
| `opaque-patches` | `opaque.api.patches.*`; façade `opaque.patches` | Torch checkpoint patches + HF Transformers compat (vmap-safe attention, KV cache) + fused Triton kernels (SwiGLU, GeGLU, RoPE, fused CE, LoRA) | `opaque-engine` |
| `opaque-transformers` | `opaque.api.transformers.*`; façade `opaque.transformers` | HF trainer + integration | `opaque-engine`, `opaque-patches`, transformers, peft |
| `opaque-federated` | `opaque.api.federated.*`; façade `opaque.federated` | Federated DP on IFED: population/cohort types, min-separation cohort sampler, round loader, per-client clipped-sum strategy, round driver | `opaque-engine`, ifed-sdk |

Sub-packages are independently installable; `pip install opaque-dpsgd`
pulls only `opaque-engine`, `opaque-accounting`, and their transitive
deps. `pip install opaque-accounting` alone is **torch-free** (only
`opaque-base` + the Rust extension).

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
uv run pytest packages/opaque-base/tests/
uv run pytest packages/opaque-engine/tests/
uv run pytest packages/opaque-optimizers/tests/
uv run pytest packages/opaque-dpsgd/tests/
uv run pytest packages/opaque-dpftrl/tests/
uv run pytest packages/opaque-auditing/tests/
uv run pytest packages/opaque-patches/tests/
uv run pytest packages/opaque-transformers/tests/
uv run pytest packages/opaque-federated/tests/
uv run pytest packages/opaque-accounting/tests/  # smoke; PLD factory tests live under dpsgd/dpftrl
```

## Installation matrix

```bash
pip install opaque-base                  # serialization registry only (stdlib-only, torch-free)
pip install opaque-engine                # torch substrate (types, pytree, clipping, distributed, ...)
pip install opaque-optimizers            # torchopt-based functional optimizers
pip install opaque-accounting            # PLD accounting (torch-free standalone)
pip install opaque-dpsgd                 # DP-SGD mechanisms
pip install opaque-dpsgd[optimizers]     # DP-SGD + opaque-optimizers
pip install opaque-dpftrl                # MF (DP-FTRL) mechanisms
pip install opaque-patches               # PyTorch checkpoint + HF compat patches
pip install opaque-patches[transformers] # + HF Transformers + PEFT extras
pip install opaque-transformers          # HF trainer integration
pip install opaque-federated             # federated DP on IFED
pip install "opaque[all]"                # everything
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
| `opaque-patches[transformers]` | `transformers`, `peft` |
| `opaque-dpsgd[optimizers]` | `opaque-optimizers` (torchopt-based functional optimizers) |
| `opaque-dpftrl[optimizers]` | `opaque-optimizers` |
| `opaque-accounting[cross-validation]` | `dp-accounting`, `riskcal` |
| `opaque[all]` | everything |

## Patching model (on-import)

`opaque.patches` exposes explicit entry points. `opaque.transformers`
does not patch Hugging Face globals at import time; `DPTrainer`
applies runtime and model patches during construction, and non-trainer
flows should call `opaque.patches.apply_runtime_patches()` once plus
`opaque.patches.apply_model_patches(model)` for each model instance.
There is no top-level `opaque.patch_all()`.

Patch submodules:

- `opaque.patches.torch` — gradient-checkpointing for `torch.utils.checkpoint`.
- `opaque.patches.kernels` — fused Triton kernels (SwiGLU, GeGLU, RoPE,
  fused CE, LoRA).
- `opaque.patches.transformers` — HF Transformers model patches
  (vmap-safe attention, KV cache, per-model component replacements).
- `opaque.patches.peft` — PEFT/LoRA patches (vmap-safe linear, MLP, QKV).
- `opaque.transformers` — compatibility-only runtime (Poisson-collator
  compat, trainer integration).

## Key architectural notes

### Kernel pattern (`opaque.performance.kernels`)

Triton kernels use a two-level `autograd.Function` for `vmap(grad())`
support: `Opaque_Foo` main entry + `_FooBackward` with their own `vmap()`
methods. New-style API (`setup_context()`); **not** compatible with
`@torch.amp.custom_fwd`/`@custom_bwd` (PyTorch #132388). Forward runs
under caller's autocast, backward has autocast OFF.

### Accounting native module

- Rust crate name: `opaque_accounting` (Cargo `[lib].name`, valid Rust
  identifier; used by doctests via `use opaque_accounting::...`).
- PyO3 `#[pymodule]` function: `opaque_accounting` → compiled artifact is
  `opaque/accounting/opaque_accounting.abi3.so`.
- maturin `module-name = "opaque.accounting.opaque_accounting"`,
  `python-packages = ["opaque.accounting"]`.
- The Python facade at `opaque.accounting/__init__.py` does
  `from . import opaque_accounting as _native`; all submodules continue to
  use `_native` as the private-impl alias. No top-level `opaque_accounting`
  Python module exists anywhere.

### Partition policy

`opaque-engine` holds algorithm-agnostic torch-using primitives.
Anything that only one algorithm would construct (DP-SGD adaptive
clipping, truncated Poisson; MF b-min-sep / cyclic / balls-in-bins /
sequential sampling, BLT/BSR/BiSR/band-MF/λ-CGD noise, private
second-moment streams) lives with that algorithm.

AUTO-S clipping (`auto_clipped_grad`) lives in `opaque-engine` because
its per-record sensitivity bound is constant and data-independent
(`sup_g ‖R · g / (‖g‖ + γ)‖ ≤ R`), making it compatible with both
DP-SGD's Gaussian mechanism and DP-FTRL's matrix-factorization
mechanisms — exactly like fixed clipping. Adaptive clipping is the only
clipping rule whose threshold drifts across steps, so it is the only one
that violates the constant per-step sensitivity assumption MF privacy
proofs require, and it correctly stays in `opaque.dpsgd.clipping`.

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
`tests/_support/opaque_test_support.py` module. It is a
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

- `uv sync` triggers a full Rust build of `opaque-accounting` via maturin
  (first run ~30s; cached afterwards).
- Pure library — no application server or database; testing is entirely
  `pytest` + `cargo test`.
- CUDA/MPS tests auto-skip when the accelerator is unavailable (marker-
  driven). HuggingFace compat tests also skip via `pytest.importorskip()`
  when `transformers` / `peft` aren't installed.
- CI guardrail: a single shell step in `.github/workflows/ci.yml`
  enforces that no sub-package ships `src/opaque/__init__.py` (the
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
- **Rust stable** (≥ 1.83) is pre-installed for the `opaque-accounting` PyO3 build.
- **uv** must be on `PATH` (`$HOME/.local/bin`). Install via
  `curl -LsSf https://astral.sh/uv/install.sh | sh` if missing.

### Running services

There are no long-running services. See the **Key commands** section above for
the canonical lint / test / Rust-test commands.

### Non-obvious gotchas

- The first `uv sync` triggers a full Rust/maturin build of `opaque-accounting`
  (~30 s cold, cached afterwards). Subsequent syncs are fast (~seconds).
- The namespace is PEP 420 — there is **no** `opaque.core` import path.
  Public primitives live at `opaque.{types,pytree,random,distributed,
  functional,scheduling,profiling,serialization,optimizers}` (provided by
  `opaque-base` + `opaque-engine` + `opaque-optimizers`); stack code
  imports clipping via `opaque.dpsgd.clipping` / `opaque.dpftrl.clipping`.
- `gaussian_noise` returns `(noise_fn, state)` and the inner `noise_fn` signature
  is `noise_fn(clipped_pytree, state) -> (noised_pytree, new_state)` (positional args).
- `clipped_grad` returns `(clip_fn, clip_state)` and `clip_fn` is called as
  `clip_fn(params, batch, state=clip_state) -> (ClippedPytree, new_state)`.
- `opaque.accounting` is the cross-cutting surface (composition, calibration,
  generic mechanisms, native Rust extension). Algorithm-specific factories
  (`gaussian`, `poisson`, `adaclip`, etc.) live in `opaque.dpsgd.accounting`;
  MF-specific ones (`band_mf`, `blt`, `bisr`, etc.) live in
  `opaque.dpftrl.accounting`.
- CUDA/MPS tests auto-skip; no special handling needed on CPU-only VMs.
- Running the `examples/` training scripts requires the `examples` dependency
  group (`uv sync --group examples --all-packages --extra all`).
- The example scripts download models and datasets from the Hugging Face Hub.
  Two constraints apply with the pinned `transformers` / `huggingface_hub`
  versions: the model must belong to a supported family (listed above), as
  unsupported architectures such as GPT-2 fail inside the opaque patches; and
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
