# GitHub Actions Workflows

This directory contains Opaque's CI/CD configuration. Workflow files define
triggers, permissions, and pipeline policy; shared implementation lives in
private callable workflows described below.

## Entry-point workflows

| Workflow | Trigger | Responsibility |
|---|---|---|
| `pr.yml` | Pull requests to `main`, manual dispatch | Required Linux amd64, dependency-boundary, macOS arm64, Linux arm64, and CUDA checks plus preview-wheel artifacts. |
| `ci.yml` | Pushes to `main` | Linux amd64, dependency-boundary, macOS arm64, Linux arm64, and CUDA validation plus development-wheel publication. |
| `prepare-release.yml` | Manual dispatch from `main` or `release/X.Y` | Resolves a release line, reuses a successful `CI (main)` run for its exact SHA when available (otherwise runs the complete release test matrix), builds and validates its exact SHA, and either stops as a non-mutating dry run or creates the maintenance branch and complete draft Release. |
| `release.yml` | Published GitHub Release, manual tag recovery | Verifies attached Release assets, publishes them idempotently to JetBrains Packages, and deploys immutable documentation. |
| `docs.yml` | Pushes to `main`, manual dispatch, callable workflow | Builds rolling documentation or deploys a caller-selected immutable release version. |
| `autoformat.yml` | Pull requests to `main` | Checks and, for trusted PRs, applies Python and Rust formatting fixes. |
| `junie-review.yml` | Pull requests to `main` | Runs Junie as a repository reviewer using the branch's Junie guidance and architecture contracts. |
| `junie.yml` | Trusted `@junie-agent` or `/junie` commands in issues and pull requests | Runs interactive Junie tasks, including code changes and pull-request updates. |
| `build-devcontainer.yaml` | Devcontainer changes and manual dispatch | Builds, smoke-tests, and publishes the development container. |

## Callable components

### `.github/actions/setup-python`

This composite action installs the pinned Python version and optionally uv. Use
`install-uv: "false"` for Python-only jobs. uv cache restore/save is disabled
repository-wide: caching a full all-extras environment benefits only one
platform/dependency combination, while the pruned cache omits the pre-built
wheels that dominate installation time. Release-capable called workflows keep
pinned external setup steps inline so checking out an older release tag cannot
remove the local action implementation.

### `.github/actions/setup-rust`

This composite action installs Rust stable, optional toolchain components, and
a shared Cargo dependency cache. Use `components: clippy, rustfmt` for
formatting jobs.

### `.github/actions/run-qodana`

This composite action downloads the selected coverage artifacts, runs Qodana,
and uploads its SARIF report. The PR workflow calls it through `qodana.yml` as
a required check, except on Dependabot pull requests because they cannot access
the `QODANA_TOKEN` Actions secret. The main workflow uses the same action in an
allowed-to-fail job so policy violations remain visible without gating wheel
publication.

### `.github/workflows/build-distributions.yml`

This private `workflow_call` workflow builds all Python wheels, native wheels,
the umbrella wheel, and the accounting sdist from caller-supplied package
matrices. `pr.yml`, `ci.yml`, and release preparation discover the matrices once
and supply them with an artifact prefix, retention period, and, when necessary,
an explicit build version. This renders checks as `Build / <distribution>`
without an internal discovery child.

Each build job validates its own wheel metadata. Native artifact jobs also
validate accounting policy, and the sdist job proves that the source artifact
can rebuild a wheel. The callable workflow intentionally does not own
credentials, publication, cross-package validation, or pipeline gates.

### `.github/workflows/python-tests.yml`

This private `workflow_call` workflow runs the shared Python test matrix:
Rust/Python/uv setup, dependency synchronization, a CUDA preflight, pytest with
coverage, and Codecov upload. Each caller provides one runner environment plus
its Python version, dependency selection, pytest marker filter, and xdist
arguments. Its five fixed test-group matrix legs keep each caller collapsed as
one environment block in the Actions graph while the groups still run in
parallel.
Dependency selection is `locked`, `minimum` (uv `lowest-direct`), or `latest`
(`uv` `highest`). Validation callers report every test phase taking at
least five seconds, so newly slow tests cannot disappear behind a fixed-size
duration table. Every selected pytest failure, dependency resolution failure,
and workflow failure blocks its caller.

`pr.yml`, `ci.yml`, and release preparation invoke the callable workflow for
Linux amd64 tests, minimum and latest dependency boundaries, and macOS arm64,
Linux arm64, and CUDA platform coverage. Main and release include slow tests in
platform lanes. Fork pull requests never receive the self-hosted CUDA runner.

Lanes selecting `cuda` run `.github/scripts/assert_cuda_available.sh` and fail
when the host has no usable CUDA device; lanes excluding `cuda` skip it.

### `.github/workflows/rust-tests.yml`

This private `workflow_call` workflow runs the accounting crate's unit and doc
tests with the shared Rust setup and dependency cache. Entry workflows call it
as `Rust`, so checks render as `Rust / opaque-accounting`. Tests above five
seconds use `#[ignore = "slow"]`: PRs run the default set, while main and
release add `cargo test --lib -- --ignored` after the default unit/doc-test run.

### `.github/workflows/validate-distributions.yml`

This private `workflow_call` workflow downloads a complete caller-selected
artifact family, installs `opaque[all]` using only built Opaque wheels, and runs
a representative DP-SGD + DP-FTRL cross-stack accounting scenario without
checking out the source tree. PR, main, and release differ only in artifact
prefix.

### `.github/workflows/prepare-release-implementation.yml`

`prepare-release.yml` is a stable branch-selected dispatcher. It calls the
implementation on `main` while passing the selected branch and SHA explicitly,
so workflow-only fixes do not need backports. Before running release source
tests, the implementation looks for a successful `CI (main)` run for the
exact target SHA. It reuses that coverage when present, waits up to two hours
for an in-progress matching run, fails if that run fails, and otherwise runs
the full release matrix. This lets maintenance commits without a main CI run
retain their normal release validation. The implementation rejects sources
other than `main` and `release/X.Y`, creates a missing maintenance branch or
safely fast-forwards an existing compatible branch only after the candidate is
green, and preserves handwritten release prose outside the three generated
fences.

## Artifact contracts

The distribution workflow uploads artifacts named `<prefix>-<distribution>`.
Callers preserve the existing prefixes:

| Caller | Prefix | Retention |
|---|---|---|
| PR previews | `preview-wheels` | 14 days |
| Main development builds | `wheels` | 30 days |
| Release preparation | `candidate` | 90 days |

Release preparation merges the `candidate-*` family into a checksummed
`release-candidate` Actions artifact and attaches its distributions plus
`release-manifest.json` to the draft GitHub Release. The published-Release
workflow consumes only those attached assets; it never rebuilds them.

## Security and maintenance

Actions remain pinned to immutable commit SHAs. The maintenance-branch
dispatcher deliberately calls this repository's current `main` release
workflow; its selected source SHA remains immutable and is the only code tested
or packaged. Entry-point workflows default to read-only permissions and
elevate permissions only for trusted preparation or publishing jobs. Registry
credentials exist only in `release.yml` after a maintainer publishes the draft.
Fork pull requests do not run untrusted code on the self-hosted GPU runner and
never receive repository, package, or cloud credentials.

The active `main` ruleset requires `Build documentation`, `Format Python`,
`Format Rust`, `Conventional Commits PR title`, the selected individual Python
environment/package checks, `Rust / opaque-accounting`, and `Junie review`.
The review workflow uses the `JUNIE_API_KEY` Actions secret. Fork and Dependabot
pull requests cannot receive the secret-backed Junie review; the job records
that limitation and completes without invoking Junie.

Interactive Junie sessions follow `.junie/guidelines.md`; automated reviews use
`.junie/review-guidelines.md`. The review file is also the canonical policy for
Copilot code review through `.github/instructions/code-review.instructions.md`.
Reviewers must apply every relevant active architecture contract.
Privacy-sensitive reviews trace the guarantee end to end and use read-only web
search and URL fetching to verify primary literature before reporting
theorem-dependent findings.

The automated review and interactive workflows use the same SHA-pinned upstream
Junie action and `JUNIE_API_KEY`. Only non-bot authors with an `OWNER`,
`MEMBER`, or `COLLABORATOR` association reach the action, which then verifies
that the actor has repository write or admin access. The job grants `contents`,
pull-request, and issue write access explicitly; repository-wide workflow
permissions remain read-only. Repository Settings > Actions > General must also
allow GitHub Actions to create and approve pull requests.

Interactive Junie tasks use the default `GITHUB_TOKEN`, so comments and commits
are attributed to `github-actions[bot]`. GitHub does not start new `push` or
`pull_request` workflow runs for changes made with that token. After Junie
changes a branch, a maintainer must trigger CI for the new head, for example by
closing and reopening the pull request or by pushing a maintainer-authored
commit. The automated repository review uses the same identity.
