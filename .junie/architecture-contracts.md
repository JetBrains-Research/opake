# Architecture contracts

This document is the source of truth for Opake's package and API architecture.
The rules distinguish mechanically verifiable distribution properties from
decisions that require semantic review. Do not introduce source-tree inventories
of packages, modules, imports, or exports as proxies for these contracts.

## Enforcement model

Each contract has one primary enforcement mode:

- **Artifact** checks inspect generated wheels, sdists, and their metadata after
  CI/CD builds them.
- **Behavior** tests exercise a concrete supported operation in the owning
  package or against generated distributions.
- **Junie review** evaluates the pull-request diff semantically. Junie reports
  findings as code-review comments; humans decide how to resolve them.
- **Advisory** rules are design preferences. Junie may raise them as
  non-blocking feedback when a change introduces a relevant public surface.

Deterministic checks must operate on artifacts, declared metadata, or concrete
behavior. They must not rely on hand-maintained maps from distributions to
Python modules.

## Agent entry points

All supported development agents use this document as the architecture source
of truth:

| Agent | Repository entry point |
| --- | --- |
| Junie | `.junie/guidelines.md`; `.junie/review-guidelines.md` for review |
| Claude Code | `CLAUDE.md` |
| Copilot agents | `AGENTS.md` |
| Copilot code review | `.github/instructions/code-review.instructions.md` |

Agent-specific files contain only loading and workflow instructions; they do not
duplicate the contracts. Each entry point also loads
`.junie/differential-privacy-review.md` for mathematical and
privacy-sensitive changes.

Contracts marked **active** apply to current changes. Contracts marked
**planned** are accepted designs whose supporting metadata or post-build
infrastructure has not landed yet. Junie must not report the absence of planned
infrastructure as a violation.

| Contract | Status |
| --- | --- |
| ARC-001: Shared namespaces | Planned |
| ARC-002: Public façade separation | Active |
| ARC-003: Public documentation paths | Active |
| ARC-004: Backend-neutral distributions | Planned |
| ARC-005: Capability dependency graph | Planned |
| ARC-006: Test placement | Active |
| ARC-007: Deliberate public exports | Active |
| ARC-008: Accounting API ownership | Active |
| ARC-009: Legal artifacts and provenance | Planned |
| ARC-010: Import-time behavior | Active |
| ARC-011: Supported installation options | Planned |
| ARC-012: Callable strategy behavior | Active |

## Normative contracts

### ARC-001: Shared namespaces

Every published wheel must coexist with the other Opake wheels while
contributing to the shared `opake`, `opake.api`, and
`opake.api.accounting` PEP 420 namespaces.

**Enforcement:** Artifact validation inspects every generated wheel for regular
package initializers at those namespace roots, then installs the generated
artifacts together and runs a representative cross-wheel scenario. Installation
validation belongs to the post-build pipeline, not the repository pytest suite.

### ARC-002: Public façade separation

Public `opake.*` façade code adapts or re-exports supported APIs. Algorithm
implementation belongs under the owning `opake.api.*` tree. Façades may
contain re-exports, public type aliases, version metadata, deprecation adapters,
and justified lazy-loading infrastructure.

**Enforcement:** Junie reviews changed façade files for newly introduced
implementation behavior. Concrete runtime behavior may have focused tests; no
AST node allowlist enforces this rule.

### ARC-003: Public documentation paths

User-facing documentation and examples use supported public façades.
Contributor documentation, implementation documentation, tests, and traceback
discussion may reference `opake.api.*` when the internal path is relevant.
Stack walkthroughs use their corresponding public stack façade.

**Enforcement:** Junie reviews changed documentation and examples in context.
There is no fixed-directory regular-expression scan.

### ARC-004: Backend-neutral distributions

A distribution explicitly marked as backend-neutral must install and execute
its declared neutral scenario without any backend runtime.

**Enforcement:** Each distribution will declare provided and required
capabilities in its `pyproject.toml`. Post-build validation will discover
backend-neutral generated artifacts, verify that their dependency metadata has
no backend-runtime edge, install them without backend packages, and execute
their declared neutral scenario. Source import scans are not a substitute.

### ARC-005: Capability dependency graph

Each distribution declares the capabilities it provides and requires. A
distribution must not introduce a dependency edge forbidden by the capability
policy. Foundation and backend-neutral capabilities cannot acquire dependencies
on higher-level mechanisms, patches, optimizers, alignment integrations, or
trainers.

**Enforcement:** Repository CI validates per-package capability declarations and
dependency edges. Post-build validation confirms that generated dependency
metadata matches the declared distribution dependencies. Junie reviews semantic
coupling that metadata cannot represent. The capability vocabulary and
constraints belong in this document; distribution declarations belong in their
own `pyproject.toml`.

### ARC-006: Test placement

A wheel-local test must be runnable with that wheel, its declared
dependencies/extras, its backend requirements, and shared test tooling. Tests
that intentionally combine independent capabilities belong under
`tests/integration`.

**Enforcement:** Junie reviews added or moved tests against the owning
distribution metadata. Isolated per-wheel test environments may replace this
review only if they can be generated without a duplicate dependency manifest
and have acceptable CI cost.

### ARC-007: Deliberate public exports

Stable public façade package initializers declare a deliberate `__all__`.
Internal `opake.api.*` packages and ordinary modules are not universally
required to declare `__all__`.

**Enforcement:** Junie reviews changed public façade `__init__.py` files. There
is no recursive import or runtime-name parity test.

### ARC-008: Accounting API ownership

`opake.accounting` exposes backend- and stack-independent accounting algebra
and generic mechanisms. Stack- or mechanism-specific factories are exposed from
their owning public accounting façade.

**Enforcement:** Junie reviews changed accounting exports and documentation.
Focused package-local behavior tests exercise documented factories through
their owning façade. There is no global negative export inventory.

### ARC-009: Legal artifacts and provenance

Every published distribution contains the required license and notice material.
Repository provenance references resolve to real attributed sources.

**Enforcement:** Post-build validation dynamically inspects every generated
wheel and sdist. A machine-readable provenance manifest will become the
repository source of truth and will generate or validate NOTICE content; NOTICE
prose is not the primary data model.

### ARC-010: Import-time behavior

Importing a public Opake package must not automatically patch third-party
globals. Optional integrations must fail or no-op as their public API
documents. Lazy loading is an implementation detail unless a separate,
measurable startup or optional-dependency budget is adopted.

**Enforcement:** Junie reviews changes to package initializers and
patch/bootstrap paths. This policy is not enforced with generic import smoke
tests or import subprocess tests.

### ARC-011: Supported installation options

Every documented installation option has one meaningful post-build usage
scenario using generated artifacts. A scenario performs a minimal supported
operation rather than importing every module.

**Enforcement:** Tagged executable scenarios in the installation documentation
are the source of truth. Post-build validation extracts each scenario, installs
the generated artifacts for that option, and runs it.

### ARC-012: Callable strategy behavior

Public strategy factories accept their documented arguments and return
callable/state objects that satisfy their documented numerical contract.

**Enforcement:** Ordinary behavior tests live in the package that owns the
strategy. They test real calls and numerical properties, not package structure.

## Advisory design rules

The following preferences have legitimate exceptions. Junie may comment on
them when reviewing a newly introduced public API, but they are not blocking
contracts.

### ADV-001: Concern naming

New concern namespaces generally use singular names. A plural name is
appropriate when a collection is itself the public API.

### ADV-002: Type placement

Concern-specific public types generally live beside their concern. Genuinely
cross-cutting public types belong in `opake.types`.

### ADV-003: Factory-oriented APIs

APIs that build behavior generally prefer factory functions returning
callables. Classes remain appropriate for inert state/configuration and when
identity or lifecycle is itself the abstraction.

## Junie review protocol

Junie reviews only active contracts whose scope intersects the pull-request
diff. Each finding must:

1. cite the contract ID;
2. identify the affected changed lines;
3. explain the architectural consequence rather than only matching syntax; and
4. suggest a concrete remediation or request an explicit design decision.

Normative findings and advisory suggestions must be clearly distinguished.
Adding a package or module is never a violation by itself. Junie posts a code
review; successful completion of the review job does not mean that its findings
are automatically accepted or rejected.
