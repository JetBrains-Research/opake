# Opake development guidelines

Before planning, implementing, or reviewing a change:

1. Read `AGENTS.md` for repository layout, commands, and workflow.
2. Read `.junie/architecture-contracts.md` and apply every normative
   active contract whose scope intersects the change.
3. For privacy-sensitive or mathematical work, read
   `.junie/differential-privacy-review.md`. Trace the claim through
   sensitivity, clipping, noise, sampling, composition, accounting, and auditing
   as applicable.
4. Treat the advisory rules in the architecture document as design feedback, not
   automatic blockers.
5. Do not report missing infrastructure for contracts marked planned.

For implementation work, preserve the architecture contracts while making the
smallest complete change. Use the owning package's public façade in user-facing
code and its `opake.api.*` implementation tree for implementation. Add or move
tests according to ARC-006 and run the smallest relevant existing validation.

For code review, read and follow `.junie/review-guidelines.md`.

If a requested change intentionally revises an architecture contract, stop and
propose the policy change explicitly instead of silently working around it.
