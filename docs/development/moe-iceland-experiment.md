# Iceland MoE experiment branch

The experiment lives on **`david-stan/moe-iceland-experiment`**, based directly on
PR [#1080](https://github.com/JetBrains-Research/opaque/pull/1080) head
`bde08c91247cc2fe127e776bc72e402d8e7fef83`. The PR's source branch is
`claude/friendly-pasteur-uiiokv`.

The local worktree is `.temp/moe-iceland/`. The original checkout and its existing
changes are intentionally left alone. Do not copy its virtual environment or
legacy ZenML dependency pins into this worktree.

This is a **stacked experiment change**, not a modification of the mechanism PR:

1. Keep experiment runner/configuration, deployment, tests, and runbook changes
   on this branch; do not push to the PR author's branch.
2. Once reviewed locally, commit and push the experiment branch. A suitable
   Conventional Commits title is
   `feat(experiments): add an Iceland private MoE feasibility experiment`.
3. Open a draft stacked PR targeting `claude/friendly-pasteur-uiiokv` while #1080
   is open. Its description should explain the three-arm utility/routing question
   and distinguish local validation from cluster results. Do not present #1080's
   implementation changes as part of the experiment diff.
4. When #1080 lands, replay **only the experiment commits** onto the resulting
   `main`, retarget the PR, update the pinned mechanism provenance, rebuild the
   image if dependencies changed, and rerun the smoke/gate tests. A squash merge
   is not a reason to replay the entire mechanism branch.

No commit, push, PR creation, image push, or cluster submission is implicit in
preparing these files. Follow the repository's review/CI workflow when publishing
the branch; handle Copilot comments inline and leave resolution to the author.

The [experiment runbook](../../examples/moe_privacy/README.md) defines the data,
privacy unit, controls, metrics, local commands, and interpretation. The dedicated
[Iceland deployment runbook](../../deploy/zenml/moe_iceland/README.md) covers image
preparation, bounded source upload, resource gating, and artifact persistence.