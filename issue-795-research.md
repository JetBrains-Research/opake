# Issue 795: bounded-memory BISR noise execution

Research snapshot: 2026-08-31 UTC

## Post-research implementation update (2026-09-01 UTC)

The implementation work was rebased on the then-current `origin/main` at
[`e64846c999701d58095a8800e50434dc922a1de1`](https://github.com/JetBrains-Research/opake/commit/e64846c999701d58095a8800e50434dc922a1de1).
This matters because [PR #857](https://github.com/JetBrains-Research/opake/pull/857)
had since merged as
[`e87a0c217728da8a25086359c715d30f0d7d66c1`](https://github.com/JetBrains-Research/opake/commit/e87a0c217728da8a25086359c715d30f0d7d66c1),
so the signed-encoder accounting safeguards described below as pending are part
of the implementation base and must be preserved rather than cherry-picked.
The typed, namespaced RNG folding work and realized-standard-deviation coverage
were also already present on this base.

The selected implementation follows the report's focused recommendation: a
BISR-local, immutable, newest-first window of at most
`min(p, n_steps) - 1` iid noise pytrees; signed direct FIR convolution; and the
finite-horizon normalization scale folded into each per-step effective tap
`d_t q_k` before tensor multiplication. It delegates iid sampling and step-key
derivation to the generic MF streaming engine, preserving the existing
`opake.dpftrl.mf_gaussian` namespace and one fresh draw per call.
`BisrStrategy.streaming_matrix()` remains unchanged as the dense numerical
reference.

That ring is a complete solution to #795's stated `O(p)`-buffer and `O(p*d)`
requirements. It is bounded in the horizon, not memory-free: its persistent
tensor state still scales with bandwidth and model size. Eliminating those
buffers through PRNG replay, or consolidating BISR into a reusable generic
banded-inverse executor, would be useful follow-up work but is not a condition
for closing #795.

The checkpoint policy is intentional incompatibility, not replay migration.
The bounded inner state has an explicit layout version and legacy dense BISR
states fail with a targeted error. Reconstructing a correct ring from an old
state is not generally safe: the dense state contains correlated outputs rather
than iid draws, historical per-step scales are not stored in the standalone
noise state, and older checkpoints do not prove the RNG-layout and internal
compute-dtype provenance needed for a faithful inversion. New-layout
checkpoints retain the iid window, but exact continuation additionally requires
the same BISR execution identity and base noise scale as the original run. The
layout version prevents an incompatible history restore; it does not prove that
a rebuilt mechanism uses the same calibrated multiplier. Open
[issue #789](https://github.com/JetBrains-Research/opake/issues/789) is the
separate, urgent calibrated-resume defect. It is not fixed by this bounded-state
change.

## Executive conclusion

[Issue #795](https://github.com/JetBrains-Research/opake/issues/795) is a
confirmed implementation defect. BISR is defined by a banded inverse strategy
matrix, so its runtime noise operator is a finite impulse-response convolution
over at most `p` iid noise vectors. Opake instead recovers the dense,
full-horizon forward strategy and feeds it to the generic forward-substitution
executor. That executor is mathematically correct, but it retains
`n_steps - 1` gradient-sized correlated outputs and performs work proportional
to that history on every step. The result has the intended covariance, but the
implementation loses the central `O(p)`-buffer scalability property of BISR.

The desired production operator is

\[
  y_t = d_t \sum_{k=0}^{\min(t,p-1)} q_k z_{t-k},
\]

where `q[0:p]` are the signed coefficients of the banded `C^{-1}`, `z_t` is
the iid Gaussian draw already generated for step `t`, and `d_t = 1` for
unnormalized BISR. For normalized BISR, `d_t` is the norm of column `t` of the
unnormalized forward matrix `C`. A ring containing the previous `p - 1` iid
draws gives `O(p*d)` state and work for a gradient with `d` elements. The full
dense forward-substitution implementation should remain available as the
reference oracle requested by the issue and for existing low-level consumers;
the production BISR noise path must stop using it for model-shaped state.

This is the complete conceptual correction requested by #795. A replay
implementation could reduce persistent model-shaped state further, but it would
solve a stronger zero-buffer problem than the issue's accepted ring-or-replay
contract.

The most important facts that change how this should be fixed are:

1. The old `O(p)` implementation was wrong because it truncated the *forward*
   strategy to `p` coefficients. [Issue #360](https://github.com/JetBrains-Research/opake/issues/360)
   and merged [PR #509](https://github.com/JetBrains-Research/opake/pull/509)
   fixed that correctness defect by using the full horizon. Restoring the old
   code would reintroduce a numerical/covariance bug. The production path must
   convolve the banded *inverse*, not truncate the dense forward strategy.
2. The exact bounded-state algorithm and useful scale tests already exist in
   commit [`8c36115`](https://github.com/JetBrains-Research/opake/commit/8c36115afdee3414af098017d00b0c06520cf560),
   originally closed [PR #658](https://github.com/JetBrains-Research/opake/pull/658).
   Closure is not evidence that its BISR design was rejected: open
   [PR #722](https://github.com/JetBrains-Research/opake/pull/722) says it
   consolidated the stacked series, preserved every commit, and closed the
   stack in favor of the integration PR. Its direct-convolution branch and
   bounded-state tests remain, although surrounding engine and RNG code changed.
   The PR now conflicts with `main`, so it is prior art rather than an imminent
   drop-in merge.
3. A narrow fix must preserve main's existing namespaced iid stream. The
   issue's suggestion to replay noise "as lambda-CGD does" refers to the
   technique, not the current key derivation: sibling
   [issue #793](https://github.com/JetBrains-Research/opake/issues/793) says
   lambda-CGD's unrooted `fold_in(key, step)` is itself a privacy-relevant RNG
   collision defect.
4. Changing the state from `n_steps - 1` past correlated outputs to `p - 1`
   past iid draws changes tensor shapes *and their meaning*. Upstream checkpoint
   loading is template-driven, with strict tensor shapes only at matching
   structural paths; a changed layout can instead ignore old descendants. Old
   BISR checkpoints will not automatically resume safely into a ring-buffer
   implementation. The issue's acceptance criteria omit this compatibility
   decision.
5. Open [PR #857](https://github.com/JetBrains-Research/opake/pull/857) changes
   the same BISR files and hardens and formally documents conservative
   accounting for already-permitted signed custom inverse coefficients. Runtime
   convolution must retain those signs. The absolute-value majorants in #857
   belong only to conservative privacy accounting; applying `abs()` to runtime
   coefficients would implement a different mechanism.

The focused mainline solution is therefore the direct inverse convolution in
BISR's existing dedicated raw noise factory, while leaving
`BisrStrategy.streaming_matrix()` as the dense reference. The selected ring
preserves the current one-draw-per-step stream and stores each historical draw
with the base scale that actually produced it. The implementation intentionally
rejects legacy dense-history state through an explicit layout version. Replay,
generic executor consolidation, and legacy migration are possible follow-ups,
not blockers for this solution.

## Research scope and repository snapshot

The public issue, its timeline, related issues and pull requests, the BISR
paper, repository documentation, upstream code, tests, history, release tags,
and unmerged pull-request heads were inspected.

The authoritative upstream code snapshot for this report is `origin/main` at
[`e89e858a31ca5e279f796f2d322d323c8d59665e`](https://github.com/JetBrains-Research/opake/commit/e89e858a31ca5e279f796f2d322d323c8d59665e),
fetched on 2026-08-31. The investigation began from the older WIP branch
`wip/dpftrl-353-595-final` at `4b54a39`; that revision contains related
accounting work but is not treated as the definition of current upstream
behavior.

Relevant live refs were rechecked at the end of the investigation:

| Item | State at snapshot | Revision / detail |
| --- | --- | --- |
| Issue #795 | Open | No comments, assignees, linked branch, or linked PR |
| `main` | Current | `e89e858a31ca5e279f796f2d322d323c8d59665e` |
| PR #658 | Closed, unmerged as a standalone PR | `8c36115afdee3414af098017d00b0c06520cf560` |
| PR #722 | Open, conflicted with `main` | Head `9ca8d3e6368d238bd2dcf7cbe0686265efb2bcee`; contains `8c36115`; API state `dirty` |
| PR #857 | Open, currently mergeable | Head `1c0a28a9e61a1707c39fa232e336622df249aa34`; API state `clean` |

No source implementation was changed as part of this research.

## Issue record

The issue is titled **“Implement BISR noise with O(p) buffers instead of
n_steps gradient-sized buffers.”** It was opened by collaborator `evgri243` on
2026-08-29 at 03:17:13 UTC and remains open with no state reason. It is unlocked,
unassigned, has no public project, and is a `Bug` in the open `Bazalt` milestone,
which has no due date. Its labels are:

- `bug`
- `source: audit`
- `severity: medium`
- `pkg: dpftrl`
- `impact: performance`

Its audit identifier is `OPQ-316`; no separate public artifact for that audit
identifier was found, so the issue is the only public source for its benchmark
method and results. There are no public comments, reactions, child issues,
dependencies, commit references, cross-references, or formally linked
development items. GitHub's Development panel accurately reports no linked
branch or pull request. PR #722 is semantically relevant but predates #795 and
was never formally linked to it.

GitHub records `updated_at` one second after creation. The public timeline has
only the milestone, parent, five label-addition events, and one issue-type event;
there is no evidence of a later body edit. Creation was at 03:17:13, followed by
the milestone at 03:17:14, parent at 03:17:15, all five labels at 03:17:16, and
the `Bug` type at 03:17:17 UTC. Creation, the milestone, and label events record
the Claude GitHub App; the parent and issue-type events record no app provenance.
The issue's cited `_bisr.py:198-211` range is also indirect and has drifted with
later edits: BISR selects the length-`n_steps` operator there, but the
model-shaped allocation actually occurs in `_toeplitz.py`.

The issue is a child of open tracking
[issue #766](https://github.com/JetBrains-Research/opake/issues/766), **“Correct
DP-FTRL mechanism identity, noise streams, and strategy state.”** The parent has
five independently closable children. Unlike #795's medium-severity performance
classification, #766 is labeled `severity: high`, `impact: epsilon`, and
`impact: numerical`, as well as `source: audit` and `pkg: dpftrl`. Its children
are:

- #793: root lambda-CGD under a namespaced RNG stream;
- #794: make learning-rate schedule identity participate in strategy equality;
- #795: bound BISR runtime state;
- #796: remove or implement the warned no-op `acc.cached()` path;
- #797: reject discarded `mf_identity` keyword arguments.

This parent relationship matters. #795 is scoped as a performance/availability
repair, but it sits in a re-audit whose completion criterion says no mechanism
may fold directly into caller key space and every privacy-material strategy
input must participate in identity. A seemingly local replay implementation can
therefore violate the parent even while satisfying #795's memory test.

The issue reports the following behavior for `bisr_strategy(bandwidth=4)` and a
200,000-parameter gradient:

| `n_steps` | Reported state | Reported time per noise step |
| ---: | ---: | ---: |
| 256 | 204 MB | 0.055 s |
| 1,024 | 818 MB | 0.411 s |

The allocation numbers match the implementation exactly. The timing result is
directionally consistent with the code, but the issue does not give hardware,
device, dtype, warm-up procedure, benchmark source, or exact commit. Also,
`0.411 / 0.055` is about 7.47 while the horizon grows by 4. The phrase “exactly
linear” is exact for persistent state and asymptotic per-step work, not literally
for those two wall-clock samples.

The issue asks for four outcomes:

1. Apply banded `C^{-1}` directly, using either a `p - 1` history of iid draws
   or deterministic replay, and apply `d_t` in normalized mode.
2. Use `O(p)` gradient buffers and `O(p*d)` work per step.
3. Retain dense forward substitution as an equivalence reference.
4. Test that runtime state size does not grow with `n_steps`.

Those criteria are correct but should be strengthened for checkpoint behavior,
RNG identity, signed coefficients, realized row norms, and the `p > n_steps`
edge case.

## What BISR is supposed to execute

Let `C` be the lower-triangular Toeplitz strategy matrix and let

\[
  C^{-1} = \operatorname{LTT}(q_0,q_1,\ldots,q_{p-1},0,\ldots,0).
\]

Although `C` is generally dense out to the finite horizon, its inverse has only
`p` nonzero diagonals. Given independent Gaussian gradient-shaped columns
`z_0, ..., z_{n-1}`, the unnormalized noising operator is

\[
  x_t = (C^{-1}z)_t
      = \sum_{k=0}^{\min(t,p-1)} q_k z_{t-k}.
\]

The values retained between calls must be the previous iid inputs `z`, not the
previous correlated outputs `x`. At step `t`, the current `z_t` is already in
hand, so persistent state needs at most `p - 1` prior gradient pytrees. For a
finite horizon shorter than the bandwidth, the effective tap count is
`p_eff = min(p, n_steps)` and only `p_eff - 1` prior values are useful.

### Normalized BISR

Opake column-normalizes the forward strategy. If
`D = diag(d_0, ..., d_{n-1})`, where `d_t = ||C[:,t]||_2`, then the normalized
strategy is `C_bar = C D^{-1}` and

\[
  \bar C^{-1} = D C^{-1}.
\]

The diagonal therefore scales the *output row* after inverse convolution:

\[
  y_t = d_t x_t.
\]

For a lower-triangular Toeplitz `C` whose first-column coefficients are
`c_0, ..., c_{n-1}`, the finite-horizon scale is

\[
  d_t = \sqrt{\sum_{j=0}^{n-1-t} c_j^2}.
\]

This matches upstream's reversed cumulative-sum implementation in
[`_toeplitz.py`](https://github.com/JetBrains-Research/opake/blob/e89e858a31ca5e279f796f2d322d323c8d59665e/packages/opaque-dpftrl/src/opaque/api/dpftrl/noise/_toeplitz.py#L271-L275).
The realized row norm reported to callers must be

\[
  ||\operatorname{row}_t(\bar C^{-1})||_2
    = d_t\sqrt{\sum_{k=0}^{\min(t,p-1)}q_k^2}.
\]

That value is not diagnostic-only. `mf_gaussian_noise` publishes
`base_stddev * row_l2_at(t)` as `NoisedPytree.noise_stddev`, and downstream
adaptive-optimizer behavior can use the realized value. A fix that emits the
right samples but reports the old or unscaled row norm is incomplete.

### Paper support

The primary BISR paper,
[“Back to Square Roots: An Optimal Bound on the Matrix Factorization Error for
Multi-Epoch Differentially Private SGD”](https://arxiv.org/html/2505.12128),
independently confirms this interpretation:

- Algorithm 1 injects the finite sum of the current and previous `p - 1` iid
  Gaussian vectors.
- Figure 1 describes taking older noise vectors from a buffer, evicting the
  oldest one, and inserting the current draw.
- Section 3 says streaming multiplication needs only `p` rows of the iid-noise
  matrix at a time, including the current row.
- Lemma 2 proves exact streaming space complexity `p` for `n >= 2p - 1` when
  coefficient storage is excluded; the full unbanded inverse has linear space
  complexity.

The paper assumes `0 <= beta < alpha <= 1`. Opake's BISR implementation uses
the `alpha = 1` case and exposes `momentum` as `beta`; its built-in parameter
domain is therefore consistent with the formula it implements. Supporting the
paper's `alpha < 1` workload is outside #795.

## Current upstream control flow and root cause

The dedicated BISR runtime hook is currently only a wrapper around the generic
executor:

1. [`_make_bisr_noise`](https://github.com/JetBrains-Research/opake/blob/e89e858a31ca5e279f796f2d322d323c8d59665e/packages/opaque-dpftrl/src/opaque/api/dpftrl/noise/_bisr.py#L116-L145)
   converts `n_steps` with `int(...)`, calls
   `strategy.streaming_matrix(n_steps=n_steps)`, and passes the result to the
   generic matrix-factorization noise engine.
2. [`BisrStrategy.streaming_matrix`](https://github.com/JetBrains-Research/opake/blob/e89e858a31ca5e279f796f2d322d323c8d59665e/packages/opaque-dpftrl/src/opaque/api/dpftrl/noise/_bisr.py#L207-L220)
   recovers a forward-strategy first column of length `n_steps` and calls
   `inverse_as_streaming_matrix(...)`.
3. The same call supplies the known `p` inverse coefficients, but the code
   comment and implementation make clear that this is only a validation and
   closed-form row-norm hint.
4. [`inverse_as_streaming_matrix`](https://github.com/JetBrains-Research/opake/blob/e89e858a31ca5e279f796f2d322d323c8d59665e/packages/opaque-dpftrl/src/opaque/api/dpftrl/noise/_toeplitz.py#L220-L267)
   sets `bands = len(coef) = n_steps`. Its initializer allocates
   `(bands - 1, *gradient_leaf.shape)` for every tensor leaf.
5. Each call computes the dense recurrence

   \[
     x_t = \frac{z_t - \sum_{j=1}^{n-1} c_j x_{t-j}}{c_0},
   \]

   using `torch.tensordot`, then calls `torch.roll` on the entire history and
   stores the newest correlated output.
6. [`_streaming_mf_noise`](https://github.com/JetBrains-Research/opake/blob/e89e858a31ca5e279f796f2d322d323c8d59665e/packages/opaque-dpftrl/src/opaque/api/dpftrl/noise/_engine.py#L378-L431)
   initializes that state while the noise factory is constructed, before the
   first training step.

For `d` float32 gradient elements, unnormalized persistent tensor storage is
exactly `4 * (n_steps - 1) * d` bytes. Normalized execution composes the
inverse with a row-diagonal stream and adds one int64 step counter per tensor
leaf. At the issue's `d = 200,000`, this gives:

- `(256 - 1) * 200,000 * 4 = 204,000,000` bytes;
- `(1,024 - 1) * 200,000 * 4 = 818,400,000` bytes.

A local allocation check with a 10,000-element float32 leaf and normalized
`bandwidth=4` observed:

| `n_steps` | Gradient-history shape | Total tensor bytes in streaming state |
| ---: | ---: | ---: |
| 16 | `(15, 10000)` | 600,008 |
| 64 | `(63, 10000)` | 2,520,008 |
| 256 | `(255, 10000)` | 10,200,008 |

The extra eight bytes in this one-leaf reproduction are the row-diagonal
stream's scalar counter; a general normalized pytree adds eight bytes per tensor
leaf. The state not only consumes `O(n_steps*d)` persistent memory; the
contraction and history roll perform `O(n_steps*d)` arithmetic/copy work and can
create a similarly large transient allocation each step. With a realistic
model, factory construction can fail before training begins.

This allocation check ran against the pinned `main` revision in a detached
worktree on CPU with PyTorch `2.9.1+cu128`. Its template was one
`torch.float32` tensor of shape `(10000,)`; bytes are the sum of
`numel() * element_size()` across tensor leaves immediately after
`StreamingMatrix.init_multiply`.

The inverse hint introduced by PR #572 does not change this. Its use is confined
to [`row_norms_squared`](https://github.com/JetBrains-Research/opake/blob/e89e858a31ca5e279f796f2d322d323c8d59665e/packages/opaque-dpftrl/src/opaque/api/dpftrl/noise/_toeplitz.py#L277-L309).
It fixed scalar setup work, not gradient-sized execution state.

### Correctness of the current output

This is primarily a performance, availability, and checkpoint-size defect—not
evidence by itself that current BISR runtime under-noises. Solving the full dense
`C x = z` recurrence realizes the selected `C^{-1}z` operator as direct inverse
convolution does, subject to ordinary floating-point evaluation-order
differences. Built-in BISR and nonnegative theorem-domain configurations retain
matching runtime/accounting semantics; already-accepted signed custom encoders
have the separate conservative-accounting defect addressed by #857.

An independent deterministic comparison fed identical float32 iid columns to
the current dense solver and the direct formula:

| Horizon | Bandwidth | Momentum | Normalized | Maximum absolute difference |
| ---: | ---: | ---: | :---: | ---: |
| 12 | 4 | 0.0 | No | `4.77e-7` |
| 12 | 4 | 0.0 | Yes | `7.15e-7` |
| 256 | 4 | 0.9 | Yes | `1.19e-5` |
| 3 | 5 | 0.3 | Yes | `4.77e-7` |

This reproducible spot check also used the pinned `main` worktree, CPU, and
PyTorch `2.9.1+cu128`. For each row it regenerated a `(n_steps, 10000)`
float32 iid tensor using a CPU `torch.Generator` seeded with `795`, applied
`multiply_array(strategy.streaming_matrix(...), z)`, and compared that with an
explicit signed tap sum using the same `z`, inverse coefficients, and
finite-horizon column norms. It is a deterministic equivalence check, not a
performance benchmark.

The last case also confirms the finite-horizon `p > n_steps` truncation. A
direct implementation can preserve the same iid draws and distribution, but
should not promise bitwise-identical outputs because it changes summation order.
A bad direct implementation—wrong normalization side, history type, tap order,
sign, or RNG namespace—could become privacy-material even though the present
defect is not.

## How the defect arose

### BISR introduction: PR #121

[PR #121](https://github.com/JetBrains-Research/opake/pull/121), merged on
2026-04-16, introduced lambda-CGD and BISR. Its design correctly identified
`C^{-1}` as the banded Toeplitz object, and the documentation already advertised
bounded memory. The implementation nevertheless recovered only `bandwidth`
entries of the *forward* strategy before delegating to the generic streaming
solver. It therefore happened to retain only `p - 1` outputs, but did not
represent the full finite-horizon operator.

### Full-horizon correctness: issue #360 and PR #509

[Issue #360](https://github.com/JetBrains-Research/opake/issues/360) identified
that truncation on 2026-08-02. It was a high-severity numerical issue: plausible
per-step variances did not establish that runtime covariance matched the
selected strategy. Its criteria asked for the full operator, empirical
covariance over multiple horizons/bandwidths, and dense/streaming agreement.

[PR #509](https://github.com/JetBrains-Research/opake/pull/509), merged as
[`2507548`](https://github.com/JetBrains-Research/opake/commit/2507548b20222769958fff574168b0bd68232133)
on 2026-08-06, changed BISR to recover all `n_steps` forward coefficients. The
title says it applies the runtime operator “directly,” but the code still routes
through generic forward substitution. The correctness repair thus created the
present scaling failure.

An automated review on that PR explicitly warned that the tests only checked
dispatch and did not pin the full-horizon behavioral regression. It requested
empirical covariance or dense-versus-streaming agreement across horizons and
bandwidths. The author added the current horizon-sensitive matrix test. That
test proves the reference matrix is full-horizon, but it still does not compare
actual generated runtime sequences from the dedicated raw hook. #795 should
finally add that missing end-to-end equivalence coverage. The review thread is
[here](https://github.com/JetBrains-Research/opake/pull/509#discussion_r3726858053).

The #509 behavior shipped by release `v0.13.1`, so the scaling problem affects
released versions rather than only an unreleased branch.

### Row-norm setup optimization: issue #355 and PR #572

[Issue #355](https://github.com/JetBrains-Research/opake/issues/355) concerned
eager quadratic row-norm probing. Merged
[PR #572](https://github.com/JetBrains-Research/opake/pull/572) replaced it with
a closed-form cumulative sum and let BISR supply its known banded inverse as a
hint. GitHub records merge commit
[`8440327d`](https://github.com/JetBrains-Research/opake/commit/8440327d582ec883a17188215bf6341d2e1e0eeb)
on the PR's stacked base; the equivalent commit present on current `main` is
[`4106222`](https://github.com/JetBrains-Research/opake/commit/4106222d0de5c3df5d57d7fc4276e2370a1f7f27).
It explicitly documents the crucial duality: the forward coefficients are dense
to `n_steps`, while the inverse is banded.

That PR made noise-factory setup practical but left multiplication unchanged.
It also found that `bandwidth > n_steps` is valid and that only the first
`n_steps` inverse taps/product terms belong to the finite matrix. The current
code truncates the hint accordingly. Any #795 implementation that demands
`p <= n_steps` or validates all `p` terms against an `n_steps` matrix would
regress this established behavior.

The row-norm improvement shipped in `v0.15.0`.

## Existing direct implementation in PR #658 / PR #722

Commit [`8c36115`](https://github.com/JetBrains-Research/opake/commit/8c36115afdee3414af098017d00b0c06520cf560)
was authored on 2026-08-19, committed on 2026-08-21 at 09:49 UTC, and present
at #658's final force-push at 09:56 UTC; it already implements the requested
operator. It appeared as part 7 of the backend-split stack in
[PR #658](https://github.com/JetBrains-Research/opake/pull/658). The relevant
parts are:

- [`_plan.py`](https://github.com/JetBrains-Research/opake/blob/8c36115afdee3414af098017d00b0c06520cf560/packages/opaque-dpftrl/src/opaque/api/dpftrl/noise/_plan.py#L110-L174)
  validates the known inverse against the dense forward strategy over the
  finite horizon, truncates a longer hint when `p > n_steps`, computes `d_t`,
  and computes closed-form row norms.
- [`_engine.py`](https://github.com/JetBrains-Research/opake/blob/8c36115afdee3414af098017d00b0c06520cf560/packages/opaque-dpftrl/src/opaque/api/dpftrl/noise/_engine.py#L211-L243)
  chooses direct convolution when the inverse has fewer nonzero bands than the
  forward strategy. It retains iid `z` values newest-first and applies output
  scaling after the convolution.
- [`test_streaming_execution.py`](https://github.com/JetBrains-Research/opake/blob/8c36115afdee3414af098017d00b0c06520cf560/packages/opaque-torch/tests/dpftrl/test_streaming_execution.py)
  asserts BISR state length, actual state bytes, deterministic dense-reference
  equivalence, and a tolerant early-versus-late cost bound.
- `BisrStrategy.streaming_matrix()` remains the dense reference while its
  execution plan carries the efficient production representation.

One review initially found the same `p > n_steps` validation error previously
encountered in #572. The final commit contains the corrected horizon-truncated
validation, making it especially valuable prior art for #795.

PR #658 is closed, but closure is not evidence that its BISR design was
rejected. Open [PR #722](https://github.com/JetBrains-Research/opake/pull/722),
**“split a backend-neutral engine from the Torch provider,”** explicitly says it
replaces the stacked series, preserves every commit, and closes the stack in
favor of the consolidated integration branch. Its body calls this a “22-PR”
series, although the three listed ranges enumerate 18 PR numbers. Commit
`8c36115` is an ancestor of #722's current head
[`9ca8d3e`](https://github.com/JetBrains-Research/opake/commit/9ca8d3e6368d238bd2dcf7cbe0686265efb2bcee).

A comparison of `8c36115` with that head found:

- `_plan.py`, the BISR runtime tests, and the bounded-state tests are unchanged;
- `_bisr.py` only gained a named minimum-bandwidth constant;
- the direct convolution itself is unchanged;
- later engine changes add the consolidated RNG namespace and backend
  activation behavior.

Thus, a matching implementation remains on open, unlinked #722. As of this
snapshot, it has 41 commits and no submitted human review, but GitHub reports
`mergeable: false`, `mergeable_state: dirty`, and `rebaseable: false` against
current `main`. A bot's
[“Ready to merge” comment](https://github.com/JetBrains-Research/opake/pull/722#issuecomment-3254981668)
was posted on 2026-08-22 against an earlier tip and is now stale; it was never
maintainer approval. The implementation should be coordinated with—not silently
duplicated—but #722 cannot be assumed to land soon without conflict resolution.

### Why not cherry-pick #658 wholesale

PR #658 changes 67 files as part of a provider-neutral engine redesign. It
moves tests between packages, introduces execution plans, changes backend
selection, and deliberately changes per-leaf RNG and checkpoint layouts. Its
algorithm and regression tests are directly reusable; its architectural and
randomness changes are not a suitable focused fix for current `main`.

There are two reasonable integration strategies:

1. If maintainers intend to resolve and rebase #722 and land it first, retain
   its existing BISR tests, link it to #795, reconcile it with #857, and close
   #795 through that work.
2. If #795 should land independently, port the small direct-convolution seam and
   tests into current `main`, then explicitly resolve the overlap when #722 is
   rebased. Do not create a third algorithm with different state/RNG semantics.

## Recommended focused design on current `main`

Opake already has the right extension seam. The dispatcher gives a strategy's
`raw_noise_factory(...)` priority over the generic strategy path, and BISR
already implements that hook. A narrow solution can remain in the BISR module:

1. Validate `n_steps` with the shared strict positive-horizon helper instead of
   calling `int(n_steps)`.
2. Let `q = strategy._inv_coefs()[:n_steps]` and retain every signed value,
   including a non-unit `q[0]` from a valid custom strategy.
3. Build a direct inverse `StreamingMatrix` whose state contains the previous
   `len(q) - 1` iid inputs, not correlated outputs.
4. In `multiply_next`, compute `q[0] * z_t` plus the weighted prior iid values,
   update the bounded history, then apply `d_t` in normalized mode.
5. Pass that direct streaming operator to the existing generic
   `_matrix_factorization_noise` / `_streaming_mf_noise` engine. This preserves
   the surrounding horizon, pytree, one-draw-per-step, output-casting, and key
   integration. The new operator must nevertheless allocate its history and
   accumulate explicitly in the captured `compute_dtype`: the existing
   `StreamingMatrix` initializer starts from the gradient-template dtype, so
   generic-engine reuse alone does not preserve a float64 override for a
   float32 template.
6. Compute `row_l2_at(t)` from the closed form above. Dense forward
   coefficients may still be used to obtain the normalization diagonal, but no
   gradient-shaped object should be sized from their length.
7. Leave `BisrStrategy.streaming_matrix(n_steps)` unchanged as the dense
   forward-substitution reference and for existing low-level consumers/tests.
   Production `raw_noise_factory` should no longer use it for gradient state.

The direct operator may store history as a stacked tensor per leaf or a small
pytree of prior tensors. Under functional/immutable state semantics, updating a
stacked cyclic ring requires cloning the stack; an in-place slot write would
alias and mutate the prior state. A newest-first tuple of tensor references, as
#658 uses, avoids `torch.roll` and full-stack copying. A cloned stack still meets
the required `O(p*d)` asymptotic bound but has a larger practical copy cost.

No Rust accountant, sensitivity, Gram, or privacy calibration change is needed
for #795 itself. The strategy is unchanged; only an equivalent runtime
realization replaces the dense solve. Accounting changes from #857 should merge
orthogonally.

### Complexity claim should be precise

For fixed `p` and gradient size `d`, the desired *gradient-sized state* is

\[
  O((\min(p,n)-1)d),
\]

and each call is `O(min(p,n)*d)`. Normalized execution may still retain
`O(n_steps)` scalar forward coefficients, column scales, or row norms. That is
negligible next to model-shaped state and is explicitly excluded by the paper's
streaming-space lemma, but it means “all memory is independent of `n_steps`” is
too strong if interpreted literally. The regression should measure tensor
`numel * element_size` for gradient-shaped state and separately account for
scalar metadata.

For a 100-million-parameter float32 model, `p=4` means about 1.2 GB for three
past full-model draws, independent of horizon. That is vastly smaller than the
hundreds of GB implied by a thousand-step history, but it is not zero and may
still be operationally significant. PRNG replay is the zero-buffer alternative
if that trade-off is unacceptable.

## Ring buffer versus PRNG replay

Both approaches can satisfy the issue's asymptotic criterion, but they are not
interchangeable operationally. Because the issue explicitly accepts either
one, the selected ring is a complete #795 solution; replay would be a separate
zero-buffer optimization.

| Property | Ring of prior iid draws | Replay prior iid draws |
| --- | --- | --- |
| Gradient-sized persistent state | `p_eff - 1` pytrees | None |
| Gaussian generation per call | One fresh pytree | Up to `p_eff` pytrees |
| Linear combination work | `O(p*d)` | `O(p*d)` plus RNG/sampling overhead |
| Accelerator behavior | One current sample/transfer | Repeated sampling/transfers can dominate |
| Existing stream compatibility | Natural if generic engine is reused | Requires exact reconstruction of every historical key |
| Varying raw `stddev` semantics | Stores the actual historical scaled draws | Replaying old draws with the current scale is wrong |
| Checkpoint state size | `O(p*d)` | Constant |
| Legacy-state policy | Explicit rejection or migration is required | Replay still requires an explicit compatibility decision |

The public `mf_gaussian_noise` wrapper latches the first clipping norm and
rejects later changes, so its base standard deviation is constant. However, the
internal raw factory accepts a `stddev` argument on each call, and second-moment
composition delegates through it. A ring stores the actual historical draw and
therefore remains correct even if an internal caller changes scale. A naive
replay implementation would regenerate every old draw using today's scale and
would not be the same operator. Replay would need to regenerate standard
normals and recover the historical scales or formally narrow the raw contract.

On the current Torch path, repeated replay also means repeated random tensor
generation and potentially repeated CPU-to-device transfer. Given the current
engine and #795's acceptance criteria, a ring is the lower-risk focused choice.
A hybrid design could keep a ring during normal execution and reconstruct it
from the immutable key and step only during restore, but that adds serialization
complexity and should be justified by a separate compatibility requirement.

The same direct convolution could eventually live in a generic banded-inverse
executor shared by more strategies. The provider-neutral work in #658/#722 is
useful prior art for that direction, but importing or recreating that broader
abstraction is not necessary to fix BISR's horizon-dependent state on `main`.

## RNG and privacy invariants

Current upstream generic BISR execution derives its step key as shown in
[`_engine.py`](https://github.com/JetBrains-Research/opake/blob/e89e858a31ca5e279f796f2d322d323c8d59665e/packages/opaque-dpftrl/src/opaque/api/dpftrl/noise/_engine.py#L402-L418):

```text
fold_in(base_key,
        "opake.dpftrl.mf_gaussian",
        "mf_gaussian_column",
        step)
```

A focused ring-buffer implementation can preserve this byte-for-byte by reusing
the generic streaming engine. The iid draw created on step `t` is stored and
reused; no new key convention is necessary. This also preserves one current
draw per call and the existing relationship between tensor leaves.

Do not copy current lambda-CGD's replay code. At upstream main it derives
`fold_in(base_key, step)` and `fold_in(base_key, step - 1)` without a mechanism
root. Sibling #793 explains that these keys can collide byte-for-byte with keys
a caller legitimately derives in its own namespace. If replay is chosen for
#795, it must replay the *existing namespaced BISR/MF keys*, not lambda-CGD's
current keys. Ideally it should land after or alongside #793's namespacing
tests.

The paired second-moment implementation already derives independent first and
second base streams before invoking each strategy's raw factory. The BISR fix
must preserve those outer roots and keep independent ring state for each stream.

PR #658/#722 uses a different provider-neutral per-leaf derivation and declares
the broader RNG/state change breaking. That is appropriate inside its major
backend refactor, but it should not leak into a focused #795 patch. Preserving
the current base draw also makes deterministic comparisons and release behavior
much easier to explain. With the same internal `compute_dtype` preserved, the
only expected numeric difference is floating-point reduction order.

## Checkpoint and resume compatibility

The issue omits checkpoint-layout behavior, so the implementation must make it
explicit rather than relying on incidental template matching.

Opake's
[`Trainer` checkpoint path](https://github.com/JetBrains-Research/opake/blob/e89e858a31ca5e279f796f2d322d323c8d59665e/packages/opaque-transformers/src/opaque/api/transformers/trainer/_checkpoint.py#L198-L368)
persists the DP-FTRL state, including matrix-factorization noise history needed
to continue the same correlated stream. Restore is template-driven: the
[`base` structural walker](https://github.com/JetBrains-Research/opake/blob/e89e858a31ca5e279f796f2d322d323c8d59665e/packages/opaque-base/src/opaque/api/base/serialization/_structural.py#L82-L114)
visits the freshly constructed template, and the
[`engine` tensor loader](https://github.com/JetBrains-Research/opake/blob/e89e858a31ca5e279f796f2d322d323c8d59665e/packages/opaque-engine/src/opaque/api/engine/serialization/_structural.py#L44-L73)
raises `CheckpointError` when a saved tensor at the same structural path has a
different shape. It does **not** independently verify every path in the saved
state.

Current normalized BISR state contains a scalar row index and a tensor of shape
`(n_steps - 1, *leaf.shape)` for each leaf. A new ring template contains a
tensor or tuple representing `(p_eff - 1, *leaf.shape)`. Reusing the same stacked
tensor path will reject an old checkpoint because its shape differs. Changing
to a tuple, ring object, or primitive `None` may instead cause old descendants
that are absent from the new template to be ignored and fresh history to remain
initialized. That silent partial restore is worse than a clear rejection.
Even if shapes happen to match, accepting the old tensor would be wrong:

- old history stores correlated outputs `x` from dense forward substitution;
- new history stores iid inputs `z` for direct inverse convolution.

The implemented policy is intentional, versioned incompatibility: the bounded
BISR state carries an explicit layout marker, and a legacy dense-history state
fails with a targeted checkpoint error. This is sufficient for #795 and avoids
silently interpreting correlated outputs as iid draws. The alternatives
considered during research were:

1. **Versioned incompatibility (selected).** Use a mechanism-local layout marker
   or a broader checkpoint version, fail with a targeted BISR state-layout
   message, and document that pre-fix runs must resume under their original
   Opake version. This is the simplest and safest policy, but it is a
   user-visible incompatibility.
2. **Explicit migration.** Reconstruct the last `p_eff - 1` iid draws from the
   saved base key, step counter, restored first clipping norm, configured noise
   multiplier, dtype, and tree shape. This is likely simpler and less
   numerically fragile than algebraically recovering `z` from the saved dense
   `x` history, but it needs mechanism-specific loading or a lazy first-call
   migration. It can preserve the iid columns and mathematical operator, but
   dense recurrence versus direct convolution may differ by floating-point
   roundoff, so old-to-new continuation should use a justified tolerance rather
   than promise bit identity.
3. **Replay-only state.** Use `_inner_state=None` and regenerate history every
   call. The generic template walker may ignore descendant keys from an old
   nested `_inner_state` when the new template is primitive `None`, while
   restoring the step and base key. That can be a valid migration only if replay
   reconstructs the exact prior iid columns; otherwise it is a silent history
   reset. It needs an explicit version/layout discriminator, a real old fixture,
   and an end-to-end resumed-suffix test.

Whichever policy is chosen, do not rely on incidental template shape to detect
the transition. Add an explicit BISR state-layout/version marker so structural
changes cannot silently initialize a new history.

For a new bounded-layout checkpoint, restoring the iid window is necessary but
not sufficient for exact continuation. The rebuilt mechanism must have the same
BISR execution identity—including its strategy/horizon, RNG derivation, and
numeric execution choices—and the same base noise scale. Otherwise the retained
old-scale draws and newly generated draws describe a different process.
[Issue #789](https://github.com/JetBrains-Research/opake/issues/789) tracks the
high-severity case where calibrated DP-FTRL resume can silently choose a new
noise multiplier and mix scales across the boundary. #795 neither fixes that
calibration/resume defect nor makes such a resume safe.

PR #658 explicitly treats its changed state/RNG layout as breaking, and #722
uses a newer checkpoint bundle version in its larger refactor. It therefore
provides algorithmic precedent but does not solve focused migration from
current main automatically.

Checkpoint size and I/O are additional impacts of the current bug. At the issue's
example sizes, serializing `noise_state` writes hundreds of megabytes solely for
BISR history. A regression should inspect serialized state size as well as live
memory if the chosen policy retains a ring.

## Other facts that constrain the solution

### Signed custom inverse coefficients and PR #857

Open [PR #857](https://github.com/JetBrains-Research/opake/pull/857), head
[`1c0a28a`](https://github.com/JetBrains-Research/opake/commit/1c0a28a9e61a1707c39fa232e336622df249aa34),
hardens privacy accounting for signed encoders. It also changes `_bisr.py`, the
native BISR path, strategy serialization, tests, and docs. In particular, it:

- uses conservative absolute majorants for privacy bounds where signed terms
  could otherwise cancel;
- validates finite custom inverse coefficients and a non-negligible leading
  coefficient;
- rejects legacy learning-rate-weighted BISR accounting that does not match the
  deployed encoder;
- documents signed custom `inv_coefficients` as a supported conservative
  extension.

#795 must rebase on or deliberately reconcile #857. Direct runtime must compute
the signed sum `sum(q[k] * z[t-k])` exactly. It must not copy accounting's
absolute coefficients into the noise filter. It also must not hard-code
`q[0] = 1`: built-in BISR has that value, but a validated custom inverse may
not. The safest implementation obtains coefficients through the strategy's
post-#857 validated path.

### Horizon validation is currently weaker in BISR

`_make_bisr_noise` currently performs `n_steps = int(n_steps)`. That silently
turns `4.2` into `4` and accepts booleans, unlike the generic MF engine and
lambda-CGD, which use the shared positive-integer horizon validator. Since #795
must rewrite this factory, it should remove the cast and use the shared strict
helper. BISR is also absent from the main parametrized streaming-horizon test.

This is a small adjacent correctness hardening at the exact touched boundary,
not a reason to expand #795 into an accounting redesign.

### Full-horizon scalar metadata remains valid

Normalized BISR still needs the finite-horizon column norm `d_t`; accounting and
reference construction still need dense scalar coefficients. Retaining an
`O(n_steps)` float64 vector is compatible with the paper and the practical goal.
The defect is multiplying horizon length by model dimension. Avoid weakening
normalization merely to make every captured scalar independent of `n_steps`.

### Pytree, per-group, dtype, and second-moment behavior

The generic streaming engine currently handles nested tensor pytrees, promotes
half/bfloat16 internal computation to at least float32, casts the final value
back to input dtype, and accepts scalar or `PerGroup` noise scales. Reusing it
reduces the risk of a BISR-only divergence, but the direct operator's history
initializer and accumulation must still use the engine-selected
`compute_dtype`, including a float64 override on a float32 template.

A direct state must be per tensor leaf and retain the historical, already-scaled
iid draw for that leaf. In paired second-moment mode, BISR can be selected for
either or both streams; the present defect then allocates the excessive history
twice. Each stream needs its own bounded history and checkpoint coverage.

Distributed MF synchronization validates key, step, and latched clipping norm;
[`_inner_state` is marked local](https://github.com/JetBrains-Research/opake/blob/e89e858a31ca5e279f796f2d322d323c8d59665e/packages/opaque-dpftrl/src/opaque/api/dpftrl/noise/_distributed.py#L86-L120),
so synchronization does not compare ring history tensors. A direct ring built
from the same namespaced draw on every rank should preserve deterministic state
evolution, but a schema-acceptance test cannot prove it. A distributed
regression must compare deterministic outputs/state evolution across ranks or
otherwise establish equality.

## Current test coverage and its gaps

Upstream's
[`test_bisr_noise.py`](https://github.com/JetBrains-Research/opake/blob/e89e858a31ca5e279f796f2d322d323c8d59665e/packages/opaque-dpftrl/tests/noise/test_bisr_noise.py#L85-L171)
currently verifies:

- the dedicated raw factory exists and is dispatched;
- the forward reference is constructed at full horizon;
- materialized reference matrices agree;
- closed-form row norms agree with generic probing.

It does **not** assert runtime inner-state bytes, direct runtime outputs over a
sequence, actual covariance, constant late-step cost, checkpoint migration, or
the semantic difference between iid and correlated history. The runtime test
can pass when the raw factory merely delegates back to the generic dense path,
which is exactly the current situation.

Existing broader tests exercise BISR indirectly in realized-noise standard
deviation, `PerGroup`, second-moment, and trainer checkpoint/resume paths. No
BISR-specific distributed test was found on the pinned revision. The existing
consumers are useful regressions, but most make one call or inspect only step
zero; they do not prove bounded-state sequence behavior.

## Recommended verification matrix

The following tests should gate closure of #795.

### 1. Deterministic operator equivalence

Feed a controlled sequence of iid pytrees to both implementations and compare
every output row:

- direct signed inverse convolution;
- retained dense full-horizon forward substitution.

Cover normalized and unnormalized BISR, `p=2` and a larger bandwidth, default
and high momentum, explicit signed custom coefficients, `n_steps < p`, one- and
multi-leaf pytrees, float32 and float64. Use a justified tolerance because
summation order differs. This deterministic same-input test is stronger and
less noisy than covariance alone.

### 2. End-to-end generated-noise equivalence/covariance

Drive the public noise factory across multiple steps and confirm the observed
sequence agrees with the intended dense covariance for several horizons and
bandwidths. This closes the still-partial test request from #360/#509 and catches
RNG/history wiring that a pure matrix helper test cannot.

### 3. State-size invariant

For fixed `p`, gradient template, dtype, and tree shape, construct BISR with two
widely different horizons that are both at least `p`. Assert both at
initialization and after more than `p` calls that:

- the number of model-shaped retained elements/bytes is identical;
- it equals at most `(min(p,n_steps)-1) * template_bytes` per stream;
- no state leaf retains an `n_steps - 1` dimension;
- normalized scalar index metadata is accounted separately.

Separately test `n_steps < p` and assert the general upper bound of
`(min(p,n_steps)-1) * template_bytes` per stream; a correct finite-horizon state
grows from `n_steps - 1` histories up to `p - 1`, so byte equality is not
expected across that boundary.

Checking only tuple length is insufficient. Include serialized checkpoint bytes
if ring state is persisted.

### 4. Flat cost/complexity gate

Use the tolerant #658/#722 pattern to compare median early-step and late-step
cost at a horizon large enough to expose `O(step)` behavior. Prefer the
state-size test as the non-flaky primary gate; keep wall-clock performance as a
coarse regression or benchmark.

### 5. RNG-domain and draw-count regression

Pin the existing BISR namespaced key derivation and confirm one fresh iid pytree
is generated per public call when using the ring. Ensure it cannot collide with
caller folds or lambda-CGD after #793. If replay is selected, test every replayed
key against the existing generic BISR stream rather than current lambda-CGD.

### 6. Realized row norm at every phase

Compare `row_l2_at(t)` with the dense reference at step zero, during ring fill,
after the ring is full, and at the last horizon step. Cover both normalization
modes and `p > n_steps`; verify the public `noise_stddev` field uses it.

### 7. Checkpoint policy and continuation

For new checkpoints, compare an uninterrupted suffix with a save/load/resumed
suffix bit-for-bit under the same implementation, device, and dtype, including
nested trees and normalized BISR. Add either:

- a real pre-fix dense-state fixture that migrates the same iid columns/operator
  and agrees with the uninterrupted dense suffix within a justified tolerance;
  or
- a real pre-fix fixture that fails with the intentional targeted version/layout
  error promised in release notes.

Do not test only state-dict round-trip shape. Lost or misinterpreted history
changes future covariance.

### 8. Public consumers

Retain or expand coverage for:

- `PerGroup` scaling and multiple leaves;
- BISR in one or both second-moment streams for several steps;
- float16/bfloat16 input with float32 internal computation and float64 override;
- exact output dtype;
- deterministic distributed output and local-state evolution across ranks, not
  only step/key schema synchronization;
- calls exactly at and one step beyond the calibrated horizon;
- rejection of float, boolean, zero, and negative `n_steps`;
- finite/signed custom coefficient validation from #857.

### 9. Production/reference separation

Add a test proving the public BISR raw path no longer initializes the dense
`n_steps - 1` streaming state. The dense `streaming_matrix()` must remain
callable as a test/reference oracle, but production should not invoke it for
model-shaped execution state.

## Strengthened acceptance criteria

The issue can safely close when all of the following are true:

1. Production BISR emits the signed direct inverse convolution with the correct
   finite-horizon `d_t` row scale.
2. It retains at most `min(p,n_steps) - 1` gradient-shaped iid histories and
   performs `O(min(p,n_steps)*d)` work per call.
3. For fixed `p` and horizons `n_steps >= p`, gradient-shaped state bytes are
   independent of `n_steps`; for every horizon they stay within
   `(min(p,n_steps)-1) * template_bytes` per stream. Any `O(n_steps)` scalar
   metadata is documented and measured separately.
4. The full-horizon dense forward-substitution implementation remains a
   reference and agrees with direct execution in all supported modes.
5. The existing mainline MF RNG namespace and one-fresh-draw-per-step behavior
   are preserved, or any deliberate RNG change is versioned and documented.
6. Normalized row norms and public realized standard deviations agree with the
   dense reference for every step.
7. `p > n_steps`, signed custom coefficients, nested pytrees, per-group noise,
   compute dtypes, second-moment streams, and horizon failures are covered.
8. Legacy dense-history checkpoints are intentionally rejected through an
   explicit layout version; new-layout checkpoints resume to the exact same
   future sequence only under the same BISR execution identity and base noise
   scale.
9. The patch preserves #857's signed-accounting hardening and records the
   overlap with #722. A future generic executor may replace the focused helper,
   but that consolidation does not block #795.

The calibrated-resume failure in #789 remains urgent, but it is a separate
acceptance surface: closing #795 must not be represented as fixing #789.

## Non-goals

The focused fix should not:

- restore the old `p`-length forward recurrence;
- change BISR sensitivity, Gram construction, or privacy calibration except to
  merge compatible #857 hardening;
- apply absolute values to runtime inverse coefficients;
- copy lambda-CGD's current unrooted replay keys;
- remove or weaken finite-horizon normalization;
- require the entire provider-neutral backend split to land;
- claim zero-buffer or memory-free execution—the selected ring is `O(p*d)`;
- implement PRNG replay or a cross-strategy generic executor;
- fix or weaken the calibrated-resume guard tracked by #789;
- claim bitwise cross-version equality when only floating accumulation order
  has changed.

## Maintainer assessment

The defect is well scoped and the mathematical correction is low ambiguity.
A minimal direct FIR operator inside BISR's existing raw factory preserves the
current RNG stream and avoids touching accounting or the generic dense solver.
The bounded iid ring is the selected and complete #795 implementation. The
#658/#722 code remains useful prior art, but neither the provider-neutral split
nor generic executor consolidation should delay this focused fix.

The maintainer landing requirements are therefore limited to preserving #857's
signed runtime/accounting split, retaining the dense equivalence oracle,
testing bounded state and downstream consumers, documenting the selected ring,
and intentionally rejecting legacy dense-history state. PRNG replay and a
generic banded-inverse executor should be tracked as follow-up work. Exact
resume additionally depends on an unchanged execution identity and base noise
scale; the urgent enforcement gap is #789, outside this patch.

With those constraints addressed, #795 can be fixed without changing the
selected strategy or its privacy accounting and with a reduction from
`O(n_steps*d)` to `O(p*d)` model-shaped state and per-step work.

## Primary sources

### Issue and audit context

- [Issue #795: Implement BISR noise with O(p) buffers](https://github.com/JetBrains-Research/opake/issues/795)
- [Issue #795 REST metadata](https://api.github.com/repos/JetBrains-Research/opake/issues/795)
- [Issue #795 public timeline](https://api.github.com/repos/JetBrains-Research/opake/issues/795/timeline)
- [Parent issue #766: Correct DP-FTRL mechanism identity, noise streams, and strategy state](https://github.com/JetBrains-Research/opake/issues/766)
- [Parent #766 sub-issues](https://api.github.com/repos/JetBrains-Research/opake/issues/766/sub_issues)
- [Sibling issue #793: namespace the lambda-CGD noise stream](https://github.com/JetBrains-Research/opake/issues/793)
- [Issue #789: stop calibrated DP-FTRL resume from changing sigma](https://github.com/JetBrains-Research/opake/issues/789)
- [Issue #360: full-horizon BISR runtime operator](https://github.com/JetBrains-Research/opake/issues/360)
- [Issue #355: row-norm setup complexity](https://github.com/JetBrains-Research/opake/issues/355)
- [Issue #353: signed Toeplitz accounting](https://github.com/JetBrains-Research/opake/issues/353)

### Paper and documentation

- [BISR paper, arXiv HTML](https://arxiv.org/html/2505.12128)
- [BISR paper abstract and version history](https://arxiv.org/abs/2505.12128)
- [Upstream BISR mechanism documentation](https://github.com/JetBrains-Research/opake/blob/e89e858a31ca5e279f796f2d322d323c8d59665e/docs/mechanisms/dp-ftrl/bisr.md)
- [Upstream mechanism comparison and memory claims](https://github.com/JetBrains-Research/opake/blob/e89e858a31ca5e279f796f2d322d323c8d59665e/docs/mechanisms/index.md#L39-L46)

### Current upstream implementation

- [BISR raw factory and strategy](https://github.com/JetBrains-Research/opake/blob/e89e858a31ca5e279f796f2d322d323c8d59665e/packages/opaque-dpftrl/src/opaque/api/dpftrl/noise/_bisr.py#L116-L239)
- [Generic dense Toeplitz inverse executor](https://github.com/JetBrains-Research/opake/blob/e89e858a31ca5e279f796f2d322d323c8d59665e/packages/opaque-dpftrl/src/opaque/api/dpftrl/noise/_toeplitz.py#L177-L309)
- [Generic namespaced streaming MF engine](https://github.com/JetBrains-Research/opake/blob/e89e858a31ca5e279f796f2d322d323c8d59665e/packages/opaque-dpftrl/src/opaque/api/dpftrl/noise/_engine.py#L378-L431)
- [Dedicated raw-factory dispatch](https://github.com/JetBrains-Research/opake/blob/e89e858a31ca5e279f796f2d322d323c8d59665e/packages/opaque-dpftrl/src/opaque/api/dpftrl/noise/_mf_gaussian_noise.py#L206-L275)
- [Paired second-moment stream construction](https://github.com/JetBrains-Research/opake/blob/e89e858a31ca5e279f796f2d322d323c8d59665e/packages/opaque-dpftrl/src/opaque/api/dpftrl/noise/_second_moment.py#L87-L109)
- [Template-driven structural checkpoint traversal](https://github.com/JetBrains-Research/opake/blob/e89e858a31ca5e279f796f2d322d323c8d59665e/packages/opaque-base/src/opaque/api/base/serialization/_structural.py#L82-L114)
- [Same-path tensor shape validation](https://github.com/JetBrains-Research/opake/blob/e89e858a31ca5e279f796f2d322d323c8d59665e/packages/opaque-engine/src/opaque/api/engine/serialization/_structural.py#L44-L73)
- [Trainer checkpoint persistence and restore](https://github.com/JetBrains-Research/opake/blob/e89e858a31ca5e279f796f2d322d323c8d59665e/packages/opaque-transformers/src/opaque/api/transformers/trainer/_checkpoint.py#L198-L368)
- [Distributed MF state schema](https://github.com/JetBrains-Research/opake/blob/e89e858a31ca5e279f796f2d322d323c8d59665e/packages/opaque-dpftrl/src/opaque/api/dpftrl/noise/_distributed.py#L86-L120)
- [Current BISR tests](https://github.com/JetBrains-Research/opake/blob/e89e858a31ca5e279f796f2d322d323c8d59665e/packages/opaque-dpftrl/tests/noise/test_bisr_noise.py)

### Historical and active pull requests

- [PR #121: initial lambda-CGD/BISR implementation](https://github.com/JetBrains-Research/opake/pull/121)
- [PR #509 / commit 2507548: full-horizon BISR correctness](https://github.com/JetBrains-Research/opake/pull/509)
- [PR #509 review requesting behavioral covariance/equivalence coverage](https://github.com/JetBrains-Research/opake/pull/509#discussion_r3726858053)
- [PR #572 / commit 4106222: closed-form row norms](https://github.com/JetBrains-Research/opake/pull/572)
- [PR #658 / commit 8c36115: direct bounded-state execution](https://github.com/JetBrains-Research/opake/pull/658)
- [PR #722: consolidated backend-neutral integration branch](https://github.com/JetBrains-Research/opake/pull/722)
- [PR #728 / commit 6f6d132: mechanism RNG stream roots](https://github.com/JetBrains-Research/opake/pull/728)
- [PR #857: signed-encoder privacy-accounting hardening](https://github.com/JetBrains-Research/opake/pull/857)
