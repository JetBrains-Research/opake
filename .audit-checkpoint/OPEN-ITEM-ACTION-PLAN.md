# OPEN-ITEM ACTION PLAN — Opaque third audit

**Scope:** the 24 issues still open across milestone **Slate** (#2) and milestone **Bazalt** (#4), reconsidered against the tree at `12146ec4` after the 124-commit delta. Every claim below was re-verified in the current working tree; label sets are the live GitHub label sets read from `JetBrains-Research/opaque` on 2026-09-13.

**Net effect if executed:** 3 issues closed as completed, 3 replaced by 4 new issues (one split), 17 corrected in place, 1 left untouched. Open audit issues go from 24 to 22, and 8 of the new findings become unfileable-as-new because they are absorbed into corrected bodies (see §6).

**Milestone Marble (#3)** contributes no open items — its 15 issues are all closed — but note that three of them (#553, #558, #566, #574, #578, #586) were verified *partial / not-fixed / regressed* in this run. Those are new findings against closed issues, not open-item work, and are out of scope here.

---

## 1. Summary table

| # | Milestone | Current title | Verdict | One-line reason |
| --- | --- | --- | --- | --- |
| 764 | Bazalt | Make accounting state fail closed across serialization, discretization, and composition | **Close — completed** | All 7 children closed; every completion clause verified at `12146ec4` (template budget preserved, partial discretization update, `u32` guard, calibration re-evaluation, asymmetric ∞-mass). |
| 767 | Bazalt | Correct DP-SGD sampling, adaptive-clipping, and accounting validation | **Close — completed** | All 6 children closed; schema-ordered per-group reduction, `AdaClip.__post_init__`, `sample_rate ∈ (0,1]`, namespaced sampler streams all verified. |
| 768 | Bazalt | Correct engine clipping, precision, and RNG boundary behavior | **Close — completed** | All 7 children closed; loss scaler deleted by #889 with no dangling reference, `global_norm` raises on wrappers, `fold_in` collision no longer reproduces. |
| 769 | Bazalt | Restore patched-model fidelity to upstream | **Keep + correct** | Real but wider: three families also have *no forward* parity, gpt2 is the documented quick-start model — and its "gradient defect" is harness dropout, not the patch. |
| 773 | Bazalt | Restore auditing statistical validity | **Keep + correct** | 3 of 4 children verified done; only the `v_k` truncation clause is unmet, so narrow the criteria to it plus the documentation fallback. |
| 774 | Bazalt | Close CI fail-open gates and restore test integrity | **Keep + correct** | The "all children closed" premise is wrong (#880 is open), and the zero-collected-tests clause was *declined* (#836/#841 `not_planned`) — record the residual risk instead of claiming the bar. |
| 775 | Bazalt | Reconcile documentation and citations with the post-remediation code | **Keep + correct** | 5 of 6 Sweep C sites are present verbatim, and the executable-docs gate that landed does not run Markdown snippets, so clause 1 is still false. |
| 813 | Bazalt | Convert parity skips to strict xfail and resolve the gpt2 gradient mismatch | **Keep + correct** | Test-gap half exactly true; gpt2 diagnosis **wrong** (shared-RNG dropout), two exemptions stale, one exemption hides a real attention-sink defect. |
| 828 | Bazalt | Bound v_k as a function of mu so strong audits converge instead of raising | **Keep + correct** | Reproduced (raise after 103.9 s at u=3800); new proof the p-value is μ-blind across 12 orders of magnitude; chunked-exact route makes it engineering → drop `needs-design`. |
| 845 | Bazalt | Sweep C — verify every citation against its record and its hypotheses | **Restate (split in two)** | All six sites live, but three mechanical citation fixes are blocked behind four theorem-hypothesis questions; two of the six claims also need correcting. |
| 911 | Bazalt | Monte Carlo calibration performance may be unusable | **Keep + correct** | "May be" is now measured (303.8 s toy calibration, ~1 h 50 m projected); #1033 fixes only the Balls-in-Bins half; the b-min-sep corpus is inert above 60 steps. |
| 956 | Bazalt | Optimize patched-model memory and throughput end to end | **Keep (no change)** | 7 of 14 children open, no benchmark harness exists anywhere, #880's markers still absent; body accurate as written. |
| 958 | Bazalt | Batch fused SDPA across the DP vmap dimension | **Keep + correct** | Only the *backward* ops lack batching rules; #985 multiplies the fallback by `num_key_value_heads`; needs a blocking per-example isolation gate. |
| 963 | Bazalt | Tune compact sliding-window attention and support padded batches | **Keep + correct** | Every defect live, and #985 made dispatch 8× worse than the issue states (measured 32 SDPA calls for one layer at S=1024). |
| 964 | Bazalt | Reuse MoE route plans and redesign streamed weight gradients | **Keep + correct** | All mechanisms live; needs a blocking gate that the route plan preserves the `samples * E + experts` per-example mapping. |
| 966 | Bazalt | Make MoE backend dispatch geometry- and evidence-driven | **Keep + correct** | `_SPARSE_MOE_MIN_EXPERTS = 16` is still the sole speed term, but as written the issue cannot start (no harness, contradicting evidence taken on Torch 2.10) — narrow it. |
| 968 | Bazalt | Pack LoRA QKV and MLP projections into fewer GEMMs | **Keep + correct (weakest; close if not measured)** | No packing exists, but the persistent packed-weight cost contradicts landed #933 and the win is unmeasured — needs an explicit kill criterion. |
| 969 | Bazalt | Make activation offload selective and overlap transfers | **Keep + correct** | Materially worse than titled: measured 94% of offloaded bytes are model *weights*, pageable, every microbatch. |
| 970 | Bazalt | Optimize MoE expert casts and full-size FP32 accumulators | **Keep + correct** | Both halves live; split the safe FP32-accumulator win from the weight-shadow half that re-opens what #933 closed. |
| 319 | Slate | Restore DP-FTRL mechanism and strategy invariants | **Keep + correct** | 6 of 7 children now closed, but criterion 2 is violated by `IdentityStrategy.sensitivity` (measured 145× ε under-report) — a path no child covers. |
| 359 | Slate | Define BandMF participation sensitivity and price second moments consistently | **Restate** | #998 answered the design question; what remains is `IdentityStrategy` plus the unaccounted paired second-moment release (measured 1.31× under-noised). |
| 376 | Slate | Calibrate one-run auditing under the null | **Restate** | Measured 4–5× Type-I inflation; the `v_k` half belongs to #828; the options are enumerable, so `needs-design` is wrong. |
| 407 | Slate | Implement proper JME second moments for DP-SGD | **Keep + correct** | Still releases mean-of-per-record-squares (reproduced: `[g, -g]` → 1.0); DP-FTRL supplies plumbing, not JME mathematics; the misattribution half is landable today. |
| 417 | Slate | Add a reproducible benchmark harness for every published number | **Keep + correct** | Its stated trigger is inert (no numeric claim survives), but 20 unmeasured perf commits landed and the only perf gate passes a 20% slowdown — re-scope and raise to medium. |

---

## 2. CLOSE

Three issues close. All three close as **completed** (`state_reason: completed`) — the work landed, it is not a scope decision. Closing an umbrella does not touch its children; all of them are already closed.

### #764 — Make accounting state fail closed across serialization, discretization, and composition
**Close as:** completed. **Labels:** unchanged. **Children:** 7/7 closed.

```
Closing: every sub-issue is landed and verified against the current tree (12146ec4), and the completion criteria hold clause by clause.

- Caller-supplied guarantee no longer discarded: `_accountant.py:258` passes the template into the loader and `:239-250` keeps `template._budget` when the checkpoint has none, logging when a checkpoint budget overrides one (OPQ-302).
- No silent narrowing: `discretization.py:163-180` is a genuine partial update (`replace(base, **overrides)`, `None` sentinels, bare call is a no-op) (OPQ-304); `numerics/fft.rs:292-299` rejects `count > u32::MAX` before `powu(count as u32)` at `:331`, with tests at `:440-457` (OPQ-303).
- No value reported under a different discretization: `calibration.py:320-329` re-evaluates the calibrated process and `achieved` under `overall_config` outside the probe context (OPQ-323); `pld/metrics.rs:314-326` folds `negative_infinity_mass` into the asymmetric CDFs, matching `pmf_beta` at `:216-218` (OPQ-305).
- Packaging: `Cargo.toml:39-40` is `[lints] workspace = true` (OPQ-308); no `opaque.dpftrl` import remains under `packages/opaque-accounting/tests/` (OPQ-307).

Audit source: OPQ-302, OPQ-303, OPQ-304, OPQ-305, OPQ-307, OPQ-308, OPQ-323.
```

### #767 — Correct DP-SGD sampling, adaptive-clipping, and accounting validation
**Close as:** completed. **Labels:** unchanged. **Children:** 6/6 closed.

```
Closing: every sub-issue is landed and verified against the current tree (12146ec4).

- Identical per-rank thresholds: `dpsgd/clipping/_distributed.py:64-70` reduces per-group counts over `sorted(current_pg.values)`, so the collective schedule comes from the schema rather than a rank-local dict order, and the per-group noise key folds the sorted index (`:76-78`) (OPQ-309). No other `reduce_scalar` call site in `packages/*/src` is driven by dict iteration order.
- Validation at construction and on deserialization: `accounting/dpsgd/mechanisms/_adaclip.py:26-67` adds `AdaClip.__post_init__` rejecting `num_groups < 1`, non-positive `fraction_noise_std` / `expected_batch_size` (OPQ-310). Checked the sibling too: `Gaussian(-1.0)` raises and `Gaussian(0.0)` reports `inf`.
- Runtime/accounting parity: `accounting/dpsgd/amplification/_poisson.py:49-58` accepts `(0, 1]`, matching `sampling/_poisson.py` and `docs/user-guide/sampling.md:42` (OPQ-312); samplers now root under namespaced tags (`POISSON_STREAM_FOLD`, `K_OUT_OF_T_STREAM_FOLD`, `CYCLIC_POISSON_STREAM_FOLD`, `B_MIN_SEP_STREAM_FOLD`, `BALLS_IN_BINS_STREAM_FOLD`) (OPQ-311); `num_groups` is documented in both reference pages and the per-group clipping recipe (OPQ-356); the epoch transform is memoised by #881 (OPQ-313).

Audit source: OPQ-309, OPQ-310, OPQ-311, OPQ-312, OPQ-313, OPQ-356.
```

### #768 — Correct engine clipping, precision, and RNG boundary behavior
**Close as:** completed. **Labels:** unchanged. **Children:** 7/7 closed.

```
Closing: every sub-issue is landed and verified against the current tree (12146ec4).

- Adaptive precision: 7f55aae5 (#889) removed `engine/precision/_loss_scaler.py` outright, so the unaccounted `all_finite`-driven scale schedule no longer exists (OPQ-330). The removal is clean — no reference to `LossScaler` / `unscale_grads` / `all_finite` survives anywhere in source, docs or CI.
- Wrapper types: `engine/pytree.py:377-400` raises `InputTypeError` pointing at `.pytree` when `global_norm` receives a DP wrapper (confirmed by execution); `unscale_grads` is gone with the scaler (OPQ-333).
- No silent degradation: `clipping/_per_group.py:166-179` raises on patterns that match no parameter, with `allow_unused_patterns` as the explicit opt-out (OPQ-332); `_clipped_grad.py:256-263` derives aux `clipping_rate` presence from the clipping schema so a partially-empty Poisson round no longer aborts a DDP run (OPQ-331); `_empty_batch_response` (`:223-234`) applies `pre_clipping_transform` and the configured `dtype`, and `dpsgd/clipping/_adaptive.py:457` uses the same helper (OPQ-334).
- RNG boundary: `engine/random/_engine.py:29-39` adds the `b"i"` / `b"s"` type discriminator; the reported 16-byte-string/integer collision no longer reproduces (OPQ-335). `profiling/_memory.py:268-276` documents the MPS peak semantics (OPQ-336).

Audit source: OPQ-330, OPQ-331, OPQ-332, OPQ-333, OPQ-334, OPQ-335, OPQ-336.
```

**Attached caveat (do not put it in the close comment):** the one residue of the OPQ-333 wrapper-blindness class — `tree_leaves` returning `[]` for Opaque wrapper types (`packages/opaque-engine/src/opaque/api/engine/pytree.py:111-129`, low) — must be filed as a **standalone** issue (`pkg: engine`, `severity: low`, `impact: api`), not as a child of #768. It is documented behaviour on a non-clipping primitive and does not block this umbrella.

**Superseded issues also closed in §3 (not-planned):** #845, #359, #376.

---

## 3. RESTATE

Each restate is executed as: file the replacement(s) → link the parent → close the superseded issue as `not_planned` with the comment given. If you would rather preserve the number (relevant for #359, which is referenced from #319 and from `docs/reference/accounting.md`), apply the same title/body/labels **in place** and skip the close; the replacement text is identical either way.

**Milestone for replacements:** keep the superseded issue's milestone (Bazalt for the #845 pair, Slate for the #359 and #376 replacements) so audit-cycle bookkeeping stays truthful. If a third-audit milestone is opened, move all four there instead.

### 3.1 #845 → two issues (split)

**Why split:** OPQ-371, OPQ-372 and the bibliographic half of OPQ-375 are mechanical (wrong author, wrong title, missing identifier, a docstring contradicting its own function) and are checkable by a reviewer with a browser. OPQ-373, OPQ-374, the newly found equal-bins contradiction, and the residual of OPQ-376 require reading theorem hypotheses against the deployed default — DP-review work under `.junie/differential-privacy-review.md`, with one adjacency question (`dataset_size` public and fixed) that touches what the library may compose. Keeping them in one issue has meant the cheap half blocks behind the expensive half for two audit cycles.

#### Replacement A — supersedes #845

**Title:** `Sweep C1 — make every citation resolve to the right record`
**Labels:** `documentation`, `source: audit`, `severity: low`, `area: docs`
**Parent:** #775 (add as sub-issue). **Milestone:** Bazalt.

```markdown
## Problem

Three citation sites still name the wrong record. Closed #418 swept the references once and missed `_schedule_free.py`; #503 re-introduced entries #502 had just deleted. A manual pass demonstrably does not hold.

| ID | Site | Defect |
| --- | --- | --- |
| OPQ-371 | `opaque-optimizers/src/opaque/api/optimizers/_schedule_free.py:10-11` | Cites "The Road Less Scheduled" (arXiv:2405.15682) as "Defazio, Yaida, Cutkosky"; Yaida is not an author. Still unfixed from July's OPQ-136. |
| OPQ-372 | `opaque-alignment/src/opaque/api/alignment/dpo/loss/_discopop.py:5-7`; `logprob/_sequence.py:15-16` | DiscoPOP attributed to "Azar, M. G., et al. (2024) ... Using Self-Supervised Feedback"; the real record is Lu et al., "Discovering Preference Optimization Algorithms with and for Large Language Models", arXiv:2406.08414. Separately, `_sequence.py:15-16` states the `ld_alpha` (LD-DPO) split "is not implemented here" while the same module implements it at `:37, 54-62, 82-95`. |
| OPQ-375 (bibliographic half) | `opaque-dpftrl/src/opaque/api/dpftrl/sampling/_balls_in_bins.py:3-5, 10, 27, 48` | Cites "Definition 3.1" and "Lemma 3.2" of "Choquette-Choo et al. (2024), *Privacy Amplification for Matrix Mechanisms*" with no identifier. Those results are in a **different** paper: Choquette-Choo, Ganesh, Haque, Steinke, Thakurta, *Near Exact Privacy Amplification for Matrix Mechanisms*, arXiv:2410.06266 (Definition 3.1 p. 7, Lemma 3.2 p. 8). The accounting module (`accounting/dpftrl/amplification/_balls_in_bins.py:47-48`) and `matrix_factorization/mod.rs:41` already use the correct title/id, so the tree is internally inconsistent. |

## Impact

A reviewer checking a sensitivity or amplification claim is sent to a paper that does not contain the cited result, or to no resolvable record at all. Two of the three are on privacy-relevant paths. The `_sequence.py` docstring additionally tells a user a shipped feature does not exist.

## Acceptance criteria

- Every citation in the three sites above resolves to a real record with correct authors, title and a stable identifier (arXiv id or DOI).
- `_sequence.py`'s module docstring describes the `ld_alpha` support the module actually implements, and names the fused path that does not support it (`:134`).
- The check is a per-file checklist generated from the source (every docstring/comment reference paired with an identifier) rather than a manual pass, so it survives the next refactor.

## Split out

The hypothesis half of Sweep C — OPQ-373, OPQ-374, the residual of OPQ-376, and a newly found contradiction at the OPQ-375 site — is tracked in Sweep C2 (link the issue number once filed). Those are defects where the citation resolves but the cited result does not cover the deployed configuration; they need a mathematical answer, not a reference fix.

Audit source: OPQ-371, 372, 375.
```

#### Replacement B — supersedes #845

**Title:** `Sweep C2 — check every cited hypothesis against the deployed default`
**Labels:** `documentation`, `source: audit`, `severity: medium`, `area: docs`, `pkg: accounting`, `pkg: dpftrl`, `impact: privacy`, `impact: epsilon`
**Parent:** #775 (add as sub-issue). **Milestone:** Bazalt.
`impact: epsilon` is warranted here and not on C1: the truncated-Poisson site conditions its guarantee on an adjacency (`|D|` fixed and public) that the library's advertised add-or-remove model does not state, so a reported ε may not hold as composed.

```markdown
## Problem

Four sites cite a real record whose stated result does not cover the configuration Opaque deploys by default.

- **OPQ-373** — `opaque-accounting/src/amplification/truncated_poisson.rs:18, 124` cites the bare key `[Gan25]`; nothing in the repository defines it (the source is Ganesh, *Tighter Privacy Analysis for Truncated Poisson Sampling*, arXiv:2508.15089). That paper's guarantee is stated for a restricted add-or-remove-1-of-n adjacency with |D| fixed and public, and cautions against composing the result with other mechanisms. `p_trunc = Pr[Binom(n-1,q) >= B]` makes the analysis n-specific, yet `docs/reference/accounting.md:242-259` describes the truncated form only as "not better than plain Poisson" and the accountant composes it freely. The same key appears at `packages/opaque-dpsgd/tests/accounting/test_cross_validation.py:460`.
- **OPQ-374** — `docs/mechanisms/dp-ftrl/lambda-cgd.md:70-73` and `matrix_factorization/lambda_cgd.rs:8, 65, 130` attribute the sensitivity to "Theorem 1, eq 15", but the default is `normalized=True` (`dpftrl/noise/_lambda_cgd.py:106, 175-176`) and the Rust normalized path documents a different result — "the Gram matrix from Lemma 8 of the paper" (`lambda_cgd.rs:154-161`). The docs cite a theorem the deployed default does not use, and eq (15) is a utility metric.
- **OPQ-375 (hypothesis half)** — `accounting/dpftrl/amplification/_balls_in_bins.py:3-4` says the dataset is "randomly partitioned into `num_bins` **equally-sized** bins", contradicting its own sampler (`sampling/_balls_in_bins.py:4-6`: sizes are Binomial(N, 1/num_bins)) and Definition 3.1 of arXiv:2410.06266. Which participation model holds is what the dominating pair depends on. (The signed-encoder hypothesis of Lemma 3.2 — C lower-triangular with non-negative entries — was already addressed by e87a0c21 / #857 via the |C| triangle-inequality extension; document that as the assumption it is.)
- **OPQ-376 (residual)** — `dpftrl/noise/_bisr.py:232-241` implements Lemma 1 with α hard-pinned to 1 and exposes no α on `BisrStrategy` (`:629-688`). The β half of the `0 <= β < α <= 1` hypothesis IS now enforced (`:660-663`, added by #857), so only the un-exposed α and the unvalidated `inv_coefficients` override (`:643, 674-688`) remain: a caller-supplied C⁻¹ is checked for length, finiteness and leading magnitude, but never against the lemma's hypotheses.

## Impact

Each of these is a place where a reviewer can follow the citation, find the theorem, and still not learn whether the deployed default is covered by it. OPQ-373 is the most consequential: it conditions a privacy guarantee on a public, fixed dataset size under a different adjacency from plain Poisson, and the accountant composes it as though it were interchangeable. OPQ-375's equal-bins contradiction is a disagreement about the participation model the dominating-pair argument rests on — the sampler and the accountant currently describe different experiments.

## Acceptance criteria

- For each site the cited result's hypotheses are written out and checked against the deployed default; any gap is closed in code (a validation) or stated as an assumption in the user-facing docs.
- `[Gan25]` is replaced with the full citation, and the truncated-Poisson docstring and `docs/reference/accounting.md` state that the guarantee is under add-or-remove-1-of-n with `dataset_size` public and fixed, that this is a different adjacency from plain Poisson, and that the source cautions against composing it.
- The λ-CGD docs cite the result the normalized default actually uses (Lemma 8 / eq (12)), not Theorem 1 / eq (15).
- The balls-in-bins bin-size description is made consistent across sampler, accountant and paper, and the |C| triangle-inequality extension from #857 is documented as the assumption it is.
- `BisrStrategy` either exposes α or documents that it is pinned to 1 in `docs/mechanisms/dp-ftrl/bisr.md`, and `inv_coefficients` is validated against Lemma 1's hypotheses or rejected.

Audit source: OPQ-373, OPQ-374, OPQ-375, OPQ-376.
```

#### Close comment for #845 (`state_reason: not_planned`)

```
Superseded. Re-verified site by site against 12146ec4: five of the six findings are present verbatim (OPQ-371 `_schedule_free.py:10-11`, OPQ-372a `_discopop.py:5-7`, OPQ-372b `logprob/_sequence.py:15-16`, OPQ-373 `truncated_poisson.rs:18,124`, OPQ-374 `lambda-cgd.md` + `lambda_cgd.rs:8,65,130`, OPQ-375 `_balls_in_bins.py`), so nothing here is done. Two claims needed correcting: Definition 3.1 / Lemma 3.2 do exist, but in arXiv:2410.06266 (*Near Exact* Privacy Amplification for Matrix Mechanisms), not the paper cited at the site — this is a wrong-title/missing-id defect, not an unfindable result; and OPQ-376's "enforces none of its 0 <= beta < alpha <= 1 hypothesis" is stale, e87a0c21 (#857) added the beta check at `_bisr.py:660-663`.

Splitting rather than re-scoping in place: the three bibliographic fixes are mechanical and reviewable with a browser, while OPQ-373/374/376 and a newly found equal-bins contradiction at the OPQ-375 site need theorem hypotheses checked against the deployed default under `.junie/differential-privacy-review.md`. Bundled, the cheap half has blocked behind the expensive half for two audit cycles.

Continued in Sweep C1 (citations resolve to the right record) and Sweep C2 (cited hypotheses cover the deployed default), both under #775.
```

### 3.2 #359 → replacement

**Title:** `Make every MF strategy honour the participation schema and tie the paired second-moment release to an accountant`
**Labels:** `bug`, `source: audit`, `severity: high`, `pkg: accounting`, `pkg: dpftrl`, `impact: epsilon`, `impact: privacy`
**Remove from the current set:** `blocked`, `needs-design` — e44d6ab6 (#998) decided the design question, and both residues are concrete.
**Parent:** #319 (add as sub-issue). **Milestone:** Slate.

```markdown
## Problem

e44d6ab6 (#998) settled the design question behind #359: MF strategy sensitivity is **participation-aware**, and a caller that wants the single-participation column norm asks for it explicitly with `min_sep=n_steps, max_participations=1`. `BandMfStrategy` now implements that contract (`packages/opaque-dpftrl/src/opaque/api/dpftrl/noise/_band_mf.py:138-180`), and the two amplifier call sites were updated to match. Two residues remain.

**1. `IdentityStrategy` does not implement the contract.**

```python
# packages/opaque-dpftrl/src/opaque/api/dpftrl/noise/_identity.py:45-46
def sensitivity(self, **_) -> float:
    return 1.0
```

For `C = I_n`, a record participating in `k` steps contributes `k` distinct columns, so the schema sensitivity is `sqrt(k)`, not 1. `MfGaussian.pld` consumes this value as participation-aware (`.../accounting/dpftrl/mechanisms/_mf_gaussian.py:132-140`) and `max_participations=None` resolves to `n_steps`, so the *default* bare call charges one participation for an `n_steps`-participation release.

```python
mf_gaussian(1.0, identity_strategy(), n_steps=1000).epsilon_at(delta=1e-5)  # -> 4.3772
(dpsgd_acc.gaussian(1.0) * 1000).epsilon_at(delta=1e-5)                     # -> 633.93
```

The API accepts `min_sep` and `max_participations` and silently discards them. The only warning is one sentence at `docs/reference/accounting.md:420-421`, whose BandMF half is now stale.

**2. The paired second-moment release is not tied to any accountant.**

`make_second_moment_mf_noise` (`.../dpftrl/noise/_second_moment.py:140-150`) calibrates the joint Mahalanobis budget from single-participation `c1`/`c2`. That cancels correctly only when both strategies scale identically with `k`. Nothing enforces that, and `docs/user-guide/noise.md:305-307` explicitly blesses a differing second-moment workload while `docs/user-guide/dp-ftrl.md:177-178` claims 'the joint allocation is accounted with the same mechanism PLD as the first-moment release'. Measured counter-example (`n=64, min_sep=8, max_participations=8, nm=3.0`, first=`band_mf_strategy(bands=8)`, second=`bisr_strategy(bandwidth=8)`): actual joint budget 1.5279 against a charged 0.8889 -- the effective noise multiplier is over-stated by 1.311x.

## Impact

Under-reported epsilon in two configurations both public APIs accept without a warning. (1) is unbounded in the horizon (`sqrt(n_steps)`); (2) is bounded by the ratio of the two strategies' participation growth factors but is reachable from a documented, recommended configuration.

## Acceptance criteria

- `IdentityStrategy.sensitivity` either returns the schema-correct value or raises for any schema it does not price; no strategy silently ignores `min_sep`/`max_participations`.
- Charged privacy cost is non-decreasing in `max_participations` for every strategy, with a test that sweeps all six shipped strategies.
- The paired second-moment release either (a) exposes an accountant that consumes both strategies and is cross-validated against the realized joint Mahalanobis budget, or (b) rejects a second strategy whose participation growth factor differs from the first's, or (c) is removed with #407.
- `docs/reference/accounting.md:420-421` is corrected: BandMF bare sensitivity is participation-aware since #998; only Identity still is not (until fixed).
- `docs/user-guide/dp-ftrl.md:177-178` states the condition under which the paired PLD claim actually holds.

Audit source: OPQ-177 (AUD-084, AUD-186); re-scoped September 2026 after #998 decided the design question.
```

#### Close comment for #359 (`state_reason: not_planned`)

```
Superseded — the blocking design question is answered, so the "blocked + needs-design" framing is obsolete, but two concrete defects this title does not name are still live.

Decided by e44d6ab6 (#998): `BandMfStrategy.sensitivity` is participation-aware (`_band_mf.py:138-180`), and the two amplifier call sites now ask explicitly for the single-participation column norm (`_poisson.py:180-184`, `_b_min_sep/__init__.py:166-170`) with the theorem each relies on named in-line. Criterion 1 is met for every correlated strategy — measured at n_steps=64, min_sep=8 for k=1/2/4/8: bandmf(8) 1.0/1.414/2.0/2.828, blt(3) 1.543/2.361/3.542/5.218, bsr 1.080/1.528/2.161/3.056, bisr 1.0/1.966/3.003/4.416, lcgd(0.9) 1.0/1.691/2.736/4.157. Criterion 3 is met via the non-increasing-majorant upper bound (`minsep_sensitivity_upper_bound`, `_toeplitz.py:520-557`), which is safe for min_sep < bands.

Still broken: `IdentityStrategy.sensitivity` returns 1.0 for any schema (`_identity.py:45-46`), so `mf_gaussian(1.0, identity_strategy(), n_steps=1000)` reports eps=4.377 against a true 633.93; and criterion 2 is untouched — `_second_moment.py:140-150` pins the joint budget to single-participation c1/c2, the accountant never sees `second_moment_strategy`, and a documented mismatched pair (first=band_mf(8), second=bisr(8), n=64, min_sep=8, k=8) is 1.311x under-noised relative to the charged budget.

Leaving this issue under its current title would invite a reader to conclude it is done. Continued in the replacement, under #319.
```

### 3.3 #376 → replacement

**Title:** `Correct the Type-I rate of one-run audits: label-selected thresholds have no multiplicity treatment`
**Labels:** `bug`, `source: audit`, `severity: medium`, `pkg: auditing`, `impact: epsilon`, `impact: privacy`, `impact: test-gap`
**Remove from the current set:** `needs-design` — the three remedies are enumerable and stated in the criteria.
**Parent:** none (standalone; the July auditing umbrella #327 is closed and #773's scope is now only #828). **Milestone:** Slate.

```markdown
## Problem

`OneRunEstimate._best_r_u` selects the reporting threshold by maximising accuracy against the **true** membership labels, then feeds the resulting `(r, u)` to a test run at the raw significance:

```python
# packages/opaque-auditing/src/opaque/api/auditing/one_run/_estimate.py:166-168
correct = (self.n_in - self.fn_counts) + self.tn_counts
best_c = int(np.max(correct))
return m, m - best_c
```

This is the default path: `epsilon_at`, `delta_at`, `beta_at` and `advantage` all take `threshold=None`. No Bonferroni, max-statistic, or sample-splitting correction exists anywhere in the estimator (`_eps_delta.py`, `_gdp.py`), so the reported value is not a valid lower bound at the stated significance.

Measured on the current tree (400 trials per cell, scores drawn independently of membership, so the true epsilon is 0; `eps_delta().epsilon_at(significance=0.05, delta=0)`):

| m | n_in | empirical P(eps_hat > 0) | nominal alpha |
| --- | --- | --- | --- |
| 100 | 50 | 0.210 | 0.05 |
| 500 | 250 | 0.247 | 0.05 |

`docs/reference/auditing.md:404-409` presents this default as 'Largest eps the audit can certify at the given delta' and mentions only that 'the Pareto-optimal threshold maximising TP + TN is used'. There is no null-calibration, Type-I, or coverage test in `packages/opaque-auditing/tests/auditing/`.

## Impact

A practitioner who audits a correctly-noised mechanism gets a non-zero eps_hat roughly one run in four and has no way to tell that from real leakage. Because the audit is used to *check* a privacy claim, an inflated Type-I rate makes the check report violations that are not there -- and, once a correction is applied, some currently-reported eps_hat values will fall, which is the honest direction.

Scope note: the anti-conservative `v_k = 0.5` truncated-rank substitution in the mu-GDP estimator is the *other* half of the original #376 and is tracked separately in #828; it is not in scope here.

## Acceptance criteria

- One of three explicit decisions is taken and documented: (a) `threshold` becomes required for a certified estimate and the label-selected default is demoted to an explicitly-labelled exploratory statistic; (b) a max-statistic / Bonferroni correction over the realized Pareto threshold set is applied inside `_best_r_u`'s consumers; or (c) the canaries are split into a selection half and a test half.
- Across at least 200 null runs at m in {100, 500, 2000}, the false-positive rate is at most nominal alpha plus documented binomial slack, for both `eps_delta()` and `gdp()`.
- Negative controls and both adjacency directions are covered.
- `docs/reference/auditing.md` states which statistic is certified at the stated significance and which is not.

Audit sources: AUD-136, AUD-171 / gate G6; re-scoped September 2026 (v_k truncation split out to #828).
```

#### Close comment for #376 (`state_reason: not_planned`)

```
Superseded and re-scoped. The defect is real and now quantified: with `threshold=None`, `_best_r_u` takes `best_c = int(np.max(correct))` over every Pareto threshold using the true labels (`one_run/_estimate.py:166-168`) with no multiplicity treatment anywhere. Measured on the current tree, 400 null trials per cell with scores drawn independently of membership, the empirical false-positive rate of `eps_delta().epsilon_at(significance=0.05, delta=0) > 0` is 84/400 = 0.210 at m=100 and 99/400 = 0.247 at m=500 — 4-5x the nominal 0.05 — while `docs/reference/auditing.md:404-409` presents the default as the "largest eps the audit can certify" with no caveat. There is still no null / Type-I / coverage test in `packages/opaque-auditing/tests/auditing/`.

Two reasons to restate rather than keep: this issue bundled the `v_k = 0.5` truncated-rank substitution, which is byte-unchanged (`one_run/_gdp.py:39-41, 336-339, 389`) but is exactly the subject of #828 — and merging in the other direction would be wrong too, since #828 is a bounded numerical fix while threshold multiplicity is a separate estimator-validity defect with three enumerable remedies. The `needs-design` label no longer fits.

Continued in the replacement; the v_k clause is delegated to #828.
```

---

## 4. KEEP WITH CORRECTION

17 issues. For each: what changes, and why. Bodies are given in full wherever the body changes. Labels are stated as **final desired set** so they can be applied with one call.

### #813 — parity skips / gpt2

- **Title →** `Make grad-parity exemptions strict and visible, and fix the dropout non-determinism that fakes the gpt2 failure`
- **Labels:** unchanged — `bug`, `source: audit`, `severity: medium`, `pkg: patches`, `impact: numerical`, `impact: test-gap`.
- **Severity:** unchanged (medium). The high-severity part of what was seen here is the attention-sink deletion, which becomes its own issue rather than inflating this one.
- **Why:** the test-gap half is exactly true (`_GRAD_PARITY_SKIP_FAMILIES` at `test_parity.py:97-103`, consumed as plain `pytest.skip` at `:278-279` and `:483-484`; the coverage guard at `:509-520` subtracts only `_PARITY_SKIP_FAMILIES`). The *causal diagnosis is wrong*: with the skip set cleared gpt2 does fail, but running `assert_parity_grad` directly shows the cause is the harness — `get_tiny_config_kwargs()` (`_test_utils.py:14-27`) sets no dropout keys, and both grad helpers put the two models in `.train()` and run them sequentially off the shared global RNG, so they draw different dropout masks. GPT-2 is the only registered family with nonzero default dropout. With `resid_pdrop=embd_pdrop=attn_pdrop=0.0` the two losses are bit-identical and 0 of 28 gradients mismatch. Acting on the existing second acceptance criterion would either hunt a non-existent backward bug or de-register a family that is bit-exact. Two exemptions are also stale (`gpt_oss`, `deepseek_v4` pass vmap-grad today) and one hides a real defect. Line refs in the current text have drifted.

**Body (replace in full):**

```markdown
Parent: #769

## Problem

The parity suite exempts families from gradient and vmap-grad parity with bare `pytest.skip`, and one of those exemptions is caused by the harness rather than by the patches.

**1. The exemptions are invisible and cannot rot-detect.** `_GRAD_PARITY_SKIP_FAMILIES = {gpt2, qwen3_next, deepseek_v4, gpt_oss}` (`packages/opaque-patches/tests/transformers/models/test_parity.py:97-103`) is consumed as a plain `pytest.skip` at `:278-279` and `:483-484`. `_STRICT_FORWARD_PARITY_SKIP_FAMILIES` (`:88-92`) exempts three more families from forward parity. The coverage guard `test_parity_families_covered` (`:509-520`) subtracts only `_PARITY_SKIP_FAMILIES` (`:82-84`), so neither exemption set is visible to it. Nothing reports when an exempt family starts passing.

**2. Two exemptions are already stale.** Clearing `_GRAD_PARITY_SKIP_FAMILIES` and rerunning: `test_vmap_grad_parity[gpt_oss]` and `test_vmap_grad_parity[deepseek_v4]` PASS today.

**3. gpt2 is not a patch defect — the harness is non-deterministic under dropout.** `assert_parity_grad` (`_test_utils.py:461-482`) and `assert_parity_vmap_grad` (`:525-556`) put both models in `.train()` and run them sequentially off the shared global RNG, and `get_tiny_config_kwargs()` (`:14-27`) sets no dropout keys. GPT-2 is the only registered family whose config defaults to nonzero dropout (`resid_pdrop=embd_pdrop=attn_pdrop=0.1`), so the two models draw different dropout masks. Measured on a tiny GPT-2 pair: at the default dropout all 28 parameters mismatch and even the two losses differ (4.198711 vs 4.208967); with the three dropout probabilities set to 0.0, the losses are bit-identical, 0 of 28 gradients mismatch, and `assert_parity_vmap_grad` passes. `models/gpt2.py:41-49` applies only `batchify` + `kv_cache` and disables every family shim (`*_replacement=None`) — there is no altered backward graph.

**4. The deepseek_v4 / gpt_oss backward-grads exemption hides a real defect.** With the skip cleared, `test_backward_grads_parity` for both fails with `missing={'model.layers.0.self_attn.sinks'}` — the patched model produces no gradient for the attention-sink parameter at all. The exemption reason ("runtime compatibility shims change sink gradients") understates this: the shim removes the sink from the softmax. Tracked separately; this issue must not convert that entry to a permanent xfail before it is fixed.

## Impact

The grad-parity suite reports green while (a) it is silently non-deterministic for any family with dropout, (b) two exemptions no longer correspond to any failure, and (c) one exemption conceals a genuine fidelity break in the family a user would train. Because the exemptions are `skip`, the hole is permanent by construction. `test_vmap_grad_parity` is the DP-relevant lane — `vmap(grad(...))` is exactly how per-record gradients reach `clipped_grad`.

## Acceptance criteria

- `get_tiny_config_kwargs()` (or `_base_config_kwargs`) forces every dropout probability to 0 for every family, and/or both grad helpers reseed immediately before each model's forward, so the comparison is deterministic. A regression test asserts parity on a family with nonzero default dropout.
- `gpt2` is removed from `_GRAD_PARITY_SKIP_FAMILIES` once dropout is pinned; it passes both grad-parity tests. It is NOT de-registered from `supported_families()` — it is bit-exact.
- `gpt_oss` and `deepseek_v4` are removed from the vmap-grad exemption (they pass today).
- Remaining exemptions become `pytest.param(family, marks=pytest.mark.xfail(strict=True, reason=...))` with an issue reference in the reason, so a family that starts passing fails the suite.
- `test_parity_families_covered` asserts that every family in `_GRAD_PARITY_SKIP_FAMILIES` and `_STRICT_FORWARD_PARITY_SKIP_FAMILIES` is a registered family and carries a reason, so the exemption sets cannot silently accumulate.

Audit source: OPQ-339.
```

### #828 — μ-GDP estimator refuses strong audits

- **Title →** `Stop the mu-GDP estimator from refusing strong audits: compute v_k for all ranks (chunked) or bound the truncated tail`
- **Labels (final):** `bug`, `source: audit`, `severity: medium`, `pkg: auditing`, `impact: epsilon`, `impact: numerical` — **remove `needs-design`**.
- **Why:** fully reproducible (`_MAX_EXACT_RANKS = 2000` at `_gdp.py:41`; raise after 103.9 s at (r=10000, u=3800) versus ε=11.5598 at u=3900). New evidence upgrades the diagnosis from "slow convergence" to a proof: `_p_value(10000, 10000, 3800, μ, 10000)` returns 4.5211e-05 at μ=10, 100, 1e4 **and** 1e12 — the statistic is μ-blind below the `n_trunc*0.5` floor, so no bracket width can ever succeed. `needs-design` is no longer justified: `_compute_v_k` (`:342-379`) reduces independent per-rank integrals with `logsumexp(axis=1)`, so chunking over ranks is exact at the same ~160 MB peak, with no new mathematics. Also: the raise is now `opaque.exceptions.OperationError` after 7b26f655 (#760), the line refs have drifted, and #829's grid work (3ccf15ba, #890) changed nothing here.

**Body (replace in full):**

```markdown
Parent: #773

## Problem

`_MAX_EXACT_RANKS = 2000` (`packages/opaque-auditing/src/opaque/api/auditing/one_run/_gdp.py:41`) pins every rank past the top 2000 at `v_k = 0.5` (`:336-338`), so the Chernoff expectation has a hard floor of `n_trunc * 0.5` (`:389`). Once the observed error count `u` falls below that floor, **the p-value stops depending on μ at all** — `_p_value(m=10000, r=10000, u=3800, μ, grid=10000)` returns 4.5211e-05 at μ=10, μ=100, μ=10⁴ and μ=10¹², identical to six significant figures. The bracket search at `:79-92` therefore cannot succeed for any finite μ and raises `OperationError` after exhausting 60 doublings.

Measured on the default `epsilon_at` surface with n_in=n_out=5000:

| (r, u) | attack accuracy | result |
|---|---|---|
| (10000, 3900) | 61% | ε = 11.5598 in 16.1 s |
| (10000, 3800) | 62% | `OperationError` after 103.9 s |

The cliff is at roughly 62% attack accuracy at m = 10 000 — the scale `docs/user-guide/auditing.md:220-228` recommends, on a page that explicitly tells the reader its ceiling table "does not apply to the μ-GDP estimator". Neither auditing doc page mentions 2000, truncation, or any limit (`grep` over both returns nothing).

## Impact

The estimator fails exactly where it is most informative: the stronger the attack, the more likely the refusal. A practitioner following the guide's own canary-count advice on an under-noised mechanism gets an exception after ~100 s instead of the large ε the run actually demonstrates. It is fail-closed (no wrong ε is emitted), but the audit cannot do its job and nothing warns the user in advance.

## Acceptance criteria

- The default `epsilon_at` surface returns a finite ε for (m=10000, r=10000, u=3800), i.e. the p-value becomes a strictly increasing function of μ again for every reachable (r, u). Two acceptable routes, not mutually exclusive:
  - **Exact, no new mathematics:** chunk `_compute_v_k` (`:342-379`) over `k_vals` and compute all `r` ranks. Each row of the log-integrand is an independent integral, so chunking is exact and keeps peak memory at the current ~160 MB. Runtime is linear in the rank count (~5× at r=10000), so pair it with a cheaper fast path.
  - **μ-dependent bound:** any valid upper bound on `v_k` for truncated ranks that decays with μ restores p-value growth; state and cite the bound.
- A regression test pins the μ-monotonicity property directly (`_p_value` strictly increasing in μ at fixed (m, r, u) in the truncated regime), not just the absence of an exception.
- If any truncation survives, `docs/user-guide/auditing.md` states the limit next to the canary-count table, and the error message points at it.
- Remove the `needs-design` label: the chunked-exact route is a known-correct implementation task.

Statistical calibration under the null remains tracked by open #376.

Audit source: OPQ-350.
```

> Note: the trailer line above says "#376". If §3.3 is executed first, replace it with the replacement issue's number.

### #769 — patched-model fidelity (umbrella)

- **Title →** `Restore patched-model fidelity to upstream (gpt2 gradients, and the families with no parity reference)`
- **Labels:** unchanged — `source: audit`, `severity: high`, `pkg: patches`, `impact: numerical`, `impact: test-gap`.
- **Why:** six sub-issues landed (#811, #812, #814, #815, #816, #817); #813 is unchanged; #898 was added as an eighth child. Three things the current text does not carry: (1) `_STRICT_FORWARD_PARITY_SKIP_FAMILIES` (`test_parity.py:88-92`, skipped at `:251-252`) leaves three registered families with **no forward parity either**, so the criterion "every registered family passes forward … parity" cannot be met as written and needs an exemption mechanism; (2) gpt2 is not a corner case — `supported_families()` includes it and the user guide uses it as the quick-start model (`docs/user-guide/huggingface/index.md:54`, `dptrainer.md:21-23`, `model-patches.md:180`), and #814's fail-closed signal only covers *unregistered* families; (3) the criteria must not order gpt2's de-registration.
- **Reconciliation I applied (deliberate deviation from the reconsider text):** the reconsider pass produced two incompatible statements — this umbrella's draft says "gpt2 genuinely fails", and #813's deeper investigation proves the failure is the harness's shared-RNG dropout and that gpt2 is bit-exact once dropout is pinned. #813's evidence is strictly more specific (it ran `assert_parity_grad` directly with dropout zeroed: 0 of 28 mismatches, bit-identical losses). I have therefore amended the gpt2 bullet and the third completion criterion below to match #813. **Consequence for the new-findings list: the candidate finding "GPT-2 is the documented quick-start model … its per-example gradients diverge from upstream by 53× relative error" (high) must be withdrawn, not filed.** What remains true and worth stating is that gpt2 is promoted in the docs while unverified by the suite — which is what the amended bullet says.

**Body (replace in full):**

```markdown
## Goal

Guarantee that a patched model computes the same function as its upstream implementation, that the parity suite can prove it, and that no family the documentation promotes is silently exempt from that proof.

## Scope

Six of the original sub-issues are landed (#811, #812, #814, #815, #816, #817). What remains, re-scoped against the tree at 12146ec4:

- **#813 — grad-parity skips and the gpt2 mismatch.** `packages/opaque-patches/tests/transformers/models/test_parity.py:97-102` still skips `gpt2`, `qwen3_next`, `deepseek_v4`, `gpt_oss` from `test_backward_grads_parity` (`:278-279`) and `test_vmap_grad_parity` (`:483-484`) with bare `pytest.skip`. Re-running with the set cleared shows the exemption list has rotted in three directions: `gpt_oss` and `deepseek_v4` now **pass** vmap-grad parity; `qwen3_next` fails for the stated upstream reason (`masking_utils` data-dependent control flow under vmap); and `gpt2` fails (`transformer.h.0.attn.c_attn.bias mismatch: abs_err=1.93e-02, rel_err=5.30e+01` against `rtol=1e-05`) **because the harness is non-deterministic under dropout, not because the patch changes the backward graph** — with `resid_pdrop=embd_pdrop=attn_pdrop=0.0` the two models' losses are bit-identical and 0 of 28 gradients mismatch. See #813 for the measurement.
- **New: the forward-parity hole.** `test_parity.py:88-92` also defines `_STRICT_FORWARD_PARITY_SKIP_FAMILIES = {deepseek_v4, gpt_oss, qwen3_next}`, and `test_forward_logits_parity:251-252` skips them, so three registered families have no forward-parity coverage either. No sub-issue covers this and the completion criteria as originally written cannot be satisfied for them.
- **New: gpt2 is promoted while unverified.** `supported_families()` includes `gpt2`, so `DPTrainer` patches it with no signal (#814's fail-closed path only covers *unregistered* families), and the user guide uses it as the quick-start model (`docs/user-guide/huggingface/index.md:54`, `dptrainer.md:21-23`) and lists it in the compatibility matrix (`model-patches.md:180`) — while the suite's only gradient evidence for it is a `skip`.
- **New: the sink-deletion defect behind one exemption.** With the grad skips cleared, `test_backward_grads_parity` for `gpt_oss` and `deepseek_v4` fails on `missing={'model.layers.0.self_attn.sinks'}`: the family patch replaces sink-aware attention with the llama-shaped shim, dropping `s_aux` and freezing `sinks`. Tracked as its own sub-issue; it must be fixed rather than converted to a permanent xfail.
- **#898** (fused-kernel repeated-backward gap) is tracked as an added sub-issue.

## Completion criteria

- Every registered family either passes forward, gradient and vmap-gradient parity against a genuinely unpatched reference, **or** appears in a single reviewed exemption constant that names the reason and an issue number, is asserted by `test_parity_families_covered`, and is reflected in `docs/user-guide/huggingface/model-patches.md`.
- No parity exemption uses bare `pytest.skip`: each is `pytest.param(..., marks=pytest.mark.xfail(strict=True, reason=...))` so a family that starts passing fails the suite.
- `gpt2` passes gradient and vmap-gradient parity once the harness pins dropout (#813), and stays registered and documented — it is bit-exact, so de-registering it or removing it from the quick-start would be the wrong fix.
- The attention-sink parameter receives a gradient under the patched path for every sink-bearing family, or that family is de-registered.

This is a tracking issue; implementation belongs in its independently closable sub-issues.

Audit source: OPQ-339.
```

### #773 — auditing statistical validity (umbrella)

- **Title →** `Restore auditing statistical validity: bound v_k for truncated ranks`
- **Labels:** unchanged — `source: audit`, `severity: medium`, `pkg: auditing`, `impact: epsilon`, `impact: numerical`.
- **Why:** two of three criteria clauses are met and verified (`_estimate.py:44` now `_MIN_GRID_SIZE = 1_000` with the raise at `:192-194`; `attacks/_helpers.py:159-175` records and checks `coin_flip.dataset_size` and bounds-checks canary indices; the stale "Bonferroni" comment is gone repo-wide). The third is not: `_gdp.py:41`, `:336-338`, `:385-397`, `:88-91` are unchanged and the cap is documented nowhere. Narrow the criteria to the one live clause and add the documentation fallback #828 itself names as the minimum, so the umbrella cannot be closed on a partial.

**Body (replace in full):**

```markdown
## Goal

Make the one-run μ-GDP surface usable at its own documented scale, or honest about where it stops working.

## Scope

Three of the four sub-issues are landed and verified at 12146ec4:

- #829 — `one_run/_estimate.py:44` raises `_MIN_GRID_SIZE` to `1_000` and `:192-194` rejects coarser grids (OPQ-351).
- #830 — `attacks/_helpers.py:159-175` records `CoinFlip.dataset_size`, checks it against `len(dataset)` and bounds-checks the canary indices before building the scoring loader (OPQ-352).
- #831 — the stale "Bonferroni (default)" comment is gone (OPQ-353).

Only **#828** remains, and it is unchanged: `one_run/_gdp.py:41` still pins `_MAX_EXACT_RANKS = 2000`, `:336-338` pins every rank past that at `v_k = 0.5`, and `:385-397` folds `n_trunc * 0.5` into the Chernoff expectation, so no finite μ lifts the p-value to significance once the attack is strong enough. `:88-91` raises rather than returning a bound. The cap is documented nowhere: `docs/user-guide/auditing.md` and `docs/reference/auditing.md` contain no occurrence of `2000`, `truncat` or `_MAX_EXACT`, while the guide's own canary-count table recommends 10^4 and above.

## Completion criteria

- A strong attack at the guide's recommended canary count returns a μ̂/ε estimate rather than raising, via a μ-dependent upper bound on `v_k` for truncated ranks.
- Failing that (the fallback #828 itself names as the minimum): the rank truncation, the attack strength at which the estimator stops converging, and the recommended workaround are documented on both auditing pages, and the raised error points at them.

This is a tracking issue; implementation belongs in #828.

Audit source: OPQ-350, OPQ-351, OPQ-352, OPQ-353.
```

### #774 — CI fail-open gates (umbrella)

- **Title →** `Close CI fail-open gates and restore test integrity (residual: kernel CUDA markers)`
- **Labels:** unchanged — `source: audit`, `severity: medium`, `area: ci`, `impact: test-gap`.
- **Why:** two clauses are met and verified (31ffed65 / #871 ties realized RMS noise to `output.noise_stddev` within six sampling standard errors and asserts a 0.97× perturbation fails, plus `packages/opaque-dpftrl/tests/noise/test_realized_stddev.py`; 039fc012 / #876 added `.github/scripts/assert_cuda_available.sh`, invoked from `python-tests.yml:142-143` and `run_python_test_package.sh:14`, keyed off the marker expression with a real device round-trip). The third — "no blocking lane can pass on zero collected tests" — was **declined**, not missed: #836 and #841 are closed `not_planned`, `allow-empty-test-selection: true` is still on `pr.yml:81, 145, 160, 177`, and `run_python_test_package.sh:43-46` converts exit 5 to exit 0. `python-tests-distributed` is a required input to the blocking `python-tests` gate (`pr.yml:190-203`) whose jq filter (`:207-222`) only excuses `skipped` for `python-tests-cuda*`. The issue must record the accepted decline and its residual risk instead of standing on an unmet bar. It also keeps one open child, #880 (confirmed: `sub_issues_summary` 12 total / 11 completed, and `issue_read get_parent` on #880 returns #774).

**Body (replace in full):**

```markdown
## Goal

Ensure the lanes that are supposed to block a regression can actually fail, and that the suite can detect the defect classes the re-audit found.

## Scope

Landed and verified at 12146ec4:

- #832 — `.github/scripts/assert_cuda_available.sh`, called from `python-tests.yml:142-143` and again from `run_python_test_package.sh:14`; it keys off the marker expression rather than the runner label and reads a value back off the device (OPQ-377).
- #833 — `packages/opaque-dpsgd/tests/noise/test_noise.py` now ties realized RMS noise to `noise_stddev` within six sampling standard errors and asserts a 0.97x perturbation fails; `packages/opaque-dpftrl/tests/noise/test_realized_stddev.py` covers the MF row-norm path (OPQ-379).
- #834, #835, #837, #838, #839, #840, #842 — all landed.

**Declined, with residual risk recorded rather than left implicit:**

- #836 (closed not-planned). `allow-empty-test-selection: true` remains on the distributed lane (`.github/workflows/pr.yml:81`) and the three CUDA lanes (`:145`, `:160`, `:177`), and `run_python_test_package.sh:43-46` turns pytest's exit 5 into exit 0 under it. `python-tests-distributed` is a required input to the blocking `python-tests` gate (`pr.yml:190-203`), whose jq filter (`:207-222`) only excuses `skipped` for `python-tests-cuda*`. **Residual risk:** a marker rename, a collection break, or a deleted `distributed`/`cuda` marker that drops a required lane to zero collected tests still leaves the gate green. The CUDA preflight from #832 catches an unavailable device but not an empty selection on an available one.
- #841 (closed not-planned). No lane exercises Python 3.12, which every wheel advertises. **Residual risk:** a 3.12-only incompatibility ships undetected.

**Open:** #880 — restore CUDA marker coverage for the LoRA and linear-cross-entropy kernel tests.

## Completion criteria

- #880 lands, so the kernel tests it names are selected by the CUDA lanes.
- The two declines above are recorded in the CI documentation (or as a comment next to each `allow-empty-test-selection: true`) so the next reader does not mistake the flag for an oversight; re-opening either is a deliberate decision, not a bug report.

This is a tracking issue; implementation belongs in its independently closable sub-issues.

Audit source: OPQ-377, OPQ-378, OPQ-379, OPQ-380, OPQ-381, OPQ-382, OPQ-383, OPQ-384, OPQ-385, OPQ-386, OPQ-387.
```

### #775 — documentation and citations (umbrella)

- **Title →** `Reconcile citations with their sources (Sweep C) and close the executable-docs gap`
- **Labels (final):** `documentation`, `source: audit`, `severity: medium`, `area: docs`, `area: ci` (**added** — the remaining criterion is a CI gate), `impact: epsilon`.
- **Why:** Sweeps A (#843) and B (#844) are closed, but five of #845's six sites are present verbatim — the manual-pass failure mode this issue's own note predicted. Separately, clause 1 needs correcting: the executable-docs gate that landed (`pr.yml:290-301`, #1034) runs `examples/quickstart.py`, `python -m doctest` on one façade module, and three tutorial notebooks — it does **not** execute the Markdown snippets under `docs/user-guide/` or `docs/reference/`, which is what "every documented snippet executes in CI" meant.

**Body (replace in full):**

```markdown
## Goal

Bring published documentation, reference tables and scholarly citations back into agreement with the code, and make the agreement mechanically checkable so it does not drift again.

## Scope

Sweeps A (#843) and B (#844) are closed. Sweep C is open and, checked against 12146ec4, five of its six findings are present verbatim — the manual-pass failure mode this issue's own note warned about:

- OPQ-371 — `packages/opaque-optimizers/src/opaque/api/optimizers/_schedule_free.py:10`: `Defazio, Yaida, Cutkosky` — "Yaida" is not an author of arXiv:2405.15682 (it reads as a corruption of "Yang"). Sole repo-wide hit.
- OPQ-372a — `alignment/dpo/loss/_discopop.py:6-7` and `:41` attribute DiscoPOP to Azar et al. / NeurIPS 2024; DiscoPOP is Lu et al., arXiv:2406.08414, and Azar et al. is IPO.
- OPQ-372b — `alignment/logprob/_sequence.py:15-16` says the `ld_alpha` split "is not implemented here" while `sequence_logp` accepts (`:37`), documents (`:54-62`) and implements it (`:82-95`); only the fused path lacks it (`:134`).
- OPQ-373 — `amplification/truncated_poisson.rs:18,124` (and `packages/opaque-dpsgd/tests/accounting/test_cross_validation.py:460`) cite the key `[Gan25]`, which is defined nowhere. The dataset-size-dependent adjacency the guarantee assumes is still undocumented.
- OPQ-374 — `docs/mechanisms/dp-ftrl/lambda-cgd.md:62` and `matrix_factorization/lambda_cgd.rs:8,65,130` cite "Theorem 1, eq 15"; eq (15) is a utility quantity and Theorem 1's Toeplitz hypothesis does not cover the default `normalized=True` path (Lemma 8 / eq (12) does).
- OPQ-375 — `dpftrl/sampling/_balls_in_bins.py:26` and `accounting/dpftrl/amplification/_balls_in_bins.py:46` give only arXiv:2412.16802, which contains neither "Definition 3.1" nor "Lemma 3.2"; those are in arXiv:2410.06266.

Done: OPQ-376's validation half — `dpftrl/noise/_bisr.py:660-663` now enforces `0.0 <= momentum < 1.0`. Its documentation half (α pinned to 1) is still open: the note exists at `_bisr.py:235` but not in `docs/mechanisms/dp-ftrl/bisr.md`.

Sweep C is now split into two independently closable children: a bibliographic sweep (citations resolve to the right record) and a hypothesis sweep (the cited result covers the deployed default).

**Correction to the first completion clause.** The executable-docs gate that landed (`.github/workflows/pr.yml:290-301`, #1034) runs `examples/quickstart.py`, `python -m doctest` on `packages/opaque-accounting/src/opaque/accounting/__init__.py`, and three tutorial notebooks. It does **not** execute the Markdown snippets under `docs/user-guide/` or `docs/reference/`, so "every documented snippet executes in CI" is not yet true.

## Completion criteria

- Every citation in source and docs resolves to a record whose stated result covers the use made of it, verified from a **generated** per-file checklist rather than a manual pass.
- The Markdown snippets under `docs/user-guide/` and `docs/reference/` execute in CI, not only the tutorial notebooks and the quickstart script.
- Every published default is generated from, or checked against, its dataclass.

This is a tracking issue; implementation belongs in the two Sweep C children and a follow-up for the Markdown-snippet gate.

Audit source: OPQ-371, OPQ-372, OPQ-373, OPQ-374, OPQ-375, OPQ-376.
```

### #319 — DP-FTRL mechanism and strategy invariants (umbrella)

- **Title:** unchanged.
- **Labels:** unchanged — `source: audit`, `severity: high`, `pkg: accounting`, `pkg: dpftrl`, `impact: epsilon`, `impact: numerical`.
- **Why:** the "5 closed" premise is stale — six of seven children are closed (#347, #353, #355, #360, #361, #362), only #359 is open. The umbrella must stay open because criterion 2 ("accounting charges the deployed sensitivity") is measurably violated by a path **no child covers**: #998 fixed `BandMfStrategy.sensitivity` but left `IdentityStrategy.sensitivity` returning `1.0` (`_identity.py:45-46`), so `mf_gaussian(1.0, identity_strategy(), n_steps=1000).epsilon_at(delta=1e-5)` reports 4.377 against a true 633.93 — a 145× under-report. Closing #359 alone would otherwise close #319 with the criterion unsatisfied.

**Body (replace in full):**

```markdown
## Goal

Restore the sensitivity, correlation, operator, and cache invariants required by DP-FTRL matrix-factorization mechanisms, so that the accountant charges exactly the sensitivity the deployed noise mechanism realizes.

## Scope

Six of the original seven sub-issues are closed: #347 (immutable `PerGroup` state), #353 (Toeplitz Gram against absolute coefficients), #355 (non-quadratic MF row norms), #360 (full BISR operator at runtime), #361 (column-keyed Gaussian reuse), #362 (LR schedules on the MF step axis + weighted Grams).

Remaining scope:

- #359 -- participation sensitivity and the paired second-moment release. The *design* half of #359 was decided and implemented by e44d6ab6 (#998): strategy sensitivity is participation-aware, and a caller that wants the single-participation column norm asks for it with `min_sep=n_steps, max_participations=1`. What remains are two concrete residues, not a design question.
- **Sibling gap not covered by any child:** `IdentityStrategy.sensitivity` (`packages/opaque-dpftrl/src/opaque/api/dpftrl/noise/_identity.py:45-46`) still returns `1.0` for every schema, so bare `mf_gaussian(nm, identity_strategy(), n_steps=N)` charges one participation for an N-participation release. Measured: `n_steps=1000, nm=1.0` reports eps=4.377 at delta=1e-5 against a true 633.93.
- **Paired second-moment release:** `make_second_moment_mf_noise` (`packages/opaque-dpftrl/src/opaque/api/dpftrl/noise/_second_moment.py:140-150`) calibrates the joint Mahalanobis budget from single-participation `c1`/`c2`, while `MfGaussian.pld` (`packages/opaque-dpftrl/src/opaque/api/accounting/dpftrl/mechanisms/_mf_gaussian.py:132-140`) charges the schema sensitivity of the *first* strategy only. Nothing ties `second_moment_strategy` to an accountant.

## Completion criteria

- The deployed noise covariance matches the specified matrix mechanism.
- Accounting charges the deployed sensitivity for **every** strategy, including `IdentityStrategy`, or the unsupported context is rejected rather than silently priced at one participation.
- The paired first/second-moment release is either tied to an accountant that sees both strategies, or removed (see #407).
- Strategy identity covers every privacy-material input.

This is a tracking issue; implementation belongs in its independently closable sub-issues.

Audit source: OPQ-177 (AUD-084, AUD-186); September reconsideration added the `IdentityStrategy` sibling gap.
```

### #407 — JME second moments for DP-SGD

- **Title →** `Implement proper JME second moments for DP-SGD (and stop attributing the current statistic to JME)`
- **Labels (final):** `bug`, `source: audit`, `needs-design` (kept — scope item 2 needs a real derivation), `severity: medium`, `pkg: dpsgd`, `pkg: optimizers`, `impact: numerical`, `area: docs` (**added** — scope item 1 is landable documentation work).
- **Why:** still fully real and reproduced with the issue's own acceptance test (two records `[g, -g]`, `clipping_norm=1.0`, `normalize_by=2.0`: aggregate `x_t = 0.0`, clean statistic `0.0`, released "private second moment" `1.0`), with the squaring site at `_clipped_fun.py:534-541` and the per-record bound at `:389-394`. The assumed shortcut does not exist: DP-FTRL's private second-moment work supplies plumbing only — a repo-wide grep for `2502.06597` returns three citation sites and no implementation, and no aggregate projection to a public radius exists anywhere. The misattribution is live in three user-facing surfaces (`_adam.py:26-30`, `opaque/dpftrl/noise/__init__.py:26`, `examples/train_dpftrl.py:99`), and that half is landable today — it should not sit behind the derivation.

**Body (replace in full):**

```markdown
## Problem

Opaque's `second_moment=True` path releases the normalized aggregate `x_t = sum_i clip(g_{t,i}, C) / z` and `q_t = sum_i clip(g_{t,i}, C)^2 / z` -- a **mean of per-record squares**. That is neither the clean Adam statistic `x_t * x_t` nor the JME query of Kalinin, Upadhyay and Lampert ([arXiv:2502.06597](https://arxiv.org/abs/2502.06597)).

Reproduced on the current tree (two records with gradients `[g, -g]`, `clipping_norm=1.0`, `normalize_by=2.0`, `second_moment=True`):

```
aggregate x_t                 = tensor([0.])
clean Adam statistic x_t*x_t  = tensor([0.])
released 'second moment'      = tensor([1.0000])
```

The squaring site is `packages/opaque-engine/src/opaque/api/engine/clipping/_clipped_fun.py:534-541` (`clipped_value.square()` **inside** the per-example function, before reduction), with the matching per-record sensitivity `squared_bound = (current * current) / normalize_by` at `_clipped_fun.py:389-394`.

**The defect now spans three packages and is misattributed in user-facing docs:**

- `opaque-engine` produces the statistic (`_clipped_fun.py:534-541`).
- `opaque-dpftrl` re-noises the same statistic through a paired MF release (`.../dpftrl/noise/_second_moment.py`) and cites the JME paper at `packages/opaque-dpftrl/src/opaque/dpftrl/noise/__init__.py:26`.
- `opaque-optimizers` consumes it and describes it as 'a privately-estimated `g^2`' citing arXiv:2502.06597 at `packages/opaque-optimizers/src/opaque/api/optimizers/_adam.py:26-30`; `examples/train_dpftrl.py:99` repeats the citation.

**opaque-dpftrl does not shortcut this work.** Its paired-stream machinery (`opaque.api.engine.noise_allocation.paired_noise_stddevs`, the two domain-separated RNG streams, the `SecondMomentNoiseOutput` handoff) is reusable plumbing, but it supplies none of the JME content: no aggregate projection to a public L2 radius `R`, no square-of-aggregate query, no `lambda`, and no constrained add/remove sensitivity. A repo-wide grep for `2502.06597` returns citation sites only.

## Impact

Mean-of-per-record-squares upper-bounds the square of the aggregate by Jensen, so Adam's `v_t` is systematically inflated and the effective step size systematically shrunk -- a utility defect, not a privacy one (the released quantity is correctly clipped and noised for its own sensitivity). The *documentation* defect is separate and immediate: three user-facing surfaces attribute a statistic to a paper that defines a different one.

## Revised scope

1. **Landable now, independently:** correct the attribution. State in `_adam.py`, `opaque.dpftrl.noise.__init__`, `examples/train_dpftrl.py` and the optimizer docs exactly which statistic is released (`sum_i clip(g_i)^2 / z`), that it is not the JME query, and what its bias direction is.
2. **Research + implementation, DP-SGD only:** clip per-record gradients, form the normalized aggregate, project it to a fixed public L2 radius `R`, form `x_t * x_t` after projection, and release the pair with the JME lambda-weighted joint Gaussian mechanism.
3. Remove `second_moment_strategy` and the paired private-second-moment claims from DP-FTRL/MF APIs, accounting docs, examples and tests (this also retires acceptance criterion 2 of #359).

## Privacy derivation required

The paper's theorem uses replace-one neighboring streams. Opaque uses record-level add/remove adjacency. Derive the exact diagonal-query sensitivity for `sup ||x-y||^2 + lambda * ||x*x - y*y||^2` subject to `||x|| <= R`, `||y|| <= R`, `||x-y|| <= Delta = C / z`; choose `lambda` and the two noise scales from that constrained sensitivity; verify the closed form against numerical maximization; and confirm the whitened per-step mechanism is dominated by the same Gaussian PLD used by `dpsgd_acc.gaussian(noise_multiplier)`.

## API and architecture

- JME construction lives in `opaque-dpsgd`, not the algorithm-neutral engine.
- Replace `second_moment=True` with JME aggregate-square semantics through a DP-SGD-owned API that requires an explicit aggregate bound.
- Keep `SecondMomentNoiseOutput` as the optimizer handoff if its names and metadata stay accurate.
- Scalar global clipping only at first; reject `PerGroup` until a group-coupled JME sensitivity is derived.

## Acceptance criteria

- Scope item 1 has landed and is not blocked on the derivation.
- Public docs state the exact clean statistics, normalization, aggregate projection, adjacency model, `lambda`, and realized noise scales.
- A checked derivation and a numerical optimizer agree on the constrained add/remove joint sensitivity.
- Toy batches distinguish square-of-aggregate from mean-of-squares: `[g, -g]` must produce a zero clean JME square (today it produces 1.0).
- Zero-noise JME updates match ordinary Adam updates for the same bounded aggregate.
- Statistical tests cover both Gaussian streams and their domain-separated RNG state.
- Poisson-sampled accounting is cross-validated against the derived whitened Gaussian mechanism.
- Unsupported DP-FTRL, MF, repeated-input-row and per-group configurations fail explicitly rather than silently using the old allocator.

Audit source: AUD-093; re-verified September 2026 against commit 12146ec4.
```

### #417 — benchmark harness

- **Title →** `Measure and regression-gate the kernel and patch performance paths`
- **Labels (final):** `enhancement`, `bug` (**added** — criterion 1 fixes a live fail-open assertion), `source: audit`, `severity: medium` (**raised from low**), `area: ci`, `area: docs`, `pkg: patches` (**added**), `impact: performance`, `impact: test-gap` (**added**).
- **Why:** nothing satisfying it exists (15 workflow files, none perf-related; no `benchmarks/` directory anywhere), but its *stated trigger* is inert — a regex sweep of `docs/`, `README.md` and the package READMEs for numeric speed/memory claims returns zero hits, and all 20 `perf(...)` commits in the delta have empty bodies, so nothing numeric reaches `git-cliff` either. Meanwhile the exposure inverted: 20 perf commits landed, the #956–970 series proposes more, and the only in-repo measurement infrastructure — `_measure_time_and_memory` / `_assert_perf_benefit` at `packages/opaque-patches/tests/kernels/conftest.py:130-178`, which the issue does not mention — is CUDA-only and fail-open (`has_benefit = speedup > (1.0 - max_perf_overhead) or mem_reduction > 1.0`, default `max_perf_overhead=0.20`, so a kernel 20% slower at equal memory asserts "benefit"). So the perf series does not supersede #417; it raises its priority for a different reason than the issue states.
- **Also flag (outside these 24):** #511 covers overlapping ground and is still open. Consolidate into #417 and close #511 as a duplicate, or #966's blocked-on note will point at two issues for one missing artifact.

**Body (replace in full):**

```markdown
## Problem

The original framing -- a harness to substantiate precise published numbers -- has no work to do: the audit removed every numeric throughput/memory/overhead claim and none has been restored. A regex sweep of `docs/`, `README.md` and the package READMEs for numeric speed or memory claims returns zero hits, and all 20 `perf(...)` commits in `4b13d82..12146ec4` have empty bodies, so nothing numeric reaches `git-cliff` or the release notes.

The live exposure is the opposite one. Twenty performance commits landed in this delta (fused CE, RMSNorm, RoPE, GQA K/V, LoRA QKV fusion, evaluation transfer overlap, compiled-graph reuse) and the #956-970 perf-audit series proposes more, with **no measurement artifact on any lane a PR actually runs**. Nothing in CI can distinguish a speedup from a regression.

The only in-repo measurement infrastructure is undiscoverable from this issue and fail-open:

```python
# packages/opaque-patches/tests/kernels/conftest.py:174
has_benefit = speedup > (1.0 - max_perf_overhead) or mem_reduction > 1.0
```

With the default `max_perf_overhead=0.20`, a kernel that is 20% **slower** at identical memory asserts 'benefit'. `_measure_time_and_memory` (`conftest.py:130-160`) records no hardware, torch/triton version, commit or configuration, emits no artifact, and every consumer is CUDA-marked -- so on the PR CUDA lane it produces console prose and on every other lane it does not run at all.

## Impact

Performance work is landing unmeasured. A change that regresses throughput by up to 20% passes the only gate that looks at throughput; a change that regresses it by more fails with a one-line message and no recorded baseline to compare against. When the #956-970 series lands there will be no way to say whether it helped.

## Acceptance criteria

- `_assert_perf_benefit` stops accepting a slowdown: separate the speed and memory assertions, or require an explicit per-case expectation, so a regression cannot pass by satisfying the other disjunct.
- A repository-owned harness records hardware, software versions, configuration, commit and command for every measured case, and writes committed result data rather than console output.
- At least the kernel and patch paths touched by the 20 delta perf commits and by #956-970 have a harness case each.
- A scheduled workflow (not the PR gate) runs the harness and detects regressions against the committed baseline.
- No numeric performance claim is restored to `docs/` or a release body without a harness case behind it.

Source: remediation issues J/W / gate G10; re-scoped September 2026 after 20 unmeasured perf commits.
```

### #911 — Monte Carlo calibration cost

- **Title →** `Make Monte Carlo accounting affordable for calibration: reuse sigma-independent draws and unblock the dead b-min-sep corpus budget`
- **Labels (final):** the issue currently has **no labels at all** — set `bug`, `source: audit`, `severity: medium`, `pkg: accounting`, `pkg: dpftrl`, `impact: performance`.
- **Keep the number.** PR #1033 is linked to it.
- **Why:** the "may be" is settled by measurement (defaults resolve to 2,940,252 samples per adjacency direction; one BnB+BLT `epsilon_at` takes 4.15 s / 22.65 s / 127.99 s at 25/100, 100/1000, 250/2500 bins/steps; a *toy* 25-bin/100-step `calibrate` takes 303.8 s over 52 process evaluations at the default tolerance, projecting to ~1 h 50 m at 250/2500). Two causes are live, not one: the BnB redraw (#1033's subject) **and** the b-min-sep reuse cache that is inert at defaults (`_transcript_cache.py:13-20` sizes `3 * num_mc_samples * n_steps * 8` against a 4 GiB cap → only `n_steps <= 60`; `:91-94` swallows the `ValueError` and `_b_min_sep/__init__.py:196-203` silently falls back per probe). Two amplifiers the issue never mentions: `calibration.py:303` divides `mc_failure_probability` by `max_iterations + 2` for every probe regardless of how many run, and the `if not runtime_safe:` fallback at `:331` re-runs the entire search (`invocations=2` on an ordinary default call). d0257cbf (#913) does not address any of it. Closing on #1033 alone would leave the b-min-sep cliff in place — hence the explicit final criterion.

**Body (replace in full):**

```markdown
## Problem

Measured on 12146ec4 (4 vCPU, torch 2.14):

- Default MC settings (`mc_resolution=1e-5`, `mc_failure_probability=1e-6`, `packages/opaque-accounting/src/opaque/api/accounting/core/discretization.py:37-39`) resolve to **2,940,252 samples per adjacency direction**.
- One `balls_in_bins(mf_gaussian(nm, blt_strategy(momentum=0.95)), ...).epsilon_at(1e-5)` at defaults: **4.15 s** at 25 bins/100 steps, **22.65 s** at 100/1000, **127.99 s** at 250/2500.
- `calibrate(epsilon_budget(8.0, delta=1e-5), proc, 0.5, 8.0)` on the *toy* 25-bin/100-step process: **123.2 s / 10 iterations / 25 process evaluations** at `tolerance=1e-2`, and **303.8 s / 22 iterations / 52 process evaluations** at the default `tolerance=1e-6`. Projecting those 52 evaluations onto the 250-bin/2500-step per-call cost gives **~1 h 50 m** for a horizon that is still small by DP-FTRL standards.
- Root cause 1 - Balls-in-Bins redraws the sigma-independent transcripts for every noise-multiplier probe. PR #1033 measures 40.35 s -> 6.92 s (5.8x) for a 512-bin/2048-step lambda-CGD calibration at the *loose* `mc_resolution=1e-3`; at the default 1e-5 the absolute numbers are ~3 orders of magnitude more samples.
- Root cause 2 - b-min-sep already has the corresponding reuse cache, and it is **inert at default settings**. `_b_min_sep/_transcript_cache.py:13-20` sizes a corpus at `3 * num_mc_samples * n_steps * 8` bytes against a 4 GiB default cap (`OPAQUE_B_MIN_SEP_TRANSCRIPT_CACHE_MAX_BYTES`). At 2,940,252 samples that admits only `n_steps <= 60` (6.6 GiB already at `n_steps=100`, 164 GiB at 2500). Above that, `with_handle` raises, `_transcript_cache.py:91-94` swallows the `ValueError`, and `_b_min_sep/__init__.py:196-203` silently falls back to a fresh full-horizon `bandmf_b_min_sep_warm_mc_pld` per probe. Nothing warns. As a lower bound on what that costs: a single `b_min_sep(mf_gaussian(1.0, band_mf_strategy(bands=8)), n_steps=100, p0=0.001).epsilon_at(1e-5)` at default MC settings had not returned after 15 minutes on 4 vCPU - and `n_steps=100` is already past the corpus limit, so a calibration would repeat that per probe.
- Amplifier 1 - `calibration.py:303` sets `probe_count = max_iterations + 2` and divides `mc_failure_probability` by it for *every* probe, regardless of how many probes the search actually performs. At defaults that raises each probe from 2,940,252 to 3,417,797 samples even when the search converges in 22 iterations.
- Amplifier 2 - the `if not runtime_safe:` fallback at `calibration.py:331` **runs the entire binary search a second time**. Instrumenting `_calibrate_impl` on the run above gives `invocations=2` for a plain default-tolerance calibration, so the measured 303.8 s is already a doubled cost, not a worst case. (That fallback also under-reports its Monte Carlo confidence; see the separate finding.)
- `d0257cbf` (#913) does not address any of this: it only hoists `WarmLogLikelihoodRatio::new` out of the per-sample closures in `amplification/b_min_sep/mc.rs`, and touches no Balls-in-Bins code.

## Impact

Calibrating a realistic DP-FTRL horizon under a correlated strategy (BLT / BSR / BISR / lambda-CGD) or under b-min-sep is an hours-long offline job at library defaults. The one reuse mechanism that already exists is disabled by its own byte budget at any horizon a user would actually train, and the fallback is silent, so a user cannot tell whether they are on the fast or the slow path.

## Acceptance criteria

- Sigma-independent draws are reused across calibration probes for Balls-in-Bins with correlated strategies, with PLDs bit-identical to the one-shot path across several sigma values (PR #1033).
- The b-min-sep corpus budget is reconciled with the default sample count: either the corpus is stored compactly enough (or streamed/sharded) to serve realistic horizons, or the fallback emits a `RuntimeWarning` naming the byte estimate and the cap so the cliff is observable rather than silent.
- The per-probe `mc_failure_probability` split reflects the probes actually performed, or the worst-case union bound is documented together with its sample-count cost in `docs/user-guide/accounting.md`.
- The `runtime_safe` fallback at `calibration.py:331` no longer routinely doubles the search for Monte Carlo processes (e.g. by comparing against the runtime estimate with a tolerance-aware margin instead of an exact `<=` at the converged point).
- A recorded benchmark (hardware, commit, config, raw samples) for `calibrate` on at least one correlated-strategy Balls-in-Bins process **and** one b-min-sep process at default MC settings, before and after.
- Any reuse of draws across probes must not change the reported privacy guarantee: assert identical epsilon/delta output versus the non-cached path, not merely 'close'.
- This issue is not satisfied by #1033 alone; the b-min-sep path must be covered before closing.

Audit source: team-filed (Bazalt performance review); re-scoped with measurements by the 2026-09 reconsideration pass.
```

### #958 — fused SDPA under vmap

- **Title →** `Batch fused SDPA backward across the DP vmap dimension (forward rules already exist)`
- **Labels (final):** `enhancement`, `source: audit`, `needs-validation`, `pkg: patches`, `pkg: transformers`, `impact: performance`, `impact: numerical` (**added**), `impact: epsilon` (**added** — a wrong batching rule breaks per-example clipping and therefore the sensitivity the accountant assumes).
- **Why:** measured — in torch 2.14 `FuncTorchBatched` is True for all three fused forward ops and False for all three `_backward` ops, and a minimal `vmap(grad(...))` emits exactly two fallback warnings. PR #993 (661 additions) was closed unmerged on 2026-09-09 with nothing landed. Two corrections: "flash attention is not selected" reads as an accident but is deliberate (`attention.py:116-130` `_can_use_native_gqa` returns False for any batched tensor), and since b9fb8674 (#985) `_grouped_sdpa` issues one SDPA call **per key/value head**, multiplying the per-example backward fallback by `num_key_value_heads` — a factor the issue does not account for. The issue also lacks a hard correctness criterion, which matters most here.

**Body (replace in full):**

```markdown
## Problem

In the pinned PyTorch (2.14), the fused SDPA **forward** operators already have functorch batching rules, but their **backward** operators do not:

```
_scaled_dot_product_efficient_attention          FuncTorchBatched: True
_scaled_dot_product_efficient_attention_backward FuncTorchBatched: False
_scaled_dot_product_cudnn_attention              True  / _backward False
_scaled_dot_product_flash_attention              True  / _backward False
```

So under `vmap(grad(...))` the forward runs batched and the backward falls back to one backend dispatch per DP example per layer. A minimal `vmap(grad(...))` over `F.scaled_dot_product_attention` on CPU emits exactly two fallback warnings, one of them for `_scaled_dot_product_flash_attention_for_cpu_backward`.

Two facts change the shape of the problem relative to the original write-up:

1. Flash is not 'not selected' by accident. `packages/opaque-patches/src/opaque/api/patches/transformers/components/attention.py:116-130` (`_can_use_native_gqa`) returns `False` for any batched tensor, so under the DP vmap the native `enable_gqa` flash path is disabled deliberately and `_grouped_sdpa` runs instead.
2. Since `b9fb8674` (#985, closing child #960) replaced physical K/V replication with a per-KV-head loop, `_grouped_sdpa` (`attention.py:75-110`, loop at `:97`) issues **one SDPA call per key/value head** whenever `num_key_value_groups > 1`. For every GQA model the per-example backward fallback is therefore multiplied by `num_key_value_heads`, on top of the microbatch factor.

PR #993 attempted this and was closed unmerged on 2026-09-09; no part of it is in `main`.

## Impact

Attention backward under DP costs `microbatch x num_key_value_heads` backend dispatches per layer instead of one, which is launch-bound at short and medium sequences and prevents the backend from ever seeing the full physical batch.

## Acceptance criteria

- Capability-gated batching rules for `_scaled_dot_product_{efficient,cudnn,flash}_attention_backward` that merge the DP vmap dimension with the logical model batch, installed only when the native rule is absent and retiring themselves when PyTorch ships one.
- **Correctness gate (blocking):** per-example gradient isolation is asserted, not assumed. A test must compare per-sample gradients from the batched rule against a reference computed one example at a time, for MHA and GQA, causal and padded masks, and with at least one example whose inputs differ only in one position - a batching rule that leaks across the vmap dimension silently corrupts per-example clipping and therefore the sensitivity the accountant assumes.
- Dropout randomness semantics are preserved: either dropout is refused on the batched path or the Philox stream is shown to stay per-example.
- The GQA multiplier is addressed or explicitly scoped out: state whether `_grouped_sdpa`'s per-KV-head loop stays, and if so, measure the combined `chunks x kv_heads x microbatch` dispatch count.
- Benchmarks record the backend that actually executed (profiler-observed), not the configured one, across MATH/efficient/cuDNN/flash, microbatch {1,2,4,8}, sequence {512,2048,8192}.

Audit source: team-filed (Bazalt performance review); corrected by the 2026-09 reconsideration pass.
```

### #963 — compact sliding-window attention

- **Title →** `Tune compact sliding-window attention: chunk size, mask reuse, padded batches, and the per-KV-head SDPA multiplier`
- **Labels (final):** `enhancement`, `source: audit`, `needs-validation`, `pkg: patches`, `pkg: transformers`, `impact: performance`, `impact: numerical` (**added**).
- **Why:** every cited defect is still in the tree (`_GEMMA2_QUERY_CHUNK = 64` / `_SDPA_QUERY_CHUNK = 64` at `attention.py:9-10`; fixed loops at `:255-256`, `:514-515`, `:584-585`; masks rebuilt per chunk at `:146-162` called from `:258`; `runtime/masking.py:95` still gates on `attention_mask is None`), and #985 made one of them worse: measured 32 Python-level SDPA calls for a single layer at S=1024, window 256, 8 query / 2 KV heads — `ceil(1024/64) × 2`, i.e. 1024 per layer at S=8192 with 8 KV heads, **8× the 128 the issue states** — each fanning out again per DP example in backward. PR #991 (2055 additions) closed unmerged.

**Body (replace in full):**

```markdown
## Problem

All of the original findings are still present in `main`, and one of them got worse.

- Fixed chunk constants: `packages/opaque-patches/src/opaque/api/patches/transformers/components/attention.py:9` (`_GEMMA2_QUERY_CHUNK = 64`) and `:10` (`_SDPA_QUERY_CHUNK = 64`), driving the Python loops at `:255-256` (compact SDPA), `:514-515` and `:584-585` (Gemma2 softcap forward/backward).
- Masks are rebuilt for every chunk: `_chunked_sliding_window_mask` (`attention.py:146-162`) is called inside the loop at `:258` and repeats `torch.ones(...).tril_().triu_()` for every steady-state chunk.
- Padded batches are still excluded: `packages/opaque-patches/src/opaque/api/patches/transformers/runtime/masking.py:95` gates the compact route on `attention_mask is None`, so an ordinary variable-length SFT batch falls back to the dense-mask route.
- **New since the issue was filed:** `b9fb8674` (#985, which closed child #960) replaced physical K/V replication with a per-KV-head loop. Each chunk now calls `_grouped_sdpa` (`attention.py:75-110`), which loops over `key.shape[-3]` at `:97`; Gemma2's chunked path has the same loop at `:517` and `:587`. Measured on the current tree, `vmap(grad(...))` through `vmap_sdpa_attention_forward_sliding_window` at sequence 1024, window 256, 8 query / 2 KV heads issues **32** Python-level SDPA calls for one layer (`ceil(1024/64) x 2`). At sequence 8192 with 8 KV heads that is **1024 per layer**, eight times the 128 the original issue assumed - and under `vmap(grad)` each of those fans out again per DP example in backward (#958). Before #985 the same loop issued one SDPA call per chunk.

Mistral and Ministral route here (`models/mistral.py:32`, `models/ministral.py:32`). PR #991 attempted this and was closed unmerged on 2026-09-09.

## Impact

Long-sequence sliding-window training is launch-bound, the cost now scales with `ceil(seq/64) x num_key_value_heads x microbatch`, and padded SFT batches - the common case - do not get the compact path at all.

## Acceptance criteria

- Chunk size is chosen from sequence, window, head geometry, dtype, backend and an explicit workspace budget, with the chosen value and the resulting SDPA/mask kernel counts observable in benchmark output.
- Steady-state chunk masks are reused rather than rebuilt per chunk.
- The compact route accepts left/right-padded batches, with the padding folded into the chunk-local band, and `runtime/masking.py`'s `attention_mask is None` gate relaxed accordingly.
- The per-KV-head loop is addressed explicitly: either batch the KV groups back into one call on paths where it is safe, or record the combined `chunks x kv_heads x microbatch` dispatch count and justify keeping it.
- Output and Q/K/V gradient parity against the dense-mask path for fully masked rows, causal offsets, cache decoding, Gemma2 softcap and dropout-zero training; **and** per-example gradient parity under `vmap(grad)` against a one-example-at-a-time reference, since this path sits on the per-sample gradient used for clipping.
- Peak memory must not regress relative to the current bounded path (#925/#954).

Audit source: team-filed (Bazalt performance review); corrected by the 2026-09 reconsideration pass.
```

### #964 — MoE route plans and streamed weight gradients

- **Title →** `Reuse MoE route plans and redesign streamed weight gradients (preserving per-example virtual groups)`
- **Labels (final):** `enhancement`, `source: audit`, `needs-validation`, `pkg: patches`, `impact: performance`, `impact: numerical` (**added**), `impact: epsilon` (**added**).
- **Why:** every mechanism is still in the tree (`_grouped_AtB` at `_grouped_moe.py:59-71` with `bounds = seg_offs.tolist()` at `:64` and one Python matmul per virtual group at `:65-70`; route orderings rebuilt in forward `:87-115`, backward `:346-371` and a third pass at `:377-384`; unconditional synchronize at `:394-399`), and the `B × E` figure is exact (`:526-527`). The correction the issue needs is a **correctness criterion**: per-example weight-gradient isolation rests entirely on `_route_groups` (`:117-124`) mapping each route to `samples * E + experts`. A route plan keyed only on expert id would merge per-example weight gradients and silently destroy per-example clipping — the original text mentions per-example gradients only as one of a dozen metrics to *record*.

**Body (replace in full):**

```markdown
## Problem

Still present in `main`, unchanged by any delta commit:

- `packages/opaque-patches/src/opaque/api/patches/kernels/_grouped_moe.py:59-71` (`_grouped_AtB`) copies segment offsets to the host (`bounds = seg_offs.tolist()`, `:64`) and runs **one Python matmul per virtual group** (`:65-70`), each with a full fp32 conversion. Under DP vmap there are `B x E` virtual groups (`:526-527` passes `n_groups=current_batch * E`, `tokens_per_sample=T`), so two expert matrices need up to `2 x B x E` tiny matmuls - 4,096 at `B=32, E=64`.
- Forward rebuilds the route ordering per chunk (`_route_sort` at `:87-115`), backward rebuilds it again (`:346-371`), and the weight-gradient path performs a third ordering pass (`_route_groups` + `argsort` + `_seg_offsets`, `:377-384`).
- The streamed fallback still synchronizes unconditionally (`torch.cuda.synchronize` / `torch.mps.synchronize`, `:394-399`) and re-derives the workspace budget inside the scan, giving `O(groups x routes x tiles)` effective route work.

## Impact

Trainable-expert MoE on CPU/MPS, and any CUDA run whose accumulators do not fit the workspace budget, hits a throughput cliff dominated by Python dispatch and host synchronization rather than arithmetic.

## Acceptance criteria

- One bounded on-device route plan per operation, reused across forward, backward and both weight matrices; ordinary backward reuses the already expert-sorted order instead of re-sorting.
- Grouped `A^T B` without a host `.tolist()` and without one Python matmul per virtual group.
- The workspace budget is computed once per operation; unconditional device synchronization is replaced by event/stream-scoped lifetime management.
- **Correctness gate (blocking):** the route plan must keep the per-example virtual-group mapping `samples * E + experts` from `_route_groups` (`_grouped_moe.py:117-124`) exactly. Add a test that, for `B >= 2` under `vmap(grad)`, compares each example's expert weight gradients against a single-example reference and fails if any example's gradient is contaminated by another's. A plan keyed only on expert id would silently merge per-example weight gradients, destroying per-example clipping and the sensitivity bound the accountant assumes.
- Existing invariants preserved: deterministic ordering, duplicate top-k handling, empty-group zeros, and the hard workspace bound (no increase in peak memory versus #948).
- Benchmarks report route/sort time, host synchronization count, Python and device launch counts, and forward/backward tokens/s for `E={8,16,64}`, `K={1,2,4}`, frozen and trainable experts, balanced/skewed/empty groups.

Audit source: team-filed (Bazalt performance review); corrected by the 2026-09 reconsideration pass.
```

### #966 — MoE backend dispatch

- **Title →** `Replace the universal E>=16 MoE speed gate with per-backend thresholds and a selected-route diagnostic`
- **Labels (final):** `enhancement`, `source: audit`, `needs-validation`, `pkg: patches`, `impact: performance`, `impact: numerical` (**added**). Do **not** add `blocked` — the re-scope removes the blocked half from scope; the blocked-on note lives in the body.
- **Why:** the premise is intact (`moe.py:575` `_SPARSE_MOE_MIN_EXPERTS = 16`, consumed at `:627`, decision reduces to `return experts >= min_experts and grouped_fits` at `_moe_memory.py:134`; CUDA bf16/fp16 with Triton selects `Opaque_FusedMoE` unconditionally at `:606-613`; `estimate_moe_workspace` is unchunked). But as written the issue **cannot start**: it asks for a cost model validated with the #511/#417 evidence framework, and no benchmark harness exists anywhere while both #511 and #417 are open. Its contradicting M5 Max evidence was taken on Torch 2.10 from an unmerged branch, against a grouped path #948 has since changed, so it must be re-measured before it can justify anything.

**Body (replace in full):**

```markdown
## Problem

The non-CUDA MoE dispatcher still decides on expert count alone. `packages/opaque-patches/src/opaque/api/patches/kernels/moe.py:575` defines `_SPARSE_MOE_MIN_EXPERTS = 16`, passed at `:627` into `use_grouped_route`, whose decision reduces to `return experts >= min_experts and grouped_fits` (`packages/opaque-patches/src/opaque/api/patches/kernels/_moe_memory.py:134`). Tokens/routes per expert, `K`, `H`, `I`, dtype, trainability, route balance and the achievable bounded chunk size play no part; `estimate_moe_workspace` (`_moe_memory.py:57-118`) is unchunked, so even the memory term is not the one the kernel will run with.

Conversely, CUDA bf16/fp16 with Triton available selects `Opaque_FusedMoE` **unconditionally** (`moe.py:606-613`), with no lower bound, including token counts where routing overhead dominates.

The evidence originally cited against the universal threshold (grouped 45-66% slower at `E=16` on Apple M5 Max) came from an unmerged branch on Torch 2.10. The tree now pins Torch 2.14 and #948 changed the grouped path, so that evidence establishes only that *expert count alone is the wrong model* - the crossover itself must be re-measured.

## Impact

On at least one supported backend the default dispatch is known to pick the slower kernel, and on CUDA small MoE workloads always pay Triton routing overhead. Users have no way to see which route ran.

## Scope correction

The original proposal (a full cost model validated with the #511/#417 evidence framework) cannot start today: there is no benchmark harness anywhere in the repository and both #511 and #417 are open. This issue is narrowed to the part that stands alone; the full cost model should be re-raised against #956 once an evidence harness exists.

## Acceptance criteria

- Separate CPU, MPS and CUDA dispatch policies replace the single `_SPARSE_MOE_MIN_EXPERTS` constant, each backed by a recorded measurement on current code (Torch 2.14, post-#948) rather than an inherited constant.
- The CUDA Triton branch gains a lower bound on routed rows so tiny workloads fall back to the dense path.
- The selected route is observable (a diagnostic hook or returned metadata), so a benchmark can attribute a regression to dispatch.
- Dispatch changes must not alter numerical or per-example-gradient semantics: assert grouped, dense and Triton routes agree within the documented dtype floor **and** produce identical per-example gradients under `vmap(grad)` for `B >= 2`.
- Blocked-on note: the full geometry cost model depends on #417/#511 and stays out of scope until an evidence harness lands.

Audit source: team-filed (Bazalt performance review); re-scoped by the 2026-09 reconsideration pass.
```

### #968 — LoRA GEMM packing

- **Title →** `Pack frozen LoRA QKV/MLP base projections into fewer GEMMs, only if it pays for its persistent memory`
- **Labels (final):** `enhancement`, `source: audit`, `needs-validation`, `blocked` (**added** — its kill measurement needs #417's harness), `pkg: patches`, `impact: performance`, `impact: numerical` (**added**).
- **Why:** the factual premise holds (three separate base GEMMs at `lora.py:517, 521, 525`, repeated in the vmap path at `:633, 637, 641`; two same-input MLP GEMMs at `:1084, 1088`, `:1140/1143`, `:1254/1258`; a grep for `pack`/`torch.cat` in that file returns **0** hits). The delta commits people assume closed it did not (#947/#949/#950/#951 extend fused-QKV *coverage*; #934 reduces backward memory; #977 preserves mixed precision), and PR #992 closed unmerged. What the issue never priced is the cost: persistent packed base weights are a permanent extra device allocation of the full Q/K/V (and gate/up) bytes per layer, in the package whose umbrella exists to *reduce* memory, and directly against 97a72920 (#933), which removed retained weight replicas — re-materialised per compute dtype under autocast.
- **Decision note — this is the one item in the 24 I would accept closing.** The defect is real but the *value* is unmeasured, PR #992 already failed once, and the fix contradicts landed memory work. Keep it only with the kill criterion and the `blocked` label below; if the maintainer prefers a shorter open list, close it `not_planned` with: *"No measurement supports this and the packed-weight cost contradicts #933. Re-file when #417's harness can show a step-time win in the DP regime net of persistent memory."*

**Body (replace in full):**

```markdown
## Problem

Fused LoRA QKV still launches three independent base GEMMs - `packages/opaque-patches/src/opaque/api/patches/kernels/lora.py:517` (`Q = F.linear(X, Wq, bq)`), `:521`, `:525`, mirrored in the vmap path at `:633`, `:637`, `:641` - plus up to six adapter GEMMs. Fused MLP performs two same-input base GEMMs (`:1084`, `:1088`; repeated at `:1140`/`:1143` and `:1254`/`:1258`). There is no packing anywhere in the file. Existing 'fusion' reduces Python/autograd overhead and shares staging; it is not packed-GEMM fusion.

None of the LoRA delta work closed this: `#947`/`#949`/`#950`/`#951` extend fused-QKV *family coverage*, `#934` reduces backward memory, `#977` preserves mixed-precision behaviour. PR #992 attempted the packing and was closed unmerged on 2026-09-09.

## Impact

At short sequences, decode-like calls and small DP microbatches - the regime DP training is often forced into, because per-sample gradients cap the physical microbatch - individual projection GEMMs are underfilled and the extra launches show up in step time.

## Cost that the original proposal did not price

Packing frozen base weights at setup means **permanently retaining a second copy** of the Q/K/V (and gate/up) weight bytes per layer, in the package whose umbrella (#956) exists to reduce memory, and in direct tension with `97a72920` (#933), which removed retained autocast weight replicas precisely because they blew the memory budget. Under autocast the packed copy must additionally be re-materialised per compute dtype. This makes the item conditional, not unconditional.

## Acceptance criteria

- Restricted to **frozen** base weights: pack Q/K/V (and gate/up) once at patch/model setup, use one wider GEMM plus output views, and concatenate upstream gradients for a single base `dX` GEMM. Packing of low-rank adapter projections is out of scope unless measured separately.
- A hard memory budget: net persistent device memory increase is reported, and the change is rejected if it exceeds an explicitly stated per-layer cap. Packed storage is versioned and invalidated on device, dtype, `state_dict` load, and any weight mutation; tied/shared storage is covered by a test.
- **Correctness gate (blocking):** per-example gradient parity under `vmap(grad)` for `B >= 2` against a one-example-at-a-time reference, plus preservation of unequal GQA widths, missing adapters, active-adapter selection, adapter-dropout fallback, and `state_dict` round-trip.
- Dispatch retains the unpacked path when measurements favour it (long sequences, saturated GEMMs).
- **Kill criterion:** if the first measurement on the DP regime (microbatch 1-4, sequence 512-4096, rank 8-64) does not show a clear step-time win net of the memory cost, close this issue rather than extending it to more families. This is a medium-value item; it should not be carried indefinitely.

Audit source: team-filed (Bazalt performance review); corrected by the 2026-09 reconsideration pass.
```

### #969 — activation offload

- **Title →** `Activation offload copies model weights, not just activations: make it selective and overlap transfers`
- **Labels (final):** `enhancement`, `source: audit`, `needs-validation`, `needs-design`, `pkg: patches`, `pkg: transformers`, `impact: performance`, `impact: numerical` (**added**), `severity: medium` (**added** — a documented feature currently does the opposite of its stated purpose).
- **Why:** materially worse than the title suggests. `trainer/_dp_trainer.py:1422-1430` builds `save_on_cpu(pin_memory=False)` and `:2204` wraps the *entire* `ctx.grad_fn(...)` call — `vmap(grad)` plus clipping — in it, while the patched hook at `patches/torch/checkpoint/save_on_cpu.py:29-35` has **no** size, role or device test. Measured: 8 tensors packed, largest is the `(256, 64)` weight matrix — 65,536 of 69,632 packed bytes, **94% weights** — packed unbatched, i.e. once per layer per microbatch, all of it pageable because `pin_memory=False` is forced. The issue lists "parameter aliases" as one bullet; it is the dominant term. PR #994 closed unmerged.

**Body (replace in full):**

```markdown
## Problem

`packages/opaque-transformers/src/opaque/api/transformers/trainer/_dp_trainer.py:1422-1430` builds `torch.autograd.graph.save_on_cpu(pin_memory=False)` and `:2204` wraps the whole transformed gradient execution (`vmap(grad)` + clipping) in it. The patched hook at `packages/opaque-patches/src/opaque/api/patches/torch/checkpoint/save_on_cpu.py:29-35` copies **every** saved tensor to CPU with no size, role, alias or recomputation-context test.

Measured on the current tree: with a `saved_tensors_hooks` probe around `vmap(grad(functional_call(...)))` on a two-layer MLP, 8 tensors are packed and the single largest is the **weight matrix** `(256, 64)` at 65,536 of 69,632 total packed bytes - **94% of the offloaded bytes are weights, not activations**. Weights are packed unbatched, so under `activation_offloading=True` the model's base weights are copied device -> host -> device once per layer per microbatch, on top of the activations the feature is meant to move.

`pin_memory=False` is forced (documented host-OOM rationale at `:1424-1429`), so every one of those transfers is pageable and cannot overlap with compute.

The legacy checkpoint path additionally snapshots and re-enters functional reparameterizations per recomputed segment (`packages/opaque-patches/src/opaque/api/patches/torch/checkpoint/reparametrize_recompute.py`) on Torch versions without native support.

PR #994 attempted this and was closed unmerged on 2026-09-09.

## Impact

Enabling activation offloading pays PCIe for the entire parameter set every step. For a 7B bf16 model that is on the order of tens of GiB of pageable traffic per microbatch, which can easily be slower than simply reducing the microbatch - the opposite of the feature's purpose - while also inflating host RSS toward the host-OOM the `pin_memory=False` comment was trying to avoid.

## Acceptance criteria

- A selection policy that keeps parameter-storage-aliased tensors and small/cheap-to-recompute tensors on device and offloads only large activation-like tensors. Parameter exclusion must be explicit and tested, not incidental.
- Selection is observable: number of tensors and bytes offloaded versus skipped is reported per step, so the 94%-weights result above can be re-measured.
- An optional bounded pinned double-buffer / transfer-stream mode for throughput, with the existing pageable host-memory-safe mode kept as the default until bounded host-memory behaviour is demonstrated.
- Gradient parity, tied weights, views/version counters, saved-tensor lifetimes, and the repeated-backward guards (#874) are preserved; per-example gradients under `vmap(grad)` are unchanged bit-for-bit versus offload-off on CPU.
- Legacy reparameterization snapshots are cached or narrowed where native support is absent.

Audit source: team-filed (Bazalt performance review); corrected by the 2026-09 reconsideration pass.
```

### #970 — MoE casts and FP32 accumulators

- **Title →** `Bound MoE FP32 accumulators per token chunk; treat cached low-precision expert shadows as a separate, gated change`
- **Labels (final):** `enhancement`, `source: audit`, `needs-validation`, `needs-design` (kept — Part B), `pkg: patches`, `impact: performance`, `impact: numerical` (**added**).
- **Why:** both halves of the premise hold exactly (`moe.py:524-526` and `:549-551`; `fused_moe.py:489-491` and `:514-516`; `_cast_to_dtype` at `moe.py:51-58` with no caching; `fused_moe.py:188` and `:232-236` full-size FP32 buffers retained across the chunk loop). The correction is to **split by risk**: the token-chunk-local FP32 accumulation is a pure memory-and-bandwidth win with no lifetime hazard and should land alone, whereas "cache a versioned low-precision shadow for frozen expert banks" re-opens exactly what 97a72920 (#933) closed and needs a headroom policy plus invalidation tests. As written, one issue mixes a safe win with a change that can regress the memory work the umbrella exists to protect.

**Body (replace in full):**

```markdown
## Problem

Two distinct costs, still present in `main`:

**A. Full-size FP32 accumulators (low risk).** `packages/opaque-patches/src/opaque/api/patches/kernels/fused_moe.py:188` allocates `out = torch.zeros(N, H, dtype=torch.float32)` for the whole forward and `:232-236` allocates `dx = torch.zeros(N, H, dtype=torch.float32)` for the whole backward, each retained across the chunk loop and overlapping its converted output. At `N=16384, H=4096` each is 256 MiB on top of the bf16 result.

**B. Repeated whole-bank dtype conversion (higher risk to fix).** `packages/opaque-patches/src/opaque/api/patches/kernels/moe.py:524-526` casts the stacked expert banks in `Opaque_MoE.forward` and `:549-551` casts them again in `Opaque_MoE.backward`; `fused_moe.py:489-491` and `:514-516` do the same for `Opaque_FusedMoE`. `_cast_to_dtype` (`moe.py:51-58`) converts whole tensors with no caching. For `E=8, H=4096, I=14336` that is roughly 15.7 GiB of conversion traffic across forward and backward. Frozen experts cannot go stale within a step, so for them it is pure overhead.

## Scope correction

These should not be one change. (A) is a bounded-memory improvement with no lifetime hazard. (B) - 'cache a versioned low-precision shadow for frozen expert banks' - re-introduces retained weight replicas, which is exactly what `97a72920` (#933, 'bound autocast weight replica lifetimes') removed to fix an OOM. Landing (B) carelessly would regress the memory work that umbrella #956 exists to protect.

## Acceptance criteria

**Part A (land first, independently):**
- Route chunks are aligned to complete token top-k groups; each chunk accumulates locally in FP32 and casts once into its final output/gradient slice, so no full-size `N x H` FP32 buffer is retained.
- Numerical parity within the documented dtype floor versus the current full-FP32 accumulation, including skewed and empty groups.
- Peak memory measured before/after; it must strictly decrease.

**Part B (gated, separate PR):**
- A versioned low-precision shadow keyed on storage + version counter, device and autocast dtype, applied only to expert banks with `requires_grad=False`.
- An explicit headroom policy: the shadow is created only when measured free memory exceeds a stated margin, and is released under pressure. Retained bytes must be reported.
- Invalidation tests covering optimizer updates, in-place mutation, `state_dict` loading, device/dtype moves, tied/shared storage, nested autocast and saved-tensor hooks.
- Must not regress the peak-memory numbers established by #933 and #948.

**Both parts:**
- Per-example gradient parity under `vmap(grad)` for `B >= 2` against a one-example-at-a-time reference.

Audit source: team-filed (Bazalt performance review); re-scoped by the 2026-09 reconsideration pass.
```

### #956 — patched-model memory and throughput (umbrella) — NO CHANGE

- **Title, body, labels:** all unchanged. This is the only one of the 24 that needs no edit.
- **Why:** 7 of 14 children are still open (#958, #963, #964, #966, #968, #969, #970 — each verified individually), so the "close an umbrella whose children are all closed" rule does not apply, and its completion criteria are independently unmet: there is no benchmark harness anywhere (`find . -iname '*bench*'` outside `.git`/`.venv`/`target` returns nothing; #417 and #511 both open), and #880's markers are still missing — `packages/opaque-patches/tests/kernels/test_lora.py:44` and `test_linear_cross_entropy.py:38` still read `pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), ...)` with no `pytest.mark.cuda`, so all 44 tests are deselected on every CUDA lane and skipped everywhere else. The body's problem statement is accurate as written.

---

## 5. The "all children closed" umbrellas

The maintainer's premise holds for three of the four. It is **wrong for #774**.

| Umbrella | Children | Criteria genuinely met? |
| --- | --- | --- |
| #764 | 7 / 7 closed completed | **Yes** |
| #767 | 6 / 6 closed completed | **Yes** |
| #768 | 7 / 7 closed completed | **Yes** |
| #774 | 12 total, **11** closed (2 `not_planned`), **#880 open** | **No** |

**#764 — met.** The single criterion is: *"No public accounting entry point discards a caller-supplied guarantee, silently narrows a tightened parameter, or reports a value computed under a different discretization than the one in force."* All three clauses now hold at `12146ec4`: `_accountant.py:258` registers a loader that receives the template and `:239-250` keeps `template._budget` when the checkpoint carries none (first clause); `discretization.py:163-180` is a genuine `replace(base, **overrides)` partial update with `None` sentinels and a no-op bare call, and `numerics/fft.rs:292-299` rejects `count > u32::MAX` before `powu` (second clause); `calibration.py:320-329` re-evaluates the calibrated process under `overall_config` outside the probe context, and `pld/metrics.rs:314-326` folds `negative_infinity_mass` into the asymmetric CDFs (third clause). I looked for a sibling template-ignoring deserializer and found only `_strategy_codec.py:191`, which is correct because that codec serialises every dataclass field. **Close.**

**#767 — met.** *"Adaptive clipping derives identical thresholds on every rank, invalid accounting parameters are rejected at construction and on deserialization, and the runtime and accounting surfaces accept the same configurations."* The deciding clause is the first, because it was the distributed-correctness one: `dpsgd/clipping/_distributed.py:64-70` now derives the collective schedule from the schema (`group_names = sorted(current_pg.values)`) and folds the per-group noise key by the sorted index at `:76-78`, so ordering no longer depends on how `_num_clipped` was built — and no other `reduce_scalar`/`all_reduce` call site under `packages/*/src` is driven by a rank-local dict iteration order. Clause 2 is `AdaClip.__post_init__` (`_adaclip.py:26-67`), clause 3 is `sample_rate ∈ (0,1]` parity (`_poisson.py:49-58`). **Close.**

**#768 — met.** *"Every clipping and precision primitive either handles the Opaque wrapper types correctly or raises, no configuration silently degrades to a fallback, and adaptive precision decisions are either data-independent or accounted."* The third clause is what decides it, and it was resolved by deletion: 7f55aae5 ("delete(engine)!: remove data-dependent loss scaling (#889)") removed `precision/_loss_scaler.py` entirely, and a repo-wide grep for `unscale_grads|LossScaler|all_finite` over `.py/.md/.yml/.toml/.ipynb` returns nothing outside `.audit-checkpoint/` and `.venv/` — so the defect is genuinely unreachable rather than hidden behind a dangling reference. Clause 1 now holds because `global_norm` raises `InputTypeError` naming `.pytree` (`engine/pytree.py:377-400`, confirmed by execution) and `unscale_grads` no longer exists; clause 2 holds via `_per_group.py:166-179`, `_clipped_grad.py:256-263` and `:223-234`. The one residue of the wrapper-blindness class (`tree_leaves` returning `[]`) is documented behaviour on a non-clipping primitive and is filed standalone. **Close.**

**#774 — NOT met, on two counts.** Its criterion is: *"No blocking lane can pass on zero collected tests or an unavailable accelerator, and a deliberately injected noise-scale or composition error fails the suite."* The **zero-collected-tests** half is unmet and was *declined*: #836 and #841 are closed `not_planned` with no code change, `allow-empty-test-selection: true` is still on `pr.yml:81, 145, 160, 177`, `run_python_test_package.sh:43-46` still converts pytest's exit 5 to exit 0, and `python-tests-distributed` remains a required input to the blocking `python-tests` gate (`pr.yml:190-203`) whose jq filter (`:207-222`) only excuses `skipped` for `python-tests-cuda*`. The other two halves are met and verified (#832's `assert_cuda_available.sh`; #871's realized-stddev conformance test that fails a 0.97× perturbation). Separately, the premise that all children are closed is simply false — `sub_issues_summary` reports 12 total / 11 completed and `get_parent` on **#880** returns #774. So #774 stays open as the tracker for #880, with the two declines recorded as residual risk rather than left implicit (§4).

---

## 6. Sequencing

The ordering constraint that matters is **absorption**: eight of the new candidate findings are already written into corrected bodies above. If the findings are filed first, they become duplicates and the fix gets split across two issues. Correct the host issue first, then file only what is left.

**Tier 0 — before filing any new finding (same sitting).**

1. **Close #764, #767, #768** (§2). No dependencies, no children to touch; immediately drops the open audit set from 24 to 21.
2. **#813 + #769** (§4) — then file the attention-sink defect (`attention.py:365-387`, `missing={'…self_attn.sinks'}`, high) as a **new sub-issue of #769**. And **withdraw** the candidate finding that says gpt2's per-example gradients diverge from upstream by 53×: it is the harness's shared-RNG dropout, proven by 0-of-28 mismatches with dropout pinned. Filing it would create a high-severity issue for a defect that does not exist, and #813's second acceptance criterion already points the other way. The grad-parity-harness and forward-parity-hole findings are absorbed; do not file them.
3. **#319 correction + #359 restate** (§3.2, §4) — absorbs the two high, ε-affecting findings (`IdentityStrategy.sensitivity`, 145× under-report; paired second-moment release, 1.31× under-noised) and the `docs/reference/accounting.md:420-421` staleness. These are the highest-severity items in the whole set; they should be the first *code* work scheduled.
4. **#376 restate** (§3.3) — absorbs the measured 4–5× Type-I inflation finding.
5. **#911 correction** (§4) — absorbs the b-min-sep inert-corpus finding. Then file the `calibrate()` runtime-unsafe-fallback confidence finding (`calibration.py:331-341, 353`, medium, ε-relevant) **separately**: it is a distinct defect about reported Monte Carlo confidence, not about cost.
6. **#417 correction** (§4) — absorbs the fail-open `_assert_perf_benefit` finding, and #417 is the gate the rest of the perf series depends on, so re-scoping it first stops #966/#968 from being re-litigated on evidence nobody can produce.
7. **#963 and #969 corrections** (§4) — absorb the GQA dispatch-multiplier finding and the activation-offload-copies-weights finding respectively.

**Tier 1 — this week, no new-finding dependency.** #773, #774, #775, #828 (drop `needs-design`), #845 → Sweep C1 + C2, #407. Sweep C1 is the cheapest real improvement on the list and has been blocked behind the hypothesis work for two cycles; it can be done by one person in an afternoon. #407's scope item 1 (correct the JME attribution in `_adam.py`, `opaque/dpftrl/noise/__init__.py`, `examples/train_dpftrl.py`) is likewise landable immediately and must not wait on the derivation.

**Tier 2 — can wait.** #956 (no edit), #958, #964, #966, #968, #970. Their corrections are re-scoping only, none absorbs a finding, and several of their acceptance criteria reference benchmark output that does not exist yet — so they should be edited *after* #417 lands a harness, or the new criteria will be unverifiable on arrival. #968 in particular should not be scheduled at all until its kill measurement is possible; if the harness slips, close it rather than carry it.

**Independent of all of the above:** file the `tree_leaves` wrapper-blindness finding (low, `pkg: engine`) as a standalone issue, not under the now-closed #768.