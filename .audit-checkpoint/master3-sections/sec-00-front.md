# Opaque third audit — new findings master list

Post-remediation tree at **12146ec4** (delta `4b13d82..12146ec4`, 124 commits). Format follows the
July and August master lists; numbering continues at **OPQ-400** (August's highest was OPQ-388) and
runs to **OPQ-538**, so the three lists never collide. Findings are grouped by package/subsystem and
severity-ordered within each group. Near-duplicates reported by more than one sweep are merged under
a single ID, with the merged titles noted.

Legend: `file:line` · category · severity · ε:yes/no (does the defect move a reported or deployed
epsilon) · **VERDICT** (CONFIRMED / PLAUSIBLE — every finding here survived independent adversarial
verification; duplicates of open tracked issues are in §8 and refuted candidates in §9).

---

## 1. Summary

**125 new findings**, IDs **OPQ-400 … OPQ-538**.

| Severity | Count | ε-affecting |
| --- | --- | --- |
| Critical | 1 | 0 |
| High | 15 | 6 |
| Medium | 26 | 8 |
| Low | 83 | 1 |
| **Total** | **125** | **15** |

Verdicts: 124 CONFIRMED, 1 PLAUSIBLE (OPQ-414 — the mechanics reproduce, but an explicit test pins
the current behaviour as intended, so it may be a design decision rather than a defect).

Pipeline: 149 candidates entered verification; **131 survived**, 16 duplicate an open tracked issue
(§8) and 2 were refuted (§9). The 131 surviving reports collapse to these 125 IDs after merges; the
verifier's report-level tally was 1 critical / 16 high / 28 medium / 86 low.

**By subsystem**

| Subsystem | Findings |
| --- | --- |
| Accounting — sensitivity, amplification, calibration, removal coherence (`opaque-accounting`, Rust + Python) | 18 |
| Patched-model fidelity and fused kernels (`opaque-patches`) | 19 |
| Public API surface, error contracts and documentation truth | 26 |
| Packaging, CI gates, test integrity, release posture | 24 |
| Trainer checkpoint / resume / distributed state (`opaque-transformers`) | 10 |
| Engine — clipping, pytree, distributed execution (`opaque-engine`) | 8 |
| Alignment and auditing fix-verification residue | 8 |
| DP-SGD (`opaque-dpsgd`) | 5 |
| Optimizers (`opaque-optimizers`) | 4 |
| DP-FTRL (`opaque-dpftrl`) | 3 |

Eleven defects were reported independently by two or three sweeps looking through different lenses and
are written up more than once, because each write-up is the view a different owner will arrive
through. They are **one fix each**, not two or three: OPQ-401 / OPQ-433 / OPQ-447 (the DDP resume
sampler collapse, reached from accounting, engine-distributed and resume), OPQ-403 / OPQ-420 (the
lost `poisson()` inner guard), OPQ-408 / OPQ-505 (the `calibrate()` fallback union bound),
OPQ-417 / OPQ-513 (the stale `bound` allowlist entry), OPQ-430 / OPQ-512 (`tree_leaves` fail-open),
OPQ-406 / OPQ-424 (`random_allocation` in the accounting guide), OPQ-425 / OPQ-524 (the bare-BandMF
participation warning), OPQ-478 / OPQ-483 (Python 3.12 advertised and never executed),
OPQ-480 / OPQ-521 (the extras install commands), OPQ-426 / OPQ-496 (the research diary at the
repository root) and OPQ-495 / OPQ-520 (the privacy-auditing tutorial's amplification). §11 groups
them so no one pays twice.

### The findings that matter most

1. **OPQ-445 (critical) — PEFT/LoRA resume silently restores no adapter weights.** `strict=False` is
   applied to a checkpoint whose key space does not overlap the live module's at all, so the resumed
   run continues from fresh random init while the accountant, noise counter, sampler cursor and
   optimizer state are all restored faithfully — every signal a user checks says the resume worked,
   and in a DP setting the lost steps cannot be re-run because their budget is already spent.
2. **OPQ-401 / OPQ-433 / OPQ-447 — DDP resume installs rank 0's sampler snapshot on every rank**,
   collapsing the per-rank Poisson streams the rank fold exists to keep independent while the
   accountant keeps pricing i.i.d. Poisson amplification: the only ε-affecting defect in the delta
   that destroys an amplification *hypothesis* rather than mispricing a parameter.
3. **OPQ-400 — adaptive clipping releases a `{0,1}` clipped-count query while AdaClip accounting
   prices it at the `{−1/2,+1/2}` sensitivity**, understating the quantile stream's µ by exactly 2×;
   measured with the repo's own accountant, documented per-group configurations report 1.354 against
   a correctly-priced 5.356 (3.96×), and the pairing is what `DPTrainer(clipping_mode="adaptive")`
   wires by default.
4. **OPQ-517 — the flagship LLM fine-tuning tutorial publishes ε = 1.09 for a run whose accounted
   cost is 8.99**, because it prices an adaptively-clipped loop with the plain Gaussian chain: a
   wrong privacy number served today on a live public page, not a latent code defect.
5. **OPQ-465 + OPQ-466 — the generic fused LoRA QKV wrapper trains a different architecture than the
   one loaded**, applying RoPE on Cohere2's NoPE full-attention layers (25 % of layers at the default
   config) and dropping `sliding_window` / `softcap` for Mistral, Ministral and Gemma2 — silent
   numerical divergence on the GPU training path at default patch settings.
6. **OPQ-460 — the now-default fused-CE forward wrapper's non-fused fallback recomputes logits
   without the family's post-`lm_head` transform**, so Cohere/Cohere2/Granite/Gemma2 train against
   an unscaled, unsoftcapped objective on a documented first-class precision mode (fp32 weights on
   CUDA), with no warning and no test reaching the branch.
7. **OPQ-448 — full-finetune resume and `load_best_model_at_end` crash for every tied-embedding
   model**, because the checkpoint omits `lm_head.weight` and the loader is strict: loud rather than
   silent, but it means the documented resume path is broken for both model shapes at once — PEFT
   silently (OPQ-445) and full fine-tunes noisily.

**Honest note on the criticals.** Three findings arrived reported as critical; two were reduced to
**high** on verification, and both reductions were about reachability, not about whether the defect
is real. OPQ-460 (fused-CE fallback dropping the family logit transform) needs fp32 model weights on
CUDA to reach the fallback — a documented mode with an example flag, but not what the most literal
default flow produces, since `from_pretrained(dtype="auto")` on bf16 Cohere/Granite/Gemma2
checkpoints routes to the correct fused branch; impact is utility, not ε. OPQ-465 (RoPE on Cohere2's
NoPE layers) needs one specific family plus LoRA on all of q/k/v plus CUDA, and likewise has no
privacy effect. Only OPQ-445 kept the critical label: it fires on the documented default
fine-tuning shape, on every host, with no configuration required and no observable symptom.

### The theme of this round

The July and August audits found defects in code that had been wrong since it was written. This
round is different, and the difference should shape who fixes what. Six of the fifteen high
findings sit in `opaque-patches`, and **every one of them is a correctness regression introduced by
a commit whose stated purpose was throughput** — `c0f95ced` (#988, "activate fused loss-only
cross-entropy") made a pre-existing wrong fallback branch reachable at trainer defaults (OPQ-460)
and made `torch_compile` unusable for every registered dense family (OPQ-462); the four QKV-fusion
commits (#947/#949/#951/#950) built faithful wrappers for two families and left six more on a
generic wrapper that is faithful to none of them (OPQ-465, OPQ-466); `d80d3f99` (#954) and
`4e61b09d` (#925) extended the compact-SDPA mask elision and reopened OPQ-337 in a sibling path
(OPQ-472). The privacy core, by contrast, held: clipping-bound enforcement survived every mutation
injected into it, per-example isolation through the microbatch and vmap paths is exact, and the new
Poisson PLD math (#850) was conservative at every one of 3,360 points checked against an independent
60-digit closed form and every one of 142 checked against Google's `dp_accounting` (§10).

The second half of the theme is that the Marble milestone (#3, public-release readiness) was closed
15/15 and **verified for the first time this round** — and it does not hold. The repository is
public, the docs site is live, and the documented extras install commands resolve to an unrelated
PyPI project named `opaque` (OPQ-480 / OPQ-521) while six of the ten published distribution names sit
unregistered and claimable (OPQ-481). Closing a readiness milestone without executing its own
instructions is how a supply-chain exposure and a wrong published epsilon (OPQ-517) end up shipping
together.
