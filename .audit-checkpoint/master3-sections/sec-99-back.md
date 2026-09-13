## 8. Tracked duplicates

Sixteen verified reports were withheld from §2–§7 because they restate a defect that is already
tracked by an **open** issue. Several are materially wider than the issue title suggests, and that
widening is recorded here so the owner can restate the scope — but **none of these may be re-filed
as a new issue.** Add the detail as a comment on the tracking issue instead.

| Verified report (as written by the sweep that found it) | Tracked by |
| --- | --- |
| `IdentityStrategy.sensitivity` ignores the participation schema, so bare `mf_gaussian` under-reports ε by √k | **#359** — Define BandMF participation sensitivity and price second moments consistently |
| `IdentityStrategy.sensitivity` ignores the participation schema: bare MF accounting under-reports ε by √k (145× at the default horizon) | **#359** (same defect, reported again by a second sweep at a different horizon) |
| Paired second-moment accounting is documented as unconditionally equal to the first-moment mechanism, but the runtime sizes both streams from single-participation column norms | **#359** |
| Paired second-moment MF release is never tied to an accountant; a documented mismatched-strategy pair is 1.31× under-noised relative to the charged budget | **#359** |
| `allow-empty-test-selection` still lets two distributed-lane legs report green on zero collected tests | **#774** — Close CI fail-open gates and restore test integrity |
| OPQ-381 was closed (#836) without a change: exit-5 is still converted to success, and two of five distributed-lane legs plus the foundation leg of every CUDA lane still pass on zero tests | **#774** |
| OPQ-386 was closed (#841) without a change: Python 3.12 still executes zero tests while all ten wheels advertise it | **#774** (also written up as OPQ-478 / OPQ-483 where the wheel-classifier half is new) |
| Open issue #880 is materially wider than its title: 87 CUDA-requiring tests across 13 files in 5 packages carry a `skipif` but no `cuda` marker, so they run in no CI lane | **#880** — Restore CUDA marker coverage for the LoRA and linear-cross-entropy kernel tests |
| RMSNorm / fused-add-RMSNorm / RoPE backwards overwrite the upstream gradient buffer on a single backward, not only on a repeated one (wider than the tracked scope) | **#898** — Audit the remaining fused kernels for the repeated-backward gap |
| Patching `gpt_oss` / `deepseek_v4` silently deletes the attention sink and freezes the `sinks` parameter | **#813** — Convert parity skips to strict xfail and resolve the gpt2 gradient mismatch |
| Grad-parity harness is non-deterministic under dropout and mis-attributes the failure to the patches | **#813** |
| Measured 4–5× Type-I inflation in the default one-run audit path — #376 is materially worse than its title suggests | **#376** — Calibrate one-run auditing under the null |
| The only in-repo performance gate is fail-open: a kernel 20 % slower at equal memory asserts "benefit" | **#417** — Add a reproducible benchmark harness for every published number |
| b-min-sep Monte Carlo transcript reuse is silently inert at default settings for any horizon above 60 steps | **#911** — Monte Carlo calibration performance may be unusable |
| The GQA de-replication fix (#985) multiplies SDPA dispatch count by `num_key_value_heads` on every vmapped attention path | **#963** — Tune compact sliding-window attention and support padded batches |
| `activation_offloading` copies model weights, not only activations, on every microbatch | **#969** — Make activation offload selective and overlap transfers |

Four of the sixteen point at **#359**, which is the oldest open high-severity item in the backlog
(July, `needs-design`). Two independent sweeps reproduced its √k under-report this round and one
measured it at 145× at the default horizon. It is not new, and it is not getting smaller.

---

## 9. Refuted candidates

Two candidates were reported and did not survive verification. They are recorded so the
consideration is on file and nobody re-derives them.

| Candidate | Why it was refuted |
| --- | --- |
| The published LLM tutorial is built on `distilgpt2`, a family the documentation elsewhere declares unsupported | The load-bearing claim is false: `gpt2` is a first-class registered family (`register_family('gpt2', ...)`, present in `supported_families()`, with its own row in the `model-patches.md` compatibility table) and is the model used by both canonical quickstarts. The two sources cited as authoritative — a README feature bullet and the `AGENTS.md` family list — are non-exhaustive prose that also omits eight other registered families. The only residual is open **#813**. |
| GPT-2's per-example gradients diverge from upstream by 53× relative error | The 53× reproduces but is an artifact of the parity harness, not of the patched model: it compares a dropout-0.1 reference against a patched model whose dropout the default `compat=True` patch intentionally zeroes. Given both sides the same dropout-free contract, patched and upstream gradients are **bit-identical** (0 of 16 tensors differ under `torch.equal`). The real residual is a one-line documentation gap — `model-patches.md:58` does not list dropout-zeroing among the compat-bucket effects. |

---

## 10. Checked and found sound

This section is a coverage record, not a compliment. It says what property was checked, by what
method, so that a future reader can tell the difference between "audited and sound" and "not looked
at". Where a claim could not be established either way, that is stated too.

### 10.1 Privacy invariants (end-to-end, executed)

- **Per-example clipping sensitivity bound**, fixed and AUTO-S: ran `clipped_grad` /
  `auto_clipped_grad` with `return_aux` over fp16/bf16/fp32 × gradient scales 1e-4…1e3 × C ∈
  {1e-4, 1, 1e4} × γ ∈ {1e-8, 1e-2} × microbatch {off, 2} and read the post-clip norms; maximum
  observed ‖clip(g)‖/C = 0.99999990, never above 1.
- **Per-group noise allocation**: derived the Mahalanobis budget — σ_g = nm·√(B_g·Σ_h B_h) gives
  Σ_g (B_g/σ_g)² = 1/nm² exactly, so the per-group Gaussian has the same PLD as `gaussian(nm)` with
  no composition penalty in the group count — and verified the clip-group and noise-group path
  assignments cannot diverge.
- **Paired first/second-moment budget**: derived Σ_g[(Δ₁/σ₁)² + (Δ₂/σ₂)²] = (c₁/nm)² for the shipped
  allocation and confirmed the shipped squared-stream bound C²/`normalize_by` is a valid bound on
  ‖g ⊙ g‖₂, including the per-group case.
- **k-out-of-t block sampler vs its accountant**: `KOutOfTSampler.block_sizes` matches the Rust
  `random_allocation_gaussian_pld` decomposition exactly — unequal blocks are priced at their own
  higher per-step probability, not at an averaged t/k. The "block bound dominates total allocation"
  claim was re-derived independently and then Monte-Carlo'd over 4M draws at five (t,k,σ) points.
- **BandMF min-separation sensitivity**: hand-verified `max_participation_for_linear_fn` as the
  correct at-most-k / min-sep dynamic program, and that `minsep_sensitivity_squared` reconstructs
  ‖Σ_{j∈π} C[:,j]‖² for the equally-spaced pattern including the k-truncation subtraction.
- **b-min-sep and balls-in-bins samplers vs their hypotheses**: traced the cooldown Markov chain for
  b = 2, 3 and confirmed the realized separation is exactly `bands` and that the warm start matches
  the chain's stationary law; confirmed balls-in-bins draws its bin assignment once in `__init__`
  and reuses it every epoch, which is the fixed-assignment hypothesis the dominating pair requires.
- **DDP order of operations**: clip → `sum_gradients_` (op `sum`, not `mean`) → sync → `noise_fn`
  with a rank-shared key → optimizer. Noise is applied after the cross-rank sum exactly once, and an
  empty Poisson round still runs the full path and still releases.
- **RNG domain separation**: `fold_in` is a blake2b chain with disjoint `b'i'`/`b's'` type prefixes
  and a `b'|'` separator, so integer folds (steps, ranks, leaf/group indices, `split()`) cannot
  reach a key derived through a string tag; every mechanism roots itself with a namespaced tag.
- **MF constant-per-step-sensitivity latch**: `_first_max_norm` and its sync fingerprint are in the
  codec's required-field set, are round-tripped, and are re-attached on every call, so a
  clipping-norm change across a resume still fails closed.
- **PLD cache soundness**: `pld_cache` keys on the fully resolved `DiscretizationConfig` together
  with a process-free structural key over every dataclass field, so a cached looser result cannot be
  served to a stricter query.

### 10.2 The new Poisson PLD amplification math (#850)

- **Derived the generalized amplification statement from first principles** and confirmed it against
  the cited primary source; verified the REMOVE transform algebraically against `poisson.rs:311-339`
  atom by atom, including the +∞ atom and the dual-residual atom at log(1−q).
- **Conservatism, Gaussian paths**: checked against an independent 60-digit `mpmath` closed form for
  the subsampled-Gaussian hockey stick, derived independently of the repo's formulas, over
  σ ∈ {0.3 … 3.0} × q ∈ {1e-4 … 0.9} × ε ∈ 0.0…5.9 — **3,360 points × 2 paths**, no
  under-estimate.
- **Conservatism, non-Gaussian bases** it now accepts: compared `delta_at` against the exact
  two-atom transformed support computed by hand for `eps_delta_pld` over 5 × 4 × 5 parameter
  combinations × ε ∈ −2.0…6.0.
- **Connect-the-Dots projection** (`poisson.rs:233-306`): verified the grid brackets the whole
  transformed support, that δ at each knot is the exact tail sum plus a positive buffer, and that
  the piecewise-linear-in-eᵉ reconstruction is conservative by convexity.
- **Realization repair** (`poisson.rs:125-224`): confirmed moving mass to +∞ dominates the input at
  every ε, that the repair is capped at 3 rounds and fails closed, and that it is not needed on the
  Gaussian fast path.
- **External cross-validation**: 34 grid points plus a 108-point wider sweep against Google
  `dp_accounting` 0.6.0 — Opaque exceeds the reference at **every** point (minimum signed gap
  +9.2e-8) — plus 12 advantage/β points against `riskcal`.
- **Self-consistency and robustness**: `poisson(poisson(g,q₁),q₂)` reproduces the rate-q₁q₂ map
  exactly; rates 1e-12…0.999999 and 199 q-points produce no panics and monotone `delta_at`; the
  `gaussian_source` fast-path tag is cleared by every operation that would invalidate it.
- `cargo test -p opaque-accounting --lib amplification::` → 79 passed, including the new
  generic-Poisson conservatism tests.

### 10.3 Checkpoint and resume state

- **MF noise-state continuity through the real trainer**: `mf_band(bands=4)` + `cyclic_poisson`,
  8 steps, `save_steps=4`, resumed at checkpoint-4 — checkpoint-8's `dp_state.pt` is byte-identical
  between the continuous and resumed runs across all four band-history tensors, `_step_counter`,
  `_rng_key`, latch, fingerprint, sampler state, clip state, horizon-process state and the
  calibrated multiplier.
- **Gaussian noise-stream continuity**: steps 3–5 of a resumed stream are bit-identical to the
  uninterrupted run, and none of steps 0–2 is replayed.
- **Structural-drift fail-closed**: a 16-case save/restore matrix across band_mf/blt/bisr/bsr/
  identity/lambda_cgd — bands 2↔4, bandwidth 3↔5, band_mf↔blt, band_mf↔identity all raise
  `CheckpointError` with field-level diagnostics. BISR's `_BisrExecutionIdentity` is the strongest
  codec in the tree and rejects unversioned legacy dense-history checkpoints by name.
- **Removed-type codecs fail closed**: `PerStep` and `RandomAllocation` raise
  `Unknown DpProcess type`, renamed `KOutOfT` fields raise `unexpected keys` / `missing required
  field`, so no pre-removal `accountant.json` can be silently reinterpreted. `DP_STATE_BUNDLE_VERSION`
  7 → 8 rejects every released bundle before any state is applied.
- **Prefetch cursor clamp (#1009)** verified empirically at `dataloader_num_workers` 0 and 2:
  `consumed == global_step` for both `poisson` and `k_out_of_t`, with a load-side re-clamp.
- **Atomic publication**: staging dir + `os.replace`, `^checkpoint-(\d+)$` discovery ignores
  `.tmp`, and weights-only exports lacking `dp_state.pt` / `dp_optimizer.pt` / `accountant.json`
  are refused rather than silently restarting the noise stream.
- **MF horizon-calibration resume guard (#789)** is in place and closes the σ-mixing path.

### 10.4 Fused kernels and patched-model parity

- **Chunked portable linear CE** numerical parity: every feature combination (label smoothing, logit
  softcapping, logit scale, token scaling and combinations) at float64 against an independent
  `logits → F.cross_entropy` reference for `chunk_vocab` ∈ {None, 4, 1}, including a sample whose
  labels are all −100 — loss, d_hidden and d_weight agree to ≤ 7.1e-15.
- **Per-sample isolation under `vmap(grad(...))`** for the same kernel: max diff **0.0** against an
  explicit per-sample loop; perturbing sample 2 changed samples 0/1/3 gradients by exactly 0.0, so
  the streamed LSE, the token/vocab tiling and the `cat`-based accumulators leak nothing across
  samples.
- **Triton shifted-index translation (#945)**: hand-checked `offs_b + offs_b // tokens_per_sequence`
  in both forward and backward — no read crosses a sequence boundary into the next sample's hidden
  states or labels, which is the specific cross-sample leak the copy-free shift could have caused.
- **Buffer separation and mutation**: the fused-CE backward now writes only to a distinct
  `grad_logits_ptr` (int64-safe indexing) and reads saved logits read-only; the LoRA MLP
  saved-activation mutation behind OPQ-343 is structurally gone after #934; the fused-add-RMSNorm
  RSTD reuse was checked store by store.
- **GQA de-replication (#985)** exact parity on CPU at float64 against a `vmap_repeat_kv`
  reference: max |diff| = **0.0** for four mask shapes and `is_causal=True`, with the head-order
  contract and per-sample `vmap(grad)` gradients also at 0.0.
- **Whole-model fused-vs-unfused LoRA QKV parity**, run end to end per family with the fused branch
  forced: **Llama, Gemma, Granite, GLM4, Qwen2, Qwen3 and Gemma3 are bit-exact** (max |diff| = 0.0)
  under both `eager` and `sdpa`. Only Mistral, Gemma2 and Cohere2 diverge (OPQ-465, OPQ-466). The
  dedicated Qwen3 and Gemma3 wrappers were also checked line-by-line against transformers 5.15.1.
- **Fusion eligibility gates fail safe** at patch time: `bias='all'`, `bias='lora_only'`,
  `lora_dropout=0.1` and partial q/k coverage all decline fusion; families outside the eligibility
  sets are skipped rather than routed to a wrong wrapper; a scan of all 478 importable
  `transformers.models.*` modules found no look-alike class-name mis-fusion.
- **Test-weakening sweep** across the delta: no tolerance was widened on an existing assertion in
  `opaque-patches/tests`; the one `pytestmark` change *removes* a CUDA skip.

### 10.5 Compiled and dynamic-shape paths

- **Dynamic-shape graph reuse, numerics**: the strict chunk compiler (`fullgraph=True`,
  `dynamic=True`) against eager `clipped_grad` over 11 realized batch sizes with
  `microbatch_size=4`, so full, remainder and size-1 chunks all occur — summed gradients,
  per-example norms, clipping rate and batch size bit-identical to eager.
- Same differential test repeated for `adaptive_clipped_grad` (threshold crossing the compiled
  boundary as a tensor, 7 steps including an empty draw) and `PerGroup` clipping (2 groups,
  5 batch sizes including 0 and 1) — exact agreement on gradients, `max_norm`, the adapted
  threshold sequence and the per-group counts.
- **Empty-batch short-circuit (#808)** still precedes any vmap or compiled kernel in all three
  clipping entry points, with structure, key names and dtype preserved (the sanitization gap inside
  that path is OPQ-428).
- **Eval accumulator ordering** unit-tested over 24 combinations of `eval_accumulation_steps` ×
  batch counts × ragged sequence lengths: the balanced chunk tree plus the cold-chunk reversal
  restores strict oldest-to-newest row order. The CUDA transfer pipeline was read for
  read-before-write and use-after-free (stream waits, `record_stream`, pinned-destination
  retention) and telemetry division guards exist.
- **DDP × strict compilation**: a first-call compile failure on a non-empty rank while a sibling
  skipped the kernel on an empty draw routes through `_synchronize_grad_failure` before the gradient
  AllReduce, so a compile error is rank-symmetric rather than a hang.

### 10.6 Public API surface and error contracts

- **`__all__` integrity across the whole surface**: imported all 36 façade modules and every
  documented sub-façade and `*.types` module and resolved every `__all__` entry with `hasattr` —
  **zero dangling names**.
- **mkdocstrings**: all 54 `::: <identifier>` directives resolve; `mkdocs.yml` nav has 0 missing
  files; 0 broken relative links in `docs/`.
- **Exception taxonomy (#760)**: the seven names, their `ValueError`/`TypeError`/`RuntimeError`
  bases and the `CalibrationError`/`PrivacyBudgetError < ConfigurationError` and
  `CheckpointError < OperationError` chains match `docs/reference/exceptions.md` exactly.
- **Domain validation, probed with bad inputs rather than read**: `opaque.random` (14 inputs),
  `opaque.auditing` (20 inputs), `opaque.scheduling` composed factories, and every clipping / noise
  / sampler / DP-FTRL strategy factory — each raises `ConfigurationError` or `InputTypeError` naming
  the offending knob. (The accounting *metric* queries are the exception: OPQ-510.)
- **`TrainingArguments` privacy-kwargs validation (#1024)**: 19 malformed configurations all fire
  with the mode and allowed set named; renamed adaptive knobs fail closed rather than being ignored;
  `mf_band` + `sampling_mode='poisson'` is now rejected and `mf_band` defaults to `cyclic_poisson`.
- **Registered HF family coverage** under the pinned transformers 5.15.1: imported every
  `module_path` referenced by the 26 model files and asserted `eager_attention_forward`,
  `repeat_kv`, `apply_rotary_pos_emb` and every class in each `classes={...}` map exists — 0
  missing, so the silent-skip paths reported elsewhere are latent under the pin, not live.
- **Distributed collectives** fail closed as documented: uninitialized group → `OperationError`
  with the remedy, bad op → `ConfigurationError` listing the five valid ops, unregistered sync type
  → `InputTypeError` listing the registered types.
- **Composition operators**: `proc * 0`, `proc * -2`, `repeat(proc, 0)` all raise pointing at
  `identity()`, and integer repeats compose correctly (ε 4.377 → 6.573 → 8.385 for k = 1,2,3).

### 10.7 The three breaking removals

- **Bounded Gaussian (#1021, 873de427)**: a surviving-reference grep over `packages/`, `docs/`,
  `examples/`, `README.md`, `AGENTS.md` and `CONTRIBUTING.md` returns zero hits; the API raises
  `TypeError` and the trainer surface raises a targeted `ConfigurationError` for all four `bound`
  spellings under every mechanism; checked for false positives by signature-inspecting all six
  strategy factories. The replacement sampler is distributionally **wider**, not narrower — the old
  clamped inverse CDF truncated even the "unbounded" path at ±5.1σ in fp32 and ±2.4σ in bf16, and
  200,000 draws per dtype now show no analytic clamp.
- **Horizon-prefix accounting (#894, 87404a1b)**: every surviving `pld()` was compared line by line
  against the branch of the deleted `pld_at()` it replaced — full-horizon ε is numerically
  unchanged; legacy state fails loud (`Unknown DpProcess type: PerStep`); the trainer installs the
  horizon process once and privacy-based early stopping is disabled in code, not only in prose; the
  removal also resolved OPQ-328 at the root (measured: 5.79 s first query, 0.000 s thereafter, same
  value).
- **Data-dependent loss scaling (#889, 7f55aae5)**: no dangling references anywhere;
  `TrainingArguments.fp16` is now a read-only `False` property and `from_hf` refuses `fp16=True`;
  `LossScalerState` was never a `RuntimeCheckpoint` field so no migration was needed; the OPQ-333
  sibling was fixed properly (`global_norm` now raises on DP wrapper types) rather than deleted with
  its partner.
- Regression run over all three removals' surviving code with a freshly built native extension:
  `ruff check packages/` clean; 397 + 210 + 504 + 43 tests pass under the PR-gate marker.

### 10.8 Test integrity — mutation results

Method: edit the shipped source at HEAD `12146ec4`, clear `__pycache__`, run the owning package
suites under the PR-gate marker (`-m "not cuda and not mps and not slow and not distributed"`,
`-n 8`), restore the file. Clean baseline: 4,590 passed / 191 skipped / 2 failed, both failures
network flakes that pass in isolation. The stale in-tree `.so` (missing #850's `poisson_pld`) was
rebuilt first, which removed 88 spurious failures.

**Defects the suite caught — this is the evidence of real coverage:**

| Injected defect | Caught by |
| --- | --- |
| **M3** clipping bound loosened 5 % (`ratio = C/norm * 1.05`) | **42 failures** — `test_clipping_numerics.py::test_clip_pytree_bound_holds_in_stored_dtype` across 4 dtypes × 5 shapes, leaf counts 1…600, per-group bounds, complex leaves |
| **M4** conservative ULP margin removed in `_guard_scale` | **38 failures**, same module, plus `test_guard_is_negligible_at_float32` |
| **M5** microbatch slice reuse (`x[s:e]` → `x[0:e-s]`, summed sensitivity becomes k·C) | **24 failures** across `opaque-engine` and `opaque-dpsgd` — the microbatch/full-batch equivalence tests for all four clipping entry points, second moments and low-precision accumulation |
| **M13** reported per-example sensitivity 10 % too small while the output is unchanged | **11 failures** spanning engine, dpsgd and dpftrl, including scalar, per-group, AUTO-S, adaptive and paired second-moment metadata |
| **M1** scalar Gaussian noise 1 % short of its reported σ | **1 failure** — `test_noise.py::TestGaussian::test_realized_noise_matches_reported_stddev` (n = 1e6, tolerance 4.24e-3, observed 9.78e-3); the test carries its own negative control so the tolerance cannot silently widen |
| **M6** Poisson draw replaced by a fixed-size sample with the same mean | **5 failures** including `test_statistical_properties_variance` and the cross-rank independence test — the suite pins the batch-size *distribution*, not just its mean |
| **M7** composition count off by one in `Repeated.pld` | **39 failures**, essentially all in `test_cross_validation.py` against `dp_accounting` and `riskcal`. Worth knowing what does *not* catch it: the trainer-level `test_accountant_composes_exactly_total_steps` builds its "independent" reference the same way the trainer does, so it pins the step count but not the arithmetic. The independent oracle is the cross-validation suite, and it works. |
| **M8** resume replay dropped from `_from_state_dict_poisson` | **4 failures** including a test that loads into a template built from a *different* seed, so it cannot be satisfied by the template's own stream |
| **M9** noise stream step index frozen | **2 failures** (`test_uniqueness`, `test_state_evolution`) — thin but sufficient |

**Defects the suite did not catch** — four injected mutations survived (per-group Gaussian noise,
the paired second-moment stream, per-group MF noise, and the AdaClip quantile release: 0 failures in
3,677 / 1,346 / 620 / 1,523 tests respectively). They are one defect class and are filed once, as
**OPQ-490**.

Also verified: all six test deletions in the delta accompany a feature removal or rename, not a
coverage retreat; exactly one tolerance was loosened across 154 changed test files and it was
re-tightened per-metric; every added `skip`/`xfail` is a platform or availability gate; the 19
`perf(...)` commits removed no assertion without replacing it with an equal-or-stronger one; and
`set_discretization` — the only process-global privacy knob tests mutate — is restored at every
mutating site, so `pytest-xdist` cannot leak it.

### 10.9 Documentation truth (the parts that hold)

- **Every ε and noise-multiplier value in `accounting_and_calibration.ipynb` and
  `dp_ftrl_training.ipynb` was recomputed against HEAD and matches to the published precision**
  (0.7181 / 6.67e-07, 1.2656 / 3.0000, 1.8283 / 2.61e-03 / 0.1612, …; sensitivity 1.0000,
  multiplier 4.1981, achieved 3.00). `dp_sgd_training.ipynb` is the correctly fixed version of
  OPQ-355 — every cell read, no residual shuffled/fixed-batch loop.
- **OPQ-354 is genuinely fixed**: `_bare_n_steps` raises `ConfigurationError` when `n_steps` is
  `None`, so the one-step single-participation mispricing is no longer constructible.
- **Mathematics re-derived rather than trusted**: the k-out-of-t "block dominates total" claim, the
  paired second-moment Mahalanobis identity at `docs/reference/noise.md:56-90`, the Gaussian
  privacy-loss and hockey-stick expressions in `docs/mechanisms/dp-sgd/gaussian.md`, the auditing
  canary-ceiling table at `docs/user-guide/auditing.md:226-230`, and the b-min-sep rate conversion
  `p = p₀/(1 − p₀(bands−1))` all check out.
- **The RNG domain-tag table** at `docs/reference/rng.md:218-232` matches the source exactly — all
  twelve published tags, no extras, no duplicates, including #1018's new namespacing.
- **A static kwarg/positional checker over all 370 extracted docs snippets** found exactly one
  signature drift (reported as OPQ-522); no snippet imports a name that does not exist and no
  `alias.attr` reference fails to resolve. Executing all 370 in isolation produced 296 failures, of
  which every one except the four reported is a deliberate placeholder, a signature fragment, or an
  environment precondition.
- **`examples/quickstart.py`**, the file the Quick Start page now includes, runs green end to end,
  reaches exactly ε = 3.00 at step 310, and carries two real assertions.
- **The documented JetBrains index is genuinely public and complete**: it returns 200 anonymously and
  serves all ten distribution names, so the pinned `--index-url` form of the install instructions
  works as written (which is precisely why dropping the flag in the extras block, OPQ-480, is a
  defect rather than a general packaging failure).
- **The trainer's sampler allow-lists match their documentation** row for row.

### 10.10 What could not be established

- A **200-trial null-calibration study of the one-run auditor** (random scores independent of
  membership, m = 1000, both the µ-GDP and ε-δ methods) was started and timed out before finishing.
  No claim is made here about the calibration of `OneRunEstimate.epsilon_at` under the null; that
  question belongs to open **#376**, which the tracked-duplicate above says is already worse than
  its title.
- **CUDA and Triton execution**: the audit host is CPU-only. Every fused kernel gates on
  `hidden_states.is_cuda` / `fused_kernels_available()`, so the GPU-only branches were verified by
  reading plus CPU-forced-branch reproductions (monkeypatching `is_cuda`), not by running on a card.
  OPQ-460's reachability argument in particular rests on a readable boolean, not an execution.
- **ARC-001 (the PEP 420 namespace invariant)** is still enforced behaviourally by
  `validate-distributions.yml`, but `AGENTS.md`'s claim about a `ci.yml` shell guard no longer
  matches `.github/` — nothing there greps for `__init__.py` any more.

---

## 11. Suggested triage order

Ordered by **guarantee-at-stake per hour of engineering**, not by severity. Items that share a file,
a call site or a root cause are paired, and the pairing is stated — several are literally one edit.
The documentation and release items are folded into coordinated sweeps rather than listed one by one,
because filing thirty individual documentation tickets is how the last two rounds produced thirty
individually-closed issues and a docs site that still publishes a wrong epsilon.

### Band A — this week: highest stake, bounded work, no open design question

1. **OPQ-445 + OPQ-448 — one code pair.** Both live in the same two functions,
   `_load_model_weights` and `_load_best_model`, and both are caused by the same line
   (`strict = not self._is_peft`). Fix them together: branch on `_is_peft` and call
   `set_peft_model_state_dict` / `load_adapter`, and for the non-PEFT branch load with
   `strict=False` plus an explicit tied-key coverage check. **The cheapest and highest-leverage
   single change in this entire list** is the shared half: capture the `_IncompatibleKeys` return at
   both sites and raise when *no* key matched. That one assertion converts this whole failure class
   from silent to loud for any future key-space skew. Add a weight assertion to the existing resume
   tests, which today assert only `global_step` and `privacy_epsilon`.
2. **OPQ-401 / OPQ-433 / OPQ-447 — one defect, three write-ups; do the interim guard now.** Land the
   fail-closed half first: refuse to restore a sampler snapshot when `world_size > 1` unless the
   snapshot is this rank's own. That is about an hour and it closes the ε exposure immediately. The
   durable fix — per-rank snapshot files, or persisting the rank-independent base key and re-applying
   `fold_in(base, rank)` at restore time — is a design decision (see Band E) and should not block
   the guard.
3. **OPQ-517 + OPQ-495 / OPQ-520 — wrong epsilons on live public pages.** Two notebooks; the fix is
   editing the mechanism construction and re-executing in place so the published outputs carry the
   honest number. Do it in the same PR as the gate that prevents recurrence (see D3), because
   without the gate this is the third round in a row that a tutorial's published ε has drifted.
4. **OPQ-436 + OPQ-437 — the Adadelta/RAdam second-moment pair.** Same defect in two sibling files:
   both private second-moment branches lack the negative-`v` fallback that `_adam.py:197` documents
   and applies. Two one-line `torch.where` guards copied from the Adam branch, plus extending each
   test from one update to six or more so the divergence has room to appear. Cheap, mechanical, no
   decision required.
5. **OPQ-480 / OPQ-521 + OPQ-481 — the install path.** The doc half is a few `--index-url` flags (or
   one persistent-index restructure) plus a sentence warning that `opaque` on PyPI is someone else's
   project; minutes of work on the single most-read page in the repository. The registration half
   (six unclaimed distribution names) needs an external owner, so open that request today even
   though your own code work is zero — the exposure is live until someone else acts.

### Band B — the performance-regression cluster: one owner, one sweep

These are all in `opaque-patches`, all introduced by throughput commits in this delta, and all
cheaper to fix as one body of work than as six tickets, because they share files and test harnesses.

6. **OPQ-465 + OPQ-466 — the LoRA QKV per-family losses.** One file
   (`patches/peft/components/qkv.py`), one root cause: the generic wrapper routes by attention class
   name and forwards no per-family kwargs. Immediate action is small and safe — remove
   `"Cohere2Attention"` from `_FUSEABLE_QKV_ATTENTION_CLASSES` and forward `sliding_window` /
   `softcap` — and both halves are verified by the same device-independent pipeline test pattern
   that Gemma3 already has. Whether to keep name-set routing at all is a design decision (Band E).
7. **OPQ-460 + OPQ-461 — the two fused-CE logit-scaling reports.** Same missing concept (the
   family's post-`lm_head` transform) on two paths. OPQ-460 is nearly free: make the fallback
   delegate to `original(self, ...)`, which the wrapper already does two branches away. OPQ-461 needs
   the scale threaded into `linear_nll_sum` / `fused_dft_loss`, so gate `_fused_dft` off for scaled
   families in the same PR and do the threading behind a decision.
8. **OPQ-462 + OPQ-452 + OPQ-453 — `torch_compile` is out, reported three times.** One feature
   outage: the deprecated `return_dict` read, the deleted fullgraph fallback and the `dynamic=True`
   interaction with AUTO-S. One owner, one decision on fullgraph policy, one parametrised end-to-end
   test over `clipping_mode` × `torch_compile` that actually runs the trainer — the current compile
   tests only build the arguments, which is why three separate regressions shipped.
9. **OPQ-474 (cheap) then OPQ-473, OPQ-472 (masking).** OPQ-474 is a small high-leverage fix — teach
   `_is_hf_pretrained_model` to unwrap PEFT, after which DP-LoRA over an unregistered family fails
   closed as intended. OPQ-473 can also fail closed cheaply (raise when a mask hook is present
   rather than ignoring it). OPQ-472 is the one that needs the predicate unified and belongs with
   the design items.
10. **OPQ-463, OPQ-467, OPQ-468, OPQ-469, OPQ-471** — the remaining kernel-wrapper defects, same
    files and same harnesses as items 6–7. Batch them behind that work; OPQ-463 deserves priority
    within the batch because it silently disables `opaque.auditing.gradient_scores`; OPQ-464 (a
    functorch guard that never fires under the DP transform) is a one-line read in the same sweep.

### Band C — ε-affecting accounting: small diffs, but each needs its decision first

11. **OPQ-400 — decide which side moves before writing any code.** Either shift the released
    statistic to the ±1/2 form the analysis assumes (and divide by the public
    `expected_batch_size`), which fixes both halves at once, or keep the 0/1 indicator and correct
    `adaclip.rs` to drop the `/4` and charge the realized-batch worst case. The diff is small either
    way; choosing wrong means touching the Rust accountant and the runtime release twice.
12. **OPQ-403 / OPQ-420 — one guard, reported twice.** The code is trivial (restore a check in
    `Poisson.__post_init__`, so deserialization is covered too). The decision is which inner classes
    to reject and whether an explicit opt-in for the genuine dataset-level-coin model is worth
    keeping.
13. **OPQ-446 + OPQ-421 + OPQ-449 — sampler/accountant agreement.** OPQ-446 is mechanical and
    should go first: mirror the #1019 allocation-law guard into every remaining sampler codec and
    make `sample_rate` drift fatal rather than warn-only. OPQ-421 is a small derivation change.
    OPQ-449 (no derivation-version tag on MF noise state outside BISR) needs a versioning scheme and
    belongs with the design items — but BISR already shows what the scheme looks like. Fold
    **OPQ-450** in here too: optimizer-state restore silently degrading to template defaults is the
    same "resume succeeded, state did not" shape as item 1, and the same loud-on-mismatch remedy
    applies.
14. **OPQ-408 / OPQ-505 — the same `calibrate()` fallback defect, reported twice.** One decision:
    re-run the fallback search under the probe config, or report the honest union
    (`mc_failure_probability × probes_performed`). Either is a few lines plus a forced-fallback
    regression test.
15. **OPQ-428 + OPQ-429 + OPQ-430 / OPQ-512 — three engine sibling-path defects in two modules.**
    OPQ-428 is a `nan_to_num` call restoring the pre-#808 invariant; OPQ-429 and OPQ-430/512 are
    fail-closed guards of the kind `global_norm` already received. Cheap as a group, awkward as
    three tickets.

### Band D — coordinated sweeps, not individual tickets

16. **D1 — install and distribution sweep** (with item 5): OPQ-480/521, OPQ-481, OPQ-482 (no
    coverage guard on the wheel-build inventory), OPQ-484 (the torch-free `opaque-accounting`
    guarantee has no enforcement anywhere), OPQ-485 (`uv.lock` pins the last `accelerate` affected
    by a live advisory). One owner, one PR plus one external request.
17. **D2 — accounting and API documentation-truth sweep.** One editor, one pass over
    `docs/reference/` and `docs/user-guide/`, every claim re-executed as it is edited: OPQ-402,
    OPQ-406/424, OPQ-425/524, OPQ-497, OPQ-514, OPQ-516, OPQ-518, OPQ-519, OPQ-522, OPQ-523,
    OPQ-525, OPQ-526, OPQ-527, OPQ-536, OPQ-431, OPQ-432, OPQ-434, OPQ-498, OPQ-515. Carve out one code half:
    OPQ-519 also needs `CyclicPoissonSampler` to reject `truncated_batch_size` when `bands > 1`, so
    the unaccountable configuration cannot be constructed at all.
18. **D3 — notebook and executable-docs gate** (do with item 3): re-execute all seven tutorials in
    place and add every one to the CI execution list — OPQ-487, OPQ-494, OPQ-538, OPQ-528 (a
    maintainer's absolute local path and OS username are being served publicly). This is the sweep
    that turns "a published epsilon drifted again" into a build failure, which is why it ranks above
    everything else in Band D.
19. **D4 — CI fail-open sweep.** Small YAML changes, each individually trivial, collectively the
    difference between a green lane and a lane that ran something: OPQ-478/483 (Python 3.12
    advertised by eleven classifiers, executed by nothing), OPQ-486 (a GPU under 24 GB silently
    skips the whole Triton suite and still reports green), OPQ-488, OPQ-489 (the required Qodana
    check can never report on a fork PR, so the documented contribution path cannot merge).
20. **D5 — test-integrity sweep.** Lead with **OPQ-490**: the parametrised realized-stddev test is
    the single highest-value test in this list, because four injected noise-shortfall mutations
    survived the entire suite (§10.8) and today a 5 % shortfall on any of those sites ships green.
    Then OPQ-491, OPQ-492, OPQ-493 (two checkpoint tests that skip on exactly the symptom they
    exist to catch), OPQ-470, OPQ-475 + OPQ-537 (paired: both are the parity-exemption bookkeeping,
    same guard, same file), OPQ-476, OPQ-439, OPQ-423, OPQ-410, OPQ-411, OPQ-404.
21. **D6 — error-contract sweep** over the #760 taxonomy, one pass: OPQ-506, OPQ-507, OPQ-508 (a
    Rust `clamp` panic escaping as `PanicException`, which `except Exception` cannot catch),
    OPQ-509, OPQ-510 (a NaN laundered into a publishable-looking privacy number), OPQ-511,
    OPQ-412, OPQ-415, OPQ-417/513 (delete one stale allowlist entry — seconds of work, and it stops
    the error message recommending a removed, privacy-unsound knob), OPQ-416, OPQ-405, OPQ-409,
    OPQ-422, OPQ-407, OPQ-413, OPQ-451 (pre-release checkpoints failing with a bare `KeyError` or a
    bare version number instead of a migration instruction).
22. **D7 — alignment and auditing residue sweep**, one owner who holds both fix-verification
    contexts: OPQ-529, OPQ-530, OPQ-531, OPQ-532, OPQ-533 + OPQ-534 (paired: both are the #895
    dataset guard, one covers the missing half, the other strengthens the comparison from length to
    identity), OPQ-535, OPQ-536.
23. **D8 — repository and release hygiene sweep**, all public-facing, none technical:
    OPQ-426/496 (a 60 KB internal research diary at the public repository root, shipped inside the
    sdist), OPQ-427, OPQ-499, OPQ-500, OPQ-501, OPQ-503, OPQ-435, OPQ-477.

### Cheap and high-leverage — pull these forward regardless of band

- The `_IncompatibleKeys` assertion in item 1. One condition; converts a whole class of silent
  state loss into a raise.
- The `world_size > 1` sampler-restore refusal in item 2. About an hour; closes an ε exposure while
  the real design is debated.
- The two `torch.where` guards in item 4, copied from an existing sibling.
- Removing `"Cohere2Attention"` from the fuseable set (item 6).
- Delegating the fused-CE fallback to `original(...)` (item 7).
- The PEFT unwrap in `_is_hf_pretrained_model` (item 9).
- Deleting the stale `bound` allowlist entry (D6).
- The `--index-url` flags (item 5).
- **OPQ-490's** parametrised realized-stddev test (D5) — the highest guarantee-per-hour item in the
  list that is a *test* rather than a fix.

### Needs a design decision before any code is written

Do not assign these as implementation tickets. Each one has at least two defensible resolutions, and
picking the wrong one costs more than the defect.

- **OPQ-400** — move the released statistic, or move the accountant.
- **OPQ-403 / OPQ-420** — which inner processes `poisson()` must reject, and whether a
  dataset-level-coin opt-in survives.
- **OPQ-408 / OPQ-505** — probe-config re-search versus an honest union bound.
- **OPQ-438** — raise, warn once, or let the noise mechanism learn the optimizer's capability, when
  a paid-for second-moment stream reaches a scaler that cannot consume it.
- **OPQ-447** — per-rank snapshot files versus a rank-independent base key folded at restore.
- **OPQ-449** — the derivation-version / execution-identity scheme for MF noise state (BISR is the
  precedent to generalize).
- **OPQ-454** — the `compute_loss_func` rank contract: vmap the user callable, or keep the batched
  call and validate it.
- **OPQ-461 / OPQ-466 / OPQ-472** — whether per-family behaviour lives in a routing table that
  forces an explicit decision per family, or stays in name sets and duplicated predicates. These
  three are the same architectural question asked in three places, and answering it once is what
  stops the next throughput commit from producing the next OPQ-465.
- **OPQ-481** — who owns defensive name registration, and where that becomes a release checklist
  item rather than a person's memory.
- **OPQ-502** — the repository squash setting discards every PR body, which contradicts `AGENTS.md`,
  the PR template and `git-cliff`'s `BREAKING CHANGE` detection. Repository-admin decision, not a
  code change.
- **OPQ-414 (PLAUSIBLE)** — decide whether checkpoint-budget-wins is intended. If it is, document it
  and keep the test; if it is not, it is a fail-open. Either outcome closes the finding; leaving it
  undecided is the only wrong answer.
