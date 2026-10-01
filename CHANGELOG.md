# Lab notebook

Chronological log of meaningful progress, failed approaches, and major
verification results for the DP-clipping optimization workstream. Newest last.

## 2026-09-29 — baseline breakdown + oracle rig created (MPS)

- Added `experiments/clip_breakdown/bench_clip_breakdown.py` (step split:
  non-DP backward / vmap-grad / clip-stream / clip-legacy / microbatched) and
  `experiments/clip_oracle/run_oracle.py` (fp64-oracle parity + `‖clipped‖ ≤ C`
  stored-value invariant + noise-convention probe).
- First full matrix run on Apple-silicon MPS. Clip pipeline ≈ 82–89% of DP
  step time in large-leaf configs; legacy clip path 7× slower than streaming
  at B=64 (memory-bound on MPS). Harness written for cpu/mps only.

## 2026-09-29 — CUDA replication: first run DISCARDED (timing methodology bug)

- Ran the breakdown on the A100 40GB Coder workspace. Initial interpretation
  ("materialization ≈ +1.5 ms; legacy faster than streaming; A100 is 60×
  faster than MPS at non-DP backward") was **wrong**: the harness's `sync()`
  only called `torch.mps.synchronize()`, so CUDA timings measured kernel
  enqueue plus incidental syncs, with queue drain bleeding across paths.
  Flat batch-invariant `t_nodp`/`t_grad` was the tell.
- Fix (`7a1fa0e8`): `torch.cuda.synchronize()` in `sync()`, plus empty-queue
  sync before each timed iteration and a per-path CUDA peak-memory probe.
  Commit + remote pull + full matrix re-run.

## 2026-09-29 — corrected CUDA baseline + oracle (A100 40GB, torch 2.14.0)

- Corrected split at few-large B=64: materialization +9.5 ms vs clip machinery
  +171.5 ms — the clip-machinery ≫ materialization conclusion survives but
  with an 18× margin, not the artifact-inflated 144×.
- New fact from the memory probes: B=64 clip passes peak at 27 GB (stream) /
  35 GB (nostream) of the 40 GB card at seq=64 synthetic data. Memory, not
  time, is where per-example materialization hurts on CUDA.
- Legacy nostream path is faster than streaming on CUDA in all four configs
  (opposite of MPS large-leaf) → fused-kernel bar on CUDA is min(stream,
  nostream) at ≤ stream-path memory, not "beat streaming".
- Oracle rig gained `--cuda`; full CUDA run: **0 failures**, including strict
  fp32 1e-4 near-C configs that marginally missed on MPS — confirms those MPS
  dings are the fp32 accumulator budget (backend lacks fp64), not a path bug.
- Plan updated (r3 CUDA section); v1 = fused in-vmap clip confirmed with
  target data; v2 (per-layer) demoted to memory-lever candidate pending
  long-sequence/large-B evidence.

## Open / next

- Prototype fused in-vmap norm+scale kernel (Phase 1.1 alternative path) —
  correctness-gated by `run_oracle.py` before any perf claim.
- Longer-sequence / larger-B CUDA configs to test whether the materialization
  time share ever becomes dominant (v2 gate).
## 2026-09-29 — fused clip prototype v0: 8/8 correctness, 7–25× op-level (A100)

- `experiments/clip_fused/fused_clip.py`: 3-launch deterministic replacement
  of the per-leaf clip+sum chain (K1 partial sq-sums, C1 tiny combine+scale
  in fp64, K2 column-slab fused sanitize+scale+sum with b-ascending order,
  no atomics, no clipped-copy or sanitized-copy materialization).
- All 8 matrix cases PASS on A100: fp64-oracle parity, chain-parity, stored
  bound ‖clipped‖ ≤ C (bf16 lands 0.995–0.997, i.e. under), bitwise
  determinism. bf16 fused-vs-oracle deviation is IDENTICAL to the current
  chain's (shared bf16 storage rounding, not kernel error).
- Speed vs reconstructed chain: big fp32 leaf x7.2, big bf16 leaf x25.4
  (chain was launch-bound), total matrix x12.5. Near roofline: 4.3GB leaf,
  2-pass streaming ≈ 84–85% of A100 HBM bandwidth.
- Peak memory per big leaf: 4.2GB fused vs 20.5GB chain (no materialization).
- Infra note: Coder SSH ProxyCommand became flaky mid-session; working path
  is `coder ssh mihajlolinic/kernel -- "cmd"` (direct CLI, bypass ssh config),
  file transfer via base64 inline with md5 verify.
- NOT yet done: integration through the real `clipped_grad`/pipeline seam
  (needs a chosen override point), e2e step benchmark, per-tree roundoff
  constant, small-leaf launch diet.

## Task ledger (current)

- [x] baseline MPS + CUDA breakdowns (synced), oracle rig (cpu/mps/cuda)
- [x] fused clip prototype v0 — correctness gates green, op-level speedup
- [ ] pipeline-seam integration + e2e step benchmark (A100)
- [ ] small-leaf launch diet (fold C1; maybe CUDA graphs for 48-leaf trees)
- [ ] long-sequence configs to test v2 (per-layer) dominance gate

## 2026-09-29 — fused engine via real clipped_grad seam: 5/5 e2e PASS (A100)

- Architecture decision (user): kernel engine lives in opake-engine
  ("custom triton kernel engine", algorithm-agnostic), the mechanism CHOICE
  lives in opake-dpsgd. Partition-policy compliant; DP-FTRL can consume the
  same seam later.
- experiments/clip_fused/fused_engine.py: drop-in replacement for
  _stream_clip_and_sum (tree-level per-example norm, fp64 scalar chain with
  tree-level roundoff + conservative guard shrink, fp32 accumulation,
  no value division; PerGroup/AUTO-S/second_moment/aux/complex -> original).
  Monkeypatched at the module-global seam in bench_fused_e2e.py.
- e2e (real clipped_grad, synced): few-large B=16 56.3->15.0ms (x3.75);
  few-large B=64 fp32 219.5->55.4ms (x3.96), peak 27.6->9.8GB;
  few-large B=64 bf16 176.1->16.3ms (x10.8), peak 23->4.9GB; many-small
  fp32/bf16 x2.7/x2.8. maxrel dev 3e-7 (fp32) / 1e-3 (bf16); drift=0 all.
- Known gaps before packaging: aux/second-moment fallback, per-leaf scale
  for mixed-dtype trees (currently conservative), many-small still
  launch-bound (2.7x only), small remaining B=64 peak dominated by the
  input [B,...] stack (removable only by v2 per-layer fusion).

## 2026-10-01 — correctness review of the fused engine + real 7B workload (A100-40GB)

- Review found the v0 engine (d6be8dca) was bypassed by every trainer mode:
  `train_dpsgd.py` passes `return_aux=True` (adaptive also forces inner
  `return_stats`), which v0 treated as unsupported. The 3.96x was real but
  only for fixed-mode clipped_grad without aux. Fixed: aux/stats supported.
- Privacy-relevant: v0 accepted fp16 leaves with fp32 guard constants and no
  fp16 subnormal nudge. New strict test (check_bound_strict.py: B=1 trick,
  fp64 norm, zero tolerance) shows 297/1900 violations for v0, all fp16
  (100/100 just above C: a ~0.9999 scale is below fp16 half-ulp, so the
  example passes unclipped). Current engine: 0, production: 0. fp16 now
  falls back. Latent: no prior result used fp16.
- Also fixed: guard is now production `_finalize_scale` per storage dtype
  (exact for mixed trees; v0 over-shrank mixed trees 14x), compute_dtype /
  output-dtype / requires_grad / B>65535 fallbacks, microbatch reducer fp32
  output, K1 grid axes. Independent reviewer subagent: 2 findings rejected
  with evidence (adaptive != AUTO-S; bf16 ulp is 7.8e-3 not 1e-3), B>65535
  guard accepted.
- Doc corrections: MPS mb8 slowdown was 13.3x not "250%"; roofline ~90% fp32 /
  ~84% bf16 (not 84-85%); memory figures were MiB mislabeled as GB; "step"
  speedups were clipped_grad-call speedups.
- e2e 12/12 PASS incl. adaptive+aux+mb8 (trainer path) with fused dispatch.
- 7B LoRA (seq 512, mb=2, B=16, trainer as-is): seam 478 -> 78.5 ms/call;
  step 11.35 -> 7.66 s; throughput 1.34 -> 1.98 samples/s; peak unchanged
  22.43 GiB. Shadow A/B on real grads: norms bitwise equal. A step-4
  GradNorm blip (0.228 vs 0.216) in one fused run was trajectory
  nondeterminism (second fused run 0.217; shadow proves outputs equal).
- Interpretation caveat (from the checkpointing investigation): the trainer
  runs in eval mode, so every per-example forward builds a KV cache; both
  arms pay it. Next bottleneck after the engine is the vmapped per-example
  fwd/bwd (~6.5 of ~7.1 s fused clip phase), not clipping.

## 2026-10-01 — adversarial bound review: v0 fp16 severity + production underflow corner

- check_bound_adversarial.py (zero tolerance, B=1 trick): v0 fp16 violated
  in every regime the guard exists for: normal-range round-back +4.4e-4
  (deterministic, ~fp16 half-ulp), subnormal round-up +19% (1.19C passes
  unclipped), and a realistic C=0.01 / 4M-element leaf +1.0e-3. The earlier
  "+1.1e-4" came from a benign normal-range config and understated it.
  Blast radius verified zero: no v0 commit touched packages/, no PR, no
  v0-era rig or engine references fp16, every real run saw fp32 leaves only.
- Production (and the current engine, which reuses production's guard)
  also fail case A2 for bf16/fp32: per-example squares underflow in the
  fp32 accumulator, computed norm = 0, example left unclipped
  (probe_norm_underflow.py). Onset at per-element RMS ~1e-21 (fp32) /
  ~1e-23 (bf16), i.e. C <~ 1e-18 / 1e-20 for N = 1M. Unreachable in
  practice (lost squared mass <= N*1.2e-38, harmless while
  C >> ~2.4e-16*sqrt(N); adaptive clipping floors C at 0.01), but it
  contradicts clip_pytree's "holds for every input" docstring. Not changed:
  production edits need owner sign-off + DP review.
- Production utility note: for fp16 storage, the subnormal guard ZEROES a
  rounded-up element rather than rounding it to the next lower subnormal.
  With small C this deletes most of the signal (A3: an example 0.1% over C
  is stored at 3% of C). Private but utility-harsh.
