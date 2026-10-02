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

## 2026-09-30 — per-phase memory profiling added; checkpointing variants measured (A100-40GB)

- Added per-phase memory instrumentation: `step_perf(..., phase_memory=True)`
  records a peak per `.mark()` segment (high-water since the previous mark, not
  a cumulative step peak), `perf_tracker` gained `phase_memory`, and
  `start_snapshot`/`finish_snapshot` wrap CUDA `_record_memory_history` /
  `_dump_snapshot`. All three are opt-in and no-ops off CUDA.
- `train_dpsgd.py --memory-profile PREFIX` (CUDA-only, others skip with a
  notice) records `--memory-profile-steps` steps after skipping
  `--memory-profile-skip`, and writes a per-phase peak/time CSV, a 1 ms
  live-memory timeline CSV, and the CUDA snapshot pickle for that window.
  `examples/plot_memory_profile.py` and `examples/plot_memory_compare.py`
  render them; both read CSVs only, so they run anywhere. Tests in
  `packages/opake-engine/tests/profiling/test_phase_memory.py`.
- Dropped the scratch `bench_memory.py` harness. The three claims it was built
  on (autograd graph retained through microbatching worth 6.2 GB;
  per-microbatch peak equal to full-batch peak divided by mb;
  frozen-params-cast worth 14.2 GB) were assumptions, not measurements, and
  none of them survived contact with the profiler. Nothing in the repo ever
  imported `dpsgd_stages`, which that harness imported.
- Four variants measured on Qwen2.5-Coder-7B LoRA, seq 512, mb=2, same
  instrumentation (details in `notes/checkpointing-open-questions.md`):
  no checkpointing 22.43 GiB / 10.19 s; attention-only checkpointing 22.43 GiB
  / 10.17 s; full-layer checkpointing 16.39 GiB / 14.12 s; eager (no Triton
  kernels) 24.40 GiB / 8.26 s. Static floor 15.0 GiB in all four.
- Reading: the peak is **sawtooth, not cumulative** — one spike per microbatch
  chunk returning to the floor, flat across steps, so activations are released
  during backward and there is no retained-reference leak. Attention-only
  checkpointing never fired: the model trains in eval mode (`from_pretrained`
  returns eval, PEFT leaves base submodules in eval, `train_dpsgd.py` never calls
  `.train()`), so the wrapper's `module.training` guard bypasses every call.
  Full-layer checkpointing escapes this only because opake's
  `gradient_checkpointing_enable` patch flips `.training` on the layers.
  Confirmed on a CPU replica; see `notes/memory-profiling-report.md` §5.
  (Fixed and re-measured in the next entry.) Full-layer checkpointing works but costs 38%
  step time to save 6 GiB. The Triton kernels cost throughput at seq 512, the
  opposite sign to the seq-4096 sweep. At this shape ~16 GiB sat unused while
  GPU utilisation averaged 32.1% (50 one-second `nvidia-smi` samples, max 64%,
  16 at the 11% floor), so memory was not the constraint.

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

## 2026-10-01 — train-mode fix in `train_dpsgd.py`; attention-only checkpointing measured (A100-40GB)

- Fix: `examples/train_dpsgd.py` now calls `model.train()` before
  `make_functional`. Without it the base model trained in eval mode
  (`from_pretrained` returns eval; PEFT keeps base submodules in eval), which
  made training-gated code no-op: the attention-checkpoint probe bypassed every
  call, and the `kv_cache` patch left `use_cache=True`. Safe because dropout is
  already zeroed in the config and `clipped_grad` vmaps with
  `randomness="same"`. Per-example grads are bitwise equal in eval and train mode.
- `experiments/mem_profile/check_attn_ckpt_fires.py`: CPU self-check that
  replicates the `train_dpsgd.py` model setup and asserts the probe is inert in
  eval mode, fires on every layer in train mode, and leaves grads bitwise
  unchanged. Passes locally and on the workspace.
- Re-profiled all four variants in one session (Qwen2.5-Coder-7B LoRA, seq 512,
  mb=2): no checkpointing 22.43 GiB / 11.67 s; attention-only checkpointing
  **21.60 GiB / 14.80 s**; full-layer 16.39 GiB / 15.94 s; eager 24.40 GiB /
  9.25 s. Attention-only saves 0.83 GiB for +27% step time, i.e. 14% of
  full-layer's saving for 73% of its time cost, so it isn't worth it at this
  shape. Peaks of the unchanged variants match the 09-30 batch to four decimals,
  so train mode changed no memory. Times run ~12–15% slower across the board this
  session, so compare within a batch only.

## 2026-10-01 — underflow note, issues #1119/#1120, kernel-optimizations review + port

- Underflow: documented as a limitation (clip_pytree docstring, precision.md,
  notes/clip-norm-underflow.md); no behavior change.
- Issues: #1119 (float16 subnormal guard zeroes instead of rounding down;
  utility only) and #1120 (per-leaf clip-and-reduce overhead; links the
  fused-engine evidence). None opened for the eval-mode trainer (owner's) or
  the underflow (now a note).
- mihajlo/kernel-optimizations review (experiments/kernel_opt_review/):
  ported the BLT/Toeplitz coefficient cache (fixed: (device, dtype) keys,
  lazy fill, version- and grad-aware) and elementwise BLT decay as 65fd7f28.
  Bitwise-equal outputs; A100 BLT multiply_next ~112 -> ~38 ms. Not ported:
  broadcast-sum (slower on CUDA), flat i.i.d. draw (slower, 2x memory,
  changes the RNG stream for non-16-multiple leaves, a λ-CGD resume
  hazard), unused generator_from_key(device=). Its "pre-existing" failures
  were introduced on that branch (main 623/623). 18 regression tests fail on
  the source branch, pass on main.
- Ported 3a9bdcdc (attention-only checkpointing, lora_mlp_recompute) as
  a05b0b8e: reviewer's CRITICAL claim refuted by instrumentation;
  save_intermediates honored under vmap(grad()); no new failures in A/B.
  The 36 failures in both arms also fail on main in that environment. The
  doc now states that attention checkpointing needs training mode and no KV
  cache. Stale sweep notes not ported.

## 2026-10-01 — attention-only checkpointing removed

- Removed the `attention_checkpointing` option ported in `a05b0b8e`: the
  `apply_model_patches` kwarg, `transformers/components/attention_checkpoint.py`
  and its tests, the `TrainingArguments` torch_compile guard and its two tests,
  and the docs row. Also removed the `--attention-checkpointing` probe and
  `experiments/mem_profile/`. `lora_mlp_recompute` and the `train_dpsgd.py`
  `model.train()` fix stay.
- Why: selective recomputation (Korthikanti et al., arXiv 2205.05198 §5)
  recomputes only the s×s core-attention ops and stores the projections. Opake
  runs `sdpa`, which dispatched to its fused flash kernel on our A100 runs.
  That kernel already never stores the attention matrix, so wrapping the whole
  attention block only re-ran the Q/K/V/O projections: −0.83 GiB for +27% step
  time at seq 512, mb=2. The docs now say why no attention-only option exists.
- Unreleased (in no tag, only on this branch), so no deprecation path. The
  `opake-patches` + `opake-transformers` PR-marker suites pass: 1436 passed,
  0 failed.

## 2026-10-01 — `lora_mlp_recompute` measured; DPTrainer eval-mode and fused-LoRA dtype issues filed

- DPTrainer path (`train_dpsgd_trainer.py`, bf16 autocast, Qwen2.5-Coder-7B
  LoRA on all 7 projections, A100, seq 512, mb=2, identical Poisson batches):
  `lora_mlp_recompute=False` gives **−9.9% step time (14.60 → 13.16 s) for
  +2.08 GB step peak (17.02 → 19.10 GB)**, faster on all 6 steps. That matches
  the 2 × 28 × 18944 × 2 B per token per example estimate. Worth enabling when
  the extra memory fits. New flag `--lora-mlp-recompute` in
  `train_dpsgd_trainer.py`.
- The same knob is inert in `train_dpsgd.py` (identical peaks and times): bf16
  base + PEFT's default fp32 adapters + no autocast fail the fused LoRA
  MLP/Linear dtype gate, so those kernels never run there. The briefly added
  `train_dpsgd.py` flag was removed again. The user guide now documents the
  dtype condition and the memory cost. Issue #1122.
- Confirmed that DPTrainer (and SFTTrainer) trains with the base model in eval
  mode: a CPU repro via `from_pretrained` + PEFT shows
  `self_attn.training=False` and a `DynamicCache` in the training forward. Test
  fixtures construct models directly and hide it. Issue #1121, not fixed here.
- Workspace synced to the branch via git bundle; its prior local state is
  backed up in `~/opaque_backup_20261001`.

## 2026-10-02 — fused clip backend: full-step profile (A100-40GB, report §11)

- `--clip-backend {torch,auto,triton}` added to `train_dpsgd.py` and
  `train_dpsgd_trainer.py` (`b6f83956`). Ten runs in one session with
  identical Poisson batches per script; triton runs strict, so every clipping
  call ran fused.
- `train_dpsgd.py` (Qwen2.5-Coder-7B LoRA, seq 512, mb 2, profiler on):
  11.10 → 7.35 s per step (×1.51, faster on 6/6 steps), the `clip` phase
  −3.76 s, peak unchanged at 22.43 GiB. Eager: 9.10 → 5.43 s (×1.68).
  Eager + triton against the kernels + torch baseline: ×2.05,
  1.37 → 2.80 smp/s. Clean runs: 1.50 → 2.30 smp/s (+53%).
- `DPTrainer`: 14.42 → 10.18 s (×1.42, peak unchanged at 17.03 GB).
  `lora_mlp_recompute=False` adds only −1.5% on top of triton (−5.5% without).
- Next bottleneck: the vmapped per-example forward/backward (GPU util ≤44%).
  Workspace stopped after the runs.

## 2026-10-02 — `lora_mlp_recompute` removed; runtime header fixed

- With the fused clip backend, `lora_mlp_recompute=False` saved only 0.16 s
  per step (steps 2–6, paired; −1.5%) for +2.08 GB, down from 0.48 s without
  it. It was removed (`c9966971`): patches files restored to `main`, example
  flag and user-guide rows dropped. Report §10, §11 and the header updated.
- `train_dpsgd.py` printed "Kernel optimizations: enabled" even with
  `--no-kernel-patches`, because only the environment switches were checked.
  Fixed (`89ea8f88`). The §11 "eager" runs were confirmed eager by their
  later `Kernel patches: DISABLED` line.
- Report terminology: "eager" means `--no-kernel-patches`. Attention is
  `sdpa` in every run.

## 2026-10-02 — multi-tensor clip kernel and vmap(chunk_size) explored (report §12)

- Multi-tensor prototype (`experiments/clip_fused/mt_engine.py`): every leaf
  of one dtype per launch via tile + pointer tables, plus shared dtype markers.
  393 of the remaining 423 kernels per call were per-leaf marker fills.
  Per call: 93.6 → 13.1 ms, 2,767 → 31 kernels. 25/25 gates. (Corrected
  later: clipped sums and norms are bitwise identical; the `clipped_norms`
  diagnostic differs by up to 2.4e-7.)
- 7B full steps, same session, identical batches: eager 5.59 → 4.80 s
  (−14.2%, 6/6 steps), kernels 7.59 → 6.93 s (−8.8%), DPTrainer 10.60 →
  9.84 s (−7.2%). Peak memory unchanged. Clean eager throughput 3.10 →
  3.60 smp/s.
- `vmap(chunk_size=2)`: −10–13% alone, but +2.3 GiB peak, and it adds
  nothing meaningful on top of the multi-tensor kernel (+1.7% eager, −4.8%
  kernels). Not pursued.
- Workspace stopped after the runs.

## 2026-10-02 — multi-tensor clip kernel packaged (report §12.3)

- `6f339050`: packaged the multi-tensor clip kernel in `_clip_sum`, with
  shared markers, a same-device guard and an LRU plan cache. A foreach
  microbatch accumulator went in the same commit.
- Found an overclaim and corrected it. "Output bitwise identical" was wrong:
  clipped sums and norms are bitwise equal (16 cases); the `clipped_norms`
  diagnostic differs by up to 2.4e-7. `bench_mt.py` never compared norms
  bitwise. Report, plan and this log are corrected. Docs and docstring now
  state where the fused path can differ from torch (`ddd3bece`).
- 7B, same session: eager 5.49 → 4.88 s, kernels 7.68 → 6.84 s, DPTrainer
  10.62 → 9.68 s, clean 3.10 → 3.60 smp/s. Memory unchanged.
- Foreach accumulator: 4.88 vs 4.85 s/step, no gain. Reverted
  (`a530ca2d`). This also refutes the §12.2 inference about where chunking's
  remaining gain comes from.
- Suites on A100: engine/dpsgd/dpftrl 2,166 passed, 0 failed. Transformers
  and patches: the isolated failing subset has 60 failures (new) vs 62 (old);
  the extra two old-code failures are flaky timing tests. No new failures.
- Workspace stopped after GPU work.

## 2026-10-02: kernel and host-overhead targets (outputs/kernel-optimization-targets.md)

- Filed #1123: the clipping-norm underflow decision, with the evidence and the four fix options. #1119 already holds the float16 decision.
- Kernel-level profiles of full 7B steps (per-step `torch.profiler` hook in `run_train_variant.py`). GPU busy: 51% (eager, mb 2), 33% (model kernels on), 26% (`DPTrainer`). GPU work is about the same in all three (2.1–2.5 s); the rest is host time.
- Root cause: each custom `autograd.Function` costs about 0.6 ms of host time per call under `vmap(grad)` on the A100 host (75 µs on a Mac), about 20× a plain op. That is why Opake's kernels make the step slower under DP-SGD.
  - Caching functorch's per-call generated classes is correct (bitwise equal) but saves only 13%.
  - `torch.library.custom_op` with `register_autograd` fails under `torch.func.grad`.
- Measured levers (`train_dpsgd.py`, clean baseline 4.86 s):
  - mb 4: −33%.
  - Stock PEFT LoRA: −13% at mb 2.
  - TF32: +2% at mb 2, −6.8% on top of mb 4.
  - Best: stock PEFT + TF32 + mb 4, 2.92 s (−40%).
  - `DPTrainer`: mb 4 + stock PEFT is ×2.06 (9.76 → 4.74 s).
- `torch.compile`: needed `g++` on the workspace (installed). It fails with Opake's PEFT kernels (in-place `addmm_` under Dynamo). With stock PEFT it is ×0.36 of eager in steady state, with a graph break at `_clipped_fun.py:213` and recompiles on batch size. `reduce-overhead` is worse.
- SDPA backward has no functorch batching rule in torch 2.14 (upstream PR #176265 is open); attention is only about 2% of GPU time here.
- Hypothesis refuted: class generation is *not* the dominant part of the custom-Function overhead.
- Blocked: the `pi-subagents` install is missing runner files, so no subagent delegation; the research was done directly.
- Workspace stopped.

## 2026-10-02: T3, CUDA-graph capture of the per-microbatch chunk (outputs/kernel-optimization-targets.md §8)

- Prototype `experiments/cuda_graph/cudagraph_chunk.py`, passed as `_chunk_compiler`.
  - Keyed on structure, tensor metadata and non-tensor values.
  - Static input copy (including the clipping-norm tensor), output clones.
  - Warm-up only before the first capture.
- Production fixes needed for capture, each with a CUDA test that fails on the old code:
  - `1af7ab04`: vmap causal mask without host-to-device copies.
  - `846db2cd` + `876ac00b`: fused clip pointer table through a pinned arena. A first version tripped a PyTorch host-allocator assert when graphs were destroyed; fixed.
- Measured on 7B, same session, identical batches:
  - `train_dpsgd.py`: −40.6% at mb 2 (estimate was −40%).
  - Best: graph + fused clip + model kernels + TF32 at mb 4, 2.26 s and 6.41 smp/s, ×2.21 vs eager mb 2 and ×1.27 vs the best eager configuration.
  - Opake model kernels: +42% eager → −6.6% in the graph.
  - `DPTrainer` (fused linear CE off): ×3.14 at mb 2, ×1.86 at mb 4.
- Correctness:
  - Small model bitwise equal (fixed and adaptive, partial microbatches, torch and fused clip).
  - 7B: graph-vs-eager equals eager-vs-eager.
  - New finding: the eager mb-2 kernel itself is nondeterministic at 0.2–1% relative L2.
- Blockers / findings:
  - Fused linear CE uses `nonzero()` (`_utils.py:138`), so it cannot be captured; `DPTrainer` needs a mask-based variant.
  - Check mode runs out of memory at mb 4 (two extra eager runs).
  - Mistake: my first summarizer missed the third capture row at mb 4; fixed with a predicted-and-asserted capture schedule.
- Workspace stopped.

## 2026-10-02: next-edit smoke comparison on the new opake (in progress)

- Baseline found: 2026-09-25 ckpt-sweep `baseline` (ZenML `06ef7070`, W&B `ada8c57f`, image `…training@sha256:87fab1f2…`, opaque 0.15.6rc1).
  - H100, seq ≤ 4096, LoRA rank 384, logical batch ~256 in microbatches of 2, checkpointing on, `chunked_nll`.
  - Steps 208–229 s, batches `[265, 270, 249, 251, 272]`.
  - Parameters saved to `notes/smoke-compare/baseline-zenml-config.json`. Its trainer args include `log_completion_metrics: false`, i.e. the uncommitted next-edit pass-through.
- next-edit branch `mihajlo/opake-smoke-new` (`f1fc4ec4a`, a worktree off `9321ea13a`; the user's working tree is untouched).
  - Migration: `opaque` → `opake`; clipping keys `target_quantile` / `clipping_norm_max`; `clipbound_learning_rate` accepted only at 0.2; Opake-only trainer-args pass-through; `privacy_accounting: true` (0.16 keeps no accountant for fixed-noise DP-SGD by default).
  - Vendored opake `0.16.1.dev80+gef183f25` wheels (manylinux 2_28 x86_64 + macOS arm64 accounting + sdist). The lock diff is exactly the opaque → opake swap.
  - Configs: `sft_opake_dp_cmp_{a,b,c}` (A = baseline replica, B = + fused clip, C = + CUDA graphs).
  - Verified by composing the configs and diffing them against the baseline parameters.
- Wheels validated on the next-edit stack (torch 2.9.0, Triton 3.5.0, transformers 5.11.0): 260 passed.
- next-edit tests: 261 passed; 6 DPO/KTO failures are pre-existing on `9321ea13a`.
- Image #1 (A/B): GitHub run 37054674512.
- CUDA graphs packaged (`0159293d`, tests fixed in `6f7c3bf4`):
  - `CudaGraphChunkCompiler`, a `DPTrainer` `cuda_graphs` option, a capture-safe fused-CE mask path, and checkpointing without RNG state, guarded by a dropout check.
  - A100 `DPTrainer` with checkpointing and fused CE on: ×3.7 steady.
  - Capture probe: checkpoint RNG saving is not capturable; `preserve_rng_state=False` is exact without dropout.
  - Fused-CE gradients are not bitwise-reproducible even eager-vs-eager (atomics; max 3.8e-6).
- The user confirmed speed is the target and bitwise equality is not required, as long as privacy holds.

## 2026-10-02: next-edit smoke results (report §9)

- A (opake, same recipe): ε identical to the baseline (10293.44); eval loss within 0.3%; peak memory 46.8 vs 54.0 GB. The per-sample time ratio (×0.95) is confounded because opake 0.16's sampler draws different batches.
- B (+ fused clip): −9.2% step time vs A, exact and paired (0.908–0.910 on every step).
- C and D (`cuda_graphs`): failed in the Adam update after step 1 (likely OOM; 73 GB reserved). Variable-length microbatches mean a capture per shape, each holding ~3.9 GB of rank-384 outputs. Next: shared static buffers and length bucketing.
- ZenML runs: A `870a5a5a`, B `416ecfd8`, C `879b748b`, D `86979dc4`. Images: `sha256:dc825d30…` (A, B), `sha256:488960c5…` (C, D).
- Workspace stopped.
