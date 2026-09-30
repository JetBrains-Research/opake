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