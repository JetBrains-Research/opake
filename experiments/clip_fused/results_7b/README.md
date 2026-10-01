# Fused clip engine on the real 7B workload — raw artifacts

Date: 2026-10-01. Host: Coder workspace `kernel`, A100-SXM4-40GB, torch 2.14.0.
Workload (as in `notes/memory-profiling-report.md`): Qwen2.5-Coder-7B, LoRA r16
on q/k/v/o/gate/up/down, seq 512, microbatch 2, batch 16, bf16 base, adaptive
clipping with `return_aux=True` (trainer default). The trainer runs the model in
eval mode (KV cache built each forward) — identical for both arms.

Engine: `experiments/clip_fused/fused_engine.py` md5 `b6fe7bc1…` for all runs
below (the later `818ecab7` adds only an unreachable B>65535 fallback; it was
re-validated with `check_bound_strict.py` and `bench_fused_e2e.py --quick`).

Driver: `fe_driver.sh` (copied here), common flags:
`--preset qwen-7b-kstack --max-seq-len 512 --microbatch-size 2 --batch-size 16
--num-train-samples 2000 --num-eval-samples 100 --eval-steps 100`

| File | Run | Notes |
|---|---|---|
| `fe_base.{log,csv,timeline.csv}` | R1 stream, `--memory-profile` skip 3 / steps 2, 6 steps | report protocol |
| `fe_fused.{log,csv,timeline.csv}` | R2 fused, same | identical Poisson batches |
| `fe_attr_base.log` | R3 stream, `OPAKE_SEAM_TIMING=1` | synced seam timing (attribution only) |
| `fe_attr_fused.log` | R4 fused, same | |
| `fe_clean_{stream,fused}.{log,smi}` | R5/R6, 8 steps, no instrumentation, `nvidia-smi -l 1` | throughput + utilization |
| `fe_shadow.log` | both engines on identical inputs at every seam call | correctness on real grads |

CUDA snapshot pickles (`fe_base.pickle`, `fe_fused.pickle`, ~26 MB each) remain on
the workspace in `~/opaque/runs/`.

Utilization caveat: `.smi` summaries use "memory.used > 20 GiB" as the filter for
the stepping phase, which also includes eval/setup while the model is resident;
compare stream vs fused with the same filter only, not against the report's 32.1%.
