# Selective recomputation sweep — Qwen2.5-Coder-7B LoRA r384 DP-SGD (2026-09-25)

Recipe: next-edit `sft_opaque_dp_trace_v20260605_accepted_n001` via the smoke
recipe (`sft_opaque_dp_ckpt_sweep_*.yaml`, branch `mihajlo/opaque-ckpt-sweep`):
microbatch 2, gradient accumulation 128, seq 4096, adaptive clipping, noise
multiplier 0.01, 5 optimizer steps, trace stack (1× H100 80 GB). Image built
with `opaque-patches 0.15.6rc1+ckptopt1` (this branch's patches backported to
0.15.6rc1), torch 2.9.0, transformers 5.11.0.

## Results (mean of 5 logical steps, `step_time_sec` from the trainer log)

| Variant | Step time | vs current | Per-step peak allocated | Peak reserved |
|---|---:|---:|---:|---:|
| Current: full layer checkpointing | 221.2 s | 100% | 53.98 GB | 55.3 GB |
| Checkpointing off | 174.7 s | 79.0% | 54.2 GB | 55.7 GB |
| Off + Triton kernels (`use_performance_kernels`) | 163.3 s | 73.8% | 53.98 GB | 55.0 GB |
| Off + kernels + `lora_mlp_recompute=False` | **152.7 s** | **69.0%** | 60.4–66.7 GB | 67.8 GB |
| … + `attention_checkpointing=True` | 152.3 s | 68.9% | 60.4–66.7 GB | 67.8 GB |
| Off + kernels + `attention_checkpointing=True` | 163.5 s | 73.9% | 53.98 GB | 55.0 GB |

ZenML runs: baseline `06ef7070`, off `dc60eb01`, off_kernels `f925f0eb`,
mlp_save `3e8b55ce`, mlp_save_attn `cc588991`, off_kernels_attn `c0848f3e`.

- Training is unchanged across variants: final loss 0.0462–0.0463,
  eval loss 0.0266 ± 0.0001, epsilon 10293 in all six runs. Clip rate is
  0.029–0.033 at the last step because floating-point differences move
  borderline examples across the adaptive threshold.
- Peak memory is the same (≈54 GB) with or without full checkpointing, so the
  peak happens during per-example gradient clipping, not while activations
  are saved. The fused LoRA path was active: with it, the saved activations
  are about 14 GiB at microbatch 2, not the ≈40 GiB assumed earlier.
- Keeping the MLP `gate`/`up` tensors (`lora_mlp_recompute=False`) costs
  +12.8 GB of peak memory; the estimate was ≈17 GB. It is the fastest variant
  that fits.
- Compared with the compute-only estimates: checkpointing off gave 79% of the
  current step time (estimated 75%), and the MLP save mode gave 69%
  (estimated 61%). Clipping, noise and optimizer time do not shrink with
  recompute savings, which accounts for the gap.

## Open issue: attention checkpointing did not engage on the GPU runs

The attention variants match their counterparts to within ±1 s per step, and
their per-step peaks are identical to 0.01 GB. No recompute happened.
Locally (CPU), the same trainer path does checkpoint every attention call
(`past_key_values=None`, `use_cache=False`, grad enabled). That holds both on
current opake and on the image's exact torch/transformers versions with the
patched 0.15.6 sources. The difference is therefore specific to CUDA with the
Triton kernels. Candidates are the fused-add RMSNorm decoder forward and the
CUDA-only fused LoRA QKV path. Next diagnostic on a GPU: run one microbatch
with a spy on `attention_checkpoint._has_kv_cache` and
`torch.utils.checkpoint.checkpoint`, following the local reproduction pattern.
Since peak memory is not activation-bound, attention checkpointing is not
needed for the current microbatch.

## Assumptions made while unattended

- "Fused RMSNorm" was taken to mean `use_performance_kernels: true`. That
  enables all Opake Triton kernels (RMSNorm, fused add-RMSNorm, RoPE, SwiGLU),
  not RMSNorm alone.
- Measurements come from smoke-sized runs (20k-row source limit, 5 steps, post-training
  metrics disabled); no full run was launched.
- Step time is taken from the trainer's `step_time_sec`. `train_runtime` also
  includes about 180 s of evaluation.

## Suggested next steps

1. Adopt `gradient_checkpointing: false`, `use_performance_kernels: true` and
   `performance_kernels_config: {lora_mlp_recompute: false}` for the full
   recipe: about 31% faster per step, ≈68 GB peak reserved on an 80 GB H100.
2. Because the peak is set by clipping, try microbatch 4 with checkpointing
   off and kernels on, using the default `lora_mlp_recompute`. There is about
   25 GB of headroom, so it may fit.
3. Debug attention checkpointing on a GPU before relying on it.
