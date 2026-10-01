# Checkpointing: Open Questions

**Date:** 2026-09-29
**Branch:** `mihajlo/kernel-optimizations`

## Known facts

- Attention checkpointing (`attention_checkpointing=True`) installs correctly — verified `[2, 2]` forward/backward call counts per layer.
- Attention checkpointing yielded **0 GB peak memory savings** in the sweep.
- `vmap(grad())` processes mb=2 samples in parallel (batched matmuls), not sequentially.
- `vmap(grad())` interleaves forward and backward per sample: forward computes activations, backward immediately needs them.
- `apply_runtime_patches(compat=True)` patches torch checkpoint to work under `vmap(grad())`.
- No persisted sweep data exists — notes files from the investigation session were not written to disk.

## Questions

### 1. Why did attention checkpointing yield 0 GB peak savings?

Checkpointing should save memory during backward: instead of holding all 28 layers' attention weights (~X GB), it recomputes them layer-by-layer (~X/28 GB at a time). The sweep showed no measurable reduction.

**Possible explanations:**
- `vmap(grad())`'s fused backward cannot free activations layer-by-layer (gradient graph holds references)
- Per-sample gradients dominate the peak and swamp checkpointing savings
- Checkpointing overhead (extra tensors, grad check) adds back the savings
- The checkpoint wrapper is installing but not actually bypassing activation storage

**What to measure:**
- Peak memory with/without checkpointing for individual layers (not all 28)
- Are attention weights actually freed after forward, or retained by autograd?
- Does the checkpoint wrapper's `_recompute_layer` actually fire, or does it short-circuit?

### 2. What is the actual activation memory breakdown?

No calibrated numbers exist. The 9 GB attention / 13 GB per-sample gradient figures used in the chart were fabricated.

**What to measure:**
- Per-layer activation memory (attention, MLP, norms) at mb=2, seq=4096
- Total saved activations across all 28 layers during vmap forward
- Per-sample gradient memory for mb=2
- Transient compute memory during attention forward (not saved, just used)

### 3. Does `vmap(grad())` prevent layer-by-layer freeing during backward?

If backward processes all 28 layers as a fused operation, activations from all layers may remain live simultaneously, neutralizing checkpointing.

**What to verify:**
- Trace `vmap(grad())` backward: does it free per-layer activations as each layer completes, or hold them all?
- Compare peak memory of a 1-layer model with/without checkpointing — isolates the vmap behavior from the gradient accumulation

### 4. Where does the ~54 GB peak actually go?

Without a proper `torch.cuda.max_memory_allocated` sweep across the training step, the 54 GB number is an observation, not a decomposition.

**What to measure:**
- `torch.cuda.max_memory_allocated()` at each phase: baseline, vmap forward, vmap backward, clip, noise, opt step
- Per-component breakdown: frozen params, optimizer state, trainable params, activations, gradients, aux

## Method

Run `bench.py` (or a minimal reproduction script) with `torch.cuda.memory_stats()` tracing enabled. Capture:

1. Baseline memory allocation (params + optimizer state)
2. Peak during vmap forward (with/without checkpointing)
3. Peak during vmap backward (with/without checkpointing)
4. Peak during clip + noise + opt step
5. Per-layer peak (single layer, 28 layers, 49 layers)

## Not needed yet

- Charts or visualizations — need real numbers first
- Implementations of "fixes" — need to understand the problem before patching

---

## Measured 2026-09-30 (A100-40GB, torch 2.14.0, Qwen2.5-Coder-7B + LoRA r16 on q/k/v/o/gate/up/down, seq 512, mb=2, batch 16, 6 steps)

Run on `feat/dp-fused-per-layer-clipping`, not the branch named above. That
branch's `attention_checkpointing=True` knob does not exist here; the
"attention-only" row used the same wrapper copied to
`experiments/mem_profile/attn_ckpt.py` plus a temporary
`--attention-checkpointing` flag on `train_dpsgd.py`, applied after the kernel
patches. 28 `self_attn` blocks reported as wrapped, so installation is not the
thing that failed.

Instrumentation added for this: `train_dpsgd.py --memory-profile PREFIX` writes a
per-phase peak/time CSV plus a 1 ms live-memory timeline, and
`examples/plot_memory_compare.py` plots the variants against each other. All
four rows below used identical instrumentation (timeline on), so they are
mutually comparable; step times therefore include sampler overhead and are not
comparable to the flagless sweep numbers.

| variant | step peak | working set above floor | mean step time |
|---|---:|---:|---:|
| kernels, no checkpointing | 22.43 GiB | 7.39 GiB | 10.19 s |
| attention-only checkpointing (inert — see Q1; superseded below) | 22.43 GiB | 7.39 GiB | 10.17 s |
| full-layer checkpointing | 16.39 GiB | 1.36 GiB | 14.12 s |
| eager (no Triton kernels) | 24.40 GiB | 9.36 GiB | 8.26 s |

Static floor — weights + optimizer state, resident before any per-step work —
was **15.0 GiB in every variant**. Only 7.4 GiB was ever variable.

### Answers to the open questions

**Q1 (why 0 GB from attention checkpointing).** It never fired. The model
trains in eval mode: `from_pretrained` returns eval, `get_peft_model` wraps it
in a `PeftModel` whose own `.training` is `True` while the base submodules stay
`False`, and `train_dpsgd.py` never calls `.train()`. The wrapper's guard
(`not module.training or not torch.is_grad_enabled() or _has_kv_cache(kwargs)`)
returns the plain forward on every call. Full-layer checkpointing works only
because opake's `gradient_checkpointing_enable` patch flips `.training = True`
on the decoder layers and `Qwen2Model`. Flipping only `self_attn` is not
enough: `Qwen2Model` stays in eval, so the `kv_cache` patch leaves
`use_cache=True`, a `DynamicCache` reaches every `self_attn`, and the KV guard
trips instead. Confirmed on a CPU replica of the `train_dpsgd.py` setup
(from_pretrained → PEFT → patches → probe → make_functional → vmap(grad)):
guard outcomes were `eval` as run, `kv-cache` with only `self_attn` flipped, and
`CHECKPOINTED` after `model.train()`, with bitwise-equal per-example grads
throughout. The `[2, 2]` call counts in "Known facts" came from tests that call
`model.train()` and pass `use_cache=False` explicitly, which removes both
guards. The H100 sweep likely hit the same cause (inferred from the branch's
code, not executed).

**Fixed and re-measured 2026-10-01.** `train_dpsgd.py` now calls `model.train()`
before `make_functional`. All four variants were rerun in one session:

| variant | step peak | working set | mean step time |
|---|---:|---:|---:|
| kernels, no checkpointing | 22.43 GiB | 7.39 GiB | 11.67 s |
| **attention-only checkpointing** | **21.60 GiB** | **6.56 GiB** | **14.80 s** |
| full-layer checkpointing | 16.39 GiB | 1.36 GiB | 15.94 s |
| eager (no Triton kernels) | 24.40 GiB | 9.36 GiB | 9.25 s |

Attention-only checkpointing saves **0.83 GiB for +27% step time**: 14% of
full-layer's memory saving for 73% of its time cost. Peaks of the unchanged
variants match the first batch to four decimals. Times are about 12–15% slower
across the board in this session, so compare within a batch only. Full
write-up: `notes/memory-profiling-report.md` §4–§5.

**Closed: removed 2026-10-01.** SDPA's fused kernel (flash on our runs) already
recomputes the s×s attention matrix, which is the part that selective
recomputation (arXiv 2205.05198 §5) targets. The attention-only wrapper
therefore only re-ran the expensive projections. The probe and the library
`attention_checkpointing` option are both gone; see
`notes/memory-profiling-report.md` §9.

**Q2 (activation breakdown).** The per-step working set at mb=2 is 7.4 GiB with
kernels, 9.4 GiB eager. Full-layer checkpointing removes 6.0 GiB of the 7.4, so
most of the working set is saved activations. Attention-only checkpointing
(once fixed, Q1) removes 0.83 GiB, so attention's saved activations are about
0.8 GiB of the 7.4 and the remaining ~5.2 GiB that full-layer removes sits in
the MLP and norms. That is consistent with the MLP intermediates (3 × 512 ×
18944 per example per layer) dwarfing attention's saved tensors.

**Q3 (does vmap prevent layer-by-layer freeing).** Partial answer: memory is
*sawtooth, not monotonic*. Each step shows one spike per microbatch chunk that
returns all the way to the floor afterwards, so activations are being released
during backward. There is no evidence of a fused backward holding all 28 layers
live. A "6.2 GB retained gradient references" style leak would show as a
rising staircase across steps; peak was flat at 22.43 GiB over all 6 steps.

**Q4 (where the peak goes).** Decomposed, at this shape (step 3 of the
no-checkpointing run): 15.0 GiB static (weights + optimizer state) + 7.4 GiB
one chunk's working set, all of it reached inside the single `grad_fn` call
(10.37 s). `noise` took 0.34 s peaking at 15.03 GiB, `optimizer` 0.27 s at
15.19 GiB — together 0.61 s and at most 0.16 GiB above the floor. So the
ceiling is 15 GiB of untouchable residency plus one chunk; peak scales with
microbatch size, not with batch size.

### Consequences

- On a 40 GB card at this shape ~16 GiB went unused while GPU utilisation
  averaged **32.1 %** during the step (50 one-second `nvidia-smi` samples,
  max 64 %, 16 of them at the 11 % floor, allocator steady at 24326 MiB).
  Memory was not the binding constraint; the per-chunk compute was. Freeing
  memory bought nothing here.
- Full-layer checkpointing is a net loss at this shape: −6.0 GiB for +38 % step
  time, with headroom to spare.
- The Triton kernels cost throughput at seq 512 (10.19 s with, 8.26 s without)
  for 1.97 GiB. This is the opposite sign to the seq-4096/H100 sweep, so do not
  carry that result over without re-measuring at the target shape.
- mb=2 sequential-chunked and full-batch vmap were both tried at seq 512;
  full-batch vmap OOM'd at 39.5 GiB, so microbatching is load-bearing for the
  7B on this card even though it is slower.
