# Memory Profiling Report: DP-SGD step, Qwen2.5-Coder-7B LoRA, mb=2

**Dates:** 2026-09-30 (first batch), 2026-10-01 (re-run after the train-mode fix)
**Branch:** `feat/dp-fused-per-layer-clipping`. Profiling tooling and the fix
were uncommitted at time of writing (see §8).
**Hardware:** A100-SXM4-40GB (Coder workspace `kernel`, repo at `~/opaque`)
**Software:** torch 2.14.0+cu130, transformers 5.17.0, Python 3.11

Everything below is measured unless marked *estimate* or *inferred*.

**Outcome:** attention-only checkpointing has been **removed**. That covers both
the probe used here and the library option `attention_checkpointing` that
`a05b0b8e` ported to this branch. See §9. The `model.train()` fix in
`train_dpsgd.py` stays. `lora_mlp_recompute=False` measured **−9.9% step time
for +2.08 GB** through DPTrainer, and is inert in `train_dpsgd.py` (§10).
Follow-ups are tracked in
[#1121](https://github.com/JetBrains-Research/opake/issues/1121) (DPTrainer
trains in eval mode) and
[#1122](https://github.com/JetBrains-Research/opake/issues/1122) (fused LoRA
MLP/Linear never run in `train_dpsgd.py`).

## 1. Workload

`examples/train_dpsgd.py --preset qwen-7b-kstack` with CLI overrides:

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen2.5-Coder-7B`, bf16 |
| LoRA | r=16, alpha=16, on q/k/v/o/gate/up/down (preset) |
| Dataset | `JetBrains/KStack` (preset), 2000 train / 100 eval samples |
| Sequence length | 512 |
| Expected batch | 16 (Poisson; realised 14–16) |
| Microbatch | 2 (`clipped_grad` chunks of 2 under `vmap(grad)`) |
| Steps | 6 |

Step structure in `train_dpsgd.py`: one `grad_fn` call (vmap forward + backward
+ per-example clip, all fused) → `noise_fn` → `base_opt.update` +
`apply_updates`. The profiler's `clip` phase is the **whole** `grad_fn` call.

## 2. Tooling

### 2.1 Library: `opake.profiling`

`packages/opake-engine/src/opake/api/engine/profiling/_memory.py`, re-exported
from `opake.profiling`:

- `step_perf(device, *, batch_size=0, track_memory=True, phase_memory=False)`.
  With `phase_memory=True`, each `.mark(name)` records a **segment peak**
  (high-water since the previous mark) and then resets the CUDA peak counter.
  It is not a cumulative peak. The step peak is the max over segments. Every
  mark device-syncs, with or without `phase_memory`.
- `perf_tracker(device, phase_memory=...)`: the same option on the tracker.
- `StepPerf.mem_marks`, `StepPerf.phase_table()`: per-phase results.
- `start_snapshot(device)` / `finish_snapshot(path, device)`: wrap
  `torch.cuda.memory._record_memory_history` / `_dump_snapshot`. No-ops off
  CUDA (`start_snapshot` returns `False`, `finish_snapshot` returns `None`).

Tests: `packages/opake-engine/tests/profiling/test_phase_memory.py` (7 tests;
2 need CUDA). Full profiling suite: 83 pass / 3 skip on macOS, 76 pass / 10 skip
on the A100.

### 2.2 Entry point: `train_dpsgd.py --memory-profile`

| Flag | Default | Meaning |
|---|---|---|
| `--memory-profile PREFIX` | off | Enable. CUDA-only; prints a skip notice elsewhere |
| `--memory-profile-skip S` | 3 | Steps to run before recording (warm-up) |
| `--memory-profile-steps N` | 2 | Steps to record |

Outputs:

- `PREFIX.csv`: one row per step, with `step_time_sec`, `samples_per_second`,
  `memory_peak_gb`, `clip_sec`/`noise_sec`/`optimizer_sec` and
  `clip_peak_gb`/`noise_peak_gb`/`optimizer_peak_gb`.
- `PREFIX.timeline.csv`: long format `step,t_ms,phase,alloc_bytes`, from a
  daemon thread that samples `torch.cuda.memory_allocated()` every 1 ms during
  the recorded steps only.
- `PREFIX.pickle`: CUDA allocation history for the recorded steps, for
  <https://pytorch.org/memory_viz>.

The sampler thread contends for the GIL, and its overhead was **not measured**.
Compare step times between profiled runs from the same session only (§4.3).

### 2.3 Plotting (reads CSVs, no GPU needed)

```bash
python examples/plot_memory_profile.py notes/profiles/fix_attn      # one run
python examples/plot_memory_compare.py notes/profiles \
    "kernels, no ckpt:fix_nockpt" "attention-only ckpt:fix_attn" \
    "full-layer ckpt:fix_full" "eager, no kernels:fix_eager" \
    --step 3 --out notes/profiles/compare.png
```

`plot_memory_compare.py` takes the static floor to be the max `noise_peak_gb`
(the noise phase starts after `grad_fn` has released its working set). The
working set is `clip_peak_gb − floor`.

### 2.4 Mechanism self-check (CPU, since removed)

`experiments/mem_profile/check_attn_ckpt_fires.py` built a tiny random Qwen2
the same way `train_dpsgd.py` builds the real model: `from_pretrained` → PEFT
→ opake patches → probe → `make_functional` → `vmap(grad)`. It asserted that
the probe was inert without `model.train()`, that it fired on every layer with
it, and that per-example grads were bitwise equal either way. It passed locally
and on the workspace, and was deleted with the feature (§9).

## 3. Variants and runs

| Variant | Extra flags | Batch 1 (09-30) | Batch 2 (10-01, fixed) |
|---|---|---|---|
| Triton kernels on, no checkpointing | (none) | `q7bE` | `fix_nockpt` |
| + attention-only checkpointing | `--attention-checkpointing` | `q7b_attnckpt` (**inert**) | `fix_attn` |
| + HF full-layer gradient checkpointing | `--gradient-checkpointing` | `q7bF` | `fix_full` |
| kernels off ("before our upgrades") | `--no-kernel-patches` | `q7b_eager` | `fix_eager` |

Batch 2 is batch 1 rerun with `model.train()` added to `train_dpsgd.py` (§5).
It was run sequentially from `run_fix.sh` on the workspace:

```bash
WANDB_MODE=disabled .venv/bin/python examples/train_dpsgd.py \
    --preset qwen-7b-kstack --max-seq-len 512 --microbatch-size 2 \
    --batch-size 16 --stop-at-step 6 --num-train-samples 2000 \
    --num-eval-samples 100 --eval-steps 100 \
    --memory-profile runs/<PREFIX> --memory-profile-skip 3 --memory-profile-steps 2 \
    <extra flags>
```

`--attention-checkpointing` was a probe flag. It imported
`experiments/mem_profile/attn_ckpt.py` (a copy of the
`mihajlo/kernel-optimizations` wrapper) and applied it after the kernel patches.
When both batches ran it was not yet a library feature on this branch; `a05b0b8e`
ported it later the same day. The flag, the probe and the library option are
now all removed (§9), so the attention-only rows cannot be re-run from the
current tree.

## 4. Results

### 4.1 Per variant, batch 2 (the authoritative numbers)

All rows use identical profiling instrumentation and come from one session.
Means are over 6 steps.

| Variant | Step peak | Floor | Working set | Mean step | Throughput |
|---|---:|---:|---:|---:|---:|
| kernels, no ckpt | 22.43 GiB | 15.03 | 7.39 GiB | 11.67 s | 1.30 smp/s |
| attention-only ckpt | 21.60 GiB | 15.04 | 6.56 GiB | 14.80 s | 1.02 smp/s |
| full-layer ckpt | 16.39 GiB | 15.04 | 1.36 GiB | 15.94 s | 0.95 smp/s |
| eager, no kernels | 24.40 GiB | 15.04 | 9.36 GiB | 9.25 s | 1.64 smp/s |

Against kernels with no checkpointing:

| Variant | Δ peak | Δ step time |
|---|---:|---:|
| attention-only ckpt | **−0.83 GiB** (−11% of working set) | **+3.13 s (+27%)** |
| full-layer ckpt | −6.03 GiB (−82% of working set) | +4.27 s (+37%) |
| eager (kernels off) | +1.97 GiB | −2.42 s (−21%) |

Attention-only checkpointing buys **14% of full-layer checkpointing's memory
saving for 73% of its time cost**.

Figure: `notes/profiles/compare.png` (batch 2). Single runs: `fix_*.png`.
The batch-1 figure is kept as `compare_inert.png`. Its attention-only panel is
the inert run.

### 4.2 Phase breakdown (`q7bE`, step 3)

| Phase | Time | Segment peak |
|---|---:|---:|
| clip (`grad_fn`: fwd + bwd + clip) | 10.37 s | 22.42 GiB |
| noise | 0.34 s | 15.03 GiB |
| optimizer | 0.27 s | 15.19 GiB |
| step | 10.99 s | 22.42 GiB |

In every variant `noise` + `optimizer` stay under 0.7 s and 0.2 GiB above the
floor. The variants differ only inside `grad_fn`.

### 4.3 Batch 1 vs batch 2: what changed, what didn't

- **Memory: identical.** Peaks for the three variants whose mechanism did not
  change match to four decimals (22.4263 / 16.3943 / 24.4039 GiB). Running in
  train mode, and therefore with the `kv_cache` patch active and no
  `DynamicCache`, had no measurable effect on peak. The cache size I estimated
  earlier (~59 MB/chunk) therefore never showed up at the peak. *Inferred:* the
  cache holds the same K/V tensors autograd already saves.
- **Time: about 12–15% slower in batch 2 for all three unchanged variants**
  (no ckpt 10.19→11.67 s, full 14.12→15.94 s, eager 8.26→9.25 s). A uniform
  shift is consistent with session-to-session drift. The two can't be fully
  separated, because the no-ckpt and eager runs also switched to train mode.
  Compare times within a batch only.
- **Attention-only: batch 1 is invalid** (inert; §5). Use batch 2.

### 4.4 Shape of the timeline

Inside `grad_fn`, live memory is a **sawtooth**: one spike per microbatch chunk
(7 full chunks of 2 plus 1 partial). Each spike rises from the 15 GiB floor to
the step peak and returns. Peak is flat across steps, so nothing accumulates
across chunks or steps.

- Full-layer checkpointing flattens the spikes to about 1.4 GiB above the floor.
- Attention-only lowers each spike by about 0.8 GiB and lengthens its backward
  (descending) half, which is where the recompute shows up.

### 4.5 GPU utilisation (batch 1 session)

This was a separate unprofiled 8-step run (kernels on, no checkpointing),
sampled with `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -l 1`:

- mean **32.1%** over 50 one-second samples; max 64%; 16 samples at 11%
- `nvidia-smi` memory.used steady at 24326 MiB. This includes the CUDA context
  and the allocator reserve, so it exceeds the allocated peak.
- 1.7 samples/s; peak allocated 22.43 GB

This run was before the fix. It was not repeated in batch 2.

### 4.6 Full-batch vmap

With `--microbatch-size` unset (one vmap over the whole batch) at seq 512, the
run went **OOM** at 39.5 GiB on a 518 MiB allocation. Microbatching is required
for this model on 40 GB.

## 5. The attention-only no-op: cause, fix, effect

### 5.1 Cause

**The model trained in eval mode, and the probe only checkpoints in training
mode.**

1. `AutoModelForCausalLM.from_pretrained` returns the model in eval mode.
2. `get_peft_model` wraps it in a new `PeftModel`. The wrapper's own
   `.training` is `True`, but the base submodules stay `False`, so checking
   `model.training` at the top level is misleading.
3. `train_dpsgd.py` never called `model.train()`.
4. The probe's guard is
   `if not module.training or not torch.is_grad_enabled() or _has_kv_cache(kwargs): return forward(...)`.
   `self_attn.training` was `False`, so every call took the plain forward.

Full-layer checkpointing was unaffected. Opake's patch of
`PreTrainedModel.gradient_checkpointing_enable`
(`packages/opake-patches/src/opake/api/patches/torch/checkpoint/huggingface.py`,
`_force_non_reentrant`) sets `.training = True` on every module that has a
`gradient_checkpointing` attribute (the decoder layers and `Qwen2Model`). Its
docstring notes that PEFT "keeps base layers in eval, which otherwise makes
checkpoint a no-op".

There is a second guard behind the first. Flipping only `self_attn.training`
is not enough. `Qwen2Model` stays in eval, so the `kv_cache` patch (which forces
`use_cache=False` only in training) leaves `config.use_cache=True`. A
`DynamicCache` then reaches every `self_attn`, and the KV guard trips instead.

CPU replica, guard outcome per attention call:

| Setup | Outcome |
|---|---|
| as originally run | `eval` (no checkpoint) |
| only `self_attn.training = True` | `kv-cache` (no checkpoint) |
| `model.train()` | `CHECKPOINTED` |

The `mihajlo/kernel-optimizations` tests call `model.train()` and pass
`use_cache=False`, which removes both guards. That is how its `[2, 2]` call
counts passed while the real training path stayed inert. *Inferred, not
executed:* the H100 sweep's null result for `attention_checkpointing=True`
likely has the same cause. On that branch the wrapper is applied from
`opake.patches.__init__` with no `.train()` nearby, and `DPTrainer` only calls
`self._model.train()` to restore mode after an eval pass.

### 5.2 Fix

In `examples/train_dpsgd.py`, `model.train()` is now called unconditionally
just before `make_functional`, after PEFT, the patches, the probe and
gradient-checkpointing setup. It is safe because:

- `train_dpsgd.py` already zeroes every dropout attribute in the model config;
- the `vmap`s in `clipped_grad` use `randomness="same"`;
- per-example grads were bitwise equal in eval and train mode (CPU check).

Batch-2 losses at step 2 match across variants to 1e-3 (0.8693–0.8699), as
expected for unchanged math under bf16.

### 5.3 Effect (batch 2)

Attention-only checkpointing now fires (28 blocks wrapped; memory and time
both move):

- peak 22.43 → **21.60 GiB (−0.83 GiB)**
- step 11.67 → **14.80 s (+27%)**

It is a poor trade at this shape. *Estimate* of why the saving is small: the
MLP intermediates (3 tensors of 512 × 18944 per example per layer) are far
larger than attention's saved tensors (q and out at 512 × 3584, k and v at
512 × 512 with 4 KV heads), and attention-only leaves the MLP's saved
activations in place.

## 6. Conclusions

1. **Peak = 15.0 GiB static floor + one microbatch chunk's working set.** The
   floor (bf16 weights + LoRA optimizer state) is identical in every variant.
   Only the working set (7.4 GiB with kernels, 9.4 GiB eager) can be reduced.
2. **The working set is mostly saved activations.** Full-layer checkpointing
   removes 6.0 of the 7.4 GiB. Attention accounts for only 0.8 GiB of that.
   An earlier claim in this workstream that the peak is "clip-bound, not
   activation-bound" is wrong.
3. **No leak.** The timeline is a sawtooth per chunk and flat across steps.
4. **Attention-only checkpointing: −0.83 GiB for +27% step time.** It is not
   worth it at this shape. Full-layer costs +37% for −6.03 GiB.
5. **Kernels vs eager at seq 512: −1.97 GiB for +26% step time** (11.67 vs
   9.25 s, batch 2; batch 1 showed +23%). The seq-4096/H100 sweep found the
   opposite sign, so re-measure at the target shape before generalising. In
   `train_dpsgd.py` this toggles only the non-LoRA kernels (QKV, RoPE,
   RMSNorm, activation, CE). The fused LoRA MLP and Linear never run there
   (§10, #1122).
6. **At this shape, time is the binding constraint, not memory.** The card had
   ~16 GiB unused, mean GPU utilisation was 32% (batch 1), and about 95% of
   step time is spent inside `grad_fn`.
7. **Training-gated code was silently off in `train_dpsgd.py` until this fix.**
   Check every training-mode-dependent patch and wrapper against the real
   `from_pretrained` + PEFT path, not only against tests that call
   `model.train()` themselves.

## 7. Open items

- [#1121](https://github.com/JetBrains-Research/opake/issues/1121): confirmed
  that `DPTrainer` (and `SFTTrainer`, which subclasses it) trains with the base
  model in eval mode. A CPU repro through `from_pretrained` + PEFT +
  `DPTrainer.train()` sees `self_attn.training=False` and a `DynamicCache` in
  the training forward. LoRA dropout is unaffected and base-model dropout is
  off. The test fixtures construct models directly (`training=True`), which
  hides it. Not fixed here.
- [#1122](https://github.com/JetBrains-Research/opake/issues/1122): the fused
  LoRA MLP/Linear kernels never run in `train_dpsgd.py` (§10).
- At seq 512 neither checkpointing variant pays off. The regime where
  activation memory binds (seq 4096) doesn't fit on 40 GB at mb=2 without
  checkpointing and has not been profiled here.

## 8. Reproducing

Everything is committed on `feat/dp-fused-per-layer-clipping`: profiling
`87a2c0c5`, `--memory-profile` `1562e33f`, train-mode fix `5a1131a2`,
attention-checkpointing removal `a2ce1712`, `--lora-mlp-recompute` for the
DPTrainer example `6d4a352f`. Note that `c165da35` added the flag to
`train_dpsgd.py` and `6f6fbf2a` removed it again as inert. The workspace (`coder ssh kernel`, `~/opaque`) is
synced to the same commit through git, not by copying files. Its pre-sync local
state (a modified `experiments/clip_breakdown/results.json`, a modified
`experiments/clip_fused/fused_clip.py`, `run_fix.sh`) is preserved in
`~/opaque_backup_20261001`. `runs/` still holds every run's CSV, timeline and
snapshot.

Run the §3 command per variant on the workspace (Homebrew `coder` CLI,
`coder ssh kernel -- '...'`), copy `runs/<PREFIX>.{csv,timeline.csv}` back, and
plot locally per §2.3. Only the figures (`notes/profiles/*.png`) are committed.
The CSVs behind them live in the workspace's `runs/` and are regenerable.

Remote-shell pitfalls: piping data into `coder ssh` stdin hangs and leaves the
target file truncated, and inside a `while read` loop `coder ssh` needs
`< /dev/null` or it swallows the loop's input. To sync code, commit locally and
then either push or ship a `git bundle` (`git bundle create b origin/<branch>..HEAD`),
fetch it on the workspace, and `git reset --hard FETCH_HEAD`, after backing up
anything local there.

## 9. Decision: attention-only checkpointing removed (2026-10-01)

**Why.** The selective activation recomputation of Korthikanti et al.
([arXiv 2205.05198](https://arxiv.org/abs/2205.05198), §5) recomputes only the
"core attention": QKᵀ, softmax, softmax dropout and attention over V. Those
produce the s×s tensors (`5as²b` per layer), which are large in memory but
low in FLOPs. Everything else is stored, including the inputs to the
Q/K/V/output projections, because those are expensive to recompute. Applied to
this setup:

- Opake runs `attn_implementation="sdpa"`. HF `flash_attention_2` (the
  `flash-attn` package) is not vmap-compatible. On our runs SDPA dispatched to
  its built-in flash kernel. Both A100 logs carry the
  `_scaled_dot_product_flash_attention_backward` batching-rule warning, and
  attention dropout is 0. A fused SDPA backend never stores the s×s matrix and
  recomputes it in its own backward, so the paper's recompute target is already
  gone from memory.
- The attention-only wrapper therefore re-ran only what the paper says to
  store: the Q/K/V/O projections, their LoRA adapters and RoPE. That matches the
  measurement, −0.83 GiB for +27% step time (§5.3).

**Removed:**

- the `attention_checkpointing` kwarg of `apply_model_patches` and
  `packages/opake-patches/src/opake/api/patches/transformers/components/attention_checkpoint.py`
  with its test file;
- the `torch_compile` × `attention_checkpointing` `ConfigurationError` in
  `TrainingArguments`, its two tests in `test_compile_and_kernel_args.py`, and
  the key from the `_performance_kernels.py` docstring;
- the attention row and prose in `docs/user-guide/memory-optimizations.md`,
  replaced by a short note explaining why no attention-only option exists;
- the probe: `--attention-checkpointing` in `train_dpsgd.py` and
  `experiments/mem_profile/`.

**Kept:** `lora_mlp_recompute` (the other half of `a05b0b8e`) and the
`model.train()` fix.

No deprecation path is needed: `a05b0b8e` is in no tag and only on this feature
branch. Passing `attention_checkpointing=True` now is silently ignored, as is
any unknown `apply_model_patches` kwarg.

**Verified:** no remaining references in `packages/`, `docs/` or `examples/`;
lint and format unchanged against HEAD. `opake-patches` + `opake-transformers`
PR-marker suites pass: 1436 passed, 42 skipped, 0 failed (macOS).

**When it would matter:** only with `attn_implementation="eager"` or SDPA's
`MATH` backend, which materialize the attention matrix. Even then, the paper's
answer is to recompute just the core-attention ops, not the whole block.

## 10. `lora_mlp_recompute` (2026-10-01)

**What it trades.** By default (`True`) the fused LoRA MLP saves only its input
and re-runs the two `hidden → intermediate` matmuls (gate and up) in backward.
With `False` it saves gate and up instead. *Estimate* for Qwen2.5-Coder-7B:
2 tensors × 28 layers × 18944 × 2 bytes ≈ 2.1 MB per token per example, i.e.
2.17 GB at seq 512, mb=2.

**Measured through DPTrainer** (`examples/train_dpsgd_trainer.py`, bf16
autocast, `--use-performance-kernels`, LoRA r16 on all 7 projections, seq 512,
batch 16, mb 2, 6 steps). Both runs used the same seed and therefore identical
Poisson batches. Step times are DPTrainer's own `step_time_sec`, steps 1–6:

| | Step peak (`memory_peak_gb`) | Mean step | Per-step times (s) |
|---|---:|---:|---|
| `lora_mlp_recompute=True` (default) | 17.02 GB | 14.60 s | 15.93 14.88 12.37 12.51 18.16 13.77 |
| `lora_mlp_recompute=False` | 19.10 GB | 13.16 s | 12.10 14.23 11.88 11.81 15.73 13.23 |

That is **−9.9% step time for +2.08 GB**, with the save variant faster on every
step. The memory delta matches the estimate. The H100 sweep (seq 4096) found
−6.5% for +6.4–12.7 GB.

**Verdict:** worth it whenever the extra `≈ 2 × layers × intermediate × bytes`
per token per example fits in spare memory. At seq 512, mb=2 on a 40 GB A100
it does, by a wide margin. At seq 4096 the cost grows 8×. The default stays
memory-safe.

**Inert in `train_dpsgd.py`.** The same two runs through `train_dpsgd.py` gave
byte-identical peaks (22.43 GiB) and equal step times. A CUDA probe showed why:
the base model is bf16, PEFT 0.21 upcasts LoRA adapters to fp32, and the loop
has no autocast. The fused LoRA MLP (and Linear) forward requires autocast or
adapters in the hidden states' dtype, so it falls back to the PEFT forward on
every call (adapters cast to bf16 → fused path taken). The flag was therefore
removed from `train_dpsgd.py` again (`6f6fbf2a`). It lives in
`train_dpsgd_trainer.py` (`6d4a352f`), and the user guide now states the dtype
condition (`fbc214a1`). Tracked in
[#1122](https://github.com/JetBrains-Research/opake/issues/1122).

*Not established:* DPTrainer's step peak at this shape (17.02 GB) is 5.4 GB
below `train_dpsgd.py`'s (22.43 GiB). The fused LoRA kernels are one plausible
cause, but the loops differ in other ways, so this is not attributed.
