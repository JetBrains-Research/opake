# Plan: Fused Per-Layer Clipping Kernels (FlashDP-style)

## Status

**Draft (r3, 2026-09-29)** — r2's blocking correctness constraints (§2.6) stand. r3 adds measured baseline evidence + the oracle rig (new §4) and re-scopes v1's *first deliverable* accordingly. Benchmark scripts/results: `experiments/clip_breakdown/` (MPS + CUDA A100 results JSONs), `experiments/clip_oracle/` (all synthetic, two hosts: Apple-silicon MPS + A100 40GB CUDA — ratios are triage, not paper numbers).

**Branch:** `feat/dp-fused-per-layer-clipping` (branched from `main` at `5657d6ce`)

**Parent research:** `outputs/dp-clipping-optimization.md`

## Motivation

Opake's current clipping pipeline (`vmap(grad())` + `clip_pytree`) **reportedly** consumes ~87% of DP step time (user-reported figure; not yet reproduced — Phase 2.3 requires re-measuring the baseline first). The hypothesized dominant cost (inferred from code structure, not a profile) is per-example gradient materialization plus the two-pass norm-then-scale clipping loop.

**FlashDP** (Wang et al., NeurIPS 2025, arXiv:2507.01154) demonstrates that fusing gradient computation + clipping into per-layer CUDA kernels achieves **90% of non-DP throughput** on Llama-13B with 4× A100, with **zero added memory overhead** vs. non-DP training.

This plan adapts that approach to Opake's architecture — per-layer fused kernels as an **opt-in performance path** alongside the existing `vmap(grad())` pipeline.

## Goals

### In scope (v1)

1. **Triton kernel for fused per-layer Linear clipping** — compute weight gradient, per-sample norm, clip, aggregate, and add noise in fewer, coarser kernel passes (the reference implementation is two kernels per sample plus separate accumulate/noise steps; "single fused pass" is paper framing, not code reality)
2. **`DPLinear` layer wrapper** — `autograd.Function`-based drop-in replacement for `torch.nn.Linear`
3. **Integration API** — `opake.dpsgd.fused` public façade with `wrap_model()` and factory functions
4. **Triton kernel for fused per-layer LayerNorm clipping** — same pattern for normalization layers
5. **Tests** — numerical correctness (matches existing `clip_pytree` output), performance regression, dtype coverage

### Out of scope (future)

- Conv1D (HuggingFace-specific, deferred to v2)
- Attention-specific kernels (QKV projection handled by DPLinear; attention backprop deferred)
- Distributed (DDP/FSDP) integration
- DP-FTRL compatibility (per-layer clipping is DP-SGD-only)
- AMP/mixed precision hardening (basic bf16 support only in v1)

## Architecture

### 1.2 Where this lives

New home: **`opake-dpsgd`** (impl `opake.api.dpsgd.fused`, façade `opake.dpsgd.fused`), **not** under `opake.patches`.

**Decision (revised in r2):** per AGENTS.md partition policy, anything only one algorithm constructs (here: DP-SGD per-sample clip+noise inside backward) lives with that algorithm; `opake-patches` kernels are model-compat perf patches, whereas "clipping + noise inside backward" is a **privacy mechanism** and must not hide behind the patch façade (which is also entangled with the vmap-safety model patches). Generic Triton plumbing may be reused from the kernels module, but DP semantics stay in `opake-dpsgd`. Placement and dependency edges must be checked against `.junie/architecture-contracts.md` ARC-005 before implementation.

**Package layout:**

```
packages/opake-dpsgd/src/opake/api/dpsgd/fused/
  __init__.py           # public factory: wrap_model(), DPLinear, DPLayerNorm
  _linear.py            # autograd.Function + DPLinear module wrapper
  _layernorm.py         # autograd.Function + DPLayerNorm module wrapper
  kernel/
    __init__.py
    mm_clip.py          # Triton: fused matmul + norm + clip for Linear weight grad
    bmtm_clip.py        # Triton: batched matmul + clip (per-sample loop)
    layernorm_clip.py   # Triton: fused LayerNorm backward + clip
    clip_fn.py          # Shared clip factor computation (Torch + Triton)
    utils.py            # Shared Triton utilities (dtype maps, autotuner)
```

### 1.3 Design principle: Alternative path, not replacement

The fused per-layer kernels are an **alternative clipping path**, not a replacement for the existing `vmap(grad())` pipeline:

- **Default:** Existing `clipped_grad()` + `clip_pytree()` (works everywhere).
- **Opt-in:** `opake.dpsgd.fused.wrap_model()` swaps selected layers to fused variants. Because this *changes the privacy mechanism* (§2.6), the fused path must be mutually exclusive with the `clipped_grad`/DPTrainer pipeline and refuse double-stacking (wrap + clipped_grad = double clip/noise and an unmodeled mechanism).
- **Fallback:** If Triton is unavailable or kernel fails, fall back to `torch.einsum` + PyTorch clipping path (already in FlashDP codebase as `_weight_grad_block_clip`).

This preserves Opake's existing API surface and guarantees correctness via the existing path.

## Kernel Design

### 2.1 Core insight

For a Linear layer with `input ∈ ℝ^{B×T×P}`, `weight ∈ ℝ^{D×P}`, `grad_output ∈ ℝ^{B×T×D}`:

**Standard backward:** `grad_weight = Σ_{b,t} grad_output[b,t]^T × input[b,t]` → shape `(D, P)`

**DP-SGD backward:** Compute `G[b] = Σ_t grad_output[b,t]^T × input[b,t]` → shape `(B, D, P)`, clip each `G[b]`, then sum.

**FlashDP trick:** Compute `G[b]` in blocks that fit in SRAM, compute per-sample norm in SRAM, all-reduce norms across blocks, clip in SRAM, accumulate clipped gradients. Never materialize full `(B, D, P)` tensor.

### 2.2 Triton kernel: `_mtm_prenorm_kernel`

Computes per-sample weight gradient matrix and its squared norm simultaneously:

```
Input:  grad_output[b] ∈ ℝ^{M×D}, input[b] ∈ ℝ^{M×P}  (M = seq_length)
Output: grad_weight[b] ∈ ℝ^{D×P}, sq_norm[b] ∈ ℝ (scalar)

For each (M,N) tile of the output matrix:
    accumulator = dot(grad_output_tile, input_tile^T)   # GEMM in SRAM
    sq_norm += sum(accumulator^2)                        # intra-block reduction
    write accumulator to HBM                             # partial gradient
    atomic_add sq_norm to global norm buffer             # inter-block reduction
```

### 2.3 Triton kernel: `_mat_clip_kernel`

Applies clipping using the all-reduced norm:

```
Input:  grad_weight ∈ ℝ^{D×P}, global_norm ∈ ℝ, clip_threshold C
Output: clipped_grad_weight ∈ ℝ^{D×P}

clip_factor = min(1, C / (global_norm + ε))
clipped_grad_weight = grad_weight × clip_factor
```

### 2.4 Python orchestration: `bmtm_clip`

Loops over batch dimension, calling `mtm_clip` (Triton kernel) or `_weight_grad_block_clip` (PyTorch fallback) per sample, **accumulating into a single `(D, P)` buffer — the `[B, D, P]` stack is never allocated** (r1's pseudocode materialized it, which would have defeated the memory goal and contradicted §2.1):

```python
def bmtm_clip(grad_output, input, clip_args):
    B = grad_output.shape[0]
    out = torch.zeros((D, P), device=...)          # one accumulator, not [B, D, P]
    for b in range(B):
        out += mtm_clip(grad_output[b], input[b], ...)  # compute→norm→clip→accumulate
    return out
```

**Optimization:** `RuntimeAutoTuner` benchmarks Triton vs. PyTorch paths on first few iterations and selects the faster variant per matrix shape. **Do not port verbatim:** the reference defaults to 10 warmup + 100 measured iterations *per candidate per layer* (≈220 untimed calls per Linear before locking in), is nondeterministic across runs, and times work inside the training path. v1 should use a static shape/dtype heuristic with optional pre-tuning outside timed steps.

### 2.5 Numerical correctness

Gate (r2): two separate requirements — (a) approximate agreement with an **fp64 oracle** of the reference path (tolerances set by experiment; starting points `rtol=1e-4, atol=1e-5` fp32 / `rtol=1e-2, atol=1e-3` bf16), and (b) the **hard invariant** `‖clipped‖ ≤ C` on stored values in every dtype (§2.6.4). Approximate agreement alone is insufficient: Opake's contract is the bound, not the bits. Test data must be drawn in a regime where clipping is actually active — random Gaussian tensors at B=2–4 mostly exercise the no-clip path.

```python
# fp64 oracle for (a); exact enough to check fp32/bf16 kernels
def reference_clip(grad_output, input, C):
    B, T, D, P = grad_output.shape[0], grad_output.shape[1], grad_output.shape[-1], input.shape[-1]
    G = torch.einsum('btd,btp->bdp', grad_output, input)  # (B, D, P)
    norms = G.norm(dim=(-1, -2), keepdim=True)             # (B, 1, 1)
    scale = torch.clamp(C / (norms + 1e-10), max=1.0)
    clipped = G * scale
    return clipped.sum(0)                                  # (D, P)
```

### 2.6 Privacy-correctness constraints (BLOCKING — resolve before kernel code)

Verified against the reference code (`flashdp/layers/linear.py`, `flashdp/core/bmtm_clip_loop.py`, `flashdp/core/clip_fn.py`) and Opake's noise contract (`opake/api/dpsgd/noise/_gaussian.py`, `docs/user-guide/noise.md`). These are requirements, not open questions.

1. **Noise scale.** The reference adds noise **after** `.div_(B)` with `std = noise_multiplier/√B` on the *averaged* gradient, while clipping to C happens on the per-sample sum. Opake's convention is `std = noise_multiplier × max_norm` on the **aggregated** gradient (`C/B`-scaled for mean aggregation via `normalize_by`). The scales differ by a factor `√B/C`: for `C=1, B≥4` the reference over-noises (wasted utility + accounting mismatch); for `C > √B` (e.g. `C=10, B=64`) it **under-noises — a privacy violation** against any accounting calibrated to the sensitivity. The fused path must implement `N(0, (σ·C)²)` on the clipped sum before normalization (or route noise back through `gaussian_noise`), and the accounting must describe the executed mechanism.
2. **Composition.** Per-layer independent noise is *one Gaussian channel per layer per step*, not a single flat Gaussian. `opake.dpsgd.accounting` currently exposes flat-mechanism factories (`gaussian`, `adaclip`, `poisson`, `parallel_poisson`, `k_out_of_t`) that model the flat pipeline; per-layer composition (advanced composition or a per-layer accountant) must be designed and verified before release.
3. **Explicit RNG.** Noise must come from a threaded `RngKey`/generator, not global `torch.normal` state — required by Opake's explicit-state design and its reproducibility tests.
4. **Stored-value norm bound.** `clip_pytree` guarantees `‖clipped‖ ≤ C` on *stored* values via the `_guard_scale` ULP shrink, and sanitizes NaN/Inf to zero first. A fused `x * clamp(C/(‖x‖+1e-10), max=1)` in bf16 can round back above C; one NaN in a shared atomic norm accumulator poisons every sample's clip factor. Replicate the guard + sanitization, or document a weaker contract and keep the fused path out of anything feeding `gaussian_noise`.
5. **Reference bug: uninitialized norm accumulator.** `mtm_clip` allocates the norm buffer with `torch.empty([1])` and accumulates via `tl.atomic_add`; `reset_to_zero=['norm_ptr']` only applies to autotune benchmarking, not production launches. Port must zero the accumulator per launch and include a nondeterminism/parity test that would catch this class of bug.
6. **Block clipping semantics.** The PyTorch fallback `_weight_grad_block_clip` computes the norm over a *block of BK samples* and clips the block sum when `BK > 1` — that is not per-example DP-SGD. Pin BK=1 semantics in v1 (or handle per-example norms inside the block loop).

## API Design

### 3.1 Public factory: `wrap_model()`

```python
from opake.dpsgd.fused import wrap_model

model = my_llm_model.cuda()
dp_model = wrap_model(
    model,
    target_modules=[torch.nn.Linear],       # which module types to replace
    skip_modules=["lm_head", "embed_tokens"],  # which named modules to skip
    C=1.0,                                   # clipping threshold (per-layer)
    noise_multiplier=1.0,                    # σ, applied per Opake's convention (§2.6.1), NOT the reference's σ/√B
)

# Train with standard optimizer — DP is handled inside the fused layer
optimizer = torch.optim.Adam(dp_model.parameters(), lr=1e-4)
```

### 3.2 Per-layer module: `DPLinear`

```python
from opake.dpsgd.fused import DPLinear

layer = DPLinear(
    in_features=4096,
    out_features=4096,
    C=1.0,
    noise_multiplier=1.0,
)
# Drop-in replacement for nn.Linear — same forward signature.
# Clipping + noise happens in backward.
```

### 3.3 Integration with Opake accounting

The fused path does **not** use Opake's `clipped_grad()` pipeline, so accounting must be assembled explicitly — and per §2.6 the accountant must describe the mechanism that actually executes. The real building blocks are the factories in `opake.dpsgd.accounting` (verified exports: `gaussian`, `adaclip`, `poisson`, `parallel_poisson`, `k_out_of_t`); the r1 `gaussian_accounting(...)` sketch was invented and does not exist. None of the current factories model per-layer-per-step independent noise (§2.6.2): v1 must either (a) prove per-layer channel composition and encode it, or (b) route noise back out of backward through `gaussian_noise` on the assembled sum, which preserves the existing accountant at the cost of some memory savings. This decision gates Phase 3.

**Open question:** How to reconcile per-layer noise addition with Opake's existing noise allocation (per-group/structured noise)? In v1, we use simple per-layer isotropic Gaussian noise. Structured noise allocation is a v2 feature.

## Baseline & measurement rig (r3, 2026-09-29)

### What was measured

`experiments/clip_breakdown/bench_clip_breakdown.py` splits a step into
non-DP fwd+bwd (`t_nodp`) → vmap(grad) without clipping (`t_grad`) → clipped_grad
(streaming / pre-#1108 legacy / microbatched). Key results (median ms, MPS, synthetic):

| config | t_nodp | t_grad | clip (stream) | clip (legacy) | clip (mb) |
|---|---|---|---|---|---|
| 12×Linear128, B=128 | 4.5 | 4.1 | 9.9 | 8.9 | **131.3** |
| 2×Linear4096, B=16 | 14.2 | 20.0 | **126.9** | 120.6 | 128.8 |
| 2×Linear4096, B=64 | 67.2 | 87.2 | **502.3** | 3488.7 | 503.0 |

And on the real stack (`benchmarks/bench_e2e.py`, SmolLM2-135M LoRA, DP-FTRL/BLT):
`clip=158ms / step=213ms median ≈ 74%` — where the "clip" bucket is the **whole
`clipped_grad` call, i.e. vmap(grad) materialization *plus* clipping**, not clipping alone.

### CUDA baseline (A100 40GB, PyTorch 2.14.0, Triton 3.8.0)

Same rig on the target hardware (`results_cuda_a100.json`). Process note: the *first*
CUDA run was discarded — the harness synced only MPS, so CUDA timings measured kernel
enqueue, not execution (`t_nodp` looked batch-invariant at 1.1 ms). `cuda.synchronize()`
was added (commit `7a1fa0e8`); the table below is post-fix, device-synced per iteration,
plus per-path peak memory:

| config | t_nodp | t_grad | stream | nostream | mb8 | peak stream / nostream |
|---|---|---|---|---|---|---|
| many-small B=32 | 4.6 | 10.3 | 51.4 | 41.9 | 205.5 | 79 / 98 MB |
| many-small B=128 | 4.5 | 10.1 | 50.7 | **33.8** | 797.0 | 264 / 337 MB |
| few-large B=16 | 12.9 | 12.6 | 56.3 | 53.8 | 57.9 | 7.0 / 9.0 GB |
| few-large B=64 | 38.4 | 47.9 | 219.4 | **208.0** | 230.1 | **26.5 / 34.4 GiB** |

### What this says (both backends)

1. **The ~87% premise reproduces in order of magnitude everywhere it was measured** (74% real-harness, 77–91% synced CUDA, 86–91% MPS large-leaf), but it is dominated by the clip machinery, not by per-example grad materialization. MPS few-large B=64: materializing the `[B, ...]` grad stack costs ~+20 ms over non-DP while the clip pipeline adds **+415 ms**. CUDA (synced) few-large B=64: materialization **+9.5 ms** (47.9 vs 38.4) vs clip machinery **+171.5 ms** — ratio ~18×. At few-large B=16 on CUDA, `t_grad ≈ t_nodp` (12.6 vs 12.9 ms): the vmap per-example backward already matches batched autograd wall-time.
2. **Memory tells the opposite story from time.** Few-large B=64 on a 40 GB card peaks at **26.5 GiB (stream) / 34.4 GiB (nostream)** at seq=64 synthetic data; the pre-fix run logged an 8 GB allocation failure during the B=64 case. So per-example materialization is time-cheap but memory-dense on CUDA: Ghost/FlashDP-class no-materialize designs remain relevant as a *memory* lever and for longer sequences, not as the primary time lever at tested shapes.
3. **Legacy (nostream) is faster than streaming on CUDA in all four configs** (e.g. 33.8 vs 50.7 ms at many-small B=128; 208 vs 219 ms at few-large B=64) — the reverse of MPS few-large, where legacy was 7× slower (3489 vs 502 ms) because memory, not compute, was the wall there. The legacy path also uses ~1.3× stream's peak memory. Consequence: the fused-kernel bar on CUDA is *min(stream, nostream)* — beat ~34 ms (many-small) and ~208 ms (few-large B=64) at ≤ stream-path memory. Per-leaf two-pass elementwise + reduction with per-leaf launches and Python traversal is the shared shape of the cost on both backends — exactly what report §4.1/§4.4 (fused in-vmap norm+scale) targets, without mechanism change.
4. **Microbatching is a loser on both backends** (mb8: 250% slower at 128 small examples on MPS, 1500% on CUDA; parity elsewhere, no config where it wins on time; it only reduces memory — 3.8 GiB at few-large B=64). vmap `chunk_size` would inherit the same shape of cost.
5. Consequence for scope: a design that only removes the *grad-materialization* overhead (Ghost/FlashDP-class) attacks the minority time term on this hardware; a fused clip kernel attacks the majority term. FlashDP's 90%-of-non-DP was measured against its own stack's non-DP baseline — where per-sample materialization WAS the bottleneck, unlike post-#1108 Opake.

### Oracle rig (`experiments/clip_oracle/run_oracle.py`)

fp64-oracle parity + stored-value bound invariant + noise-convention probe, run against
today's clip path across {stream, legacy, mb} × {fp32, bf16} × {cpu, mps, cuda}, with
clipping-active data regimes (per-sample norms straddling C):
- All parity and bound checks pass on CPU and CUDA (strict fp32 1e-4 tolerance included,
  the near-C configs pass clean on CUDA where the sq-norm accumulator is fp64). On MPS,
  the same near-C fp32 config marginally exceeds tolerance, with **all three paths
  bit-identical to each other** (verified) — the deviation is MPS's deliberate fp32
  accumulator (no fp64 backend support), not a path bug. The rig therefore needs a
  same-backend reference for regression detection; fp64 stays the absolute sanity check.
- Bound invariant `‖clipped‖ ≤ C` holds on stored values everywhere (bf16 clipped norms
  land slightly *below* C via `_guard_scale`); noise probes confirm Opake's convention:
  `std = σ·max_norm` on the aggregate, `max_norm = C/B` under mean aggregation.
  Any fused kernel must reproduce both, per §2.6.

### Re-scoped v1 (CUDA gate passed — see baselines above)

- **v1 = fused *in-vmap* clip path**: single-pass per-leaf norm+scale in the existing
  `vmap(grad()) + clip` pipeline (report §4.1/§4.4 + Tier-1 #2/#4 micro-items), measured
  against the rig; no privacy-mechanism change, keeps max_norm/guard contract, DP-FTRL
  still compatible, no §2.6 blockers.
  Perf targets until a first prototype exists (set from baseline, revise after):
  beat **2× of the faster current path** per rig config (many-small: <~17 ms vs 33.8;
  few-large B=64: <~104 ms vs 208) at ≤ stream-path peak memory. The CUDA bar is the
  *nostream* path, not stream — unlike MPS where the fused design can only target stream.
- **v2 = per-layer fused backward (FlashDP-style)**: demoted from "pending evidence" to
  *memory-lever candidate*: on tested CUDA shapes its time prize is the minority term
  (~4% of step at B=64; ~0% at B=16, where vmap-grad already ties batched backward).
  Revisit when (a) peak memory (26.5–34.4 GiB measured at synthetic B=64/seq=64) blocks real
  configs, or (b) long-sequence/large-B profiles shift dominance to materialization.
  §2.6 constraints re-apply at that point.
- Before any of this: one CUDA profiling run (same breakdown script) to confirm the
  two-term split holds on the target hardware.

## Milestones

### Phase 1: Foundation (Week 1–2)

- [x] **1.0 ✅ DONE (A100 40GB, synced timing)** CUDA-baseline gate on target hardware (`results_cuda_a100.json`). **Result:** clip share 77–91%; clip machinery ≫ materialization in time (7–18×) but materialization dominates *memory* (26.5–34.4 GiB peaks at B=64); nostream is the faster CUDA baseline (fused kernel must beat it, ≤ stream memory); microbatching never wins on time. **Gate passed: v1 = fused in-vmap clip, confirmed.**

- [x] **1.1a ✅ Prototype v0 landed** (`experiments/clip_fused/fused_clip.py`, A100): 3-launch deterministic fused clip+sum; 8/8 correctness gates green (fp64-oracle parity, chain parity, stored-bound ≤ C in fp32+bf16, bitwise determinism); op-level speedup x7.2 (fp32 big leaf) / x25.4 (bf16 big leaf) / x1.7–3.4 (small leaves) vs the reconstructed stream chain; peak memory 4.2 GB vs 20.5 GB per big leaf; ~84–85% of HBM roofline. Baseline bar §4-baselines was met and exceeded at op level.
- [ ] **1.1b** Pipeline-seam integration: route the real `clipped_fun`/`clipped_grad` stream path through the fused kernel behind a flag; e2e step benchmark vs the synced CUDA baselines (33.8–208 ms clip buckets); per-tree `norm_roundoff` constant; small-leaf launch diet (fold C1 combine+scale into K2 prologue or one extra fused kernel; then consider CUDA graphs for 48-leaf trees).
  - Adapt to Opake's Triton kernel patterns (existing kernels in `opake-patches/src/opake/api/patches/kernels/`)
  - Add dtype support: fp32, bf16 (defer fp16)
  - Add `autograd.Function` compatibility checks

- [ ] **1.2** Implement `DPLinear` layer
  - `autograd.Function` wrapper with fused backward
  - Fallback to PyTorch path when Triton unavailable
  - `RuntimeAutoTuner` integration

- [ ] **1.3** Numerical correctness tests
  - `test_linear_clip_correctness.py` — compare fused vs. fp64 oracle reference path (§2.5), including the stored-value bound invariant and clipping-active data regimes
  - Tolerance: starting points `rtol=1e-4, atol=1e-5` (fp32), `rtol=1e-2, atol=1e-3` (bf16); final gate = invariant + oracle agreement, not bit-equality
  - Shape coverage: various (B, T, D, P) combinations

### Phase 2: Integration (Week 3–4)

- [ ] **2.1** Implement `DPLayerNorm` layer
  - Triton kernel for LayerNorm backward + clip
  - Same `autograd.Function` pattern as DPLinear

- [ ] **2.2** Implement `wrap_model()` factory
  - Module replacement logic (target/skip by type and name)
  - State dict compatibility (load weights from original module)
  - Device/dtype preservation

- [ ] **2.3** Performance benchmark
  - **Prerequisite:** reproduce the baseline first — the "87% step time in clipping" figure is user-reported and unverified. Record current-pipeline step time + peak memory via `StepPerf`/profiling tests before touching kernels; all impact claims are unfalsifiable otherwise
  - Single-layer benchmark: fused vs. current `clip_pytree` path vs. non-DP
  - Full-model benchmark: GPT-2 small / Llama-2 7B (if GPU available)
  - Target: ≥50% of **non-DP** throughput for Linear-dominated layers. For context, FlashDP's 90% was measured on its own stack against its own non-DP baseline; the primary comparison baseline is the **current Opake pipeline**

### Phase 3: Integration with Opake (Week 5–6)

- [ ] **3.1** Public API surface under `opake.dpsgd.fused` (impl under `opake.api.dpsgd.fused`; ARC-005/ARC-002 review)
  - `__all__` declaration in façade
  - Factory function signatures finalized
  - Deprecation/guard for unsupported dtypes

- [ ] **3.2** Integration tests
  - End-to-end training step: fused model + standard optimizer produces valid updates
  - Privacy accounting integration: verify realized noise stddev equals `σ × sensitivity` per Opake's convention (§2.6.1) and that the accounting model describes the per-layer channel composition actually executed (§2.6.2)
  - Mixed precision: bf16 forward + fp32 clipping (if supported)

- [ ] **3.3** Documentation
  - User guide: when to use fused vs. vmap clipping
  - API reference: `DPLinear`, `DPLayerNorm`, `wrap_model()`
  - Performance guide: expected speedups, hardware requirements

### Phase 4: Polish (Week 7)

- [ ] **4.1** Edge cases
  - Batch size 1
  - Very small layers (D < 64) — may be slower than PyTorch path
  - Reshape handling: input dim > 3 (current Opake reshapes to 3D)

- [ ] **4.2** CI integration
  - CUDA tests in `packages/opake-dpsgd/tests/fused/`
  - Auto-skip on non-CUDA / non-Triton hosts
  - Performance regression guard (fused must be ≥X% faster than reference)

- [ ] **4.3** Review and merge preparation
  - Internal review against `.junie/architecture-contracts.md`
  - DP review against `.junie/differential-privacy-review.md`
  - Benchmark results documented

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Triton kernel produces numerically different results for edge shapes | HIGH | Comprehensive shape/dtype test matrix; PyTorch fallback always available |
| Performance regression on small matrices | MEDIUM | `RuntimeAutoTuner` auto-selects faster path; document minimum matrix size |
| Integration with Opake's existing noise allocation breaks structured noise | MEDIUM | v1 uses isotropic noise only; structured noise is v2 |
| FlashDP's Triton kernels have bugs for bf16 | MEDIUM | Test against reference path; defer bf16 if needed |
| Porting the reference noise scale (σ/√B on the mean, global RNG) breaks the privacy mechanism — can **under-noise** when `C > √B` | CRITICAL | §2.6 constraints are blocking; parity tests: realized noise stddev == σ×sensitivity per Opake convention, RNG determinism, accounting model matches executed mechanism |
| Uninitialized norm accumulator in reference kernels (`torch.empty` + `tl.atomic_add`) | HIGH | Zero accumulator per launch; nondeterminism parity test |
| GPU memory at large batch/long sequences | MEDIUM (was LOW — now measured) | 26.5 GiB (stream) / 34.4 GiB (nostream) peak at synthetic B=64, seq=64 on 40 GB card; 8 GB allocation failure logged pre-fix. Keep memory probes in the bench; fused path gated at ≤ stream-path peak; v2 memory-lever remains the fallback |

## Open Questions

1. **Per-layer vs. flat clipping:** FlashDP uses per-layer clipping. Opake currently supports both. Should the fused path support flat clipping (requires cross-layer norm accumulation)? **Decision updated in r3: the v1 candidate is a fused CLIP kernel inside the existing flat/per-group vmap path (mechanism unchanged); per-layer clipping is deferred to v2 behind the CUDA evidence gate (Phase 1.0). Flat/per-group contract stays authoritative.**

2. **Noise addition timing and scale:** FlashDP adds noise during backward at `σ/√B` on the mean with global RNG; Opake adds `σ × max_norm` on the aggregate from an explicit `RngKey`. **Decision (revised in r2): v1 follows Opake's convention (§2.6.1/2.6.3, blocking), not the reference's. Per-layer channel accounting must be settled per §2.6.2 before any release; the r1 decision to copy FlashDP's noise path is withdrawn.**

3. **Attention layers:** Opake's patches include fused attention kernels. Should we fuse clipping into attention backward? **Decision: Deferred to v2. Linear layers cover ~80% of model params in transformers.**

4. **Distributed training:** How does per-layer fused clipping interact with FSDP/DDP? **Decision: out of scope for v1.** Correction to r1: "DDP should work if each rank clips its own shards" is wrong as stated — per-replica clip+noise before all-reduce is a *different mechanism* (replica-level clipping, noise shrunk by world size) that needs its own privacy analysis, and FSDP sharded flat buffers don't preserve the per-layer tensor identity the wrap scheme assumes. Do not enable the fused path under either without a DP review.

## Verification Plan

### Numerical correctness (gate)

Two separate gates per §2.5/§2.6: oracle agreement AND the stored-value bound invariant, run in a **clipping-active regime** (scale inputs so most per-sample norms exceed `C`; plain `randn` at small B mostly exercises the no-clip path).

```python
# Must pass before any performance claims
@torch.no_grad()
def test_correctness(dtype=torch.float32):
    for B, T, D, P in [(2, 64, 512, 512), (4, 128, 4096, 4096), (1, 32, 256, 1024)]:
        # large std ⇒ most per-sample grad norms exceed C ⇒ clipping is exercised
        input = 10.0 * torch.randn(B, T, P, dtype=torch.float64)
        grad_output = 10.0 * torch.randn(B, T, D, dtype=torch.float64)
        C = 1.0

        fused_result = fused_linear_backward(input.to(dtype), grad_output.to(dtype), C, rng_key=...)  # includes noise per §2.6.1
        ref_debiased = reference_clip(grad_output, input, C)  # fp64 oracle, noise removed for the parity check

        # (a) oracle parity (noise-free comparison of clip+sum; tolerance from experiment)
        assert torch.allclose(fused_result.debiased, ref_debiased, rtol=1e-4, atol=1e-5), \
            f"Shape {(B,T,D,P)}: max_diff={torch.abs(fused_result.debiased - ref_debiased).max()}"
        # (b) hard invariant on stored values, per §2.6.4 — check BEFORE noise is added
        assert norm_of_clipped_sum_if_unclipped_exceeds_C_is_clipped(fused_result.clipped_parts, C, dtype)
```

Additional required gates (not sketched here): realized noise stddev matches `σ × sensitivity` per Opake's convention (§2.6.1); same `RngKey` ⇒ bitwise-identical noise, different key ⇒ different noise (§2.6.3); nondeterminism parity test for the norm accumulator (§2.6.5).

### Performance (non-blocking gate)

```python
def benchmark_fused_vs_current_pipeline():
    # Primary comparison: fused path vs the CURRENT Opake vmap+clip_pytree path
    # on identical model/batch. Secondary: both vs non-DP (FlashDP's 90% was
    # vs its own stack's non-DP baseline, not vs Opake's current pipeline).
    # If fused is not faster, the static heuristic must select the current path.
    pass
```

### Integration (gate)

```python
def test_end_to_end():
    model = GPT2Small()
    dp_model = wrap_model(model, target_modules=[nn.Linear], C=1.0, noise_multiplier=1.0)
    # Guard: wrap_model must refuse to stack on top of an active
    # clipped_grad/DPTrainer configuration (double clip/noise, §1.3).
    assert model.__opake_dp_mode__ == "fused"  # illustrative exclusivity check
    optimizer = torch.optim.Adam(dp_model.parameters(), lr=1e-4)

    # Single training step must complete without error
    x = torch.randint(0, vocab_size, (2, 64))
    loss = dp_model(x, labels=x).loss
    loss.backward()
    optimizer.step()

    # Gradients must be finite
    for p in dp_model.parameters():
        assert p.grad is not None and p.grad.isfinite().all()
```

## References

- FlashDP paper: Wang et al., "Private Training Large-scale Models with Efficient DP-SGD", NeurIPS 2025, https://arxiv.org/abs/2507.01154
- FlashDP code: https://github.com/kaustpradalab/flashdp (r2 checks used `flashdp/layers/linear.py`, `flashdp/core/bmtm_clip_loop.py`, `flashdp/core/clip_fn.py`)
- Opake noise convention: `packages/opake-dpsgd/src/opake/api/dpsgd/noise/_gaussian.py` (realized std = `noise_multiplier * clipped.max_norm` on the aggregate, explicit `RngKey`), `docs/user-guide/noise.md`
- Opake clipping internals: `packages/opake-engine/src/opake/api/engine/clipping/` (`_pytree.py` `_guard_scale`/`_finalize_scale`; `_clipped_fun.py` microbatching + `_chunk_compiler`)
- Opake architecture contracts: `.junie/architecture-contracts.md`
