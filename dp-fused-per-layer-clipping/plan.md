# Plan: Fused Per-Layer Clipping Kernels (FlashDP-style)

## Status

**Draft** — awaiting explicit approval before implementation.

**Branch:** `feat/dp-fused-per-layer-clipping` (branched from `main` at `5657d6ce`)

**Parent research:** `outputs/dp-clipping-optimization.md`

## Motivation

Opake's current clipping pipeline (`vmap(grad())` + `clip_pytree`) consumes ~87% of DP step time. The dominant bottleneck is the per-example gradient materialization and the two-pass norm-then-scale clipping loop.

**FlashDP** (Wang et al., NeurIPS 2025, arXiv:2507.01154) demonstrates that fusing gradient computation + clipping into per-layer CUDA kernels achieves **90% of non-DP throughput** on Llama-13B with 4× A100, with **zero added memory overhead** vs. non-DP training.

This plan adapts that approach to Opake's architecture — per-layer fused kernels as an **opt-in performance path** alongside the existing `vmap(grad())` pipeline.

## Goals

### In scope (v1)

1. **Triton kernel for fused per-layer Linear clipping** — compute weight gradient, per-sample norm, clip, aggregate, and add noise in a single kernel pass
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

New distribution: **`opake-kernels`** (or submodule within `opake-patches`).

**Decision:** We place the fused kernels as a **new submodule** under `opake-patches` rather than a new distribution, because:

1. The kernels depend on `opake-engine` for types and random state.
2. The existing `opake-patches` already contains Triton kernels (fused CE, RoPE, etc.).
3. This avoids the overhead of a new wheel while keeping the functionality discoverable.

**Package layout:**

```
packages/opake-patches/src/opake/api/patches/fused/
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
- **Opt-in:** `opake.patches.fused.wrap_model()` swaps selected layers to fused variants.
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

Loops over batch dimension, calling `mtm_clip` (Triton kernel) or `_weight_grad_block_clip` (PyTorch fallback) per sample:

```python
def bmtm_clip(grad_output, input, clip_args):
    B, M, D, P = batch, seq_len, out_features, in_features
    out = zeros([B, D, P], device=...)
    for b in range(B):
        out[b] = mtm_clip(grad_output[b], input[b], ...)
    return out
```

**Optimization:** `RuntimeAutoTuner` benchmarks Triton vs. PyTorch paths on first few iterations and selects the faster variant per matrix shape.

### 2.5 Numerical correctness

The fused kernel must produce results within `1e-5` relative tolerance of the reference PyTorch path:

```python
# Reference (matches Opake's existing clip_pytree behavior for a single linear layer)
def reference_clip(grad_output, input, C):
    B, T, D, P = grad_output.shape[0], grad_output.shape[1], grad_output.shape[-1], input.shape[-1]
    G = torch.einsum('btd,btp->bdp', grad_output, input)  # (B, D, P)
    norms = G.norm(dim=(-1, -2), keepdim=True)             # (B, 1, 1)
    scale = torch.clamp(C / (norms + 1e-10), max=1.0)
    clipped = G * scale
    return clipped.sum(0)                                  # (D, P)
```

## API Design

### 3.1 Public factory: `wrap_model()`

```python
from opake.patches.fused import wrap_model

model = my_llm_model.cuda()
dp_model = wrap_model(
    model,
    target_modules=[torch.nn.Linear],       # which module types to replace
    skip_modules=["lm_head", "embed_tokens"],  # which named modules to skip
    C=1.0,                                   # clipping threshold (per-layer)
    noise_multiplier=1.0,                    # σ for Gaussian noise
    sample_size=None,                        # batch size (inferred at runtime if None)
)

# Train with standard optimizer — DP is handled inside the fused layer
optimizer = torch.optim.Adam(dp_model.parameters(), lr=1e-4)
```

### 3.2 Per-layer module: `DPLinear`

```python
from opake.patches.fused import DPLinear

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

The fused path does **not** use Opake's `clipped_grad()` pipeline, so privacy accounting must be handled separately:

```python
from opake.dpsgd.accounting import gaussian_accounting

accountant = gaussian_accounting(
    sample_rate=1.0 / num_steps,
    noise_multiplier=1.0,
    num_steps=num_steps,
    clipping_norm=1.0,
)
epsilon = accountant.epsilon(delta=1e-5)
```

The `noise_multiplier` is passed directly to each `DPLinear` instance via `wrap_model()`. This matches FlashDP's approach — the noise is added per-layer during backward, not in a centralized optimizer step.

**Open question:** How to reconcile per-layer noise addition with Opake's existing noise allocation (per-group/structured noise)? In v1, we use simple per-layer isotropic Gaussian noise. Structured noise allocation is a v2 feature.

## Milestones

### Phase 1: Foundation (Week 1–2)

- [ ] **1.1** Port Triton kernels from FlashDP (`mm_clip.py`, `bmtm_clip.py`, `clip_fn.py`, `utils.py`)
  - Adapt to Opake's Triton kernel patterns (existing kernels in `opake-patches/src/opake/api/patches/kernels/`)
  - Add dtype support: fp32, bf16 (defer fp16)
  - Add `autograd.Function` compatibility checks

- [ ] **1.2** Implement `DPLinear` layer
  - `autograd.Function` wrapper with fused backward
  - Fallback to PyTorch path when Triton unavailable
  - `RuntimeAutoTuner` integration

- [ ] **1.3** Numerical correctness tests
  - `test_linear_clip_correctness.py` — compare fused vs. reference PyTorch path
  - Tolerance: `rtol=1e-4, atol=1e-5` for fp32, `rtol=1e-2, atol=1e-3` for bf16
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
  - Single-layer benchmark: fused vs. `clip_pytree` vs. non-DP
  - Full-model benchmark: GPT-2 small / Llama-2 7B (if GPU available)
  - Target: ≥50% of non-DP throughput for Linear-dominated layers

### Phase 3: Integration with Opake (Week 5–6)

- [ ] **3.1** Public API surface under `opake.patches.fused`
  - `__all__` declaration in façade
  - Factory function signatures finalized
  - Deprecation/guard for unsupported dtypes

- [ ] **3.2** Integration tests
  - End-to-end training step: fused model + standard optimizer produces valid updates
  - Privacy accounting integration: verify noise magnitude matches `noise_multiplier`
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
  - CUDA tests in `packages/opake-patches/tests/fused/`
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
| GPU memory regression for very large batches | LOW | Per-block processing limits SRAM usage; HBM pressure is bounded by output shape |

## Open Questions

1. **Per-layer vs. flat clipping:** FlashDP uses per-layer clipping. Opake currently supports both. Should the fused path support flat clipping (requires cross-layer norm accumulation)? **Decision: v1 = per-layer only. Flat clipping deferred.**

2. **Noise addition timing:** FlashDP adds noise during backward (per-layer). Opake's existing path adds noise after clipping in the optimizer step. Which is correct for Opake's accounting? **Decision: v1 follows FlashDP's per-layer noise. Accounting must be adjusted to match.**

3. **Attention layers:** Opake's patches include fused attention kernels. Should we fuse clipping into attention backward? **Decision: Deferred to v2. Linear layers cover ~80% of model params in transformers.**

4. **Distributed training:** How does per-layer fused clipping interact with FSDP/DDP? **Decision: Out of scope for v1. The per-layer design is inherently local, so DDP should work if each rank clips its own shards.**

## Verification Plan

### Numerical correctness (gate)

```python
# Must pass before any performance claims
@torch.no_grad()
def test_correctness(dtype=torch.float32):
    for B, T, D, P in [(2, 64, 512, 512), (4, 128, 4096, 4096), (1, 32, 256, 1024)]:
        input = torch.randn(B, T, P, dtype=dtype)
        grad_output = torch.randn(B, T, D, dtype=dtype)
        C = 1.0

        fused_result = fused_linear_backward(input, grad_output, C)
        ref_result = reference_clip(grad_output, input, C)

        assert torch.allclose(fused_result, ref_result, rtol=1e-4, atol=1e-5), \
            f"Shape {(B,T,D,P)}: max_diff={torch.abs(fused_result - ref_result).max()}"
```

### Performance (non-blocking gate)

```python
def benchmark_fused_vs_reference():
    # Fused must be >= 50% faster than reference PyTorch path
    # or the RuntimeAutoTuner should select the reference path automatically
    pass
```

### Integration (gate)

```python
def test_end_to_end():
    model = GPT2Small()
    dp_model = wrap_model(model, target_modules=[nn.Linear], C=1.0, noise_multiplier=1.0)
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
- FlashDP code: https://github.com/kaustpradalab/flashdp
- Opake existing clipping: `packages/opake-engine/src/opake/api/engine/clipping/`
- Opake existing Triton kernels: `packages/opake-patches/src/opake/api/patches/kernels/`
- Opake architecture contracts: `.junie/architecture-contracts.md`
