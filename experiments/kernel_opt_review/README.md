# Review of `mihajlo/kernel-optimizations` (6 commits) for `feat/dp-fused-per-layer-clipping`

Date: 2026-10-01. Branch base `5657d6ce` (#1101), so it predates #1108. None of
its 17 files changed on `main` or here since, so there were no conflicts and
nothing had been upstreamed. Protocol: `.junie/review-guidelines.md` and
`.junie/differential-privacy-review.md` (noise and RNG are privacy-sensitive).
CUDA numbers: A100-SXM4-40GB, 392-leaf / 40.4M-element LoRA-7B-like fp32 tree.

## Verdicts

| Commit | Change | Verdict | Evidence |
|---|---|---|---|
| `07a10621` | BLT/Toeplitz: per-device coefficient cache; `tensordot` → broadcast-sum; `diag(decay) @ state` → elementwise | **Partly ported** (with `2913e229`, `d1fbb21e`) as `65fd7f28` | `bench_noise_variants.py`, `bench_mf_port.py` |
| `2913e229` | Toeplitz cache by (device, dtype); `generator_from_key(device=)`; `benchmarks/bench.py` | Cache idea ported (fixed); `device=` kwarg **not ported**; bench not ported | see below |
| `883a9e3f` + `617d1bfa` | `_iid_normal_noise`: one flat CPU draw per device | **Not ported** | `check_rng_stream.py`, `bench_noise_variants.py` |
| `d1fbb21e` | Fix multi-D broadcast in BLT/Toeplitz | Folded into the port; the bug was introduced by `07a10621` (main passes 623/623; `07a10621` fails 28) | dpftrl suite at each commit |
| `3a9bdcdc` | Fused LoRA MLP `save_intermediates` option, attention-only checkpointing, trainer flags, docs, notes | **Ported** (`a05b0b8e`), without the notes file and with a doc fix. The attention-only checkpointing part was later removed in `a2ce1712`: once it fired, it saved 0.83 GiB for +27% step time, because SDPA never stores the attention matrix. `lora_mlp_recompute` remains. | `check_save_intermediates.py`, A/B test runs |

## Why each part was or wasn't taken

**Ported, numerically identical to main** (bitwise-equal outputs, A100):
- Coefficient cache keyed by (device, dtype), filled on first use. Toeplitz
  entries track `coef._version` and are bypassed for grad-carrying `coef`.
- Elementwise BLT buffer decay instead of the diagonal matmul.
- `multiply_next` over the tree: BLT **~112 → ~38 ms (~3.0×)**, Toeplitz
  C⁻¹ **~80 → ~59 ms (1.34×)**.

**Not ported:**
- **`tensordot` → broadcast-multiply-sum:** **1.8–2.3× slower** on CUDA at
  40M elements (it materializes an `[nb, *leaf]` temporary).
- **Device-only BLT cache filled only in `_init`:**
  - `KeyError` when a fresh matrix resumes from saved state (checkpoint
    restore) or a state changes device.
  - Wrong coefficient dtype for mixed fp32/fp64 trees.
- **Unconditional Toeplitz cache:** serves stale coefficients after an
  in-place (optimizer) update when a dtype conversion is involved.
- **Flat i.i.d. draw:**
  - **Slower** on CUDA: 385 vs 350 ms.
  - **2× peak device memory**: 310 vs 157 MiB.
  - "Bit-for-bit determinism" holds only if every leaf size is a multiple of
    16. Otherwise the stream changes, and λ-CGD regenerates the previous
    step's noise from its key, so a checkpoint from an older version would
    resume with a different z_{t-1} (a privacy-relevant mechanism change at
    the resume step).
  - Non-tensor leaves are silently returned without noise instead of
    raising.
  - A stream-preserving batched variant (per-leaf draws into one buffer) is
    bit-exact but not faster on CUDA (397 ms): CPU `randn` dominates, not
    transfers.
- **`generator_from_key(device=...)`:** unused public-API parameter added
  "for future support". AGENTS.md forbids forward references.

**Regression tests added** (`test_buffered_toeplitz.py::TestBLTStreamingCoefficients`,
`test_toeplitz.py::TestInverseStreamingCoefficients`, 18 tests): main and the
port pass. The k-opt tip fails the restore, mixed-dtype,
optimizer-update-staleness, in-place-staleness and direct-update cases;
`07a10621` additionally fails the multi-D broadcast cases.

**`3a9bdcdc` (independent reviewer + verification):**
- The reviewer's CRITICAL finding ("`save_intermediates` ignored under
  `vmap(grad())`") is **refuted**. Instrumented backward sees
  `ctx.save_intermediates=True` with 12 saved tensors (False → 10).
- Per-example grads match an unfused reference to 2.6e-7 / 2.3e-7 relative.
  Recompute vs save differ by 1.9e-9 abs.
- Valid finding: attention checkpointing is a silent no-op unless the model is
  in training mode, grad is enabled, and no KV cache is passed.
  `examples/train_dpsgd.py` runs in eval mode, so it never fires there, and HF
  causal LMs default to `use_cache=True`, so `DPTrainer` is unverified too.
  The ported user doc states the condition. No code change, per instruction.
- `notes/checkpointing-sweep-2026-09.md` is not ported: its null-result
  conclusion was overturned by that root cause.
- A/B on the A100, branch head with vs without `3a9bdcdc`: the commit's own
  tests pass (39 passed, 3 skipped).
  - **Zero new failures** in opake-patches (CUDA and CPU) or
    opake-transformers (CPU).
  - The 36 failures in both arms (Triton kernel/checkpoint/offload tests,
    `gpt_oss` parity, TRL SFT/DPO) also fail on `main` `a3214a97`, so they are
    pre-existing in that environment, not caused by this branch.
