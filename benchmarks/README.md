# Clipping memory diagnostic

`clipping_memory.py` exercises the installed native `clipped_grad` path, not a
reimplementation or trainer. It uses fixed `C=1`, `return_aux=True`, first moments
only, and `normalize_by=256`. One call processes at least three consecutive
microbatches. Each physical-batch/checkpoint pair runs in a **fresh subprocess**.
There is no optimizer, noise, parameter update, activation offload, compilation,
dataset, tokenizer, or experiment tracker. This is not a private training run.

## Bounded CPU smoke

From the repository root, using the existing environment:

```bash
.venv/bin/python benchmarks/clipping_memory.py --label cpu-native --output cpu-native.jsonl
```

This runs batches **4 and 5**, checkpoint **off and on**, three chunks each,
with a seeded two-block FP32 residual MLP (131,072 trainables), 16 positions,
width 128, hidden width 256, bounded inputs in `[-1, 1]`, and one CPU thread.
It deliberately does not approximate a 7B model's memory consumption.
`--chunks N` accepts `N >= 3`; `--batch-sizes 4` and `--checkpoint on` select
individual cases. Run `--help` for the full CLI. Output files refuse overwrite.

The 12 or 15 records are a diagnostic slice, **not a full logical batch of
256 records**. Its summed gradient is still normalized by 256. Chunk count,
batch size, seed, normalization, environment versions, source paths, git revision,
dirty status, and clipping-source fingerprints are recorded in JSONL. A source
change during the measured call fails the case. Do not edit measured worktrees.

## Baseline, lifetime-only, and streaming refs

Use clean, separate worktrees to distinguish the original baseline from the
commit that changes buffer lifetimes and the commit that adds streaming. The
refs used for the recorded CPU evidence are:

```bash
BASELINE_REF=5c31065facb5c0115fd708ae731101629c274ce4
LIFETIME_REF=ae2c6e003599e2cc5ed1377dee021ce057f1b4d2
STREAM_REF=98e0eb64a08a1615e0c0b33d7b2cc2db635693f6

git worktree add --detach ../clipping-baseline "$BASELINE_REF"
git worktree add --detach ../clipping-lifetime "$LIFETIME_REF"
git worktree add --detach ../clipping-stream "$STREAM_REF"

.venv/bin/python benchmarks/clipping_memory.py --target cpu \
  --source-root ../clipping-baseline --label baseline --output baseline-cpu.jsonl
.venv/bin/python benchmarks/clipping_memory.py --target cpu \
  --source-root ../clipping-lifetime --label lifetime-only --output lifetime-cpu.jsonl
.venv/bin/python benchmarks/clipping_memory.py --target cpu \
  --source-root ../clipping-stream --label stream --output stream-cpu.jsonl
```

Each command runs the entire 4/5 × off/on matrix. The benchmark script need not
exist in those refs: `--source-root` prepends their `packages/*/src` paths in the
worker's environment, using the **same existing interpreter and dependencies**.
This targets refs with the current split-package layout and compatible APIs;
it does not rebuild native extensions or install dependencies. Check `engine_file`
and `engine_git_head` in every result. Without `--source-root`, imports resolve
normally to the installed/editable packages; the script does not change them.

For an additional comparison within a streaming-capable ref:

```bash
.venv/bin/python benchmarks/clipping_memory.py --source-root ../clipping-stream \
  --variant original --label stream-disabled --output stream-disabled-cpu.jsonl
```

`--variant original` sets the private `_streaming_supported` selector to false
**only inside the benchmark worker**, if that selector exists. On earlier refs
it is a no-op. It does **not** undo lifetime fixes and therefore is not the
historical baseline. `--variant native` (default) never overrides dispatch;
`clipping_function_calls` records whether `_stream_clip_and_sum` actually ran.
No new public engine switches are required. These private observation seams may
need adjustment on incompatible future refs.

## Recorded CPU evidence

[results/clipping-memory-cpu.json](results/clipping-memory-cpu.json) preserves
the measured case data, complete clipping-source SHA-256 manifests, benchmark
fingerprint, exact revisions, input hashes, versions, numerical check counts,
and decoder body-entry counts. Shared fields are stored once, with their scope
documented in the artifact. Absolute local paths and process IDs are removed;
original capture basenames and digests retain provenance.

Baseline **`5c31065f`** and lifetime-only **`ae2c6e00`** were measured in clean
worktrees. The canonical streaming matrix was measured at committed
**`98e0eb64`**, with every measured clipping-source hash checked against the
commit. The original streaming measurement was **uncommitted but fingerprinted**
on top of `ae2c6e00`; it is retained as superseded evidence. Its clipping-source
hashes and the measurements below match the committed run. The stream-disabled
comparison also used those uncommitted sources with the benchmark-only selector
override; it was not rerun at the final commit and is not the historical baseline.

All memory values below are **sampled live-storage proxy bytes on CPU**, not
allocator usage, RSS, or CUDA peaks. Each row contains checkpoint off/on cases;
only the forward peak differs between them. Backward includes recomputation.
Clipping/reduction was the largest phase in every case, so its peak equals the
overall sampled peak. “Prior-live at chunk 3” is the sum of earlier-chunk origins
just before the third forward, excluding persistent model/input storage.

| Variant | Batch | Forward peak off / on | Backward peak | Clipping/reduction = overall peak | Prior-live at chunk 3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline `5c31065f` | 4 | 1,900,676 / 1,835,124 | 3,834,004 | 9,273,628 | 1,048,688 |
| Baseline `5c31065f` | 5 | 1,982,624 / 1,900,684 | 4,399,284 | 11,198,812 | 1,048,712 |
| Lifetime-only `ae2c6e00` | 4 | 1,376,372 / 1,310,820 | 3,309,700 | 8,749,324 | 524,384 |
| Lifetime-only `ae2c6e00` | 5 | 1,458,320 / 1,376,380 | 3,874,980 | 10,674,508 | 524,408 |
| Stream `98e0eb64` | 4 | 1,376,372 / 1,310,820 | 3,309,700 | 5,996,824 | 524,384 |
| Stream `98e0eb64` | 5 | 1,458,320 / 1,376,380 | 3,874,980 | 7,135,576 | 524,408 |
| Stream-disabled (uncommitted) | 4 | 1,376,372 / 1,310,820 | 3,309,700 | 8,749,324 | 524,384 |
| Stream-disabled (uncommitted) | 5 | 1,458,320 / 1,376,380 | 3,874,980 | 10,674,508 | 524,408 |

The artifact also retains the between-chunk phase peak and the third-forward
total live bytes and origin-phase breakdown. Prior-live values include expected
running sums and auxiliary values; they must not be interpreted as leaked bytes.

Every case passed the nonempty/finite checks: **4 gradient tensors, 131,072
gradient elements**, and **12 (batch 4) or 15 (batch 5) elements in each** of
`loss_values`, `grad_norms`, `clipped_grad_norms`, and `loss_aux`. Each of the two
decoder bodies entered forward three times; checkpoint-on had three additional
backward re-entries per body, checkpoint-off had zero. Each native streaming
case called `_stream_clip_and_sum` three times; baseline, lifetime-only, and
stream-disabled called `clip_pytree` and `global_norm` three times each. Input
hashes and scalar checks matched across variants, not a tensorwise equivalence
proof. Versions were torch **2.14.0**, Transformers **5.17.0**, and PEFT **0.20.0**.
CUDA allocated/reserved metrics and throughput are **null**, not zero.

**Not established:** CUDA/H100 behavior, execution of the target Qwen model,
sustained reliability, batch-5 fit on that target, production Inductor CUDA
memory, throughput, or publication-quality performance claims. These are
single instrumented, three-chunk CPU invocations with no optimizer or noise,
not full training runs. Sanitized finite pre-clipping norms can conceal overflow.

## Optional CUDA target

Only run on a suitably large, BF16-capable CUDA GPU with the model already
cached. This workload can OOM even with checkpointing. No model is downloaded
unless `--allow-download` is explicitly supplied.

```bash
for variant in baseline lifetime stream; do
  .venv/bin/python benchmarks/clipping_memory.py --target qwen \
    --source-root "../clipping-$variant" --label "$variant" \
    --output "$variant-qwen.jsonl"
done
```

The target is `Qwen/Qwen2.5-Coder-7B`, pinned to revision
`0396a76181e127dfc13e5c5ec48a8cee09938b02`, sequence length 3072, frozen BF16
base parameters and FP32 LoRA parameters. LoRA uses rank 384, alpha 352,
dropout 0, no trainable bias, and `q_proj`, `k_proj`, `v_proj`, `o_proj`,
`gate_proj`, `up_proj`, `down_proj`. The script asserts **968,884,224 trainable
parameters** and the parameter dtypes. It uses Opake runtime/model patches,
kernel patches, loss-only fused linear cross entropy, SDPA, disabled KV cache,
BF16 autocast, and non-reentrant checkpointing. It calls `make_functional` with
trainable/frozen partitioning and detached parameters.

Token IDs are seeded synthetic uniform samples across the vocabulary; labels
match the tokens and causal-LM loss performs the shift. Checkpoint settings
receive identical input hashes and initialization. Deterministic algorithms are
required, TF32 is disabled, and cuBLAS workspace configuration is fixed. Exact
reproducibility across hardware/library versions or custom kernels is not
guaranteed. The CPU smoke does not validate CUDA kernels or this model path.

## Reading results and limitations

- `cuda_allocator_peak_bytes_by_phase` reports **allocated and reserved** CUDA
  allocator peaks separately, with synchronization and peak resets at phase
  boundaries. Values are absolute, not deltas; reserved memory includes cached
  blocks and need not fall when tensors die. No `empty_cache` occurs between
  chunks. Setup and post-call output validation are excluded from successful
  measured peaks. The maximum of the reported phase peaks is the measured
  invocation peak. External CUDA allocations are not included.
- `observed_live_storage_peak_bytes_by_phase` is a **sampled storage-byte proxy**,
  not RSS, allocator usage, or a CUDA peak. Dispatch observations deduplicate
  shared storage and keep only native **weak storage handles** plus numeric
  metadata, never tensors or frames. They include parameters, buffers, inputs,
  and observed operation results. They can miss allocations inside fused
  operators, unobservable tensor wrappers, and allocations between observations;
  `unobservable_storage_observations` counts access failures. On CPU these are
  the only memory measurements; CUDA fields are null.
- `forward_entries` shows live bytes immediately before each chunk's forward,
  including `prior_chunks_live_bytes_by_origin_phase`. These are surviving
  storages first observed in earlier chunks, **not necessarily leaked gradients**:
  the running sum and requested per-record auxiliary values should survive.
  Persistent model/input storages are excluded from this overlap breakdown.
  Origin phases describe first observation, not semantic ownership.
- Phase attribution is approximate: forward ends when the scalar loss returns;
  backward includes checkpoint recomputation; clipping/reduction starts at the
  first observed native norm/clip/stream helper and includes aggregation and
  auxiliary construction until the next chunk. Unsupported helper layouts can
  leave clipping attributed to backward; check `clipping_function_calls`.
- `decoder_forward_body_entries` counts calls to the actual decoder forward
  code, not checkpoint API calls or module hooks. Every block must enter forward
  once per chunk. Checkpoint-on must also re-enter in backward at least once per
  chunk; checkpoint-off must not. Recomputation can stop before the body returns.
- `ClippedPytree.pytree` and `ClippedGradAux` tensor fields are explicitly
  unwrapped and checked for **nonempty, finite** results. Cardinalities,
  post-clipping bound, and normalization metadata are checked. Sanitized finite
  pre-clipping norms can conceal overflow; this check is not proof of finite
  unsanitized gradients. Scalar checksums are diagnostic, not an equivalence test.
- `instrumented_elapsed_seconds` includes dispatch, weak-lifetime sampling,
  Python profiling, synchronization, and cold-call overhead. **Throughput is not
  measured** (`throughput: null`); do not derive speed claims from these timings.
  Instrumentation itself can change scheduling and memory lifetimes.
- OOM/error records include the current phase, chunk, and last dispatched
  operation; the controller returns nonzero but continues the other cases.
  Setup OOMs have chunk `-1`. A killed process may have only a start record and
  `worker_exit`, not a phase or peak measurement. CUDA asynchronous failures can
  be reported at the next synchronization boundary.
