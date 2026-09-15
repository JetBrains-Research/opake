# Patched-model performance evidence

Performance changes to model patches need evidence from the same workload on a
baseline commit and the candidate commit. The repository tool
`tools.performance.patched_model_evidence` records that evidence without making
noisy wall-clock measurements a pull-request gate.

This focused cross-revision layer complements the broader benchmark suite
tracked in #417 and #511. It does not maintain a second benchmark registry,
claim inventory, scheduled runner, or table generator.

## Recording comparable runs

Use the same command, machine, environment, seed, and JSON configuration in two
worktrees. Only `--variant` and `--output` should change:

```bash
# At the main baseline.
uv run python -m tools.performance.patched_model_evidence record \
  --variant main \
  --device cuda \
  --warmup 3 \
  --repeats 10 \
  --config-json '{"batch_size": 4, "sequence_length": 1024}' \
  --output artifacts/main.json

# At the candidate commit.
uv run python -m tools.performance.patched_model_evidence record \
  --variant candidate \
  --device cuda \
  --warmup 3 \
  --repeats 10 \
  --config-json '{"batch_size": 4, "sequence_length": 1024}' \
  --output artifacts/candidate.json

uv run python -m tools.performance.patched_model_evidence compare \
  artifacts/main.json artifacts/candidate.json \
  --output artifacts/comparison.json
```

The built-in tiny Llama workload has random weights and downloads nothing. It
uses production `apply_model_patches()` and reports `forward`, `backward`,
`per_example_gradient_and_clipping`, `optimizer`, and `evaluation` separately.
It is a harness smoke case, not representative performance evidence. Child PRs
should provide a workload plug-in for the relevant model and shape.

Recording requires a clean worktree so the commit identifies every
implementation input. `--allow-dirty` exists only for local diagnostics and
preserves the dirty paths in metadata.

## Artifact contract

Run artifacts use schema `opaque.patched-model-evidence.v1` and contain:

- the full commit, dirty paths, command, seed, warmup/repeat counts, source
  digest, package versions, selected environment variables, platform, and
  accelerator properties; allocator, thread-count, determinism, and compiler
  settings participate in the comparison fingerprint;
- workload configuration, selected backend per stage, and compiler/Dynamo/
  Inductor counter deltas when the installed PyTorch exposes them; workload
  plug-ins may also expose route, graph, or kernel counters;
- every synchronized raw duration, throughput, allocator peak, incremental
  peak, peak reservation, end allocation, end reservation, and selected-backend
  sample;
- median and p95 duration/throughput plus median and maximum memory;
- numerical output and parameter-gradient checks with tolerances and observed
  errors.

CUDA allocator peaks are exact. MPS peaks are labeled with
`peak_exact=false` when the installed PyTorch cannot reset allocator peaks. CPU
does not pretend to report allocator memory. Inexact observations remain in the
raw artifact but produce `memory_status="unavailable"` rather than an enforceable
memory verdict.

The comparison fingerprint covers workload shape, stage units, hardware,
platform, Python, and relevant dependency versions. The comparator refuses
different fingerprints rather than producing an attractive but invalid ratio.
Selected backends are reported but excluded from the fingerprint because a
candidate may intentionally change dispatch.
Dirty diagnostic artifacts cannot be compared.

An abbreviated comparison looks like:

```json
{
  "correctness_passed": true,
  "performance_passed": false,
  "stages": [
    {
      "name": "backward",
      "baseline_backend": "opaque-patched:cuda-triton-auto",
      "candidate_backend": "opaque-patched:cuda-triton-auto",
      "throughput_ratio": 0.93,
      "throughput_status": "regression",
      "peak_memory_ratio": 0.81,
      "memory_status": "pass"
    }
  ]
}
```

A memory improvement never masks a throughput regression, and a throughput
improvement never masks a memory regression.

## Regression policy

Schema, comparability, and numerical/gradient checks are stable correctness
contracts and may run in ordinary CI. Timing and accelerator-memory results are
evidence:

- the default comparator reports independent throughput and memory statuses but
  exits successfully when only a performance budget is exceeded;
- `--enforce-performance` is opt-in for a controlled, repeated benchmark host;
- `--max-throughput-regression` and `--max-memory-regression` set explicit
  independent budgets (fractions, not a speed-or-memory disjunction);
- raw samples remain the review artifact so medians, p95s, variance, warmup,
  dispatch changes, and allocator precision can be audited.

This avoids flaky PR gates while still allowing an H200 or MPS evidence job to
enforce a predeclared budget.

## Adding a child workload

Pass `--workload package.module:create_workload`. The factory receives
`(device, config, seed)` and returns a `Workload`. Each `Stage` declares its
unit count and actual selected backend, an untimed `prepare`, and the measured
callable. A backward stage should build its graph in `prepare`; an end-to-end
training stage should build and execute the whole step in the measured callable.
Its optional `counters` callback can report route-plan reuse, backend dispatch,
graph count, or kernel launches when that workload can observe them reliably.
For dynamic dispatch, `backend` may be a callback that reports the route chosen
by the completed sample; mixed routes remain visible in the raw samples.

The workload's correctness callback must return at least one `numerical` and one
`gradient` `CheckResult`. MoE children should add frozen/trainable-expert cases
and record dense/grouped dispatch. Attention children should encode attention
implementation, GQA ratio, and sliding-window shape. Dense-kernel children
should encode dtype, vocabulary, LoRA rank, and trainable parameters. Integration
children should use the same contract for full trainer stages.

Keep generated artifacts outside the repository unless a PR or the benchmark
evidence suite intentionally commits them. Do not turn a one-host result into a
general performance claim.
