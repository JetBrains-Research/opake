# Pretrained MoE balancing comparison

This experiment tests a hypothesis, not a promised positive result. A pretrained
MoE need not gain prediction accuracy from more uniform expert use. The earlier
single-seed CodeAlpaca pilot did not establish a balancing benefit.

## Model and training task

- Pinned `JetBrains/Mellum2-12B-A2.5B-Base`, 64 experts/top-8, 28 sparse layers.
- Completion-only SFT on pinned public Magicoder OSS-Instruct Python records.
  The source file is labelled decontaminated by its publisher; this is not proof
  that the checkpoint's pretraining excluded any evaluation benchmark.
- Expert LoRA (rank 4, alpha 8) and unwrapped FP32 routers are trained. The
  approximately 12B base parameters remain frozen. This is adapter SFT, not
  full-model fine-tuning or pretraining.
- Whole prompt/solution records, maximum 512 tokens, no packing. Overlength
  answers are rejected rather than truncated. Fixed, prompt-deduplicated
  partitions: 8,192 train, 256 development, 1,024 final test, 64 diagnostic.
  The test partition is reserved before training and never selects coefficients.
- A pre-confirmation exact/13-word overlap audit against HumanEval+/MBPP+ flagged
  five potential training records; their prompt hashes are excluded in the
  configuration. Deterministic refill keeps 8,192 training records and
  leaves every reserved validation/test/diagnostic record unchanged. Such a
  text audit does not establish semantic independence or clean pretraining.

## Four matched controls

| Arm | Gradient noise | Balancing state |
| --- | --- | --- |
| `reference` | None | Off |
| `reference_aux` | None | Lagged exact loads |
| `dp` | Gaussian | Off |
| `dp_aux` | Gaussian | Lagged private loads |

All four use the same per-record clipping bound, SGD optimizer, trainable
parameters, dataset partitions and public initialization seed. **The non-private
arms are matched clipped controls, not optimized ordinary/unclipped SFT.** The
balanced non-private control deliberately uses the same lagged surrogate, not a
different batch-global Switch objective. This isolates noise and load-release
effects without attributing an objective change to privacy.

The PR pools noisy load estimates across layers before constructing its shared
expert-index penalty. **Pooled index balance is not per-layer resource balance:**
two layers can each use a different single expert and look perfectly balanced
after pooling. Report pooled-index CV and per-layer CV separately. A reduction
in the pooled objective alone does not meet the practical criterion below.

Each private run targets epsilon 8, delta 1e-5, at 256 Poisson-sampled updates.
The balanced accountant includes both releases. Equal privacy does not mean
equal gradient noise. Guarantees are sequence-level add/remove, per run, and do
not retroactively cover pretraining. Public data permits these controls and
diagnostics; releasing multiple models on truly private data needs composition.

## Diagnostic before confirmation

Measure actual task versus balancing router gradients on the diagnostic split,
alongside the theoretical standard deviation of the private load estimate and
its EMA. Check that the configured coefficient is not effectively zero, that
the sign encourages overloaded experts to lose load, and that the mechanism
survives the real GPU forward/backward path. The uniform initial load state gives
zero balancing gradient on the first update by design.

The frozen-public diagnostic at coefficient 0.2 measured an auxiliary/task
router-gradient norm ratio of 0.118. Before test evaluation, the confirmation
coefficient was set to 2.0, making its initial unclipped router contribution
approximately 1.18 times the task contribution rather than a weak perturbation.
This is not a measurement of its post-clipping contribution. The other settings
are ratio 1.0 and EMA beta 0.95. Compared with the old pilot, more privacy is
allocated to load estimation, with the extra gradient noise correctly accounted. These are
explicit hypotheses, not known-optimal settings. Any diagnostic changes must
be frozen and recorded before main test-set evaluation, with all attempted
diagnostic configurations retained. Do not rerun until a positive test result.

## Measurements and interpretation

1. **Generated code:** greedy HumanEval+ `pass@1` (all 164 tasks), original-test
   and extended-test correctness, confidence intervals, syntax and generation
   truncation rates. MBPP+ (all 378 tasks) is the secondary benchmark when the
   runtime budget permits. Same prompts, decoding and token limits for the
   unchanged checkpoint and every evaluated trained checkpoint. Small subsets
   are explicitly smoke checks and must not enter the main results table.
   The fixed `entry_point_top_level_v1` policy stops after the requested function,
   before unrelated top-level code; it is identical across arms and never uses
   test outcomes to select an answer. Raw and graded programs are preserved,
   no sanitizer repairs the implementation, and genuine coding errors count as
   failures. The maximum remains 512 generated tokens.
2. **Prediction:** held-out example-averaged and token-weighted completion NLL,
   plus teacher-forced token accuracy. Perplexity and NLL are not independent
   evidence; teacher-forced accuracy is not program correctness.
3. **Routing:** per-layer/global CV, effective experts, max/mean expert load,
   underused-expert fraction, and batch-local load skew. Measure at initialization
   and intermediate/final updates, not just one pooled final mean.
4. **Cost:** update time, actual evaluation throughput, peak GPU memory, and
   accounting. Single-GPU load balance is not evidence of distributed speedup.

Predeclared practical balancing target: at least 10% lower mean layer load CV,
without more than 2% relative test-NLL degradation or 2 percentage points of
HumanEval+ pass@1 degradation. Report observed differences and uncertainty even
when these thresholds are missed. A single run or overlapping uncertainty is
inconclusive, not proof of equivalence. Prediction improvement is desirable but
not assumed to follow from load balance.

Repeat matched arms with public initialization seeds 0, 1 and 2, fresh unreported
private sampling/noise randomness, and fixed splits/configuration. Report all
seeds, mean, standard deviation and paired differences. The initial four-arm
comparison takes priority over extra seeds. Incomplete seeds, failures and the
72-hour runtime ceiling remain visible; do not silently exclude poor runs.

## Execution and outputs

`configs/sft_balancing.json` defines the starting configuration. Run each arm in
a fresh directory with native W&B tracking; keep old pilot sources/results
separate. See `TRACKING.md` for credentials and logging. Raw training telemetry,
private RNG state and optimizer state are not published.

`python -m examples.moe_privacy.campaign run --help` describes the serial
controller. It starts reference SFT immediately, then evaluates the unchanged
checkpoint before scoring the first fine-tuned checkpoint. All four arms run
for each requested seed, with full HumanEval+ evaluation after each checkpoint.
The secondary MBPP+ stage starts only after the primary comparison is complete,
if time remains. Its scores are appended without replacing HumanEval+ results.
`report.md` and `report.json` are regenerated after each stage, and missing
measurements remain explicitly pending. The controller stops on its first
failure or absolute deadline; it does not silently retry/select successful runs.

Generated code executes only inside a resource-limited, network-disabled
container, never on the workspace host. A successful container gate is required
before the benchmark. The report must distinguish mechanism checks, diagnostic
runs, complete confirmation runs, and pending measurements.