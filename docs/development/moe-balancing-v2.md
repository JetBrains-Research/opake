# Mellum2 balancing: confirmation campaign

## September 22 addition: native TRL controls

Two genuine `trl.SFTTrainer` controls were **queued at 13:24:10 UTC** under
`opaque-moe-trl-baselines-20260922.service`. The recorded state is
`waiting_for_private_followup`; neither new training arm has started yet.
The coefficient-0.2 private follow-up and its original source are unchanged.

- `trl_reference`: non-private native SFT, auxiliary coefficient zero.
- `trl_reference_aux`: non-private native SFT with Transformers' current-minibatch
  auxiliary loss, coefficient **0.2**, not Opaque's unnoised lagged surrogate.
- Same pinned Mellum2 checkpoint, Magicoder partition hashes, expert LoRA/router
  trainability, initialization zero, learning rate and 256-update budget. No
  additional initialization repetitions or private-baseline reruns.
- Conventional training differences are explicit: shuffled batches, physical
  batch 4 / accumulation 8, averaged physical-batch token-mean losses, and norm
  1.0 clipping of the accumulated batch gradient. This is a standard-SFT quality
  reference, not a comparison changing only privacy noise.
- Task metrics remain **HumanEval+ pass@1, held-out answer NLL, teacher-forced
  token accuracy, training time and GPU memory**. Both training arms precede
  their saved-checkpoint code evaluations, with the same full scoring protocol.
- Pinned `trl==1.13.0` and the new source live separately under the remote
  campaign's `trl-baselines-20260922/{deps,source}/`; the active environment was
  not upgraded. The source transfer was bounded to 65,661 bytes.
- **66 focused tests passed**, including five isolated actual-TRL CPU cases,
  native auxiliary-gradient/accumulation checks, adapters, tracking and queue
  validation. Linux imports/configuration passed. The full pretrained CUDA
  two-update engineering gate remains queued before the two full controls.
- W&B group: `mellum2-native-trl-controls-20260922` in `federated-compute/opaque`.
  Training runs appear only when they start, not while the controller waits.
  Results go to `trl-baselines-20260922/results/<arm>/`; `comparison.json` and
  `comparison.md` retain pending scores and distinguish these controls from the
  earlier Opaque non-private arms.
- The original **September 22, 20:10 UTC cutoff remains in force**, including
  the wait. No workspace timeout extension was made; completion of both new
  controls and code scoring before that cutoff is not guaranteed.

See [native TRL protocol and commands](../../examples/moe_privacy/TRL.md).

## September 22 follow-up: coefficient 0.2

One private auxiliary-loss follow-up was submitted at **12:40:46 UTC**, using
the [coefficient-0.2 configuration](../../examples/moe_privacy/configs/sft_balancing_aux02.json).
Its only configuration change is `router_aux_loss_coef: 2.0 -> 0.2`.
The completed private baseline (`82540020`) and coefficient-2 run (`b97e2db1`)
are retained; no baseline retraining or initialization repetitions are scheduled.

- [Live W&B run](https://jetbrains.wandb.io/federated-compute/opaque/runs/2e891b34):
  `magicoder-dp-aux-coef0.2-s0-20260922`. At **12:45 UTC**, at least **2/256**
  optimizer updates were observed on the H100; results remain pending.
- This is completion-only **SFT of an already pretrained Mellum2 base model** on
  Magicoder Python problem/solution pairs, not pretraining. Only expert LoRA
  adapters and routers are trainable; the remaining backbone stays frozen.
- The existing non-private balanced control uses Opaque's `DPTrainer` with the
  lagged auxiliary objective and noise disabled, retaining per-record clipping.
  It is not TRL's conventional current-batch balancing baseline.
- Total target remains epsilon 8, delta 1e-5, per private run. Load-noise ratio
  1.0 and filter beta 0.95 are unchanged: the smaller coefficient does not reduce
  the additional accounting cost or gradient-noise allocation.
- Success is assessed through **HumanEval+ pass@1**, held-out answer NLL,
  teacher-forced token accuracy and training cost, not routing uniformity.
  The same full 164-task generation/scoring protocol runs after training; an
  automatic comparison checks dataset hashes, matched privacy and protocol identity.
- This is an exploratory single-run follow-up on a previously evaluated benchmark,
  not an independent statistical confirmation or a causal utilization claim.
- Unit: `opaque-moe-aux02-20260922.service`; original cutoff remains
  **2026-09-22 20:10 UTC**. Training source and existing results are unchanged.
- Outputs are under the remote campaign's `followup-aux02-20260922/`: `dp_aux/`
  contains metrics, weights and code scores; `logs/` contains training/scoring
  logs; `campaign_state.json` and `comparison.json` track execution and the result.

## September 21 continuation: one run per arm

The user narrowed the campaign to the three remaining arms, with **no seed
replications or reference retraining**. The completed `reference` checkpoint
(`c962cc71`) is retained. At **09:58 UTC**, the non-private balanced arm
[`reference_aux`](https://jetbrains.wandb.io/federated-compute/opaque/runs/c385ff45)
had completed **4/256 updates** on the H100; `dp` and `dp_aux` were queued serially.
The training configuration and source files remain unchanged. Public initialization
zero is retained only to match the existing reference, not to launch repetitions.

- Training unit: `opaque-moe-remaining-20260921.service`, started at 09:52 UTC,
  bounded to 20 hours.
- Scoring unit: `opaque-moe-scoring-20260921.service`, waits for all three training
  arms to finish before running HumanEval+ on the four saved checkpoints;
  bounded to 30 hours and the original absolute campaign deadline.
- The manifest now contains only initialization zero; original manifest and
  failure state are preserved as `*.before-20260921.json`. MBPP+ and additional
  initialization runs are not scheduled by this continuation.
- Both units run independently of the laptop connection. Progress is in
  `confirmation/logs/s0-reference_aux-train.log`, then the corresponding `dp`
  and `dp_aux` logs. Controller receipts are `remaining-20260921.log`,
  `scoring-20260921.log`, and `campaign_state.json`.

The original interruption was an evaluator resource limit, not training:
EvalPlus's full HumanEval ground-truth cache is **254,640,043 bytes**, exceeding
the previous 128 MiB file limit. The isolated evaluator now permits files up to
1 GiB with 2 GiB temporary storage; networking, privileges and the other sandbox
constraints are unchanged. Its 105 focused tests and a real CPU-only evaluation
of all 164 cached base-model samples passed. The failed attempt remains intact
under `confirmation/base/`; recovery evidence is under
`evaluation/runtime-fix-20260921/`. No model generation or training was repeated
for this repair, and the patched evaluator is separate from frozen training
source.

Single-run comparisons are descriptive, not replicated evidence. The report's
existing replication requirements for a supported balancing-benefit conclusion
are not relaxed. Learning and balancing conclusions remain pending.

## Execution receipt

- Submitted **2026-09-19 23:20:11 UTC** on `opaque-moe`, H100 80GB.
- Managed unit: `opaque-moe-balancing-v2-20260919.service`.
- Absolute cutoff: **2026-09-22 20:10 UTC**, within the approved 72-hour budget.
- W&B: <https://jetbrains.wandb.io/federated-compute/opaque>, group
  `mellum2-balancing-v2`. Diagnostic runs use a separate `-diagnostics` group.
- Remote root:
  `/home/david.stanojevic/opaque-moe-experiment-20260918/campaigns/balancing-20260919`.
- Frozen configuration SHA-256:
  `b741d751e874910ef8f46e9b46a390a4d13aacdb8ec17c1261da750fcd37e473`.
- Opaque core revision: `bde08c91247cc2fe127e776bc72e402d8e7fef83`.
- EvalPlus image:
  `sha256:68b93cd727b300feb7b6cb7b583e49e78c85e5eba2878266879e4462ab02cd17`.

The controller starts reference SFT immediately. It evaluates the unchanged
checkpoint before scoring the first trained checkpoint, then completes the
remaining primary runs serially. Full campaign results are **pending**, not
implied by successful execution gates. Sources, configuration, dataset and
benchmark checksums are recorded in `submission.json`, `confirmation/campaign.json`
and `evaluation/final-data-audit-v2.json` on the workspace.

At **23:25 UTC**, [seed-0 reference](https://jetbrains.wandb.io/federated-compute/opaque/runs/c962cc71)
had reached **3/256 optimizer updates**, and the service was active/running.
Initial development token NLL was 0.346048, mean per-layer CV 1.280030 and
pooled-index CV 0.247781. These are initialization measurements, not improvements
or directly comparable losses to the older CodeAlpaca dataset.
The final handoff check confirmed **24/256 updates**, still active/running.

## Fixed experiment

See the [protocol](../../examples/moe_privacy/BALANCING.md) and
[configuration](../../examples/moe_privacy/configs/sft_balancing.json).

All arms use pretrained `JetBrains/Mellum2-12B-A2.5B-Base`, revision
`271755e48ab6b2ed0ef224eaaabe2d25275fb8ee`, with 64 experts/top-8 routing
in 28 sparse layers. Trainable scope is **56,426,496 parameters**: expert LoRA
and FP32 routers. Original expert weights, attention and the other backbone
weights stay frozen; this is not full-model fine-tuning or pretraining.

| Arm | Privacy noise | Balancing |
| --- | --- | --- |
| `reference` | None | Off |
| `reference_aux` | None | Lagged exact pooled loads |
| `dp` | Gaussian | Off |
| `dp_aux` | Gaussian | Lagged private pooled loads |

Originally three public initialization seeds, `0`, `1`, `2`; the September 21
continuation retains only `0`. Each arm still has 256 updates.
Private randomness is fresh and not exported. Each private run targets
epsilon 8, delta 1e-5, with joint gradient/load accounting where applicable.
This is a per-run, sequence-level statement, not a campaign-wide budget.

**The non-private arms retain the same per-record clipping.** They are matched
controls, not tuned ordinary/unclipped SFT. Both balanced arms use the same
lagged surrogate, not a conventional current-batch Switch loss.

Public Python Magicoder OSS-Instruct data is pinned at
`5f839b1f368a76b161028bb9edff055db34022b2`. Splits are 8,192 training,
256 development, 1,024 final test and 64 diagnostic records; maximum 512 tokens,
completion-only loss, no packing or truncated answers. Five conservative
benchmark-overlap exclusions were deterministically refilled. The final audit
resolved all 8,192 records, found no further exact/13-word matches under its
specified check, and verified unchanged reserved partitions. This does not
establish semantic independence or absence from pretraining.

## Why the old result was insufficient

The PR optimizes expert-index loads **pooled across layers**. Mean per-layer
imbalance is a different quantity. Pooled uniformity can coexist with collapsed
individual layers, so both are now reported explicitly.

The old public pilot was reanalysed post hoc using its complete 28-by-64 expert
share arrays. Its private balanced arm reduced pooled CV by only **0.1505%**
relative to private unbalanced, versus **0.5200%** for mean per-layer CV. Thus
the metric distinction does **not** reveal a previously hidden strong benefit;
both differences remain unconvincing single-seed observations.

The new frozen-public gradient diagnostic measured auxiliary/task router-gradient
norm ratio **0.118 at coefficient 0.2**. The confirmation coefficient was set to
**2.0 before test scoring**, targeting approximately equal initial router-gradient
scales. This is an unclipped router-only measurement, not proof of its
post-clipping contribution or future utility. Load-noise ratio is 1.0 and EMA
beta is 0.95; extra gradient noise is included in the privacy budget.

## Measurements

- Full HumanEval+ (164 tasks): greedy `pass@1`, original and extended tests,
  Wilson intervals, syntax, truncations and timeouts.
- MBPP+ (378 tasks): secondary stage after all primary runs, budget permitting;
  377 tasks have extended tests, one has base tests only.
- Final-test token-weighted and example-mean completion NLL; teacher-forced token
  accuracy, which is not generated-program correctness.
- Pooled-index and per-layer CV, worst-layer imbalance, max/mean load, effective
  experts, underused experts and batch-local skew.
- Runtime, generation throughput, GPU memory and actual privacy accounting.
- Per-seed results, mean/SD and exploratory paired intervals. The reporter requires
  at least three complete pairs and the predefined margins before declaring a
  practical balancing benefit; missing runs remain visible.

The generation protocol is `entry_point_top_level_v1`: identical canonical
prompts, greedy decoding and a 512-token ceiling, stopping before unrelated
top-level continuations after the requested function. It never uses test outcomes
to select an answer. Raw and graded text are retained; no sanitizer repairs code.
Programs run only in a network-disabled, resource-bounded container, not on the host.

## Gates and evidence

- Final focused local suite: **261 passed, 1 CUDA-only test skipped** on the Mac;
  real H100 execution was separately validated below. Lint and whitespace checks passed.

- [Public diagnostic](https://jetbrains.wandb.io/federated-compute/opaque/runs/43c6c9a1):
  35.6 effective experts per layer out of 64; predicted mature EMA noise/signal
  0.273 under the starting diagnostic assumptions. No optimizer updates.
- [Private execution gate](https://jetbrains.wandb.io/federated-compute/opaque/runs/8b161f55):
  two real updates at coefficient 2, epsilon 7.999391, delta 1e-5;
  about 50.7 GiB peak allocation. Final-test data were reserved and not evaluated.
- Reference harness: known correct code passed; deliberately wrong code failed;
  container cleanup verified.
- Restored private-checkpoint generation: the original unbounded completions
  continued into unrelated code and failed syntax checks. Those failures remain
  preserved. The fixed boundary produced identical answer prefixes: one valid
  solution passes, one genuine nested-parentheses algorithm error still fails.
  These two-task gates are not benchmark performance estimates.

## Monitoring and outputs

In the workspace terminal:

```bash
cd /home/david.stanojevic/opaque-moe-experiment-20260918/campaigns/balancing-20260919
sudo systemctl status opaque-moe-balancing-v2-20260919.service --no-pager
tail -f confirmation/logs/s0-reference-train.log
```

- `confirmation/campaign_state.json`: current stage/failure/deadline outcome.
- `confirmation/report.md` and `report.json`: report regenerated after each stage.
- `confirmation/main/seed-N/ARM/`: metrics, public per-record evaluation,
  trainable weights and benchmark results.
- `confirmation/base/`: unchanged-checkpoint code baseline.

In W&B, use `trainer/step` for training curves. Relevant keys include
`eval/nll_token_mean`, `eval/teacher_forced_token_accuracy`,
`eval/global_load_cv_mean`, `eval/pooled_load_cv`, final `test/*`, and
`code/humaneval/plus_pass1`. Code scores appear after the full benchmark, not
from the smoke checks. The controller stops on its first execution failure or
budget exhaustion; it does not tune against test results or hide failed seeds.