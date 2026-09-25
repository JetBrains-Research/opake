# Mellum2 auxiliary coefficients at matched privacy

## Question and execution boundary

Does the private auxiliary objective improve **task performance over auxiliary-free
DP-SGD at the same total privacy budget**? This round tests coefficient and
gradient/load-noise allocation, not maximal expert balance or a new mechanism.

The four standalone configurations below are fixed for this round. This document
records their protocol. Cluster execution steps, image build, and gates live in
[`moe-zenml-reproduction.md`](moe-zenml-reproduction.md). Allow roughly 12–14
GPU-hours for the three private trainings at previously observed Coder speed;
native training, Iceland overhead and scoring are additional. Configuration
edits alone do not launch jobs.

## Published coefficient: primary-source evidence

Verified against the versioned sources on September 23, 2026:

- [Mellum 2 technical report, v1, §3.4.5](https://arxiv.org/html/2605.31268v1#S3.SS4.SSS5)
  specifies **global auxiliary load-balancing coefficient `0.001`** and a
  **separate router z-loss coefficient `0.001`**, with FP32 routers. The z-loss
  is a distinct stability objective, not another contribution to the balancing
  coefficient. This round adds **no z-loss**.
- The [exact checkpoint configuration](https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Base/blob/271755e48ab6b2ed0ef224eaaabe2d25275fb8ee/config.json)
  independently confirms `router_aux_loss_coef: 0.001` at revision
  `271755e48ab6b2ed0ef224eaaabe2d25275fb8ee`. It does not independently establish
  the report's z-loss setting.
- [Report §3.6, “Cluster migration and load-balancing loss shift”](https://arxiv.org/html/2605.31268v1#S3.SS6)
  explains Megatron's running average of expert counts across accumulation
  microbatches, reset at gradient finalization. The loss uses that running
  estimate, not a true global count recomputed for every microbatch.

The earlier `0.2` and `2.0` coefficients are numerically **200× and 2,000×** the
published balancing coefficient. Their effective strengths are not directly
comparable across objectives and normalizations. `0.001` is a justified anchor,
**not a reproduction of Mellum pretraining**. These experiments fine-tune an
already pretrained checkpoint, with the backbone frozen except for expert LoRA
adapters and trainable routers.

## Fixed configurations and controls

Here `α` is `router_aux_loss_coef`; `ρ` is `load_noise_ratio`. Paths in this table
are under `examples/moe_privacy/configs/`.

| Configuration | Backend / arm | α | Effective ρ | Purpose |
| --- | --- | ---: | ---: | --- |
| Existing `sft_balancing_aux02.json` | Opaque / `dp` | 0 | None | Auxiliary-free private baseline; no load release |
| Existing `sft_balancing_aux02.json` | Opaque / `dp_aux` | 0.2 | 1 | Moderate-coefficient private anchor |
| [New `sft_balancing_aux0001_ratio1.json`](../../examples/moe_privacy/configs/sft_balancing_aux0001_ratio1.json) | Opaque / `dp_aux` | 0.001 | 1 | Change only coefficient versus the moderate anchor |
| [New `sft_balancing_aux0001_ratio01.json`](../../examples/moe_privacy/configs/sft_balancing_aux0001_ratio01.json) | Opaque / `dp_aux` | 0.001 | 0.1 | Published coefficient with less gradient noise |
| [New `sft_balancing_aux02_ratio01.json`](../../examples/moe_privacy/configs/sft_balancing_aux02_ratio01.json) | Opaque / `dp_aux` | 0.2 | 0.1 | Isolate allocation at the moderate coefficient |
| Existing `sft_balancing.json` | Opaque / `dp_aux` | 2.0 | 1 | Retain as context, not a new run |
| Existing `sft_trl_baselines.json` | Native TRL / `trl_reference` | 0 | N/A | Conventional non-private auxiliary-off control |
| [New `sft_trl_aux0001.json`](../../examples/moe_privacy/configs/sft_trl_aux0001.json) | Native TRL / `trl_reference_aux` | 0.001 | N/A | Published-coefficient conventional control |
| Existing `sft_trl_baselines.json` | Native TRL / `trl_reference_aux` | 0.2 | N/A | Moderate-coefficient conventional control |

The three private files derive from `sft_balancing_aux02.json`, changing only the
name and the listed coefficient/ratio. The native file derives from
`sft_trl_baselines.json`, changing only name and coefficient. Keep the templates,
historical reports, completed checkpoints and failed artifacts unchanged. Verify
completion receipts, configuration, data and adapter metadata before reusing
controls; a queued receipt is not proof of completion. Do not duplicate completed
training or interrupt running jobs. There is **one run per configuration**, with
no initialization repetitions, model switch, broad sweep or pretraining campaign.

Auxiliary-off is selected with `--arm dp`, not by setting a configuration's
positive coefficient or ratio to zero. `sft_run.training_arguments` overrides the
coefficient to zero and supplies no auxiliary kwargs for this arm. Native files
retain privacy/ratio/filter fields only to share the configuration schema;
`trl_run` does not use them for privacy calibration or load releases. Report native
privacy as non-private and its effective ratio as N/A, not `1` or epsilon 8.

### Matched task, trainability and optimization

- Model: `JetBrains/Mellum2-12B-A2.5B-Base`, revision
  `271755e48ab6b2ed0ef224eaaabe2d25275fb8ee`.
- Data: `ise-uiuc/Magicoder-OSS-Instruct-75K`, revision
  `5f839b1f368a76b161028bb9edff055db34022b2`; Python only, complete answers,
  unchanged five benchmark-overlap exclusions and partition seeds
  `20260919` / `20260920`.
- Completion-only task loss, prompt tokens included in the router mask, maximum
  sequence length 512. Partitions: 8,192 train, 256 validation, **already-used
  1,024-record test partition**, 64 public diagnostic records.
- Same expert `gate_up_proj` / `down_proj` LoRA tensors, rank 4 / LoRA alpha 8,
  and actual FP32 routers; the remaining backbone is frozen. LoRA alpha is not
  the auxiliary coefficient `α`. Use public model/adapter initialization
  `--seed 0`; do not copy or publish private sampling/noise seeds.
- Constant-learning-rate SGD at `0.05`, no weight decay, expected/effective batch
  32, **256 optimizer updates**, public evaluation every 64 updates.
- Private runs retain Poisson sampling (`q = 32 / 8192 = 1 / 256`), per-record
  clipping norm `0.1`, microbatch 1, no gradient checkpointing and no performance
  kernels. Target **epsilon 8, delta 10⁻⁵ per run**, including the load release.

The saved `.temp/balancing-20260919/final-data-audit-v2.json` supplies these
reference hashes (relative to the experiment worktree):

| Partition | SHA-256 |
| --- | --- |
| Train | `c1211fe46f56249b47e9a5a5394e3ca8b8bd5c3eda4eb2fcd7028f03e96ff1de` |
| Validation | `a609ad356206992fbdfa1857ffb99eae030b3e63a304d2e8a228439087287046` |
| Test | `a685873689e5a2d85a09e769b5ba52a5378c29ad0fe446a19f60937b823b9b0d` |
| Diagnostic | `d6cf608cfa2ef61321dd754ffa6d6e8260c19cf29e1651145789494803d1df5c` |

Identical configuration inputs do not replace checking each execution's actual
partition hashes and trainability receipt. The audit found no matches under its
stated lexical checks; it does not establish absence of semantic duplicates or
pretraining contamination.

## Normalization and native-control limits

Opaque retains the existing [lagged load-balancing mechanism](../mechanisms/dp-sgd/moe-load-balancing.md).
For record `x`, its auxiliary contribution before joint task-plus-auxiliary
gradient clipping is

$$
\alpha E\,\frac{T_x}{\bar T}\sum_e (\tilde f_e-k/E)P_e(x),
\qquad \bar T=T_{\max}=512.
$$

`T_x` counts attended tokens, including the prompt; `P_e(x)` is the record's
mean router probability. The layer-pooled estimate `f_tilde` is derived from prior
noisy load releases, with fixed `filter_beta = 0.95` and
`mean_tokens = max_tokens = 512`. Summed per-record gradients are divided by
expected batch size 32, not the realized token total. If both objectives use the
same frozen current-batch loads, this normalization scales the unclipped Switch
gradient by `T_batch / (32 * 512)`; using a lagged noisy estimate is an additional
difference. The first private step has a uniform load estimate and zero auxiliary
gradient.

The [native TRL control](../../examples/moe_privacy/TRL.md) instead uses the
current **physical batch of 4**, with 8 accumulation microsteps and non-reentrant
gradient checkpointing. It uses shuffled fixed-size batches, the mean of
physical-batch completion-token-mean losses, and clipping norm **1.0 on the
accumulated batch gradient**. It has no private load release, Gaussian noise,
per-record clipping or Opaque auxiliary patches. Its physical-batch loss does
not reproduce Megatron's pretraining accumulation semantics. It is a standard-SFT
quality reference, **not a control changing only privacy noise**.

### Saved public signal: coefficient scaling only

`sft_metrics.probe_balancing_signal` already measures the frozen public surrogate
and current-batch Switch reference separately. The saved
`.temp/balancing-20260919/diagnostic/public-signal-v1.log` records
`router_aux_to_task_ratio = 0.11791976608075179` at coefficient `0.2`
([diagnostic run](https://jetbrains.wandb.io/federated-compute/opaque/runs/43c6c9a1)).
At the **same frozen weights, records, loads and normalization**, unscaled
auxiliary gradients are unchanged, so scaling the coefficient to `0.001` gives
`0.11791976608075179 * 0.001 / 0.2 ≈ 0.000589599` (about 0.059% of the task
router-gradient norm). This is arithmetic reuse of saved evidence, not a new
measurement or a prediction of the noisy training trajectory.

These are **router-gradient diagnostics, not whole-model clipping measurements**.
The log's estimated full-gradient buffer exceeds the probe's 256 MiB threshold;
router-only measurements cannot establish whether the combined trainable tree
clips at `0.1`. The probe applies neither training clipping nor optimizer updates,
and its illustrative unit clip scales use `C = 1`. It does not measure realized
private filter quality or establish a useful coefficient by itself.

## Matched privacy and the noise-allocation trade-off

Keep the existing joint accountant and calibrate every private run through
`sft_run.training_arguments` and `run_experiment`. If `σ₀` is the auxiliary-free
multiplier at the same `q`, 256 steps and target budget, the current accountant
uses

$$
\sigma_{\mathrm{effective}}=\frac{\sigma_{\mathrm{gradient}}}{\sqrt{1+\rho}},
\qquad \sigma_{\mathrm{gradient}}=\sigma_0\sqrt{1+\rho}.
$$

| Load allocation | Gradient-noise factor versus auxiliary-free | Extra gradient noise |
| --- | ---: | ---: |
| `ρ = 1` | `sqrt(2) ≈ 1.4142` | 41.4% |
| `ρ = 0.1` | `sqrt(1.1) ≈ 1.0488` | 4.9% |

At fixed geometry, token bounds and expected batch, raw load-release standard
deviation is `σ_gradient * Δ_load / (sqrt(ρ) * 32)`. Thus the `ρ = 0.1` release
has `sqrt((1.1 / 0.1) / (2 / 1)) = sqrt(5.5) ≈ 2.35` times the standard deviation
of `ρ = 1` at matched privacy. The same factor applies to the fixed linear
pre-clamp filter-noise forecast; it is not a measured error ratio for the
nonlinear filtered estimate during training.

**Changing coefficient alone does not lower privacy cost.** A positive auxiliary
coefficient still enables the load release. Reusing the auxiliary-free multiplier
with that release would exceed the target budget. Record the resolved multiplier,
actual epsilon, delta, sampling rate and completed steps, rather than reporting
the target as an observed result.

The protected unit is one preprocessed prompt/answer pair under add/remove
adjacency. This is public-data experimentation: on genuinely private data,
multiple released runs and tuning require composition. Individual epsilon-8
results do **not** give the whole campaign an epsilon-8 guarantee.

## Execution and evaluation protocol

- Training and answer generation use the Iceland ZenML GPU profiles described in
  the [deployment runbook](../../deploy/zenml/moe_iceland/README.md), with
  `models-rd`, separate CPU-only orchestration, bounded scratch and an explicitly
  reviewed registry/secret contract. Deployment reuse is pinned to
  `0e42b111d621256cc84d82f2b8000a939315db91` (feature commit
  `604318995c350cb6215771c8da60afcb9e45b8f1`); it does not replace the trainers,
  dependencies, expert adapters or privacy mechanism with the branch's SFT job.
- The previous September 22, 20:10 UTC authorization does not cover this migration.
  A new aggregate GPU-hour cap and deadline are required before any GPU probe,
  training or answer generation. Actual GPU, storage and pull availability must
  be demonstrated; static manifests and the earlier CPU smoke are not evidence.
- Generated programs execute only in the existing isolated Docker evaluator on
  the Mac, never on the host, in Coder, or inside a training pod. Record hardware
  migration and evaluator platform differences as comparison limitations; where
  parity fails, re-score retained answers under one image/limit contract rather
  than regenerate answers or retrain models.
- Mac Docker verification passed four synthetic isolation fixtures after the
  user-approved inert-tunnel preflight adaptation. The current local evaluator is
  `sha256:12657bbffce037d5e73ae30e852b0901c58cba6602ac0f709fbd8d859a6c9929`,
  an amd64 image on an arm64 Docker engine. This verifies the sandbox path, not
  Mellum performance or timing parity. Re-score all compared retained answers
  under one common image/limit contract and preserve the historical scores.
- After authorization, use `sft_run.py` for each new `dp_aux` file and
  `trl_run.py` for `sft_trl_aux0001.json` with `trl_reference_aux`. Use fresh
  output directories under a new comparison root, preserving the existing
  train-then-score workflow and adapter exports; no new sweep controller.
- Use distinct W&B names containing coefficient and ratio (native: `ratio-na`),
  retaining the existing `federated-compute/opaque` project. The new config names
  encode both; when supplying a shared `--wandb-group`, also supply an explicit
  `--wandb-name`, since the tracker's default name otherwise uses the group and
  arm rather than the config name. Keep historical tracking records unchanged.
- Freeze these candidates **before revealing MBPP+ scores**. Verify whether that
  benchmark has previously informed selection before describing it as
  unconsulted; the overlap audit alone does not answer that question. Do not tune
  against its outcomes in this round.
- Score full **164-task HumanEval+** with the identical existing decoding and
  protocol hashes; label it exploratory because earlier results informed this
  follow-up. Score all **378 MBPP+ tasks** for the fixed candidates and reused
  controls. Use `campaign.evaluate_checkpoint`, `code_eval.py` and isolated
  Docker scoring without weakening sandbox constraints. HumanEval and HumanEval+
  are not independent benchmarks.
- Retain held-out **answer-token NLL** and **teacher-forced token accuracy**, clearly
  identifying the already-used 1,024-record partition. Report training/evaluation
  GPU-hours, peak memory, syntax failures, timeouts and truncations. Expert-load
  metrics are secondary diagnostics.
- Require identical task IDs and evaluation protocols for per-task wins, losses
  and ties and paired task-bootstrap intervals. These intervals describe
  benchmark-task uncertainty **conditional on these checkpoints**, not
  training-seed reproducibility. Do not silently drop failed or missing tasks;
  missing results remain pending, not zero or completed.
- Prefer executable-correctness improvements across benchmarks that also reduce
  the existing prediction-loss penalty. Otherwise report a trade-off or no
  demonstrated benefit. Include unsuccessful configurations. Do not use
  `campaign_report.py`'s balance-based benefit criterion, and do not infer that
  improved utilization caused a net task-quality gain at matched privacy.

Focused configuration tests compare the complete JSON objects to their templates,
exercise private/off argument wiring and calibrate all three private variants
with the real accountant. Native comparison checks retain model/data/trainable
scope and initialization invariants. Existing accounting tests cover joint
inflation at both allocations. These CPU checks validate the configuration and
accounting contract, not execution receipts or pretrained task performance.