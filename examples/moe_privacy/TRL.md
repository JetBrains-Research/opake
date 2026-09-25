# Native TRL non-private controls

These controls use **`trl.SFTTrainer`**, not Opaque's `DPTrainer` with zero noise.
All start from the pinned pretrained `Mellum2-12B-A2.5B-Base` checkpoint and
train the same expert LoRA tensors and routers as the private experiments.
They do not retrain the frozen backbone or claim differential privacy.

| Arm | Configuration | Coefficient |
| --- | --- | ---: |
| `trl_reference` | `sft_trl_baselines.json` | 0 (overridden) |
| `trl_reference_aux` | `sft_trl_baselines.json` | 0.2 |
| `trl_reference_aux` | `sft_trl_aux0001.json` | 0.001 |

The balanced arm backpropagates the native Transformers current-minibatch
load-balancing term through TRL. Merely setting a configuration field or logging
expert loads is insufficient. The focused tests load both configurations into
tiny CPU models and exercise real TRL optimizer updates, coefficient-scaled
native auxiliary gradients, accumulation and adapter exports without Opaque
patches. These tests are not completion evidence for the full experiment.

## What is matched

- Pretrained model and dataset revisions; all four partition hashes, including
  the benchmark-overlap exclusions and the already-used 1,024-record test set.
- Magicoder Python problem/solution SFT, completion-only labels, maximum 512
  tokens, no packing, and the same 8,192 training records.
- Public initialization zero, expert LoRA rank 4 / alpha 8, trainable routers,
  constant-learning-rate SGD at 0.05, no weight decay, 256 optimizer updates,
  effective batch size 32. One run per configuration; no repetitions.
- Identical held-out evaluation and full 164-task HumanEval+ generation/scoring
  protocol, including the isolated executable-code sandbox.

## Deliberate differences from the private method

These are conventional SFT references, not another attempt to isolate only noise:

- TRL uses fixed-size shuffled batches without Poisson sampling.
- Physical batch size is 4, with 8 gradient-accumulation microsteps. Native
  balancing observes each **physical minibatch of 4**, not a synthetic global
  batch of 32. Both TRL arms use the same accumulation settings.
- Native task loss is token-weighted within each physical minibatch, then those
  minibatch means are averaged during accumulation. It is not a global token
  mean over 32 records. This scales the complete native task-plus-auxiliary loss
  together, rather than multiplying the auxiliary contribution by accumulation.
  Opaque's private objective averages per-record losses.
- The norm bound is **1.0 on the accumulated batch gradient**, not Opaque's
  per-record bound of 0.1. No per-record clipping, Gaussian noise, lagged load
  estimate, private load release or privacy accountant is used.
- Non-reentrant gradient checkpointing bounds memory. Static LoRA remains the
  same adapter parameterization; per-expert-module caching avoids repeated
  materialization, and a precision wrapper executes the original router in FP32.
  Neither substitutes Opaque's auxiliary objective or expert implementation.

The configuration shares the existing experiment schema, so it contains privacy
calibration fields that are **unused** in these non-private runs. W&B and the
result summary explicitly report no privacy guarantee and no privacy spending.

The native off/on pair measures the effect of conventional balancing. The
private off/on pair remains the test of the PR at equal total privacy. Comparing
TRL and Opaque also changes sampling, loss weighting and clipping; it cannot
attribute the whole quality difference to privacy noise alone. An equal numeric
coefficient does not make native and lagged auxiliary objectives identical.

## Execution and measurements

Install the pinned `trl==1.13.0` in an isolated dependency directory when sharing
a workspace with an active experiment; do not upgrade its training environment.

```bash
PYTHONUNBUFFERED=1 python -m examples.moe_privacy.trl_run \
  --config examples/moe_privacy/configs/sft_trl_aux0001.json \
  --arm trl_reference_aux --seed 0 \
  --output-dir results/trl-aux-coef0.001-ratio-na \
  --wandb-mode online --wandb-group mellum2-native-trl-coefficients \
  --wandb-name magicoder-native-trl-aux-coef0.001-ratio-na-s0
```

Use `sft_trl_baselines.json` for missing off/`0.2` controls, with distinct
`coef0-ratio-na`/`coef0.2-ratio-na` output directories and W&B names. A shared
W&B group does not supply coefficient-specific names automatically.

Before launching, obtain a newly confirmed runtime deadline and inspect real
completion receipts and active jobs. Reuse matching completed controls; do not
repeat their training. Check `summary.json` with `trl_campaign.validate_result`
against the control's original configuration and the verified reference, and
verify its `trainable/adapter_spec.json` and `trainable/trainable.safetensors`
exports. A queued receipt or W&B name is not completion evidence. Preserve active
jobs and failed outputs. The single-run runner refuses existing output
directories; it does not enforce a runtime deadline itself, so bound execution
externally using the existing `campaign.run_stage` deadline.

Report **HumanEval+ pass@1**, held-out answer NLL, teacher-forced token accuracy,
training time and peak GPU memory. Evaluation NLL excludes the auxiliary term;
expert-load metrics are diagnostics, not the success criterion.

`trl_campaign` waits for the preceding private follow-up and its scoring to
finish, runs a two-update GPU engineering check without evaluating the final
test set, then trains both controls before scoring them. It checks partition,
trainable-scope and scoring-protocol identity and writes `comparison.json` and
`comparison.md`. It never stops the preceding service, retries a failed run,
changes Coder's timeout, or runs beyond the recorded authorization deadline.
That legacy launcher schedules both arms using the old campaign manifest; it is
not a completion-aware coefficient-follow-up launcher. Do not restart it to add
the `0.001` anchor: run only the missing configuration through `trl_run` after
authorization, retaining the train-then-score workflow.

A single run per configuration on an already-used benchmark is an exploratory
comparison, not a statistically established advantage. Missing scores remain
pending.