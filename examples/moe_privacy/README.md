# Private MoE feasibility experiment

For completion-only fine-tuning of a pretrained Mellum MoE on public code
instructions, see [Pretrained MoE SFT](SFT.md). The tiny model below is also
available as a correctness gate for that experiment.

A fully trainable, randomly initialized `MellumForCausalLM`: two layers, hidden
size 64, eight experts, top-2 routing, 313,152 parameters. FP32, eager
attention, no performance kernels, pretrained downloads, adapters, or credentials.
The runner uses Opaque's `DPTrainer`; it does not implement a second training loop.
It defaults to CPU; `--device cuda` requires an available CUDA device without
silently falling back to CPU.

## Experiment

All records are **synthetic and public**. Four domain tags select different
stochastic token-transition grammars. Training and validation use independent
public generation seeds; the same datasets and model initialization are used
across the three arms for each model seed.

| Arm | Gradient clipping | Gaussian noise | Router balancing |
| --- | --- | --- | --- |
| `reference` | Fixed global norm | None; **not private** | Disabled |
| `dp` | Fixed global norm | Calibrated to the target budget | Disabled; no load release |
| `dp_aux` | Task + lagged surrogate gradient, before clipping | Joint gradient/load calibration | Private load release and lagged balancing |

`configs/pilot.json` specifies 4,096 training and 1,024 validation sequences of
32 tokens, expected Poisson batch size 32 (`q=1/128`), microbatches of one,
256 updates, SGD at constant learning rate 0.25, clipping norm 1, and a per-run
target `(epsilon=8, delta=1e-5)`. The private balancing coefficient is 0.05,
load-noise ratio 0.02, and filter beta 0.99. These are fixed pilot choices,
not optimized hyperparameters or predicted results.

The privacy unit is **one sequence under add/remove adjacency**, not a token or
a user. There is no record packing. Both releases use public expected-batch
normalization; `max_tokens=mean_tokens=sequence_length`. The `dp_aux` accountant
wraps the Gaussian with `moe_aux` *before* Poisson amplification and composition.
Its effective multiplier is `sigma / sqrt(1 + ratio)`, so matched privacy requires
about 1% more gradient noise than `dp`, not identical gradient noise.

The model's ordinary, batch-coupled auxiliary loss is disabled. The PR's lagged
private load state supplies the balancing gradient; it starts uniformly, so the
first balancing gradient is zero. All router and expert parameters remain
trainable and enter clipping/noising, including experts receiving no tokens.

## Local preparation and smoke gate

Use the experiment worktree, **not the original checkout's older environment**:

```bash
cd .temp/moe-iceland
uv sync --frozen --no-default-groups --package opaque-transformers --python 3.12
PYTHONUNBUFFERED=1 uv run --no-sync python -m examples.moe_privacy.run \
  --config examples/moe_privacy/configs/smoke.json \
  --arm dp_aux --seed 0 --output-dir .temp/results/smoke-dp-aux --checks
```

The lock pins `torch==2.14.0` and `transformers==5.17.0`. Mellum and the PR APIs
are mandatory: missing capabilities cause errors, not a fallback to a dense/toy
model or a skipped experiment. `--seed` controls public model initialization;
fresh, separate, **unpublished** seeds drive Poisson sampling and privacy noise.
Repeated invocations are not bit-for-bit identical private runs. The gate uses
deterministic public test keys, never the training keys.

`--checks` runs the actual Mellum model through:

- `vmap(grad)` versus an explicit single-record gradient loop;
- first-record gradient invariance under changing another batched record;
- masking/padding invariance and complete-vector clipping versus an explicit sum;
- ragged full-batch versus microbatch equivalence with the same lagged state;
- nonzero router task/balancing gradients and expert task gradients;
- gradient/load noise scale checks and noise on every forced-unused expert;
- empty-batch gradient noise, load release, and state advancement.

An output directory must be new. A failed run keeps any diagnostics already
written but has no completed `summary.json`; do not treat it as a successful gate.
The two-update smoke checks execution and instrumentation, **not learning**.

## Pilot and replication

Only after the runtime gate succeeds, execute one arm at a time. For local runs:

```bash
for arm in reference dp dp_aux; do
  PYTHONUNBUFFERED=1 uv run --no-sync python -m examples.moe_privacy.run \
    --config examples/moe_privacy/configs/pilot.json \
    --arm "$arm" --seed 0 --output-dir ".temp/results/pilot-${arm}-s0" || break
done
```

Inspect the one-seed pilot before repeating with public model seeds 1 and 2.
Keep the configuration unchanged across arms; do not silently tune on validation
and call the result preregistered. No campaign is launched automatically.

Each successful run produces:

- `summary.json`: complete configuration, versions/revision, dataset hashes,
  per-run accounting, initial/final held-out CE and routing metrics, time and RSS;
- `metrics.jsonl`: held-out CE at initialization, configured intervals, and the
  final update; routing metrics at initialization and the final update;
- `checks.json`, when requested: real-model gate results;
- `model/`: a **weights-only** Hugging Face export, not a resumable trainer state.

Private seeds, sampler/optimizer state, raw training losses, and raw auxiliary
losses are not saved. The trainer's final raw-loss printer is disabled. Runtime
and resource diagnostics are experimental telemetry, not additional DP releases
covered by the accountant. Do not replace this dataset with sensitive data and
assume the entire experiment, its diagnostics, or its non-private control is DP.

Compare complete matched sets, including all three arms for every seed:

```bash
uv run --no-sync python -m examples.moe_privacy.compare \
  .temp/results/pilot-reference-s0 .temp/results/pilot-dp-s0 \
  .temp/results/pilot-dp_aux-s0 --output .temp/results/pilot-comparison.json
```

The comparison rejects mismatched configurations, data, initialization, privacy
budgets, incomplete horizons, duplicate runs, and missing arms. It reports per-arm
means/sample standard deviations and paired CE/load differences. With one seed,
standard deviation is `null`, not an invented error bar. On public/synthetic data
the controls are safe to publish; on private data, multiple released private runs
compose, and the non-private reference invalidates any all-arms privacy claim.

The learning target is at least **10% held-out CE reduction from initialization**.
Report failure as well as success. A successful `dp` arm establishes implementation
and utility evidence for private MoE training regardless of whether `dp_aux`
improves load CV, maximum expert share, or normalized routing entropy. Assess
balancing alongside its CE cost; lower imbalance alone is not a learning result.
These tests and curves are **not a mathematical proof of differential privacy**.

## Native W&B tracking

For existing runs, see [backfilling and live mirroring](TRACKING.md).

Both prepared runners support optional W&B tracking. To add the pinned SDK to
an already prepared environment:

```bash
uv pip install --python .venv/bin/python wandb==0.30.0
```

The command-line default is `--wandb-mode auto`: it respects `WANDB_MODE`,
otherwise choosing online when `WANDB_API_KEY` is set and offline otherwise.
Without the SDK, `auto` warns and disables tracking; explicit online/offline
requests fail with installation guidance. The default destination is
`https://jetbrains.wandb.io`, entity `federated-compute`, project `opaque`.

- `--wandb-mode online` sends public telemetry to W&B.
- `--wandb-mode offline` records W&B data locally without connecting.
- `--wandb-mode disabled` disables W&B entirely.

Use the same `--wandb-group` for matched arms/seeds, keeping distinct fresh
output directories:

```bash
uv run --no-sync python -m examples.moe_privacy.run \
  --config examples/moe_privacy/configs/smoke.json --arm dp_aux --seed 0 \
  --output-dir .temp/results/tracked-smoke-dp-aux \
  --wandb-mode offline --wandb-group moe-smoke
```

Existing JSON and weights-only exports are always kept locally. Enabled tracking
also writes `wandb_run.json` in the output directory. It records public evaluation
metrics (including initial routing), optimizer-step progress, public experiment
configuration and the final outcome. The configured `target_epsilon` is not the
actual accounted epsilon: the latter comes from the completed run's `privacy`
summary, and the `reference` arm remains non-private.

No raw training/auxiliary losses, trainer arguments, private RNG seeds, model
uploads or artifact uploads are sent to W&B. Timing/progress are operational
telemetry; tracking does **not** expand the per-run DP claim or make sensitive-data
diagnostics safe to publish. Python calls keep tracking inert by default:
`run_experiment(..., tracking=None)`; pass `TrackingOptions` explicitly to enable it.

## Iceland execution

Use the dedicated [Iceland launcher](../../deploy/zenml/moe_iceland/README.md),
not the legacy GPU-oriented launcher. Start with a dry run, an explicitly
confirmed project and the shared Iceland stack, a rebuilt pinned image, and one
smoke job. Both orchestrator and step pods need the external ZenML URL; the
laptop client must keep the internal URL. Do not request GPUs, volumes, or the
absent Kubernetes `ai-for-code` secret.

The known schedulable `200m CPU / 512Mi` profile is an **infrastructure gate**, not
a claim that PyTorch fits. A local macOS smoke with checks peaked around 1.1 GiB
RSS; Linux and ZenML overhead must be measured separately. If the default gate
OOMs or cannot schedule, stop and obtain a confirmed memory/CPU allocation from
Andrei. Larger jobs must not sit pending indefinitely. The launcher must upload
the result bundle to the artifact store before short-lived pods disappear.

## Tests

```bash
uv sync --frozen --group dev --package opaque-transformers --python 3.12
uv run --no-sync pytest tests/integration/experiments/test_moe_privacy.py \
  tests/integration/experiments/test_moe_privacy_checks.py \
  tests/integration/experiments/test_moe_privacy_compare.py \
  tests/integration/experiments/test_moe_tracking_integration.py
```

The real-model smoke/gate tests have the `slow` marker and are explicitly included
in this command. No cluster or W&B access is needed.