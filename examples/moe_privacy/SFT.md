# Pretrained MoE SFT

`sft_run.py` fine-tunes `JetBrains/Mellum2-12B-A2.5B-Base` on public
`HuggingFaceH4/CodeAlpaca_20K` instruction/completion pairs. Both revisions are
pinned in the configuration. This is adaptation of a pretrained model, not
randomly initialized synthetic training.

## Trainable state

The 12.15B-parameter backbone is frozen in BF16. Rank-4, alpha-8 LoRA updates
adapt both stacked expert projections in every sparse layer; all 28 router
weights remain trainable in FP32. Together these are **56,426,496 trainable
parameters**. The expert adapters use permanent PyTorch parametrizations:
registration happens before training, and forward evaluation is pure tensor
arithmetic. Attention weights are not adapted.

The runner uses Opaque's `DPTrainer` with completion-masked labels, fixed joint
clipping, Poisson sampling, Gaussian noise and SGD. It does not implement an
alternative DP training loop. Performance kernels are disabled, attention is
eager, and router computations use FP32.

## Data and privacy

- One preprocessed prompt/answer pair is one add/remove privacy record. No packing.
- Only answer tokens contribute to task loss. Prompts and padding have label `-100`.
- Router statistics count every attended token, including the prompt. Public
  `max_tokens=mean_tokens=sequence_length` bounds the load release.
- Train/test prompt duplicates are excluded before selection; processed-record
  fingerprints are saved. Long prompts without answer space are rejected and
  answers are right-truncated to the public length bound.
- The private arms target the same **per-run** `epsilon=8`, `delta=1e-5`.
  `dp_aux` accounts jointly for gradient noise and the lagged private load release.
- The guarantee does not cover pretraining, private preprocessing, a whole user,
  or the combined release of several runs. This public-data experiment includes
  a non-private control and operational telemetry; it is not a privacy-safe
  template for publishing diagnostics from sensitive data.

Held-out loss is token-weighted answer cross-entropy on the public test subset.
It is not code execution accuracy, nor proof that the records were absent from
pretraining. A two-update smoke is not evidence of sustained learning.

## Execution

Use the pinned workspace environment and a CUDA GPU. Run the tiny correctness
gate first, then `configs/sft_smoke.json` with `--arm dp_aux` to validate the
pretrained BF16 path before the pilot:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python -m examples.moe_privacy.run \
  --config examples/moe_privacy/configs/smoke.json --arm dp_aux \
  --seed 0 --device cuda --output-dir results/cuda-checks --checks

PYTHONUNBUFFERED=1 .venv/bin/python -m examples.moe_privacy.sft_run \
  --config examples/moe_privacy/configs/sft_smoke.json --arm dp_aux \
  --seed 0 --output-dir results/pretrained-check
```

For the pilot, run `reference`, `dp`, then `dp_aux` serially using
`configs/sft_pilot.json` and distinct output directories. Each arm starts from
the same pinned checkpoint and public adapter seed; secret sampling/noise seeds
are separate and never exported. The pilot uses 4,096 training and 1,024 held-out
records, length 256, expected batch 32, microbatch one, 256 updates, SGD learning
rate 0.05, clipping norm 0.1, and private balancing coefficient 0.05 with ratio
0.02. These are exploratory, untuned choices.

`summary.json`, `metrics.jsonl`, `data.json`, and `trainability.json` preserve
metrics, configuration, provenance and trainable scope. `trainable/` contains
only `trainable.safetensors` and `adapter_spec.json`, not frozen base weights,
optimizer state or random state. Reconstruct the same pinned base, call
`configure_expert_lora` with the saved rank/alpha, then `load_trainable` to restore
both adapters and routers. This export is not a PEFT adapter checkpoint.

Output directories must be fresh. The synthetic `compare.py` expects synthetic
configurations; compare SFT summaries directly, checking identical configurations,
base revisions, public seeds, dataset hashes and equal private budgets.

## Native W&B tracking

The SFT runner uses the same [optional tracking settings](README.md#native-wb-tracking)
as the synthetic runner. Install the SDK only if wanted in the prepared environment:

```bash
uv pip install --python .venv/bin/python wandb==0.30.0
```

Add `--wandb-mode online`, `--wandb-mode offline`, or `--wandb-mode disabled` to
the commands above. Use a common `--wandb-group mellum2-sft-pilot` for the three
matched arms. The default `auto` mode respects `WANDB_MODE`, then selects online
with `WANDB_API_KEY` or offline without it; missing SDK support warns and disables
`auto`, but explicit online/offline requests fail. Defaults target
`https://jetbrains.wandb.io`, entity `federated-compute`, project `opaque`.

Per-step JSON stdout and all local exports are retained regardless of tracking
mode. Enabled tracking adds `wandb_run.json`; it sends public held-out loss and
routing metrics (including initialization), step progress, public configuration
and the completed outcome. The target budget is distinct from actual accounted
epsilon in the final privacy summary; the reference arm has no DP guarantee.

Raw training/auxiliary losses, trainer arguments and private RNG seeds are never
forwarded, and no models or artifacts are uploaded. Operational telemetry remains
outside the accountant's guarantee: enabling tracking does not broaden the DP
claim above. Python `run_experiment` calls remain inert unless passed explicit
`TrackingOptions` through `tracking=`.