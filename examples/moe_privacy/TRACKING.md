# W&B tracking for MoE experiments

Both the synthetic runner and pretrained SFT runner support
[native W&B logging](README.md#native-wb-tracking). All three arms and arbitrary
public model seeds use the same interface. The default destination is
`https://jetbrains.wandb.io/federated-compute/opaque`.

Install the optional SDK in the environment that performs logging:

```bash
uv pip install --python .venv/bin/python wandb==0.30.0
```

For online logging, use `--wandb-mode online` with a valid server-specific login
or `WANDB_API_KEY`. `WANDB_MODE=disabled` disables the default automatic mode;
an explicit command-line mode takes precedence. `--wandb-mode offline` stores
W&B run data locally for later `wandb sync`. It does not require credentials.

## Mirror an already running campaign

The external mirror reads existing files only: it never imports PyTorch,
changes training code, or restarts training. It can run in a separate virtual
environment with just W&B installed. Run from the repository root:

```bash
python -m examples.moe_privacy.wandb_mirror \
  --campaign-dir results/pilot-20260918-s0 \
  --config examples/moe_privacy/configs/sft_pilot.json \
  --seed 0 --follow \
  --wandb-mode online --wandb-group pilot-20260918-s0
```

This expects the serial `reference`, `dp`, `dp_aux` layout produced by the
experiment campaign. It backfills `metrics.jsonl` and the structured progress
events in `campaign.log`, then polls every 15 seconds. Each arm gets its own
run when its output directory appears. On completion, `summary.json` supplies
the actual privacy result, public routing diagnostics, resolved configuration,
dependency/source provenance and trainable-parameter counts.

Use `--training-unit NAME.service` on a systemd workspace to stop the mirror
if training exits without producing all summaries. Otherwise the default
mirror deadline is 24 hours (`--max-seconds`). A persistent supervisor is needed
to keep a live mirror running after disconnecting from SSH.

The mirror saves `wandb_run.json` and `wandb_mirror_state.json` beside each
arm's results. Restart it with the same destination, paths and arguments to
resume the same run IDs and skip already processed records. One mirror per
campaign is enforced with a file lock. Do not mirror a run that is already
logging natively. On an unclean interruption, retain the SDK's local W&B files
as well as the cursors; these remain available for recovery with `wandb sync`.

## Backfill a completed smoke or individual run

Omit `--follow` for a one-shot upload:

```bash
python -m examples.moe_privacy.wandb_mirror \
  --run-dir results/cuda-smoke --arm dp_aux --seed 0 \
  --experiment synthetic_moe \
  --config examples/moe_privacy/configs/smoke.json \
  --wandb-mode online --wandb-group moe-correctness-gates
```

For pretrained SFT use `--experiment pretrained_moe_sft` (the default) and its
original SFT config. An incomplete one-shot upload is labelled `incomplete`,
not successful training; use `--follow` for active jobs.

## What is and is not uploaded

- `eval/loss` and public routing metrics use `trainer/step` as their chart axis.
  Backfilled records are uploaded now, not assigned fabricated wall-clock times.
- Progress and timing update during training. Actual `privacy/epsilon`,
  `privacy/delta` and the calibrated multiplier appear in the completed summary;
  a configured target budget is not a measured privacy result.
- Existing files remain the source of truth. No prompts, completions, arbitrary
  console logs, raw training/auxiliary losses, private RNG seeds, model weights
  or checkpoints are uploaded. Automatic code/Git/machine capture is disabled.
- These runners use public data. Operational timing and diagnostics are not an
  additional DP mechanism and must not be assumed safe on sensitive data. Each
  private arm's budget is separate; W&B grouping does not compose privacy.