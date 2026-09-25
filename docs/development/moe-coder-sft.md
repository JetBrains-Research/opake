# Coder pretrained MoE execution receipt

## Runtime

- Workspace: `https://coder.dev-dws-jbr-europe-west4-gke.intellij.net/@davidstanojevic/opaque-moe`.
- VM: `coder-davidstanojevic-opaque-moe-56cbe5b7`, project `gke-dev-dws-jbr`,
  zone `europe-west4-b`; authenticated SSH through GCP IAP.
- Remote directory: `/home/david.stanojevic/opaque-moe-experiment-20260918`.
- GPU: NVIDIA H100 80GB HBM3; driver `580.178.04`.
- Python `3.12.10`, Torch `2.14.0+cu130`, Transformers `5.17.0`, Rust `1.89.0`.
- Opaque source: PR #1080 revision `bde08c91247cc2fe127e776bc72e402d8e7fef83`.
- Local worktree: `.temp/moe-iceland`, branch `david-stan/moe-iceland-experiment`.
  No commits or pushes were made for this execution.

## Completed gates

The tiny CUDA gate completed two private updates and all correctness checks,
including per-record gradient parity, independence, microbatching, empty batches
and noise on unused experts. Epsilon was `7.999689022418134` at delta `1e-5`.
Local evidence: `.temp/coder-cuda-gate/` inside the experiment worktree.

The pretrained `dp_aux` SFT gate also completed two updates:

| Measurement | Observed value |
| --- | --- |
| Checkpoint | `JetBrains/Mellum2-12B-A2.5B-Base` |
| Model revision | `271755e48ab6b2ed0ef224eaaabe2d25275fb8ee` |
| Dataset revision | `798c567f69c8f4b12fc191015e59ee34e9afe00d` |
| Records / length | 64 train, 16 held out, maximum 128 tokens |
| Trainable scope | 56,426,496 expert-adapter/router parameters; base frozen |
| Epsilon / delta | `7.999263844883154` / `1e-5` |
| Gradient noise multiplier | `0.4113283920288086` |
| Held-out CE | `1.066605281084776` → `1.006005808711052` |
| Held-out load CV | `1.2248935577402456` → `1.2036274684305857` |
| Peak allocated GPU memory | 48,935,901,184 bytes (45.6 GiB) |
| Training-phase wall time | 31.073 seconds, including scheduled evaluation |

This short gate establishes execution, not sustained utility, a balancing
advantage or a mathematical privacy proof. Results and weight export remain in
`results/pretrained-smoke-dp_aux/` on the VM. Metadata, logs and the deployed
pilot source were copied to local `.temp/coder-pretrained-evidence/`. Later local
lint-only edits were not copied over the running campaign's frozen source.

The ordinary PEFT parameter wrapper failed under functional differentiation.
The example's static LoRA parametrization fixed that incompatibility without
changing Opaque's DP mechanism. Both failing reproduction tests then passed;
the focused adapter suite passed 20 tests and the data/config suite passed 18.

## Launched campaign

At `2026-09-18 10:45:26 UTC`, the serial `reference` → `dp` → `dp_aux`
campaign was launched under `opaque-moe-sft-20260918.service`. Each arm uses
`configs/sft_pilot.json`: 256 updates, 4,096 train / 1,024 held-out records,
length 256, expected batch 32, microbatch one. Each private arm independently
targets epsilon 8 at delta `1e-5`; this is not a combined-campaign privacy budget.

The service runs as `david.stanojevic`, stops on an arm failure, never restarts
automatically, and has a 24-hour campaign deadline and 128 GiB host-memory cap.
It survives an SSH disconnect or the end of the agent session. The complete
three-arm comparison is **still pending**; only completed per-arm
`summary.json` files count as completed runs.

At `2026-09-18 11:18:22 UTC`, the reference arm had completed **32/256 updates**,
with initial held-out CE `0.6445080766`. The service was active/running without
a traceback; GPU use was 53,105 MiB with 41% instantaneous utilization. No arm
had a completed summary at that observation.

With explicit user approval, Coder's current workspace shutdown deadline was
extended from `2026-09-18 12:47:18 UTC` to **`2026-09-19 10:45:00 UTC`**. The API
returned HTTP 200 and a follow-up read confirmed the exact deadline, unchanged
build ID and `running` status. The default three-hour timeout (`ttl_ms=10800000`)
was not changed; no workspace restart occurred.

From the workspace terminal:

```bash
cd /home/david.stanojevic/opaque-moe-experiment-20260918
systemctl status opaque-moe-sft-20260918.service --no-pager
tail -f results/pilot-20260918-s0/campaign.log
```

Outputs are under `results/pilot-20260918-s0/{reference,dp,dp_aux}/`.
To cancel only this campaign:

```bash
sudo systemctl stop opaque-moe-sft-20260918.service
```

No Iceland submission, workspace restart or pre-existing job was involved in
this execution.

## W&B mirror

At `2026-09-18 14:34:32 UTC`, `opaque-moe-wandb-20260918.service` started
alongside the unchanged training service. It uses an isolated `.tracking-venv`
with `wandb==0.30.0` and logging-only source under `.tracking/source`; neither
the campaign's frozen training source nor its environment was replaced.

- Instance/project: `https://jetbrains.wandb.io/federated-compute/opaque`.
- Group: `pilot-20260918-s0`; one run per arm, created as that arm starts.
- Reference: [e200190e](https://jetbrains.wandb.io/federated-compute/opaque/runs/e200190e).
- Server-side verification showed 242 completed updates and loss history at
  step 0 (`0.6445080766`) and step 128 (`0.5308847872`). These are reference-arm
  observations, not a private-training result.
- The mirror polls every 15 seconds and automatically follows `dp` and `dp_aux`.
  Completed summaries add actual accounting, routing, execution and provenance
  metadata. No training examples, weights, private RNG state or raw training
  loss telemetry are uploaded.

The mirror has a 20-hour service deadline, 512 MiB host-memory limit, half-CPU
quota and bounded failure retries. It uses a project-scoped, mode-0600 login
file inside the mode-0700 `.tracking` directory. Credentials were not copied
into source, command-line arguments or the experiment's recorded configuration.

```bash
systemctl status opaque-moe-wandb-20260918.service --no-pager
tail -f results/pilot-20260918-s0/wandb-mirror.log
# Stop only logging; training continues:
sudo systemctl stop opaque-moe-wandb-20260918.service
```

Native W&B options are available in both local experiment runners for future
jobs. See `examples/moe_privacy/TRACKING.md` for native logging, offline mode,
single-run backfill and restartable mirroring. The running campaign continues
to use its original file-only runners plus this external mirror.

Both completed gates were backfilled and verified through the W&B API:

| Run | W&B | Final held-out loss | Accounted epsilon |
| --- | --- | --- | --- |
| Tiny CUDA correctness gate | [b4558da4](https://jetbrains.wandb.io/federated-compute/opaque/runs/b4558da4) | 4.9346558452 | 7.9996890224 |
| Pretrained SFT gate | [a0d5bcec](https://jetbrains.wandb.io/federated-compute/opaque/runs/a0d5bcec) | 1.0060058087 | 7.9992638449 |

Both server states were `finished`, with `status=completed` and delta `1e-5`.
No training was repeated for these uploads. After reloading only the mirror to
include resolved configuration defaults, the reference retained run ID
`e200190e`, advanced to step 248, and still had exactly the original two loss
evaluation records. The campaign's start time remained `10:45:26 UTC`.

Validation: 75 focused tests passed, two CUDA-only tests were skipped locally;
the nine shared-tracking tests were rerun successfully after the metadata
change. Lint and formatting checks passed for all seven changed Python files.