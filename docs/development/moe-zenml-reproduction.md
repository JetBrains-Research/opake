# Reproduce MoE private-auxiliary experiments on Iceland ZenML

This document is the single entry point for rebuilding and re-running the
coefficient / noise-allocation campaign from branch
`david-stan/moe-iceland-experiment`.

It stacks:

1. **PR #1080 private MoE load-balancing mechanism** (Opaque lagged auxiliary
   objective + joint accounting).
2. **Experiment trainers and configs** under `examples/moe_privacy/`.
3. **Iceland ZenML GPU deployment** under `deploy/zenml/moe_iceland/`, selectively
   adapted from Evgeny’s
   [`junie/zenml-sft-pipeline` @ `0e42b111`](https://github.com/JetBrains-Research/opake/tree/0e42b111d621256cc84d82f2b8000a939315db91)
   (feature commit `604318995c350cb6215771c8da60afcb9e45b8f1`): external ZenML URL
   on **both** pods, `gpu-binpack-scheduler`, narrow `nvidia.com/gpu` toleration,
   separate CPU orchestrator vs GPU step resources. It does **not** adopt that
   branch’s general SFT trainer, attention-only LoRA defaults, or 200 Gi scratch.

Related protocol notes:

- Coefficient agenda: [`moe-aux-coefficients.md`](moe-aux-coefficients.md)
- Iceland launcher details: [`../../deploy/zenml/moe_iceland/README.md`](../../deploy/zenml/moe_iceland/README.md)
- Local SFT / TRL runners: [`../../examples/moe_privacy/`](../../examples/moe_privacy/)

## What the experiment answers

At matched total privacy (`ε≈8`, `δ=10⁻⁵` **per private run**):

> Does Opaque’s privately accounted auxiliary objective improve **task quality**
> enough to offset its extra noise, versus ordinary DP-SGD with auxiliary off?

Judge by HumanEval+/MBPP+ pass@1, held-out NLL, token accuracy, and cost—not by
expert-balance diagnostics alone.

## Fixed scientific contract

| Item | Value |
| --- | --- |
| Model | `JetBrains/Mellum2-12B-A2.5B-Base` @ `271755e48ab6b2ed0ef224eaaabe2d25275fb8ee` |
| Data | Magicoder OSS Python @ `5f839b1f368a76b161028bb9edff055db34022b2` |
| Trainables | Expert LoRA (`gate_up_proj`/`down_proj`, r=4, α=8) + FP32 routers |
| Updates | 256 per run, seed 0, no seed repetitions |
| Private budget | `ε=8`, `δ=10⁻⁵` per run when load release is on |
| Stack | `jbr-cmk-dev-jbr-eu-iceland1-fed-comp` (`28320977-dcbd-423d-b52e-a59888d8a24f`) |
| Project | `models-rd` (`357b8eb4-6c65-40d1-a4de-ea48a3279288`) |
| Namespace | `federated-compute` |
| Registry | `europe-docker.pkg.dev/grazie-development/zenml-generated` |
| Laptop ZenML | `https://zenml.labs.jb.gg` |
| Pod ZenML | `https://zenml-external.labs.jb.gg` (both orchestrator and step) |
| Scheduler | `gpu-binpack-scheduler` |
| GPU taint | `nvidia.com/gpu` / `Exists` / `NoSchedule` (all GPU nodes) |
| W&B | `https://jetbrains.wandb.io/federated-compute/opaque` |

### Configurations (α = auxiliary coef, ρ = load_noise_ratio)

| Role | Config | Backend / arm | α | ρ |
| --- | --- | --- | ---: | ---: |
| Private baseline (reuse if valid) | `sft_balancing_aux02.json` | opaque / `dp` | 0 | none |
| Private aux 0.2/1 (reuse if valid) | `sft_balancing_aux02.json` | opaque / `dp_aux` | 0.2 | 1 |
| **New private A** | `sft_balancing_aux0001_ratio1.json` | opaque / `dp_aux` | 0.001 | 1 |
| **New private B** | `sft_balancing_aux0001_ratio01.json` | opaque / `dp_aux` | 0.001 | 0.1 |
| **New private C** | `sft_balancing_aux02_ratio01.json` | opaque / `dp_aux` | 0.2 | 0.1 |
| Native TRL off (reuse if valid) | `sft_trl_baselines.json` | trl / `trl_reference` | 0 | n/a |
| Native TRL 0.2 (reuse if valid) | `sft_trl_baselines.json` | trl / `trl_reference_aux` | 0.2 | n/a |
| **New native 0.001** | `sft_trl_aux0001.json` | trl / `trl_reference_aux` | 0.001 | n/a |

Joint private accounting: `σ_effective = σ_gradient / √(1+ρ)`.
At matched privacy, `ρ=1` adds ~41% gradient noise; `ρ=0.1` adds ~4.9%.

## Branch layout (committed)

```text
examples/moe_privacy/          # trainers, configs, generation/scoring handoff
deploy/zenml/moe_iceland/      # Iceland GPU/CPU launcher (no venvs/secrets)
tests/integration/experiments/ # deployment + experiment tests
docs/development/moe-*.md      # protocol and this runbook
packages/...                   # PR #1080 mechanism (already on branch base)
```

**Not committed** (local only): `.client-venv/`, `.credentials/`, `source-stages/`,
`submissions/`, GPU authorization ledgers, model caches, recovered weights.

## One-time local setup

```bash
# Work from the experiment branch checkout
git checkout david-stan/moe-iceland-experiment
cd /path/to/opaque   # or the worktree that tracks this branch

# Experiment Python (trainers / unit tests)
uv sync --frozen --no-default-groups --package opaque-transformers --python 3.12

# Isolated ZenML 0.96.4 client (do not mix into the training venv)
python3.12 -m venv deploy/zenml/moe_iceland/.client-venv
deploy/zenml/moe_iceland/.client-venv/bin/pip install -r \
  deploy/zenml/moe_iceland/requirements-client.txt
# Plus jb-mlops[zenml]==0.0.45 from the private index used by the team
```

Authenticate:

```bash
# Laptop client: internal URL only
deploy/zenml/moe_iceland/.client-venv/bin/zenml login \
  https://zenml.labs.jb.gg --api-key --refresh

# Optional: gcloud for registry connector / artifact download
gcloud auth login
```

Cached Pro login can be passed through the launcher with `--cached-login`
(memory-only stdin transfer; no credentials in images or archives).

## Build the CUDA image

```bash
export PYTHONPATH=deploy/zenml/moe_iceland/.client-venv/lib/python3.12/site-packages:deploy/zenml:.
CLIENT=deploy/zenml/moe_iceland/.client-venv/bin/python

# 1) Stage a bounded build context (no secrets)
$CLIENT -m deploy.zenml.moe_iceland.launch \
  --profile gpu-probe-v1 --prepare iceland-cuda-v1

# 2) Resolve / verify the CUDA wheel lock (cu126 for Iceland driver 12.8)
$CLIENT -m deploy.zenml.moe_iceland.resolve_gpu_lock --check

# 3) Build linux/amd64 with BuildKit secrets for the private index only
CONTEXT=deploy/zenml/moe_iceland/source-stages/iceland-cuda-v1
TAG=europe-docker.pkg.dev/grazie-development/zenml-generated/opaque-moe-cuda:YYYYMMDD-v1
docker buildx build --platform linux/amd64 --load \
  --secret id=uv_config,src=.../uv.toml \
  --secret id=space_tools_username,src=... \
  --secret id=space_tools_password,src=... \
  -f "$CONTEXT/Dockerfile.cuda" -t "$TAG" "$CONTEXT"

# 4) Push via the stack’s GCP service connector (not plain gcloud user IAM)
#    Connector id on this stack: d83ed4ee-f66a-4345-8339-fa52e8b014a5
#    Record the resulting digest: .../opaque-moe-cuda@sha256:...
```

Image requirements that already bit us once:

- Torch **CUDA 12.6** wheels (host driver was 12.8; CUDA 13 wheels failed).
- **gcc/g++** in the final image (Torch 2.14 still Triton-JITs some eager ops).
- Digest-pinned parent image; launcher never builds on the cluster.

Record:

- `IMAGE=...@sha256:...`
- `SOURCE_SHA` from `source-stages/.../source/moe-source.json`

## Serial GPU execution (required order)

Authorization must be live: `--authorization-id`, `--gpu-seconds`,
`--deadline-utc`, `--timeout-seconds`. No CPU/Coder fallback, no automatic
retries, no seed sweeps.

Common flags:

```bash
COMMON=(
  --project-name models-rd
  --project-id 357b8eb4-6c65-40d1-a4de-ea48a3279288
  --confirm-project models-rd
  --confirm-stack-id 28320977-dcbd-423d-b52e-a59888d8a24f
  --image "$IMAGE"
  --source-sha256 "$SOURCE_SHA"
  --service-account zenml-orchestrator
  --step-service-account zenml-step-runner
  --service-account-verification admission
  --image-pull-secret crusoe-regcred-ro
  --image-pull-secret space-regcred-ro
  --wandb-secret-name jbr-fed
  --wandb-secret-key WANDB_API_KEY
  --reviewed-target-sha256 "$TARGET_FP"   # from cluster.target_fingerprint
  --authorization-id "$AUTH_ID"
  --deadline-utc "$DEADLINE"
  --gpu-seconds "$GPU_SECONDS"
  --acknowledge-confirmed-resources
  --cached-login
  --submit
)
```

### 1. Infrastructure probe

```bash
$CLIENT -m deploy.zenml.moe_iceland.launch \
  --profile gpu-probe-v1 --backend opaque --arm dp_aux --seed 0 \
  --timeout-seconds 1800 "${COMMON[@]}"
# → GATE_PROBE=<run_id>
```

Success means real CUDA matmul, W&B connectivity, and a sealed GCS bundle—not
merely a scheduled pod.

### 2. Pretrained two-update smokes

```bash
# Opaque private auxiliary smoke
CONFIG=examples/moe_privacy/configs/sft_balancing_aux0001_ratio1.json
$CLIENT -m deploy.zenml.moe_iceland.launch \
  --profile sft-smoke-v1 --backend opaque --arm dp_aux --seed 0 \
  --config-name sft_balancing_aux0001_ratio1.json \
  --config-sha256 "$(sha256sum "$CONFIG" | awk '{print $1}')" \
  --gate-run-id "$GATE_PROBE" --timeout-seconds 3600 "${COMMON[@]}"
# → GATE_PRIVATE=<run_id>

# Native TRL auxiliary smoke
CONFIG=examples/moe_privacy/configs/sft_trl_aux0001.json
$CLIENT -m deploy.zenml.moe_iceland.launch \
  --profile sft-smoke-v1 --backend trl --arm trl_reference_aux --seed 0 \
  --config-name sft_trl_aux0001.json \
  --config-sha256 "$(sha256sum "$CONFIG" | awk '{print $1}')" \
  --gate-run-id "$GATE_PROBE" --timeout-seconds 3600 "${COMMON[@]}"
# → GATE_NATIVE=<run_id>
```

### 3. Full 256-update trainings (serial)

```bash
# Native 0.001
$CLIENT -m deploy.zenml.moe_iceland.launch \
  --profile sft-train-v1 --backend trl --arm trl_reference_aux --seed 0 \
  --config-name sft_trl_aux0001.json --config-sha256 ... \
  --gate-run-id "$GATE_NATIVE" --timeout-seconds 28800 "${COMMON[@]}"

# Private (0.001, 1), (0.001, 0.1), (0.2, 0.1)
for cfg in \
  sft_balancing_aux0001_ratio1.json \
  sft_balancing_aux0001_ratio01.json \
  sft_balancing_aux02_ratio01.json
do
  $CLIENT -m deploy.zenml.moe_iceland.launch \
    --profile sft-train-v1 --backend opaque --arm dp_aux --seed 0 \
    --config-name "$cfg" --config-sha256 ... \
    --gate-run-id "$GATE_PRIVATE" --timeout-seconds 28800 "${COMMON[@]}"
done
```

Do **not** retrain completed historical controls (`dp` aux-off, `dp_aux` 0.2/1,
native off/0.2) if valid exports already exist—recover them instead.

### 4. Generation on Iceland, scoring on Mac

```bash
# GPU generation only (one benchmark per job)
$CLIENT -m deploy.zenml.moe_iceland.launch \
  --profile generate-v1 --backend opaque --arm dp_aux --seed 0 \
  --config-name ... --config-sha256 ... \
  --gate-run-id "$TRAIN_RUN" --benchmark humaneval \
  --timeout-seconds 14400 "${COMMON[@]}"

# Mac Docker scoring (network-none evaluator image)
python -m examples.moe_privacy.campaign score \
  --samples-dir path/to/answers \
  --output-dir path/to/score-attempt \
  --eval-image "$PINNED_EVAL_IMAGE" \
  --expected-manifest-sha256 "$MANIFEST_SHA" \
  --wandb-mode offline
```

Never execute generated code on the host. Training completion is independent of
scoring failures.

## Local verification before cluster spend

```bash
# Focused deployment + experiment suites (CPU)
.venv/bin/python -m pytest tests/integration/experiments/test_moe_iceland_gpu.py \
  tests/integration/experiments/test_moe_iceland_auth.py \
  tests/integration/experiments/test_moe_iceland_sft_bundle.py \
  tests/integration/experiments/test_moe_handoff.py -q

# Accounting / native readiness (as available in the worktree)
.venv/bin/python -m pytest packages/opaque-dpsgd/tests/accounting/test_moe_aux.py \
  tests/integration/experiments/test_moe_trl.py -q
```

## Comparison report

After trainings and scores exist:

```bash
python -m examples.moe_privacy.trl_campaign write_comparison \
  --output-dir .temp/coefficient-comparison \
  --named-dirs ...   # explicit result directories for each arm
```

Report HumanEval+/MBPP+, held-out NLL/accuracy, actual ε, noise multipliers,
runtime/memory, and hardware. Keep expert-load metrics diagnostic only.

## Operational pitfalls (Iceland)

1. **Pods cannot use `zenml.labs.jb.gg`.** Both orchestrator and step pods need
   `https://zenml-external.labs.jb.gg`. Laptop stays on the internal URL.
2. **All GPU nodes are tainted**; only the narrow `nvidia.com/gpu` toleration is
   required. Kyverno defaults GPU workloads onto `gpu-binpack-scheduler`.
3. **Registry push** must use the stack service connector; plain user gcloud IAM
   may lack `uploadArtifacts`.
4. **Do not archive `.venv`** or credentials into ZenML code uploads.
5. **Capacity ≠ configuration.** Correct scheduler/toleration does not create free
   GPUs; failed placement is a bounded blocker, not a reason to add blanket
   tolerations or fall back to CPU.
6. **Scratch sizes** may differ across probe/smoke/train; CPU/memory/GPU requests
   stay matched. Ephemeral stay ≤ ~100 Gi (node ceiling 128 Gi).

## Historical W&B anchors (do not duplicate blindly)

| Run | Meaning |
| --- | --- |
| [82540020](https://jetbrains.wandb.io/federated-compute/opaque/runs/82540020) | Private aux **off** (standard DP-SGD baseline) |
| [2e891b34](https://jetbrains.wandb.io/federated-compute/opaque/runs/2e891b34) | Private aux **0.2 / ρ=1** (60/164 HumanEval+) |
| [b97e2db1](https://jetbrains.wandb.io/federated-compute/opaque/runs/b97e2db1) | Private aux **2.0 / ρ=1** (63/164) |
| [28a80a84](https://jetbrains.wandb.io/federated-compute/opaque/runs/28a80a84) | Native TRL aux off (metrics present; overall failed label) |
| [cca2c97a](https://jetbrains.wandb.io/federated-compute/opaque/runs/cca2c97a) | Native TRL aux 0.2 finished |

Verify exports before reuse. Missing local receipts do not justify retraining.

## Authorization template

Replace placeholders; do not invent capacity:

```text
Authorization ID: <uuid>
Aggregate GPU-seconds: <e.g. 259200 for 72h>
Hard stop (UTC): <ISO-8601>
Allowed work: probe + smokes + missing trains + missing generation only
Forbidden: seed sweeps, CPU/Coder fallback, auto-retry, stack mutation
```
