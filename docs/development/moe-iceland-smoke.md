# Iceland smoke execution record

## Scope and target

One native CPU `dp_aux` smoke, two optimizer updates with correctness checks,
with a 15-minute Kubernetes Job deadline. No pilot or retry was authorized after
this outcome. All records are synthetic/public; a smoke is not learning evidence
or a mathematical proof of differential privacy.

- Project: `models-rd`, `357b8eb4-6c65-40d1-a4de-ea48a3279288`.
- Stack: `jbr-cmk-dev-jbr-eu-iceland1-fed-comp`,
  `28320977-dcbd-423d-b52e-a59888d8a24f`.
- Namespace: `federated-compute`.
- Step request/limit: `200m CPU / 2Gi`; orchestrator: `200m CPU / 512Mi`.
- Published image:
  `europe-docker.pkg.dev/grazie-development/zenml-generated/opaque-moe-iceland-cpu@sha256:06dd19a7b8ecc5a7d7bf2f8a9125dd3ae5b29851ffbdcd6a3fef7ca4a4221aef`.

## Native cluster outcome

[Run `7086669c-615b-4ce3-8c78-b6f93e9e171c`](https://cloud.zenml.io/workspaces/jcp-prod/projects/357b8eb4-6c65-40d1-a4de-ea48a3279288/runs/7086669c-615b-4ce3-8c78-b6f93e9e171c),
named `moe-iceland-smoke-dp_aux-s0-b7d125a51716`, was submitted once.
The orchestrator scheduled, but its image pull failed with `ErrImagePull`:
the anonymous token request to `https://europe-docker.pkg.dev/v2/token`
returned `403 Forbidden`. The pod had `imagePullSecrets: []`.

No container started, no training updates or correctness checks executed on
Iceland, and no GCS result bundle was produced. Only the owned Job
`moe-iceland-eb5872bf-moe-iceland` was deleted; Job and pod absence were verified.
The backend still reported `provisioning` at `2026-09-17 23:09:49 UTC`, despite
that cleanup. The local monitor was stopped after exceeding its wait limit;
this was not a training timeout.

Sanitized logs, pod/events evidence, and cleanup receipts are retained under
`.temp/iceland-smoke-results/7086669c-615b-4ce3-8c78-b6f93e9e171c/` in the
experiment worktree. Registry ownership must confirm an authorized pull route
or appropriate image-pull credentials for this actual repository before a new
submission. The existing Crusoe pull secret is not proven to authorize direct
pulls from `europe-docker.pkg.dev`; do not guess a replacement repository or
modify the shared stack implicitly.

## Separate local evidence

- All 54 deployment tests passed with the normal repository integration fixtures.
- The Linux image's real Mellum correctness checks passed at `0.2 CPU / 2Gi`,
  but that checks-plus-training invocation timed out at 295 seconds.
- A separate training-only invocation at `1 CPU / 2Gi` completed two updates:
  epsilon `7.999689022418134`, delta `1e-5`, noise multiplier
  `0.5194172668457031`, training time `9.400s`, runner peak RSS `919.98 MiB`.
- Its public loss changed from `4.855314` to `4.877843`. It did not meet the
  learning target and did not rerun the already completed correctness checks.
- Local evidence is under `.temp/image-smoke-v2/`. It is not a native cluster
  pass, a combined checks-and-training pass, or an uploaded ZenML result bundle.


## Capacity blocker — 2026-09-24

Settings match the cluster docs:
- toleration `nvidia.com/gpu` / `Exists` / `NoSchedule`
- scheduler `gpu-binpack-scheduler` (Kyverno default for GPU workloads)

Published CUDA image (gcc + cu126):
`europe-docker.pkg.dev/grazie-development/zenml-generated/opaque-moe-cuda@sha256:843e3337a7b7141f3a5af894fa7f649462280fcf451aa17ecec18aae9d64ab09`

Latest probe run `e62ccc86-1e64-4bbb-bc0e-83e4ca6296bf` failed placement:
`0/19 nodes Insufficient nvidia.com/gpu` (also cpu/memory/ephemeral on some nodes).
Autoscaler: did not free a usable GPU (`NotTriggerScaleUp`).

Coder `opaque-moe` temporary start also failed:
zone `europe-west4-b` had no `a3-highgpu-1g` / `nvidia-h100-80gb` capacity.
Workspace was stopped again; persistent disk remains.

No automatic retries. Resume when a free Iceland GPU or Coder H100 is available.
