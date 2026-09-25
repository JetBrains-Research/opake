# Iceland MoE: isolated native ZenML deployment

This launcher uses ZenML's native Kubernetes orchestrator, not the legacy GPU
launcher. It submits **one checked arm/seed job**, waits for it, and uploads the
actual runner directory using ZenML 0.96.4's `PathMaterializer` (`data.tar.gz`).
It does not build/push images or persistently switch the operator's project/stack.
CPU profiles do not download Hugging Face data/models. A dry-run imports only the Python standard
library and does not authenticate or contact Docker, Kubernetes, GCS, or ZenML.

## Preconditions: not a declaration of access or capacity

The operator must explicitly confirm an accessible **non-learn, non-TRACE**
project name and UUID. No project is selected by default. Expected deployment:

| Setting | Required value |
| --- | --- |
| Workspace / laptop endpoint | `prod` / `https://zenml.labs.jb.gg` |
| Stack | `jbr-cmk-dev-jbr-eu-iceland1-fed-comp` |
| Stack UUID | `28320977-dcbd-423d-b52e-a59888d8a24f` |
| Kubernetes namespace | `federated-compute` |
| Artifact store | `gs://gke-dev-dws-jbr-zenml` |
| Registered container repository | `europe-docker.pkg.dev/grazie-development/zenml-generated` |
| Both pods' `ZENML_STORE_URL` | `https://zenml-external.labs.jb.gg` |
| Parent source revision | `bde08c91247cc2fe127e776bc72e402d8e7fef83` |

Real submission reads the actual server version, accessible project/stack IDs,
component flavors, namespace, artifact-store path, and stored pod configuration
before invoking the pipeline. A reported non-prod Pro workspace is rejected;
self-hosted servers without Pro workspace metadata are bound to the exact prod
endpoint above. There is no automatic fallback to another project, stack, or
endpoint. Existing orchestrator and step service-account names are mandatory, so
ZenML cannot fall back to creating its default account and `edit` RoleBinding.

For CPU profiles, stored pod settings, custom component/stack environments, and secret bindings
require owner review and are rejected rather than merged or changed. The stack
must not use `pass_zenml_token_as_secret`. Cluster RBAC, registry pulls, GCS
connector/Workload Identity access, namespace admission, and capacity still need
a **real smoke gate**. This code does not assert they are already available.

## GPU coefficient-experiment profiles

`gpu-probe-v1`, `sft-smoke-v1`, `sft-train-v1`, and `generate-v1` are separate
CUDA-only contracts; the `smoke`/`pilot` CPU behavior below is unchanged.
They require `models-rd` (`357b8eb4-6c65-40d1-a4de-ea48a3279288`) and the
same Iceland stack. Only the four missing coefficient follow-ups are allowed
for full training. Historical configurations can be used for generation, not
as permission to retrain completed controls.

Deployment behavior is selectively adapted from
[`junie/zenml-sft-pipeline` at `0e42b111`](https://github.com/JetBrains-Research/opake/tree/0e42b111d621256cc84d82f2b8000a939315db91),
whose deployment feature commit is `604318995c350cb6215771c8da60afcb9e45b8f1`.
No general SFT trainer, production defaults, checkpoint/resume subsystem, or
attention-only LoRA configuration is reused. The experiment still uses the PR
worktree and its original expert-adapter/router training contract.

- Step: **4 CPUs / 64Gi host memory / one GPU with at least 80 GB device memory**.
  Orchestration: **200m CPU / 512Mi**, no GPU and no runtime W&B credential.
- Step scheduling: `gpu-binpack-scheduler`, with only
  `nvidia.com/gpu`, `Exists`, `NoSchedule`. No broad taint tolerations.
- Scratch: one bounded `80Gi` emptyDir; step ephemeral limit `96Gi`, plus `4Gi`
  for orchestration, below the reported `128Gi` node ceiling. This is an envelope,
  not evidence of available storage or schedulable resources.
- Both pod types explicitly use the external ZenML URL and confirmed existing
  image-pull secrets. The laptop continues to use the internal URL. The namespace
  service accounts, registry secret types and W&B key presence are checked without
  printing their values. A successful GPU probe, not a secret reference alone,
  establishes registry pull/runtime access.
- Inherited stack/pod settings require a reviewed `cluster.target_fingerprint`;
  only a safe subset of the explicit pod contract is accepted. A matching digest
  does not authorize GPU-bearing orchestrators, extra mounts or broad tolerations.
  The shared stack is never modified.
- No retries, automatic private-state resume, CPU fallback or Coder fallback.
  An in-progress run blocks submission. Recorded full-training configurations,
  including failed attempts, are not silently resubmitted.

GPU submissions additionally require `--backend`, an allowlisted `--config-name`,
`--config-sha256`, `--source-sha256`, an immutable `--image`, and explicit resource
acknowledgment. Secret arguments are references only: `--image-pull-secret` (repeat
for each approved existing reference), `--wandb-secret-name`, `--wandb-secret-key`.
`--reviewed-target-sha256` binds the inspected inherited settings. All profiles
remain **dry-run by default**; only `--submit` contacts the runtime.

**GPU execution always needs an explicit authorization.** Supply
`--authorization-id`, `--gpu-seconds`, `--deadline-utc`, and a per-job
`--timeout-seconds`. Maximum single-job ceilings are 30 minutes for the GPU
probe (including scheduling headroom), one hour for pretrained smoke, eight
hours for training, and four hours for generation; ceilings are not
authorization. A persistent local ledger conservatively reserves the entire
allocation, including failed/lost submissions; it never refunds or expands an
existing authorization automatically.

For the full end-to-end recipe (image build, gates, trainings, Mac scoring),
see [`docs/development/moe-zenml-reproduction.md`](../../../docs/development/moe-zenml-reproduction.md).

New Kubernetes Jobs initially have a 15-minute deadline. The observing client
extends only this run's running Jobs, within the approved allocation. Pending or
image-pull failures stop at 15 minutes; loss of the client before startup fails
closed. GPU submission source, receipts and bounded pod/event logs remain under
`submissions/<run-name>/`, rather than disappearing with a temporary directory.
Operators must still serialize submissions across laptops; this is not a
distributed campaign scheduler.

### CUDA build and execution contract

Prepare a bounded CUDA build context without submitting a job:

```bash
python -m deploy.zenml.moe_iceland.launch --prepare cuda-v1 --profile gpu-probe-v1
python -m deploy.zenml.moe_iceland.resolve_gpu_lock --check
```

`Dockerfile.cuda` uses digest-pinned Linux/amd64 Python/Rust bases, locked CUDA
wheels and the same PR workspace/native accounting build. The GPU lock retains
Torch `2.14.0`, Transformers `5.17.0`, PEFT `0.20.0` and native TRL `1.13.0`;
it does not install the other branch's released Opaque distribution. GPU image
receipts use an explicit CUDA contract, with wheel/native checksums; runtime
validation checks installed CUDA files as well as dependencies. Preparing the
context or validating the lock is not an image build, a driver-compatibility
check, or evidence of GPU availability.

The dispatcher checks exactly one CUDA device, at least 80 GB device memory,
BF16 support and a synchronized BF16 matrix multiplication. The infrastructure
probe also checks both trainer imports in **separate interpreters**, publishes
public W&B telemetry, and materializes a checksummed artifact. One successful
probe supports the two backend-specific pretrained gates; a full job requires
its own backend's real two-update gate with matching source/image/resources.

Training prefetches only the exact pinned model/data commits into a bounded
56Gi writable cache, primes the dataset cache, then launches the existing
`sft_run.py` or `trl_run.py` with explicit `--device cuda` and offline model/data
loading. The smoke changes only `steps=2` and `eval_every=1`; all four audited
partition hashes remain unchanged. Full jobs retain 256 updates. No warm start,
private-state resume, data truncation, or smaller adapter substitution is allowed.

Before upload, another fresh CUDA process reloads the weights-only export and
checks its expert/router layout and a finite forward pass. The directory artifact
contains selected public metadata, metrics and `trainable/`, not optimizer state,
private randomness, W&B caches, or unrelated outputs. A manifest binds every
payload file's size and SHA-256; SFT artifacts are limited to 1Gi, probes to 16Mi.
The submitting client reloads the GCS artifact and verifies the manifest rather
than trusting an ephemeral pod path.

Runtime tracking uses distinct backend/coefficient/ratio names and public ZenML
run/source/image links. Only W&B's required key reaches the fresh trainer; no
ZenML, GCS, HF or unrelated credentials are forwarded. After a successful gate,
explicit online fail-open tracking retains completed training and local metrics
if an upload subsequently fails. `wandb_run.json` distinguishes training status
from upload status; `tracking_failure.json` records only the failing operation
and exception type. Such a result is not a successful tracking gate and does not
authorize another full job, but its valid training export remains recoverable.

### Independent generation and Mac scoring

`generate-v1` requires a matching pretrained gate and two immutable inputs in
the approved bucket: `--checkpoint-uri` / `--checkpoint-sha256` for the weights
artifact, and `--benchmark-uri` / `--benchmark-sha256` for the previously frozen
uncompressed full benchmark JSONL. Select one `--benchmark humaneval` or `mbpp`.
Downloads are bounded and verified before the credential-stripped generation
process starts. The same common inference loader, exported trainables, prompts,
greedy decoding, stopping policy and 512-token budget are used; there is no Docker
daemon or generated-code execution in this pod.

The generation artifact contains `answers/` plus public stage/runtime/tracking
receipts. It binds checkpoint/config/model revisions, benchmark bytes, task IDs,
decoding and sample checksums. Its runtime receipt includes actual CUDA device,
inference time and peak allocated/reserved GPU memory. Generation does not copy
weights into the answer artifact. Training, generation and scoring remain separate
stages: an evaluation failure does not rewrite a completed training receipt.

After downloading and validating the directory artifact, score on the Mac:

```bash
python -m examples.moe_privacy.campaign score \
  --samples-dir DOWNLOADED_ARTIFACT/answers \
  --expected-manifest-sha256 GENERATION_MANIFEST_SHA256 \
  --output-dir NEW_SCORING_ATTEMPT \
  --eval-image sha256:12657bbffce037d5e73ae30e852b0901c58cba6602ac0f709fbd8d859a6c9929 \
  --cpus 2 --memory-gb 8 --timeout-seconds 3600 --wandb-mode off
```

Use `stage_receipt.json`'s **`generation_manifest_sha256`**, not the outer bundle
hash. The image above is a verified **local** image ID, not a published registry
image. Every score attempt uses a fresh directory, retains per-task results and
an evaluator identity receipt, and refuses corrupt/partial/duplicate task sets.
Retry scoring from the same saved answers; do not regenerate or retrain merely
because scoring failed. The combined `campaign evaluate` interface remains
available for compatibility, and keeps historical scores/artifacts intact.

On the Mac's Docker kernel, network-none namespaces expose inert default tunnel
interfaces. The user-approved preflight permits only the known kernel defaults
when they are down, have no IPv4/IPv6 addresses and no non-loopback routes. Unknown,
active, addressed or routed interfaces fail. `--network none`, read-only root,
unprivileged UID/GID, capability dropping, no-new-privileges, process/memory/CPU
limits, and the no-Docker-socket/no-credential boundary remain unchanged.

Build and verify the evaluator with its bounded source context:

```bash
docker buildx build --platform linux/amd64 --load \
  -t opaque-code-eval:mac-network-none-20260923 \
  -f examples/moe_privacy/Dockerfile.eval examples/moe_privacy
python -m deploy.zenml.moe_iceland.verify_evaluator \
  --image opaque-code-eval:mac-network-none-20260923 --output-dir NEW_FIXTURE_RECEIPT
```

The live verification used four synthetic identity-function/isolation fixtures,
**not Mellum outputs or benchmark performance**. It recorded Docker `29.8.0`,
an `amd64` image on an `arm64` Linux engine, and a 300-second fixture-job timeout.
Full scoring has the separately declared common 3600-second job timeout. Because
the evaluator image and host timing differ from Coder, re-score retained answers
for **all compared arms** under the same Mac image/limits; never mix the new scores
with historical scores and claim matched protocol. Preserve both result sets.

For recovered Coder exports, `checkpoint.package_export` validates the completed
summary, audited partitions, privacy receipt and actual FP32 safetensor layout,
then creates a new weights-only transport directory. It never changes the original
failed tracking receipt or assumes a W&B failure means failed training. Missing or
invalid weights remain a blocker, not permission for duplicate training.

## Resource envelope and sequencing

- Default **both pods**: requests = limits = **0.2 CPU, 512Mi memory**; numeric
  UID/GID `1000`; Linux amd64; no GPU requests, configured volumes/PVCs, image-pull
  secret references, or environment secret references.
- Each pod requests `1Gi` and limits `4Gi` ephemeral storage (8Gi combined),
  comfortably below 128Gi. Admission-injected service-account projections are
  cluster-controlled; the manifests themselves do not add volumes or secrets.
- The step can have explicit CPU/memory overrides, but all four quantities plus
  `--acknowledge-confirmed-resources` are required. Limits are at most 8 CPU and
  16Gi memory. The orchestrator retains its small default envelope. Confirm
  scheduling separately from runtime memory fit; this profile does not use GPUs.
- A local macOS smoke plus correctness checks peaked around 1.1 GiB RSS; Linux
  image memory and additional ZenML overhead remain unmeasured. A 512Mi gate may
  OOM when ZenML and the real Mellum runner overlap. Do not
  silently increase it or treat an OOM as a pass. Obtain resource approval, then
  rerun smoke with all four acknowledged step quantities.
- Smoke/pilot Job active deadlines are 900/3600 seconds, including Pending time.
  Step subprocesses have shorter timeouts; submission/waiting is also bounded.
  Retries/backoff are disabled, disruptions fail the Job, and max parallelism is
  one. Finished Jobs have a one-hour TTL for investigation.
- A pilot requires a completed **`dp_aux` smoke run with `--checks`**, on the
  same project/stack, exact source hash, image digest, and image receipt. Its
  actual stored bundle is loaded and checked; a local success file is not proof.
- A local file lock and a server-side in-progress-run check prevent routine
  overlapping invocations. They are not a cross-host distributed lock: operators
  must serialize submissions from different laptops. There is no sweep, mapping,
  schedule, automatic retry, or implicit multi-arm fan-out.

## Dependencies and image

Use **Python 3.12.10** for both client and image. `requirements.in` pins
ZenML **0.96.4** and private **`jb-mlops[zenml]==0.0.45`**. Both client/runtime
requirements include the **same reviewed, hash-locked `requirements.lock`**.
The full lock resolves the public and private dependency closure; installation
requires authenticated access to the `space-tools` index.

The public profile includes Kubernetes 25.3.0, gcsfs 2024.12.0, GCS 2.19.0,
Container API 2.61.0, Artifact Registry API 1.21.0, Torch 2.14.0, Transformers
5.17.0, and PEFT 0.20.0. Two deliberate differences from a fresh/root-lock
resolution are necessary:

- `huggingface-hub==1.5.0`: meets Transformers 5.17.0's minimum. Hub 1.28.0 requires
  Click >=8.4.2, conflicting with ZenML 0.96.4's Click <=8.2.1.
- `google-api-core==2.34.0`: avoids the newer core's OpenTelemetry API >=1.44
  requirement, incompatible with ZenML's pinned SDK 1.43.0.

The public universal profile resolves, installs, passes dependency checking, and
imports real Mellum classes plus the GCP/Kubernetes service connectors. The
Linux amd64 CPython 3.12 **`torch==2.14.0+cpu`** wheel URL and SHA-256 in
`requirements-public.in` were verified against the PyTorch CPU index; no CUDA
wheel is selected for that platform. macOS arm64 uses Torch 2.14.0.
The parent runner must still validate the complete private deployment profile.

`Dockerfile.cpu` pins Python/Rust base-image digests, Rust 1.89.0, and uv 0.7.2.
It builds workspace wheels from bounded source, installs no unresolved runtime
dependencies, uses numeric `USER 1000:1000`, and never overrides `HOME`.
Private-index configuration is mounted only as a BuildKit secret: no credential
`ARG`, `ENV`, `COPY`, or committed index URL. Do not put credentials in any source
or requirements file. A local `.credentials/uv.toml` is excluded from Git and
source staging; configure its index/authentication according to your index's
documented settings. Its named `space-tools` index uses
`UV_INDEX_SPACE_TOOLS_USERNAME` and `UV_INDEX_SPACE_TOOLS_PASSWORD` from the
operator's environment. Docker receives these values only as BuildKit secrets,
never as build arguments or persistent image environment variables.

From the repository root, after obtaining private-index access:

```bash
D=deploy/zenml/moe_iceland
UV_CONFIG="$D/.credentials/uv.toml"
uv --config-file "$UV_CONFIG" pip compile --python-version 3.12 --universal \
  --generate-hashes --no-header --no-annotate --no-emit-index-url \
  --output-file "$D/requirements.lock" "$D/requirements.in"
uv venv --python 3.12.10 "$D/.client-venv"
uv --config-file "$UV_CONFIG" pip install --python "$D/.client-venv/bin/python" \
  --require-hashes -r "$D/requirements-client.txt"
uv pip check --python "$D/.client-venv/bin/python"
```

Review the entire lock, including private transitive dependencies. Any conflict
is a blocker, not a reason to use `--no-deps` for third-party packages. Keep test
tools in a separate environment; unexpected installed packages fail submission.

Only after the runner files are joined and image building is separately approved:

```bash
"$D/.client-venv/bin/python" -m deploy.zenml.moe_iceland.launch --prepare cpu-v1
docker buildx build --platform linux/amd64 --load \
  --secret "id=uv_config,src=$UV_CONFIG" --tag "$CONFIRMED_IMAGE_TAG" \
  --secret id=space_tools_username,env=UV_INDEX_SPACE_TOOLS_USERNAME \
  --secret id=space_tools_password,env=UV_INDEX_SPACE_TOOLS_PASSWORD \
  -f "$D/source-stages/cpu-v1/Dockerfile.cpu" "$D/source-stages/cpu-v1"
```

The registry project/repository path must match the live stack configuration.
The confirmed stack uses the `europe-docker.pkg.dev` repository above; a west4
hostname is accepted only if it matches the registered repository. Publish only
with explicit authorization and use an immutable `image@sha256:...` reference.
Never rewrite it to a registry-cache hostname. Before submitting, that exact
digest must already be present locally: image inspection uses
`docker run --pull=never --network=none
--read-only` with no mounts. It reads `/opt/moe-image.json`, not source from the
image. Rebuild and re-gate on dependency/build-recipe or workspace-source changes;
the launcher compares the actual image receipt with client versions and source
hashes before calling the pipeline. A changed runner/config needs a new smoke
gate, but not a new image when installed code/dependencies are unchanged.

## Dry-run, smoke, then pilot

No credentials, dependency installation, or image is needed for:

```bash
python3.12 -m deploy.zenml.moe_iceland.launch --dry-run
```

It prints the full pod settings, concrete checked runner command with a fresh
output directory, source inventory/readiness, and (when supplied) exact image
inspection command. Supply intended target/image flags to preview them; absent
values remain visibly unconfirmed. `--help` lists all flags.

For a **separately authorized real submission**, securely provide
exactly one of `ZENML_STORE_API_KEY` or a short-lived `ZENML_STORE_API_TOKEN` to
this process; never put credential values on the command line or in Git.
An authenticated SDK client can obtain its current session token with
`client.zen_store.get_or_generate_api_token()` and pass it directly in the
launcher's child-process environment, without exporting a persistent API key.
That token must remain valid for the bounded submission/wait period. ZenML
creates run-scoped workload tokens after a pipeline run exists; an unscoped
workload-token request is not a substitute for client login.
Keep any laptop `ZENML_STORE_URL` at the prod laptop endpoint. The launcher creates
an ephemeral child-only ZenML configuration with per-invocation
`ZENML_ACTIVE_PROJECT_ID` and `ZENML_ACTIVE_STACK_ID`; it never calls project/stack
activation APIs or writes the operator's existing client configuration.

```bash
"$D/.client-venv/bin/python" -m deploy.zenml.moe_iceland.launch --submit \
  --workspace prod --profile smoke --arm dp_aux --seed 0 \
  --project-name "$PROJECT_NAME" --project-id "$PROJECT_ID" \
  --confirm-project "$PROJECT_NAME" \
  --confirm-stack-id 28320977-dcbd-423d-b52e-a59888d8a24f \
  --image "$IMAGE_DIGEST" --service-account "$ORCHESTRATOR_SA" \
  --step-service-account "$STEP_SA"
```

Wait for successful completion and the printed GCS artifact URI. For the next
single pilot arm/seed invocation, keep the target flags and replace/add:

```bash
--profile pilot --arm dp_aux --seed 0 --gate-run-id "$SMOKE_RUN_ID" \
--cpu-request "$APPROVED_CPU_REQUEST" --cpu-limit "$APPROVED_CPU_LIMIT" \
--memory-request "$APPROVED_MEMORY_REQUEST" --memory-limit "$APPROVED_MEMORY_LIMIT" \
--acknowledge-confirmed-resources
```

This is an acknowledgment of **already confirmed** resources, not a request for
approval. Do not submit subsequent jobs while a previous run is active. On a
failure/timeout, check the printed run ID/name and actual run status before
inspecting steps/logs. Inspect the namespace's Jobs and Pods (including Pending
events/OOMs) and cancel confirmed leftover Jobs before retrying. The launcher
does not automatically delete remote resources or assume a timed-out submission
never reached the server.

## Source and artifact guarantees

Source staging explicitly initializes ZenML's source root without `zenml init`
or files in `.junie`. It includes only the runner/checks/configs, deployment
modules/requirements, root build metadata, and workspace packages named by the
root metadata. `.venv`, vendor/target, `.git`, `.temp`, `.worktrees`, caches,
credentials, source stages, test trees, binaries, and results are excluded.
Links are rejected except package `LICENSE`/`NOTICE` links to the corresponding
regular root files; staging copies their contents as regular files. The actual
ZenML archive file set and compressed/uncompressed
32MiB / 4096-entry caps are checked **before upload**.

`source-stages/` is deliberately not Git-ignored: ZenML's native `CodeArchive`
uses Git ignore rules even for an explicit source root. It is excluded by the
source allowlist instead. Submission stages are fresh and removed when the
invocation ends; named build contexts are retained for operator inspection.
No Git branch/index operation is performed.

In each step, Python workspace code comes from the verified downloaded archive,
not baked source. The one Rust accounting extension is copied from the attested
image wheel and hash-checked. The runner receives the archive-first `PYTHONPATH`;
its output directory must not exist. Its only seed argument is the public
model/data-generation seed; this deployment does not save or reconstruct secret
noise/sampling randomness.

The bundle contains `metrics.jsonl`, `summary.json`, `checks.json`, weights-only
`model/` files, and an additional `deployment.json` with source/dependency/image
provenance and output hashes. Unexpected/resumable-state filenames, links,
missing outputs, oversized metadata, and oversized bundles fail before upload.
Caps are 16MiB for smoke and 128MiB for pilot, with at most 256 entries. This is a
deployment check, not an independent audit of weights or mathematical privacy:
the real runner owns its checks and secret-randomness contract.

## Offline validation

`tests/integration/experiments/test_moe_iceland_deployment.py` exercises actual
ZenML models, Kubernetes manifests, source archives, and `PathMaterializer`
save/load after removing the original directory. Network access is denied in
tests. Use a separate test environment with the public profile, pytest/ruff, and
real pinned `opaque-base`/`opaque-accounting` (the repository's integration fixture
imports accounting); do not weaken or override that fixture.

```bash
mkdir -p deploy/zenml/moe_iceland/source-stages
PYTHONDONTWRITEBYTECODE=1 ZENML_ANALYTICS_OPT_IN=false \
  deploy/zenml/moe_iceland/.check-venv/bin/python -m pytest \
  tests/integration/experiments/test_moe_iceland_deployment.py \
  --basetemp=deploy/zenml/moe_iceland/source-stages/test-offline \
  -o cache_dir=deploy/zenml/moe_iceland/.pytest-cache -q
```

Neither these tests nor public dependency resolution verifies private jb-mlops
compatibility, Linux image construction, server permissions, GCS upload, cluster
admission, or that Mellum fits the default memory budget. Those remain explicit
prerequisites and must be verified with an authorized real smoke gate.