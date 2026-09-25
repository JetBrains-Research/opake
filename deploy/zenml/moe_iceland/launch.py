"""Offline-first, single-job launcher for isolated Iceland experiments."""

import argparse
import fcntl
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager, nullcontext
from pathlib import Path
from uuid import UUID, uuid4

from .auth import (
    DISABLE_CACHE,
    collect_cached_auth,
    protected_auth,
    restore_cached_auth,
    validate_cached_login,
)
from .errors import refuse
from .gpu_policy import (
    CONFIGS,
    GPU_PROFILES,
    RESOURCE_PROFILE,
    gpu_resources,
    reserve_budget,
    validate_config,
)
from .settings import (
    ARTIFACT_STORE,
    CLIENT_URL,
    CPU_DEFAULTS,
    DEADLINES,
    DEPLOY_PATH,
    NAMESPACE,
    PARENT_REVISION,
    STACK_ID,
    STACK_NAME,
    pipeline_settings,
    resources,
    runner_command,
    validate_execution_plan,
    validate_image,
    validate_project,
)
from .source import inventory, stage_source


def parser() -> argparse.ArgumentParser:
    """Expose versioned single-job profiles, with dry-run as the default action."""
    result = argparse.ArgumentParser(description=__doc__)
    mode = result.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", help="Default; no ZenML import/contact."
    )
    mode.add_argument(
        "--submit", action="store_true", help="Explicitly authorize one real run."
    )
    mode.add_argument(
        "--prepare", metavar="NAME", help="Create a new bounded build context."
    )
    result.add_argument(
        "--profile", choices=("smoke", "pilot", *GPU_PROFILES), default="smoke"
    )
    result.add_argument(
        "--cached-login",
        action="store_true",
        help="GPU only: transfer the confirmed cached ZenML login through stdin.",
    )
    result.add_argument(
        "--arm",
        choices=(
            "reference",
            "reference_aux",
            "dp",
            "dp_aux",
            "trl_reference",
            "trl_reference_aux",
        ),
        default="dp_aux",
    )
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--workspace", choices=("prod",), default="prod")
    result.add_argument("--project-name")
    result.add_argument("--project-id", type=UUID)
    result.add_argument("--confirm-project", metavar="NAME")
    result.add_argument("--confirm-stack-id", type=UUID)
    result.add_argument(
        "--image", help="Confirmed European registry image@sha256:digest."
    )
    result.add_argument(
        "--service-account", help="Confirmed existing orchestrator account."
    )
    result.add_argument(
        "--step-service-account", help="Confirmed existing step account."
    )
    result.add_argument("--gate-run-id", type=UUID)
    for name in ("cpu-request", "cpu-limit", "memory-request", "memory-limit"):
        result.add_argument(f"--{name}")
    result.add_argument("--acknowledge-confirmed-resources", action="store_true")
    result.add_argument("--backend", choices=("opaque", "trl"), default="opaque")
    result.add_argument("--config-name", choices=tuple(CONFIGS))
    result.add_argument("--config-sha256")
    result.add_argument("--source-sha256")
    result.add_argument("--authorization-id", type=UUID)
    result.add_argument("--deadline-utc")
    result.add_argument(
        "--gpu-seconds",
        type=int,
        help="Renewed aggregate allocation, including gates/generation.",
    )
    result.add_argument("--timeout-seconds", type=int)
    result.add_argument("--image-pull-secret", action="append", default=[])
    result.add_argument(
        "--service-account-verification",
        choices=("inspect", "admission"),
        default="inspect",
    )
    result.add_argument("--wandb-secret-name")
    result.add_argument("--wandb-secret-key")
    result.add_argument(
        "--reviewed-target-sha256",
        help="Digest of explicitly reviewed inherited settings.",
    )
    result.add_argument("--checkpoint-uri")
    result.add_argument("--checkpoint-sha256")
    result.add_argument("--benchmark", choices=("humaneval", "mbpp"))
    result.add_argument(
        "--benchmark-uri",
        help="Approved GCS object containing the frozen full JSONL snapshot.",
    )
    result.add_argument("--benchmark-sha256")
    result.add_argument("--_submit-plan", type=Path, help=argparse.SUPPRESS)
    result.add_argument(
        "--_cached-auth-stdin", action="store_true", help=argparse.SUPPRESS
    )
    return result


def make_plan(args: argparse.Namespace) -> dict:
    """Build an offline plan and require explicit acknowledgment for resource changes."""
    if not 0 <= args.seed < 2**32:
        refuse("The public generation seed must be in [0, 2**32).")
    supplied = (
        args.cpu_request,
        args.cpu_limit,
        args.memory_request,
        args.memory_limit,
    )
    if any(supplied) and not all(supplied):
        refuse(
            "Supply all four CPU/memory requests and limits, not a partial override."
        )
    gpu = args.profile in GPU_PROFILES
    if args.cached_login and not gpu:
        refuse("Cached login is restricted to GPU profiles.")
    if gpu and any(supplied):
        refuse(
            "The versioned GPU profile has fixed resources; CPU overrides are not accepted."
        )
    if args.profile == "pilot" and not all(supplied):
        refuse("Pilot requires explicit CPU/memory requests and limits.")
    if (
        any(supplied) or args.profile == "pilot"
    ) and not args.acknowledge_confirmed_resources:
        refuse(
            "Resource scheduling is unconfirmed: --acknowledge-confirmed-resources required."
        )
    if args.profile == "pilot" and not args.gate_run_id:
        refuse("Pilot requires --gate-run-id from a successful dp_aux smoke gate.")
    if args.project_name:
        validate_project(args.project_name)
    if args.image:
        validate_image(args.image)
    for name in (args.service_account, args.step_service_account):
        if name and not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", name):
            refuse("Provide an existing Kubernetes service account name.")
    job_id = uuid4().hex
    output = f"/tmp/moe-iceland-{job_id}/output"
    command = runner_command(args.profile, args.arm, args.seed, output)
    plan = {
        "schema": 1,
        "workspace": args.workspace,
        "client_url": CLIENT_URL,
        "project_name": args.project_name,
        "project_id": str(args.project_id) if args.project_id else None,
        "project_confirmed": bool(args.project_name)
        and args.confirm_project == args.project_name,
        "stack_confirmed": str(args.confirm_stack_id) == STACK_ID,
        "stack_name": STACK_NAME,
        "stack_id": STACK_ID,
        "namespace": NAMESPACE,
        "artifact_store": ARTIFACT_STORE,
        "parent_revision": PARENT_REVISION,
        "profile": args.profile,
        "arm": args.arm,
        "seed": args.seed,
        "image": args.image,
        "service_account": args.service_account,
        "step_service_account": args.step_service_account,
        "gate_run_id": str(args.gate_run_id) if args.gate_run_id else None,
        "resources": gpu_resources(args.profile)
        if gpu
        else resources(supplied if all(supplied) else CPU_DEFAULTS),
        "resources_acknowledged": args.acknowledge_confirmed_resources,
        "run_name": f"moe-iceland-{args.profile}-{args.arm}-s{args.seed}-{job_id[:12]}",
        "output_dir": output,
        "runner_command": command,
    }
    if gpu:
        plan.update(
            accelerator="cuda",
            resource_profile=RESOURCE_PROFILE,
            backend=args.backend,
            config_name=args.config_name,
            config_sha256=args.config_sha256,
            source_sha256=args.source_sha256,
            timeout_seconds=args.timeout_seconds,
            authorization={
                "id": str(args.authorization_id) if args.authorization_id else None,
                "deadline_utc": args.deadline_utc,
                "gpu_seconds": args.gpu_seconds,
            },
            image_pull_secrets=args.image_pull_secret,
            service_account_verification=args.service_account_verification,
            wandb_secret={"name": args.wandb_secret_name, "key": args.wandb_secret_key},
            reviewed_target_sha256=args.reviewed_target_sha256,
            checkpoint={"uri": args.checkpoint_uri, "sha256": args.checkpoint_sha256}
            if args.checkpoint_uri or args.checkpoint_sha256
            else None,
            benchmark=args.benchmark,
            benchmark_input={"uri": args.benchmark_uri, "sha256": args.benchmark_sha256}
            if args.benchmark_uri or args.benchmark_sha256
            else None,
        )
    return plan


def image_inspection_command(image: str) -> list[str]:
    """Read an already-present immutable image receipt without networking or mounts."""
    validate_image(image)
    return [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--read-only",
        "--cpus=0.2",
        "--memory=512m",
        "--user=1000:1000",
        "--entrypoint=/opt/venv/bin/python",
        image,
        "-c",
        "from pathlib import Path; print(Path('/opt/moe-image.json').read_text())",
    ]


def submission_environment(
    plan: dict, config: Path, root: Path, *, cached_login: bool = False
) -> dict:
    """Return a child environment; never mutate os.environ or the user's client config."""
    env = os.environ.copy()
    if env.get("ZENML_STORE_URL", CLIENT_URL).rstrip("/") != CLIENT_URL:
        refuse("Keep the laptop ZENML_STORE_URL at https://zenml.labs.jb.gg.")
    credentials = (
        () if cached_login else ("ZENML_STORE_API_KEY", "ZENML_STORE_API_TOKEN")
    )
    if cached_login:
        validate_cached_login(plan, env)
    elif sum(bool(env.get(name)) for name in credentials) != 1:
        refuse(
            "Provide exactly one of ZENML_STORE_API_KEY or ZENML_STORE_API_TOKEN "
            "securely for this invocation. The launcher "
            "does not copy credentials or switch the user's persisted ZenML config."
        )
    env = {
        name: value
        for name, value in env.items()
        if not name.startswith("ZENML_") or name in credentials
    }
    env.update(
        {
            "ZENML_STORE_URL": CLIENT_URL,
            "ZENML_ACTIVE_PROJECT_ID": plan["project_id"],
            "ZENML_ACTIVE_STACK_ID": STACK_ID,
            "ZENML_CONFIG_PATH": str(config),
            "ZENML_STORE_HTTP_TIMEOUT": "30",
            "ZENML_ANALYTICS_OPT_IN": "false",
            "ZENML_ENABLE_REPO_INIT_WARNINGS": "false",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(root),
        }
    )
    if cached_login:
        env[DISABLE_CACHE] = "true"
    return env


def authorize(args: argparse.Namespace, plan: dict) -> None:
    """Require affirmative target confirmation before any submission-side actions."""
    if not args.submit:
        refuse("Real submission requires --submit.")
    if not plan["project_id"] or args.confirm_project != plan["project_name"]:
        refuse("Provide project name/UUID and repeat its name in --confirm-project.")
    validate_project(plan["project_name"])
    if str(args.confirm_stack_id) != STACK_ID:
        refuse(f"Confirm the exact stack with --confirm-stack-id {STACK_ID}.")
    if not plan["image"]:
        refuse("A confirmed image digest is required; no image is built automatically.")
    if not plan["service_account"] or not plan["step_service_account"]:
        refuse(
            "Explicit existing service accounts are required; no automatic RBAC creation."
        )
    validate_execution_plan(plan)


@contextmanager
def submission_context(scope: Path, plan: dict):
    """Keep GPU submission receipts/evidence after errors and successful downloads."""
    if plan["profile"] in GPU_PROFILES:
        directory = scope / "submissions" / plan["run_name"]
        directory.mkdir(parents=True, exist_ok=False)
        yield directory
    else:
        with tempfile.TemporaryDirectory(
            prefix="run-", dir=scope / "source-stages"
        ) as temporary:
            yield Path(temporary)


def _submit_staged(plan_file: Path, *, cached_auth_stdin: bool = False) -> None:
    if cached_auth_stdin:
        os.environ[DISABLE_CACHE] = "true"
    root = Path(__file__).resolve().parents[3]
    if root != plan_file.parent / "source" or os.environ.get(
        "ZENML_CONFIG_PATH"
    ) != str(plan_file.parent / "zenml-config"):
        refuse(
            "The internal submission entry point requires a launcher-created isolated stage."
        )
    if (
        os.environ.get("ZENML_STORE_URL") != CLIENT_URL
        or os.environ.get("ZENML_ACTIVE_STACK_ID") != STACK_ID
    ):
        refuse(
            "The isolated submission process has inconsistent target environment settings."
        )
    plan = json.loads(plan_file.read_text())
    if os.environ.get("ZENML_ACTIVE_PROJECT_ID") != plan["project_id"]:
        refuse("The isolated process project differs from the confirmed plan.")
    validate_execution_plan(plan)
    if cached_auth_stdin:
        restore_cached_auth(sys.stdin.buffer, plan)

    from .provenance import validate_contract
    from .remote import submit_one
    from .settings import validate_native_settings
    from .source import validate_code_archive

    accelerator = plan.get("accelerator", "cpu")
    validate_contract(root, plan["image_contract"], accelerator=accelerator)
    if accelerator == "cuda":
        validate_config(root, plan)
    validate_native_settings(plan)
    archive = validate_code_archive(root, plan_file.parent / "code-check.tar.gz")
    print(json.dumps({"source_archive": archive}), flush=True)
    result = submit_one(root, plan)
    print(json.dumps(result, indent=2), flush=True)


def main(argv: list[str] | None = None) -> int:
    """Print a dry-run, prepare source, or explicitly authorize one isolated submission."""
    args = parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    scope = root / DEPLOY_PATH
    try:
        if args._cached_auth_stdin and not args._submit_plan:
            refuse("Cached authentication stdin is only valid for the staged child.")
        if args.cached_login and (
            args.profile not in GPU_PROFILES or args.prepare or args._submit_plan
        ):
            refuse("--cached-login is only valid for parent GPU launch profiles.")
        if args._submit_plan:
            if not args.submit:
                refuse("The staged child requires explicit --submit authorization.")
            # Restore credentials under protected_auth only; keep submit/wait errors
            # visible so scheduling, CUDA and artifact failures are not mislabeled.
            _submit_staged(
                args._submit_plan, cached_auth_stdin=args._cached_auth_stdin
            )
            return 0
        if args.prepare:
            if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.prepare):
                refuse("Choose a fresh simple build-context name, not a path.")
            context = scope / "source-stages" / args.prepare
            context.mkdir(parents=True, exist_ok=False)
            source = stage_source(
                root,
                context / "source",
                accelerator="cuda" if args.profile in GPU_PROFILES else "cpu",
            )
            recipe = (
                "Dockerfile.cuda" if args.profile in GPU_PROFILES else "Dockerfile.cpu"
            )
            shutil.copyfile(scope / recipe, context / recipe)
            shutil.copyfile(scope / ".dockerignore", context / ".dockerignore")
            print(
                json.dumps({"build_context": str(context), "source": source}, indent=2)
            )
            return 0
        plan = make_plan(args)
        if not args.submit:
            try:
                source = inventory(root)
            except (ValueError, FileNotFoundError) as error:
                source = {"not_ready": str(error)}
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "plan": plan,
                        "pipeline_settings": pipeline_settings(plan),
                        "runner_command": shlex.join(plan["runner_command"]),
                        "image_check_command": shlex.join(
                            image_inspection_command(plan["image"])
                        )
                        if plan["image"]
                        else None,
                        "source": source,
                        "not_verified": [
                            "prod server access, actual project/stack IDs and stored settings",
                            "existing service accounts, RBAC, registry pull and GCS permissions",
                            "private dependency lock, image build and image/client provenance",
                            "smoke execution and uploaded artifact; larger resources are unconfirmed",
                        ],
                    },
                    indent=2,
                )
            )
            return 0
        authorize(args, plan)
        if plan["profile"] in GPU_PROFILES:
            validate_config(root, plan)
        # Validate authentication before any local image inspection or source staging.
        submission_environment(
            plan, scope / ".local-zenml", root, cached_login=args.cached_login
        )
        cached_auth = collect_cached_auth(plan) if args.cached_login else None
        contexts = scope / "source-stages"
        contexts.mkdir(exist_ok=True)
        with (scope / ".submission.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with submission_context(scope, plan) as context:
                staged = context / "source"
                source = stage_source(
                    root, staged, accelerator=plan.get("accelerator", "cpu")
                )
                if (
                    plan["profile"] in GPU_PROFILES
                    and plan["source_sha256"] != source["source_sha256"]
                ):
                    refuse(
                        "Source hash changed since approval; inspect the new dry-run."
                    )
                inspection = subprocess.run(
                    image_inspection_command(plan["image"]),
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=45,
                )
                if len(inspection.stdout) > 512 * 1024:
                    refuse("Oversized image receipt.")
                plan["image_contract"] = json.loads(inspection.stdout)
                plan["source_sha256"] = source["source_sha256"]
                if plan["profile"] in GPU_PROFILES:
                    from .provenance import validate_contract

                    validate_contract(
                        staged, plan["image_contract"], accelerator="cuda"
                    )
                    reserve_budget(scope / "gpu-authorizations", plan)
                plan_file = context / "plan.json"
                plan_file.write_text(json.dumps(plan, indent=2) + "\n")
                env = submission_environment(
                    plan,
                    context / "zenml-config",
                    staged,
                    cached_login=args.cached_login,
                )
                command = [
                    sys.executable,
                    "-m",
                    "deploy.zenml.moe_iceland.launch",
                    "--submit",
                    "--_submit-plan",
                    str(plan_file),
                ]
                if args.cached_login:
                    command.append("--_cached-auth-stdin")
                print(
                    json.dumps(
                        {"plan": plan, "pipeline_settings": pipeline_settings(plan)},
                        indent=2,
                    ),
                    flush=True,
                )
                # Do not wrap the child process in protected_auth: that would rewrite
                # archive, scheduling, and target failures as a cached-login error.
                subprocess.run(
                    command,
                    cwd=staged,
                    env=env,
                    check=True,
                    timeout=(
                        plan["timeout_seconds"]
                        if plan["profile"] in GPU_PROFILES
                        else DEADLINES[plan["profile"]]
                    )
                    + 240,
                    **({"input": cached_auth} if args.cached_login else {}),
                )
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"Deployment refused/failed: {error}", file=sys.stderr)
        print(
            "If submission may have started, inspect the printed run ID/name and its "
            "Kubernetes Jobs before retrying. No retry or remote cleanup is automatic.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
