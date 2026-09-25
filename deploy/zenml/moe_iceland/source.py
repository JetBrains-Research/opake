"""Allowlisted, size-bounded source staging and content provenance."""

import hashlib
import json
import os
import shutil
import subprocess
import tomllib
from pathlib import Path

from .errors import refuse
from .settings import DEPLOY_PATH, PARENT_REVISION, SOURCE_FILE_LIMIT, SOURCE_LIMIT

EXCLUDED = {
    ".git",
    ".junie",
    ".temp",
    ".venv",
    ".worktrees",
    ".zen",
    "__pycache__",
    "vendor",
    "target",
    "node_modules",
    ".check-venv",
    ".client-venv",
    ".local-zenml",
    "staging",
    "source-stages",
    "receipts",
    "dist",
    "build",
}
ROOT_FILES = (
    "pyproject.toml",
    "uv.lock",
    "Cargo.toml",
    "Cargo.lock",
    "LICENSE",
    "NOTICE",
)
MANIFEST = "moe-source.json"
SFT_SOURCES = {
    "sft_run.py",
    "sft_data.py",
    "sft_adapters.py",
    "sft_metrics.py",
    "tracking.py",
    "trl_run.py",
    "campaign.py",
    "campaign_report.py",
    "code_eval.py",
    "handoff.py",
}
SOURCE_SUFFIXES = {
    ".py",
    ".pyi",
    ".rs",
    ".toml",
    ".lock",
    ".json",
    ".md",
    ".txt",
    ".in",
}
WORKSPACE_MEMBER_DEPTH = 2
PACKAGE_TEST_DEPTH = 4
DEPLOY_FILES = {
    "README.md",
    "Dockerfile.cpu",
    "Dockerfile.cuda",
    "requirements-gpu.in",
    "requirements-gpu.lock",
    "requirements-gpu-runtime.txt",
    "requirements-gpu-build.txt",
    "requirements.in",
    "requirements-public.in",
    "requirements-client.txt",
    "requirements-runtime.txt",
    "requirements.lock",
}


def sha256(path: Path) -> str:
    """Hash a file without loading it into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest_json(value: object) -> str:
    """Hash canonical JSON, independently of dictionary insertion order."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def package_paths(root: Path) -> list[str]:
    """Use workspace metadata rather than a parallel hand-maintained package map."""
    with (root / "pyproject.toml").open("rb") as stream:
        members = tomllib.load(stream)["tool"]["uv"]["workspace"]["members"]
    if not members or any(
        not isinstance(member, str)
        or not member.startswith("packages/opaque-")
        or len(Path(member).parts) != WORKSPACE_MEMBER_DEPTH
        or any(char in member for char in "*?[]\\")
        for member in members
    ):
        refuse("Expected explicit packages/opaque-* workspace members.")
    return sorted(members)


def source_files(root: Path) -> dict[str, Path]:
    """Collect only bounded experiment, deployment, and pinned workspace source."""
    files = {}
    for name in ROOT_FILES:
        path = root / name
        if not path.is_file() or path.is_symlink():
            refuse(f"Missing regular source file: {name}")
        files[name] = path
    roots = ["examples/moe_privacy", DEPLOY_PATH, *package_paths(root)]
    for directory in roots:
        base = root / directory
        if (
            not base.is_dir()
            or base.is_symlink()
            or not base.resolve().is_relative_to(root.resolve())
        ):
            refuse(f"Missing regular source directory: {directory}")
        for current, dirs, names in os.walk(base, followlinks=False):
            dirs[:] = sorted(
                name
                for name in dirs
                if name not in EXCLUDED
                and not name.startswith(".")
                and name != "tests"
                and not name.endswith(".egg-info")
            )
            for name in [*dirs, *names]:
                path = Path(current) / name
                package_license = (
                    directory.startswith("packages/")
                    and path.parent == base
                    and name in {"LICENSE", "NOTICE"}
                    and path.resolve() == (root / name).resolve()
                )
                if path.is_symlink() and not package_license:
                    refuse(f"Symlink in source allowlist: {path}")
            for name in sorted(names):
                path = Path(current) / name
                if name.startswith(".") or name.endswith(".local.json"):
                    continue
                relative = path.relative_to(base)
                if directory == DEPLOY_PATH and (
                    len(relative.parts) != 1
                    or (path.suffix != ".py" and name not in DEPLOY_FILES)
                ):
                    continue
                if directory == "examples/moe_privacy" and not (
                    relative.as_posix()
                    in {
                        "__init__.py",
                        "run.py",
                        "checks.py",
                        "compare.py",
                        "README.md",
                        *SFT_SOURCES,
                    }
                    or (relative.parent == Path("configs") and path.suffix == ".json")
                ):
                    continue
                if path.suffix not in SOURCE_SUFFIXES and name not in {
                    "Dockerfile.cpu",
                    "Dockerfile.cuda",
                    "py.typed",
                    "LICENSE",
                    "NOTICE",
                }:
                    continue
                if not path.is_file():
                    refuse(f"Non-regular source file: {path}")
                files[path.relative_to(root).as_posix()] = path
    if len(files) > SOURCE_FILE_LIMIT:
        refuse(f"Source exceeds {SOURCE_FILE_LIMIT} files.")
    if sum(path.stat().st_size for path in files.values()) > SOURCE_LIMIT:
        refuse("Uncompressed source exceeds the 32MiB cap.")
    return dict(sorted(files.items()))


def inventory(root: Path) -> dict:
    """Record source, workspace, and root-lock hashes for provenance checks."""
    files = source_files(root)
    hashes = {name: sha256(path) for name, path in files.items()}
    workspace = {
        name: value
        for name, value in hashes.items()
        if name.startswith("packages/") or name in ROOT_FILES
    }
    return {
        "schema": 1,
        "parent_revision": PARENT_REVISION,
        "files": hashes,
        "source_sha256": digest_json(hashes),
        "workspace_sha256": digest_json(workspace),
        "uv_lock_sha256": hashes["uv.lock"],
        "bytes": sum(path.stat().st_size for path in files.values()),
    }


def require_pinned_workspace(root: Path, *, allow_test_changes: bool = False) -> str:
    """Reject workspace changes relative to the pinned parent without changing Git."""

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True, timeout=30
        ).strip()

    head = git("rev-parse", "HEAD")
    if git("rev-parse", f"{PARENT_REVISION}^{{commit}}") != PARENT_REVISION:
        refuse("The pinned parent revision is unavailable.")
    git("merge-base", "--is-ancestor", PARENT_REVISION, head)
    paths = [*ROOT_FILES, "packages"]
    changed = git("diff", "--name-only", PARENT_REVISION, "--", *paths).splitlines()
    if changed and (
        not allow_test_changes or any(not _package_test(path) for path in changed)
    ):
        refuse("Workspace source or root dependencies drifted; review and rebuild.")
    untracked = git(
        "ls-files", "--others", "--exclude-standard", "--", "packages"
    ).splitlines()
    if untracked and (
        not allow_test_changes or any(not _package_test(path) for path in untracked)
    ):
        refuse("Untracked workspace source; review and rebuild before deployment.")
    return head


def _package_test(path: str) -> bool:
    parts = Path(path).parts
    return (
        len(parts) >= PACKAGE_TEST_DEPTH
        and parts[0] == "packages"
        and parts[2] == "tests"
    )


def stage_source(root: Path, destination: Path, *, accelerator: str = "cpu") -> dict:
    """Copy only approved source. The destination must be new and is never overwritten."""
    head = (
        require_pinned_workspace(root, allow_test_changes=True)
        if accelerator == "cuda"
        else require_pinned_workspace(root)
    )
    before = inventory(root)
    if accelerator not in {"cpu", "cuda"}:
        refuse("Unknown source accelerator contract.")
    if (
        accelerator == "cuda"
        and not {
            *(f"examples/moe_privacy/{name}" for name in SFT_SOURCES),
            f"{DEPLOY_PATH}/gpu_runner.py",
            f"{DEPLOY_PATH}/Dockerfile.cuda",
            f"{DEPLOY_PATH}/requirements-gpu.lock",
        }
        <= before["files"].keys()
    ):
        refuse("CUDA source is missing its exact trainer/build dependencies.")
    for name in (
        "__init__.py",
        "run.py",
        "checks.py",
        "configs/smoke.json",
        "configs/pilot.json",
    ):
        if f"examples/moe_privacy/{name}" not in before["files"]:
            refuse(f"Runner contract is incomplete: examples/moe_privacy/{name}")
    if f"{DEPLOY_PATH}/requirements.lock" not in before["files"]:
        refuse("Resolve and review the private dependency lock before staging.")
    destination.mkdir(parents=True, exist_ok=False)
    for name, path in source_files(root).items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    after = inventory(destination)
    if after != before:
        refuse("Source changed during staging; discard this stage and retry.")
    before["checkout_revision"] = head
    (destination / MANIFEST).write_text(json.dumps(before, indent=2) + "\n")
    # ZenML's CodeArchive walks parent Git repositories. GPU submissions live under
    # an ignored directory, so seed a local repository that tracks only staged files.
    _seed_archive_repository(destination)
    return before


def _seed_archive_repository(destination: Path) -> None:
    """Make staged source archivable even when the parent path is gitignored."""
    if (destination / ".git").exists():
        refuse("Staged source must not already contain a .git directory.")
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "GIT_DIR",
            "GIT_WORK_TREE",
            "GIT_COMMON_DIR",
            "GIT_INDEX_FILE",
            "GIT_OBJECT_DIRECTORY",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        }
    }
    env.update(
        GIT_CONFIG_GLOBAL="/dev/null",
        GIT_CONFIG_SYSTEM="/dev/null",
        GIT_CONFIG_NOSYSTEM="1",
    )

    def git(*args: str) -> None:
        completed = subprocess.run(
            ["git", "-C", str(destination), *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if completed.returncode != 0:
            refuse(
                "Failed to seed a local Git repository for the staged source archive."
            )

    git("init", "--quiet")
    git("config", "user.email", "moe-iceland@local")
    git("config", "user.name", "moe-iceland")
    git("add", "-A")
    git("commit", "--quiet", "-m", "staged source archive")


def verify_source(root: Path) -> dict:
    """Verify that a staged or downloaded source tree matches its manifest."""
    manifest = json.loads((root / MANIFEST).read_text())
    actual = inventory(root)
    if any(manifest.get(key) != value for key, value in actual.items()):
        refuse("Downloaded source does not match its content manifest.")
    return manifest


def validate_code_archive(root: Path, output: Path) -> dict:
    """Exercise ZenML's actual archive implementation before allowing upload."""
    from zenml.utils.code_utils import CodeArchive

    manifest = verify_source(root)
    # CodeArchive joins absolute Git paths with the provided root; a relative root
    # under an ignored parent path yields an empty archive even when files exist.
    archive = CodeArchive(root=str(root.resolve(strict=True)))
    expected = {*manifest["files"], MANIFEST}
    files = archive.get_files()
    if set(files) != expected:
        refuse(
            "ZenML would archive a different file set (possibly an ignored staging "
            "directory). Do not upload; use the isolated stage instructions."
        )
    if (
        len(files) > SOURCE_FILE_LIMIT
        or sum(Path(path).stat().st_size for path in files.values()) > SOURCE_LIMIT
    ):
        refuse("Source plus manifest exceeds the file/32MiB archive cap.")
    with output.open("x+b") as stream:
        archive.write_archive(stream)
    if output.stat().st_size > SOURCE_LIMIT:
        refuse("Compressed source archive exceeds the 32MiB cap.")
    return {"sha256": sha256(output), "bytes": output.stat().st_size}
