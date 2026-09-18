#!/usr/bin/env python3
"""Resolve releases, preserve generated notes, and verify release assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

VERSION_RE = re.compile(
    r"^(?P<major>0|[1-9][0-9]*)\."
    r"(?P<minor>0|[1-9][0-9]*)\."
    r"(?P<patch>0|[1-9][0-9]*)"
    r"(?:\.(?P<kind>alpha|beta|rc|post)(?P<number>0|[1-9][0-9]*))?$"
)
MAINTENANCE_BRANCH_RE = re.compile(
    r"^refs/heads/release/(?P<major>0|[1-9][0-9]*)\."
    r"(?P<minor>0|[1-9][0-9]*)$"
)
FENCES = (
    ("<!-- ai:begin -->", "<!-- ai:end -->"),
    ("<!-- auto:begin -->", "<!-- auto:end -->"),
    ("<!-- contributors:begin -->", "<!-- contributors:end -->"),
)
MANIFEST_NAME = "release-manifest.json"
MANIFEST_SCHEMA = 1


@dataclass(frozen=True)
class Version:
    """Supported Opake release version."""

    major: int
    minor: int
    patch: int
    kind: str | None = None
    number: int | None = None

    @classmethod
    def parse(cls, value: str) -> Version:
        match = VERSION_RE.fullmatch(value)
        if match is None:
            raise ValueError(
                f"unsupported release version {value!r}; expected "
                "X.Y.Z or X.Y.Z.{alphaN,betaN,rcN,postN}"
            )
        return cls(
            major=int(match["major"]),
            minor=int(match["minor"]),
            patch=int(match["patch"]),
            kind=match["kind"],
            number=int(match["number"]) if match["number"] is not None else None,
        )

    @property
    def series(self) -> str:
        return f"{self.major}.{self.minor}"

    @property
    def is_stable(self) -> bool:
        return self.kind is None

    @property
    def is_prerelease(self) -> bool:
        return self.kind in {"alpha", "beta", "rc"}

    def ordering_key(self) -> tuple[int, int, int, int, int]:
        phase = {
            "alpha": 0,
            "beta": 1,
            "rc": 2,
            None: 3,
            "post": 4,
        }[self.kind]
        return self.major, self.minor, self.patch, phase, self.number or 0

    def __str__(self) -> str:
        suffix = "" if self.kind is None else f".{self.kind}{self.number}"
        return f"{self.major}.{self.minor}.{self.patch}{suffix}"


@dataclass(frozen=True)
class ReleasePlan:
    """Immutable inputs and derived state for one release candidate."""

    source_ref: str
    source_branch: str
    target_sha: str
    version: str
    tag: str
    series: str
    maintenance_branch: str
    maintenance_branch_sha: str
    base_tag: str
    branch_action: str
    commit_range: str
    prerelease: bool


def _git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ValueError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def _ref_exists(repo: Path, ref: str) -> bool:
    return (
        subprocess.run(
            ["git", "-C", str(repo), "show-ref", "--verify", "--quiet", ref],
            check=False,
        ).returncode
        == 0
    )


def _maintenance_branch_sha(repo: Path, branch: str) -> str:
    for ref in (f"refs/remotes/origin/{branch}", f"refs/heads/{branch}"):
        if _ref_exists(repo, ref):
            return _git(repo, "rev-parse", f"{ref}^{{commit}}")
    return ""


def _is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    return (
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "merge-base",
                "--is-ancestor",
                ancestor,
                descendant,
            ],
            check=False,
        ).returncode
        == 0
    )


def _tags_since(repo: Path, base_sha: str, target_sha: str) -> list[str]:
    return _git(
        repo,
        "tag",
        "--merged",
        target_sha,
        "--no-merged",
        base_sha,
    ).splitlines()


def _reachable_versions(repo: Path, target_sha: str) -> dict[str, Version]:
    tags = _git(repo, "tag", "--merged", target_sha, "--list", "v*")
    versions: dict[str, Version] = {}
    for tag in tags.splitlines():
        try:
            versions[tag] = Version.parse(tag.removeprefix("v"))
        except ValueError:
            continue
    return versions


def _closest_tag(
    repo: Path,
    target_sha: str,
    versions: dict[str, Version],
) -> str:
    if not versions:
        return ""

    def key(item: tuple[str, Version]) -> tuple[int, tuple[int, int, int, int, int]]:
        tag, version = item
        distance = int(_git(repo, "rev-list", "--count", f"{tag}..{target_sha}"))
        return -distance, version.ordering_key()

    return max(versions.items(), key=key)[0]


def resolve_plan(
    *,
    repo: Path,
    source_ref: str,
    source_sha: str,
    requested_version: str,
) -> ReleasePlan:
    """Validate the selected branch and derive the exact release range."""

    if source_ref == "refs/heads/main":
        source_branch = "main"
        branch_series: str | None = None
    else:
        branch_match = MAINTENANCE_BRANCH_RE.fullmatch(source_ref)
        if branch_match is None:
            raise ValueError(
                "release preparation must run from main or release/X.Y, "
                f"not {source_ref!r}"
            )
        branch_series = f"{branch_match['major']}.{branch_match['minor']}"
        source_branch = f"release/{branch_series}"

    target_sha = _git(repo, "rev-parse", f"{source_sha}^{{commit}}")
    reachable = _reachable_versions(repo, target_sha)

    if source_branch == "main":
        if not requested_version:
            raise ValueError("an explicit version is required when releasing from main")
        version = Version.parse(requested_version)
    elif requested_version:
        version = Version.parse(requested_version)
        if version.series != branch_series:
            raise ValueError(
                f"version {version} belongs to {version.series}, but the selected "
                f"branch is release/{branch_series}"
            )
    else:
        stable = [
            candidate
            for candidate in reachable.values()
            if candidate.series == branch_series and candidate.is_stable
        ]
        if not stable:
            raise ValueError(
                f"release/{branch_series} has no reachable stable tag; "
                "provide an explicit version"
            )
        previous = max(stable, key=Version.ordering_key)
        version = Version(previous.major, previous.minor, previous.patch + 1)

    tag = f"v{version}"
    maintenance_branch = f"release/{version.series}"
    maintenance_branch_sha = _maintenance_branch_sha(repo, maintenance_branch)

    if source_branch == "main":
        if not maintenance_branch_sha:
            branch_action = "create"
        elif not _is_ancestor(repo, maintenance_branch_sha, target_sha):
            raise ValueError(
                f"{maintenance_branch} at {maintenance_branch_sha} cannot be "
                f"fast-forwarded to main target {target_sha}; dispatch preparation "
                f"from {maintenance_branch}"
            )
        else:
            intervening_tags = _tags_since(
                repo,
                maintenance_branch_sha,
                target_sha,
            )
            if intervening_tags:
                raise ValueError(
                    f"{maintenance_branch} cannot be fast-forwarded to main target "
                    f"{target_sha}; tags found after its current head: "
                    f"{', '.join(intervening_tags)}"
                )
            branch_action = "fast-forward"
    elif maintenance_branch != source_branch:
        raise ValueError(
            f"version {version} must be released from {maintenance_branch}, "
            f"not {source_branch}"
        )
    else:
        branch_action = "none"

    tag_ref = f"refs/tags/{tag}"
    tag_exists = _ref_exists(repo, tag_ref)
    if tag_exists:
        tagged_sha = _git(repo, "rev-parse", f"{tag_ref}^{{commit}}")
        if tagged_sha != target_sha:
            can_recover_tagged_candidate = (
                source_branch != "main"
                and bool(requested_version)
                and subprocess.run(
                    [
                        "git",
                        "-C",
                        str(repo),
                        "merge-base",
                        "--is-ancestor",
                        tagged_sha,
                        target_sha,
                    ],
                    check=False,
                ).returncode
                == 0
            )
            if not can_recover_tagged_candidate:
                raise ValueError(
                    f"{tag} already points to {tagged_sha}, "
                    f"not selected SHA {target_sha}"
                )
            target_sha = tagged_sha
            reachable = _reachable_versions(repo, target_sha)

    same_series = {
        candidate_tag: candidate
        for candidate_tag, candidate in reachable.items()
        if candidate.series == version.series and candidate_tag != tag
    }
    comparison_versions = (
        {
            candidate_tag: candidate
            for candidate_tag, candidate in reachable.items()
            if candidate_tag != tag
        }
        if source_branch == "main"
        else same_series
    )
    if comparison_versions and not tag_exists:
        latest = max(comparison_versions.values(), key=Version.ordering_key)
        if version.ordering_key() <= latest.ordering_key():
            raise ValueError(
                f"version {version} does not follow latest reachable release {latest}"
            )

    all_previous = {
        candidate_tag: candidate
        for candidate_tag, candidate in reachable.items()
        if candidate_tag != tag
    }
    released_candidates = {
        candidate_tag: candidate
        for candidate_tag, candidate in all_previous.items()
        if not candidate.is_prerelease
    }
    base_candidates = (
        released_candidates
        if source_branch == "main"
        else {
            candidate_tag: candidate
            for candidate_tag, candidate in released_candidates.items()
            if candidate.series == version.series
        }
    )
    # If the first release-line preparation stopped after creating its branch
    # but before completing its tag/draft, a retry comes from release/X.Y and
    # still compares against the prior line's closest published release.
    if source_branch != "main" and not base_candidates:
        base_candidates = released_candidates
    base_tag = _closest_tag(repo, target_sha, base_candidates)
    commit_range = f"{base_tag}..{target_sha}" if base_tag else target_sha

    return ReleasePlan(
        source_ref=source_ref,
        source_branch=source_branch,
        target_sha=target_sha,
        version=str(version),
        tag=tag,
        series=version.series,
        maintenance_branch=maintenance_branch,
        maintenance_branch_sha=maintenance_branch_sha,
        base_tag=base_tag,
        branch_action=branch_action,
        commit_range=commit_range,
        prerelease=version.is_prerelease,
    )


def _splice_fence(body: str, begin: str, end: str, content: str) -> str:
    fenced = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    replacement = f"{begin}\n{content.rstrip()}\n{end}"
    if fenced.search(body):
        return fenced.sub(lambda _: replacement, body)
    return body.rstrip() + "\n\n" + replacement + "\n"


def render_release_notes(
    *,
    existing_body: str,
    ai_highlights: str,
    auto_notes: str,
    contributors: str,
) -> str:
    """Update generated fences without changing handwritten release prose."""

    if not existing_body:
        return (
            "<!-- ai:begin -->\n"
            f"{ai_highlights.rstrip()}\n"
            "<!-- ai:end -->\n\n"
            "## What's changed\n\n"
            "<!-- auto:begin -->\n"
            f"{auto_notes.rstrip()}\n"
            "<!-- auto:end -->\n\n"
            "## Contributors\n\n"
            "<!-- contributors:begin -->\n"
            f"{contributors.rstrip()}\n"
            "<!-- contributors:end -->\n"
        )

    body = existing_body
    replacements = (
        (FENCES[0], ai_highlights),
        (FENCES[1], auto_notes),
        (FENCES[2], contributors),
    )
    for (begin, end), content in replacements:
        body = _splice_fence(body, begin, end, content)
    return body


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_manifest(
    *,
    asset_dir: Path,
    repository: str,
    tag: str,
    version: str,
    target_sha: str,
    source_branch: str,
    maintenance_branch: str,
    base_tag: str,
    preparation_run_id: str,
) -> dict[str, Any]:
    """Describe the complete immutable distribution asset set."""

    parsed = Version.parse(version)
    if tag != f"v{parsed}":
        raise ValueError(f"tag {tag!r} does not match version {version!r}")

    artifacts = sorted(
        (
            path
            for path in asset_dir.iterdir()
            if path.is_file()
            and path.name != MANIFEST_NAME
            and (path.suffix == ".whl" or path.name.endswith(".tar.gz"))
        ),
        key=lambda path: path.name,
    )
    if not artifacts:
        raise ValueError(f"no wheel or sdist artifacts found in {asset_dir}")

    return {
        "schema": MANIFEST_SCHEMA,
        "repository": repository,
        "tag": tag,
        "version": version,
        "target_sha": target_sha,
        "source_branch": source_branch,
        "maintenance_branch": maintenance_branch,
        "base_tag": base_tag,
        "preparation_run_id": str(preparation_run_id),
        "files": [
            {
                "name": path.name,
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in artifacts
        ],
    }


def write_manifest(manifest: dict[str, Any], output: Path) -> None:
    """Write a manifest with stable ordering and formatting."""

    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def verify_manifest(
    *,
    manifest_path: Path,
    asset_dir: Path,
    expected_repository: str = "",
    expected_tag: str = "",
    expected_target_sha: str = "",
) -> dict[str, Any]:
    """Verify manifest metadata and every local release asset."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"unsupported manifest schema: {manifest.get('schema')!r}")

    required_strings = (
        "repository",
        "tag",
        "version",
        "target_sha",
        "source_branch",
        "maintenance_branch",
        "preparation_run_id",
    )
    for key in required_strings:
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise ValueError(f"manifest field {key!r} must be a non-empty string")

    version = Version.parse(manifest["version"])
    if manifest["tag"] != f"v{version}":
        raise ValueError("manifest tag and version do not match")
    if expected_repository and manifest["repository"] != expected_repository:
        raise ValueError("manifest repository does not match the current repository")
    if expected_tag and manifest["tag"] != expected_tag:
        raise ValueError("manifest tag does not match the requested release")
    if expected_target_sha and manifest["target_sha"] != expected_target_sha:
        raise ValueError("manifest target SHA does not match the release tag")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("manifest files must be a non-empty list")

    names: list[str] = []
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("manifest file entries must be objects")
        name = entry.get("name")
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name == MANIFEST_NAME
        ):
            raise ValueError(f"invalid manifest filename: {name!r}")
        if name in names:
            raise ValueError(f"duplicate manifest filename: {name}")
        names.append(name)

        path = asset_dir / name
        if not path.is_file():
            raise ValueError(f"release asset is missing: {name}")
        if path.stat().st_size != entry.get("size"):
            raise ValueError(f"release asset size mismatch: {name}")
        if _sha256(path) != entry.get("sha256"):
            raise ValueError(f"release asset digest mismatch: {name}")

    if names != sorted(names):
        raise ValueError("manifest files are not sorted by name")

    actual_names = sorted(path.name for path in asset_dir.iterdir() if path.is_file())
    expected_names = sorted([*names, MANIFEST_NAME])
    if actual_names != expected_names:
        extra = sorted(set(actual_names) - set(expected_names))
        missing = sorted(set(expected_names) - set(actual_names))
        raise ValueError(
            f"release asset set does not match manifest; extra={extra}, missing={missing}"
        )
    return manifest


def _write_outputs(path: Path | None, values: dict[str, Any]) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as stream:
        for key, value in values.items():
            rendered = str(value).lower() if isinstance(value, bool) else str(value)
            stream.write(f"{key}={rendered}\n")


def _read(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--repo", type=Path, default=Path.cwd())
    plan.add_argument("--source-ref", required=True)
    plan.add_argument("--source-sha", required=True)
    plan.add_argument("--version", default="")
    plan.add_argument("--github-output", type=Path)

    notes = subparsers.add_parser("render-notes")
    notes.add_argument("--existing-body", type=Path)
    notes.add_argument("--ai-highlights", type=Path, required=True)
    notes.add_argument("--auto-notes", type=Path, required=True)
    notes.add_argument("--contributors", type=Path, required=True)
    notes.add_argument("--output", type=Path, required=True)

    create = subparsers.add_parser("create-manifest")
    create.add_argument("--asset-dir", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--repository", required=True)
    create.add_argument("--tag", required=True)
    create.add_argument("--version", required=True)
    create.add_argument("--target-sha", required=True)
    create.add_argument("--source-branch", required=True)
    create.add_argument("--maintenance-branch", required=True)
    create.add_argument("--base-tag", default="")
    create.add_argument("--preparation-run-id", required=True)

    verify = subparsers.add_parser("verify-manifest")
    verify.add_argument("--asset-dir", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--expected-repository", default="")
    verify.add_argument("--expected-tag", default="")
    verify.add_argument("--expected-target-sha", default="")
    verify.add_argument("--github-output", type=Path)

    return parser


def main() -> int:
    """Run the requested release helper command."""

    args = _build_parser().parse_args()
    if args.command == "plan":
        plan = resolve_plan(
            repo=args.repo,
            source_ref=args.source_ref,
            source_sha=args.source_sha,
            requested_version=args.version,
        )
        values = asdict(plan)
        print(json.dumps(values, indent=2, sort_keys=True))
        _write_outputs(args.github_output, values)
    elif args.command == "render-notes":
        rendered = render_release_notes(
            existing_body=_read(args.existing_body),
            ai_highlights=_read(args.ai_highlights),
            auto_notes=_read(args.auto_notes),
            contributors=_read(args.contributors),
        )
        args.output.write_text(rendered, encoding="utf-8")
    elif args.command == "create-manifest":
        manifest = create_manifest(
            asset_dir=args.asset_dir,
            repository=args.repository,
            tag=args.tag,
            version=args.version,
            target_sha=args.target_sha,
            source_branch=args.source_branch,
            maintenance_branch=args.maintenance_branch,
            base_tag=args.base_tag,
            preparation_run_id=args.preparation_run_id,
        )
        write_manifest(manifest, args.output)
    elif args.command == "verify-manifest":
        manifest = verify_manifest(
            manifest_path=args.manifest,
            asset_dir=args.asset_dir,
            expected_repository=args.expected_repository,
            expected_tag=args.expected_tag,
            expected_target_sha=args.expected_target_sha,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        _write_outputs(
            args.github_output,
            {key: value for key, value in manifest.items() if key != "files"},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
