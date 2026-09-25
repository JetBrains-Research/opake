"""Dependency/image attestation and an archive-first workspace overlay."""

import argparse
import base64
import csv
import importlib.metadata as metadata
import io
import json
import platform
import re
import shutil
import sys
import sysconfig
from pathlib import Path

from .errors import refuse
from .settings import DEPLOY_PATH, PARENT_REVISION
from .source import digest_json, package_paths, sha256, verify_source

IMAGE_CONTRACT = Path("/opt/moe-image.json")
PINS = {
    "zenml": "0.96.4",
    "jb-mlops": "0.0.45",
    "kubernetes": "25.3.0",
    "gcsfs": "2024.12.0",
    "transformers": "5.17.0",
    "peft": "0.20.0",
    "torch": "2.14.0",
    "huggingface-hub": "1.5.0",
    "google-api-core": "2.34.0",
}
# Iceland node driver reports 12080 (CUDA 12.8). Prefer the 12.6 runtime wheel so
# torch can initialize without requiring a CUDA 13 host driver.
CUDA_VERSION = "12.6"
CUDA_TORCH_SHA256 = "e9922441cb2ed269742a46246840c53634ca963f8b0a38d30182c28b4e8388da"
CUDA_PINS = {**PINS, "trl": "1.13.0", "torch": "2.14.0+cu126"}
WHEEL_RECORD_COLUMNS = 3
CLIENT_REQUIREMENTS = (
    "zenml",
    "jb-mlops[zenml]",
    "kubernetes",
    "gcsfs",
    "google-api-core",
    "google-cloud-container",
    "google-cloud-artifact-registry",
    "google-cloud-storage",
)


def _accelerator(value: str) -> None:
    if value not in {"cpu", "cuda"}:
        refuse("Choose accelerator='cpu' or accelerator='cuda' explicitly.")


def _cuda_dependencies(lock: Path) -> dict[str, str]:
    """Read the entire Linux CUDA lock, without consulting the client's platform."""
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name

    result = {}
    for line in lock.read_text().replace("\\\n", " ").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirement_text, *hashes = re.split(r"\s+--hash=", line)
        if not hashes or any(
            not re.fullmatch(r"sha256:[0-9a-f]{64}", value.strip()) for value in hashes
        ):
            refuse("Resolve a complete CUDA hash lock without embedded index options.")
        try:
            requirement = Requirement(requirement_text)
        except InvalidRequirement:
            refuse("Invalid requirement in the CUDA hash lock.")
        name = canonicalize_name(requirement.name)
        if (
            requirement.url
            or requirement.marker
            or requirement.extras
            or not re.fullmatch(r"==[a-zA-Z0-9.!+-]+", str(requirement.specifier))
        ):
            refuse("The CUDA lock requires exact pins, no URLs, extras, or markers.")
        if name in result:
            refuse(f"Duplicate dependency in CUDA lock: {name}")
        if name in {"opaque", "opake"} or name.startswith(("opaque-", "opake-")):
            refuse(
                "Build the PR workspace; do not install published workspace packages."
            )
        if name == "torch" and {value.strip() for value in hashes} != {
            f"sha256:{CUDA_TORCH_SHA256}"
        }:
            refuse(
                "The CUDA lock must select the verified CPython 3.12 Linux torch wheel."
            )
        result[name] = str(requirement.specifier)[2:]
    for name, version in CUDA_PINS.items():
        if result.get(name) != version:
            refuse(f"The CUDA dependency lock must install {name}=={version}.")
    if not {"cuda-toolkit", "cuda-bindings", "triton"} <= result.keys() or not any(
        name.startswith("nvidia-") for name in result
    ):
        refuse("The CUDA dependency lock is missing its CUDA runtime libraries.")
    return dict(sorted(result.items()))


def _client_platform() -> None:
    if sys.version_info[:2] != (3, 12):
        refuse("Use Python 3.12 for both client and image.")
    if (sys.platform, platform.machine()) not in {
        ("linux", "x86_64"),
        ("darwin", "arm64"),
    }:
        refuse("Only Linux amd64 and macOS arm64 clients are verified.")


def _installed_version(name: str, expected: str) -> str:
    try:
        installed = metadata.version(name)
    except metadata.PackageNotFoundError:
        refuse(f"Missing pinned dependency: {name}; install the reviewed lock.")
    if installed != expected:
        refuse(f"Dependency drift: {name}=={installed}; reinstall the lock.")
    return installed


def _client_versions(expected: dict[str, str]) -> dict[str, str]:
    """Check orchestration and its transitive extras, not laptop training packages."""
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    _client_platform()
    environment = default_environment()
    pending = [Requirement(value) for value in CLIENT_REQUIREMENTS]
    seen = set()
    result = {}
    while pending:
        requirement = pending.pop()
        name = canonicalize_name(requirement.name)
        key = (name, frozenset(requirement.extras))
        if name not in expected or requirement.url:
            refuse(f"Client orchestration dependency is outside the CUDA lock: {name}")
        installed = result.get(name) or _installed_version(name, expected[name])
        if installed not in requirement.specifier:
            refuse(
                f"Client orchestration metadata conflicts with the CUDA lock: {name}"
            )
        result[name] = installed
        if key in seen:
            continue
        seen.add(key)
        for value in metadata.requires(name) or ():
            dependency = Requirement(value)
            if not dependency.marker or any(
                dependency.marker.evaluate({**environment, "extra": extra})
                for extra in {"", *requirement.extras}
            ):
                pending.append(dependency)
    return dict(sorted(result.items()))


def _cuda_installed_versions(expected: dict[str, str]) -> dict[str, str]:
    from packaging.utils import canonicalize_name

    result = {
        name: _installed_version(name, version) for name, version in expected.items()
    }
    installed = {
        canonicalize_name(distribution.metadata["Name"])
        for distribution in metadata.distributions(
            path={sysconfig.get_path("purelib"), sysconfig.get_path("platlib")}
        )
    }
    unexpected = {
        name
        for name in installed - result.keys() - {"pip", "uv"}
        if not name.startswith("opaque-")
    }
    if unexpected:
        refuse(f"Packages outside the CUDA lock: {sorted(unexpected)}")
    return result


def _cuda_wheel_names(dependencies: dict[str, str]) -> set[str]:
    return {
        name
        for name in dependencies
        if name in {"torch", "triton"} or name.startswith(("cuda-", "nvidia-"))
    }


def _cuda_wheel_hashes(dependencies: dict[str, str]) -> dict[str, str]:
    """Verify RECORD hashes and attest installed bytes, not just version metadata."""
    result = {}
    for name in sorted(_cuda_wheel_names(dependencies)):
        distribution = metadata.distribution(name)
        try:
            files = list(
                csv.reader(
                    io.StringIO(distribution.read_text("RECORD") or ""), strict=True
                )
            )
        except csv.Error:
            refuse(f"Malformed CUDA wheel RECORD: {name}")
        if any(len(row) != WHEEL_RECORD_COLUMNS or not row[0] for row in files):
            refuse(f"Malformed CUDA wheel RECORD: {name}")
        records = [row[0] for row in files if row[0].endswith(".dist-info/RECORD")]
        if len(records) != 1:
            refuse(f"CUDA wheel has no unique RECORD: {name}")
        site = Path(distribution.locate_file("")).resolve()
        scripts = Path(sysconfig.get_path("scripts")).resolve()
        hashes = {}
        for entry, recorded_hash, recorded_size in files:
            if entry.endswith(".pyc"):
                continue
            path = Path(distribution.locate_file(entry))
            if (
                Path(entry).is_absolute()
                or path.is_symlink()
                or not path.is_file()
                or not (
                    path.resolve().is_relative_to(site)
                    or path.resolve().parent == scripts
                )
                or entry in hashes
            ):
                refuse(f"Invalid installed CUDA wheel file: {name}")
            if recorded_size and recorded_size != str(path.stat().st_size):
                refuse(f"CUDA wheel RECORD size drift: {name}")
            value = sha256(path)
            if recorded_hash:
                encoded = (
                    base64.urlsafe_b64encode(bytes.fromhex(value)).decode().rstrip("=")
                )
                if recorded_hash != f"sha256={encoded}":
                    refuse(f"CUDA wheel RECORD hash drift: {name}")
            elif entry != records[0] and not (
                Path(entry).parent.name.endswith(".dist-info")
                and Path(entry).name in {"INSTALLER", "REQUESTED", "direct_url.json"}
            ):
                refuse(f"Unhashed installed CUDA wheel file: {name}")
            hashes[entry] = value
        result[name] = digest_json(hashes)
    return result


def _cuda_platform() -> None:
    if (sys.platform, platform.machine()) != ("linux", "x86_64"):
        refuse("Build and run the CUDA image on Linux amd64.")


def _cuda_image() -> None:
    import torch

    if torch.version.cuda != CUDA_VERSION:
        refuse(f"The image must use the pinned CUDA {CUDA_VERSION} torch wheel.")


def _cuda_receipt_hashes(contract: dict, dependencies: dict[str, str]) -> None:
    wheels = contract.get("cuda_wheels")
    native = contract.get("native_files")
    if (
        not isinstance(wheels, dict)
        or wheels.keys() != _cuda_wheel_names(dependencies)
        or not isinstance(native, dict)
        or len(native) != 1
        or not all(
            isinstance(name, str)
            and re.fullmatch(
                r"opaque/api/accounting/core/opaque_accounting\.[\w.-]+\.so", name
            )
            for name in native
        )
        or not all(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
            for value in [*wheels.values(), *native.values()]
        )
    ):
        refuse(
            "The CUDA receipt needs complete CUDA wheel and native accounting hashes."
        )


def locked_versions(lock: Path, *, accelerator: str = "cpu") -> dict[str, str]:
    """Check installed pins; CUDA on macOS checks only orchestration's closure."""
    _accelerator(accelerator)
    if accelerator == "cuda":
        _client_platform()
        expected = _cuda_dependencies(lock)
        if sys.platform == "darwin":
            return _client_versions(expected)
        return _cuda_installed_versions(expected)

    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    from packaging.version import Version

    if sys.version_info[:2] != (3, 12):
        refuse("Use Python 3.12 for both client and image.")
    if (sys.platform, platform.machine()) not in {
        ("linux", "x86_64"),
        ("darwin", "arm64"),
    }:
        refuse("Only Linux amd64 CPU and macOS arm64 clients are verified.")
    result = {}
    for line in lock.read_text().replace("\\\n", " ").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-") or "--hash=sha256:" not in line:
            refuse("Resolve a complete hash lock without embedded index options.")
        requirement = Requirement(re.sub(r"\s+--hash=sha256:[0-9a-f]{64}", "", line))
        name = canonicalize_name(requirement.name)
        if requirement.url and not (
            name == "torch"
            and requirement.url.startswith(
                "https://download-r2.pytorch.org/whl/cpu/torch-2.14.0%2Bcpu-"
            )
            and "#sha256=" in requirement.url
        ):
            refuse("Only the pinned public CPU torch URL is allowed in the lock.")
        if not requirement.url and not re.fullmatch(
            r"==[a-zA-Z0-9.+!-]+", str(requirement.specifier)
        ):
            refuse(f"Unpinned dependency in lock: {name}")
        if requirement.marker and not requirement.marker.evaluate():
            continue
        installed = metadata.version(name)
        if not requirement.url and Version(installed) not in requirement.specifier:
            refuse(f"Dependency drift: {name}=={installed}; reinstall the lock.")
        if requirement.url and installed != "2.14.0+cpu":
            refuse("The Linux image/client must use the pinned CPU torch wheel.")
        result[name] = installed
    for name, expected in PINS.items():
        if result.get(name, "").split("+", 1)[0] != expected:
            refuse(f"The dependency lock must install {name}=={expected}.")
    installed_names = {
        canonicalize_name(distribution.metadata["Name"])
        for distribution in metadata.distributions(
            path={sysconfig.get_path("purelib"), sysconfig.get_path("platlib")}
        )
    }
    unexpected = {
        name
        for name in installed_names - result.keys() - {"pip", "uv"}
        if not name.startswith("opaque-")
    }
    if unexpected:
        refuse(
            f"Packages outside the lock: {sorted(unexpected)}; use a fresh client environment."
        )
    return dict(sorted(result.items()))


def dependency_fingerprint(root: Path, *, accelerator: str = "cpu") -> str:
    """Bind the dependency lock and build recipe to an image receipt."""
    _accelerator(accelerator)
    names = (
        "requirements.in",
        "requirements-public.in",
        "requirements.lock",
        "requirements-client.txt",
        "requirements-runtime.txt",
        "Dockerfile.cpu",
        "provenance.py",
        "build_workspace.py",
    )
    if accelerator == "cuda":
        names = (
            "requirements-gpu.in",
            "requirements-gpu.lock",
            "requirements-gpu-runtime.txt",
            "requirements-gpu-build.txt",
            "requirements.lock",
            "Dockerfile.cuda",
            "resolve_gpu_lock.py",
            "provenance.py",
            "source.py",
            "build_workspace.py",
        )
    return digest_json({name: sha256(root / DEPLOY_PATH / name) for name in names})


def native_files() -> dict[str, Path]:
    """Locate the single installed accounting extension through wheel metadata."""
    distribution = metadata.distribution("opaque-accounting")
    files = {
        str(path): Path(distribution.locate_file(path))
        for path in distribution.files or ()
        if str(path).startswith("opaque/api/accounting/core/opaque_accounting.")
        and str(path).endswith(".so")
    }
    if len(files) != 1:
        refuse("Expected one installed opaque-accounting native extension.")
    return files


def build_contract(root: Path, *, accelerator: str = "cpu") -> dict:
    """Record image dependencies and installed binaries without probing a GPU."""
    _accelerator(accelerator)
    source = verify_source(root)
    lock = (
        root
        / DEPLOY_PATH
        / ("requirements-gpu.lock" if accelerator == "cuda" else "requirements.lock")
    )
    if accelerator == "cuda":
        _cuda_platform()
        dependencies = locked_versions(lock, accelerator="cuda")
    else:
        dependencies = locked_versions(lock)
        import torch

        if sys.platform != "linux" or torch.version.cuda is not None:
            refuse("Build the Linux amd64 CPU image, without CUDA dependencies.")
    contract = {
        "schema": 1,
        "parent_revision": PARENT_REVISION,
        "python": platform.python_version(),
        "platform": "linux/amd64",
        "workspace_sha256": source["workspace_sha256"],
        "uv_lock_sha256": source["uv_lock_sha256"],
        "dependency_sha256": dependency_fingerprint(root, accelerator=accelerator),
        "requirements_lock_sha256": sha256(lock),
        "dependencies": dependencies,
        "native_files": {name: sha256(path) for name, path in native_files().items()},
    }
    if accelerator == "cuda":
        contract.update(
            schema=2,
            accelerator="cuda",
            torch_cuda=CUDA_VERSION,
            cuda_wheels=_cuda_wheel_hashes(dependencies),
        )
        _cuda_image()
    return contract


def validate_contract(
    root: Path, contract: dict, *, runtime: bool = False, accelerator: str = "cpu"
) -> dict:
    """Check a receipt; CUDA clients need orchestration pins, not a CUDA installation.

    CUDA runtime validation checks the full installed lock and rehashes CUDA wheel
    contents and the PR-built accounting extension. It does not check GPU availability.
    CPU receipts retain their original schema and client/image parity rules.
    """
    _accelerator(accelerator)
    if contract.get("accelerator", "cpu") != accelerator:
        refuse("Image accelerator differs from the requested accelerator.")
    source = verify_source(root)
    lock = (
        root
        / DEPLOY_PATH
        / ("requirements-gpu.lock" if accelerator == "cuda" else "requirements.lock")
    )
    expected = {
        "schema": 2 if accelerator == "cuda" else 1,
        "parent_revision": PARENT_REVISION,
        "platform": "linux/amd64",
        "workspace_sha256": source["workspace_sha256"],
        "uv_lock_sha256": source["uv_lock_sha256"],
        "dependency_sha256": dependency_fingerprint(root, accelerator=accelerator),
        "requirements_lock_sha256": sha256(lock),
    }
    if accelerator == "cuda":
        expected.update(accelerator="cuda", torch_cuda=CUDA_VERSION)
    if any(contract.get(key) != value for key, value in expected.items()):
        refuse("Image/source/dependency mismatch. Review and rebuild the image.")
    if contract.get("python") != platform.python_version():
        refuse("Use the same exact Python patch version as the confirmed image.")
    if accelerator == "cuda":
        dependencies = _cuda_dependencies(lock)
        if contract.get("dependencies") != dependencies:
            refuse("Image dependencies differ from the complete CUDA lock.")
        _cuda_receipt_hashes(contract, dependencies)
        if not runtime:
            return _client_versions(dependencies)
        _cuda_platform()
        actual = locked_versions(lock, accelerator="cuda")
        if contract != json.loads(IMAGE_CONTRACT.read_text()):
            refuse("The running image differs from the confirmed image receipt.")
        if contract["cuda_wheels"] != _cuda_wheel_hashes(dependencies):
            refuse("Installed CUDA wheel hashes drifted; rebuild the image.")
        if contract["native_files"] != {
            name: sha256(path) for name, path in native_files().items()
        }:
            refuse("Native accounting binary drifted; rebuild the image.")
        _cuda_image()
        return actual
    actual = locked_versions(root / DEPLOY_PATH / "requirements.lock")

    def normalized(versions: dict) -> dict:
        return {
            key: value.split("+", 1)[0] if key == "torch" else value
            for key, value in versions.items()
        }

    if normalized(actual) != normalized(contract["dependencies"]):
        refuse(
            "Client/image transitive dependencies differ. Reinstall the same lock; "
            "do not submit using an independently resolved environment."
        )
    if runtime:
        if contract != json.loads(IMAGE_CONTRACT.read_text()):
            refuse("The running image differs from the confirmed image receipt.")
        if contract["native_files"] != {
            name: sha256(path) for name, path in native_files().items()
        }:
            refuse("Native accounting binary drifted; rebuild the image.")
    return actual


def overlay_workspace(root: Path) -> str:
    """Use archived Python sources, retaining only the attested image's Rust binary."""
    for name, binary in native_files().items():
        target = root / "packages/opaque-accounting/src" / name
        if target.exists():
            refuse("A native binary unexpectedly arrived in the source archive.")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(binary, target)
    return ":".join([str(root), *(str(root / p / "src") for p in package_paths(root))])


def main() -> None:
    """Write a fresh image receipt inside the builder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accelerator", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    with args.output.open("x") as stream:
        json.dump(
            build_contract(args.root, accelerator=args.accelerator), stream, indent=2
        )
        stream.write("\n")


if __name__ == "__main__":
    main()
