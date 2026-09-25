"""Extend the reviewed CPU lock using public metadata, never private credentials.

Run with uv 0.7.2: python -m deploy.zenml.moe_iceland.resolve_gpu_lock.
Existing GPU pins constrain regeneration; --check compares without writing.
The authenticated jb-mlops hashes and every shared pin remain those of the CPU
lock. Only public CUDA/TRL additions are resolved, on Linux amd64 / Python 3.12.
"""

import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from .errors import refuse
from .provenance import CUDA_TORCH_SHA256, _cuda_dependencies
from .settings import DEPLOY_PATH
from .source import sha256


def entries(text: str) -> dict:
    """Select locked Linux requirements, retaining authenticated artifact hashes."""
    environment = {
        **default_environment(),
        "os_name": "posix",
        "sys_platform": "linux",
        "platform_system": "Linux",
        "platform_machine": "x86_64",
        "platform_python_implementation": "CPython",
        "implementation_name": "cpython",
        "implementation_version": "3.12.10",
        "python_version": "3.12",
        "python_full_version": "3.12.10",
    }
    result = {}
    for line in text.replace("\\\n", " ").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirement_text, *hashes = re.split(r"\s+--hash=", line)
        if not hashes or any(
            not re.fullmatch(r"sha256:[0-9a-f]{64}", value.strip()) for value in hashes
        ):
            refuse("Expected only requirements with SHA256 hashes in the input lock.")
        requirement = Requirement(requirement_text)
        name = canonicalize_name(requirement.name)
        if requirement.marker and not requirement.marker.evaluate(environment):
            continue
        if name == "torch" and requirement.url:
            continue
        if (
            requirement.url
            or requirement.extras
            or not re.fullmatch(r"==[a-zA-Z0-9.!+-]+", str(requirement.specifier))
            or name in result
        ):
            refuse("Expected unique exact dependency pins in the input lock.")
        result[name] = (
            str(requirement.specifier)[2:],
            {value.strip() for value in hashes},
        )
    return result


def resolve(root: Path) -> str:
    """Resolve public additions with the CPU closure pinned and no private index."""
    deploy = root / DEPLOY_PATH
    cpu_lock = deploy / "requirements.lock"
    inherited = entries(cpu_lock.read_text())
    if inherited.get("jb-mlops", (None,))[0] != "0.0.45":
        refuse("The reviewed CPU lock must supply jb-mlops==0.0.45 and its hashes.")
    requirements = [
        f"{name}=={version}"
        for name, (version, _) in inherited.items()
        if name not in {"jb-mlops", "torch"}
    ]
    for line in (deploy / "requirements-gpu.in").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        requirement = Requirement(line)
        if (
            requirement.url
            or requirement.marker
            or not re.fullmatch(r"==[a-zA-Z0-9.!+-]+", str(requirement.specifier))
        ):
            refuse("GPU inputs must use exact index pins without URLs or markers.")
        if canonicalize_name(requirement.name) == "jb-mlops":
            if str(requirement.specifier) != "==0.0.45" or requirement.extras != {
                "zenml"
            }:
                refuse("Preserve the reviewed jb-mlops[zenml] pin.")
        elif canonicalize_name(requirement.name) == "torch":
            # Request the Iceland-compatible CUDA 12.6 build explicitly.
            requirements.append("torch==2.14.0+cu126")
        else:
            requirements.append(str(requirement))
    command = [
        "uv",
        "--no-config",
        "pip",
        "compile",
        "-",
        "--python-version",
        "3.12.10",
        "--python-platform",
        "x86_64-manylinux_2_28",
        "--index-url",
        "https://pypi.org/simple",
        "--extra-index-url",
        "https://download.pytorch.org/whl/cu126",
        "--index-strategy",
        "unsafe-best-match",
        "--keyring-provider",
        "disabled",
        "--only-binary",
        ":all:",
        "--no-binary",
        "kfp-server-api",
        "--generate-hashes",
        "--no-header",
        "--no-annotate",
        "--no-sources",
    ]
    previous = deploy / "requirements-gpu.lock"
    # Do not constrain against a previous CUDA-13 lock when retargeting the runtime.
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("UV_", "PIP_"))
    }
    try:
        completed = subprocess.run(
            command,
            input="\n".join(requirements) + "\n",
            text=True,
            capture_output=True,
            check=True,
            timeout=300,
            env=env,
            cwd=root,
        )
    except subprocess.CalledProcessError as error:
        refuse("Public CUDA dependency resolution failed:\n" + error.stderr)
    resolved = entries(completed.stdout)
    if f"sha256:{CUDA_TORCH_SHA256}" not in resolved["torch"][1]:
        refuse(
            "The cu126 index did not return the verified Linux amd64 CUDA torch wheel."
        )
    resolved["torch"] = ("2.14.0+cu126", {f"sha256:{CUDA_TORCH_SHA256}"})
    for name, (version, hashes) in inherited.items():
        if name == "torch":
            continue
        if name != "jb-mlops" and resolved.get(name, (None,))[0] != version:
            refuse(f"Public resolution changed an inherited dependency: {name}")
        resolved[name] = (version, hashes)
    lines = [
        "# Linux amd64 / CPython 3.12.10; generated by resolve_gpu_lock.py with uv 0.7.2.",
        f"# Inherited CPU lock SHA256: {sha256(cpu_lock)}",
        "# Private jb-mlops hashes are reused, not fetched from a public index.",
    ]
    for name, (version, hashes) in sorted(resolved.items()):
        lines.append(
            " \\\n    ".join(
                [f"{name}=={version}", *(f"--hash={value}" for value in sorted(hashes))]
            )
        )
    return "\n".join(lines) + "\n"


def build_requirements(lock: str) -> str:
    """Bootstrap the pinned backends before building the source-only KFP dependency."""
    locked = entries(lock)
    lines = [
        "# Build backends from requirements-gpu.lock; install with --require-hashes."
    ]
    for name in ("maturin", "packaging", "setuptools", "setuptools-scm", "wheel"):
        version, hashes = locked[name]
        lines.append(
            " \\\n    ".join(
                [f"{name}=={version}", *(f"--hash={value}" for value in sorted(hashes))]
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Regenerate both CUDA locks without installing anything into the client."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if subprocess.check_output(["uv", "--version"], text=True).split()[1] != "0.7.2":
        refuse("Resolve the reviewed CUDA lock with uv 0.7.2.")
    root = args.root.resolve()
    text = resolve(root)
    outputs = {
        "requirements-gpu.lock": text,
        "requirements-gpu-build.txt": build_requirements(text),
    }
    for name, contents in outputs.items():
        output = root / DEPLOY_PATH / name
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix="requirements-gpu-",
            suffix=".lock",
            dir=output.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(contents)
        try:
            if name == "requirements-gpu.lock":
                _cuda_dependencies(temporary)
            if args.check:
                if output.read_text() != contents:
                    refuse(f"CUDA build input differs from public resolution: {name}")
            else:
                temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Verified CUDA hash lock: {len(entries(text))} dependencies.")


if __name__ == "__main__":
    main()
