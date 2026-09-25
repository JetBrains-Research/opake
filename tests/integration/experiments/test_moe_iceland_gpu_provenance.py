"""Offline CUDA receipts: no image builds, GPU probes, or service connections."""

import base64
import builtins
import copy
import csv
import hashlib
import importlib.metadata as metadata
import json
import re
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from deploy.zenml.moe_iceland import (
    build_workspace,
    provenance,
    resolve_gpu_lock,
    source,
)
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[3]
CPU_INPUTS = (
    "requirements.in",
    "requirements-public.in",
    "requirements.lock",
    "requirements-client.txt",
    "requirements-runtime.txt",
    "Dockerfile.cpu",
    "provenance.py",
    "build_workspace.py",
)
GPU_INPUTS = (
    "requirements-gpu.in",
    "requirements-gpu.lock",
    "requirements-gpu-runtime.txt",
    "requirements-gpu-build.txt",
    "Dockerfile.cuda",
    "resolve_gpu_lock.py",
)
GPU_VERSIONS = {
    **provenance.PINS,
    "trl": "1.13.0",
    "cuda-toolkit": "13.0.3.0",
    "cuda-bindings": "13.4.1",
    "triton": "3.8.0",
    "nvidia-cuda-runtime": "13.0.96",
    "google-cloud-container": "2.61.0",
    "google-cloud-artifact-registry": "1.21.0",
    "google-cloud-storage": "2.19.0",
    "fixture-transport": "1.2.3",
}
TORCH_HASH = "fecffb58f51fd643d213acd68da21cc3fc19bea05a3bc64b4ee55128f47a4963"
NATIVE = "opaque/api/accounting/core/opaque_accounting.abi3.so"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny_network(*args, **kwargs):
        raise AssertionError("Provenance tests must never contact a service.")

    monkeypatch.setattr(socket.socket, "connect", deny_network)


def write_lock(path, versions):
    path.write_text(
        "".join(
            f"{name}=={version} --hash=sha256:"
            f"{TORCH_HASH if name == 'torch' else 'a' * 64}\n"
            for name, version in sorted(versions.items())
        )
    )


def manifest(root):
    (root / source.MANIFEST).write_text(json.dumps(source.inventory(root)))


@pytest.fixture
def staged(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    for name in source.ROOT_FILES:
        (root / name).write_text("fixture\n")
    (root / "pyproject.toml").write_text(
        '[tool.uv.workspace]\nmembers = ["packages/opaque-accounting", '
        '"packages/opaque-base"]\n'
    )
    for name in (
        "examples/moe_privacy/run.py",
        "packages/opaque-base/src/opaque/serialization/__init__.py",
        "packages/opaque-accounting/src/lib.rs",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# source fixture\n")
    deploy = root / source.DEPLOY_PATH
    deploy.mkdir(parents=True)
    for name in {*CPU_INPUTS, *GPU_INPUTS, "source.py"}:
        (deploy / name).write_text(f"# {name}\n")
    write_lock(deploy / "requirements-gpu.lock", GPU_VERSIONS)
    write_lock(deploy / "requirements.lock", provenance.PINS)
    manifest(root)
    return root


def install_distribution(site, name, version, *, requires=(), files=None):
    info = site / f"{name.replace('-', '_')}-{version}.dist-info"
    info.mkdir(parents=True)
    text = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    text += "".join(f"Requires-Dist: {requirement}\n" for requirement in requires)
    contents = {str(info.relative_to(site) / "METADATA"): text.encode()}
    contents.update(files or {f"{name.replace('-', '_')}/__init__.py": b"VALUE = 1\n"})
    rows = []
    for name, data in contents.items():
        path = site / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode()
        rows.append((name, f"sha256={digest.rstrip('=')}", str(len(data))))
    record = info / "RECORD"
    rows.append((str(record.relative_to(site)), "", ""))
    with record.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    return metadata.PathDistribution(info)


@pytest.fixture
def installed(tmp_path, monkeypatch):
    site = tmp_path / "venv/lib/python3.12/site-packages"
    site.mkdir(parents=True)
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.setattr(
        provenance.sysconfig,
        "get_path",
        lambda name: str(tmp_path / "venv/bin") if name == "scripts" else str(site),
    )
    for name, version in GPU_VERSIONS.items():
        requirements = (
            ('fixture-transport==1.2.3; extra == "zenml"',)
            if name == "jb-mlops"
            else ()
        )
        files = None
        if name in {"torch", "nvidia-cuda-runtime", "triton"}:
            module = name.replace("-", "_")
            files = {
                f"{module}/__init__.py": b"VALUE = 1\n",
                f"{module}/lib/runtime.so": b"CUDA binary fixture\n",
            }
        install_distribution(site, name, version, requires=requirements, files=files)
    install_distribution(
        site, "opaque-accounting", "0.0.0.dev0", files={NATIVE: b"native Rust fixture"}
    )
    return site


def host(monkeypatch, *, system="linux", machine="x86_64", cuda="13.0"):
    monkeypatch.setattr(provenance.sys, "platform", system)
    monkeypatch.setattr(provenance.sys, "version_info", (3, 12, 10))
    monkeypatch.setattr(provenance.platform, "machine", lambda: machine)
    monkeypatch.setattr(provenance.platform, "python_version", lambda: "3.12.10")
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(version=SimpleNamespace(cuda=cuda))
    )


def no_training_imports(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "trl", "transformers", "peft"}
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


@pytest.mark.parametrize(
    "api",
    [
        "dependency_fingerprint",
        "build_contract",
        "locked_versions",
        "validate_contract",
    ],
)
def test_unknown_accelerator_fails_closed(api, staged):
    args = (staged, {}) if api == "validate_contract" else (staged,)
    with pytest.raises(ValueError, match="accelerator"):
        getattr(provenance, api)(*args, accelerator="auto")


def test_cpu_fingerprint_keeps_its_original_inputs(staged):
    deploy = staged / source.DEPLOY_PATH
    expected = source.digest_json(
        {name: source.sha256(deploy / name) for name in CPU_INPUTS}
    )
    assert provenance.dependency_fingerprint(staged) == expected
    assert provenance.dependency_fingerprint(staged, accelerator="cpu") == expected
    (deploy / "requirements-gpu.lock").write_text("changed CUDA lock\n")
    assert provenance.dependency_fingerprint(staged) == expected


@pytest.mark.parametrize("name", GPU_INPUTS)
def test_each_gpu_input_is_fingerprinted(staged, name):
    before = provenance.dependency_fingerprint(staged, accelerator="cuda")
    (staged / source.DEPLOY_PATH / name).write_text("changed build input\n")
    assert provenance.dependency_fingerprint(staged, accelerator="cuda") != before


def test_cuda_receipt_build_and_runtime_hash_roundtrip(
    staged, installed, monkeypatch, tmp_path
):
    host(monkeypatch)
    contract = provenance.build_contract(staged, accelerator="cuda")
    assert contract["accelerator"] == "cuda"
    assert contract["dependencies"] == GPU_VERSIONS
    assert contract["torch_cuda"] == "13.0"
    assert contract["native_files"] == {NATIVE: source.sha256(installed / NATIVE)}
    assert set(contract["cuda_wheels"]) == {
        "torch",
        "triton",
        "cuda-toolkit",
        "cuda-bindings",
        "nvidia-cuda-runtime",
    }
    receipt = tmp_path / "image.json"
    receipt.write_text(json.dumps(contract))
    monkeypatch.setattr(provenance, "IMAGE_CONTRACT", receipt)
    assert (
        provenance.validate_contract(staged, contract, runtime=True, accelerator="cuda")
        == GPU_VERSIONS
    )


def test_mac_checks_orchestration_and_full_recorded_lock_separately(
    staged, installed, monkeypatch
):
    host(monkeypatch)
    contract = provenance.build_contract(staged, accelerator="cuda")
    host(monkeypatch, system="darwin", machine="arm64", cuda=None)
    no_training_imports(monkeypatch)
    original = provenance.metadata.version

    def client_version(name):
        assert name not in {
            "torch",
            "trl",
            "peft",
            "transformers",
            *contract["cuda_wheels"],
        }
        return original(name)

    monkeypatch.setattr(provenance.metadata, "version", client_version)
    actual = provenance.validate_contract(staged, contract, accelerator="cuda")
    assert actual["zenml"] == "0.96.4"
    assert actual["fixture-transport"] == "1.2.3"
    assert "torch" not in actual
    assert (
        provenance.locked_versions(
            staged / source.DEPLOY_PATH / "requirements-gpu.lock", accelerator="cuda"
        )
        == actual
    )


@pytest.mark.parametrize("change", ["missing", "version", "extra", "cpu-torch"])
def test_cuda_client_rejects_incomplete_or_drifted_image_dependencies(
    staged, installed, monkeypatch, change
):
    host(monkeypatch)
    contract = provenance.build_contract(staged, accelerator="cuda")
    if change == "missing":
        del contract["dependencies"]["nvidia-cuda-runtime"]
    elif change == "version":
        contract["dependencies"]["trl"] = "1.12.0"
    elif change == "extra":
        contract["dependencies"]["unreviewed"] = "1.0"
    else:
        contract["dependencies"]["torch"] = "2.14.0+cpu"
    host(monkeypatch, system="darwin", machine="arm64")
    no_training_imports(monkeypatch)
    with pytest.raises(ValueError, match=r"dependencies|lock"):
        provenance.validate_contract(staged, contract, accelerator="cuda")


@pytest.mark.parametrize(
    "path",
    [
        "torch/__init__.py",
        "torch/lib/runtime.so",
        "nvidia_cuda_runtime/lib/runtime.so",
        "triton/lib/runtime.so",
        NATIVE,
    ],
)
def test_cuda_runtime_rejects_changed_wheel_or_native_bytes(
    staged, installed, monkeypatch, tmp_path, path
):
    host(monkeypatch)
    contract = provenance.build_contract(staged, accelerator="cuda")
    receipt = tmp_path / "image.json"
    receipt.write_text(json.dumps(contract))
    monkeypatch.setattr(provenance, "IMAGE_CONTRACT", receipt)
    (installed / path).write_bytes(b"modified binary bytes")
    no_training_imports(monkeypatch)
    with pytest.raises(ValueError, match=r"drift|hash|RECORD"):
        provenance.validate_contract(staged, contract, runtime=True, accelerator="cuda")


@pytest.mark.parametrize(
    ("system", "machine", "cuda"),
    [
        ("darwin", "arm64", None),
        ("linux", "aarch64", "13.0"),
        ("linux", "x86_64", None),
        ("linux", "x86_64", "12.8"),
    ],
)
def test_cuda_build_rejects_wrong_platform_or_cpu_wheel(
    staged, installed, monkeypatch, system, machine, cuda
):
    host(monkeypatch, system=system, machine=machine, cuda=cuda)
    with pytest.raises(ValueError, match=r"CUDA|Linux amd64"):
        provenance.build_contract(staged, accelerator="cuda")


@pytest.mark.parametrize(
    "line",
    [
        "trl==1.13.0",
        "--index-url https://invalid.example/simple",
        "trl>=1.13.0 --hash=sha256:" + "a" * 64,
        "trl @ https://invalid.example/trl.whl --hash=sha256:" + "a" * 64,
    ],
)
def test_cuda_lock_rejects_unhashed_unpinned_and_url_inputs(
    tmp_path, monkeypatch, line
):
    host(monkeypatch)
    lock = tmp_path / "requirements-gpu.lock"
    lock.write_text(line + "\n")
    with pytest.raises(ValueError, match=r"CUDA (hash )?lock"):
        provenance.locked_versions(lock, accelerator="cuda")


def test_gpu_inputs_are_staged_without_broadening_other_files(staged):
    deploy = staged / source.DEPLOY_PATH
    for name in (
        "credentials.toml",
        "requirements-unreviewed.txt",
        "Dockerfile.other",
        "gpu-token.txt",
        "receipt.json",
    ):
        (deploy / name).write_text("not source\n")
    files = source.source_files(staged)
    assert {f"{source.DEPLOY_PATH}/{name}" for name in GPU_INPUTS} <= files.keys()
    assert not any(
        name.endswith(
            (
                "credentials.toml",
                "requirements-unreviewed.txt",
                "Dockerfile.other",
                "gpu-token.txt",
                "receipt.json",
            )
        )
        for name in files
    )
    assert "packages/opaque-accounting/src/lib.rs" in files


def test_gpu_build_uses_pr_workspace_and_locked_native_build(staged, monkeypatch):
    calls = []
    monkeypatch.chdir(staged)
    monkeypatch.setattr(
        build_workspace.subprocess,
        "run",
        lambda args, **kwargs: calls.append((args, kwargs)),
    )
    build_workspace.main()
    native_command, native_options = calls[0]
    assert native_command[-1] == str(staged / "packages/opaque-accounting")
    assert "build-args=--locked" in native_command
    assert {"--no-deps", "--no-build-isolation", "--no-sources"} <= set(native_command)
    assert native_options["env"]["SETUPTOOLS_SCM_PRETEND_VERSION"] == "0.0.0.dev0"
    assert calls[1][0][-1] == str(staged / "packages/opaque-base")
    assert calls[2][0][:3] == ["uv", "pip", "check"]


def test_legacy_cpu_contract_still_roundtrips(staged, tmp_path, monkeypatch):
    host(monkeypatch, cuda=None)
    site = tmp_path / "cpu-site"
    for name, version in provenance.PINS.items():
        install_distribution(site, name, version)
    install_distribution(
        site, "opaque-accounting", "0.0.0.dev0", files={NATIVE: b"CPU native"}
    )
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.setattr(provenance.sysconfig, "get_path", lambda name: str(site))
    contract = provenance.build_contract(staged)
    assert contract["schema"] == 1
    assert "accelerator" not in contract
    assert "cuda_wheels" not in contract
    assert provenance.build_contract(staged, accelerator="cpu") == contract
    receipt = tmp_path / "cpu-image.json"
    receipt.write_text(json.dumps(contract))
    monkeypatch.setattr(provenance, "IMAGE_CONTRACT", receipt)
    assert (
        provenance.validate_contract(staged, contract, runtime=True) == provenance.PINS
    )
    assert (
        provenance.validate_contract(staged, contract, accelerator="cpu")
        == provenance.PINS
    )
    host(monkeypatch, system="darwin", machine="arm64", cuda=None)
    contract["dependencies"]["torch"] += "+cpu"
    assert provenance.validate_contract(staged, contract) == provenance.PINS


@pytest.mark.parametrize(
    ("recorded", "requested"), [("cuda", "cpu"), (None, "cuda"), ("cpu", "cuda")]
)
def test_cpu_and_cuda_receipts_are_not_interchangeable(staged, recorded, requested):
    contract = {} if recorded is None else {"accelerator": recorded}
    with pytest.raises(ValueError, match="accelerator"):
        provenance.validate_contract(staged, contract, accelerator=requested)


@pytest.mark.parametrize(
    ("system", "machine"), [("darwin", "arm64"), ("linux", "x86_64")]
)
def test_client_does_not_require_installed_training_packages(
    staged, installed, monkeypatch, system, machine
):
    host(monkeypatch)
    contract = provenance.build_contract(staged, accelerator="cuda")
    host(monkeypatch, system=system, machine=machine, cuda=None)
    no_training_imports(monkeypatch)
    original = provenance.metadata.version

    def client_version(name):
        if name in {"torch", "trl", "transformers", "peft", *contract["cuda_wheels"]}:
            raise metadata.PackageNotFoundError(name)
        return original(name)

    monkeypatch.setattr(provenance.metadata, "version", client_version)
    assert (
        provenance.validate_contract(staged, contract, accelerator="cuda")["jb-mlops"]
        == "0.0.45"
    )


@pytest.mark.parametrize("name", ["zenml", "jb-mlops", "fixture-transport"])
@pytest.mark.parametrize("missing", [False, True])
def test_client_rejects_orchestration_and_extra_dependency_drift(
    staged, installed, monkeypatch, name, missing
):
    host(monkeypatch)
    contract = provenance.build_contract(staged, accelerator="cuda")
    host(monkeypatch, system="darwin", machine="arm64")
    original = provenance.metadata.version

    def client_version(dependency):
        if dependency == name:
            if missing:
                raise metadata.PackageNotFoundError(name)
            return "0.0.1"
        return original(dependency)

    monkeypatch.setattr(provenance.metadata, "version", client_version)
    with pytest.raises(ValueError, match=re.escape(name)):
        provenance.validate_contract(staged, contract, accelerator="cuda")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", 1),
        ("platform", "linux/arm64"),
        ("torch_cuda", "12.8"),
        ("workspace_sha256", "0" * 64),
        ("uv_lock_sha256", "0" * 64),
        ("dependency_sha256", "0" * 64),
        ("requirements_lock_sha256", "0" * 64),
        ("parent_revision", "0" * 40),
        ("python", "3.12.9"),
    ],
)
def test_cuda_receipt_rejects_contract_field_drift(
    staged, installed, monkeypatch, field, value
):
    host(monkeypatch)
    contract = provenance.build_contract(staged, accelerator="cuda")
    contract[field] = value
    with pytest.raises(ValueError, match=r"mismatch|Python patch"):
        provenance.validate_contract(staged, contract, accelerator="cuda")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cuda_wheels", {}),
        ("cuda_wheels", None),
        ("native_files", {}),
        ("native_files", {NATIVE: "invalid"}),
        ("native_files", {"../native.so": "a" * 64}),
        ("native_files", {NATIVE: "a" * 64, "other": "a" * 64}),
    ],
)
def test_cuda_client_requires_complete_well_formed_binary_attestations(
    staged, installed, monkeypatch, field, value
):
    host(monkeypatch)
    contract = provenance.build_contract(staged, accelerator="cuda")
    contract[field] = value
    with pytest.raises(ValueError, match="hashes"):
        provenance.validate_contract(staged, contract, accelerator="cuda")


def test_cuda_runtime_rejects_different_on_image_receipt(
    staged, installed, monkeypatch, tmp_path
):
    host(monkeypatch)
    contract = provenance.build_contract(staged, accelerator="cuda")
    recorded = copy.deepcopy(contract)
    recorded["cuda_wheels"]["torch"] = "0" * 64
    receipt = tmp_path / "image.json"
    receipt.write_text(json.dumps(recorded))
    monkeypatch.setattr(provenance, "IMAGE_CONTRACT", receipt)
    with pytest.raises(ValueError, match="running image differs"):
        provenance.validate_contract(staged, contract, runtime=True, accelerator="cuda")


def test_modified_wheel_and_updated_record_cannot_bypass_receipt(
    staged, installed, monkeypatch, tmp_path
):
    host(monkeypatch)
    contract = provenance.build_contract(staged, accelerator="cuda")
    receipt = tmp_path / "image.json"
    receipt.write_text(json.dumps(contract))
    monkeypatch.setattr(provenance, "IMAGE_CONTRACT", receipt)
    path = installed / "torch/lib/runtime.so"
    path.write_bytes(b"different CUDA binary")
    record = installed / "torch-2.14.0.dist-info/RECORD"
    with record.open(newline="") as stream:
        rows = list(csv.reader(stream))
    for row in rows:
        if row[0] == "torch/lib/runtime.so":
            digest = (
                base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest())
                .decode()
                .rstrip("=")
            )
            row[1:] = [f"sha256={digest}", str(path.stat().st_size)]
    with record.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    with pytest.raises(ValueError, match="CUDA wheel hashes drifted"):
        provenance.validate_contract(staged, contract, runtime=True, accelerator="cuda")


@pytest.mark.parametrize(
    "change", ["missing-record", "missing-file", "missing-hash", "symlink", "escape"]
)
def test_cuda_build_refuses_unverifiable_wheel_contents(
    staged, installed, monkeypatch, change
):
    host(monkeypatch)
    record = installed / "torch-2.14.0.dist-info/RECORD"
    path = installed / "torch/lib/runtime.so"
    if change == "missing-record":
        record.unlink()
    elif change == "missing-file":
        path.unlink()
    elif change == "symlink":
        path.unlink()
        path.symlink_to(installed / "triton/lib/runtime.so")
    else:
        with record.open(newline="") as stream:
            rows = list(csv.reader(stream))
        if change == "missing-hash":
            for row in rows:
                if row[0] == "torch/lib/runtime.so":
                    row[1] = ""
        else:
            rows.append(("../../../../outside.so", "", ""))
        with record.open("w", newline="") as stream:
            csv.writer(stream).writerows(rows)
    with pytest.raises(ValueError, match=r"RECORD|CUDA wheel file"):
        provenance.build_contract(staged, accelerator="cuda")


def test_cuda_lock_rejects_extra_runtime_installations(staged, installed, monkeypatch):
    host(monkeypatch)
    install_distribution(installed, "unreviewed-package", "1.0")
    with pytest.raises(ValueError, match="outside the CUDA lock"):
        provenance.locked_versions(
            staged / source.DEPLOY_PATH / "requirements-gpu.lock", accelerator="cuda"
        )


@pytest.mark.parametrize(
    "change",
    [
        "duplicate",
        "marker",
        "wrong-torch-hash",
        "missing-trl",
        "published-workspace",
        "missing-cuda",
        "invalid-hash",
    ],
)
def test_cuda_lock_fails_closed_before_installation_checks(staged, monkeypatch, change):
    lock = staged / source.DEPLOY_PATH / "requirements-gpu.lock"
    text = lock.read_text()
    if change == "duplicate":
        text += "jb_mlops==0.0.45 --hash=sha256:" + "a" * 64 + "\n"
    elif change == "marker":
        text = text.replace("trl==1.13.0", "trl==1.13.0 ; sys_platform == 'linux'")
    elif change == "wrong-torch-hash":
        text = text.replace(TORCH_HASH, "b" * 64)
    elif change == "missing-trl":
        text = "\n".join(
            line for line in text.splitlines() if not line.startswith("trl==")
        )
    elif change == "published-workspace":
        text += "opaque==0.1.0 --hash=sha256:" + "a" * 64 + "\n"
    elif change == "missing-cuda":
        text = "\n".join(
            line for line in text.splitlines() if not line.startswith("cuda-toolkit==")
        )
    else:
        text = text.replace("--hash=sha256:" + "a" * 64, "--hash=sha256:bad", 1)
    lock.write_text(text)
    host(monkeypatch, system="darwin", machine="arm64")
    no_training_imports(monkeypatch)
    with pytest.raises(ValueError, match=r"lock|workspace"):
        provenance.locked_versions(lock, accelerator="cuda")


@pytest.mark.parametrize("name", GPU_INPUTS)
def test_gpu_input_symlinks_remain_forbidden(staged, name):
    path = staged / source.DEPLOY_PATH / name
    path.unlink()
    path.symlink_to(staged / "uv.lock")
    with pytest.raises(ValueError, match="Symlink"):
        source.source_files(staged)


@pytest.mark.parametrize("cap", ["SOURCE_LIMIT", "SOURCE_FILE_LIMIT"])
def test_gpu_inputs_cannot_bypass_source_caps(staged, monkeypatch, cap):
    monkeypatch.setattr(source, cap, 1)
    with pytest.raises(ValueError, match=r"Source exceeds|source exceeds"):
        source.source_files(staged)


def test_native_binaries_and_excluded_trees_still_stay_out_of_source(staged):
    deploy = staged / source.DEPLOY_PATH
    for directory in (".venv", "target", "source-stages", "receipts"):
        path = deploy / directory / "requirements-gpu.lock"
        path.parent.mkdir()
        path.write_text("not source\n")
    native = staged / "packages/opaque-accounting/src" / NATIVE
    native.parent.mkdir(parents=True)
    native.write_bytes(b"compiled Rust is not source")
    files = source.source_files(staged)
    assert not any(source.EXCLUDED.intersection(Path(name).parts) for name in files)
    assert native.relative_to(staged).as_posix() not in files


def test_checked_in_gpu_lock_preserves_pins_private_hashes_and_build_backends():
    deploy = ROOT / source.DEPLOY_PATH
    cpu = resolve_gpu_lock.entries((deploy / "requirements.lock").read_text())
    text = (deploy / "requirements-gpu.lock").read_text()
    gpu = resolve_gpu_lock.entries(text)
    assert cpu.items() <= gpu.items()
    assert gpu["torch"] == ("2.14.0", {f"sha256:{TORCH_HASH}"})
    assert (
        "sha256:7c7ae0c72e969450221cf4c7cfe69c532c5a83c727204f1c8ced38a9b79134ec"
        in gpu["trl"][1]
    )
    assert (
        provenance._cuda_dependencies(deploy / "requirements-gpu.lock")["trl"]
        == "1.13.0"
    )
    for line in (deploy / "requirements-gpu.in").read_text().splitlines():
        if line and not line.startswith("#"):
            requirement = Requirement(line)
            assert gpu[canonicalize_name(requirement.name)][0] in requirement.specifier
    bootstrap = (deploy / "requirements-gpu-build.txt").read_text()
    assert bootstrap == resolve_gpu_lock.build_requirements(text)
    assert resolve_gpu_lock.entries(bootstrap).items() <= gpu.items()
    assert not {"opake", "opaque"}.intersection(gpu)
    assert not any(name.startswith(("opake-", "opaque-")) for name in gpu)


def test_regeneration_uses_public_index_without_private_input_or_environment(
    monkeypatch,
):
    deploy = ROOT / source.DEPLOY_PATH
    monkeypatch.setenv("UV_INDEX", "https://invalid.example/simple")
    monkeypatch.setenv("PIP_EXTRA_INDEX_URL", "https://invalid.example/simple")
    calls = []

    def resolver(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=(deploy / "requirements-gpu.lock").read_text())

    monkeypatch.setattr(resolve_gpu_lock.subprocess, "run", resolver)
    assert (
        resolve_gpu_lock.resolve(ROOT) == (deploy / "requirements-gpu.lock").read_text()
    )
    command, options = calls[0]
    assert command[command.index("--index-url") + 1] == "https://pypi.org/simple"
    assert command[command.index("--python-platform") + 1] == "x86_64-manylinux_2_28"
    assert "--no-config" in command
    assert "jb-mlops" not in options["input"]
    assert not any(key.startswith(("UV_", "PIP_")) for key in options["env"])


def test_cuda_recipe_has_pinned_bases_and_only_copies_installed_artifacts():
    recipe = (ROOT / source.DEPLOY_PATH / "Dockerfile.cuda").read_text()
    instructions = [
        line.strip()
        for line in recipe.replace("\\\n", " ").splitlines()
        if line and not line.startswith("#")
    ]
    bases = [line.split()[2] for line in instructions if line.startswith("FROM ")]
    assert len(bases) == 3
    assert all(re.search(r"@sha256:[0-9a-f]{64}$", image) for image in bases)
    assert bases[1] == bases[2]
    final = instructions[
        max(
            index for index, line in enumerate(instructions) if line.startswith("FROM ")
        ) :
    ]
    copies = [line for line in final if line.startswith("COPY ")]
    assert copies == [
        "COPY --from=builder /opt/venv /opt/venv",
        "COPY --from=builder /opt/moe-image.json /opt/moe-image.json",
    ]
    assert "USER 1000:1000" in final


def test_receipt_cli_passes_accelerator_without_loading_training(monkeypatch, tmp_path):
    output = tmp_path / "image.json"
    calls = []

    def record(root, *, accelerator="cpu"):
        calls.append((root, accelerator))
        return {"schema": 2, "accelerator": accelerator}

    no_training_imports(monkeypatch)
    monkeypatch.setattr(provenance, "build_contract", record)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "provenance",
            "--root",
            str(tmp_path),
            "--output",
            str(output),
            "--accelerator",
            "cuda",
        ],
    )
    provenance.main()
    assert calls == [(tmp_path, "cuda")]
    assert json.loads(output.read_text())["accelerator"] == "cuda"
    with pytest.raises(FileExistsError):
        provenance.main()


@pytest.mark.parametrize("conflict", [False, True])
def test_client_checks_repeated_constraints_and_handles_dependency_cycles(
    staged, installed, monkeypatch, conflict
):
    host(monkeypatch)
    contract = provenance.build_contract(staged, accelerator="cuda")
    host(monkeypatch, system="darwin", machine="arm64")
    original = provenance.metadata.requires

    def requirements(name):
        if name == "fixture-transport":
            return [
                f"jb-mlops[zenml]=={'0.0.44' if conflict else '0.0.45'}",
                "zenml==0.96.4",
            ]
        return original(name)

    monkeypatch.setattr(provenance.metadata, "requires", requirements)
    if conflict:
        with pytest.raises(ValueError, match="metadata conflicts"):
            provenance.validate_contract(staged, contract, accelerator="cuda")
    else:
        assert (
            provenance.validate_contract(staged, contract, accelerator="cuda")[
                "fixture-transport"
            ]
            == "1.2.3"
        )
