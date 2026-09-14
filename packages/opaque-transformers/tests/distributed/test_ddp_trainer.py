"""DDP end-to-end tests for :class:`Trainer`.

Each test launches N rank workers via ``subprocess.Popen`` (one per rank)
running :mod:`_ddp_runner` with the right ``RANK`` / ``LOCAL_RANK`` /
``WORLD_SIZE`` env vars. We use subprocess rather than ``mp.spawn``
because pytest's ``--import-mode=importlib`` mode renames test modules
in a way the spawned worker can't unpickle.

Markers:
- ``cuda``: skipped automatically on CPU-only machines (top-level conftest).
- ``slow``: launches multiple processes, each pulling a fresh torch import.

Hardware: validated on 4× H100. ``torch.cuda.device_count() < 2`` skips
the multi-rank scenarios.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch

RUNNER = str(Path(__file__).resolve().parent / "_ddp_runner.py")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_ddp(
    scenario: str,
    world_size: int,
    *,
    output_dir: str | None = None,
    backend: str | None = None,
    timeout: float = 240.0,
) -> None:
    """Launch ``world_size`` rank workers and wait for completion.

    Raises on any non-zero exit. Stderr from each rank is captured and
    surfaced on failure.
    """
    port = _free_port()
    procs: list[subprocess.Popen] = []
    common = [
        sys.executable,
        RUNNER,
        "--world-size",
        str(world_size),
        "--port",
        str(port),
        "--scenario",
        scenario,
    ]
    if output_dir is not None:
        common += ["--output-dir", output_dir]
    if backend is not None:
        common += ["--backend", backend]
    env = os.environ.copy()
    # Quiet HF/torch in workers; keep our own stderr pristine.
    env["TRANSFORMERS_VERBOSITY"] = "error"
    try:
        procs = [
            subprocess.Popen(
                [*common, "--rank", str(rank)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for rank in range(world_size)
        ]
        results = []
        for p in procs:
            stdout, stderr = p.communicate(timeout=timeout)
            results.append((p.returncode, stdout.decode(), stderr.decode()))
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
                p.wait()
    failures = [
        (i, rc, out, err) for i, (rc, out, err) in enumerate(results) if rc != 0
    ]
    if failures:
        msg = "\n".join(
            f"rank={i} rc={rc}\nstdout:\n{out}\nstderr:\n{err}"
            for i, rc, out, err in failures
        )
        raise AssertionError(f"DDP scenario {scenario!r} failed:\n{msg}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _skip_if_no_multi_gpu() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires ≥2 CUDA devices")


@pytest.mark.cuda
@pytest.mark.slow
def test_runtime_foundation_rank_world_world2(tmp_path) -> None:
    _skip_if_no_multi_gpu()
    _run_ddp("runtime_foundation", world_size=2, output_dir=str(tmp_path))


@pytest.mark.slow
@pytest.mark.distributed
def test_gloo_runtime_foundation_trains_and_checkpoints(tmp_path) -> None:
    _run_ddp(
        "runtime_foundation",
        world_size=2,
        output_dir=str(tmp_path),
        backend="gloo",
    )


@pytest.mark.cuda
@pytest.mark.slow
def test_runtime_foundation_rank_world_world4(tmp_path) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        pytest.skip("requires ≥4 CUDA devices")
    _run_ddp("runtime_foundation", world_size=4, output_dir=str(tmp_path))


@pytest.mark.cuda
@pytest.mark.slow
def test_per_rank_shards_partition_dataset() -> None:
    _skip_if_no_multi_gpu()
    _run_ddp("per_rank_partition", world_size=2)


@pytest.mark.cuda
@pytest.mark.slow
def test_eval_gather_returns_cluster_wide_predictions(tmp_path) -> None:
    _skip_if_no_multi_gpu()
    _run_ddp("eval_gather", world_size=2, output_dir=str(tmp_path))


@pytest.mark.slow
@pytest.mark.distributed
def test_gloo_eval_gather_matches_reference_for_uneven_shards(tmp_path) -> None:
    _run_ddp(
        "eval_gather",
        world_size=2,
        output_dir=str(tmp_path),
        backend="gloo",
    )


@pytest.mark.slow
@pytest.mark.distributed
def test_gloo_eval_gather_supports_empty_rank_shard(tmp_path) -> None:
    _run_ddp(
        "eval_gather_empty_rank",
        world_size=2,
        output_dir=str(tmp_path),
        backend="gloo",
    )


@pytest.mark.cuda
@pytest.mark.slow
def test_batch_eval_metrics_runs_cluster_wide(tmp_path) -> None:
    _skip_if_no_multi_gpu()
    _run_ddp("batch_eval_metrics", world_size=2, output_dir=str(tmp_path))


@pytest.mark.slow
@pytest.mark.distributed
def test_gloo_rank_gating_and_worker_seed(tmp_path) -> None:
    _run_ddp(
        "rank_gating_and_worker_seed",
        world_size=2,
        output_dir=str(tmp_path),
        backend="gloo",
    )


@pytest.mark.slow
@pytest.mark.distributed
def test_gloo_gather_fastpath_and_fallback() -> None:
    _run_ddp("gather_paths", world_size=2, backend="gloo")


@pytest.mark.slow
@pytest.mark.distributed
def test_gloo_push_to_hub_does_not_deadlock(tmp_path) -> None:
    """Regression for #1004: rank zero must not hang in save_model's barrier."""
    _run_ddp(
        "push_to_hub_no_deadlock",
        world_size=2,
        output_dir=str(tmp_path),
        backend="gloo",
        timeout=60.0,
    )


@pytest.mark.slow
@pytest.mark.distributed
def test_vendor_backend_fails_fast_without_runtime(tmp_path) -> None:
    _run_ddp(
        "env_backend_diagnostic",
        world_size=2,
        output_dir=str(tmp_path),
        backend="gloo",
    )


@pytest.mark.slow
@pytest.mark.distributed
def test_gloo_checkpoint_save_failure_propagates_to_all_ranks(tmp_path) -> None:
    """Regression for issue #1002: a saving-rank ``on_save`` failure must not
    leave the non-saving rank hanging at the checkpoint barrier. On the old
    code, rank 1 hangs there and this times out; the scenario itself asserts
    each rank sees the expected error and exits 0 once fixed.
    """
    _run_ddp(
        "checkpoint_save_failure",
        world_size=2,
        output_dir=str(tmp_path),
        backend="gloo",
        timeout=60,
    )


@pytest.mark.slow
@pytest.mark.distributed
def test_gloo_non_divisible_population_trimmed() -> None:
    """Trainer trims N % world_size != 0 to an equal shard length on every rank."""
    _run_ddp("non_divisible_population_trimmed", world_size=2, backend="gloo")


@pytest.mark.slow
def test_mpi_launcher_smoke_when_available() -> None:
    if not torch.distributed.is_mpi_available():
        pytest.skip("PyTorch was built without MPI backend support")
    mpirun = shutil.which("mpirun") or shutil.which("mpiexec")
    if mpirun is None:
        pytest.skip("mpirun/mpiexec is not available on this host")

    cmd = [
        mpirun,
        "-n",
        "2",
        sys.executable,
        "-c",
        (
            "import torch.distributed as dist;"
            "dist.init_process_group('mpi');"
            "assert dist.get_world_size()==2;"
            "dist.barrier();"
            "dist.destroy_process_group()"
        ),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip(f"MPI launcher/runtime unavailable: {proc.stderr.strip()}")
