"""Shared test fixtures for kernel tests.

Provides pytest fixtures for:
- mellum_config: Mellum-4b model dimensions for realistic testing
- assert_precision: Assert precision using torch.testing.assert_close (atol+rtol)
- measure_time_and_memory: Benchmark execution time and peak CUDA memory
- assert_perf_benefit: Assert performance improvement (and optionally record
  the measured numbers to a JSON acceptance log when ``--record-perf=PATH``
  is passed; default behavior is unchanged when the option is absent).
"""

import gc
import json
import time
from pathlib import Path

import pytest
import torch

MIN_KERNEL_CUDA_MEM_GB = 24


def pytest_collection_modifyitems(config, items):
    """Skip CUDA kernel stress suite when runner GPU memory is insufficient."""
    if not torch.cuda.is_available():
        return

    try:
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except Exception:
        return

    if total_gb >= MIN_KERNEL_CUDA_MEM_GB:
        return

    reason = (
        f"Kernel stress tests require >= {MIN_KERNEL_CUDA_MEM_GB}GB CUDA memory "
        f"(found {total_gb:.2f}GB)."
    )
    skip_marker = pytest.mark.skip(reason=reason)
    # `pytest_collection_modifyitems` receives the WHOLE session's items, not
    # just this directory's — so scope the skip to the kernel stress suite
    # (this conftest's directory). Without this guard a <24GB GPU would skip
    # every CUDA test in the repo, not just the kernel stress tests.
    kernels_dir = Path(__file__).resolve().parent
    for item in items:
        if Path(str(item.fspath)).resolve().is_relative_to(kernels_dir):
            item.add_marker(skip_marker)


# ============================================================================
# Mellum-4b model configuration
# ============================================================================

MELLUM_CONFIG = {
    "batch_size": 4,
    "seq_len": 1024,
    "hidden_dim": 3072,
    "intermediate_dim": 8256,  # FFN intermediate size
    "n_heads": 24,
    "head_dim": 128,
    "vocab_size": 128256,
    "rank": 16,  # LoRA rank
    "vmap_batch": 4,
}


# ============================================================================
# Utility functions
# ============================================================================


def _assert_precision(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float,
    atol: float,
    label: str = "",
) -> dict:
    """Assert precision using combined atol+rtol formula with diagnostic reporting.

    Uses torch.testing.assert_close: |actual - expected| <= atol + rtol * |expected|
    This gracefully handles near-zero values (falls back to atol) and large values
    (uses rtol), avoiding the near-zero blowup of pure max relative error.

    Returns dict with abs_err and rel_err for diagnostic purposes.
    """
    diff = (actual.float() - expected.float()).abs()
    abs_err = diff.max().item()

    # Compute max relative error for diagnostics (not used for assertion)
    expected_abs = expected.float().abs()
    mask = expected_abs >= 1e-6
    if mask.sum() > 0:
        rel_err = (diff[mask] / expected_abs[mask]).max().item()
    else:
        rel_err = float("nan")

    prefix = f"  {label}: " if label else "  "
    print(
        f"{prefix}abs={abs_err:.2e}, rel={rel_err:.2e} (rtol={rtol:.0e}, atol={atol:.0e})"
    )

    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    return {"abs_err": abs_err, "rel_err": rel_err}


def _measure_incremental_cuda_memory(fn, warmup=2):
    """Measure allocations made by ``fn`` above already-resident inputs."""
    for _ in range(warmup):
        result = fn()
        torch.cuda.synchronize()
        del result

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    result = fn()
    torch.cuda.synchronize()
    return result, {
        "allocated_bytes": torch.cuda.memory_allocated() - baseline,
        "peak_bytes": torch.cuda.max_memory_allocated() - baseline,
    }


def _measure_time_and_memory(fn, *args, warmup=3, runs=10):
    """Measure execution time and peak CUDA memory.

    Runs warmup iterations, then timed iterations. Each iteration clones
    inputs and runs forward + backward (if output requires grad).
    """
    for _ in range(warmup):
        inputs = [a.detach().clone().requires_grad_(a.requires_grad) for a in args]
        out = fn(*inputs)
        if isinstance(out, torch.Tensor) and out.requires_grad:
            out.sum().backward()
        torch.cuda.synchronize()

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(runs):
        inputs = [a.detach().clone().requires_grad_(a.requires_grad) for a in args]
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = fn(*inputs)
        if isinstance(out, torch.Tensor) and out.requires_grad:
            out.sum().backward()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    return {
        "time_ms": sum(times) / len(times) * 1000,
        "memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }


def _assert_perf_benefit(pt_stats, op_stats, label="", max_perf_overhead=0.20):
    """Assert that opake kernel shows benefit in speed or memory."""
    speedup = pt_stats["time_ms"] / op_stats["time_ms"]
    mem_reduction = pt_stats["memory_mb"] / op_stats["memory_mb"]

    print(f"\n{label} performance:")
    print(f"  PyTorch: {pt_stats['time_ms']:.2f}ms, {pt_stats['memory_mb']:.1f}MB")
    print(f"  Opake:  {op_stats['time_ms']:.2f}ms, {op_stats['memory_mb']:.1f}MB")
    print(f"  Speedup: {speedup:.2f}x, Memory reduction: {mem_reduction:.2f}x")

    has_benefit = speedup > (1.0 - max_perf_overhead) or mem_reduction > 1.0
    assert has_benefit, (
        f"No benefit for {label}: speedup={speedup:.2f}x, "
        f"mem_reduction={mem_reduction:.2f}x"
    )


# ============================================================================
# Pytest fixtures
# ============================================================================


@pytest.fixture(scope="session")
def mellum_config():
    """Mellum-4b model configuration for realistic testing."""
    return MELLUM_CONFIG


@pytest.fixture(scope="session")
def assert_precision():
    """Assert precision using torch.testing.assert_close with diagnostics."""
    return _assert_precision


@pytest.fixture(scope="session")
def measure_time_and_memory():
    """Benchmark execution time and peak CUDA memory."""
    return _measure_time_and_memory


@pytest.fixture(scope="session")
def measure_incremental_cuda_memory():
    """Measure CUDA allocation and peak growth above resident inputs."""
    return _measure_incremental_cuda_memory


# ============================================================================
# Acceptance recorder (--record-perf)
# ============================================================================

# Per-session list populated by the ``assert_perf_benefit`` fixture when the
# ``--record-perf=PATH`` option is set. Flushed in ``pytest_sessionfinish``.
_PERF_ROWS: list[dict] = []


def pytest_addoption(parser):
    """Register the ``--record-perf=PATH`` CLI option."""
    parser.addoption(
        "--record-perf",
        action="store",
        default=None,
        metavar="PATH",
        help=(
            "Append assert_perf_benefit measurements to a JSON acceptance log "
            "at PATH (one row per call). Default: no recording."
        ),
    )


def _record_perf_row(
    *,
    nodeid: str,
    label: str,
    pt_stats: dict,
    op_stats: dict,
) -> None:
    """Append one acceptance row to the in-memory buffer.

    Kernel and op identifiers are derived from the test ``nodeid`` and the
    ``label`` argument; numeric fields come straight from
    ``measure_time_and_memory`` stats.
    """
    speedup = pt_stats["time_ms"] / op_stats["time_ms"]
    mem_reduction = pt_stats["memory_mb"] / op_stats["memory_mb"]
    device_name = ""
    if torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception:
            device_name = ""
    _PERF_ROWS.append(
        {
            "nodeid": nodeid,
            "label": label,
            "device": device_name,
            "time_pt_ms": pt_stats["time_ms"],
            "time_op_ms": op_stats["time_ms"],
            "mem_pt_mb": pt_stats["memory_mb"],
            "mem_op_mb": op_stats["memory_mb"],
            "speedup": speedup,
            "mem_reduction": mem_reduction,
        }
    )


def pytest_sessionfinish(session, exitstatus):
    """On session end, flush ``_PERF_ROWS`` to ``--record-perf`` if set.

    The output file is a single JSON array; if it already exists the new rows
    are appended to the array so reruns accumulate measurements.
    """
    path = session.config.getoption("--record-perf")
    if not path or not _PERF_ROWS:
        return

    abs_path = Path(path).resolve()
    abs_path.parent.mkdir(parents=True, exist_ok=True)

    existing: list = []
    if abs_path.exists():
        try:
            with abs_path.open() as fh:
                loaded = json.load(fh)
            if isinstance(loaded, list):
                existing = loaded
        except (json.JSONDecodeError, OSError):
            existing = []

    existing.extend(_PERF_ROWS)

    with abs_path.open("w") as fh:
        json.dump(existing, fh, indent=2)
        fh.write("\n")


@pytest.fixture
def assert_perf_benefit(request):
    """Assert performance improvement; record a row when ``--record-perf`` is set.

    Default behavior is unchanged: the fixture exposes ``_assert_perf_benefit``
    so existing assertions keep firing. When ``--record-perf=PATH`` is passed
    on the pytest CLI, each call additionally appends a row to the in-memory
    buffer flushed by :func:`pytest_sessionfinish`.

    Recording happens *before* the assertion so failed buckets still appear in
    the acceptance log (the run can then waive or widen the bucket).
    """
    record_path = request.config.getoption("--record-perf")

    if not record_path:
        return _assert_perf_benefit

    nodeid = request.node.nodeid

    def _record_then_assert(pt_stats, op_stats, label="", max_perf_overhead=0.20):
        _record_perf_row(
            nodeid=nodeid, label=label, pt_stats=pt_stats, op_stats=op_stats
        )
        _assert_perf_benefit(
            pt_stats, op_stats, label=label, max_perf_overhead=max_perf_overhead
        )

    return _record_then_assert
