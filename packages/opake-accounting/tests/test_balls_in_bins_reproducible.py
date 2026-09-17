"""Balls-in-bins Monte Carlo PLDs are reproducible across Rayon thread counts."""

from __future__ import annotations

import os
import subprocess
import sys

_THREAD_COUNTS = ("1", "2", "8")

_SCRIPT = """
from opake.api.accounting.core import _native
from opake.api.accounting.core.discretization import get_discretization

num_bins = 8
gram = [
    1.0 if i == j else 0.2 ** min(abs(i - j), num_bins - abs(i - j))
    for i in range(num_bins)
    for j in range(num_bins)
]
config = get_discretization(
    discretization=1e-3,
    seed=173,
    mc_resolution=5e-3,
    mc_failure_probability=1e-2,
).to_native()
pld = _native.bnb_mc_pld(gram, num_bins, 1.3, config)
print(" ".join(repr(pld.epsilon_at(delta)) for delta in (1e-2, 2e-2, 5e-2)))
"""


def _epsilons(num_threads: str) -> str:
    env = {**os.environ, "RAYON_NUM_THREADS": num_threads}
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, (
        f"RAYON_NUM_THREADS={num_threads} failed:\n{proc.stderr}"
    )
    return proc.stdout.strip()


def test_pld_is_bit_identical_across_thread_counts():
    results = {n: _epsilons(n) for n in _THREAD_COUNTS}
    baseline = results[_THREAD_COUNTS[0]]
    assert baseline, "subprocess produced no output"
    for n, got in results.items():
        assert got == baseline, (
            f"RAYON_NUM_THREADS={n} gave {got}, "
            f"RAYON_NUM_THREADS={_THREAD_COUNTS[0]} gave {baseline}"
        )
