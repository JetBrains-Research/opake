"""b-min-sep Monte Carlo PLDs are reproducible across Rayon thread counts."""

from __future__ import annotations

import os
import subprocess
import sys

_THREAD_COUNTS = ("1", "2", "8")

_SCRIPT = """
from opake.api.accounting.core import _native
from opake.api.accounting.core.discretization import get_discretization

coef = [0.7 ** 0.5, 0.3 ** 0.5]
n_steps = 50
p = 0.05
sigma = 1.0
config = get_discretization(
    discretization=1e-4,
    seed=173,
    mc_resolution=5e-3,
    mc_failure_probability=1e-2,
).to_native()

one_shot = _native.bandmf_b_min_sep_warm_mc_pld(
    coef, n_steps, p, sigma, config
)
handle = _native.register_b_min_sep_transcript_corpus(
    coef, n_steps, p, config.resolved_num_mc_samples, config.seed
)
try:
    cached = _native.bandmf_b_min_sep_pld_from_transcript_handle(
        handle, coef, p, sigma, config
    )

    def fingerprint(pld):
        values = (
            pld.epsilon_at(1e-2),
            pld.epsilon_at(2e-2),
            pld.epsilon_at(5e-2),
            pld.infinity_mass,
            pld.mc_failure_probability,
            pld.mc_resolution,
        )
        return tuple(value.hex() for value in values)

    result = fingerprint(one_shot)
    assert fingerprint(cached) == result
    print(" ".join(result))
finally:
    _native.drop_b_min_sep_transcript_corpus(handle)
"""


def _fingerprint(num_threads: str) -> str:
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
    results = {n: _fingerprint(n) for n in _THREAD_COUNTS}
    baseline = results[_THREAD_COUNTS[0]]
    assert baseline, "subprocess produced no output"
    for n, got in results.items():
        assert got == baseline, (
            f"RAYON_NUM_THREADS={n} gave {got}, "
            f"RAYON_NUM_THREADS={_THREAD_COUNTS[0]} gave {baseline}"
        )
