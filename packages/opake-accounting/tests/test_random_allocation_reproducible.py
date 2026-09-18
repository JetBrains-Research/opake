"""Random-allocation ε is bit-identical across thread counts.

The Monte Carlo primitive this replaced could not satisfy this: every MC
driver shards by ``rayon::current_num_threads()`` and seeds ``seed + tid``,
so its ε depends on how many cores the machine happened to have.  Two
people auditing the same run on different hardware got different numbers.

That is the whole point of the deterministic transform, so it is a
regression test rather than a nice-to-have.  Each thread count runs in a
fresh subprocess because ``RAYON_NUM_THREADS`` is read once when the global
pool is built — setting it in-process after the first PLD would be a no-op
and the test would pass vacuously.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from opake.api.accounting.core import _native
from opake.api.accounting.core.discretization import get_discretization

_THREAD_COUNTS = ("1", "2", "8")

_SCRIPT = """
from opake.api.accounting.core import _native
from opake.api.accounting.core.discretization import get_discretization

cfg = get_discretization().to_native()
out = []
for sigma, t in ((1.0, 8), (2.0, 16)):
    pld = _native.random_allocation_gaussian_pld(sigma, t, 1, cfg)
    out.append(repr(pld.epsilon_at(1e-8)))
print(" ".join(out))
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


@pytest.mark.slow
def test_epsilon_is_bit_identical_across_thread_counts():
    results = {n: _epsilons(n) for n in _THREAD_COUNTS}
    baseline = results[_THREAD_COUNTS[0]]
    assert baseline, "subprocess produced no output"
    for n, got in results.items():
        assert got == baseline, (
            f"RAYON_NUM_THREADS={n} gave {got}, "
            f"RAYON_NUM_THREADS={_THREAD_COUNTS[0]} gave {baseline}"
        )


def test_native_allocation_api_has_no_optimistic_selector():
    cfg = get_discretization().to_native()
    with pytest.raises(TypeError):
        _native.random_allocation_gaussian_pld(1.0, 8, 1, True, cfg)
