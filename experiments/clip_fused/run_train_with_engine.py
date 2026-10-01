"""Run examples/train_dpsgd.py unmodified, with the clip seam instrumented and
optionally the fused engine installed.

Env:
  OPAKE_CLIP_ENGINE  stream (default) | fused
  OPAKE_SEAM_TIMING  1 -> cuda-sync-time every _stream_clip_and_sum call.
                     Adds two syncs per call (removes CPU/GPU overlap at the
                     seam): use for ATTRIBUTION runs only, never for headline
                     step times.

At exit prints: seam calls, flags seen, leaves/dtypes per call, engine
dispatch (fused vs fallback reasons), and seam time (if timing on), split into
warmup calls and steady-state calls.

Usage (CUDA host, from repo root):
  OPAKE_CLIP_ENGINE=fused .venv/bin/python experiments/clip_fused/run_train_with_engine.py \
      --preset qwen-7b-kstack ... (same flags as train_dpsgd.py)
"""

from __future__ import annotations

import atexit
import os
import runpy
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import fused_engine  # noqa: E402

from opake.api.engine.clipping import _clipped_fun as CF  # noqa: E402
from opake.api.engine.pytree import tree_flatten  # noqa: E402

ENGINE = os.environ.get("OPAKE_CLIP_ENGINE", "stream")
TIMING = os.environ.get("OPAKE_SEAM_TIMING", "0") == "1"
WARMUP_CALLS = int(os.environ.get("OPAKE_SEAM_WARMUP_CALLS", "16"))
_BASE = (fused_engine.fused_stream_clip_and_sum if ENGINE == "fused"
         else fused_engine._ORIGINAL_STREAM)

REC = {"calls": 0, "ms": [], "flags": Counter(), "shape": Counter()}


def _seam(values, clipping_norm, **kw):
    REC["calls"] += 1
    leaves, _ = tree_flatten(values)
    REC["flags"][(f"aux={kw['return_aux']}", f"stats={kw['return_stats']}",
                  f"m2={kw['second_moment']}", type(clipping_norm).__name__)] += 1
    REC["shape"][(len(leaves), kw["batch_size"],
                  ",".join(sorted({str(leaf.dtype).split('.')[-1] for leaf in leaves})),
                  sum(leaf.numel() for leaf in leaves) // max(kw["batch_size"], 1))] += 1
    if not TIMING:
        return _BASE(values, clipping_norm, **kw)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = _BASE(values, clipping_norm, **kw)
    torch.cuda.synchronize()
    REC["ms"].append((time.perf_counter() - t0) * 1e3)
    return out


CF._stream_clip_and_sum = _seam


@atexit.register
def _report():
    print(f"\n[seam] engine={ENGINE} timing={TIMING} calls={REC['calls']}")
    for k, v in REC["flags"].items():
        print(f"[seam] flags {k}: {v} calls")
    for (n_leaves, b, dts, per_ex), v in REC["shape"].items():
        print(f"[seam] tree: {n_leaves} leaves, batch {b}, dtypes {dts}, "
              f"{per_ex:,} elems/example: {v} calls")
    print(f"[seam] dispatch: fused={fused_engine.STATS['fused']} "
          f"fallback={dict(fused_engine.STATS['fallback'])}")
    if REC["ms"]:
        steady = REC["ms"][WARMUP_CALLS:] or REC["ms"]
        print(f"[seam] time: total {sum(REC['ms']) / 1e3:.2f} s over {len(REC['ms'])} calls; "
              f"steady-state (after {WARMUP_CALLS} calls) median {statistics.median(steady):.2f} ms, "
              f"mean {statistics.mean(steady):.2f} ms, n={len(steady)}")


script = Path("examples/train_dpsgd.py")
sys.argv = [str(script)] + sys.argv[1:]
runpy.run_path(str(script), run_name="__main__")
