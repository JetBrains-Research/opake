"""Shadow A/B at the clip seam on a real training run.

Every _stream_clip_and_sum call runs BOTH the production streaming path and the
fused engine on the identical per-example values, compares every output
(per-leaf reduced sums, diagnostics "norms" and "clipped_norms"), and returns
the FUSED result (so the run proceeds exactly as a fused run would).

This answers "does the engine agree with production on this workload's actual
per-example gradients?" independently of run-to-run trajectory divergence.

Usage (CUDA host, repo root; same flags as examples/train_dpsgd.py):
  .venv/bin/python experiments/clip_fused/run_train_shadow.py --preset ... 
"""

from __future__ import annotations

import atexit
import runpy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import fused_engine  # noqa: E402

from opake.api.engine.clipping import _clipped_fun as CF  # noqa: E402
from opake.api.engine.pytree import tree_flatten  # noqa: E402

REC = {"calls": 0, "rows": []}


def _rel(a, b):
    a = a.detach().double()
    b = b.detach().double()
    d = (a - b).abs().max().item()
    return d / max(a.abs().max().item(), 1e-300)


def _seam(values, clipping_norm, **kw):
    out_p = fused_engine._ORIGINAL_STREAM(values, clipping_norm, **kw)
    out_f = fused_engine.fused_stream_clip_and_sum(values, clipping_norm, **kw)
    REC["calls"] += 1
    red_p, _ = tree_flatten(out_p[0])
    red_f, _ = tree_flatten(out_f[0])
    worst_red = max(_rel(a, b) for a, b in zip(red_p, red_f))
    row = {"call": REC["calls"], "batch": kw["batch_size"], "reduced": worst_red,
           "dtypes": (red_p[0].dtype, red_f[0].dtype)}
    dp, df = out_p[4], out_f[4]
    if isinstance(dp, dict):
        assert set(dp) == set(df), f"diag keys differ: {set(dp)} vs {set(df)}"
        for k in sorted(dp):
            if isinstance(dp[k], torch.Tensor):
                row[k] = _rel(dp[k], df[k])
                row[k + "_dtype"] = (dp[k].dtype, df[k].dtype)
        row["norm_max"] = dp["norms"].max().item() if "norms" in dp else None
        row["C"] = float(clipping_norm)
    REC["rows"].append(row)
    return out_f


CF._stream_clip_and_sum = _seam


@atexit.register
def _report():
    rows = REC["rows"]
    print(f"\n[shadow] calls={REC['calls']} dispatch fused={fused_engine.STATS['fused']} "
          f"fallback={dict(fused_engine.STATS['fallback'])}")
    if not rows:
        return
    for key in ("reduced", "norms", "clipped_norms"):
        vals = [r[key] for r in rows if key in r]
        if vals:
            worst = max(range(len(vals)), key=lambda i: vals[i])
            print(f"[shadow] {key:14s} worst rel {vals[worst]:.3e} (call {rows[worst]['call']}), "
                  f"median {sorted(vals)[len(vals) // 2]:.3e}, n={len(vals)}")
    dts = {(r["dtypes"], r.get("norms_dtype"), r.get("clipped_norms_dtype")) for r in rows}
    print(f"[shadow] dtypes (prod, fused) reduced/norms/clipped_norms: {dts}")
    print("[shadow] first 3 / last 3 calls:")
    for r in rows[:3] + rows[-3:]:
        print("   ", {k: (f"{v:.2e}" if isinstance(v, float) else v)
                      for k, v in r.items() if not k.endswith("dtype") and k != "dtypes"})


script = Path("examples/train_dpsgd.py")
sys.argv = [str(script)] + sys.argv[1:]
runpy.run_path(str(script), run_name="__main__")
