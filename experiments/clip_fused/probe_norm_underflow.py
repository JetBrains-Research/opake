"""Probe: production norm underflow (fp32/bf16) — evidence for the theoretical
stored-value-bound hole found during the adversarial review (2026-10-01).

(1) On subnormal-magnitude inputs the computed per-example norm is 0 (squares
underflow in the fp32 accumulator), so the example is not clipped.
(2) Magnitude scan: per-element RMS a, C = true_norm/1.19 -> stored/C - 1.
Run on CUDA: .venv/bin/python experiments/clip_fused/probe_norm_underflow.py
"""
import math, torch
from opake.api.engine.clipping import _clipped_fun as CF
from opake.api.engine.clipping._streaming import _FIXED_SCALE, _stream_clip_and_sum as prod
from opake.api.engine.pytree import tree_flatten
red = lambda x: CF._sum_clipped_tensor(x, dim=0, output_dtype=None, compute_dtype=None)
def run(x, C):
    out = prod({"w": x}, torch.tensor(C, dtype=torch.float64, device="cuda"), scale=_FIXED_SCALE,
               batch_size=1, reduce_leaf=red, compute_dtype=None, second_moment=False,
               return_aux=False, return_stats=True)
    s = math.sqrt(sum(float((l.double().cpu()**2).sum()) for l in tree_flatten(out[0])[0]))
    return s / C - 1.0, float(out[4]["norms"][0])
# (1) mechanism on the A2 inputs: is the computed norm 0?
for dt, sub in [(torch.bfloat16, 2.0**-133), (torch.float32, 2.0**-149)]:
    n = 1024; v = 3*sub; x = torch.full((1, n), v, dtype=torch.float64).to(dt).cuda()
    C = v*math.sqrt(n)*0.84
    r, nrm = run(x, C)
    print(f"A2 {str(dt):15s} true norm {v*math.sqrt(n):.3e}  computed norm {nrm:.3e}  stored/C-1 {r:+.3e}")
# (2) magnitude scan: Gaussian leaf (N=1M), example norm = 1.19*C; per-element RMS a
print("\nscan: per-element RMS a, C = a*sqrt(N)/1.19  ->  production stored/C - 1")
g = torch.Generator().manual_seed(0); n = 1 << 20
raw = torch.randn(n, generator=g, dtype=torch.float64)
for dt in (torch.float32, torch.bfloat16):
    for e in range(-14, -27, -1):
        a = 10.0**e
        x64 = raw * (a / raw.pow(2).mean().sqrt())
        x = x64.to(dt).reshape(1, n).cuda()
        true = math.sqrt(float((x.double().cpu()**2).sum()))
        C = true / 1.19
        r, nrm = run(x, C)
        print(f"  {str(dt):15s} a=1e{e:+d}  C={C:.2e}  computed/true norm {nrm/true:.4f}  stored/C-1 {r:+.3e}{'  VIOL' if r > 0 else ''}")
