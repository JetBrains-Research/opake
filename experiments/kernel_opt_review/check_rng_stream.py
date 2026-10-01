"""Does a single flat CPU randn reproduce per-leaf randn draws bit for bit?

Reviewed claim (mihajlo/kernel-optimizations 883a9e3f): flattening
_iid_normal_noise into one draw per device "preserves bit-for-bit determinism
with the existing CPU generator stream". Also checks the stream-preserving
alternative: per-leaf draws written into one preallocated buffer.
Run anywhere: python experiments/kernel_opt_review/check_rng_stream.py
"""
import math

import torch


def per_leaf(shapes, dt, seed):  # main's behaviour
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(s, dtype=dt, generator=g) for s in shapes]


def flat(shapes, dt, seed):  # 883a9e3f
    g = torch.Generator().manual_seed(seed)
    f = torch.randn((sum(math.prod(s) for s in shapes),), dtype=dt, generator=g)
    out, o = [], 0
    for s in shapes:
        n = math.prod(s)
        out.append(f[o : o + n].reshape(s))
        o += n
    return out


def batched(shapes, dt, seed):  # per-leaf draws into one buffer
    g = torch.Generator().manual_seed(seed)
    buf = torch.empty((sum(math.prod(s) for s in shapes),), dtype=dt)
    out, o = [], 0
    for s in shapes:
        n = math.prod(s)
        v = buf[o : o + n].view(s)
        torch.randn(s, dtype=dt, generator=g, out=v)
        out.append(v)
        o += n
    return out


CASES = {
    "all multiples of 16": [(64, 64), (128,), (16, 32), (4096,)],
    "LoRA-like (r=16)": [(16, 3584), (512, 16), (16, 18944), (3584, 16)],
    "has leaf < 16 (bias 10)": [(64, 64), (10,), (128,)],
    "has non-multiple-16 leaf": [(37, 3), (64, 64), (128,)],
    "odd conv-like": [(3, 3, 3, 8), (50257,), (8,)],
}
if __name__ == "__main__":
    same = lambda a, b: all(torch.equal(x, y) for x, y in zip(a, b))  # noqa: E731
    for name, shapes in CASES.items():
        for dt in (torch.float32, torch.float64):
            a = per_leaf(shapes, dt, 1234)
            print(f"{name:26s} {str(dt):14s} flat==per-leaf {same(a, flat(shapes, dt, 1234))!s:5s} "
                  f"batched==per-leaf {same(a, batched(shapes, dt, 1234))}")
