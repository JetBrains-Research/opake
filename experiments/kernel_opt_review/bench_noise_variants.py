"""A100 microbench for the kernel-optimizations noise changes (DP-FTRL paths)."""
import math, time, torch
dev = torch.device("cuda")
def timeit(fn, it=20, warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(it):
        torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort(); return ts[len(ts) // 2]
def peak(fn):
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated(); r = fn(); torch.cuda.synchronize()
    p = (torch.cuda.max_memory_allocated() - base) / 2**20; del r; return p
# ---- (1) i.i.d. draw strategies over a LoRA-7B-like tree (392 leaves, 40.4M elems) ----
lora = []
for _ in range(28):
    for (a, b) in [(16, 3584), (3584, 16), (16, 3584), (512, 16), (16, 3584), (512, 16), (16, 3584), (3584, 16),
                   (16, 3584), (18944, 16), (16, 3584), (18944, 16), (16, 18944), (3584, 16)]:
        lora.append((a, b))
tree = [torch.empty(s, device=dev) for s in lora]
print(f"tree: {len(tree)} leaves, {sum(t.numel() for t in tree)/1e6:.1f}M elems")
std = 0.5
def old_draw():  # main: per-leaf try device draw (raises for CPU gen) -> CPU draw -> .to(dev)
    g = torch.Generator().manual_seed(0); out = []
    for t in tree:
        try: n = torch.randn(t.shape, device=dev, generator=g)
        except RuntimeError: n = torch.randn(t.shape, generator=g).to(dev)
        out.append(n * std)
    return out
def kopt_flat():
    g = torch.Generator().manual_seed(0); tot = sum(t.numel() for t in tree)
    f = torch.randn((tot,), generator=g).to(dev); out, o = [], 0
    for t in tree: n = t.numel(); out.append(f[o:o+n].reshape(t.shape) * std); o += n
    return out
def batched(chunk_bytes=256 * 2**20):  # proposal: per-leaf draws into one pinned buffer, one copy, in-place scale, views
    g = torch.Generator().manual_seed(0); tot = sum(t.numel() for t in tree)
    buf = torch.empty((tot,)); o = 0; views = []
    for t in tree:
        n = t.numel(); torch.randn(t.shape, generator=g, out=buf[o:o+n].view(t.shape)); o += n
    d = buf.to(dev); d.mul_(std); out, o = [], 0
    for t in tree: n = t.numel(); out.append(d[o:o+n].view(t.shape)); o += n
    return out
for name, fn in [("main per-leaf (try/except)", old_draw), ("k-opt flat", kopt_flat), ("batched, stream-preserving", batched)]:
    print(f"  {name:30s} {timeit(fn, it=10, warm=3):8.1f} ms   peak device {peak(fn):7.1f} MiB")
# ---- (2) streaming-matrix ops: tensordot vs broadcast-multiply-sum, diag-matmul vs elementwise ----
print("\nBLT _read / Toeplitz inner (sum over nb buffers) and _update:")
for nb in (4, 8):
    for M in (1_000_000, 40_000_000):
        state = torch.randn(nb, M, device=dev); coef = torch.randn(nb, device=dev); rhs = torch.randn(M, device=dev)
        r_td = timeit(lambda: torch.tensordot(coef, state, dims=([0], [0])))
        r_bc = timeit(lambda: (coef.view(-1, 1) * state).sum(dim=0))
        u_dg = timeit(lambda: (torch.diag(coef) @ state.reshape(nb, -1)).reshape(state.shape) + rhs)
        u_el = timeit(lambda: state * coef.view(-1, 1) + rhs)
        print(f"  nb={nb} M={M/1e6:>4.0f}M  read: tensordot {r_td:7.3f} ms | broadcast-sum {r_bc:7.3f} ms    "
              f"update: diag@ {u_dg:7.3f} ms | elementwise {u_el:7.3f} ms")
        del state, rhs
