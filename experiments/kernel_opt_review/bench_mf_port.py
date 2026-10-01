"""A100: MF streaming multiply over a LoRA-7B-like tree, main vs port (same process)."""
import importlib.util, sys, time, torch
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m
import opake.api.dpftrl.noise._blt_math as blt_main
import opake.api.dpftrl.noise._toeplitz as tp_main
blt_port = load("opake.api.dpftrl.noise._port_blt_math", sys.argv[1] + "/_blt_math.py")
tp_port = load("opake.api.dpftrl.noise._port_toeplitz", sys.argv[1] + "/_toeplitz.py")
dev = torch.device("cuda")
shapes = []
for _ in range(28):
    shapes += [(16, 3584), (3584, 16), (16, 3584), (512, 16), (16, 3584), (512, 16), (16, 3584), (3584, 16),
               (16, 3584), (18944, 16), (16, 3584), (18944, 16), (16, 18944), (3584, 16)]
g = torch.Generator(device="cuda").manual_seed(0)
tree = {f"p{i}": torch.randn(s, device=dev, generator=g) for i, s in enumerate(shapes)}
def timeit(fn, it=15, warm=4):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(it):
        torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort(); return ts[len(ts) // 2]
def bench(sm):
    state = sm.init_multiply(tree)
    out, state = sm.multiply_next(tree, state)  # warm the coefficient cache
    return out, timeit(lambda: sm.multiply_next(tree, state))
print(f"tree: {len(tree)} leaves, {sum(t.numel() for t in tree.values())/1e6:.1f}M elems, fp32, CUDA")
for label, nb in [("BLT 4 buffers", 4), ("BLT 8 buffers", 8)]:
    decay = [0.95 * (0.7 ** k) for k in range(nb)]; scale = [0.3 / (k + 1) for k in range(nb)]
    for kind in ("C", "C^-1"):
        mk = (lambda m: m.as_streaming_matrix if kind == "C" else m.inverse_as_streaming_matrix)
        b_main = blt_main.BufferedToeplitz.build(buf_decay=decay, output_scale=scale)
        b_port = blt_port.BufferedToeplitz.build(buf_decay=decay, output_scale=scale)
        om, tm = bench(mk(blt_main)(b_main)); op, tpt = bench(mk(blt_port)(b_port))
        same = all(torch.equal(om[k], op[k]) for k in om)
        print(f"  {label:14s} {kind:5s} main {tm:7.2f} ms  port {tpt:7.2f} ms  (x{tm/tpt:4.2f})  outputs bitwise equal: {same}")
for bands in (4, 16):
    coef = torch.tensor([1.0] + [0.5 / (k + 1) for k in range(bands - 1)], dtype=torch.float64)
    om, tm = bench(tp_main.inverse_as_streaming_matrix(coef)); op, tpt = bench(tp_port.inverse_as_streaming_matrix(coef))
    same = all(torch.equal(om[k], op[k]) for k in om)
    print(f"  Toeplitz C^-1 bands={bands:2d}  main {tm:7.2f} ms  port {tpt:7.2f} ms  (x{tm/tpt:4.2f})  outputs bitwise equal: {same}")
