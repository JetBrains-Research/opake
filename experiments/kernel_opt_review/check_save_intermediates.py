"""Does lora_mlp(save_intermediates=True) save gate/up under torch.func.vmap(grad())?

Instruments Opake_LoRA_MLP.backward to record ctx.save_intermediates and the
number of saved tensors (12 = gate/up saved, 10 = recompute), then compares
per-example gradients between the two settings and against a pure-PyTorch
reference (no fused kernel)."""
import torch, torch.nn.functional as F
from torch.func import grad, vmap
import opake.api.patches.kernels.lora as L

torch.manual_seed(0)
dev, dt = "cuda", torch.float32
B, T, D, H, r = 4, 8, 64, 160, 8
X = torch.randn(B, T, D, device=dev, dtype=dt)
W = lambda o, i: torch.randn(o, i, device=dev, dtype=dt) / i**0.5
Wg, Wu, Wd = W(H, D), W(H, D), W(D, H)
Ag, Bg = W(r, D).T.contiguous(), (torch.randn(H, r, device=dev, dtype=dt) * 0.1).T.contiguous()
Au, Bu = W(r, D).T.contiguous(), (torch.randn(H, r, device=dev, dtype=dt) * 0.1).T.contiguous()
Ad, Bd = W(r, H).T.contiguous(), (torch.randn(D, r, device=dev, dtype=dt) * 0.1).T.contiguous()
S = 0.5
seen = []
orig_backward = L.Opake_LoRA_MLP.backward
def spy(ctx, *g):
    seen.append((ctx.save_intermediates, len(ctx.saved_tensors)))
    return orig_backward(ctx, *g)
L.Opake_LoRA_MLP.backward = staticmethod(spy)

def loss(adapters, x, save):
    Ag_, Bg_, Au_, Bu_, Ad_, Bd_ = adapters
    out = L.opake_lora_mlp(x, Wg, Ag_, Bg_, S, Wu, Au_, Bu_, S, Wd, Ad_, Bd_, S,
                     activation="swiglu", save_intermediates=save)
    return out.float().pow(2).mean()

def ref_loss(adapters, x):  # unfused reference, A:[in,r] B:[r,out]
    Ag_, Bg_, Au_, Bu_, Ad_, Bd_ = adapters
    g = x @ Wg.T + S * (x @ Ag_) @ Bg_
    u = x @ Wu.T + S * (x @ Au_) @ Bu_
    h = F.silu(g) * u
    o = h @ Wd.T + S * (h @ Ad_) @ Bd_
    return o.pow(2).mean()

ad = (Ag, Bg, Au, Bu, Ad, Bd)
per_ex = lambda f: vmap(grad(f), in_dims=(None, 0))(ad, X)
results = {}
for save in (False, True):
    seen.clear()
    results[save] = per_ex(lambda a, x: loss(a, x, save))
    print(f"save_intermediates={save}: backward saw (ctx.save_intermediates, n_saved) = {sorted(set(seen))}")
ref = per_ex(ref_loss)
for save in (False, True):
    worst = max(((g - r).abs().max() / r.abs().max().clamp_min(1e-30)).item() for g, r in zip(results[save], ref))
    print(f"save_intermediates={save}: max rel diff vs unfused reference over 6 per-example grads = {worst:.2e}")
same = max(((a - b).abs().max()).item() for a, b in zip(results[False], results[True]))
print(f"recompute vs save: max abs diff = {same:.2e}")
