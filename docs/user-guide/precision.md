# Numerical Precision

Opaque's recommended training dtypes are `bfloat16` and `float32`. Low-level
primitives can process `float16` tensors, but Opaque does not provide dynamic
loss scaling and `Trainer` does not support fp16 training.

Two independent knobs control numerical precision:

| Knob | Where | What it controls |
|------|-------|------------------|
| `torch.autocast(...)` | Around the forward / loss closure | Op-level dtype dispatch (matmul, conv, and similar operations). Standard PyTorch. |
| `compute_dtype` | Kwarg on clipping and noise factories | Precision used for sensitivity-bound reductions, accumulation, and noise sampling. **DP-critical**: it must be high enough for the accountant to be calibrated to the realized mechanism. |

## Recommended `bfloat16` training

`bfloat16` has the same exponent range as `float32`, so it does not need loss
scaling. Enter an autocast context around the loss closure:

```python
def per_example_loss(params, x, y):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return loss(model_fn(params, x), y)

grad_fn, clip_state = clipped_grad(
    per_example_loss,
    argnums=0,
    batch_argnums=(1, 2),
    clipping_norm=C,
)
```

Clipping defaults to safe accumulation behavior and the noise factories sample
in `float32` by default, even though the forward uses `bfloat16`.

## `float32` training

No autocast or precision helper is needed. Clipping preserves float32 reduction
precision by default, while noise factories also default to
`compute_dtype=torch.float32`.

## The fp16 boundary

`float16` has a narrow exponent range, so gradients can underflow or overflow.
Native PyTorch commonly compensates with `torch.amp.GradScaler`, whose scale
schedule reacts to an un-noised finiteness check. In private training, that
data-dependent state can alter later numerical gradient queries without being
represented in the privacy accountant. Opaque therefore does not expose a loss
scaler or an aggregate pre-clipping finiteness signal.

Low-level clipping and noise primitives still accept fp16 tensors and promote
their privacy-critical reductions by default. This protects the numerical
sensitivity and noise calculations, but it does not recover gradient
information already lost during an fp16 forward or backward pass. Prefer bf16
on supported hardware and float32 otherwise.

## The `compute_dtype` contract

`compute_dtype` is **not** an autocast surface. Autocast intercepts operation
dispatch inside the forward pass; clipping and noise reductions run outside
that region and produce the sensitivity bound and noise scale the privacy
accountant relies on.

- **Use the safe defaults.** Clipping defaults to `compute_dtype=None`, which
  promotes `bfloat16` and `float16` inputs to `float32` for L2 reductions and
  leaf scaling while leaving `float32` and `float64` inputs unchanged. Both
  `gaussian_noise(...)` and `mf_gaussian_noise(...)` default to
  `compute_dtype=torch.float32`.
- **Do not lower precision without a separate numerical argument.** Setting
  `compute_dtype=torch.bfloat16` or `torch.float16` can make the realized
  sensitivity and noise scale diverge from the accountant's model.
- **You can raise it.** Pass `compute_dtype=torch.float64` explicitly to both
  clipping and noise when higher-precision reductions and sampling are needed.
- **The final output cast still matters.** Clipping multiplies in the wider of
  `compute_dtype` and the leaf's storage precision, then casts the result once
  to the leaf dtype. That final cast can round up past the bound,
  so clipping conservatively shrinks each leaf's scale and rounds outward
  subnormal casts toward zero to preserve `norm(output) <= clipping_norm` on
  the values as stored.

Under microbatching, the running sum is held at the accumulation precision and
cast once at the end. A `bfloat16` run therefore uses one model-sized float32
accumulator by default; explicitly lowering `compute_dtype` trades away this
protection.

## Compatibility with `torch.amp`

| `torch.amp` primitive | Opaque behavior |
|-----------------------|-----------------|
| `torch.amp.autocast(device_type, dtype=...)` | Used directly around the loss closure. |
| `torch.amp.GradScaler` | No Opaque counterpart. Data-dependent scale adaptation is outside Opaque's accounting model. |
| `torch.amp.custom_fwd` / `custom_bwd` | Not used by the functional DP path, which differentiates per-example losses with `vmap(grad(...))`. |
