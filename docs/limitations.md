# Known Limitations

## Matrix factorization (MF) workloads

Correlated-noise mechanisms are analyzed for a **specific linear map** (the strategy
matrix) and optional subsampling model. **DP correctness** requires that accounting,
noise, and sampling match that map. **Utility** can depend on how closely the
encoded workload matches your optimizer (for example, BandMF/BLT apply
`lr_schedule` on the training-step axis). Private second moments add a second MF
stream; pass its strategy explicitly.

See [Matrix factorization (MF)](user-guide/dp-ftrl.md).

## Randomness and the threat model

Random keys are reproducible PRNG state, not cryptographically secure
randomness. Exposing seeds or noise state can reveal the noise realization.
Keep them private and strip the random and noise state before publishing.
See [Random keys](user-guide/rng-key.md).

## Telemetry outside the guarantee

DP accounting does not cover exact diagnostics such as un-noised mean loss,
pre-clip gradient norm, clip rate, batch size, or token and reward metrics.
Keep them private or disable external logging. Hub uploads exclude the default
`runs/` directory; review other outputs before publishing.

## Gradient checkpointing

Supported under `vmap(grad(...))` via automatic patches. See
[Memory Optimizations](user-guide/memory-optimizations.md#gradient-checkpointing)
for usage and limitations.

## Flash Attention 2 incompatibility

Flash Attention 2 uses `torch.nonzero` internally for sequence unpadding,
which produces dynamic output shapes that are incompatible with vmap.

**Solution:** Use SDPA or eager attention when loading Hugging Face models:

```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    attn_implementation="sdpa",
)
```

See [Model Patches — Attention implementations](user-guide/huggingface/model-patches.md#attention-implementations)
for supported attention implementations.

## DDP only

Opaque supports `torch.nn.parallel.DistributedDataParallel` (DDP). FSDP,
Tensor Parallel, and Pipeline Parallel are not supported. Multi-node DDP
should work but is not extensively tested. First-class distributed backends are
NCCL, Gloo, and MPI. Vendor/runtime-specific backends require external stacks
and are not covered by default CI.

## Kernel patching lives in `opaque.patches`

Kernel optimization and Hugging Face model patches are provided by
`opaque.patches`. Triton kernels require CUDA; on CPU, MPS, or without
Triton, compatible non-kernel patches remain available. Configure them with
the explicit flags described in [Model Patches — Trainer
integration](user-guide/huggingface/model-patches.md#trainer-integration).

Public standalone kernels (`opaque_swiglu`, `opaque_cross_entropy_loss`,
`opaque_lora_w`, `opaque_lora_qkv`, `opaque_lora_mlp`) are importable
from `opaque.patches.kernels`.

## In-place operations under vmap

`torch.func.vmap` does not support in-place tensor operations. Models that
use in-place operations (for example, `x.add_()`, `x[:, 0] = 0`) in their
forward pass will fail. Replace them with out-of-place equivalents
(`x = x + y`, `x = torch.cat([zeros, x[:, 1:]], dim=1)`).

## Data-dependent control flow under vmap

vmap requires the same operations to execute for every example in the
batch. Conditional branches that depend on tensor values
(`if x.sum() > 0`) will fail. This is a fundamental limitation of
vectorized execution.
