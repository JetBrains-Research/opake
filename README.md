# Opaque

Functional DP-SGD and DP-FTRL for PyTorch.

Opaque provides composable primitives for differentially private model
training in PyTorch: per-example gradient clipping, calibrated noise
injection, privacy accounting, and Poisson sampling. Built on `torch.func`,
it uses a functional API with explicit state — no hooks, no subclassing, no
hidden mutation.

> **Work in progress:** Opaque is research software under active development.
> Its differential-privacy mechanisms, accounting, and privacy guarantees are
> still being validated and may change. Do not rely on it for production or
> compliance-sensitive privacy guarantees without independent validation for
> your use case.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11-3.13](https://img.shields.io/badge/python-3.11--3.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.9+](https://img.shields.io/badge/pytorch-2.9+-red.svg)](https://pytorch.org/)
[![JetBrains Research](https://jb.gg/badges/research.svg)](https://confluence.jetbrains.com/display/ALL/JetBrains+on+GitHub)
[![CI](https://github.com/JetBrains-Research/opaque/actions/workflows/ci.yml/badge.svg)](https://github.com/JetBrains-Research/opaque/actions/workflows/ci.yml)

**[Documentation](https://jetbrains-research.github.io/opaque/)**

## Packages

Install and depend on `opaque` only. The repository is implemented as
[PEP 420] namespace packages under the shared `opaque.*` namespace:

| Distribution | Import roots | Purpose |
|---|---|---|
| `opaque` | — | Convenience installer; pulls in a curated bundle of sub-packages |
| `opaque-base` | `opaque.serialization` | Pure-Python serialization registry + dispatcher; the seam every other wheel registers handlers against |
| `opaque-engine` | `opaque.{types,pytree,random,distributed,functional,scheduling,profiling}` | Torch substrate: pytree wrappers (`ClippedPytree` / `NoisedPytree` / `PerGroup`), `RngKey`, fixed + AUTO-S clipping, schedules + warmup, DDP plumbing, profiler |
| `opaque-optimizers` | `opaque.optimizers` | Torchopt-based functional optimizer chain (DP-aware AdamW-BC and friends) |
| `opaque-dpsgd` | `opaque.dpsgd` | Gaussian / per-group noise, Poisson samplers, adaptive clipping, DP-SGD-specific accounting factories |
| `opaque-dpftrl` | `opaque.dpftrl` | DP-FTRL mechanisms (BLT, BSR, BiSR, band-MF, λ-CGD), private second moments, correlated-noise samplers, DP-FTRL-specific accounting factories |
| `opaque-auditing` | `opaque.auditing` | Empirical privacy auditing (one-run, coin-flip, loss attacks) |
| `opaque-patches` | `opaque.patches` | Unified patching entrypoint for PyTorch checkpointing, Hugging Face compat wrappers, Triton kernels, and PEFT/LoRA fusion |
| `opaque-transformers` | `opaque.transformers` | Hugging Face trainer + integration; TRL-style `SFTTrainer` / `DPOTrainer` (`opaque.transformers.trl`) built on `DPTrainer` |
| `opaque-alignment` | `opaque.alignment` | Functional, mechanism-agnostic DP-safe SFT / DPO primitives: per-example losses, log-prob helpers, collators, reference helpers, reward metrics |
| `opaque-federated` | `opaque.federated` | Federated DP on IFED: cohort sampler, round loader, per-client clipped-sum strategy, round driver |
| `opaque-accounting` | `opaque.accounting` | PLD privacy accounting (Rust/PyO3 backend); torch-free standalone |

[PEP 420]: https://peps.python.org/pep-0420/

### Import layout

```
opaque.serialization                                       <- opaque-base
opaque.{types,pytree}                                      <- opaque-engine
opaque.{random,distributed}                                <- opaque-engine
opaque.{functional,scheduling,profiling}                   <- opaque-engine
opaque.optimizers                                          <- opaque-optimizers
opaque.dpsgd.{clipping,noise,sampling,accounting}          <- opaque-dpsgd
opaque.dpftrl.{clipping,noise,sampling,accounting}         <- opaque-dpftrl
opaque.auditing                                            <- opaque-auditing
opaque.patches.{kernels,torch,transformers,peft}           <- opaque-patches
opaque.transformers{,.trl}                                 <- opaque-transformers
opaque.alignment.{sft,dpo,data,metric}                     <- opaque-alignment
opaque.federated.{clipping,data,sampling}                  <- opaque-federated
opaque.accounting                                          <- opaque-accounting
```

## Installation

```bash
# From JetBrains Packages
pip install opaque \
  --index-url https://packages.jetbrains.team/pypi/p/fed/python/simple/

# Or with uv
uv add opaque \
  --index https://packages.jetbrains.team/pypi/p/fed/python/simple/
```

Extras:

```bash
pip install "opaque[auditing]"      # empirical privacy auditing
pip install "opaque[dpftrl]"        # correlated-noise DP-FTRL components
pip install "opaque[transformers]"  # Hugging Face + patching components
pip install "opaque[federated]"     # federated DP on IFED
pip install "opaque[all]"           # all optional components
```

### Patching

Hugging Face and checkpoint patches are applied explicitly through
`opaque.patches`:

```python
from opaque.patches import apply_model_patches, apply_runtime_patches

apply_runtime_patches()

# ... build / wrap the model, then patch the concrete instance
apply_model_patches(model)
```

`apply_runtime_patches()` enables the runtime-side checkpoint, collator, and
loss-mapping fixes. `apply_model_patches(model)` wires compat wrappers and
Triton kernels into the specific model instance, including PEFT/LoRA modules.

See the [model-patches guide](https://jetbrains-research.github.io/opaque/latest/user-guide/huggingface/model-patches/)
for patching details, model compatibility, and tuning knobs.

## Example

A minimal DP-SGD training loop:

```python
import torch
import opaque.accounting as acc                # cross-cutting (calibrate, budget)
import opaque.dpsgd.accounting as dpsgd_acc    # DP-SGD per-step factories
from opaque.dpsgd.clipping import clipped_grad
from opaque.dpsgd.noise import gaussian_noise
from opaque.random import key

def loss_fn(params, x, y):
    return ((x @ params - y) ** 2).sum()

# Calibrate noise for target privacy budget
result = acc.calibrate(
    acc.epsilon_budget(3.0, delta=1e-5),
    lambda nm: dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), sample_rate=0.01) * 1000,
    param_min=0.1, param_max=10.0,
)
batch_size = 64  # expected batch size for Poisson sampling

# DP-SGD components
grad_fn, clip_state = clipped_grad(
    loss_fn, clipping_norm=1.0, batch_argnums=(1, 2),
    normalize_by=batch_size,
)
noise_fn, noise_state = gaussian_noise(
    noise_multiplier=result.param, key=key(42),
)

# Training loop
params = torch.randn(10, requires_grad=False)
lr = 0.01
for batch_x, batch_y in dataloader:
    grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    params = params - lr * noisy_grads.pytree  # or wire opaque.optimizers
```

## Features

- **Per-example gradient clipping** via `torch.func.vmap` + `torch.func.grad`,
  with fixed, adaptive (Andrew et al. 2021), and AUTO-S (Bu et al. 2023) variants.
- **Noise injection**: Gaussian and correlated matrix-factorization noise
  (band-MF, BLT, BSR, BiSR, DP-λCGD), including private second-moment streams
  for adaptive optimizers.
- **Privacy accounting**: Rust-based PLD engine with tight composition,
  multiple privacy metrics (ε-δ, f-DP advantage, error rates), and noise
  calibration via binary search.
- **Sampling**: DP-SGD Poisson (plain or capped), DP-FTRL Poisson (identity or banded),
  balls-in-bins, b-min-separation, and sequential batch samplers.
- **Privacy auditing**: empirical privacy validation via membership inference.
- **Distributed training**: DDP-compatible with synchronized noise and
  gradient aggregation via `opaque.distributed`.
- **Hugging Face compatibility**: automatic `vmap` patching for LLaMA, Mistral,
  Qwen2/3, Phi-3, Gemma/Gemma2, Granite, Cohere/Cohere2, plus fused Triton
  kernels via `opaque.patches`.

## Documentation

- [Documentation](https://jetbrains-research.github.io/opaque/)
- [Getting Started](https://jetbrains-research.github.io/opaque/latest/getting-started/quickstart/)
- [User Guide](https://jetbrains-research.github.io/opaque/latest/user-guide/)
- [Tutorials](https://jetbrains-research.github.io/opaque/latest/tutorials/)
- [API Reference](https://jetbrains-research.github.io/opaque/latest/reference/)
- [Examples](examples)

## Development

```bash
uv sync --group dev --all-packages --extra all
uv run pytest -m "not cuda and not mps and not slow"        # PR-equivalent suite
uv run ruff format packages/                                # Format
uv run ruff check packages/                                 # Lint
uv run --group docs mkdocs build --strict                   # Build docs
cargo test --workspace
```

See [CONTRIBUTING.md](./CONTRIBUTING.md) for the full development workflow.
The [development guide](docs/development/index.md) includes fork-based setup.

## References

- [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133) (Abadi et al. 2016)
- [JAX-Privacy](https://github.com/google-deepmind/jax_privacy) — original inspiration
- [Opacus](https://opacus.ai/) — alternative PyTorch DP library (hook-based design)

## License

Apache 2.0. See [LICENSE](./LICENSE).
