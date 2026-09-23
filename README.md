# Opake

Functional DP-SGD and DP-FTRL for PyTorch.

Opake provides composable primitives for differentially private model
training in PyTorch: per-example gradient clipping, calibrated noise
injection, privacy accounting, and Poisson sampling. Built on `torch.func`,
it uses a functional API with explicit state — no hooks, no subclassing, no
hidden mutation.

> **Work in progress:** Opake is research software under active development.
> Its differential-privacy mechanisms, accounting, and privacy guarantees are
> still being validated and may change. Do not rely on it for production or
> compliance-sensitive privacy guarantees without independent validation for
> your use case.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11-3.13](https://img.shields.io/badge/python-3.11--3.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.9+](https://img.shields.io/badge/pytorch-2.9+-red.svg)](https://pytorch.org/)
[![JetBrains Research](https://jb.gg/badges/research.svg)](https://confluence.jetbrains.com/display/ALL/JetBrains+on+GitHub)
[![CI](https://github.com/JetBrains-Research/opake/actions/workflows/ci.yml/badge.svg)](https://github.com/JetBrains-Research/opake/actions/workflows/ci.yml)

**[Documentation](https://jetbrains-research.github.io/opake/)**

## Packages

Install and depend on `opake` only. The repository is implemented as
[PEP 420] namespace packages under the shared `opake.*` namespace:

| Distribution | Import roots | Purpose |
|---|---|---|
| `opake` | — | Convenience installer; pulls in a curated bundle of sub-packages |
| `opake-base` | `opake.serialization` | Pure-Python serialization registry + dispatcher; the seam every other wheel registers handlers against |
| `opake-engine` | `opake.{types,pytree,random,distributed,functional,scheduling,profiling}` | Torch substrate: pytree wrappers (`ClippedPytree` / `NoisedPytree` / `PerGroup`), `RngKey`, fixed + AUTO-S clipping, schedules + warmup, DDP plumbing, profiler |
| `opake-optimizers` | `opake.optimizers` | Torchopt-based functional optimizer chain (DP-aware AdamW-BC and friends) |
| `opake-dpsgd` | `opake.dpsgd` | Gaussian / per-group noise, Poisson samplers, adaptive clipping, DP-SGD-specific accounting factories |
| `opake-dpftrl` | `opake.dpftrl` | DP-FTRL mechanisms (BLT, BSR, BiSR, band-MF, λ-CGD), private second moments, correlated-noise samplers, DP-FTRL-specific accounting factories |
| `opake-auditing` | `opake.auditing` | Empirical privacy auditing (one-run, coin-flip, loss attacks) |
| `opake-patches` | `opake.patches` | Unified patching entrypoint for PyTorch checkpointing, Hugging Face compat wrappers, Triton kernels, and PEFT/LoRA fusion |
| `opake-transformers` | `opake.transformers` | Hugging Face trainer + integration; TRL-style `SFTTrainer` / `DPOTrainer` (`opake.transformers.trl`) built on `DPTrainer` |
| `opake-alignment` | `opake.alignment` | Functional, mechanism-agnostic DP-safe SFT / DPO primitives: per-example losses, log-prob helpers, collators, reference helpers, reward metrics |
| `opake-accounting` | `opake.accounting` | PLD privacy accounting (Rust/PyO3 backend); torch-free standalone |

[PEP 420]: https://peps.python.org/pep-0420/

### Import layout

```
opake.serialization                                       <- opake-base
opake.{types,pytree}                                      <- opake-engine
opake.{random,distributed}                                <- opake-engine
opake.{functional,scheduling,profiling}                   <- opake-engine
opake.optimizers                                          <- opake-optimizers
opake.dpsgd.{clipping,noise,sampling,accounting}          <- opake-dpsgd
opake.dpftrl.{clipping,noise,sampling,accounting}         <- opake-dpftrl
opake.auditing                                            <- opake-auditing
opake.patches.{kernels,torch,transformers,peft}           <- opake-patches
opake.transformers{,.trl}                                 <- opake-transformers
opake.alignment.{sft,dpo,data,metric}                     <- opake-alignment
opake.accounting                                          <- opake-accounting
```

## Installation

```bash
# From JetBrains Packages
pip install opake \
  --index-url https://packages.jetbrains.team/pypi/p/fed/python/simple/

# Or with uv
uv add opake \
  --index https://packages.jetbrains.team/pypi/p/fed/python/simple/
```

Extras (pass `--index-url` on every command — it does not persist from the
block above; without it, pip resolves from public PyPI, where the unrelated
`opaque` OPAQUE-PAKE wrapper also lives):

```bash
pip install "opake[auditing]" \      # empirical privacy auditing
  --index-url https://packages.jetbrains.team/pypi/p/fed/python/simple/
pip install "opake[dpftrl]" \        # correlated-noise DP-FTRL components
  --index-url https://packages.jetbrains.team/pypi/p/fed/python/simple/
pip install "opake[transformers]" \  # Hugging Face + patching components
  --index-url https://packages.jetbrains.team/pypi/p/fed/python/simple/
pip install "opake[all]" \           # all optional components
  --index-url https://packages.jetbrains.team/pypi/p/fed/python/simple/
```

### Patching

Hugging Face and checkpoint patches are applied explicitly through
`opake.patches`:

```python
from opake.patches import apply_model_patches, apply_runtime_patches

apply_runtime_patches()

# ... build / wrap the model, then patch the concrete instance
apply_model_patches(model)
```

`apply_runtime_patches()` enables the runtime-side checkpoint, collator, and
loss-mapping fixes. `apply_model_patches(model)` wires compat wrappers and
Triton kernels into the specific model instance, including PEFT/LoRA modules.

See the [model-patches guide](https://jetbrains-research.github.io/opake/latest/user-guide/huggingface/model-patches/)
for patching details, model compatibility, and tuning knobs.

## Example

A minimal DP-SGD training loop:

```python
import torch
import opake.accounting as acc                # cross-cutting (calibrate, budget)
import opake.dpsgd.accounting as dpsgd_acc    # DP-SGD per-step factories
from opake.dpsgd.clipping import clipped_grad
from opake.dpsgd.noise import gaussian_noise
from opake.random import key

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
    params = params - lr * noisy_grads.pytree  # or wire opake.optimizers
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
  gradient aggregation via `opake.distributed`.
- **Hugging Face compatibility**: automatic `vmap` patching for LLaMA, Mistral,
  Qwen2/3, Phi-3, Gemma/Gemma2, Granite, Cohere/Cohere2, plus fused Triton
  kernels via `opake.patches`.

## Documentation

- [Documentation](https://jetbrains-research.github.io/opake/)
- [Getting Started](https://jetbrains-research.github.io/opake/latest/getting-started/quickstart/)
- [User Guide](https://jetbrains-research.github.io/opake/latest/user-guide/)
- [Tutorials](https://jetbrains-research.github.io/opake/latest/tutorials/)
- [API Reference](https://jetbrains-research.github.io/opake/latest/reference/)
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
