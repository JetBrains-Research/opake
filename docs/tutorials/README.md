# Tutorials

These Jupyter notebooks provide hands-on practice with Opake. Each is
self-contained: you build something concrete and see it work.

For conceptual explanations, see the [User Guide](../user-guide/index.md).
For complete function signatures, see the
[API Reference](../reference/index.md).

## DP-SGD track

Independent per-step Gaussian noise. The classical DP training pipeline.

| Tutorial | Task |
| -------- | ---- |
| [DP-SGD Training](dp_sgd_training.ipynb) | Train a model with DP-SGD from scratch — the canonical reference. |
| [Privacy Accounting & Calibration](accounting_and_calibration.ipynb) | Compose mechanisms, calibrate the noise multiplier, query ε at the end. |
| [Sampling & Microbatching](sampling_and_microbatching.ipynb) | Plain vs truncated Poisson, microbatch size, the privacy/memory trade-off. |
| [Fine-tuning an LLM](llm_finetuning.ipynb) | LoRA + adaptive clipping + truncated Poisson on a real Hugging Face model. |

## DP-FTRL track

Correlated noise across the run via matrix factorization. Better
privacy/utility on cumulative updates, at the cost of fixing the
training length up front.

| Tutorial | Task |
| -------- | ---- |
| [DP-FTRL Training](dp_ftrl_training.ipynb) | Train a model with DP-FTRL — strategy choice, whole-process accounting, MF noise. |

The DP-SGD-track accounting and sampling notebooks share their core
ideas with DP-FTRL; the strategy-and-`n_steps`-up-front difference is
covered explicitly in the DP-FTRL training tutorial above.

## Cross-cutting

Concerns that apply to either stack.

| Tutorial | Task |
| -------- | ---- |
| [Privacy Auditing](privacy_auditing.ipynb) | Validate privacy guarantees empirically with one-run / coin-flip auditing. |
| [Distributed Training](distributed_training.ipynb) | Multi-GPU DP training under DDP — synchronized noise, sharded sampling. |

## Running the tutorials

Install Opake and its tutorial dependencies from a repository checkout:

```bash
git clone https://github.com/JetBrains-Research/opaque.git
cd opake
uv sync --group examples --all-packages --extra all
jupyter lab docs/tutorials/
```

The `examples` group ships Jupyter, matplotlib, torchvision, torchopt,
datasets, and wandb; HuggingFace integration comes via `--extra all`
(or `--extra transformers`). The distributed-training tutorial requires
multiple processes via `torchrun`.
