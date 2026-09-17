# BSR (Banded Square Root)

**BSR** (Kalinin & Lampert, NeurIPS 2024) is a matrix factorization mechanism whose **strategy matrix** is a **banded lower-triangular Toeplitz** matrix obtained from the **matrix square root** of the paper's workload \(A_{\alpha,\beta}\) (multiplicative decay \(\alpha\), Polyak momentum \(\beta\)).

Opake exposes **closed-form coefficients** (Theorem 1 in the paper): no L-BFGS optimization step at initialization.

- **Paper**: [Banded Square Root Matrix Factorization for Differentially Private Model Training](https://arxiv.org/abs/2405.13763)
- **Strategy matrix**: Banded lower-triangular Toeplitz \(C^{|p|}_{\alpha,\beta}\)
- **Sensitivity**: Closed form under min-separation participation (Theorem 2; Rust `toeplitz_minsep_sensitivity_squared`)
- **Amplification**: Balls-in-Bins with Monte Carlo PLD (same Gram path as BLT/BISR Toeplitz Gram)

**API (paper-native):** `bsr_strategy(..., *, alpha=α, beta=β)` — keyword-only, **`alpha` before `beta`**. These are **not** PyTorch `AdamW(weight_decay=...)` or a generic `momentum` name from other MF strategies; they are exactly the paper’s \((\alpha,\beta)\). Training scripts should **bind** `beta` from SGD Polyak momentum or Adam’s \(\beta_1\) when you want the noise workload to match that optimizer choice.

## Accounting

```python
from opake.dpftrl.noise import bsr_strategy
import opake.accounting as acc           # cross-cutting balls_in_bins
import opake.dpftrl.accounting as dpftrl_acc  # DP-FTRL factories

strategy = bsr_strategy(
    bandwidth=8,
    n_steps=total_steps,
    min_sep=steps_per_epoch,
    max_participations=num_epochs,
    alpha=1.0,
    beta=0.95,
)

training = dpftrl_acc.balls_in_bins(
    dpftrl_acc.mf_gaussian(noise_multiplier, strategy),
    num_bins=steps_per_epoch,
    n_steps=steps_per_epoch * num_epochs,
)
eps = training.epsilon_at(1e-5)
```

### Strategy parameters

| Parameter | Description |
|-----------|-------------|
| `bandwidth` | Bandwidth \(p\) (≥ 1). Coefficients \(c_j\) for \(j \ge p\) are zero. |
| `n_steps` | Total training steps |
| `min_sep` | Minimum separation between participations (typically steps per epoch) |
| `max_participations` | Maximum number of participations per user (epochs) |
| `alpha` | Paper \(\alpha \in (0, 1]\) |
| `beta` | Paper \(\beta \in [0, 1)\); must satisfy \(\alpha > \beta\) |

### Accounting parameters

| Parameter | Description |
|-----------|-------------|
| `noise_multiplier` | Raw noise \(\sigma\) |
| `sensitivity` | From `strategy.sensitivity(n_steps=...)` |
| `gram_matrix` | From `strategy.gram_matrix` for BnB Monte Carlo |

## Noise generation

```python
from opake.dpftrl.noise import mf_gaussian_noise, bsr_strategy
from opake.random import key

strategy = bsr_strategy(
    bandwidth=8,
    alpha=1.0,
    beta=0.95,
)
noise_fn, state = mf_gaussian_noise(
    grad_template=params,
    strategy=strategy,
    n_steps=1000,
    min_sep=195,
    max_participations=8,
    noise_multiplier=noise_multiplier,
    key=key(seed),
)
```

## Assumptions and limitations

- **Closed-form regime only**: \(0 \leq \beta < 1\), \(0 < \alpha \leq 1\), \(\alpha > \beta\). Other hyperparameters raise `ValueError` with guidance to use `band_mf_strategy`.
- **`examples/train_dpftrl.py`**: workload **`--bsr-alpha`** → `alpha`; **`--momentum`** (SGD) or **`--beta1`** (Adam) → `beta`. Optimizer **`--weight-decay`** is separate (default `0.0`).
- **No learning-rate schedule**: BSR coefficients assume the paper’s workload; use BandMF/BLT with `lr_schedule` if you need schedule-shaped workloads in the optimizer.
- **vs BISR**: BISR bands the **inverse** square root of the workload (different coefficient family). BSR bands the **forward** square root factors from Theorem 1.
- **`normalized=True`**: Not currently supported.

## References

- Kalinin, Lampert (2024) — [arXiv:2405.13763](https://arxiv.org/abs/2405.13763)
