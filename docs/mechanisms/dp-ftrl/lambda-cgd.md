# DP-λCGD

**DP-λCGD** (Kalinin et al., 2026) is a correlated noise mechanism that uses a
single parameter \(\lambda \in [0, 1)\) to control noise correlation across training steps.
Its key advantage is **zero extra memory** — noise correlation is achieved via
PRNG seed replay instead of storing previous noise vectors.

- **Paper**: [DP-λCGD: Efficient Noise Correlation for Differentially Private Model Training](https://arxiv.org/abs/2601.22334)
- **Strategy matrix**: Lower-triangular Toeplitz with entries \(C_{i,j} = \lambda^{i-j}\)
- **Inverse**: Bidiagonal — 1 on the diagonal, \(-\lambda\) on the subdiagonal (bandwidth 2)
- **Memory**: Zero extra (PRNG replay regenerates previous noise from its seed)
- **Amplification**: Balls-in-Bins (BnB) with MC dominating pair accounting

## Accounting

The noise **strategy** and the privacy **accounting** are connected through two
values that the strategy computes: `sensitivity` and `gram_matrix`. The strategy
is a recipe for C; the amplifier supplies the participation pattern and horizon
when it derives those values.

This separation keeps accounting constructors simple and avoids duplicating
mechanism parameters in two places.

```python
from opaque.dpftrl.noise import lambda_cgd_strategy
import opaque.accounting as acc           # cross-cutting balls_in_bins
import opaque.dpftrl.accounting as dpftrl_acc  # DP-FTRL factories

# 1. Create a strategy recipe
strategy = lambda_cgd_strategy(lambda_=0.9)

# 2. The amplifier supplies n_steps and the participation geometry
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
| `lambda_` | Correlation coefficient in \([0, 1)\). \(\lambda = 0\) is DP-SGD. |
| `normalized` | Column-normalize C (default True, gives sensitivity=1 for k=1) |
| `lr_schedule` | Deprecated compatibility argument; only `None` is accepted |

### Amplifier and accounting parameters

| Parameter | Description |
|-----------|-------------|
| `noise_multiplier` | Raw noise σ (calibrated or fixed) |
| `n_steps` | Total training steps; supplied by the amplifier |
| `num_bins` | Steps per epoch; determines minimum separation and participation count |

### Sensitivity

The sensitivity and Balls-in-Bins Gram depend only on the strategy matrix C,
not on the optimizer workload. Learning-rate schedules belong to the optimizer
and are deterministic post-processing of the private mechanism; they must not
weight the privacy Gram. For the unnormalized Toeplitz strategy, Theorem 1 and
Appendix C, equations (15)-(16) of arXiv v1 give the closed form in terms of λ,
`min_sep`, and `max_participations`. The normalized default is not Toeplitz; its
sensitivity is covered separately by Lemma 8, equation (12) of the current
arXiv v2 and is computed from normalized column inner products.

## Assumptions and limitations

- Bandwidth is **fixed** (bidiagonal inverse); correlation is controlled by a single \(\lambda\). Does **not** accept `momentum` (use `bisr_strategy` with bandwidth > 2 for momentum-aware coefficients).
- Learning-rate schedules are not strategy parameters. The temporary `lr_schedule` compatibility argument rejects non-`None` values; pass the schedule only to the optimizer.
- Uses **Balls-in-Bins** amplification like other epoch-structured MF mechanisms; sampler semantics must match accounting.
- **Private second moments (DP-Adam)**: auto-deriving the second-moment strategy is **not supported** for λCGD. Pass `second_moment_strategy` explicitly.
- Broader MF context: [Correlated noise (DP-FTRL)](../../user-guide/dp-ftrl.md).

## Noise generation

```python
from opaque.dpftrl.noise import mf_gaussian_noise, lambda_cgd_strategy
from opaque.random import key

strategy = lambda_cgd_strategy(lambda_=0.9)
noise_fn, state = mf_gaussian_noise(
    grad_template, strategy,
    n_steps=total_steps,
    min_sep=steps_per_epoch,
    max_participations=num_epochs,
    noise_multiplier=noise_multiplier,
    key=key(seed),
)
```

At each step t, the noise function:
1. Generates z_t from step t's PRNG seed
2. Regenerates z_{t-1} from step t-1's PRNG seed (PRNG replay)
3. Computes correlated noise: n_t = z_t - λ · z_{t-1}
4. Optionally applies column-norm scaling: n_t *= d_t

## Relationship to BISR

DP-λCGD is the bandwidth-2 special case of [BISR](bisr.md). For bandwidth > 2,
use `bisr_strategy(...)`, which generalizes the correlation structure.
