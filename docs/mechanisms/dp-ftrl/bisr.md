# BISR (Banded Inverse Square Root)

**BISR** (Kalinin et al., ICLR 2026) generalizes [DP-λCGD](lambda-cgd.md) from
bandwidth 2 to an arbitrary bandwidth \(p \geq 2\). The inverse of the strategy matrix
\(C^{-1}\) is a banded Toeplitz matrix whose coefficients are analytically determined
from the inverse square root of the workload matrix.

- **Paper**: [Back to Square Roots: An Optimal Bound on the Matrix Factorization Error for Multi-Epoch Differentially Private SGD](https://arxiv.org/abs/2505.12128)
- **Strategy matrix**: Inverse is banded Toeplitz with bandwidth p
- **Memory**: `min(p, n_steps) - 1` previous iid noise pytrees (bounded,
  but not memory-free)
- **Optimality**: Asymptotically optimal (matches upper and lower bounds)
- **Amplification**: Balls-in-Bins (BnB) with MC dominating pair accounting

## Accounting

Like all MF mechanisms, the noise **strategy** is a recipe. The amplifier
supplies the horizon and participation pattern when it asks the recipe for its
`sensitivity` and `gram_matrix`. This keeps those accounting-owned parameters
in one place.

```python
from opaque.dpftrl.noise import bisr_strategy
import opaque.accounting as acc           # cross-cutting balls_in_bins
import opaque.dpftrl.accounting as dpftrl_acc  # DP-FTRL factories

# 1. Create a strategy recipe
strategy = bisr_strategy(
    bandwidth=4,
    momentum=0.9,
)

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
| `bandwidth` | Number of bands p (≥ 2). Higher can improve utility, at the cost of more retained noise buffers and convolution work. |
| `momentum` | The paper's β in the implemented α=1 specialization. Must satisfy \(0 \leq \beta < 1\), the specialized form of Lemma 1's \(0 \leq \beta < \alpha \leq 1\) hypothesis. |
| `lr_schedule` | Deprecated compatibility argument; only `None` is accepted |

### Amplifier and accounting parameters

| Parameter | Description |
|-----------|-------------|
| `noise_multiplier` | Raw noise σ |
| `n_steps` | Total training steps; supplied by the amplifier |
| `num_bins` | Steps per epoch; determines minimum separation and participation count |

The amplifier derives the BISR sensitivity and BnB Gram matrix from these
values. For a signed or non-monotone custom `inv_coefficients` override,
unnormalized sensitivity majorizes the absolute forward coefficients;
normalized sensitivity majorizes their prefix-normalized envelope. These
bounds may be conservative, but remain safe.

### BISR coefficients

The inverse coefficients c̃_k are computed from the inverse square root
formula (Lemma 1 of the paper), specialized to \(\alpha=1\):

- Base sequence: r̃_0 = 1, r̃_j = ((j - 3/2) / j) · r̃_{j-1}
- With momentum β: c̃_k = Σ_{j=0}^{k} r̃_j · β^j · r̃_{k-j}

For β=0 (FTRL), the c̃ sequence starts with 1, -0.5, -0.125, and -0.0625.

Momentum enters the coefficient computation, changing the strategy matrix C
itself. The privacy analysis then uses the resulting C columns; momentum is
not applied a second time as a separate workload operator.

## Noise generation

```python
from opaque.dpftrl.noise import mf_gaussian_noise, bisr_strategy
from opaque.random import key

strategy = bisr_strategy(
    bandwidth=4,
    momentum=0.9,
)
noise_fn, state = mf_gaussian_noise(
    grad_template, strategy,
    n_steps=total_steps,
    min_sep=steps_per_epoch,
    max_participations=num_epochs,
    noise_multiplier=noise_multiplier,
    key=key(seed),
)
```

At step `t`, the noise function draws one fresh iid noise pytree and applies the
signed banded inverse directly:

\[
y_t = d_t \sum_{k=0}^{\min(t,p-1)} q_k z_{t-k}.
\]

Here `q_k` are the inverse coefficients, `z_t` is the fresh iid draw, and `d_t`
is `1` for unnormalized BISR or the finite-horizon column-normalization factor
for normalized BISR. Runtime state keeps only the newest
`min(p, n_steps) - 1` iid draws. It therefore uses
`O(min(p, n_steps) * model_size)` tensor storage and the same asymptotic work
per step, independent of the training horizon once `n_steps >= p`.

This bounded ring is the complete `O(p)` execution design required by
[issue #795](https://github.com/JetBrains-Research/opaque/issues/795). It is not
a zero-buffer design: a large model or bandwidth can still make the retained
pytrees significant. PRNG replay could trade that persistent tensor storage for
extra noise generation, but replay and a reusable generic banded-inverse
executor are follow-up directions rather than requirements for this bounded
BISR path.

The full-horizon forward-substitution operator remains available through
`BisrStrategy.streaming_matrix()` as a numerical reference, but the public
noise path does not allocate its `n_steps - 1` model-shaped history.

!!! warning "Checkpoint layout change"

    BISR runtime checkpoints written before bounded-state support contain past
    correlated outputs, while current checkpoints contain a versioned window of
    past iid draws. These layouts are not safely interchangeable. Restoring a
    standalone legacy BISR noise state fails with a targeted BISR checkpoint
    error; a full trainer checkpoint may instead fail first at its outer bundle
    version. Resume either form with the Opaque version that created it. This
    rejection is intentional; the bounded-state change does not attempt a
    legacy-state migration.

    Exact continuation from a bounded-layout checkpoint requires the same BISR
    execution identity and base noise scale as the original run, in addition to
    the saved iid window. Open
    [issue #789](https://github.com/JetBrains-Research/opaque/issues/789) tracks
    the separate, urgent problem where calibrated DP-FTRL resume can rebuild a
    mechanism with a different noise multiplier. This bounded-state change does
    not fix or relax that resume requirement.

## Assumptions and limitations

- **BnB sampling**: pair with a sampler and accounting consistent with Balls-in-Bins (fixed partition semantics where required).
- **Momentum** enters the **inverse** coefficient construction (Lemma 1); sensitivity and Gram use the resulting strategy matrix.
- **Learning-rate schedules are optimizer-only**: they do not change BISR's encoder or noise and therefore must not weight its privacy Gram. The temporary `lr_schedule` compatibility argument rejects non-`None` values; pass the schedule only to the optimizer.
- **Not BSR**: BISR bands the **inverse** square-root construction (generalized λCGD). [BSR](bsr.md) uses the **forward** square-root closed form for SGD with momentum and weight decay.
- For a high-level comparison of MF mechanisms, see [Matrix factorization (MF)](../../user-guide/dp-ftrl.md).

## Bandwidth selection

Bandwidth `p` retains up to `p - 1` noise vectors. That makes memory independent
of a longer training horizon, not independent of model size or bandwidth.
Measure the runtime and utility trade-off on your workload.
