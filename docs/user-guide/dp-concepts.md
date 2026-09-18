# Differential Privacy Concepts

This page covers the theory behind differential privacy (DP) and DP-SGD at the
level needed to use Opake effectively. For the API itself, see the topic-specific
guides linked throughout.

## What differential privacy guarantees

Differential privacy is a mathematical property of a *mechanism* (an algorithm
that takes a dataset and produces an output). A mechanism $\mathcal{M}$ is
$(\varepsilon, \delta)$-differentially private if, for any two datasets $D$ and $D'$ that
differ in a single record, and for any set of outputs $S$:

$$P[\mathcal{M}(D) \in S] \leq e^{\varepsilon} \cdot P[\mathcal{M}(D') \in S] + \delta$$

Informally: the output distribution barely changes when one person's data is
added or removed. An adversary observing the output cannot confidently determine
whether any specific individual was in the training set.

**$\varepsilon$** (privacy loss) controls how much the distributions can differ. Smaller
$\varepsilon$ means stronger privacy. At $\varepsilon=0$, the mechanism reveals nothing
about individuals; at $\varepsilon=\infty$, there is no privacy guarantee.

**$\delta$** bounds the probability that the $\varepsilon$ guarantee fails. It should be
cryptographically small, typically smaller than $1/n$ where $n$ is the dataset size.
Common values: $10^{-5}$ to $10^{-6}$.

### Practical interpretation

| $\varepsilon$ | Interpretation |
|---------|----------------|
| 0.1     | Strong privacy. Significant accuracy loss expected. |
| 1.0     | Moderate privacy. Reasonable accuracy on large datasets. |
| 3.0     | Pragmatic privacy. Common target for ML training. |
| 10.0    | Weak privacy. Some protection against memorization. |

These are rough guidelines. The actual privacy-utility trade-off depends on the
dataset size, model capacity, and training procedure.

## The DP-SGD algorithm

DP-SGD ([Abadi et al. 2016](https://arxiv.org/abs/1607.00133)) modifies
standard stochastic gradient descent to provide differential privacy. The key
insight is that if we bound the influence of each training example on the
gradient update and add enough noise, the resulting model satisfies DP.

### Standard SGD vs DP-SGD

Standard SGD computes the average gradient over a batch:

```
gradient = mean([grad(loss, x_i) for x_i in batch])
params = params - lr * gradient
```

One outlier example can produce a large gradient that dominates the update.
Removing that person's data would noticeably change the trained model,
violating privacy.

DP-SGD adds two operations: **clipping** and **noise**.

```
per_example_grads = [grad(loss, x_i) for x_i in batch]
clipped_grads = [clip(g, max_norm=C) for g in per_example_grads]
noisy_gradient = sum(clipped_grads) + N(0, sigma^2 * I)
params = params - lr * noisy_gradient
```

### Step 1: Per-example gradients

Each training example produces its own gradient vector. This is more expensive
than standard batch-gradient computation but necessary for bounding per-example
influence.

Opake computes per-example gradients efficiently using `torch.func.vmap` and
`torch.func.grad`. See [Gradient Clipping](clipping.md) for details.

### Step 2: Clipping

Each per-example gradient is clipped to a maximum $\ell_2$ norm $C$ (the *clip norm*).
If the gradient's norm exceeds $C$, it is scaled down to have norm exactly $C$.
Gradients with norm below $C$ are left unchanged.

Clipping bounds the *sensitivity* of the gradient query: changing one training
example changes the sum of clipped gradients by at most $C$ (under add-or-remove
neighboring relation) or $2C$ (under replace-one).

### Step 3: Noise addition

Gaussian noise with standard deviation $\sigma_{\text{abs}}$ is added to the
sum of clipped gradients. The absolute noise scale is proportional to the
sensitivity (the clip norm): for clip norm $C$ and noise multiplier $\sigma$,
$\sigma_{\text{abs}} = \sigma C$.

The dimensionless ratio $\sigma = \sigma_{\text{abs}} / C$ is the *noise
multiplier*. Larger noise multiplier means stronger privacy (smaller
$\varepsilon$) but more gradient corruption.

### Step 4: Parameter update

The noisy gradient is applied to update model parameters, exactly as in
standard SGD. Any optimizer (SGD, Adam, AdamW) can be used.

## Privacy budget and composition

### The budget is finite

Each DP-SGD step consumes some privacy budget. Over $T$ training steps, the total
privacy cost composes. More steps means larger total $\varepsilon$ for the same noise
level, or more noise needed to maintain the same $\varepsilon$.

This creates a fundamental trade-off: training longer improves model quality
but costs more privacy. The **calibration** step (see
[Privacy Accounting](accounting.md)) finds the noise level that achieves a
target epsilon for a given number of steps.

### Composition

When the same dataset is used for multiple DP mechanisms (e.g., $T$ steps of
DP-SGD), the total privacy loss is bounded by *composition theorems*.

**Basic composition**: $\varepsilon$ values add linearly. $T$ steps of
$\varepsilon_0$-DP gives $T \cdot \varepsilon_0$ total privacy loss. For the
Gaussian mechanism, the single-step privacy cost is controlled by the **noise
multiplier** $\sigma = \sigma_{\text{abs}} / C$: increasing $\sigma$ decreases
$\varepsilon_0$. Holding $\sigma$ fixed keeps the per-step guarantee unchanged even
if you later rescale both the clip norm and the absolute noise standard
deviation together. The linear $T \cdot \varepsilon_0$ bound is correct but
extremely loose.

**Advanced composition** ([Dwork, Rothblum, and Vadhan 2010](https://theory.stanford.edu/~salil/papers/compose-private.pdf)):
for $T$ applications of an $(\varepsilon_0, \delta_0)$-DP mechanism, the total
privacy satisfies $(\varepsilon, T\delta_0 + \delta)$-DP where:

$$\varepsilon = \sqrt{2T \ln(1/\delta)} \cdot \varepsilon_0 + T \cdot \varepsilon_0 (e^{\varepsilon_0} - 1)$$

For small $\varepsilon_0$, the total grows as $\sqrt{T}$ rather than $T$ — a
significant improvement. But it still uses worst-case per-step bounds.

**PLD composition**: Opake uses *Privacy Loss Distributions* (PLD), which
track the full probability distribution of the privacy loss random variable

$$\ell(o) = \ln \frac{P[\mathcal{M}(D) = o]}{P[\mathcal{M}(D') = o]}$$

rather than just worst-case bounds. Composing two mechanisms corresponds to
convolving their PLDs. This is numerically tight — it gives the exact
privacy guarantee up to discretization error — and dominates both basic and
advanced composition. In practice, PLD composition gives 2-5x tighter
$\varepsilon$ than advanced composition for typical DP-SGD training runs.

The `*` and `|` operators on `DpProcess` objects compute PLD composition
in Opake's Rust PLD engine. See [Privacy Accounting](accounting.md).

### Subsampling amplification

If each training step samples a random subset of the dataset (rather than using
the full dataset), the privacy cost per step is reduced. This is *privacy
amplification by subsampling*.

With Poisson sampling at rate $q$ (each example included independently with
probability $q$), the effective noise multiplier is amplified by approximately
$1/q$. For $q=0.01$ (1% sample rate), this is a 100x amplification.

Opake supports several subsampling schemes:

| Scheme | Description | Use case |
|--------|-------------|----------|
| **Poisson** | Each example included independently with probability $q$ | Standard DP-SGD. Variable batch size. |
| **Truncated Poisson** | Poisson draw capped at a maximum batch size | DP-SGD when you want stable batch sizes; privacy is weaker than plain Poisson at the same $q$. |
| **Cyclic Poisson (DP-FTRL)** | ``opake.dpftrl.sampling.CyclicPoissonSampler``: ``bands`` disjoint groups, step ``i`` uses group ``i % bands``, inclusion prob. ``q`` per eligible example. ``bands=1`` is identity (full data each step); larger ``bands`` match BandMF-style rotation. | ``mf_gaussian_noise`` + ``dpftrl_acc.poisson`` (whole-process accountant). |

The key distinction is between *Poisson* and *fixed-size* sampling. Poisson
sampling produces variable-size batches but has a clean privacy analysis.
Fixed-size sampling (drawing exactly $B$ examples) has a slightly different
privacy profile. Opake uses Poisson-style sampling throughout.

Truncated Poisson keeps Poisson-style randomness but caps realized batch size,
which stabilizes memory and batch norms at the cost of **weaker** privacy than
unconditional Poisson at the same inclusion probability $q$ (use the
truncated-Poisson accountant).

Opake's `PoissonSampler` implements Poisson subsampling. The accounting module
accounts for this amplification via `dpsgd_acc.poisson(mechanism, sample_rate)`.
See [Sampling & Microbatching](sampling.md) and the
[Mechanisms](../mechanisms/index.md) reference for per-mechanism amplification
details.

## Privacy metrics

Opake supports four families of privacy metrics, all derived from the same
underlying PLD. Different metrics suit different audiences and use cases.

### ($\varepsilon$, $\delta$)-DP

The standard DP definition. Given a target $\delta$, compute the smallest $\varepsilon$
such that the mechanism is $(\varepsilon, \delta)$-DP.

```python
eps = training.epsilon_at(delta=1e-5)
```

This is the most widely reported metric in the DP literature. Use it for
compliance reporting and comparison with published results.

### f-DP and the trade-off function

f-DP ([Dong et al. 2019](https://arxiv.org/abs/1905.02383)) characterizes
privacy via a *trade-off function* $f$. For neighboring datasets $D, D'$ and
any hypothesis test $\phi$ that distinguishes between them:

$$f(\alpha) = \inf_\phi \{ \beta_\phi : \alpha_\phi \leq \alpha \}$$

where $\alpha_\phi$ is the Type-I error (false positive) and $\beta_\phi$ is
the Type-II error (false negative). The function $f$ traces the best possible
ROC curve an adversary can achieve. f-DP is strictly more informative than
$(\varepsilon, \delta)$-DP: you can derive $(\varepsilon, \delta)$ from $f$
but not vice versa.

**Advantage** is a scalar summary of the trade-off function:

$$\text{Adv} = \sup_\phi \left| P[\phi(o)=1 \mid D] - P[\phi(o)=1 \mid D'] \right|$$

An advantage of 0 means the adversary cannot distinguish at all (perfect
privacy); an advantage of 1 means perfect distinguishing.

```python
adv = training.advantage()
```

### ($\alpha$, $\beta$) error rates

The hypothesis-testing interpretation. Given a Type-I error rate $\alpha$ (false
positive rate), compute the Type-II error rate $\beta$ (false negative rate) for
the optimal distinguishing test.

Higher $\beta$ means stronger privacy: the adversary cannot reject the null
hypothesis (that the data was not in the training set) without high false
negative rates.

```python
beta = training.beta_at(alpha=0.01)
```

This metric is useful for understanding the operational meaning of a privacy
guarantee: "an adversary accepting 1% false positives will miss at least
$\beta$% of true members."

### Bayes risk

Given a prior probability $\pi$ that a record is in the dataset, the Bayes
risk measures the adversary's expected error under the optimal decision rule:

```python
risk = training.risk_at(prior=0.5)
```

A risk of 0.5 means the adversary does no better than random guessing.
This metric is natural for decision-theoretic reasoning about privacy.

### Choosing a metric

| Metric | Best for | Opake method |
|--------|----------|---------------|
| $(\varepsilon, \delta)$-DP | Compliance, published comparisons | `.epsilon_at(delta)` |
| Advantage | Quick scalar privacy summary | `.advantage()` |
| $(\alpha, \beta)$ | Understanding operational privacy | `.beta_at(alpha)` |
| Bayes risk | Decision-theoretic analysis | `.risk_at(prior)` |

All four metrics are derived from the same PLD, so they are mutually
consistent. You can query all of them from the same `DpProcess` object.

## Key trade-offs in DP training

### Privacy vs accuracy

More noise means stronger privacy but degrades gradient signal. The model may
converge to a worse solution or not converge at all.

Strategies to improve accuracy at fixed privacy budget:

- **Increase batch size.** Larger batches give stronger subsampling amplification
  (smaller $q$) and more gradient signal per unit of noise. The total noise added
  per step is $\sigma \cdot C$ regardless of batch size, but it is averaged over
  more examples. Physical batch sizes of 1000+ are common in DP training.
- **Use correlated noise (DP-FTRL).** Matrix factorization mechanisms (BandMF,
  BLT) inject correlated noise that partially cancels across steps,
  reducing the effective noise on cumulative model updates.
- **Use adaptive clipping** to avoid over-clipping gradients.
- **Use LoRA** or other parameter-efficient methods to reduce gradient
  dimensionality, concentrating the noise budget on fewer parameters.

### Epoch budget

The total number of training steps determines the privacy cost:

$$T = \text{epochs} \times \lceil n / B \rceil$$

where $n$ is the dataset size and $B$ is the (expected) batch size. For Poisson
sampling, $B = q \cdot n$, so $T = \text{epochs} / q$. More epochs or smaller
batches (smaller $q$) means more steps and higher privacy cost.

### Privacy vs compute

Per-example gradients are more expensive than batch gradients. Opake uses
`torch.func.vmap` for efficient vectorized computation. Microbatching trades
compute time for memory by processing the batch in smaller chunks.

### Clip norm

The clip norm C controls the sensitivity-noise trade-off:
- Higher C: less clipping distortion, but more noise needed for the same privacy
- Lower C: more clipping distortion, but less noise needed

A common heuristic is to set C to the median gradient norm observed during
non-private training. Opake's `adaptive_clipped_grad` automates this by
adjusting C to maintain a target fraction of clipped gradients (typically the
median, i.e., 50%).

### Number of steps

More training steps consume more privacy budget. For a fixed budget, longer
training requires more noise per step, which can reduce accuracy. There is
often an optimal number of steps that balances convergence with noise level.

## Noise mechanisms

The choice of noise mechanism affects both the privacy guarantee and the noise
impact on model quality.

### Independent noise (DP-SGD)

Standard DP-SGD adds independent Gaussian noise at each training step. See
[Mechanisms](../mechanisms/index.md) for the mathematical details.

### Correlated noise (DP-FTRL)

Matrix factorization (MF) mechanisms add *correlated* noise across training
steps. Instead of independent noise $z_t$ at each step, the noise is
generated as $z = B \cdot \xi$ where $B$ is a lower-triangular strategy
matrix and $\xi$ is i.i.d. Gaussian. The correlations are designed so that
the effective noise on the *cumulative* model update is smaller than what
independent noise would give.

Three MF strategies are available:

| Strategy | Memory | Best for |
|----------|--------|----------|
| **BandMF** | $O(b)$ | Streaming, long training runs |
| **BLT** | $O(b)$ | Multi-epoch training |

MF mechanisms use ``opake.dpftrl.sampling.CyclicPoissonSampler`` (and other FTRL
samplers) with amplification that depends on the mechanism; identity runs use
``bands=1``. See the
[Mechanisms](../mechanisms/index.md) reference for details.

## Neighboring relations

The privacy guarantee depends on what "differ in one record" means:

| Relation | Meaning | Sensitivity |
|----------|---------|-------------|
| Add or remove | $D' = D \pm$ one record | $C$ |
| Replace one | $D' = D$ with one record swapped | $2C$ |

Opake uses the **add-or-remove** convention: clipped outputs carry
`grads.max_norm = C / normalize_by`. When `normalize_by` is set to the expected
batch size $B$, the bound is $C/B$. If your analysis uses replace-one
semantics, double the bound when calibrating noise.

## References

- [Abadi et al. 2016 - Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133)
- [Dong et al. 2019 - Gaussian Differential Privacy](https://arxiv.org/abs/1905.02383)
- [Dwork, Rothblum, and Vadhan 2010 - Boosting and Differential Privacy](https://theory.stanford.edu/~salil/papers/compose-private.pdf)
- [Kairouz et al. 2015 - The Composition Theorem for Differential Privacy](https://arxiv.org/abs/1311.0776)
- [Balle et al. 2020 - Hypothesis Testing Interpretations and Renyi Differential Privacy](https://arxiv.org/abs/1905.09982)
- [Andrew et al. 2021 - Differentially Private Learning with Adaptive Clipping](https://arxiv.org/abs/1905.03871)
- [Koskela et al. 2020 - Computing Tight DP Guarantees Using FFT](https://arxiv.org/abs/1906.03049)
- [Denisov et al. 2022 - Improved Differential Privacy for SGD via Optimal Private Linear Operators on Adaptive Streams](https://arxiv.org/abs/2202.08312)
