# Gaussian Mechanism

The standard noise mechanism for differential privacy. Adds independent
Gaussian noise $\mathcal{N}(0, \sigma^2)$ with unbounded support to
sensitivity-1 queries. This is the default mechanism for DP-SGD and the
building block on which the bounded variants improve.

## Idea

Given a function $f$ with sensitivity $\Delta$ (the maximum change in
$f$ when one training example is added or removed), adding noise from
$\mathcal{N}(0, \sigma^2)$ where $\sigma = \text{noise\_multiplier}
\times \Delta$ makes the output $(\varepsilon, \delta)$-differentially
private. The ratio $\sigma / \Delta$ is the **noise multiplier** — larger
values mean more privacy (lower $\varepsilon$) but more gradient corruption.

**Advantages**: Simple, well-understood, no extra parameters beyond the
noise multiplier.

**Limitation**: Unbounded support means the worst-case privacy loss is
technically infinite; the $\delta$ parameter bounds the probability of
extreme privacy loss. The bounded variant (`gaussian_noise(bound=...)`)
eliminates this tail risk.

## Mathematics

### Noise distribution

$$f(x; \mu) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

where $\mu$ is the true query answer (0 for the "null" dataset, $\Delta$
for the neighboring dataset).

### Privacy loss function

The privacy loss random variable at point $x$ is:

$$\ell(x) = \log \frac{f(x; 0)}{f(x; \Delta)} = \frac{\Delta}{\sigma^2}\!\left(\frac{\Delta}{2} - x\right)$$

This is linear in $x$ with slope $-\Delta/\sigma^2$.

### Hockey-stick divergence

The $(\varepsilon, \delta)$ guarantee is characterized by:

$$\delta(\varepsilon) = \Phi\!\left(\frac{\Delta}{2\sigma} - \frac{\varepsilon\sigma}{\Delta}\right) - e^{\varepsilon}\,\Phi\!\left(-\frac{\Delta}{2\sigma} - \frac{\varepsilon\sigma}{\Delta}\right)$$

where $\Phi$ is the standard normal CDF. This has a closed-form expression,
making the Gaussian mechanism the most analytically tractable.

## Supported amplifications

### Poisson subsampling (`poisson`)

The standard DP-SGD amplification. Each example is included independently
with probability $q = \text{batch\_size} / \text{dataset\_size}$.

**What it means**: instead of applying the Gaussian mechanism to the full
dataset, we apply it to a random $q$-fraction. The privacy loss of the
mixture is:

$$L_{\text{rem}}(x) = \log\!\left(1 - q + q \cdot e^{\ell(x)}\right)$$

For small $q$, this gives roughly $1/q$ amplification — a sample rate of 1%
makes the mechanism behave as if noise were 100$\times$ larger.

```python
step = dpsgd_acc.poisson(dpsgd_acc.gaussian(1.0), sample_rate=0.01)
training = step * 1000
eps = training.epsilon_at(delta=1e-5)
```

### Truncated Poisson subsampling (`poisson` with a batch cap)

Production variant that caps batch size. Capping stabilizes batches and
memory but **weakens** privacy versus plain Poisson at the same
`sample_rate` unless noise is recalibrated—use `truncated_batch_size` and
`dataset_size` together with `opaque.dpsgd.accounting.poisson`.

```python
n = 50_000
batch = 256
step = dpsgd_acc.poisson(
    dpsgd_acc.gaussian(1.0),
    batch / n,
    truncated_batch_size=batch,
    dataset_size=n,
)
training = step * 1000
eps = training.epsilon_at(delta=1e-5)
```

### Parallel Poisson subsampling (`parallel_poisson`)

For multi-worker training where each worker independently draws a Poisson
sample. A given example may appear on multiple workers.

```python
step = dpsgd_acc.parallel_poisson(
    dpsgd_acc.gaussian(1.0), sample_rate=0.01, num_workers=4,
)
```

## Code examples

### Noise injection

```python
from opaque.dpsgd.noise import gaussian_noise
from opaque.random import key

noise_fn, noise_state = gaussian_noise(
    noise_multiplier=noise_multiplier,
    key=key(42),
)

for batch in dataloader:
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    params = params - lr * noisy_grads
```

### Privacy accounting

```python
import opaque.accounting as acc
import opaque.dpsgd.accounting as dpsgd_acc

# Single step
step = dpsgd_acc.poisson(dpsgd_acc.gaussian(1.0), sample_rate=0.01)

# Composed over training
training = step * 1000

# Query multiple metrics
eps   = training.epsilon_at(delta=1e-5)
delta = training.delta_at(epsilon=3.0)
adv   = training.advantage()
beta  = training.beta_at(alpha=0.01)
```

### Calibration

```python
result = acc.calibrate(
    acc.epsilon_budget(3.0, delta=1e-5),
    lambda nm: dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), sample_rate=0.01) * 1000,
    param_min=0.1,
    param_max=10.0,
)
noise_multiplier = result.param
print(f"σ/Δ = {noise_multiplier:.4f}, achieved ε = {result.achieved:.4f}")
```

## Parameter guide

| Parameter | Range | Effect |
|-----------|-------|--------|
| `noise_multiplier` | 0.1 – 10.0 (after calibration) | Higher = more privacy, more noise. Calibrate to target $\varepsilon$. |

**Tips**:

- Start with `acc.calibrate()` — don't guess the noise multiplier.
- For a given privacy budget, increasing the sample rate (larger batches)
  or reducing the number of steps lets you use less noise per step.
- If $\varepsilon > 10$ after calibration, the privacy guarantee is weak.
  Consider more noise, fewer steps, or a larger dataset.

## Bounded noise variant

Pass `bound=B` (or `bound=(low, high)`) to `gaussian_noise()` to sample
from a Gaussian renormalized over $[-B, B]$ (or $[\text{low}, \text{high}]$)
per coordinate — the *bounded Gaussian mechanism* of
[Chen and Hale (2024)](https://arxiv.org/abs/2211.17230).
Bounds are absolute (same scale as the gradient / clip norm), not multiples
of $\sigma$. Treat `bound=` as experimental: `dpsgd_acc.gaussian()` does not
cover this variant.

```python
from opaque.dpsgd.noise import gaussian_noise
from opaque.random import key

# Noise injection: bounded support, symmetric absolute bound
noise_fn, noise_state = gaussian_noise(
    noise_multiplier=noise_multiplier, bound=3.0, key=key(42),
)
noisy_grads, noise_state = noise_fn(grads, noise_state)

# Asymmetric bound:
noise_fn, noise_state = gaussian_noise(
    noise_multiplier=noise_multiplier, bound=(-1.0, 4.0), key=key(42),
)
```

## References

- **Abadi et al. (2016)** — [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133).
  Introduced DP-SGD with the Gaussian mechanism.
- **Balle, Barthe, Gaboardi (2018)** — [Privacy Amplification by Subsampling: Tight Analyses via Couplings and Divergences](https://arxiv.org/abs/1807.01647).
  Poisson subsampling amplification analysis.
- **Chen and Hale (2024)** — [The Bounded Gaussian Mechanism for Differential Privacy](https://arxiv.org/abs/2211.17230).
  Introduces the bounded Gaussian mechanism.

For a jointly noised clean aggregate square, see
[Projected JME](jme.md).
