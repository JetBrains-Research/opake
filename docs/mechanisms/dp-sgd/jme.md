# Projected Joint Moment Estimation

Projected Joint Moment Estimation (JME) privately releases the normalized
clipped gradient aggregate and its clean element-wise square for optimizers such
as Adam:

\[
a_t=\frac{1}{z}\sum_{i\in S_t}\operatorname{clip}(g_{t,i},C),\qquad
x_t=\Pi_R(a_t),\qquad q_t=x_t\odot x_t.
\]

Here `z` and the projection radius `R` are fixed public values.
`normalize_by=z` on `clipped_grad` produces \(a_t\); `jme_noise` performs the
global projection, forms \(q_t\), and noises both values. This is different from
\(\sum_i \operatorname{clip}(g_i,C)^2/z\), which is a mean of per-record
squares rather than Adam's square of the aggregate.

## Usage

```python
from opaque.dpsgd.clipping import clipped_grad
from opaque.dpsgd.noise import jme_noise
from opaque.dpsgd.noise.types import JmeAllocation
from opaque.random import key

grad_fn, clip_state = clipped_grad(
    loss_fn,
    clipping_norm=1.0,
    normalize_by=expected_batch_size,
)
noise_fn, noise_state = jme_noise(
    noise_multiplier=noise_multiplier,
    aggregate_norm=1.0,
    allocation=JmeAllocation.first_variance_cap(1.5),
    key=key(42),
)

grads, clip_state = grad_fn(params, batch, state=clip_state)
# In distributed training, sum `grads` across ranks here.
moments, noise_state = noise_fn(grads, noise_state)
updates, opt_state = optimizer.update(moments, opt_state, params=params)
```

`moments` is a `SecondMomentNoiseOutput`. Its `noisy_grads` field contains the
noised projected aggregate; `noisy_squared_grads` contains the separately
noised clean square. Optimizers route the latter directly into their second
moment and do not subtract first-stream noise variance in this branch.

## Add/remove sensitivity

Under record-level add/remove adjacency, fixed per-record clipping and fixed
normalization give

\[
\Delta=C/z.
\]

Projection onto the L2 ball is non-expansive. Neighboring projected aggregates
\(x,y\) therefore satisfy

\[
\lVert x\rVert_2,\lVert y\rVert_2\le R,\qquad
\lVert x-y\rVert_2\le D:=\min(\Delta,2R).
\]

For model dimension \(d\ge2\), the exact maximum squared displacement of the
diagonal square query at fixed \(r=\lVert x-y\rVert_2\) is

\[
M_R(r)=
\begin{cases}
r^2(2R-r)^2,&0\le r\le 2R/3,\\
2r^2(R^2-r^2/4),&2R/3\le r\le2R.
\end{cases}
\]

The first branch is attained by \(x=Re_1,\ y=(R-r)e_1\); the second is attained
by a two-coordinate pair with both norms equal to \(R\). For \(d=1\), only the
first expression applies.

For a positive allocation parameter \(\lambda\), define

\[
E(D,\lambda)=D^2+\lambda M_R(D).
\]

For \(d\ge2\), the exact joint sensitivity squared of
\((x,\sqrt{\lambda}(x\odot x))\) is

\[
S_\lambda^2=
\begin{cases}
2R^2+2\lambda R^4+\frac{1}{2\lambda},
&D^2>2R^2+\frac1\lambda,\\
E(D,\lambda),&\text{otherwise}.
\end{cases}
\]

In the first case the maximum occurs at
\(r=\sqrt{2R^2+1/\lambda}\). The one-dimensional implementation evaluates the
endpoint and the explicit stationary candidates of its quartic.

## Allocation policies

There is no universally optimal nonzero lambda when \(D<2R\): increasing
lambda raises first-stream variance and lowers second-stream variance. Opaque
therefore requires an explicit dimensionless `JmeAllocation`.

`JmeAllocation.paper_reference()` uses the identity-strategy convention from
Kalinin, Upadhyay, and Lampert:

\[
\lambda R^2=\frac12\quad(d\ge2),\qquad
\lambda R^2=\frac{11+5\sqrt5}{8}\quad(d=1).
\]

The first value preserves first-stream noise in the paper's unrestricted
replace-one geometry \(D=2R\). It is a reference allocation, not a universal
optimum for add/remove DP-SGD. For example, at \(D=R\) it increases
first-stream variance by 75 percent.

`JmeAllocation.first_variance_cap(kappa)` requires \(\kappa>1\) and selects the
largest lambda satisfying

\[
S_\lambda^2\le\kappa D^2.
\]

It therefore minimizes second-stream variance subject to the declared cap on
first-stream variance relative to releasing only the projected aggregate.

## Noise scales and accounting

The mechanism samples independent Gaussian streams with realized standard
deviations

\[
\sigma_1=\text{noise\_multiplier}\,S_\lambda,\qquad
\sigma_2=\frac{\text{noise\_multiplier}\,S_\lambda}{\sqrt{\lambda}}.
\]

Whitening by these scales turns the complete release into a sensitivity-one
Gaussian query. The actual nonlinear sampled mechanism is therefore
**dominated by**, but need not equal, the standard sampled-Gaussian privacy
loss distribution. For ordinary independent Poisson sampling, account with

```python
step = dpsgd_acc.poisson(
    dpsgd_acc.gaussian(noise_multiplier),
    sample_rate=sample_rate,
)
training = step * num_steps
```

Adaptive clipping remains valid when its threshold depends only on the prior DP
transcript; its clipping-rate release must still be included through
`dpsgd_acc.adaclip`.

## Scope and restrictions

- Use scalar global clipping. `PerGroup` needs a group-coupled sensitivity
  derivation and is rejected.
- Use standard unbounded Gaussian noise. Bounded Gaussian output has a
  different privacy profile and is not supported by `jme_noise`.
- Use fixed public `z`, `R`, and allocation configuration. Normalizing by the
  realized Poisson batch size is not covered.
- Empty Poisson draws still release noised zero aggregates and advance state.
- Sum disjoint distributed shards before calling `jme_noise`; projecting each
  local aggregate computes a different query.
- The documented accounting covers plain independent Poisson sampling.
  Truncated, replicated/parallel, horizon-allocation, and DP-FTRL/MF sampling
  need separate multi-contribution analyses.

## Source

JME is based on Kalinin, Upadhyay, and Lampert,
[Continual Release Moment Estimation with Differential Privacy](https://arxiv.org/abs/2502.06597v2),
especially Algorithm 1, Definition 3.4, Lemma 3.5, Theorem 4.3, and
Appendix A. The paper analyzes replace-one bounded stream rows. The projection,
distance constraint, and plain-Poisson domination above are the add/remove
DP-SGD adaptation used by Opaque. The paper publishes pseudocode rather than an
executable JME reference implementation, and its DP-Adam pseudocode does not
project the summed minibatch aggregate.
