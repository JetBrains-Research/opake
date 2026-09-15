# MoE load balancing

A mixture-of-experts checkpoint trained with a Switch-style
load-balancing loss needs the batch load vector, the fraction of the
batch's tokens routed to each expert, in its objective. That vector is
the only term of the objective that couples the examples of a batch, so a
per-example DP pipeline cannot express it directly. This mechanism
releases it privately alongside the gradient, inside the clipper, the way
adaptive clipping releases its clipped fraction: one Gaussian per step,
one accountant call, no change to the noise function or the optimizer.

Implemented by `opaque.dpsgd.clipping.moe_clipped_grad` (the clipper) and
`opaque.dpsgd.accounting.moe_aux` (the accountant).

## Idea

The Switch load-balancing loss over a batch $B$ is

$$L_{\text{aux}}(B) = E \sum_{e=1}^{E} f_e(B)\, P_e(B),$$

where $f_e(B)$ is the fraction of the batch's tokens whose top-$k$ set
contains expert $e$ and $P_e(B)$ is the mean router probability of expert
$e$ over the batch's tokens
([Fedus, Zoph, Shazeer, 2022](https://arxiv.org/abs/2101.03961), eqs. 4
to 6; stated there per layer for top-1 routing, pooled over layers and
generalised to top-$k$ here). The load $f$ is a normalised count of
argmax outcomes, so it is piecewise constant in the parameters and its
gradient is zero almost everywhere. The router mass
$P_e(B) = \frac{1}{T_B} \sum_{x \in B} T_x\, P_e(x)$ is differentiable, a
token-weighted mean of per-example router masses over the batch's token
total $T_B$. Freezing $f$ at its batch value therefore gives

$$\nabla_\theta L_{\text{aux}}(B) = E \sum_e f_e(B)\, \nabla_\theta P_e(B)
  = \sum_{x \in B} \nabla_\theta \Big[ E\, \frac{T_x}{T_B} \sum_e f_e(B)\, P_e(x) \Big].$$

Two batch-level quantities remain, $T_B$ and $f(B)$, and a per-example
pipeline can use neither. The mechanism replaces the first by a public
constant, $T_x / T_B \to w_x / \bar B$ with $w_x = T_x / \bar T$ for a
public token constant $\bar T$ and $\bar B$ the expected batch size the
summed gradients are divided by, and the second by the lagged private
estimate $\tilde f$ of the next sections. Every example then contributes
the gradient of the surrogate

$$S(x; \tilde f) = E\, w_x \sum_e \big(\tilde f_e - k/E\big)\, P_e(x),$$

whose normalised batch sum is the frozen-load Switch gradient scaled by
$T_B / (\bar B \bar T)$: the Switch gradient itself when the realised
token total equals the public normalisation, a public rescaling of it
otherwise. What is trained is therefore a one-step-lagged,
public-normalised, top-$k$, layer-pooled generalisation of the Switch
objective, not an identity with it. Centring by $k/E$ changes nothing,
since $\sum_e P_e(x) = 1$, and makes the signal read as imbalance: a
balanced estimate produces a zero gradient. The only private object the
pipeline lacks is $\tilde f$, and the mechanism below is how it obtains
one.

## The released statistic

For example $x$ and routed layer $l$, the load fraction and its centred
form are

$$h^l_e(x) = \frac{1}{T_x} \sum_t m_{x,t}\, \mathbb{1}[e \in S^l_{x,t}],
\qquad s(x) = w_x \big(h(x) - k/E\big) \in \mathbb{R}^{L \times E},$$

with $m$ the token mask and $S^l_{x,t}$ the executed top-$k$ set. Because
$0 \le h^l_e \le 1$ and $\sum_e h^l_e = k$ on every layer,

$$\lVert s(x) \rVert_2 \le \Delta_L = \frac{T_{\max}}{\bar T}
  \sqrt{k\, L\, (1 - k/E)}$$

for every input, with equality when every token of every layer is routed
to the same $k$ experts. The bound is structural, so it holds for
adversarial inputs, and the rescale the clipper applies at $\Delta_L$
only ever removes floating-point round-off: the released mean is
unbiased. A row longer than $T_{\max}$ is clipped to the bound, so it is
under-weighted rather than a violation.

The released quantity of step $t$ is the batch mean of $s(x)$ over the
expected batch size $\bar B$, plus Gaussian noise of per-entry standard
deviation

$$\sigma_h = \frac{\text{nm}}{\sqrt{\rho}} \cdot \frac{\Delta_L}{\bar B},$$

where $\text{nm}$ is the gradient noise multiplier and $\rho$ is the
share of the whitened sensitivity given to the load (default $0.02$).

## One Gaussian, one accountant

Whiten the gradient half by $\sigma_g = \text{nm}\, C_g / \bar B$ and the
load half by $\sigma_h$. One record's contribution to the whitened pair
has squared norm at most

$$\frac{(C_g/\bar B)^2}{\sigma_g^2} + \frac{(\Delta_L/\bar B)^2}{\sigma_h^2}
  = \frac{1}{\text{nm}^2} + \frac{\rho}{\text{nm}^2},$$

so the step is a sensitivity-one Gaussian mechanism at the joint
multiplier

$$\text{nm}_{\text{eff}} = \frac{\text{nm}}{\sqrt{1 + \rho}}.$$

That is what `moe_aux(gaussian(nm), ratio=rho)` prices, and Poisson
amplification applies to it as one mechanism because both halves are
computed from the same sampled batch. At a fixed privacy budget the
calibrated gradient multiplier grows by exactly $\sqrt{1 + \rho}$, one
percent at the default. The estimate that enters the surrogate is
post-processing of earlier releases and costs nothing further.

The construction is the same one adaptive clipping uses, with a vector
in place of a scalar:

| | adaptive clipping | MoE load |
|---|---|---|
| side statistic | clipped fraction per group | mean centred load, $(L, E)$ |
| per-record sensitivity | one half | $\Delta_L$, structural |
| the parameter | `fraction_noise_std` | `ratio` |
| accountant | `adaclip(gaussian(nm), ...)` | `moe_aux(gaussian(nm), ratio)` |
| whitened sensitivity | $1/\text{nm}^2 + K/(2\sigma_b)^2$ | $1/\text{nm}^2 + \rho/\text{nm}^2$ |
| clipper | `adaptive_clipped_grad(..., fraction_noise_std, key)` | `moe_clipped_grad(..., ratio, key)` |
| consumed next step as | the clipping threshold | the constant in the surrogate |

The joint-release argument is the one
[Andrew et al. (2021)](https://arxiv.org/abs/1905.03871) use for the
clipped count and [McMahan et al. (2018)](https://arxiv.org/abs/1812.06210)
state for an arbitrary tuple of vector queries. Two things it needs are
worth naming. Noising each half at its own multiplier $\text{nm}$
independently, the natural first implementation, is a Gaussian at
$\text{nm}/\sqrt{2}$, not $\text{nm}$; the allocation above is what keeps
the accountant honest. And $\rho$ is fixed at setup: a share chosen from
the data would make the noise distribution data-dependent and void the
argument.

## The estimate

The noised release $y_t$ is post-processed into the next step's
constant:

1. pool over layers, $\hat d_t = \frac{1}{L} \sum_l y_t^{l}$, which
   divides the per-entry noise by $\sqrt{L}$;
2. project onto the sum-zero subspace, which removes the pure noise
   component along $\mathbf{1}$ and costs the signal nothing;
3. filter with a bias-corrected exponential moving average,
   $m_{t+1} = \beta m_t + (1 - \beta) \hat d_t$,
   $\tilde d_{t+1} = m_{t+1} / (1 - \beta^{t+1})$;
4. $\tilde f_{t+1} = \operatorname{clamp}(k/E + \tilde d_{t+1}, 0, 1)$.

The estimate is consumed one step late, so there is no dependency inside
a step: gradient and load come out of the same per-example transform,
and at $\beta = 0.99$ the current batch has one percent weight in the
estimate that multiplies its own gradient. The noise standard deviation
of the latent estimate $k/E + \tilde d_t$, before the clamp, is known
exactly at every step: `MoeClipState.filtered_noise_std` tracks the
filter's noise variance recursively,
$v_{t+1} = \beta^2 v_t + (1-\beta)^2 s_h^2$ with
$s_h = \frac{\sigma_h}{\sqrt{L}} \sqrt{\frac{E-1}{E}}$ the per-entry
noise of one pooled, projected release, and divides by the bias
correction, so it stays exact when the noise scale or $\beta$ changes
between steps (a resume with different arguments). $\tilde f$ itself is
the clamp of that latent value, so its error is Gaussian only away from
the boundaries. At constant parameters the closed form is

$$s_{t} = \frac{\sigma_h}{\sqrt{L}} \sqrt{\frac{E-1}{E}}
  \cdot \frac{\sqrt{(1-\beta)^2 (1 - \beta^{2t}) / (1 - \beta^2)}}{1 - \beta^{t}},$$

stationary at $\frac{\sigma_h}{\sqrt{L}} \sqrt{\frac{E-1}{E}}
\sqrt{\frac{1-\beta}{1+\beta}}$. The clipper logs that stationary value
as a share of $k/E$ at setup, so a misconfigured run is visible before
the first step. `MoeClipState.imbalance`, the public monitor
$\max_e |\tilde f_e - k/E| / (k/E)$, is post-processing of the same
estimate.

## Cost

At $\bar B = 256$, $k = 8$, $E = 64$, $L = 28$, $\bar T = T_{\max}$,
$\beta = 0.99$ and a budget where the plain DP-SGD multiplier is
$0.5622$ (so $\Delta_L = 14$):

| $\rho$ | gradient-noise inflation $\sqrt{1+\rho}$ | load multiplier $\text{nm}/\sqrt{\rho}$ | one release, per entry | after the moving average |
|---|---|---|---|---|
| 0.5 | 1.225 | 0.97 | 8.0 % of $k/E$ | 0.57 % of $k/E$ |
| 0.2 | 1.095 | 1.38 | 11.3 % | 0.80 % |
| 0.1 | 1.049 | 1.87 | 15.3 % | 1.08 % |
| 0.05 | 1.025 | 2.58 | 21.1 % | 1.50 % |
| **0.02** (default) | **1.010** | 4.02 | 32.9 % | **2.33 %** |
| 0.01 | 1.005 | 5.65 | 46.3 % | 3.29 % |

Overshooting $\rho$ is cheap and undershooting is not: the penalty of a
too-large share is at most the inflation in the second column, while a
too-small share leaves the estimate noise-dominated. When in doubt, round
$\rho$ up; $0.1$ is a reasonable value when the router or the experts
themselves are trainable, because then the load signal matters more and
five percent more gradient noise is cheap by comparison. Pick the
smallest $\rho$ whose stationary filtered noise, as logged at setup, is
below about a third of the imbalance you expect to correct.

## Using it

### Functional loop

```python
import opaque.dpsgd.accounting as dpsgd_acc
from opaque.accounting import calibration as cal
from opaque.dpsgd.clipping import moe_clipped_grad
from opaque.dpsgd.noise import gaussian_noise
from opaque.random import key

ratio = 0.02
nm = cal.calibrate(
    cal.epsilon_budget(3.0, delta=1e-6),
    lambda nm: dpsgd_acc.poisson(dpsgd_acc.moe_aux(dpsgd_acc.gaussian(nm), ratio=ratio), q) * steps,
    0.3, 5.0,
).param

noise_fn, noise_state = gaussian_noise(noise_multiplier=nm, key=key(1))
grad_fn, clip_state = moe_clipped_grad(
    loss_fn,                              # (params, input_ids, attention_mask, labels) -> (loss, router_logits, attention_mask)
    clipping_norm=C_g, normalize_by=B_bar, batch_argnums=(1, 2, 3),
    noise_multiplier=nm, ratio=ratio, key=key(2),
    top_k=k, num_experts=E, num_layers=L, max_tokens=T_max,
    alpha=alpha,
)

for batch in loader:
    grads, clip_state = grad_fn(params, *batch, state=clip_state)
    noisy, noise_state = noise_fn(grads, noise_state)
    params, opt_state = optimizer.update(noisy, ...)
```

One number appears twice, `ratio`, exactly as `fraction_noise_std` does
for adaptive clipping; `noise_multiplier` is the value the noise function
already uses. `grads` is a plain `ClippedPytree` at bound
`clipping_norm / normalize_by`; the load never appears in it, in the
noised pytree, or in the diagnostics `return_aux` hands back.

Under DDP pass the same `key` on every rank and synchronize the state
after every step, as for adaptive clipping: `grad_fn` leaves the
rank-local un-noised mean in the state, and `opaque.distributed.sync`
all-reduces it, adds the noise once from the shared key and step, and
filters. A second call before the sync fails rather than dropping the
pending release.

### The router-logits contract

The loss function is evaluated on one example and returns
`(loss, router_logits, attention_mask)`: the scalar loss, one `(T, E)`
logits tensor per routed layer in layer order, holding the logits the
router actually executed, and the example's token mask (`None` counts
every position; any mask is read as `mask != 0`). The clipper does the
rest: fp32 softmax, top-$k$ recovery, masking, the load vector, the
surrogate. The surrogate enters value-neutrally, so the loss values in
the diagnostics stay the plain loss.

With a Hugging Face model and the Opaque patches applied, the loss
function is four lines:

```python
def loss_fn(params, input_ids, attention_mask, labels):
    out = fmodel(params, input_ids, attention_mask=attention_mask, labels=labels,
                 loss_only=True, output_router_logits=True)
    return out.loss, out.router_logits, attention_mask
```

`loss_only=True` selects the per-example loss forward and HF's own
`output_router_logits=True` asks the backbone for the router logits; on
that path the batch-coupled auxiliary loss is neither computed nor added
(see [model patches](../../user-guide/huggingface/model-patches.md)).
`opaque.patches.transformers.moe_geometry(model)` reads `top_k`,
`num_experts` and `num_layers` off the model as a mapping that unpacks
into the clipper. For a model with no patches, the loss function calls
the backbone and the head itself through `functional_call` and computes
the cross-entropy inline; HF's auxiliary path never runs.

### DPTrainer

`TrainingArguments(router_load=True, router_load_max_tokens=<row length>)`
switches the trainer's clipper to `moe_clipped_grad` and wraps its
accountant in `moe_aux` with `router_load_ratio` (default 0.02). The
geometry is read off the model, `alpha` defaults to the model config's
`router_aux_loss_coef`, the loss closure passes the two forward keywords,
and every logged step carries `router_load_imbalance` and
`router_load_noise_std`. The release state rides in the DP runtime
checkpoint with the other clip states, and DDP ranks synchronize it
through the trainer's existing state sync. It needs the Gaussian
mechanism, fixed clipping, and the base per-example causal-LM loss; a
subclass that overrides `compute_per_example_loss` is rejected at
construction because its forward cannot hand the router logits to the
clipper.

### Choosing the public constants

`max_tokens` is the collator's row length, a public constant of the data
pipeline like the expected batch size. `mean_tokens` defaults to it; a
public mean length (from a held-out split, never from the training rows)
makes the token weight $w_x$ average closer to one at the cost of a
larger $T_{\max} / \bar T$ in the bound. `alpha` is the model's router
auxiliary-loss coefficient. `filter_beta` sets the lag of the estimate:
$0.99$ lags about a hundred steps.

## Router precision

The clipper recovers the executed top-$k$ set by recomputing
$\operatorname{topk}(\operatorname{softmax}(z))$ in fp32. Hugging Face
routers do the same from bf16 logits, and bf16 ties can make the
recovered set differ from the executed one on a small fraction of
tokens. The opt-in fp32 router swap
(`apply_model_patches(model, router_fp32=True)`) removes the ties. This
is a fidelity question, not a privacy one: the bound holds either way,
it only decides whether the released load is exactly the executed one.

## Privacy statement

Each step releases one Gaussian on the concatenation of the clipped
per-example gradients and the per-example token-weighted centred
router-load vectors, with per-record bounds $C_g$ and $\Delta_L$ and
noise scales $\text{nm}\, C_g / \bar B$ and
$(\text{nm}/\sqrt{\rho})\, \Delta_L / \bar B$. The two are one mechanism
at multiplier $\text{nm}/\sqrt{1+\rho}$ under add/remove adjacency, which
is what `moe_aux` accounts and what the subsampling amplifiers price. The
load estimate consumed by the surrogate and the imbalance monitor are
post-processing of previous releases. Everything computed from private
examples inside the gradient transform, router logits, probabilities,
executed routes, the per-example load, stays inside it: the per-example
load is stripped from the returned diagnostics, and the gradient norms
they carry are norms of the gradient alone. The gradient half uses fixed
clipping; adaptive clipping has no constant per-record bound and cannot
host this release.

## References

- Fedus, Zoph, Shazeer. *Switch Transformers: Scaling to Trillion
  Parameter Models with Simple and Efficient Sparsity.* JMLR 2022.
  [arXiv:2101.03961](https://arxiv.org/abs/2101.03961). The
  load-balancing loss, eqs. 4 to 6.
- Andrew, Thakkar, McMahan, Ramaswamy. *Differentially Private Learning
  with Adaptive Clipping.* NeurIPS 2021.
  [arXiv:1905.03871](https://arxiv.org/abs/1905.03871). The joint
  release of a gradient and a side statistic under one Gaussian.
- McMahan, Andrew, Erlingsson, Chien, Mironov, Papernot, Kairouz. *A
  General Approach to Adding Differential Privacy to Iterative Training
  Procedures.* 2018.
  [arXiv:1812.06210](https://arxiv.org/abs/1812.06210). Allocation of
  one Gaussian across a tuple of vector queries.
- Abadi et al. *Deep Learning with Differential Privacy.* CCS 2016.
  [arXiv:1607.00133](https://arxiv.org/abs/1607.00133). DP-SGD.
