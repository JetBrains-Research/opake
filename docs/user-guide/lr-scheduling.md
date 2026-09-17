# Learning-rate Scheduling

Standard ML LR schedules — linear / cosine decay with warmup — work
without modification under DP-SGD: they scale the already-clipped,
already-noised update direction. The privacy guarantee is
unchanged.

This page explains the schedules and warmup composition Opake
provides via [`opake.scheduling`](../reference/schedules.md), and the
pattern for using them with TorchOpt functional optimizers.

## Why warmup matters under DP-SGD

Early DP-SGD updates can be noise-dominated. A short warmup can improve
stability by gradually increasing the learning rate; tune its length
with the optimizer and noise multiplier.

## Constant LR

```python
import torchopt

opt = torchopt.adamw(lr=1e-3)
```

A float passes straight through TorchOpt; no schedule object is
needed.

## Pure decay

```python
import torchopt
from opake.scheduling import linear_schedule

# Linear decay from 1e-3 to 0 over 10,000 steps.
schedule = linear_schedule(1e-3, 0.0, transition_steps=10_000)
opt = torchopt.adamw(lr=schedule)
```

`opake.scheduling` ships the common decay curves directly:
[`linear_schedule`](../reference/schedules.md#linear_schedule),
[`polynomial_schedule`](../reference/schedules.md#polynomial_schedule),
[`exponential_schedule`](../reference/schedules.md#exponential_schedule),
[`cosine_schedule`](../reference/schedules.md#cosine_schedule),
[`inverse_sqrt_schedule`](../reference/schedules.md#inverse_sqrt_schedule),
and [`one_minus_sqrt_schedule`](../reference/schedules.md#one_minus_sqrt_schedule).

## Adding warmup

[`with_warmup`](../reference/schedules.md#with_warmup) multiplies a
schedule by a linear ramp `0 → 1` during the first `transition_steps`
steps and leaves it untouched afterwards.

For the standard "warmup, then decay" shape, configure the decay
with `transition_begin = num_warmup_steps`.  Schedules in
`opake.scheduling` return their `init_value` while
`step < transition_begin`, so the multiplicative ramp turns that
leading plateau into the warmup ramp.

```python
import torchopt
from opake.scheduling import linear_schedule, with_warmup

base_lr, W, N = 1e-3, 500, 10_000

decay = linear_schedule(
    base_lr, 0.0,
    transition_steps=N - W,
    transition_begin=W,
)
schedule = with_warmup(decay, transition_steps=W)
opt = torchopt.adamw(lr=schedule)
```

`with_warmup` raises `ConfigurationError` for `transition_steps <= 0`, so
gate it on a runtime warmup length:

```python
schedule = (
    with_warmup(decay, transition_steps=W) if W > 0 else decay
)
```

The default warmup ramp is linear; `ramp="cosine"`, `ramp="1-sqrt"`,
or any `Callable[[float], float]` mapping `progress ∈ [0, 1]` to a
factor in `[0, 1]` selects a different shape:

```python
schedule = with_warmup(decay, transition_steps=W, ramp="cosine")
```

### Non-zero warmup floor

The default warmup ramps from `0`. Some optimizers (StableAdamW, for
example) prefer a non-trivial first step; pass `init_value` to start
the ramp at a fraction of the inner schedule's value instead of zero:

```python
# Ramp factor goes 0.1 → 1.0 over W steps, so the first step's LR
# is 0.1 * decay(0) rather than 0.
schedule = with_warmup(decay, transition_steps=W, init_value=0.1)
```

`init_value=1.0` collapses to the inner schedule unchanged.

## Three worked examples

### Linear with warmup

```python
from opake.scheduling import linear_schedule, with_warmup

base_lr, W, N = 1e-3, 500, 10_000
decay = linear_schedule(
    base_lr, 0.0, transition_steps=N - W, transition_begin=W,
)
schedule = with_warmup(decay, transition_steps=W)
```

### Cosine with warmup

```python
from opake.scheduling import with_warmup, cosine_schedule

base_lr, W, N = 1e-3, 500, 10_000
decay = cosine_schedule(
    base_lr, 0.0, transition_steps=N - W, transition_begin=W,
)
schedule = with_warmup(decay, transition_steps=W)
```

### Constant after warmup

`with_warmup` accepts a scalar `float` as shorthand for a constant
schedule, so warm-up-then-plateau is a one-liner:

```python
from opake.scheduling import with_warmup

schedule = with_warmup(1e-3, transition_steps=500)
```

## Restarts

[`with_restarts`](../reference/schedules.md#with_restarts) replays a
schedule `num_cycles` times over a window. Combined with cosine you get
[SGDR](https://arxiv.org/abs/1608.03983) (Loshchilov & Hutter, 2017):

```python
from opake.scheduling import cosine_schedule, with_restarts

base_lr, N, k = 1e-3, 4000, 4
cycle_length = N / k

inner = cosine_schedule(base_lr, 0.0, transition_steps=cycle_length)
schedule = with_restarts(inner, transition_steps=N, num_cycles=k)
```

Each cycle the schedule snaps back to `inner(0)` and runs the curve
over its own `cycle_length`. It composes with `with_warmup` the same
way as any other decay.

## Warmup-Stable-Decay (WSD)

[`warmup_stable_decay`](../reference/schedules.md#warmup_stable_decay)
implements the three-phase schedule from Hägele et al.'s 2024
[Scaling Laws and Compute-Optimal Training Beyond Fixed Training
Durations](https://arxiv.org/abs/2405.18392). MiniCPM also adopts this
schedule:

1. **Warmup** — ramp from `0` to `init_value` over `num_warmup_steps`.
2. **Stable** — constant at `init_value` for `num_stable_steps`.
3. **Decay** — drop from `init_value` to `end_value` over
   `num_decay_steps` according to `decay_shape`.

```python
import torchopt
from opake.scheduling import warmup_stable_decay

schedule = warmup_stable_decay(
    init_value=1e-3,                # peak LR
    end_value=0.0,                  # decay target (often 0.1 * init_value)
    num_warmup_steps=500,
    num_stable_steps=8_000,
    num_decay_steps=1_500,
    # decay_shape="1-sqrt" by default — Hägele et al.'s recommendation.
)
opt = torchopt.adamw(lr=schedule)
```

The stable middle plateau enables **decay-only fine-tuning**: resume
from any checkpoint inside the stable region and run only the decay
tail without re-training the warmup. That makes WSD a natural fit
for "anytime" training where the final compute budget isn't fixed up
front.

`decay_shape` accepts `"1-sqrt"` (default — concave-down, fast initial
drop then slow finish), `"linear"`, `"cosine"` (half-cosine from `init`
to `end`), or a callable `f(progress) -> factor in [0, 1]` mapping
progress to the factor applied to `(init_value - end_value)`.

## Mapping from `transformers` schedule names

`transformers` exposes a number of named cosine variants for
historical reasons. Opake's `cosine_schedule` and the composition
primitives subsume all of them; the table below shows the recipe for
each. No engine-side alias is needed — pass the equivalent
construction directly.

The HF schedules all include a `0 → base_lr` warmup over the first
`W` steps. Each Opake recipe configures the inner cosine with
`transition_begin=W` (so the cosine returns `init_value` during the
warmup window) and wraps it with `with_warmup(..., transition_steps=W)`
so that leading plateau is rescaled into the ramp.

| `transformers` name | Opake recipe |
|---|---|
| `cosine` | `with_warmup(cosine_schedule(base_lr, 0.0, transition_steps=N - W, transition_begin=W), transition_steps=W)` |
| `cosine_with_min_lr` | Same as `cosine`, but pass `end_value=min_lr` (the second positional arg of `cosine_schedule`). |
| `cosine_with_restarts` | `with_warmup(with_restarts(cosine_schedule(base_lr, 0.0, transition_steps=cycle_length), transition_steps=N - W, num_cycles=k, transition_begin=W), transition_steps=W)` |
| `cosine_warmup_with_min_lr` | `with_warmup(cosine_schedule(base_lr, min_lr, transition_steps=N - W, transition_begin=W), transition_steps=W)` |
| `warmup_stable_decay` | [`warmup_stable_decay(...)`](#warmup-stable-decay-wsd) — direct primitive. |

If your application doesn't use warmup (`W == 0`), drop the
`with_warmup` wrapper and the `transition_begin=W` argument.

## Reading the current LR

Schedules are plain callables — `schedule(step)` returns the LR at
any step:

```python
for step in range(num_steps):
    # ... clipping, noise, update ...
    if step % 100 == 0:
        print(f"step={step}  lr={schedule(step):.6f}")
```

The optimizer maintains its own counter inside `opt_state` (via
`scale_by_neg_lr`), so calling `schedule(step)` for logging does
not interfere with how the optimizer applies the schedule.

## See also

- [Schedules API](../reference/schedules.md) — full signatures and behavior.
- [Optimizers User Guide](optimizers.md) — bias correction, weight decay, AdamW-BC.
