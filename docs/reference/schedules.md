# Schedules

Pure-Python step-indexed scalar schedules. Each public function
returns a plain `Callable[[int], float]` that plugs straight into
`torchopt.adamw(lr=...)` or Opake factories such as
[`adamw`](optimizers.md), `adam`, and
`sgd`.
TorchOpt's `scale_by_neg_lr` advances the schedule via the
optimizer-state step counter — no manual `.step()` call.

Public surface:

- pure curves: `constant_schedule`, `linear_schedule`,
  `polynomial_schedule`, `exponential_schedule`, `cosine_schedule`,
  `inverse_sqrt_schedule`, `one_minus_sqrt_schedule`;
- composition primitives: `with_warmup`, `with_restarts`,
  `warmup_stable_decay`.

---

## Composing a schedule

A decay curve is configured by `transition_begin` (when decay starts)
and `transition_steps` (how long it takes); it returns its
`init_value` during `[0, transition_begin)` and decays during
`[transition_begin, transition_begin + transition_steps]`. Wrap the
decay with `with_warmup` to multiply it by a `0 → 1` ramp during the
first `transition_steps` steps — turning the leading plateau into a
warmup ramp.

```python
import torchopt
from opake.scheduling import with_warmup, cosine_schedule

W, N, base_lr = 100, 10000, 1e-3

# Cosine that plateaus during [0, W), then decays over [W, N).
decay = cosine_schedule(
    init_value=base_lr, end_value=0.0,
    transition_steps=N - W, transition_begin=W,
)

# Replace the plateau with a 0 → base_lr ramp.
schedule = with_warmup(decay, transition_steps=W)

opt = torchopt.adamw(lr=schedule)
```

---

## `constant_schedule`

```python
constant_schedule(value: float) -> Callable[[int], float]
```

Returns `value` at every step. Equivalent to passing a float
directly to TorchOpt's `lr` argument; `with_warmup` accepts a float
as the same shorthand.

```python
from opake.scheduling import constant_schedule, with_warmup

schedule = constant_schedule(1e-3)

# Warmup, then plateau.
schedule = with_warmup(1e-3, transition_steps=100)
```

---

## `linear_schedule`

```python
linear_schedule(
    init_value: float,
    end_value: float,
    transition_steps: int,
    transition_begin: int = 0,
) -> Callable[[int], float]
```

Linear interpolation from `init_value` to `end_value` over
`transition_steps`, starting at `transition_begin`. Steps before
`transition_begin` hold at `init_value`; steps after the transition
hold at `end_value`.

```python
from opake.scheduling import linear_schedule

# 1e-3 → 0 over 1000 steps.
sched = linear_schedule(1e-3, 0.0, transition_steps=1000)
```

---

## `polynomial_schedule`

```python
polynomial_schedule(
    init_value: float,
    end_value: float,
    power: float,
    transition_steps: int,
    transition_begin: int = 0,
) -> Callable[[int], float]
```

Polynomial transition from `init_value` to `end_value`:
`end + (init - end) * (1 - count/T)^power`. `power=1` reduces to
[`linear_schedule`](#linear_schedule); larger `power` produces a
flatter early phase and steeper late drop.

```python
from opake.scheduling import polynomial_schedule

sched = polynomial_schedule(1e-3, 1e-7, power=2.0, transition_steps=1000)
```

---

<a id="exponential_decay"></a>
## `exponential_schedule`

```python
exponential_schedule(
    init_value: float,
    decay_rate: float,
    transition_begin: int = 0,
    transition_steps: int = 1,
    staircase: bool = False,
    end_value: float | None = None,
) -> Callable[[int], float]
```

Geometric schedule: `init * decay_rate^((step - transition_begin) / transition_steps)`.
The shape is exponential; the direction depends on `decay_rate` —
`< 1` decays, `> 1` grows, `== 1` stays constant. With
`staircase=True` the exponent is floored, so the value moves in
discrete jumps every `transition_steps`. `end_value`, when set,
clamps the result (lower bound for `decay_rate < 1`, upper bound for
`decay_rate > 1`).

```python
from opake.scheduling import exponential_schedule

# Halve LR every 1000 steps, but never below 1e-5.
sched = exponential_schedule(1e-3, decay_rate=0.5, transition_steps=1000, end_value=1e-5)

# Or grow a knob from 0.1 to (eventually) 1.0.
warm = exponential_schedule(0.1, decay_rate=1.5, transition_steps=500, end_value=1.0)
```

> **Renamed from `exponential_decay`.**  The old name implied decay, but
> the function also produces growth when `decay_rate > 1`, so it was
> renamed to match the `<shape>_schedule` pattern of the rest of this
> module.  The previous name is no longer exported.

---

## `cosine_schedule`

```python
cosine_schedule(
    init_value: float,
    end_value: float,
    transition_steps: int,
    transition_begin: int = 0,
    num_cycles: float = 0.5,
) -> Callable[[int], float]
```

Cosine annealing from `init_value` to `end_value` over
`transition_steps`.  Steps before `transition_begin` hold at
`init_value`.  `num_cycles=0.5` (the default) is a single half-cosine
that bottoms out at `end_value` exactly when `progress == 1`; values
greater than `0.5` produce additional oscillations clamped at zero.

```python
from opake.scheduling import cosine_schedule

# Single half-cosine from 1e-3 to 0 over 1,000 steps.
sched = cosine_schedule(1e-3, 0.0, transition_steps=1000)

# 1.5 oscillations over 1000 steps.
sched = cosine_schedule(1e-3, 0.0, transition_steps=1000, num_cycles=1.5)
```

---

## `inverse_sqrt_schedule`

```python
inverse_sqrt_schedule(
    init_value: float,
    transition_steps: int,
    transition_begin: int = 0,
) -> Callable[[int], float]
```

Inverse-square-root decay
`init_value * sqrt(T / (s + T))`, where `T = transition_steps` is the
timescale and `s = max(0, step - transition_begin)`.  At `s = 0`
returns `init_value`; at `s = T` returns `init_value / sqrt(2)`; at
`s = 3T` returns `init_value / 2`.

```python
from opake.scheduling import inverse_sqrt_schedule

# Decay from 1e-3 with timescale 1000.  At step 1000: ~7.07e-4.
sched = inverse_sqrt_schedule(1e-3, transition_steps=1000)
```

---

## `one_minus_sqrt_schedule`

```python
one_minus_sqrt_schedule(
    init_value: float,
    end_value: float,
    transition_steps: int,
    transition_begin: int = 0,
) -> Callable[[int], float]
```

Decay following `factor = 1 - sqrt(progress)` from `init_value` at
`transition_begin` to `end_value` at
`transition_begin + transition_steps`.  Concave decreasing — drops
faster early than late.  Held at `end_value` after the transition.

```python
from opake.scheduling import one_minus_sqrt_schedule

# Drops 1e-3 -> 1e-5 with concave shape over 1000 steps.
sched = one_minus_sqrt_schedule(1e-3, 1e-5, transition_steps=1000)
```

---

## `with_warmup`

```python
with_warmup(
    schedule: Callable[[int], float] | float,
    transition_steps: int,
    *,
    ramp: str | Callable[[float], float] = "linear",
    init_value: float = 0.0,
) -> Callable[[int], float]
```

Multiply `schedule` by an `init_value → 1` ramp over the first
`transition_steps` steps; afterwards return `schedule(step)` unchanged.
A scalar `float` for `schedule` is treated as
`constant_schedule(value)`.

The `ramp` kwarg picks the warmup curve shape:

- `"linear"` (default): `progress`
- `"cosine"`: `0.5 * (1 - cos(pi * progress))`
- `"1-sqrt"`: `1 - sqrt(1 - progress)`
- a callable `f(progress)` taking progress in `[0, 1]` and returning
  a factor in `[0, 1]` — for any custom shape.

`init_value` (default `0.0`) is the starting factor for the ramp.
The default produces a standard `0 → 1` warmup.  A positive value
(must lie in `[0, 1]`) produces a `floor → 1` ramp that starts at
`init_value * schedule(0)` instead of zero — useful when the
optimizer needs a non-trivial step from the very first update (e.g.
StableAdamW), or to match `warmup_lr_rate` semantics from HF's
`cosine_warmup_with_min_lr`.  `init_value=1.0` collapses the wrapper
to the inner schedule unchanged.

All schedules in this module return `init_value` for
`step < transition_begin`, so configuring a decay with
`transition_begin == transition_steps` of the warmup produces the
standard "warmup, then decay" shape:

| step range                                         | result                              |
|----------------------------------------------------|-------------------------------------|
| `[0, transition_steps)`                            | `(step / transition_steps) * init_value` (linear ramp 0 → init_value) |
| `[transition_steps, decay's transition_begin)`     | `init_value` (plateau, only if there's a gap) |
| later                                              | `schedule(step)` as configured      |

```python
from opake.scheduling import with_warmup, linear_schedule, cosine_schedule

base_lr, W, N = 1e-3, 100, 1000
decay_steps = N - W

# Linear-decay with warmup.
sched = with_warmup(
    linear_schedule(base_lr, 0.0, decay_steps, transition_begin=W),
    transition_steps=W,
)

# Cosine-decay with warmup.
sched = with_warmup(
    cosine_schedule(base_lr, 0.0, decay_steps, transition_begin=W),
    transition_steps=W,
)

# Warmup-then-constant via the float shorthand.
sched = with_warmup(base_lr, transition_steps=W)

# Floor → 1 ramp (start at 0.1 * schedule(0)).
sched = with_warmup(base_lr, transition_steps=W, init_value=0.1)
```

Raises `ValueError` if `transition_steps <= 0`, `ramp` is an
unknown string, or `init_value` is outside `[0, 1]`.

---

## `with_restarts`

```python
with_restarts(
    schedule: Callable[[int], float],
    transition_steps: int,
    num_cycles: int,
    transition_begin: int = 0,
) -> Callable[[int], float]
```

Repeat `schedule` `num_cycles` times across
`[transition_begin, transition_begin + transition_steps)`.  Each
cycle has length `transition_steps / num_cycles`; within a cycle,
`schedule` is evaluated at the cycle-local step
(`relative_step % cycle_length`).  Configure `schedule` to produce
its full curve over a single cycle of that length.

Before `transition_begin` returns `schedule(0)`; after the final
cycle, returns `schedule(cycle_length)`.

```python
from opake.scheduling import cosine_schedule, with_restarts

# SGDR: cosine annealing repeated 4 times over 4000 steps.
inner = cosine_schedule(1e-3, 0.0, transition_steps=1000)
sched = with_restarts(inner, transition_steps=4000, num_cycles=4)
```

Raises `ValueError` if `num_cycles <= 0` or `transition_steps <= 0`.

---

## `warmup_stable_decay`

```python
warmup_stable_decay(
    init_value: float,
    end_value: float = 0.0,
    *,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int,
    warmup_ramp: str | Callable[[float], float] = "linear",
    decay_shape: str | Callable[[float], float] = "1-sqrt",
) -> Callable[[int], float]
```

Three-phase **W**armup → **S**table → **D**ecay schedule (Hägele
et al. 2024, MiniCPM).  Over
`num_warmup_steps + num_stable_steps + num_decay_steps` total steps:

1. `[0, num_warmup_steps)` — ramp from `0` to `init_value`.
2. `[num_warmup_steps, num_warmup_steps + num_stable_steps)` —
   constant at `init_value`.
3. `[num_warmup_steps + num_stable_steps, total)` — decay from
   `init_value` down to `end_value` under `decay_shape`.

Beyond `total` returns `end_value`.

`decay_shape` selects the decay curve:

- `"1-sqrt"` (default — Hägele et al.'s recommendation):
  `factor = 1 - sqrt(progress)`; concave-down, fast initial drop,
  slow finish.
- `"linear"`: `factor = 1 - progress`.
- `"cosine"`: `factor = 0.5 * (1 + cos(pi * progress))`; half-cosine
  from init to end.
- a callable `f(progress) -> factor in [0, 1]` mapping progress
  (where `0` is the start of decay and `1` is the end) to the
  factor applied to `(init_value - end_value)`.

The stable middle plateau enables **decay-only fine-tuning**: resume
from any checkpoint inside the stable region and run only the decay
tail without re-running the warmup.

```python
from opake.scheduling import warmup_stable_decay

schedule = warmup_stable_decay(
    init_value=1e-3,
    end_value=1e-4,                 # min-lr floor
    num_warmup_steps=500,
    num_stable_steps=8_000,
    num_decay_steps=1_500,
)
```

Raises `ValueError` if `num_warmup_steps <= 0`, `num_stable_steps < 0`,
`num_decay_steps <= 0`, or if `warmup_ramp` / `decay_shape` is an
unknown string.

---

## See also

- [Optimizers API](optimizers.md) — pass any of these schedules to TorchOpt's `lr` argument.
- [LR Scheduling User Guide](../user-guide/lr-scheduling.md) — patterns and DP-specific guidance.
