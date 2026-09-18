# opake-auditing

Empirical privacy auditing for Opake: one-run estimator
(Steinke et al. 2023), coin-flip canary partitioning, and
loss-based membership-inference attacks.

## Install

Install the root package as described in the [repository installation guide](https://github.com/JetBrains-Research/opaque#installation).
Use its `auditing` extra to include this component.

`opake-auditing` depends on `opake-engine` and `scipy`; both install
automatically with the extra.

## Quick start

```python
import opake.auditing as auditing
from opake.random import key

cf = auditing.coin_flip(dataset, num_canaries=1000, key=key(42))
# ... DP-SGD training loop ...
estimate = auditing.one_run(scores, coin_flip=cf)
```

By default, canaries are sampled from the dataset's natural rows. Pass
`candidate_indices=` to restrict selection to a precommitted pool of constructed
or selected records.

## Layout

- `opake.auditing.one_run` — one-run estimator (Steinke et al. 2023)
- `opake.auditing.coin_flip()` — coin-flip canary partitioning
- `opake.auditing.loss_scores()` — loss-based membership inference via `vmap`
