# Differential privacy review protocol

Opaque is a functional differential privacy library. A change can be
well-structured and fully tested while still invalidating a privacy guarantee.
Reviews of privacy-sensitive code must therefore treat mathematical correctness,
implementation correctness, and architecture as separate requirements.

## Evidence standard

Repository documentation explains the implemented mechanisms and links to their
primary sources. It is an implementation map, not a substitute for the
literature.

For a change that modifies a mechanism, privacy accountant, sampler, clipping or
noise rule, audit, or claimed privacy property:

1. Identify the exact privacy claim and the assumptions it requires.
2. Follow the repository citation to the primary paper. Use web search and URL
   fetching to verify the relevant theorem, equation, algorithm, or experimental
   condition before accepting a mathematical claim.
3. Cite the source and, when practical, the relevant section, theorem, equation,
   or algorithm in review feedback.
4. Check the implementation end to end. A local formula can match a paper while
   the composed system violates its assumptions.
5. Never invent a citation or imply that a paper proves more than it does. If a
   source is unavailable or ambiguous, state what could not be verified and ask
   for a derivation, reference, or focused numerical validation.

Findings based on a direct contradiction of a theorem assumption or the
implemented privacy model are correctness issues. Differences from a paper's
workload, optimizer, or recommended parameters may be utility concerns rather
than privacy violations; label them accordingly.

## End-to-end privacy invariants

Trace every affected path through the following questions.

### Adjacency and protected unit

- What is the protected unit: record, example, user, sequence, or another
  contribution?
- Which neighboring relation is assumed? Opaque uses add-or-remove adjacency by
  default. With a fixed clipping threshold, replace-one analyses generally
  require twice the sensitivity (`2 * max_norm`).
- Do grouping, repeated participation, distributed execution, or data
  preprocessing change the effective contribution model?

### Query and sensitivity

- Is the loss or query evaluated per protected unit before aggregation?
- Is clipping applied before summation, and does the norm match the analysis?
- Do `normalize_by`, microbatching, distributed reductions, and per-group
  allocation preserve the advertised `max_norm`?
- Does adaptive state change the bound? DP-FTRL matrix mechanisms require a
  constant per-step contribution bound; AUTO-S remains bounded by its fixed
  radius, while adaptive clipping does not.

### Noise and randomness

- Is absolute noise scaled from the same sensitivity used by accounting?
- Are first- and second-moment releases allocated and accounted jointly?
- Are random keys domain-separated, advanced exactly once, serialized, and
  restored without reuse?
- Does bounded, truncated, correlated, or otherwise nonstandard noise still
  match the accountant? Similar parameter names do not establish equivalence.

### Sampling and amplification

- Does the runtime sampler exactly match the amplification theorem and
  accounting factory?
- Poisson, truncated Poisson, random allocation, cyclic Poisson, b-min-separation,
  balls-in-bins, and deterministic sequential sampling are distinct models.
- Are dataset size, sample rate, band count, bin count, participation separation,
  and horizon derived from the same execution plan?
- Are empty or partial batches handled without silently changing participation
  probabilities or step separation?

### Composition and accounting

- Is the mechanism per-step or whole-process? DP-SGD composes per-step costs;
  DP-FTRL matrix mechanisms describe a fixed horizon and must not be composed
  again as if every step were independent.
- Are all releases, adaptive queries, phases, restarts, early-stopping decisions,
  and hyperparameter searches included in the privacy statement?
- Do discretization, tail truncation, numerical tolerances, and calibration
  brackets preserve the stated guarantee?
- Is a result an analytic upper bound, a deterministic numerical bound, or a
  Monte Carlo point estimate? Conservative PLD discretization does not turn a
  Monte Carlo estimate into an upper confidence bound.

### Matrix mechanisms

- Do noise generation and accounting use the same strategy, Gram matrix,
  sensitivity, horizon, and participation model?
- Do optimizer momentum and learning-rate schedule match the workload encoded by
  the strategy when a utility claim depends on that match?
- Are `min_sep`, `max_participations`, bands, bins, and epochs consistent with
  the actual data order?
- Distinguish privacy correctness for the instantiated linear operator from
  workload fidelity and utility relative to a paper.

### Auditing

- Does the attack and confidence procedure match the audited privacy family?
- Are canary selection, inclusion coins, scoring direction, significance, and
  the number of canaries handled as assumed by the audit?
- Treat an empirical lower bound above the theoretical upper bound as a likely
  implementation, sensitivity, noise, accounting, or leakage defect.

## Functional implementation review

Opaque's functional design is part of correctness:

- state returned by clipping, noise, optimizers, samplers, and accountants must
  be threaded without hidden mutation;
- callable strategies must preserve their documented numerical contract;
- serialization must restore privacy-relevant state, counters, horizons, and RNG
  position;
- distributed and microbatched paths must be equivalent to the analyzed
  single-process mechanism;
- tests should exercise numerical properties and end-to-end invariants, not only
  imports or source structure.

## Literature map

Start with the source closest to the changed mechanism, then follow its
dependencies.

| Area | Repository map | Primary literature |
| --- | --- | --- |
| DP definition, composition, f-DP, PLD | [`dp-concepts.md`](../docs/user-guide/dp-concepts.md), [`accounting.md`](../docs/user-guide/accounting.md) | [Dwork, Rothblum, Vadhan (2010)](https://theory.stanford.edu/~salil/papers/compose-private.pdf); [Kairouz et al. (2015)](https://arxiv.org/abs/1311.0776); [Dong, Roth, Su (2019)](https://arxiv.org/abs/1905.02383); [Koskela et al. (2020)](https://arxiv.org/abs/1906.03049) |
| DP-SGD and Gaussian noise | [`dp-sgd.md`](../docs/user-guide/dp-sgd.md), [`gaussian.md`](../docs/mechanisms/dp-sgd/gaussian.md) | [Abadi et al. (2016)](https://arxiv.org/abs/1607.00133); [Balle, Barthe, Gaboardi (2018)](https://arxiv.org/abs/1807.01647) |
| Clipping | [`clipping.md`](../docs/user-guide/clipping.md) | [Andrew et al. (2021)](https://arxiv.org/abs/1905.03871); [Bu et al. (2023)](https://arxiv.org/abs/2206.07136) |
| MoE load-balancing release | [`moe-load-balancing.md`](../docs/mechanisms/dp-sgd/moe-load-balancing.md) | [Fedus, Zoph, Shazeer (2022)](https://arxiv.org/abs/2101.03961); [Andrew et al. (2021)](https://arxiv.org/abs/1905.03871); [McMahan et al. (2018)](https://arxiv.org/abs/1812.06210) |
| DP-FTRL and matrix factorization | [`dp-ftrl.md`](../docs/user-guide/dp-ftrl.md), [`dp-ftrl mechanisms`](../docs/mechanisms/dp-ftrl/index.md) | [Kairouz et al. (2021)](https://arxiv.org/abs/2103.00039); [Denisov et al. (2022)](https://arxiv.org/abs/2202.08312) |
| BandMF and b-min-separation | [`band-mf.md`](../docs/mechanisms/dp-ftrl/band-mf.md), [`sampling.md`](../docs/user-guide/sampling.md) | [Choquette-Choo et al. (2023)](https://arxiv.org/abs/2306.08153); [Dong, Ganesh (2026)](https://arxiv.org/abs/2602.09338) |
| BLT, BSR, BISR, and DP-lambda-CGD | [`blt.md`](../docs/mechanisms/dp-ftrl/blt.md), [`bsr.md`](../docs/mechanisms/dp-ftrl/bsr.md), [`bisr.md`](../docs/mechanisms/dp-ftrl/bisr.md), [`lambda-cgd.md`](../docs/mechanisms/dp-ftrl/lambda-cgd.md) | [Dvijotham et al. (2024)](https://arxiv.org/abs/2404.16706); [McMahan et al. (2024)](https://arxiv.org/abs/2408.08868); [Kalinin, Lampert (2024)](https://arxiv.org/abs/2405.13763); [Kalinin et al. (2025)](https://arxiv.org/abs/2505.12128); [Kalinin et al. (2026)](https://arxiv.org/abs/2601.22334) |
| Privacy auditing | [`auditing.md`](../docs/user-guide/auditing.md) | [Steinke, Nasr, Jagielski (2023)](https://arxiv.org/abs/2305.08846); [Xiang et al. (2025)](https://arxiv.org/abs/2509.08704); [Carlini et al. (2022)](https://arxiv.org/abs/2112.03570) |

The map is intentionally curated rather than exhaustive. A new mechanism or
mathematical claim must add its primary source and review assumptions to the
closest mechanism documentation.
