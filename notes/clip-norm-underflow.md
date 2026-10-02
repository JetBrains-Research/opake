# Clipping norm underflow: stored-value bound outside the representable range

**Status:** tracked in [#1123](https://github.com/JetBrains-Research/opake/issues/1123); documented limitation (docstring note in `clip_pytree`, bullet in
`docs/user-guide/precision.md`). No code change. A fix needs owner sign-off and
DP review.
**Found:** 2026-10-01, adversarial bound review of the fused clip engine
(branch `feat/dp-fused-per-layer-clipping`).

## Claim affected

`clip_pytree` documents that `norm(output) <= clipping_norm` holds on the stored
values for every input. The guard (`_guard_scale` / `_norm_roundoff`) budgets
*relative* rounding error of the norm reduction. Underflow is an *absolute*
error that the budget does not cover.

## Mechanism

Per-example squares are formed in `compute_dtype`, which is float32 for
float32, bfloat16 and float16 leaves (`_leaf_sq_sum`: `x = leaf.to(acc_dtype);
sq = x * x`), and only then summed in the float64 accumulator.

- `|x| < 2^-63 ≈ 1.08e-19`: `x*x` is a float32 subnormal, with absolute error
  up to `2^-150` per element.
- `|x| < 2^-75 ≈ 2.65e-23`: `x*x` rounds to zero.

So the squared norm is under-measured by at most `numel * 2^-150`. If every
entry is in that range, the computed norm can be 0. The ratio is then
`C / 0 = inf`, the clamped scale is 1, and the example is returned unchanged.

## Evidence

`experiments/clip_fused/probe_norm_underflow.py`, A100, production streaming path:

| Input | True norm | Computed norm | stored/C − 1 |
|---|---|---|---|
| bf16, 1024 × 3·2⁻¹³³, C = 0.84·true | 8.8e-39 | **0** | **+0.19** |
| fp32, 1024 × 3·2⁻¹⁴⁹, C = 0.84·true | 1.3e-43 | **0** | **+0.19** |

Magnitude scan (Gaussian, 1M elements, true norm = 1.19·C):

| Per-element RMS | fp32 computed/true | fp32 stored/C − 1 | bf16 stored/C − 1 |
|---|---|---|---|
| 1e-20 | 1.0000 | −2.0e-7 | −3.9e-3 |
| 1e-21 | 1.0000 | **+1.4e-6** | −3.9e-3 |
| 1e-22 | 0.9987 | **+1.3e-3** | −2.7e-3 |
| 1e-23 | 0.339 | **+0.19** | **+0.19** |

`experiments/clip_fused/check_bound_adversarial.py` (case A2) reproduces it for
production and for the fused engine, which reuses production's guard.

## Reachability

A violation needs the true norm above C while the lost squared mass exceeds the
guard's relative margin:

- Inputs made only of underflowing entries have norm `<= 1.08e-19 * sqrt(numel)`.
  So a violation needs `C` below that (for example `C < 1e-13` at `numel = 1e12`).
- For mixed inputs the relative deficit is at most `numel * 2^-150 / C^2`. At
  `C >= 1e-6` and `numel <= 1e12` that is `<= 7e-22`.
- Adaptive clipping floors C at `clipping_norm_min = 0.01` by default.

**Practical impact: none for realistic thresholds. The issue is formal:** the
docstring promised the bound for every input.

## Scope

| Path | Affected? |
|---|---|
| Fixed clipping: `clip_pytree`, per-group, streaming (#1108), legacy vmap, microbatched | Yes (same `_leaf_sq_sum`) |
| MPS | Yes (squares also float32) |
| AUTO-S (`auto_scale_pytree`) | Only if `gamma < ~2.6e-23 * sqrt(numel)`: the bound `R` holds while the norm deficit `<= sqrt(numel * 2^-150)` stays below `gamma` |
| `compute_dtype=torch.float64` | Limit moves to ~`1e-154` |

## Fix options (not implemented)

1. **Max-abs pre-scaling** (LAPACK `nrm2` style): divide each example by its
   largest absolute entry before squaring, then multiply back. Robust for all
   magnitudes. Costs an extra reduction pass, plus an argument that the
   roundoff budget still covers the scaled reduction.
2. **Square in float64 for float32/bfloat16/float16 leaves.** Simple and moves
   the limit to ~1e-154. Doubles square bandwidth; MPS has no float64, so it
   needs option 1 or 3 there.
3. **Detect and fall back:** if the computed norm is below a safe floor (for
   example `sqrt(numel) * 2^-63`) while any entry is non-zero, recompute with
   option 1. Cheap on the common path.
4. **Validate C:** reject or warn when `clipping_norm < 1e-19 * sqrt(numel)`.
   Doesn't fix `clip_pytree`'s general contract, but closes every
   training-relevant path.

Any of these changes the privacy-relevant guard. They need the
`.junie/differential-privacy-review.md` protocol and the zero-tolerance bound
tests (`check_bound_strict.py`, `check_bound_adversarial.py`) as a gate.
