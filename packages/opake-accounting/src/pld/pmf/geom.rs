//! Exponentiated-PLD carrier on a geometric grid.
//!
//! The random-allocation transform (Feldman & Shenfeld, arXiv:2602.17284)
//! needs the law of a **sum of exponentiated** privacy losses,
//! `e^{L₁} + e^{L₂}`. This is not composition and **not** an FFT:
//!
//! - Additive convolution in index space gives the law of `L₁ + L₂`, i.e. of
//!   the *product* `e^{L₁}·e^{L₂}`. That is composition.
//! - The values here are geometric (`v_i = v_min·rⁱ`), and a *sum* of two such
//!   values is not on the grid.
//!
//! Hence a direct `O(n·m)` pass plus a range renormalisation. The saving grace
//! is that renormalisation keeps the bin count fixed: if both operands span
//! `[A, A·r^{n-1}]` and `[B, B·r^{n-1}]`, every sum lies in
//! `[A+B, (A+B)·r^{n-1}]` — the same `n` points at the same ratio, only the
//! anchor moves. So `n` does not grow through the recursion.
//!
//! Everything is held in log-space: `σ = 1`, `β = 1e-10` already puts the
//! exponentiated grid across ~950 decades.
//!
//! This type is never a `PrivacyLossDistribution`. It is entered via
//! [`GeomPmf::from_pmf_exp`] and left via [`GeomPmf::into_pmf_log`].

use crate::error::{PldError, Result};
use crate::numerics::logspace::log_add;
use crate::pld::pmf::Pmf;
use crate::pld::realization::Rounding;
use rayon::prelude::*;

/// Row chunks the `O(n²)` convolution core is split into.
///
/// Fixed rather than derived from the thread count, so the order in which
/// partial sums are combined — and hence the exact bits of the result — is
/// the same on every machine. Large enough to keep every core busy, small
/// enough that the `ROW_CHUNKS · n` scratch stays modest (8 MB at n = 16,384).
const ROW_CHUNKS: usize = 64;

/// Move a finite floating-point value one representable step in `dir`.
///
/// Rust 1.70 does not expose `f64::next_up` / `next_down`, so keep the tiny
/// bit-level implementation local to the directional geometric arithmetic.
fn directed_next(value: f64, dir: Rounding) -> f64 {
    if value.is_nan() {
        return value;
    }

    match dir {
        Rounding::Up if value == f64::INFINITY => value,
        Rounding::Down if value == f64::NEG_INFINITY => value,
        Rounding::Up if value == 0.0 => f64::from_bits(1),
        Rounding::Down if value == 0.0 => -f64::from_bits(1),
        Rounding::Up if value > 0.0 => f64::from_bits(value.to_bits() + 1),
        Rounding::Up => f64::from_bits(value.to_bits() - 1),
        Rounding::Down if value > 0.0 => f64::from_bits(value.to_bits() - 1),
        Rounding::Down => f64::from_bits(value.to_bits() + 1),
    }
}

/// Directionally round `ln(exp(a) + exp(b))`.
fn directed_log_add(a: f64, b: f64, dir: Rounding) -> f64 {
    directed_next(log_add(a, b), dir)
}

/// PMF over `[0, v_min·r⁰, …, v_min·r^{n-1}, +∞]`, stored in log-space.
#[derive(Debug, Clone)]
pub(crate) struct GeomPmf {
    /// `ln(v_min)` — the smallest interior value.
    log_v_min: f64,
    /// `ln(r)` — constant ratio between consecutive interior points.
    log_ratio: f64,
    /// Interior masses.
    probs: Vec<f64>,
    /// Mass at value 0 (the image of a `−∞` privacy loss under `exp`).
    zero_mass: f64,
    /// Mass at `+∞`.
    infinity_mass: f64,
}

impl GeomPmf {
    #[inline]
    fn log_value(&self, i: usize) -> f64 {
        self.log_v_min + (i as f64) * self.log_ratio
    }

    #[inline]
    fn n(&self) -> usize {
        self.probs.len()
    }

    /// `exp(L)` for a PLD on Opake's uniform loss grid.
    ///
    /// The uniform grid `l_i = (lower + i)·Δ` exponentiates to a geometric grid
    /// with `ln r = Δ`. `−∞` mass becomes the 0-atom; `+∞` mass is carried
    /// through.
    pub(crate) fn from_pmf_exp(pmf: &Pmf) -> Result<Self> {
        if pmf.probs.is_empty() {
            return Err(PldError::InvalidParameter("empty PMF".into()));
        }
        Ok(GeomPmf {
            log_v_min: pmf.lower_loss_index as f64 * pmf.discretization,
            log_ratio: pmf.discretization,
            probs: pmf.probs.clone(),
            zero_mass: pmf.negative_infinity_mass,
            infinity_mass: pmf.infinity_mass,
        })
    }

    /// `exp(-L)`: negate the loss, which reverses the grid.
    pub(crate) fn from_pmf_exp_neg(pmf: &Pmf) -> Result<Self> {
        if pmf.probs.is_empty() {
            return Err(PldError::InvalidParameter("empty PMF".into()));
        }
        let n = pmf.probs.len();
        let hi = pmf.lower_loss_index + n as i64 - 1;
        let mut probs = pmf.probs.clone();
        probs.reverse();
        Ok(GeomPmf {
            // Smallest value of e^{-l} corresponds to the largest l.
            log_v_min: -(hi as f64) * pmf.discretization,
            log_ratio: pmf.discretization,
            probs,
            // e^{-(+inf)} = 0 and e^{-(-inf)} = +inf: the atoms swap.
            zero_mass: pmf.infinity_mass,
            infinity_mass: pmf.negative_infinity_mass,
        })
    }

    /// Extend the interior grid upward with zero mass so two operands match.
    fn padded_to(&self, n_target: usize) -> GeomPmf {
        let mut out = self.clone();
        if out.probs.len() < n_target {
            out.probs.resize(n_target, 0.0);
        }
        out
    }

    /// Directional output bin offsets indexed by `d = i - j`.
    ///
    /// With a common geometric ratio `R`, `logsumexp(A+iR, B+jR)` is
    /// `jR + logsumexp(A+dR, B)`. Therefore the output bin is
    /// `j + K[d]`, and this O(n) table replaces the transcendental work in
    /// every one of the O(n²) input pairs.
    fn convolution_index_table(
        a: &GeomPmf,
        b: &GeomPmf,
        log_v_min: f64,
        n: usize,
        dir: Rounding,
    ) -> Vec<i64> {
        let center = n - 1;
        (0..2 * n - 1)
            .map(|offset| {
                let d = offset as i64 - center as i64;
                // `A + B` is exactly the new grid anchor algebraically. The
                // directionally rounded stored anchor can differ by an ulp,
                // so preserve the identity rather than turning it into ±1.
                if d == 0 {
                    return 0;
                }

                let log_sum =
                    directed_log_add(a.log_v_min + (d as f64) * a.log_ratio, b.log_v_min, dir);
                let pos = directed_next((log_sum - log_v_min) / a.log_ratio, dir);
                match dir {
                    Rounding::Up => pos.ceil() as i64,
                    Rounding::Down => pos.floor() as i64,
                }
            })
            .collect()
    }

    /// Convolve two exp-PLDs. `dir` decides which way off-grid sums are moved.
    pub(crate) fn conv(&self, other: &GeomPmf, dir: Rounding) -> Result<GeomPmf> {
        if (self.log_ratio - other.log_ratio).abs() > 1e-12 * self.log_ratio.abs().max(1.0) {
            return Err(PldError::InvalidParameter(format!(
                "geometric grids must share a ratio: {} vs {}",
                self.log_ratio, other.log_ratio
            )));
        }
        let n = self.n().max(other.n());
        let a = self.padded_to(n);
        let b = other.padded_to(n);

        // Range-renorm: the anchor moves to a[0] + b[0]; ratio and bin count
        // are unchanged (see the module docstring). Store the anchor in the
        // requested direction so the grid itself preserves the bound.
        let log_v_min = directed_log_add(a.log_v_min, b.log_v_min, dir);
        let log_ratio = a.log_ratio;
        let index_table = Self::convolution_index_table(&a, &b, log_v_min, n, dir);
        let mut probs = vec![0.0f64; n];
        let mut infinity_mass = 0.0f64;
        let mut zero_mass = 0.0f64;

        // interior × interior — the O(n²) core, parallel over rows of `a`.
        //
        // Rows are split into a fixed number of chunks derived from `n` alone,
        // and the per-chunk accumulators are folded back in chunk-index order.
        // rayon's `fold`/`reduce` would instead split by thread count and
        // work-stealing, so the summation order — and with it the last ulp of
        // the reported ε — would depend on how many cores the machine has.
        // Determinism across machines is the reason this transform replaced a
        // Monte Carlo primitive, so it is worth the fixed chunking.
        // Guarded by tests/test_random_allocation_reproducible.py.
        let chunk_size = (n + ROW_CHUNKS - 1) / ROW_CHUNKS.max(1);
        let partials: Vec<(Vec<f64>, f64, f64)> = (0..n)
            .into_par_iter()
            .chunks(chunk_size.max(1))
            .map(|rows| {
                let mut acc = vec![0.0f64; n];
                let mut inf = 0.0f64;
                let mut zero = 0.0f64;
                for i in rows {
                    let pa = a.probs[i];
                    if pa == 0.0 {
                        continue;
                    }
                    for j in 0..n {
                        let pb = b.probs[j];
                        if pb == 0.0 {
                            continue;
                        }
                        let idx = j as i64 + index_table[i + n - 1 - j];
                        let mass = pa * pb;
                        if idx < 0 {
                            match dir {
                                Rounding::Up => acc[0] += mass,
                                Rounding::Down => zero += mass,
                            }
                        } else if idx >= n as i64 {
                            match dir {
                                Rounding::Up => inf += mass,
                                Rounding::Down => acc[n - 1] += mass,
                            }
                        } else {
                            acc[idx as usize] += mass;
                        }
                    }
                }
                (acc, inf, zero)
            })
            .collect();
        // `collect` on an indexed parallel iterator preserves chunk order, so
        // this fold is the same sequence of additions on every machine.
        for (acc, inf, zero) in &partials {
            for (p, q) in probs.iter_mut().zip(acc.iter()) {
                *p += *q;
            }
            infinity_mass += *inf;
            zero_mass += *zero;
        }

        let mut place = |log_val: f64, mass: f64, probs: &mut Vec<f64>, inf: &mut f64| {
            if mass == 0.0 {
                return;
            }
            let pos = directed_next((log_val - log_v_min) / log_ratio, dir);
            let idx = match dir {
                Rounding::Up => pos.ceil(),
                Rounding::Down => pos.floor(),
            };
            if idx < 0.0 {
                // Only reachable via the 0-atom identity below; round toward
                // the requested direction.
                match dir {
                    Rounding::Up => probs[0] += mass,
                    Rounding::Down => zero_mass += mass,
                }
            } else if idx >= n as f64 {
                match dir {
                    Rounding::Up => *inf += mass,
                    Rounding::Down => probs[n - 1] += mass,
                }
            } else {
                probs[idx as usize] += mass;
            }
        };

        // 0 is the additive identity: 0 + v = v.
        if a.zero_mass > 0.0 {
            for j in 0..n {
                place(
                    b.log_value(j),
                    a.zero_mass * b.probs[j],
                    &mut probs,
                    &mut infinity_mass,
                );
            }
        }
        if b.zero_mass > 0.0 {
            for i in 0..n {
                place(
                    a.log_value(i),
                    b.zero_mass * a.probs[i],
                    &mut probs,
                    &mut infinity_mass,
                );
            }
        }
        zero_mass += a.zero_mass * b.zero_mass;

        // +∞ absorbs.
        infinity_mass += a.infinity_mass + b.infinity_mass - a.infinity_mass * b.infinity_mass;

        Ok(GeomPmf {
            log_v_min,
            log_ratio,
            probs,
            zero_mass,
            infinity_mass,
        })
    }

    /// `t`-fold self-convolution by exponentiation by squaring.
    ///
    /// Uses exactly `⌊log₂ t⌋ + popcount(t) − 1` pairwise convolutions.
    pub(crate) fn self_conv(&self, t: usize, dir: Rounding) -> Result<GeomPmf> {
        if t == 0 {
            return Err(PldError::InvalidParameter("t must be >= 1".into()));
        }
        let mut base = self.clone();
        let mut acc: Option<GeomPmf> = None;
        let mut rem = t;
        while rem > 0 {
            if rem % 2 == 1 {
                acc = Some(match acc {
                    None => base.clone(),
                    Some(a) => base.conv(&a, dir)?,
                });
            }
            rem /= 2;
            if rem > 0 {
                base = base.conv(&base, dir)?;
            }
        }
        Ok(acc.expect("t >= 1"))
    }

    /// `ln(V / scale)` back onto Opake's uniform loss grid of width
    /// `discretization`.
    pub(crate) fn into_pmf_log(
        self,
        scale: f64,
        discretization: f64,
        max_grid_size: usize,
        dir: Rounding,
    ) -> Result<Pmf> {
        if scale.is_nan() || scale <= 0.0 {
            return Err(PldError::InvalidParameter("scale must be > 0".into()));
        }
        let shift = scale.ln();
        let n = self.n();

        let loss_at = |i: usize| self.log_value(i) - shift;
        let (l_lo, l_hi) = (loss_at(0), loss_at(n - 1));
        let idx_lo = (l_lo / discretization).floor() as i64;
        let idx_hi = (l_hi / discretization).ceil() as i64;
        let out_n = (idx_hi - idx_lo + 1) as usize;
        if out_n > max_grid_size {
            return Err(PldError::InvalidParameter(format!(
                "output grid of {} points exceeds max_grid_size {}",
                out_n, max_grid_size
            )));
        }

        let mut probs = vec![0.0f64; out_n];
        for i in 0..n {
            if self.probs[i] == 0.0 {
                continue;
            }
            let pos = directed_next(loss_at(i) / discretization, dir);
            let k = match dir {
                Rounding::Up => pos.ceil(),
                Rounding::Down => pos.floor(),
            } as i64
                - idx_lo;
            let k = k.clamp(0, out_n as i64 - 1) as usize;
            probs[k] += self.probs[i];
        }

        Ok(Pmf {
            discretization,
            lower_loss_index: idx_lo,
            probs,
            infinity_mass: self.infinity_mass,
            // ln(0) = -inf: the 0-atom becomes a -inf privacy loss.
            negative_infinity_mass: self.zero_mass,
            max_grid_size,
            right_tail_budget: 0.0,
            left_tail_budget: 0.0,
        })
    }

    /// `-ln(V / scale)` — the add direction's final step.
    pub(crate) fn into_pmf_neg_log(
        self,
        scale: f64,
        discretization: f64,
        max_grid_size: usize,
        dir: Rounding,
    ) -> Result<Pmf> {
        // Negating the loss flips the rounding direction it must respect.
        let flipped = match dir {
            Rounding::Up => Rounding::Down,
            Rounding::Down => Rounding::Up,
        };
        let pmf = self.into_pmf_log(scale, discretization, max_grid_size, flipped)?;
        let n = pmf.probs.len();
        let hi = pmf.lower_loss_index + n as i64 - 1;
        let mut probs = pmf.probs;
        probs.reverse();
        Ok(Pmf {
            discretization: pmf.discretization,
            lower_loss_index: -hi,
            probs,
            infinity_mass: pmf.negative_infinity_mass,
            negative_infinity_mass: pmf.infinity_mass,
            max_grid_size,
            right_tail_budget: 0.0,
            left_tail_budget: 0.0,
        })
    }

    #[cfg(test)]
    pub(crate) fn total_mass(&self) -> f64 {
        self.probs.iter().sum::<f64>() + self.zero_mass + self.infinity_mass
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pld::realization::{disc_dist, NormalLoss};
    use approx::assert_relative_eq;

    fn gauss_exp(sigma: f64, alpha: f64) -> GeomPmf {
        let pmf = disc_dist(&NormalLoss::gaussian(sigma), alpha, 1e-10, 10_000_000).unwrap();
        GeomPmf::from_pmf_exp(&pmf).unwrap()
    }

    #[test]
    fn test_conv_conserves_mass() {
        let g = gauss_exp(1.0, 5e-2);
        let c = g.conv(&g, Rounding::Up).unwrap();
        assert_relative_eq!(c.total_mass(), 1.0, epsilon = 1e-9);
    }

    #[test]
    fn test_bin_count_is_stable() {
        // The renormalisation must keep n fixed — this is what makes large t
        // tractable.
        let g = gauss_exp(1.0, 5e-2);
        let n = g.n();
        let mut cur = g.clone();
        for _ in 0..6 {
            cur = cur.conv(&g, Rounding::Up).unwrap();
            assert_eq!(cur.n(), n, "bin count drifted");
        }
    }

    #[test]
    fn test_self_conv_matches_sequential() {
        let g = gauss_exp(1.0, 5e-2);
        for t in 1..=9usize {
            let by_squaring = g.self_conv(t, Rounding::Up).unwrap();
            let mut seq = g.clone();
            for _ in 1..t {
                seq = seq.conv(&g, Rounding::Up).unwrap();
            }
            // Each reduction path is directionally rounded independently, so
            // their anchors may differ by a few ulps. Both remain safe.
            assert!(
                (by_squaring.log_v_min - seq.log_v_min).abs() < 1e-10,
                "{} vs {}",
                by_squaring.log_v_min,
                seq.log_v_min
            );
            assert_relative_eq!(by_squaring.total_mass(), 1.0, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_self_conv_convolution_count() {
        // ⌊log₂ t⌋ + popcount(t) − 1, not the 2⌈log₂ t⌉ analysis bound.
        for &(t, want) in &[(1usize, 0usize), (2, 1), (8, 3), (13, 5), (100, 8)] {
            let got = (t.ilog2() as usize) + (t.count_ones() as usize) - 1;
            assert_eq!(got, want, "t={}", t);
        }
    }

    #[test]
    fn test_round_trip_identity() {
        // exp then log with scale 1 recovers the original support.
        let alpha = 1e-2;
        let pmf = disc_dist(&NormalLoss::gaussian(1.0), alpha, 1e-10, 10_000_000).unwrap();
        let back = GeomPmf::from_pmf_exp(&pmf)
            .unwrap()
            .into_pmf_log(1.0, alpha, 10_000_000, Rounding::Up)
            .unwrap();
        assert_eq!(back.lower_loss_index, pmf.lower_loss_index);
        let m: f64 = back.probs.iter().sum::<f64>() + back.infinity_mass;
        assert_relative_eq!(m, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn test_index_table_preserves_direction_for_unequal_anchors() {
        let a = GeomPmf {
            log_v_min: -1.3,
            log_ratio: 0.07,
            probs: vec![0.2, 0.3, 0.5],
            zero_mass: 0.0,
            infinity_mass: 0.0,
        };
        let b = GeomPmf {
            log_v_min: 0.4,
            log_ratio: 0.07,
            probs: vec![0.5, 0.2, 0.3],
            zero_mass: 0.0,
            infinity_mass: 0.0,
        };
        let n = a.n();

        for dir in [Rounding::Up, Rounding::Down] {
            let anchor = directed_log_add(a.log_v_min, b.log_v_min, dir);
            let table = GeomPmf::convolution_index_table(&a, &b, anchor, n, dir);
            assert_eq!(table[n - 1], 0, "d=0 must be exact");

            for i in 0..n {
                for j in 0..n {
                    let idx = j as i64 + table[i + n - 1 - j];
                    let selected = match (dir, idx) {
                        (Rounding::Up, idx) if idx >= n as i64 => f64::INFINITY,
                        (Rounding::Down, idx) if idx < 0 => f64::NEG_INFINITY,
                        (_, idx) if idx < 0 => anchor,
                        (_, idx) if idx >= n as i64 => anchor + (n - 1) as f64 * a.log_ratio,
                        (_, idx) => anchor + idx as f64 * a.log_ratio,
                    };
                    let direct = log_add(a.log_value(i), b.log_value(j));
                    match dir {
                        Rounding::Up => {
                            assert!(selected >= direct, "i={i}, j={j}: {selected} < {direct}")
                        }
                        Rounding::Down => {
                            assert!(selected <= direct, "i={i}, j={j}: {selected} > {direct}")
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_downward_zero_identity_routes_to_zero_mass() {
        let a = GeomPmf {
            log_v_min: 0.0,
            log_ratio: 0.1,
            probs: vec![0.5],
            zero_mass: 0.5,
            infinity_mass: 0.0,
        };
        let b = GeomPmf {
            log_v_min: 0.0,
            log_ratio: 0.1,
            probs: vec![1.0],
            zero_mass: 0.0,
            infinity_mass: 0.0,
        };

        let out = a.conv(&b, Rounding::Down).unwrap();
        assert_relative_eq!(out.zero_mass, 0.5, epsilon = 1e-15);
        assert_relative_eq!(out.probs[0], 0.5, epsilon = 1e-15);
    }

    #[test]
    fn test_rejects_mismatched_ratio() {
        let a = gauss_exp(1.0, 1e-2);
        let b = gauss_exp(1.0, 2e-2);
        assert!(a.conv(&b, Rounding::Up).is_err());
    }
}
