//! FFT-based convolution operations
//!
//! This module provides efficient convolution using Fast Fourier Transform (FFT).
//! The convolution is O(n log n) complexity, compared to O(n²) for direct convolution.

use realfft::RealFftPlanner;
use std::sync::Mutex;
use std::sync::OnceLock;

use crate::error::{PldError, Result};

/// Global cached RealFFT planner (initialized on first use)
static REAL_FFT_PLANNER: OnceLock<Mutex<RealFftPlanner<f64>>> = OnceLock::new();

/// Get or initialize the global RealFFT planner
fn get_planner() -> &'static Mutex<RealFftPlanner<f64>> {
    REAL_FFT_PLANNER.get_or_init(|| Mutex::new(RealFftPlanner::new()))
}

fn pow_usize(mut base: f64, mut exponent: usize) -> f64 {
    let mut result = 1.0;

    while exponent > 0 {
        if exponent % 2 == 1 {
            result *= base;
        }
        base *= base;
        exponent /= 2;
    }

    result
}

/// Convolve two real-valued sequences using FFT
///
/// # Arguments
///
/// * `a` - First sequence
/// * `b` - Second sequence
///
/// # Returns
///
/// Convolution of `a` and `b` with length `a.len() + b.len() - 1`
///
/// # Example
///
/// ```
/// use opake_accounting::numerics::fft::convolve;
///
/// let a = vec![1.0, 2.0, 3.0];
/// let b = vec![4.0, 5.0];
/// let result = convolve(&a, &b);
/// assert_eq!(result.len(), 4);
/// ```
///
/// # Algorithm
///
/// Uses RealFFT for real-valued inputs (1.6x faster than complex FFT):
/// 1. Forward real FFT on both inputs (produces complex frequency spectrum)
/// 2. Element-wise multiply in frequency domain
/// 3. Inverse real FFT back to time domain
pub fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }

    if a.len() == 1 && b.len() == 1 {
        return vec![a[0] * b[0]];
    }

    let result_len = a.len() + b.len() - 1;

    // Use next power of 2 for FFT efficiency
    let fft_len = result_len.next_power_of_two();

    // Get cached planner
    let planner = get_planner();
    let mut planner_guard = planner.lock().unwrap();

    // Create forward and inverse FFT plans
    let r2c = planner_guard.plan_fft_forward(fft_len);
    let c2r = planner_guard.plan_fft_inverse(fft_len);

    // Drop the lock before heavy computation
    drop(planner_guard);

    // Prepare input buffers (real-valued, zero-padded)
    let mut a_real = a.to_vec();
    a_real.resize(fft_len, 0.0);

    let mut b_real = b.to_vec();
    b_real.resize(fft_len, 0.0);

    // Allocate output buffers for complex frequency domain
    // RealFFT only needs (fft_len/2 + 1) complex values!
    let mut a_freq = r2c.make_output_vec();
    let mut b_freq = r2c.make_output_vec();

    // Forward FFT: real time domain → complex frequency domain
    r2c.process(&mut a_real, &mut a_freq)
        .expect("RealFFT forward failed");
    r2c.process(&mut b_real, &mut b_freq)
        .expect("RealFFT forward failed");

    // Element-wise multiplication in the frequency domain (in-place on a_freq)
    for i in 0..a_freq.len() {
        a_freq[i] *= b_freq[i];
    }
    // b_freq can be dropped here (saves memory)
    drop(b_freq);

    // Allocate output buffer for inverse FFT
    let mut result = c2r.make_output_vec();

    // Inverse FFT: complex frequency domain → real time domain
    c2r.process(&mut a_freq, &mut result)
        .expect("RealFFT inverse failed");

    // Normalize by FFT length and truncate to result length
    result
        .iter()
        .take(result_len)
        .map(|&x| x / fft_len as f64)
        .collect()
}

/// Self-convolve a sequence `count` times using FFT power
///
/// This is more efficient than repeatedly calling `convolve` for large counts.
/// Uses the property that convolution in time domain = multiplication in frequency domain.
///
/// # Arguments
///
/// * `a` - Sequence to convolve with itself
/// * `count` - Number of times to convolve (must be >= 1)
///
/// # Returns
///
/// Result of convolving `a` with itself `count` times
///
/// # Example
///
/// ```
/// use opake_accounting::numerics::fft::self_convolve;
///
/// let a = vec![1.0, 2.0, 3.0];
/// let result = self_convolve(&a, 3);
/// // Equivalent to convolve(convolve(a, a), a)
/// ```
///
/// # Algorithm
///
/// Uses FFT power method (O(log n) FFTs vs. O(n) for repeated convolution):
/// 1. Forward FFT once
/// 2. Raise frequency spectrum to power `count`
/// 3. Inverse FFT once
///
/// # Performance
///
/// For count=100: 2 FFTs total vs. 199 FFTs for repeated convolution!
pub fn self_convolve(a: &[f64], count: usize) -> Result<Vec<f64>> {
    self_convolve_with_bounds(a, count, None, false)
}

/// Maximum FFT buffer size for linear (non-wrapping) convolution.
///
/// When computing `IFFT(FFT(a)^k)`, the FFT operates on a fixed-length
/// buffer. If that buffer is shorter than the true result of convolving
/// `a` with itself `k` times, the result "wraps around" — high-index
/// entries alias onto low-index entries, corrupting the output.
///
/// To prevent this, the FFT buffer must be ≥ `full_result_len` (zero-padded
/// to next power of 2). This constant caps how large that buffer can be:
/// 32M f64 elements ≈ 256 MB, sufficient for k ≈ 240 with typical 140k grids.
///
/// Beyond this limit, only compositions with a tail-truncation budget may use
/// a smaller circular FFT buffer. Exact compositions return an error instead
/// of allowing wrapping artifacts.
const MAX_LINEAR_FFT_SIZE: usize = 32 * 1024 * 1024;

#[derive(Debug, PartialEq, Eq)]
enum SelfConvolutionStrategy {
    Linear { fft_len: usize },
    Circular { fft_len: usize },
}

pub(crate) fn self_convolution_full_result_len(input_len: usize, count: usize) -> Result<usize> {
    input_len
        .checked_add((count - 1).checked_mul(input_len - 1).ok_or_else(|| {
            PldError::NumericalError("self-composition result length overflow".into())
        })?)
        .ok_or_else(|| PldError::NumericalError("self-composition result length overflow".into()))
}

fn select_self_convolution_strategy(
    input_len: usize,
    count: usize,
    truncated_len: usize,
    allow_circular_fallback: bool,
) -> Result<SelfConvolutionStrategy> {
    let full_result_len = self_convolution_full_result_len(input_len, count)?;
    let alias_free_fft_len = full_result_len
        .checked_next_power_of_two()
        .ok_or_else(|| PldError::NumericalError("self-composition FFT length overflow".into()))?;

    if allow_circular_fallback {
        let circular_fft_len = truncated_len
            .max(input_len)
            .checked_next_power_of_two()
            .ok_or_else(|| {
                PldError::NumericalError("self-composition FFT length overflow".into())
            })?;
        let materially_smaller = circular_fft_len <= alias_free_fft_len / 4;
        if circular_fft_len <= MAX_LINEAR_FFT_SIZE
            && (alias_free_fft_len > MAX_LINEAR_FFT_SIZE || materially_smaller)
        {
            return Ok(SelfConvolutionStrategy::Circular {
                fft_len: circular_fft_len,
            });
        }
    }

    if alias_free_fft_len <= MAX_LINEAR_FFT_SIZE {
        Ok(SelfConvolutionStrategy::Linear {
            fft_len: alias_free_fft_len,
        })
    } else {
        Err(PldError::SelfCompositionTooLarge {
            requested: alias_free_fft_len,
            maximum: MAX_LINEAR_FFT_SIZE,
        })
    }
}

/// Self-convolve with optional truncation bounds.
///
/// Computes `a` convolved with itself `count` times, optionally returning
/// only the `[lower, upper]` window of the result.
///
/// # Linear vs circular convolution
///
/// FFT-based exponentiation (`FFT(a)^k`) computes **circular** convolution
/// modulo the FFT buffer length. If the buffer is shorter than the true
/// linear convolution result (`a.len() + (k-1)*(a.len()-1)`), high-index
/// output wraps around and corrupts the result.
///
/// This function uses a full-size buffer (≥ `full_result_len`, rounded to
/// next power of 2) to guarantee correct linear convolution. When that
/// buffer is materially larger than the bounded circular transform, a
/// composition with a positive tail-truncation budget uses the smaller buffer
/// (`max(truncated_len, a.len())`). The excluded tail mass is already covered
/// by that budget. Exact composition rejects transforms above
/// [`MAX_LINEAR_FFT_SIZE`] instead: circular aliasing is invalid without a
/// tail-error budget.
///
/// # Arguments
///
/// * `a` - Sequence to convolve with itself
/// * `count` - Number of times to convolve (must be >= 1, and `<= u32::MAX`
///   for multi-entry inputs; the singleton path supports full `usize`
///   exponents)
/// * `bounds` - Optional (lower, upper) indices to compute. If None, computes full result.
/// * `allow_circular_fallback` - Whether a positive tail-truncation budget
///   permits a materially smaller bounded circular FFT.
///
/// # Returns
///
/// Result of convolving `a` with itself `count` times, truncated to [lower, upper]
/// if bounds provided. The returned vector has length (upper - lower + 1).
pub fn self_convolve_with_bounds(
    a: &[f64],
    count: usize,
    bounds: Option<(usize, usize)>,
    allow_circular_fallback: bool,
) -> Result<Vec<f64>> {
    if count == 0 {
        return Ok(vec![1.0]); // Identity for convolution
    }

    if count == 1 {
        return Ok(a.to_vec());
    }

    if a.is_empty() {
        return Ok(vec![]);
    }

    if a.len() == 1 {
        return Ok(vec![pow_usize(a[0], count)]);
    }

    // The FFT power path raises each frequency bin with `powu(count as u32)`;
    // reject counts above u32 instead of silently truncating them, mirroring
    // the Python composition-count boundary.
    if count > u32::MAX as usize {
        return Err(PldError::InvalidParameter(format!(
            "self-composition count must not exceed {}",
            u32::MAX
        )));
    }

    let full_result_len = self_convolution_full_result_len(a.len(), count)?;
    let (lower_bound, upper_bound) = bounds.unwrap_or((0, full_result_len - 1));
    let truncated_len = upper_bound - lower_bound + 1;

    let fft_len = match select_self_convolution_strategy(
        a.len(),
        count,
        truncated_len,
        allow_circular_fallback,
    )? {
        SelfConvolutionStrategy::Linear { fft_len }
        | SelfConvolutionStrategy::Circular { fft_len } => fft_len,
    };

    let planner = get_planner();
    let mut planner_guard = planner.lock().unwrap();
    let r2c = planner_guard.plan_fft_forward(fft_len);
    let c2r = planner_guard.plan_fft_inverse(fft_len);
    drop(planner_guard);

    let mut a_real = a.to_vec();
    a_real.resize(fft_len, 0.0);

    let mut a_freq = r2c.make_output_vec();
    r2c.process(&mut a_real, &mut a_freq)
        .expect("RealFFT forward failed");

    // Raise to power element-wise
    for c in a_freq.iter_mut() {
        *c = c.powu(count as u32);
    }

    let mut result = c2r.make_output_vec();
    c2r.process(&mut a_freq, &mut result)
        .expect("RealFFT inverse failed");

    // Extract window using circular indexing (needed for both strategies:
    // for alias-free the modulo is a no-op since fft_len >= full_result_len)
    Ok((0..truncated_len)
        .map(|i| {
            let idx = (lower_bound + i) % fft_len;
            result[idx] / fft_len as f64
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_convolve_simple() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0];
        let result = convolve(&a, &b);

        // Hand-calculated: [1*4, 1*5+2*4, 2*5+3*4, 3*5]
        assert_eq!(result.len(), 4);
        assert_relative_eq!(result[0], 4.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 13.0, epsilon = 1e-10);
        assert_relative_eq!(result[2], 22.0, epsilon = 1e-10);
        assert_relative_eq!(result[3], 15.0, epsilon = 1e-10);
    }

    #[test]
    fn test_convolve_identity() {
        let a = vec![1.0, 2.0, 3.0];
        let identity = vec![1.0];
        let result = convolve(&a, &identity);

        assert_eq!(result.len(), 3);
        for (i, &val) in result.iter().enumerate() {
            assert_relative_eq!(val, a[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_convolve_commutative() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let result1 = convolve(&a, &b);
        let result2 = convolve(&b, &a);

        assert_eq!(result1.len(), result2.len());
        for (v1, v2) in result1.iter().zip(result2.iter()) {
            assert_relative_eq!(v1, v2, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_self_convolve() {
        let a = vec![1.0, 2.0];

        // Self-convolve twice
        let result = self_convolve(&a, 2).unwrap();

        // Should equal convolve(a, a)
        let expected = convolve(&a, &a);

        assert_eq!(result.len(), expected.len());
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(r, e, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_self_convolve_three_times() {
        let a = vec![1.0, 1.0];

        // Convolve [1, 1] with itself 3 times
        let result = self_convolve(&a, 3).unwrap();

        // Manual: conv([1,1], [1,1]) = [1,2,1]
        //         conv([1,2,1], [1,1]) = [1,3,3,1]
        assert_eq!(result.len(), 4);
        assert_relative_eq!(result[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 3.0, epsilon = 1e-10);
        assert_relative_eq!(result[2], 3.0, epsilon = 1e-10);
        assert_relative_eq!(result[3], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_singleton_self_convolution_supports_maximum_exponent() {
        let result = self_convolve(&[1.0], u32::MAX as usize).unwrap();

        assert_eq!(result, vec![1.0]);
    }

    #[test]
    fn test_singleton_self_convolution_preserves_large_odd_exponent() {
        let result = self_convolve(&[-1.0], usize::MAX).unwrap();

        assert_eq!(result, vec![-1.0]);
    }

    #[test]
    fn test_fft_self_convolution_rejects_count_above_u32_max() {
        // Counts above u32::MAX must be rejected on the FFT path (including
        // the budgeted circular fallback) instead of silently truncating the
        // exponent, mirroring the Python composition-count boundary.
        let count = u32::MAX as usize + 1;

        let err = self_convolve_with_bounds(&[0.5, 0.5], count, Some((0, 15)), true).unwrap_err();
        assert!(matches!(err, PldError::InvalidParameter(_)));

        let err = self_convolve(&[0.5, 0.5], count).unwrap_err();
        assert!(matches!(err, PldError::InvalidParameter(_)));
    }

    #[test]
    fn test_circular_self_convolution_accepts_count_at_u32_max() {
        let result =
            self_convolve_with_bounds(&[0.5, 0.5], u32::MAX as usize, Some((0, 15)), true).unwrap();

        assert_eq!(result.len(), 16);
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_empty_sequences() {
        assert!(convolve(&[], &[1.0]).is_empty());
        assert!(convolve(&[1.0], &[]).is_empty());
        assert!(self_convolve(&[], 5).unwrap().is_empty());
    }

    #[test]
    fn test_oversized_exact_convolution_is_rejected_without_allocating() {
        let count = MAX_LINEAR_FFT_SIZE + 1;

        assert!(matches!(
            select_self_convolution_strategy(2, count, count + 1, false),
            Err(PldError::SelfCompositionTooLarge {
                requested,
                maximum,
            }) if requested > MAX_LINEAR_FFT_SIZE && maximum == MAX_LINEAR_FFT_SIZE
        ));
    }

    #[test]
    fn test_oversized_truncated_convolution_uses_bounded_circular_fft() {
        let count = MAX_LINEAR_FFT_SIZE + 1;

        assert_eq!(
            select_self_convolution_strategy(2, count, 16, true).unwrap(),
            SelfConvolutionStrategy::Circular { fft_len: 16 }
        );
    }

    #[test]
    fn test_truncated_convolution_prefers_smaller_circular_fft() {
        assert_eq!(
            select_self_convolution_strategy(20_000, 1_000, 64_000, true).unwrap(),
            SelfConvolutionStrategy::Circular { fft_len: 65_536 }
        );
        assert_eq!(
            select_self_convolution_strategy(20_000, 1_000, 64_000, false).unwrap(),
            SelfConvolutionStrategy::Linear {
                fft_len: 33_554_432
            }
        );
        assert_eq!(
            select_self_convolution_strategy(100, 2, 100, true).unwrap(),
            SelfConvolutionStrategy::Linear { fft_len: 256 }
        );
    }

    #[test]
    fn test_large_sequence() {
        // Test with larger sequences to verify FFT efficiency
        let a: Vec<f64> = (0..1000).map(|i| (i as f64) / 1000.0).collect();
        let b: Vec<f64> = (0..1000).map(|i| 1.0 - (i as f64) / 1000.0).collect();

        let result = convolve(&a, &b);

        assert_eq!(result.len(), 1999);
        // Just verify it completes without panicking
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_planner_caching() {
        // Test that the planner is properly cached
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0];

        // First call initializes planner
        let result1 = convolve(&a, &b);

        // Second call should reuse cached planner
        let result2 = convolve(&a, &b);

        // Results should be identical
        assert_eq!(result1.len(), result2.len());
        for (v1, v2) in result1.iter().zip(result2.iter()) {
            assert_relative_eq!(v1, v2, epsilon = 1e-10);
        }
    }
}
