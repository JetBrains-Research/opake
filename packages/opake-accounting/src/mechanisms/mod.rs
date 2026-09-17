//! Privacy mechanisms: flat functions producing PLDs.
//!
//! Each function takes scalar parameters and a discretization config,
//! and returns a `PrivacyLossDistribution`. No structs, no traits.
//!
//! - [`gaussian`]: Gaussian mechanism (base noise for DP-SGD)
//! - [`eps_delta`]: Fixed (ε, δ)-DP mechanism
//! - [`identity`]: Identity (zero privacy loss) mechanism
//! - [`non_private`]: Non-private mechanism (infinite privacy loss)

mod eps_delta;
mod gaussian;
mod identity;
mod non_private;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Hard floor for noise multiplier — prevents degenerate grids.
///
/// Extremely small but positive noise multipliers cause epsilon bounds to
/// explode and discretization grids to grow unboundedly. This floor is a
/// numerical safety net; the Python layer warns at a higher threshold.
pub(crate) const MIN_NOISE_MULTIPLIER: f64 = 1e-6;

pub(crate) fn validate_noise_multiplier(nm: f64) -> crate::error::Result<()> {
    if nm < MIN_NOISE_MULTIPLIER {
        return Err(crate::error::PldError::InvalidParameter(format!(
            "noise_multiplier must be >= {}, got {}",
            MIN_NOISE_MULTIPLIER, nm
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

pub use eps_delta::eps_delta_pld;
pub use gaussian::gaussian_pld;
pub use identity::identity_pld;
pub use non_private::non_private_pld;
