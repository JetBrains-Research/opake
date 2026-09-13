"""Type stubs for the ``opaque.accounting`` Rust extension module.

``opaque.accounting`` provides flat PLD-based privacy accounting.
Rust computes PLDs; Python owns composition and dispatch.

The extension exports two classes (:class:`Pld` and :class:`DiscretizationConfig`)
and several functions for creating PLDs from mechanism parameters.

Example::

    import opaque.accounting as dp

    pld = dp.gaussian_pld(1.1)
    composed = pld.self_compose(1000)
    print(composed.epsilon_at(1e-5))
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Pld — opaque PLD handle
# ---------------------------------------------------------------------------

class Pld:
    """An opaque Privacy Loss Distribution (PLD).

    Created by mechanism/amplification functions.  Supports privacy metric
    queries and composition.

    Example::

        import opaque.accounting as dp
        pld = dp.gaussian_pld(1.1)
        print(pld.epsilon_at(1e-5))
        composed = pld.self_compose(1000)
        print(composed.epsilon_at(1e-5))
    """

    def epsilon_at(self, delta: float) -> float:
        """Smallest ε achieving (ε, δ)-DP.

        Args:
            delta: Failure probability (typically 1e-5 to 1e-7).

        Returns:
            Epsilon value.
        """

    def delta_at(self, epsilon: float) -> float:
        """Smallest δ achieving (ε, δ)-DP.

        Always returns at least ``infinity_mass``; if the returned value equals
        ``infinity_mass``, the tail-truncation budget is exhausted and the true
        δ may be smaller than what is reported.

        Args:
            epsilon: Privacy budget. Must be >= 0.

        Returns:
            Delta value.
        """

    @property
    def infinity_mass(self) -> float:
        """Worst-case infinity mass over both adjacency types.

        ``delta_at(ε)`` is always ≥ ``infinity_mass``; equality means the
        configured tail-truncation budget is exhausted and the true δ may be
        smaller.
        """

    @property
    def mc_failure_probability(self) -> float:
        """Failure probability of the Monte Carlo confidence event."""

    @property
    def mc_confidence(self) -> float:
        """Confidence level of the Monte Carlo PLD bound."""

    @property
    def mc_resolution(self) -> float:
        """Unresolved Monte Carlo probability mass in delta units."""

    def advantage(self) -> float:
        """Total-variation advantage.

        Returns:
            Advantage in [0, 1]. Lower is more private.
        """

    def beta_at(self, alpha: float) -> float:
        """Type-II error (β) at a given Type-I error (α).

        Args:
            alpha: Type-I error rate in [0, 1].

        Returns:
            Beta value in [0, 1].
        """

    def risk_at(self, prior: float) -> float:
        """Bayes risk under an optimal adversary.

        Args:
            prior: Prior probability (typically 0.5).

        Returns:
            Risk value in [0, 0.5].
        """

    def compose(self, other: Pld) -> Pld:
        """Compose this PLD with another (heterogeneous composition).

        Args:
            other: The other PLD to compose with.

        Returns:
            A new composed PLD.

        Raises:
            ValueError: If composition fails (mismatched discretization).
        """

    def self_compose(self, count: int) -> Pld:
        """Self-compose this PLD *count* times (homogeneous repetition).

        Args:
            count: Repetition count. Must satisfy ``1 <= count <= 2**32 - 1``.

        Raises:
            ValueError: If count is not positive.
            OverflowError: If count exceeds 2**32 - 1.

        Returns:
            A new self-composed PLD.
        """

    def __mul__(self, count: int) -> Pld:
        """``pld * k`` is shorthand for ``pld.self_compose(k)``."""

    def __rmul__(self, count: int) -> Pld:
        """``k * pld`` also works."""

    def __or__(self, other: Pld) -> Pld:
        """``a | b`` is shorthand for ``a.compose(b)``."""

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

# ---------------------------------------------------------------------------
# DiscretizationConfig — discretization configuration
# ---------------------------------------------------------------------------

class DiscretizationConfig:
    """Discretization configuration for PLD computation.

    Args:
        discretization: Grid spacing for the PLD PMF. Default: 1e-4.
            Smaller values give tighter bounds but use more memory.
        log_mass_truncation_bound: Log tail mass cutoff. Default: -50.0.
            Tails below exp(bound) are truncated.
        max_grid_size: Maximum grid bins before automatic coarsening.
            Default: 10,000,000.
        tail_mass_truncation: Total tail mass budget for Chernoff truncation
            during composition. Default: 1e-15.
        seed: RNG seed for reproducibility across worker counts within a fixed
            Opaque and dependency build. Exact streams may change across
            releases. Default: 42.
        max_conv_grid: Maximum convolution grid size for the
            random-allocation PLD transform. Default: 32,768.
        mc_resolution: Maximum unresolved Monte Carlo mass. Default: 1e-5.
        mc_failure_probability: Failure probability of the simultaneous Monte
            Carlo confidence band. Default: 1e-6.

    Example::

        config = dp.DiscretizationConfig(discretization=0.001)
        pld = dp.gaussian_pld(1.1, config=config)
    """

    def __init__(
        self,
        discretization: float = 1e-4,
        log_mass_truncation_bound: float = -50.0,
        max_grid_size: int = 10_000_000,
        tail_mass_truncation: float = 1e-15,
        seed: int = 42,
        max_conv_grid: int = 32_768,
        mc_resolution: float = 1e-5,
        mc_failure_probability: float = 1e-6,
    ) -> None: ...
    @property
    def discretization(self) -> float:
        """Grid spacing for the PLD PMF."""

    @property
    def log_mass_truncation_bound(self) -> float:
        """Log-probability below which tails are truncated."""

    @property
    def max_grid_size(self) -> int:
        """Maximum bins before automatic coarsening."""

    @property
    def tail_mass_truncation(self) -> float:
        """Total tail mass budget for composition truncation."""

    @property
    def seed(self) -> int:
        """RNG seed for Monte Carlo."""

    @property
    def max_conv_grid(self) -> int:
        """Maximum convolution grid size for random-allocation PLD."""

    @property
    def mc_resolution(self) -> float:
        """Maximum unresolved Monte Carlo mass."""

    @property
    def mc_failure_probability(self) -> float:
        """Failure probability of the simultaneous Monte Carlo confidence band."""

    @property
    def resolved_num_mc_samples(self) -> int:
        """Sample count required by the configured confidence settings."""

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

# ---------------------------------------------------------------------------
# Mechanism functions
# ---------------------------------------------------------------------------

def gaussian_pld(
    noise_multiplier: float,
    config: DiscretizationConfig,
) -> Pld:
    """Compute the PLD for a Gaussian mechanism with sensitivity 1.

    Args:
        noise_multiplier: Ratio of noise std to sensitivity (σ/Δ).
        config: PLD discretization configuration.

    Returns:
        The privacy loss distribution.

    Example::

        pld = dp.gaussian_pld(1.1, config)
        pld.epsilon_at(1e-5)  # ~3.92
    """

def eps_delta_pld(
    epsilon: float,
    delta: float,
    config: DiscretizationConfig,
) -> Pld:
    """Compute the PLD for a fixed (ε, δ)-mechanism.

    Args:
        epsilon: Privacy loss, >= 0.
        delta: Failure probability, in [0, 1].
        config: PLD discretization configuration.

    Returns:
        The privacy loss distribution.
    """

def identity_pld(
    config: DiscretizationConfig,
) -> Pld:
    """Compute the PLD for the identity (zero-privacy-loss) mechanism.

    Args:
        config: PLD discretization configuration.

    Returns:
        The identity PLD (neutral element for composition).
    """

def non_private_pld(
    config: DiscretizationConfig,
) -> Pld:
    """Compute the PLD for a non-private mechanism (ε = ∞, δ = 1).

    All mass sits at +∞, representing a mechanism that provides no privacy
    guarantee.  This is the annihilator for composition.

    Args:
        config: PLD discretization configuration.

    Returns:
        A PLD with all mass at +∞.
    """

# ---------------------------------------------------------------------------
# Amplification functions
# ---------------------------------------------------------------------------

def poisson_pld(
    base: Pld,
    rate: float,
) -> Pld:
    """Apply plain Poisson subsampling to an existing PLD.

    Args:
        base: PLD of the base mechanism.
        rate: Poisson sampling probability, in (0, 1).

    Returns:
        The amplified privacy loss distribution.
    """

def truncated_poisson_gaussian_pld(
    noise_multiplier: float,
    rate: float,
    batch_size_max: int,
    dataset_size: int,
    config: DiscretizationConfig,
) -> Pld:
    """Compute the PLD for a truncated Poisson-subsampled Gaussian mechanism.

    Production DP-SGD sampling: caps batch at *batch_size_max* for
    predictable memory usage and tighter privacy bounds.

    Args:
        noise_multiplier: σ/Δ ratio.
        rate: Poisson sampling probability, in (0, 1].
        batch_size_max: Maximum batch size.
        dataset_size: Total dataset size.
        config: PLD discretization configuration.

    Returns:
        The amplified privacy loss distribution.
    """

def parallel_poisson_gaussian_pld(
    noise_multiplier: float,
    rate: float,
    microbatches: int,
    config: DiscretizationConfig,
) -> Pld:
    """Compute the PLD for a parallel Poisson-subsampled Gaussian mechanism.

    Models summing multiple independent Poisson samples before adding noise once.
    Use cases: gradient accumulation (m microbatches) or parallel workers (K workers).

    Args:
        noise_multiplier: σ/Δ ratio.
        rate: Poisson sampling probability, in (0, 1].
        microbatches: Number of independent samples, > 0.
        config: PLD discretization configuration.

    Returns:
        The amplified privacy loss distribution.
    """

def random_allocation_gaussian_pld(
    noise_multiplier: float,
    t: int,
    k: int,
    config: DiscretizationConfig,
) -> Pld:
    """PLD for random allocation applied to the Gaussian mechanism.

    In k-out-of-t random allocation each record is used in k steps chosen
    uniformly at random from t. Deterministic, composable, and reproducible
    across thread counts, unlike the Monte Carlo balls-in-bins accountant.

    Exact for ``k = 1``. For ``k > 1`` the result is a valid **upper bound**
    rather than the exact k-out-of-t PLD: the t steps are split into k blocks
    and the record is placed once per block. Joint convexity of the
    hockey-stick divergence makes that an upper bound, because a uniformly
    random partition into blocks induces exactly the uniform distribution over
    k-subsets.

    Args:
        noise_multiplier: σ/Δ ratio, must be > 0.
        t: Steps per allocation round (number of bins), > 0.
        k: Steps each record participates in, in [1, t]. Values above 1 return
            the block upper bound described above.
        config: PLD discretization configuration.

    Returns:
        The amplified privacy loss distribution.

    Raises:
        ValueError: If a parameter is out of range or the grid is too large.
    """

def balls_in_bins_gaussian_pld(
    noise_multiplier: float,
    num_bins: int,
    config: DiscretizationConfig,
) -> Pld:
    """Compute the per-epoch PLD for a Balls-in-Bins Gaussian mechanism.

    The dataset is randomly partitioned into ``num_bins`` equally-sized bins
    each epoch. Each bin is processed once with a Gaussian mechanism.
    Uses a conservative Poisson per-step approximation composed ``num_bins`` times.

    Args:
        noise_multiplier: σ/Δ ratio.
        num_bins: Number of bins (k ≥ 2).
        config: PLD discretization configuration.

    Returns:
        The per-epoch privacy loss distribution.
    """

def balls_in_bins_gaussian_pld_epochs(
    noise_multiplier: float,
    num_bins: int,
    num_epochs: int,
    config: DiscretizationConfig,
) -> Pld:
    """Compute the multi-epoch BnB Gaussian PLD.

    Exact per-bin Poisson composition across all ``num_epochs`` epochs.
    The returned PLD covers the full training run.

    Args:
        noise_multiplier: σ/Δ ratio.
        num_bins: Number of bins (k ≥ 2).
        num_epochs: Number of training epochs.
        config: PLD discretization configuration.

    Returns:
        The privacy loss distribution for the entire training run.
    """

def bandmf_b_min_sep_warm_mc_pld(
    strategy_coef: list[float],
    n_steps: int,
    p: float,
    sigma: float,
    config: DiscretizationConfig,
) -> Pld:
    """Monte Carlo PLD for BandMF + warm-start b-min-sep (arXiv:2602.09338)."""

def register_b_min_sep_transcript_corpus(
    strategy_coef: list[float],
    n_steps: int,
    p: float,
    num_samples: int,
    seed: int,
) -> int:
    """Allocate MC transcripts in Rust; return handle for reuse across σ."""

def drop_b_min_sep_transcript_corpus(handle: int) -> None:
    """Free a corpus allocated by ``register_b_min_sep_transcript_corpus``."""

def bandmf_b_min_sep_pld_from_transcript_handle(
    handle: int,
    strategy_coef: list[float],
    p: float,
    sigma: float,
    config: DiscretizationConfig,
) -> Pld:
    """Build PLD from a registered corpus at σ."""

def bnb_mc_pld(
    gram: list[float],
    num_bins: int,
    sigma: float,
    config: DiscretizationConfig,
) -> Pld:
    """Compute the BnB PLD via Monte Carlo sampling of the dominating pair.

    Uses Algorithm 2 of Choquette-Choo et al. (2024) for near-exact
    BnB privacy accounting with matrix mechanisms.

    Args:
        gram: Flattened row-major b×b Gram matrix.
        num_bins: Number of bins b.
        sigma: Noise multiplier.
        config: PLD discretization and Monte Carlo confidence configuration.

    Returns:
        The privacy loss distribution (asymmetric).
    """

def mf_gaussian_pld(
    noise_multiplier: float,
    sensitivity: float,
    config: DiscretizationConfig,
) -> Pld:
    """Compute the PLD for a matrix factorization Gaussian mechanism.

    Computes the privacy guarantee for the entire MF training run as a
    single Gaussian mechanism with effective noise multiplier σ/S.

    Args:
        noise_multiplier: Raw noise std σ (before MF). Must be positive.
        sensitivity: L2 sensitivity of the encoder matrix. Must be positive.
        config: PLD discretization configuration.

    Returns:
        The privacy loss distribution for the entire MF training run.

    Example::

        config = dp.DiscretizationConfig()
        pld = dp.mf_gaussian_pld(1.0, 2.5, config)
        pld.epsilon_at(1e-5)
    """

def max_participation_for_linear_fn(
    x: list[float],
    min_sep: int = 1,
    max_participations: int | None = None,
) -> float:
    """Solve max_u <x, u> where u respects min-sep participation.

    Uses dynamic programming (Algorithm 3, VecSens) from
    Choquette-Choo et al. (2023).

    Args:
        x: Vector of values to optimize over.
        min_sep: Minimum separation between selections (>= 1).
        max_participations: Optional upper bound on selections.

    Returns:
        The optimal inner product.
    """

def minsep_true_max_participations(
    n: int,
    min_sep: int,
    max_participations: int | None = None,
) -> int:
    """Maximum participations under a min-sep constraint.

    Args:
        n: Number of rounds.
        min_sep: Minimum separation between participations.
        max_participations: Optional upper bound.

    Returns:
        Effective maximum participations.
    """

def single_participation_sensitivity(
    column_norms: list[float],
) -> float:
    """L2 sensitivity under single participation.

    Args:
        column_norms: L2 norms of encoder matrix columns.

    Returns:
        Maximum column norm (the sensitivity).
    """

def banded_sensitivity(
    gram_diag: list[float],
    min_sep: int = 1,
    max_participations: int | None = None,
) -> float:
    """Exact L2 sensitivity for banded Gram matrices under min-sep participation.

    Args:
        gram_diag: Diagonal of Gram matrix X = C^T C.
        min_sep: Minimum separation between participations.
        max_participations: Optional upper bound.

    Returns:
        The exact L2 sensitivity.
    """

def general_sensitivity_upper_bound(
    gram_matrix: list[float],
    n: int,
    min_sep: int = 1,
    max_participations: int | None = None,
) -> float:
    """Upper bound on L2 sensitivity for general Gram matrices.

    Args:
        gram_matrix: Flattened row-major Gram matrix X = C^T C.
        n: Matrix dimension.
        min_sep: Minimum separation between participations.
        max_participations: Optional upper bound.

    Returns:
        An upper bound on the L2 sensitivity.
    """

def fixed_epoch_sensitivity(
    gram_matrix: list[float],
    n: int,
    epochs: int,
) -> float:
    """L2 sensitivity under fixed-epoch participation.

    Args:
        gram_matrix: Flattened row-major Gram matrix X = C^T C.
        n: Matrix dimension (total rounds).
        epochs: Number of epochs (must divide n).

    Returns:
        The L2 sensitivity under fixed-epoch participation.
    """

def blt_sensitivity_squared(
    buf_decay: list[float],
    output_scale: list[float],
    n: float,
) -> float:
    """Sensitivity squared for a BLT strategy matrix.

    Implements Lemma 5.3 of the BLT paper.

    Args:
        buf_decay: Decay factors for each buffer, each in (0, 1).
        output_scale: Scale factors for each buffer.
        n: Number of iterations (use float('inf') for asymptotic limit).

    Returns:
        The sensitivity squared.
    """

def toeplitz_minsep_sensitivity_squared(
    strategy_coef: list[float],
    n: int,
    min_sep: int = 1,
    max_participations: int | None = None,
) -> float:
    """Sensitivity squared for a Toeplitz matrix under min-sep participation.

    Implements BSR Theorem 2 closed-form for non-negative, non-increasing
    Toeplitz coefficients.

    Args:
        strategy_coef: Toeplitz coefficients (non-negative, non-increasing).
        n: Matrix dimension (total rounds).
        min_sep: Minimum separation between participations.
        max_participations: Optional upper bound.

    Returns:
        The sensitivity squared.
    """

def lambda_cgd_sensitivity_squared(
    lambda_: float,
    n_steps: int,
    min_sep: int = 1,
    max_participations: int | None = None,
    momentum: float = 0.0,
) -> float:
    """Squared L2 sensitivity of the DP-λCGD strategy matrix.

    Uses the closed-form expression completed in Appendix C, equation (16) of
    arXiv:2601.22334v1.

    Note: the standard Python API always passes momentum=0.
    Sensitivity is workload-independent (BandMF paper, Thm 1).

    Args:
        lambda_: Correlation coefficient in [0, 1). λ=0 is DP-SGD.
        n_steps: Total number of training steps.
        min_sep: Minimum separation between participations (>= 1).
        max_participations: Optional upper bound on participations.
        momentum: Unused in standard API (default 0.0).

    Returns:
        The squared L2 sensitivity.
    """

def lambda_cgd_normalized_sensitivity_squared(
    lambda_: float,
    n_steps: int,
    min_sep: int = 1,
    max_participations: int | None = None,
    momentum: float = 0.0,
) -> float:
    """Squared L2 sensitivity of the column-normalized DP-λCGD.

    Column normalization: C̃_λ = C_λ · D⁻¹ where D = diag(‖C_λ[:,j]‖).
    For single participation (k=1), always returns 1.0.
    The normalized result is covered by Lemma 8, equation (12) of the current
    arXiv:2601.22334v2, rather than its Toeplitz-only Theorem 1.

    Args:
        lambda_: Correlation coefficient in [0, 1).
        n_steps: Total number of training steps.
        min_sep: Minimum separation between participations (>= 1).
        max_participations: Optional upper bound on participations.
        momentum: Unused in standard API (default 0.0).

    Returns:
        The squared L2 sensitivity of the column-normalized matrix.
    """

def lambda_cgd_max_column_norm(
    lambda_: float,
    n_steps: int,
) -> float:
    """Max column L2 norm of the DP-λCGD strategy matrix.

    The first column has the largest norm: sqrt((1 - λ^{2n}) / (1 - λ²)).

    Args:
        lambda_: Correlation coefficient in [0, 1).
        n_steps: Total number of steps.

    Returns:
        The max column L2 norm.
    """

def lambda_cgd_gram_matrix(
    lambda_: float,
    n_steps: int,
    min_sep: int = 1,
    max_participations: int | None = None,
    normalized: bool = True,
    momentum: float = 0.0,
) -> list[float]:
    """Compute the BnB Gram matrix for DP-λCGD.

    G_{ij} = ⟨m_i, m_j⟩ where m_i = Σ_epoch C[:,b·epoch+i].

    Note: the standard Python API always passes momentum=0.
    Gram matrix is workload-independent.

    Args:
        lambda_: Correlation coefficient in [0, 1).
        n_steps: Total steps (= bins_per_epoch × num_epochs).
        min_sep: Bins per epoch (= b).
        max_participations: Number of epochs. None infers.
        normalized: Whether to use column-normalized matrix.
        momentum: Unused in standard API (default 0.0).

    Returns:
        Flattened row-major b×b Gram matrix.
    """

def lambda_cgd_gram_matrix_lr(
    lambda_: float,
    momentum: float,
    n_steps: int,
    min_sep: int,
    max_participations: int | None,
    normalized: bool,
    lr_weights: list[float],
) -> list[float]:
    """Compatibility tombstone for LR-weighted DP-lambda-CGD accounting.

    Args:
        lambda_: Correlation coefficient in [0, 1).
        momentum: Optimizer momentum β in [0, 1).
        n_steps: Total steps.
        min_sep: Bins per epoch.
        max_participations: Number of epochs.
        normalized: Whether to use column-normalized matrix.
        lr_weights: Per-step LR weights, length = n_steps.

    Raises:
        ValueError: Always. Omit the schedule and recalibrate, or use BandMF/BLT.
    """

# ---------------------------------------------------------------------------
# BISR (Banded Inverse Square Root)
# ---------------------------------------------------------------------------

def bisr_sensitivity_squared(
    coefficients: list[float],
    n_steps: int,
    min_sep: int = 1,
    max_participations: int | None = None,
    momentum: float = 0.0,
) -> float:
    """Squared L2 sensitivity upper bound for BISR under min-sep participation."""

def bisr_normalized_sensitivity_squared(
    coefficients: list[float],
    n_steps: int,
    min_sep: int = 1,
    max_participations: int | None = None,
    momentum: float = 0.0,
) -> float:
    """Conservative squared L2 sensitivity bound for column-normalized BISR."""

def bisr_gram_matrix(
    coefficients: list[float],
    n_steps: int,
    min_sep: int = 1,
    max_participations: int | None = None,
    normalized: bool = True,
    momentum: float = 0.0,
) -> list[float]:
    """BnB Gram matrix for BISR with optional momentum."""

def bisr_gram_matrix_lr(
    coefficients: list[float],
    momentum: float,
    n_steps: int,
    min_sep: int,
    max_participations: int | None,
    normalized: bool,
    lr_weights: list[float],
) -> list[float]:
    """Compatibility tombstone for LR-weighted BISR accounting.

    Raises:
        ValueError: Always. Omit the schedule and recalibrate, or use BandMF/BLT.
    """

# ---------------------------------------------------------------------------
# Toeplitz Gram matrix (for BnB with BandMF/BLT strategy coefs)
# ---------------------------------------------------------------------------

def toeplitz_gram_matrix(
    strategy_coef: list[float],
    n_steps: int,
    min_sep: int = 1,
    max_participations: int | None = None,
    normalized: bool = True,
) -> list[float]:
    """BnB Gram matrix for banded Toeplitz strategy with known forward coefficients.

    For BandMF/BLT mechanisms where the optimized strategy coefficients
    are known directly.

    Args:
        strategy_coef: Toeplitz strategy coefficients [c_0, ..., c_{p-1}].
        n_steps: Total steps.
        min_sep: Bins per epoch.
        max_participations: Number of epochs.
        normalized: Whether to column-normalize.

    Returns:
        Flattened row-major b x b Gram matrix.
    """

def bisr_strategy_coefficients(
    coefficients: list[float],
    n: int,
) -> list[float]:
    """Recover strategy matrix column from banded C^{-1} coefficients.

    Computes the first n entries of column 0 of the strategy matrix C
    defined by the banded inverse C^{-1} with the given coefficients.

    Args:
        coefficients: Inverse coefficients [c_tilde_0, ..., c_tilde_{p-1}].
        n: Number of entries to compute.

    Returns:
        First n entries of column 0.

    Raises:
        ValueError: If coefficients are invalid or n is zero.
    """

# ---------------------------------------------------------------------------
# AdaClip utility
# ---------------------------------------------------------------------------

def adaclip_sensitivity(
    noise_multiplier: float,
    quantile_noise_std: float,
    num_groups: int = 1,
) -> float:
    """Combined sensitivity for adaptive clipping (Andrew et al. 2021).

    Returns z̃ = sqrt(1/z² + 1/(4·σ_b²)).

    Args:
        noise_multiplier: Gradient noise multiplier z.
        quantile_noise_std: Std of quantile estimator noise σ_b.
        num_groups: Number of independent quantile queries (default 1).

    Returns:
        Combined sensitivity z̃.
    """
