"""AUTO-S automatic clipping for differential privacy.

Implements the automatic clipping scheme of Bu et al., "Automatic Clipping:
Differentially Private Deep Learning Made Easier and Stronger" (NeurIPS
2023), which replaces the standard clip threshold with a per-example scaling

.. math::

    \\tilde g_i = R \\cdot g_i / (\\lVert g_i \\rVert + \\gamma)

where ``R`` is a fixed sensitivity bound and ``\\gamma`` is a small
denominator stabilizer. The output has L2 norm at most ``R`` by
construction, so the clipping threshold is no longer a tunable
hyperparameter — it is absorbed into the learning rate.

AUTO-S is algorithm-agnostic: the per-record bound ``R`` is fixed at
construction and data-independent (``\\sup_g \\lVert\\tilde g\\rVert \\le R``
holds uniformly), so each ``ClippedPytree`` carries a ``max_norm`` that
does not depend on the batch — the correct per-step sensitivity for
``gaussian_noise`` and other per-step Gaussian mechanisms.  DP-FTRL's
``mf_gaussian_noise`` additionally requires that ``max_norm`` stay *unchanged
across training steps* (the dispatcher latches the first call for the
matrix-factorization privacy proof); AUTO-S satisfies that latch because
``R`` and ``normalize_by`` are fixed, unlike adaptive clipping.  Privacy
accounting is the standard Gaussian PLD parameterized by
``noise_multiplier``; AUTO-S contributes no additional privacy cost
because the scaling is fully per-example (depends only on the sample's
own gradient).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from opaque.api.engine.clipping._clipped_fun import ClippedFunAux, clipped_fun
from opaque.api.engine.clipping._clipped_grad import ClippedGradAux, clipped_grad
from opaque.api.engine.clipping._pytree import auto_scale_pytree
from opaque.api.engine.types import ClipState, PerGroup
from opaque.exceptions import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Callable

_DEFAULT_GAMMA = 0.01


@dataclass(frozen=True)
class AutoClippedFunAux(ClippedFunAux):
    """Diagnostic outputs from ``auto_clipped_fun``.

    Inherits all fields from :class:`ClippedFunAux` with AUTO-S
    semantics:

    - ``norms``: per-example L2 norms before scaling (i.e. ``||v||``).
    - ``clipped_norms``: per-example L2 norms after AUTO-S scaling
      (``R * ||v|| / (||v|| + gamma)``), clipped by ``R``.
    - ``clipping_rate``: fraction of examples with ``||v|| > R``
      (approximately the fraction where the scale factor was < 1).
    - ``values``, ``value_aux``, ``batch_size``, ``group_norms``:
      unchanged from :class:`ClippedFunAux`.
    """


@dataclass(frozen=True)
class AutoClippedGradAux(ClippedGradAux):
    """Diagnostic outputs from ``auto_clipped_grad``.

    Inherits all fields from :class:`ClippedGradAux` with AUTO-S
    semantics:

    - ``grad_norms``: per-example gradient L2 norms before scaling.
    - ``clipped_grad_norms``: per-example gradient L2 norms after
      AUTO-S scaling (``R * ||g|| / (||g|| + gamma)``), clipped by ``R``.
    - ``clipping_rate``: fraction of examples with ``||g|| > R``
      (approximately the fraction where the scale factor was < 1).
    - ``loss_values``, ``loss_aux``, ``batch_size``, ``group_norms``:
      unchanged from :class:`ClippedGradAux`.
    """


@dataclass(frozen=True)
class AutoClipState(ClipState):
    """Marker state for AUTO-S automatic clipping."""


def _make_auto_scale_fn(R: float | PerGroup, gamma: float) -> Callable:
    """Build a vmap-compatible per-example AUTO-S scaling closure."""

    def scale(value):
        return auto_scale_pytree(value, R=R, gamma=gamma)

    return scale


def _validate_auto_params(R: float | PerGroup, gamma: float) -> None:
    if isinstance(R, PerGroup):
        for gname, val in R.values.items():
            if val <= 0:
                raise ConfigurationError(
                    *(
                        f"R must be positive for all groups, got {val} for group '{gname}'",
                    )
                )
    elif R <= 0:
        raise ConfigurationError(*(f"R must be positive, got {R}",))
    if gamma <= 0:
        raise ConfigurationError(*(f"gamma must be positive, got {gamma}",))


def auto_clipped_fun(
    fun: Callable,
    has_aux: bool = False,
    *,
    batch_argnums: int | tuple[int, ...] = 0,
    R: float | PerGroup = 1.0,
    gamma: float = _DEFAULT_GAMMA,
    normalize_by: float = 1.0,
    return_aux: bool = False,
    return_stats: bool = False,
    microbatch_size: int | None = None,
    dtype: Any = None,
    _chunk_compiler: Callable | None = None,
) -> tuple[Callable, AutoClipState]:
    r"""Transform a function so each per-example output is scaled to clipped norm via AUTO-S.

    Mirrors :func:`clipped_fun` but replaces threshold-based clipping with
    AUTO-S automatic scaling ``R \cdot v / (\|v\| + \gamma)``.  The output of
    the returned function is the sum of the scaled per-example outputs.

    Args:
        fun: The per-example function to be transformed.
        has_aux: If True, ``fun`` returns ``(value, aux)``; only ``value``
            is scaled and aggregated.
        batch_argnums: Which arguments have a batch dimension.
        R: Sensitivity max_norm for the scaled output.  When ``PerGroup``,
            each group is scaled independently.
        gamma: Denominator stabilizer :math:`\gamma` (default 0.01).
        normalize_by: Divisor applied to the scaled sum.
        return_aux: If True, the returned callable returns an
            :class:`AutoClippedFunAux` alongside the summed value.
        return_stats: If True, return :class:`ClippingStats` alongside the
            summed value. Cannot be combined with ``return_aux``.
        microbatch_size: Process the batch in chunks of this size.
        dtype: Optional accumulation dtype for the sum.

    Returns:
        ``(auto_fn, state)``.  ``auto_fn`` has the signature
        ``(*args, state, **kwargs) -> (value, state)`` or
        ``(*args, state, **kwargs) -> ((value, aux), state)`` when
        ``return_aux=True``.

    Raises:
        ConfigurationError: If the batch is empty.

    Formal guarantee:
        Under add/remove or zero-out DP, the L2 sensitivity of the first
        output with respect to the batch arguments is the returned
        ``ClippedPytree.max_norm`` metadata.  The bound is constant and
        data-independent, so per-step Gaussian calibration is correct.
        ``mf_gaussian_noise`` additionally requires that ``max_norm`` not drift across
        steps; AUTO-S satisfies that latch because ``R`` and ``normalize_by``
        are fixed.
    """
    _validate_auto_params(R, gamma)

    scale_fn = _make_auto_scale_fn(R, gamma)
    inner_fn, _ = clipped_fun(
        fun,
        has_aux=has_aux,
        batch_argnums=batch_argnums,
        clipping_norm=R,
        normalize_by=normalize_by,
        return_aux=return_aux,
        return_stats=return_stats,
        microbatch_size=microbatch_size,
        dtype=dtype,
        _scale_fn=scale_fn,
        _chunk_compiler=_chunk_compiler,
    )

    state = AutoClipState()

    if not return_aux:

        def auto_fn(*args, state, **kwargs):
            result, _ = inner_fn(*args, state=None, **kwargs)
            return result, state

        return auto_fn, state

    def auto_fn(*args, state, **kwargs):
        (result, fun_aux), _ = inner_fn(*args, state=None, **kwargs)
        aux = AutoClippedFunAux(
            values=fun_aux.values,
            norms=fun_aux.norms,
            clipped_norms=fun_aux.clipped_norms,
            value_aux=fun_aux.value_aux,
            clipping_rate=fun_aux.clipping_rate,
            batch_size=fun_aux.batch_size,
            group_norms=fun_aux.group_norms,
        )
        return (result, aux), state

    return auto_fn, state


def auto_clipped_grad(
    loss_fn: Callable,
    argnums: int | tuple[int, ...] = 0,
    has_aux: bool = False,
    *,
    R: float | PerGroup = 1.0,
    gamma: float = _DEFAULT_GAMMA,
    normalize_by: float = 1.0,
    batch_argnums: int | tuple[int, ...] = 1,
    return_aux: bool = False,
    return_stats: bool = False,
    pre_clipping_transform: Callable = lambda x: x,
    microbatch_size: int | None = None,
    dtype: Any = None,
    second_moment: bool = False,
    _chunk_compiler: Callable | None = None,
) -> tuple[Callable, AutoClipState]:
    r"""Create a function that computes the sum of AUTO-S scaled per-example gradients.

    AUTO-S (Bu et al., 2023) replaces the standard clipping threshold with
    a per-example scaling

    .. math::

        \tilde g_i = R \cdot g_i / (\lVert g_i \rVert + \gamma)

    so every per-example gradient has L2 norm at most ``R`` by
    construction.  There is no threshold to tune; the effective step size
    is absorbed into the optimizer learning rate.

    Args:
        loss_fn: Scalar loss function.  If ``has_aux``, returns
            ``(scalar, loss_aux)``.
        argnums: Which argument(s) to differentiate w.r.t.
        has_aux: If True, ``loss_fn`` returns ``(scalar, loss_aux)``.
        R: Sensitivity max_norm (default 1.0).  When ``PerGroup``, each
            parameter group is scaled independently to its own ``R_k``.
        gamma: Denominator stabilizer :math:`\gamma` (default 0.01,
            strictly positive).
        normalize_by: Divisor applied to the summed gradients (set to
            expected batch size for averaged gradients).
        batch_argnums: Which arguments have a batch dimension.
        return_aux: If True, returns per-example diagnostics as
            :class:`AutoClippedGradAux`.
        pre_clipping_transform: An optional function to apply to the
            per-example gradients before AUTO-S scaling. The function
            should consume the gradient pytree for a single example
            and return a new pytree (possibly with different structure).
            Can be used to e.g., scale the leaves of the pytree to
            accommodate preconditioner clipping. Does not affect the
            sensitivity guarantee. Default is identity function.
        microbatch_size: Process the batch in chunks of this size.
        dtype: Optional accumulation dtype for the summed gradient.
        second_moment: If True, also accumulate the per-example sum of
            element-wise squared scaled gradients and return a
            :class:`~opaque.types.SecondMomentClippingOutput` carrying
            both streams.  Privacy accounting is unchanged
            (``gaussian(noise_multiplier)`` for DP-SGD or any standard MF
            mechanism for DP-FTRL); the sensitivity-proportional joint
            Mahalanobis allocation gives the paired release the same PLD
            as a single first-moment release.
        return_stats: If True, return :class:`ClippingStats` beside the
            gradient. Cannot be combined with ``return_aux``.

    Returns:
        ``(grad_fn, state)``.  ``grad_fn`` has the signature
        ``(*args, state, **kwargs) -> (grad, state)``, or
        ``(*args, state, **kwargs) -> ((grad, aux), state)`` when
        ``return_aux=True``. When ``return_stats=True``, it returns
        ``((grad, stats), state)``.

    Formal guarantee:
        Under add/remove or zero-out DP, the L2 sensitivity of the
        summed gradients is the returned ``ClippedPytree.max_norm`` metadata —
        a constant independent of the input data.  Per-step Gaussian noise
        (``gaussian_noise``, …) reads that value
        each step; PLD composition does not require ``max_norm`` to be
        identical across steps when the accountant models step-varying
        sensitivity (for example adaptive clipping with ``adaclip``).
        ``mf_gaussian_noise``'s matrix-factorization correlated noise *does* require
        ``max_norm`` to stay fixed for the whole run — the dispatcher latches
        the first-call bound — and AUTO-S satisfies that because ``R`` and
        ``normalize_by`` do not drift.  AUTO-S scaling is per-example and adds
        no privacy cost beyond ``gaussian(noise_multiplier)`` for the
        gradient release.

    Example:
        >>> import torch
        >>> from opaque.api.engine.clipping import auto_clipped_grad
        >>> def loss_fn(params, x, y):
        ...     return ((x @ params - y) ** 2).mean()
        >>> grad_fn, state = auto_clipped_grad(
        ...     loss_fn, argnums=0, batch_argnums=(1, 2),
        ...     R=1.0, normalize_by=32,
        ... )
        >>> params = torch.randn(10)
        >>> batch_x = torch.randn(32, 10)
        >>> batch_y = torch.randn(32)
        >>> grads, state = grad_fn(params, batch_x, batch_y, state=state)

    The returned ``ClippedPytree`` carries ``max_norm = R / normalize_by``
    (constant across steps), so it composes with ``mf_gaussian_noise`` (DP-FTRL)
    and ``gaussian_noise`` (DP-SGD) the same way ``clipped_grad`` does at
    the same bound.

    References:
        Bu, Wang, Zha, Karypis.  "Automatic Clipping: Differentially
        Private Deep Learning Made Easier and Stronger." NeurIPS 2023.
        https://arxiv.org/abs/2206.07136
    """
    _validate_auto_params(R, gamma)

    scale_fn = _make_auto_scale_fn(R, gamma)
    inner_fn, _ = clipped_grad(
        loss_fn,
        argnums=argnums,
        has_aux=has_aux,
        clipping_norm=R,
        normalize_by=normalize_by,
        batch_argnums=batch_argnums,
        return_aux=return_aux,
        return_stats=return_stats,
        pre_clipping_transform=pre_clipping_transform,
        microbatch_size=microbatch_size,
        dtype=dtype,
        second_moment=second_moment,
        _scale_fn=scale_fn,
        _chunk_compiler=_chunk_compiler,
    )

    state = AutoClipState()

    if not return_aux:

        def grad_fn(*args, state, **kwargs):
            result, _ = inner_fn(*args, state=None, **kwargs)
            return result, state

        return grad_fn, state

    def grad_fn(*args, state, **kwargs):
        inner_result, _ = inner_fn(*args, state=None, **kwargs)
        grads, grad_aux = inner_result
        auto_aux = AutoClippedGradAux(
            loss_values=grad_aux.loss_values,
            grad_norms=grad_aux.grad_norms,
            clipped_grad_norms=grad_aux.clipped_grad_norms,
            loss_aux=grad_aux.loss_aux,
            clipping_rate=grad_aux.clipping_rate,
            batch_size=grad_aux.batch_size,
            group_norms=grad_aux.group_norms,
        )
        return (grads, auto_aux), state

    return grad_fn, state


__all__ = [
    "AutoClipState",
    "AutoClippedFunAux",
    "AutoClippedGradAux",
    "auto_clipped_fun",
    "auto_clipped_grad",
]
