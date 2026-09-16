"""Adaptive gradient clipping with explicit state-passing."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch.autograd.profiler import record_function

from opaque.api.engine.clipping._clipped_fun import ClippingStats, _RuntimeClipState
from opaque.api.engine.clipping._clipped_grad import (
    ClippedGradAux,
    _validate_static_args,
    clipped_grad,
)
from opaque.api.engine.clipping._helpers import (
    batch_size_from_args,
    empty_clipped_grads_like,
    normalize_to_tuple,
)
from opaque.api.engine.pytree import tree_map
from opaque.exceptions import ConfigurationError
from opaque.random import fold_in, generator_from_key
from opaque.random.types import RngKey
from opaque.types import ClipState, PerGroup, SecondMomentClippingOutput, clipped

_DEFAULT_FRACTION_NOISE_STD = 0.05

# Without a root the obvious derivation — ``fold_in(key, step)`` — is what
# every mechanism writes.  See ``docs/reference/rng.md``.
ADAPTIVE_CLIPPING_STREAM_FOLD = "opaque.dpsgd.adaptive_clipping"


@dataclass(frozen=True)
class AdaptiveClippedGradAux(ClippedGradAux):
    """Diagnostic outputs from adaptive_clipped_grad.

    All fields are diagnostic — they reflect pre-noise, pre-aggregation
    values and must not be fed back into private computation.  Use the
    returned ``ClippedPytree.max_norm`` metadata for noise calibration.

    Inherits all fields from :class:`ClippedGradAux`.
    """


@dataclass(frozen=True)
class AdaptiveClipState(ClipState):
    """Immutable state for adaptive gradient clipping.

    This state is passed explicitly to the clipping function and returned
    as part of the output, enabling pure functional composition.

    The fields are internal implementation details. Adaptive clipping must
    carry the current threshold, next-step threshold, step counter, RNG/config,
    and local clipping counts so distributed synchronization can aggregate
    counts and recompute the global next threshold.
    """

    _current_clipping_norm: float | PerGroup
    _next_clipping_norm: float | PerGroup
    _step: int

    # -- internal (config carried for distributed sync) --
    _rng_key: RngKey
    _fraction_noise_std: float
    _learning_rate: float
    _target_quantile: float
    _clipping_norm_min: float
    _clipping_norm_max: float

    # -- internal (per-step counts for distributed aggregation) --
    # Scalar for global clipping, dict[str, float] for per-group clipping.
    _num_clipped: float | dict[str, float]
    _batch_size: float

    def __post_init__(self):
        """Validate state values."""
        if isinstance(self._next_clipping_norm, PerGroup):
            for gname, val in self._next_clipping_norm.values.items():
                if val <= 0:
                    raise ConfigurationError(
                        *(
                            f"next_clipping_norm must be positive for all groups, "
                            f"got {val} for group '{gname}'",
                        )
                    )
        elif self._next_clipping_norm <= 0:
            raise ConfigurationError(
                *(
                    f"next_clipping_norm must be positive, got {self._next_clipping_norm}",
                )
            )
        if self._fraction_noise_std <= 0:
            raise ConfigurationError(
                *(f"fraction_noise_std must be > 0, got {self._fraction_noise_std}",)
            )


def _compute_clipping_stats(
    grad_norms: torch.Tensor, clipping_norm: float
) -> tuple[float, float, float]:
    """Compute local clipping statistics from per-example gradient norms."""
    total = float(grad_norms.numel())
    if total == 0:
        return 0.0, 0.0, 0.0
    num_clipped = float((grad_norms > clipping_norm).sum().item())
    clipping_rate = num_clipped / total
    return num_clipped, total, clipping_rate


def _sample_noisy_clipping_rate(
    clipping_rate: float,
    *,
    key: RngKey,
    step: int,
    fraction_noise_std: float,
) -> float:
    """Add DP Gaussian noise to clipping rate using step-folded RNG key."""
    step_key = fold_in(key, ADAPTIVE_CLIPPING_STREAM_FOLD, step)
    generator = generator_from_key(step_key)
    noise = torch.randn(1, generator=generator).item() * fraction_noise_std
    return clipping_rate + noise


def _adaptive_clipping_norm_update(
    *,
    base_clipping_norm: float,
    noisy_clipping_rate: float,
    target_quantile: float,
    learning_rate: float,
    clipping_norm_min: float,
    clipping_norm_max: float,
) -> float:
    """Compute geometric adaptive clipping update: C * exp(η * (ρ̃ - γ))."""
    update_factor = torch.exp(
        torch.tensor(learning_rate * (noisy_clipping_rate - target_quantile))
    ).item()
    new_clipping_norm = base_clipping_norm * update_factor
    return float(max(clipping_norm_min, min(clipping_norm_max, new_clipping_norm)))


def _empty_batch_grads(
    zeros: Any,
    bound: float,
    squared_bound: float,
    second_moment: bool,
):
    """Wrap zero gradients in the same output structure as a non-empty step."""
    grads = clipped(zeros, max_norm=bound)
    if not second_moment:
        return grads
    return SecondMomentClippingOutput(
        grads=grads,
        squared_grads=clipped(
            tree_map(
                lambda x: torch.zeros_like(x) if isinstance(x, torch.Tensor) else x,
                zeros,
            ),
            max_norm=squared_bound,
        ),
    )


def adaptive_clipped_grad(
    loss_fn: Callable,
    argnums: int | tuple[int, ...] = 0,
    has_aux: bool = False,
    *,
    initial_clipping_norm: float | PerGroup = 0.1,
    target_quantile: float = 0.5,
    learning_rate: float = 0.2,
    clipping_norm_min: float = 0.01,
    clipping_norm_max: float = 100.0,
    fraction_noise_std: float = _DEFAULT_FRACTION_NOISE_STD,
    key: RngKey,
    return_aux: bool = False,
    return_stats: bool = False,
    pre_clipping_transform: Callable = lambda x: x,
    **clipped_grad_kwargs: Any,
) -> tuple[Callable, AdaptiveClipState]:
    """Create function for adaptive gradient clipping with explicit state-passing.

    Returns ``(clipped_grad_fn, initial_state)``.  The returned function
    takes ``state`` as an explicit parameter and returns
    ``(grad, new_state)`` or ``((grad, aux), new_state)``.

    Args:
        loss_fn: Loss function (scalar output). If ``has_aux``, returns
            ``(scalar, loss_aux)``.
        argnums: Which argument(s) to differentiate w.r.t.
        has_aux: If True, ``loss_fn`` returns ``(value, loss_aux)``.
        initial_clipping_norm: Initial clipping threshold C_0.  When
            ``PerGroup``, each group adapts independently.
        target_quantile: Target fraction of clipped gradients.
        learning_rate: Step size for geometric adaptation.
        clipping_norm_min: Minimum allowed clipping threshold.
        clipping_norm_max: Maximum allowed clipping threshold.
        fraction_noise_std: Std of Gaussian noise added to the clipping
            fraction (default 0.05).
        key: RNG key for quantile noise generation.
        return_aux: If True, return per-example aux with loss values,
            gradient norms, and clipping rate.
        return_stats: If True, return aggregate clipping statistics alongside
            the gradient. Cannot be combined with ``return_aux``.
        pre_clipping_transform: An optional function to apply to the
            per-example gradients before clipping. The function should
            consume the gradient pytree for a single example and return
            a new pytree (possibly with different structure). Can be
            used to e.g., scale the leaves of the pytree to accommodate
            preconditioner clipping. Does not affect the sensitivity guarantee.
            Default is identity function.
        **clipped_grad_kwargs: Passed to ``clipped_grad()``
            (``batch_argnums``, ``normalize_by``, etc).

    Note:
        **Empty-batch parity (``second_moment``):** the empty-batch
        short-circuit mirrors :func:`~opaque.dpsgd.clipping.clipped_grad` and
        :func:`~opaque.dpsgd.clipping.auto_clipped_grad` — when ``second_moment=True``
        it returns a :class:`~opaque.types.SecondMomentClippingOutput` of zeros
        (squared-stream sensitivity ``C²/normalize_by``), so paired-stream
        noise and optimizer dispatch are stable across empty and non-empty
        steps under Poisson sampling.

    Returns:
        A tuple of (clipped_grad_fn, initial_state) where:
            - clipped_grad_fn: Function with signature
                (*args, state, **kwargs) -> (grad, new_state) or
                (*args, state, **kwargs) -> ((grad, aux), new_state)
                or ``((grad, stats), new_state)`` when ``return_stats=True``.
            - initial_state: Initial AdaptiveClipState

    Example (single device):
        >>> import torch
        >>> from opaque.dpsgd.clipping import adaptive_clipped_grad
        >>> from opaque.dpsgd.noise import gaussian_noise
        >>> from opaque.random import key
        >>> import torchopt
        >>>
        >>> def loss_fn(params, x, y):
        ...     pred = x @ params
        ...     return ((pred - y) ** 2).mean()
        >>>
        >>> # Create an adaptive clipping function with explicit local state.
        >>> # Distributed callers must synchronize state explicitly.
        >>> grad_fn, clip_state = adaptive_clipped_grad(
        ...     loss_fn,
        ...     initial_clipping_norm=0.1,
        ...     target_quantile=0.5,
        ...     key=key(0),
        ...     batch_argnums=(1, 2),
        ... )
        >>>
        >>> # Training loop with explicit state-passing
        >>> params = torch.randn(10, requires_grad=False)
        >>> optimizer = torchopt.adamw(lr=1e-3)
        >>> opt_state = optimizer.init(params)
        >>>
        >>> noise_fn, noise_state = gaussian_noise(noise_multiplier=1.1, key=key(1))
        >>> # ``dataloader`` yields (x, y) batches (e.g. a torch DataLoader).
        >>> dataloader = [(torch.randn(4, 10), torch.randn(4)) for _ in range(3)]
        >>> for batch_x, batch_y in dataloader:
        ...     # Compute clipped gradients - state passed explicitly
        ...     grad, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
        ...
        ...     # Add DP noise scaled to current threshold
        ...     noisy_grad, noise_state = noise_fn(grad, noise_state)
        ...
        ...     # Optimizer step
        ...     updates, opt_state = optimizer.update(noisy_grad, opt_state, params=params)
        ...     params = torchopt.apply_updates(params, updates)
        ...
        ...     # Monitor adaptation
        ...     # The current DP max_norm is attached to the clipped output.
        ...     current_bound = grad.max_norm

    Example with distributed training (DDP with Poisson sampling):
        >>> import torch.distributed as dist
        >>> from torch.utils.data import DataLoader
        >>> from opaque.dpsgd.clipping import adaptive_clipped_grad
        >>> from opaque.dpsgd.noise import gaussian_noise
        >>> from opaque.dpsgd.sampling import PoissonSampler
        >>> from opaque.distributed import local_shard, sum_gradients, sync
        >>> from opaque.random import fold_in, key
        >>>
        >>> # Initialize distributed first (typically via torchrun).
        >>> dist.init_process_group(backend='nccl')
        >>> rank, world_size = dist.get_rank(), dist.get_world_size()
        >>>
        >>> # Create adaptive clipping (local-only function).
        >>> # Distributed callers must synchronize the state explicitly.
        >>> grad_fn, clip_state = adaptive_clipped_grad(
        ...     loss_fn,
        ...     key=key(0),
        ...     batch_argnums=(1, 2),
        ... )
        >>> noise_fn, noise_state = gaussian_noise(noise_multiplier=1.1, key=key(2))
        >>>
        >>> # Poisson sampling: shard per rank, derive a per-rank sampling key.
        >>> shard = local_shard(dataset, rank=rank, world_size=world_size)
        >>> sampler = PoissonSampler(
        ...     shard, sample_rate=0.01, key=fold_in(key(42), rank)
        ... )
        >>> loader = DataLoader(shard, batch_sampler=sampler)
        >>>
        >>> for batch_x, batch_y in loader:
        ...     # Each device: compute clipped gradients and local adaptive state
        ...     grad, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
        ...     # Explicit sync step for adaptive clipping state.
        ...     clip_state = sync(clip_state)
        ...
        ...     # Sum clipped gradients across devices
        ...     grad = sum_gradients(grad)
        ...
        ...     # Add noise and update
        ...     noisy_grad, noise_state = noise_fn(grad, noise_state)
        ...     # ... optimizer step

    References:
        Andrew et al., "Differentially Private Learning with Adaptive
        Clipping", NeurIPS 2021. https://arxiv.org/abs/1905.03871
    """
    if return_aux and return_stats:
        raise ConfigurationError(
            *("return_stats cannot be combined with return_aux=True",)
        )

    # Validate parameters
    if isinstance(initial_clipping_norm, PerGroup):
        for gname, val in initial_clipping_norm.values.items():
            if val <= 0:
                raise ConfigurationError(
                    *(
                        f"initial_clipping_norm must be positive for all groups, "
                        f"got {val} for group '{gname}'",
                    )
                )
    elif initial_clipping_norm <= 0:
        raise ConfigurationError(
            *(f"initial_clipping_norm must be positive, got {initial_clipping_norm}",)
        )
    if not 0 < target_quantile < 1:
        raise ConfigurationError(
            *(f"target_quantile must be in (0, 1), got {target_quantile}",)
        )
    if learning_rate <= 0:
        raise ConfigurationError(
            *(f"learning_rate must be positive, got {learning_rate}",)
        )
    if clipping_norm_min <= 0:
        raise ConfigurationError(
            *(f"clipping_norm_min must be positive, got {clipping_norm_min}",)
        )
    if clipping_norm_max <= clipping_norm_min:
        raise ConfigurationError(
            *(
                f"clipping_norm_max ({clipping_norm_max}) must be > clipping_norm_min ({clipping_norm_min})",
            )
        )
    if fraction_noise_std <= 0:
        raise ConfigurationError(
            *(f"fraction_noise_std must be positive, got {fraction_noise_std}",)
        )

    # Normalize argnums/batch_argnums for empty-batch detection.
    argnums_tuple = normalize_to_tuple(argnums)
    batch_argnums_raw = clipped_grad_kwargs.get("batch_argnums", 1)
    batch_argnums_tuple = normalize_to_tuple(batch_argnums_raw)

    # Read static config that the empty-batch path needs to compute max_norms
    # without going through ``clipped_grad`` itself.
    normalize_by = clipped_grad_kwargs.get("normalize_by", 1.0)
    second_moment = bool(clipped_grad_kwargs.get("second_moment", False))

    # Validate the same static args ``clipped_grad`` validates at factory time,
    # so misconfigurations (e.g. ``normalize_by <= 0`` or argnums/batch_argnums
    # collisions) fail at construction here rather than first surfacing on the
    # next non-empty step — keeping empty- and non-empty-batch failure modes
    # consistent.
    _validate_static_args(argnums, batch_argnums_raw, normalize_by)
    output_bound = lambda clipping_norm: clipping_norm / normalize_by  # noqa: E731

    def _output_squared_bound(clipping_norm):
        return (clipping_norm * clipping_norm) / normalize_by

    config = {
        "target_quantile": target_quantile,
        "learning_rate": learning_rate,
        "clipping_norm_min": clipping_norm_min,
        "clipping_norm_max": clipping_norm_max,
        "fraction_noise_std": fraction_noise_std,
    }

    is_per_group = isinstance(initial_clipping_norm, PerGroup)

    def _empty_num_clipped():
        if is_per_group:
            return dict.fromkeys(sorted(initial_clipping_norm.values), 0.0)
        return 0.0

    def _empty_batch_state(state: AdaptiveClipState) -> AdaptiveClipState:
        """Build new state for an empty batch: clip_norm preserved, step bumped."""
        return AdaptiveClipState(
            _current_clipping_norm=state._next_clipping_norm,
            _next_clipping_norm=state._next_clipping_norm,
            _step=state._step + 1,
            _rng_key=state._rng_key,
            _fraction_noise_std=config["fraction_noise_std"],
            _learning_rate=config["learning_rate"],
            _target_quantile=config["target_quantile"],
            _clipping_norm_min=config["clipping_norm_min"],
            _clipping_norm_max=config["clipping_norm_max"],
            _num_clipped=_empty_num_clipped(),
            _batch_size=0.0,
        )

    inner_return_aux = return_aux
    inner_fn, _ = clipped_grad(
        loss_fn,
        argnums=argnums,
        has_aux=has_aux,
        clipping_norm=initial_clipping_norm,
        return_aux=inner_return_aux,
        return_stats=not inner_return_aux,
        pre_clipping_transform=pre_clipping_transform,
        **clipped_grad_kwargs,
    )

    def grad_fn(*args, state: AdaptiveClipState, **kwargs):
        """Compute clipped gradients with adaptive threshold.

        Args:
            *args: Positional arguments to pass to the loss function.
            state: Current AdaptiveClipState.
            **kwargs: Keyword arguments to pass to the loss function.

        Returns:
            If return_aux:
                ((grad, aux), new_state)
            Else:
                (grad, new_state)
        """
        with record_function("opaque::adaptive_clipped_grad"):
            return _grad_fn_impl(*args, state=state, **kwargs)

    def _grad_fn_impl(*args, state: AdaptiveClipState, **kwargs):
        # Empty batch: zero grads, no adaptation, step still incremented.
        # When ``second_moment=True`` was forwarded to the inner ``clipped_grad``,
        # mirror its empty-batch shape (paired ``SecondMomentClippingOutput``) so
        # downstream noise + optimizer dispatch stays paired across empty and
        # non-empty steps. The squared-stream sensitivity is C^2/normalize_by, the
        # same bound ``clipped_grad`` would attach on a non-empty step.
        if batch_size_from_args(args, batch_argnums_tuple) == 0:
            new_state = _empty_batch_state(state)
            # Build the zero aggregate in the transformed gradient schema.
            zeros = empty_clipped_grads_like(
                args,
                argnums_tuple,
                pre_clipping_transform,
                clipped_grad_kwargs.get("dtype"),
            )
            grads = _empty_batch_grads(
                zeros,
                output_bound(state._next_clipping_norm),
                _output_squared_bound(state._next_clipping_norm),
                second_moment,
            )
            if return_aux:
                empty = torch.empty(0)
                # ``clipping_rate`` presence is schema-driven: None for
                # per-group, like non-empty steps.
                adaptive_aux = AdaptiveClippedGradAux(
                    loss_values=empty,
                    grad_norms=empty,
                    clipped_grad_norms=empty,
                    loss_aux=None,
                    clipping_rate=None if is_per_group else 0.0,
                )
                return (grads, adaptive_aux), new_state
            if return_stats:
                if is_per_group:
                    empty_counts = dict.fromkeys(
                        sorted(initial_clipping_norm.values), 0.0
                    )
                    empty_rates = dict.fromkeys(
                        sorted(initial_clipping_norm.values), 0.0
                    )
                    stats = ClippingStats(
                        num_clipped=empty_counts,
                        clipping_rate=empty_rates,
                        batch_size=0,
                    )
                else:
                    stats = ClippingStats(
                        num_clipped=0.0,
                        clipping_rate=0.0,
                        batch_size=0,
                    )
                return (grads, stats), new_state
            return grads, new_state

        # Compute gradients with the current threshold. The engine transform is
        # constructed once; its tensor-only chunk kernel receives the changing
        # threshold as data so compilation does not specialize on its value.
        user_wants_return_aux = return_aux
        result, _ = inner_fn(
            *args,
            state=_RuntimeClipState(state._next_clipping_norm),
            **kwargs,
        )

        # Extract gradients and auxiliary output
        aux = None
        stats: ClippingStats | None = None
        if isinstance(result, tuple):
            if inner_return_aux:
                grads, aux = result
                grad_norms = aux.grad_norms
            else:
                grads, stats = result
                grad_norms = None
        else:
            grads = result
            grad_norms = None

        batch_size = (
            aux.batch_size
            if aux is not None
            else (stats.batch_size if stats is not None else 0)
        )

        if is_per_group and (grad_norms is not None or stats is not None):
            # --- Per-group adaptive path ---
            current_pg = state._next_clipping_norm
            if (
                stats is not None
                and isinstance(stats.num_clipped, dict)
                and batch_size > 0
            ):
                per_group_num_clipped = {
                    gname: stats.num_clipped[gname]
                    for gname in sorted(current_pg.values)
                }
                per_group_rates = {
                    gname: (
                        stats.clipping_rate.get(gname, 0.0)
                        if isinstance(stats.clipping_rate, dict)
                        else 0.0
                    )
                    for gname in sorted(current_pg.values)
                }
                new_values: dict[str, float] = {}
                for i, gname in enumerate(sorted(current_pg.values)):
                    threshold = current_pg.values[gname]
                    rate = per_group_rates.get(gname, 0.0)

                    group_key = fold_in(
                        state._rng_key, ADAPTIVE_CLIPPING_STREAM_FOLD, state._step, i
                    )
                    generator = generator_from_key(group_key)
                    noise = (
                        torch.randn(1, generator=generator).item()
                        * config["fraction_noise_std"]
                    )
                    noisy_rate = rate + noise

                    new_values[gname] = _adaptive_clipping_norm_update(
                        base_clipping_norm=threshold,
                        noisy_clipping_rate=noisy_rate,
                        target_quantile=config["target_quantile"],
                        learning_rate=config["learning_rate"],
                        clipping_norm_min=config["clipping_norm_min"],
                        clipping_norm_max=config["clipping_norm_max"],
                    )
                new_clipping_norm = PerGroup(
                    groups=current_pg.groups, values=new_values
                )
                num_clipped = per_group_num_clipped
            elif aux is not None and aux.group_norms is not None and batch_size > 0:
                per_group_num_clipped: dict[str, float] = {}
                new_values: dict[str, float] = {}
                for i, gname in enumerate(sorted(current_pg.values)):
                    threshold = current_pg.values[gname]
                    gnorms = aux.group_norms[gname]
                    nc = float((gnorms > threshold).sum().item())
                    per_group_num_clipped[gname] = nc
                    rate = nc / max(1.0, float(batch_size))

                    # Independent noise per group: fold in both step and group index
                    group_key = fold_in(
                        state._rng_key, ADAPTIVE_CLIPPING_STREAM_FOLD, state._step, i
                    )
                    generator = generator_from_key(group_key)
                    noise = (
                        torch.randn(1, generator=generator).item()
                        * config["fraction_noise_std"]
                    )
                    noisy_rate = rate + noise

                    new_values[gname] = _adaptive_clipping_norm_update(
                        base_clipping_norm=threshold,
                        noisy_clipping_rate=noisy_rate,
                        target_quantile=config["target_quantile"],
                        learning_rate=config["learning_rate"],
                        clipping_norm_min=config["clipping_norm_min"],
                        clipping_norm_max=config["clipping_norm_max"],
                    )

                new_clipping_norm = PerGroup(
                    groups=current_pg.groups, values=new_values
                )
                num_clipped = per_group_num_clipped
            else:
                new_clipping_norm = state._next_clipping_norm
                num_clipped = _empty_num_clipped()
        elif grad_norms is not None or stats is not None:
            # --- Scalar adaptive path (original) ---
            if stats is not None and isinstance(stats.num_clipped, float):
                num_clipped = stats.num_clipped
                clipping_rate = (
                    stats.clipping_rate
                    if isinstance(stats.clipping_rate, float)
                    else 0.0
                )
            else:
                num_clipped = float(
                    (grad_norms > state._next_clipping_norm).sum().item()
                )
                clipping_rate = aux.clipping_rate

            noisy_clipping_rate = _sample_noisy_clipping_rate(
                clipping_rate,
                key=state._rng_key,
                step=state._step,
                fraction_noise_std=config["fraction_noise_std"],
            )

            new_clipping_norm = _adaptive_clipping_norm_update(
                base_clipping_norm=state._next_clipping_norm,
                noisy_clipping_rate=noisy_clipping_rate,
                target_quantile=config["target_quantile"],
                learning_rate=config["learning_rate"],
                clipping_norm_min=config["clipping_norm_min"],
                clipping_norm_max=config["clipping_norm_max"],
            )
        else:
            new_clipping_norm = state._next_clipping_norm
            num_clipped = _empty_num_clipped()

        new_state = AdaptiveClipState(
            _current_clipping_norm=state._next_clipping_norm,
            _next_clipping_norm=new_clipping_norm,
            _step=state._step + 1,
            _rng_key=state._rng_key,
            _fraction_noise_std=config["fraction_noise_std"],
            _learning_rate=config["learning_rate"],
            _target_quantile=config["target_quantile"],
            _clipping_norm_min=config["clipping_norm_min"],
            _clipping_norm_max=config["clipping_norm_max"],
            _num_clipped=num_clipped,
            _batch_size=float(batch_size),
        )

        if user_wants_return_aux:
            adaptive_aux = AdaptiveClippedGradAux(
                loss_values=aux.loss_values if aux is not None else None,
                grad_norms=aux.grad_norms if aux is not None else None,
                clipped_grad_norms=(
                    aux.clipped_grad_norms if aux is not None else None
                ),
                loss_aux=aux.loss_aux if aux is not None else None,
                clipping_rate=aux.clipping_rate if aux is not None else None,
                batch_size=batch_size,
                group_norms=aux.group_norms if aux is not None else None,
            )
            return (grads, adaptive_aux), new_state

        if return_stats:
            assert stats is not None
            return (grads, stats), new_state
        return grads, new_state

    # Create initial state
    initial_state = AdaptiveClipState(
        _current_clipping_norm=initial_clipping_norm,
        _next_clipping_norm=initial_clipping_norm,
        _step=0,
        _rng_key=key,
        _fraction_noise_std=fraction_noise_std,
        _learning_rate=learning_rate,
        _target_quantile=target_quantile,
        _clipping_norm_min=clipping_norm_min,
        _clipping_norm_max=clipping_norm_max,
        _num_clipped=_empty_num_clipped(),
        _batch_size=0,
    )

    return grad_fn, initial_state


__all__ = [
    "AdaptiveClipState",
    "AdaptiveClippedGradAux",
    "adaptive_clipped_grad",
]
