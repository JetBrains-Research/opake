"""Per-example clipping and summing for arbitrary functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch.func import vmap as _vmap

from opaque.api.engine.clipping._helpers import batch_size_from_args, normalize_to_tuple
from opaque.api.engine.clipping._pytree import clip_pytree
from opaque.api.engine.pytree import global_norm, tree_leaves, tree_map
from opaque.api.engine.types import (
    ClippedPytree,
    PerGroup,
    SecondMomentClippingOutput,
    clipped,
)
from opaque.api.engine.types import ClipState as _ClipState
from opaque.exceptions import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class FixedClipState(_ClipState):
    """Marker state for fixed (non-adaptive) clipping.

    Returned by :func:`clipped_fun` and :func:`opaque.api.engine.clipping.clipped_grad`.
    Carries no fields; the configured clipping threshold flows through
    the ``ClippedPytree.max_norm`` metadata, not through the state.
    """


@dataclass(frozen=True)
class _RuntimeClipState(FixedClipState):
    """Internal state carrying a changing threshold into a reusable kernel."""

    clipping_norm: float | PerGroup


@dataclass(frozen=True)
class ClippedFunAux:
    """Diagnostic outputs from clipped_fun.

    All fields are diagnostic — they reflect pre-noise, pre-aggregation
    values and must not be fed back into private computation.  Use the
    returned ``ClippedPytree.max_norm`` metadata for noise calibration.

    Fields:
        values: Per-example function values before clipping.
        norms: Per-example L2 norms before clipping.
        clipped_norms: Per-example L2 norms after clipping.
        value_aux: Per-example auxiliary payload returned by the wrapped function.
        clipping_rate: Fraction of per-example outputs whose norm exceeded the
            clipping threshold.  Equal to ``num_clipped / batch_size``.
        batch_size: Number of examples in the batch.
        group_norms: Per-group per-example L2 norms before clipping
            (dict[str, Tensor] with shape [batch_size] per group), or None
            when global clipping is used.
    """

    values: Any | None = None
    norms: Any | None = None
    clipped_norms: Any | None = None
    value_aux: Any | None = None
    clipping_rate: float | None = None
    batch_size: int = 0
    group_norms: dict[str, torch.Tensor] | None = None


@dataclass(frozen=True)
class ClippingStats:
    """Aggregated clipping statistics without per-example materialization."""

    num_clipped: float | dict[str, float]
    clipping_rate: float | dict[str, float] | None
    batch_size: int = 0


def _resolve_compute_dtype(
    tensor: torch.Tensor,
    compute_dtype: torch.dtype | None,
) -> torch.dtype | None:
    """Resolve safe compute dtype for reductions.

    If compute_dtype is explicitly requested, use it. Otherwise, promote
    low-precision floating reductions (fp16/bf16) to float32 for numerical
    stability.  Returns ``None`` to mean "no promotion needed" — the caller
    can pass that directly to ``torch.sum(dtype=None)`` (default behavior).
    """
    if compute_dtype is not None:
        return compute_dtype
    if torch.is_floating_point(tensor) and tensor.dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        return torch.float32
    return None


def _sum_clipped_tensor(
    tensor: torch.Tensor,
    *,
    dim: int,
    output_dtype: torch.dtype | None,
    compute_dtype: torch.dtype | None,
) -> torch.Tensor:
    """Sum with separate compute (accumulation) and output dtype.

    ``compute_dtype`` controls the reduction precision; ``output_dtype`` the
    caller-visible result dtype.  Defaults preserve the type-stable contract
    (output dtype = input dtype) with auto-fp32 promotion for bf16/fp16 inputs.
    """
    accum_dtype = _resolve_compute_dtype(tensor, compute_dtype)
    summed = torch.sum(tensor, dim=dim, dtype=accum_dtype)

    target = output_dtype if output_dtype is not None else tensor.dtype
    if summed.dtype != target:
        return summed.to(dtype=target)
    return summed


def _accumulation_dtype(
    tensor: torch.Tensor,
    output_dtype: torch.dtype | None,
    compute_dtype: torch.dtype | None,
) -> torch.dtype | None:
    """Choose the chunk reduction dtype without losing output precision."""
    resolved = _resolve_compute_dtype(tensor, compute_dtype)
    if output_dtype is None:
        return resolved
    if resolved is None:
        resolved = tensor.dtype
    return torch.promote_types(resolved, output_dtype)


class _MicrobatchAccumulator:
    """Running sum over microbatches, held at the accumulation precision.

    The sum stays in the wider of ``compute_dtype`` and the output dtype and is
    cast to the output dtype once, at the end.
    """

    __slots__ = ("_output_dtype", "_targets", "_total")

    def __init__(self, *, output_dtype: torch.dtype | None) -> None:
        self._output_dtype = output_dtype
        self._total: Any | None = None
        self._targets: Any | None = None

    def add_reduced(self, values: Any, dtype_markers: Any) -> None:
        """Add one already-reduced chunk without materializing example gradients.

        ``dtype_markers`` are scalar tensors carrying the dtypes of the
        per-example leaves.  The chunk kernel may reduce in a wider dtype, so
        those markers preserve the caller-visible dtype without returning the
        microbatch-sized unclipped or clipped values.
        """
        if self._targets is None:
            self._targets = tree_map(
                lambda marker: (
                    marker.dtype if self._output_dtype is None else self._output_dtype
                ),
                dtype_markers,
            )
        self._total = (
            values
            if self._total is None
            else tree_map(lambda acc, new: acc + new, self._total, values)
        )

    def result(self) -> Any:
        """The accumulated sum in the caller-visible dtype, or None if unused."""
        if self._total is None:
            return None
        return tree_map(
            lambda acc, target: acc if acc.dtype == target else acc.to(dtype=target),
            self._total,
            self._targets,
        )


def _validate_clipping_norm(clipping_norm: float | PerGroup) -> None:
    if isinstance(clipping_norm, PerGroup):
        for group_name, value in clipping_norm.values.items():
            if value <= 0:
                raise ConfigurationError(
                    *(
                        "clipping_norm must be positive for all groups, "
                        f"got {value} for group '{group_name}'",
                    )
                )
        return
    if clipping_norm <= 0:
        raise ConfigurationError(
            *(f"clipping_norm must be positive, got {clipping_norm}",)
        )


def _prepare_kernel_clipping_norm(
    clipping_norm: float | PerGroup,
    args: tuple[Any, ...],
    batch_argnums: tuple[int, ...],
) -> torch.Tensor | PerGroup:
    """Require a non-empty batch and place its clipping norm on the batch device."""
    batch_leaves = tree_leaves(args[batch_argnums[0]])
    tensor = next(
        (leaf for leaf in batch_leaves if isinstance(leaf, torch.Tensor)), None
    )
    if tensor is None:
        raise ConfigurationError(
            *("Could not determine the device for the runtime clipping norm",)
        )
    if tensor.shape[0] == 0:
        raise ConfigurationError(
            *(
                "Function-output clipping requires a non-empty batch because "
                "the output pytree cannot be inferred otherwise.",
            )
        )

    if isinstance(clipping_norm, PerGroup):
        return PerGroup(
            clipping_norm.groups,
            {
                name: torch.as_tensor(value, device=tensor.device)
                for name, value in clipping_norm.values.items()
            },
        )
    return torch.as_tensor(clipping_norm, device=tensor.device)


def _microbatch_accumulate_reduced(
    chunk_fn: Callable,
    kernel_clipping_norm: torch.Tensor | PerGroup,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    batch_argnums: tuple[int, ...],
    microbatch_size: int,
    return_aux: bool,
    return_stats: bool,
    dtype: torch.dtype | None,
    clipping_norm: float | PerGroup,
    second_moment: bool,
) -> tuple[Any, Any, Any, ClippingStats | None]:
    """Run the tensor-only chunk kernel and combine its compact reductions eagerly."""
    batch_size = batch_size_from_args(args, batch_argnums)
    grad_acc = _MicrobatchAccumulator(output_dtype=dtype)
    squared_acc = _MicrobatchAccumulator(output_dtype=dtype)
    aux_list: list[Any] = []
    total_batch_size = 0
    if isinstance(clipping_norm, PerGroup):
        total_num_clipped: float | dict[str, float] = dict.fromkeys(
            sorted(clipping_norm.values), 0.0
        )
    else:
        total_num_clipped = 0.0

    for start_idx in range(0, batch_size, microbatch_size):
        end_idx = min(start_idx + microbatch_size, batch_size)
        microbatch_args = list(args)
        for i in batch_argnums:
            microbatch_args[i] = tree_map(
                lambda x, s=start_idx, e=end_idx: (
                    x[s:e] if isinstance(x, torch.Tensor) else x
                ),
                args[i],
            )

        reduced, markers, squared_reduced, squared_markers, diagnostics = chunk_fn(
            kernel_clipping_norm, *microbatch_args, **kwargs
        )
        grad_acc.add_reduced(reduced, markers)
        if second_moment:
            squared_acc.add_reduced(squared_reduced, squared_markers)

        if return_aux:
            aux_list.append(diagnostics)
        elif return_stats:
            chunk_stats = _compute_clipping_stats(
                diagnostics["norms"],
                clipping_norm=clipping_norm,
                group_norms_dict=diagnostics.get("group_norms"),
            )
            total_batch_size += chunk_stats.batch_size
            if isinstance(total_num_clipped, dict):
                assert isinstance(chunk_stats.num_clipped, dict)
                for name, count in chunk_stats.num_clipped.items():
                    total_num_clipped[name] += count
            else:
                assert isinstance(chunk_stats.num_clipped, float)
                total_num_clipped += chunk_stats.num_clipped

    if return_aux:

        def concat_leaves(*leaf_values):
            if all(isinstance(v, torch.Tensor) for v in leaf_values):
                return torch.cat(leaf_values, dim=0)
            return leaf_values[0]

        aux = tree_map(concat_leaves, *aux_list)
    else:
        aux = ()

    stats = None
    if return_stats:
        clipping_rate: float | dict[str, float]
        if isinstance(total_num_clipped, dict):
            clipping_rate = {
                name: count / max(1.0, float(total_batch_size))
                for name, count in total_num_clipped.items()
            }
        else:
            clipping_rate = total_num_clipped / max(1.0, float(total_batch_size))
        stats = ClippingStats(
            num_clipped=total_num_clipped,
            clipping_rate=clipping_rate,
            batch_size=total_batch_size,
        )

    return grad_acc.result(), squared_acc.result(), aux, stats


def _compute_clipping_stats(
    norms: torch.Tensor | None,
    *,
    clipping_norm: float | PerGroup,
    group_norms_dict: dict[str, torch.Tensor] | None,
) -> ClippingStats:
    """Compute aggregated clipping statistics from materialized norm tensors."""
    batch_size = norms.numel() if isinstance(norms, torch.Tensor) else 0
    if batch_size == 0:
        if isinstance(clipping_norm, PerGroup):
            group_names = sorted(clipping_norm.values)
            empty_counts = dict.fromkeys(group_names, 0.0)
            empty_rates = dict.fromkeys(group_names, 0.0)
            return ClippingStats(
                num_clipped=empty_counts,
                clipping_rate=empty_rates,
                batch_size=0,
            )
        return ClippingStats(
            num_clipped=0.0,
            clipping_rate=0.0,
            batch_size=0,
        )

    if isinstance(clipping_norm, PerGroup) and group_norms_dict is not None:
        counts = {
            gname: float(
                (group_norms_dict[gname] > clipping_norm.values[gname]).sum().item()
            )
            for gname in sorted(clipping_norm.values)
        }
        rates = {gname: count / float(batch_size) for gname, count in counts.items()}
        return ClippingStats(
            num_clipped=counts,
            clipping_rate=rates,
            batch_size=batch_size,
        )

    effective_cn = (
        clipping_norm.effective
        if isinstance(clipping_norm, PerGroup)
        else clipping_norm
    )
    num_clipped = float((norms > effective_cn).sum().item())
    return ClippingStats(
        num_clipped=num_clipped,
        clipping_rate=num_clipped / float(batch_size),
        batch_size=batch_size,
    )


def _prepare_clipped_fun(
    batch_argnums: int | tuple[int, ...],
    clipping_norm: float | PerGroup,
    return_aux: bool,
    return_stats: bool,
) -> tuple[int, ...]:
    if return_aux and return_stats:
        raise ConfigurationError(
            *("return_stats cannot be combined with return_aux=True",)
        )
    normalized_batch_argnums = normalize_to_tuple(batch_argnums)
    _validate_clipping_norm(clipping_norm)
    return normalized_batch_argnums


def _resolve_runtime_clipping_norm(
    configured: float | PerGroup,
    runtime: float | PerGroup | None,
    normalize_by: float,
    second_moment: bool,
) -> tuple[float | PerGroup, float | PerGroup, float | PerGroup | None]:
    current = configured if runtime is None else runtime
    output_bound = current / normalize_by
    squared_bound = (current * current) / normalize_by if second_moment else None
    return current, output_bound, squared_bound


def clipped_fun(
    fun: Callable[..., Any],
    has_aux: bool = False,
    *,
    batch_argnums: int | tuple[int, ...] = 0,
    clipping_norm: float | PerGroup = 1.0,
    normalize_by: float = 1.0,
    return_aux: bool = False,
    return_stats: bool = False,
    second_moment: bool = False,
    microbatch_size: int | None = None,
    dtype: torch.dtype | None = None,
    compute_dtype: torch.dtype | None = None,
    _scale_fn: Callable | None = None,
    _chunk_compiler: Callable | None = None,
) -> tuple[Callable, FixedClipState]:
    """Transform a function to clip its output and sum across a batch.

    This is the primary API for per-example clipping in DP-SGD. It wraps a function
    to clip each per-example output to a maximum L2 norm, then sums the clipped outputs.

    The returned pytree is wrapped as :class:`ClippedPytree` (single-stream)
    or :class:`SecondMomentClippingOutput` (paired-stream when
    ``second_moment=True``), carrying the post-normalization
    ``max_norm`` for downstream noise calibration.  The bound is part of
    the contract: consumers (``gaussian_noise``, ``mf_gaussian_noise``) read it
    directly without the caller threading a separate ``sensitivity``
    argument.  Unwrap to a raw pytree via ``.pytree`` if you need the
    summed values without metadata.

    Example Usage:
        >>> from opaque.api.engine.clipping._clipped_fun import clipped_fun
        >>> data = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        >>> clipped_mean, clip_state = clipped_fun(torch.mean, clipping_norm=1.0)
        >>> result, clip_state = clipped_mean(data, state=clip_state)
        >>> result.pytree
        tensor(5.)
        >>> result.max_norm
        1.0

    Formal Guarantees:
        For the first function output:
          The L2 sensitivity of the returned function with respect to the batch
          arguments (specified by `batch_argnums`) under add/remove or zero-out
          differential privacy definitions is guaranteed to be `clipping_norm`.
          Under replace-one DP, the sensitivity is doubled (2 * `clipping_norm`).
        Extra auxiliary outputs (aux, norms) are per-example. This function
          guarantees that per-example outputs only depend on the data for the same
          example. This allows maximum flexibility for the caller to aggregate
          these as desired (possibly with a DP mean, median, quantile, or histogram
          mechanism).

    Args:
        fun: The function to be clipped.
        has_aux: If True, `fun` is expected to return a tuple `(value, loss_aux)`. Only
            the value will be clipped + aggregated, `loss_aux` will be returned on a
            per-example basis. Exercise caution when using this as the sensitivity
            guarantees of the returned Callable are only provided w.r.t. `value`.
        batch_argnums: Specifies which argument(s) of `fun` contain the batch
            dimension. All arguments specified here must have the same size along the
            0th axis.
        clipping_norm: The maximum L2 norm allowed.
        normalize_by: Divide the clipped output by this value before returning.
        return_aux: If True, the returned Callable will return a per-example aux
            dataclass containing the original per-example values, per-example norms
            before clipping, and any auxiliary data returned by `fun`.
        return_stats: If True, return :class:`ClippingStats` with aggregate
            clipping counts, rates, and batch size. Cannot be combined with
            ``return_aux``.
        second_moment: If True, also accumulate the element-wise sum of
            per-example squared clipped values, i.e. ``Σᵢ gᵢ²``.  The
            squaring happens inside the per-example loop so the
            second-stream sensitivity is the per-record squared bound
            ``C²`` (averaged: ``C² / normalize_by``).  The wrapped
            output becomes :class:`SecondMomentClippingOutput` with both
            streams.  Per-group ``clipping_norm`` is supported and yields
            ``SecondMomentClippingOutput`` with per-group ``max_norm``
            on both streams.
        microbatch_size: If set, the batch is split up into microbatches of this
            size for memory-efficient processing. Processes each microbatch separately
            and accumulates results without materializing the full batch of gradients.
            Set this to reduce peak memory usage at the cost of slightly slower computation.
            The running sum is held in the wider of ``compute_dtype`` and the
            output dtype, so a bf16/fp16 run keeps one float32 copy of the summed
            output. Pass ``compute_dtype=torch.bfloat16`` to give that memory back,
            at the cost of accumulation precision when the output dtype is also low
            precision.
        dtype: Optional dtype for the clipped+aggregated pytree. If None, the dtype
            will be the same as the dtypes of the function output.
        compute_dtype: Internal reduction and accumulation dtype. ``None``
            (default) auto-promotes bf16/fp16 to float32 for numerical stability;
            an explicit dtype selects the reduction precision. Leaf scaling and
            microbatch accumulation use the wider of this dtype and the leaf or
            output storage dtype, so explicit lower precision never narrows a wider
            value. Independent of ``dtype`` (which controls the *output* dtype).
            Applies across microbatches too, so microbatched and non-microbatched
            runs agree at their resolved accumulation precision.
    Returns:
        A tuple ``(clip_fn, FixedClipState)`` where ``clip_fn(*args, state=...)``
        clips the output of ``fun`` and sums across the batch.  The exact
        return shape depends on ``second_moment`` and ``return_aux``:

        | ``second_moment`` | ``return_aux`` | ``clip_fn`` returns                                |
        | :---------------- | :------------- | :------------------------------------------------- |
        | False             | False          | ``(ClippedPytree, state)``                         |
        | False             | True           | ``((ClippedPytree, ClippedFunAux), state)``        |
        | True              | False          | ``(SecondMomentClippingOutput, state)``            |
        | True              | True           | ``((SecondMomentClippingOutput, ClippedFunAux), state)`` |

        With ``return_stats=True`` and ``return_aux=False``, the clipped value
        is paired with :class:`ClippingStats`.

    Raises:
        ConfigurationError: If the batch is empty.
    """
    batch_argnums = _prepare_clipped_fun(
        batch_argnums, clipping_norm, return_aux, return_stats
    )

    # Wrap function to handle has_aux - use empty tuple () not None!
    if not has_aux:

        def fun_with_aux(*args, **kwargs):
            return (fun(*args, **kwargs), ())

    else:
        fun_with_aux = fun

    clip_state = FixedClipState()

    def _per_example_fn(kernel_clipping_norm, *args_single, **call_kwargs):
        value, aux = fun_with_aux(*args_single, **call_kwargs)
        if _scale_fn is None:
            clipped_value, norm = clip_pytree(
                value,
                clipping_norm=kernel_clipping_norm,
                compute_dtype=compute_dtype,
            )
        else:
            clipped_value, norm = _scale_fn(value)
        squared_value = (
            tree_map(
                lambda x: x.square() if isinstance(x, torch.Tensor) else x,
                clipped_value,
            )
            if second_moment
            else None
        )
        if return_aux or return_stats:
            diagnostics = {"norms": norm.norm.detach()}
            if return_aux:
                diagnostics["clipped_norms"] = global_norm(
                    clipped_value, compute_dtype=compute_dtype
                ).detach()
            if norm.group_norms is not None:
                diagnostics["group_norms"] = {
                    key: group_norm.detach()
                    for key, group_norm in norm.group_norms.items()
                }

            if return_aux and isinstance(aux, dict):
                if "values" in aux:
                    aux_value = aux["values"]
                    diagnostics["values"] = (
                        aux_value.detach()
                        if isinstance(aux_value, torch.Tensor)
                        else aux_value
                    )
                else:
                    diagnostics["values"] = (
                        value.detach() if isinstance(value, torch.Tensor) else value
                    )
                if has_aux:
                    diagnostics["value_aux"] = aux.get("value_aux", aux)
            elif return_aux:
                diagnostics["values"] = (
                    value.detach() if isinstance(value, torch.Tensor) else value
                )
                if has_aux:
                    diagnostics["value_aux"] = aux

            if second_moment:
                return clipped_value, squared_value, diagnostics
            return clipped_value, diagnostics
        if second_moment:
            return clipped_value, squared_value
        return clipped_value

    if _chunk_compiler is not None and microbatch_size is None:
        raise ConfigurationError(
            *("_chunk_compiler requires a finite microbatch_size",)
        )

    def _make_chunk_kernel(in_dims):
        # Fixed and AUTO-S scaling are construction-time constants.  A
        # stateful scaling parameter must cross this private boundary as a
        # tensor input rather than be captured as changing Python state.
        def _chunk_kernel(kernel_clipping_norm, *chunk_args, **call_kwargs):
            """Run vmap, clipping, diagnostics, and reduction for one chunk."""
            n_outputs = 1 + int(bool(second_moment)) + int(return_aux or return_stats)
            out_dims = 0 if n_outputs == 1 else (0,) * n_outputs
            vmapped = _vmap(
                _per_example_fn,
                in_dims=(None, *in_dims),
                out_dims=out_dims,
                randomness="same",
            )
            outputs = vmapped(kernel_clipping_norm, *chunk_args, **call_kwargs)
            if n_outputs == 1:
                clipped_values = outputs
                squared_values = None
                diagnostics = ()
            else:
                output_index = 0
                clipped_values = outputs[output_index]
                output_index += 1
                squared_values = outputs[output_index] if second_moment else None
                if second_moment:
                    output_index += 1
                diagnostics = (
                    outputs[output_index] if (return_aux or return_stats) else ()
                )

            reduced = tree_map(
                lambda x: torch.sum(
                    x,
                    dim=0,
                    dtype=_accumulation_dtype(x, dtype, compute_dtype),
                ),
                clipped_values,
            )
            dtype_markers = tree_map(lambda x: x.new_zeros(()), clipped_values)
            if second_moment:
                squared_reduced = tree_map(
                    lambda x: torch.sum(
                        x,
                        dim=0,
                        dtype=_accumulation_dtype(x, dtype, compute_dtype),
                    ),
                    squared_values,
                )
                squared_dtype_markers = tree_map(
                    lambda x: x.new_zeros(()), squared_values
                )
            else:
                squared_reduced = ()
                squared_dtype_markers = ()
            return (
                reduced,
                dtype_markers,
                squared_reduced,
                squared_dtype_markers,
                diagnostics,
            )

        return _chunk_kernel

    # Cache by positional batching structure so callers may use any valid
    # positional/default-argument arrangement without reusing incompatible
    # vmap ``in_dims``.
    chunk_kernels: dict[tuple[int | None, ...], Callable] = {}

    def clipped_fn(runtime_clipping_norm, *args, **kwargs):
        current_clipping_norm, output_max_norm, output_squared_max_norm = (
            _resolve_runtime_clipping_norm(
                clipping_norm,
                runtime_clipping_norm,
                normalize_by,
                second_moment,
            )
        )
        kernel_clipping_norm = _prepare_kernel_clipping_norm(
            current_clipping_norm, args, batch_argnums
        )
        in_dims = tuple(0 if i in batch_argnums else None for i in range(len(args)))
        per_example_fn = _per_example_fn

        # Choose execution path based on microbatch_size
        stats = None
        if microbatch_size is None:
            # Fast path: vmap entire batch at once.  Output shape depends
            # on the (second_moment, return_aux) flags — see the per_example_fn
            # branches above.  When n_outputs == 1, vmap returns the single
            # pytree (which may itself be a tuple of tensors for tuple
            # params); when n_outputs > 1 the per_example_fn returns a
            # tuple of n_outputs pytrees.
            n_outputs = 1 + int(bool(second_moment)) + int(return_aux or return_stats)
            out_dims = 0 if n_outputs == 1 else (0,) * n_outputs
            vmapped = _vmap(
                per_example_fn,
                in_dims=(None, *in_dims),
                out_dims=out_dims,
                randomness="same",
            )
            outputs = vmapped(kernel_clipping_norm, *args, **kwargs)
            if n_outputs == 1:
                clipped_values = outputs
                squared_values = None
                aux = ()
            else:
                idx = 0
                clipped_values = outputs[idx]
                idx += 1
                squared_values = outputs[idx] if second_moment else None
                if second_moment:
                    idx += 1
                aux = outputs[idx] if (return_aux or return_stats) else ()

            # Sum clipped values across batch dimension
            result = tree_map(
                lambda x: _sum_clipped_tensor(
                    x, dim=0, output_dtype=dtype, compute_dtype=compute_dtype
                ),
                clipped_values,
            )
            squared_result = (
                tree_map(
                    lambda x: _sum_clipped_tensor(
                        x, dim=0, output_dtype=dtype, compute_dtype=compute_dtype
                    ),
                    squared_values,
                )
                if second_moment
                else None
            )
        else:
            chunk_kernel = chunk_kernels.get(in_dims)
            if chunk_kernel is None:
                eager_chunk_kernel = _make_chunk_kernel(in_dims)
                if _chunk_compiler is None:
                    chunk_kernel = eager_chunk_kernel
                else:
                    # Compilation policy belongs to the owner; the kernel
                    # itself remains tensor-only. Static chunk variants bound
                    # recompilation by ``microbatch_size`` rather than by every
                    # realized Poisson batch size.
                    chunk_kernel = _chunk_compiler(eager_chunk_kernel)
                chunk_kernels[in_dims] = chunk_kernel
            result, squared_result, aux, stats = _microbatch_accumulate_reduced(
                chunk_fn=chunk_kernel,
                kernel_clipping_norm=kernel_clipping_norm,
                args=args,
                kwargs=kwargs,
                batch_argnums=batch_argnums,
                microbatch_size=microbatch_size,
                return_aux=return_aux,
                return_stats=return_stats,
                dtype=dtype,
                clipping_norm=current_clipping_norm,
                second_moment=second_moment,
            )

        # Normalize
        if normalize_by != 1.0:
            result = tree_map(lambda x: x / normalize_by, result)
            if second_moment:
                squared_result = tree_map(lambda x: x / normalize_by, squared_result)

        if second_moment:
            output = SecondMomentClippingOutput(
                grads=ClippedPytree(pytree=result, max_norm=output_max_norm),
                squared_grads=ClippedPytree(
                    pytree=squared_result, max_norm=output_squared_max_norm
                ),
            )
        else:
            output = clipped(result, max_norm=output_max_norm)

        if not return_aux and not return_stats:
            return output

        if return_aux:
            aux_dict = aux if isinstance(aux, dict) else {}
            norms = aux_dict.get("norms")
            group_norms_dict = aux_dict.get("group_norms")
            stats = _compute_clipping_stats(
                norms,
                clipping_norm=current_clipping_norm,
                group_norms_dict=group_norms_dict,
            )

            aux = ClippedFunAux(
                values=aux_dict.get("values"),
                norms=norms,
                clipped_norms=aux_dict.get("clipped_norms"),
                value_aux=aux_dict.get("value_aux"),
                clipping_rate=(
                    stats.clipping_rate
                    if isinstance(stats.clipping_rate, float)
                    or stats.clipping_rate is None
                    else None
                ),
                batch_size=stats.batch_size,
                group_norms=aux_dict.get("group_norms"),
            )
            return output, aux

        if microbatch_size is not None:
            assert stats is not None
            return output, stats

        aux_dict = aux if isinstance(aux, dict) else {}
        return output, _compute_clipping_stats(
            aux_dict.get("norms"),
            clipping_norm=current_clipping_norm,
            group_norms_dict=aux_dict.get("group_norms"),
        )

    # Wrap function to accept and return state
    def stateful_clipped_fn(*args, state, **kwargs):
        runtime_clipping_norm = (
            state.clipping_norm if isinstance(state, _RuntimeClipState) else None
        )
        result = clipped_fn(runtime_clipping_norm, *args, **kwargs)
        return result, state  # State unchanged for fixed clipping

    # Return wrapped function with state
    return stateful_clipped_fn, clip_state


__all__ = ["ClippedFunAux", "ClippingStats", "clipped_fun"]
