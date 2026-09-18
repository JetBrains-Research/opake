"""Functional utilities for PyTorch models.

This module provides helpers for working with PyTorch's functional API,
particularly for converting stateful nn.Module objects to functional form
compatible with torch.func transformations.
"""

import functools
from collections.abc import Callable

import torch
import torch.nn as nn


def make_functional(
    mod: nn.Module,
    disable_autograd_tracking: bool = False,
    partition_trainable: bool = False,
) -> tuple[Callable, tuple[torch.Tensor, ...]] | tuple[Callable, dict, dict]:
    """Convert a PyTorch module to functional form.

    This helper mimics the behavior of the deprecated `functorch.make_functional()`.
    It creates a stateless version of the module that can be called with external
    parameters, enabling use with `torch.func` transformations like `vmap` and `grad`.

    The returned functional model can be used with per-example gradient computation,
    which is essential for differential privacy.

    Args:
        mod: PyTorch module to convert to functional form.
        disable_autograd_tracking: If True, detach parameters from autograd graph.
            Useful when you only need to compute gradients via torch.func.grad,
            not through standard .backward(). Default: False.
        partition_trainable: If True, return (fmodel, trainable_params, frozen_params)
            as dicts instead of (fmodel, params) as tuple. Partitioning is based on
            the `requires_grad` attribute of the original module's parameters.
            This is ideal for LoRA-style fine-tuning. Default: False.

    Returns:
        If partition_trainable=False:
            A tuple containing:
            - fmodel: Functional version of the module. Takes parameters as first argument,
              followed by the module's normal forward arguments.
            - params: Tuple of the module's parameter tensors.

        If partition_trainable=True:
            A tuple containing:
            - fmodel: Functional version of the module. Takes parameters dict as first argument.
            - trainable_params: Dict of parameters where requires_grad=True.
            - frozen_params: Dict of parameters where requires_grad=False.

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> from opake.api.engine.functional_utils import make_functional
        >>>
        >>> # Create a simple model
        >>> model = nn.Linear(10, 1)
        >>>
        >>> # Convert to functional form
        >>> fmodel, params = make_functional(model)
        >>>
        >>> # Use with new parameters
        >>> x = torch.randn(5, 10)
        >>> output = fmodel(params, x)
        >>> print(output.shape)
        torch.Size([5, 1])
        >>>
        >>> # Works with torch.func transformations
        >>> from torch.func import grad, vmap
        >>>
        >>> def loss_fn(params, x, y):
        ...     pred = fmodel(params, x)
        ...     return ((pred - y) ** 2).mean()
        >>>
        >>> # Compute gradient w.r.t. parameters
        >>> x_batch = torch.randn(3, 10)
        >>> y_batch = torch.randn(3, 1)
        >>> grads = grad(loss_fn)(params, x_batch, y_batch)

    Example with per-example gradients:
        >>> # Per-example loss function
        >>> def loss_single(params, x, y):
        ...     pred = fmodel(params, x.unsqueeze(0))
        ...     return ((pred - y) ** 2).mean()
        >>>
        >>> # Compute per-example gradients with vmap
        >>> per_example_grads = vmap(grad(loss_single), in_dims=(None, 0, 0))(
        ...     params, x_batch, y_batch
        ... )
        >>> # per_example_grads is a tuple with shape (batch_size, *param_shape) for each param

    Example with LoRA partitioning:
        >>> import torch.nn as nn
        >>> from opake.dpsgd.clipping import clipped_grad
        >>> from opake.dpsgd.noise import gaussian_noise
        >>> from opake.api.engine.random import key
        >>> from opake.api.engine.functional import make_functional, merge
        >>>
        >>> # Create model and freeze backbone
        >>> model = nn.Sequential(
        ...     nn.Linear(10, 5, bias=False),  # Pretrained backbone
        ...     nn.Linear(5, 1, bias=False),   # Pretrained backbone
        ... )
        >>>
        >>> # Freeze first layer, keep second trainable
        >>> model[0].weight.requires_grad = False  # Frozen
        >>> model[1].weight.requires_grad = True   # Trainable
        >>>
        >>> # Convert with automatic partitioning
        >>> fmodel, trainable, frozen = make_functional(
        ...     model, partition_trainable=True
        ... )
        >>>
        >>> # Training loop
        >>> def loss_fn(train_params, x, y):
        ...     all_params = merge(frozen, train_params)
        ...     pred = fmodel(all_params, x)
        ...     return ((pred - y) ** 2).mean()
        >>>
        >>> grad_fn, clip_state = clipped_grad(
        ...     loss_fn, argnums=0, batch_argnums=(1, 2), clipping_norm=1.0
        ... )
        >>> noise_fn, noise_state = gaussian_noise(noise_multiplier=1.1, key=key(42))
        >>> optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        >>>
        >>> for x_batch, y_batch in dataloader:
        ...     grads, clip_state = grad_fn(trainable, x_batch, y_batch, state=clip_state)
        ...     noisy_grads, noise_state = noise_fn(grads, noise_state)
        ...     # ... assign grads and step optimizer

    Note:
        Gradient checkpointing compatibility is handled by the
        ``reparametrize_recompute`` patch in
        ``opake.api.patches.torch.checkpoint``, which re-binds the
        ``functional_call`` parameters during checkpoint recomputation in
        backward (on a PyTorch with native support the patch is skipped and
        torch handles this itself).  This wrapper is purely functional — it
        delegates entirely to ``torch.func.functional_call``.

    See Also:
        - PyTorch migration guide: https://pytorch.org/docs/master/func.migrating.html
    """
    # Extract parameters with their requires_grad status
    params_dict = dict(mod.named_parameters())

    stateless_mod = mod

    # Optionally detach parameters from autograd graph
    if disable_autograd_tracking:
        params_dict = {name: param.detach() for name, param in params_dict.items()}

    if partition_trainable:
        # Partition based on requires_grad (BEFORE detaching!)
        # We need to check the original parameters
        original_params = dict(mod.named_parameters())
        trainable_params = {
            name: params_dict[name]
            for name, param in original_params.items()
            if param.requires_grad
        }
        frozen_params = {
            name: params_dict[name]
            for name, param in original_params.items()
            if not param.requires_grad
        }

        def fmodel_dict(params_dict_input, *args, **kwargs):
            return torch.func.functional_call(
                stateless_mod, params_dict_input, args, kwargs
            )

        return fmodel_dict, trainable_params, frozen_params

    else:
        # Original behavior: return tuple
        params_names = list(params_dict.keys())
        params_values = tuple(params_dict.values())

        def fmodel_tuple(new_params_values, *args, **kwargs):
            new_params_dict = dict(zip(params_names, new_params_values, strict=True))
            return torch.func.functional_call(
                stateless_mod, new_params_dict, args, kwargs
            )

        return fmodel_tuple, params_values


def _squeeze_output(output):
    """Squeeze the leading batch dim from model output tensors.

    Handles torch.Tensor, dict-like (HF ModelOutput), tuple, and namedtuple.
    Scalars (0-dim tensors) are left untouched.
    """
    if isinstance(output, torch.Tensor):
        return output.squeeze(0) if output.ndim > 0 else output

    # Dict-like (HF ModelOutput): squeeze top-level tensor values in-place
    if hasattr(output, "__setitem__") and hasattr(output, "items"):
        for k, v in output.items():
            if isinstance(v, torch.Tensor) and v.ndim > 0:
                output[k] = v.squeeze(0)
        return output

    # Namedtuple / tuple
    if isinstance(output, tuple):
        squeezed = tuple(
            v.squeeze(0) if isinstance(v, torch.Tensor) and v.ndim > 0 else v
            for v in output
        )
        # Preserve namedtuple type
        if hasattr(output, "_fields"):
            return type(output)(*squeezed)
        return squeezed

    return output


def with_batch_dim(
    fn: Callable,
    batch_argnums: int | tuple[int, ...] = (),
    batch_kwargs: tuple[str, ...] | dict[str, int] = (),
    *,
    min_ndim: int | None = None,
) -> Callable:
    """Wrap a function to add a leading batch dimension to specified arguments.

    Use this when a model expects a batch dimension but the function is called
    per-example under ``vmap`` where the batch dim has been stripped.

    Two modes controlled by ``min_ndim``:

    - ``min_ndim=None`` (default): always unsqueeze specified args, never
      touch output. This is the user-facing mode for wrapping loss functions.

    - ``min_ndim=N`` (idempotent mode): only unsqueeze tensors with
      ``ndim < threshold``, and squeeze top-level output tensors when any
      unsqueezing happened. This is the model-forward mode — a no-op for
      already-batched inputs.

    Args:
        fn: Function to wrap.
        batch_argnums: Which positional arguments should get an extra leading
            dimension via ``unsqueeze(0)``. Int or tuple of ints.
        batch_kwargs: Keyword arguments to process. A tuple of names uses the
            global ``min_ndim`` threshold. A dict maps each kwarg name to its
            own ndim threshold (e.g., ``{"input_ids": 2, "inputs_embeds": 3}``).
        min_ndim: Global ndim threshold. ``None`` means always unsqueeze.
            An int means only unsqueeze when ``tensor.ndim < min_ndim``.

    Returns:
        Wrapped function.

    Example — loss function mode (always unsqueeze)::

        >>> def loss_fn(params, tokens):
        ...     output = model(params, tokens)  # model expects (batch, seq)
        ...     return output.loss
        >>> # Under vmap, tokens is (seq,); wrap so model sees (1, seq)
        >>> loss_fn = with_batch_dim(loss_fn, batch_argnums=1)

    Example — model forward mode (idempotent)::

        >>> # Only unsqueeze when inputs lack batch dim; squeeze output
        >>> model.forward = with_batch_dim(
        ...     model.forward,
        ...     batch_kwargs={"input_ids": 2, "inputs_embeds": 3},
        ...     min_ndim=2,
        ... )
    """
    if getattr(fn, "_opake_batchified", False):
        return fn

    import inspect

    from opake.api.engine.pytree import tree_map

    if isinstance(batch_argnums, int):
        batch_argnums = (batch_argnums,)

    # Normalize batch_kwargs: tuple → dict using global min_ndim
    if isinstance(batch_kwargs, tuple):
        batch_kwargs_dict: dict[str, int | None] = dict.fromkeys(batch_kwargs, min_ndim)
    else:
        batch_kwargs_dict = dict(batch_kwargs)

    conditional = min_ndim is not None

    # Pre-compute: positional indices of batch_kwargs parameters in fn's
    # signature, so we can catch them even when passed positionally.
    _positional_kwarg_indices: dict[int, str] = {}
    if batch_kwargs_dict:
        try:
            sig = inspect.signature(fn)
            for idx, (name, param) in enumerate(sig.parameters.items()):
                if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
                    if name in batch_kwargs_dict:
                        _positional_kwarg_indices[idx] = name
                elif param.kind == param.VAR_POSITIONAL:
                    break
        except (ValueError, TypeError):
            pass

    def _needs_unsqueeze(tensor: torch.Tensor, threshold: int | None) -> bool:
        if threshold is None:
            return True
        return tensor.ndim < threshold

    # ``functools.wraps`` propagates ``fn.__wrapped__`` (and sets it to
    # ``fn`` itself) so callers can walk the chain back to the original
    # function via ``inspect.signature(wrapper, follow_wrapped=True)`` — the
    # default behavior of :func:`inspect.signature`.  Concretely: when this
    # wraps a transformers ``forward(input_ids=None, attention_mask=None,
    # …)``, ``inspect.signature`` resolves to the real named parameters
    # instead of just ``(*args, **kwargs)``.  HF-style column pruning that
    # introspects ``model.forward`` (and any other downstream consumer of
    # the signature) keeps working after batchifying.
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        unsqueezed = False
        args_list = list(args)

        # Move batch_kwargs that were passed positionally into kwargs.
        for idx in sorted(_positional_kwarg_indices, reverse=True):
            if idx < len(args_list):
                name = _positional_kwarg_indices[idx]
                if name not in kwargs:
                    kwargs[name] = args_list.pop(idx)

        # Process positional args
        for i in batch_argnums:

            def _unsqueeze_arg(x, _threshold=min_ndim):
                nonlocal unsqueezed
                if isinstance(x, torch.Tensor) and _needs_unsqueeze(x, _threshold):
                    unsqueezed = True
                    return x.unsqueeze(0)
                return x

            if i < len(args_list):
                args_list[i] = tree_map(_unsqueeze_arg, args_list[i])

        # Process keyword args
        for name, threshold in batch_kwargs_dict.items():
            if name not in kwargs or kwargs[name] is None:
                continue
            val = kwargs[name]
            if isinstance(val, torch.Tensor) and _needs_unsqueeze(val, threshold):
                unsqueezed = True
                kwargs[name] = val.unsqueeze(0)

        result = fn(*args_list, **kwargs)

        if conditional and unsqueezed:
            result = _squeeze_output(result)

        return result

    wrapper._opake_batchified = True
    return wrapper


from opake.api.engine.functional._collate import empty_collate  # noqa: E402

__all__ = ["empty_collate", "make_functional", "with_batch_dim"]
