"""Adagrad with optional DP-aware cumulative noise-variance subtraction.

Adagrad follows Duchi, Hazan, and Singer, `Adaptive Subgradient Methods for
Online Learning and Stochastic Optimization
<https://jmlr.org/papers/v12/duchi11a.html>`_ (2011).

Standard Adagrad: per-coordinate cumulative second moment, no decay::

    v_acc[i] += g_t[i]²
    update = g_t / (√v_acc + eps)

DP failure mode.  Under noised gradients ``g̃ = g + ξ`` with
``ξ ~ N(0, σ²)``, ``E[g̃²] = g² + σ²``, so::

    E[v_acc_t[i]] = ∑_{s≤t} g_s²[i]   +   t·σ²

The ``t·σ²`` term grows linearly in step count with no decay.  After
enough steps the denominator is dominated by accumulated noise; the
update direction becomes effectively random and the per-coordinate
LR shrinks indefinitely.  Vanilla Adagrad is **unsafe** for DP
training.

DP correction.  Track a parallel cumulative sum of the per-step noise
variance::

    Φ_acc[i] += σ_t²[i]
    v_acc_corrected = v_acc − Φ_acc   where that is positive
                      v_acc           elsewhere
    update = g_t / (√v_acc_corrected + eps)

Where ``Φ_acc`` has overtaken ``v_acc`` the signal-variance estimate is
non-positive, hence meaningless, and that coordinate reverts to
*uncorrected* Adagrad.  It is deliberately not clamped to a small positive
floor: a floor would divide by ``≈ eps`` and amplify a coordinate that
carries no usable signal by ~1/eps.  The additive ``eps`` in the
denominator is what keeps the step finite.

No bias-correction division is needed (Adagrad doesn't apply
``1/(1−β^t)``); ``v_acc`` and ``Φ_acc`` accumulate at the same rate.
The correction makes the optimizer usable under DP noise — the
denominator now tracks only the signal contribution and the per-
coordinate LR adapts as intended.

This module is mechanism-agnostic: the noise mechanism carries realized
``noise_stddev`` metadata on ``NoisedPytree`` updates; the noise injection
lives elsewhere (``opake.dpsgd.noise``, ``opake.dpftrl.noise``).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from opake.exceptions import ConfigurationError

try:
    from torchopt.base import GradientTransformation
except ImportError as exc:
    raise ImportError(  # noqa: TRY003 - preserve standard Python error contract
        "torchopt is required for opake.optimizers. "
        "Install it with: pip install 'torchopt>=0.7.3'"
    ) from exc

from opake.api.optimizers._bias_correction import (
    init_per_group_phi,
    is_per_group,
    resolve_noise_variance,
)
from opake.api.optimizers._chain import make_optimizer_chain
from opake.pytree import tree_map

if TYPE_CHECKING:
    from opake.pytree import ParamPath
    from opake.types import PerGroup, TensorPytree

_LR = float | Callable[[int], float]


@dataclasses.dataclass(frozen=True)
class AdagradState:
    """State for Adagrad with optional DP correction.

    Attributes:
        v_acc: Per-coordinate cumulative ``∑ g²`` (pytree matching
            params).
        phi_acc: Cumulative noise variance ``∑ σ²`` (scalar, or
            ``dict[ParamPath, float]`` when BC is enabled).  Stays at
            zero unless ``NoisedPytree`` updates supply realized σ
            metadata.
        step: Number of completed updates.
    """

    v_acc: TensorPytree
    phi_acc: float | dict[ParamPath, float]
    step: int


def _scale_by_adagrad(
    eps: float,
    initial_accumulator_value: float,
    noise_bias_correction: bool,
) -> GradientTransformation:
    def init_fn(params: Any) -> AdagradState:
        v_acc = tree_map(
            lambda p: torch.full_like(p, initial_accumulator_value),
            params,
        )
        phi_acc: float | dict = (
            init_per_group_phi(params) if noise_bias_correction else 0.0
        )
        return AdagradState(v_acc=v_acc, phi_acc=phi_acc, step=0)

    def update_fn(
        updates: Any,
        state: AdagradState,
        *,
        params: Any = None,
        inplace: bool = False,
        noise_stddev: float | PerGroup | None = None,
    ) -> tuple[Any, AdagradState]:
        t = state.step + 1

        # Cumulative second moment: v_acc += g².
        new_v = tree_map(lambda v, g: v + g * g, state.v_acc, updates)

        effective = noise_stddev if noise_stddev is not None else 0.0
        if not noise_bias_correction:
            result = tree_map(lambda g, v: g / (v.sqrt() + eps), updates, new_v)
            return result, AdagradState(
                v_acc=new_v,
                phi_acc=state.phi_acc,
                step=t,
            )

        per_group = is_per_group(effective) or isinstance(state.phi_acc, dict)

        if per_group:
            from opake.api.optimizers._bias_correction import map_leaves_with_path

            new_phi: dict = {}

            def _bc_leaf(path, g_node, v_node):
                nv = resolve_noise_variance(effective, path)
                old_phi_k = (
                    state.phi_acc.get(path, 0.0)
                    if isinstance(state.phi_acc, dict)
                    else state.phi_acc
                )
                new_phi_k = old_phi_k + nv
                new_phi[path] = new_phi_k
                if new_phi_k > 0:
                    corrected = v_node - new_phi_k
                    v_corrected = torch.where(corrected > 0, corrected, v_node)
                else:
                    v_corrected = v_node
                return g_node / (v_corrected.sqrt() + eps)

            result = map_leaves_with_path(_bc_leaf, updates, new_v)
        else:
            scalar_var = float(effective) ** 2
            new_phi = float(state.phi_acc) + scalar_var

            if new_phi > 0:

                def _compute(g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                    corrected = v - new_phi
                    v_corrected = torch.where(corrected > 0, corrected, v)
                    return g / (v_corrected.sqrt() + eps)
            else:

                def _compute(g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                    return g / (v.sqrt() + eps)

            result = tree_map(_compute, updates, new_v)

        return result, AdagradState(v_acc=new_v, phi_acc=new_phi, step=t)

    return GradientTransformation(init_fn, update_fn)


def adagrad(
    lr: _LR = 1e-2,
    eps: float = 1e-10,
    weight_decay: float = 0.0,
    initial_accumulator_value: float = 0.0,
    *,
    decoupled_weight_decay: bool = True,
    noise_bias_correction: bool = False,
) -> GradientTransformation:
    """Create an Adagrad optimizer with optional DP-aware correction.

    Args:
        lr: Learning rate, scalar or schedule.
        eps: Denominator stability constant.
        weight_decay: Weight-decay coefficient.
        initial_accumulator_value: Constant added to every leaf of
            ``v_acc`` at init time (matches PyTorch's
            ``initial_accumulator_value``).  Default 0.
        decoupled_weight_decay: ``True`` selects decoupled WD;
            ``False`` folds ``wd·params`` into the gradient.
        noise_bias_correction: If ``True``, subtract a cumulative
            ``Φ_acc`` of the realized noise variance from ``v_acc`` when
            ``NoisedPytree`` updates are passed.  Adagrad does not decay
            its accumulator, so without correction the un-decaying
            ``v_acc`` absorbs ``t·σ²`` of noise variance and learning
            halts; turn this on for any DP run.  Defaults to ``False``
            for consistency with the rest of the package.

    Returns:
        A ``torchopt.base.GradientTransformation``.
    """
    if eps <= 0:
        raise ConfigurationError(*(f"eps must be positive, got {eps}",))
    if weight_decay < 0:
        raise ConfigurationError(
            *(f"weight_decay must be non-negative, got {weight_decay}",)
        )
    if initial_accumulator_value < 0:
        raise ConfigurationError(
            *(
                "initial_accumulator_value must be non-negative, got "
                f"{initial_accumulator_value}",
            )
        )
    moment = _scale_by_adagrad(
        eps=eps,
        initial_accumulator_value=initial_accumulator_value,
        noise_bias_correction=noise_bias_correction,
    )
    return make_optimizer_chain(
        moment,
        lr=lr,
        weight_decay=weight_decay,
        decoupled_weight_decay=decoupled_weight_decay,
        update_rms_clip=None,
    )


__all__ = ["AdagradState", "adagrad"]
