"""Cross-package DP-flow types.

Single canonical home for the data types that connect clipping →
noise → optimizer:

- **Pytree wrappers**: ``ClippedPytree`` (post-clipping), ``NoisedPytree``
  (post-noise), and the paired-stream outputs
  ``SecondMomentClippingOutput`` / ``SecondMomentNoiseOutput``.
- **Per-group container**: ``PerGroup`` — a dict-like that flows through
  the entire pipeline carrying per-parameter-group scalar values.
- **Abstract state bases**: ``ClipState`` and ``NoiseState`` — markers
  shared across DP-SGD and DP-FTRL implementations.
- **Aliases**: ``MaxNorm``, ``NoiseStddev`` — opake-typed unions used
  in wrapper metadata fields.
- **Factories**: ``clipped()`` and ``noised()`` — manual wrapper
  constructors for callers that already produced privatised values.

Concrete state classes live with the factories that produce them:
``FixedClipState`` in :mod:`opake.api.engine.clipping.types`,
``AdaptiveClipState`` in :mod:`opake.dpsgd.clipping._adaptive`,
``GaussianNoiseState`` in :mod:`opake.dpsgd.noise._gaussian`,
``MFNoiseState`` in :mod:`opake.dpftrl.noise._engine`.
"""

from __future__ import annotations

import math
from abc import ABC
from dataclasses import dataclass, replace
from numbers import Real
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import torch

from opake.api.engine.pytree import ParamPath, tree_map
from opake.exceptions import ConfigurationError, InputTypeError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from opake.api.engine.random.types import RngKey

# ===========================================================================
# Pytree alias
# ===========================================================================


# A pytree whose leaves are ``torch.Tensor``.  Restricted to the
# concrete container types the rest of the library actually rebuilds
# and serialises (``dict``, ``list``, ``tuple``); custom ``Mapping`` /
# ``Sequence`` subclasses are intentionally excluded.
#
# Real ``|`` union (not a string) so ``typing.get_type_hints()`` and
# ``typing.get_args()`` see the alias correctly.  Recursion is encoded
# with forward references; static checkers expand the alias to ``Any``,
# which is fine — the alias is documentation-grade typing.
TensorPytree = (
    torch.Tensor
    | dict[str, "TensorPytree"]
    | list["TensorPytree"]
    | tuple["TensorPytree", ...]
)


# ===========================================================================
# Per-group container
# ===========================================================================


@dataclass(frozen=True)
class PerGroup:
    """Per-group values with pre-resolved leaf-path → group assignment.

    Flows through the DP pipeline as clipping norms, output bounds,
    noise multipliers, and noise stddevs.  Supports arithmetic so
    training loop code reads naturally::

        noise_multiplier * clipping_norm  # PerGroup when clipping_norm is PerGroup

    For MSE-optimal per-group noise allocation, prefer
    :meth:`ClippedPytree.noise_stddev_for` (``allocation="optimal"``) instead
    of a plain product.

    Attributes:
        groups: Read-only mapping from :data:`~opake.pytree.ParamPath`
            (optree leaf path) to group name.  Flat ``named_parameters`` keys
            are stored as one-segment paths ``(name,)``; nested trees use
            multi-segment paths such as ``("layer", "weight")``.  A bare
            ``str`` key is accepted at construction and normalized to
            ``(str,)``.
        values: Read-only mapping from group name to the per-group value.

    Checkpointing uses :func:`opake.serialization.state_dict` /
    :func:`opake.serialization.from_state_dict`; the template must use
    the same ``groups`` / ``values`` keys as at save time.
    """

    groups: Mapping[ParamPath | str, str]
    values: Mapping[str, float]

    def __post_init__(self) -> None:
        from opake.api.engine.pytree import param_path

        normalized: dict[ParamPath, str] = {}
        for key, group in self.groups.items():
            normalized[param_path(key)] = group
        object.__setattr__(self, "groups", MappingProxyType(normalized))
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))

    @property
    def effective(self) -> float:
        """``sqrt(sum v²)`` — effective global value for accounting."""
        return math.sqrt(sum(v**2 for v in self.values.values()))

    @property
    def num_groups(self) -> int:
        """Number of independently configured groups."""
        return len(self.values)

    def __rmul__(self, scalar: float) -> PerGroup:
        return PerGroup(self.groups, {k: scalar * v for k, v in self.values.items()})

    def __mul__(self, other: float | PerGroup) -> PerGroup:
        """Multiply by a scalar (broadcast to every group) or by another
        :class:`PerGroup` element-wise.

        Element-wise multiplication requires the same ``groups`` mapping
        and the same set of group names; the result has each group's
        value multiplied with its peer.  This is what makes
        ``clipping_norm * clipping_norm`` produce per-group squared
        sensitivities for the paired second-moment release.
        """
        if isinstance(other, PerGroup):
            if other.groups != self.groups:
                raise ConfigurationError(
                    *(
                        "PerGroup × PerGroup requires identical group mappings; "
                        f"got groups with {len(self.groups)} vs "
                        f"{len(other.groups)} parameter assignments.",
                    )
                )
            if set(other.values) != set(self.values):
                raise ConfigurationError(
                    *(
                        "PerGroup × PerGroup requires identical group sets; "
                        f"got {sorted(self.values)} vs {sorted(other.values)}.",
                    )
                )
            return PerGroup(
                self.groups,
                {k: v * other.values[k] for k, v in self.values.items()},
            )
        return PerGroup(self.groups, {k: v * other for k, v in self.values.items()})

    def __truediv__(self, scalar: float) -> PerGroup:
        return PerGroup(self.groups, {k: v / scalar for k, v in self.values.items()})

    def for_path(self, path: ParamPath | str) -> float:
        """Look up the per-group value for a leaf :data:`~opake.pytree.ParamPath`."""
        from opake.api.engine.pytree import param_path

        return self.values[self.groups[param_path(path)]]


# ===========================================================================
# Abstract state bases (cross-family markers)
# ===========================================================================


class ClipState(ABC):  # noqa: B024 — intentional empty marker base
    """Base class for clipping state.

    The explicit state token returned by clipping transforms.  Fixed
    clipping uses an empty marker; adaptive schemes carry the
    threshold and counters needed for the next step.  Privacy
    calibration metadata lives on the returned :class:`ClippedPytree`,
    not on the state object.
    """


class NoiseState(ABC):
    """Base class for noise state.

    All noise functions (Gaussian and MF) return a state object that
    inherits from this class, providing a unified
    interface for step tracking and RNG key management.

    Attributes:
        _step_counter: Number of ``noise_fn`` calls made.
        _rng_key: Immutable RNG key for deterministic per-step derivation.
    """

    _step_counter: int
    """Number of ``noise_fn`` calls made."""

    _rng_key: RngKey
    """Immutable RNG key for deterministic per-step derivation."""


# ===========================================================================
# DP wrapper types
# ===========================================================================


MaxNorm = Any
"""Type of the ``ClippedPytree.max_norm`` field — ``float`` or ``PerGroup``."""

NoiseStddev = Any
"""Type of the ``NoisedPytree.noise_stddev`` field — ``float | PerGroup | None``."""


def _validate_public_scalar(scalar: Any, *, op: str) -> float:
    if isinstance(scalar, bool) or not isinstance(scalar, Real):
        raise InputTypeError(
            *(
                f"{op} only supports public real-number scalars. "
                "Operate on `.pytree` and reconstruct the clipped value with an "
                "explicit max_norm when the clipped interpretation is unclear.",
            )
        )
    return float(scalar)


def _scale_tensor_leaves(pytree: Any, scalar: float) -> Any:
    return tree_map(
        lambda leaf: leaf * scalar if isinstance(leaf, torch.Tensor) else leaf,
        pytree,
    )


def _apply_tensor_method(pytree: Any, method: str, *args: Any, **kwargs: Any) -> Any:
    def _apply(leaf: Any) -> Any:
        if isinstance(leaf, torch.Tensor):
            return getattr(leaf, method)(*args, **kwargs)
        return leaf

    return tree_map(_apply, pytree)


def _scale_max_norm(max_norm: MaxNorm, factor: float) -> MaxNorm:
    return abs(factor) * max_norm


def _scale_stddev(stddev: NoiseStddev, factor: float) -> NoiseStddev:
    if stddev is None:
        return None
    return abs(factor) * stddev


def _unsupported_message(op: str) -> str:
    return (
        f"ClippedPytree {op} does not preserve DP max_norm semantics "
        "automatically. Operate on `.pytree` and reconstruct the clipped "
        "value with an explicit max_norm."
    )


@dataclass(frozen=True)
class ClippedPytree:
    """A pytree with a public maximum L2 norm on one record's contribution.

    Arithmetic is intentionally narrow.  Public scalar multiplication,
    division, and negation preserve the clipped-query interpretation.
    Other operations should be applied to ``.pytree`` directly, followed
    by explicit reconstruction with the correct ``max_norm``.
    """

    pytree: Any
    max_norm: MaxNorm

    @property
    def sensitivity(self) -> float:
        """Scalar effective L2 sensitivity implied by ``max_norm``."""
        effective = getattr(self.max_norm, "effective", None)
        if effective is not None:
            return float(effective)
        return float(self.max_norm)

    def noise_stddev_for(
        self,
        *,
        noise_multiplier: float,
        allocation: Literal["isotropic", "optimal"] = "optimal",
    ) -> float | PerGroup:
        """Noise standard deviation that ``gaussian(noise_multiplier)`` would apply.

        Named ``noise_stddev_for`` rather than ``noise_stddev`` to leave the
        ``noise_stddev`` slot free for the realised-stddev *field* on the
        :class:`NoisedPytree` subclass — a method on the parent and a field
        on the child cannot share a name.

        Scalar ``max_norm``: returns ``noise_multiplier * max_norm``; the
        ``allocation`` argument is validated but does not change the result.

        ``PerGroup`` ``max_norm``:

        - ``allocation="optimal"`` (default): MSE-optimal Mahalanobis allocation
          ``σᵢ = noise_multiplier · √(Cᵢ · ΣⱼCⱼ)``.  Returns a :class:`PerGroup`
          of per-group standard deviations.
        - ``allocation="isotropic"``: returns the scalar
          ``noise_multiplier * max_norm.effective`` — uniform stddev applied
          to all leaves.

        Privacy accounting under either allocation is ``gaussian(noise_multiplier)``;
        per-group allocation costs nothing in the privacy accountant because the
        Mahalanobis constraint is satisfied with equality.

        Args:
            noise_multiplier: Gaussian noise multiplier.  Must be non-negative.
            allocation: How to spread noise across groups when ``max_norm``
                is :class:`PerGroup`.  Ignored for scalar ``max_norm``.

        Returns:
            Scalar (``float``) or :class:`PerGroup` of standard deviations,
            ready to feed to a Gaussian noise mechanism.

        Raises:
            ValueError: ``noise_multiplier`` negative, ``allocation`` unknown,
                or any per-group bound negative.
        """
        if noise_multiplier < 0:
            raise ConfigurationError(
                *(f"noise_multiplier must be non-negative, got {noise_multiplier}",)
            )
        if allocation not in ("isotropic", "optimal"):
            raise ConfigurationError(
                *(f"allocation must be 'isotropic' or 'optimal', got {allocation!r}.",)
            )
        if isinstance(self.max_norm, PerGroup):
            if allocation == "isotropic":
                return noise_multiplier * self.max_norm.effective
            # Local import: ``opake.api.engine.noise_allocation`` imports these types at
            # module load; importing it at ``types`` import time would cycle.
            from opake.api.engine.noise_allocation import per_group_noise_stddev

            return per_group_noise_stddev(self.max_norm, noise_multiplier)
        return noise_multiplier * float(self.max_norm)

    def _scaled(self, scalar: float) -> ClippedPytree:
        return replace(
            self,
            pytree=_scale_tensor_leaves(self.pytree, scalar),
            max_norm=_scale_max_norm(self.max_norm, scalar),
        )

    def __mul__(self, scalar: Any) -> ClippedPytree:
        return self._scaled(_validate_public_scalar(scalar, op="ClippedPytree *"))

    def __rmul__(self, scalar: Any) -> ClippedPytree:
        return self._scaled(_validate_public_scalar(scalar, op="* ClippedPytree"))

    def __truediv__(self, scalar: Any) -> ClippedPytree:
        factor = _validate_public_scalar(scalar, op="ClippedPytree /")
        if factor == 0.0:
            raise ZeroDivisionError("ClippedPytree division by zero")  # noqa: TRY003 - preserve standard Python error contract
        return self._scaled(1.0 / factor)

    def __rtruediv__(self, scalar: Any) -> ClippedPytree:
        raise InputTypeError(*(_unsupported_message("reverse division"),))

    def __neg__(self) -> ClippedPytree:
        return self._scaled(-1.0)

    def __add__(self, other: Any) -> ClippedPytree:
        raise InputTypeError(*(_unsupported_message("addition"),))

    def __radd__(self, other: Any) -> ClippedPytree:
        raise InputTypeError(*(_unsupported_message("addition"),))

    def __sub__(self, other: Any) -> ClippedPytree:
        raise InputTypeError(*(_unsupported_message("subtraction"),))

    def __rsub__(self, other: Any) -> ClippedPytree:
        raise InputTypeError(*(_unsupported_message("subtraction"),))

    def __pow__(self, exponent: Any) -> ClippedPytree:
        raise InputTypeError(*(_unsupported_message("power"),))

    def clone(self) -> ClippedPytree:
        """Clone tensor leaves while preserving metadata."""
        return replace(self, pytree=_apply_tensor_method(self.pytree, "clone"))

    def detach(self) -> ClippedPytree:
        """Detach tensor leaves while preserving metadata."""
        return replace(self, pytree=_apply_tensor_method(self.pytree, "detach"))

    def to(self, *args: Any, **kwargs: Any) -> ClippedPytree:
        """Call ``Tensor.to`` on tensor leaves while preserving metadata."""
        return replace(
            self,
            pytree=_apply_tensor_method(self.pytree, "to", *args, **kwargs),
        )


@dataclass(frozen=True)
class NoisedPytree(ClippedPytree):
    """A privatised pytree carrying max-norm and realised noise metadata.

    Extends :class:`ClippedPytree` with the per-step ``noise_stddev``
    recorded by the noise mechanism.  ``max_norm`` still describes the
    original record-impact bound, not a bound on the noised output values.
    """

    noise_stddev: NoiseStddev = None

    def _scaled(self, scalar: float) -> NoisedPytree:
        return replace(
            self,
            pytree=_scale_tensor_leaves(self.pytree, scalar),
            max_norm=_scale_max_norm(self.max_norm, scalar),
            noise_stddev=_scale_stddev(self.noise_stddev, scalar),
        )


# ===========================================================================
# Paired-stream outputs (private second moment)
# ===========================================================================


class SecondMomentClippingOutput(NamedTuple):
    """Pre-noise paired-stream input to a noise mechanism.

    Symmetric with :class:`SecondMomentNoiseOutput` but on the
    *pre-noise* side: where the output pairs two ``NoisedPytree``s
    (post-noise), this pairs two ``ClippedPytree``s (pre-noise).
    Each carries its own ``max_norm``.

    Constructed by clipping when the user requests paired-stream
    output (per-example squaring inside the vmap loop).  The presence
    of this type at a noise mechanism's input switches the mechanism
    into paired-stream mode without an explicit constructor flag.

    Attributes:
        grads: Clipped per-example summed gradients (``Σᵢ gᵢ``).
        squared_grads: Clipped per-example summed squared gradients
            (``Σᵢ gᵢ²``).  The squaring happens per-example inside the
            clipping loop so the second-stream sensitivity is ``C²``
            (per record) and the streams are jointly DP-accountable.
    """

    grads: ClippedPytree
    squared_grads: ClippedPytree


class SecondMomentNoiseOutput(NamedTuple):
    """Noise output with private first and second moment streams.

    Attributes:
        noisy_grads: Noised clipped gradients for the optimizer's first
            moment / update direction.
        noisy_squared_grads: Noised element-wise squared clipped gradients
            for optimizers with a second moment accumulator.
    """

    noisy_grads: NoisedPytree
    noisy_squared_grads: NoisedPytree


# ===========================================================================
# Manual wrapper factories
# ===========================================================================


def clipped(pytree: Any, *, max_norm: MaxNorm) -> ClippedPytree:
    """Manually wrap a pytree with public DP max-norm metadata."""
    return ClippedPytree(pytree=pytree, max_norm=max_norm)


def noised(
    pytree: Any,
    *,
    max_norm: MaxNorm,
    noise_stddev: NoiseStddev,
) -> NoisedPytree:
    """Manually wrap an already-privatised pytree with noise metadata."""
    return NoisedPytree(pytree=pytree, max_norm=max_norm, noise_stddev=noise_stddev)


__all__ = [
    "ClipState",
    "ClippedPytree",
    "MaxNorm",
    "NoiseState",
    "NoiseStddev",
    "NoisedPytree",
    "ParamPath",
    "PerGroup",
    "SecondMomentClippingOutput",
    "SecondMomentNoiseOutput",
    "TensorPytree",
    "clipped",
    "noised",
]
