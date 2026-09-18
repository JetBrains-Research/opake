"""Callable contracts for extensible Transformers model patch factories."""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import ParamSpec, Protocol, TypeAlias, TypeVar

__all__ = [
    "FamilyPatchFn",
    "ForwardFactory",
    "ForwardFn",
    "ModelPatchFn",
    "ModulePatcher",
]

P = ParamSpec("P")
R = TypeVar("R")


ForwardFn: TypeAlias = Callable[P, R]
"""A model or component ``forward`` implementation."""

ForwardFactory: TypeAlias = Callable[[ForwardFn[P, R]], ForwardFn[P, R]]
"""A callback that replaces a component's bound ``forward`` method."""

ModulePatcher: TypeAlias = Callable[[ModuleType], bool]
"""A callback that applies patches to an imported Transformers module."""


class FamilyPatchFn(Protocol):
    """Function returned by :func:`make_apply_family_patches`."""

    def __call__(
        self,
        *,
        performance: bool = True,
        compat: bool = True,
        kernels: bool | None = None,
        **kwargs: object,
    ) -> None: ...


class ModelPatchFn(Protocol):
    """Function returned by :func:`make_apply_model_patches`."""

    def __call__(
        self,
        model: object | None = None,
        *,
        performance: bool = True,
        compat: bool = True,
        kernels: bool | None = None,
        **kwargs: object,
    ) -> None: ...
