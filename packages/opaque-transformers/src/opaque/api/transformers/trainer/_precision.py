"""Precision helpers for Trainer.

Eval-time full-cast: HF parity for ``bf16_full_eval=True`` casts the model
in place for the eval scope and restores it on exit.  This is distinct
from training-time precision, where ``bf16=True`` enables autocast (no
model cast) — see
:meth:`opaque.transformers.trainer.Trainer._setup_precision`.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ._training_arguments import TrainingArguments


@contextmanager
def eval_dtype(
    model: torch.nn.Module,
    args: TrainingArguments,
    train_dtype: torch.dtype,
) -> Iterator[None]:
    """Cast ``model`` to the eval dtype for the duration of the context.

    On entry, casts the model in place if ``args.bf16_full_eval`` is set
    *and* the model isn't already at the target dtype.  On exit, restores
    the model to ``train_dtype`` (captured at trainer ``__init__``).

    Args:
        model: The model to cast.  Mutated in place (and restored on exit).
        args: Training arguments; reads ``bf16_full_eval``.
        train_dtype: The dtype the model should be in *outside* this
            context — captured by ``Trainer.__init__`` so nested calls
            and the no-op case behave correctly.

    Yields:
        Nothing.  The context manager is purely for the side effect.

    HF-parity note: HF's ``bf16_full_eval`` casts the model rather than
    using autocast (see ``transformers.trainer.py``), and we mirror that.
    """
    if args.bf16_full_eval:
        target = torch.bfloat16
    else:
        yield
        return

    first_param = next(model.parameters(), None)
    if first_param is None or first_param.dtype == target:
        # Parameterless module, or already at the target dtype (e.g.
        # ``bf16=True`` was also set) — nothing to cast.
        yield
        return

    model.to(dtype=target)
    try:
        yield
    finally:
        model.to(dtype=train_dtype)
