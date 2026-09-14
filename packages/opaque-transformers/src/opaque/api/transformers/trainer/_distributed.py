"""DDP runtime support for :class:`opaque.transformers.trainer.Trainer`.

Owns the rank/world resolution, the per-process gating for I/O sites
(logging, saving, hub push), and the small collection of cross-rank
collectives the trainer's training loop needs.

The trainer never calls ``torch.distributed.init_process_group``; the
launcher (``torchrun`` or test-side ``mp.spawn``) owns that. We just read
``LOCAL_RANK`` / ``RANK`` / ``WORLD_SIZE`` from the environment for the
``__init__`` device pick, and use :mod:`opaque.distributed` once the
process group is up.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any

import torch

from opaque.distributed import get_rank, get_world_size, is_distributed
from opaque.distributed.collectives import all_reduce_ as _opaque_all_reduce_
from opaque.distributed.collectives import barrier as _opaque_barrier
from opaque.exceptions import ConfigurationError, OperationError
from transformers.utils import logging as _hf_logging

__all__ = [
    "DDPState",
    "apply_logging",
    "barrier",
    "checkpoint_barrier",
    "resolve_ddp_state",
    "should_log",
    "should_save",
    "validate_ddp_backend",
]

_BACKEND_ENV_DEPENDENT_HINTS = {
    "xccl": "Intel XPU/XCCL runtime",
    "hccl": "Habana Gaudi/HCCL runtime",
    "cncl": "Cambricon MLU/CNCL runtime",
    "mccl": "Moore Threads MUSA/MCCL runtime",
}

# HF trainer_log_levels parity: level-name -> int, with "passive" = -1
# ("leave the current verbosity unchanged").
_TRAINER_LOG_LEVELS = {**_hf_logging.get_log_levels_dict(), "passive": -1}


@dataclasses.dataclass(frozen=True)
class DDPState:
    """Snapshot of the rank/world topology for one Trainer instance.

    ``is_distributed`` reflects ``torch.distributed.is_initialized()`` at
    construction time. When it's ``False``, the trainer behaves
    identically to the pre-DDP single-process path.

    The ``device`` field is the device this rank's model parameters live
    on. For NCCL/CUDA, that's ``cuda:{local_rank}``; for the
    single-process / CPU path it's whatever
    :class:`TrainingArguments._setup_devices` resolved.
    """

    is_distributed: bool
    rank: int
    local_rank: int
    world_size: int
    backend: str | None
    device: torch.device

    @property
    def is_world_zero(self) -> bool:
        return self.rank == 0

    @property
    def is_local_zero(self) -> bool:
        return self.local_rank == 0


def resolve_ddp_state(device: torch.device, args: Any | None = None) -> DDPState:
    """Build a :class:`DDPState` for the current process.

    If ``torch.distributed`` has been initialised, the rank/world fields
    come from the live process group. Otherwise:
    - When ``WORLD_SIZE > 1`` and ``args`` is provided, auto-initialise
      ``torch.distributed`` using ``args.ddp_backend`` (defaulting to
      ``nccl`` on CUDA, ``gloo`` elsewhere) and ``args.ddp_timeout``.
      This mirrors HF Trainer's eager init via Accelerate, without us
      depending on Accelerate.
    - Otherwise (single-process), fall back to env vars so a 1-rank
      ``torchrun`` smoke test still reports ``world_size==1``.
    """
    if not is_distributed() and args is not None:
        env_world = int(os.environ.get("WORLD_SIZE", "1"))
        if env_world > 1:
            backend = args.ddp_backend or (
                "nccl" if torch.cuda.is_available() else "gloo"
            )
            timeout_seconds = int(args.ddp_timeout)
            import logging as _logging
            from datetime import timedelta as _td

            torch.distributed.init_process_group(
                backend=backend, timeout=_td(seconds=timeout_seconds)
            )
            _logging.getLogger(__name__).info(
                "Auto-initialised torch.distributed (backend=%s, world_size=%d, rank=%d)",
                backend,
                torch.distributed.get_world_size(),
                torch.distributed.get_rank(),
            )

    if is_distributed():
        rank = get_rank()
        world_size = get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        backend = torch.distributed.get_backend()
        return DDPState(
            is_distributed=True,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            backend=backend,
            device=device,
        )
    # No process group — single-process. Honour LOCAL_RANK if set so a
    # 1-rank torchrun reports the same shape as 1-rank no-launcher.
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    return DDPState(
        is_distributed=False,
        rank=0,
        local_rank=max(local_rank, 0),
        world_size=1,
        backend=None,
        device=device,
    )


def validate_ddp_backend(args: Any, ddp: DDPState) -> None:
    """Validate configured backend against active process-group backend.

    ``args.ddp_backend`` mirrors HF surface values.  If a process group is
    already initialized, the configured value (when set) must match the live
    backend to avoid silent misconfiguration.
    """
    configured = args.ddp_backend
    if configured is None:
        return
    configured_backend = str(configured).lower()
    if not ddp.is_distributed:
        if configured_backend in _BACKEND_ENV_DEPENDENT_HINTS:
            raise ConfigurationError(
                *(
                    f"ddp_backend={configured_backend!r} requires a distributed process "
                    f"group initialized with vendor runtime "
                    f"({_BACKEND_ENV_DEPENDENT_HINTS[configured_backend]}), but no "
                    "process group is currently initialized.",
                )
            )
        return
    live_backend = (ddp.backend or "").lower()
    if live_backend and configured_backend != live_backend:
        raise ConfigurationError(
            *(
                "Configured ddp_backend does not match initialized process group: "
                f"ddp_backend={configured_backend!r}, live_backend={live_backend!r}.",
            )
        )
    # (Once the group is initialized ``get_backend()`` is always non-empty, so
    # the mismatch check above already covers env-dependent backends — no
    # additional empty-``live_backend`` branch is reachable here.)


def should_log(args: Any, ddp: DDPState) -> bool:
    """Return ``True`` if this rank should emit log payloads.

    Mirrors HF's ``TrainingArguments.should_log`` (training_args.py:1992-2002):
    if ``log_on_each_node`` is set the gate fires on each node's local zero
    (typical multi-node default); otherwise only world-rank-0 logs.
    """
    if not ddp.is_distributed:
        return True
    if args.log_on_each_node:
        return ddp.is_local_zero
    return ddp.is_world_zero


def apply_logging(args: Any, ddp: DDPState) -> int:
    """Set this rank's logging verbosity (HF parity) and return the raw level.

    Mirrors HF's ``TrainingArguments.get_process_log_level``: the main process
    uses ``args.log_level`` and replicas use ``args.log_level_replica`` (the
    main/replica split reuses :func:`should_log`). ``"passive"`` (-1) leaves the
    current verbosity untouched. Concrete levels are applied to both the
    transformers-namespace logger and opaque's root logger, so
    ``log_level_replica`` actually quiets replica emissions from either.
    """
    raw = (
        _TRAINER_LOG_LEVELS[args.log_level]
        if should_log(args, ddp)
        else _TRAINER_LOG_LEVELS[args.log_level_replica]
    )
    if raw != -1:
        _hf_logging.set_verbosity(raw)
        logging.getLogger("opaque").setLevel(raw)
    return raw


def should_save(args: Any, ddp: DDPState) -> bool:
    """Return ``True`` if this rank should write checkpoint / artefact files.

    Mirrors HF's ``TrainingArguments.should_save`` (training_args.py:2009-2015).
    Default ``save_on_each_node=False`` => only world-rank-0 writes; under
    ``save_on_each_node=True`` each node's local zero writes.
    """
    if not ddp.is_distributed:
        return True
    if args.save_on_each_node:
        return ddp.is_local_zero
    return ddp.is_world_zero


def barrier(ddp: DDPState) -> None:
    """Block until every rank reaches this point.

    No-op outside a process group.
    """
    if ddp.is_distributed:
        _opaque_barrier()


def checkpoint_barrier(ddp: DDPState, local_error: Exception | None) -> None:
    """Synchronise one checkpoint-save stage, propagating failure to all ranks.

    Every rank calls this once per stage, passing the exception it caught in
    that stage (or ``None`` on success). If any rank failed, every rank
    raises: the rank whose own segment failed re-raises ``local_error``,
    every other rank raises :class:`OperationError`. Replaces plain
    ``barrier()`` calls that let a saving-rank failure leave peers blocked
    forever.
    """
    if not ddp.is_distributed:
        if local_error is not None:
            raise local_error
        return
    failed = torch.tensor(
        [1 if local_error is not None else 0], dtype=torch.int64, device=ddp.device
    )
    _opaque_all_reduce_(failed, op="max")
    if local_error is not None:
        raise local_error
    if int(failed.item()) > 0:
        raise OperationError(
            *(
                "Checkpoint save failed on another rank; aborting on rank "
                f"{ddp.rank} instead of waiting at the checkpoint barrier.",
            )
        )
