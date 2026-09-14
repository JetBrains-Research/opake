"""Checkpoint helpers for Trainer.

Defines the on-disk layout (parallel to HuggingFace ``Trainer``), discovery and
rotation utilities, an RNG snapshot helper, and the DP-side runtime bundle
stored under ``dp_state.pt``.  Clip / noise slices use
:func:`opaque.serialization.state_dict`; resume merges them with the live
training context via :func:`opaque.serialization.from_state_dict`.
"""

from __future__ import annotations

import dataclasses
import logging
import random
import re
import shutil
from dataclasses import field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from opaque.exceptions import CheckpointError
from opaque.serialization import state_dict as opaque_state_dict
from opaque.types import ClipState, NoiseState
from transformers.trainer import TRAINER_STATE_NAME
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME

log = logging.getLogger(__name__)

# Filename layout: ``training_args.bin`` matches HF's ``TRAINING_ARGS_NAME``.
TRAINING_ARGS_NAME = "training_args.bin"
DP_OPTIMIZER_NAME = "dp_optimizer.pt"
DP_STATE_NAME = "dp_state.pt"
DP_ACCOUNTANT_NAME = "accountant.json"
RNG_STATE_NAME = "rng_state.pth"

# Version 8 requires the current DP runtime state format.
DP_STATE_BUNDLE_VERSION = 8

_CHECKPOINT_RE = re.compile(rf"^{re.escape(PREFIX_CHECKPOINT_DIR)}\-(\d+)$")

__all__ = [
    "DP_ACCOUNTANT_NAME",
    "DP_OPTIMIZER_NAME",
    "DP_STATE_BUNDLE_VERSION",
    "DP_STATE_NAME",
    "PREFIX_CHECKPOINT_DIR",
    "RNG_STATE_NAME",
    "SAFE_WEIGHTS_NAME",
    "TRAINER_STATE_NAME",
    "TRAINING_ARGS_NAME",
    "WEIGHTS_NAME",
    "RuntimeCheckpoint",
    "get_last_checkpoint",
    "list_checkpoints",
    "load_dp_runtime_state",
    "parse_checkpoint_step",
    "restore_rng_state",
    "rng_state_path",
    "rotate_checkpoints",
    "save_dp_runtime_state",
    "snapshot_rng_state",
]


def rng_state_path(ckpt_dir: str, *, rank: int = 0, world_size: int = 1) -> str:
    """Resolve the RNG-snapshot path for ``rank`` in a ``world_size``-process run."""
    if world_size <= 1:
        return str(Path(ckpt_dir) / RNG_STATE_NAME)
    return str(Path(ckpt_dir) / f"rng_state_{int(rank)}.pth")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def parse_checkpoint_step(path: str) -> int | None:
    """Extract the step number from a ``checkpoint-N`` directory name."""
    m = _CHECKPOINT_RE.match(Path(path.rstrip("/\\")).name)
    return int(m.group(1)) if m is not None else None


def list_checkpoints(folder: str) -> list[str]:
    """Return ``checkpoint-N`` subdirectories of ``folder`` sorted by step ascending."""
    folder_path = Path(folder)
    if not folder_path.is_dir():
        return []
    found: list[tuple[int, str]] = []
    for child in folder_path.iterdir():
        if not child.is_dir():
            continue
        step = parse_checkpoint_step(child.name)
        if step is not None:
            found.append((step, str(child)))
    return [path for _, path in sorted(found)]


def get_last_checkpoint(folder: str) -> str | None:
    """Return the ``checkpoint-N`` with the highest step, or ``None``."""
    paths = list_checkpoints(folder)
    return paths[-1] if paths else None


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------


def rotate_checkpoints(
    output_dir: str,
    save_total_limit: int | None,
    best_model_checkpoint: str | None = None,
) -> None:
    """Delete oldest checkpoints to honor ``save_total_limit``.

    Always protects the most recent checkpoint and the best-model checkpoint
    (when supplied). Effective keep count is
    ``max(save_total_limit, len(protected))`` to avoid deleting either.
    """
    if save_total_limit is None or save_total_limit <= 0:
        return

    checkpoints = list_checkpoints(output_dir)
    if len(checkpoints) <= save_total_limit:
        return

    protected: set[str] = {checkpoints[-1]}
    if best_model_checkpoint is not None:
        best_abs = Path(best_model_checkpoint).resolve()
        for path in checkpoints:
            if Path(path).resolve() == best_abs:
                protected.add(path)
                break

    keep = max(save_total_limit, len(protected))
    num_to_delete = max(0, len(checkpoints) - keep)
    deleted = 0
    for path in checkpoints:
        if deleted >= num_to_delete:
            break
        if path in protected:
            continue
        log.info(
            "Deleting older checkpoint %s due to save_total_limit=%d",
            path,
            save_total_limit,
        )
        shutil.rmtree(path, ignore_errors=True)
        deleted += 1


# ---------------------------------------------------------------------------
# RNG snapshot
# ---------------------------------------------------------------------------


def snapshot_rng_state() -> dict[str, Any]:
    """Capture python / numpy / torch CPU + per-CUDA-device + MPS RNG states."""
    snap: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        snap["cuda"] = torch.cuda.random.get_rng_state_all()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        snap["mps"] = torch.mps.get_rng_state()
    return snap


def restore_rng_state(snap: dict[str, Any]) -> None:
    """Apply a previously captured RNG snapshot."""
    random.setstate(snap["python"])
    np.random.set_state(snap["numpy"])
    torch.set_rng_state(snap["cpu"])
    cuda_states = snap.get("cuda")
    if cuda_states is not None and torch.cuda.is_available():
        for i, st in enumerate(cuda_states):
            if i < torch.cuda.device_count():
                torch.cuda.set_rng_state(st, i)
    mps_state = snap.get("mps")
    if (
        mps_state is not None
        and hasattr(torch, "mps")
        and torch.backends.mps.is_available()
    ):
        torch.mps.set_rng_state(mps_state)


# ---------------------------------------------------------------------------
# DP runtime bundle (clip_state + noise_state + scheduling + sampler state)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RuntimeCheckpoint:
    """Typed payload of ``dp_state.pt`` — the DP-side runtime bundle.

    Replaces the prior ad-hoc ``dict[str, Any]`` so the resume code is
    type-driven (attribute access, ``dataclasses.fields(...)`` iteration)
    instead of string-keyed.  Adding a new field is a single edit on
    this dataclass; ``_apply_runtime_state`` and ``_warn_on_arg_drift``
    in the trainer pick it up automatically.

    Each field tagged ``compare_on_resume=True`` carries a ``drift``
    disposition that controls what happens when phase-2's value differs
    from phase-1's saved value:

    - ``"dp_relevant"`` — affects privacy accounting. Independent-step
      DP-SGD warns; whole-horizon processes (including Gaussian k-out-of-t)
      and DP-FTRL raise because accounting uses the original schedule.
    - ``"shape"`` — affects training trajectory (LR schedule, etc.) but
      not privacy.  Warns.
    - ``"intentional_extend"`` — silently allowed.  Used for ``total_steps``
      in independent-step DP-SGD, where extending training is allowed.
    - dict form, e.g. ``{"gaussian": "intentional_extend", "default":
      "dp_relevant"}`` — per-mechanism override, resolved by looking up
      the saved ``mechanism_kind`` (the ``"default"`` key catches
      anything not listed; ``"gaussian"`` is the DP-SGD path).
    """

    version: int
    clip_state: dict[str, Any]
    noise_state: dict[str, Any]
    sampler_state: dict[str, Any] | None

    # --- DP-accounting scalars (privacy-relevant) ---------------------
    sample_rate: float = field(
        metadata={"compare_on_resume": True, "drift": "dp_relevant"}
    )
    target_delta: float = field(
        metadata={"compare_on_resume": True, "drift": "dp_relevant"}
    )
    noise_multiplier: float = field(
        metadata={"compare_on_resume": True, "drift": "dp_relevant"}
    )
    expected_steps_per_epoch: int = field(
        metadata={"compare_on_resume": True, "drift": "dp_relevant"}
    )
    expected_batch_size: int = field(
        metadata={"compare_on_resume": True, "drift": "dp_relevant"}
    )
    # ``total_steps`` differs by mechanism: DP-SGD users extend training
    # routinely (RDP composes step-by-step); DP-FTRL builds an MF
    # strategy for a specific T, so extending means a different strategy
    # and a different ε.
    total_steps: int = field(
        metadata={
            "compare_on_resume": True,
            "drift": {
                "gaussian": "intentional_extend",
                "default": "dp_relevant",
            },
        }
    )

    # --- DP-FTRL provenance + MF strategy params ---------------------
    mechanism_kind: str = field(
        default="gaussian",
        metadata={"compare_on_resume": True, "drift": "dp_relevant"},
    )
    is_horizon_process: bool = field(
        default=False,
        metadata={"compare_on_resume": True, "drift": "dp_relevant"},
    )
    calibration_source: str = "fixed"
    target_epsilon: float | None = None
    horizon_process_state: dict[str, Any] | None = field(
        default=None,
        metadata={"compare_on_resume": True, "drift": "dp_relevant"},
    )
    mf_n_steps: int | None = field(
        default=None,
        metadata={"compare_on_resume": True, "drift": "dp_relevant"},
    )
    mf_min_sep: int | None = field(
        default=None,
        metadata={"compare_on_resume": True, "drift": "dp_relevant"},
    )
    mf_max_participations: int | None = field(
        default=None,
        metadata={"compare_on_resume": True, "drift": "dp_relevant"},
    )

    # --- LR-schedule shape (trajectory-relevant, privacy-neutral) ---
    lr_scheduler: str | None = field(
        default=None,
        metadata={"compare_on_resume": True, "drift": "shape"},
    )
    learning_rate: float | None = field(
        default=None,
        metadata={"compare_on_resume": True, "drift": "shape"},
    )
    warmup_steps: int | float | None = field(
        default=None,
        metadata={"compare_on_resume": True, "drift": "shape"},
    )
    lr_scheduler_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={"compare_on_resume": True, "drift": "shape"},
    )


def save_dp_runtime_state(  # noqa: PLR0913
    path: str,
    *,
    clip_state: Any,
    noise_state: Any,
    sampler_state: dict[str, Any] | None,
    sample_rate: float,
    target_delta: float,
    noise_multiplier: float,
    expected_steps_per_epoch: int,
    expected_batch_size: int,
    total_steps: int,
    mechanism_kind: str = "gaussian",
    is_horizon_process: bool = False,
    calibration_source: str = "fixed",
    target_epsilon: float | None = None,
    horizon_process_state: dict[str, Any] | None = None,
    mf_n_steps: int | None = None,
    mf_min_sep: int | None = None,
    mf_max_participations: int | None = None,
    lr_scheduler: str | None = None,
    learning_rate: float | None = None,
    warmup_steps: int | float | None = None,
    lr_scheduler_kwargs: dict[str, Any] | None = None,
) -> None:
    """Save the DP runtime bundle as a :class:`RuntimeCheckpoint`."""
    if not isinstance(clip_state, ClipState):
        raise CheckpointError(
            *(
                f"clip_state must be a ClipState instance, got {type(clip_state).__name__}",
            )
        )
    if not isinstance(noise_state, NoiseState):
        raise CheckpointError(
            *(
                f"noise_state must be a NoiseState instance, got {type(noise_state).__name__}",
            )
        )
    bundle = RuntimeCheckpoint(
        version=DP_STATE_BUNDLE_VERSION,
        clip_state=opaque_state_dict(clip_state),
        noise_state=opaque_state_dict(noise_state),
        sampler_state=sampler_state,
        sample_rate=float(sample_rate),
        target_delta=float(target_delta),
        noise_multiplier=float(noise_multiplier),
        expected_steps_per_epoch=int(expected_steps_per_epoch),
        expected_batch_size=int(expected_batch_size),
        total_steps=int(total_steps),
        mechanism_kind=str(mechanism_kind),
        is_horizon_process=bool(is_horizon_process),
        calibration_source=str(calibration_source),
        target_epsilon=(float(target_epsilon) if target_epsilon is not None else None),
        horizon_process_state=horizon_process_state,
        mf_n_steps=int(mf_n_steps) if mf_n_steps is not None else None,
        mf_min_sep=int(mf_min_sep) if mf_min_sep is not None else None,
        mf_max_participations=(
            int(mf_max_participations) if mf_max_participations is not None else None
        ),
        lr_scheduler=lr_scheduler,
        learning_rate=(float(learning_rate) if learning_rate is not None else None),
        warmup_steps=(float(warmup_steps) if warmup_steps is not None else None),
        lr_scheduler_kwargs=lr_scheduler_kwargs,
    )
    # ``torch.save`` of a dataclass round-trips via pickle.  Kept as
    # pickle to handle the heterogeneous types (tensors inside
    # ``clip_state`` / ``noise_state``, Python objects in
    # ``sampler_state``).  A future migration to safetensors+JSON
    # sidecar would eliminate the ``weights_only=False`` requirement.
    torch.save(bundle, path)


def load_dp_runtime_state(path: str) -> RuntimeCheckpoint:
    """Load the DP runtime bundle from disk as a typed :class:`RuntimeCheckpoint`.

    ``clip_state`` and ``noise_state`` are flat serialisation dicts; merge them
    into live objects with :func:`opaque.serialization.from_state_dict` using
    the templates produced by the current training setup.

    ``weights_only=False`` is required for PyTorch 2.6+ defaults when tensors
    or dataclasses appear in the bundle.
    """
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(bundle, RuntimeCheckpoint):
        raise CheckpointError(
            *(
                f"dp_state.pt at {path} did not deserialize to RuntimeCheckpoint "
                f"(got {type(bundle).__name__}); checkpoint may be from an older "
                "trainer version.",
            )
        )
    if bundle.version != DP_STATE_BUNDLE_VERSION:
        raise CheckpointError(
            *(
                f"unsupported dp_state bundle version {bundle.version} "
                f"(expected {DP_STATE_BUNDLE_VERSION})",
            )
        )
    return bundle
