"""Evaluation helpers for :class:`Trainer`.

This module hosts the Opaque-owned eval container plus pure helpers used
by the eval loop:

- :class:`EvaluationResult` — single dataclass returned by
  :meth:`Trainer.evaluate`, :meth:`Trainer.predict`, and
  :meth:`Trainer.evaluation_loop`.  Replaces HF's split
  ``EvalLoopOutput`` / ``PredictionOutput`` pair.
- :class:`_PredictionAccumulator` — collects per-batch losses, predictions,
  labels, and (optionally) inputs across an eval loop, with bounded device and
  pinned staging plus balanced CPU chunk trees that avoid prefix rebuilding.
- :func:`with_metric_prefix` — adds ``{prefix}_`` to keys that don't already
  start with it (HF parity).
- :func:`speed_metrics` — pure helper mirroring HF's
  ``transformers.trainer_utils.speed_metrics`` so eval/predict reports
  expose ``{prefix}_runtime``, ``{prefix}_samples_per_second``,
  ``{prefix}_steps_per_second``.

``EvalPrediction`` is re-exported from ``transformers.trainer_utils`` as the
canonical input shape for user-supplied ``compute_metrics`` callbacks.
"""

from __future__ import annotations

import dataclasses
import time
from collections import deque
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from typing import Any

import torch
from torch import Tensor

from transformers.trainer_pt_utils import (
    find_batch_size,
    nested_detach,
    nested_numpify,
    nested_truncate,
)
from transformers.trainer_utils import (
    EvalPrediction,
    denumpify_detensorize,
)

from .types import EvaluationResult  # re-export; canonical home is types.py

__all__ = [
    "EvalPrediction",
    "EvaluationResult",
    "_PredictionAccumulator",
    "denumpify_detensorize",
    "find_batch_size",
    "nested_numpify",
    "nested_truncate",
    "resolve_eval_num_samples",
    "speed_metrics",
    "with_metric_prefix",
]


# ``-100`` is HF's universal padding sentinel for eval tensors — labels,
# logits, and inputs alike.  ``compute_metrics`` users mask this value to
# skip ignored positions; padding logits with ``0`` would silently leak
# ignored positions into accuracy / perplexity computations.
_HF_PAD_VALUE = -100


# ---------------------------------------------------------------------------
# Evaluation transfer and timing
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _EvaluationTelemetry:
    """Accumulate evaluation phase timings without serializing CUDA batches."""

    device: torch.device
    model_time_sec: float = 0.0
    gather_time_sec: float = 0.0
    transfer_time_sec: float = 0.0
    finalization_time_sec: float = 0.0
    metric_time_sec: float = 0.0
    transfer_bytes: int = 0
    transfer_overlap_sec: float = 0.0
    _origin: Any | None = dataclasses.field(default=None, init=False, repr=False)
    _model_events: list[tuple[Any, Any]] = dataclasses.field(
        default_factory=list, init=False, repr=False
    )
    _transfer_events: list[tuple[Any, Any]] = dataclasses.field(
        default_factory=list, init=False, repr=False
    )
    _finished: bool = dataclasses.field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        if self.device.type == "cuda":
            self._origin = torch.cuda.Event(enable_timing=True)
            self._origin.record(torch.cuda.current_stream(self.device))

    @contextmanager
    def model(self) -> Iterator[None]:
        if self.device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            stream = torch.cuda.current_stream(self.device)
            start.record(stream)
            yield
            end.record(stream)
            self._model_events.append((start, end))
            return

        if self.device.type == "mps":
            torch.mps.synchronize()
        started = time.perf_counter()
        yield
        if self.device.type == "mps":
            torch.mps.synchronize()
        self.model_time_sec += time.perf_counter() - started

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        yield
        setattr(
            self,
            f"{name}_time_sec",
            getattr(self, f"{name}_time_sec") + time.perf_counter() - started,
        )

    def add_transfer(
        self,
        *,
        num_bytes: int,
        events: tuple[Any, Any] | None = None,
        elapsed_sec: float = 0.0,
    ) -> None:
        self.transfer_bytes += num_bytes
        self.transfer_time_sec += elapsed_sec
        if events is not None:
            self._transfer_events.append(events)

    def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        if self.device.type != "cuda":
            return

        torch.cuda.synchronize(self.device)
        self.model_time_sec += sum(
            start.elapsed_time(end) / 1000.0 for start, end in self._model_events
        )
        self.transfer_time_sec += sum(
            start.elapsed_time(end) / 1000.0 for start, end in self._transfer_events
        )
        if self._origin is None:
            return
        model_intervals = [
            (
                self._origin.elapsed_time(start) / 1000.0,
                self._origin.elapsed_time(end) / 1000.0,
            )
            for start, end in self._model_events
        ]
        transfer_intervals = [
            (
                self._origin.elapsed_time(start) / 1000.0,
                self._origin.elapsed_time(end) / 1000.0,
            )
            for start, end in self._transfer_events
        ]
        self.transfer_overlap_sec = _interval_overlap(
            model_intervals,
            transfer_intervals,
        )

    def to_dict(self) -> dict[str, float | int]:
        self.finish()
        overlap_ratio = (
            self.transfer_overlap_sec / self.transfer_time_sec
            if self.transfer_time_sec > 0.0
            else 0.0
        )
        return {
            "model_time_sec": self.model_time_sec,
            "gather_time_sec": self.gather_time_sec,
            "transfer_time_sec": self.transfer_time_sec,
            "finalization_time_sec": self.finalization_time_sec,
            "metric_time_sec": self.metric_time_sec,
            "transfer_bytes": self.transfer_bytes,
            "transfer_overlap_sec": self.transfer_overlap_sec,
            "transfer_overlap_ratio": overlap_ratio,
        }


def _interval_overlap(
    left: list[tuple[float, float]],
    right: list[tuple[float, float]],
) -> float:
    """Measure overlap between two chronologically ordered interval streams."""
    overlap = 0.0
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        left_start, left_end = left[left_index]
        right_start, right_end = right[right_index]
        overlap += max(0.0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return overlap


@dataclasses.dataclass
class _PendingCpuTransfer:
    payload: Any
    completed: Any
    started: Any
    num_bytes: int


class _CpuTransferPipeline:
    """Bound CUDA D2H staging while allowing copies to overlap model work."""

    def __init__(
        self,
        telemetry: _EvaluationTelemetry | None,
        *,
        max_pending: int = 2,
    ) -> None:
        self._telemetry = telemetry
        self._max_pending = max_pending
        self._pending: deque[_PendingCpuTransfer] = deque()
        self._stream: Any | None = None
        self.max_pending_observed = 0

    def submit(self, payload: Any) -> list[Any]:
        device = _first_cuda_device(payload)
        if device is None:
            started = time.perf_counter()
            cpu_payload = _to_cpu_nested(payload)
            if self._telemetry is not None:
                self._telemetry.add_transfer(
                    num_bytes=_nested_num_bytes(payload),
                    elapsed_sec=time.perf_counter() - started,
                )
            return [cpu_payload]

        completed_payloads: list[Any] = []
        if len(self._pending) >= self._max_pending:
            completed_payloads.append(self._drain_one())

        if self._stream is None:
            self._stream = torch.cuda.Stream(device=device)
        producer = torch.cuda.current_stream(device)
        self._stream.wait_stream(producer)
        started = torch.cuda.Event(enable_timing=True)
        completed = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self._stream):
            started.record(self._stream)
            staged, num_bytes = _stage_cuda_nested(payload, self._stream)
            completed.record(self._stream)
        self._pending.append(
            _PendingCpuTransfer(
                payload=staged,
                completed=completed,
                started=started,
                num_bytes=num_bytes,
            )
        )
        self.max_pending_observed = max(
            self.max_pending_observed,
            len(self._pending),
        )
        return completed_payloads

    def drain_all(self) -> list[Any]:
        return [self._drain_one() for _ in range(len(self._pending))]

    def _drain_one(self) -> Any:
        pending = self._pending.popleft()
        pending.completed.synchronize()
        if self._telemetry is not None:
            self._telemetry.add_transfer(
                num_bytes=pending.num_bytes,
                events=(pending.started, pending.completed),
            )
        return _to_pageable_nested(pending.payload)


# ---------------------------------------------------------------------------
# _PredictionAccumulator
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _PredictionAccumulator:
    """Collects per-batch eval tensors with optional CPU-offload cadence.

    The accumulator is bypassed when ``prediction_loss_only`` is ``True``:
    only per-example losses are retained and ``finalize`` returns
    ``predictions=None`` and ``label_ids=None``.

    Storage layout (HF parity):

    - ``_hot_*`` lists hold the *current* on-device batch group.
    - ``_cold_*`` lists are binary chunk trees of previously-flushed CPU
      payloads, with at most one frozen chunk at each power-of-two level.

    An unset ``eval_accumulation_steps`` flushes after every batch, bounding
    on-device prediction storage to one batch. An explicit value retains at
    most that many batches. ``flush_to_cpu`` moves every hot payload to CPU
    before concatenating it, inserts each result into its cold chunk tree,
    then resets the hot buffers. CUDA uses a two-entry pinned staging queue;
    balanced merging keeps CPU chunk count logarithmic and copy volume
    subquadratic without allocating a full on-device concatenation result.

    When ``eval_do_concat_batches`` is ``True`` (HF default), ``finalize``
    concatenates the cold + trailing-hot chunks into one tensor.  When
    ``False``, the per-batch list is returned (no concat) so
    ``compute_metrics`` can introspect each batch separately.

    ``EvalPrediction.losses`` is populated with **per-example** losses —
    each batch's reduced loss is repeated by its batch size before being
    appended (HF parity, mirroring ``losses.repeat(batch_size)``).  This
    removes the silent corruption that would otherwise occur when a
    custom head returns a sum-reduced loss.
    """

    prediction_loss_only: bool = False
    eval_accumulation_steps: int | None = None
    eval_do_concat_batches: bool = True
    include_inputs: bool = False
    include_losses: bool = False
    telemetry: _EvaluationTelemetry | None = dataclasses.field(
        default=None,
        repr=False,
    )

    # On-device "hot" buffers — current batch group; flushed periodically.
    _hot_losses: list[Tensor] = dataclasses.field(default_factory=list)
    _hot_logits: list[Any] = dataclasses.field(default_factory=list)
    _hot_labels: list[Any] = dataclasses.field(default_factory=list)
    # ``inputs`` is the *bare* main-input tensor (HF parity:
    # ``EvalLoopContainer`` collects ``inputs_decode = inputs[main_input_name]``
    # — a single tensor, not a dict).
    _hot_inputs: list[Tensor] = dataclasses.field(default_factory=list)

    # On-CPU binary chunk trees. Level i is empty or stores 2**i flushes,
    # bounding retained chunk count logarithmically without prefix rebuilding.
    _cold_losses: list[Any | None] = dataclasses.field(default_factory=list)
    _cold_logits: list[Any | None] = dataclasses.field(default_factory=list)
    _cold_labels: list[Any | None] = dataclasses.field(default_factory=list)
    _cold_inputs: list[Any | None] = dataclasses.field(default_factory=list)

    _num_batches: int = 0
    _transfer_pipeline: _CpuTransferPipeline = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self._transfer_pipeline = _CpuTransferPipeline(self.telemetry)

    def add(
        self,
        loss: Tensor | None,
        logits: Any | None,
        labels: Any | None,
        inputs: Tensor | None,
        *,
        batch_size: int,
    ) -> None:
        """Append one batch's tensors to the on-device "hot" buffers.

        ``batch_size`` is required so per-example losses can be produced
        even when the model returns a scalar, batch-reduced loss
        (``losses.repeat(bs)``, HF parity).

        ``inputs`` is the bare main-input tensor (HF parity:
        ``inputs_decode = inputs[main_input_name]``), or ``None`` if the
        collator didn't emit the primary input.
        """
        del batch_size  # reserved for HF-parity scalar-loss expansion
        # ``loss`` is either scalar (the standard ``prediction_step``
        # path: a batch-mean reduced by the model's ``forward``) or 1-D
        # of length ``batch_size`` (the vmap'd eval closure path
        # triggered by ``'loss' in include_for_metrics``).  We store
        # only the real per-example track; scalar losses are discarded
        # at this point because they carry no per-example information
        # — populating ``EvalPrediction.losses`` from a replicated
        # batch-mean is fake-by-construction and HF-misleading.
        if self.include_losses and loss is not None and loss.ndim > 0:
            self._hot_losses.append(loss.detach())

        if not self.prediction_loss_only:
            if logits is not None:
                self._hot_logits.append(nested_detach(logits))
            if labels is not None:
                self._hot_labels.append(nested_detach(labels))
            if self.include_inputs and inputs is not None:
                self._hot_inputs.append(inputs.detach())

        self._num_batches += 1

        # In ``eval_do_concat_batches=False`` mode we MUST keep per-batch
        # tensors as separate cold chunks (the contract is "list of
        # per-batch tensors").  Force a flush after every add so each
        # cold chunk corresponds to exactly one batch.
        flush_now = not self.eval_do_concat_batches
        if not flush_now:
            accumulation_steps = (
                self.eval_accumulation_steps
                if self.eval_accumulation_steps is not None
                else 1
            )
            flush_now = self._num_batches % accumulation_steps == 0
        if flush_now:
            self.flush_to_cpu()

    def flush_to_cpu(self) -> None:
        """Schedule the current hot group for bounded CPU offload."""
        if not any(
            (self._hot_losses, self._hot_logits, self._hot_labels, self._hot_inputs)
        ):
            return

        payload = {
            "losses": self._hot_losses or None,
            "logits": self._hot_logits or None,
            "labels": self._hot_labels or None,
            "inputs": self._hot_inputs or None,
        }
        self._hot_losses = []
        self._hot_logits = []
        self._hot_labels = []
        self._hot_inputs = []
        for completed in self._transfer_pipeline.submit(payload):
            self._consume_transferred(completed)

    def _consume_transferred(self, payload: Mapping[str, Any]) -> None:
        if payload["losses"]:
            self._append_cold_chunk(
                self._cold_losses,
                _freeze_cpu_chunk(payload["losses"]),
                pad_value=0,
            )
        if payload["logits"]:
            self._append_cold_chunk(
                self._cold_logits,
                _freeze_cpu_chunk(payload["logits"], pad_value=_HF_PAD_VALUE),
                pad_value=_HF_PAD_VALUE,
            )
        if payload["labels"]:
            self._append_cold_chunk(
                self._cold_labels,
                _freeze_cpu_chunk(payload["labels"], pad_value=_HF_PAD_VALUE),
                pad_value=_HF_PAD_VALUE,
            )
        if payload["inputs"]:
            self._append_cold_chunk(
                self._cold_inputs,
                _freeze_cpu_chunk(payload["inputs"], pad_value=_HF_PAD_VALUE),
                pad_value=_HF_PAD_VALUE,
            )

    def _append_cold_chunk(
        self,
        levels: list[Any | None],
        chunk: Any,
        *,
        pad_value: int | float,
    ) -> None:
        if not self.eval_do_concat_batches:
            levels.append(chunk)
            return
        level = 0
        while level < len(levels) and levels[level] is not None:
            chunk = _concat_nested_chunks(
                [levels[level], chunk],
                padding_value=pad_value,
            )
            levels[level] = None
            level += 1
        if level == len(levels):
            levels.append(chunk)
        else:
            levels[level] = chunk

    def _drain_transfers(self) -> None:
        for completed in self._transfer_pipeline.drain_all():
            self._consume_transferred(completed)

    def _phase(self, name: str) -> Any:
        if self.telemetry is None:
            return nullcontext()
        return self.telemetry.phase(name)

    def finalize(
        self,
        *,
        num_samples: int | None = None,
        gather: bool = False,
    ) -> tuple[
        Any | None,
        Any | None,
        Any | None,
        Any | None,
    ]:
        """Return ``(predictions, label_ids, inputs, losses)`` as numpy arrays.

        ``predictions`` / ``label_ids`` are ``None`` when
        ``prediction_loss_only`` is ``True`` or no logits / labels were
        ever added.  ``inputs`` is a single tensor (HF parity:
        ``inputs_decode`` collected from ``inputs[main_input_name]``)
        or ``None`` when the user didn't request it / the collator
        didn't emit a primary input.

        ``losses`` is a 1-D numpy array of length ``total_samples`` when
        ``include_losses`` is ``True``, otherwise ``None``.  Per-example
        — *not* per-batch — semantics (HF parity).

        ``num_samples`` (when set) truncates each leading-dim of the
        returned tensors to that length via HF's
        :func:`~transformers.trainer_pt_utils.nested_truncate` — drops
        gather-padding rows distributed gather may introduce.  Pass
        ``None`` to skip truncation.

        ``gather=True`` all-gathers each tensor pytree
        across DDP ranks via :func:`opaque.distributed.gather_pytree`
        *before* numpify, so per-rank shards are concatenated into the
        cluster-wide result.  Single-process eval should leave the
        default (``False``).

        Tensors are converted to numpy arrays via HF's
        :func:`~transformers.trainer_pt_utils.nested_numpify` so user
        ``compute_metrics`` callbacks receive the same types as HF
        ``Trainer`` would deliver (works for the ``evaluate`` /
        ``sklearn`` / ``seqeval`` ecosystem).  When
        ``eval_do_concat_batches=False`` the per-batch chunks are
        preserved as a list (mirrors HF's ``EvalLoopContainer`` with
        ``do_nested_concat=False``).
        """
        # Final flush and drain make all local payloads available before
        # concatenation or distributed collection.
        self.flush_to_cpu()
        self._drain_transfers()

        with self._phase("finalization"):
            predictions = self._collect_chunks(
                self._cold_logits,
                empty_ok=self.prediction_loss_only,
                pad_value=_HF_PAD_VALUE,
            )
            labels = self._collect_chunks(
                self._cold_labels,
                empty_ok=self.prediction_loss_only,
                pad_value=_HF_PAD_VALUE,
            )

            inputs: Any | None
            if self.include_inputs and self._cold_inputs:
                inputs = self._collect_chunks(
                    self._cold_inputs,
                    empty_ok=False,
                    pad_value=_HF_PAD_VALUE,
                )
            else:
                inputs = None

            losses: Any | None
            if self.include_losses and self._cold_losses:
                losses = _concat_nested_chunks(
                    self._ordered_cold_chunks(self._cold_losses),
                    padding_value=0,
                )
            else:
                losses = None

        # All ranks enter the same four collectives even when a rank-local
        # shard produced no payload. ``gather_pytree`` treats local ``None`` as
        # an empty contribution while preserving rank order.
        if gather:
            from opaque.api.engine.distributed._state import gather_pytree

            with self._phase("gather"):
                predictions = gather_pytree(predictions)
                labels = gather_pytree(labels)
                inputs = gather_pytree(inputs)
                losses = gather_pytree(losses)

        # HF parity: ``compute_metrics`` consumes numpy arrays, not
        # ``torch.Tensor``.  ``nested_numpify`` recurses into lists / dicts
        # / tuples so all four containers are converted uniformly.
        with self._phase("finalization"):
            if predictions is not None:
                predictions = nested_numpify(predictions)
            if labels is not None:
                labels = nested_numpify(labels)
            if inputs is not None:
                inputs = nested_numpify(inputs)
            if losses is not None:
                losses = nested_numpify(losses)

        # HF parity (trainer.py: end of ``evaluation_loop``): truncate
        # leading dim to the dataset's true sample count so user
        # ``compute_metrics`` callbacks see ``predictions.shape[0] ==
        # num_samples`` regardless of any padding the gather/pad path
        # introduced upstream.  No-op for single-process eval (where
        # ``sum(batch_sizes) == num_samples`` already); needed under
        # distributed eval where the pad makes the gather rectangular.
        if num_samples is not None:
            with self._phase("finalization"):
                if predictions is not None:
                    predictions = nested_truncate(predictions, num_samples)
                if labels is not None:
                    labels = nested_truncate(labels, num_samples)
                if inputs is not None:
                    inputs = nested_truncate(inputs, num_samples)
                if losses is not None:
                    losses = nested_truncate(losses, num_samples)

        return predictions, labels, inputs, losses

    def _ordered_cold_chunks(self, cold: list[Any | None]) -> list[Any]:
        chunks = [chunk for chunk in cold if chunk is not None]
        if self.eval_do_concat_batches:
            chunks.reverse()
        return chunks

    def _collect_chunks(
        self,
        cold: list[Any | None],
        *,
        empty_ok: bool,
        pad_value: int | float,
    ) -> Any | list[Any] | None:
        if not cold:
            # An empty container always collapses to ``None`` (HF parity:
            # ``compute_metrics`` receives ``None`` for predictions/labels in
            # loss-only mode).  ``empty_ok`` is retained as caller intent —
            # predictions/labels pass ``prediction_loss_only``; ``inputs`` is
            # only collected behind a non-empty guard.
            del empty_ok
            return None
        chunks = self._ordered_cold_chunks(cold)
        if self.eval_do_concat_batches:
            return _concat_nested_chunks(chunks, padding_value=pad_value)
        return chunks


def _to_cpu_nested(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.to("cpu")
    if isinstance(value, tuple):
        return type(value)(_to_cpu_nested(v) for v in value)
    if isinstance(value, list):
        return type(value)(_to_cpu_nested(v) for v in value)
    if isinstance(value, Mapping):
        return type(value)({k: _to_cpu_nested(v) for k, v in value.items()})
    return value


def _first_cuda_device(value: Any) -> torch.device | None:
    if isinstance(value, Tensor):
        return value.device if value.device.type == "cuda" else None
    if isinstance(value, (tuple, list)):
        for item in value:
            if (device := _first_cuda_device(item)) is not None:
                return device
    elif isinstance(value, Mapping):
        for item in value.values():
            if (device := _first_cuda_device(item)) is not None:
                return device
    return None


def _nested_num_bytes(value: Any) -> int:
    if isinstance(value, Tensor):
        if value.device.type == "cpu":
            return 0
        return value.numel() * value.element_size()
    if isinstance(value, (tuple, list)):
        return sum(_nested_num_bytes(item) for item in value)
    if isinstance(value, Mapping):
        return sum(_nested_num_bytes(item) for item in value.values())
    return 0


def _stage_cuda_nested(value: Any, stream: Any) -> tuple[Any, int]:
    if isinstance(value, Tensor):
        if value.device.type != "cuda":
            return value.to("cpu"), _nested_num_bytes(value)
        staged = torch.empty_like(value, device="cpu", pin_memory=True)
        staged.copy_(value, non_blocking=True)
        value.record_stream(stream)
        return staged, value.numel() * value.element_size()
    if isinstance(value, tuple):
        staged_items = [_stage_cuda_nested(item, stream) for item in value]
        return type(value)(item for item, _ in staged_items), sum(
            size for _, size in staged_items
        )
    if isinstance(value, list):
        staged_items = [_stage_cuda_nested(item, stream) for item in value]
        return type(value)(item for item, _ in staged_items), sum(
            size for _, size in staged_items
        )
    if isinstance(value, Mapping):
        staged_items = {
            key: _stage_cuda_nested(item, stream) for key, item in value.items()
        }
        return type(value)({key: item for key, (item, _) in staged_items.items()}), sum(
            size for _, size in staged_items.values()
        )
    return value, 0


def _to_pageable_nested(value: Any) -> Any:
    if isinstance(value, Tensor):
        if not value.is_pinned():
            return value
        pageable = torch.empty_like(value, device="cpu", pin_memory=False)
        pageable.copy_(value)
        return pageable
    if isinstance(value, tuple):
        return type(value)(_to_pageable_nested(item) for item in value)
    if isinstance(value, list):
        return type(value)(_to_pageable_nested(item) for item in value)
    if isinstance(value, Mapping):
        return type(value)(
            {key: _to_pageable_nested(item) for key, item in value.items()}
        )
    return value


def _concat_tensor_chunks(
    tensors: list[Tensor],
    *,
    padding_value: int | float,
) -> Tensor:
    chunks = [torch.atleast_1d(tensor) for tensor in tensors]
    first = chunks[0]
    for tensor in chunks[1:]:
        if tensor.dtype != first.dtype or tensor.device != first.device:
            raise TypeError("Evaluation tensor chunks must share dtype and device.")  # noqa: TRY003
        if tensor.ndim != first.ndim or tensor.shape[2:] != first.shape[2:]:
            raise ValueError("Evaluation tensor chunks have incompatible shapes.")  # noqa: TRY003

    output_shape = list(first.shape)
    output_shape[0] = sum(tensor.shape[0] for tensor in chunks)
    needs_padding = first.ndim > 1 and any(
        tensor.shape[1] != first.shape[1] for tensor in chunks[1:]
    )
    if needs_padding:
        output_shape[1] = max(tensor.shape[1] for tensor in chunks)
        output = first.new_full(output_shape, padding_value)
    else:
        output = first.new_empty(output_shape)

    offset = 0
    for tensor in chunks:
        target = [slice(offset, offset + tensor.shape[0])]
        target.extend(slice(0, size) for size in tensor.shape[1:])
        output[tuple(target)].copy_(tensor)
        offset += tensor.shape[0]
    return output


def _concat_nested_chunks(
    tensors: list[Any],
    *,
    padding_value: int | float = _HF_PAD_VALUE,
) -> Any:
    if not tensors:
        raise ValueError("Cannot concatenate an empty evaluation chunk list.")  # noqa: TRY003
    if len(tensors) == 1:
        return tensors[0]
    first = tensors[0]
    if isinstance(first, Tensor):
        if not all(isinstance(tensor, Tensor) for tensor in tensors):
            raise TypeError("Evaluation chunks must have matching nested structures.")  # noqa: TRY003
        return _concat_tensor_chunks(tensors, padding_value=padding_value)
    if isinstance(first, (tuple, list)):
        if not all(type(tensor) is type(first) for tensor in tensors):
            raise TypeError("Evaluation chunks must have matching nested structures.")  # noqa: TRY003
        if not all(len(tensor) == len(first) for tensor in tensors):
            raise ValueError("Evaluation chunk sequences must have matching lengths.")  # noqa: TRY003
        return type(first)(
            _concat_nested_chunks(
                [tensor[index] for tensor in tensors],
                padding_value=padding_value,
            )
            for index in range(len(first))
        )
    if isinstance(first, Mapping):
        if not all(type(tensor) is type(first) for tensor in tensors):
            raise TypeError("Evaluation chunks must have matching nested structures.")  # noqa: TRY003
        keys = tuple(first)
        key_set = set(keys)
        if not all(set(tensor) == key_set for tensor in tensors):
            raise ValueError("Evaluation chunk mappings must have matching keys.")  # noqa: TRY003
        return type(first)(
            {
                key: _concat_nested_chunks(
                    [tensor[key] for tensor in tensors],
                    padding_value=padding_value,
                )
                for key in keys
            }
        )
    raise TypeError(f"Unsupported evaluation chunk type: {type(first).__name__}")  # noqa: TRY003


def _freeze_cpu_chunk(
    tensors: list[Any],
    *,
    pad_value: int | float = 0,
) -> Any:
    return _concat_nested_chunks(tensors, padding_value=pad_value)


def _freeze_hot_chunk(
    tensors: list[Any],
    *,
    pad_value: int | float = 0,
) -> Any:
    """Synchronously move and freeze a hot chunk (CPU/MPS fallback helper)."""
    return _freeze_cpu_chunk(
        [_to_cpu_nested(tensor) for tensor in tensors],
        pad_value=pad_value,
    )


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def speed_metrics(
    prefix: str,
    start_time: float,
    *,
    num_samples: int | None = None,
    num_steps: int | None = None,
) -> dict[str, float]:
    """Compute throughput metrics for an eval / predict pass.

    Pure helper mirroring ``transformers.trainer_utils.speed_metrics`` —
    we don't import HF's version because it is part of an unstable
    private surface.  Returned keys (when the corresponding count is
    available):

    - ``{prefix}_runtime`` — wall time in seconds (always emitted).
    - ``{prefix}_samples_per_second`` — ``num_samples / runtime``.
    - ``{prefix}_steps_per_second`` — ``num_steps / runtime``.

    All values are rounded to three decimals (HF parity —
    ``transformers.trainer_utils.speed_metrics`` rounds with
    ``round(x, 4)`` for runtime but ``round(x, 3)`` for derived rates;
    we round all three uniformly to 3 to keep dashboard parity tight).
    """
    runtime = max(time.monotonic() - start_time, 1e-9)
    out: dict[str, float] = {f"{prefix}_runtime": round(runtime, 4)}
    if num_samples is not None:
        out[f"{prefix}_samples_per_second"] = round(float(num_samples) / runtime, 3)
    if num_steps is not None:
        out[f"{prefix}_steps_per_second"] = round(float(num_steps) / runtime, 3)
    return out


def resolve_eval_num_samples(dataloader: Any, *, observed: int) -> int:
    """Resolve the ``num_samples`` field of an :class:`EvaluationResult`.

    HF parity (``transformers.trainer.Trainer.evaluation_loop``,
    trainer.py:4757-4769): prefer the dataset's reported length, then the
    dataloader-driven count, then the observed batch sums.  Streaming
    iterables expose neither length so we land on the observed count.

    ``observed`` is the running batch-size sum recorded by the eval loop;
    we use it as both the final fallback and as a non-zero rescue when
    upstream length probes return zero (HF does the same).
    """
    dataset = getattr(dataloader, "dataset", None)
    # 1. dataset.__len__ (map-style or finite IterableDataset).
    try:
        if dataset is not None:
            return len(dataset)
    except TypeError:
        pass
    # 2. Sharded iterable datasets carry a usable ``num_examples`` attr.
    if dataset is not None:
        n = getattr(dataset, "num_examples", 0)
        if isinstance(n, int) and n > 0:
            return n
    # 3. Dataloader's own length × declared batch size.
    try:
        n_batches = len(dataloader)
        bs = int(getattr(dataloader, "batch_size", 0) or 0)
        if n_batches > 0 and bs > 0:
            return n_batches * bs
    except TypeError:
        pass
    # 4. Observed batch sums.
    return int(observed)


def with_metric_prefix(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Return a copy of ``metrics`` with ``{prefix}_`` prepended where missing.

    Mirrors HF's behavior: keys that already start with ``{prefix}_`` pass
    through unchanged.  This keeps user-supplied ``compute_metrics`` outputs
    HF-compatible regardless of whether the user pre-prefixes their keys.
    """
    if not prefix:
        return dict(metrics)
    out: dict[str, Any] = {}
    head = f"{prefix}_"
    for k, v in metrics.items():
        out[k if k.startswith(head) else f"{head}{k}"] = v
    return out
