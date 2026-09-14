"""Unit tests for opaque.api.transformers.trainer._eval helpers.

Covers ``_PredictionAccumulator`` and the reporting helpers without
instantiating a model.  Evaluation cadence comes from HF's
``DefaultFlowCallback`` and is covered end-to-end in
``tests/validation/test_trainer.py``.
"""

from __future__ import annotations

import pytest
import torch

import opaque.api.transformers.trainer._eval as _eval_mod
from opaque.api.transformers.trainer._eval import (
    EvalPrediction,
    EvaluationResult,
    _PredictionAccumulator,
    with_metric_prefix,
)

# ---------------------------------------------------------------------------
# _PredictionAccumulator
# ---------------------------------------------------------------------------


def _b(loss: float, logits_shape=(2, 4), labels_shape=(2, 4)) -> tuple:
    """Build a fake batch's tensors.

    Inputs is the bare main-input tensor (HF parity: the accumulator
    collects ``inputs[main_input_name]`` as a single tensor, not a
    dict).
    """
    return (
        torch.tensor(loss),
        torch.randn(*logits_shape),
        torch.randint(0, 10, labels_shape),
        torch.zeros(logits_shape[0], dtype=torch.long),
    )


def _add_batch(acc: _PredictionAccumulator, loss: float, **kw) -> None:
    """Helper that derives ``batch_size`` from the synthetic batch shape."""
    args = _b(loss, **kw)
    acc.add(*args, batch_size=int(args[1].shape[0]))


class TestPredictionAccumulator:
    def test_default_concatenates_at_finalize(self):
        acc = _PredictionAccumulator()
        for _ in range(3):
            _add_batch(acc, 0.5)
        preds, labels, inputs, losses = acc.finalize()
        assert preds.shape == (6, 4)
        assert labels.shape == (6, 4)
        assert inputs is None  # include_inputs=False
        assert losses is None  # include_losses=False

    def test_loss_only_short_circuits(self):
        acc = _PredictionAccumulator(prediction_loss_only=True, include_losses=True)
        # ``include_losses`` now stores only **real** 1-D per-example
        # losses (produced by the vmap'd eval closure).  Pass per-example
        # tensors of length ``batch_size`` directly.
        for vals in ([0.1, 0.2], [0.3, 0.4], [0.5, 0.6]):
            acc.add(
                loss=torch.tensor(vals),
                logits=torch.randn(2, 4),
                labels=torch.randint(0, 10, (2, 4)),
                inputs=torch.zeros(2, dtype=torch.long),
                batch_size=2,
            )
        preds, labels, _inputs, losses = acc.finalize()
        assert preds is None
        assert labels is None
        assert losses is not None
        assert losses.shape == (6,)
        # Real per-example: each value distinct, not replicated.
        assert losses[0].item() == pytest.approx(0.1)
        assert losses[1].item() == pytest.approx(0.2)
        assert losses[5].item() == pytest.approx(0.6)

    def test_do_concat_false_returns_lists(self):
        acc = _PredictionAccumulator(eval_do_concat_batches=False)
        for _ in range(3):
            _add_batch(acc, 0.5)
        preds, labels, _, _ = acc.finalize()
        assert isinstance(preds, list)
        assert len(preds) == 3
        assert isinstance(labels, list)
        assert len(labels) == 3

    def test_accumulation_steps_flush_cadence(self):
        # 5 batches with eval_accumulation_steps=2 → flushes after batch 2
        # and batch 4 (2 cold chunks).  ``finalize`` performs a third flush
        # for the trailing batch (5th), producing 3 cold chunks total.
        acc = _PredictionAccumulator(eval_accumulation_steps=2)
        for _ in range(5):
            _add_batch(acc, 0.5)
        # Two flushes before finalize have merged into one balanced-tree node.
        assert len(acc._cold_logits) == 2
        assert sum(chunk is not None for chunk in acc._cold_logits) == 1
        acc.finalize()
        assert len(acc._cold_logits) == 2
        assert sum(chunk is not None for chunk in acc._cold_logits) == 2

    def test_accumulation_steps_none_flushes_each_batch(self):
        acc = _PredictionAccumulator(eval_accumulation_steps=None)
        for _ in range(4):
            _add_batch(acc, 0.5)
        # The default effective window is one batch, bounding device retention;
        # four cold flushes compact into one level-2 node.
        assert len(acc._cold_logits) == 3
        assert sum(chunk is not None for chunk in acc._cold_logits) == 1
        acc.finalize()
        assert len(acc._cold_logits) == 3

    def test_cold_chunk_tree_stays_logarithmically_bounded(self):
        acc = _PredictionAccumulator()
        for value in range(1024):
            logits = torch.tensor([value])
            acc.add(
                loss=None,
                logits=logits,
                labels=None,
                inputs=None,
                batch_size=1,
            )

        assert len(acc._cold_logits) == 11
        assert sum(chunk is not None for chunk in acc._cold_logits) == 1
        predictions, _, _, _ = acc.finalize()
        torch.testing.assert_close(
            torch.from_numpy(predictions),
            torch.arange(1024),
        )

    def test_freeze_moves_payloads_before_cpu_concatenation(self, monkeypatch):
        events: list[str] = []
        original_to_cpu = _eval_mod._to_cpu_nested
        original_concat = _eval_mod._concat_nested_chunks

        def recording_to_cpu(value):
            events.append("to_cpu")
            return original_to_cpu(value)

        def recording_concat(tensors, *, padding_value):
            assert events
            events.append("concat")
            return original_concat(tensors, padding_value=padding_value)

        monkeypatch.setattr(_eval_mod, "_to_cpu_nested", recording_to_cpu)
        monkeypatch.setattr(_eval_mod, "_concat_nested_chunks", recording_concat)

        _eval_mod._freeze_hot_chunk([torch.ones(1), torch.ones(1)])

        assert events[-1] == "concat"

    def test_accumulation_windows_preserve_finalized_values(self):
        batches = [
            (torch.tensor([[1.0, 2.0]]), torch.tensor([[1, 2]])),
            (torch.tensor([[3.0], [4.0]]), torch.tensor([[3], [4]])),
            (torch.tensor([[5.0, 6.0, 7.0]]), torch.tensor([[5, 6, 7]])),
        ]

        outputs = []
        for window in (None, 2):
            acc = _PredictionAccumulator(eval_accumulation_steps=window)
            for logits, labels in batches:
                acc.add(
                    loss=None,
                    logits=logits,
                    labels=labels,
                    inputs=None,
                    batch_size=int(logits.shape[0]),
                )
            outputs.append(acc.finalize())

        default_preds, default_labels, _, _ = outputs[0]
        explicit_preds, explicit_labels, _, _ = outputs[1]
        assert torch.equal(
            torch.from_numpy(default_preds), torch.from_numpy(explicit_preds)
        )
        assert torch.equal(
            torch.from_numpy(default_labels), torch.from_numpy(explicit_labels)
        )

    def test_include_inputs_populates_finalize(self):
        # HF parity: ``EvalPrediction.inputs`` is a bare tensor (the
        # main input column collected via
        # ``inputs_decode = inputs[main_input_name]``), not a dict.
        acc = _PredictionAccumulator(include_inputs=True)
        for _ in range(3):
            _add_batch(acc, 0.5)
        _, _, inputs, _ = acc.finalize()
        assert inputs is not None
        assert inputs.shape == (6,)

    def test_include_losses_is_per_example(self):
        # Real per-example losses: each entry distinct.
        acc = _PredictionAccumulator(include_losses=True)
        for batch_vals in ([0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]):
            acc.add(
                loss=torch.tensor(batch_vals),
                logits=torch.randn(2, 4),
                labels=torch.randint(0, 10, (2, 4)),
                inputs=torch.zeros(2, dtype=torch.long),
                batch_size=2,
            )
        _, _, _, losses = acc.finalize()
        assert losses is not None
        assert losses.shape == (8,)
        # Per-example values are stored as-is, not replicated batch-mean.
        for idx, expected in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]):
            assert losses[idx].item() == pytest.approx(expected)

    def test_include_losses_drops_scalar_batch_mean(self):
        # Scalar (batch-mean) losses carry no per-example information so
        # the accumulator silently drops them — populating
        # ``EvalPrediction.losses`` from a replicated mean is fake by
        # construction.  Users wanting real per-example losses must set
        # ``include_for_metrics=['loss']`` so ``prediction_step`` takes
        # the vmap'd eval path.
        acc = _PredictionAccumulator(include_losses=True)
        for v in (0.1, 0.2, 0.3):
            _add_batch(acc, v)
        _, _, _, losses = acc.finalize()
        assert losses is None

    def test_missing_loss_still_collects_predictions(self):
        acc = _PredictionAccumulator(include_losses=True)
        logits = torch.randn(2, 3)
        labels = torch.arange(2)
        acc.add(
            loss=None,
            logits=logits,
            labels=labels,
            inputs=None,
            batch_size=2,
        )
        preds, label_ids, _, losses = acc.finalize()
        assert preds.shape == (2, 3)
        assert label_ids.shape == (2,)
        assert losses is None

    def test_tuple_predictions_and_labels_round_trip(self):
        acc = _PredictionAccumulator()
        acc.add(
            loss=torch.tensor(0.5),
            logits=(torch.randn(2, 3), torch.randn(2, 4)),
            labels=(torch.arange(2), torch.arange(2) + 10),
            inputs=None,
            batch_size=2,
        )
        acc.add(
            loss=torch.tensor(0.6),
            logits=(torch.randn(1, 3), torch.randn(1, 4)),
            labels=(torch.arange(1), torch.arange(1) + 10),
            inputs=None,
            batch_size=1,
        )
        preds, label_ids, _, _ = acc.finalize()
        assert isinstance(preds, tuple)
        assert isinstance(label_ids, tuple)
        assert preds[0].shape == (3, 3)
        assert preds[1].shape == (3, 4)
        assert label_ids[0].shape == (3,)
        assert label_ids[1].shape == (3,)


class TestLinearNestedAssembly:
    def test_nested_mapping_sequence_and_scalar_semantics(self):
        chunks = [
            {
                "scores": torch.tensor([[1.0, 2.0]]),
                "aux": [torch.tensor(3.0)],
            },
            {
                "scores": torch.tensor([[4.0], [5.0]]),
                "aux": [torch.tensor(6.0)],
            },
        ]

        result = _eval_mod._concat_nested_chunks(chunks)

        assert isinstance(result, dict)
        assert isinstance(result["aux"], list)
        torch.testing.assert_close(
            result["scores"],
            torch.tensor([[1.0, 2.0], [4.0, -100.0], [5.0, -100.0]]),
        )
        torch.testing.assert_close(result["aux"][0], torch.tensor([3.0, 6.0]))

    def test_mapping_key_order_may_differ_between_chunks(self):
        chunks = [
            {"scores": torch.tensor([1]), "labels": torch.tensor([2])},
            {"labels": torch.tensor([3]), "scores": torch.tensor([4])},
        ]

        result = _eval_mod._concat_nested_chunks(chunks)

        assert tuple(result) == ("scores", "labels")
        torch.testing.assert_close(result["scores"], torch.tensor([1, 4]))
        torch.testing.assert_close(result["labels"], torch.tensor([2, 3]))

    def test_each_nested_leaf_is_assembled_once(self, monkeypatch):
        calls = 0
        original = _eval_mod._concat_tensor_chunks

        def recording_concat(tensors, *, padding_value):
            nonlocal calls
            calls += 1
            return original(tensors, padding_value=padding_value)

        monkeypatch.setattr(_eval_mod, "_concat_tensor_chunks", recording_concat)
        chunks = [
            {"left": torch.full((1, 2), i), "right": (torch.tensor(i),)}
            for i in range(64)
        ]

        result = _eval_mod._concat_nested_chunks(chunks)

        assert calls == 2
        assert result["left"].shape == (64, 2)
        assert result["right"][0].shape == (64,)

    def test_incompatible_trailing_shapes_are_rejected(self):
        with pytest.raises(ValueError, match="incompatible shapes"):
            _eval_mod._concat_nested_chunks([torch.ones(1, 2, 3), torch.ones(1, 2, 4)])


class TestEvaluationTelemetry:
    def test_interval_overlap_uses_ordered_streams(self):
        model = [(0.0, 2.0), (3.0, 5.0), (7.0, 8.0)]
        transfer = [(1.0, 4.0), (4.5, 6.0), (8.0, 9.0)]

        assert _eval_mod._interval_overlap(model, transfer) == pytest.approx(2.5)

    def test_cpu_transfer_and_phase_metrics(self):
        telemetry = _eval_mod._EvaluationTelemetry(torch.device("cpu"))
        acc = _PredictionAccumulator(telemetry=telemetry)
        with telemetry.model():
            _add_batch(acc, 0.5)
        acc.finalize()

        metrics = telemetry.to_dict()
        assert metrics.keys() == {
            "model_time_sec",
            "gather_time_sec",
            "transfer_time_sec",
            "finalization_time_sec",
            "metric_time_sec",
            "transfer_bytes",
            "transfer_overlap_sec",
            "transfer_overlap_ratio",
        }
        assert metrics["model_time_sec"] >= 0.0
        assert metrics["finalization_time_sec"] >= 0.0
        assert metrics["transfer_bytes"] == 0

    @pytest.mark.cuda
    def test_cuda_transfer_queue_is_bounded_and_stream_safe(self):
        telemetry = _eval_mod._EvaluationTelemetry(torch.device("cuda"))
        acc = _PredictionAccumulator(telemetry=telemetry)
        expected = []
        expected_bytes = 0
        for i in range(6):
            with telemetry.model():
                logits = torch.full((2, 4), float(i), device="cuda")
                labels = torch.full((2, 4), i, dtype=torch.long, device="cuda")
            expected.append(torch.full((2, 4), float(i)))
            expected_bytes += logits.numel() * logits.element_size()
            expected_bytes += labels.numel() * labels.element_size()
            acc.add(
                loss=None,
                logits=logits,
                labels=labels,
                inputs=None,
                batch_size=2,
            )

        predictions, label_ids, _, _ = acc.finalize()
        metrics = telemetry.to_dict()

        assert acc._transfer_pipeline.max_pending_observed <= 2
        assert all(
            not chunk.is_pinned() for chunk in acc._cold_logits if chunk is not None
        )
        torch.testing.assert_close(
            torch.from_numpy(predictions), torch.cat(expected, dim=0)
        )
        assert label_ids.shape == (12, 4)
        assert metrics["transfer_bytes"] == expected_bytes
        assert metrics["transfer_time_sec"] >= 0.0
        assert 0.0 <= metrics["transfer_overlap_ratio"] <= 1.0


# ---------------------------------------------------------------------------
# with_metric_prefix
# ---------------------------------------------------------------------------


class TestWithMetricPrefix:
    def test_adds_missing_prefix(self):
        out = with_metric_prefix({"acc": 0.9, "f1": 0.8}, "eval")
        assert out == {"eval_acc": 0.9, "eval_f1": 0.8}

    def test_preserves_existing_prefix(self):
        out = with_metric_prefix({"eval_acc": 0.9, "f1": 0.8}, "eval")
        assert out == {"eval_acc": 0.9, "eval_f1": 0.8}

    def test_empty_prefix_is_passthrough(self):
        out = with_metric_prefix({"acc": 0.9}, "")
        assert out == {"acc": 0.9}


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_eval_prediction_accepts_all_fields(self):
        ep = EvalPrediction(
            predictions=torch.zeros(2, 3),
            label_ids=torch.zeros(2, 3, dtype=torch.long),
            inputs={"input_ids": torch.zeros(2, dtype=torch.long)},
            losses=torch.zeros(1),
        )
        assert ep.predictions is not None
        assert ep.label_ids is not None
        # ``inputs`` and ``losses`` are positional/keyword fields across HF
        # versions; accessing them must not raise.
        assert getattr(ep, "inputs", None) is not None
        assert getattr(ep, "losses", None) is not None

    def test_evaluation_result_constructible(self):
        out = EvaluationResult(
            predictions=None,
            label_ids=None,
            metrics={"eval_loss": 0.0},
            num_samples=0,
        )
        assert out.metrics == {"eval_loss": 0.0}
