"""Regression tests for structurally fixed sync(aux) collectives.

An empty Poisson batch on one rank used to take a different all-reduce
schedule than non-empty ranks (one ``reduce_scalar`` vs two), permanently
desynchronizing the process group. These tests assert the collective
sequence is identical regardless of local batch size, without needing a
multi-GPU process group.
"""

from __future__ import annotations

from dataclasses import fields

import torch

from opake.api.engine.clipping._clipped_fun import ClippedFunAux
from opake.api.engine.clipping._clipped_grad import ClippedGradAux
from opake.api.engine.clipping._distributed import (
    _SCALAR_AUX_FIELDS,
    _merge_gathered_values,
    _split_aux_fields,
    _sync_clipping_rate,
    sync_clipped_fun_aux,
    sync_clipped_grad_aux,
)


def _recording_reduce(calls: list[tuple[float, str]]):
    def _reduce(value: float, op: str = "mean", device=None) -> float:
        calls.append((float(value), op))
        return float(value)

    return _reduce


class TestSyncClippingRateCollectiveSequence:
    def test_empty_and_nonempty_issue_same_reduce_count(self, monkeypatch):
        import opake.api.engine.clipping._distributed as dist_mod

        empty_calls: list[tuple[float, str]] = []
        monkeypatch.setattr(dist_mod, "reduce_scalar", _recording_reduce(empty_calls))
        empty_rate = _sync_clipping_rate(0.0, torch.empty(0))

        nonempty_calls: list[tuple[float, str]] = []
        monkeypatch.setattr(
            dist_mod, "reduce_scalar", _recording_reduce(nonempty_calls)
        )
        nonempty_rate = _sync_clipping_rate(0.5, torch.ones(4))

        assert len(empty_calls) == len(nonempty_calls) == 4
        assert [op for _, op in empty_calls] == ["min", "max", "sum", "sum"]
        assert [op for _, op in nonempty_calls] == ["min", "max", "sum", "sum"]
        assert empty_rate == 0.0
        assert nonempty_rate == 0.5

    def test_empty_contributes_zero_weight(self, monkeypatch):
        import opake.api.engine.clipping._distributed as dist_mod

        calls: list[tuple[float, str]] = []
        monkeypatch.setattr(dist_mod, "reduce_scalar", _recording_reduce(calls))
        _sync_clipping_rate(0.0, torch.empty(0))

        assert calls[0] == (1.0, "min")  # clipping-rate presence
        assert calls[1] == (1.0, "max")  # clipping-rate presence
        assert calls[2] == (0.0, "sum")  # rate * n
        assert calls[3] == (0.0, "sum")  # n

    def test_none_rate_uses_presence_collectives(self, monkeypatch):
        import opake.api.engine.clipping._distributed as dist_mod

        calls: list[tuple[float, str]] = []
        monkeypatch.setattr(dist_mod, "reduce_scalar", _recording_reduce(calls))
        assert _sync_clipping_rate(None, torch.empty(0)) is None
        assert calls == [(0.0, "min"), (0.0, "max")]


class TestSplitAuxFieldsSchema:
    def test_none_tensor_fields_stay_in_gather_map(self):
        aux = ClippedGradAux(
            loss_values=None,
            grad_norms=torch.empty(0),
            clipped_grad_norms=torch.empty(0),
            loss_aux=None,
            clipping_rate=0.0,
            batch_size=0,
            group_norms=None,
        )
        tensor_fields, scalar_fields = _split_aux_fields(aux, ClippedGradAux)

        assert set(scalar_fields) == _SCALAR_AUX_FIELDS
        assert "loss_values" in tensor_fields
        assert tensor_fields["loss_values"] is None
        assert "group_norms" in tensor_fields
        assert tensor_fields["group_norms"] is None
        assert "grad_norms" in tensor_fields
        assert set(tensor_fields) | set(scalar_fields) == {f.name for f in fields(aux)}

    def test_empty_and_nonempty_grad_aux_share_field_partition(self):
        empty = ClippedGradAux(
            loss_values=torch.empty(0),
            grad_norms=torch.empty(0),
            clipped_grad_norms=torch.empty(0),
            loss_aux=None,
            clipping_rate=0.0,
            batch_size=0,
        )
        nonempty = ClippedGradAux(
            loss_values=torch.ones(3),
            grad_norms=torch.ones(3),
            clipped_grad_norms=torch.ones(3),
            loss_aux=None,
            clipping_rate=0.25,
            batch_size=3,
        )
        empty_t, empty_s = _split_aux_fields(empty, ClippedGradAux)
        nonempty_t, nonempty_s = _split_aux_fields(nonempty, ClippedGradAux)
        assert set(empty_t) == set(nonempty_t)
        assert set(empty_s) == set(nonempty_s)


class TestMergeGatheredValues:
    def test_none_plus_per_group_dict_uses_optree_unflatten(self):
        device = torch.device("cpu")
        merged = _merge_gathered_values(
            [
                None,
                {
                    "attn": torch.tensor([0.5, 1.5]),
                    "mlp": torch.tensor([0.2, 0.3]),
                },
            ],
            device,
        )
        assert set(merged) == {"attn", "mlp"}
        assert torch.equal(merged["attn"], torch.tensor([0.5, 1.5]))
        assert torch.equal(merged["mlp"], torch.tensor([0.2, 0.3]))

    def test_matching_dicts_concatenate_leaves(self):
        merged = _merge_gathered_values(
            [
                {"attn": torch.tensor([1.0]), "mlp": torch.tensor([2.0])},
                {"attn": torch.tensor([3.0]), "mlp": torch.tensor([4.0])},
            ],
            torch.device("cpu"),
        )
        assert torch.equal(merged["attn"], torch.tensor([1.0, 3.0]))
        assert torch.equal(merged["mlp"], torch.tensor([2.0, 4.0]))

    def test_nested_none_placeholders_are_preserved(self):
        merged = _merge_gathered_values(
            [
                {"attn": torch.tensor([1.0]), "skip": None, "mlp": torch.tensor([2.0])},
                {"attn": torch.tensor([3.0]), "skip": None, "mlp": torch.tensor([4.0])},
            ],
            torch.device("cpu"),
        )
        assert merged["skip"] is None
        assert torch.equal(merged["attn"], torch.tensor([1.0, 3.0]))
        assert torch.equal(merged["mlp"], torch.tensor([2.0, 4.0]))

    def test_structure_mismatch_among_nonempty_raises(self):
        import pytest

        with pytest.raises(TypeError, match="matching pytree structures"):
            _merge_gathered_values(
                [
                    {"attn": torch.tensor([1.0])},
                    {"attn": torch.tensor([1.0]), "mlp": torch.tensor([2.0])},
                ],
                torch.device("cpu"),
            )

    def test_non_tensor_leaf_fails_closed(self):
        import pytest

        with pytest.raises(TypeError, match="tensor leaves only"):
            _merge_gathered_values(
                [{"tag": "rank0"}, {"tag": "rank1"}],
                torch.device("cpu"),
            )


class TestGatherAuxFieldDevice:
    def test_none_local_field_uses_sibling_tensor_device(self, monkeypatch):
        import opake.api.engine.clipping._distributed as dist_mod

        cuda = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        remote_group_norms = {
            "attn": torch.tensor([1.0]),
            "mlp": torch.tensor([2.0]),
        }

        monkeypatch.setattr(dist_mod, "get_world_size", lambda: 2)

        def _fake_all_gather(payloads, local):
            payloads[0] = local
            if local is None:
                payloads[1] = remote_group_norms
            elif isinstance(local, torch.Tensor):
                payloads[1] = local.detach().cpu().clone()
            else:
                payloads[1] = local

        monkeypatch.setattr(dist_mod.dist, "all_gather_object", _fake_all_gather)

        fields = {
            "grad_norms": torch.tensor([0.5], device=cuda),
            "group_norms": None,
        }
        gathered = dist_mod._gather_aux_fields(fields)
        assert gathered["group_norms"]["attn"].device == cuda
        assert gathered["group_norms"]["mlp"].device == cuda


class TestSyncAuxCollectiveParity:
    def test_empty_and_nonempty_grad_aux_same_reduce_schedule(self, monkeypatch):
        import opake.api.engine.clipping._distributed as dist_mod

        monkeypatch.setattr(dist_mod, "is_distributed", lambda: True)
        monkeypatch.setattr(dist_mod, "_gather_aux_fields", lambda tree: tree)

        def _run(aux: ClippedGradAux) -> list[tuple[float, str]]:
            calls: list[tuple[float, str]] = []
            monkeypatch.setattr(dist_mod, "reduce_scalar", _recording_reduce(calls))
            sync_clipped_grad_aux(aux)
            return calls

        empty = ClippedGradAux(
            loss_values=torch.empty(0),
            grad_norms=torch.empty(0),
            clipped_grad_norms=torch.empty(0),
            loss_aux=None,
            clipping_rate=0.0,
            batch_size=0,
        )
        nonempty = ClippedGradAux(
            loss_values=torch.tensor([1.0, 2.0]),
            grad_norms=torch.tensor([0.5, 1.5]),
            clipped_grad_norms=torch.tensor([0.5, 1.0]),
            loss_aux=None,
            clipping_rate=0.5,
            batch_size=2,
        )

        empty_ops = [op for _, op in _run(empty)]
        nonempty_ops = [op for _, op in _run(nonempty)]
        # clipping-rate presence: min/max, weighted rate: 2× sum, batch-size: sum
        assert empty_ops == nonempty_ops == ["min", "max", "sum", "sum", "sum"]

    def test_empty_and_nonempty_fun_aux_same_reduce_schedule(self, monkeypatch):
        import opake.api.engine.clipping._distributed as dist_mod

        monkeypatch.setattr(dist_mod, "is_distributed", lambda: True)
        monkeypatch.setattr(dist_mod, "_gather_aux_fields", lambda tree: tree)

        def _run(aux: ClippedFunAux) -> list[str]:
            calls: list[tuple[float, str]] = []
            monkeypatch.setattr(dist_mod, "reduce_scalar", _recording_reduce(calls))
            sync_clipped_fun_aux(aux)
            return [op for _, op in calls]

        empty = ClippedFunAux(
            values=torch.empty(0),
            norms=torch.empty(0),
            clipped_norms=torch.empty(0),
            value_aux=None,
            clipping_rate=0.0,
            batch_size=0,
        )
        nonempty = ClippedFunAux(
            values=torch.ones(2),
            norms=torch.ones(2),
            clipped_norms=torch.ones(2),
            value_aux=None,
            clipping_rate=1.0,
            batch_size=2,
        )
        assert _run(empty) == _run(nonempty) == ["min", "max", "sum", "sum", "sum"]
