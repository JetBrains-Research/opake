"""Tests for ``TrainerState`` JSON round-trip."""

from __future__ import annotations

import json

from opaque.api.transformers.trainer._state import TrainerState


class TestTrainerStateJson:
    def test_default_roundtrip(self):
        st = TrainerState()
        encoded = json.dumps(st.to_json())
        restored = TrainerState.from_json(json.loads(encoded))
        assert restored == st

    def test_populated_roundtrip(self):
        st = TrainerState(
            global_step=42,
            max_steps=100,
            epoch=1.5,
            log_history=[{"loss": 0.7, "step": 10}, {"loss": 0.6, "step": 20}],
            logging_steps=10,
            eval_steps=20,
            save_steps=20,
            best_metric=0.6,
            best_global_step=20,
            best_model_checkpoint="/tmp/output/checkpoint-20",
        )
        encoded = json.dumps(st.to_json())
        restored = TrainerState.from_json(json.loads(encoded))
        assert restored == st

    def test_unknown_keys_ignored(self):
        st = TrainerState(global_step=5)
        data = st.to_json()
        data["future_field"] = "ignored"
        restored = TrainerState.from_json(data)
        assert restored == st

    def test_best_fields_default_none(self):
        st = TrainerState()
        assert st.best_metric is None
        assert st.best_global_step is None
        assert st.best_model_checkpoint is None
