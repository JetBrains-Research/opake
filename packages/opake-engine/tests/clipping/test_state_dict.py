"""Round-trip tests for ``state_dict`` / ``from_state_dict`` on ``PerGroup``."""

from __future__ import annotations

import json

from opake.serialization import from_state_dict, state_dict
from opake.types import PerGroup


class TestPerGroupStateDict:
    def test_roundtrip_via_json(self):
        pg = PerGroup(
            groups={"layer.0.q": "attn", "layer.0.mlp": "mlp"},
            values={"attn": 1.5, "mlp": 0.75},
        )
        sd = state_dict(pg)
        # JSON-serializable
        decoded = json.loads(json.dumps(sd))
        # Template-driven restore — same shape of groups/values keys.
        template = PerGroup(
            groups={"layer.0.q": "", "layer.0.mlp": ""},
            values={"attn": 0.0, "mlp": 0.0},
        )
        restored = from_state_dict(template, decoded)
        assert restored == pg

    def test_state_dict_flat_dotted_keys(self):
        pg = PerGroup(groups={"a": "g"}, values={"g": 2.0})
        sd = state_dict(pg)
        # Path tuples stringify under the structural walker.
        assert sd == {"groups.('a',)": "g", "values.g": 2.0}

    def test_roundtrip_preserves_float_values(self):
        pg = PerGroup(groups={"a": "g"}, values={"g": 0.5})
        sd = state_dict(pg)
        template = PerGroup(groups={"a": ""}, values={"g": 0.0})
        restored = from_state_dict(template, sd)
        assert isinstance(restored.values["g"], float)
        assert restored.values["g"] == 0.5
