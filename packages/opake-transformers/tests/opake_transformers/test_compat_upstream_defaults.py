"""Conversion of *pristine* upstream HF/TRL configs.

The other compat modules build their sources through fixtures that pin a
handful of fields. That hides a whole class of breakage: a field the user
never touched blocking the conversion outright. These tests construct
configs with nothing set beyond what a CPU test host requires, so an
upstream default that opake's manifest doesn't classify shows up here.
"""

from __future__ import annotations

import pytest

trl = pytest.importorskip("trl")

from opake.transformers.trl import DPOConfig, SFTConfig  # noqa: E402

# ``bf16`` resolves to True on a pristine TRL config even with
# ``use_cpu=True``, and opake's precision check rejects that on a host
# without a bf16 accelerator. That check is unrelated to what these tests
# cover, so keep it out of the way.
_HOST = {"bf16": False, "use_cpu": True}

_FLAVORS = (
    pytest.param(trl.SFTConfig, SFTConfig, id="sft"),
    pytest.param(trl.DPOConfig, DPOConfig, id="dpo"),
)


@pytest.mark.parametrize(("trl_cls", "opake_cls"), _FLAVORS)
def test_pristine_upstream_config_converts(tmp_path, trl_cls, opake_cls):
    """A default upstream config converts — no field the user never set blocks it."""
    cfg = opake_cls.from_trl(
        trl_cls(output_dir=str(tmp_path), **_HOST),
        privacy_noise_multiplier=0.8,
    )
    assert cfg is not None


def _fractional_warmup_config(trl_cls, tmp_path):
    """Build a config with a fractional warmup, or skip where upstream refuses.

    Transformers only started reading ``warmup_steps`` as a fraction in 5.x;
    4.x rejects a non-integer outright, so there is nothing to translate.
    """
    try:
        return trl_cls(output_dir=str(tmp_path), warmup_steps=0.05, **_HOST)
    except ValueError:
        pytest.skip("upstream treats warmup_steps as an integer step count only")


@pytest.mark.parametrize(("trl_cls", "opake_cls"), _FLAVORS)
def test_fractional_warmup_steps_convert(tmp_path, trl_cls, opake_cls):
    """A fractional ``warmup_steps`` means 5% of training, and stays that."""
    cfg = opake_cls.from_trl(
        _fractional_warmup_config(trl_cls, tmp_path),
        privacy_noise_multiplier=0.8,
    )
    assert cfg.warmup_steps == pytest.approx(0.05)


@pytest.mark.parametrize(("trl_cls", "opake_cls"), _FLAVORS)
def test_absolute_warmup_steps_stay_absolute(tmp_path, trl_cls, opake_cls):
    """``warmup_steps >= 1`` is a step count and must not become a fraction."""
    cfg = opake_cls.from_trl(
        trl_cls(output_dir=str(tmp_path), warmup_steps=25, **_HOST),
        privacy_noise_multiplier=0.8,
    )
    assert cfg.warmup_steps == 25


@pytest.mark.parametrize(("trl_cls", "opake_cls"), _FLAVORS)
def test_legacy_warmup_ratio_converts_to_warmup_steps(tmp_path, trl_cls, opake_cls):
    """Transformers 4.x's ``warmup_ratio`` lands on opake's ``warmup_steps``.

    Opake follows 5.x, where a value below 1 already *is* the fractional
    encoding, so the 4.x ratio carries across untouched rather than being
    dropped as an unsupported alias.
    """
    if "warmup_ratio" not in trl_cls.__dataclass_fields__:
        pytest.skip("upstream dropped warmup_ratio in favour of a float warmup_steps")
    cfg = opake_cls.from_trl(
        trl_cls(output_dir=str(tmp_path), warmup_ratio=0.05, **_HOST),
        privacy_noise_multiplier=0.8,
    )
    assert cfg.warmup_steps == pytest.approx(0.05)
