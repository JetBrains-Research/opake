"""Tests for ``TrainingArguments.from_hf`` — HF → opake conversion."""

from __future__ import annotations

import dataclasses
import math
import multiprocessing
import warnings

import pytest

from opake.api.transformers.trainer import TrainingArguments
from opake.api.transformers.trainer._convert import (
    _apply_manifest,
    _is_default,
    _normalize_dp_overrides,
)
from opake.api.transformers.trainer._hf_convert import _warmup_collapse
from opake.api.transformers.trainer._scheduler import get_warmup_steps

# ``transformers`` is a required dep of opake-transformers.
hf = pytest.importorskip("transformers")
_HF_FIELDS = hf.TrainingArguments.__dataclass_fields__


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _hf_args(tmp_path, **overrides):
    """Construct an HF TrainingArguments with sensible defaults for tests."""
    return hf.TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=overrides.pop("per_device_train_batch_size", 8),
        learning_rate=overrides.pop("learning_rate", 1e-4),
        max_steps=overrides.pop("max_steps", 10),
        seed=overrides.pop("seed", 42),
        save_strategy="no",
        report_to=[],
        **overrides,
    )


def _convert(tmp_path, **dp_overrides):
    """Helper: minimal default ``from_hf`` invocation."""
    return TrainingArguments.from_hf(
        _hf_args(tmp_path),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
        **dp_overrides,
    )


def test_is_default_does_not_pass_dataclass_classes_to_asdict(monkeypatch):
    @dataclasses.dataclass
    class DefaultValue:
        pass

    @dataclasses.dataclass
    class ActualValue:
        pass

    @dataclasses.dataclass
    class Config:
        value: type = DefaultValue

    calls = []
    real_asdict = dataclasses.asdict

    def spy_asdict(value):
        calls.append(value)
        return real_asdict(value)

    monkeypatch.setattr(dataclasses, "asdict", spy_asdict)

    assert not _is_default(ActualValue, dataclasses.fields(Config)[0])
    assert calls == []


# ---------------------------------------------------------------------------
# Required DP knobs
# ---------------------------------------------------------------------------


def test_missing_dp_knob_raises(tmp_path):
    with pytest.raises(ValueError, match="privacy_noise_multiplier"):
        TrainingArguments.from_hf(_hf_args(tmp_path))


def test_either_noise_multiplier_or_target_epsilon_accepted(tmp_path):
    # noise_multiplier path
    opake = TrainingArguments.from_hf(
        _hf_args(tmp_path),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert opake.privacy_noise_multiplier == 0.8

    # target_epsilon path
    opake = TrainingArguments.from_hf(
        _hf_args(tmp_path),
        privacy_target_epsilon=8.0,
        clipping_norm=1.0,
    )
    assert opake.privacy_target_epsilon == 8.0


# ---------------------------------------------------------------------------
# DIRECT — copy as-is
# ---------------------------------------------------------------------------


def test_direct_field_learning_rate_carries_through(tmp_path):
    opake = TrainingArguments.from_hf(
        _hf_args(tmp_path, learning_rate=3e-4),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert opake.learning_rate == 3e-4


def test_direct_field_max_steps_carries_through(tmp_path):
    opake = TrainingArguments.from_hf(
        _hf_args(tmp_path, max_steps=123),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert opake.max_steps == 123


@pytest.mark.skipif(
    "dataloader_multiprocessing_context" not in _HF_FIELDS,
    reason="transformers does not expose dataloader_multiprocessing_context",
)
def test_dataloader_multiprocessing_context_carries_through(tmp_path):
    context = multiprocessing.get_all_start_methods()[0]
    opake = TrainingArguments.from_hf(
        _hf_args(
            tmp_path,
            dataloader_num_workers=1,
            dataloader_multiprocessing_context=context,
        ),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert opake.dataloader_multiprocessing_context == context


@pytest.mark.skipif(
    "dataloader_in_order" not in _HF_FIELDS,
    reason="transformers does not expose dataloader_in_order",
)
def test_dataloader_in_order_false_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="dataloader_in_order must be True"):
        TrainingArguments.from_hf(
            _hf_args(tmp_path, dataloader_in_order=False),
            privacy_noise_multiplier=0.8,
            clipping_norm=1.0,
        )


# bf16 needs a bf16-capable GPU: HF's TrainingArguments rejects ``bf16=True`` at
# construction on CPU/MPS runners, so this carry-through check can only exercise a
# real bf16 input on the CUDA lane.
@pytest.mark.cuda
def test_direct_field_bf16_carries_through(tmp_path):
    opake = TrainingArguments.from_hf(
        _hf_args(tmp_path, bf16=True),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert opake.bf16 is True


# ---------------------------------------------------------------------------
# RENAME
# ---------------------------------------------------------------------------


def test_evaluation_strategy_renamed_to_eval_strategy(tmp_path):
    # HF 4.41+ uses ``eval_strategy``; the rename is the inverse for old
    # configs. HF's TrainingArguments accepts both currently.
    opake = TrainingArguments.from_hf(
        _hf_args(tmp_path, eval_strategy="steps", eval_steps=5),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert opake.eval_strategy == "steps"
    assert opake.eval_steps == 5


# ---------------------------------------------------------------------------
# TRANSFORM — batch collapse
# ---------------------------------------------------------------------------


def test_batch_collapse_grad_accum_multiplies_into_logical_batch(tmp_path):
    """``(per_device=2, grad_accum=4)`` → ``(opake.per_device=8, microbatch=2)``."""
    opake = TrainingArguments.from_hf(
        _hf_args(
            tmp_path,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
        ),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert opake.per_device_train_batch_size == 8  # 2 × 4
    assert opake.microbatch_size == 2


def test_batch_no_collapse_when_grad_accum_is_one(tmp_path):
    """At grad_accum=1, logical batch == per_device and microbatch_size
    stays at its default (None → vmap over the full batch)."""
    opake = TrainingArguments.from_hf(
        _hf_args(
            tmp_path,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=1,
        ),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert opake.per_device_train_batch_size == 8
    assert opake.microbatch_size is None


def test_optim_adamw_torch_collapses_to_adamw(tmp_path):
    opake = TrainingArguments.from_hf(
        _hf_args(tmp_path, optim="adamw_torch"),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert opake.optim == "adamw"


def test_optim_adamw_torch_fused_collapses_to_adamw_with_warning(tmp_path):
    # opake's functional AdamW has no ``fused`` kernel arg, so the fused HF
    # optimizer must translate to plain adamw without forwarding fused — and
    # surface the dropped kernel request rather than silently rewrite (#389).
    with pytest.warns(RuntimeWarning, match="fused AdamW kernel"):
        opake = TrainingArguments.from_hf(
            _hf_args(tmp_path, optim="adamw_torch_fused"),
            privacy_noise_multiplier=0.8,
            clipping_norm=1.0,
        )
    assert opake.optim == "adamw"
    assert not opake.optim_args  # nothing forwarded (no fused flag)


# ---------------------------------------------------------------------------
# REJECT_IF_SET
# ---------------------------------------------------------------------------


def test_reject_fp16(tmp_path):
    with pytest.raises(ValueError, match="bf16"):
        TrainingArguments.from_hf(
            _hf_args(tmp_path, fp16=True),
            privacy_noise_multiplier=0.8,
            clipping_norm=1.0,
        )


def test_reject_neftune_noise_alpha(tmp_path):
    with pytest.raises(ValueError, match="NEFTune"):
        TrainingArguments.from_hf(
            _hf_args(tmp_path, neftune_noise_alpha=0.5),
            privacy_noise_multiplier=0.8,
            clipping_norm=1.0,
        )


def test_auto_find_batch_size_maps_to_microbatch(tmp_path):
    """HF ``auto_find_batch_size`` → opake ``auto_find_microbatch_size``."""
    opake = TrainingArguments.from_hf(
        _hf_args(tmp_path, auto_find_batch_size=True),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert opake.auto_find_microbatch_size is True


def test_max_grad_norm_maps_to_clipping_norm(tmp_path):
    """HF ``max_grad_norm`` loosely maps to opake ``clipping_norm`` (no DP
    override given, so it isn't overwritten)."""
    opake = TrainingArguments.from_hf(
        _hf_args(tmp_path, max_grad_norm=0.5),
        privacy_noise_multiplier=0.8,
    )
    assert opake.clipping_norm == 0.5


def test_liger_maps_to_performance_kernels(tmp_path):
    """HF ``use_liger_kernel`` → opake ``use_performance_kernels=True``."""
    opake = TrainingArguments.from_hf(
        _hf_args(tmp_path, use_liger_kernel=True),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert opake.use_performance_kernels is True


def test_performance_kernels_off_by_default_on_conversion(tmp_path):
    """Converting an HF config (no Liger) leaves perf-kernels OFF to match
    upstream, even though opake's own default is True."""
    opake = TrainingArguments.from_hf(
        _hf_args(tmp_path),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert opake.use_performance_kernels is False
    # ...but a name override wins.
    opake2 = TrainingArguments.from_hf(
        _hf_args(tmp_path),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
        use_performance_kernels=True,
    )
    assert opake2.use_performance_kernels is True


def test_reject_paged_optim(tmp_path):
    with pytest.raises(ValueError, match="paged"):
        TrainingArguments.from_hf(
            _hf_args(tmp_path, optim="paged_adamw_8bit"),
            privacy_noise_multiplier=0.8,
            clipping_norm=1.0,
        )


# ---------------------------------------------------------------------------
# DROP_WITH_WARN
# ---------------------------------------------------------------------------


def test_drop_do_train_warns_when_non_default(tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # HF defaults do_train=False; explicitly setting it to True triggers
        # the drop warning.
        TrainingArguments.from_hf(
            _hf_args(tmp_path, do_train=True),
            privacy_noise_multiplier=0.8,
            clipping_norm=1.0,
        )
    assert any(
        "do_train" in str(w.message) and issubclass(w.category, RuntimeWarning)
        for w in caught
    )


def test_drop_silent_in_lenient_mode(tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        TrainingArguments.from_hf(
            _hf_args(tmp_path, do_train=True),
            privacy_noise_multiplier=0.8,
            clipping_norm=1.0,
            strict=False,
        )
    # ``strict=False`` should suppress the drop warnings entirely.
    assert not any("do_train" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# DP overrides
# ---------------------------------------------------------------------------


def test_dp_overrides_layered_on_top(tmp_path):
    opake = TrainingArguments.from_hf(
        _hf_args(tmp_path),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
        privacy_noise_mechanism="gaussian",
        clipping_mode="fixed",
    )
    assert opake.privacy_noise_multiplier == 0.8
    assert opake.clipping_norm == 1.0
    assert opake.privacy_noise_mechanism == "gaussian"


def test_opake_overrides_win_over_hf_derived(tmp_path):
    # HF sets per_device=2, grad_accum=4 → opake.microbatch_size=2 via
    # transform. Then user passes microbatch_size=1 as an opake override,
    # which should win.
    opake = TrainingArguments.from_hf(
        _hf_args(
            tmp_path,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
        ),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
        microbatch_size=1,
    )
    assert opake.per_device_train_batch_size == 8  # still HF-derived
    assert opake.microbatch_size == 1  # opake override wins


# ---------------------------------------------------------------------------
# Type checks
# ---------------------------------------------------------------------------


def test_typeerror_on_non_hf_input(tmp_path):
    with pytest.raises(TypeError, match="TrainingArguments"):
        TrainingArguments.from_hf(
            {"learning_rate": 1e-4},
            privacy_noise_multiplier=0.8,
            clipping_norm=1.0,
        )


def test_normalize_dp_overrides_rejects_empty():
    with pytest.raises(ValueError, match="privacy_noise_multiplier"):
        _normalize_dp_overrides({})


def test_normalize_dp_overrides_accepts_noise_only():
    result = _normalize_dp_overrides({"privacy_noise_multiplier": 0.5})
    assert result["privacy_noise_multiplier"] == 0.5


def test_normalize_dp_overrides_accepts_epsilon_only():
    result = _normalize_dp_overrides({"privacy_target_epsilon": 8.0})
    assert result["privacy_target_epsilon"] == 8.0


# ---------------------------------------------------------------------------
# Manifest engine — fields no bucket claims
# ---------------------------------------------------------------------------


def _apply(values, defaults):
    """Drive the engine with an empty manifest, so no field is classified."""
    return _apply_manifest(
        source_values=values,
        source_defaults=defaults,
        direct=frozenset(),
        rename={},
        transform={},
        reject={},
        drop={},
        source_label="src",
        strict=True,
    )


def test_unclassified_field_at_default_is_skipped():
    """An untouched field carries no instruction to drop and cannot affect ε."""
    assert _apply({"novel_knob": 0.0}, {"novel_knob": 0.0}) == {}


def test_unclassified_none_default_is_skipped():
    """Upstream deprecating a field to a ``None`` placeholder must not break callers."""
    assert _apply({"novel_knob": None}, {"novel_knob": None}) == {}


def test_unclassified_field_the_user_set_raises():
    """Silently dropping a deliberately configured knob could invalidate accounting."""
    with pytest.raises(ValueError, match="not classified"):
        _apply({"novel_knob": 0.05}, {"novel_knob": 0.0})


def test_unclassified_field_with_unknown_default_raises():
    """No default to compare against means we cannot prove the user left it alone."""
    with pytest.raises(ValueError, match="not classified"):
        _apply({"novel_knob": 0.05}, {})


# ---------------------------------------------------------------------------
# Warmup: HF's two knobs (4.x) / one knob (5.x) → opake's warmup_steps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hf", "expected"),
    [
        ({}, 0),
        ({"warmup_steps": None}, 0),
        ({"warmup_steps": 0}, 0),
        ({"warmup_steps": 0.05}, 0.05),
        ({"warmup_steps": 25}, 25),
        # transformers 4.x shape: the fraction lived in its own field, and
        # opake's warmup_steps now reads exactly that encoding.
        ({"warmup_steps": 0, "warmup_ratio": 0.05}, 0.05),
        ({"warmup_steps": 0, "warmup_ratio": 0.0}, 0),
    ],
)
def test_warmup_collapse(hf, expected):
    """Both upstream shapes land on a single ``warmup_steps``, value unchanged."""
    assert _warmup_collapse(hf)["warmup_steps"] == pytest.approx(expected)


def test_warmup_collapse_prefers_steps_when_both_set():
    """Matches HF 4.x's ``get_warmup_steps``: a non-zero step count wins."""
    assert _warmup_collapse({"warmup_steps": 25, "warmup_ratio": 0.05}) == {
        "warmup_steps": 25
    }


@pytest.mark.parametrize("warmup", [0, 0.05, 0.5, 1, 25, 25.7])
def test_warmup_resolves_to_the_same_step_count_as_hf(warmup):
    """Opake's resolution of ``warmup_steps`` must agree with HF's, exactly."""
    total = 200
    expected = int(warmup) if warmup >= 1 else math.ceil(total * warmup)
    assert get_warmup_steps(total, warmup) == expected


def test_warmup_matches_upstream_get_warmup_steps(tmp_path):
    """Cross-check the resolution against the installed transformers itself."""
    total = 200
    try:
        hf_args = _hf_args(tmp_path, warmup_steps=0.05)
    except ValueError:
        pytest.skip("upstream treats warmup_steps as an integer step count only")
    collapsed = _warmup_collapse({"warmup_steps": hf_args.warmup_steps})
    assert get_warmup_steps(
        total, collapsed["warmup_steps"]
    ) == hf_args.get_warmup_steps(total)
