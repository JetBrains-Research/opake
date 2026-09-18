"""Tests for ``opake.transformers.trl.DPOConfig.from_trl``."""

from __future__ import annotations

import warnings

import pytest

# TRL is the optional ``opake[trl]`` extra.
trl = pytest.importorskip("trl")
_DPO_FIELDS = trl.DPOConfig.__dataclass_fields__

from opake.transformers.trl import DPOConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _trl_dpo_args(tmp_path, **overrides):
    return trl.DPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=overrides.pop("per_device_train_batch_size", 8),
        learning_rate=overrides.pop("learning_rate", 1e-6),
        max_steps=overrides.pop("max_steps", 10),
        seed=overrides.pop("seed", 42),
        save_strategy="no",
        report_to=[],
        use_cpu=True,
        **overrides,
    )


# ---------------------------------------------------------------------------
# Required DP knobs
# ---------------------------------------------------------------------------


def test_missing_dp_knob_raises(tmp_path):
    with pytest.raises(ValueError, match="privacy_noise_multiplier"):
        DPOConfig.from_trl(_trl_dpo_args(tmp_path))


def test_typeerror_on_wrong_input_type(tmp_path):
    with pytest.raises(TypeError, match="DPOConfig"):
        DPOConfig.from_trl(
            {"beta": 0.1},
            privacy_noise_multiplier=0.8,
            clipping_norm=1.0,
        )


# ---------------------------------------------------------------------------
# DIRECT — TRL DPO-specific fields carry through
# ---------------------------------------------------------------------------


def test_beta_carries_through(tmp_path):
    cfg = DPOConfig.from_trl(
        _trl_dpo_args(tmp_path, beta=0.05),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert cfg.beta == 0.05


def test_label_smoothing_carries_through(tmp_path):
    cfg = DPOConfig.from_trl(
        _trl_dpo_args(tmp_path, label_smoothing=0.1, loss_type=["robust"]),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert cfg.label_smoothing == 0.1


def test_sync_ref_model_carries_through(tmp_path):
    cfg = DPOConfig.from_trl(
        _trl_dpo_args(
            tmp_path,
            sync_ref_model=True,
            ref_model_mixup_alpha=0.6,
            ref_model_sync_steps=16,
        ),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert cfg.sync_ref_model is True
    assert cfg.ref_model_mixup_alpha == 0.6
    assert cfg.ref_model_sync_steps == 16


# ---------------------------------------------------------------------------
# TRANSFORM — loss_type validation
# ---------------------------------------------------------------------------


def test_loss_type_sigmoid_passes(tmp_path):
    cfg = DPOConfig.from_trl(
        _trl_dpo_args(tmp_path, loss_type=["sigmoid"]),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert cfg.loss_type == ["sigmoid"]


def test_loss_type_mpo_blend_passes(tmp_path):
    # TRL's ``sft`` head is translated to opake's ``chosen_nll`` convention.
    cfg = DPOConfig.from_trl(
        _trl_dpo_args(tmp_path, loss_type=["sigmoid", "sft"], loss_weights=[1.0, 0.5]),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert cfg.loss_type == ["sigmoid", "chosen_nll"]
    assert cfg.loss_weights == [1.0, 0.5]


def test_loss_type_aot_rejected(tmp_path):
    """TRL 1.x added ``aot``; opake doesn't implement it."""
    with pytest.raises(ValueError, match="aot"):
        DPOConfig.from_trl(
            _trl_dpo_args(tmp_path, loss_type=["aot"]),
            privacy_noise_multiplier=0.8,
            clipping_norm=1.0,
        )


@pytest.mark.skipif(
    "router_aux_loss_coef" not in _DPO_FIELDS,
    reason="TRL does not expose router_aux_loss_coef",
)
def test_router_aux_loss_is_dropped_with_a_warning(tmp_path):
    """A deliberately set MoE aux-loss coefficient converts, loudly, to no-op."""
    with pytest.warns(RuntimeWarning, match="router_aux_loss_coef"):
        cfg = DPOConfig.from_trl(
            _trl_dpo_args(tmp_path, router_aux_loss_coef=0.5),
            privacy_noise_multiplier=0.8,
            clipping_norm=1.0,
        )
    assert cfg is not None


@pytest.mark.skipif(
    "router_aux_loss_coef" not in _DPO_FIELDS,
    reason="TRL does not expose router_aux_loss_coef",
)
def test_trl_default_router_aux_loss_is_dropped_silently(tmp_path):
    """TRL's own non-zero default is not a user request, so it warns about nothing."""
    default_coef = trl.DPOConfig.__dataclass_fields__["router_aux_loss_coef"].default
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = DPOConfig.from_trl(
            _trl_dpo_args(tmp_path, router_aux_loss_coef=default_coef),
            privacy_noise_multiplier=0.8,
            clipping_norm=1.0,
        )
    assert cfg is not None
    assert not [w for w in caught if "router_aux_loss_coef" in str(w.message)]


@pytest.mark.skipif(
    "router_aux_loss_coef" not in _DPO_FIELDS,
    reason="TRL does not expose router_aux_loss_coef",
)
def test_router_aux_loss_switched_off_is_dropped_silently(tmp_path):
    """Asking for no aux loss gets no aux loss, so there is nothing to warn about."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = DPOConfig.from_trl(
            _trl_dpo_args(tmp_path, router_aux_loss_coef=0.0),
            privacy_noise_multiplier=0.8,
            clipping_norm=1.0,
        )
    assert cfg is not None
    assert not [w for w in caught if "router_aux_loss_coef" in str(w.message)]


@pytest.mark.skipif(
    "trust_remote_code" not in _DPO_FIELDS,
    reason="TRL does not expose trust_remote_code",
)
def test_trust_remote_code_overrides_model_init_kwargs(tmp_path):
    cfg = DPOConfig.from_trl(
        _trl_dpo_args(
            tmp_path,
            trust_remote_code=True,
            model_init_kwargs={"trust_remote_code": False, "revision": "main"},
        ),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert cfg.trust_remote_code is True
    assert cfg.model_init_kwargs["trust_remote_code"] is True
    assert cfg.model_init_kwargs["revision"] == "main"


def test_loss_type_aot_unpaired_rejected(tmp_path):
    with pytest.raises(ValueError, match="aot_unpaired"):
        DPOConfig.from_trl(
            _trl_dpo_args(tmp_path, loss_type=["aot_unpaired"]),
            privacy_noise_multiplier=0.8,
            clipping_norm=1.0,
        )


# ---------------------------------------------------------------------------
# REJECT_IF_SET — TRL DPO fields opake doesn't implement
# ---------------------------------------------------------------------------


def test_reject_padding_free(tmp_path):
    with pytest.raises(ValueError, match="padding_free"):
        DPOConfig.from_trl(
            _trl_dpo_args(tmp_path, padding_free=True),
            privacy_noise_multiplier=0.8,
            clipping_norm=1.0,
        )


def test_reject_truncation_mode_keep_end(tmp_path):
    with pytest.raises(ValueError, match="keep_start"):
        DPOConfig.from_trl(
            _trl_dpo_args(tmp_path, truncation_mode="keep_end"),
            privacy_noise_multiplier=0.8,
            clipping_norm=1.0,
        )


# ---------------------------------------------------------------------------
# HF base translation inherited
# ---------------------------------------------------------------------------


def test_inherits_hf_base_batch_collapse(tmp_path):
    cfg = DPOConfig.from_trl(
        _trl_dpo_args(
            tmp_path,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
        ),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert cfg.per_device_train_batch_size == 8
    assert cfg.microbatch_size == 2


def test_inherits_hf_base_rejection_of_fp16(tmp_path):
    with pytest.raises(ValueError, match="bf16"):
        DPOConfig.from_trl(
            _trl_dpo_args(tmp_path, fp16=True),
            privacy_noise_multiplier=0.8,
            clipping_norm=1.0,
        )


def test_inherits_max_grad_norm_to_clipping_norm(tmp_path):
    """HF ``max_grad_norm`` flows through the base manifest to opake
    ``clipping_norm`` when no explicit ``clipping_norm`` override is given."""
    cfg = DPOConfig.from_trl(
        _trl_dpo_args(tmp_path, max_grad_norm=0.5),
        privacy_noise_multiplier=0.8,
    )
    assert cfg.clipping_norm == 0.5


def test_inherits_perf_kernels_off_by_default(tmp_path):
    """Converting a TRL config leaves perf-kernels OFF to match upstream,
    even though opake's own DPO default is True. A name override wins."""
    cfg = DPOConfig.from_trl(
        _trl_dpo_args(tmp_path),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert cfg.use_performance_kernels is False
    cfg2 = DPOConfig.from_trl(
        _trl_dpo_args(tmp_path),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
        use_performance_kernels=True,
    )
    assert cfg2.use_performance_kernels is True


def test_inherits_liger_to_perf_kernels(tmp_path):
    """HF ``use_liger_kernel`` flows through to opake
    ``use_performance_kernels=True``."""
    cfg = DPOConfig.from_trl(
        _trl_dpo_args(tmp_path, use_liger_kernel=True),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert cfg.use_performance_kernels is True
