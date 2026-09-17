"""trl.DPOConfig → opaque DPOConfig manifest + converter.

TRL-specific fields only; the HF-inherited subset is delegated to the HF
manifest by ``_convert_trl_config``. Every TRL ``DPOConfig`` field appears in
exactly one bucket, enforced by ``test_compat_manifest_exhaustive.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opaque.api.transformers.trainer._convert import _normalize_dp_overrides
from opaque.exceptions import ConfigurationError, InputTypeError

from ._convert import (
    _convert_trl_config,
    _import_trl,
    _reject_if_truthy,
    _reject_pad_token,
    _reject_truncation_mode,
)

if TYPE_CHECKING:
    from collections.abc import Callable

TRL_DPO_DIRECT_FIELDS: frozenset[str] = frozenset(
    {
        "router_aux_loss_coef",
        "model_init_kwargs",
        "trust_remote_code",
        "disable_dropout",
        "dataset_num_proc",
        "max_length",
        "pad_to_multiple_of",
        "precompute_ref_batch_size",
        "beta",
        "label_smoothing",
        "loss_weights",
        "f_divergence_type",
        "f_alpha_divergence_coef",
        "ld_alpha",
        "use_weighting",
        "discopop_tau",
        "sync_ref_model",
        "ref_model_mixup_alpha",
        "ref_model_sync_steps",
        # On TRL's DPOConfig but not on HF base TrainingArguments.
        "activation_offloading",
        # SimPO / CPO / ORPO head-specific tunables (same names/semantics here).
        "simpo_gamma",
        "cpo_alpha",
        "orpo_lambda",
    }
)


TRL_DPO_RENAME_MAP: dict[str, str] = {}


# Opaque's implemented DPO heads (opaque's own naming convention).
_OPAQUE_DPO_LOSS_TYPES = frozenset(
    {
        "sigmoid",
        "hinge",
        "ipo",
        "robust",
        "exo_pair",
        "nca_pair",
        "bco_pair",
        "sppo_hard",
        "apo_zero",
        "apo_down",
        "discopop",
        "chosen_nll",
        "sigmoid_norm",
        # CPO / ORPO / SimPO are assembled specially but are accepted
        # ``DPOConfig.loss_type`` values.
        "cpo",
        "orpo",
        "simpo",
    }
)

# TRL head name → opaque head name, where opaque pursues its own (clearer)
# convention. TRL's ``"sft"`` is the chosen-completion NLL regulariser, which
# opaque calls ``"chosen_nll"``. The converter translates on the way in.
_TRL_TO_OPAQUE_LOSS_TYPE = {"sft": "chosen_nll"}


def _loss_type_transform(trl: dict[str, Any]) -> dict[str, Any]:
    """Translate + validate every entry in ``loss_type`` against opaque's heads.

    TRL's ``"sft"`` head is renamed to opaque's ``"chosen_nll"`` (see
    ``_TRL_TO_OPAQUE_LOSS_TYPE``). Adversarial Optimal Transport heads (``aot``,
    ``aot_unpaired``), which opaque does not implement, are rejected.
    """
    loss_type = trl.get("loss_type")
    if loss_type is None:
        return {}
    # TRL stores loss_type as list[str] in 1.x; coerce singletons.
    values = [loss_type] if isinstance(loss_type, str) else list(loss_type)
    mapped = [_TRL_TO_OPAQUE_LOSS_TYPE.get(v, v) for v in values]
    unsupported = [v for v in mapped if v not in _OPAQUE_DPO_LOSS_TYPES]
    if unsupported:
        raise ConfigurationError(
            *(
                f"trl_dpo_config.loss_type contains unsupported heads: "
                f"{sorted(set(unsupported))}. Opaque implements: "
                f"{sorted(_OPAQUE_DPO_LOSS_TYPES)}. The Adversarial Optimal "
                f"Transport family (``aot``, ``aot_unpaired``) added in TRL 1.x "
                f"is not in opaque.",
            )
        )
    return {"loss_type": mapped}


TRL_DPO_TRANSFORM_MAP: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "loss_type": _loss_type_transform,
}


TRL_DPO_REJECTED_FIELDS: dict[str, Callable[[Any], str | None]] = {
    "truncation_mode": _reject_truncation_mode,
    "padding_free": _reject_if_truthy(
        "``padding_free`` is not supported for DPO — see the SFT rationale."
    ),
    # ``precompute_ref_log_probs`` is not rejected here: opaque always
    # precomputes, matching TRL's True mode; a False setting is caught by the
    # trainer-side runtime check.
    "pad_token": _reject_pad_token,
}


TRL_DPO_DROP_FIELDS: dict[str, str | Callable[[Any], str | None]] = {
    "precompute_ref_log_probs": (
        "Opaque always precomputes reference logps under DP for static-"
        "reference heads; this TRL flag is silently honored at its True "
        "mode and ignored otherwise."
    ),
}


def _convert_trl_dpo_config(
    trl_cfg: Any,
    *,
    strict: bool = True,
    **dp_overrides: Any,
) -> dict[str, Any]:
    """Translate a ``trl.DPOConfig`` instance into opaque ``DPOConfig`` kwargs."""
    trl = _import_trl()
    if not isinstance(trl_cfg, trl.DPOConfig):
        raise InputTypeError(
            *(f"Expected ``trl.DPOConfig`` instance, got {type(trl_cfg).__name__}.",)
        )

    dp_layer = _normalize_dp_overrides(dp_overrides)
    return _convert_trl_config(
        trl_cfg,
        trl_direct=TRL_DPO_DIRECT_FIELDS,
        trl_rename=TRL_DPO_RENAME_MAP,
        trl_transform=TRL_DPO_TRANSFORM_MAP,
        trl_reject=TRL_DPO_REJECTED_FIELDS,
        trl_drop=TRL_DPO_DROP_FIELDS,
        source_label="trl_dpo_config",
        strict=strict,
        dp_overrides=dp_layer,
    )
