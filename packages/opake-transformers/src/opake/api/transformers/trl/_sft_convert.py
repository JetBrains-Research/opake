"""trl.SFTConfig → opake SFTConfig manifest + converter.

TRL-specific fields only; the HF-inherited subset is delegated to the HF
manifest by ``_convert_trl_config``. Every TRL ``SFTConfig`` field appears in
exactly one bucket, enforced by ``test_compat_manifest_exhaustive.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opake.api.transformers.trainer._convert import _normalize_dp_overrides
from opake.exceptions import InputTypeError

from ._convert import (
    _convert_trl_config,
    _drop_router_aux_loss,
    _import_trl,
    _reject_if_truthy,
    _reject_pad_token,
    _reject_truncation_mode,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# DIRECT — TRL field name matches opake, same semantics.
TRL_SFT_DIRECT_FIELDS: frozenset[str] = frozenset(
    {
        "model_init_kwargs",
        "trust_remote_code",
        "chat_template_path",
        "dataset_text_field",
        "dataset_num_proc",
        "eos_token",
        "max_length",
        "pad_to_multiple_of",
        "completion_only_loss",
        "assistant_only_loss",
        "loss_type",
        # On TRL's SFTConfig/DPOConfig but not on HF base ``TrainingArguments``;
        # opake's base has the same field with the same semantics.
        "activation_offloading",
    }
)


# RENAME — TRL SFT field name differs from opake.
TRL_SFT_RENAME_MAP: dict[str, str] = {}


# TRANSFORM — TRL SFT field requires a derivation step.
TRL_SFT_TRANSFORM_MAP: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}


# REJECT_IF_SET — TRL has the field but opake does not implement it.
TRL_SFT_REJECTED_FIELDS: dict[str, Callable[[Any], str | None]] = {
    "truncation_mode": _reject_truncation_mode,
    "packing": _reject_if_truthy(
        "Sequence packing breaks the fixed per-example batch shape DP-SGD's "
        "per-example vmap requires. Disable ``packing`` or use opake without "
        "the converter."
    ),
    "padding_free": _reject_if_truthy(
        "``padding_free`` flattens the batch and skips the attention mask "
        "DP-SGD's per-example accounting depends on. Not supported."
    ),
    "eval_packing": _reject_if_truthy(
        "Eval-side sequence packing is not supported for the same reason as "
        "``packing``."
    ),
    "shuffle_dataset": _reject_if_truthy(
        "Opake's Poisson sampler controls ordering; explicit shuffling "
        "would defeat the DP accounting."
    ),
    "pad_token": _reject_pad_token,
}


# DROP_WITH_WARN — TRL has the field but it's irrelevant on opake path.
TRL_SFT_DROP_FIELDS: dict[str, str | Callable[[Any], str | None]] = {
    "dataset_kwargs": (
        "Opake does not expose a ``datasets.map`` hook; the collator and "
        "tokenizer handle preprocessing."
    ),
    "packing_strategy": (
        "Only meaningful with ``packing=True``, which opake does not "
        "support; silently dropped."
    ),
    "router_aux_loss_coef": _drop_router_aux_loss,
}


def _convert_trl_sft_config(
    trl_cfg: Any,
    *,
    strict: bool = True,
    **dp_overrides: Any,
) -> dict[str, Any]:
    """Translate a ``trl.SFTConfig`` instance into opake ``SFTConfig`` kwargs."""
    trl = _import_trl()
    if not isinstance(trl_cfg, trl.SFTConfig):
        raise InputTypeError(
            *(f"Expected ``trl.SFTConfig`` instance, got {type(trl_cfg).__name__}.",)
        )

    dp_layer = _normalize_dp_overrides(dp_overrides)
    return _convert_trl_config(
        trl_cfg,
        trl_direct=TRL_SFT_DIRECT_FIELDS,
        trl_rename=TRL_SFT_RENAME_MAP,
        trl_transform=TRL_SFT_TRANSFORM_MAP,
        trl_reject=TRL_SFT_REJECTED_FIELDS,
        trl_drop=TRL_SFT_DROP_FIELDS,
        source_label="trl_sft_config",
        strict=strict,
        dp_overrides=dp_layer,
    )
