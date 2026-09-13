# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
#
# Trainer structure and method layout adapted from Hugging Face TRL
# (Apache-2.0; https://github.com/huggingface/trl), then modified for
# Opaque's per-example DP training flow.
# See ../../../../../NOTICE in this package for the full attribution.
"""``DPOTrainer`` — Direct Preference Optimization on :class:`DPTrainer`.

Mirrors ``trl.DPOTrainer`` in structure and method names — ``_prepare_dataset``
/ ``tokenize_row`` for data prep, ``compute_ref_log_probs`` for the reference
pass, ``dpo_loss`` for the loss dispatch — but routes the gradient through
Opaque's per-example :meth:`DPTrainer.compute_per_example_loss_and_metrics` seam
(loss + reward telemetry in one forward; rewards ride the clipped-grad aux
channel and the symmetric eval aux). The two forwards (chosen + rejected) are
folded into the seam since a batched forward has no per-example DP meaning.

The reference policy enters via precompute: a one-shot pass attaches per-example
``ref_chosen_logps`` / ``ref_rejected_logps`` columns the collator emits as
constants, so the per-example loss reads them without a second model inside
``vmap``. TR-DPO (``sync_ref_model``) instead recomputes the reference logps
each step from an EMA reference, via the :meth:`DPTrainer._augment_inputs`
pre-``vmap`` hook.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from operator import attrgetter
from typing import TYPE_CHECKING, Any

import torch

from opaque.alignment.dpo.collator import preference_collator
from opaque.alignment.dpo.data import extract_prompt
from opaque.alignment.dpo.loss import (
    apo_down_loss,
    apo_zero_loss,
    bco_loss,
    chosen_nll_loss,
    discopop_loss,
    exo_loss,
    f_divergence_remap,
    fused_sequence_logp,
    hinge_loss,
    ipo_loss,
    mpo_combine,
    nca_loss,
    odds_ratio_loss,
    robust_loss,
    sequence_logp,
    sigmoid_loss,
    simpo_loss,
    sppo_loss,
    wpo_weights,
)
from opaque.alignment.dpo.reference import (
    compute_ref_logprobs_for_dataset,
    ema_update_reference,
    null_ref_context,
)
from opaque.alignment.metric import entropy_from_logits, mean_token_accuracy
from opaque.api.transformers.trainer import DPTrainer
from opaque.api.transformers.trainer._distributed import resolve_ddp_state

# Single source of truth for PEFT detection (handles PeftModel + PeftMixedModel).
from opaque.api.transformers.trainer._dp_trainer import _is_peft_model
from opaque.exceptions import ConfigurationError, InputTypeError

from ._dpo_config import _REFERENCE_FREE_HEADS, DPOConfig

if TYPE_CHECKING:
    from collections.abc import Callable

# TRL ``loss_type`` name → ``opaque.alignment.dpo.loss`` head. An unknown value
# raises a standard ``KeyError`` at dispatch.
_DPO_HEADS: dict[str, Callable] = {
    "sigmoid": sigmoid_loss,
    "sigmoid_norm": sigmoid_loss,  # length-normalized log-ratio (see dpo_loss)
    "hinge": hinge_loss,
    "ipo": ipo_loss,
    "robust": robust_loss,
    "apo_zero": apo_zero_loss,
    "apo_down": apo_down_loss,
    "exo_pair": exo_loss,
    "nca_pair": nca_loss,
    "bco_pair": bco_loss,
    "sppo_hard": sppo_loss,
    "discopop": discopop_loss,
    "simpo": simpo_loss,  # reference-free, length-normalized policy logps
    "chosen_nll": chosen_nll_loss,  # special-cased: consumes chosen_logp, not the ratio
    # ``cpo`` / ``orpo`` are reference-free composites (preference/odds-ratio term
    # plus a per-token-mean NLL); special-cased in ``dpo_loss``, no registry entry.
}

# Loss variants that score the *length-normalized* log-ratio / policy logp
# (per-token mean). Normalization is applied per head in ``dpo_loss`` so an MPO
# list may mix normalized and summed variants (the reference is always
# precomputed summed). ``simpo`` consumes the length-normalized reference-free
# policy logp pair.
_NORM_LOSSES = frozenset({"ipo", "sigmoid_norm", "simpo"})

# Heads ``dpo_loss`` builds by hand (preference / odds-ratio term plus a
# per-token-mean NLL); they have no registry entry and are exempt from the eager
# head lookup.
_SPECIAL_CASED_HEADS = frozenset({"cpo", "orpo"})

_REF_COLUMNS = ("ref_chosen_logps", "ref_rejected_logps")


def _json_value(value: Any, *, path: str = "value") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigurationError(*(f"{path} must not contain non-finite floats",))
        return value
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise InputTypeError(
                    *(f"{path} mapping keys must be strings, got {type(key)!r}",)
                )
            normalized[key] = _json_value(item, path=f"{path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, (set, frozenset)):
        normalized = [_json_value(item, path=f"{path}[]") for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        )
    raise InputTypeError(
        *(f"{path} contains unsupported value of type {type(value)!r}",)
    )


def _tensor_state_digest(model: Any, *, exclude_adapter: bool = False) -> str:
    """Hash model parameters and buffers without depending on device placement."""
    hasher = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if exclude_adapter and _is_adapter_state_name(name):
            continue
        if not isinstance(tensor, torch.Tensor):
            raise InputTypeError(*(f"model state entry {name!r} is not a tensor",))
        if tensor.is_meta:
            raise ConfigurationError(*(f"cannot fingerprint meta tensor {name!r}",))
        value = tensor.detach().cpu().contiguous()
        hasher.update(name.encode("utf-8"))
        hasher.update(str(value.dtype).encode("ascii"))
        hasher.update(json.dumps(list(value.shape)).encode("ascii"))
        start = value.storage_offset() * value.element_size()
        end = start + value.numel() * value.element_size()
        hasher.update(bytes(value.untyped_storage()[start:end]))
    return hasher.hexdigest()


def _is_adapter_state_name(name: str) -> bool:
    adapter_markers = (
        ".lora_",
        ".ia3_",
        ".loha_",
        ".lokr_",
        ".oft_",
        ".boft_",
        ".vera_",
        ".fourierft_",
        ".hra_",
        ".modules_to_save.",
        "prompt_encoder.",
        "prompt_tokens",
    )
    return any(marker in name for marker in adapter_markers)


def _model_cache_identity(model: Any, *, adapter_mode: str) -> dict[str, Any]:
    """Return TRL-style identity for the reference behavior used at inference."""
    is_peft = _is_peft_model(model)
    identity: dict[str, Any] = {
        "adapter_mode": adapter_mode,
        "state_sha256": _tensor_state_digest(
            model, exclude_adapter=is_peft and adapter_mode == "disabled"
        ),
    }
    if is_peft and adapter_mode != "disabled":
        get_model_status = getattr(model, "get_model_status", None)
        if not callable(get_model_status):
            raise ConfigurationError(
                *(
                    "cannot fingerprint the effective PEFT reference state: "
                    "model does not expose get_model_status()",
                )
            )
        adapter_status = get_model_status()
        identity["adapter_runtime"] = {
            "enabled": adapter_status.enabled,
            "active_adapters": _json_value(adapter_status.active_adapters),
            "merged_adapters": _json_value(adapter_status.merged_adapters),
        }
    return identity


def _resolve_fused_handles(model: Any, eligible: bool) -> tuple[str | None, str | None]:
    """Resolve the backbone-prefix + lm_head param-key for the fused logp path.

    Returns ``(backbone_prefix, lm_head_param_name)`` — the dotted attribute path
    from ``model`` to the backbone submodule (e.g. ``"model"`` on a bare
    causal-LM, ``"base_model.model.model"`` on a PEFT-wrapped one) and the key of
    the lm_head weight in ``model.named_parameters()``. The path doubles as the
    params-key prefix when slicing the backbone-scoped sub-dict. The lm_head key
    is found by id-matching ``get_output_embeddings().weight`` against the named
    params, so it resolves correctly even under tied embeddings (where
    ``lm_head.weight`` is absent and the key is the embedding's
    ``model.embed_tokens.weight``).

    Under PEFT the inner causal-LM at ``peft_model.base_model.model`` is the
    target: calling the backbone (not the inner causal-LM) functionally yields
    ``BaseModelOutputWithPast`` rather than ``CausalLMOutputWithPast``. PEFT is
    detected via the ``peft_config`` marker and the ``base_model.model.`` path is
    prepended so the returned dotted prefix walks all the way to the backbone.

    Returns ``(None, None)`` when the run is ineligible or the model does not
    expose the ``backbone + output-embeddings`` shape the fused path needs (in
    which case the seam keeps the eager logits path).
    """
    if not eligible or model is None:
        return None, None
    # Unwrap PEFT: PeftModel.base_model is the adapter wrapper, whose ``.model``
    # is the original *ForCausalLM whose ``base_model_prefix`` we want.
    # ``peft_config`` is the PEFT marker (PreTrainedModel.base_model also exists
    # but as a self-reference, so it cannot be used for detection).
    if hasattr(model, "peft_config"):
        inner = getattr(getattr(model, "base_model", None), "model", None)
        path_to_inner = "base_model.model."
    else:
        inner = model
        path_to_inner = ""
    if inner is None:
        return None, None
    prefix = getattr(inner, "base_model_prefix", None)
    get_oe = getattr(inner, "get_output_embeddings", None)
    if not prefix or get_oe is None or getattr(inner, prefix, None) is None:
        return None, None
    output_embeddings = get_oe()
    weight = getattr(output_embeddings, "weight", None) if output_embeddings else None
    if weight is None:
        return None, None
    # ``params`` at training time is keyed off the OUTER model's named_parameters
    # tree (PEFT-wrapped keys live under ``base_model.model.``), so the id-lookup
    # must run on the outer model — not the unwrapped inner.
    name_by_id = {id(p): n for n, p in model.named_parameters()}
    lm_head_param_name = name_by_id.get(id(weight))
    if lm_head_param_name is None:
        return None, None
    return path_to_inner + prefix, lm_head_param_name


class DPOTrainer(DPTrainer):
    """DP Direct Preference Optimization trainer.

    Reference-logprob caching follows TRL: the prepared dataset fingerprint and
    the effective reference-model state determine cache reuse.
    """

    def __init__(
        self,
        model: Any = None,
        ref_model: Any = None,
        args: DPOConfig | None = None,
        data_collator: Callable | None = None,
        train_dataset: Any = None,
        eval_dataset: Any = None,
        processing_class: Any = None,
        compute_metrics: Callable | None = None,
        callbacks: list[Any] | None = None,
        optimizers: tuple[Any | None, Any | None] = (None, None),
        optimizer_cls_and_kwargs: tuple[Any, dict[str, Any]] | None = None,
        preprocess_logits_for_metrics: Callable | None = None,
        peft_config: Any = None,
    ) -> None:
        # ---- args normalization -------------------------------------------
        if args is None:
            args = DPOConfig(output_dir="trainer_output")
        elif not isinstance(args, DPOConfig):
            args = DPOConfig(
                **{f.name: getattr(args, f.name) for f in dataclasses.fields(args)}
            )

        # Kwargs reused for both the string-policy load and the string/auto
        # reference load.
        self._model_init_kwargs = dict(args.model_init_kwargs or {})

        if isinstance(model, str):
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                model, **self._model_init_kwargs
            )
        if model is ref_model and ref_model is not None:
            raise ConfigurationError(
                *(
                    "`model` and `ref_model` must be different objects (the reference "
                    "is a frozen copy of the policy).",
                )
            )

        # ---- capture loss configuration onto self -------------------------
        self._beta = float(args.beta)
        self._loss_type: list[str] = list(args.loss_type)
        self._loss_weights: list[float] = list(args.loss_weights)
        self._label_smoothing = float(args.label_smoothing)
        self._f_divergence_type = args.f_divergence_type
        self._f_alpha_coef = float(args.f_alpha_divergence_coef)
        self._ld_alpha = args.ld_alpha
        self._discopop_tau = float(args.discopop_tau)
        self._simpo_gamma = float(args.simpo_gamma)
        self._cpo_alpha = float(args.cpo_alpha)
        self._orpo_lambda = float(args.orpo_lambda)
        self._use_weighting = bool(args.use_weighting)
        self._log_completion_metrics = bool(args.log_completion_metrics)
        # FUSED-PATH GATE (static eligibility). The logits-free fused log-prob
        # primitive may be selected only when no logits-consuming feature is
        # active in the loss path: LD-DPO needs per-token logits; WPO reads
        # per-token logps; a non-reverse-KL f-divergence remaps them. Rich
        # completion telemetry (``log_completion_metrics``) still needs logits,
        # but it is gated per-step at the call site rather than disabling the
        # fused path wholesale.
        self._fused_logp_eligible = (
            self._ld_alpha is None
            and not self._use_weighting
            and self._f_divergence_type == "reverse_kl"
        )
        # Fused-path handles (backbone prefix + lm_head key) are resolved after
        # the PEFT wrapping below, so the prefix targets the final module tree.
        # Reference-need is intrinsic to each head: a run needs a reference iff
        # any configured head is *not* in the reference-free set.
        self._needs_reference = any(
            lt not in _REFERENCE_FREE_HEADS for lt in self._loss_type
        )
        # Build the head dispatch eagerly so an unknown loss_type fails now with a
        # standard KeyError. ``cpo`` / ``orpo`` are special-cased in dpo_loss and
        # have no registry entry, so they're exempt from the eager lookup.
        self._heads = [
            _DPO_HEADS[name] if name not in _SPECIAL_CASED_HEADS else None
            for name in self._loss_type
        ]

        # ---- TR-DPO ----
        self._sync_ref_model = bool(args.sync_ref_model)
        self._ref_mixup_alpha = float(args.ref_model_mixup_alpha)
        self._ref_sync_steps = int(args.ref_model_sync_steps)
        self._tr_ref: Any = None  # EMA reference module (kept on device)

        # ---- tokenizer ----------------------------------------------------
        processing_class = self._resolve_tokenizer(
            model, processing_class, self._model_init_kwargs
        )
        self._pad_token_id = processing_class.pad_token_id

        # ---- dropout / PEFT ----------------------------------------------
        if args.disable_dropout:
            self._disable_dropout_in_model(model)
        if peft_config is not None:
            from peft import get_peft_model

            model = get_peft_model(model, peft_config)
        self._is_peft = _is_peft_model(model)

        # Resolve fused-path handles on the FINAL model: with a ``peft_config``
        # the policy was just wrapped above, so resolving earlier would record the
        # unwrapped ``"model"`` prefix and ``attrgetter`` would hit the inner
        # causal-LM (``CausalLMOutputWithPast``, no ``last_hidden_state``) at fused
        # time. ``None`` handles (ineligible / no backbone) keep the eager path.
        self._backbone_prefix, self._lm_head_param_name = _resolve_fused_handles(
            model, self._fused_logp_eligible
        )
        self._use_fused_logp = (
            self._fused_logp_eligible and self._lm_head_param_name is not None
        )
        if self._sync_ref_model and self._is_peft:
            raise ConfigurationError(
                *(
                    "sync_ref_model (TR-DPO) requires full fine-tuning, not PEFT "
                    "(the EMA reference tracks the full policy).",
                )
            )

        # ---- reference resolvability (before tokenize/precompute) ---------
        self._precompute_device = args.device
        name_or_path = getattr(model.config, "_name_or_path", "") or ""
        model_id = name_or_path
        batch_size = args.precompute_ref_batch_size or args.per_device_train_batch_size

        # A reference-using loss needs a resolvable reference. When none can be
        # auto-loaded (in-memory policy with no path, no explicit ref_model, not
        # PEFT) fail now, before the (potentially long) tokenize/precompute.
        if (
            self._needs_reference
            and ref_model is None
            and not self._is_peft
            and not name_or_path
        ):
            raise ConfigurationError(
                *(
                    "No reference available for a reference-using loss_type: pass "
                    "ref_model=, use a PEFT policy, use a reference-free loss_type "
                    "(simpo/cpo/orpo), or load the policy from a path so a reference "
                    "copy can be auto-loaded.",
                )
            )

        # ---- collator (reused for precompute and training) ---------------
        if data_collator is None:
            data_collator = preference_collator(
                pad_token_id=self._pad_token_id,
                max_length=args.max_length,
                pad_to_multiple_of=args.pad_to_multiple_of,
            )

        # ---- tokenize datasets (before super) -----------------------------
        train_dataset = self._prepare_dataset(train_dataset, processing_class, args)
        if eval_dataset is not None and not isinstance(eval_dataset, dict):
            eval_dataset = self._prepare_dataset(eval_dataset, processing_class, args)

        # ---- reference logps (precompute, or seed for TR-DPO) -------------
        if self._sync_ref_model or self._needs_reference:
            # The precompute shards across a live process group, which
            # ``super().__init__`` below brings up too late.
            # ``resolve_ddp_state`` is idempotent, so the later call reads the
            # group established here.
            resolve_ddp_state(self._precompute_device, args)
        if self._sync_ref_model:
            train_dataset, eval_dataset = self._setup_tr_dpo(
                model,
                ref_model,
                train_dataset,
                eval_dataset,
                collator=data_collator,
                batch_size=batch_size,
                disable_dropout=args.disable_dropout,
            )
        elif self._needs_reference:
            train_dataset = self._precompute_ref_logps(
                train_dataset,
                model=model,
                ref_model=ref_model,
                model_id=model_id,
                collator=data_collator,
                batch_size=batch_size,
                disable_dropout=args.disable_dropout,
            )
            if eval_dataset is not None and not isinstance(eval_dataset, dict):
                eval_dataset = self._precompute_ref_logps(
                    eval_dataset,
                    model=model,
                    ref_model=ref_model,
                    model_id=model_id,
                    collator=data_collator,
                    batch_size=batch_size,
                    disable_dropout=args.disable_dropout,
                )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

    # ------------------------------------------------------------------
    # Tokenizer / dropout helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_tokenizer(
        model: Any, processing_class: Any, model_init_kwargs: dict[str, Any]
    ) -> Any:
        if processing_class is None:
            from transformers import AutoTokenizer

            name = getattr(model.config, "_name_or_path", None)
            if not name:
                raise ConfigurationError(
                    *(
                        "processing_class is None and the model config has no "
                        "_name_or_path; pass processing_class.",
                    )
                )
            processing_class = AutoTokenizer.from_pretrained(
                name,
                trust_remote_code=bool(
                    model_init_kwargs.get("trust_remote_code", False)
                ),
            )
        if processing_class.pad_token_id is None:
            processing_class.pad_token = processing_class.eos_token
        return processing_class

    @staticmethod
    def _disable_dropout_in_model(model: Any) -> None:
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0

    # ------------------------------------------------------------------
    # Dataset preparation (TRL-shaped)
    # ------------------------------------------------------------------
    def _prepare_dataset(
        self, dataset: Any, processing_class: Any, args: DPOConfig
    ) -> Any:
        if dataset is None:
            return None
        column_names = list(getattr(dataset, "column_names", []) or [])
        if "chosen_input_ids" in column_names:
            return dataset  # already tokenized
        # Extract the shared prompt prefix when only (chosen, rejected) are given.
        if "prompt" not in column_names:
            dataset = dataset.map(extract_prompt, num_proc=args.dataset_num_proc)
            column_names = list(dataset.column_names)

        max_length = args.max_length
        tokenize_fn = self.tokenize_row

        def tokenize_row(example: dict) -> dict:
            return tokenize_fn(example, processing_class, max_length)

        return dataset.map(
            tokenize_row,
            remove_columns=column_names,
            num_proc=args.dataset_num_proc,
            desc="Tokenizing preference dataset",
        )

    @staticmethod
    def tokenize_row(
        example: dict, processing_class: Any, max_length: int | None
    ) -> dict:
        """Tokenize one preference example into the collator's input schema.

        Produces ``chosen_input_ids`` / ``rejected_input_ids`` (prompt +
        completion) and the matching ``*_completion_mask`` (``0`` over prompt,
        ``1`` over completion). Lifted from ``examples/train_dpo.py``.
        """
        prompt = example.get("prompt", [])
        chosen = example["chosen"]
        rejected = example["rejected"]

        def _apply_template(messages: Any) -> list[int]:
            if isinstance(messages, list):
                return processing_class.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=False
                )
            return processing_class.encode(messages, add_special_tokens=False)

        if isinstance(prompt, list) and prompt:
            prompt_ids = processing_class.apply_chat_template(
                prompt, tokenize=True, add_generation_prompt=True
            )
        elif isinstance(prompt, str) and prompt:
            prompt_ids = processing_class.encode(prompt, add_special_tokens=True)
        else:
            prompt_ids = []

        if isinstance(chosen, list):
            chosen_ids = _apply_template(prompt + chosen)
            rejected_ids = _apply_template(prompt + rejected)
        else:
            if prompt_ids:
                chosen_ids = prompt_ids + processing_class.encode(
                    chosen, add_special_tokens=False
                )
                rejected_ids = prompt_ids + processing_class.encode(
                    rejected, add_special_tokens=False
                )
            else:
                chosen_ids = processing_class.encode(chosen, add_special_tokens=True)
                rejected_ids = processing_class.encode(
                    rejected, add_special_tokens=True
                )

        prompt_len = len(prompt_ids)
        chosen_cmask = [0] * min(prompt_len, len(chosen_ids)) + [1] * max(
            0, len(chosen_ids) - prompt_len
        )
        rejected_cmask = [0] * min(prompt_len, len(rejected_ids)) + [1] * max(
            0, len(rejected_ids) - prompt_len
        )

        if max_length is not None:
            chosen_ids = chosen_ids[:max_length]
            rejected_ids = rejected_ids[:max_length]
            chosen_cmask = chosen_cmask[:max_length]
            rejected_cmask = rejected_cmask[:max_length]

        return {
            "chosen_input_ids": chosen_ids,
            "rejected_input_ids": rejected_ids,
            "chosen_completion_mask": chosen_cmask,
            "rejected_completion_mask": rejected_cmask,
        }

    # ------------------------------------------------------------------
    # Reference log-probs (precompute / TR-DPO; outside vmap)
    # ------------------------------------------------------------------
    def _precompute_ref_logps(
        self,
        dataset: Any,
        *,
        model: Any,
        ref_model: Any,
        model_id: str,
        collator: Callable,
        batch_size: int,
        disable_dropout: bool,
    ) -> Any:
        """Attach ``ref_{chosen,rejected}_logps`` columns via a one-shot pass.

        Resolves the reference as an explicit ``ref_model`` (an object or a path
        string), the PEFT base model (adapter disabled via ``null_ref_context``),
        or an auto-loaded copy of the policy. A user-supplied ``ref_model``
        *object* is left in the device/mode it started in; a string ``ref_model``
        and an auto-loaded copy are both instantiated here and freed to CPU after
        the pass.

        The no-reference-available case is checked early in ``__init__`` (before
        tokenization), so it never reaches here.
        """
        null_ref = False
        owns_ref = False  # loaded by us → safe to drop / move to CPU
        if isinstance(ref_model, str):
            from transformers import AutoModelForCausalLM

            ref = AutoModelForCausalLM.from_pretrained(
                ref_model, **self._model_init_kwargs
            )
            owns_ref = True
            adapter_mode = "explicit"
        elif ref_model is not None:
            ref = ref_model
            adapter_mode = "explicit"
        elif self._is_peft:
            ref = model  # base model with the adapter disabled at forward time
            null_ref = True
            adapter_mode = "disabled"
        else:
            from transformers import AutoModelForCausalLM

            ref = AutoModelForCausalLM.from_pretrained(
                model_id, **self._model_init_kwargs
            )
            owns_ref = True
            adapter_mode = "auto-policy-copy"

        resolved_cache_identity = {
            "kind": "dpo-reference-logprobs",
            "reference": _model_cache_identity(ref, adapter_mode=adapter_mode),
        }

        orig_device = next(ref.parameters()).device
        was_training = ref.training
        if disable_dropout:
            self._disable_dropout_in_model(ref)
        ref.to(self._precompute_device).eval()

        def ref_callable(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return self.compute_ref_log_probs(batch, ref, null_ref=null_ref)

        dataset = compute_ref_logprobs_for_dataset(
            dataset,
            ref=ref_callable,
            collator=collator,
            output_columns=_REF_COLUMNS,
            batch_size=batch_size,
            cache_identity=resolved_cache_identity,
        )

        # Restore caller-visible state. Training reads the cached columns, not
        # the model, so we don't keep the reference around.
        if null_ref:
            if was_training:
                ref.train()
        elif owns_ref:
            ref.to("cpu")
        else:
            ref.to(orig_device)  # explicit user ref_model: leave as we found it
            if was_training:
                ref.train()
        return dataset

    def _setup_tr_dpo(
        self,
        model: Any,
        ref_model: Any,
        train_dataset: Any,
        eval_dataset: Any,
        *,
        collator: Callable,
        batch_size: int,
        disable_dropout: bool,
    ) -> tuple[Any, Any]:
        """Build the EMA reference (kept on device) and seed the ref columns.

        TR-DPO recomputes the reference logps each step (``_augment_inputs``), so
        the seed values don't matter for correctness — they only populate the
        columns so ``batch_keys`` includes them. The seed is not persisted.
        """
        ref = ref_model if ref_model is not None else copy.deepcopy(model)
        if disable_dropout:
            self._disable_dropout_in_model(ref)
        ref.to(self._precompute_device).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        self._tr_ref = ref

        def seed(dataset: Any) -> Any:
            if dataset is None or isinstance(dataset, dict):
                return dataset
            return compute_ref_logprobs_for_dataset(
                dataset,
                ref=lambda b: self.compute_ref_log_probs(b, ref, null_ref=False),
                collator=collator,
                output_columns=_REF_COLUMNS,
                batch_size=batch_size,
                cache_identity={"kind": "trdpo-seed"},
                use_cache=False,
            )

        return seed(train_dataset), seed(eval_dataset)

    def compute_ref_log_probs(
        self,
        batch: dict[str, torch.Tensor],
        ref_model: Any,
        *,
        null_ref: bool,
        to_cpu: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Reference chosen/rejected *summed* sequence logps for one batch.

        Always summed (``length_normalized=False``); ``dpo_loss`` derives any
        length-normalized log-ratio by dividing by the completion length, so a
        single precompute serves both normalized and summed losses. With
        ``to_cpu=True`` (precompute) returns CPU float tensors; with
        ``to_cpu=False`` (the per-step TR-DPO path) returns device tensors.
        """
        device = next(ref_model.parameters()).device
        cids = batch["chosen_input_ids"].to(device)
        cmask = batch["chosen_attention_mask"].to(device)
        ccmask = batch["chosen_completion_mask"].to(device)
        rids = batch["rejected_input_ids"].to(device)
        rmask = batch["rejected_attention_mask"].to(device)
        rcmask = batch["rejected_completion_mask"].to(device)

        def forward(ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return ref_model(input_ids=ids, attention_mask=mask).logits

        if null_ref:
            with null_ref_context(ref_model):
                c_logits = forward(cids, cmask)
                r_logits = forward(rids, rmask)
        else:
            c_logits = forward(cids, cmask)
            r_logits = forward(rids, rmask)

        c_lp_kwargs: dict[str, Any] = {}
        r_lp_kwargs: dict[str, Any] = {}
        if self._ld_alpha is not None:
            c_sp, r_sp = self._ld_public_length(ccmask, rcmask)
            c_lp_kwargs = {"ld_alpha": self._ld_alpha, "shared_prefix_len": c_sp}
            r_lp_kwargs = {"ld_alpha": self._ld_alpha, "shared_prefix_len": r_sp}

        ref_chosen = sequence_logp(c_logits, cids, ccmask, **c_lp_kwargs)
        ref_rejected = sequence_logp(r_logits, rids, rcmask, **r_lp_kwargs)
        if to_cpu:
            return {
                "ref_chosen_logps": ref_chosen.detach().float().cpu(),
                "ref_rejected_logps": ref_rejected.detach().float().cpu(),
            }
        return {
            "ref_chosen_logps": ref_chosen.detach(),
            "ref_rejected_logps": ref_rejected.detach(),
        }

    # ------------------------------------------------------------------
    # TR-DPO: per-step reference refresh + EMA update (outside vmap)
    # ------------------------------------------------------------------
    def _augment_inputs(
        self, inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # Training step: advance the EMA reference on cadence, then refresh the
        # per-step ref logps from it (overwriting the seeded columns).
        if not self._sync_ref_model or self._tr_ref is None:
            return inputs
        step = int(self.state.global_step)
        if step > 0 and step % self._ref_sync_steps == 0:
            self._ema_update_reference()
        return self._inject_tr_ref_logps(inputs)

    def _inject_tr_ref_logps(
        self, inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Overwrite ``ref_*_logps`` with the *current* EMA reference (no EMA step).

        Used by both the training augment hook and eval ``prediction_step`` so
        eval scores against the same evolving reference as training, not the
        stale seed columns.

        Poisson rounds can realize an empty batch (like every other DP-SGD
        batch path in :class:`DPTrainer`); the reference forward is skipped
        then and the zero-shaped seed columns pass through unchanged.
        """
        if int(inputs["chosen_input_ids"].shape[0]) == 0:
            return inputs
        with torch.no_grad():
            refs = self.compute_ref_log_probs(
                inputs, self._tr_ref, null_ref=False, to_cpu=False
            )
        device = inputs["chosen_input_ids"].device
        return {
            **inputs,
            "ref_chosen_logps": refs["ref_chosen_logps"].to(device),
            "ref_rejected_logps": refs["ref_rejected_logps"].to(device),
        }

    def _ema_update_reference(self) -> None:
        """``ref ← (1-α)·ref + α·policy`` over the live functional policy params."""
        ctx = self._ctx
        if ctx is None:
            return
        policy = {**ctx.frozen_params, **ctx.trainable_params}
        ref_named = dict(self._tr_ref.named_parameters())
        keys = [k for k in ref_named if k in policy]
        with torch.no_grad():
            updated = ema_update_reference(
                {k: ref_named[k].detach() for k in keys},
                {k: policy[k].detach().to(ref_named[k].device) for k in keys},
                self._ref_mixup_alpha,
            )
            for k in keys:
                ref_named[k].data.copy_(updated[k])

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _head_kwargs(self, name: str) -> dict[str, Any]:
        if name in ("sigmoid", "sigmoid_norm", "robust", "exo_pair"):
            return {"label_smoothing": self._label_smoothing}
        if name == "simpo":
            return {
                "gamma": self._simpo_gamma,
                "label_smoothing": self._label_smoothing,
            }
        if name == "discopop":
            return {"discopop_tau": self._discopop_tau}
        return {}

    @staticmethod
    def _completion_len(completion_mask: torch.Tensor) -> torch.Tensor:
        """Shifted completion-token count (matches ``sequence_logp``'s divisor)."""
        return (completion_mask[..., 1:] != 0).sum(-1).clamp(min=1)

    @staticmethod
    def _ld_public_length(
        chosen_completion_mask: torch.Tensor,
        rejected_completion_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Derive the paired LD-DPO public length in completion tokens.

        The paper defines this as the shorter completion length. Tokens through
        that position keep weight ``1``; the verbose tail is damped by
        ``ld_alpha`` inside ``sequence_logp``.
        """
        c = chosen_completion_mask[..., 1:]
        r = rejected_completion_mask[..., 1:]
        shared = torch.minimum((c != 0).sum(-1), (r != 0).sum(-1))
        return shared, shared

    def dpo_loss(
        self,
        chosen_logratio: torch.Tensor,
        rejected_logratio: torch.Tensor,
        *,
        chosen_logp: torch.Tensor,
        rejected_logp: torch.Tensor,
        chosen_completion_mask: torch.Tensor,
        rejected_completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Combine the configured loss head(s) into one per-example scalar.

        Two input pairs flow in so a mixed MPO list stays coherent:

        - ``chosen_logratio`` / ``rejected_logratio`` are the *summed*
          ``policy_logp - ref_logp`` — consumed by the **reference-using** heads
          (``sigmoid``, ``ipo``, …). They equal the policy logps when the run is
          reference-free (no head needs a reference).
        - ``chosen_logp`` / ``rejected_logp`` are the *summed* policy logps —
          consumed by the **reference-free** heads (``chosen_nll``, ``simpo``,
          ``cpo``, ``orpo``), which score the policy's own (length-normalized) logp.

        Length-normalized heads (``ipo`` / ``sigmoid_norm`` on the log-ratio,
        ``simpo`` on the policy logp) divide by the completion length, so an MPO
        list may freely mix normalized and summed variants. ``chosen_nll`` /
        ``cpo`` / ``orpo`` are assembled by hand (NLL + a preference / odds-ratio
        term).
        """
        c_len = self._completion_len(chosen_completion_mask)
        r_len = self._completion_len(rejected_completion_mask)

        # Length-normalized views, computed lazily only when a head needs them.
        lr_norm: tuple[torch.Tensor, torch.Tensor] | None = None
        logp_norm: tuple[torch.Tensor, torch.Tensor] | None = None

        def _lr_norm() -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal lr_norm
            if lr_norm is None:
                lr_norm = (chosen_logratio / c_len, rejected_logratio / r_len)
            return lr_norm

        def _logp_norm() -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal logp_norm
            if logp_norm is None:
                logp_norm = (chosen_logp / c_len, rejected_logp / r_len)
            return logp_norm

        parts: dict[str, torch.Tensor] = {}
        weights: dict[str, float] = {}
        for name, weight, head in zip(
            self._loss_type, self._loss_weights, self._heads, strict=False
        ):
            weights[name] = weight
            # ---- reference-free composites (assembled by hand) ----
            if name == "chosen_nll":
                parts[name] = chosen_nll_loss(chosen_logp)
                continue
            if name == "cpo":
                # sigmoid on the SUMMED policy logps + per-token-mean chosen NLL.
                c_norm, _ = _logp_norm()
                parts[name] = sigmoid_loss(
                    chosen_logp, rejected_logp, beta=self._beta
                ) + self._cpo_alpha * chosen_nll_loss(c_norm)
                continue
            if name == "orpo":
                # per-token-mean chosen NLL + odds-ratio on normalized policy logps.
                c_norm, r_norm = _logp_norm()
                parts[name] = chosen_nll_loss(c_norm) + self._orpo_lambda * (
                    odds_ratio_loss(c_norm, r_norm)
                )
                continue
            if name == "simpo":
                # reference-free, length-normalized policy logps; no f-divergence.
                c_norm, r_norm = _logp_norm()
                parts[name] = head(
                    c_norm, r_norm, beta=self._beta, **self._head_kwargs(name)
                )
                continue

            # ---- reference-using heads (log-ratio pair) ----
            if name in _NORM_LOSSES:
                clr, rlr = _lr_norm()
            else:
                clr, rlr = chosen_logratio, rejected_logratio
            if self._f_divergence_type != "reverse_kl":
                clr = f_divergence_remap(
                    clr,
                    f_divergence_type=self._f_divergence_type,
                    alpha=self._f_alpha_coef,
                )
                rlr = f_divergence_remap(
                    rlr,
                    f_divergence_type=self._f_divergence_type,
                    alpha=self._f_alpha_coef,
                )
            parts[name] = head(clr, rlr, beta=self._beta, **self._head_kwargs(name))
        return mpo_combine(parts, weights)

    # ------------------------------------------------------------------
    # Fused logits-free policy logp (plan §E)
    # ------------------------------------------------------------------
    def _last_hidden_state(
        self,
        params: dict[str, Any],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Backbone last hidden state ``(T, H)`` only — no lm_head, no all-layers.

        Calls the backbone submodule (resolved via :func:`_resolve_fused_handles`)
        functionally with the backbone-scoped slice of ``params`` (the keys under
        the ``"<prefix>."`` namespace, re-rooted to the submodule). This returns
        just the last-layer hidden state — unlike ``output_hidden_states=True``,
        which stacks all ``L+1`` layers and can exceed the ``(T, V)`` logits the
        fused path is avoiding. The HF backbones in scope return the hidden state
        as ``out[0]`` / ``out.last_hidden_state``.

        The backbone-prefix is a dotted path so ``attrgetter`` walks PEFT wrappers
        correctly — calling the unwrapped backbone (not the inner causal-LM)
        guarantees ``BaseModelOutputWithPast`` rather than
        ``CausalLMOutputWithPast``.

        The ``batchify`` vmap-safety patch is applied to the causal-LM class, not
        the backbone, so under ``vmap`` (per-example ``(T,)`` inputs) the batch
        dim is added here and stripped on exit. Already-batched ``(B, T)`` inputs
        pass through untouched.
        """
        prefix = self._backbone_prefix + "."
        backbone_params = {
            k[len(prefix) :]: v for k, v in params.items() if k.startswith(prefix)
        }
        backbone = attrgetter(self._backbone_prefix)(self._model)
        unbatched = input_ids.ndim < 2  # noqa: PLR2004 - batchless token IDs are 1D
        if unbatched:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
        out = torch.func.functional_call(
            backbone,
            backbone_params,
            (),
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )
        hidden = out[0] if isinstance(out, tuple) else out.last_hidden_state
        return hidden.squeeze(0) if unbatched else hidden

    def _fused_logp(
        self,
        fmodel: Callable[..., Any],
        params: dict[str, Any],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """One side's summed policy logp via the logits-free fused primitive.

        ``length_normalized=False`` (summed): ``dpo_loss`` derives any per-token
        mean per head, so a single summed logp serves every head — matching the
        eager ``sequence_logp`` call it replaces. ``fmodel`` is accepted for
        signature symmetry with the eager branch but unused (we call the backbone
        directly to avoid the lm_head projection).
        """
        del fmodel  # the backbone forward is what we need, not the lm_head forward
        hidden = self._last_hidden_state(params, input_ids, attention_mask)
        lm_head_weight = params[self._lm_head_param_name]
        return fused_sequence_logp(
            hidden, lm_head_weight, input_ids, completion_mask, length_normalized=False
        )

    def compute_per_example_loss_and_metrics(
        self,
        fmodel: Callable[..., Any],
        params: dict[str, Any],
        inputs: dict[str, Any],
    ) -> tuple[Any, dict[str, Any]]:
        """One preference pair's DPO ``(loss, rewards/*)`` (vmap-batched).

        This is the rich :class:`DPTrainer` seam: two policy forwards (chosen,
        rejected) → per-sequence summed logps → the configured head(s), plus the
        per-example reward telemetry. The harness carries the telemetry through
        the clipped-grad aux channel (logged every training step) and aggregates
        it in the eval loop (``eval_rewards/*``). The reference enters as the
        constant ``ref_*_logps`` (precomputed, or TR-DPO's per-step values from
        :meth:`_augment_inputs`), so no second model runs inside ``vmap``.
        """
        c_cmask = inputs["chosen_completion_mask"]
        r_cmask = inputs["rejected_completion_mask"]

        # FUSED PATH: on an eligible run compute the policy logps through
        # ``fused_sequence_logp`` over the backbone's last hidden state, never
        # materialising the ``(T, V)`` logits. Static eligibility (LD-DPO / WPO /
        # f-divergence) is captured in ``_use_fused_logp``; the per-step
        # ``log_completion_metrics`` check disables the fused branch when rich
        # telemetry is on, because ``entropy`` / ``logits/*`` / ``mean_token_acc``
        # all consume full logits.
        if self._use_fused_logp and not self._log_completion_metrics:
            chosen_logp = self._fused_logp(
                fmodel,
                params,
                inputs["chosen_input_ids"],
                inputs["chosen_attention_mask"],
                c_cmask,
            )
            rejected_logp = self._fused_logp(
                fmodel,
                params,
                inputs["rejected_input_ids"],
                inputs["rejected_attention_mask"],
                r_cmask,
            )
            chosen_out = rejected_out = None
        else:
            chosen_out = fmodel(
                params,
                input_ids=inputs["chosen_input_ids"],
                attention_mask=inputs["chosen_attention_mask"],
            )
            rejected_out = fmodel(
                params,
                input_ids=inputs["rejected_input_ids"],
                attention_mask=inputs["rejected_attention_mask"],
            )

            c_lp_kwargs: dict[str, Any] = {}
            r_lp_kwargs: dict[str, Any] = {}
            if self._ld_alpha is not None:
                c_sp, r_sp = self._ld_public_length(c_cmask, r_cmask)
                c_lp_kwargs = {"ld_alpha": self._ld_alpha, "shared_prefix_len": c_sp}
                r_lp_kwargs = {"ld_alpha": self._ld_alpha, "shared_prefix_len": r_sp}

            chosen_logp = sequence_logp(
                chosen_out.logits, inputs["chosen_input_ids"], c_cmask, **c_lp_kwargs
            )
            rejected_logp = sequence_logp(
                rejected_out.logits,
                inputs["rejected_input_ids"],
                r_cmask,
                **r_lp_kwargs,
            )

        # Log-ratio pair for reference-using heads (== policy logp when the run
        # has no reference). Reference-free heads read the policy-logp pair.
        if self._needs_reference:
            chosen_lr = chosen_logp - inputs["ref_chosen_logps"]
            rejected_lr = rejected_logp - inputs["ref_rejected_logps"]
        else:
            chosen_lr, rejected_lr = chosen_logp, rejected_logp

        loss = self.dpo_loss(
            chosen_lr,
            rejected_lr,
            chosen_logp=chosen_logp,
            rejected_logp=rejected_logp,
            chosen_completion_mask=c_cmask,
            rejected_completion_mask=r_cmask,
        )

        # WPO (arXiv:2406.11827): reweight by the policy's detached average
        # completion probability on each side. The weight is per-example and
        # carries no gradient (``wpo_weights`` detaches), so per-example DP is
        # preserved.
        if self._use_weighting:
            weight = self._wpo_weight(
                chosen_out.logits, inputs["chosen_input_ids"], c_cmask
            ) * self._wpo_weight(
                rejected_out.logits, inputs["rejected_input_ids"], r_cmask
            )
            loss = loss * weight

        # On the fused branch ``chosen_out`` / ``rejected_out`` are ``None`` (no
        # logits were materialised). The logits-consuming telemetry inside
        # ``_reward_aux`` is gated on ``log_completion_metrics`` — off whenever the
        # fused path is active — so passing ``None`` here is safe.
        chosen_logits = chosen_out.logits if chosen_out is not None else None
        rejected_logits = rejected_out.logits if rejected_out is not None else None
        return loss, self._reward_aux(
            chosen_logratio=chosen_lr,
            rejected_logratio=rejected_lr,
            chosen_logp=chosen_logp,
            rejected_logp=rejected_logp,
            chosen_logits=chosen_logits,
            rejected_logits=rejected_logits,
            chosen_input_ids=inputs["chosen_input_ids"],
            chosen_completion_mask=c_cmask,
            rejected_completion_mask=r_cmask,
        )

    def compute_per_example_loss(
        self,
        fmodel: Callable[..., Any],
        params: dict[str, Any],
        inputs: dict[str, Any],
        *,
        return_logits: bool = False,
    ) -> Any:
        """Loss-only view of the seam (the DP grad path uses the rich seam)."""
        loss, _aux = self.compute_per_example_loss_and_metrics(fmodel, params, inputs)
        if return_logits:
            return loss, None  # DPO eval routes through prediction_step, not logits
        return loss

    @staticmethod
    def _masked_mean_logit(
        logits: torch.Tensor, completion_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean logit over (shifted) completion positions — TRL ``logits/*``.

        vmap-safe: a masked weighted mean (no boolean/dynamic indexing), so it
        runs inside the per-example closure. For one example ``logits`` is
        ``(T, V)`` and ``completion_mask`` is ``(T,)``.
        """
        shifted = logits[..., :-1, :]
        mask = (completion_mask[..., 1:] != 0).to(shifted.dtype)
        pos_mean = shifted.mean(dim=-1)
        return (pos_mean * mask).sum(-1) / mask.sum(-1).clamp(min=1)

    def _reward_aux(
        self,
        *,
        chosen_logratio: torch.Tensor,
        rejected_logratio: torch.Tensor,
        chosen_logp: torch.Tensor,
        rejected_logp: torch.Tensor,
        chosen_logits: torch.Tensor,
        rejected_logits: torch.Tensor,
        chosen_input_ids: torch.Tensor,
        chosen_completion_mask: torch.Tensor,
        rejected_completion_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Per-example DPO telemetry (detached) — the full TRL logged set.

        ``rewards/*`` from the implicit-reward log-ratios and ``logps/*`` (summed
        policy sequence logps) are always logged — they're free byproducts of the
        loss. The logits-consuming diagnostics — ``logits/*`` (mean completion
        logit), ``entropy`` (mean next-token entropy), and ``mean_token_accuracy``
        — are gated on ``log_completion_metrics`` (default on); when off they're
        skipped so the loss path stays logits-light. Every tensor is
        ``detach()``-ed, so the telemetry rides the clipped-grad aux channel
        without leaking gradient; the harness means each across the DDP-synced
        batch (train) and aggregates the same dict in the eval loop (``eval_*``).
        """
        beta = self._beta
        aux = {
            "rewards/chosen": (beta * chosen_logratio).detach(),
            "rewards/rejected": (beta * rejected_logratio).detach(),
            "rewards/accuracies": (chosen_logratio > rejected_logratio)
            .to(chosen_logratio.dtype)
            .detach(),
            "rewards/margins": (beta * (chosen_logratio - rejected_logratio)).detach(),
            "logps/chosen": chosen_logp.detach(),
            "logps/rejected": rejected_logp.detach(),
        }
        if self._log_completion_metrics:
            entropy = 0.5 * (
                entropy_from_logits(chosen_logits, chosen_completion_mask)
                + entropy_from_logits(rejected_logits, rejected_completion_mask)
            )
            aux.update(
                {
                    "logits/chosen": self._masked_mean_logit(
                        chosen_logits, chosen_completion_mask
                    ).detach(),
                    "logits/rejected": self._masked_mean_logit(
                        rejected_logits, rejected_completion_mask
                    ).detach(),
                    "entropy": entropy.detach(),
                    "mean_token_accuracy": mean_token_accuracy(
                        chosen_logits, chosen_input_ids, chosen_completion_mask
                    ),
                }
            )
        return aux

    @staticmethod
    def _wpo_weight(
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Detached WPO weight for one side (causal-LM shift + completion mask)."""
        shifted_logits = logits[..., :-1, :]
        shifted_ids = input_ids[..., 1:]
        shifted_mask = completion_mask[..., 1:]
        # Per-token logp of the realised next token (public log_softmax + gather;
        # equivalent to selective_log_softmax, kept on the public boundary).
        per_token_logps = (
            torch.log_softmax(shifted_logits, dim=-1)
            .gather(-1, shifted_ids.unsqueeze(-1))
            .squeeze(-1)
        )
        return wpo_weights(per_token_logps, shifted_mask, shifted_logits)

    # ------------------------------------------------------------------
    # Evaluation: plug into the inherited eval harness via prediction_step
    # ------------------------------------------------------------------
    def prediction_step(
        self,
        model: Any,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[Any, None, None]:
        """One eval batch → per-example DPO loss (+ rewards via the aux channel).

        A preference batch (``chosen_input_ids`` / ``rejected_input_ids``, no
        ``labels``) can't run the inherited LM-shaped prediction path, so DPO
        plugs in here: it computes per-example ``(loss, rewards)`` through the
        rich seam, using the **same** functional context as training
        (``self._ctx`` → live policy weights, not the stale ``self.model``), and
        publishes the per-example rewards on ``self._pending_eval_aux`` for
        :meth:`DPTrainer.evaluation_loop` to aggregate into ``eval_rewards/*``.
        Returns ``(per_example_loss, None, None)`` — no predictions/labels.
        """
        del model, prediction_loss_only, ignore_keys
        from opaque.functional import make_functional

        ctx = self._ctx
        if ctx is not None:
            fmodel, frozen, keys = ctx.fmodel, ctx.frozen_params, ctx.batch_keys
            trainable = ctx.trainable_params
        else:
            fmodel, trainable, frozen = make_functional(
                self._model, partition_trainable=True
            )
            keys = self._discover_batch_keys()
        params = {**frozen, **trainable}

        inputs = self._prepare_input(inputs)
        # TR-DPO: score eval against the current EMA reference, not the seed
        # columns (mirrors training's _augment_inputs, without an EMA step).
        if self._sync_ref_model and self._tr_ref is not None:
            inputs = self._inject_tr_ref_logps(inputs)
        batch_args = tuple(inputs[k] for k in keys)

        def per_example(p, *args):
            return self.compute_per_example_loss_and_metrics(
                fmodel, p, dict(zip(keys, args, strict=False))
            )

        vmapped = torch.vmap(per_example, in_dims=(None,) + (0,) * len(keys))

        amp_dtype = self._amp_dtype
        was_training = self._model.training
        if was_training:
            self._model.eval()
        try:
            with torch.no_grad():
                if amp_dtype is not None:
                    with torch.autocast(device_type=self._device.type, dtype=amp_dtype):
                        loss, aux = vmapped(params, *batch_args)
                else:
                    loss, aux = vmapped(params, *batch_args)
        finally:
            if was_training:
                self._model.train()

        self._pending_eval_aux = {name: v.detach() for name, v in aux.items()}
        return loss.detach(), None, None
