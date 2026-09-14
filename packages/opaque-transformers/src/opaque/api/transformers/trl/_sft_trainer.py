# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
#
# Trainer structure and method layout adapted from Hugging Face TRL
# (Apache-2.0; https://github.com/huggingface/trl), then modified for
# Opaque's per-example DP training flow.
# See ../../../../../NOTICE in this package for the full attribution.
"""``SFTTrainer`` — supervised fine-tuning on :class:`Trainer`.

Mirrors ``trl.SFTTrainer`` in structure and method names — ``_prepare_dataset``
/ ``tokenize_row`` for data prep, a language-modeling collator, and an
``nll``/``dft`` loss dispatch — but routes the training gradient through
Opaque's per-example :meth:`Trainer.compute_per_example_loss` hook.
"""

from __future__ import annotations

import inspect
from operator import attrgetter
from typing import TYPE_CHECKING, Any

import torch

from opaque.alignment.data import (
    apply_chat_template_with_mask,
    clone_chat_template,
    get_training_chat_template,
)
from opaque.alignment.metric import entropy_from_logits, mean_token_accuracy
from opaque.alignment.sft.collator import language_modeling_collator
from opaque.alignment.sft.loss import dft_loss, fused_dft_loss, nll_loss
from opaque.api.transformers.trainer import Trainer
from opaque.exceptions import ConfigurationError

from ._sft_config import SFTConfig

_IGNORE_INDEX = -100

if TYPE_CHECKING:
    from collections.abc import Callable

# Loss dispatch (``loss_type`` → alignment head); unknown values raise KeyError.
# ``chunked_nll`` is handled separately (fused linear-CE forward, logits-free).
_SFT_LOSSES: dict[str, Callable] = {"nll": nll_loss, "dft": dft_loss}

# Columns that may carry chat-format conversations.
_CHAT_COLUMNS = ("messages", "conversations", "chat")


def _detect_chat_column(row: dict) -> str | None:
    """Return the chat-message column in *row*, or ``None`` for plain text."""
    for col in _CHAT_COLUMNS:
        value = row.get(col)
        if (
            isinstance(value, list)
            and value
            and isinstance(value[0], dict)
            and "role" in value[0]
            and "content" in value[0]
        ):
            return col
    return None


def _resolve_fused_handles(model: Any, eligible: bool) -> tuple[str | None, str | None]:
    """Resolve the backbone-prefix + lm_head param-key for the fused loss path.

    Returns ``(backbone_prefix, lm_head_param_name)`` — the **dotted attribute
    path** from ``model`` to the backbone submodule (e.g. ``"model"`` on a bare
    causal-LM, ``"base_model.model.model"`` on a PEFT-wrapped one) and the key of
    the lm_head weight in ``model.named_parameters()``. The path doubles as the
    params-key prefix when slicing the backbone-scoped sub-dict. The lm_head
    key is found by id-matching ``get_output_embeddings().weight`` against the
    named params, so it resolves correctly even under **tied embeddings** (where
    ``lm_head.weight`` is absent and the key is the embedding's
    ``model.embed_tokens.weight``).

    Under PEFT the dotted prefix must walk to the real backbone at
    ``peft_model.base_model.model.<prefix>``: calling the inner causal-LM
    directly would return ``CausalLMOutputWithPast`` rather than the
    ``BaseModelOutputWithPast`` that ``_last_hidden_state`` needs.

    Returns ``(None, None)`` when the run is ineligible or the model does not
    expose the ``backbone + output-embeddings`` shape the fused path needs (the
    seam then keeps the eager logits path).
    """
    if not eligible or model is None:
        return None, None
    # Unwrap PEFT: PeftModel.base_model is the adapter wrapper (LoraModel etc.),
    # whose ``.model`` is the original *ForCausalLM with the ``base_model_prefix``
    # we want. ``peft_config`` is the PEFT marker (PreTrainedModel.base_model is a
    # self-reference, so it cannot be used for detection).
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
    # ``params`` is keyed off the OUTER model's named_parameters tree (PEFT keys
    # live under ``base_model.model.``), so the id-lookup runs on the outer model.
    name_by_id = {id(p): n for n, p in model.named_parameters()}
    lm_head_param_name = name_by_id.get(id(weight))
    if lm_head_param_name is None:
        return None, None
    return path_to_inner + prefix, lm_head_param_name


class SFTTrainer(Trainer):
    """DP supervised fine-tuning trainer.

    Args mirror ``trl.SFTTrainer.__init__`` for the supported subset; the model
    forward and gradient go through the DP per-example path.
    """

    def __init__(
        self,
        model: Any = None,
        args: SFTConfig | None = None,
        data_collator: Callable | None = None,
        train_dataset: Any = None,
        eval_dataset: Any = None,
        processing_class: Any = None,
        compute_loss_func: Callable | None = None,
        compute_metrics: Callable | None = None,
        callbacks: list[Any] | None = None,
        optimizers: tuple[Any | None, Any | None] = (None, None),
        optimizer_cls_and_kwargs: tuple[Any, dict[str, Any]] | None = None,
        preprocess_logits_for_metrics: Callable | None = None,
        peft_config: Any = None,
        formatting_func: Callable[[dict], str] | None = None,
    ) -> None:
        # ---- args normalization (accept a plain TrainingArguments) --------
        if args is None:
            args = SFTConfig(output_dir="trainer_output")
        elif not isinstance(args, SFTConfig):
            import dataclasses

            args = SFTConfig(
                **{f.name: getattr(args, f.name) for f in dataclasses.fields(args)}
            )

        # ---- model (string ⇒ load) ----------------------------------------
        if isinstance(model, str):
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                model, **(args.model_init_kwargs or {})
            )

        # ---- tokenizer / processing_class ---------------------------------
        processing_class = self._resolve_tokenizer(model, processing_class, args)

        # ---- chat template clone (resizes embeddings) ---------------------
        added_tokens: list[int] = []
        if args.chat_template_path is not None:
            model, processing_class, added_tokens = clone_chat_template(
                model, processing_class, args.chat_template_path
            )

        # ---- PEFT ----------------------------------------------------------
        if peft_config is not None:
            from peft import get_peft_model

            # Newly cloned-in special tokens have randomly-initialised embedding
            # rows a frozen base would never learn. Mark exactly those rows
            # trainable and keep the lm_head in modules_to_save (with a warning)
            # so the model can learn to emit them.
            if added_tokens:
                self._mark_added_tokens_trainable(peft_config, added_tokens)
            model = get_peft_model(model, peft_config)

        # ---- custom-loss guard --------------------------------------------
        # A custom compute_loss_func is only meaningful on the ``nll`` path, which
        # has the model logits to hand it. ``dft`` computes its own token-weighted
        # loss and ``chunked_nll`` is logits-free, so neither can route a custom
        # loss.
        if args.loss_type in ("dft", "chunked_nll") and compute_loss_func is not None:
            raise ConfigurationError(
                *(
                    f"loss_type={args.loss_type!r} computes its own loss; pass "
                    "loss_type='nll' to use a custom compute_loss_func.",
                )
            )
        # Loss path: ``chunked_nll`` computes its own loss via the fused linear-CE
        # kernel (logits-free on CUDA, eager fallback elsewhere); ``nll`` / ``dft``
        # dispatch to an alignment head.
        self._loss_type: str = args.loss_type
        # ``dft`` is a per-example objective in its own right (token-normalized
        # with a DP-safe per-example divisor): eval aggregates it as the plain
        # per-example mean matching training, not the per-token CE
        # reconstruction, so no realized token-count divisor enters
        # best-model selection (#384 review).
        self._eval_token_weighted_loss = args.loss_type != "dft"
        # Gate logits-derived completion telemetry (entropy / mean_token_accuracy
        # / logits/*). When off, the (loss, aux) seam returns an empty aux dict so
        # the loss path runs without materialising those metrics.
        self._log_completion_metrics: bool = args.log_completion_metrics

        # Fused-loss gate: when telemetry is off and no logits-consuming feature is
        # active, the loss is computed logits-free — ``nll`` via the model-level
        # ``fused_linear_cross_entropy`` forward (as ``chunked_nll`` does), ``dft``
        # via the ``fused_dft_loss`` primitive over the backbone's last hidden
        # state. A custom ``compute_loss_func`` needs logits. Metric callbacks do
        # not affect eligibility: the fused forward is requested only for
        # loss-only calls, while prediction-producing evaluation remains eager.
        self._fused_loss_eligible = (
            args.loss_type in ("nll", "dft")
            and not self._log_completion_metrics
            and compute_loss_func is None
        )
        # ``nll`` rides the model-level fused forward (enables the kernel patch);
        # ``dft`` rides the primitive over the last hidden state. Both fall back to
        # eager on CPU / non-half automatically.
        self._fused_nll = self._fused_loss_eligible and args.loss_type == "nll"
        self._fused_dft = self._fused_loss_eligible and args.loss_type == "dft"
        # Resolve the backbone-prefix + lm_head param-key once for the ``dft``
        # fused primitive (robust to tied embeddings). Loss-only evaluation does
        # not consume telemetry or predictions, so it can use this path even
        # when those features keep training or metric evaluation eager.
        self._backbone_prefix, self._lm_head_param_name = _resolve_fused_handles(
            model, args.loss_type == "dft"
        )
        self._fused_dft_loss_only = (
            args.loss_type == "dft" and self._lm_head_param_name is not None
        )
        self._fused_dft = self._fused_dft and self._lm_head_param_name is not None

        if args.loss_type == "chunked_nll" or self._fused_nll:
            # Both let the model compute its own loss via the fused linear-CE
            # forward (``chunked_nll`` always; eligible ``nll`` when telemetry off).
            self._loss_fn: Callable | None = (
                None if args.loss_type == "chunked_nll" else _SFT_LOSSES[args.loss_type]
            )
            cfg = dict(args.performance_kernels_config or {})
            cfg["fused_linear_cross_entropy"] = True
            args.performance_kernels_config = cfg
        else:
            self._loss_fn = _SFT_LOSSES[args.loss_type]

        # ---- dataset preprocessing (before super().__init__) --------------
        self._formatting_func = formatting_func
        completion_only = self._resolve_completion_only(train_dataset, args)
        mask_labels = completion_only or args.assistant_only_loss

        train_dataset = self._prepare_dataset(
            train_dataset, processing_class, args, "train"
        )
        if eval_dataset is not None and not isinstance(eval_dataset, dict):
            eval_dataset = self._prepare_dataset(
                eval_dataset, processing_class, args, "eval"
            )

        # ---- collator ------------------------------------------------------
        if data_collator is None:
            data_collator = language_modeling_collator(
                pad_token_id=processing_class.pad_token_id,
                max_length=args.max_length,
                completion_only_loss=mask_labels,
                pad_to_multiple_of=args.pad_to_multiple_of,
            )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        forward_parameters = inspect.signature(self._model.forward).parameters.values()
        self._fused_forward_uses_marker = any(
            parameter.name == "loss_only"
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in forward_parameters
        )

    # ------------------------------------------------------------------
    # PEFT: trainability of cloned-in token embeddings
    # ------------------------------------------------------------------
    @staticmethod
    def _mark_added_tokens_trainable(peft_config: Any, added_tokens: list[int]) -> None:
        """Make newly cloned-in token embeddings trainable under PEFT.

        Points ``peft_config.trainable_token_indices['embed_tokens']`` at the new
        token ids and ensures ``lm_head`` is in ``modules_to_save`` (warning when
        it has to be added). A no-op when ``added_tokens`` is empty.
        """
        if not added_tokens:
            return
        tti = getattr(peft_config, "trainable_token_indices", None)
        if tti is None:
            peft_config.trainable_token_indices = {"embed_tokens": list(added_tokens)}
        elif "embed_tokens" not in tti:
            tti["embed_tokens"] = list(added_tokens)
        else:
            tti["embed_tokens"] = list(tti["embed_tokens"]) + list(added_tokens)

        mts = getattr(peft_config, "modules_to_save", None)
        if mts is None or "lm_head" not in mts:
            import warnings

            warnings.warn(
                "New tokens were added to the chat template but 'lm_head' is not "
                "in the PEFT config's modules_to_save; adding it so the model can "
                "learn to generate them. Pass modules_to_save=['lm_head'] to "
                "silence this.",
                stacklevel=2,
            )
            if mts is None:
                peft_config.modules_to_save = ["lm_head"]
            else:
                mts.append("lm_head")

    # ------------------------------------------------------------------
    # Tokenizer / format resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_tokenizer(model: Any, processing_class: Any, args: SFTConfig) -> Any:
        if processing_class is None:
            from transformers import AutoTokenizer

            name = getattr(model.config, "_name_or_path", None)
            if not name:
                raise ConfigurationError(
                    *(
                        "processing_class is None and the model config has no "
                        "_name_or_path to load a tokenizer from; pass processing_class.",
                    )
                )
            processing_class = AutoTokenizer.from_pretrained(
                name,
                trust_remote_code=bool(
                    (args.model_init_kwargs or {}).get("trust_remote_code", False)
                ),
            )
        if args.eos_token is not None:
            processing_class.eos_token = args.eos_token
        if processing_class.pad_token_id is None:
            processing_class.pad_token = processing_class.eos_token
        return processing_class

    def _resolve_completion_only(self, dataset: Any, args: SFTConfig) -> bool:
        """Auto-detect completion-only loss when ``args`` leaves it ``None``.

        ``True`` for prompt-completion datasets and ``False`` for language-
        modeling datasets, including conversational data, matching TRL.
        """
        if dataset is None or len(dataset) == 0:
            return bool(args.completion_only_loss)
        row = dataset[0]
        is_chat = _detect_chat_column(row) is not None
        is_prompt_completion = "prompt" in row and "completion" in row
        has_completion_mask = "completion_mask" in row
        if args.assistant_only_loss and not is_chat:
            ConfigurationError.raise_(
                "assistant_only_loss=True requires a conversational dataset "
                "with a messages, conversations, or chat column."
            )
        if (
            args.completion_only_loss
            and not args.assistant_only_loss
            and not is_prompt_completion
            and not has_completion_mask
        ):
            ConfigurationError.raise_(
                "completion_only_loss=True requires a prompt-completion dataset "
                "or examples with a completion_mask. Use assistant_only_loss=True "
                "for conversational datasets."
            )
        if args.completion_only_loss is not None:
            return bool(args.completion_only_loss)
        return is_prompt_completion

    # ------------------------------------------------------------------
    # Dataset preparation (TRL-shaped)
    # ------------------------------------------------------------------
    def _prepare_dataset(
        self, dataset: Any, processing_class: Any, args: SFTConfig, dataset_name: str
    ) -> Any:
        """Tokenize *dataset* into ``input_ids`` (+ ``completion_mask``).

        Mirrors ``trl.SFTTrainer._prepare_dataset``: skip already-tokenized
        datasets, optionally apply a ``formatting_func``, then dispatch by
        format (plain text / prompt-completion / chat).
        """
        if dataset is None or len(dataset) == 0:
            # Empty dataset: skip tokenization so ``Trainer``'s clearer
            # "train_dataset is empty" validation fires later, rather than an
            # IndexError on the ``dataset[0]`` chat-column probe below.
            return dataset
        column_names = list(getattr(dataset, "column_names", []) or [])
        if "input_ids" in column_names:
            return dataset  # already tokenized

        formatting_func = self._formatting_func
        if formatting_func is not None:
            dataset_text_field = args.dataset_text_field
            dataset = dataset.map(
                lambda ex: {dataset_text_field: formatting_func(ex)},
                num_proc=args.dataset_num_proc,
            )
            column_names = list(dataset.column_names)

        row = dataset[0]
        chat_col = _detect_chat_column(row)

        # Assistant-only chat data needs generation markers so
        # ``apply_chat_template_with_mask`` can recover its token mask.
        if chat_col is not None and args.assistant_only_loss:
            processing_class.chat_template = get_training_chat_template(
                processing_class
            )

        tokenize_fn = self.tokenize_row

        def tokenize_row(example: dict) -> dict:
            return tokenize_fn(example, processing_class, args, chat_col=chat_col)

        return dataset.map(
            tokenize_row,
            remove_columns=column_names,
            num_proc=args.dataset_num_proc,
            desc=f"Tokenizing {dataset_name} dataset",
        )

    @staticmethod
    def tokenize_row(
        example: dict,
        processing_class: Any,
        args: SFTConfig,
        *,
        chat_col: str | None,
    ) -> dict:
        """Tokenize one SFT example into model-ready columns."""
        max_length = args.max_length

        if chat_col is not None:
            if args.assistant_only_loss:
                encoded = apply_chat_template_with_mask(
                    processing_class,
                    example[chat_col],
                    max_length=max_length,
                    truncation=max_length is not None,
                )
            else:
                encoded = processing_class.apply_chat_template(
                    example[chat_col],
                    tokenize=True,
                    return_dict=True,
                    max_length=max_length,
                    truncation=max_length is not None,
                )
            ids = encoded["input_ids"]
            if max_length is not None:
                ids = ids[:max_length]
            result = {"input_ids": ids}
            if args.assistant_only_loss:
                cmask = encoded["completion_mask"]
                result["completion_mask"] = (
                    cmask[:max_length] if max_length is not None else cmask
                )
            return result

        if "prompt" in example and "completion" in example:
            prompt_ids = processing_class(example["prompt"], add_special_tokens=True)[
                "input_ids"
            ]
            full_ids = (
                prompt_ids
                + processing_class(example["completion"], add_special_tokens=False)[
                    "input_ids"
                ]
            )
            cmask = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))
            if max_length is not None:
                full_ids, cmask = full_ids[:max_length], cmask[:max_length]
            return {"input_ids": full_ids, "completion_mask": cmask}

        # Plain text field. Append the EOS token so the model learns to stop:
        # the explicit ``args.eos_token`` when set, else the tokenizer's own
        # ``eos_token`` (TRL parity). When neither exists, nothing is appended.
        text = example[args.dataset_text_field]
        eos = (
            args.eos_token if args.eos_token is not None else processing_class.eos_token
        )
        if eos is not None and not text.endswith(eos):
            text = text + eos
        ids = processing_class(
            text, truncation=max_length is not None, max_length=max_length
        )["input_ids"]
        return {"input_ids": ids}

    # ------------------------------------------------------------------
    # Fused logits-free loss (plan §E) — last hidden state only
    # ------------------------------------------------------------------
    def _last_hidden_state(
        self,
        params: dict[str, Any],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Backbone last hidden state ``(T, H)`` only — no lm_head, no all-layers.

        Calls the backbone submodule (resolved via :func:`_resolve_fused_handles`)
        functionally with the backbone-scoped slice of ``params`` (keys under the
        ``"<prefix>."`` namespace, re-rooted to the submodule). Returns just the
        last-layer hidden state — unlike ``output_hidden_states=True``, which
        stacks all ``L+1`` layers and can exceed the ``(T, V)`` logits the fused
        path is avoiding. The HF backbones in scope return it as ``out[0]`` /
        ``out.last_hidden_state``.

        Calling the unwrapped backbone (not the inner causal-LM) guarantees a
        ``BaseModelOutputWithPast`` rather than ``CausalLMOutputWithPast``; the
        dotted prefix lets ``attrgetter`` walk PEFT wrappers to reach it.

        The ``batchify`` vmap-safety patch is applied to the *causal-LM* class,
        not the backbone, so under ``vmap`` (per-example ``(T,)`` inputs) we add
        the batch dim here and strip it on exit. Already-batched ``(B, T)`` inputs
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

    # ------------------------------------------------------------------
    # Loss (DP per-example hook)
    # ------------------------------------------------------------------
    def compute_per_example_loss(
        self,
        fmodel: Callable[..., Any],
        params: dict[str, Any],
        inputs: dict[str, Any],
        *,
        return_logits: bool = False,
    ) -> Any:
        """One example's SFT loss (vmap-batched by :class:`Trainer`).

        The collator already folds ``completion_mask`` into ``-100`` labels, so
        this is a single forward + the configured head (``nll`` / ``dft``); both
        heads use a DP-safe per-example divisor. A custom ``compute_loss_func``
        (only valid on the ``nll`` path; ``dft`` / ``chunked_nll`` are guarded at
        init) is honoured here per ``Trainer``'s contract
        ``compute_loss_func(outputs, labels) -> scalar`` — it runs inside vmap,
        so it must be a pure per-example tensor op (no ``num_items_in_batch`` /
        batch coupling). Per-example telemetry lives in
        :meth:`compute_per_example_loss_and_metrics`.
        """
        # Fused logits-free per-example loss when eligible.
        if self._loss_type == "chunked_nll" or self._fused_nll:
            # The model computes the fused NLL when given labels (``chunked_nll``
            # always; eligible ``nll`` reuses the same forward, per-example equal
            # to ``nll_loss``). Falls back to the eager projection on CPU / non-half.
            out = fmodel(
                params,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
                **(
                    {"loss_only": not return_logits}
                    if self._fused_forward_uses_marker
                    else {}
                ),
            )
            loss = out["loss"]
            logits = out.get("logits")  # None on the fused path
        elif self._fused_dft or (not return_logits and self._fused_dft_loss_only):
            # ``dft`` has no model-level fused forward, so project the backbone's
            # last hidden state ``(T, H)`` through ``fused_dft_loss`` (falls back
            # to the eager logits form on CPU / non-half).
            hidden = self._last_hidden_state(
                params, inputs["input_ids"], inputs["attention_mask"]
            )
            lm_head_weight = params[self._lm_head_param_name]
            loss = fused_dft_loss(hidden, lm_head_weight, inputs["labels"])
            logits = None  # logits-free path
        else:
            out = fmodel(
                params,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            logits = out.logits
            if self._compute_loss_func is not None:
                loss = self._compute_loss_func(out, inputs["labels"])
            else:
                loss = self._loss_fn(out.logits, inputs["labels"])
        if return_logits:
            return loss, logits
        return loss

    def prediction_step(
        self,
        model: Any,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[Any, Any, Any]:
        """Evaluate SFT with the *training* objective.

        For ``loss_type="dft"`` the model's built-in cross-entropy differs from
        the detached-confidence-weighted DFT loss used in training, so eval —
        and best-model selection — must score with the per-example DFT loss.
        The base loop only routes through the per-example loss when predictions
        are also requested (``include_for_metrics=['loss']`` and not
        ``prediction_loss_only``); the common best-model-selection path
        (``prediction_loss_only=True``) would otherwise read the model's CE.
        This override always scores DFT via the base per-example eval closure,
        while still surfacing logits (``ignore_keys``-filtered, base parity)
        and labels so ``compute_metrics`` keeps working.  ``eval_loss``
        aggregates as the plain per-example mean in every mode
        (``_eval_token_weighted_loss = False``): the DFT scalar is already
        token-normalized with a DP-safe per-example divisor, so no
        token-count reweighting is applied.  ``nll`` / ``chunked_nll`` equal
        the model's CE head and keep the inherited path (#384).
        """
        if self._loss_type != "dft":
            if (
                self._loss_type == "chunked_nll" or self._fused_nll
            ) and self._fused_forward_uses_marker:
                inputs = {
                    **inputs,
                    "loss_only": prediction_loss_only,
                }
            return super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys
            )
        del model
        label_keys = list(self._label_names) if self._label_names else []
        has_labels = bool(label_keys) and all(
            inputs.get(k) is not None for k in label_keys
        )
        inputs = self._prepare_input(inputs)

        # Loss-only evaluation must retain the per-example DFT scalars for plain
        # per-example aggregation, but has no consumer for logits. Ask the base
        # helper for its separate no-logits closure so this call can also take
        # DFT's fused logits-free path despite telemetry being enabled elsewhere.
        vmapped_fn, _argnums, batch_keys = self._get_eval_per_example_loss_fn(
            return_logits=not prediction_loss_only
        )
        if self._ctx is not None:
            trainable = self._ctx.trainable_params
        else:
            trainable = {
                name: p for name, p in self._model.named_parameters() if p.requires_grad
            }
        # Fail loudly on a mismatched eval collator (base-path parity) —
        # a missing key would otherwise surface as an opaque vmap error.
        missing = [k for k in batch_keys if inputs.get(k) is None]
        if missing:
            raise ConfigurationError(
                *(
                    "DFT eval expects the eval batch to carry the train-discovered "
                    f"keys {list(batch_keys)!r}, but {missing!r} are absent (or "
                    "None); align the eval collator/dataset with the training one.",
                )
            )
        batch_args = tuple(inputs[k] for k in batch_keys)

        amp_dtype = self._amp_dtype
        was_training = self._model.training
        if was_training:
            self._model.eval()
        try:
            with torch.no_grad():
                if amp_dtype is not None:
                    with torch.autocast(device_type=self._device.type, dtype=amp_dtype):
                        output = vmapped_fn(trainable, *batch_args)
                else:
                    output = vmapped_fn(trainable, *batch_args)
        finally:
            if was_training:
                self._model.train()

        if prediction_loss_only:
            return output.detach(), None, None
        loss, logits = output
        loss = loss.detach()
        # Same logits filtering as the base per-example path
        # (``Trainer.prediction_step``).
        preds = (
            None
            if (ignore_keys and "logits" in ignore_keys)
            else (logits.detach() if logits is not None else None)
        )
        if has_labels:
            label_values = tuple(inputs[k].detach() for k in label_keys)
            labels = label_values[0] if len(label_values) == 1 else label_values
        else:
            labels = None
        return loss, preds, labels

    def compute_per_example_loss_and_metrics(
        self,
        fmodel: Callable[..., Any],
        params: dict[str, Any],
        inputs: dict[str, Any],
    ) -> tuple[Any, dict[str, Any]]:
        """One example's SFT ``(loss, telemetry)`` (vmap-batched).

        Adds the per-example diagnostics TRL logs each step — ``entropy`` and
        ``mean_token_accuracy`` over the supervised (non-``-100``) tokens —
        riding the clipped-grad aux channel (and the symmetric eval aux). They
        are computed from the model logits, **independent of the loss head**, so
        a custom ``compute_loss_func`` that returns only a scalar still gets
        telemetry (the aux fail-safe). When logits are unavailable (the
        logits-free ``chunked_nll`` path), or when ``log_completion_metrics`` is
        ``False``, the telemetry dict is empty and only the loss is logged — the
        harness handles an empty aux dict.
        """
        # Telemetry gated off: skip materialising logits-derived metrics entirely
        # and do not request logits. The loss path still runs (and may take the
        # logits-free fused route in ``compute_per_example_loss``).
        if not self._log_completion_metrics:
            return self.compute_per_example_loss(fmodel, params, inputs), {}
        loss, logits = self.compute_per_example_loss(
            fmodel, params, inputs, return_logits=True
        )
        if logits is None:
            return loss, {}
        labels = inputs["labels"]
        # ``mask`` is the FULL-length supervised mask; ``entropy_from_logits`` and
        # ``mean_token_accuracy`` both shift internally to next-token alignment
        # (logits[..., :-1, :] vs mask[..., 1:]), so we pass the full mask here.
        mask = labels != _IGNORE_INDEX
        return loss, {
            "entropy": entropy_from_logits(logits, mask),
            "mean_token_accuracy": mean_token_accuracy(logits, labels, mask),
        }
