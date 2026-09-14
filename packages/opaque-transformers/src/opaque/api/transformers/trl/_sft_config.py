# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
#
# Config surface and field semantics adapted from Hugging Face TRL
# (Apache-2.0; https://github.com/huggingface/trl), then modified for
# Opaque's per-example DP training flow.
# See ../../../../../NOTICE in this package for the full attribution.
"""``SFTConfig`` — training arguments for :class:`SFTTrainer`.

Mirrors ``trl.SFTConfig`` for the subset meaningful under per-example DP,
extending Opaque's standalone
:class:`~opaque.api.transformers.trainer._training_arguments.TrainingArguments`. Fields with
no DP meaning (DeepSpeed/FSDP/Accelerate knobs, VLM args, packing / padding-free)
are absent, so passing them is an unexpected-keyword ``TypeError``; an unknown
``loss_type`` fails at the trainer's dispatch table, not a curated check here.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from opaque.api.transformers.trainer._training_arguments import TrainingArguments

from ._sft_convert import _convert_trl_sft_config


@dataclasses.dataclass
class SFTConfig(TrainingArguments):
    """Arguments for supervised fine-tuning on :class:`Trainer`.

    Adds TRL-parity data-prep / loss fields on top of
    :class:`TrainingArguments`, keeping the TRL field names and defaults so the
    two configs read as analogues.
    """

    # ---- Learning rate override (TRL default differs from HF) ------------
    learning_rate: float = 2e-5

    # ---- Model loading ---------------------------------------------------
    #: Extra kwargs forwarded to ``AutoModelForCausalLM.from_pretrained`` when
    #: ``model`` is passed as a string (e.g. ``torch_dtype``, ``attn_implementation``).
    #: Ignored when ``model`` is an already-instantiated module.
    model_init_kwargs: dict | None = None
    #: Allow loading a model or tokenizer that ships custom Hub Python code.
    #: This is forwarded only when the trainer receives a string model identifier.
    trust_remote_code: bool = False

    # ---- Data preparation ------------------------------------------------
    #: Name of the column holding raw text on a language-modeling dataset.
    dataset_text_field: str = "text"
    # ``truncation_mode`` is intentionally absent: tokenization keeps the start
    # of the sequence (``keep_start``, TRL's default and forward path); TRL's
    # deprecated ``keep_end`` is not offered, so passing it is a TypeError.
    #: Maximum tokenized sequence length; ``None`` disables truncation.
    max_length: int | None = 1024
    #: Compute the loss only over completion tokens (prompt-completion data).
    #: ``None`` auto-detects from the dataset format at trainer-init time.
    completion_only_loss: bool | None = None
    #: EOS token appended to plain-text examples so the model learns to stop.
    #: When set, this exact token overrides ``tokenizer.eos_token``; when ``None``
    #: the tokenizer's own ``eos_token`` is used, or nothing if it has none.
    eos_token: str | None = None
    #: Pad the collated batch length up to a multiple of this value.
    pad_to_multiple_of: int | None = None
    #: Number of processes for ``datasets.map`` during preprocessing.
    dataset_num_proc: int | None = None
    #: Compute the loss only over assistant turns (conversational data). Uses
    #: the ``{% generation %}``-marked training chat template + the assistant
    #: token mask. Implies completion-only masking for chat data.
    assistant_only_loss: bool = False
    #: Path to a tokenizer dir or Jinja file whose chat template (and special
    #: tokens) is cloned onto ``processing_class`` before tokenization.
    chat_template_path: str | None = None

    # ---- Loss ------------------------------------------------------------
    #: ``"nll"`` (standard CE), ``"dft"`` (Dynamic Fine-Tuning), or the fused,
    #: logits-free ``"chunked_nll"`` (enables the ``fused_linear_cross_entropy``
    #: kernel). Unknown values fail at the trainer's loss-dispatch table.
    loss_type: str = "nll"

    # ``activation_offloading`` is inherited from the base ``TrainingArguments``
    # (shared by SFT and DPO); the base ``Trainer`` reads it.

    # ---- Telemetry -------------------------------------------------------
    #: Log the per-step completion-metric telemetry (``entropy``,
    #: ``mean_token_accuracy``, ``logits/*``). When ``False`` these logits-derived
    #: diagnostics are skipped, which also clears the way for the fused,
    #: logits-free loss path (see :class:`SFTTrainer`).
    log_completion_metrics: bool = True

    # ---- TRL base-default overrides (shared base default unchanged for
    # other trainers) ----------------------------------------------------
    #: TRL logs every 10 steps by default (vs. HF's 500).
    logging_steps: float = 10
    #: GC disabled by default: the vmap DP path recomputes activations once per
    #: microbatch, so GC multiplies that cost without memory benefit on models
    #: that fit. Enable explicitly when memory is the bottleneck.
    gradient_checkpointing: bool = False
    #: Opaque-specific (no TRL analogue): enable the model-level Triton kernels
    #: (``rope`` / ``rms_norm`` / ``activation`` / non-fused ``cross_entropy``)
    #: by default — they cut per-sample-gradient memory/compute under the vmap DP
    #: path. CUDA + Triton only; no-op on CPU/MPS. The conditional fused-linear
    #: CE wrapper is installed automatically; SFT activates it only for eligible
    #: logits-free NLL calls, while eager objectives and telemetry retain logits.
    use_performance_kernels: bool = True

    def __post_init__(self) -> None:
        # DP-driven: the collator consumes raw dataset columns
        # (``completion_mask`` is folded into ``-100`` labels) that are not
        # ``model.forward`` parameters and would be stripped by column pruning.
        self.remove_unused_columns = False
        # TRL parity: default bf16 on when supported and no precision was set.
        # The base still raises if bf16 was set explicitly on unsupported
        # hardware, so only auto-enable when it's actually available.
        if not self.use_cpu and not self.bf16 and not self.bf16_full_eval:
            import torch

            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.bf16 = True
        if self.trust_remote_code:
            self.model_init_kwargs = dict(self.model_init_kwargs or {})
            self.model_init_kwargs["trust_remote_code"] = True
        super().__post_init__()

    @classmethod
    def from_trl(
        cls,
        trl_cfg: Any,
        *,
        # DP knobs (one of noise_multiplier / target_epsilon required)
        privacy_noise_multiplier: float | None = None,
        privacy_target_epsilon: float | None = None,
        privacy_target_delta: float | None = None,
        clipping_norm: float | dict[str, float] | None = None,
        privacy_noise_mechanism: str = "gaussian",
        clipping_mode: str = "fixed",
        clipping_kwargs: dict[str, Any] | None = None,
        sampling_mode: str = "auto",
        sampling_kwargs: dict[str, Any] | None = None,
        noise_calibration_kwargs: dict[str, Any] | None = None,
        # Behavior
        strict: bool = True,
        **opaque_overrides: Any,
    ) -> SFTConfig:
        """Convert a ``trl.SFTConfig`` to an opaque ``SFTConfig``.

        Requires the optional ``trl`` extra:
        ``pip install opaque[trl]``.

        TRL-specific fields (``dataset_text_field``, ``chat_template_path``,
        ``completion_only_loss``, ``assistant_only_loss``, ``loss_type``,
        ``eos_token``, ``max_length``, ``pad_to_multiple_of``,
        ``dataset_num_proc``, ``model_init_kwargs``, ``trust_remote_code``) are
        copied directly.

        Fields TRL has that opaque does not implement raise
        ``ValueError``: ``packing``, ``padding_free``, ``eval_packing``,
        ``shuffle_dataset``, ``truncation_mode='keep_end'``, and ``pad_token``.
        A nonzero ``router_aux_loss_coef`` warns and trains as if it were 0.

        HF-inherited fields go through the same translation as
        :meth:`TrainingArguments.from_hf` — same DP-knob requirement, same
        ``per_device * gradient_accumulation_steps`` → opaque logical
        batch collapse.

        Raises
        ------
        ImportError
            If the optional ``trl`` dependency is not installed.
        TypeError
            If ``trl_cfg`` is not a ``trl.SFTConfig`` instance.
        ValueError
            If a required DP knob is missing, or any REJECT_IF_SET field
            is set to a non-default value.
        """
        dp_overrides: dict[str, Any] = {
            "privacy_noise_multiplier": privacy_noise_multiplier,
            "privacy_target_epsilon": privacy_target_epsilon,
            "privacy_noise_mechanism": privacy_noise_mechanism,
            "clipping_mode": clipping_mode,
            "sampling_mode": sampling_mode,
        }
        # ``clipping_norm`` only overrides when explicitly passed; otherwise the
        # value derived from HF ``max_grad_norm`` (or opaque's own default)
        # stands.
        if clipping_norm is not None:
            dp_overrides["clipping_norm"] = clipping_norm
        if privacy_target_delta is not None:
            dp_overrides["privacy_target_delta"] = privacy_target_delta
        if clipping_kwargs is not None:
            dp_overrides["clipping_kwargs"] = clipping_kwargs
        if sampling_kwargs is not None:
            dp_overrides["sampling_kwargs"] = sampling_kwargs
        if noise_calibration_kwargs is not None:
            dp_overrides["noise_calibration_kwargs"] = noise_calibration_kwargs

        converted = _convert_trl_sft_config(trl_cfg, strict=strict, **dp_overrides)
        converted.update(opaque_overrides)
        return cls(**converted)
