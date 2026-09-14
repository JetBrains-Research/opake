"""DP Direct Preference Optimization via the class-based ``DPOTrainer``.

Hands ``opaque.alignment`` DPO primitives to
:class:`opaque.transformers.trl.DPOTrainer`, which precomputes the reference
log-probs and orchestrates the per-example DP path on top of ``Trainer``.

The reference need is derived from ``loss_type``: the reference-free heads
``simpo`` / ``cpo`` / ``orpo`` / ``sft`` skip the reference precompute, every
other head requires one. A reference is resolved from ``--ref-model`` (a
path/repo id), from the PEFT base model when ``--peft`` is set, or auto-loaded
from the policy's own path.

Exposes TR-DPO reference sync (``--sync-ref-model``), MPO multi-loss blends
(``--loss-type a b ... --loss-weights w1 w2 ...``), WPO weighting
(``--use-weighting``), LD-DPO (``--ld-alpha``), f-divergence regularisers
(``--f-divergence-type``), the reference-free heads, and the completion-metric
telemetry opt-in (``--log-completion-metrics``).

Examples::

    # Vanilla DPO (LoRA policy, base model as the reference)
    uv run python examples/train_dpo_trainer.py \\
      --model HuggingFaceTB/SmolLM2-135M-Instruct \\
      --dataset CyberNative/Code_Vulnerability_Security_DPO \\
      --beta 0.1 --max-length 512 --batch-size 8 --microbatch-size 2 \\
      --max-steps 50 --learning-rate 1e-4 --peft \\
      --clipping-norm 1.0 --noise-multiplier 0.8 --log-steps 5 --seed 42

    # Reference-free SimPO (no reference precompute)
    uv run python examples/train_dpo_trainer.py --loss-type simpo --simpo-gamma 0.5

    # MPO blend: weighted sigmoid + sft regulariser
    uv run python examples/train_dpo_trainer.py \\
      --loss-type sigmoid sft --loss-weights 1.0 0.5

    # TR-DPO (EMA reference sync; full fine-tuning, not PEFT)
    uv run python examples/train_dpo_trainer.py \\
      --sync-ref-model --ref-model-mixup-alpha 0.6 --ref-model-sync-steps 64
"""

from __future__ import annotations

import argparse
import os

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig


def _configure_reporting(no_wandb: bool) -> list[str]:
    """Set W&B env defaults and return TrainingArguments.report_to.

    Enables ``"wandb"`` reporting when ``WANDB_NAME`` / ``WANDB_PROJECT`` /
    ``WANDB_ENTITY`` / ``WANDB_TAGS`` are configured, since the DPOConfig
    default of ``[]`` would otherwise keep runs stdout-only.
    """
    if no_wandb:
        return []
    if not os.environ.get("WANDB_MODE"):
        os.environ["WANDB_MODE"] = (
            "online" if os.environ.get("WANDB_API_KEY") else "offline"
        )
    return ["wandb"]


from transformers import AutoModelForCausalLM, AutoTokenizer

from opaque.transformers.trl import DPOConfig, DPOTrainer


# A run is reference-free only when *every* head is in this set, so the trainer
# skips the reference precompute and ``--ref-model`` is not required.
_REFERENCE_FREE_HEADS = frozenset({"sft", "simpo", "cpo", "orpo"})


def _to_trl_canonical_dpo(row: dict) -> dict:
    """Collapse non-TRL preference shapes into (prompt, chosen, rejected).

    Two non-canonical schemas we handle:

    - CyberNative/Code_Vulnerability_Security_DPO: (system, question, chosen,
      rejected). Non-empty ``system`` prefixed onto ``question`` as a free-text
      preamble.
    - zed-industries/zeta: (events, input, output, rejected, assertions).
      ``events`` carries the user-edit history, ``input`` is the
      cursor-positioned code, ``output`` is the chosen completion, and
      ``rejected`` is an incorrect completion.

    chosen/rejected are passed through as plain strings (no chat template).
    """
    if "output" in row and "input" in row and "rejected" in row:
        # zed-industries/zeta schema
        events = (row.get("events") or "").strip()
        input_code = row["input"]
        prompt = f"{events}\n\n{input_code}" if events else input_code
        return {
            "prompt": prompt,
            "chosen": row["output"],
            "rejected": row["rejected"],
        }
    # CyberNative-style (system, question, chosen, rejected)
    system = (row.get("system") or "").strip()
    question = row["question"]
    return {
        "prompt": f"{system}\n\n{question}" if system else question,
        "chosen": row["chosen"],
        "rejected": row["rejected"],
    }


def _require_configured(parser, args, required=("model", "dataset")):
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        flags = ", ".join("--" + name.replace("_", "-") for name in missing)
        parser.error(f"missing required configuration: {flags}. Pass them directly.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DP DPO with the class-based DPOTrainer")
    p.add_argument(
        "--model",
        default=None,
        help="HuggingFace model name or local path (required)",
    )
    p.add_argument(
        "--ref-model",
        default=None,
        help="Reference policy path/repo id. Defaults to the policy's own path "
        "(auto-loaded) for reference-using heads; ignored for reference-free "
        "heads and when --peft is set (the PEFT base serves as the reference).",
    )
    p.add_argument(
        "--dataset",
        default=None,
        help="HuggingFace preference dataset (chosen/rejected columns; required)",
    )
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--num-train-samples", type=int, default=256)
    # --- Loss --------------------------------------------------------------
    p.add_argument(
        "--loss-type",
        nargs="+",
        default=["sigmoid"],
        help="One or more TRL-canonical heads. Multiple ⇒ MPO blend "
        "(pair with --loss-weights). Reference-free: simpo/cpo/orpo/sft.",
    )
    p.add_argument(
        "--loss-weights",
        nargs="+",
        type=float,
        default=None,
        help="Per-head weights for an MPO blend; must match --loss-type length.",
    )
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument(
        "--f-divergence-type",
        default="reverse_kl",
        choices=["reverse_kl", "forward_kl", "js_divergence", "alpha_divergence"],
        help="f-divergence regulariser on the log-ratios.",
    )
    p.add_argument(
        "--f-alpha-divergence-coef",
        type=float,
        default=0.5,
        help="alpha coefficient for --f-divergence-type alpha_divergence.",
    )
    p.add_argument(
        "--ld-alpha",
        type=float,
        default=None,
        help="LD-DPO verbose-token weight in [0, 1]; None disables LD-DPO.",
    )
    p.add_argument(
        "--use-weighting",
        action="store_true",
        help="WPO: reweight each pair by the policy's average completion prob.",
    )
    p.add_argument("--simpo-gamma", type=float, default=0.5)
    p.add_argument("--cpo-alpha", type=float, default=1.0)
    p.add_argument("--orpo-lambda", type=float, default=1.0)
    # --- TR-DPO (reference sync) ------------------------------------------
    p.add_argument(
        "--sync-ref-model",
        action="store_true",
        help="TR-DPO: EMA-sync the reference toward the policy (full FT, not PEFT).",
    )
    p.add_argument("--ref-model-mixup-alpha", type=float, default=0.6)
    p.add_argument("--ref-model-sync-steps", type=int, default=512)
    # --- Telemetry ---------------------------------------------------------
    p.add_argument(
        "--log-completion-metrics",
        dest="log_completion_metrics",
        action="store_true",
        help="Log entropy / mean_token_accuracy / logits/* per-pair. "
        "Forces the eager logits-materialising loss path (~12x microbatch hit "
        "on big-vocab models) — rewards/* and logps/* are still logged "
        "unconditionally on the fused path.",
    )
    p.set_defaults(log_completion_metrics=False)
    # --- Training ----------------------------------------------------------
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    # ``None`` → vmap over the full batch (no chunking). Override
    # explicitly if a model's per-example memory footprint requires
    # splitting the logical batch into smaller chunks.
    p.add_argument("--microbatch-size", type=int, default=None)
    p.add_argument(
        "--precompute-ref-batch-size",
        type=int,
        default=None,
        help="Batch for the one-shot reference-logp precompute (a regular "
        "batched forward, NOT vmap). Defaults to --batch-size; set small "
        "(e.g. 8) when --batch-size is large to avoid lm_head OOM in precompute.",
    )
    p.add_argument(
        "--stop-at-step",
        type=int,
        default=8,
        help="Stop the training loop after this many optimizer steps "
        "(early-stop knob, not a privacy-accounting target — privacy is "
        "calibrated from target_epsilon × steps × sample_rate regardless).",
    )
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--clipping-norm", type=float, default=1.0)
    p.add_argument(
        "--noise-multiplier",
        type=float,
        default=None,
        help=(
            "Fixed noise multiplier (skips calibration). Set to 0 for a "
            "non-private baseline: the chosen mechanism/sampler are kept, no "
            "noise is added, and the accountant reports epsilon=inf. Leave "
            "unset (the default) to let ``--target-epsilon`` drive calibration."
        ),
    )
    p.add_argument(
        "--target-epsilon",
        type=float,
        default=8.0,
        help=(
            "Target ε for noise calibration. Active when ``--noise-multiplier`` "
            "is left unset; the accountant solves for the σ that achieves this "
            "ε at the configured ``(max_steps, batch_size, num_train_samples)``. "
            "Default matches train_dpo.py / train_sft.py."
        ),
    )
    p.add_argument("--log-steps", type=int, default=1)
    p.add_argument("--output-dir", default="trainer_output/dpo")
    # --- Memory knobs ------------------------------------------------------
    p.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Recompute activations during the backward pass instead of "
        "storing them. Trades compute (~25%% slower) for memory (typically "
        "2-4× less activation memory). Necessary for big models at large batch.",
    )
    p.add_argument(
        "--activation-offloading",
        action="store_true",
        help="Offload saved-for-backward activations to CPU via "
        "``torch.autograd.graph.save_on_cpu``. Adds PCIe transfer overhead "
        "but lets activations exceed the GPU memory cap.",
    )
    p.add_argument(
        "--auto-find-microbatch-size",
        action="store_true",
        help="On CUDA OOM mid-step, halve ``microbatch-size`` and retry "
        "until the step fits (the per-rank logical batch is preserved, "
        "so privacy accounting is unchanged).",
    )
    p.add_argument(
        "--no-performance-kernels",
        action="store_true",
        help="Disable the model-level Triton kernels (rope/rms_norm/activation/"
        "cross_entropy). Use to isolate kernel-vs-eager behaviour — e.g. the "
        "RMSNorm vmap dW is batch-summed (DP-sensitivity bug for full-FT), so "
        "eager gives correct per-example norm-weight grads.",
    )
    # --- PEFT --------------------------------------------------------------
    p.add_argument(
        "--peft",
        action="store_true",
        help="Train a LoRA adapter (the frozen base serves as the reference via "
        "the PEFT null-ref path; incompatible with --sync-ref-model).",
    )
    p.add_argument("--lora-modules", nargs="+", default=["q_proj", "v_proj"])
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    # --- Eval --------------------------------------------------------------
    p.add_argument(
        "--eval-steps",
        type=int,
        default=10,
        help="Evaluate every N steps on a held-out slice (disjoint slice of "
        "the same dataset, after the --num-train-samples). Set to ``0`` to "
        "disable eval entirely. Default matches train_dpo.py / "
        "train_dpsgd_trainer.py.",
    )
    p.add_argument(
        "--num-eval-samples",
        type=int,
        default=64,
        help="Held-out preference-pair count for eval. Matches train_dpo.py. "
        "rewards/accuracies is a binary signal so wants a larger eval set than "
        "scalar loss; 2000 pairs ≈ 0.7%% std-err on the accuracy estimate.",
    )
    p.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=None,
        help="Eval batch size; defaults to --batch-size.",
    )
    p.add_argument(
        "--eval-on-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run an evaluation pass at step 0 before training begins, "
        "providing a pre-training anchor for the eval curve. "
        "``--no-eval-on-start`` skips it.",
    )
    # --- Optim -------------------------------------------------------------
    p.add_argument(
        "--noise-bias-correction",
        action="store_true",
        help="Enable DP-noise-aware Adam bias correction in the opaque AdamW "
        "variant. Only meaningful at noise_multiplier > 0; default off so the "
        "noise=0 path matches stock PyTorch AdamW semantics.",
    )
    # --- W&B ---------------------------------------------------------------
    p.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging; defaults to enabled when WANDB_PROJECT / "
        "WANDB_API_KEY env vars are set.",
    )
    args = p.parse_args()
    _require_configured(p, args)
    return args


def main() -> int:
    args = parse_args()

    loss_type = list(args.loss_type)
    reference_free = all(lt in _REFERENCE_FREE_HEADS for lt in loss_type)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        # Base models (e.g. JetBrains/Mellum-4b-base) ship no chat template, but
        # DPOTrainer.tokenize_row calls apply_chat_template on list-of-message
        # preference rows (ultrafeedback). Install a minimal ChatML so the run
        # validates DPO mechanics without depending on an instruct variant.
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        )

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model_dtype = torch.bfloat16 if bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=model_dtype)

    # PEFT policy: pass the LoRA config to the trainer (it calls get_peft_model),
    # so the frozen base can serve as the reference via the null-ref path.
    peft_config = (
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_modules,
            lora_dropout=0.0,
        )
        if args.peft
        else None
    )

    # Reference policy. Reference-free heads need none; a PEFT policy uses the
    # frozen base via the null-ref path (ref_model=None). For full fine-tuning,
    # --ref-model is loaded (DPOTrainer also accepts a path/repo-id string).
    if reference_free or args.peft:
        ref_model = None
    elif args.ref_model is not None:
        ref_model = args.ref_model  # DPOTrainer accepts a string and loads it
    else:
        # Auto-load a frozen copy from the policy's own path.
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=model_dtype
        )

    raw = load_dataset(args.dataset, split=args.dataset_split, streaming=True)
    eval_count = args.num_eval_samples if args.eval_steps else 0
    take_total = args.num_train_samples + eval_count
    all_rows = [row for _, row in zip(range(take_total), raw)]
    # Canonicalize column shape so non-TRL-canonical code-DPO datasets work
    # without a fork. CyberNative ships (system, question, chosen, rejected);
    # zed-industries/zeta ships (events, input, output, rejected); both
    # are remapped to (prompt, chosen, rejected) so the collator + tokenize_row
    # see a TRL-canonical row. ultrafeedback already has ``prompt``.
    if (
        all_rows
        and "prompt" not in all_rows[0]
        and ("question" in all_rows[0] or "input" in all_rows[0])
    ):
        all_rows = [_to_trl_canonical_dpo(row) for row in all_rows]
    train_dataset = Dataset.from_list(all_rows[: args.num_train_samples])
    eval_dataset = (
        Dataset.from_list(all_rows[args.num_train_samples : take_total])
        if eval_count
        else None
    )

    report_to = _configure_reporting(args.no_wandb)
    run_name = os.environ.get("WANDB_NAME") or os.environ.get("RUN_NAME")

    optim_args = "noise_bias_correction=True" if args.noise_bias_correction else None
    eval_kwargs: dict = {}
    if args.eval_steps:
        eval_kwargs = {
            "eval_strategy": "steps",
            "eval_steps": args.eval_steps,
            "per_device_eval_batch_size": (
                args.per_device_eval_batch_size or args.batch_size
            ),
            "eval_on_start": args.eval_on_start,
        }

    dpo_args = DPOConfig(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        # --- Loss ---
        loss_type=loss_type,
        loss_weights=args.loss_weights,
        beta=args.beta,
        label_smoothing=args.label_smoothing,
        f_divergence_type=args.f_divergence_type,
        f_alpha_divergence_coef=args.f_alpha_divergence_coef,
        ld_alpha=args.ld_alpha,
        use_weighting=args.use_weighting,
        simpo_gamma=args.simpo_gamma,
        cpo_alpha=args.cpo_alpha,
        orpo_lambda=args.orpo_lambda,
        # --- TR-DPO (reference sync) ---
        sync_ref_model=args.sync_ref_model,
        ref_model_mixup_alpha=args.ref_model_mixup_alpha,
        ref_model_sync_steps=args.ref_model_sync_steps,
        # --- Telemetry ---
        log_completion_metrics=args.log_completion_metrics,
        # --- Training ---
        max_length=args.max_length,
        per_device_train_batch_size=args.batch_size,
        microbatch_size=args.microbatch_size,
        precompute_ref_batch_size=args.precompute_ref_batch_size,
        max_steps=args.stop_at_step,
        learning_rate=args.learning_rate,
        logging_steps=args.log_steps,
        save_strategy="no",
        seed=args.seed,
        use_cpu=not torch.cuda.is_available(),
        bf16=bf16,
        model_init_kwargs={"torch_dtype": model_dtype},
        use_performance_kernels=not args.no_performance_kernels,
        gradient_checkpointing=args.gradient_checkpointing,
        activation_offloading=args.activation_offloading,
        auto_find_microbatch_size=args.auto_find_microbatch_size,
        # W&B
        report_to=report_to,
        run_name=run_name,
        # DP knobs
        clipping_norm=args.clipping_norm,
        privacy_noise_multiplier=args.noise_multiplier,
        # ε target drives calibration only when no fixed multiplier is given;
        # at a fixed nm (incl. the nm=0 non-private path) the config layer
        # rejects a dangling target.
        privacy_target_epsilon=(
            args.target_epsilon if args.noise_multiplier is None else None
        ),
        optim="adamw",
        optim_args=optim_args,
        **eval_kwargs,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    out = trainer.train()
    print("Training complete:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
