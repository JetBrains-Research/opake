"""DP Supervised Fine-Tuning via the class-based ``SFTTrainer``.

Hands ``opaque.alignment`` primitives to
:class:`opaque.transformers.trl.SFTTrainer`, which orchestrates the per-example
DP path on top of ``Trainer``.

Exposes the loss paths (``nll`` / ``dft`` / the fused logits-free
``chunked_nll``), a custom ``compute_loss_func`` (``--compute-loss-func``, valid
only on ``nll``), the PEFT added-token path (``--chat-template-path`` clones a
chat template and its special tokens, then the LoRA config keeps those new
embedding rows trainable), assistant-only masking on chat data
(``--assistant-only-loss``), a meaningful ``--eos-token``, and the
completion-metric telemetry opt-in (``--log-completion-metrics``).

Examples::

    # Plain LoRA SFT on raw text
    uv run python examples/train_sft_trainer.py \\
      --model-name HuggingFaceTB/SmolLM2-135M \\
      --dataset JetBrains/KExercises --dataset-text-field solution \\
      --num-train-samples 2000 --loss-type nll \\
      --max-length 512 --batch-size 8 --num-steps 50 \\
      --learning-rate 1e-4 --clipping-norm 1.0 --noise-multiplier 0.8 \\
      --lora-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \\
      --seed 42

    # Fused logits-free loss (CUDA fast path, eager fallback elsewhere)
    uv run python examples/train_sft_trainer.py --loss-type chunked_nll

    # Custom per-example loss (nll path only)
    uv run python examples/train_sft_trainer.py --loss-type nll --compute-loss-func

    # PEFT added-token path: clone a chat template + special tokens, train the
    # new embedding rows alongside the LoRA adapter, mask to assistant turns.
    uv run python examples/train_sft_trainer.py \\
      --chat-template-path HuggingFaceTB/SmolLM2-135M-Instruct --assistant-only-loss
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from opaque.transformers.trl import SFTConfig, SFTTrainer


def _configure_reporting(no_wandb: bool) -> list[str]:
    """Set W&B env defaults and return TrainingArguments.report_to.

    Enables ``"wandb"`` reporting when ``WANDB_NAME`` / ``WANDB_PROJECT`` /
    ``WANDB_ENTITY`` / ``WANDB_TAGS`` are configured, since the SFTConfig
    default of ``[]`` would otherwise keep runs stdout-only.
    """
    if no_wandb:
        return []
    if not os.environ.get("WANDB_MODE"):
        os.environ["WANDB_MODE"] = (
            "online" if os.environ.get("WANDB_API_KEY") else "offline"
        )
    return ["wandb"]


def _require_configured(parser, args, required=("model_name", "dataset")):
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        flags = ", ".join("--" + name.replace("_", "-") for name in missing)
        parser.error(f"missing required configuration: {flags}. Pass them directly.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DP SFT with the class-based SFTTrainer")
    p.add_argument(
        "--model-name",
        default=None,
        help="HuggingFace model name or local path (required)",
    )
    p.add_argument(
        "--dataset", default=None, help="HuggingFace dataset name (required)"
    )
    p.add_argument("--dataset-config", default=None)
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--dataset-text-field", default="text")
    p.add_argument("--num-train-samples", type=int, default=256)
    # --- Loss --------------------------------------------------------------
    p.add_argument(
        "--loss-type",
        default="nll",
        choices=["nll", "dft", "chunked_nll"],
        help="nll = standard CE; dft = Dynamic Fine-Tuning; chunked_nll = fused "
        "logits-free linear-CE (CUDA fast path, eager fallback).",
    )
    p.add_argument(
        "--compute-loss-func",
        action="store_true",
        help="Wire a custom per-example loss (label-smoothed CE). Valid only on "
        "--loss-type nll (dft / chunked_nll compute their own loss).",
    )
    p.add_argument("--completion-only", action="store_true")
    p.add_argument(
        "--assistant-only-loss",
        action="store_true",
        help="Mask the loss to assistant turns on conversational data "
        "(uses the generation-marked training chat template).",
    )
    p.add_argument(
        "--eos-token",
        default=None,
        help="EOS token appended to plain-text examples (overrides the "
        "tokenizer's eos_token); None uses the tokenizer's own.",
    )
    # --- Telemetry ---------------------------------------------------------
    p.add_argument(
        "--log-completion-metrics",
        dest="log_completion_metrics",
        action="store_true",
        help="Log entropy / mean_token_accuracy / logits/* per example. "
        "Forces the eager logits-materialising loss path (~12x microbatch hit "
        "on big-vocab models) — train/loss is still logged unconditionally "
        "on the fused path.",
    )
    p.set_defaults(log_completion_metrics=False)
    # --- Training ----------------------------------------------------------
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--microbatch-size", type=int, default=None)
    p.add_argument(
        "--stop-at-step",
        type=int,
        default=8,
        help="Stop the training loop after this many optimizer steps "
        "(early-stop knob, not a privacy-accounting target — privacy is "
        "calibrated from target_epsilon × steps × sample_rate regardless).",
    )
    p.add_argument("--learning-rate", type=float, default=1e-5)
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
            "ε at the configured ``(num_steps, batch_size, num_train_samples)``."
        ),
    )
    p.add_argument("--log-steps", type=int, default=1)
    p.add_argument("--output-dir", default="trainer_output/sft")
    # --- PEFT --------------------------------------------------------------
    p.add_argument(
        "--chat-template-path",
        default=None,
        help="Tokenizer dir / Jinja file whose chat template + special tokens "
        "are cloned onto the tokenizer before training. New tokens' embedding "
        "rows are kept trainable alongside the LoRA adapter.",
    )
    p.add_argument("--lora-modules", nargs="+", default=["q_proj", "v_proj"])
    p.add_argument("--lora-r", type=int, default=4)
    p.add_argument("--lora-alpha", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-performance-kernels",
        action="store_true",
        help="Disable Opaque's fused performance kernels (RMSNorm, LoRA QKV, "
        "fused linear+CE). Needed for models like Mellum-2.0 whose q_norm/k_norm "
        "shapes don't fit the current Triton row/block path.",
    )
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
    # --- Eval --------------------------------------------------------------
    p.add_argument(
        "--eval-steps",
        type=int,
        default=10,
        help="Evaluate every N steps on a held-out slice (disjoint slice of "
        "the same dataset, after the --num-train-samples). Set to ``0`` to "
        "disable eval entirely. Default matches train_sft.py / "
        "train_dpsgd_trainer.py.",
    )
    p.add_argument(
        "--num-eval-samples",
        type=int,
        default=64,
        help="Held-out sample count for eval. Matches train_sft.py / "
        "train_dpsgd_trainer.py. At 100 the per-eval loss is too noisy "
        "to read learning dynamics; 1000 keeps the per-eval std-err around 3%%.",
    )
    p.add_argument(
        "--eval-on-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run an evaluation pass at step 0 before training begins, "
        "providing a pre-training anchor for the eval curve. "
        "``--no-eval-on-start`` skips it.",
    )
    p.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=None,
        help="Eval batch size; defaults to --batch-size.",
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


def label_smoothed_ce(outputs, labels: torch.Tensor) -> torch.Tensor:
    """Custom per-example loss: label-smoothed next-token cross-entropy.

    Runs inside the trainer's per-example ``vmap`` path, so it sees a single
    example's ``outputs.logits`` ``(T, V)`` and ``labels`` ``(T,)`` and must
    return a scalar. Standard causal-LM shift (predict token ``t+1`` from ``t``)
    with ``-100`` ignored positions. Only valid on ``--loss-type nll``.
    """
    logits = outputs.logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    mask = shift_labels != -100
    safe_labels = shift_labels.clamp(min=0)
    logp = F.log_softmax(logits, dim=-1)
    nll = -logp.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    smooth = -logp.mean(dim=-1)
    loss = 0.9 * nll + 0.1 * smooth
    masked = loss * mask
    return masked.sum() / mask.sum().clamp(min=1)


def main() -> int:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model_dtype = torch.bfloat16 if bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=model_dtype
    )

    # When a chat template is cloned in, the trainer marks the added tokens'
    # embedding rows trainable on this config (trainable_token_indices + lm_head)
    # so the new special tokens are learned under DP.
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_modules,
        lora_dropout=0.0,
    )

    raw = load_dataset(
        args.dataset, args.dataset_config, split=args.dataset_split, streaming=True
    )
    eval_count = args.num_eval_samples if args.eval_steps else 0
    take_total = args.num_train_samples + eval_count
    all_rows = [row for _, row in zip(range(take_total), raw)]
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

    sft_args = SFTConfig(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        dataset_text_field=args.dataset_text_field,
        completion_only_loss=True if args.completion_only else None,
        assistant_only_loss=args.assistant_only_loss,
        chat_template_path=args.chat_template_path,
        eos_token=args.eos_token,
        loss_type=args.loss_type,
        log_completion_metrics=args.log_completion_metrics,
        max_length=args.max_length,
        per_device_train_batch_size=args.batch_size,
        microbatch_size=args.microbatch_size,
        max_steps=args.stop_at_step,
        learning_rate=args.learning_rate,
        logging_steps=args.log_steps,
        save_strategy="no",
        seed=args.seed,
        use_cpu=not torch.cuda.is_available(),
        bf16=bf16,
        # W&B
        report_to=report_to,
        run_name=run_name,
        # DP knobs
        clipping_norm=args.clipping_norm,
        privacy_noise_multiplier=args.noise_multiplier,
        # ε target drives calibration only when no fixed multiplier is given;
        # at a fixed nm (incl. nm=0) the config layer rejects a dangling target.
        privacy_target_epsilon=(
            args.target_epsilon if args.noise_multiplier is None else None
        ),
        optim="adamw",
        optim_args=optim_args,
        use_performance_kernels=not args.no_performance_kernels,
        gradient_checkpointing=args.gradient_checkpointing,
        activation_offloading=args.activation_offloading,
        auto_find_microbatch_size=args.auto_find_microbatch_size,
        **eval_kwargs,
    )

    # A custom compute_loss_func is only meaningful on the standard ``nll`` path
    # (dft / chunked_nll compute their own loss); the trainer guards this.
    compute_loss_func = label_smoothed_ce if args.compute_loss_func else None

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_loss_func=compute_loss_func,
        peft_config=peft_config,
    )
    out = trainer.train()
    print("Training complete:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
