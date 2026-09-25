"""Public-only gradient and load-noise diagnostic before a confirmation campaign."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch
from examples.moe_privacy.run import execution_metadata, write_json
from examples.moe_privacy.sft_data import CompletionCollator, prepare_partitions
from examples.moe_privacy.sft_metrics import evaluate_public, probe_balancing_signal
from examples.moe_privacy.sft_run import SFTExperimentConfig, load_model
from examples.moe_privacy.tracking import (
    ExperimentTracker,
    add_tracking_arguments,
    options_from_args,
)
from transformers import AutoTokenizer

from opaque.accounting import calibrate, epsilon_budget
from opaque.dpsgd.accounting import gaussian, poisson


def run_diagnostic(config, output_dir, *, tracking=None, records=8):
    """Use only the public diagnostic partition; no optimizer updates or test metrics."""
    if type(records) is not int or records <= 0:
        message = "records must be a positive integer"
        raise ValueError(message)
    torch.set_num_threads(4)
    output_dir.mkdir(parents=True, exist_ok=False)
    with ExperimentTracker(
        output_dir,
        config=asdict(config),
        arm="reference_aux",
        seed=0,
        experiment="public_balancing_diagnostic",
        options=tracking,
    ) as tracker:
        tracker.start()
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id, revision=config.model_revision
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        _, _, _, diagnostic, data = prepare_partitions(config, tokenizer)
        if len(diagnostic) < records:
            message = (
                "the public diagnostic partition is smaller than the requested probe"
            )
            raise ValueError(message)
        write_json(output_dir / "data.json", data)
        effective_noise = calibrate(
            epsilon_budget(config.target_epsilon, delta=config.delta),
            lambda nm: (
                poisson(
                    gaussian(nm), config.expected_batch_size / config.train_sequences
                )
                * config.steps
            ),
            0.11,
            10.0,
            tolerance=1e-4,
        ).param
        print(
            json.dumps(
                {
                    "event": "diagnostic_data_ready",
                    "records": len(diagnostic),
                    "effective_noise": effective_noise,
                }
            ),
            flush=True,
        )
        model, trainability = load_model(config, 0)
        torch.cuda.reset_peak_memory_stats()
        collator = CompletionCollator(config.sequence_length, tokenizer.pad_token_id)
        public_metrics = evaluate_public(
            model, diagnostic, collator, batch_size=config.eval_batch_size
        )
        tracker.log_metrics({"step": 0, **public_metrics})
        probe = probe_balancing_signal(
            model,
            diagnostic,
            collator,
            coefficient=config.router_aux_loss_coef,
            mean_tokens=config.sequence_length,
            max_tokens=config.sequence_length,
            expected_batch_size=config.expected_batch_size,
            noise_multiplier=effective_noise * math.sqrt(1 + config.load_noise_ratio),
            ratio=config.load_noise_ratio,
            filter_beta=config.filter_beta,
            steps=config.steps,
            records=records,
        )
        write_json(output_dir / "probe.json", probe)
        if tracker.run is not None:
            tracker.run.summary.update(
                {
                    f"diagnostic/{key}": value
                    for key, value in probe.items()
                    if isinstance(value, (float, int))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                }
            )
        summary = {
            "status": "completed",
            "config": asdict(config),
            "scope": "public gradient diagnostic, not a training or utility result",
            "trainability": trainability,
            "execution": execution_metadata(
                next(model.parameters()).device,
                peak_cuda_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
            ),
            "effective_noise_for_confirmation": effective_noise,
            "privacy": {"private": False, "epsilon": None, "steps": 0},
            "initial": {"step": 0, **public_metrics},
            "final": {"step": 0, **public_metrics},
            "probe": probe,
        }
        write_json(output_dir / "summary.json", summary)
        tracker.complete(summary)
        print(
            json.dumps(
                {
                    "event": "diagnostic_completed",
                    "router_aux_to_task_ratio": probe[
                        "router_aux_to_task_grad_ratio_at_coefficient"
                    ],
                    "ema_noise_to_signal": probe["predicted_ema_noise_to_signal_ratio"],
                }
            ),
            flush=True,
        )
        return summary


def main():
    """Run a bounded public-only diagnostic on the pretrained GPU model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--records", type=int, default=8)
    add_tracking_arguments(parser)
    args = parser.parse_args()
    config = SFTExperimentConfig(**json.loads(args.config.read_text()))
    run_diagnostic(
        config, args.output_dir, tracking=options_from_args(args), records=args.records
    )


if __name__ == "__main__":
    main()
