"""Compare complete, matched three-arm campaigns without importing training code."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from examples.moe_privacy import ARMS, LEARNING_TARGET


def _statistics(values: list[float]) -> dict:
    if not values or not all(math.isfinite(value) for value in values):
        message = "comparison requires nonempty, finite metrics"
        raise ValueError(message)
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
        "values": values,
    }


def compare(summaries: list[dict]) -> dict:
    """Validate matched arms and report descriptive utility/routing statistics."""
    if not summaries:
        message = "no run summaries supplied"
        raise ValueError(message)
    config = summaries[0]["config"]
    data = summaries[0]["data"]
    provenance = summaries[0]["provenance"]
    runs: dict[str, dict[int, dict]] = {arm: {} for arm in ARMS}
    for run in summaries:
        if run["status"] != "completed" or run["schema_version"] != 1:
            message = "all runs must be completed, schema-version-1 results"
            raise ValueError(message)
        if run["config"] != config or run["data"] != data:
            message = "configuration or public dataset mismatch"
            raise ValueError(message)
        if run["provenance"] != provenance:
            message = "code or dependency provenance mismatch"
            raise ValueError(message)
        if run["arm"] not in ARMS:
            message = f"unknown arm: {run['arm']}"
            raise ValueError(message)
        arm, seed = run["arm"], run["model_seed"]
        if seed in runs[arm]:
            message = f"duplicate {arm} seed {seed}"
            raise ValueError(message)
        if run["privacy"]["steps"] != config["steps"]:
            message = "incomplete training horizon"
            raise ValueError(message)
        if arm == "reference":
            if run["privacy"]["private"] or run["privacy"]["epsilon"] is not None:
                message = "the reference arm must be labeled non-private"
                raise ValueError(message)
        else:
            privacy = run["privacy"]
            epsilon = privacy["epsilon"]
            if (
                not privacy["private"]
                or not math.isfinite(epsilon)
                or not 0 < epsilon <= config["target_epsilon"]
            ):
                message = "private run exceeds its target budget"
                raise ValueError(message)
            if (
                privacy["delta"] != config["delta"]
                or privacy["target_epsilon"] != config["target_epsilon"]
            ):
                message = "privacy target mismatch"
                raise ValueError(message)
            if privacy["load_release"] != (arm == "dp_aux"):
                message = "load-release accounting mismatch"
                raise ValueError(message)
        runs[arm][seed] = run
    seeds = sorted(runs["reference"])
    if not seeds or any(set(runs[arm]) != set(seeds) for arm in ARMS):
        message = "each model seed needs reference, dp and dp_aux results"
        raise ValueError(message)
    for seed in seeds:
        plain, joint, reference = (
            runs[arm][seed] for arm in ("dp", "dp_aux", "reference")
        )
        for run in (plain, joint):
            if run["parameters"] != reference["parameters"] or not math.isclose(
                run["initial"]["eval_loss"],
                reference["initial"]["eval_loss"],
                rel_tol=1e-6,
            ):
                message = "model initialization mismatch"
                raise ValueError(message)
        ratio = (
            joint["privacy"]["noise_multiplier"] / plain["privacy"]["noise_multiplier"]
        )
        if not math.isclose(
            ratio, math.sqrt(1 + config["load_noise_ratio"]), rel_tol=5e-4
        ):
            message = "arms do not have matched effective gradient-plus-load noise"
            raise ValueError(message)
        if not math.isclose(
            plain["privacy"]["epsilon"], joint["privacy"]["epsilon"], rel_tol=1e-3
        ):
            message = "private arms do not have matched achieved epsilon"
            raise ValueError(message)
    metrics = (
        "eval_loss",
        "eval_load_cv",
        "eval_max_expert_share",
        "eval_routing_entropy",
    )
    by_arm = {}
    for arm in ARMS:
        group = [runs[arm][seed] for seed in seeds]
        improvement = _statistics([run["relative_ce_improvement"] for run in group])
        by_arm[arm] = {
            "final": {
                metric: _statistics([run["final"][metric] for run in group])
                for metric in metrics
            },
            "relative_ce_improvement": improvement,
            "mean_learning_target_met": improvement["mean"] >= LEARNING_TARGET,
            "epsilon": [run["privacy"]["epsilon"] for run in group],
        }
    paired = {}
    for comparison, left, right in (
        ("dp_minus_reference", "dp", "reference"),
        ("dp_aux_minus_dp", "dp_aux", "dp"),
    ):
        paired[comparison] = {
            metric: _statistics(
                [
                    runs[left][seed]["final"][metric]
                    - runs[right][seed]["final"][metric]
                    for seed in seeds
                ]
            )
            for metric in metrics
        }
    private_runs = [run for run in summaries if run["arm"] != "reference"]
    return {
        "model_seeds": seeds,
        "runs_per_arm": len(seeds),
        "arms": by_arm,
        "paired_differences": paired,
        "balancing_cv_improved_on_average": paired["dp_aux_minus_dp"]["eval_load_cv"][
            "mean"
        ]
        < 0,
        "privacy": {
            "reported_budgets_are_per_run": True,
            "all_arms_private": False,
            "basic_composition_of_private_arms_only": {
                "epsilon": sum(run["privacy"]["epsilon"] for run in private_runs),
                "delta": sum(run["privacy"]["delta"] for run in private_runs),
            },
        },
        "scope": "synthetic/public data only; descriptive pilot statistics, not significance tests or a DP proof",
    }


def main() -> None:
    """Write one comparison report from completed run summaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "summaries",
        nargs="+",
        type=Path,
        help="summary.json files or their run directories",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        message = f"refusing to overwrite {args.output}"
        raise FileExistsError(message)
    summaries = [
        json.loads((path / "summary.json" if path.is_dir() else path).read_text())
        for path in args.summaries
    ]
    result = compare(summaries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps({"output": str(args.output), "runs_per_arm": result["runs_per_arm"]})
    )


if __name__ == "__main__":
    main()
