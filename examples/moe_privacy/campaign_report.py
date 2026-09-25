"""Read a fixed four-arm campaign; never launch experiments or select seeds."""

import argparse
import json
import math
import statistics
import tempfile
from pathlib import Path

ARMS = ("reference", "reference_aux", "dp", "dp_aux")
BENCHMARKS = {"humaneval": 164, "mbpp": 378}
CODE = tuple(f"{name}_{kind}_pass1" for name in BENCHMARKS for kind in ("plus", "base"))
TEST = (
    "test_nll",
    "test_token_accuracy",
    "test_load_cv",
    "test_max_mean_load",
    "test_effective_experts",
)
POOLED = (
    "test_pooled_load_cv",
    "test_pooled_max_mean_load",
    "test_pooled_effective_experts",
)
METRICS = (*CODE, *TEST, *POOLED, "training_hours", "peak_memory_gb", "epsilon")
HEADERS = (
    "HumanEval+ pass@1",
    "HumanEval pass@1",
    "MBPP+ pass@1",
    "MBPP pass@1",
    "Test NLL",
    "Token accuracy",
    "Mean per-layer load CV",
    "Max/mean load",
    "Effective experts",
    "Pooled-index load CV",
    "Pooled-index max/mean load",
    "Pooled-index effective experts",
    "Time h",
    "Peak CUDA GB",
    "ε",
)
T95 = (0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228)
MAX_SEEDS, MIN_PAIRS, EPSILON_TOLERANCE = 10, 3, 1e-3
MARGINS = {
    "test_load_cv_relative_pct": -10,
    "test_nll_relative_pct": 2,
    "humaneval_plus_pass1_pp": -2,
}
NOTES = [
    "Scope: expert LoRA + routers; matched per-record clipping in all four arms, including non-private controls.",
    "reference_aux uses lagged unnoised loads; dp_aux uses lagged noisy loads. This is not conventional batch-global Switch balancing.",
    "test_load_cv is the mean of per-layer expert-load CVs. test_pooled_load_cv computes CV after averaging loads across layers by expert index. Uniform pooled loads can hide fully collapsed individual layers; these metrics are not interchangeable.",
    "PR_MoE averages released layer-by-expert loads (noisy for dp_aux) across layers before f_tilde, so its objective targets pooled expert-index load. Pooled objective response is descriptive only, not evidence of practical per-layer benefit or speedup.",
    "Sequence-level DP applies per released fine-tuning run, not to the campaign or pretraining. The reference arms are non-private; no campaign privacy budget is claimed.",
    "Single-GPU routing balance is not a distributed speedup measurement. Prior pretraining contamination is unknown.",
    "Pass@1 and token accuracy are fractions (0–1); GB means 10^9 bytes. Missing values are unknown, never zero. Base means no adaptation.",
    "Optional pooled metrics have separate pending statuses and counts; their absence does not change completion of the original required metrics. Pooled response uses available paired CVs and labels improvement/worsening only with >=3 pairs and a 95% interval wholly below/above zero.",
    "Aggregates use available completed-training rows. Effects are treatment minus control, paired within seed; relative percentages divide by that seed's control (zero denominators are unknown).",
    "95% paired t-intervals are exploratory, unadjusted, small-n intervals across independent training seeds, not across benchmark tasks or 1024 test examples. One seed has no SD or CI; no p-values or equivalence claims.",
    "Predeclared practical balance criterion: >=10% mean per-layer CV reduction, <=2% relative test NLL worsening, and <=2 percentage-point HumanEval+ loss. Supported requires >=3 complete paired seeds and all three 95% intervals fully within these margins; otherwise inconclusive or observed trade-off.",
    "Criterion evidence uses only complete pairs; available-case effect tables may have more pairs. Planned seeds are never selected or ranked. Campaign state is operational only: no summary means no completed training evidence.",
]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _number(value, name):
    if value is not None:
        _require(
            type(value) in (int, float) and math.isfinite(value) and value >= 0,
            f"{name} must be finite and nonnegative",
        )
    return value


def _same(pins, name, value):
    _require(pins.setdefault(name, value) == value, f"{name} mismatch across campaign")


def _code(path, protocols):
    source, metrics = _read(path), dict.fromkeys(CODE)
    for name, count in BENCHMARKS.items():
        if name not in source:
            continue
        result = source[name]
        _require(isinstance(result, dict), f"invalid benchmark: {path}/{name}")
        _require(
            result.get("smoke") is False, f"smoke benchmark forbidden: {path}/{name}"
        )
        _require(
            type(result.get("task_count")) is int and result["task_count"] == count,
            f"task_count mismatch: {path}/{name}",
        )
        protocol = result.get("protocol_sha256")
        _require(
            isinstance(protocol, str) and bool(protocol),
            f"missing benchmark protocol: {path}/{name}",
        )
        _require(
            protocols.setdefault(name, protocol) == protocol,
            f"benchmark protocol mismatch: {path}/{name}",
        )
        for kind in ("plus", "base"):
            key = f"{kind}_pass1"
            value = _number(result.get(key), key)
            _require(value is None or value <= 1, f"{key} must lie in [0, 1]")
            metrics[f"{name}_{key}"] = value
    return metrics


def _validate(summary, manifest, seed, arm, pins):
    config, privacy = manifest["config"], summary.get("privacy", {})
    _require(
        summary.get("arm") == arm and summary.get("model_seed") == seed,
        "arm/model_seed mismatch",
    )
    _require(
        _canonical(summary.get("config")) == _canonical(config),
        "configuration differs from fixed manifest",
    )
    hashes = {
        name: summary.get("data", {}).get(name)
        for name in ("train_sha256", "validation_sha256", "test_sha256")
    }
    _require(
        all(isinstance(value, str) and value for value in hashes.values()),
        "missing data hashes",
    )
    _same(pins, "data", hashes)
    parameters = summary.get("trainability", {}).get("trainable_parameters")
    _require(type(parameters) is int and parameters > 0, "missing trainability count")
    _same(pins, "trainability", parameters)
    private = arm in ("dp", "dp_aux")
    _require(privacy.get("private") is private, "private flag disagrees with arm")
    rate = _number(privacy.get("sample_rate"), "sampling rate")
    _require(
        privacy.get("steps") == config["steps"]
        and rate is not None
        and math.isclose(
            rate,
            config["expected_batch_size"] / config["train_sequences"],
            rel_tol=0,
            abs_tol=1e-12,
        ),
        "sampling bounds differ from manifest",
    )
    noise = _number(privacy.get("noise_multiplier"), "noise multiplier")
    _require(
        noise is not None and (noise > 0 if private else noise == 0),
        "noise multiplier disagrees with arm",
    )
    if private:
        epsilon = _number(privacy.get("epsilon"), "epsilon")
        _require(
            epsilon is not None and epsilon <= config["target_epsilon"],
            "epsilon exceeds target or is missing",
        )
        _require(privacy.get("delta") == config["delta"], "delta differs from manifest")
        pins.setdefault("epsilons", []).append(epsilon)
        _require(
            max(pins["epsilons"]) - min(pins["epsilons"]) <= EPSILON_TOLERANCE + 1e-12,
            "actual epsilon mismatch exceeds 1e-3",
        )
    else:
        _require(privacy.get("epsilon") is None, "non-private arm cannot claim epsilon")


def _stats(values, *, paired=False):
    values = [value for value in values if value is not None]
    n = len(values)
    mean = statistics.fmean(values) if n else None
    sd = statistics.stdev(values) if n > 1 else None
    width = T95[n - 1] * sd / math.sqrt(n) if paired and n > 1 else None
    return {
        "n": n,
        "mean": mean,
        "sample_sd": sd,
        "ci95": [mean - width, mean + width] if width is not None else None,
    }


def _comparison(rows, seeds, control, treatment):
    indexed = {(row["seed"], row["arm"]): row for row in rows}
    effects, complete = [], []
    relative_metrics = ("test_load_cv", "test_pooled_load_cv", "test_nll")
    keys = {
        f"{metric}_{'relative_pct' if metric in relative_metrics else 'pp'}": metric
        for metric in (*relative_metrics, *CODE)
    }
    for seed in seeds:
        left, right = indexed[seed, control], indexed[seed, treatment]
        if any(row["training_status"] != "completed" for row in (left, right)):
            continue
        values = {"seed": seed}
        for key, metric in keys.items():
            a, b = left["metrics"][metric], right["metrics"][metric]
            values[key] = (
                None
                if a is None or b is None or (metric in relative_metrics and a == 0)
                else 100 * (b - a) / (a if metric in relative_metrics else 1)
            )
        effects.append(values)
        if left["status"] == right["status"] == "complete" and all(
            values[key] is not None for key in MARGINS
        ):
            complete.append(seed)
    result = {
        "complete_paired_seeds": complete,
        "pending_pairs": len(seeds) - len(complete),
        "paired_values": effects,
        "effects": {
            key: _stats([row[key] for row in effects], paired=True) for key in keys
        },
    }
    if treatment.endswith("_aux"):
        evidence = {
            key: _stats(
                [row[key] for row in effects if row["seed"] in complete], paired=True
            )
            for key in MARGINS
        }
        cv, nll, he = (evidence[key] for key in MARGINS)
        cv_limit, nll_limit, he_limit = MARGINS.values()
        supported = (
            len(complete) >= MIN_PAIRS
            and cv["ci95"][1] <= cv_limit
            and nll["ci95"][1] <= nll_limit
            and he["ci95"][0] >= he_limit
        )
        harmful = bool(complete) and (
            cv["mean"] > 0 or nll["mean"] > nll_limit or he["mean"] < he_limit
        )
        result.update(
            criterion_evidence=evidence,
            conclusion="supported"
            if supported
            else "observed trade-off"
            if harmful
            else "inconclusive",
        )
        pooled = result["effects"]["test_pooled_load_cv_relative_pct"]
        response = "pending" if not pooled["n"] else "inconclusive"
        if pooled["n"] >= MIN_PAIRS:
            if pooled["ci95"][1] < 0:
                response = "improved"
            elif pooled["ci95"][0] > 0:
                response = "worsened"
        result["pooled_objective_response"] = {
            "status": response,
            "pending_pairs": len(seeds) - pooled["n"],
            "evidence": pooled,
            "interpretation": (
                "Pooled objective improved but practical per-layer benefit not demonstrated."
                if response == "improved" and not supported
                else "Descriptive pooled-index balance only; no practical benefit or speedup claim."
            ),
        }
    return result


def build_report(root: Path) -> dict:
    """Validate and aggregate only the manifest's planned main runs and base.

    Args:
        root: Campaign directory containing campaign.json.
    """
    root = Path(root)
    manifest, state = _read(root / "campaign.json"), _read(root / "campaign_state.json")
    _require(
        manifest.get("protocol") == "matched-clipped-lagged-v2",
        "unsupported campaign protocol",
    )
    _require(
        sorted(manifest.get("arms", [])) == sorted(ARMS),
        "campaign must plan exactly four arms",
    )
    seeds, config = manifest.get("seeds", []), manifest.get("config", {})
    _require(
        isinstance(seeds, list)
        and 1 <= len(seeds) <= MAX_SEEDS
        and all(type(seed) is int and seed >= 0 for seed in seeds)
        and len(set(seeds)) == len(seeds),
        "plan 1–10 distinct integer seeds",
    )
    _require(
        all(
            isinstance(config.get(key), str) and config[key]
            for key in ("model_id", "model_revision")
        ),
        "configuration must pin base model and revision",
    )
    _require(
        all(
            type(config.get(key)) is int and config[key] > 0
            for key in (
                "steps",
                "train_sequences",
                "validation_sequences",
                "test_sequences",
                "expected_batch_size",
            )
        ),
        "invalid configuration sample bounds",
    )
    _require(
        config["expected_batch_size"] <= config["train_sequences"],
        "invalid configuration sampling rate",
    )
    _require(
        _number(config.get("target_epsilon"), "target epsilon") is not None
        and config["target_epsilon"] > 0,
        "invalid target epsilon",
    )
    _require(
        _number(config.get("delta"), "delta") is not None and 0 < config["delta"] < 1,
        "invalid delta",
    )
    rows, pins, protocols = [], {}, {}
    baseline_metrics = _code(root / "base/code_metrics.json", protocols)
    baseline = {
        "arm": "base (no adaptation)",
        "seed": None,
        "status": "not_adapted",
        "metrics": {**dict.fromkeys(METRICS), **baseline_metrics},
        "code_status": "completed"
        if all(value is not None for value in baseline_metrics.values())
        else "pending",
        "pooled_status": "not_applicable",
    }
    for seed in seeds:
        for arm in ARMS:
            folder = root / "main" / f"seed-{seed}" / arm
            summary = _read(folder / "summary.json")
            training = summary.get("status")
            training = training if training in ("completed", "failed") else "pending"
            stages = state.get("stages", {})
            stage = (
                stages.get(f"main/seed-{seed}/{arm}", {})
                if isinstance(stages, dict)
                else {}
            )
            if training == "pending" and (
                stage == "failed"
                or (isinstance(stage, dict) and stage.get("status") == "failed")
            ):
                training = "failed"
            if training == "completed":
                _validate(summary, manifest, seed, arm, pins)
            else:
                summary = {}
            metrics = {
                name: _number(summary.get("measurements", {}).get(name), name)
                for name in (*TEST, *POOLED)
            }
            accuracy = metrics["test_token_accuracy"]
            _require(
                accuracy is None or accuracy <= 1,
                "test_token_accuracy must lie in [0, 1]",
            )
            for name, value, scale in (
                ("training_hours", summary.get("training_seconds"), 3600),
                (
                    "peak_memory_gb",
                    summary.get("execution", {}).get(
                        "peak_cuda_memory_allocated_bytes"
                    ),
                    1e9,
                ),
            ):
                metrics[name] = (
                    _number(value, name) / scale if value is not None else None
                )
            metrics["epsilon"] = summary.get("privacy", {}).get("epsilon")
            code = _code(folder / "code_metrics.json", protocols)
            metrics.update(code)
            code_status = (
                "completed"
                if all(value is not None for value in code.values())
                else "pending"
            )
            missing = [
                name
                for name in (*TEST, "training_hours", "peak_memory_gb")
                if metrics[name] is None
            ]
            pooled_missing = [name for name in POOLED if metrics[name] is None]
            status = (
                training
                if training != "completed"
                else "metrics_pending"
                if missing
                else "code_pending"
                if code_status == "pending"
                else "complete"
            )
            rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "status": status,
                    "training_status": training,
                    "code_status": code_status,
                    "pooled_status": "pending" if pooled_missing else "completed",
                    "missing_metrics": missing,
                    "missing_pooled_metrics": pooled_missing,
                    "metrics": metrics,
                    "privacy": summary.get("privacy", {}),
                    "wandb_url": _read(folder / "wandb_run.json").get("url"),
                }
            )
    counts = {
        "planned": len(rows),
        "complete": sum(row["status"] == "complete" for row in rows),
    }
    counts.update(
        {
            f"training_{status}": sum(row["training_status"] == status for row in rows)
            for status in ("completed", "pending", "failed")
        }
    )
    counts["code_pending"] = sum(row["code_status"] == "pending" for row in rows)
    counts["metrics_pending"] = sum(
        row["training_status"] == "completed" and bool(row["missing_metrics"])
        for row in rows
    )
    counts["incomplete"] = len(rows) - counts["complete"]
    counts["pooled_metrics_pending"] = sum(
        row["pooled_status"] == "pending" for row in rows
    )
    return {
        "protocol": manifest["protocol"],
        "config": config,
        "seeds": seeds,
        "counts": counts,
        "rows": rows,
        "baseline": baseline,
        "data": pins.get("data"),
        "trainable_parameters": pins.get("trainability"),
        "benchmark_protocols": protocols,
        "campaign_state": state,
        "criterion": {"minimum_complete_pairs": MIN_PAIRS, "margins": MARGINS},
        "notes": NOTES,
        "aggregates": {
            arm: {
                metric: _stats(
                    [
                        row["metrics"][metric]
                        for row in rows
                        if row["arm"] == arm and row["training_status"] == "completed"
                    ]
                )
                for metric in METRICS
            }
            for arm in ARMS
        },
        "comparisons": {
            f"{treatment}_minus_{control}": _comparison(rows, seeds, control, treatment)
            for control, treatment in (
                ("reference", "reference_aux"),
                ("dp", "dp_aux"),
                ("reference", "dp"),
            )
        },
    }


def _fmt(value):
    if value is None:
        return "—"
    return (
        f"{value:.5g}"
        if isinstance(value, (int, float))
        else str(value).replace("|", "\\|").replace("\n", " ")
    )


def _table(headers, rows):
    return "\n".join(
        "| " + " | ".join(_fmt(value) for value in row) + " |"
        for row in (headers, ["---"] * len(headers), *rows)
    )


def _markdown(report):
    config = report["config"]
    lines = [
        "### Four-arm campaign",
        f"Model: {_fmt(config['model_id'])} @ {_fmt(config['model_revision'])}. Seeds: {report['seeds']}.",
        "; ".join(
            f"{key.replace('_', ' ')}: {value}"
            for key, value in report["counts"].items()
        ),
        f"Unadapted base code evaluation: {report['baseline']['code_status']}.",
    ]
    for name, comparison in report["comparisons"].items():
        if "conclusion" in comparison:
            lines.append(
                f"**{name}: {comparison['conclusion']}** — {len(comparison['complete_paired_seeds'])} complete pairs; {comparison['pending_pairs']} pending/incomplete pairs."
            )
            response = comparison["pooled_objective_response"]
            evidence = response["evidence"]
            interval = (
                " to ".join(map(_fmt, evidence["ci95"]))
                if evidence["ci95"] is not None
                else "unknown"
            )
            lines.append(
                f"Pooled objective response (descriptive only): **{response['status']}** — mean paired CV change {_fmt(evidence['mean'])}%; 95% paired t-interval {interval}; n={evidence['n']}; {response['pending_pairs']} pending/unavailable pairs. {response['interpretation']}"
            )
    lines.extend(
        [
            "### Per-seed results",
            _table(
                ("Seed", "Arm", "Status", "Code", "Pooled metrics", *HEADERS, "W&B"),
                [
                    [
                        row["seed"],
                        row["arm"],
                        row["status"],
                        row["code_status"],
                        row["pooled_status"],
                        *(row["metrics"][metric] for metric in METRICS),
                        f"[run]({row['wandb_url']})" if row.get("wandb_url") else None,
                    ]
                    for row in (report["baseline"], *report["rows"])
                ],
            ),
        ]
    )
    lines.extend(
        [
            "### Across-seed mean ± sample SD (n)",
            _table(
                ("Arm", *HEADERS),
                [
                    [
                        arm,
                        *(
                            f"{_fmt(stats['mean'])} ± {_fmt(stats['sample_sd'])} (n={stats['n']})"
                            for stats in metrics.values()
                        ),
                    ]
                    for arm, metrics in report["aggregates"].items()
                ],
            ),
        ]
    )
    for name, comparison in report["comparisons"].items():
        lines.append(f"### {name}")
        for title, values in (
            ("Available matched seeds", comparison["effects"]),
            (
                "Practical criterion: complete pairs only",
                comparison.get("criterion_evidence", {}),
            ),
        ):
            if values:
                lines.extend(
                    [
                        title,
                        _table(
                            (
                                "Effect",
                                "n",
                                "Mean",
                                "Sample SD",
                                "95% paired t-interval",
                            ),
                            [
                                [
                                    key,
                                    stats["n"],
                                    stats["mean"],
                                    stats["sample_sd"],
                                    " to ".join(map(_fmt, stats["ci95"]))
                                    if stats["ci95"] is not None
                                    else "unknown",
                                ]
                                for key, stats in values.items()
                            ],
                        ),
                    ]
                )
    lines.extend(
        ["### Scope and interpretation", *(f"- {note}" for note in report["notes"])]
    )
    return "\n\n".join(lines) + "\n"


def write_report(root: Path) -> dict:
    """Build a report and atomically replace report.json and report.md individually.

    Args:
        root: Campaign directory containing campaign.json; no inputs are modified.
    """
    root = Path(root)
    report = build_report(root)
    outputs = {
        "report.json": json.dumps(report, indent=2, allow_nan=False) + "\n",
        "report.md": _markdown(report),
    }
    for name, text in outputs.items():
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=root, prefix=f".{name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            try:
                handle.write(text)
                handle.flush()
                temporary.replace(root / name)
            finally:
                temporary.unlink(missing_ok=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    write_report(parser.parse_args().root)
