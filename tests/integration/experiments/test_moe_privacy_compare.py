"""Comparison refuses incomplete or non-matched privacy experiments."""

import copy
import math

import pytest
from examples.moe_privacy.compare import compare


@pytest.fixture
def summaries():
    return [
        {
            "schema_version": 1,
            "status": "completed",
            "arm": arm,
            "model_seed": seed,
            "config": {
                "steps": 256,
                "target_epsilon": 8.0,
                "delta": 1e-5,
                "load_noise_ratio": 0.02,
            },
            "data": {"kind": "synthetic"},
            "provenance": {
                "pr_revision": "pinned-test",
                "runner_sha256": "runner-test",
            },
            "parameters": 313152,
            "initial": {"eval_loss": 4.8},
            "final": {
                "eval_loss": 4.0,
                "eval_load_cv": 0.2,
                "eval_max_expert_share": 0.2,
                "eval_routing_entropy": 0.9,
            },
            "relative_ce_improvement": 1 / 6,
            "privacy": {
                "private": arm != "reference",
                "steps": 256,
                "epsilon": 7.9999 if arm != "reference" else None,
                "target_epsilon": 8 if arm != "reference" else None,
                "delta": 1e-5 if arm != "reference" else None,
                "load_release": arm == "dp_aux",
                "noise_multiplier": math.sqrt(1.02)
                if arm == "dp_aux"
                else 1.0
                if arm == "dp"
                else 0,
            },
        }
        for seed in [0, 1, 2]
        for arm in ["reference", "dp", "dp_aux"]
    ]


def test_matched_campaign_reports_variation_and_composition(summaries):
    result = compare(summaries)
    assert result["runs_per_arm"] == 3
    assert result["arms"]["dp"]["mean_learning_target_met"]
    assert result["arms"]["dp"]["final"]["eval_loss"]["sample_std"] == 0
    assert result["paired_differences"]["dp_aux_minus_dp"]["eval_loss"]["mean"] == 0
    assert not result["privacy"]["all_arms_private"]
    composition = result["privacy"]["basic_composition_of_private_arms_only"]
    assert composition["epsilon"] == pytest.approx(6 * 7.9999)
    assert composition["delta"] == pytest.approx(6e-5)


def test_single_seed_has_no_invented_standard_deviation(summaries):
    result = compare(summaries[:3])
    assert result["arms"]["dp"]["final"]["eval_loss"]["sample_std"] is None


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("missing", "each model seed"),
        ("duplicate", "duplicate"),
        ("epsilon", "target budget"),
        ("noise", "matched effective"),
        ("config", "configuration"),
        ("provenance", "provenance"),
        ("initial", "initialization"),
        ("nonfinite", "finite metrics"),
    ],
)
def test_comparison_fails_closed(summaries, fault, message):
    runs = copy.deepcopy(summaries)
    if fault == "missing":
        runs.pop()
    elif fault == "duplicate":
        runs.append(runs[0])
    elif fault == "epsilon":
        runs[1]["privacy"]["epsilon"] = 9
    elif fault == "noise":
        runs[2]["privacy"]["noise_multiplier"] = 1
    elif fault == "config":
        runs[1]["config"]["steps"] = 257
    elif fault == "provenance":
        runs[1]["provenance"]["runner_sha256"] = "changed-runner"
    elif fault == "initial":
        runs[1]["initial"]["eval_loss"] = 2
    else:
        runs[1]["final"]["eval_loss"] = float("nan")
    with pytest.raises(ValueError, match=message):
        compare(runs)
