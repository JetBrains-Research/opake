"""The gate must exercise actual Mellum routers, experts and Opaque releases."""

import pytest
from examples.moe_privacy.checks import run_checks
from examples.moe_privacy.run import ExperimentConfig


@pytest.mark.slow
def test_real_mellum_correctness_gate():
    config = ExperimentConfig(sequence_length=12)
    result = run_checks(config)
    assert result["passed"]
    assert result["execution"]["device"] == "cpu"
    assert result["execution"]["device_name"]
    assert result["execution"]["peak_cuda_memory_allocated_bytes"] is None
    assert len(result["router_parameters"]) == config.num_layers
    assert len(result["expert_parameters"]) == 2 * config.num_layers
    assert result["unused_expert_matrices_noised"] == 24
    assert result["empty_batch_releases_verified"]
    assert result["router_task_gradient_norm"] > 0
    assert result["router_aux_gradient_norm"] > 0
    assert result["expert_task_gradient_norm"] > 0
    assert result["neighbor_independence_max_abs_error"] < 1e-5
    assert result["global_clipping_max_abs_error"] < 1e-5


@pytest.mark.slow
def test_gate_rejects_missing_gradient_noise(monkeypatch):
    from opaque.dpsgd.noise import gaussian_noise

    monkeypatch.setattr(
        "examples.moe_privacy.checks.gaussian_noise",
        lambda *, noise_multiplier, key: gaussian_noise(noise_multiplier=0, key=key),
    )
    with pytest.raises(RuntimeError, match="gradient noise scale"):
        run_checks(ExperimentConfig(sequence_length=12))
