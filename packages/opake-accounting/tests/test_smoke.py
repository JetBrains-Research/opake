"""Sanity checks for ``opake.accounting`` (+ native build, ``opake-core``).

Algorithm-specific PLD factories live in ``opake.dpsgd.accounting`` /
``opake.dpftrl.accounting``; their tests ship with those packages instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


def test_invalid_generic_mechanism_parameter_raises_value_error():
    import opake.accounting as acc

    with pytest.raises(ValueError, match="epsilon must be non-negative"):
        acc.eps_delta(-1.0).pld()


def test_incompatible_pld_operands_raise_value_error():
    import opake.accounting as acc

    process = acc.eps_delta(1.0, 1e-5)
    fine = process.pld(discretization=0.1)
    incompatible = process.pld(discretization=0.3)

    with pytest.raises(ValueError, match="Discretization intervals differ"):
        fine.compose(incompatible)


@pytest.mark.parametrize("count", [0, -1])
@pytest.mark.parametrize(
    "compose",
    [
        lambda pld, count: pld.self_compose(count),
        lambda pld, count: pld * count,
        lambda pld, count: count * pld,
    ],
)
def test_invalid_pld_self_composition_count_raises_value_error(compose, count: int):
    import opake.accounting as acc

    pld = acc.eps_delta(1.0, 1e-5).pld()

    with pytest.raises(ValueError, match="count must be greater than zero"):
        compose(pld, count)


@pytest.mark.parametrize(
    "compose",
    [
        lambda pld, count: pld.self_compose(count),
        lambda pld, count: pld * count,
        lambda pld, count: count * pld,
    ],
)
def test_overflowing_pld_self_composition_count_raises_overflow_error(compose):
    import opake.accounting as acc

    pld = acc.eps_delta(1.0, 1e-5).pld()

    with pytest.raises(OverflowError, match="count must not exceed"):
        compose(pld, 2**32)


def test_discretization_config_stub_matches_native_surface():
    stub_path = (
        Path(__file__).parents[1] / "src/opake/api/accounting/core/opake_accounting.pyi"
    )
    module = ast.parse(stub_path.read_text())
    config = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "DiscretizationConfig"
    )
    members = {
        node.name
        for node in config.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    docstring = ast.get_docstring(config)

    assert {
        "tail_mass_truncation",
        "seed",
        "mc_resolution",
        "mc_failure_probability",
        "resolved_num_mc_samples",
    } <= members
    assert docstring is not None
    for argument in (
        "tail_mass_truncation",
        "seed",
        "mc_resolution",
        "mc_failure_probability",
    ):
        assert f"{argument}:" in docstring


def test_mc_sample_count_is_derived_without_manual_override():
    import opake.accounting as acc

    config = acc.get_discretization(
        mc_resolution=1e-2,
        mc_failure_probability=1e-2,
    )
    assert config.resolved_num_mc_samples == 1236

    with pytest.raises(TypeError, match="num_mc_samples"):
        acc.get_discretization(num_mc_samples=100)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("mc_resolution", 0.0),
        ("mc_resolution", 1.0),
        ("mc_failure_probability", 0.0),
        ("mc_failure_probability", 1.0),
    ],
)
def test_invalid_mc_confidence_parameters_raise(name, value):
    from opake.api.accounting.core import _native

    with pytest.raises(ValueError, match=name):
        _native.DiscretizationConfig(**{name: value})


def test_large_derived_mc_work_emits_advisory_warning(monkeypatch):
    from opake.api.accounting.core import discretization

    config = discretization.get_discretization(
        mc_resolution=1e-2,
        mc_failure_probability=1e-2,
    )
    monkeypatch.setattr(discretization, "_MC_SAMPLE_WARNING_THRESHOLD", 1000)

    with pytest.warns(RuntimeWarning, match="1,236 samples"):
        config.warn_if_large_mc()
