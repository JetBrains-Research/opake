"""Block and total k-out-of-t horizon accounting contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import opake.dpsgd.accounting as dpsgd_acc
from opake.accounting.types import DpHorizonProcess

if TYPE_CHECKING:
    from collections.abc import Callable

_DELTA = 1e-5


def test_factory_returns_horizon_process():
    process = dpsgd_acc.k_out_of_t(
        dpsgd_acc.gaussian(1.0),
        k=2,
        t=10,
        allocation="block",
    )

    assert isinstance(process, DpHorizonProcess)
    assert process.k == 2
    assert process.t == 10
    assert process.block_sizes == (5, 5)
    assert process.allocation == "block"


@pytest.mark.slow
def test_total_full_horizon_uses_the_block_upper_bound():
    block = dpsgd_acc.k_out_of_t(
        dpsgd_acc.gaussian(1.0),
        k=3,
        t=10,
        allocation="block",
    )
    total = dpsgd_acc.k_out_of_t(
        dpsgd_acc.gaussian(1.0),
        k=3,
        t=10,
        allocation="total",
    )

    assert total.epsilon_at(_DELTA) == pytest.approx(
        block.epsilon_at(_DELTA), rel=1e-12, abs=0
    )


@pytest.mark.parametrize(
    ("k", "t", "match"),
    [
        (0, 10, "k"),
        (2, 0, "t"),
    ],
)
def test_parameters_are_validated(k: int, t: int, match: str):
    with pytest.raises(ValueError, match=match):
        dpsgd_acc.k_out_of_t(
            dpsgd_acc.gaussian(1.0),
            k=k,
            t=t,
            allocation="block",
        )


class TestKOutOfTRegressionVectors:
    """Committed epsilon values for the supported deterministic inner mechanisms."""

    @pytest.mark.parametrize(
        ("name", "factory", "expected"),
        [
            pytest.param(
                "k_out_of_t(gaussian(1.0), k=2, t=16, block)",
                lambda: dpsgd_acc.k_out_of_t(
                    dpsgd_acc.gaussian(1.0),
                    k=2,
                    t=16,
                    allocation="block",
                ),
                4.687320185091083,
                id="gaussian",
                marks=pytest.mark.slow,
            ),
            pytest.param(
                "k_out_of_t(adaclip(gaussian(1.1)), k=2, t=16, block)",
                lambda: dpsgd_acc.k_out_of_t(
                    dpsgd_acc.adaclip(
                        dpsgd_acc.gaussian(1.1),
                        expected_batch_size=250,
                        num_groups=3,
                    ),
                    k=2,
                    t=16,
                    allocation="block",
                ),
                3.9650600884447935,
                id="adaclip",
                marks=pytest.mark.slow,
            ),
        ],
    )
    def test_epsilon_matches_committed_vector(
        self,
        name: str,
        factory: Callable[[], DpHorizonProcess],
        expected: float,
    ):
        delta = 1e-8
        actual = factory().epsilon_at(delta)

        assert actual == pytest.approx(expected, rel=1e-9, abs=3e-9), (
            f"{name}, delta={delta}: epsilon drifted; "
            f"committed={expected:.17g}, observed={actual:.17g}"
        )
