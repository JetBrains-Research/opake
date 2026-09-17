"""Tests for opake.api.transformers.trainer._scheduler.

Verifies that ``build_lr_schedule`` dispatches HF's ``lr_scheduler``
strings to a schedule that reproduces HF's reference behavior pointwise,
and that deferred / unknown types and bad kwargs raise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
import torch

torchopt = pytest.importorskip("torchopt", reason="torchopt required")
transformers = pytest.importorskip(
    "transformers", reason="transformers required for HF cross-check"
)

from torch.optim import SGD
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
    get_inverse_sqrt_schedule,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_wsd_schedule,
)

from opake.api.transformers.trainer._scheduler import (
    build_lr_schedule,
    get_warmup_steps,
)
from opake.api.transformers.trainer._training_arguments import (
    _parse_dict_string,
)

BASE_LR = 1e-3


@dataclass
class _Args:
    """Minimal stand-in for TrainingArguments — only the fields the shim reads."""

    lr_scheduler: str = "linear"
    learning_rate: float = BASE_LR
    warmup_steps: int | float = 0
    lr_scheduler_kwargs: dict[str, Any] = field(default_factory=dict)
    metric_for_best_model: str | None = None


def _hf_lambda(make_scheduler):
    p = torch.zeros(1, requires_grad=True)
    return make_scheduler(SGD([p], lr=BASE_LR)).lr_lambdas[0]


def _assert_pointwise(ours, hf_lam, num_total, *, beyond=100, tol=1e-9):
    step_size = max(1, (num_total + beyond) // 200)
    for s in range(0, num_total + beyond, step_size):
        assert ours(s) == pytest.approx(hf_lam(s) * BASE_LR, abs=tol), (
            f"step={s}: ours={ours(s)} hf={hf_lam(s) * BASE_LR}"
        )


# ---------------------------------------------------------------------------
# get_warmup_steps
# ---------------------------------------------------------------------------


class TestGetWarmupSteps:
    def test_absolute_step_count(self):
        assert get_warmup_steps(1000, 50) == 50

    def test_fraction_of_total(self):
        assert get_warmup_steps(1000, 0.1) == 100

    def test_one_is_a_step_count_not_a_fraction(self):
        assert get_warmup_steps(1000, 1) == 1

    def test_fractional_step_count_truncates(self):
        assert get_warmup_steps(1000, 50.7) == 50

    def test_zero_is_no_warmup(self):
        assert get_warmup_steps(1000, 0) == 0

    def test_fraction_uses_ceil(self):
        # 1000 * 0.0501 = 50.1 -> ceil = 51
        assert get_warmup_steps(1000, 0.0501) == 51


# ---------------------------------------------------------------------------
# Per-type pointwise parity with HF
# ---------------------------------------------------------------------------


class TestPointwiseParity:
    def test_constant(self):
        ours = build_lr_schedule(_Args(lr_scheduler="constant"), num_training_steps=500)
        hf = _hf_lambda(get_constant_schedule)
        _assert_pointwise(ours, hf, 500)

    def test_constant_with_warmup(self):
        ours = build_lr_schedule(
            _Args(lr_scheduler="constant_with_warmup", warmup_steps=50),
            num_training_steps=500,
        )
        hf = _hf_lambda(
            lambda o: get_constant_schedule_with_warmup(o, num_warmup_steps=50)
        )
        _assert_pointwise(ours, hf, 500)

    def test_linear(self):
        ours = build_lr_schedule(
            _Args(lr_scheduler="linear", warmup_steps=100),
            num_training_steps=1000,
        )
        hf = _hf_lambda(lambda o: get_linear_schedule_with_warmup(o, 100, 1000))
        _assert_pointwise(ours, hf, 1000)

    def test_cosine_default_cycles(self):
        ours = build_lr_schedule(
            _Args(lr_scheduler="cosine", warmup_steps=100),
            num_training_steps=1000,
        )
        hf = _hf_lambda(lambda o: get_cosine_schedule_with_warmup(o, 100, 1000))
        _assert_pointwise(ours, hf, 1000)

    def test_cosine_custom_cycles(self):
        ours = build_lr_schedule(
            _Args(
                lr_scheduler="cosine",
                warmup_steps=100,
                lr_scheduler_kwargs={"num_cycles": 1.5},
            ),
            num_training_steps=1000,
        )
        hf = _hf_lambda(
            lambda o: get_cosine_schedule_with_warmup(o, 100, 1000, num_cycles=1.5)
        )
        _assert_pointwise(ours, hf, 1000)

    def test_polynomial_defaults(self):
        ours = build_lr_schedule(
            _Args(lr_scheduler="polynomial", warmup_steps=100),
            num_training_steps=1000,
        )
        hf = _hf_lambda(
            lambda o: get_polynomial_decay_schedule_with_warmup(o, 100, 1000)
        )
        _assert_pointwise(ours, hf, 1000)

    def test_polynomial_custom_power(self):
        ours = build_lr_schedule(
            _Args(
                lr_scheduler="polynomial",
                warmup_steps=100,
                lr_scheduler_kwargs={"power": 2.0, "lr_end": 1e-9},
            ),
            num_training_steps=1000,
        )
        hf = _hf_lambda(
            lambda o: get_polynomial_decay_schedule_with_warmup(
                o, 100, 1000, lr_end=1e-9, power=2.0
            )
        )
        _assert_pointwise(ours, hf, 1000)

    def test_inverse_sqrt(self):
        ours = build_lr_schedule(
            _Args(lr_scheduler="inverse_sqrt", warmup_steps=100),
            num_training_steps=1000,
        )
        hf = _hf_lambda(lambda o: get_inverse_sqrt_schedule(o, num_warmup_steps=100))
        _assert_pointwise(ours, hf, 1000)

    @pytest.mark.parametrize("k", [1, 2, 4])
    def test_cosine_with_restarts(self, k):
        ours = build_lr_schedule(
            _Args(
                lr_scheduler="cosine_with_restarts",
                warmup_steps=100,
                lr_scheduler_kwargs={"num_cycles": k},
            ),
            num_training_steps=1000,
        )
        hf = _hf_lambda(
            lambda o: get_cosine_with_hard_restarts_schedule_with_warmup(
                o, 100, 1000, num_cycles=k
            )
        )
        _assert_pointwise(ours, hf, 1000)

    @pytest.mark.parametrize(
        ("warmup", "total", "k"),
        [
            (101, 1000, 4),  # decay=899, 899 % 4 == 3
            (50, 997, 3),  # decay=947, 947 % 3 == 2
            (0, 1000, 7),  # decay=1000, 1000 % 7 == 6
        ],
    )
    def test_cosine_with_restarts_non_divisible(self, warmup, total, k):
        """Restart window need not be divisible by num_cycles — used to crash
        in ``with_restarts``; now matches HF's fractional-progress formula."""
        ours = build_lr_schedule(
            _Args(
                lr_scheduler="cosine_with_restarts",
                warmup_steps=warmup,
                lr_scheduler_kwargs={"num_cycles": k},
            ),
            num_training_steps=total,
        )
        hf = _hf_lambda(
            lambda o: get_cosine_with_hard_restarts_schedule_with_warmup(
                o, warmup, total, num_cycles=k
            )
        )
        _assert_pointwise(ours, hf, total)

    def test_cosine_with_min_lr_absolute(self):
        ours = build_lr_schedule(
            _Args(
                lr_scheduler="cosine_with_min_lr",
                warmup_steps=100,
                lr_scheduler_kwargs={"min_lr": 1e-4},
            ),
            num_training_steps=1000,
        )
        hf = _hf_lambda(
            lambda o: get_cosine_with_min_lr_schedule_with_warmup(
                o, 100, 1000, min_lr=1e-4
            )
        )
        _assert_pointwise(ours, hf, 1000)

    def test_cosine_with_min_lr_rate(self):
        ours = build_lr_schedule(
            _Args(
                lr_scheduler="cosine_with_min_lr",
                warmup_steps=100,
                lr_scheduler_kwargs={"min_lr_rate": 0.1},
            ),
            num_training_steps=1000,
        )
        hf = _hf_lambda(
            lambda o: get_cosine_with_min_lr_schedule_with_warmup(
                o, 100, 1000, min_lr_rate=0.1
            )
        )
        _assert_pointwise(ours, hf, 1000)

    # cosine_warmup_with_min_lr deviates from HF's lambda by a one-step
    # phase shift in the indexing (HF uses (step + 1) instead of step).
    # We test the contract directly rather than pointwise HF parity.

    def test_cosine_warmup_with_min_lr_default_warmup_floor(self):
        # warmup_lr_rate omitted → warmup ramps from 0 to base_lr.
        base_lr = 1e-3
        sched = build_lr_schedule(
            _Args(
                lr_scheduler="cosine_warmup_with_min_lr",
                warmup_steps=100,
                lr_scheduler_kwargs={"min_lr_rate": 0.1},
            ),
            num_training_steps=1000,
        )
        assert sched(0) == pytest.approx(0.0)
        assert sched(100) == pytest.approx(base_lr)
        # Cosine to floor: at step 1000, value is min_lr_rate * base_lr.
        assert sched(1000) == pytest.approx(0.1 * base_lr, abs=1e-12)

    def test_cosine_warmup_with_min_lr_with_warmup_floor(self):
        # warmup_lr_rate=0.2 → warmup ramps from 0.2*base_lr to base_lr.
        base_lr = 1e-3
        sched = build_lr_schedule(
            _Args(
                lr_scheduler="cosine_warmup_with_min_lr",
                warmup_steps=100,
                lr_scheduler_kwargs={"min_lr_rate": 0.1, "warmup_lr_rate": 0.2},
            ),
            num_training_steps=1000,
        )
        assert sched(0) == pytest.approx(0.2 * base_lr)
        assert sched(100) == pytest.approx(base_lr)
        # Cosine to floor.
        assert sched(1000) == pytest.approx(0.1 * base_lr, abs=1e-12)

    def test_cosine_warmup_with_min_lr_decay_matches_cosine_with_min_lr(self):
        # Post-warmup, the decay shape is identical to cosine_with_min_lr.
        kwargs = {"min_lr_rate": 0.1}
        ours_warmup = build_lr_schedule(
            _Args(
                lr_scheduler="cosine_warmup_with_min_lr",
                warmup_steps=100,
                lr_scheduler_kwargs=kwargs,
            ),
            num_training_steps=1000,
        )
        ours_regular = build_lr_schedule(
            _Args(
                lr_scheduler="cosine_with_min_lr",
                warmup_steps=100,
                lr_scheduler_kwargs=kwargs,
            ),
            num_training_steps=1000,
        )
        for s in range(100, 1100, 50):  # post-warmup
            assert ours_warmup(s) == pytest.approx(ours_regular(s), abs=1e-12)

    def test_cosine_warmup_with_min_lr_accepts_absolute_min_lr(self):
        # HF parity: min_lr (absolute) is accepted and converted internally.
        base_lr = 1e-3
        sched = build_lr_schedule(
            _Args(
                lr_scheduler="cosine_warmup_with_min_lr",
                warmup_steps=100,
                lr_scheduler_kwargs={"min_lr": 1e-5},
            ),
            num_training_steps=1000,
        )
        assert sched(100) == pytest.approx(base_lr)
        # End of decay: cosine reaches 1e-5.
        assert sched(1000) == pytest.approx(1e-5, abs=1e-12)

    def test_warmup_stable_decay_cosine(self):
        ours = build_lr_schedule(
            _Args(
                lr_scheduler="warmup_stable_decay",
                warmup_steps=50,
                lr_scheduler_kwargs={"num_decay_steps": 200, "min_lr_ratio": 0.1},
            ),
            num_training_steps=1000,
        )
        hf = _hf_lambda(
            lambda o: get_wsd_schedule(
                o,
                num_warmup_steps=50,
                num_decay_steps=200,
                num_training_steps=1000,
                min_lr_ratio=0.1,
            )
        )
        _assert_pointwise(ours, hf, 1000)

    def test_warmup_stable_decay_linear(self):
        ours = build_lr_schedule(
            _Args(
                lr_scheduler="warmup_stable_decay",
                warmup_steps=50,
                lr_scheduler_kwargs={"num_decay_steps": 200, "decay_type": "linear"},
            ),
            num_training_steps=1000,
        )
        hf = _hf_lambda(
            lambda o: get_wsd_schedule(
                o,
                num_warmup_steps=50,
                num_decay_steps=200,
                num_training_steps=1000,
                decay_type="linear",
            )
        )
        _assert_pointwise(ours, hf, 1000)

    def test_warmup_stable_decay_explicit_stable(self):
        ours = build_lr_schedule(
            _Args(
                lr_scheduler="warmup_stable_decay",
                warmup_steps=50,
                lr_scheduler_kwargs={"num_decay_steps": 200, "num_stable_steps": 500},
            ),
            num_training_steps=1000,
        )
        hf = _hf_lambda(
            lambda o: get_wsd_schedule(
                o,
                num_warmup_steps=50,
                num_decay_steps=200,
                num_stable_steps=500,
            )
        )
        _assert_pointwise(ours, hf, 50 + 500 + 200)

    @pytest.mark.parametrize("warmup_type", ["linear", "cosine", "1-sqrt"])
    @pytest.mark.parametrize("decay_type", ["linear", "cosine", "1-sqrt"])
    def test_warmup_stable_decay_all_combos(self, warmup_type, decay_type):
        W, D, N = 50, 200, 1000
        kw = {
            "num_decay_steps": D,
            "warmup_type": warmup_type,
            "decay_type": decay_type,
            "min_lr_ratio": 0.1,
        }
        ours = build_lr_schedule(
            _Args(
                lr_scheduler="warmup_stable_decay",
                warmup_steps=W,
                lr_scheduler_kwargs=kw,
            ),
            num_training_steps=N,
        )
        hf = _hf_lambda(
            lambda o: get_wsd_schedule(
                o,
                num_warmup_steps=W,
                num_decay_steps=D,
                num_training_steps=N,
                warmup_type=warmup_type,
                decay_type=decay_type,
                min_lr_ratio=0.1,
            )
        )
        _assert_pointwise(ours, hf, N)


# ---------------------------------------------------------------------------
# warmup resolution from a fractional warmup_steps
# ---------------------------------------------------------------------------


class TestFractionalWarmupSteps:
    def test_fraction_resolves_to_steps(self):
        # 0.05 of 1000 -> 50 warmup steps
        ours = build_lr_schedule(
            _Args(lr_scheduler="linear", warmup_steps=0.05),
            num_training_steps=1000,
        )
        hf = _hf_lambda(lambda o: get_linear_schedule_with_warmup(o, 50, 1000))
        _assert_pointwise(ours, hf, 1000)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestErrors:
    def test_cosine_with_min_lr_requires_one_of_min_args(self):
        with pytest.raises(ValueError, match="min_lr"):
            build_lr_schedule(
                _Args(lr_scheduler="cosine_with_min_lr", warmup_steps=10),
                num_training_steps=100,
            )

    def test_cosine_with_min_lr_rejects_both_min_args(self):
        with pytest.raises(ValueError, match="only one"):
            build_lr_schedule(
                _Args(
                    lr_scheduler="cosine_with_min_lr",
                    warmup_steps=10,
                    lr_scheduler_kwargs={"min_lr": 1e-4, "min_lr_rate": 0.1},
                ),
                num_training_steps=100,
            )

    def test_cosine_warmup_with_min_lr_requires_min_arg(self):
        with pytest.raises(ValueError, match="min_lr"):
            build_lr_schedule(
                _Args(lr_scheduler="cosine_warmup_with_min_lr", warmup_steps=10),
                num_training_steps=100,
            )

    def test_wsd_requires_num_decay_steps(self):
        with pytest.raises(ValueError, match="num_decay_steps"):
            build_lr_schedule(
                _Args(lr_scheduler="warmup_stable_decay", warmup_steps=10),
                num_training_steps=100,
            )

    def test_wsd_invalid_warmup_type(self):
        with pytest.raises(ValueError, match="warmup_type"):
            build_lr_schedule(
                _Args(
                    lr_scheduler="warmup_stable_decay",
                    warmup_steps=10,
                    lr_scheduler_kwargs={"num_decay_steps": 20, "warmup_type": "bogus"},
                ),
                num_training_steps=100,
            )

    def test_wsd_invalid_decay_type(self):
        with pytest.raises(ValueError, match="decay_type"):
            build_lr_schedule(
                _Args(
                    lr_scheduler="warmup_stable_decay",
                    warmup_steps=10,
                    lr_scheduler_kwargs={"num_decay_steps": 20, "decay_type": "bogus"},
                ),
                num_training_steps=100,
            )

    def test_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown lr_scheduler"):
            build_lr_schedule(_Args(lr_scheduler="bogus"), 100)

    def test_unknown_kwargs_for_cosine(self):
        with pytest.raises(ValueError, match="unsupported keys"):
            build_lr_schedule(
                _Args(
                    lr_scheduler="cosine",
                    lr_scheduler_kwargs={"power": 2.0},  # 'power' belongs to polynomial
                ),
                num_training_steps=100,
            )

    def test_unknown_kwargs_for_constant(self):
        # constant accepts no kwargs at all
        with pytest.raises(ValueError, match="unsupported keys"):
            build_lr_schedule(
                _Args(
                    lr_scheduler="constant",
                    lr_scheduler_kwargs={"num_cycles": 1.0},
                ),
                num_training_steps=100,
            )


# ---------------------------------------------------------------------------
# User-supplied Schedule recipe path
# ---------------------------------------------------------------------------


class TestUserSuppliedRecipe:
    """``lr_scheduler`` accepts a Schedule recipe directly.

    When a recipe is passed, the HF-name dispatch is bypassed and the
    recipe is returned as-is; ``warmup_steps`` / ``lr_scheduler_kwargs``
    are HF-name-dispatch-only and must be unset.
    """

    def test_recipe_returned_unchanged(self):
        from opake.scheduling import cosine_schedule

        recipe = cosine_schedule(init_value=1e-3, end_value=0.0, transition_steps=100)
        out = build_lr_schedule(_Args(lr_scheduler=recipe), num_training_steps=100)
        assert out is recipe
        # Pointwise sanity: the recipe drives LR at every step.
        for step in (0, 25, 50, 75, 99):
            assert out(step) == pytest.approx(recipe(step))

    def test_warmup_steps_rejected_with_recipe(self):
        from opake.scheduling import cosine_schedule

        recipe = cosine_schedule(init_value=1e-3, end_value=0.0, transition_steps=100)
        with pytest.raises(ValueError, match="warmup_steps is incompatible"):
            build_lr_schedule(
                _Args(lr_scheduler=recipe, warmup_steps=10),
                num_training_steps=100,
            )

    def test_fractional_warmup_rejected_with_recipe(self):
        from opake.scheduling import cosine_schedule

        recipe = cosine_schedule(init_value=1e-3, end_value=0.0, transition_steps=100)
        with pytest.raises(ValueError, match="warmup_steps is incompatible"):
            build_lr_schedule(
                _Args(lr_scheduler=recipe, warmup_steps=0.1),
                num_training_steps=100,
            )

    def test_kwargs_rejected_with_recipe(self):
        from opake.scheduling import cosine_schedule

        recipe = cosine_schedule(init_value=1e-3, end_value=0.0, transition_steps=100)
        with pytest.raises(ValueError, match="lr_scheduler_kwargs"):
            build_lr_schedule(
                _Args(
                    lr_scheduler=recipe,
                    lr_scheduler_kwargs={"num_cycles": 1.0},
                ),
                num_training_steps=100,
            )


# ---------------------------------------------------------------------------
# HF comma-form parsing (covers the optim_args string shape via the
# unified _parse_dict_string helper now used by every dict field).
# ---------------------------------------------------------------------------


class TestParseDictStringCommaForm:
    def test_empty_returns_empty_dict(self):
        assert _parse_dict_string("") == {}

    def test_basic_parse_with_coercion(self):
        out = _parse_dict_string("momentum=0.9,nesterov=True,steps=10,name=adam")
        assert out == {
            "momentum": 0.9,
            "nesterov": True,
            "steps": 10,
            "name": "adam",
        }

    def test_whitespace_tolerated(self):
        assert _parse_dict_string(" a = 1 , b = 2 ") == {"a": 1, "b": 2}

    def test_missing_equals_raises(self):
        with pytest.raises(ValueError, match="key=value"):
            _parse_dict_string("momentum")

    def test_empty_key_raises(self):
        with pytest.raises(ValueError, match="empty key"):
            _parse_dict_string("=1")


# ---------------------------------------------------------------------------
# Warmup edge cases — boundary values that frequently lurk in real configs
# (zero warmup, full warmup, over-shoot warmup, polynomial-as-linear).
# ---------------------------------------------------------------------------


class TestWarmupEdgeCases:
    """HF-parity around degenerate warmup parameter combinations."""

    def test_linear_no_warmup(self):
        # ``warmup_steps=0`` ⇒ the warmup ramp collapses to the identity
        # transform.  HF and ours must match.
        ours = build_lr_schedule(
            _Args(lr_scheduler="linear", warmup_steps=0),
            num_training_steps=200,
        )
        hf = _hf_lambda(lambda o: get_linear_schedule_with_warmup(o, 0, 200))
        _assert_pointwise(ours, hf, 200)

    def test_cosine_no_warmup(self):
        ours = build_lr_schedule(
            _Args(lr_scheduler="cosine", warmup_steps=0),
            num_training_steps=200,
        )
        hf = _hf_lambda(lambda o: get_cosine_schedule_with_warmup(o, 0, 200))
        _assert_pointwise(ours, hf, 200)

    def test_warmup_equal_to_total_training_steps(self):
        # All-warmup schedule: ``warmup_steps == num_training_steps``
        # means the LR ramps from 0 to base_lr across the entire run.
        # HF and ours agree pointwise *during* the warmup ramp — only
        # at the exact boundary step they differ by one (HF clips
        # immediately to 0; ours holds at base_lr for one step before
        # the zero-length decay floors at 0).  This test asserts the
        # ramp itself is HF-parity; the boundary behaviour is
        # documented as a known divergence.
        N = 100
        ours = build_lr_schedule(
            _Args(lr_scheduler="linear", warmup_steps=N),
            num_training_steps=N,
        )
        hf_lam = _hf_lambda(lambda o: get_linear_schedule_with_warmup(o, N, N))
        # Exercise the ramp body excluding the boundary step.
        for s in range(0, N, 5):
            assert ours(s) == pytest.approx(hf_lam(s) * BASE_LR, abs=1e-9)
        # By step N+1, both have decayed to 0.
        assert ours(N + 1) == pytest.approx(0.0, abs=1e-9)

    def test_warmup_exceeds_total_training_steps(self):
        # Over-shoot: ``warmup_steps > num_training_steps`` means the
        # ramp never reaches base_lr; the decay phase has zero length.
        # HF clips silently — its lambda just keeps ramping forever and
        # the trainer never queries it past num_training_steps.  Ours
        # matches HF on the ramp body across the whole training-step
        # range.
        N = 100
        ours = build_lr_schedule(
            _Args(lr_scheduler="linear", warmup_steps=200),
            num_training_steps=N,
        )
        hf_lam = _hf_lambda(lambda o: get_linear_schedule_with_warmup(o, 200, N))
        for s in range(0, N, 5):
            assert ours(s) == pytest.approx(hf_lam(s) * BASE_LR, abs=1e-9)

    def test_polynomial_power_one_matches_linear(self):
        # ``polynomial(power=1.0, lr_end=0.0)`` is exactly a linear
        # decay.  Confirm the dispatcher produces identical curves to
        # ``linear``.
        N = 500
        polynomial = build_lr_schedule(
            _Args(
                lr_scheduler="polynomial",
                warmup_steps=50,
                lr_scheduler_kwargs={"power": 1.0, "lr_end": 0.0},
            ),
            num_training_steps=N,
        )
        linear = build_lr_schedule(
            _Args(lr_scheduler="linear", warmup_steps=50),
            num_training_steps=N,
        )
        for s in range(0, N + 50, 25):
            assert polynomial(s) == pytest.approx(linear(s), abs=1e-9)
