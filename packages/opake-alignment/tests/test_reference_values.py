# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Closed-form reference values derived from each loss's paper equation.

Every expectation below is a constant obtained by evaluating the *published*
equation at a point where it collapses to something a reader can check by hand
— not by re-typing the implementation. Feeding the logistic heads an argument of
``log 3`` makes σ rational (3/4), so the shifted cases stay exact instead of
degenerating into a restated formula.

Symmetric points where a loss vanishes or hits a fixed value independent of its
hyper-parameters are the strongest checks available (they pin the constants that
a formula restatement cannot), but they can pass vacuously under an exponent or
sign mutation, so each is paired with an asymmetric companion.

Parity against the upstream implementations lives in ``test_trl_parity.py``;
this module is the reference for the parts of the surface that have no upstream
counterpart, and a second, independent opinion on the parts that do.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
import torch

from opake.alignment.dpo.loss import (
    apo_down_loss,
    apo_zero_loss,
    bco_loss,
    chosen_nll_loss,
    discopop_loss,
    exo_loss,
    f_divergence_logits,
    f_divergence_remap,
    hinge_loss,
    ipo_loss,
    ld_dpo_split,
    mpo_combine,
    nca_loss,
    odds_ratio_loss,
    robust_loss,
    sequence_logp,
    sigmoid_loss,
    simpo_loss,
    sppo_loss,
    wpo_weights,
)
from opake.alignment.sft.loss import dft_loss, nll_loss

if TYPE_CHECKING:
    from collections.abc import Callable

_ATOL = 1e-6

_LOG2 = math.log(2.0)
_LOG3 = math.log(3.0)
_LOG4 = math.log(4.0)
_LOG43 = math.log(4.0 / 3.0)  # -log σ(log 3), i.e. σ(log 3) = 3/4


def _t(value: float) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float32)


_ZERO = _t(0.0)

# Each entry: (id, thunk, expected). The comment above each block states the
# identity being used; the arithmetic is in the expected value, never in the
# call.
_CASES: list[tuple[str, Callable[[], torch.Tensor], float]] = [
    # DPO sigmoid (Rafailov 2023). At Δ=0 both blended terms equal -log σ(0),
    # so the smoothing weights sum back to 1 and the loss is log 2 for every ε.
    (
        "sigmoid/delta-zero-any-smoothing",
        lambda: sigmoid_loss(_ZERO, _ZERO, beta=0.3, label_smoothing=0.4),
        _LOG2,
    ),
    (
        "sigmoid/sigma-three-quarters",
        lambda: sigmoid_loss(_t(_LOG3), _ZERO, beta=1.0),
        _LOG43,
    ),
    (
        "sigmoid/sigma-three-quarters-smoothed",
        lambda: sigmoid_loss(_t(_LOG3), _ZERO, beta=1.0, label_smoothing=0.25),
        0.75 * _LOG43 + 0.25 * _LOG4,
    ),
    # DPO hinge (Liu 2023): relu(1 - βΔ) — zero once the margin is met, and
    # linear in the deficit below it.
    ("hinge/no-margin", lambda: hinge_loss(_ZERO, _ZERO, beta=1.0), 1.0),
    ("hinge/margin-exactly-met", lambda: hinge_loss(_t(1.0), _ZERO, beta=1.0), 0.0),
    ("hinge/margin-exceeded", lambda: hinge_loss(_t(3.0), _ZERO, beta=1.0), 0.0),
    ("hinge/reversed-pair", lambda: hinge_loss(_t(-1.0), _ZERO, beta=1.0), 2.0),
    # Robust: the (1-2ε) denominator exactly cancels the blend at Δ=0, so the
    # loss is log 2 across the whole admissible ε range. That invariant is the
    # point of the variant and no restated formula would expose it.
    ("robust/delta-zero-eps-0", lambda: robust_loss(_ZERO, _ZERO, beta=0.3), _LOG2),
    (
        "robust/delta-zero-eps-0.2",
        lambda: robust_loss(_ZERO, _ZERO, beta=0.3, label_smoothing=0.2),
        _LOG2,
    ),
    (
        "robust/delta-zero-eps-0.45",
        lambda: robust_loss(_ZERO, _ZERO, beta=0.3, label_smoothing=0.45),
        _LOG2,
    ),
    (
        "robust/sigma-three-quarters-smoothed",
        lambda: robust_loss(_t(_LOG3), _ZERO, beta=1.0, label_smoothing=0.25),
        (0.75 * _LOG43 - 0.25 * _LOG4) / 0.5,
    ),
    # IPO (Azar 2024, Eq. 17): a parabola with its vertex at Δ = 1/(2β).
    ("ipo/at-vertex", lambda: ipo_loss(_t(0.5), _ZERO, beta=1.0), 0.0),
    ("ipo/vertex-offset-beta-1", lambda: ipo_loss(_ZERO, _ZERO, beta=1.0), 0.25),
    ("ipo/vertex-offset-beta-quarter", lambda: ipo_loss(_ZERO, _ZERO, beta=0.25), 4.0),
    # SimPO (Meng 2024): the margin γ cancels the reward gap, leaving log 2.
    (
        "simpo/margin-cancels-gap",
        lambda: simpo_loss(_t(0.5), _ZERO, beta=1.0, gamma=0.5, label_smoothing=0.3),
        _LOG2,
    ),
    (
        "simpo/zero-margin-is-sigmoid",
        lambda: simpo_loss(_t(_LOG3), _ZERO, beta=1.0),
        _LOG43,
    ),
    # DiscoPOP (Eq. 5): gate σ(βΔ/τ) blends -log σ(βΔ) with exp(-βΔ). At Δ=0 the
    # gate is 1/2 and the two components are log 2 and 1; at βΔ = τ = log 3 the
    # gate is 3/4 and the exponential component is 1/3.
    (
        "discopop/balanced-gate",
        lambda: discopop_loss(_ZERO, _ZERO, beta=1.0),
        (1.0 + _LOG2) / 2.0,
    ),
    (
        "discopop/gate-three-quarters",
        lambda: discopop_loss(_t(_LOG3), _ZERO, beta=1.0, discopop_tau=1.0),
        0.25 * _LOG43 + 0.75 / 3.0,
    ),
    # SFT regulariser used by MPO/RPO/CPO blends: the plain NLL of the chosen
    # completion.
    ("chosen_nll/negates-logp", lambda: chosen_nll_loss(_t(-2.5)), 2.5),
    # APO (arXiv:2408.06266, Eqs. 7-8). σ(±log 3) ∈ {3/4, 1/4}.
    ("apo_zero/neutral-pair", lambda: apo_zero_loss(_ZERO, _ZERO, beta=1.0), 1.0),
    (
        "apo_zero/separated-pair",
        lambda: apo_zero_loss(_t(_LOG3), _t(-_LOG3), beta=1.0),
        0.5,
    ),
    ("apo_down/neutral-pair", lambda: apo_down_loss(_ZERO, _ZERO, beta=1.0), 1.0),
    (
        "apo_down/chosen-up-margin-negative",
        lambda: apo_down_loss(_t(_LOG3), _t(2 * _LOG3), beta=1.0),
        1.5,
    ),
    # EXO (arXiv:2402.00856, Eq. 16) is KL(q ‖ p) between the implied preference
    # q = σ(βΔ) and the smoothed target p = (1-ε, ε). It is exactly zero when
    # the two coincide — at ε = 1/4 that is βΔ = log 3 — and at Δ=0 it reduces
    # to the cross-entropy of a fair coin against the target.
    (
        "exo/kl-is-zero-when-q-equals-target",
        lambda: exo_loss(_t(_LOG3), _ZERO, beta=1.0, label_smoothing=0.25),
        0.0,
    ),
    (
        "exo/fair-coin-against-target",
        lambda: exo_loss(_ZERO, _ZERO, beta=1.0, label_smoothing=0.25),
        -_LOG2 - 0.5 * math.log(0.25 * 0.75),
    ),
    # NCA: -log σ(βc) - ½ log σ(-βc) - ½ log σ(-βr).
    ("nca/neutral-pair", lambda: nca_loss(_ZERO, _ZERO, beta=1.0), 2 * _LOG2),
    (
        "nca/separated-pair",
        lambda: nca_loss(_t(_LOG3), _t(-_LOG3), beta=1.0),
        1.5 * _LOG43 + _LOG2,
    ),
    # BCO: a binary classifier on the two shifted rewards; δ translates both.
    ("bco/neutral-pair", lambda: bco_loss(_ZERO, _ZERO, beta=1.0), 2 * _LOG2),
    (
        "bco/baseline-shifts-both-rewards",
        lambda: bco_loss(_t(_LOG3 + 0.5), _t(0.5 - _LOG3), beta=1.0, delta=0.5),
        2 * _LOG43,
    ),
    # SPPO hard labels: squared deviation from the Nash targets ±1/(2β).
    ("sppo/at-nash-targets", lambda: sppo_loss(_t(0.5), _t(-0.5), beta=1.0), 0.0),
    ("sppo/origin-beta-1", lambda: sppo_loss(_ZERO, _ZERO, beta=1.0), 0.5),
    ("sppo/origin-beta-half", lambda: sppo_loss(_ZERO, _ZERO, beta=0.5), 2.0),
    # ORPO (arXiv:2403.07691): -log σ(log odds(y_w) - log odds(y_r)). Equal
    # log-probs give zero log-odds; p = 3/4 has log odds log 3, p = 1/2 has 0.
    (
        "orpo/equal-log-probs",
        lambda: odds_ratio_loss(_t(math.log(0.5)), _t(math.log(0.5))),
        _LOG2,
    ),
    (
        "orpo/three-quarters-against-half",
        lambda: odds_ratio_loss(_t(math.log(0.75)), _t(math.log(0.5))),
        _LOG43,
    ),
    # f-divergence remaps: every g vanishes at a zero log-ratio (the policy
    # equals the reference), which is what makes them interchangeable there.
    ("f_divergence/reverse_kl-at-zero", lambda: f_divergence_remap(_ZERO), 0.0),
    (
        "f_divergence/forward_kl-at-zero",
        lambda: f_divergence_remap(_ZERO, f_divergence_type="forward_kl"),
        0.0,
    ),
    (
        "f_divergence/js-at-zero",
        lambda: f_divergence_remap(_ZERO, f_divergence_type="js_divergence"),
        0.0,
    ),
    (
        "f_divergence/alpha-at-zero",
        lambda: f_divergence_remap(
            _ZERO, f_divergence_type="alpha_divergence", alpha=2.0
        ),
        0.0,
    ),
    # g(x) = 1 - e^{-x} at x = log 2; log 2 + log σ(log 3) = log(3/2);
    # (e^{(α-1)x} - 1)/(α-1) at (α, x) = (2, log 3) and (1/2, log 4).
    (
        "f_divergence/forward_kl-at-log2",
        lambda: f_divergence_remap(_t(_LOG2), f_divergence_type="forward_kl"),
        0.5,
    ),
    (
        "f_divergence/js-at-log3",
        lambda: f_divergence_remap(_t(_LOG3), f_divergence_type="js_divergence"),
        math.log(1.5),
    ),
    (
        "f_divergence/alpha-2-at-log3",
        lambda: f_divergence_remap(
            _t(_LOG3), f_divergence_type="alpha_divergence", alpha=2.0
        ),
        2.0,
    ),
    (
        "f_divergence/alpha-half-at-log4",
        lambda: f_divergence_remap(
            _t(_LOG4), f_divergence_type="alpha_divergence", alpha=0.5
        ),
        1.0,
    ),
    (
        "f_divergence/logits-are-the-remap-difference",
        lambda: f_divergence_logits(
            _t(_LOG3), _t(_LOG2), f_divergence_type="js_divergence"
        ),
        math.log(1.5) - (_LOG2 + math.log(2.0 / 3.0)),
    ),
    # MPO blend: a plain weighted sum over the selected terms.
    (
        "mpo/weighted-sum-of-selected-terms",
        lambda: mpo_combine(
            {"pref": _t(1.0), "sft": _t(2.0)}, {"pref": 0.25, "sft": 0.5}
        ),
        1.25,
    ),
]


def _wpo(*token_distributions: tuple[float, ...]) -> torch.Tensor:
    logits = torch.tensor([token_distributions], dtype=torch.float32).log()
    logps = logits[..., 0]
    return wpo_weights(logps, torch.ones_like(logps, dtype=torch.long), logits)[0]


def _ld_split(alpha: float) -> torch.Tensor:
    """Four completion tokens of log-prob -1, the first two inside the prefix."""
    per_token = torch.full((1, 4), -1.0)
    mask = torch.ones(1, 4, dtype=torch.long)
    return ld_dpo_split(per_token, mask, 2, alpha)[0]


def _uniform_logits(seq_len: int, vocab: int) -> torch.Tensor:
    """Logits assigning every token probability 1/vocab, so log p = -log vocab."""
    return torch.zeros(seq_len, vocab)


_CASES += [
    # WPO: each realised class-0 probability is divided by the distribution's
    # collision probability before taking the geometric mean.
    (
        "wpo/uniform-tokens-align-to-one",
        lambda: _wpo((0.5, 0.5), (0.5, 0.5), (0.5, 0.5)),
        1.0,
    ),
    (
        "wpo/geometric-mean-of-aligned-token-weights",
        lambda: _wpo((0.5, 0.5), (0.125, 0.875)),
        0.4,
    ),
    # LD-DPO: prefix tokens keep weight 1, tail tokens are damped by alpha, so
    # four unit log-probs split 2 + 2 interpolate between -2 and -4.
    ("ld_dpo/alpha-one-is-the-plain-sum", lambda: _ld_split(1.0), -4.0),
    ("ld_dpo/alpha-half-damps-the-tail", lambda: _ld_split(0.5), -3.0),
    ("ld_dpo/alpha-zero-keeps-the-prefix", lambda: _ld_split(0.0), -2.0),
    # Per-sequence logp over uniform logits: three completion tokens survive the
    # causal shift, each contributing -log 4.
    (
        "sequence_logp/uniform-logits-sum",
        lambda: sequence_logp(
            _uniform_logits(5, 4),
            torch.zeros(5, dtype=torch.long),
            torch.tensor([0, 0, 1, 1, 1]),
        ),
        -3 * _LOG4,
    ),
    (
        "sequence_logp/uniform-logits-mean",
        lambda: sequence_logp(
            _uniform_logits(5, 4),
            torch.zeros(5, dtype=torch.long),
            torch.tensor([0, 0, 1, 1, 1]),
            length_normalized=True,
        ),
        -_LOG4,
    ),
    # SFT over uniform logits: the NLL of every token is log V, and DFT weights
    # it by the detached probability 1/V.
    (
        "nll/uniform-logits",
        lambda: nll_loss(_uniform_logits(5, 8), torch.tensor([-100, -100, 1, 2, 3])),
        math.log(8.0),
    ),
    (
        "dft/uniform-logits",
        lambda: dft_loss(_uniform_logits(5, 8), torch.tensor([-100, -100, 1, 2, 3])),
        math.log(8.0) / 8.0,
    ),
]


@pytest.mark.parametrize(
    ("thunk", "expected"),
    [(thunk, expected) for _, thunk, expected in _CASES],
    ids=[case_id for case_id, _, _ in _CASES],
)
def test_matches_paper_reference_value(
    thunk: Callable[[], torch.Tensor], expected: float
) -> None:
    """The loss equals the value its published equation takes at this point."""
    torch.testing.assert_close(
        thunk(), torch.tensor(expected, dtype=torch.float32), atol=_ATOL, rtol=0.0
    )
