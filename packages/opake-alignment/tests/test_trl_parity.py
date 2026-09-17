# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Parity against the upstream implementations opake's losses are ported from.

Every expected value here comes from *executing* TRL or torch, not from
restating a formula in the test. TRL is reached through three entry points, and
the SFT / log-prob references use torch's own cross-entropy:

- ``DPOTrainer._compute_loss`` holds TRL's DPO loss arithmetic inline, so the
  class is allocated without its trainer bootstrap (no model, tokenizer,
  dataset or accelerator) and handed a callable returning fabricated logits.
  Only the loss math runs.
- ``CPOTrainer.cpo_loss`` and ``ORPOTrainer.odds_ratio_loss`` are pure methods
  over log-probabilities, called with a stub ``self`` carrying just the
  hyper-parameters they read. They live under ``trl.experimental``, so they are
  imported inside the tests that need them: if upstream drops them, only those
  tests fail.
- ``trl.trainer.sft_trainer.dft_loss`` is a module-level function.

The fabricated inputs are exact rather than approximate: with uniform logits
over a vocabulary of size ``_V``, every per-token log-probability is exactly
``-log _V``, so setting ``ref_logp = policy_logp - target`` makes TRL's internal
log-ratios equal the requested values bit-for-bit, so ``assert_close``'s
dtype-aware defaults have only the loss's own float arithmetic to absorb.

Inputs stay inside the representable range on the ``exp``-clamped paths
(``discopop``, ``forward_kl``, ``alpha_divergence``); the clamps themselves are
an intentional divergence from TRL and keep their own tests in
``dpo/loss/test_discopop.py`` and ``dpo/loss/test_f_divergence.py``.
"""

from __future__ import annotations

import types
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import pytest
import torch
import torch.nn.functional as F

# TRL is the optional ``opake[trl]`` extra; CI installs it via `--extra all`.
pytest.importorskip("trl")

from trl.trainer.dpo_trainer import DPOTrainer
from trl.trainer.sft_trainer import dft_loss as trl_dft_loss

from opake.alignment.dpo.loss import (
    apo_down_loss,
    apo_zero_loss,
    bco_loss,
    discopop_loss,
    exo_loss,
    f_divergence_logits,
    fused_sequence_logp,
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
from opake.alignment.sft.loss import (
    dft_loss,
    fused_dft_loss,
    fused_nll_loss,
    nll_loss,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_BETA = 0.3
_SMOOTHING = 0.15
_TAU = 0.05
_V = 4  # vocabulary size of the fabricated logits

_CHOSEN = torch.tensor([0.7, -0.4, 1.3])
_REJECTED = torch.tensor([-0.2, 0.9, -1.1])


class _NoopAccelerator:
    """Single-process stand-in: gathers are identity, device is CPU."""

    device = torch.device("cpu")

    def gather_for_metrics(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


def _experimental(module: str, attr: str) -> Any:
    """Import a ``trl.experimental`` symbol, skipping if upstream removed it."""
    with warnings.catch_warnings():
        # The experimental namespace warns on import by design.
        warnings.simplefilter("ignore")
        mod = pytest.importorskip(module)
    return getattr(mod, attr)


def _dpo_trainer(
    *,
    loss_types: list[str] | None = None,
    loss_weights: list[float] | None = None,
    beta: float = _BETA,
    label_smoothing: float = 0.0,
    f_divergence_type: str = "reverse_kl",
    alpha: float = 1.0,
    discopop_tau: float = _TAU,
    ld_alpha: float | None = None,
    use_weighting: bool = False,
) -> DPOTrainer:
    """A ``DPOTrainer`` carrying only the attributes its loss block reads.

    Keyword-only and exhaustive so a misspelled knob raises instead of silently
    falling back to a default the comparison would then be blind to.
    """
    trainer = object.__new__(DPOTrainer)
    trainer.accelerator = _NoopAccelerator()
    trainer.model = types.SimpleNamespace(training=False)
    trainer.aux_loss_enabled = False
    trainer.precompute_ref_logps = True
    trainer.ld_alpha = ld_alpha
    trainer.f_divergence_type = f_divergence_type
    trainer.f_alpha_divergence_coef = alpha
    trainer.beta = beta
    trainer.label_smoothing = label_smoothing
    trainer.loss_types = loss_types or ["sigmoid"]
    trainer.loss_weights = loss_weights or [1.0] * len(trainer.loss_types)
    trainer.use_weighting = use_weighting
    trainer.args = types.SimpleNamespace(discopop_tau=discopop_tau)
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer._total_train_tokens = 0
    return trainer


def _trl_dpo_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    **config: Any,
) -> torch.Tensor:
    """Run TRL's DPO loss block over a ``[chosen…, rejected…]`` batch."""

    def model(**_: Any) -> types.SimpleNamespace:
        return types.SimpleNamespace(logits=logits)

    return DPOTrainer._compute_loss(
        _dpo_trainer(**config),
        model,
        {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "completion_mask": completion_mask,
            "ref_chosen_logps": ref_chosen_logps,
            "ref_rejected_logps": ref_rejected_logps,
        },
        False,
    )


def _fabricate(
    chosen_logratio: torch.Tensor, rejected_logratio: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Inputs whose reconstructed log-ratios are exactly the requested ones.

    Uniform logits give an exact per-sequence log-prob of ``-log _V``, so
    offsetting the reference log-probs by the target log-ratio is lossless.
    """
    rows, seq_len = 2 * chosen_logratio.numel(), 2
    completion_mask = torch.zeros(rows, seq_len, dtype=torch.long)
    completion_mask[:, 1] = 1  # exactly one completion token per row
    policy_logp = -torch.log(torch.tensor(float(_V)))
    return (
        torch.zeros(rows, seq_len, _V),
        torch.zeros(rows, seq_len, dtype=torch.long),
        completion_mask,
        policy_logp - chosen_logratio,
        policy_logp - rejected_logratio,
    )


def _trl_dpo_batch(
    chosen_logratio: torch.Tensor, rejected_logratio: torch.Tensor, **config: Any
) -> torch.Tensor:
    """TRL's mean DPO loss for the given per-pair log-ratios."""
    return _trl_dpo_loss(*_fabricate(chosen_logratio, rejected_logratio), **config)


def _trl_dpo_per_pair(
    chosen_logratio: torch.Tensor, rejected_logratio: torch.Tensor, **config: Any
) -> torch.Tensor:
    """TRL's per-pair DPO losses, recovered one pair at a time.

    ``_compute_loss`` only returns the batch mean, so each pair is run as its
    own single-pair batch. Every variant compared here is elementwise, so the
    per-pair values and their mean must both agree.
    """
    return torch.stack(
        [
            _trl_dpo_batch(
                chosen_logratio[i : i + 1], rejected_logratio[i : i + 1], **config
            )
            for i in range(chosen_logratio.numel())
        ]
    )


def _trl_dpo_per_pair_grads(
    chosen_logratio: torch.Tensor, rejected_logratio: torch.Tensor, **config: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    """TRL's per-pair d(loss)/d(log-ratio), from TRL's own autograd graph.

    TRL forms ``logratio = policy_logp - ref_logp`` internally, so
    differentiating its loss w.r.t. the reference log-probs and negating gives
    the derivative w.r.t. the log-ratios the opake heads take directly.
    """
    chosen_grads, rejected_grads = [], []
    for i in range(chosen_logratio.numel()):
        *batch, ref_chosen, ref_rejected = _fabricate(
            chosen_logratio[i : i + 1], rejected_logratio[i : i + 1]
        )
        ref_chosen = ref_chosen.detach().requires_grad_()
        ref_rejected = ref_rejected.detach().requires_grad_()
        loss = _trl_dpo_loss(*batch, ref_chosen, ref_rejected, **config)
        grads = torch.autograd.grad(loss, (ref_chosen, ref_rejected))
        chosen_grads.append(-grads[0])
        rejected_grads.append(-grads[1])
    return torch.cat(chosen_grads), torch.cat(rejected_grads)


def _grads(
    fn: Callable[..., torch.Tensor], *inputs: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Gradients of ``fn(*inputs).sum()`` w.r.t. each input."""
    leaves = tuple(x.detach().clone().requires_grad_() for x in inputs)
    return torch.autograd.grad(fn(*leaves).sum(), leaves)


def _trl_cpo(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    *,
    loss_type: str,
    beta: float = _BETA,
    label_smoothing: float = 0.0,
    simpo_gamma: float = 0.0,
) -> torch.Tensor:
    """TRL's per-example CPO losses (``sigmoid`` / ``hinge`` / ``ipo`` / ``simpo``)."""
    cpo_trainer = _experimental("trl.experimental.cpo.cpo_trainer", "CPOTrainer")
    trainer = object.__new__(cpo_trainer)
    trainer.accelerator = _NoopAccelerator()
    trainer.alpha = 0.0  # AlphaPO reward transform off
    trainer.loss_type = loss_type
    trainer.beta = beta
    trainer.label_smoothing = label_smoothing
    trainer.simpo_gamma = simpo_gamma
    return cpo_trainer.cpo_loss(trainer, chosen_logp, rejected_logp)[0]


def _trl_orpo(
    chosen_logp: torch.Tensor, rejected_logp: torch.Tensor, *, beta: float
) -> torch.Tensor:
    """TRL's per-example ORPO odds-ratio term (``beta * logsigmoid(log_odds)``)."""
    orpo_trainer = _experimental("trl.experimental.orpo.orpo_trainer", "ORPOTrainer")
    trainer = object.__new__(orpo_trainer)
    trainer.accelerator = _NoopAccelerator()
    trainer.beta = beta
    return orpo_trainer.odds_ratio_loss(trainer, chosen_logp, rejected_logp)[0]


# ---------------------------------------------------------------------------
# Per-pair heads against trl.DPOTrainer
# ---------------------------------------------------------------------------

_DPO_HEADS = {
    "sigmoid_loss": (
        lambda c, r: sigmoid_loss(c, r, beta=_BETA),
        {"loss_types": ["sigmoid"]},
    ),
    "hinge_loss": (
        lambda c, r: hinge_loss(c, r, beta=_BETA),
        {"loss_types": ["hinge"]},
    ),
    "robust_loss": (
        lambda c, r: robust_loss(c, r, beta=_BETA, label_smoothing=_SMOOTHING),
        {"loss_types": ["robust"], "label_smoothing": _SMOOTHING},
    ),
    "exo_loss": (
        lambda c, r: exo_loss(c, r, beta=_BETA, label_smoothing=_SMOOTHING),
        {"loss_types": ["exo_pair"], "label_smoothing": _SMOOTHING},
    ),
    "nca_loss": (
        lambda c, r: nca_loss(c, r, beta=_BETA),
        {"loss_types": ["nca_pair"]},
    ),
    "bco_loss": (
        lambda c, r: bco_loss(c, r, beta=_BETA),
        {"loss_types": ["bco_pair"]},
    ),
    "sppo_loss": (
        lambda c, r: sppo_loss(c, r, beta=_BETA),
        {"loss_types": ["sppo_hard"]},
    ),
    "apo_zero_loss": (
        lambda c, r: apo_zero_loss(c, r, beta=_BETA),
        {"loss_types": ["apo_zero"]},
    ),
    "apo_down_loss": (
        lambda c, r: apo_down_loss(c, r, beta=_BETA),
        {"loss_types": ["apo_down"]},
    ),
    "discopop_loss": (
        lambda c, r: discopop_loss(c, r, beta=_BETA, discopop_tau=_TAU),
        {"loss_types": ["discopop"]},
    ),
    # TRL divides the ipo scores by the completion-token count; the fabricated
    # batch has exactly one completion token per row, so its divisor is 1 and
    # the comparison is against opake's documented "caller normalises" form.
    "ipo_loss": (lambda c, r: ipo_loss(c, r, beta=_BETA), {"loss_types": ["ipo"]}),
}


@pytest.mark.parametrize("name", sorted(_DPO_HEADS))
def test_dpo_head_matches_trl(name: str) -> None:
    """Each DPO head reproduces trl.DPOTrainer per pair, in the mean, and in grad."""
    opake_fn, config = _DPO_HEADS[name]
    ours = opake_fn(_CHOSEN, _REJECTED)
    torch.testing.assert_close(ours, _trl_dpo_per_pair(_CHOSEN, _REJECTED, **config))
    torch.testing.assert_close(
        ours.mean(), _trl_dpo_batch(_CHOSEN, _REJECTED, **config)
    )
    # Gradients too: a stop-gradient or a sign error can leave the value intact.
    for our_grad, their_grad in zip(
        _grads(opake_fn, _CHOSEN, _REJECTED),
        _trl_dpo_per_pair_grads(_CHOSEN, _REJECTED, **config),
        strict=True,
    ):
        torch.testing.assert_close(our_grad, their_grad)


@pytest.mark.parametrize(
    ("f_divergence_type", "alpha"),
    [
        ("reverse_kl", 1.0),
        ("forward_kl", 1.0),
        ("js_divergence", 1.0),
        ("alpha_divergence", 0.5),
        ("alpha_divergence", 2.0),
    ],
)
def test_f_divergence_logits_match_trl(f_divergence_type: str, alpha: float) -> None:
    """``f_divergence_logits`` reproduces TRL's remapped ``delta_score``.

    TRL drops the additive constant from each per-side remap, so only the
    difference is comparable — which is exactly what ``f_divergence_logits``
    returns. It is fed through the sigmoid head to observe it.
    """
    delta = f_divergence_logits(
        _CHOSEN, _REJECTED, f_divergence_type=f_divergence_type, alpha=alpha
    )
    ours = sigmoid_loss(delta, torch.zeros_like(delta), beta=_BETA)
    theirs = _trl_dpo_per_pair(
        _CHOSEN,
        _REJECTED,
        f_divergence_type=f_divergence_type,
        alpha=alpha,
    )
    torch.testing.assert_close(ours, theirs)


def test_mpo_combine_matches_trl_loss_type_list() -> None:
    """``mpo_combine`` reproduces TRL's ``loss_type=list`` weighted blend."""
    weights = {"sigmoid": 0.7, "hinge": 0.3}
    blended = mpo_combine(
        {
            "sigmoid": sigmoid_loss(_CHOSEN, _REJECTED, beta=_BETA),
            "hinge": hinge_loss(_CHOSEN, _REJECTED, beta=_BETA),
        },
        weights,
    )
    theirs = _trl_dpo_batch(
        _CHOSEN,
        _REJECTED,
        loss_types=list(weights),
        loss_weights=list(weights.values()),
    )
    torch.testing.assert_close(blended.mean(), theirs)


# ---------------------------------------------------------------------------
# Reference-free heads against the TRL CPO / ORPO trainers
# ---------------------------------------------------------------------------

_CPO_HEADS = {
    "sigmoid_loss": (
        lambda c, r: sigmoid_loss(c, r, beta=_BETA, label_smoothing=_SMOOTHING),
        "sigmoid",
        {"label_smoothing": _SMOOTHING},
    ),
    "hinge_loss": (lambda c, r: hinge_loss(c, r, beta=_BETA), "hinge", {}),
    "ipo_loss": (lambda c, r: ipo_loss(c, r, beta=_BETA), "ipo", {}),
    "simpo_loss": (
        lambda c, r: simpo_loss(
            c, r, beta=_BETA, gamma=0.5, label_smoothing=_SMOOTHING
        ),
        "simpo",
        {"label_smoothing": _SMOOTHING, "simpo_gamma": 0.5},
    ),
}


@pytest.mark.parametrize("name", sorted(_CPO_HEADS))
def test_head_matches_trl_cpo(name: str) -> None:
    """TRL's CPO trainer is a second, independent implementation of these heads."""
    opake_fn, loss_type, kwargs = _CPO_HEADS[name]

    def theirs(c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return _trl_cpo(c, r, loss_type=loss_type, beta=_BETA, **kwargs)

    torch.testing.assert_close(opake_fn(_CHOSEN, _REJECTED), theirs(_CHOSEN, _REJECTED))
    for our_grad, their_grad in zip(
        _grads(opake_fn, _CHOSEN, _REJECTED),
        _grads(theirs, _CHOSEN, _REJECTED),
        strict=True,
    ):
        torch.testing.assert_close(our_grad, their_grad)


def test_odds_ratio_matches_trl_orpo() -> None:
    """``odds_ratio_loss`` is TRL's ORPO term negated and unscaled by beta.

    TRL returns ``beta * logsigmoid(log_odds)`` and *subtracts* it from the NLL
    term, so the sign flip and the beta division recover opake's head.
    """
    beta = 0.5
    chosen_logp = torch.tensor([-0.5, -1.2, -0.05])
    rejected_logp = torch.tensor([-0.9, -0.3, -2.0])
    torch.testing.assert_close(
        odds_ratio_loss(chosen_logp, rejected_logp),
        -_trl_orpo(chosen_logp, rejected_logp, beta=beta) / beta,
    )


# ---------------------------------------------------------------------------
# LD-DPO and WPO against TRL's own log-prob pipeline
# ---------------------------------------------------------------------------


def _preference_batch(
    completion_starts: tuple[int, ...], seq_len: int = 8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random logits, ids and a ragged completion mask laid out ``[chosen, rejected]``."""
    torch.manual_seed(0)
    rows = len(completion_starts)
    logits = torch.randn(rows, seq_len, _V)
    input_ids = torch.randint(0, _V, (rows, seq_len))
    completion_mask = torch.zeros(rows, seq_len, dtype=torch.long)
    for row, start in enumerate(completion_starts):
        completion_mask[row, start:] = 1
    return logits, input_ids, completion_mask


def _shifted_per_token_logps(
    logits: torch.Tensor, input_ids: torch.Tensor
) -> torch.Tensor:
    """Per-token log-probs of the realised next token, in plain torch."""
    return (
        torch.log_softmax(logits[..., :-1, :], dim=-1)
        .gather(-1, input_ids[..., 1:].unsqueeze(-1))
        .squeeze(-1)
    )


def test_ld_dpo_split_matches_trl_ld_alpha() -> None:
    """``ld_dpo_split`` reproduces TRL's shared/tail length-desensitised split.

    TRL derives the shared prefix as ``min(chosen_len, rejected_len)`` over the
    shifted completion masks and folds the split into the sequence log-probs;
    with a zero reference and ``beta=1`` the sigmoid head exposes it. Both
    rejected completions are longer than their chosen counterpart, so the
    ``ld_alpha``-weighted tail is non-empty on every pair.
    """
    ld_alpha = 0.4
    logits, input_ids, completion_mask = _preference_batch((5, 4, 2, 1))
    per_token_logps = _shifted_per_token_logps(logits, input_ids)
    shifted_mask = completion_mask[..., 1:]

    chosen_len, rejected_len = shifted_mask.sum(-1).chunk(2)
    shared = torch.minimum(chosen_len, rejected_len)
    assert (rejected_len > shared).all(), "the tail must be non-empty to be tested"

    chosen_split, rejected_split = ld_dpo_split(
        per_token_logps, shifted_mask, torch.cat([shared, shared]), ld_alpha
    ).chunk(2)
    ours = sigmoid_loss(chosen_split, rejected_split, beta=1.0)

    zeros = torch.zeros(chosen_len.numel())
    theirs = _trl_dpo_loss(
        logits, input_ids, completion_mask, zeros, zeros, beta=1.0, ld_alpha=ld_alpha
    )
    torch.testing.assert_close(ours.mean(), theirs)


def test_wpo_weights_match_trl_use_weighting() -> None:
    """The WPO pair weight must equal TRL's ``use_weighting`` factor.

    TRL multiplies the per-sequence loss by ``w_chosen * w_rejected``, so the
    ratio of the weighted to the unweighted single-pair loss is that product.
    """
    logits, input_ids, completion_mask = _preference_batch((3, 2))
    per_token_logps = _shifted_per_token_logps(logits, input_ids)
    chosen_logps, rejected_logps = per_token_logps.chunk(2)
    chosen_logits, rejected_logits = logits[..., :-1, :].chunk(2)
    chosen_mask, rejected_mask = completion_mask[..., 1:].chunk(2)
    ours = wpo_weights(chosen_logps, chosen_mask, chosen_logits) * wpo_weights(
        rejected_logps, rejected_mask, rejected_logits
    )

    zeros = torch.zeros(1)
    args = (logits, input_ids, completion_mask, zeros, zeros)
    weighted = _trl_dpo_loss(*args, beta=1.0, use_weighting=True)
    plain = _trl_dpo_loss(*args, beta=1.0, use_weighting=False)
    torch.testing.assert_close(ours, (weighted / plain).reshape(1))


# ---------------------------------------------------------------------------
# SFT losses and per-sequence log-probs
# ---------------------------------------------------------------------------


def _sft_example() -> tuple[torch.Tensor, torch.Tensor]:
    """One example's logits plus labels with a prompt span and an ignored token."""
    torch.manual_seed(1)
    logits = torch.randn(7, 11)
    labels = torch.randint(0, 11, (7,))
    labels[:2] = -100
    labels[4] = -100
    return logits, labels


def _reference_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Torch's own masked cross-entropy under the causal-LM shift.

    TRL's ``loss_type="nll"`` routes to Hugging Face's ``ForCausalLMLoss``,
    which is this call; opake's per-example divisor equals TRL's batch divisor
    when the batch is one example.
    """
    return F.cross_entropy(logits[:-1], labels[1:], ignore_index=-100)


def _reference_dft(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """TRL's own ``dft_loss``.

    TRL divides by the batch-wide non-ignored token count; for one example that
    is opake's per-example divisor, so the DP-corrected divisor is inert here.
    """
    return trl_dft_loss(
        types.SimpleNamespace(logits=logits.unsqueeze(0)), labels.unsqueeze(0).clone()
    )


def _reference_completion_logp(
    logits: torch.Tensor, input_ids: torch.Tensor, completion_mask: torch.Tensor
) -> torch.Tensor:
    """Negated masked cross-entropy *sum* over the shifted completion span."""
    targets = torch.where(
        completion_mask[1:].bool(), input_ids[1:], torch.full_like(input_ids[1:], -100)
    )
    return -F.cross_entropy(logits[:-1], targets, ignore_index=-100, reduction="sum")


@pytest.mark.parametrize(
    ("opake_fn", "reference_fn"),
    [(nll_loss, _reference_nll), (dft_loss, _reference_dft)],
    ids=["nll", "dft"],
)
def test_sft_loss_matches_upstream(opake_fn, reference_fn) -> None:
    """The eager SFT losses match their upstream reference on a single example.

    The gradient check is what pins DFT's stop-gradient on the weighting
    probability: dropping the ``detach`` leaves the loss value unchanged.
    """
    logits, labels = _sft_example()
    torch.testing.assert_close(opake_fn(logits, labels), reference_fn(logits, labels))
    torch.testing.assert_close(
        _grads(lambda x: opake_fn(x, labels), logits)[0],
        _grads(lambda x: reference_fn(x, labels), logits)[0],
    )


@pytest.mark.parametrize(
    ("fused_fn", "reference_fn"),
    [(fused_nll_loss, _reference_nll), (fused_dft_loss, _reference_dft)],
    ids=["nll", "dft"],
)
def test_fused_sft_loss_matches_upstream(fused_fn, reference_fn) -> None:
    """The fused SFT twins match the same upstream reference, not just each other."""
    torch.manual_seed(3)
    hidden_states = torch.randn(7, 5)
    lm_head_weight = torch.randn(11, 5)
    _, labels = _sft_example()
    torch.testing.assert_close(
        fused_fn(hidden_states, lm_head_weight, labels),
        reference_fn(hidden_states @ lm_head_weight.T, labels),
    )


def test_sequence_logp_matches_torch_cross_entropy() -> None:
    """``sequence_logp`` is the negated masked cross-entropy sum, normalised or not."""
    torch.manual_seed(2)
    logits = torch.randn(7, _V)
    input_ids = torch.randint(0, _V, (7,))
    completion_mask = torch.zeros(7, dtype=torch.long)
    completion_mask[3:] = 1

    theirs = _reference_completion_logp(logits, input_ids, completion_mask)
    torch.testing.assert_close(
        sequence_logp(logits, input_ids, completion_mask), theirs
    )
    torch.testing.assert_close(
        sequence_logp(logits, input_ids, completion_mask, length_normalized=True),
        theirs / completion_mask[1:].sum(),
    )


def test_fused_sequence_logp_matches_torch_cross_entropy() -> None:
    """``fused_sequence_logp`` matches the same negated cross-entropy sum."""
    torch.manual_seed(4)
    hidden_states = torch.randn(7, 5)
    lm_head_weight = torch.randn(_V, 5)
    input_ids = torch.randint(0, _V, (7,))
    completion_mask = torch.zeros(7, dtype=torch.long)
    completion_mask[3:] = 1

    torch.testing.assert_close(
        fused_sequence_logp(hidden_states, lm_head_weight, input_ids, completion_mask),
        _reference_completion_logp(
            hidden_states @ lm_head_weight.T, input_ids, completion_mask
        ),
    )
