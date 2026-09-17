# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Row-locality sweep over every public alignment loss and combinator.

Poisoning one example's inputs with NaN must leave every *other* example's
forward value and per-example gradient bit-identical. That is the per-record
independence DP-SGD / DP-FTRL rely on, and it is stronger than a finiteness
check, which a cross-row dependency can survive. Each walk also asserts the
un-poisoned run is finite, so this is the family's single vmap-safety sweep.

Gradients are driven through ``vmap(grad(...))`` — the same composition
``clipped_grad`` uses — so the assertion is about per-example gradients, not
about a batch-reduced one. For the fused losses that includes the per-example
gradient slice w.r.t. the shared ``lm_head`` weight.

The sweep is parametrised *from* the public ``__all__`` lists, so a new export
without a registered case fails here by name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import pytest
import torch
from torch.func import grad, vmap

import opake.alignment.dpo.loss as dpo_loss
import opake.alignment.sft.loss as sft_loss

if TYPE_CHECKING:
    from collections.abc import Callable

_B = 4  # batch rows
_BAD = 2  # the poisoned row
_OTHERS = [i for i in range(_B) if i != _BAD]
_SEQ, _VOCAB, _HIDDEN = 6, 5, 3
_BETA = 0.3


class _Case(NamedTuple):
    """One loss reduced to a per-example callable plus the inputs to drive it."""

    fn: Callable[..., torch.Tensor]
    args: tuple[torch.Tensor, ...]
    diff: tuple[int, ...]  # argnums carrying gradient
    shared: tuple[int, ...] = ()  # argnums with no batch axis


def _logratios() -> torch.Tensor:
    return torch.tensor([0.7, -0.4, 1.3, 0.2])


def _rejected() -> torch.Tensor:
    return torch.tensor([-0.2, 0.9, -1.1, 0.5])


def _logps() -> torch.Tensor:
    """Length-normalised log-probs (strictly negative, as ORPO/SimPO require)."""
    return torch.tensor([-0.5, -1.2, -0.05, -2.0])


def _per_token_logps() -> torch.Tensor:
    return -torch.rand(_B, _SEQ)


def _completion_mask() -> torch.Tensor:
    mask = torch.zeros(_B, _SEQ, dtype=torch.long)
    mask[:, 2:] = 1
    return mask


def _logits() -> torch.Tensor:
    return torch.randn(_B, _SEQ, _VOCAB)


def _input_ids() -> torch.Tensor:
    return torch.randint(0, _VOCAB, (_B, _SEQ))


def _labels() -> torch.Tensor:
    labels = torch.randint(0, _VOCAB, (_B, _SEQ))
    labels[:, :2] = -100  # prompt span
    return labels


def _hidden_states() -> torch.Tensor:
    return torch.randn(_B, _SEQ, _HIDDEN)


def _lm_head_weight() -> torch.Tensor:
    return torch.randn(_VOCAB, _HIDDEN)


def _pair(fn: Callable[..., torch.Tensor]) -> _Case:
    """A per-pair head taking ``(chosen_logratio, rejected_logratio)``."""
    return _Case(fn, (_logratios(), _rejected()), diff=(0, 1))


# One case per public name. ``f_divergence_*`` use ``alpha_divergence`` — the
# widest branch (clamped ``exp`` plus a division); the divergence matrix itself
# is covered by dpo/loss/test_f_divergence.py.
_CASES: dict[str, Callable[[], _Case]] = {
    "sigmoid_loss": lambda: _pair(
        lambda c, r: dpo_loss.sigmoid_loss(c, r, beta=_BETA, label_smoothing=0.1)
    ),
    "hinge_loss": lambda: _pair(lambda c, r: dpo_loss.hinge_loss(c, r, beta=_BETA)),
    "robust_loss": lambda: _pair(
        lambda c, r: dpo_loss.robust_loss(c, r, beta=_BETA, label_smoothing=0.1)
    ),
    "ipo_loss": lambda: _pair(lambda c, r: dpo_loss.ipo_loss(c, r, beta=_BETA)),
    "discopop_loss": lambda: _pair(
        lambda c, r: dpo_loss.discopop_loss(c, r, beta=_BETA)
    ),
    "chosen_nll_loss": lambda: _Case(
        lambda c, r: dpo_loss.chosen_nll_loss(c, r, beta=_BETA),
        (_logratios(), _rejected()),
        diff=(0,),
    ),
    "apo_zero_loss": lambda: _pair(
        lambda c, r: dpo_loss.apo_zero_loss(c, r, beta=_BETA)
    ),
    "apo_down_loss": lambda: _pair(
        lambda c, r: dpo_loss.apo_down_loss(c, r, beta=_BETA)
    ),
    "exo_loss": lambda: _pair(
        lambda c, r: dpo_loss.exo_loss(c, r, beta=_BETA, label_smoothing=0.1)
    ),
    "nca_loss": lambda: _pair(lambda c, r: dpo_loss.nca_loss(c, r, beta=_BETA)),
    "bco_loss": lambda: _pair(
        lambda c, r: dpo_loss.bco_loss(c, r, beta=_BETA, delta=0.1)
    ),
    "sppo_loss": lambda: _pair(lambda c, r: dpo_loss.sppo_loss(c, r, beta=_BETA)),
    "simpo_loss": lambda: _pair(
        lambda c, r: dpo_loss.simpo_loss(c, r, beta=_BETA, gamma=0.5)
    ),
    "odds_ratio_loss": lambda: _Case(
        dpo_loss.odds_ratio_loss, (_logps(), _logps() - 0.3), diff=(0, 1)
    ),
    "f_divergence_remap": lambda: _Case(
        lambda x: dpo_loss.f_divergence_remap(
            x, f_divergence_type="alpha_divergence", alpha=2.0
        ),
        (_logratios(),),
        diff=(0,),
    ),
    "f_divergence_logits": lambda: _pair(
        lambda c, r: dpo_loss.f_divergence_logits(
            c, r, f_divergence_type="alpha_divergence", alpha=2.0
        )
    ),
    "mpo_combine": lambda: _pair(
        lambda c, r: dpo_loss.mpo_combine(
            {"pref": c, "sft": r}, {"pref": 0.7, "sft": 0.3}
        )
    ),
    "wpo_weights": lambda: _Case(
        dpo_loss.wpo_weights,
        (_per_token_logps(), _completion_mask()),
        diff=(0,),
    ),
    "ld_dpo_split": lambda: _Case(
        lambda p, m: dpo_loss.ld_dpo_split(p, m, 2, 0.5),
        (_per_token_logps(), _completion_mask()),
        diff=(0,),
    ),
    "sequence_logp": lambda: _Case(
        dpo_loss.sequence_logp,
        (_logits(), _input_ids(), _completion_mask()),
        diff=(0,),
    ),
    "fused_sequence_logp": lambda: _Case(
        dpo_loss.fused_sequence_logp,
        (_hidden_states(), _lm_head_weight(), _input_ids(), _completion_mask()),
        diff=(0, 1),
        shared=(1,),
    ),
    "nll_loss": lambda: _Case(sft_loss.nll_loss, (_logits(), _labels()), diff=(0,)),
    "dft_loss": lambda: _Case(sft_loss.dft_loss, (_logits(), _labels()), diff=(0,)),
    "fused_nll_loss": lambda: _Case(
        sft_loss.fused_nll_loss,
        (_hidden_states(), _lm_head_weight(), _labels()),
        diff=(0, 1),
        shared=(1,),
    ),
    "fused_dft_loss": lambda: _Case(
        sft_loss.fused_dft_loss,
        (_hidden_states(), _lm_head_weight(), _labels()),
        diff=(0, 1),
        shared=(1,),
    ),
}

_PUBLIC = sorted(dpo_loss.__all__) + sorted(sft_loss.__all__)


def _build(name: str) -> _Case:
    builder = _CASES.get(name)
    if builder is None:
        pytest.fail(f"no row-locality case registered for public loss {name!r}")
    torch.manual_seed(0)
    return builder()


def _poisoned(case: _Case) -> tuple[torch.Tensor, ...]:
    """Replace row ``_BAD`` of the first batched differentiable input with NaN."""
    target = next(i for i in case.diff if i not in case.shared)
    args = list(case.args)
    corrupt = args[target].clone()
    corrupt[_BAD] = float("nan")
    args[target] = corrupt
    return tuple(args)


def _in_dims(case: _Case) -> tuple[int | None, ...]:
    return tuple(None if i in case.shared else 0 for i in range(len(case.args)))


@pytest.mark.parametrize("name", _PUBLIC)
def test_forward_row_locality(name: str) -> None:
    """A NaN row corrupts its own forward value and no other row's."""
    case = _build(name)
    batched = vmap(case.fn, in_dims=_in_dims(case))
    clean = batched(*case.args)
    dirty = batched(*_poisoned(case))

    assert torch.isfinite(clean).all(), f"{name}: un-poisoned forward is not finite"
    assert torch.isnan(dirty[_BAD]), f"{name}: the poisoned row stayed finite"
    assert torch.equal(clean[_OTHERS], dirty[_OTHERS]), (
        f"{name}: poisoning row {_BAD} changed another row's forward value"
    )


@pytest.mark.parametrize("name", _PUBLIC)
def test_gradient_row_locality(name: str) -> None:
    """A NaN row leaves every other row's per-example gradient untouched.

    The poisoned row's *own* gradient is deliberately not constrained: for
    ``hinge_loss`` (relu subgradient), ``chosen_nll_loss`` / ``mpo_combine`` /
    ``ld_dpo_split`` (input-independent gradients), ``wpo_weights`` (detached)
    and the ``clamp``-guarded ``f_divergence`` branches it is correctly finite.
    """
    case = _build(name)
    batched = vmap(grad(case.fn, argnums=case.diff), in_dims=_in_dims(case))
    clean = batched(*case.args)
    dirty = batched(*_poisoned(case))

    for argnum, before, after in zip(case.diff, clean, dirty, strict=True):
        assert torch.isfinite(before).all(), (
            f"{name}: un-poisoned gradient w.r.t. argument {argnum} is not finite"
        )
        assert torch.equal(before[_OTHERS], after[_OTHERS]), (
            f"{name}: poisoning row {_BAD} changed another row's gradient "
            f"w.r.t. argument {argnum}"
        )
