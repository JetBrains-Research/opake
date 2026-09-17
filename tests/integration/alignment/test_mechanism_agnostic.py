"""§11.6 mechanism-agnostic integration test for ``opake-alignment``.

Per ``docs/development/opake-alignment-plan.md`` §11.6 (and the §3.2 /
§12.5 mechanism-agnostic contract): the SAME per-example loss closure,
built only from ``opake.alignment`` primitives + ``opake-engine``
clipping, must run end-to-end under BOTH DP mechanisms —

  * DP-SGD: i.i.d. Gaussian noise (``opake.dpsgd.noise.gaussian_noise``).
  * DP-FTRL: correlated matrix-factorized noise
    (``opake.dpftrl.noise.band_mf_strategy`` + ``mf_gaussian_noise``).

``opake-alignment`` declares no dependency on either mechanism wheel
(plan §5, §12.5); the mechanism is chosen at the call site.  This smoke
test proves the package contract holds under mechanism substitution: the
clipped gradients from the shared closure are identical pre-noise, and
both post-noise gradient trees are finite — only the noise step differs.

This test imports both ``opake.dpsgd`` and ``opake.dpftrl`` (outside the
alignment wheel's dependency cone), so it lives under repo-root
``tests/integration/`` rather than ``packages/opake-alignment/tests/``.

CPU-only, tiny, deterministic; no network.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from opake.alignment.sft.loss import nll_loss
from opake.dpftrl.noise import band_mf_strategy, mf_gaussian_noise
from opake.dpsgd.clipping import clipped_grad
from opake.dpsgd.noise import gaussian_noise
from opake.functional import make_functional
from opake.random import key

# Mechanism-substitution smoke test runs in well under 5 s on CPU and needs
# no GPU, so it carries no ``slow`` / ``cuda`` marker (matching the sibling
# unmarked CPU tests under ``tests/integration``).

_SEED = 0
_BATCH, _SEQ, _VOCAB, _HIDDEN = 4, 6, 16, 8
_CLIP_NORM = 1.0
_NOISE_MULTIPLIER = 1.0
_N_STEPS = 8  # DP-FTRL training horizon for the band-MF streaming matrix.


class _ToyLM(nn.Module):
    """Two-layer toy causal LM producing logits ``(B, T, V)``."""

    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(_VOCAB, _HIDDEN)
        self.head = nn.Linear(_HIDDEN, _VOCAB)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.emb(input_ids))


def _finite_tree(noised) -> bool:
    """True iff every leaf of a noised pytree is finite."""
    return all(torch.isfinite(t).all().item() for t in noised.pytree.values())


def test_same_closure_runs_under_dpsgd_and_dpftrl() -> None:
    """One alignment loss closure -> clip -> {Gaussian, matrix-factorized} noise."""
    torch.manual_seed(_SEED)

    model = _ToyLM()
    fmodel, trainable, frozen = make_functional(model, partition_trainable=True)

    input_ids = torch.randint(0, _VOCAB, (_BATCH, _SEQ))
    labels = input_ids.clone()

    # ONE per-example loss closure, built from an alignment primitive
    # (SFT ``nll`` over the model logits — plan §7.3, Tier 1).  This is the
    # mechanism-agnostic object: nothing here knows about DP-SGD vs DP-FTRL.
    def per_example_loss(
        trainable_params: dict, ids: torch.Tensor, labs: torch.Tensor
    ) -> torch.Tensor:
        merged = {**frozen, **trainable_params}
        logits = fmodel(merged, ids)
        return nll_loss(logits, labs)

    # Shared clipping (opake-engine) — produces the per-example-clipped,
    # batch-summed gradient tree fed identically to both mechanisms.
    grad_fn, clip_state = clipped_grad(
        per_example_loss,
        clipping_norm=_CLIP_NORM,
        normalize_by=_BATCH,
        batch_argnums=(1, 2),
    )
    clipped, _ = grad_fn(trainable, input_ids, labels, state=clip_state)

    grad_keys = list(clipped.pytree.keys())
    assert grad_keys, "expected trainable gradient leaves"
    assert all(torch.isfinite(t).all() for t in clipped.pytree.values())

    # Snapshot the shared clipped grads so we can prove mechanism
    # substitution only adds noise and never mutates the closure's output.
    clipped_before = {k: clipped.pytree[k].clone() for k in grad_keys}

    # --- DP-SGD: i.i.d. Gaussian noise -------------------------------------
    sgd_noise_fn, sgd_state = gaussian_noise(
        noise_multiplier=_NOISE_MULTIPLIER, key=key(_SEED)
    )
    noisy_sgd, _ = sgd_noise_fn(clipped, sgd_state)

    # --- DP-FTRL: correlated matrix-factorized (band-MF) noise -------------
    # Mirror examples/train_dpftrl.py: band_mf_strategy(...) recipe +
    # mf_gaussian_noise(grad_template, strategy, n_steps=..., ...).
    strategy = band_mf_strategy(bands=4, momentum=0.95)
    ftrl_noise_fn, ftrl_state = mf_gaussian_noise(
        trainable,
        strategy,
        n_steps=_N_STEPS,
        min_sep=1,
        max_participations=_N_STEPS,
        noise_multiplier=_NOISE_MULTIPLIER,
        key=key(_SEED),
    )
    noisy_ftrl, _ = ftrl_noise_fn(clipped, ftrl_state)

    # Contract 1: both mechanisms consumed the SAME clipped grads, and
    # neither mutated them — the shared closure's clipped gradient tree is
    # bit-identical before and after noising (only the noise step differs).
    for k in grad_keys:
        assert torch.equal(clipped.pytree[k], clipped_before[k]), (
            f"clipped grad {k!r} mutated by a noise mechanism"
        )
    # Post-noise trees keep identical structure (same keys / shapes).
    assert set(noisy_sgd.pytree) == set(grad_keys) == set(noisy_ftrl.pytree)
    for k in grad_keys:
        assert noisy_sgd.pytree[k].shape == clipped.pytree[k].shape
        assert noisy_ftrl.pytree[k].shape == clipped.pytree[k].shape

    # Contract 2: both post-noise gradient trees are finite.
    assert _finite_tree(noisy_sgd), "DP-SGD noised grads must be finite"
    assert _finite_tree(noisy_ftrl), "DP-FTRL noised grads must be finite"


def test_dpo_closure_runs_under_both_mechanisms() -> None:
    """A DPO ``sigmoid`` closure (with ``sequence_logp``) is mechanism-agnostic too."""
    from opake.alignment.dpo.loss import sigmoid_loss
    from opake.api.alignment.logprob import sequence_logp

    torch.manual_seed(_SEED)

    model = _ToyLM()
    fmodel, trainable, frozen = make_functional(model, partition_trainable=True)

    chosen_ids = torch.randint(0, _VOCAB, (_BATCH, _SEQ))
    rejected_ids = torch.randint(0, _VOCAB, (_BATCH, _SEQ))
    completion_mask = torch.ones(_BATCH, _SEQ, dtype=torch.long)
    ref_chosen = torch.zeros(_BATCH)
    ref_rejected = torch.zeros(_BATCH)

    def per_example_loss(
        trainable_params: dict,
        c_ids: torch.Tensor,
        r_ids: torch.Tensor,
        c_mask: torch.Tensor,
        ref_c: torch.Tensor,
        ref_r: torch.Tensor,
    ) -> torch.Tensor:
        merged = {**frozen, **trainable_params}
        chosen_logp = sequence_logp(fmodel(merged, c_ids), c_ids, c_mask)
        rejected_logp = sequence_logp(fmodel(merged, r_ids), r_ids, c_mask)
        return sigmoid_loss(chosen_logp - ref_c, rejected_logp - ref_r, beta=0.1)

    grad_fn, clip_state = clipped_grad(
        per_example_loss,
        clipping_norm=_CLIP_NORM,
        normalize_by=_BATCH,
        batch_argnums=(1, 2, 3, 4, 5),
    )
    clipped, _ = grad_fn(
        trainable,
        chosen_ids,
        rejected_ids,
        completion_mask,
        ref_chosen,
        ref_rejected,
        state=clip_state,
    )

    sgd_noise_fn, sgd_state = gaussian_noise(
        noise_multiplier=_NOISE_MULTIPLIER, key=key(_SEED)
    )
    noisy_sgd, _ = sgd_noise_fn(clipped, sgd_state)

    ftrl_noise_fn, ftrl_state = mf_gaussian_noise(
        trainable,
        band_mf_strategy(bands=4, momentum=0.95),
        n_steps=_N_STEPS,
        min_sep=1,
        max_participations=_N_STEPS,
        noise_multiplier=_NOISE_MULTIPLIER,
        key=key(_SEED),
    )
    noisy_ftrl, _ = ftrl_noise_fn(clipped, ftrl_state)

    assert _finite_tree(noisy_sgd), "DP-SGD DPO noised grads must be finite"
    assert _finite_tree(noisy_ftrl), "DP-FTRL DPO noised grads must be finite"
