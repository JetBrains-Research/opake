# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Behavioural tests of ``moe_clipped_grad`` and the router statistics.

A toy two-layer router (``E = 8``, ``k = 2``) on random hidden states stands
in for a MoE backbone: its router logits are ``x @ W_l`` so the surrogate's
gradient flows to ``W``, and a squared-error token loss plays the
cross-entropy.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from opaque.api.dpsgd.clipping import moe_clipped_grad
from opaque.api.dpsgd.clipping.types import MoeClipState
from opaque.api.engine.clipping._moe import (
    centred_load,
    load_balancing_surrogate,
    load_bound,
    router_load_and_probs,
)
from opaque.api.engine.clipping.types import ClippingStats
from opaque.exceptions import ConfigurationError
from opaque.random import key
from opaque.serialization import from_state_dict, state_dict
from opaque.types import PerGroup

E, K, L, D, T = 8, 2, 2, 6, 12


def _params(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "router": torch.randn(L, D, E, generator=g),
        "head": torch.randn(D, generator=g),
    }


def _batch(lengths, seed=1):
    g = torch.Generator().manual_seed(seed)
    b = len(lengths)
    x = torch.randn(b, T, D, generator=g)
    positions = torch.arange(T)[None, :]
    mask = (positions < torch.as_tensor(lengths)[:, None]).long()
    y = torch.randn(b, generator=g)
    return x, mask, y


def _loss_fn(params, x, mask, y):
    logits = [x @ params["router"][layer] for layer in range(L)]
    m = mask.float()
    pred = x @ params["head"]
    loss = ((pred - y) ** 2 * m).sum() / m.sum().clamp(min=1.0)
    return loss, logits, mask


def _factory(**overrides):
    kwargs = {
        "clipping_norm": 1.0,
        "normalize_by": 4.0,
        "batch_argnums": (1, 2, 3),
        "noise_multiplier": 1.0,
        "ratio": 0.5,
        "key": key(3),
        "top_k": K,
        "num_experts": E,
        "num_layers": L,
        "max_tokens": T,
        "alpha": 0.1,
    }
    kwargs.update(overrides)
    return moe_clipped_grad(_loss_fn, **kwargs)


def _random_logits(seed, num_layers=L, seq_len=T):
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(seq_len, E, generator=g) for _ in range(num_layers))


def _eager_stats(router_logits, mask):
    """Token-loop reference for ``router_load_and_probs``."""
    m = [1.0 if v != 0 else 0.0 for v in mask.reshape(-1).tolist()]
    t_x = sum(m)
    h = torch.zeros(len(router_logits), E)
    probs = torch.zeros(E)
    for layer, z in enumerate(router_logits):
        z = z.reshape(-1, E)
        for t in range(z.shape[0]):
            if m[t] == 0.0:
                continue
            p = torch.softmax(z[t].float(), dim=-1)
            for e in torch.topk(p, K).indices.tolist():
                h[layer, e] += 1.0
            probs += p
    if t_x == 0:
        return h, probs, torch.tensor(0.0)
    return h / t_x, probs / (len(router_logits) * t_x), torch.tensor(t_x)


# ---------------------------------------------------------------------------
# Router statistics
# ---------------------------------------------------------------------------


class TestRouterStatistics:
    def test_structure_and_bounds(self):
        logits = _random_logits(0)
        mask = torch.ones(T, dtype=torch.long)
        mask[-3:] = 0
        h, probs, t_x = router_load_and_probs(logits, mask, top_k=K, num_layers=L)
        assert h.shape == (L, E)
        assert probs.shape == (E,)
        assert t_x.item() == T - 3
        torch.testing.assert_close(h.sum(-1), torch.full((L,), float(K)))
        assert (h >= 0).all()
        assert (h <= 1).all()
        torch.testing.assert_close(probs.sum(), torch.tensor(1.0))
        d = centred_load(h, top_k=K)
        torch.testing.assert_close(d.sum(-1), torch.zeros(L))
        assert d.norm().item() <= math.sqrt(K * L * (1 - K / E)) + 1e-6

    def test_matches_eager_loop_and_ignores_padding(self):
        logits = _random_logits(1)
        mask = torch.ones(T, dtype=torch.long)
        mask[7:] = 0
        h, probs, t_x = router_load_and_probs(logits, mask, top_k=K, num_layers=L)
        h_ref, p_ref, t_ref = _eager_stats(logits, mask)
        torch.testing.assert_close(h, h_ref)
        torch.testing.assert_close(probs, p_ref)
        torch.testing.assert_close(t_x, t_ref)
        perturbed = tuple(
            torch.cat([z[:7], torch.randn_like(z[7:]) * 50]) for z in logits
        )
        h2, p2, _ = router_load_and_probs(perturbed, mask, top_k=K, num_layers=L)
        torch.testing.assert_close(h2, h)
        torch.testing.assert_close(p2, probs)

    @pytest.mark.parametrize("variant", ["none", "weighted", "two_dim", "bool"])
    def test_masks_are_binarised_never_weighted(self, variant):
        """Any mask is read as ``mask != 0``; entries never act as weights."""
        logits = _random_logits(2)
        binary = torch.ones(T, dtype=torch.long)
        binary[9:] = 0
        if variant == "none":
            mask = None
            binary = torch.ones(T, dtype=torch.long)
        elif variant == "weighted":
            mask = torch.where(binary.bool(), -1.0, 0.0)
            mask[0] = 2.5
        elif variant == "two_dim":
            mask = binary[None, :]
        else:
            mask = binary.bool()
        expected = router_load_and_probs(logits, binary, top_k=K, num_layers=L)
        actual = router_load_and_probs(logits, mask, top_k=K, num_layers=L)
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b)
        h, probs, t_x = actual
        assert t_x.item() == binary.sum().item()
        torch.testing.assert_close(h.sum(-1), torch.full((L,), float(K)))
        torch.testing.assert_close(probs.sum(), torch.tensor(1.0))

    def test_fully_masked_row_contributes_nothing(self):
        logits = _random_logits(3)
        h, probs, t_x = router_load_and_probs(
            logits, torch.zeros(T), top_k=K, num_layers=L
        )
        assert t_x.item() == 0.0
        assert torch.equal(h, torch.zeros(L, E))
        assert torch.equal(probs, torch.zeros(E))
        assert torch.equal(
            centred_load(h, top_k=K, valid_tokens=t_x), torch.zeros(L, E)
        )

    def test_layer_count_mismatch_is_rejected(self):
        logits = _random_logits(4, num_layers=L + 1)
        with pytest.raises(ConfigurationError, match="duplicated capture"):
            router_load_and_probs(logits, None, top_k=K, num_layers=L)

    def test_bound_attained_by_collapsed_routing_and_never_exceeded(self):
        bound = load_bound(
            top_k=K, num_experts=E, num_layers=L, mean_tokens=T, max_tokens=T
        )
        # Every token of every layer routed to the same two experts.
        collapsed = torch.full((T, E), -10.0)
        collapsed[:, :K] = 10.0
        h, _, _ = router_load_and_probs(
            (collapsed, collapsed), None, top_k=K, num_layers=L
        )
        assert centred_load(h, top_k=K).norm().item() == pytest.approx(bound, rel=1e-6)
        g = torch.Generator().manual_seed(5)
        for _ in range(50):
            logits = tuple(torch.randn(T, E, generator=g) * 5 for _ in range(L))
            h, _, _ = router_load_and_probs(logits, None, top_k=K, num_layers=L)
            assert centred_load(h, top_k=K).norm().item() <= bound + 1e-6

    def test_surrogate_is_zero_at_balance_and_linear_in_the_estimate(self):
        probs = torch.softmax(torch.randn(E), -1)
        balanced = torch.full((E,), K / E)
        assert load_balancing_surrogate(
            probs, balanced, 1.0, num_experts=E, top_k=K
        ).item() == pytest.approx(0.0, abs=1e-6)
        f = torch.rand(E)
        s = load_balancing_surrogate(probs, f, 1.0, num_experts=E, top_k=K)
        expected = E * ((f - K / E) * probs).sum()
        torch.testing.assert_close(s, expected)

    def test_load_bound_validation(self):
        with pytest.raises(ConfigurationError, match="top_k"):
            load_bound(
                top_k=E, num_experts=E, num_layers=L, mean_tokens=1, max_tokens=1
            )
        with pytest.raises(ConfigurationError, match="num_layers"):
            load_bound(
                top_k=K, num_experts=E, num_layers=0, mean_tokens=1, max_tokens=1
            )
        with pytest.raises(ConfigurationError, match="positive"):
            load_bound(
                top_k=K, num_experts=E, num_layers=L, mean_tokens=0, max_tokens=1
            )
        with pytest.raises(ConfigurationError, match="finite"):
            load_bound(
                top_k=K, num_experts=E, num_layers=L, mean_tokens=1, max_tokens=math.nan
            )


# ---------------------------------------------------------------------------
# moe_clipped_grad
# ---------------------------------------------------------------------------


def _batch_aux_loss(params, x, mask):
    """HF-style batch load-balancing loss, differentiable through ``P``."""
    m = mask.float()
    total = m.sum()
    loads, probs = [], []
    for layer in range(L):
        p = torch.softmax((x @ params["router"][layer]).float(), -1)
        idx = torch.topk(p, K, -1).indices
        one_hot = (idx[..., None] == torch.arange(E)).sum(-2).float()
        loads.append((one_hot * m[..., None]).sum((0, 1)))
        probs.append((p * m[..., None]).sum((0, 1)))
    f = torch.stack(loads).sum(0) / (L * total)
    big_p = torch.stack(probs).sum(0) / (L * total)
    return E * (f.detach() * big_p).sum(), f.detach()


def _reference_grads(params, x, mask, y, f_tilde, alpha):
    """Autograd of ``mean_x loss_x + alpha * L_aux(B)`` at the frozen load."""
    leaves = {k: v.clone().requires_grad_(True) for k, v in params.items()}
    losses = torch.stack(
        [_loss_fn(leaves, x[i], mask[i], y[i])[0] for i in range(x.shape[0])]
    )
    aux, _ = _batch_aux_loss(leaves, x, mask)
    total = losses.mean() + alpha * aux
    grads = torch.autograd.grad(total, list(leaves.values()))
    return dict(zip(leaves, grads, strict=True))


class TestSeparability:
    @pytest.mark.parametrize("lengths", [[T, T, T, T], [T, 9, T, 5]])
    def test_mean_surrogate_gradient_matches_batch_aux_gradient(self, lengths):
        params = _params()
        x, mask, y = _batch(lengths)
        b = len(lengths)
        alpha = 0.3
        _, f_b = _batch_aux_loss(params, x, mask)
        grad_fn, state = _factory(
            clipping_norm=1e9,
            noise_multiplier=0.0,
            normalize_by=b,
            alpha=alpha,
            max_tokens=T,
            mean_tokens=T,
        )
        state = replace(state, f_tilde=f_b)
        grads, _ = grad_fn(params, x, mask, y, state=state)
        # Token weights w_x = T_x / T scale the aux gradient by T_tot / (B T).
        scale = mask.sum().item() / (b * T)
        ref = _reference_grads(params, x, mask, y, f_b, alpha * scale)
        for name in params:
            torch.testing.assert_close(
                grads.pytree[name], ref[name], atol=1e-5, rtol=1e-4
            )

    def test_estimate_enters_only_through_the_surrogate(self):
        params = _params()
        x, mask, y = _batch([T, T, T, T])
        grad_fn, state = _factory(clipping_norm=1e9, noise_multiplier=0.0, alpha=0.3)
        uniform, _ = grad_fn(params, x, mask, y, state=state)
        skewed = replace(state, f_tilde=torch.linspace(0.0, 0.5, E))
        moved, _ = grad_fn(params, x, mask, y, state=skewed)
        assert not torch.allclose(uniform.pytree["router"], moved.pytree["router"])
        torch.testing.assert_close(uniform.pytree["head"], moved.pytree["head"])
        grad_fn0, state0 = _factory(clipping_norm=1e9, noise_multiplier=0.0, alpha=0.0)
        off, _ = grad_fn0(
            params, x, mask, y, state=replace(state0, f_tilde=skewed.f_tilde)
        )
        torch.testing.assert_close(off.pytree["router"], uniform.pytree["router"])


class TestRelease:
    def test_gradient_stream_is_the_plain_clipped_pytree(self):
        params = _params()
        x, mask, y = _batch([T, 9, T, 5])
        grad_fn, state = _factory(clipping_norm=0.5, normalize_by=4.0)
        grads, _ = grad_fn(params, x, mask, y, state=state)
        assert set(grads.pytree) == {"router", "head"}
        assert grads.max_norm == pytest.approx(0.5 / 4.0)

    def test_noise_scale_and_filter_formula(self):
        params = _params()
        x, mask, y = _batch([T, T, T, T])
        n = 64.0
        signal = None
        samples = []
        for seed in range(300):
            grad_fn, state = _factory(
                noise_multiplier=1.0, ratio=1.0, normalize_by=n, key=key(seed)
            )
            _, new = grad_fn(params, x, mask, y, state=state)
            assert new.step == 1
            samples.append(new.f_tilde - K / E)
        grad_fn, state = _factory(noise_multiplier=0.0, ratio=1.0, normalize_by=n)
        _, clean = grad_fn(params, x, mask, y, state=state)
        signal = clean.f_tilde - K / E
        stacked = torch.stack(samples)
        bound = load_bound(
            top_k=K, num_experts=E, num_layers=L, mean_tokens=T, max_tokens=T
        )
        sigma = bound / n
        assert new.load_noise_std == pytest.approx(sigma)
        expected_var = sigma**2 / L * (E - 1) / E
        assert new.filtered_noise_std == pytest.approx(math.sqrt(expected_var))
        torch.testing.assert_close(
            stacked.mean(0), signal, atol=4 * math.sqrt(expected_var / 300), rtol=0
        )
        empirical = (stacked - signal).var(0, unbiased=True)
        assert torch.allclose(
            empirical, torch.full((E,), expected_var), rtol=0.35, atol=0
        )

    def test_ratio_sets_the_load_noise(self):
        _, a = _factory(noise_multiplier=2.0, ratio=0.25, normalize_by=8.0)
        _, b = _factory(noise_multiplier=2.0, ratio=1.0, normalize_by=8.0)
        assert a.load_noise_std == pytest.approx(2 * b.load_noise_std)
        assert a.load_noise_std == pytest.approx(2.0 / 0.5 * a._load_bound / 8.0)

    def test_deterministic_in_the_key(self):
        params = _params()
        x, mask, y = _batch([T, 9, T, 5])
        outs = []
        for seed in (3, 3, 4):
            grad_fn, state = _factory(key=key(seed))
            _, new = grad_fn(params, x, mask, y, state=state)
            outs.append(new.f_tilde)
        assert torch.equal(outs[0], outs[1])
        assert not torch.equal(outs[0], outs[2])

    def test_empty_batch_releases_noise_and_advances(self):
        params = _params()
        x, mask, y = _batch([T, 9])
        grad_fn, state = _factory()
        grads, new = grad_fn(params, x[:0], mask[:0], y[:0], state=state)
        assert new.step == 1
        assert all(torch.count_nonzero(v) == 0 for v in grads.pytree.values())
        assert not torch.equal(new.f_tilde, state.f_tilde)
        _, clean = _factory(noise_multiplier=0.0)
        _, clean_new = _factory(noise_multiplier=0.0)[0](
            params, x[:0], mask[:0], y[:0], state=clean
        )
        assert torch.equal(clean_new.f_tilde, clean.f_tilde)

    def test_microbatching_is_equivalent(self):
        params = _params()
        x, mask, y = _batch([T, 9, T, 5, 7])
        grad_fn, state = _factory(normalize_by=5.0)
        grads, new = grad_fn(params, x, mask, y, state=state)
        grad_fn_mb, state_mb = _factory(normalize_by=5.0, microbatch_size=2)
        grads_mb, new_mb = grad_fn_mb(params, x, mask, y, state=state_mb)
        for name in params:
            torch.testing.assert_close(
                grads.pytree[name], grads_mb.pytree[name], atol=1e-6, rtol=1e-5
            )
        torch.testing.assert_close(new.f_tilde, new_mb.f_tilde, atol=1e-6, rtol=1e-5)

    def test_state_round_trips_and_continues(self):
        params = _params()
        x, mask, y = _batch([T, 9, T, 5])
        grad_fn, state = _factory()
        _, s1 = grad_fn(params, x, mask, y, state=state)
        restored = from_state_dict(state, state_dict(s1))
        assert isinstance(restored, MoeClipState)
        assert torch.equal(restored.f_tilde, s1.f_tilde)
        assert restored.step == 1
        _, s2 = grad_fn(params, x, mask, y, state=s1)
        _, s2r = grad_fn(params, x, mask, y, state=restored)
        assert torch.equal(s2.f_tilde, s2r.f_tilde)

    def test_diagnostics_never_carry_the_load(self):
        params = _params()
        x, mask, y = _batch([T, 9, T, 5])
        grad_fn, state = _factory(return_aux=True)
        (grads, aux), _ = grad_fn(params, x, mask, y, state=state)
        assert aux.loss_aux is None
        assert aux.batch_size == 4
        assert set(grads.pytree) == {"router", "head"}
        plain = torch.stack(
            [_loss_fn(params, x[i], mask[i], y[i])[0] for i in range(4)]
        )
        torch.testing.assert_close(aux.loss_values, plain)
        grad_fn_s, state_s = _factory(return_stats=True, clipping_norm=0.01)
        (_, stats), _ = grad_fn_s(params, x, mask, y, state=state_s)
        assert isinstance(stats, ClippingStats)
        assert stats.batch_size == 4
        assert stats.num_clipped == 4

    def test_per_group_gradient_bound(self):
        params = _params()
        x, mask, y = _batch([T, 9, T, 5])
        bound = PerGroup(
            groups={("router",): "router", ("head",): "head"},
            values={"router": 0.3, "head": 0.2},
        )
        grad_fn, state = _factory(clipping_norm=bound, normalize_by=4.0)
        grads, new = grad_fn(params, x, mask, y, state=state)
        assert isinstance(grads.max_norm, PerGroup)
        assert set(grads.pytree) == {"router", "head"}
        assert new.step == 1

    def test_long_rows_are_clipped_to_the_bound(self):
        """A row above ``max_tokens`` is under-weighted, never over-released."""
        params = _params()
        x, mask, y = _batch([T])
        grad_fn, state = _factory(
            noise_multiplier=0.0, normalize_by=1.0, max_tokens=4, mean_tokens=4
        )
        _, new = grad_fn(params, x, mask, y, state=state)
        d = new.f_tilde - K / E
        bound = load_bound(
            top_k=K, num_experts=E, num_layers=L, mean_tokens=4, max_tokens=4
        )
        # The pooled, projected estimate is a mean of L rows of the released
        # (L, E) mean, so its norm is at most the bound.
        assert d.norm().item() <= bound + 1e-6

    def test_validation(self):
        with pytest.raises(ConfigurationError, match="ratio"):
            _factory(ratio=0.0)
        with pytest.raises(ConfigurationError, match="filter_beta"):
            _factory(filter_beta=1.0)
        with pytest.raises(ConfigurationError, match="noise_multiplier"):
            _factory(noise_multiplier=-1.0)
        with pytest.raises(ConfigurationError, match="noise_multiplier"):
            _factory(noise_multiplier=math.nan)
        with pytest.raises(ConfigurationError, match="ratio"):
            _factory(ratio=math.inf)
        with pytest.raises(ConfigurationError, match="batch_argnums"):
            _factory(batch_argnums=0)
        with pytest.raises(ConfigurationError, match="return_aux"):
            _factory(return_aux=True, return_stats=True)
