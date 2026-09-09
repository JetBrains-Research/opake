# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Portable chunked linear-CE (non-Triton MPS/CPU path) parity tests.

The complement of ``test_linear_cross_entropy.py`` (CUDA + Triton): this pins
the pure-PyTorch chunked kernel that runs where Triton is unavailable. It must
match eager ``matmul + cross_entropy`` precision staging within streaming
roundoff for every feature, on the direct call and under ``vmap(grad)`` (the
DP-SGD path), with frozen and trainable lm-head weight.

Tensors are deliberately tiny: allocation-shape checks prove the workspace is
bounded without OOM-risking peak-memory assertions on the small MPS runner.
Timing and realistic peak-memory measurements remain out-of-band benchmarks.
"""

from __future__ import annotations

import pytest
import torch
from torch.func import grad, vmap

import opaque.api.patches.kernels._linear_ce_chunked as mod
from opaque.api.patches.kernels._linear_ce_chunked import (
    linear_cross_entropy_chunked,
)

_TOL = 5e-4  # fp32 streaming vs materialized: roundoff only


def _softcap(x, sc):
    return sc * torch.tanh(x / sc) if sc else x


def _eager_mean(
    hidden,
    weight,
    labels,
    ignore_index=-100,
    logit_softcapping=0,
    label_smoothing=0.0,
    use_token_scaling=False,
    logit_scale=1.0,
):
    """Materialized, vmap-safe reference matching the kernel's math + reduction."""
    sc = logit_softcapping or None
    flat = _softcap((hidden[..., :-1, :] @ weight.transpose(-1, -2)) * logit_scale, sc)
    flat = flat.reshape(-1, weight.shape[0])
    t = labels[..., 1:].reshape(-1)
    valid = t != ignore_index
    lse = torch.logsumexp(flat, -1)
    lt = flat.gather(1, t.clamp(min=0)[:, None]).squeeze(1)
    nll = lse - lt
    if label_smoothing:
        e = label_smoothing
        loss = (1 - e) * nll + e * (lse - flat.mean(-1))
    else:
        loss = nll
    if use_token_scaling:
        loss = torch.exp(lt - lse).detach() * loss
    loss = torch.where(valid, loss, loss.new_zeros(()))
    return loss.sum() / valid.sum().clamp(min=1).to(loss.dtype)


_FEATURES = [
    pytest.param({}, id="plain"),
    pytest.param({"_ignore": True}, id="ignore_index"),
    pytest.param({"logit_softcapping": 30.0}, id="softcap"),
    pytest.param({"label_smoothing": 0.1}, id="label_smoothing"),
    pytest.param({"use_token_scaling": True}, id="token_scaling"),
    pytest.param({"logit_scale": 0.25, "logit_softcapping": 30.0}, id="scaled_softcap"),
]


def test_workspace_budget_is_fixed_on_cpu_and_bounded_on_mps(monkeypatch):
    assert mod._workspace_budget_bytes(torch.device("cpu")) == 512 * 1024**2

    mps = getattr(torch, "mps", None)
    if mps is None:
        pytest.skip("torch.mps API unavailable")
    monkeypatch.setattr(mod, "_MPS_MAX_WORKSPACE_BYTES", 8000)
    monkeypatch.setattr(mps, "recommended_max_memory", lambda: 100_000, raising=False)
    monkeypatch.setattr(mps, "driver_allocated_memory", lambda: 20_000, raising=False)
    assert mod._workspace_budget_bytes(torch.device("mps")) == 8000

    def unavailable():
        raise RuntimeError("unavailable")

    monkeypatch.setattr(mps, "recommended_max_memory", unavailable)
    assert mod._workspace_budget_bytes(torch.device("mps")) == 512 * 1024**2


def test_tile_plan_bounds_large_token_and_vocabulary_axes():
    hidden = torch.empty(8192, 3072, dtype=torch.bfloat16, device="meta")
    weight = torch.empty(128_000, 3072, dtype=torch.bfloat16, device="meta")
    plan = mod._tile_plan(hidden, weight)

    assert 1 <= plan.tokens < hidden.shape[0]
    assert 1 <= plan.vocab < weight.shape[0]
    assert plan.estimated_bytes <= plan.budget_bytes


def test_tile_plan_honors_vocab_cap_and_minimum_tile():
    hidden = torch.empty(257, 4)
    weight = torch.empty(200, 4)
    plan = mod._tile_plan(hidden, weight, chunk_vocab=64, budget_bytes=4096)
    assert plan.vocab == 64
    assert 1 <= plan.tokens < hidden.shape[0]
    assert plan.estimated_bytes <= plan.budget_bytes

    minimum = mod._tile_plan(hidden, weight, chunk_vocab=64, budget_bytes=1)
    assert (minimum.tokens, minimum.vocab) == (1, 1)
    with pytest.raises(ValueError, match="chunk_vocab must be positive"):
        mod._tile_plan(hidden, weight, chunk_vocab=0)


def test_tile_plan_includes_hidden_vmap_batch_dimension(monkeypatch):
    torch.manual_seed(4)
    hidden = torch.randn(4, 9, 4)
    weight = torch.randn(64, 4)
    labels = torch.randint(0, 64, (4, 9))
    vmapped_plans = []
    original_plan = mod._tile_plan

    def recording_plan(*args, **kwargs):
        plan = original_plan(*args, **kwargs)
        if plan.batch_factor > 1:
            vmapped_plans.append(plan)
        return plan

    monkeypatch.setattr(mod, "_workspace_budget_bytes", lambda _device: 4096)
    monkeypatch.setattr(mod, "_tile_plan", recording_plan)

    def loss(h, w, lab):
        return linear_cross_entropy_chunked(h, w, lab, chunk_vocab=32)

    vmap(grad(loss), in_dims=(0, None, 0))(hidden, weight, labels)

    assert vmapped_plans
    assert all(plan.batch_factor == hidden.shape[0] for plan in vmapped_plans)
    assert all(plan.estimated_bytes <= plan.budget_bytes for plan in vmapped_plans)


def _check_parity(device: str) -> None:
    torch.manual_seed(0)
    b, t, d = 3, 7, 16
    for feat_param in (p.values[0] for p in _FEATURES):
        feat = dict(feat_param)
        ignore = feat.pop("_ignore", False)
        # vocab 50 -> single chunk; 256 with _CHUNK_VOCAB=64 -> 4 chunks
        for vocab, chunk_vocab in ((50, 16384), (256, 64)):
            old_vocab = mod._CHUNK_VOCAB
            old_budget = mod._workspace_budget_bytes
            mod._CHUNK_VOCAB = chunk_vocab
            mod._workspace_budget_bytes = lambda _device: 4096
            try:
                h = (torch.randn(b, t, d, device=device)).requires_grad_(True)
                w = (torch.randn(vocab, d, device=device) * 0.1).requires_grad_(True)
                lab = torch.randint(0, vocab, (b, t), device=device)
                if ignore:
                    lab[:, 1] = -100

                lk = linear_cross_entropy_chunked(h, w, lab, **feat)
                lr = _eager_mean(h, w, lab, **feat)
                assert (lk - lr).abs().item() < _TOL, f"forward {feat} v{vocab}"

                gkh, gkw = torch.autograd.grad(lk, (h, w))
                grh, grw = torch.autograd.grad(lr, (h, w))
                assert (gkh - grh).abs().max().item() < _TOL, f"d_hidden {feat}"
                assert (gkw - grw).abs().max().item() < _TOL, f"d_weight {feat}"

                # vmap(grad) — the DP-SGD per-example path.
                wf = w.detach()
                hd = h.detach()

                def fk(hh, ww, ll, f=feat):
                    return linear_cross_entropy_chunked(hh, ww, ll, **f)

                def fr(hh, ww, ll, f=feat):
                    return _eager_mean(hh, ww, ll, **f)

                # frozen head: grad wrt hidden
                gk = vmap(grad(fk, 0), in_dims=(0, None, 0))(hd, wf, lab)
                gr = vmap(grad(fr, 0), in_dims=(0, None, 0))(hd, wf, lab)
                assert (gk - gr).abs().max().item() < _TOL, f"vmap d_hidden {feat}"
                # trainable head: per-example grad wrt weight [B, V, D]
                gk2 = vmap(grad(fk, (0, 1)), in_dims=(0, None, 0))(hd, wf, lab)
                gr2 = vmap(grad(fr, (0, 1)), in_dims=(0, None, 0))(hd, wf, lab)
                assert (gk2[1] - gr2[1]).abs().max().item() < _TOL, (
                    f"vmap d_weight {feat}"
                )
            finally:
                mod._CHUNK_VOCAB = old_vocab
                mod._workspace_budget_bytes = old_budget


def test_chunked_linear_ce_parity_cpu():
    _check_parity("cpu")


def test_chunked_linear_ce_backward_does_not_allocate_dense_onehot(monkeypatch):
    torch.manual_seed(0)
    h = torch.randn(2, 8, 16, requires_grad=True)
    w = torch.randn(256, 16, requires_grad=True)
    labels = torch.randint(0, 256, (2, 8))
    allocated_shapes = []
    original_zeros_like = torch.zeros_like

    def record_zeros_like(tensor, *args, **kwargs):
        allocated_shapes.append(tuple(tensor.shape))
        return original_zeros_like(tensor, *args, **kwargs)

    old = mod._CHUNK_VOCAB
    mod._CHUNK_VOCAB = 64
    try:
        loss = linear_cross_entropy_chunked(h, w, labels)
        monkeypatch.setattr(torch, "zeros_like", record_zeros_like)
        torch.autograd.grad(loss, (h, w))
    finally:
        mod._CHUNK_VOCAB = old

    assert (14, 64) not in allocated_shapes


def test_chunked_linear_ce_bounds_both_probability_tile_dimensions(monkeypatch):
    torch.manual_seed(0)
    h = torch.randn(1, 258, 4, requires_grad=True)
    w = torch.randn(200, 4)
    labels = torch.randint(0, w.shape[0], (1, 258))
    exp_shapes = []
    original_exp = torch.exp

    def record_exp(tensor, *args, **kwargs):
        if tensor.ndim == 2:
            exp_shapes.append(tuple(tensor.shape))
        return original_exp(tensor, *args, **kwargs)

    monkeypatch.setattr(mod, "_workspace_budget_bytes", lambda _device: 4096)
    monkeypatch.setattr(torch, "exp", record_exp)
    plan = mod._tile_plan(h[..., :-1, :].flatten(0, -2), w, chunk_vocab=64)
    loss = linear_cross_entropy_chunked(h, w, labels, chunk_vocab=64)
    torch.autograd.grad(loss, h)

    assert exp_shapes
    assert max(rows for rows, _ in exp_shapes) <= plan.tokens < 257
    assert max(cols for _, cols in exp_shapes) <= plan.vocab == 64


@pytest.mark.parametrize("vmapped", [False, True], ids=["direct", "vmap"])
@pytest.mark.parametrize("use_token_scaling", [False, True], ids=["plain", "scaled"])
@pytest.mark.parametrize("has_ignored", [False, True], ids=["all-valid", "ignored"])
def test_chunked_backward_reuses_forward_statistics(
    monkeypatch, vmapped, use_token_scaling, has_ignored
):
    torch.manual_seed(1)
    b, t, d, vocab = 2, 6, 8, 32
    hidden = torch.randn(b, t, d)
    weight = torch.randn(vocab, d)
    labels = torch.randint(0, vocab, (b, t))
    if has_ignored:
        labels[:, 2] = -100

    calls = 0
    original = mod._stream_lse

    def recording_stream_lse(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(mod, "_stream_lse", recording_stream_lse)

    def loss(h, w, lab):
        return linear_cross_entropy_chunked(
            h, w, lab, use_token_scaling=use_token_scaling
        )

    if vmapped:
        vmap(grad(loss, (0, 1)), in_dims=(0, None, 0))(hidden, weight, labels)
    else:
        hidden.requires_grad_(True)
        weight.requires_grad_(True)
        torch.autograd.grad(loss(hidden, weight, labels), (hidden, weight))

    assert calls == 1


def test_chunked_backward_reuses_exact_forward_tile_plan(monkeypatch):
    hidden = torch.randn(1, 9, 8, requires_grad=True)
    weight = torch.randn(32, 8)
    labels = torch.randint(0, 32, (1, 9))
    calls = 0

    def shrinking_budget(_device):
        nonlocal calls
        calls += 1
        return 4096 if calls == 1 else 1

    monkeypatch.setattr(mod, "_workspace_budget_bytes", shrinking_budget)
    loss = linear_cross_entropy_chunked(hidden, weight, labels, chunk_vocab=8)
    torch.autograd.grad(loss, hidden)

    assert calls == 1


@pytest.mark.parametrize("vmapped", [False, True], ids=["direct", "vmap"])
def test_chunked_backward_fuses_fp32_hidden_accumulation(monkeypatch, vmapped):
    torch.manual_seed(2)
    hidden = torch.randn(2, 9, 8)
    weight = torch.randn(32, 8)
    labels = torch.randint(0, 32, (2, 9))
    calls = 0
    original_addmm = torch.addmm

    def recording_addmm(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_addmm(*args, **kwargs)

    monkeypatch.setattr(mod, "_workspace_budget_bytes", lambda _device: 1024)
    monkeypatch.setattr(torch, "addmm", recording_addmm)

    def loss(h, w, lab):
        return linear_cross_entropy_chunked(h, w, lab, chunk_vocab=8)

    if vmapped:
        vmap(grad(loss, (0, 1)), in_dims=(0, None, 0))(hidden, weight, labels)
    else:
        hidden.requires_grad_(True)
        weight.requires_grad_(True)
        torch.autograd.grad(loss(hidden, weight, labels), (hidden, weight))

    assert calls > 0


def test_chunked_bf16_projection_staging_matches_forward_and_backward(monkeypatch):
    hidden = torch.randn(1, 9, 8, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(32, 8, dtype=torch.bfloat16)
    labels = torch.randint(0, 32, (1, 9))
    operand_dtypes = []
    original_linear_chunk = mod._linear_chunk

    def recording_linear_chunk(e, w):
        operand_dtypes.append((e.dtype, w.dtype))
        return original_linear_chunk(e, w)

    monkeypatch.setattr(mod, "_workspace_budget_bytes", lambda _device: 1024)
    monkeypatch.setattr(mod, "_linear_chunk", recording_linear_chunk)
    loss = linear_cross_entropy_chunked(hidden, weight, labels, chunk_vocab=8)
    torch.autograd.grad(loss, hidden)

    assert operand_dtypes
    assert set(operand_dtypes) == {(torch.bfloat16, torch.bfloat16)}


def test_chunked_bf16_backward_is_transform_safe_with_two_dimensional_tiles(
    monkeypatch,
):
    torch.manual_seed(3)
    hidden = torch.randn(2, 9, 8, dtype=torch.bfloat16)
    weight = torch.randn(32, 8, dtype=torch.bfloat16)
    labels = torch.randint(0, 32, (2, 9))
    monkeypatch.setattr(mod, "_workspace_budget_bytes", lambda _device: 1024)

    def kernel(h, w, lab):
        return linear_cross_entropy_chunked(h, w, lab, chunk_vocab=8)

    actual = vmap(grad(kernel, (0, 1)), in_dims=(0, None, 0))(hidden, weight, labels)
    expected = vmap(grad(_eager_mean, (0, 1)), in_dims=(0, None, 0))(
        hidden, weight, labels
    )
    torch.testing.assert_close(actual[0], expected[0], rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual[1], expected[1], rtol=2e-2, atol=2e-2)


def test_chunked_linear_ce_handles_no_prediction_tokens():
    hidden = torch.randn(2, 1, 4)
    weight = torch.randn(8, 4)
    labels = torch.randint(0, 8, (2, 1))

    def loss(h, w, lab):
        return linear_cross_entropy_chunked(h, w, lab, chunk_vocab=4)

    actual = vmap(grad(loss, (0, 1)), in_dims=(0, None, 0))(hidden, weight, labels)
    assert torch.count_nonzero(actual[0]) == 0
    assert torch.count_nonzero(actual[1]) == 0


@pytest.mark.mps
def test_chunked_linear_ce_parity_mps():
    _check_parity("mps")


def _check_bf16_streams_fp32(device: str) -> None:
    """BF16 linear projections are promoted before FP32 CE statistics."""
    torch.manual_seed(0)
    h = torch.randn(2, 8, 16, device=device, dtype=torch.bfloat16)
    w = (torch.randn(4096, 16, device=device) * 0.1).bfloat16()
    lab = torch.randint(0, 4096, (2, 8), device=device)
    loss = linear_cross_entropy_chunked(h, w, lab)
    assert loss.dtype == torch.float32
    logits = (h[..., :-1, :] @ w.t()).float()
    targets = lab[..., 1:]
    ref = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
    )
    assert (loss - ref).abs().item() < 1e-3, (loss.item(), ref.item())


def test_chunked_linear_ce_bf16_streams_fp32_cpu():
    _check_bf16_streams_fp32("cpu")


@pytest.mark.mps
def test_chunked_linear_ce_bf16_streams_fp32_mps():
    _check_bf16_streams_fp32("mps")


def test_chunked_linear_ce_matches_eager_under_bare_grad():
    """``torch.func.grad`` alone, not only ``vmap(grad(...))``.

    A bare ``grad`` re-enters the kernel through
    ``torch._functorch.autograd_function.custom_function_call_grad``, which
    generates a single-level Function whose ``forward`` calls back into
    ``custom_function_call`` and hands the result to ``setup_context`` as
    ``output``.  Projecting the forward's three outputs down to the loss inside
    ``apply`` therefore made that outer ``setup_context`` try to unpack the loss
    tensor itself; the projection belongs at the call site.  This pins the whole
    path against eager, which the ``vmap(grad(...))`` tests do not reach.
    """
    torch.manual_seed(0)
    n_tokens, hidden, vocab = 5, 8, 16
    h = torch.randn(1, n_tokens + 1, hidden, dtype=torch.float64)
    weight = torch.randn(vocab, hidden, dtype=torch.float64)
    labels = torch.randint(0, vocab, (1, n_tokens + 1))

    got = grad(lambda w: linear_cross_entropy_chunked(h, w, labels))(weight)
    want = grad(lambda w: _eager_mean(h, w, labels))(weight)
    assert torch.allclose(got, want, atol=1e-12, rtol=0)

    # The loss value itself, on the same path.
    assert torch.allclose(
        linear_cross_entropy_chunked(h, weight, labels),
        _eager_mean(h, weight, labels),
        atol=1e-12,
        rtol=0,
    )
