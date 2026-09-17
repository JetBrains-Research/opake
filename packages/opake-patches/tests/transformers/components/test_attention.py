# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Test different attention implementations with vmap/clipped_grad.

Tests eager and SDPA attention implementations that are compatible with vmap,
including microbatching support and numerical parity.

Known incompatibilities (not tested):
- Flash Attention 2: Uses torch.nonzero which has dynamic output shape
- flex_attention: HigherOrderOperator has no vmap support
"""

import pytest
import torch
from opake_test_support import prepare_lora_model, run_clipped_grad_test

from opake.api.engine.clipping import clipped_grad
from opake.api.patches.transformers.components import attention as attention_components
from opake.api.patches.transformers.components.attention import (
    vmap_eager_attention_forward,
    vmap_eager_attention_forward_gemma2,
    vmap_sdpa_attention_forward,
    vmap_sdpa_attention_forward_gemma2,
    vmap_sdpa_attention_forward_sliding_window,
)
from opake.functional import make_functional


class _Gemma2Attention(torch.nn.Module):
    num_key_value_groups = 1
    is_causal = False


def _gemma2_softcap_inputs():
    query = torch.tensor([[[[0.25, 0.0], [1.0, 0.0], [4.0, 0.0]]]])
    key = torch.tensor([[[[1.0, 0.0], [-1.0, 0.0], [3.0, 0.0]]]])
    value = torch.tensor([[[[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]]]])
    return query, key, value


def _expected_gemma2_softcap_attention(
    query,
    key,
    value,
    softcap,
    *,
    scaling=1.0,
    attention_mask=None,
    is_causal=False,
):
    scores = scaling * (query @ key.transpose(-2, -1))
    scores = softcap * torch.tanh(scores / softcap)
    probability_mask = None
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            probability_mask = attention_mask
            scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)
        else:
            scores = scores + attention_mask
    elif is_causal:
        probability_mask = torch.ones(
            query.shape[-2], key.shape[-2], dtype=torch.bool, device=query.device
        ).tril()
        scores = scores.masked_fill(~probability_mask, -torch.inf)
    weights = torch.softmax(scores, dim=-1)
    if probability_mask is not None:
        weights = weights.masked_fill(~probability_mask, 0.0)
    return weights @ value, weights


def _assert_gemma2_softcap_attention(attention, *, returns_weights):
    query, key, value = (tensor.requires_grad_() for tensor in _gemma2_softcap_inputs())
    output, weights = attention(
        _Gemma2Attention(), query, key, value, None, scaling=1.0, softcap=1.0
    )
    ref_query, ref_key, ref_value = (
        tensor.requires_grad_() for tensor in _gemma2_softcap_inputs()
    )
    expected_output, expected_weights = _expected_gemma2_softcap_attention(
        ref_query, ref_key, ref_value, 1.0
    )

    torch.testing.assert_close(output, expected_output.transpose(-3, -2))
    if returns_weights:
        torch.testing.assert_close(weights, expected_weights)
    else:
        assert weights is None
    actual_grads = torch.autograd.grad(output.square().sum(), (query, key, value))
    expected_grads = torch.autograd.grad(
        expected_output.square().sum(), (ref_query, ref_key, ref_value)
    )
    for actual, expected in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("mask_kind", ["none", "boolean", "additive", "per_head"])
def test_gqa_sdpa_uses_natural_kv_storage_and_matches_reference(monkeypatch, mask_kind):
    torch.manual_seed(0)
    query = torch.randn(1, 4, 5, 3, requires_grad=True)
    key = torch.randn(1, 2, 5, 3, requires_grad=True)
    value = torch.randn(1, 2, 5, 3, requires_grad=True)
    if mask_kind == "boolean":
        attention_mask = torch.ones(1, 1, 5, 5, dtype=torch.bool).tril_()
    elif mask_kind == "additive":
        attention_mask = torch.randn(1, 1, 5, 5)
    elif mask_kind == "per_head":
        attention_mask = torch.randn(1, 4, 5, 5)
    else:
        attention_mask = None
    module = _Gemma2Attention()
    module.num_key_value_groups = 2

    seen_kv = []
    real_sdpa = torch.nn.functional.scaled_dot_product_attention

    def record_sdpa(q, k, v, **kwargs):
        seen_kv.append((k, v, kwargs))
        return real_sdpa(q, k, v, **kwargs)

    monkeypatch.setattr(
        attention_components,
        "vmap_repeat_kv",
        lambda *args, **kwargs: pytest.fail("GQA must not call repeat_kv"),
    )
    monkeypatch.setattr(
        torch.nn.functional, "scaled_dot_product_attention", record_sdpa
    )
    output, weights = vmap_sdpa_attention_forward(
        module, query, key, value, attention_mask, scaling=0.5
    )

    expected = real_sdpa(
        query,
        key.repeat_interleave(2, dim=-3),
        value.repeat_interleave(2, dim=-3),
        attn_mask=attention_mask,
        scale=0.5,
    ).transpose(-3, -2)
    torch.testing.assert_close(output, expected)
    assert weights is None
    assert len(seen_kv) == key.shape[-3]
    for seen_key, seen_value, kwargs in seen_kv:
        assert seen_key.untyped_storage().data_ptr() == key.untyped_storage().data_ptr()
        assert (
            seen_value.untyped_storage().data_ptr()
            == value.untyped_storage().data_ptr()
        )
        assert seen_key.stride(-3) == 0
        assert seen_value.stride(-3) == 0
        assert "enable_gqa" not in kwargs

    actual_grads = torch.autograd.grad(output.square().sum(), (query, key, value))
    expected_grads = torch.autograd.grad(expected.square().sum(), (query, key, value))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_gqa_sdpa_uses_native_gqa_when_backend_supports_it(monkeypatch):
    query = torch.randn(1, 4, 5, 3)
    key = torch.randn(1, 2, 5, 3)
    value = torch.randn(1, 2, 5, 3)
    module = _Gemma2Attention()
    module.num_key_value_groups = 2
    calls = []
    real_sdpa = torch.nn.functional.scaled_dot_product_attention

    def record_sdpa(q, k, v, **kwargs):
        calls.append((k, v, kwargs))
        return real_sdpa(q, k, v, **kwargs)

    monkeypatch.setattr(attention_components, "_can_use_native_gqa", lambda *args: True)
    monkeypatch.setattr(
        torch.nn.functional, "scaled_dot_product_attention", record_sdpa
    )

    output, _ = vmap_sdpa_attention_forward(
        module, query, key, value, None, scaling=0.5
    )
    expected = real_sdpa(query, key, value, scale=0.5, enable_gqa=True).transpose(
        -3, -2
    )

    torch.testing.assert_close(output, expected)
    assert len(calls) == 1
    seen_key, seen_value, kwargs = calls[0]
    assert seen_key is key
    assert seen_value is value
    assert kwargs["enable_gqa"] is True


@pytest.mark.cuda
def test_native_gqa_eligibility_uses_gqa_backend_parameters():
    device = torch.device("cuda")
    query = torch.randn(1, 4, 128, 64, device=device, dtype=torch.bfloat16)
    key = torch.randn(1, 2, 128, 64, device=device, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    params = torch.backends.cuda.SDPAParams(query, key, value, None, 0.0, False, True)
    if not torch.backends.cuda.can_use_flash_attention(params):
        pytest.skip("This CUDA device cannot run native Flash GQA")

    assert attention_components._can_use_native_gqa(query, key, value, None, 0.0, False)


def test_gqa_sdpa_supports_vmap_grad():
    torch.manual_seed(0)
    query = torch.randn(3, 4, 5, 3)
    key = torch.randn(3, 2, 5, 3)
    value = torch.randn(3, 2, 5, 3)
    module = _Gemma2Attention()
    module.num_key_value_groups = 2

    def loss(q, k, v):
        output, _ = vmap_sdpa_attention_forward(module, q, k, v, None, scaling=0.5)
        return output.square().sum()

    def reference_loss(q, k, v):
        output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k.repeat_interleave(2, dim=-3),
            v.repeat_interleave(2, dim=-3),
            scale=0.5,
        )
        return output.square().sum()

    actual = torch.vmap(torch.func.grad(loss, argnums=(0, 1, 2)))(query, key, value)
    expected = torch.vmap(torch.func.grad(reference_loss, argnums=(0, 1, 2)))(
        query, key, value
    )
    for actual_grad, expected_grad in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_gqa_eager_attention_matches_reference_without_repeat_kv(monkeypatch):
    torch.manual_seed(0)
    query = torch.randn(1, 4, 5, 3, requires_grad=True)
    key = torch.randn(1, 2, 5, 3, requires_grad=True)
    value = torch.randn(1, 2, 5, 3, requires_grad=True)
    attention_mask = torch.ones(1, 1, 5, 5, dtype=torch.bool).tril_()
    module = _Gemma2Attention()
    module.num_key_value_groups = 2
    module.train()

    monkeypatch.setattr(
        attention_components,
        "vmap_repeat_kv",
        lambda *args, **kwargs: pytest.fail("GQA must not call repeat_kv"),
    )
    output, weights = vmap_eager_attention_forward(
        module, query, key, value, attention_mask, scaling=0.5
    )

    repeated_key = key.repeat_interleave(2, dim=-3)
    repeated_value = value.repeat_interleave(2, dim=-3)
    expected_weights = torch.softmax(
        (query @ repeated_key.transpose(-2, -1) * 0.5).masked_fill(
            ~attention_mask, torch.finfo(query.dtype).min
        ),
        dim=-1,
    )
    expected = (expected_weights @ repeated_value).transpose(-3, -2)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(weights, expected_weights)

    actual_grads = torch.autograd.grad(output.square().sum(), (query, key, value))
    expected_grads = torch.autograd.grad(expected.square().sum(), (query, key, value))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_sliding_window_sdpa_matches_dense_reference_without_full_mask(monkeypatch):
    query = torch.randn(1, 4, 5, 3, requires_grad=True)
    key = torch.randn(1, 2, 5, 3, requires_grad=True)
    value = torch.randn(1, 2, 5, 3, requires_grad=True)
    sliding_window = 3
    dense_mask = torch.ones((5, 5), dtype=torch.bool).tril_()
    dense_mask.triu_(diagonal=1 - sliding_window)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query,
        key.repeat_interleave(2, dim=-3),
        value.repeat_interleave(2, dim=-3),
        attn_mask=dense_mask,
        scale=0.7,
    )
    mask_shapes = []
    real_sdpa = torch.nn.functional.scaled_dot_product_attention

    def record_sdpa(*args, **kwargs):
        mask_shapes.append(kwargs["attn_mask"].shape)
        return real_sdpa(*args, **kwargs)

    monkeypatch.setattr(attention_components, "_SDPA_QUERY_CHUNK", 2)
    monkeypatch.setattr(
        torch.nn.functional, "scaled_dot_product_attention", record_sdpa
    )

    output, weights = vmap_sdpa_attention_forward_sliding_window(
        _Gemma2Attention(),
        query,
        key,
        value,
        None,
        scaling=0.7,
        sliding_window=sliding_window,
    )

    torch.testing.assert_close(output, expected.transpose(-3, -2))
    assert weights is None
    assert max(query_size * key_size for query_size, key_size in mask_shapes) < 25

    actual_grads = torch.autograd.grad(output.square().sum(), (query, key, value))
    expected_grads = torch.autograd.grad(expected.square().sum(), (query, key, value))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_gemma2_softcap_sdpa_applies_sliding_window_without_dense_mask(monkeypatch):
    monkeypatch.setattr(attention_components, "_GEMMA2_QUERY_CHUNK", 2)
    module = _Gemma2Attention()
    module.is_causal = True
    query = torch.randn(1, 1, 5, 2, requires_grad=True)
    key = torch.randn(1, 1, 5, 2, requires_grad=True)
    value = torch.randn(1, 1, 5, 2, requires_grad=True)
    mask = torch.ones((5, 5), dtype=torch.bool).tril_()
    mask.triu_(diagonal=-2)
    expected, _ = _expected_gemma2_softcap_attention(
        query, key, value, 1.0, attention_mask=mask
    )

    output, weights = vmap_sdpa_attention_forward_gemma2(
        module,
        query,
        key,
        value,
        None,
        scaling=1.0,
        softcap=1.0,
        sliding_window=3,
    )

    torch.testing.assert_close(output, expected.transpose(-3, -2))
    assert weights is None
    actual_grads = torch.autograd.grad(output.square().sum(), (query, key, value))
    expected_grads = torch.autograd.grad(expected.square().sum(), (query, key, value))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_gemma2_softcap_attention_matches_reference(monkeypatch):
    """Keep scaled scores near and far beyond the cap before softmax."""
    _assert_gemma2_softcap_attention(
        vmap_eager_attention_forward_gemma2, returns_weights=True
    )
    monkeypatch.setattr(attention_components, "_GEMMA2_QUERY_CHUNK", 2)
    _assert_gemma2_softcap_attention(
        vmap_sdpa_attention_forward_gemma2, returns_weights=False
    )


def test_gemma2_softcap_attention_accepts_boolean_sdpa_mask():
    query, key, value = (tensor.requires_grad_() for tensor in _gemma2_softcap_inputs())
    mask = torch.ones((1, 1, 3, 3), dtype=torch.bool).tril_()
    mask[..., 0, :] = False

    output, weights = vmap_sdpa_attention_forward_gemma2(
        _Gemma2Attention(), query, key, value, mask, scaling=1.0, softcap=1.0
    )

    expected_output, _ = _expected_gemma2_softcap_attention(
        query, key, value, 1.0, attention_mask=mask
    )
    torch.testing.assert_close(output, expected_output.transpose(-3, -2))
    assert weights is None
    actual_grads = torch.autograd.grad(output.square().sum(), (query, key, value))
    expected_grads = torch.autograd.grad(
        expected_output.square().sum(), (query, key, value)
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_gemma2_softcap_attention_broadcasts_additive_mask_across_chunks(
    monkeypatch,
):
    monkeypatch.setattr(attention_components, "_GEMMA2_QUERY_CHUNK", 2)
    query, key, value = (tensor.requires_grad_() for tensor in _gemma2_softcap_inputs())
    attention_mask = torch.tensor([[[[0.0, -1.0, -2.0]]]])

    output, weights = vmap_sdpa_attention_forward_gemma2(
        _Gemma2Attention(),
        query,
        key,
        value,
        attention_mask,
        scaling=0.5,
        softcap=1.0,
    )

    reference_scores = 0.5 * (query @ key.transpose(-2, -1))
    reference_scores = torch.tanh(reference_scores) + attention_mask
    reference_weights = torch.softmax(reference_scores, dim=-1)
    reference_output = reference_weights @ value
    torch.testing.assert_close(output, reference_output.transpose(-3, -2))
    assert weights is None

    actual_grads = torch.autograd.grad(output.square().sum(), (query, key, value))
    expected_grads = torch.autograd.grad(
        reference_output.square().sum(), (query, key, value)
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_gemma2_softcap_sdpa_preserves_implicit_causality_and_default_scale(
    monkeypatch,
):
    monkeypatch.setattr(attention_components, "_GEMMA2_QUERY_CHUNK", 2)
    module = _Gemma2Attention()
    module.is_causal = True
    query, key, value = (tensor.requires_grad_() for tensor in _gemma2_softcap_inputs())

    output, weights = vmap_sdpa_attention_forward_gemma2(
        module, query, key, value, None, scaling=None, softcap=1.0
    )

    ref_query, ref_key, ref_value = (
        tensor.requires_grad_() for tensor in _gemma2_softcap_inputs()
    )
    expected_output, _ = _expected_gemma2_softcap_attention(
        ref_query,
        ref_key,
        ref_value,
        1.0,
        scaling=query.shape[-1] ** -0.5,
        is_causal=True,
    )
    torch.testing.assert_close(output, expected_output.transpose(-3, -2))
    assert weights is None

    actual_grads = torch.autograd.grad(output.square().sum(), (query, key, value))
    expected_grads = torch.autograd.grad(
        expected_output.square().sum(), (ref_query, ref_key, ref_value)
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_gemma2_softcap_sdpa_decode_attends_to_all_cached_keys():
    module = _Gemma2Attention()
    module.is_causal = True
    full_query, key, value = _gemma2_softcap_inputs()
    query = full_query[..., :1, :]

    output, _ = vmap_sdpa_attention_forward_gemma2(
        module, query, key, value, None, scaling=1.0, softcap=1.0
    )
    expected_output, _ = _expected_gemma2_softcap_attention(query, key, value, 1.0)

    torch.testing.assert_close(output, expected_output.transpose(-3, -2))


def test_gemma2_softcap_sdpa_crops_unused_causal_keys():
    module = _Gemma2Attention()
    module.is_causal = True
    full_query, key, value = _gemma2_softcap_inputs()
    query = full_query[..., :2, :]

    output, _ = vmap_sdpa_attention_forward_gemma2(
        module, query, key, value, None, scaling=1.0, softcap=1.0
    )
    expected_output, _ = _expected_gemma2_softcap_attention(
        query, key[..., :2, :], value[..., :2, :], 1.0, is_causal=True
    )

    torch.testing.assert_close(output, expected_output.transpose(-3, -2))


def test_gemma2_softcap_sdpa_preserves_grouped_query_attention(monkeypatch):
    monkeypatch.setattr(attention_components, "_GEMMA2_QUERY_CHUNK", 2)
    torch.manual_seed(0)
    query = torch.randn(1, 4, 4, 3, requires_grad=True)
    key = torch.randn(1, 2, 4, 3, requires_grad=True)
    value = torch.randn(1, 2, 4, 3, requires_grad=True)
    attention_mask = torch.ones(1, 1, 4, 4, dtype=torch.bool).tril_()
    module = _Gemma2Attention()
    module.num_key_value_groups = 2

    monkeypatch.setattr(
        attention_components,
        "vmap_repeat_kv",
        lambda *args, **kwargs: pytest.fail("GQA must not call repeat_kv"),
    )
    saved_tensors = []

    def record_saved(tensor):
        saved_tensors.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(record_saved, lambda tensor: tensor):
        output, _ = vmap_sdpa_attention_forward_gemma2(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling=0.5,
            softcap=1.5,
        )

    assert any(tensor is key for tensor in saved_tensors)
    assert any(tensor is value for tensor in saved_tensors)

    ref_query = query.detach().requires_grad_()
    ref_key = key.detach().requires_grad_()
    ref_value = value.detach().requires_grad_()
    expected_output, _ = _expected_gemma2_softcap_attention(
        ref_query,
        ref_key.repeat_interleave(2, dim=-3),
        ref_value.repeat_interleave(2, dim=-3),
        1.5,
        scaling=0.5,
        attention_mask=attention_mask,
    )
    torch.testing.assert_close(output, expected_output.transpose(-3, -2))

    actual_grads = torch.autograd.grad(output.square().sum(), (query, key, value))
    expected_grads = torch.autograd.grad(
        expected_output.square().sum(), (ref_query, ref_key, ref_value)
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_gemma2_softcap_sdpa_keeps_nonzero_training_dropout_fallback(monkeypatch):
    eager_calls = []
    eager_attention = attention_components.vmap_eager_attention_forward_gemma2

    def record_eager(*args, **kwargs):
        eager_calls.append((args[4], kwargs["dropout"], kwargs["scaling"]))
        return eager_attention(*args, **kwargs)

    monkeypatch.setattr(
        attention_components, "vmap_eager_attention_forward_gemma2", record_eager
    )
    module = _Gemma2Attention()
    module.is_causal = True
    query, key, value = _gemma2_softcap_inputs()

    module.eval()
    vmap_sdpa_attention_forward_gemma2(
        module, query, key, value, None, dropout=0.25, softcap=1.0
    )
    assert eager_calls == []

    module.train()
    vmap_sdpa_attention_forward_gemma2(
        module, query, key, value, None, dropout=0.25, softcap=1.0
    )
    causal_mask, dropout, scaling = eager_calls[0]
    torch.testing.assert_close(causal_mask, torch.ones(3, 3, dtype=torch.bool).tril())
    assert dropout == 0.25
    assert scaling == query.shape[-1] ** -0.5


def test_gemma2_softcap_sdpa_supports_vmap_grad(monkeypatch):
    monkeypatch.setattr(attention_components, "_GEMMA2_QUERY_CHUNK", 2)
    torch.manual_seed(0)
    query = torch.randn(2, 4, 5, 3)
    key = torch.randn(2, 2, 5, 3)
    value = torch.randn(2, 2, 5, 3)
    module = _Gemma2Attention()
    module.num_key_value_groups = 2
    module.is_causal = True

    def loss(q, k, v):
        output, _ = vmap_sdpa_attention_forward_gemma2(
            module, q, k, v, None, scaling=1.0, softcap=1.0
        )
        return output.square().sum()

    def reference_loss(q, k, v):
        output, _ = _expected_gemma2_softcap_attention(
            q,
            k.repeat_interleave(2, dim=-3),
            v.repeat_interleave(2, dim=-3),
            1.0,
            is_causal=True,
        )
        return output.square().sum()

    actual = torch.vmap(torch.func.grad(loss, argnums=(0, 1, 2)))(query, key, value)
    expected = torch.vmap(torch.func.grad(reference_loss, argnums=(0, 1, 2)))(
        query, key, value
    )

    for actual_grad, expected_grad in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gemma2_softcap_sdpa_bounds_forward_and_backward_attention_tiles(
    monkeypatch,
    dtype,
):
    query_length = 1025
    key_length = 1537
    chunk_size = 64
    query = torch.randn(1, 2, query_length, 8, dtype=dtype, requires_grad=True)
    key = torch.randn(1, 2, key_length, 8, dtype=dtype, requires_grad=True)
    value = torch.randn(1, 2, key_length, 8, dtype=dtype, requires_grad=True)
    matmul_shapes = []
    real_matmul = torch.matmul

    def record_matmul(left, right):
        result = real_matmul(left, right)
        matmul_shapes.append(result.shape)
        return result

    monkeypatch.setattr(attention_components, "_GEMMA2_QUERY_CHUNK", chunk_size)
    monkeypatch.setattr(torch, "matmul", record_matmul)

    output, weights = vmap_sdpa_attention_forward_gemma2(
        _Gemma2Attention(), query, key, value, None, scaling=1.0, softcap=1.0
    )
    output.float().square().sum().backward()

    assert output.shape == (1, query_length, 2, 8)
    assert weights is None
    score_shapes = [shape for shape in matmul_shapes if shape[-1] == key_length]
    assert score_shapes
    assert max(shape[-2] for shape in score_shapes) <= chunk_size
    assert not any(shape[-2:] == (query_length, key_length) for shape in matmul_shapes)


def _cuda_peak_delta(fn):
    import gc

    result = fn()
    torch.cuda.synchronize()
    del result
    gc.collect()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    del result
    gc.collect()
    torch.cuda.empty_cache()
    return peak


@pytest.mark.cuda
def test_gqa_sdpa_reduces_cuda_peak_memory():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    sequence_length = 2048
    query_heads = 32
    key_value_heads = 8
    head_dim = 128
    query = torch.randn(
        1,
        query_heads,
        sequence_length,
        head_dim,
        device=device,
        dtype=dtype,
    )
    key = torch.randn(
        1,
        key_value_heads,
        sequence_length,
        head_dim,
        device=device,
        dtype=dtype,
    )
    value = torch.randn_like(key)
    module = _Gemma2Attention().eval()
    module.num_key_value_groups = query_heads // key_value_heads

    def grouped():
        return vmap_sdpa_attention_forward(
            module, query, key, value, None, scaling=None
        )

    def repeated():
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key.repeat_interleave(module.num_key_value_groups, dim=-3),
            value.repeat_interleave(module.num_key_value_groups, dim=-3),
        )

    grouped_peak = _cuda_peak_delta(grouped)
    repeated_peak = _cuda_peak_delta(repeated)
    eliminated_kv_bytes = (
        2
        * (query_heads - key_value_heads)
        * sequence_length
        * head_dim
        * query.element_size()
    )

    assert repeated_peak - grouped_peak > eliminated_kv_bytes * 0.5, (
        f"Expected at least half the expanded K/V allocation to disappear; "
        f"grouped={grouped_peak / 2**20:.1f} MiB, "
        f"repeated={repeated_peak / 2**20:.1f} MiB"
    )


@pytest.mark.cuda
@pytest.mark.parametrize("sequence_length", [1024, 4096])
def test_gemma2_softcap_sdpa_reduces_cuda_peak_memory(sequence_length):
    device = torch.device("cuda")
    dtype = torch.float16
    query = torch.randn(1, 2, sequence_length, 32, device=device, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    module = _Gemma2Attention().eval()

    def chunked():
        return vmap_sdpa_attention_forward_gemma2(
            module, query, key, value, None, scaling=None, softcap=50.0
        )

    def eager():
        return vmap_eager_attention_forward_gemma2(
            module, query, key, value, None, scaling=None, softcap=50.0
        )

    chunked_peak = _cuda_peak_delta(chunked)
    eager_peak = _cuda_peak_delta(eager)

    assert chunked_peak < eager_peak * 0.5, (
        f"Expected chunked attention to use less than half the eager peak; "
        f"chunked={chunked_peak / 2**20:.1f} MiB, "
        f"eager={eager_peak / 2**20:.1f} MiB"
    )


class TestAttentionImplementations:
    """Test different attention implementations work with clipped_grad."""

    @pytest.mark.slow
    def test_eager_attention(self, qwen2_config, qwen2_tokenizer, device):
        """Test eager attention (explicitly patched). Works on CPU and CUDA."""
        qwen2_config._attn_implementation = "eager"
        model = prepare_lora_model(qwen2_config, apply_patches=True).to(device)
        grads, _ = run_clipped_grad_test(model, qwen2_tokenizer, apply_patches=False)
        assert len(grads.pytree) > 0

    def test_sdpa_attention(self, qwen2_config, qwen2_tokenizer, device):
        """Test SDPA attention (default, uses patched repeat_kv). Works on CPU and CUDA."""
        qwen2_config._attn_implementation = "sdpa"
        model = prepare_lora_model(qwen2_config, apply_patches=True).to(device)
        grads, _ = run_clipped_grad_test(model, qwen2_tokenizer, apply_patches=False)
        assert len(grads.pytree) > 0


class TestAttentionWithMicrobatching:
    """Test attention implementations work with microbatching."""

    def _run_with_microbatch(self, config, tokenizer, device, microbatch_size=2):
        """Helper to run clipped_grad with microbatching."""
        model = prepare_lora_model(config, apply_patches=True).to(device)

        texts = ["Hello world test", "Another example", "Third sample", "Fourth one"]
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, max_length=16, truncation=True
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        labels = input_ids.clone()

        fmodel, trainable, frozen = make_functional(
            model, disable_autograd_tracking=True, partition_trainable=True
        )

        def per_example_loss(trainable_params, frozen_params, ids, mask, lbls):
            all_params = {**frozen_params, **trainable_params}
            outputs = fmodel(all_params, ids, attention_mask=mask, labels=lbls)
            return outputs.loss

        grad_fn, clip_state = clipped_grad(
            per_example_loss,
            argnums=0,
            batch_argnums=(2, 3, 4),
            clipping_norm=1.0,
            microbatch_size=microbatch_size,
        )
        grads, _state = grad_fn(
            trainable, frozen, input_ids, attention_mask, labels, state=clip_state
        )
        return grads

    def test_eager_with_microbatching(self, qwen2_config, qwen2_tokenizer, device):
        """Test eager attention with microbatching."""
        qwen2_config._attn_implementation = "eager"
        grads = self._run_with_microbatch(qwen2_config, qwen2_tokenizer, device)
        assert len(grads.pytree) > 0

    def test_sdpa_with_microbatching(self, qwen2_config, qwen2_tokenizer, device):
        """Test SDPA attention with microbatching."""
        qwen2_config._attn_implementation = "sdpa"
        grads = self._run_with_microbatch(qwen2_config, qwen2_tokenizer, device)
        assert len(grads.pytree) > 0

    def test_sdpa_microbatch_size_3(self, qwen2_config, qwen2_tokenizer, device):
        """Test SDPA with microbatch_size=3 (uneven split of batch=4)."""
        qwen2_config._attn_implementation = "sdpa"
        grads = self._run_with_microbatch(
            qwen2_config, qwen2_tokenizer, device, microbatch_size=3
        )
        assert len(grads.pytree) > 0


class TestAttentionNumericalParity:
    """Test that SDPA and eager produce similar gradients."""

    @pytest.mark.slow
    def test_sdpa_eager_gradient_parity(self, qwen2_config, qwen2_tokenizer, device):
        """Verify SDPA and eager produce numerically similar clipped gradients.

        Pins SDPA to the ``MATH`` backend so both runs use identical matmul
        order. Without the pin, SDPA picks flash / efficient / math kernels
        based on input shape, device, and dtype — different kernels produce
        different rounding and the comparison becomes flaky at the
        ``rtol=0.2`` tolerance below.
        """
        from torch.nn.attention import SDPBackend, sdpa_kernel

        # Deterministic init: LoRA and any op that reads torch's default RNG
        # need a fixed seed for a run-to-run-stable comparison.
        torch.manual_seed(0)

        # Run with eager
        qwen2_config._attn_implementation = "eager"
        model_eager = prepare_lora_model(qwen2_config, apply_patches=True).to(device)
        grads_eager, _ = run_clipped_grad_test(
            model_eager, qwen2_tokenizer, apply_patches=False
        )

        # Run with SDPA (fresh model with same weights — state is copied below)
        torch.manual_seed(0)
        qwen2_config._attn_implementation = "sdpa"
        model_sdpa = prepare_lora_model(qwen2_config, apply_patches=True).to(device)

        # Copy weights from eager model to SDPA model for fair comparison
        sdpa_state = model_sdpa.state_dict()
        eager_state = model_eager.state_dict()
        for key in sdpa_state:
            if key in eager_state:
                sdpa_state[key] = eager_state[key]
        model_sdpa.load_state_dict(sdpa_state)

        # Pin SDPA to the MATH backend so the backward pass uses the same
        # deterministic reference path eager does (rather than a flash /
        # efficient kernel with different FP rounding).
        with sdpa_kernel(SDPBackend.MATH):
            grads_sdpa, _ = run_clipped_grad_test(
                model_sdpa, qwen2_tokenizer, apply_patches=False
            )

        # Compare gradients - allow for numerical differences between backends.
        # Eager uses manual Q@K matmul; SDPA uses fused CUDA kernels (flash/efficient).
        # These use different algorithms with different floating-point rounding,
        # so we only check that gradients are in the same ballpark.
        assert set(grads_eager.pytree.keys()) == set(grads_sdpa.pytree.keys())
        for key in grads_eager.pytree:
            eager_grad = grads_eager.pytree[key]
            sdpa_grad = grads_sdpa.pytree[key]
            # Wide tolerance: eager uses manual matmul, SDPA uses fused flash/efficient
            # kernels with different FP rounding. Near-zero values can differ by ~2e-3.
            assert torch.allclose(eager_grad, sdpa_grad, rtol=0.2, atol=5e-3), (
                f"Gradient mismatch for {key}: "
                f"max_diff={torch.max(torch.abs(eager_grad - sdpa_grad)).item():.4e}"
            )
