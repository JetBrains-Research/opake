# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Autocast correctness and weight-replica lifetime for fused kernels."""

from __future__ import annotations

import gc

import pytest
import torch
import torch.nn.functional as F
from torch.func import grad, vmap

pytest.importorskip("triton")

from opaque.api.patches.kernels._moe_expert_cast import (
    clear_expert_shadow_cache,
    expert_shadow_cache_info,
)
from opaque.api.patches.kernels._utils import cast_to_dtype
from opaque.api.patches.kernels.fused_moe import opaque_fused_moe
from opaque.api.patches.kernels.lora import (
    opaque_lora_mlp,
    opaque_lora_qkv,
    opaque_lora_w,
)
from opaque.api.patches.kernels.moe import opaque_moe

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

_AMP_DTYPES = (torch.float16, torch.bfloat16)


def setup_function():
    clear_expert_shadow_cache()


def _clone_args(args, dtype):
    cloned = []
    for arg in args:
        if not isinstance(arg, torch.Tensor):
            cloned.append(arg)
            continue
        target_dtype = dtype if arg.is_floating_point() else arg.dtype
        cloned.append(
            arg.detach().to(target_dtype).clone().requires_grad_(arg.requires_grad)
        )
    return tuple(cloned)


def _outputs_and_loss(call, args):
    result = call(*args)
    outputs = result if isinstance(result, tuple) else (result,)
    loss = sum(output.float().square().mean() for output in outputs)
    return outputs, loss


def _input_grads(loss, args):
    grad_inputs = tuple(
        arg for arg in args if isinstance(arg, torch.Tensor) and arg.requires_grad
    )
    return torch.autograd.grad(loss, grad_inputs)


def _lora_case(variant: str, trainable_adapters: bool):
    hidden, intermediate, rank = 32, 48, 4
    x = torch.randn(2, 3, hidden, device="cuda", requires_grad=True)

    def base(out_features, in_features):
        return torch.randn(out_features, in_features, device="cuda") * 0.05

    def adapter(in_features, out_features):
        return (
            torch.randn(in_features, out_features, device="cuda") * 0.05
        ).requires_grad_(trainable_adapters)

    if variant == "w":
        args = (
            x,
            base(hidden, hidden),
            adapter(hidden, rank),
            adapter(rank, hidden),
            0.2,
        )
        return opaque_lora_w, args, args[1:4]

    if variant == "qkv":
        q = (base(hidden, hidden), adapter(hidden, rank), adapter(rank, hidden), 0.2)
        k = (base(hidden, hidden), adapter(hidden, rank), adapter(rank, hidden), 0.3)
        v = (base(hidden, hidden), adapter(hidden, rank), adapter(rank, hidden), 0.4)
        args = (x, *q, *k, *v)
        weights = tuple(arg for arg in args[1:] if isinstance(arg, torch.Tensor))
        return opaque_lora_qkv, args, weights

    gate = (
        base(intermediate, hidden),
        adapter(hidden, rank),
        adapter(rank, intermediate),
        0.2,
    )
    up = (
        base(intermediate, hidden),
        adapter(hidden, rank),
        adapter(rank, intermediate),
        0.3,
    )
    down = (
        base(hidden, intermediate),
        adapter(intermediate, rank),
        adapter(rank, hidden),
        0.4,
    )
    args = (x, *gate, *up, *down)
    weights = tuple(arg for arg in args[1:] if isinstance(arg, torch.Tensor))
    return opaque_lora_mlp, args, weights


@pytest.mark.parametrize("variant", ["w", "qkv", "mlp"])
@pytest.mark.parametrize("trainable_adapters", [False, True])
@pytest.mark.parametrize("amp_dtype", _AMP_DTYPES)
def test_lora_fp32_weights_match_full_cast(variant, trainable_adapters, amp_dtype):
    torch.manual_seed(0)
    call, args, _weights = _lora_case(variant, trainable_adapters)

    with torch.autocast("cuda", dtype=amp_dtype):
        outputs, loss = _outputs_and_loss(call, args)
    grads = _input_grads(loss, args)
    cast_args = _clone_args(args, amp_dtype)
    cast_outputs, cast_loss = _outputs_and_loss(call, cast_args)
    cast_grads = _input_grads(cast_loss, cast_args)

    assert all(output.dtype == amp_dtype for output in outputs)
    assert all(
        arg.dtype == torch.float32
        for arg in args
        if isinstance(arg, torch.Tensor) and arg.is_floating_point()
    )
    assert all(gradient.dtype == torch.float32 for gradient in grads)
    for actual, expected in zip(outputs, cast_outputs, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    for actual, expected in zip(grads, cast_grads, strict=True):
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=2e-2, atol=2e-3
        )


@pytest.mark.parametrize("variant", ["w", "qkv", "mlp"])
@pytest.mark.parametrize("trainable_adapters", [False, True])
def test_lora_autocast_context_saves_original_weights(variant, trainable_adapters):
    torch.manual_seed(0)
    call, args, weights = _lora_case(variant, trainable_adapters)
    saved = []

    def pack(tensor):
        saved.append(tensor)
        return tensor

    with (
        torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor),
        torch.autocast("cuda", dtype=torch.bfloat16),
    ):
        outputs = call(*args)

    saved_ptrs = {tensor.data_ptr() for tensor in saved}
    assert all(weight.data_ptr() in saved_ptrs for weight in weights)
    assert outputs is not None


@pytest.mark.parametrize("amp_dtype", _AMP_DTYPES)
def test_lora_autocast_vmap_grad_with_fp32_weights(amp_dtype):
    torch.manual_seed(0)
    batch, hidden, rank = 3, 32, 4
    x = torch.randn(batch, 2, hidden, device="cuda")
    W = torch.randn(hidden, hidden, device="cuda") * 0.05
    A = torch.randn(hidden, rank, device="cuda") * 0.05
    B = torch.randn(rank, hidden, device="cuda") * 0.05

    def loss(xi, a, b):
        with torch.autocast("cuda", dtype=amp_dtype):
            return opaque_lora_w(xi, W, a, b, 0.2).float().square().mean()

    actual = vmap(grad(loss, argnums=(0, 1, 2)), in_dims=(0, None, None))(x, A, B)
    expected = tuple(
        torch.stack(parts)
        for parts in zip(
            *(grad(loss, argnums=(0, 1, 2))(x[index], A, B) for index in range(batch)),
            strict=True,
        )
    )
    for actual_grad, expected_grad in zip(actual, expected, strict=True):
        assert actual_grad.dtype == torch.float32
        torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-2, atol=2e-3)


def _moe_args(
    trainable_experts: bool, *, hidden=32, intermediate=24, experts=8, tokens=6
):
    x = torch.randn(tokens, hidden, device="cuda", requires_grad=True)
    gate_up = (
        torch.randn(experts, 2 * intermediate, hidden, device="cuda") * 0.05
    ).requires_grad_(trainable_experts)
    down = (
        torch.randn(experts, hidden, intermediate, device="cuda") * 0.05
    ).requires_grad_(trainable_experts)
    logits = torch.randn(tokens, experts, device="cuda")
    top_k_weights, top_k_index = torch.topk(F.softmax(logits, dim=-1), 2, dim=-1)
    top_k_weights = top_k_weights.detach().requires_grad_(True)
    return x, gate_up, down, top_k_index, top_k_weights


@pytest.mark.parametrize("trainable_experts", [False, True])
@pytest.mark.parametrize("amp_dtype", _AMP_DTYPES)
def test_fused_moe_fp32_weights_match_full_cast(trainable_experts, amp_dtype):
    torch.manual_seed(0)
    args = _moe_args(trainable_experts)

    with torch.autocast("cuda", dtype=amp_dtype):
        outputs, loss = _outputs_and_loss(opaque_fused_moe, args)
    grads = _input_grads(loss, args)
    cast_args = _clone_args(args, amp_dtype)
    cast_outputs, cast_loss = _outputs_and_loss(opaque_fused_moe, cast_args)
    cast_grads = _input_grads(cast_loss, cast_args)

    assert outputs[0].dtype == amp_dtype
    assert args[1].dtype == args[2].dtype == torch.float32
    assert all(gradient.dtype == torch.float32 for gradient in grads)
    torch.testing.assert_close(outputs[0], cast_outputs[0], rtol=1e-2, atol=1e-2)
    for actual, expected in zip(grads, cast_grads, strict=True):
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=2e-2, atol=2e-2
        )


@pytest.mark.parametrize("trainable_experts", [False, True])
def test_fused_moe_autocast_context_saves_original_weights(trainable_experts):
    torch.manual_seed(0)
    args = _moe_args(trainable_experts)
    saved = []

    def pack(tensor):
        saved.append(tensor)
        return tensor

    with (
        torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor),
        torch.autocast("cuda", dtype=torch.bfloat16),
    ):
        output = opaque_moe(*args)

    saved_ptrs = {tensor.data_ptr() for tensor in saved}
    assert args[1].data_ptr() in saved_ptrs
    assert args[2].data_ptr() in saved_ptrs
    assert output is not None


@pytest.mark.parametrize("amp_dtype", _AMP_DTYPES)
def test_fused_moe_autocast_vmap_grad_with_fp32_weights(amp_dtype):
    torch.manual_seed(0)
    batch = 3
    x, gate_up, down, top_k_index, top_k_weights = _moe_args(False)
    xb = torch.randn(batch, *x.shape, device="cuda")
    index_b = top_k_index.unsqueeze(0).expand(batch, -1, -1).contiguous()
    weight_b = top_k_weights.detach().unsqueeze(0).expand(batch, -1, -1).contiguous()

    def loss(xi, index, weight):
        with torch.autocast("cuda", dtype=amp_dtype):
            return opaque_moe(xi, gate_up, down, index, weight).float().square().mean()

    actual = vmap(grad(loss, argnums=(0, 2)))(xb, index_b, weight_b)
    expected = tuple(
        torch.stack(parts)
        for parts in zip(
            *(
                grad(loss, argnums=(0, 2))(xb[index], index_b[index], weight_b[index])
                for index in range(batch)
            ),
            strict=True,
        )
    )
    for actual_grad, expected_grad in zip(actual, expected, strict=True):
        assert actual_grad.dtype == torch.float32
        torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-2, atol=2e-2)


def test_frozen_moe_shadows_follow_nested_autocast_dtype():
    torch.manual_seed(0)
    args = _moe_args(False)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        bf16 = opaque_moe(*args)
        with torch.autocast("cuda", dtype=torch.float16):
            fp16 = opaque_moe(*args)
        bf16_again = opaque_moe(*args)
    assert bf16.dtype == bf16_again.dtype == torch.bfloat16
    assert fp16.dtype == torch.float16
    assert expert_shadow_cache_info()[0] == 4


def _forward_memory(call):
    warmup = call()
    torch.cuda.synchronize()
    del warmup
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = call()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    retained = torch.cuda.memory_allocated() - baseline
    return peak, retained, output


@pytest.mark.parametrize("trainable_adapters", [False, True])
def test_lora_autocast_cast_peak_is_bounded_per_layer(trainable_adapters):
    torch.manual_seed(0)
    layer_count, hidden, rank = 4, 2048, 8
    x = torch.randn(1, hidden, device="cuda", requires_grad=True)
    layers = []
    for _ in range(layer_count):
        W = torch.randn(hidden, hidden, device="cuda") * 0.01
        A = (torch.randn(hidden, rank, device="cuda") * 0.01).requires_grad_(
            trainable_adapters
        )
        B = (torch.randn(rank, hidden, device="cuda") * 0.01).requires_grad_(
            trainable_adapters
        )
        layers.append((W, A, B))

    def forward():
        output = x
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for W, A, B in layers:
                output = opaque_lora_w(output, W, A, B, 0.2)
        return output

    replica_bytes = sum(weight.numel() * 2 for weight in layers[0])
    peak, retained, output = _forward_memory(forward)
    assert peak < 2.5 * replica_bytes
    assert retained < replica_bytes
    assert output.dtype == torch.bfloat16


@pytest.mark.parametrize("trainable_experts", [False, True])
def test_fused_moe_autocast_cast_peak_is_bounded_per_layer(trainable_experts):
    torch.manual_seed(0)
    layer_count, experts, hidden, intermediate, tokens = 4, 8, 1024, 512, 8
    x = torch.randn(tokens, hidden, device="cuda", requires_grad=True)
    layers = []
    for _ in range(layer_count):
        gate_up = (
            torch.randn(experts, 2 * intermediate, hidden, device="cuda") * 0.01
        ).requires_grad_(trainable_experts)
        down = (
            torch.randn(experts, hidden, intermediate, device="cuda") * 0.01
        ).requires_grad_(trainable_experts)
        layers.append((gate_up, down))
    logits = torch.randn(tokens, experts, device="cuda")
    top_k_weights, top_k_index = torch.topk(F.softmax(logits, dim=-1), 2, dim=-1)

    def forward():
        output = x
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for gate_up, down in layers:
                output = opaque_moe(output, gate_up, down, top_k_index, top_k_weights)
        return output

    replica_bytes = sum(weight.numel() * 2 for weight in layers[0])
    peak, retained, output = _forward_memory(forward)
    if trainable_experts:
        assert peak < 2.5 * replica_bytes
        assert retained < replica_bytes
    else:
        all_shadow_bytes = layer_count * replica_bytes
        assert peak < all_shadow_bytes + 2.5 * replica_bytes
        assert retained <= all_shadow_bytes
        assert expert_shadow_cache_info() == (2 * layer_count, all_shadow_bytes)
    assert output.dtype == torch.bfloat16


def test_matching_dtype_cast_is_identity():
    tensor = torch.randn(4, device="cuda", dtype=torch.bfloat16)
    integer = torch.arange(4, device="cuda")
    cast_tensor, cast_none, cast_integer = cast_to_dtype(
        torch.bfloat16, tensor, None, integer
    )
    assert cast_tensor is tensor
    assert cast_none is None
    assert cast_integer is integer
