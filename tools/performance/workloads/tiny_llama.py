"""Download-free patched Llama workload for evidence-harness smoke runs."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Any, Literal

import torch

from opaque.api.engine.clipping import clipped_grad
from opaque.api.engine.device import fused_kernels_available
from opaque.functional import make_functional
from opaque.patches import apply_model_patches, is_runtime_patched
from tools.performance.patched_model_evidence import CheckResult, Stage, Workload

if TYPE_CHECKING:
    from collections.abc import Mapping

_FRESH_PROCESS_REQUIRED = (
    "the built-in correctness oracle requires a fresh process before patches apply"
)


def _errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    difference = (actual.float() - expected.float()).abs()
    max_abs = difference.max().item()
    denominator = expected.float().abs().clamp_min(1e-7)
    return max_abs, (difference / denominator).max().item()


def _check(
    name: str,
    kind: Literal["numerical", "gradient"],
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> CheckResult:
    max_abs, max_rel = _errors(actual, expected)
    return CheckResult(
        name=name,
        kind=kind,
        passed=torch.allclose(actual, expected, atol=atol, rtol=rtol),
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        atol=atol,
        rtol=rtol,
    )


def create_workload(
    device: torch.device, overrides: Mapping[str, Any], seed: int
) -> Workload:
    """Build a tiny random-weight Llama and apply production model patches."""

    if is_runtime_patched():
        raise RuntimeError(_FRESH_PROCESS_REQUIRED)

    from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM

    config = {
        "batch_size": 2,
        "sequence_length": 16,
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        **overrides,
    }
    torch.manual_seed(seed)
    model_config = LlamaConfig(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        intermediate_size=config["intermediate_size"],
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        max_position_embeddings=max(config["sequence_length"], 32),
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rope_theta=10000.0,
    )
    model_config._attn_implementation = "sdpa"
    reference = LlamaForCausalLM(model_config)
    patched = LlamaForCausalLM(model_config)
    patched.load_state_dict(reference.state_dict())
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    reference = reference.to(device=device, dtype=dtype)
    patched = patched.to(device=device, dtype=dtype)

    generator = torch.Generator().manual_seed(seed)
    shape = (config["batch_size"], config["sequence_length"])
    input_ids = torch.randint(
        0,
        config["vocab_size"],
        shape,
        generator=generator,
    ).to(device)
    attention_mask = torch.ones(shape, dtype=torch.long, device=device)
    labels = input_ids.clone()
    tokens = input_ids.numel()

    reference.eval()
    with torch.no_grad():
        expected_logits = (
            reference(input_ids=input_ids, attention_mask=attention_mask)
            .logits.detach()
            .cpu()
        )
    reference(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels
    ).loss.backward()
    expected_gradients = torch.cat(
        [
            parameter.grad.float().flatten().cpu()
            for parameter in reference.parameters()
            if parameter.grad is not None
        ]
    )
    del reference
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    apply_model_patches(patched, peft=False, eager_attention=True)
    initial_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in patched.named_parameters()
    }

    def restore() -> None:
        patched.zero_grad(set_to_none=True)
        with torch.no_grad():
            for name, parameter in patched.named_parameters():
                parameter.copy_(initial_state[name])

    def clear_gradients(_: Any, __: Any) -> None:
        patched.zero_grad(set_to_none=True)

    def forward_prepare() -> None:
        restore()

    def forward(_: None):
        patched.train()
        return patched(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def backward_prepare() -> torch.Tensor:
        restore()
        patched.train()
        return patched(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        ).loss

    def backward(loss: torch.Tensor) -> None:
        loss.backward()

    def evaluation_prepare() -> None:
        patched.eval()

    def evaluation(_: None):
        with torch.no_grad():
            return patched(input_ids=input_ids, attention_mask=attention_mask)

    functional_model, trainable, frozen = make_functional(
        patched, disable_autograd_tracking=True, partition_trainable=True
    )

    def per_example_loss(params, constants, ids, mask, targets):
        outputs = functional_model(
            {**constants, **params},
            ids,
            attention_mask=mask,
            labels=targets,
        )
        return outputs.loss

    grad_fn, clip_state = clipped_grad(
        per_example_loss,
        argnums=0,
        batch_argnums=(2, 3, 4),
        clipping_norm=1.0,
    )

    def clipped_gradient(_: None):
        return grad_fn(
            trainable,
            frozen,
            input_ids,
            attention_mask,
            labels,
            state=clip_state,
        )

    optimizer = torch.optim.SGD(patched.parameters(), lr=1e-3)

    def optimizer_prepare() -> None:
        restore()
        for parameter in patched.parameters():
            parameter.grad = torch.ones_like(parameter)

    def optimizer_step(_: None) -> None:
        optimizer.step()

    kernel_backend = "triton" if fused_kernels_available() else "portable"
    backend = f"opaque-patched:sdpa+{kernel_backend}"
    stages = (
        Stage("forward", "token", tokens, backend, forward_prepare, forward),
        Stage(
            "backward",
            "token",
            tokens,
            backend,
            backward_prepare,
            backward,
            clear_gradients,
        ),
        Stage(
            "per_example_gradient_and_clipping",
            "example",
            config["batch_size"],
            backend,
            lambda: None,
            clipped_gradient,
        ),
        Stage(
            "optimizer",
            "parameter",
            sum(parameter.numel() for parameter in patched.parameters()),
            "torch.optim.SGD",
            optimizer_prepare,
            optimizer_step,
            clear_gradients,
        ),
        Stage(
            "evaluation",
            "token",
            tokens,
            backend,
            evaluation_prepare,
            evaluation,
        ),
    )

    def correctness() -> tuple[CheckResult, ...]:
        restore()
        patched.eval()
        with torch.no_grad():
            actual = (
                patched(input_ids=input_ids, attention_mask=attention_mask)
                .logits.detach()
                .cpu()
            )
        atol, rtol = (1e-2, 1e-2) if dtype == torch.bfloat16 else (1e-6, 1e-5)
        output_check = _check(
            "patched_logits",
            "numerical",
            actual,
            expected_logits,
            atol=atol,
            rtol=rtol,
        )

        patched.zero_grad(set_to_none=True)
        patched(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        ).loss.backward()
        actual_gradients = torch.cat(
            [
                parameter.grad.float().flatten().cpu()
                for parameter in patched.parameters()
                if parameter.grad is not None
            ]
        )
        gradient_check = _check(
            "patched_parameter_gradients",
            "gradient",
            actual_gradients,
            expected_gradients,
            atol=atol,
            rtol=rtol,
        )
        return output_check, gradient_check

    return Workload(
        workload_id="patched-model.tiny-llama",
        config=config,
        stages=stages,
        checks=correctness,
    )
