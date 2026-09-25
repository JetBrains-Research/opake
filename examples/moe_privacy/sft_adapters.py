"""Expert-parameter LoRA and unwrapped FP32 routers for Mellum SFT."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.nn.utils import parametrize
from transformers.models.mellum.modeling_mellum import (
    MellumExperts,
    MellumTopKRouter,
)

from opaque.patches import apply_model_patches
from opaque.patches.transformers import moe_geometry

_EXPERT_STACK_NDIM = 3
_ADAPTER_KIND = "static_parametrization_expert_lora"


class _ExpertLoRA(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, rank: int, alpha: float):
        super().__init__()
        experts, out_features, in_features = weight.shape
        self.lora_A = torch.nn.Parameter(
            torch.empty(
                experts, rank, in_features, device=weight.device, dtype=torch.float32
            )
        )
        self.lora_B = torch.nn.Parameter(
            torch.zeros(
                experts, out_features, rank, device=weight.device, dtype=torch.float32
            )
        )
        bound = 1 / math.sqrt(in_features)
        torch.nn.init.uniform_(self.lora_A, -bound, bound)
        self.scaling = alpha / rank

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight + self.scaling * (self.lora_B @ self.lora_A).to(weight.dtype)


def configure_expert_lora(
    model: torch.nn.Module,
    rank: int = 4,
    alpha: int = 8,
    *,
    apply_patches: bool = True,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Freeze the backbone and adapt both expert stacks plus the actual routers.

    Configure a fresh Mellum model before constructing the trainer. Expert LoRA
    parametrizations are registered once, outside all functional transforms;
    routers remain unwrapped. Set ``apply_patches=False`` in a fresh interpreter
    for native Transformers/TRL training without Opaque's compatibility patches.
    """
    if hasattr(model, "_expert_lora_config") or getattr(model, "peft_config", None):
        message = "configure_expert_lora requires a model without adapters"
        raise ValueError(message)
    if type(rank) is not int or rank <= 0:
        message = "rank must be a positive integer"
        raise ValueError(message)
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        message = "alpha must be finite and positive"
        raise ValueError(message)
    if not math.isfinite(alpha) or alpha <= 0:
        message = "alpha must be finite and positive"
        raise ValueError(message)

    experts = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, MellumExperts)
    ]
    routers = [
        module for module in model.modules() if isinstance(module, MellumTopKRouter)
    ]
    if not experts or len(experts) != len(routers):
        message = "require one MellumExperts stack per actual MellumTopKRouter"
        raise ValueError(message)
    geometry = moe_geometry(model)
    if geometry["num_layers"] != len(routers):
        message = "router geometry includes wrapped or unexpected routers"
        raise ValueError(message)

    targets = []
    for name, module in experts:
        gate_up, down = module.gate_up_proj, module.down_proj
        if gate_up.ndim != _EXPERT_STACK_NDIM or down.ndim != _EXPERT_STACK_NDIM:
            message = f"{name} must contain 3D expert parameter stacks"
            raise ValueError(message)
        count, hidden, intermediate = down.shape
        if count != geometry["num_experts"] or tuple(gate_up.shape) != (
            count,
            2 * intermediate,
            hidden,
        ):
            message = f"{name} has inconsistent Mellum expert geometry"
            raise ValueError(message)
        targets.extend(
            f"{name}.{parameter}" for parameter in ("gate_up_proj", "down_proj")
        )

    if apply_patches:
        apply_model_patches(
            model,
            performance=False,
            peft=False,
            router_fp32=True,
            router_aux_loss=False,
        )
    model.requires_grad_(False)
    for _, module in experts:
        for parameter in ("gate_up_proj", "down_proj"):
            parametrize.register_parametrization(
                module,
                parameter,
                _ExpertLoRA(getattr(module, parameter), rank, alpha),
            )
    for router in routers:
        router.to(dtype=torch.float32)
        router.weight.requires_grad_(True)
    model._expert_lora_config = {
        "rank": rank,
        "alpha": alpha,
        "target_parameters": sorted(targets),
    }

    _, metadata = _trainable_layout(model)
    if metadata["geometry"] != geometry:
        message = "parametrization changed the expected router geometry"
        raise ValueError(message)
    return model, metadata


def _trainable_layout(
    model: torch.nn.Module,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    config = getattr(model, "_expert_lora_config", None)
    if not isinstance(config, dict):
        message = "require configured static expert-LoRA parametrizations"
        raise ValueError(message)
    router_names = sorted(
        f"{name}.weight"
        for name, module in model.named_modules()
        if isinstance(module, MellumTopKRouter)
    )
    adapter_names = sorted(
        f"{name}.{factor}"
        for name, module in model.named_modules()
        if isinstance(module, _ExpertLoRA)
        for factor in ("lora_A", "lora_B")
    )
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(trainable) != set(router_names + adapter_names):
        message = "unexpected trainable parameters outside expert LoRA and routers"
        raise ValueError(message)
    geometry = moe_geometry(model)
    if (
        len(adapter_names) != 2 * len(config["target_parameters"])
        or len(router_names) != geometry["num_layers"]
    ):
        message = (
            "parametrization changed the expected expert-adapter or router geometry"
        )
        raise ValueError(message)
    if any(parameter.dtype != torch.float32 for parameter in trainable.values()):
        message = "trainable expert factors and routers must remain FP32"
        raise ValueError(message)
    return trainable, {
        "adapter_kind": _ADAPTER_KIND,
        "rank": config["rank"],
        "alpha": config["alpha"],
        "geometry": geometry,
        "target_parameters": sorted(config["target_parameters"]),
        "expert_adapter_names": adapter_names,
        "expert_adapter_count": len(adapter_names),
        "router_names": router_names,
        "router_count": len(router_names),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in trainable.values()
        ),
    }


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _export_description(model: torch.nn.Module):
    trainable, metadata = _trainable_layout(model)
    spec = {
        "format_version": 1,
        "adapter_kind": _ADAPTER_KIND,
        "config": model._expert_lora_config,
        "metadata": metadata,
        "tensors": {
            name: {"shape": list(parameter.shape), "dtype": str(parameter.dtype)}
            for name, parameter in sorted(trainable.items())
        },
    }
    return trainable, spec


def export_trainable(model: torch.nn.Module, output_dir: str | Path) -> None:
    """Write trainable weights and a deterministic reconstruction spec to a fresh dir.

    No frozen model copy, merge, trainer state, or private randomness is exported.
    This static-parametrization bundle is loaded with :func:`load_trainable`,
    not PEFT's adapter loader, and includes the unwrapped router weights.
    """
    trainable, spec = _export_description(model)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    tensors = {
        name: parameter.detach().to(device="cpu", copy=True).contiguous()
        for name, parameter in sorted(trainable.items())
    }
    save_file(tensors, output_dir / "trainable.safetensors")
    (output_dir / "adapter_spec.json").write_text(_json_text(spec), encoding="utf-8")


def load_trainable(model: torch.nn.Module, export_dir: str | Path) -> None:
    """Validate then restore adapters and routers, leaving frozen parameters intact.

    Reconstruct the identical base model and call :func:`configure_expert_lora`
    first. All keys, shapes, dtypes, config, and specification are checked before
    any weight is copied. Frozen base weights are neither loaded nor verified;
    this weights-only bundle cannot resume the sampler or privacy accountant.
    """
    trainable, spec = _export_description(model)
    export_dir = Path(export_dir)
    saved_spec = json.loads(
        (export_dir / "adapter_spec.json").read_text(encoding="utf-8")
    )
    if saved_spec != spec:
        message = "adapter specification does not match the reconstructed model"
        raise ValueError(message)
    with safe_open(
        export_dir / "trainable.safetensors", framework="pt", device="cpu"
    ) as saved:
        saved_names = set(saved.keys())
        expected_names = set(trainable)
        if saved_names != expected_names:
            message = (
                "trainable tensor keys differ: "
                f"missing={sorted(expected_names - saved_names)}, "
                f"unexpected={sorted(saved_names - expected_names)}"
            )
            raise ValueError(message)
        for name, parameter in trainable.items():
            if saved.get_slice(name).get_shape() != list(parameter.shape):
                message = f"trainable tensor shape mismatch: {name}"
                raise ValueError(message)
        tensors = {name: saved.get_tensor(name) for name in trainable}
        for name, parameter in trainable.items():
            if tensors[name].dtype != parameter.dtype:
                message = f"trainable tensor dtype mismatch: {name}"
                raise ValueError(message)
        with torch.no_grad():
            for name, parameter in trainable.items():
                parameter.copy_(tensors[name])
