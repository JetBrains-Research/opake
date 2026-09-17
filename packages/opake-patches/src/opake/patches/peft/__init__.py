"""PEFT / LoRA fusion patches — wires fused LoRA kernels onto PEFT modules."""

from opake.api.patches.peft import apply_peft_model_patches

__all__ = ["apply_peft_model_patches"]
