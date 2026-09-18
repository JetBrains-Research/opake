# opake-patches

Unified package for Opake patches, providing:

1. PyTorch checkpoint shim (`opake.patches.torch.runtime`)
2. Hugging Face runtime compatibility patches (`opake.patches.transformers.runtime`)
3. Hugging Face model/component patching (`opake.patches.transformers.models`, `opake.patches.transformers.components`)
4. Explicit PEFT/LoRA model patching (`opake.patches.peft`)

This package is pulled in by `opake` and should be consumed through the root
`opake` install surface.
