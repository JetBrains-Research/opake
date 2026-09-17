# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""Built-in HuggingFace model family patches.

Importing this package eagerly imports every shipped model file.  Each
file ends with a ``register_family(...)`` call, so all built-in
families land in the registry by the time
:func:`opake.patches.transformers.supported_families` is called.

The cost is bounded: each ``models/X.py`` only constructs two factory
closures and registers them — the actual ``transformers.models.X``
modeling module is imported lazily inside the apply function (when the
patch is first used on a real model), not at import time.
"""

from .cohere import apply_cohere_patches as apply_cohere_patches
from .cohere2 import apply_cohere2_patches as apply_cohere2_patches
from .deepseek_v4 import apply_deepseek_v4_patches as apply_deepseek_v4_patches
from .exaone4 import apply_exaone4_patches as apply_exaone4_patches
from .gemma import apply_gemma_patches as apply_gemma_patches
from .gemma2 import apply_gemma2_patches as apply_gemma2_patches
from .gemma3 import apply_gemma3_patches as apply_gemma3_patches
from .glm4 import apply_glm4_patches as apply_glm4_patches
from .gpt2 import apply_gpt2_patches as apply_gpt2_patches
from .gpt_oss import apply_gpt_oss_patches as apply_gpt_oss_patches
from .granite import apply_granite_patches as apply_granite_patches
from .hunyuan_v1_moe import apply_hunyuan_v1_moe_patches as apply_hunyuan_v1_moe_patches
from .llama import apply_llama_patches as apply_llama_patches
from .mellum import apply_mellum_patches as apply_mellum_patches
from .ministral import apply_ministral_patches as apply_ministral_patches
from .mistral import apply_mistral_patches as apply_mistral_patches
from .mixtral import apply_mixtral_patches as apply_mixtral_patches
from .olmo2 import apply_olmo2_patches as apply_olmo2_patches
from .olmo3 import apply_olmo3_patches as apply_olmo3_patches
from .phi3 import apply_phi3_patches as apply_phi3_patches
from .qwen2 import apply_qwen2_patches as apply_qwen2_patches
from .qwen3 import apply_qwen3_patches as apply_qwen3_patches
from .qwen3_5_moe import apply_qwen3_5_moe_patches as apply_qwen3_5_moe_patches
from .qwen3_moe import apply_qwen3_moe_patches as apply_qwen3_moe_patches
from .qwen3_next import apply_qwen3_next_patches as apply_qwen3_next_patches
from .smollm3 import apply_smollm3_patches as apply_smollm3_patches
