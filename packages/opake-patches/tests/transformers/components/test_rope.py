# Copyright (c) 2025 Opake Authors
# SPDX-License-Identifier: Apache-2.0
"""RoPE module-level patches: extended model families share Llama-style RoPE."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import pytest
import torch

_KERNELS_AVAILABLE = (
    torch.cuda.is_available() and importlib.util.find_spec("triton") is not None
)
_requires_cuda_triton = pytest.mark.skipif(
    not _KERNELS_AVAILABLE,
    reason="RoPE Triton patches apply only with CUDA + Triton",
)


def _run(script: str) -> str:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("OPAKE_SKIP_"):
            del env[key]
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, f"subprocess failed: {result.stderr}"
    return result.stdout.strip()


@_requires_cuda_triton
def test_cohere_cohere2_apply_rotary_uses_opake_kernel():
    """Cohere / Cohere2 use the same module-level API as Llama; patch must land."""
    out = _run(
        "import torch.nn as nn\n"
        "from opake.api.patches.transformers._router import apply_transformers_model_patches\n"
        "from opake.api.patches.transformers.components.rope import _opake_apply_rotary_pos_emb\n"
        "import transformers.models.cohere.modeling_cohere as mc\n"
        "import transformers.models.cohere2.modeling_cohere2 as mc2\n"
        "class C: model_type='cohere'\n"
        "class M(nn.Module): config=C()\n"
        "apply_transformers_model_patches(M())\n"
        "class C: model_type='cohere2'\n"
        "class M(nn.Module): config=C()\n"
        "apply_transformers_model_patches(M())\n"
        "assert mc.apply_rotary_pos_emb is _opake_apply_rotary_pos_emb\n"
        "assert mc2.apply_rotary_pos_emb is _opake_apply_rotary_pos_emb\n"
        "print('ok')",
    )
    assert out == "ok"


@_requires_cuda_triton
def test_skip_rope_token_leaves_cohere_stock():
    """OPAKE_SKIP_TRANSFORMERS_KERNEL_PATCHES=rope skips module-level RoPE swap."""
    out = _run(
        "import torch.nn as nn\n"
        "from opake.api.patches.transformers._router import apply_transformers_model_patches\n"
        "import transformers.models.cohere.modeling_cohere as mc\n"
        "import transformers.models.cohere2.modeling_cohere2 as mc2\n"
        "class C: model_type='cohere'\n"
        "class M(nn.Module): config=C()\n"
        "apply_transformers_model_patches(M(), rope=False)\n"
        "assert mc.apply_rotary_pos_emb.__name__ == 'apply_rotary_pos_emb'\n"
        "assert mc2.apply_rotary_pos_emb.__name__ == 'apply_rotary_pos_emb'\n"
        "print('ok')",
    )
    assert out == "ok"


@_requires_cuda_triton
def test_gemma3_apply_rotary_uses_opake_kernel():
    """Gemma3 uses the same module-level apply_rotary_pos_emb API as Llama."""
    out = _run(
        "import torch.nn as nn\n"
        "from opake.api.patches.transformers._router import apply_transformers_model_patches\n"
        "from opake.api.patches.transformers.components.rope import _opake_apply_rotary_pos_emb\n"
        "import transformers.models.gemma3.modeling_gemma3 as m3\n"
        "class C: model_type='gemma3'\n"
        "class M(nn.Module): config=C()\n"
        "apply_transformers_model_patches(M())\n"
        "assert m3.apply_rotary_pos_emb is _opake_apply_rotary_pos_emb\n"
        "print('ok')",
    )
    assert out == "ok"


@_requires_cuda_triton
def test_skip_rope_token_leaves_gemma3_stock():
    """OPAKE_SKIP_TRANSFORMERS_KERNEL_PATCHES=rope skips Gemma3 RoPE swap too."""
    out = _run(
        "import torch.nn as nn\n"
        "from opake.api.patches.transformers._router import apply_transformers_model_patches\n"
        "import transformers.models.gemma3.modeling_gemma3 as m3\n"
        "class C: model_type='gemma3'\n"
        "class M(nn.Module): config=C()\n"
        "apply_transformers_model_patches(M(), rope=False)\n"
        "assert m3.apply_rotary_pos_emb.__name__ == 'apply_rotary_pos_emb'\n"
        "print('ok')",
    )
    assert out == "ok"


@_requires_cuda_triton
def test_exaone4_apply_rotary_uses_opake_kernel():
    """Exaone4 uses the same module-level apply_rotary_pos_emb API as Llama."""
    out = _run(
        "import torch.nn as nn\n"
        "from opake.api.patches.transformers._router import apply_transformers_model_patches\n"
        "from opake.api.patches.transformers.components.rope import _opake_apply_rotary_pos_emb\n"
        "import transformers.models.exaone4.modeling_exaone4 as me4\n"
        "class C: model_type='exaone4'\n"
        "class M(nn.Module): config=C()\n"
        "apply_transformers_model_patches(M())\n"
        "assert me4.apply_rotary_pos_emb is _opake_apply_rotary_pos_emb\n"
        "print('ok')",
    )
    assert out == "ok"


@_requires_cuda_triton
def test_skip_rope_token_leaves_exaone4_stock():
    """OPAKE_SKIP_TRANSFORMERS_KERNEL_PATCHES=rope skips Exaone4 RoPE swap too."""
    out = _run(
        "import torch.nn as nn\n"
        "from opake.api.patches.transformers._router import apply_transformers_model_patches\n"
        "import transformers.models.exaone4.modeling_exaone4 as me4\n"
        "class C: model_type='exaone4'\n"
        "class M(nn.Module): config=C()\n"
        "apply_transformers_model_patches(M(), rope=False)\n"
        "assert me4.apply_rotary_pos_emb.__name__ == 'apply_rotary_pos_emb'\n"
        "print('ok')",
    )
    assert out == "ok"
