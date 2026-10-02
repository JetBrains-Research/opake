"""Run an example training script with the clip-path prototypes installed.

Env:
  CB_MT=1          multi-tensor clip kernel + shared dtype markers (mt_engine)
  CB_VMAP_CHUNK=k  streaming path uses torch.func.vmap(..., chunk_size=k);
                   pair with --microbatch-size 0 so clipping runs once per step

Usage (CUDA host, repo root):
  CB_MT=1 CB_VMAP_CHUNK=2 .venv/bin/python experiments/clip_fused/run_train_variant.py \
      examples/train_dpsgd.py <train_dpsgd.py args> --clip-backend triton --microbatch-size 0
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import mt_engine  # noqa: E402

if os.environ.get("CB_MT") == "1":
    mt_engine.install()
chunk = int(os.environ.get("CB_VMAP_CHUNK", "0"))
if chunk:
    mt_engine.install_vmap_chunk(chunk)
print(f"[variant] multi_tensor={os.environ.get('CB_MT') == '1'} vmap_chunk={chunk}", flush=True)

script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
