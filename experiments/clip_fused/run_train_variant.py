"""Run an example training script with the clip-path prototypes installed.

Env:
  CB_OLD=1         previous packaged kernels, per-leaf markers, leafwise accumulator
  CB_NOFOREACH=1   leafwise accumulator instead of torch._foreach_add
  CB_MT=1          multi-tensor clip kernel + shared dtype markers (mt_engine)
  CB_VMAP_CHUNK=k  streaming path uses torch.func.vmap(..., chunk_size=k);
                   pair with --microbatch-size 0 so clipping runs once per step

Usage (CUDA host, repo root):
  CB_MT=1 CB_VMAP_CHUNK=2 .venv/bin/python experiments/clip_fused/run_train_variant.py \
      examples/train_dpsgd.py <train_dpsgd.py args> --clip-backend triton --microbatch-size 0
"""

from __future__ import annotations

import importlib
import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import mt_engine  # noqa: E402
from opake.api.engine.pytree import tree_map  # noqa: E402

if os.environ.get("CB_MT") == "1":
    mt_engine.install()

_CS = importlib.import_module("opake.api.engine.kernels._clip_sum")
_CF = importlib.import_module("opake.api.engine.clipping._clipped_fun")


def _leafwise_add(total, new):
    return tree_map(lambda acc, value: acc + value, total, new)


if os.environ.get("CB_OLD") == "1":  # previous packaged behaviour (c13e3007)
    import _clip_sum_per_tensor as _old

    _CS.fused_clip_sum = _old.fused_clip_sum
    _CS._dtype_marker = lambda leaf: leaf.new_zeros(())
    _CF._add_trees = _leafwise_add
if os.environ.get("CB_NOFOREACH") == "1":
    _CF._add_trees = _leafwise_add
chunk = int(os.environ.get("CB_VMAP_CHUNK", "0"))
if chunk:
    mt_engine.install_vmap_chunk(chunk)
print(f"[variant] multi_tensor={os.environ.get('CB_MT') == '1'} vmap_chunk={chunk} "
      f"old={os.environ.get('CB_OLD') == '1'} noforeach={os.environ.get('CB_NOFOREACH') == '1'}", flush=True)

script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
