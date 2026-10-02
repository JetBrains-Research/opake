"""Run an example training script with the clip-path prototypes installed.

Env:
  CB_CUDAGRAPH=1   CUDA-graph the per-microbatch chunk (experiments/cuda_graph); needs a
                   clip backend other than "triton". CB_CUDAGRAPH_CHECK=1 compares replays with eager.
  CB_TF32=1        allow TF32 for float32 matmuls (cuBLAS and cuDNN)
  CB_NO_PEFT=1     apply_model_patches(..., peft=False): PEFT's stock LoRA instead of Opake's
  CB_KPROF=prefix  torch.profiler on chosen train steps; CB_KPROF_STEPS="4:cuda,5:all"
                   (cuda = CUDA activity only, accurate timeline; all = CPU+CUDA for
                   op attribution, inflates host time). Writes <prefix>.step<N>.*
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
    if hasattr(_CF, "_add_trees"):  # foreach accumulator (since reverted)
        _CF._add_trees = _leafwise_add
if os.environ.get("CB_NOFOREACH") == "1" and hasattr(_CF, "_add_trees"):
    _CF._add_trees = _leafwise_add
chunk = int(os.environ.get("CB_VMAP_CHUNK", "0"))
if chunk:
    mt_engine.install_vmap_chunk(chunk)
print(f"[variant] multi_tensor={os.environ.get('CB_MT') == '1'} vmap_chunk={chunk} "
      f"old={os.environ.get('CB_OLD') == '1'} noforeach={os.environ.get('CB_NOFOREACH') == '1'}", flush=True)

if os.environ.get("CB_TF32") == "1":
    import torch as _torch

    _torch.backends.cuda.matmul.allow_tf32 = True
    _torch.backends.cudnn.allow_tf32 = True
if os.environ.get("CB_NO_PEFT") == "1":
    import opake.patches as _patches

    _orig_amp = _patches.apply_model_patches

    def _amp_no_peft(model, *a, **k):
        k["peft"] = False
        return _orig_amp(model, *a, **k)

    _patches.apply_model_patches = _amp_no_peft
print(f"[variant] tf32={os.environ.get('CB_TF32') == '1'} no_peft={os.environ.get('CB_NO_PEFT') == '1'}", flush=True)

if os.environ.get("CB_CUDAGRAPH") == "1":
    import atexit

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cuda_graph"))
    import cudagraph_chunk as _cg

    _cg.install(check=os.environ.get("CB_CUDAGRAPH_CHECK") == "1")
    atexit.register(lambda: print(f"[cudagraph] report {_cg.report()}", flush=True))
    print(f"[variant] cudagraph=True check={os.environ.get('CB_CUDAGRAPH_CHECK') == '1'}", flush=True)

_KPROF = os.environ.get("CB_KPROF")
if _KPROF:
    import contextlib
    import csv
    import gzip

    import torch

    _MEM = importlib.import_module("opake.api.engine.profiling._memory")
    _orig_call = _MEM.PerfStage.__call__
    _plan = dict(item.split(":") for item in os.environ.get("CB_KPROF_STEPS", "4:cuda,5:all").split(","))
    _seen = {"train": 0}

    def _dump(prof, prefix, mode):
        cuda = torch.autograd.DeviceType.CUDA
        with gzip.open(f"{prefix}.kernels.csv.gz", "wt", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["name", "start_us", "dur_us"])
            for e in prof.events():
                if e.device_type == cuda:
                    w.writerow([e.name, f"{e.time_range.start:.3f}", f"{e.time_range.elapsed_us():.3f}"])
        if mode == "all":
            ka = prof.key_averages()
            Path(f"{prefix}.cpu_ops.txt").write_text(ka.table(sort_by="self_cpu_time_total", row_limit=80))
            Path(f"{prefix}.cuda_ops.txt").write_text(ka.table(sort_by="self_cuda_time_total", row_limit=80))
        print(f"[kprof] wrote {prefix}.* ({mode})", flush=True)

    @contextlib.contextmanager
    def _profiled(self, *, batch_size=0, track_memory=True):
        n = None
        if self.name == "train":
            _seen["train"] += 1
            n = _seen["train"]
        mode = _plan.get(str(n)) if n is not None else None
        if mode is None:
            with _orig_call(self, batch_size=batch_size, track_memory=track_memory) as sp:
                yield sp
            return
        acts = [torch.profiler.ProfilerActivity.CUDA]
        if mode == "all":
            acts.append(torch.profiler.ProfilerActivity.CPU)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=acts) as prof:
            with _orig_call(self, batch_size=batch_size, track_memory=track_memory) as sp:
                yield sp
            torch.cuda.synchronize()
        print(f"[kprof] step {n} batch {batch_size} wall {self.last.step_time_sec:.3f}s ({mode})", flush=True)
        _dump(prof, f"{_KPROF}.step{n}", mode)

    _MEM.PerfStage.__call__ = _profiled

script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
