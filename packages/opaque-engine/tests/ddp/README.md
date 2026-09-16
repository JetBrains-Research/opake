# NCCL / `mp.spawn` tests (`tests/ddp`)

CUDA-marked tests that initialize `torch.distributed` with `mp.spawn` live here.
`tests/conftest.py` adds this directory to `sys.path` and `PYTHONPATH` so spawn
workers can import `engine_ddp_helpers` under pytest's importlib mode.

## Layout

- `engine_ddp_helpers.py` — free port, init/destroy process group, `torch.multiprocessing.spawn` wrapper
- `test_core_utilities.py` — rank / world_size / `is_distributed` without spawn
- `test_collectives.py` — reduce / all-reduce on tensors and pytrees
- `test_local_shard.py` — `local_shard` dataset slicing
- `test_profiler_sync.py` — `TrainingProfiler` + `sync` under NCCL
- `test_second_moment_reduce.py` — gloo CPU: JME optimizer-handoff `reduce_pytree`
- `test_sync_aux_empty_batch.py` — gloo CPU: empty-vs-nonempty `sync(aux)` collective parity

## Running

From the repo root (with CUDA):

```bash
uv run pytest packages/opaque-engine/tests/ddp/ -m "cuda and not slow" -v
```

CPU-only hosts: CUDA-marked tests auto-skip via `@pytest.mark.cuda`. The gloo
CPU regressions (`test_second_moment_reduce`, `test_sync_aux_empty_batch`) still run.
