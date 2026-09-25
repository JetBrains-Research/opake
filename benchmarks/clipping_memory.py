"""Native clipped_grad memory diagnostic; see README.md beside this file."""
# ruff: noqa: INP001 -- standalone diagnostic, not an importable package

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

MODEL = "Qwen/Qwen2.5-Coder-7B"
REVISION = "0396a76181e127dfc13e5c5ec48a8cee09938b02"
TRAINABLES = 968_884_224
MIN_CHUNKS = 3


def emit(record):
    """Keep machine-readable output separate from dependency messages."""
    print(
        json.dumps(record, sort_keys=True, allow_nan=False),
        file=sys.__stdout__,
        flush=True,
    )


def git(root, *args):
    """Read revision metadata without modifying a worktree."""
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def fingerprints(directory):
    """Fingerprint clipping implementation files, including uncommitted changes."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.iterdir())
        if path.suffix == ".py"
    }


def tensors(tree):
    """Walk explicit tensor payloads, not opaque Opake dataclasses."""
    import torch

    if isinstance(tree, torch.Tensor):
        yield tree
    elif isinstance(tree, dict):
        for value in tree.values():
            yield from tensors(value)
    elif isinstance(tree, (tuple, list)):
        for value in tree:
            yield from tensors(value)


def probe_class():
    """Import torch only inside fresh workers."""
    import torch
    from torch.multiprocessing.reductions import StorageWeakRef
    from torch.utils._python_dispatch import TorchDispatchMode

    class Probe(TorchDispatchMode):
        """Observe native storage weak handles, never save tensors or frames."""

        def __init__(self, device, clipping_module):
            super().__init__()
            self.device = device
            self.phase = "setup"
            self.chunk = -1
            self.storages = {}
            self.storage_peaks = defaultdict(int)
            self.cuda_peaks = {}
            self.entries = []
            self.decoder_codes = set()
            self.decoder_counts = defaultdict(Counter)
            self.clipping_codes = {
                fn.__code__: name
                for name in ("global_norm", "clip_pytree", "_stream_clip_and_sum")
                if (fn := getattr(clipping_module, name, None)) is not None
                and hasattr(fn, "__code__")
            }
            self.unobservable = 0
            self.last_operation = None
            self.clipping_calls = Counter()

        def observe(self, tree):
            for tensor in tensors(tree):
                while torch._C._functorch.is_functorch_wrapped_tensor(tensor):
                    tensor = torch._C._functorch.get_unwrapped(tensor)
                try:
                    storage = tensor.untyped_storage()
                    identity = storage._cdata
                    if identity not in self.storages:
                        self.storages[identity] = (
                            StorageWeakRef(storage),
                            storage.nbytes(),
                            self.chunk,
                            self.phase,
                        )
                except (RuntimeError, NotImplementedError):
                    self.unobservable += 1

        def sample(self):
            self.storages = {
                key: value
                for key, value in self.storages.items()
                if not value[0].expired()
            }
            live = sum(value[1] for value in self.storages.values())
            self.storage_peaks[self.phase] = max(self.storage_peaks[self.phase], live)
            return live

        def finish_phase(self):
            self.sample()
            if self.device == "cuda":
                torch.cuda.synchronize()
                peak = self.cuda_peaks.setdefault(
                    self.phase, {"allocated_bytes": 0, "reserved_bytes": 0}
                )
                peak["allocated_bytes"] = max(
                    peak["allocated_bytes"], torch.cuda.max_memory_allocated()
                )
                peak["reserved_bytes"] = max(
                    peak["reserved_bytes"], torch.cuda.max_memory_reserved()
                )

        def set_phase(self, phase):
            if phase != self.phase:
                self.finish_phase()
                self.phase = phase
                if self.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()

        def forward_entry(self):
            self.set_phase("forward")
            self.chunk += 1
            live = self.sample()
            previous = Counter()
            for _, size, chunk, phase in self.storages.values():
                if 0 <= chunk < self.chunk:
                    previous[phase] += size
            self.entries.append(
                {
                    "chunk": self.chunk,
                    "observed_live_storage_bytes": live,
                    "prior_chunks_live_bytes_by_origin_phase": dict(previous),
                }
            )

        def profile(self, frame, event, arg):
            if event != "call":
                return
            if frame.f_code in self.decoder_codes:
                block = frame.f_locals.get("self")
                self.decoder_counts[str(block.benchmark_layer_index)][self.phase] += 1
            if frame.f_code in self.clipping_codes:
                self.clipping_calls[self.clipping_codes[frame.f_code]] += 1
                if self.phase == "backward":
                    self.set_phase("clipping_and_reduction")

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.last_operation = str(func)
            self.observe((args, kwargs))
            self.sample()
            output = func(*args, **(kwargs or {}))
            self.observe(output)
            self.sample()
            return output

        def report(self):
            return {
                "observed_live_storage_peak_bytes_by_phase": dict(self.storage_peaks),
                "cuda_allocator_peak_bytes_by_phase": self.cuda_peaks or None,
                "forward_entries": self.entries,
                "decoder_forward_body_entries": dict(self.decoder_counts),
                "clipping_function_calls": dict(self.clipping_calls),
                "unobservable_storage_observations": self.unobservable,
                "last_operation": self.last_operation,
            }

    return Probe


def model_and_input(args):
    """Build bounded synthetic data or the explicitly requested pinned CUDA target."""
    import torch
    from torch import nn

    from opake.patches import apply_model_patches, apply_runtime_patches

    apply_runtime_patches()
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    records = args.batch_size * args.chunks
    if args.target == "cpu":
        import torch.utils.checkpoint as checkpoint

        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.up = nn.Linear(128, 256, bias=False)
                self.down = nn.Linear(256, 128, bias=False)

            def forward(self, hidden):
                return hidden + self.down(torch.tanh(self.up(hidden)))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([Decoder(), Decoder()])

            def forward(self, hidden):
                for layer in self.layers:
                    if args.checkpoint == "on":
                        hidden = checkpoint.checkpoint(
                            layer, hidden, use_reentrant=False
                        )
                    else:
                        hidden = layer(hidden)
                return hidden.square().mean()

        model = Model()
        batch = torch.rand(records, 16, 128, generator=generator) * 2 - 1
        return model, list(model.layers), batch

    if not torch.cuda.is_available():
        message = "The qwen target requires CUDA; use --target cpu here"
        raise RuntimeError(message)
    if not torch.cuda.is_bf16_supported():
        message = "The qwen target requires BF16-capable CUDA"
        raise RuntimeError(message)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=REVISION,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=not args.allow_download,
    )
    model.requires_grad_(False)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=384,
            lora_alpha=352,
            lora_dropout=0.0,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            task_type="CAUSAL_LM",
        ),
    )
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == TRAINABLES
    assert all(
        p.dtype == (torch.float32 if p.requires_grad else torch.bfloat16)
        for p in model.parameters()
    )
    apply_model_patches(model, kernels=True, fused_linear_cross_entropy=True)
    model.train()
    if args.checkpoint == "on":
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        model.gradient_checkpointing_disable()
    batch = torch.randint(
        0, model.config.vocab_size, (records, 3072), generator=generator
    )
    layers = list(model.get_base_model().model.layers)
    return model.cuda(), layers, batch.cuda()


def validate(result, records):
    """Reject empty/nonfinite payloads after the measured region has ended."""
    import torch

    from opake.api.engine.clipping._clipped_grad import ClippedGradAux
    from opake.types import ClippedPytree

    (clipped, aux), _state = result
    assert isinstance(clipped, ClippedPytree)
    assert isinstance(aux, ClippedGradAux)
    gradients = list(tensors(clipped.pytree))
    assert gradients
    assert all(t.numel() > 0 for t in gradients)
    assert all(torch.isfinite(t).all().item() for t in gradients)
    # These wrappers are not generic pytree leaves: unwrap each payload explicitly.
    payloads = {
        "loss_values": aux.loss_values,
        "grad_norms": aux.grad_norms,
        "clipped_grad_norms": aux.clipped_grad_norms,
        "loss_aux": aux.loss_aux,
    }
    checks = {}
    for name, payload in payloads.items():
        leaves = list(tensors(payload))
        assert leaves, name
        assert all(t.numel() > 0 for t in leaves), name
        assert all(torch.isfinite(t).all().item() for t in leaves), name
        checks[name] = sum(t.numel() for t in leaves)
    assert aux.batch_size == records
    assert aux.loss_values.numel() == records
    assert aux.grad_norms.numel() == records
    assert aux.clipped_grad_norms.numel() == records
    assert (aux.clipped_grad_norms <= 1.0 + 1e-5).all().item()
    assert math.isfinite(aux.clipping_rate)
    assert 0.0 <= aux.clipping_rate <= 1.0
    assert clipped.max_norm == 1.0 / 256
    return {
        "gradient_tensors": len(gradients),
        "gradient_elements": sum(t.numel() for t in gradients),
        "gradient_sum_float64": sum(t.double().sum().item() for t in gradients),
        "aux_elements": checks,
        "mean_loss": aux.loss_values.double().mean().item(),
        "pre_norm_warning": "Sanitized finite pre norms can conceal overflow; not an overflow test.",
    }


def worker(args):
    """Measure one native clipped_grad invocation containing consecutive chunks."""
    import torch

    import opake.api.engine.clipping._clipped_fun as implementation
    from opake.dpsgd.clipping import clipped_grad
    from opake.functional import make_functional

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = Path(implementation.__file__).resolve()
    selector = getattr(implementation, "_streaming_supported", None)
    if args.variant == "original" and selector is not None:
        implementation._streaming_supported = (
            lambda clipping_norm, second_moment, scale_fn: False
        )
    probe = probe_class()("cpu" if args.target == "cpu" else "cuda", implementation)
    metadata = {
        "event": "start",
        "label": args.label,
        "pid": os.getpid(),
        "target": args.target,
        "batch_size": args.batch_size,
        "chunks": args.chunks,
        "records": args.batch_size * args.chunks,
        "checkpoint": args.checkpoint,
        "variant": args.variant,
        "selector_present": selector is not None,
        "normalization": 256,
        "clipping_norm": 1.0,
        "seed": args.seed,
        "engine_file": str(source),
        "engine_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "engine_source_sha256": fingerprints(source.parent),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "engine_git_head": git(source.parent, "rev-parse", "HEAD"),
        "engine_git_status": git(source.parent, "status", "--short"),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft")
        },
        "model": MODEL
        if args.target == "qwen"
        else "synthetic-two-block-128-256-128-fp32",
        "revision": REVISION if args.target == "qwen" else None,
        "cuda_device": torch.cuda.get_device_name()
        if torch.cuda.is_available() and args.target == "qwen"
        else None,
        "cuda_runtime": torch.version.cuda,
    }
    emit(metadata)
    started = None
    try:
        model, layers, batch = model_and_input(args)
        for index, layer in enumerate(layers):
            layer.benchmark_layer_index = index
            probe.decoder_codes.add(inspect.unwrap(type(layer).forward).__code__)
        fmodel, trainable, frozen = make_functional(
            model, disable_autograd_tracking=True, partition_trainable=True
        )

        def loss_fn(params, sample):
            probe.forward_entry()
            if args.target == "cpu":
                loss = fmodel({**frozen, **params}, sample)
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = fmodel(
                        {**frozen, **params},
                        sample.unsqueeze(0),
                        labels=sample.unsqueeze(0),
                        use_cache=False,
                        loss_only=True,
                    ).loss
            probe.set_phase("backward")
            return loss

        clip_fn, state = clipped_grad(
            loss_fn,
            clipping_norm=1.0,
            normalize_by=256,
            microbatch_size=args.batch_size,
            return_aux=True,
        )
        probe.observe((trainable, frozen, dict(model.named_buffers()), batch))
        if probe.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        probe.phase = "between_chunks"
        started = time.perf_counter()
        previous_profile = sys.getprofile()
        try:
            sys.setprofile(probe.profile)
            with probe:
                result = clip_fn(trainable, batch, state=state)
            probe.finish_phase()
        finally:
            sys.setprofile(previous_profile)
        elapsed = time.perf_counter() - started
        if fingerprints(source.parent) != metadata["engine_source_sha256"]:
            message = (
                "Engine sources changed during measurement; rerun a pinned worktree"
            )
            raise RuntimeError(message)
        assert len(probe.entries) == args.chunks, "Unexpected native chunk count"
        assert len(probe.decoder_counts) == len(layers), "Decoder body was not observed"
        for counts in probe.decoder_counts.values():
            assert counts["forward"] == args.chunks, counts
            if args.checkpoint == "on":
                assert counts["backward"] >= args.chunks, "Checkpoint did not recompute"
            else:
                assert counts["backward"] == 0, counts
        probe.phase = "validation"
        checks = validate(result, args.batch_size * args.chunks)
        emit(
            {
                **metadata,
                "event": "result",
                "status": "ok",
                "trainable_parameters": sum(t.numel() for t in trainable.values()),
                "input_sha256": hashlib.sha256(
                    batch.cpu().numpy().tobytes()
                ).hexdigest(),
                "instrumented_elapsed_seconds": elapsed,
                "throughput": None,
                "checks": checks,
                **probe.report(),
            }
        )
        return 0
    except Exception as error:
        oom = (
            isinstance(error, torch.OutOfMemoryError)
            or "out of memory" in str(error).lower()
        )
        if probe.device == "cpu" or torch.cuda.is_initialized():
            with contextlib.suppress(RuntimeError):
                probe.finish_phase()
        emit(
            {
                **metadata,
                "event": "result",
                "status": "oom" if oom else "error",
                "failure_phase": probe.phase,
                "failure_chunk": probe.chunk,
                "error": str(error),
                "instrumented_elapsed_seconds": time.perf_counter() - started
                if started
                else None,
                **probe.report(),
            }
        )
        traceback.print_exc(file=sys.stderr)
        return 2 if oom else 1


def main():
    """Run every selected batch/checkpoint pair in a separate interpreter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("cpu", "qwen"), default="cpu")
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, choices=(4, 5), default=[4, 5]
    )
    parser.add_argument("--checkpoint", choices=("off", "on", "both"), default="both")
    parser.add_argument("--chunks", type=int, default=3)
    parser.add_argument("--variant", choices=("native", "original"), default="native")
    parser.add_argument("--label", default="unlabelled")
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Overlay Python sources from a separate git worktree",
    )
    parser.add_argument(
        "--output", type=Path, help="Write JSONL as well as stdout (refuses overwrite)"
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Explicitly allow the pinned Qwen download",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--batch-size", type=int, choices=(4, 5), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.chunks < MIN_CHUNKS:
        parser.error("--chunks must be at least 3")
    if args.worker:
        with contextlib.redirect_stdout(sys.stderr):
            return worker(args)
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(args.seed)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if args.source_root:
        packages = args.source_root.resolve() / "packages"
        if not packages.is_dir():
            parser.error("--source-root must contain packages/")
        paths = [
            str(p / "src") for p in sorted(packages.iterdir()) if (p / "src").is_dir()
        ]
        env["PYTHONPATH"] = os.pathsep.join([*paths, env.get("PYTHONPATH", "")])
    output = args.output.open("x") if args.output else None
    failed = False
    try:
        for batch in args.batch_sizes:
            for checkpoint in (
                ("off", "on") if args.checkpoint == "both" else (args.checkpoint,)
            ):
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--target",
                    args.target,
                    "--batch-size",
                    str(batch),
                    "--checkpoint",
                    checkpoint,
                    "--chunks",
                    str(args.chunks),
                    "--variant",
                    args.variant,
                    "--label",
                    args.label,
                    "--seed",
                    str(args.seed),
                ]
                if args.allow_download:
                    command.append("--allow-download")
                process = subprocess.Popen(
                    command, env=env, stdout=subprocess.PIPE, text=True
                )
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    if output:
                        output.write(line)
                        output.flush()
                returncode = process.wait()
                if returncode:
                    record = {
                        "event": "worker_exit",
                        "batch_size": batch,
                        "checkpoint": checkpoint,
                        "returncode": returncode,
                        "note": "See preceding result/stderr; killed workers may have no phase report.",
                    }
                    emit(record)
                    if output:
                        output.write(json.dumps(record) + "\n")
                        output.flush()
                failed |= returncode != 0
    finally:
        if output:
            output.close()
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
