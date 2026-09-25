"""Verify the Mac Docker boundary using synthetic fixtures, never model scores."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from examples.moe_privacy import code_eval


def _json(path: Path, value) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def verify_evaluator(image: str, directory: Path) -> dict:
    """Run four supplied identity-function fixtures inside the unchanged sandbox."""
    image_id = code_eval.inspect_image(image)
    directory.mkdir(parents=True, exist_ok=False)
    prompt = "def identity(x):\n"
    fixture = (
        "    import os\n"
        "    assert os.getuid() == os.getgid() == 65534\n"
        "    assert os.path.exists('/.dockerenv')\n"
        "    assert not os.path.exists('/var/run/docker.sock')\n"
        "    from opaque_sandbox_network import assert_network_disabled\n"
        "    assert_network_disabled()\n"
        "    assert os.statvfs('/').f_flag & os.ST_RDONLY\n"
        "    assert not any(k.startswith('ZENML_') for k in os.environ)\n"
        "    assert 'WANDB_API_KEY' not in os.environ\n"
        "    assert 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ\n"
        "    return x\n"
    )
    benchmark_path = directory / "synthetic-humaneval-shaped.jsonl"
    with benchmark_path.open("x") as stream:
        for index in range(164):
            stream.write(
                json.dumps(
                    {
                        "task_id": f"HumanEval/{index}",
                        "prompt": prompt,
                        "entry_point": "identity",
                        "canonical_solution": "    return x\n",
                        "contract": "",
                        "atol": 0,
                        "base_input": [[1], [2]],
                        "plus_input": [[-1], [0], [3]],
                    }
                )
                + "\n"
            )
    _, benchmark = code_eval.load_benchmark("humaneval", benchmark_path)
    samples = directory / "fixture-samples"
    samples.mkdir()
    ids = [f"HumanEval/{index}" for index in range(4)]
    solution = prompt + fixture
    data = "".join(
        json.dumps({"task_id": name, "solution": solution}) + "\n" for name in ids
    ).encode()
    (samples / "samples.jsonl").write_bytes(data)
    (samples / "raw_samples.jsonl").write_bytes(data)

    def digest(value):
        return hashlib.sha256(value).hexdigest()

    _json(
        samples / "generation.json",
        {
            "format_version": code_eval.GENERATION_FORMAT_VERSION,
            "evalplus_version": code_eval.EVALPLUS_VERSION,
            "mode": "smoke",
            "smoke_limit": len(ids),
            "benchmark": benchmark,
            "model": {
                "id": "synthetic-docker-isolation-fixture-not-a-model",
                "revision": "0" * 40,
                "arm": "fixture",
            },
            "generation": {**code_eval.generation_settings(512), "eos_token_ids": [0]},
            "task_ids": ids,
            "samples_sha256": digest(data),
            "raw_samples_sha256": digest(data),
            "tasks": [
                {
                    "task_id": name,
                    "prompt_sha256": digest(prompt.encode()),
                    "solution_sha256": digest(solution.encode()),
                    "raw_solution_sha256": digest(solution.encode()),
                    "completion_boundary": None,
                    "prompt_tokens": 1,
                    "latency_seconds": 0,
                    "generated_tokens": 1,
                    "token_limit_truncated": False,
                    "stop_reason": "eos",
                }
                for name in ids
            ],
        },
    )
    result = code_eval.evaluate_samples(
        dataset="humaneval",
        dataset_path=benchmark_path,
        samples_dir=samples,
        output_dir=directory / "docker-output",
        image=image_id,
        limits=code_eval.SandboxLimits(timeout_seconds=300),
    )
    if result["plus"]["pass@1"] != 1 or result["base"]["pass@1"] != 1:
        message = (
            "Synthetic isolation fixtures failed; do not score experiment answers."
        )
        raise RuntimeError(message)
    identity = json.loads(
        subprocess.run(
            ["docker", "image", "inspect", "--format", "{{json .}}", image_id],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    )
    engine = subprocess.run(
        [
            "docker",
            "version",
            "--format",
            "{{.Server.Version}} {{.Server.Os}}/{{.Server.Arch}}",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    receipt = {
        "status": "completed",
        "synthetic_fixture_only": True,
        "fixture_count": len(ids),
        "image_id": image_id,
        "image_platform": f"{identity['Os']}/{identity['Architecture']}",
        "docker_engine": engine,
        "limits": result["limits"],
        "isolation_verified": [
            "non-root",
            "network-off",
            "read-only",
            "no-docker-socket",
            "no-cluster-credentials",
        ],
        "historical_comparison": "re-score retained samples under this exact image and limits; no cross-platform parity claim",
    }
    _json(directory / "verification.json", receipt)
    return receipt


def main() -> None:
    """Verify one explicit local image into a new, retained receipt directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_evaluator(args.image, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
