"""Measure padding waste of DPTrainer's logical-batch padding on a next-edit-like dataset.

DPTrainer collates the whole Poisson logical batch at once (padded to its longest
example) and slices microbatches along dim 0 without trimming
(``_clipped_fun._microbatch_accumulate_reduced``), so each sample is computed at
the batch's ``L_max``. This script tokenizes a next-edit-style parquet the way
the SFT pipeline formats it (PromptV0 default template + completion + EOS,
truncated to ``max_length``) and simulates Poisson batches to compare schemes:

* ``current``: every sample padded to the logical batch's L_max.
* ``trim mb=k``: microbatches of k in sampled order, each padded to its own max.
* ``sorted trim mb=k``: logical batch ordered by length, then sliced and trimmed.
* ``... bucket=b``: as above, with L rounded up to a multiple of b (bounds shapes).

Costs: "linear" = sum of padded lengths (token-proportional work), "attn" = sum
of padded lengths squared (eager attention). Ratios are padded / real (>= 1).

Approximation: the prompt is rebuilt from the parquet's ``events``/``input``
fields; NES re-composes history from raw problems (history_size=7), so real
prompts may be somewhat shorter. Lengths are tokens of prompt + completion.

Usage:
  uv run --with transformers --with pyarrow --with numpy \
    python experiments/padding_waste/measure_padding.py PATH.parquet
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-Coder-7B"
MAX_LENGTH = 4096
EXPECTED_BATCH = 256
N_STEPS = 2000
SEED = 0

INSTRUCTIONS = (
    "You are a code completion assistant and your task is to analyze user edits and then rewrite an excerpt that the "
    "user provides, suggesting the appropriate edits within the excerpt, taking into account the cursor location."
)


def prompt_of(events: str, excerpt: str) -> str:
    # PromptV0 DEFAULT_TEMPLATE; the parquet ``input`` already carries the ```filepath fence.
    return (
        f"### Instruction:\n\n{INSTRUCTIONS}\n\n### User Edits:\n\n{events}\n\n"
        f"### User Excerpt:\n\n{excerpt}\n\n### Response:\n\n"
    )


def padded_cost(lengths: np.ndarray, mb: int | None, sort: bool, bucket: int | None) -> tuple[int, int]:
    """Return (sum padded L, sum padded L^2) for one logical batch."""
    x = np.sort(lengths) if sort else lengths
    if mb is None:  # current: one pad length for the whole logical batch
        groups = [x]
    else:
        groups = [x[i : i + mb] for i in range(0, len(x), mb)]
    lin = sq = 0
    for g in groups:
        L = int(g.max())
        if bucket:
            L = min(MAX_LENGTH, -(-L // bucket) * bucket)
        lin += L * len(g)
        sq += L * L * len(g)
    return lin, sq


def main(path: str) -> None:
    table = pq.read_table(path, columns=["events", "input", "output"]).to_pydict()
    tok = AutoTokenizer.from_pretrained(MODEL)
    texts = [prompt_of(e, i) + o + tok.eos_token for e, i, o in zip(table["events"], table["input"], table["output"], strict=True)]
    ids = tok(texts, add_special_tokens=False)["input_ids"]
    raw = np.array([len(t) for t in ids])
    lengths = np.minimum(raw, MAX_LENGTH)
    n = len(lengths)
    q = EXPECTED_BATCH / n
    rng = np.random.default_rng(SEED)

    schemes = {"current (pad to logical-batch max)": (None, False, None)}
    for mb in (1, 2, 4, 8):
        schemes[f"trim mb={mb}"] = (mb, False, None)
    for mb in (2, 4, 8):
        schemes[f"sorted trim mb={mb}"] = (mb, True, None)
    for mb in (2, 4, 8):
        for b in (128, 256, 512):
            schemes[f"sorted trim mb={mb} bucket={b}"] = (mb, True, b)

    real_lin = real_sq = 0
    acc = {k: [0, 0] for k in schemes}
    batch_sizes, batch_max = [], []
    for _ in range(N_STEPS):
        sel = lengths[rng.random(n) < q]  # Poisson sampling, sampled order = dataset order
        if len(sel) == 0:
            continue
        rng.shuffle(sel)  # dataset order is arbitrary; avoid any order artefact
        batch_sizes.append(len(sel))
        batch_max.append(int(sel.max()))
        real_lin += int(sel.sum())
        real_sq += int((sel.astype(np.int64) ** 2).sum())
        for k, (mb, srt, b) in schemes.items():
            lin, sq = padded_cost(sel, mb, srt, b)
            acc[k][0] += lin
            acc[k][1] += sq

    # Oracle checks (see module docstring).
    assert acc["trim mb=1"][0] == real_lin and acc["trim mb=1"][1] == real_sq, "mb=1 must equal real tokens"
    for mb in (2, 4, 8):
        assert acc[f"sorted trim mb={mb}"][0] <= acc[f"trim mb={mb}"][0], "sorting must not increase padding"
        assert acc[f"trim mb={mb}"][0] <= acc["current (pad to logical-batch max)"][0]
    assert all(v[0] >= real_lin and v[1] >= real_sq for v in acc.values())
    # mb = whole batch reproduces current exactly, checked on one batch.
    sel = lengths[np.random.default_rng(1).random(n) < q]
    assert padded_cost(sel, len(sel), False, None) == padded_cost(sel, None, False, None)

    pct = lambda p: int(np.percentile(lengths, p))  # noqa: E731
    stats = {
        "rows": n, "truncated_at_4096": int((raw > MAX_LENGTH).sum()),
        "length_mean": float(lengths.mean()), "length_p50": pct(50), "length_p90": pct(90),
        "length_p99": pct(99), "length_max": int(lengths.max()),
        "batch_mean": float(np.mean(batch_sizes)), "batch_max_len_mean": float(np.mean(batch_max)),
        "batch_max_len_p10": int(np.percentile(batch_max, 10)), "steps": len(batch_sizes),
    }
    results = {k: {"linear_ratio": v[0] / real_lin, "attn_ratio": v[1] / real_sq} for k, v in acc.items()}
    out = Path("notes/smoke-compare/padding_waste.json")
    out.write_text(json.dumps({"dataset": path, "model": MODEL, "stats": stats, "results": results}, indent=1))

    print(json.dumps(stats, indent=1))
    print("| scheme | linear: padded/real | attention: padded/real | linear speed-up vs current |")
    print("|---|---:|---:|---:|")
    cur = results["current (pad to logical-batch max)"]["linear_ratio"]
    for k, r in results.items():
        print(f"| {k} | {r['linear_ratio']:.2f} | {r['attn_ratio']:.2f} | x{cur / r['linear_ratio']:.2f} |")
    print(f"\nwritten {out}; oracle checks passed")


if __name__ == "__main__":
    main(sys.argv[1])
