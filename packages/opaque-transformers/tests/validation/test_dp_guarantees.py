# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""DP-correctness guarantees for :class:`Trainer`.

These tests guard the *privacy* behaviour of the trainer — the surface
the rest of the suite mostly leaves unchecked.  They assert the things a
DP regression would silently break:

- the noise multiplier the trainer calibrates actually hits the target ε
  (against an independently-built reference accountant);
- the accountant composes exactly one mechanism per optimizer step, and
  resume keeps ``prefix + remaining`` on budget;
- clipping bounds per-example gradient norms;
- realized noise stddev equals ``noise_multiplier * clipping_norm``;
- ``evaluate()`` consumes no privacy budget;
- DP noise is reproducible at σ>0 and tracks ``seed`` / ``data_seed``;
- resuming a weights-only (``save_only_model``) checkpoint is refused.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from _hf_shared import build_lm_dataset, gpt2_tokenizer, make_gpt2_model
from peft import LoraConfig, TaskType, get_peft_model

from opaque.exceptions import CheckpointError
from opaque.transformers.trainer import Trainer, TrainingArguments

# ---------------------------------------------------------------------------
# Fixtures (module-scoped: GPT-2 load is the slow part)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gpt2_tok():
    tok = gpt2_tokenizer()
    tok.pad_token = tok.eos_token
    return tok


@pytest.fixture
def gpt2_lora(gpt2_tok):
    model = make_gpt2_model()
    model.config.pad_token_id = gpt2_tok.pad_token_id
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["c_attn"],
        fan_in_fan_out=True,
    )
    return get_peft_model(model, lora), gpt2_tok


@pytest.fixture
def lm_dataset(gpt2_tok):
    texts = [
        "def fibonacci(n): return n",
        "area = math.pi * r * r",
        "return os.listdir(path)",
        "self.items = []",
        "return f'hello {name}'",
        "sorted_list = sorted(xs)",
        "x = [i for i in range(10)]",
        "print('done')",
    ]
    return build_lm_dataset(texts, gpt2_tok, max_length=16)


def _args(tmp_path, **overrides) -> TrainingArguments:
    defaults = {
        "output_dir": str(tmp_path),
        "per_device_train_batch_size": 4,
        "clipping_norm": 1.0,
        "privacy_target_epsilon": 10.0,
        "privacy_noise_multiplier": 1.0,
        "use_cpu": True,
        "report_to": [],
        "max_steps": 4,
    }
    defaults.update(overrides)
    return TrainingArguments(**defaults)


# ---------------------------------------------------------------------------
# C1 — resuming a save_only_model checkpoint must be refused
# ---------------------------------------------------------------------------


def test_resume_from_save_only_model_checkpoint_is_refused(
    gpt2_lora, lm_dataset, tmp_path
):
    """save_only_model checkpoints lack DP runtime state; resuming would reuse
    the noise stream.  Training-resume must hard-error."""
    model, tok = gpt2_lora
    args = _args(
        tmp_path,
        max_steps=4,
        save_strategy="steps",
        save_steps=2,
        save_only_model=True,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=lm_dataset, processing_class=tok
    )
    trainer.train()

    # Find a checkpoint and confirm it has accountant.json but no dp_state.
    ckpts = [
        d.name for d in Path(tmp_path).iterdir() if d.name.startswith("checkpoint-")
    ]
    assert ckpts, "expected at least one checkpoint"
    ckpt_dir = Path(tmp_path) / min(ckpts)
    assert (ckpt_dir / "accountant.json").exists()

    model2 = make_gpt2_model()
    model2.config.pad_token_id = tok.pad_token_id
    model2 = get_peft_model(
        model2,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["c_attn"],
            fan_in_fan_out=True,
        ),
    )
    args2 = _args(tmp_path, max_steps=8, save_strategy="no")
    trainer2 = Trainer(
        model=model2, args=args2, train_dataset=lm_dataset, processing_class=tok
    )
    with pytest.raises(CheckpointError, match="weights-only export"):
        trainer2.train(resume_from_checkpoint=ckpt_dir)


# ---------------------------------------------------------------------------
# Calibration & accounting composition
# ---------------------------------------------------------------------------


def _reference_epsilon(nm: float, q: float, n_steps: int, delta: float) -> float:
    """Independently composed reference ε for Poisson-subsampled Gaussian."""
    import opaque.dpsgd.accounting as dpsgd_acc
    from opaque.accounting import Accountant

    acc = Accountant()
    step = dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), sample_rate=q)
    for _ in range(n_steps):
        acc |= step
    return acc.epsilon_at(delta)


def _reference_block_k_out_of_t_epsilon(
    nm: float, k: int, t: int, delta: float
) -> float:
    import opaque.dpsgd.accounting as dpsgd_acc

    return dpsgd_acc.k_out_of_t(
        dpsgd_acc.gaussian(nm),
        k=k,
        t=t,
        allocation="block",
    ).epsilon_at(delta)


def test_calibrated_noise_hits_target_epsilon(gpt2_lora, lm_dataset, tmp_path):
    """The σ the trainer calibrates reproduces the target ε against an
    independently-built accountant — guards the calibration solver, which no
    other test exercises (all others pin a fixed multiplier)."""
    model, tok = gpt2_lora
    target_eps, delta = 8.0, 1e-5
    args = _args(
        tmp_path,
        privacy_noise_multiplier=None,  # force calibration
        privacy_target_epsilon=target_eps,
        privacy_target_delta=delta,
        max_steps=4,
        save_strategy="no",
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=lm_dataset, processing_class=tok
    )
    out = trainer.train()

    sigma = trainer.state.privacy_resolved_noise_multiplier
    q = trainer.state.privacy_sample_rate
    n = trainer.state.privacy_total_steps
    assert n == 4
    assert q == pytest.approx(4 / len(lm_dataset))  # batch 4 over 8 examples

    reported = out.metrics["privacy_epsilon"]
    ref = _reference_epsilon(sigma, q, n, delta)

    # Reported ε matches an independent composition of the resolved σ ...
    assert reported == pytest.approx(ref, rel=1e-3)
    # ... and calibration landed just under the requested budget.
    assert reported <= target_eps + 1e-2
    assert reported >= target_eps - 0.5


@pytest.mark.slow
def test_fractional_epochs_calibrate_and_compose_resolved_horizon(
    gpt2_lora, lm_dataset, tmp_path
):
    """Fractional epochs calibrate and compose their ceiling step horizon."""
    model, tok = gpt2_lora
    target_eps, delta = 8.0, 1e-5
    args = _args(
        tmp_path,
        max_steps=-1,
        num_train_epochs=1.25,
        privacy_noise_multiplier=None,
        privacy_target_epsilon=target_eps,
        privacy_target_delta=delta,
        save_strategy="no",
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=lm_dataset, processing_class=tok
    )
    out = trainer.train()

    sigma = trainer.state.privacy_resolved_noise_multiplier
    q = trainer.state.privacy_sample_rate
    assert trainer.state.privacy_total_steps == 3
    assert out.global_step == 3
    assert out.metrics["privacy_epsilon"] == pytest.approx(
        _reference_epsilon(sigma, q, 3, delta), rel=1e-3
    )


@pytest.mark.slow
def test_block_k_out_of_t_calibration_and_step_accounting(
    gpt2_lora, lm_dataset, tmp_path
):
    model, tok = gpt2_lora
    target_eps, delta = 8.0, 1e-5
    k, n_steps = 3, 6
    args = _args(
        tmp_path,
        privacy_noise_multiplier=None,
        privacy_target_epsilon=target_eps,
        privacy_target_delta=delta,
        sampling_mode="k_out_of_t",
        sampling_kwargs={"k": k, "allocation": "block"},
        max_steps=n_steps,
        save_strategy="no",
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=lm_dataset, processing_class=tok
    )
    out = trainer.train()
    sigma = trainer.state.privacy_resolved_noise_multiplier
    reported = out.metrics["privacy_epsilon"]
    ref = _reference_block_k_out_of_t_epsilon(sigma, k=k, t=n_steps, delta=delta)
    assert reported == pytest.approx(ref, rel=1e-6)
    assert target_eps - 0.5 <= reported <= target_eps + 1e-2


def test_accountant_composes_exactly_total_steps(gpt2_lora, lm_dataset, tmp_path):
    """Reported ε equals an N-step reference and is distinguishable from
    N±1 — locking the one-mechanism-per-step composition count."""
    model, tok = gpt2_lora
    nm, delta = 1.0, 1e-5
    args = _args(
        tmp_path,
        privacy_noise_multiplier=nm,
        privacy_target_delta=delta,
        max_steps=5,
        save_strategy="no",
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=lm_dataset, processing_class=tok
    )
    out = trainer.train()
    q = trainer.state.privacy_sample_rate
    reported = out.metrics["privacy_epsilon"]

    assert reported == pytest.approx(_reference_epsilon(nm, q, 5, delta), rel=1e-6)
    assert reported != pytest.approx(_reference_epsilon(nm, q, 4, delta), rel=1e-3)
    assert reported != pytest.approx(_reference_epsilon(nm, q, 6, delta), rel=1e-3)


def test_resume_keeps_total_epsilon_on_budget(gpt2_lora, lm_dataset, tmp_path):
    """A run checkpointed mid-way and resumed reports the same final ε as an
    uninterrupted run (prefix + remaining stays on budget)."""
    model, tok = gpt2_lora
    nm, delta = 1.0, 1e-5

    full_dir = tmp_path / "full"
    args_full = _args(
        full_dir,
        privacy_noise_multiplier=nm,
        privacy_target_delta=delta,
        max_steps=4,
        save_strategy="no",
    )
    full = Trainer(
        model=model, args=args_full, train_dataset=lm_dataset, processing_class=tok
    )
    eps_full = full.train().metrics["privacy_epsilon"]

    # Fresh model for the interrupted run.
    model2 = make_gpt2_model()
    model2.config.pad_token_id = tok.pad_token_id
    model2 = get_peft_model(
        model2,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["c_attn"],
            fan_in_fan_out=True,
        ),
    )
    part_dir = tmp_path / "part"
    args_part = _args(
        part_dir,
        privacy_noise_multiplier=nm,
        privacy_target_delta=delta,
        max_steps=2,
        save_strategy="steps",
        save_steps=2,
    )
    part = Trainer(
        model=model2, args=args_part, train_dataset=lm_dataset, processing_class=tok
    )
    part.train()
    ckpt = str(Path(part_dir) / "checkpoint-2")
    assert Path(ckpt).is_dir()

    args_resume = _args(
        part_dir,
        privacy_noise_multiplier=nm,
        privacy_target_delta=delta,
        max_steps=4,
        save_strategy="no",
    )
    resumed = Trainer(
        model=model2, args=args_resume, train_dataset=lm_dataset, processing_class=tok
    )
    eps_resumed = resumed.train(resume_from_checkpoint=ckpt).metrics["privacy_epsilon"]

    assert eps_resumed == pytest.approx(eps_full, rel=1e-6)


# ---------------------------------------------------------------------------
# Clipping & noise primitives the trainer relies on
# ---------------------------------------------------------------------------


def test_clipping_bounds_per_example_norms():
    """Post-clip per-example gradient norms are ≤ the clip bound, and the
    clip is actually exercised (some raw norms exceed it)."""
    from opaque.api.engine.clipping import clipped_grad

    C = 1.0
    # 4 examples; scale x so per-example grads have very different norms,
    # several far above C.
    xs = torch.tensor([[10.0], [0.01], [5.0], [-8.0]])

    def per_example_loss(params, x):
        # grad wrt w is x * (residual); large |x| -> large grad norm.
        return 0.5 * ((params["w"] * x).sum() - 1.0) ** 2

    params = {"w": torch.tensor([1.0])}
    grad_fn, state = clipped_grad(
        per_example_loss,
        argnums=0,
        batch_argnums=(1,),
        clipping_norm=C,
        return_aux=True,
        normalize_by=1.0,
    )
    (_, aux), _ = grad_fn(params, xs, state=state)

    assert aux.grad_norms.max().item() > C  # clip is exercised
    assert torch.all(aux.clipped_grad_norms <= C + 1e-5)


def test_realized_noise_stddev_equals_nm_times_clip():
    """gaussian_noise reports and realizes σ = noise_multiplier * clip_norm."""
    from opaque.api.engine.types import clipped
    from opaque.dpsgd.noise import gaussian_noise
    from opaque.random import key

    nm, C = 1.3, 2.0
    big = {"w": torch.zeros(200_000)}
    cp = clipped(big, max_norm=C)
    noise_fn, st = gaussian_noise(noise_multiplier=nm, key=key(0))
    noised, _ = noise_fn(cp, st)

    assert float(noised.noise_stddev) == pytest.approx(nm * C, rel=1e-6)
    # Empirical stddev of the realized noise matches σ.
    realized = noised.pytree["w"]
    assert realized.std().item() == pytest.approx(nm * C, rel=5e-2)


def test_eval_consumes_no_privacy_budget(gpt2_lora, lm_dataset, tmp_path):
    """Repeated evaluate() calls do not advance the accountant."""
    model, tok = gpt2_lora
    delta = 1e-5
    args = _args(
        tmp_path,
        privacy_noise_multiplier=1.0,
        privacy_target_delta=delta,
        max_steps=4,
        save_strategy="no",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=lm_dataset,
        eval_dataset=lm_dataset,
        processing_class=tok,
    )
    trainer.train()
    eps_before = trainer._accountant.epsilon_at(delta)
    for _ in range(3):
        trainer.evaluate()
    eps_after = trainer._accountant.epsilon_at(delta)
    assert eps_after == pytest.approx(eps_before, rel=0, abs=0)


def test_dp_noise_reproducible_at_sigma_positive(gpt2_tok, lm_dataset, tmp_path):
    """Same seed ⇒ identical trained params at σ>0; different seed ⇒ different."""

    from transformers import set_seed

    def _train(seed, out):
        # Fix the model/LoRA init identically across runs so only the
        # trainer's DP ``seed`` (noise + sampling RNG) varies.
        set_seed(0)
        m = make_gpt2_model()
        m.config.pad_token_id = gpt2_tok.pad_token_id
        m = get_peft_model(
            m,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                target_modules=["c_attn"],
                fan_in_fan_out=True,
            ),
        )
        args = _args(
            out,
            privacy_noise_multiplier=1.0,
            seed=seed,
            max_steps=4,
            save_strategy="no",
        )
        t = Trainer(
            model=m, args=args, train_dataset=lm_dataset, processing_class=gpt2_tok
        )
        t.train()  # restores trained params into t.model in its finally block
        return {
            n: p.detach().clone()
            for n, p in t.model.named_parameters()
            if p.requires_grad
        }

    a = _train(123, tmp_path / "a")
    b = _train(123, tmp_path / "b")
    c = _train(456, tmp_path / "c")
    key0 = next(iter(a))
    assert torch.allclose(a[key0], b[key0], atol=0, rtol=0)
    assert not torch.allclose(a[key0], c[key0])


def test_truncated_poisson_accounting_differs_from_plain():
    """Truncated-Poisson amplification yields a different ε than plain Poisson
    at the same noise multiplier — the trainer's truncated branch is real."""
    import opaque.dpsgd.accounting as dpsgd_acc
    from opaque.accounting import Accountant

    nm, q, n, delta, ds = 1.0, 0.5, 4, 1e-5, 8

    def eps(step):
        acc = Accountant()
        for _ in range(n):
            acc |= step
        return acc.epsilon_at(delta)

    plain = dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), sample_rate=q)
    trunc = dpsgd_acc.poisson(
        dpsgd_acc.gaussian(nm), sample_rate=q, truncated_batch_size=3, dataset_size=ds
    )
    assert eps(plain) != pytest.approx(eps(trunc), rel=1e-3)


def test_rank_folded_key_decorrelates_per_shard_sampling():
    """Rank-folding the sampler key (the trainer's DDP fix) makes each shard's
    Poisson mask independent; a shared key would make them identical."""
    import torch
    from torch.utils.data import TensorDataset

    from opaque.dpsgd.sampling import PoissonSampler
    from opaque.random import fold_in, key

    shard = TensorDataset(torch.arange(100).reshape(-1, 1))

    def stream(k):
        s = PoissonSampler(shard, sample_rate=0.2, n_steps=10, key=k)
        return [tuple(b) for b in s]

    base = key(42)
    # Shared key (the old behaviour): both "ranks" select identical offsets.
    assert stream(base) == stream(base)
    # Rank-folded keys (the fix): the two ranks' streams differ.
    rank0 = stream(fold_in(base, 0))
    rank1 = stream(fold_in(base, 1))
    assert rank0 != rank1


def test_epoch_driven_resume_does_not_overshoot_budget(gpt2_lora, lm_dataset, tmp_path):
    """Epoch-driven (no max_steps) resume with ignore_data_skip from a
    non-epoch-aligned checkpoint must stop at total_steps, not re-run the
    partial epoch and overspend the calibrated budget."""
    model, tok = gpt2_lora
    nm, delta = 1.0, 1e-5
    # batch 2 over 8 examples -> q=0.25 -> 4 steps/epoch; 2 epochs -> 8 total.
    common = {
        "per_device_train_batch_size": 2,
        "clipping_norm": 1.0,
        "privacy_noise_multiplier": nm,
        "privacy_target_delta": delta,
        "use_cpu": True,
        "report_to": [],
        "num_train_epochs": 2,
    }
    args = TrainingArguments(
        output_dir=str(tmp_path), save_strategy="steps", save_steps=3, **common
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=lm_dataset, processing_class=tok
    )
    trainer.train()
    ckpt = str(Path(tmp_path) / "checkpoint-3")  # mid epoch 0 (steps 1-4)
    assert Path(ckpt).is_dir()

    model2 = make_gpt2_model()
    model2.config.pad_token_id = tok.pad_token_id
    model2 = get_peft_model(
        model2,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["c_attn"],
            fan_in_fan_out=True,
        ),
    )
    args2 = TrainingArguments(
        output_dir=str(tmp_path), save_strategy="no", ignore_data_skip=True, **common
    )
    resumed = Trainer(
        model=model2, args=args2, train_dataset=lm_dataset, processing_class=tok
    )
    out = resumed.train(resume_from_checkpoint=ckpt)
    assert out.global_step == 8  # total_steps, not 11
    # ε reflects exactly 8 composed mechanisms (the calibrated horizon).
    assert out.metrics["privacy_epsilon"] == pytest.approx(
        _reference_epsilon(nm, resumed.state.privacy_sample_rate, 8, delta), rel=1e-6
    )


def test_multi_dataset_eval_namespaces_metrics(gpt2_lora, lm_dataset, tmp_path):
    """A dict eval_dataset evaluates each split and namespaces metric keys
    as {prefix}_{name}_* (HF parity)."""
    model, tok = gpt2_lora
    args = _args(tmp_path, max_steps=2, save_strategy="no")
    trainer = Trainer(
        model=model, args=args, train_dataset=lm_dataset, processing_class=tok
    )
    trainer.train()
    metrics = trainer.evaluate(eval_dataset={"a": lm_dataset, "b": lm_dataset})
    assert any(k.startswith("eval_a_") for k in metrics)
    assert any(k.startswith("eval_b_") for k in metrics)
    assert "eval_a_loss" in metrics
    assert "eval_b_loss" in metrics


def test_partial_checkpoint_tmp_dir_is_ignored(gpt2_lora, lm_dataset, tmp_path):
    """A crash-leftover ``checkpoint-N.tmp`` staging dir is invisible to
    checkpoint discovery and resume (atomic-publish guarantee)."""
    from opaque.api.transformers.trainer import _checkpoint as ckpt

    model, tok = gpt2_lora
    args = _args(tmp_path, max_steps=2, save_strategy="steps", save_steps=2)
    trainer = Trainer(
        model=model, args=args, train_dataset=lm_dataset, processing_class=tok
    )
    trainer.train()
    # Simulate a crash mid-write: a half-populated staging dir for a later step.
    partial = tmp_path / "checkpoint-99.tmp"
    partial.mkdir()
    (partial / "accountant.json").write_text("{}")  # but no dp_state.pt

    found = ckpt.list_checkpoints(str(tmp_path))
    assert all(not p.endswith(".tmp") for p in found)
    last = ckpt.get_last_checkpoint(str(tmp_path))
    assert last is not None
    assert last.endswith("checkpoint-2")
