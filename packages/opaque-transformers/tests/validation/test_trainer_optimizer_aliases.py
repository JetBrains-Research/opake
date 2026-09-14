"""Optimizer name validation, HF-alias remapping, and the opaque set.

Trainer's optimizer surface has two layers:

1. Canonical opaque names (``adam``, ``adamw``, ``sgd``, ``rmsprop``,
   ``adagrad``, ``adafactor``, ``ademamix``, ``lion``, ``radam``,
   ``adadelta``, ``schedule_free``) route directly to ``opaque.optimizers.*``
   factories
   with HF-canonical ``TrainingArguments`` fields forwarded.
2. HF compat aliases (``adamw_torch``, ``adamw_torch_fused``,
   ``adamw_hf``, ``adafactor``, ``ademamix``, ``lion_32bit``,
   ``schedule_free_radam``) route to the same opaque factories — Trainer
   honours the HF name by selecting the matching DP-aware update math,
   not by substituting a different one.

Names with no DP-aware mapping (8-bit, paged, GaLore, fused-CUDA, XLA,
NPU, ``adamax``) reject with redirect messages.
"""

from __future__ import annotations

import pytest
import torch
from transformers.training_args import OptimizerNames

from opaque.api.transformers.trainer._optim import (
    build_optimizer,
    canonical_optimizer_names,
)
from opaque.transformers.trainer import Trainer, TrainingArguments


def _args(tmp_path, **overrides) -> TrainingArguments:
    """Optimizer-test args (CPU-pinned, σ=0 for bit-comparable runs)."""
    defaults = {
        "output_dir": str(tmp_path),
        "per_device_train_batch_size": 4,
        "max_steps": 3,
        "num_train_epochs": 1,
        "logging_steps": 1,
        "save_strategy": "no",
        "eval_strategy": "no",
        # NM=0.0 = non-private; target_eps is meaningless on this path.
        "privacy_noise_multiplier": 0.0,
        "clipping_norm": 1.0,
        "learning_rate": 1e-2,
        "seed": 42,
        "use_cpu": True,
    }
    defaults.update(overrides)
    return TrainingArguments(**defaults)


# ---------------------------------------------------------------------------
# HF-alias remapping (accepted) and unsupported-name rejection
# ---------------------------------------------------------------------------


# HF ``OptimizerNames`` values that route transparently onto an
# opaque factory.  Construction must succeed; the underlying update
# math is the matching opaque factory, *not* the HF-named impl.
ACCEPTED_HF_ALIASES = (
    "adamw_torch",  # ↦ opaque.optimizers.adamw
    "adamw_torch_fused",  # ↦ opaque.optimizers.adamw
    "adamw_hf",  # ↦ opaque.optimizers.adamw
    "adafactor",  # ↦ opaque.optimizers.adafactor
    "ademamix",  # ↦ opaque.optimizers.ademamix
    "lion_32bit",  # ↦ opaque.optimizers.lion
    "schedule_free_radam",  # ↦ schedule_free(radam(...))
)


# Names that Trainer cannot honour even with a remap: 8-bit / paged /
# GaLore / Apex-fused / XLA / NPU / opaque primitives without DP-aware
# modes.  Construction must raise ``ValueError`` with a redirect.
REJECTED_OPTIMIZER_NAMES = (
    "adamw_torch_xla",
    "adamw_torch_npu_fused",
    "adamw_apex_fused",
    "adamw_anyprecision",
    "adamw_bnb_8bit",
    "adamw_8bit",
    "adamw_torch_4bit",
    "adamw_torch_8bit",
    "ademamix_8bit",
    "lion_8bit",
    "paged_adamw_32bit",
    "paged_adamw_8bit",
    "paged_lion_32bit",
    "paged_lion_8bit",
    "rmsprop_bnb",
    "rmsprop_bnb_8bit",
    "rmsprop_bnb_32bit",
    "galore_adamw",
    "galore_adafactor",
    "lomo",
    "adalomo",
    "grokadamw",
    "apollo_adamw",
    # Schedule-free over adamw / sgd has a redirect to optim='schedule_free'.
    "schedule_free_adamw",
    "schedule_free_sgd",
    "stable_adamw",
    # Re-exported torchopt primitives without DP-aware modes.
    "adamax",
)


class TestOptimAcceptsHFAliases:
    """HF compat aliases route onto the matching opaque factory."""

    @pytest.mark.parametrize("name", ACCEPTED_HF_ALIASES)
    def test_hf_alias_constructs(self, tmp_path, name):
        args = _args(tmp_path, optim=name)
        # The raw ``args.optim`` keeps the user-supplied alias spelling;
        # the resolver does the redirection at optimizer-build time.
        assert args.optim == name

    def test_optimizernames_enum_adamw_torch_constructs(self, tmp_path):
        """``optim=OptimizerNames.ADAMW_TORCH`` is accepted via remap."""
        args = _args(tmp_path, optim=OptimizerNames.ADAMW_TORCH)
        assert str(args.optim) in {"adamw_torch", "OptimizerNames.ADAMW_TORCH"}


class TestOptimRejectsUnsupportedNames:
    """Names with no DP-aware mapping reject with a redirect."""

    @pytest.mark.parametrize("name", REJECTED_OPTIMIZER_NAMES)
    def test_unsupported_optim_raises(self, tmp_path, name):
        with pytest.raises(ValueError, match="is not supported by Trainer") as exc_info:
            _args(tmp_path, optim=name)
        message = str(exc_info.value)
        assert name in message, (
            f"error must name the rejected optimizer; got: {message}"
        )
        assert "Supported optimizers" in message or "optim=" in message

    def test_unknown_optim_raises(self, tmp_path):
        """Names absent from both the alias and the rejection layer
        fall through to the bare validation step."""
        with pytest.raises(ValueError, match="expected one of"):
            _args(tmp_path, optim="nonsense_optim")


# ---------------------------------------------------------------------------
# Supported torchopt-backed optimizers
# ---------------------------------------------------------------------------


# Canonical opaque names Trainer ships.  ``schedule_free`` is not in
# the smoke set because it requires an inner ``base=`` optim_args entry.
SUPPORTED_OPTIMIZERS = (
    "adam",
    "adamw",
    "sgd",
    "rmsprop",
    "adagrad",
    "adafactor",
    "ademamix",
    "lion",
    "radam",
    "adadelta",
)


class TestSupportedOptimizersConstruct:
    """All torchopt-backed names accepted by ``TrainingArguments``."""

    @pytest.mark.parametrize("name", SUPPORTED_OPTIMIZERS)
    def test_native_name_constructs(self, tmp_path, name):
        args = _args(tmp_path, optim=name)
        assert args.optim == name

    @pytest.mark.parametrize("name", canonical_optimizer_names())
    def test_canonical_optimizer_builds_and_inits(self, tmp_path, name):
        """Every canonical ``optim`` resolves to a factory with a working ``init``.

        Full ``train()`` LM integration is exercised elsewhere when the
        installed Transformers / functorch stack supports vmap over the
        model forward; here we lock the contract that each supported
        name materialises a gradient transform whose ``init`` accepts a
        small parameter pytree.
        """
        args = _args(tmp_path, optim=name)

        def lr(_step: int) -> float:
            return 0.01

        extra = dict(args.optim_args or {})
        opt = build_optimizer(args, lr, extra_kwargs=extra)
        params = {"w": torch.randn(2, 3, requires_grad=True)}
        st = opt.init(params)
        assert st is not None


# ---------------------------------------------------------------------------
# SGD weight-decay wiring
# ---------------------------------------------------------------------------


class TestSgdWeightDecay:
    """``weight_decay`` is forwarded into the functional SGD chain."""

    def test_nonzero_weight_decay_changes_optimizer_state_shape(self, tmp_path):
        """Non-zero ``weight_decay`` inserts the additive decay transform."""

        def lr(_step: int) -> float:
            return 0.1

        args0 = _args(tmp_path, optim="sgd", weight_decay=0.0)
        args1 = _args(tmp_path, optim="sgd", weight_decay=0.1)
        opt0 = build_optimizer(args0, lr, {})
        opt1 = build_optimizer(args1, lr, {})
        params = {"w": torch.ones(2, 2, requires_grad=True)}
        st0 = opt0.init(params)
        st1 = opt1.init(params)
        assert st0 != st1
        assert len(st1) >= len(st0)

    def test_zero_weight_decay_matches_bare_sgd(self, tmp_path):
        """``weight_decay=0`` keeps the minimal SGD state tuple."""

        def lr(_step: int) -> float:
            return 0.1

        args = _args(tmp_path, optim="sgd", weight_decay=0.0)
        opt = build_optimizer(args, lr, {})
        params = {"w": torch.ones(2, 2, requires_grad=True)}
        st = opt.init(params)
        assert isinstance(st, tuple)
        assert len(st) == 1


class _TinyLogitsModel(torch.nn.Module):
    """HF-shaped minimal module for Trainer constructor smoke tests."""

    main_input_name = "x"

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 2)

    def forward(self, x):
        return {"logits": self.lin(x)}


class TestOptimizerClsAndKwargsFunctional:
    """``optimizer_cls_and_kwargs`` accepts opaque/torchopt-style factories only."""

    def test_trainer_constructor_stores_functional_factory(self, tmp_path):
        import opaque.optimizers as opaque_opt

        args = _args(tmp_path, eval_strategy="no", save_strategy="no")
        trainer = Trainer(
            model=_TinyLogitsModel(),
            args=args,
            optimizer_cls_and_kwargs=(opaque_opt.adamw, {"weight_decay": 0.01}),
        )
        assert trainer._functional_optimizer_factory is not None
        assert trainer._functional_optimizer_name == "adamw"

    def test_torch_optim_subclass_rejected_in_validator(self) -> None:
        import torch.optim as optim

        from opaque.api.transformers.trainer._optim import (
            validate_functional_optimizer_cls_and_kwargs,
        )

        with pytest.raises(RuntimeError, match=r"torch.optim"):
            validate_functional_optimizer_cls_and_kwargs((optim.AdamW, {}))


class TestOptimTargetModulesRejected:
    def test_dp_training_arguments_rejects_non_none(self, tmp_path):
        with pytest.raises(TypeError, match="optim_target_modules"):
            _args(tmp_path, optim_target_modules=["q_proj"])
