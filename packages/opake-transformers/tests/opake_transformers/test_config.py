"""Tests for ``TrainingArguments.__post_init__`` HF-parity contract.

Covers:

- ``no_cuda`` is no longer accepted on the standalone dataclass.
- Auto-default of ``metric_for_best_model="loss"`` under
  ``load_best_model_at_end``.
- Cadence-alignment validation for ``load_best_model_at_end``
  (``save_strategy`` must equal ``eval_strategy``;
  ``save_steps % eval_steps == 0`` when both are step-based).
- Strategy-string validation across the four strategy fields
  (``eval_strategy`` / ``logging_strategy`` / ``save_strategy`` /
  ``lr_scheduler``).
"""

from __future__ import annotations

import math
import multiprocessing

import pytest
import torch
from transformers.debug_utils import DebugOption

from opake.api.transformers.trainer._training_arguments import _DP_OPTIMIZERS
from opake.exceptions import ConfigurationError
from opake.transformers.trainer import TrainingArguments


class TestLegacyAliases:
    """Deprecated HF kwargs are not accepted on the standalone dataclass."""

    def test_no_cuda_no_longer_accepted(self):
        # ``TrainingArguments`` is a standalone dataclass (no HF
        # inheritance).  ``no_cuda`` was a deprecated HF alias for
        # ``use_cpu``; passing it raises ``TypeError`` (unexpected kwarg).
        # Users should use ``use_cpu`` directly.
        with pytest.raises(TypeError, match="no_cuda"):
            TrainingArguments(privacy_noise_multiplier=1.0, no_cuda=True)


class TestMetricForBestModelDefaults:
    """``metric_for_best_model`` is auto-defaulted to ``"loss"`` when needed."""

    def test_load_best_model_at_end_defaults_metric(self, tmp_path):
        args = TrainingArguments(
            privacy_noise_multiplier=1.0,
            output_dir=str(tmp_path),
            load_best_model_at_end=True,
            save_strategy="steps",
            save_steps=2,
            eval_strategy="steps",
            eval_steps=2,
        )
        assert args.metric_for_best_model == "loss"
        # Loss-suffixed metric ⇒ greater_is_better=False.
        assert args.greater_is_better is False


class TestCadenceAlignment:
    """``load_best_model_at_end`` requires save / eval cadence alignment."""

    def test_mismatched_strategies_raises(self, tmp_path):
        # ``save_strategy=steps`` + ``eval_strategy=epoch`` ⇒ HF parity:
        # the save and eval strategies must match.
        with pytest.raises(ValueError, match="save and eval strategy to match"):
            TrainingArguments(
                privacy_noise_multiplier=1.0,
                output_dir=str(tmp_path),
                load_best_model_at_end=True,
                save_strategy="steps",
                save_steps=2,
                eval_strategy="epoch",
            )

    def test_save_steps_not_multiple_of_eval_steps_raises(self, tmp_path):
        with pytest.raises(ValueError, match="round multiple"):
            TrainingArguments(
                privacy_noise_multiplier=1.0,
                output_dir=str(tmp_path),
                load_best_model_at_end=True,
                save_strategy="steps",
                save_steps=3,
                eval_strategy="steps",
                eval_steps=2,
            )

    def test_aligned_step_strategies_pass(self, tmp_path):
        # 6 % 2 == 0 ⇒ aligned.
        args = TrainingArguments(
            privacy_noise_multiplier=1.0,
            output_dir=str(tmp_path),
            load_best_model_at_end=True,
            save_strategy="steps",
            save_steps=6,
            eval_strategy="steps",
            eval_steps=2,
        )
        assert args.save_steps == 6
        assert args.eval_steps == 2

    def test_save_strategy_best_skips_alignment_check(self, tmp_path):
        # ``save_strategy="best"`` is the only HF strategy that pairs
        # with any eval cadence regardless of save_steps.
        args = TrainingArguments(
            privacy_noise_multiplier=1.0,
            output_dir=str(tmp_path),
            load_best_model_at_end=True,
            save_strategy="best",
            eval_strategy="steps",
            eval_steps=3,
        )
        assert args.save_strategy == "best"


class TestStrategyCoercion:
    """The four strategy enum fields are coerced to canonical strings."""

    @pytest.mark.parametrize(
        ("field", "value", "expected"),
        [
            ("eval_strategy", "steps", "steps"),
            ("eval_strategy", "epoch", "epoch"),
            ("eval_strategy", "no", "no"),
            ("save_strategy", "steps", "steps"),
            ("save_strategy", "epoch", "epoch"),
            ("save_strategy", "no", "no"),
            ("save_strategy", "best", "best"),
            ("logging_strategy", "steps", "steps"),
            ("logging_strategy", "epoch", "epoch"),
            ("lr_scheduler", "linear", "linear"),
            ("lr_scheduler", "cosine", "cosine"),
        ],
    )
    def test_canonical_string_coercion(self, field, value, expected):
        kwargs = {field: value}
        # Side-args needed for some coercions (eval_strategy="steps" needs eval_steps).
        if field == "eval_strategy" and value == "steps":
            kwargs["eval_steps"] = 1
        # ``save_strategy='best'`` is a cross-field invariant: requires
        # ``eval_strategy != 'no'`` so the trainer can pick a best
        # checkpoint.  Pair with ``eval_strategy='steps'`` so the
        # coercion path exits cleanly.
        if field == "save_strategy" and value == "best":
            kwargs.update(eval_strategy="steps", eval_steps=1, save_steps=1)
        args = TrainingArguments(privacy_noise_multiplier=1.0, **kwargs)
        assert getattr(args, field) == expected

    def test_unknown_strategy_raises(self):
        with pytest.raises(
            ConfigurationError, match=r"eval_strategy=.*expected one of"
        ):
            TrainingArguments(
                privacy_noise_multiplier=1.0, eval_strategy="completely-bogus"
            )

    def test_unknown_lr_scheduler_raises(self):
        with pytest.raises(ValueError, match="not a valid SchedulerType"):
            TrainingArguments(
                privacy_noise_multiplier=1.0, lr_scheduler="not-a-real-scheduler"
            )


class TestDataLoaderArguments:
    def test_implicit_eval_batch_uses_explicit_microbatch(self):
        args = TrainingArguments(
            privacy_noise_multiplier=1.0,
            per_device_train_batch_size=8,
            microbatch_size=2,
        )
        assert args.per_device_eval_batch_size == 2

    def test_implicit_eval_batch_falls_back_to_logical_train_batch(self):
        args = TrainingArguments(
            privacy_noise_multiplier=1.0,
            per_device_train_batch_size=8,
        )
        assert args.per_device_eval_batch_size == 8

    def test_explicit_eval_batch_overrides_microbatch_default(self):
        args = TrainingArguments(
            privacy_noise_multiplier=1.0,
            per_device_train_batch_size=8,
            microbatch_size=2,
            per_device_eval_batch_size=6,
        )
        assert args.per_device_eval_batch_size == 6

    def test_multiprocessing_context_accepts_platform_method(self):
        context = multiprocessing.get_all_start_methods()[0]
        args = TrainingArguments(
            privacy_noise_multiplier=1.0,
            dataloader_num_workers=1,
            dataloader_multiprocessing_context=context,
        )
        assert args.dataloader_multiprocessing_context == context

    def test_multiprocessing_context_rejects_unknown_method(self):
        with pytest.raises(
            ConfigurationError, match="dataloader_multiprocessing_context"
        ):
            TrainingArguments(
                privacy_noise_multiplier=1.0,
                dataloader_num_workers=1,
                dataloader_multiprocessing_context="not-a-method",
            )

    def test_multiprocessing_context_requires_workers(self):
        context = multiprocessing.get_all_start_methods()[0]
        with pytest.raises(ValueError, match="dataloader_num_workers > 0"):
            TrainingArguments(
                privacy_noise_multiplier=1.0,
                dataloader_multiprocessing_context=context,
            )

    def test_out_of_order_loading_is_rejected(self):
        with pytest.raises(
            ConfigurationError, match="dataloader_in_order must be True"
        ):
            TrainingArguments(
                privacy_noise_multiplier=1.0,
                dataloader_in_order=False,
            )


class TestEvalAccumulationArguments:
    @pytest.mark.parametrize("value", [0, -1, False])
    def test_non_positive_or_boolean_window_is_rejected(self, value):
        with pytest.raises(
            ConfigurationError,
            match="eval_accumulation_steps must be a positive integer or None",
        ):
            TrainingArguments(
                privacy_noise_multiplier=1.0,
                eval_accumulation_steps=value,
            )


class TestHFFieldNormalization:
    def test_debug_string_coerces_to_debug_options(self):
        args = TrainingArguments(
            privacy_noise_multiplier=1.0, debug="underflow_overflow"
        )
        assert args.debug == [DebugOption.UNDERFLOW_OVERFLOW]

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(True, "all"), (False, "no"), ("non_padding", "non_padding")],
    )
    def test_include_num_input_tokens_seen_normalizes(self, value, expected):
        args = TrainingArguments(
            privacy_noise_multiplier=1.0, include_num_input_tokens_seen=value
        )
        assert args.include_num_input_tokens_seen == expected

    def test_include_num_input_tokens_seen_invalid_raises(self):
        with pytest.raises(ValueError, match="include_num_input_tokens_seen"):
            TrainingArguments(
                privacy_noise_multiplier=1.0, include_num_input_tokens_seen="sometimes"
            )

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, []),
            ("none", []),
            ([], []),
            (["none"], []),
            ("tensorboard", ["tensorboard"]),
            (["wandb"], ["wandb"]),
        ],
    )
    def test_report_to_normalizes(self, value, expected):
        args = TrainingArguments(privacy_noise_multiplier=1.0, report_to=value)
        assert args.report_to == expected

    def test_lr_scheduler_kwargs_json_object_is_parsed(self):
        args = TrainingArguments(
            privacy_noise_multiplier=1.0,
            lr_scheduler="cosine",
            lr_scheduler_kwargs='{"num_cycles": "2"}',
        )
        assert args.lr_scheduler_kwargs == {"num_cycles": 2}

    def test_torch_empty_cache_steps_must_be_positive_int(self):
        with pytest.raises(ValueError, match="torch_empty_cache_steps"):
            TrainingArguments(privacy_noise_multiplier=1.0, torch_empty_cache_steps=0)

    def test_lr_scheduler_accepts_schedule_recipe(self):
        # ``lr_scheduler`` accepts a Schedule recipe in addition to
        # HF-style name strings; the SchedulerType-enum coercion is
        # skipped for callables.
        from opake.scheduling import cosine_schedule

        recipe = cosine_schedule(init_value=1e-3, end_value=0.0, transition_steps=100)
        args = TrainingArguments(privacy_noise_multiplier=1.0, lr_scheduler=recipe)
        assert args.lr_scheduler is recipe

    def test_lr_scheduler_rejects_non_callable_non_string(self):
        with pytest.raises(TypeError, match="lr_scheduler must be"):
            TrainingArguments(privacy_noise_multiplier=1.0, lr_scheduler=42)


class TestUnknownKwargs:
    @pytest.mark.parametrize(
        "payload",
        [
            {"group_by_length": True},
            {"deepspeed": "ds_config.json"},
            {"optim_target_modules": ["q_proj"]},
            {"ddp_find_unused_parameters": True},
            {"unknown_knob_for_test_only": 1},
        ],
    )
    def test_unknown_kwargs_raise_type_error(self, payload):
        with pytest.raises(TypeError):
            TrainingArguments(privacy_noise_multiplier=1.0, **payload)

    def test_dpargs_does_not_inherit_from_hf_training_arguments(self):
        from transformers.training_args import TrainingArguments as HFTrainingArguments

        assert HFTrainingArguments not in TrainingArguments.__mro__

    @pytest.mark.parametrize(
        "backend", ["nccl", "gloo", "mpi", "xccl", "hccl", "cncl", "mccl"]
    )
    def test_ddp_backend_surface_accepts_hf_backend_values(self, backend):
        args = TrainingArguments(privacy_noise_multiplier=1.0, ddp_backend=backend)
        assert args.ddp_backend == backend

    @pytest.mark.parametrize("backend", ["invalid-backend"])
    def test_ddp_backend_unsupported_value_raises(self, backend):
        with pytest.raises(ValueError, match="ddp_backend"):
            TrainingArguments(privacy_noise_multiplier=1.0, ddp_backend=backend)


class TestOptimizerSupportSurface:
    @pytest.mark.parametrize("optim", _DP_OPTIMIZERS)
    def test_supported_dp_optimizers_construct(self, optim):
        args = TrainingArguments(privacy_noise_multiplier=1.0, optim=optim)
        assert args.optim == optim

    @pytest.mark.parametrize(
        "optim",
        [
            "adamw_torch_xla",
            "adamw_apex_fused",
            "adamw_bnb_8bit",
            "paged_adamw_32bit",
            "galore_adamw",
            "lomo",
            "grokadamw",
            "stable_adamw",
        ],
    )
    def test_unsupported_optimizers_raise_with_supported_list(self, optim):
        with pytest.raises(ValueError, match="expected one of") as exc_info:
            TrainingArguments(privacy_noise_multiplier=1.0, optim=optim)
        message = str(exc_info.value)
        assert optim in message
        # Whitelist error lists the supported set.
        assert "expected one of" in message


class TestEvalStepsFallback:
    """``eval_strategy="steps"`` without ``eval_steps`` falls back to ``logging_steps``."""

    def test_eval_steps_falls_back_to_logging_steps(self):
        args = TrainingArguments(
            privacy_noise_multiplier=1.0, eval_strategy="steps", logging_steps=10
        )
        assert args.eval_steps == 10

    def test_eval_strategy_steps_zero_logging_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            TrainingArguments(
                privacy_noise_multiplier=1.0, eval_strategy="steps", logging_steps=0
            )


class TestMaxGradNorm:
    """``clipping_norm`` is the DP clip bound: scalar or per-group dict."""

    def test_positive_scalar_accepted(self):
        args = TrainingArguments(privacy_noise_multiplier=1.0, clipping_norm=2.5)
        assert args.clipping_norm == 2.5

    def test_non_positive_scalar_raises(self):
        with pytest.raises(ValueError, match="clipping_norm"):
            TrainingArguments(privacy_noise_multiplier=1.0, clipping_norm=0.0)

    def test_dict_fallback_only_normalizes_to_float(self):
        args = TrainingArguments(
            privacy_noise_multiplier=1.0, clipping_norm={"fallback": 2.0}
        )
        assert args.clipping_norm == 2.0

    def test_dict_per_group_accepted(self):
        args = TrainingArguments(
            privacy_noise_multiplier=1.0,
            clipping_norm={"fallback": 1.0, "linear": 2.0, "bias": 0.5},
        )
        assert args.clipping_norm == {
            "fallback": 1.0,
            "linear": 2.0,
            "bias": 0.5,
        }

    def test_json_string_object_parsed(self):
        args = TrainingArguments(
            privacy_noise_multiplier=1.0,
            clipping_norm='{"fallback": 1.0, "x": 2.0}',
        )
        assert args.clipping_norm == {"fallback": 1.0, "x": 2.0}

    def test_dict_missing_fallback_raises(self):
        with pytest.raises(ValueError, match="fallback"):
            TrainingArguments(
                privacy_noise_multiplier=1.0, clipping_norm={"linear": 1.0}
            )

    @pytest.mark.parametrize("value", [math.inf, "inf", "Infinity", "1e999"])
    def test_infinity_disables_clipping(self, value):
        # ``math.inf`` is the single canonical no-clip value (string spellings
        # parse to it via float()).  Allowed only for a non-private baseline.
        args = TrainingArguments(clipping_norm=value, privacy_noise_multiplier=0.0)
        assert math.isinf(args.clipping_norm)

    def test_none_rejected(self):
        # There is exactly one way to disable clipping (math.inf); None is not
        # it and must point users at the canonical form.
        with pytest.raises(ValueError, match=r"math.inf"):
            TrainingArguments(clipping_norm=None, privacy_noise_multiplier=0.0)

    @pytest.mark.parametrize("nm", [None, 0.5, 1.0])
    def test_disabled_clipping_with_noise_rejected(self, nm):
        # Disabling clipping is unsound with noise (nm > 0) or calibration
        # (nm is None) — infinite sensitivity ⇒ infinite noise stddev.
        # Supply ``privacy_target_epsilon`` so the nm=None branch passes the
        # "at least one of NM/target_eps" validation before the clipping
        # check fires.
        with pytest.raises(ValueError, match="non-private baseline"):
            TrainingArguments(
                clipping_norm=math.inf,
                privacy_noise_multiplier=nm,
                privacy_target_epsilon=8.0 if nm is None else None,
            )

    def test_disabled_per_group_clipping_with_noise_rejected(self):
        with pytest.raises(ValueError, match="non-private baseline"):
            TrainingArguments(
                clipping_norm={"fallback": math.inf, "linear": 1.0},
                privacy_noise_multiplier=1.0,
            )

    def test_garbage_string_raises(self):
        for bad in ("banana", "none"):
            with pytest.raises(ValueError, match="clipping_norm"):
                TrainingArguments(privacy_noise_multiplier=1.0, clipping_norm=bad)


class TestClippingAndSamplingSurfaces:
    """``clipping_mode`` / ``sampling_mode`` and JSON-style arg blobs."""

    def test_invalid_clipping_mode_raises(self):
        with pytest.raises(ValueError, match="clipping_mode"):
            TrainingArguments(privacy_noise_multiplier=1.0, clipping_mode="unknown")

    def test_invalid_sampling_mode_raises(self):
        with pytest.raises(ValueError, match="sampling_mode"):
            TrainingArguments(
                privacy_noise_multiplier=1.0, sampling_mode="not_a_real_sampler"
            )

    def test_sampling_mode_not_allowed_for_mechanism_raises(self):
        # ``sequential`` is a valid sampler name on the wider surface but
        # not paired with the default ``gaussian`` mechanism.
        with pytest.raises(ValueError, match="sampling_mode"):
            TrainingArguments(privacy_noise_multiplier=1.0, sampling_mode="sequential")

    def test_gaussian_accepts_k_out_of_t(self):
        args = TrainingArguments(
            privacy_noise_multiplier=1.0,
            sampling_mode="k_out_of_t",
            sampling_kwargs={"k": 2, "allocation": "block"},
        )
        assert args.sampling_mode == "k_out_of_t"

    @pytest.mark.parametrize("mechanism", ["mf_identity", "mf_band"])
    def test_horizon_mechanism_rejects_privacy_early_stopping(self, mechanism):
        with pytest.raises(ValueError, match="whole-horizon"):
            TrainingArguments(
                privacy_noise_multiplier=1.0,
                privacy_target_epsilon=2.0,
                privacy_noise_mechanism=mechanism,
            )

    def test_k_out_of_t_rejects_privacy_early_stopping(self):
        with pytest.raises(ValueError, match="whole-horizon"):
            TrainingArguments(
                privacy_noise_multiplier=1.0,
                privacy_target_epsilon=2.0,
                sampling_mode="k_out_of_t",
                sampling_kwargs={"k": 2, "allocation": "block"},
            )

    def test_k_out_of_t_requires_k_and_allocation(self):
        with pytest.raises(ValueError, match="k"):
            TrainingArguments(
                privacy_noise_multiplier=1.0,
                sampling_mode="k_out_of_t",
            )

    def test_k_out_of_t_accepts_total_allocation(self):
        args = TrainingArguments(
            privacy_noise_multiplier=1.0,
            sampling_mode="k_out_of_t",
            sampling_kwargs={"k": 2, "allocation": "total"},
        )
        assert args.sampling_kwargs["allocation"] == "total"

    def test_k_out_of_t_rejects_truncated_poisson_kwargs(self):
        with pytest.raises(ValueError, match="truncated_batch_size"):
            TrainingArguments(
                privacy_noise_multiplier=1.0,
                sampling_mode="k_out_of_t",
                sampling_kwargs={
                    "k": 2,
                    "allocation": "block",
                    "truncated_batch_size": 4,
                },
            )

    def test_mf_identity_accepts_balls_in_bins(self):
        args = TrainingArguments(
            privacy_noise_multiplier=1.0,
            privacy_noise_mechanism="mf_identity",
            sampling_mode="balls_in_bins",
        )
        assert args.sampling_mode == "balls_in_bins"

    def test_sampling_mode_auto_resolves_to_poisson_for_gaussian(self):
        # ``"auto"`` (the default) resolves to the canonical sampler for
        # the chosen mechanism — ``"poisson"`` for the DP-SGD
        # ``"gaussian"`` baseline.
        args = TrainingArguments(privacy_noise_multiplier=1.0)
        assert args.sampling_mode == "poisson"
        args_explicit = TrainingArguments(
            privacy_noise_multiplier=1.0, sampling_mode="auto"
        )
        assert args_explicit.sampling_mode == "poisson"

    def test_clipping_kwargs_json_string_parsed(self):
        args = TrainingArguments(
            privacy_noise_multiplier=1.0,
            clipping_mode="adaptive",
            clipping_kwargs='{"clipping_norm_max": 9.0}',
        )
        assert args.clipping_kwargs == {"clipping_norm_max": 9.0}

    def test_sampling_kwargs_json_string_parsed(self):
        args = TrainingArguments(
            privacy_noise_multiplier=1.0, sampling_kwargs='{"truncated_batch_size": 8}'
        )
        assert args.sampling_kwargs == {"truncated_batch_size": 8}

    def test_sampling_kwargs_rejects_bands(self):
        """``bands`` is owned by the strategy, not the sampler kwargs."""
        with pytest.raises(ValueError, match="privacy-derived keys"):
            TrainingArguments(
                privacy_noise_multiplier=1.0,
                privacy_noise_mechanism="mf_band",
                sampling_kwargs={"bands": 4},
            )

    def test_sampling_kwargs_rejects_sampling_prob(self):
        """``sampling_prob`` is derived by the amplifier, not user input."""
        with pytest.raises(ValueError, match="privacy-derived keys"):
            TrainingArguments(
                privacy_noise_multiplier=1.0,
                privacy_noise_mechanism="mf_band",
                sampling_kwargs={"sampling_prob": 0.1},
            )

    @pytest.mark.parametrize(
        ("field", "selector"),
        [
            ("clipping_kwargs", {}),
            ("sampling_kwargs", {}),
            ("privacy_noise_mechanism_kwargs", {}),
            ("noise_calibration_kwargs", {}),
        ],
    )
    def test_privacy_kwargs_reject_unknown_keys(self, field, selector):
        with pytest.raises(
            ValueError,
            match=rf"{field} contains unsupported keys.*unknown.*Allowed:",
        ):
            TrainingArguments(
                privacy_noise_multiplier=1.0,
                **selector,
                **{field: {"unknown": 1}},
            )

    @pytest.mark.parametrize(
        ("field", "value", "selector"),
        [
            (
                "clipping_kwargs",
                {"target_clipping_rate": 0.5},
                {"clipping_mode": "adaptive"},
            ),
            (
                "clipping_kwargs",
                {"norm_max": 10.0},
                {"clipping_mode": "adaptive"},
            ),
            ("sampling_kwargs", {"max_batch_size": 8}, {}),
        ],
    )
    def test_privacy_kwargs_reject_legacy_aliases(self, field, value, selector):
        with pytest.raises(ValueError, match=rf"{field} contains unsupported keys"):
            TrainingArguments(
                privacy_noise_multiplier=1.0,
                **selector,
                **{field: value},
            )

    def test_adaptive_clipping_accepts_public_factory_names(self):
        args = TrainingArguments(
            privacy_noise_multiplier=1.0,
            clipping_mode="adaptive",
            clipping_kwargs={
                "target_quantile": 0.9,
                "clipping_norm_max": 50.0,
            },
        )
        assert args.clipping_kwargs == {
            "target_quantile": 0.9,
            "clipping_norm_max": 50.0,
        }


class TestMechanismAndSamplerDefaults:
    """``privacy_noise_mechanism`` surface + ``sampling_mode='auto'`` resolver."""

    def test_default_gaussian_resolves_to_poisson(self):
        args = TrainingArguments(privacy_noise_multiplier=1.0)
        assert args.privacy_noise_mechanism == "gaussian"
        assert args.sampling_mode == "poisson"

    def test_unknown_mechanism_rejected(self):
        with pytest.raises(ValueError, match="privacy_noise_mechanism"):
            TrainingArguments(
                privacy_noise_multiplier=1.0, privacy_noise_mechanism="laplace"
            )

    @pytest.mark.parametrize(
        ("mechanism", "expected_sampler"),
        [
            ("mf_band", "cyclic_poisson"),
            ("mf_blt", "balls_in_bins"),
            ("mf_bisr", "balls_in_bins"),
            ("mf_bsr", "balls_in_bins"),
            ("mf_lambda_cgd", "balls_in_bins"),
            ("mf_identity", "poisson"),
        ],
    )
    def test_auto_resolves_per_mechanism(self, mechanism, expected_sampler):
        args = TrainingArguments(
            privacy_noise_multiplier=1.0, privacy_noise_mechanism=mechanism
        )
        assert args.sampling_mode == expected_sampler

    def test_mf_band_accepts_b_min_sep_override(self):
        args = TrainingArguments(
            privacy_noise_multiplier=1.0,
            privacy_noise_mechanism="mf_band",
            sampling_mode="b_min_sep",
        )
        assert args.sampling_mode == "b_min_sep"

    @pytest.mark.parametrize("bands", [1, 4])
    def test_mf_band_rejects_poisson_override(self, bands):
        """``mf_band`` requires ``cyclic_poisson`` or ``b_min_sep``; a plain
        Poisson sampler does not realize BandMF's grouped participation
        pattern (issue #776)."""
        with pytest.raises(ValueError, match="sampling_mode"):
            TrainingArguments(
                privacy_noise_multiplier=1.0,
                privacy_noise_mechanism="mf_band",
                privacy_noise_mechanism_kwargs={"bands": bands},
                sampling_mode="poisson",
            )

    def test_mf_blt_rejects_poisson_override(self):
        with pytest.raises(ValueError, match="sampling_mode"):
            TrainingArguments(
                privacy_noise_multiplier=1.0,
                privacy_noise_mechanism="mf_blt",
                sampling_mode="poisson",
            )

    def test_mf_auto_resolves_adaptive_to_fixed(self, caplog):
        """``clipping_mode='adaptive'`` paired with an MF mechanism is auto-
        resolved to ``'fixed'`` with a warning rather than raising — most
        users inherit ``adaptive`` from a preset default and shouldn't be
        blocked from running MF."""
        import logging

        with caplog.at_level(logging.WARNING):
            args = TrainingArguments(
                privacy_noise_multiplier=1.0,
                privacy_noise_mechanism="mf_band",
                clipping_mode="adaptive",
            )
        assert args.clipping_mode == "fixed"
        assert any(
            "clipping_mode='adaptive' is incompatible" in record.getMessage()
            and "mf_band" in record.getMessage()
            for record in caplog.records
        ), f"expected a clipping_mode auto-resolve warning; got {caplog.records!r}"

    def test_mf_keeps_explicit_fixed_clipping(self):
        """A user-supplied ``clipping_mode='fixed'`` is not touched by the
        MF auto-resolve."""
        args = TrainingArguments(
            privacy_noise_multiplier=1.0,
            privacy_noise_mechanism="mf_band",
            clipping_mode="fixed",
        )
        assert args.clipping_mode == "fixed"

    def test_mf_kwargs_auto_filled(self):
        args = TrainingArguments(
            privacy_noise_multiplier=1.0, privacy_noise_mechanism="mf_band"
        )
        assert args.privacy_noise_mechanism_kwargs == {"bands": 16}

    def test_mf_user_kwargs_win_on_collision(self):
        args = TrainingArguments(
            privacy_noise_multiplier=1.0,
            privacy_noise_mechanism="mf_band",
            privacy_noise_mechanism_kwargs={"bands": 4},
        )
        assert args.privacy_noise_mechanism_kwargs == {"bands": 4}

    def test_mf_user_kwargs_merge_with_defaults(self):
        args = TrainingArguments(
            privacy_noise_multiplier=1.0,
            privacy_noise_mechanism="mf_bsr",
            privacy_noise_mechanism_kwargs={"bandwidth": 16},
        )
        # Defaults supply ``alpha``/``beta``; user override wins on
        # ``bandwidth``.
        assert args.privacy_noise_mechanism_kwargs == {
            "bandwidth": 16,
            "alpha": 1.0,
            "beta": 0.9,
        }


class TestRemovedGaussianBoundKwarg:
    """Removed Gaussian bound configuration is rejected at construction."""

    def test_bound_mechanism_kwarg_is_rejected(self):
        with pytest.raises(ConfigurationError, match=r"bound.*unsupported"):
            TrainingArguments(
                privacy_noise_multiplier=1.0,
                privacy_noise_mechanism_kwargs={"bound": 3.0},
            )


class TestGaussianComputeDtype:
    """Gaussian ``compute_dtype`` accepts native and string-shaped config."""

    @pytest.mark.parametrize(
        ("options", "expected"),
        [
            ({"compute_dtype": torch.float16}, torch.float16),
            ("compute_dtype=float32", torch.float32),
            ('{"compute_dtype": "torch.float64"}', torch.float64),
        ],
    )
    def test_normalizes_supported_dtype(self, options, expected):
        args = TrainingArguments(
            privacy_noise_multiplier=1.0,
            privacy_noise_mechanism_kwargs=options,
        )
        assert args.privacy_noise_mechanism_kwargs == {"compute_dtype": expected}

    @pytest.mark.parametrize("value", ["float8_e4m3fn", torch.int64, None])
    def test_rejects_unsupported_dtype(self, value):
        with pytest.raises(ConfigurationError, match="compute_dtype"):
            TrainingArguments(
                privacy_noise_multiplier=1.0,
                privacy_noise_mechanism_kwargs={"compute_dtype": value},
            )


class TestNoiseCalibrationKwargs:
    """``noise_calibration_kwargs`` defaults and overrides."""

    def test_defaults_and_override(self):
        base = TrainingArguments(privacy_noise_multiplier=1.0)
        assert base.noise_calibration_kwargs["min"] == 0.11
        assert base.noise_calibration_kwargs["max"] == 10.0
        assert base.noise_calibration_kwargs["tolerance"] == 1e-3
        tuned = TrainingArguments(
            privacy_noise_multiplier=1.0,
            noise_calibration_kwargs={"min": 0.05, "tolerance": 1e-2},
        )
        assert tuned.noise_calibration_kwargs["min"] == 0.05
        assert tuned.noise_calibration_kwargs["max"] == 10.0
        assert tuned.noise_calibration_kwargs["tolerance"] == 1e-2


# ---------------------------------------------------------------------------
# Unified dict-field input contract (Mapping / JSON / HF comma form)
# ---------------------------------------------------------------------------


from collections.abc import MutableMapping, MutableSequence  # noqa: E402


class _FakeDictConfig(MutableMapping):
    """Minimal ``Mapping`` stand-in for OmegaConf's ``DictConfig`` in tests."""

    def __init__(self, data):
        self._data = dict(data)

    def __getitem__(self, k):
        return self._data[k]

    def __setitem__(self, k, v):
        self._data[k] = v

    def __delitem__(self, k):
        del self._data[k]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


class _FakeListConfig(MutableSequence):
    """Minimal ``Sequence`` stand-in for OmegaConf's ``ListConfig``."""

    def __init__(self, data):
        self._data = list(data)

    def __getitem__(self, i):
        return self._data[i]

    def __setitem__(self, i, v):
        self._data[i] = v

    def __delitem__(self, i):
        del self._data[i]

    def __len__(self):
        return len(self._data)

    def insert(self, i, v):
        self._data.insert(i, v)


class TestDictFieldInputContract:
    """All dict-shaped fields accept Mapping / JSON / HF comma form / None."""

    def test_mapping_input_materialises_to_dict(self):
        args = TrainingArguments(
            privacy_noise_multiplier=1.0,
            clipping_mode="adaptive",
            clipping_kwargs=_FakeDictConfig({"target_quantile": 0.5}),
        )
        assert isinstance(args.clipping_kwargs, dict)
        assert args.clipping_kwargs == {"target_quantile": 0.5}

    def test_nested_listconfig_materialises_to_list(self):
        # A Mapping containing a non-list Sequence (e.g. OmegaConf
        # ListConfig).  The nested container must come back as a plain
        # list — silently fixes the OmegaConf nested-ListConfig leak.
        nested = _FakeDictConfig({"include": _FakeListConfig([1, 2, 3])})
        args = TrainingArguments(
            privacy_noise_multiplier=1.0,
            performance_kernels_config=nested,
        )
        assert args.performance_kernels_config == {"include": [1, 2, 3]}
        assert isinstance(args.performance_kernels_config["include"], list)

    def test_json_string_input_parses(self):
        args = TrainingArguments(
            privacy_noise_multiplier=1.0,
            clipping_mode="adaptive",
            clipping_kwargs='{"target_quantile": 0.5}',
        )
        assert args.clipping_kwargs == {"target_quantile": 0.5}

    def test_hf_comma_string_input_parses(self):
        args = TrainingArguments(
            privacy_noise_multiplier=1.0,
            clipping_mode="adaptive",
            clipping_kwargs="target_quantile=0.5,clipping_norm_max=10.0",
        )
        assert args.clipping_kwargs == {
            "target_quantile": 0.5,
            "clipping_norm_max": 10.0,
        }

    def test_optim_args_accepts_mapping(self):
        args = TrainingArguments(
            privacy_noise_multiplier=1.0, optim_args={"weight_decay": 0.01}
        )
        assert args.optim_args == {"weight_decay": 0.01}

    def test_optim_args_accepts_json_string(self):
        args = TrainingArguments(
            privacy_noise_multiplier=1.0, optim_args='{"weight_decay": 0.01}'
        )
        assert args.optim_args == {"weight_decay": 0.01}

    def test_optim_args_accepts_hf_comma_string(self):
        args = TrainingArguments(
            privacy_noise_multiplier=1.0, optim_args="weight_decay=0.01,nesterov=True"
        )
        assert args.optim_args == {"weight_decay": 0.01, "nesterov": True}

    def test_optim_args_none_stays_none(self):
        args = TrainingArguments(privacy_noise_multiplier=1.0, optim_args=None)
        assert args.optim_args is None

    def test_clipping_norm_accepts_dictconfig(self):
        args = TrainingArguments(
            privacy_noise_multiplier=1.0,
            clipping_norm=_FakeDictConfig({"fallback": 1.0, "attn": 0.5}),
        )
        assert args.clipping_norm == {"fallback": 1.0, "attn": 0.5}
