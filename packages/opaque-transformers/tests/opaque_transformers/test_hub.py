"""Tests for the (minimal) Hub publishing surface in Trainer.

Covers the publish-only surface (no in-training auto-push):

- ``hub_model_id`` is ``None`` when ``push_to_hub=False`` (default).
- ``init_hf_repo`` is called at construction when ``push_to_hub=True``, setting
  ``hub_model_id`` (honouring ``hub_model_id`` / ``hub_token`` /
  ``hub_private_repo``).
- ``push_to_hub`` calls ``save_model`` + ``create_model_card`` + ``upload_folder``.
- ``create_model_card`` writes ``README.md`` containing ``differential-privacy``
  and ``opaque`` tags and the ``## Privacy budget`` section.
- The DP section in ``README.md`` is replaced (not appended) on repeated calls.
- ``save_model()`` (user call) triggers ``push_to_hub`` when ``push_to_hub=True``.
- ``train_dataset`` / ``eval_dataset`` public properties.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

import opaque.api.transformers.trainer._hub as _hub
from opaque.transformers.trainer import Trainer, TrainingArguments

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _args(tmp_path, **overrides) -> TrainingArguments:
    defaults = {
        "output_dir": str(tmp_path),
        "per_device_train_batch_size": 1,
        "max_steps": 1,
        "num_train_epochs": 1,
        "save_strategy": "no",
        "use_cpu": True,
        "privacy_target_epsilon": 10.0,
        "privacy_noise_multiplier": 1.0,
    }
    defaults.update(overrides)
    return TrainingArguments(**defaults)


def _tiny_trainer(tmp_path, **arg_overrides) -> Trainer:
    model = torch.nn.Linear(4, 2)
    # TrainingSummary.from_trainer expects trainer.model.config._name_or_path.
    model.config = type("DummyConfig", (), {"_name_or_path": "dummy/local-model"})()
    args = _args(tmp_path, **arg_overrides)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=[{"x": torch.zeros(4)}],
        eval_dataset=None,
    )
    return trainer


# ---------------------------------------------------------------------------
# Public property surface
# ---------------------------------------------------------------------------


class TestPublicProperties:
    def test_train_dataset_property(self, tmp_path):
        trainer = _tiny_trainer(tmp_path)
        ds = [{"x": torch.zeros(4)}]
        trainer._train_dataset = ds
        assert trainer.train_dataset is ds

    def test_eval_dataset_property(self, tmp_path):
        trainer = _tiny_trainer(tmp_path)
        ds = [{"y": torch.zeros(2)}]
        trainer._eval_dataset = ds
        assert trainer.eval_dataset is ds


# ---------------------------------------------------------------------------
# Instance variables when push_to_hub=False (default)
# ---------------------------------------------------------------------------


class TestNoPushDefault:
    def test_hub_model_id_is_none(self, tmp_path):
        trainer = _tiny_trainer(tmp_path)
        assert trainer.hub_model_id is None


class TestInitHfRepo:
    def test_called_at_construction(self, tmp_path):
        mock_url = MagicMock()
        mock_url.repo_id = "myorg/myrepo"
        with patch(
            "opaque.api.transformers.trainer._hub._create_repo", return_value=mock_url
        ) as mock_create:
            trainer = _tiny_trainer(
                tmp_path, push_to_hub=True, hub_model_id="myorg/myrepo"
            )
        mock_create.assert_called_once_with("myorg/myrepo", token=None, private=None)
        assert trainer.hub_model_id == "myorg/myrepo"

    def test_repo_name_defaults_to_output_dir_basename(self, tmp_path):
        mock_url = MagicMock()
        mock_url.repo_id = f"myorg/{tmp_path.name}"
        with patch(
            "opaque.api.transformers.trainer._hub._create_repo", return_value=mock_url
        ) as mock_create:
            _tiny_trainer(tmp_path, push_to_hub=True)
        # repo_name should be the absolute basename of output_dir
        assert mock_create.call_args[0][0] == tmp_path.name

    def test_hub_token_passed_to_create_repo(self, tmp_path):
        mock_url = MagicMock()
        mock_url.repo_id = "org/repo"
        with patch(
            "opaque.api.transformers.trainer._hub._create_repo", return_value=mock_url
        ) as mock_create:
            _tiny_trainer(
                tmp_path, push_to_hub=True, hub_model_id="org/repo", hub_token="tok123"
            )
        mock_create.assert_called_once_with("org/repo", token="tok123", private=None)

    def test_hub_private_repo_passed_to_create_repo(self, tmp_path):
        mock_url = MagicMock()
        mock_url.repo_id = "org/private-repo"
        with patch(
            "opaque.api.transformers.trainer._hub._create_repo", return_value=mock_url
        ) as mock_create:
            _tiny_trainer(
                tmp_path,
                push_to_hub=True,
                hub_model_id="org/private-repo",
                hub_private_repo=True,
            )
        mock_create.assert_called_once_with(
            "org/private-repo", token=None, private=True
        )


class TestPushToHub:
    def _run_push(self, tmp_path, **overrides):
        mock_url = MagicMock()
        mock_url.repo_id = "org/repo"
        with patch(
            "opaque.api.transformers.trainer._hub._create_repo", return_value=mock_url
        ):
            trainer = _tiny_trainer(
                tmp_path, push_to_hub=True, hub_model_id="org/repo", **overrides
            )
        trainer.hub_model_id = "org/repo"

        with (
            patch.object(trainer, "save_model") as mock_save,
            patch(
                "opaque.api.transformers.trainer._hub.create_model_card"
            ) as mock_card,
            patch("opaque.api.transformers.trainer._hub._upload_folder") as mock_upload,
        ):
            result = _hub.push_to_hub(trainer, commit_message="End of training")

        return trainer, mock_save, mock_card, mock_upload, result

    def test_save_model_called_with_internal_flag(self, tmp_path):
        _, mock_save, _, _, _ = self._run_push(tmp_path)
        mock_save.assert_called_once_with(_internal_call=True)

    def test_create_model_card_called(self, tmp_path):
        _, _, mock_card, _, _ = self._run_push(tmp_path)
        mock_card.assert_called_once()

    def test_upload_folder_called_with_correct_kwargs(self, tmp_path):
        _, _, _, mock_upload, _ = self._run_push(tmp_path)
        mock_upload.assert_called_once()
        kw = mock_upload.call_args.kwargs
        assert kw["repo_id"] == "org/repo"
        assert kw["folder_path"] == str(tmp_path)
        assert kw["commit_message"] == "End of training"
        assert kw["run_as_future"] is False  # blocking=True by default
        assert "runs/**" in kw["ignore_patterns"]

    def test_non_blocking_sets_run_as_future(self, tmp_path):
        mock_url = MagicMock()
        mock_url.repo_id = "org/repo"
        with patch(
            "opaque.api.transformers.trainer._hub._create_repo", return_value=mock_url
        ):
            trainer = _tiny_trainer(tmp_path, push_to_hub=True, hub_model_id="org/repo")
        trainer.hub_model_id = "org/repo"
        with (
            patch.object(trainer, "save_model"),
            patch("opaque.api.transformers.trainer._hub.create_model_card"),
            patch("opaque.api.transformers.trainer._hub._upload_folder") as mock_upload,
        ):
            _hub.push_to_hub(trainer, blocking=False)
        assert mock_upload.call_args.kwargs["run_as_future"] is True


# ---------------------------------------------------------------------------
# create_model_card
# ---------------------------------------------------------------------------


class TestCreateModelCard:
    def _build_card(self, tmp_path, log_history=None):
        model_url = MagicMock()
        model_url.repo_id = "org/repo"
        with patch(
            "opaque.api.transformers.trainer._hub._create_repo", return_value=model_url
        ):
            trainer = _tiny_trainer(
                tmp_path,
                push_to_hub=True,
                hub_model_id="org/repo",
                clipping_norm=1.0,
            )
        if log_history is not None:
            trainer.state.log_history = log_history
        with patch(
            "transformers.modelcard.TrainingSummary.to_model_card",
            return_value="# Model\n\nSome card content.\n",
        ):
            _hub.create_model_card(trainer)
        readme = Path(tmp_path) / "README.md"
        return readme.read_text()

    def test_readme_written(self, tmp_path):
        content = self._build_card(tmp_path)
        assert len(content) > 0
        assert (tmp_path / "README.md").is_file()

    def test_dp_section_present(self, tmp_path):
        content = self._build_card(tmp_path)
        assert "### Privacy budget" in content
        assert "<!-- opaque-dp:begin -->" in content
        assert "<!-- opaque-dp:end -->" in content
        assert "[Opaque](https://github.com/JetBrains-Research/opaque)." in content

    def test_dp_section_contains_epsilon_from_log_history(self, tmp_path):
        log_history = [
            {
                "privacy_epsilon": 3.1415,
                "privacy_delta": 1e-5,
                "privacy_noise_multiplier": 0.8,
            }
        ]
        content = self._build_card(tmp_path, log_history=log_history)
        assert "3.1415" in content
        assert "1e-05" in content or "1e-5" in content

    def test_dp_section_idempotent(self, tmp_path):
        """Calling create_model_card twice replaces the DP section, not appends."""
        content1 = self._build_card(tmp_path)
        # Call again (README already exists).
        with patch(
            "transformers.modelcard.TrainingSummary.to_model_card",
            return_value=content1,
        ):
            _hub.create_model_card(
                _tiny_trainer_with_hub(tmp_path),
            )
        with (tmp_path / "README.md").open() as f:
            content2 = f.read()
        # Section should appear exactly once.
        assert content2.count("<!-- opaque-dp:begin -->") == 1
        assert content2.count("<!-- opaque-dp:end -->") == 1

    def test_dp_section_clipping_norm(self, tmp_path):
        content = self._build_card(tmp_path)
        assert "1.0" in content  # clipping_norm=1.0


def _tiny_trainer_with_hub(tmp_path) -> Trainer:
    model_url = MagicMock()
    model_url.repo_id = "org/repo"
    with patch(
        "opaque.api.transformers.trainer._hub._create_repo", return_value=model_url
    ):
        return _tiny_trainer(
            tmp_path, push_to_hub=True, hub_model_id="org/repo", clipping_norm=1.0
        )


# ---------------------------------------------------------------------------
# _splice_dp_section unit test
# ---------------------------------------------------------------------------


class TestSpliceDpSection:
    def test_inserts_after_framework_versions(self):
        card = (
            "## Training procedure\n\n"
            "### Training hyperparameters\n\nsome params\n\n"
            "### Framework versions\n\n- Transformers 4.x\n"
        )
        section = "<!-- opaque-dp:begin -->\n### Privacy budget\nDP stuff\n<!-- opaque-dp:end -->"
        result = _hub._splice_dp_section(card, section)
        fw_pos = result.index("### Framework versions")
        opaque_pos = result.index("- Opaque ")
        dp_pos = result.index("<!-- opaque-dp:begin -->")
        assert dp_pos > fw_pos, "DP section should appear after Framework versions"
        assert opaque_pos > fw_pos, (
            "Opaque version should be listed under Framework versions"
        )
        assert opaque_pos < dp_pos, (
            "Opaque framework version should appear before DP section"
        )
        assert result.count("<!-- opaque-dp:begin -->") == 1

    def test_framework_versions_idempotent_when_opaque_present(self):
        card = (
            "## Training procedure\n\n"
            "### Framework versions\n\n"
            "- Transformers 4.x\n"
            "- Opaque 0.0.1\n"
        )
        section = "<!-- opaque-dp:begin -->\n### Privacy budget\nDP stuff\n<!-- opaque-dp:end -->"
        result = _hub._splice_dp_section(card, section)
        assert result.count("- Opaque ") == 1

    def test_appends_when_no_framework_versions(self):
        card = "# Model\n\nSome content.\n"
        section = "<!-- opaque-dp:begin -->\nDP stuff\n<!-- opaque-dp:end -->"
        result = _hub._splice_dp_section(card, section)
        assert result.count("<!-- opaque-dp:begin -->") == 1
        assert "# Model" in result
        assert "DP stuff" in result

    def test_replaces_existing_section(self):
        old_section = "<!-- opaque-dp:begin -->\nOLD\n<!-- opaque-dp:end -->"
        card = f"# Model\n\n{old_section}\n"
        new_section = "<!-- opaque-dp:begin -->\nNEW\n<!-- opaque-dp:end -->"
        result = _hub._splice_dp_section(card, new_section)
        assert "OLD" not in result
        assert "NEW" in result
        assert result.count("<!-- opaque-dp:begin -->") == 1

    def test_multiline_section_replaced(self):
        old = "<!-- opaque-dp:begin -->\nline1\nline2\nline3\n<!-- opaque-dp:end -->"
        card = f"# Header\n\n{old}\n\n## Footer\n"
        new = "<!-- opaque-dp:begin -->\nnew content\n<!-- opaque-dp:end -->"
        result = _hub._splice_dp_section(card, new)
        assert "line1" not in result
        assert "new content" in result
        assert "## Footer" in result


# ---------------------------------------------------------------------------
# save_model push_to_hub integration
# ---------------------------------------------------------------------------


class TestSaveModelPush:
    def test_save_model_triggers_push_when_push_to_hub(self, tmp_path):
        mock_url = MagicMock()
        mock_url.repo_id = "org/repo"
        with patch(
            "opaque.api.transformers.trainer._hub._create_repo", return_value=mock_url
        ):
            trainer = _tiny_trainer(tmp_path, push_to_hub=True, hub_model_id="org/repo")
        trainer.hub_model_id = "org/repo"
        with (
            patch.object(trainer._model, "save_pretrained", create=True),
            patch("opaque.api.transformers.trainer._hub.push_to_hub") as mock_push,
        ):
            trainer.save_model(str(tmp_path))
        mock_push.assert_called_once()
        kw = mock_push.call_args
        assert kw.kwargs.get("commit_message") == "Model save"

    def test_save_model_internal_call_skips_push(self, tmp_path):
        mock_url = MagicMock()
        mock_url.repo_id = "org/repo"
        with patch(
            "opaque.api.transformers.trainer._hub._create_repo", return_value=mock_url
        ):
            trainer = _tiny_trainer(tmp_path, push_to_hub=True, hub_model_id="org/repo")
        trainer.hub_model_id = "org/repo"
        with (
            patch.object(trainer._model, "save_pretrained", create=True),
            patch("opaque.api.transformers.trainer._hub.push_to_hub") as mock_push,
        ):
            trainer.save_model(str(tmp_path), _internal_call=True)
        mock_push.assert_not_called()

    def test_save_model_no_push_when_push_to_hub_false(self, tmp_path):
        trainer = _tiny_trainer(tmp_path)
        with (
            patch.object(trainer._model, "save_pretrained", create=True),
            patch("opaque.api.transformers.trainer._hub.push_to_hub") as mock_push,
        ):
            trainer.save_model(str(tmp_path))
        mock_push.assert_not_called()
