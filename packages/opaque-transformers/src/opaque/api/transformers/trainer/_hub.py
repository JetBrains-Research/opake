"""Hub publishing for Trainer.

A deliberately minimal "publish & manage the finished model" surface — model
*publishing*, not the HF in-training auto-push machinery (per-checkpoint
uploads, async ``PushInProgress``, ``hub_strategy``) which re-couples Hub to
the checkpoint hot path for little value on a DP run.  Upload happens once at
training end (or on an explicit ``push_to_hub()`` call), synchronously.

Hub methods that mirror HF ``Trainer``:

- ``init_hf_repo`` — create (or validate) the Hub repo, set ``hub_model_id``
- ``push_to_hub`` — end-of-training (or user-triggered) upload
- ``create_model_card`` — write ``README.md`` (HF card + Opaque DP ε/δ section)

All are free functions taking the trainer as their first argument so they mix
in without subclassing overhead.

**Reuse contract**:

- ``upload_folder``          — ``huggingface_hub.upload_folder``
- ``create_repo``            — ``huggingface_hub.create_repo``
- ``ModelCard``              — ``huggingface_hub.ModelCard``
- ``TrainingSummary``        — ``transformers.modelcard.TrainingSummary``

Nothing here is re-implemented from scratch where an importable equivalent
exists.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from transformers.modelcard import TrainingSummary

if TYPE_CHECKING:
    from ._trainer import Trainer  # pragma: no cover

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — guarded so the module stays importable when
# ``huggingface_hub`` / ``transformers.modelcard`` are absent.  The guard
# raises only when the user actually calls a hub method with
# ``push_to_hub=True``, not at import time.
# ---------------------------------------------------------------------------


def _require_hub() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError as exc:
        raise ImportError(  # noqa: TRY003 - preserve standard Python error contract
            "push_to_hub=True requires `huggingface_hub`. "
            "Install it with: pip install huggingface_hub"
        ) from exc


def _upload_folder(**kwargs: Any) -> Any:
    from huggingface_hub import upload_folder

    return upload_folder(**kwargs)


def _create_repo(repo_name: str, *, token: str | None, private: bool | None) -> Any:
    from huggingface_hub import create_repo

    return create_repo(repo_name, token=token, private=private, exist_ok=True)


# ---------------------------------------------------------------------------
# init_hf_repo
# ---------------------------------------------------------------------------


def init_hf_repo(trainer: Trainer, token: str | None = None) -> None:
    """Create (or validate) the Hub repo and set ``trainer.hub_model_id``.

    Mirrors ``Trainer.init_hf_repo``.  Only runs on process-zero.
    """
    if not trainer.is_world_process_zero():
        return

    _require_hub()

    a = trainer.args
    if a.hub_model_id is None:
        repo_name = Path(a.output_dir).absolute().name
    else:
        repo_name = a.hub_model_id

    effective_token = token if token is not None else a.hub_token
    repo_url = _create_repo(
        repo_name, token=effective_token, private=a.hub_private_repo
    )
    trainer.hub_model_id = repo_url.repo_id


# ---------------------------------------------------------------------------
# push_to_hub
# ---------------------------------------------------------------------------


def push_to_hub(
    trainer: Trainer,
    commit_message: str | None = "End of training",
    blocking: bool = True,
    token: str | None = None,
    revision: str | None = None,
    **kwargs: Any,
) -> Any:
    """Upload the model to the HF Hub.

    Mirrors ``Trainer.push_to_hub``.  Calls ``save_model`` (which restores
    params into the model first), writes the model card, then uploads the
    output_dir via ``upload_folder``.

    Args:
        trainer: trainer whose saved model and output directory are uploaded.
        commit_message: Hub commit message.
        blocking: If True, wait for the upload to complete before returning.
        token: Override ``args.hub_token`` for this call.
        revision: Branch to commit to.  Defaults to ``args.hub_revision``
            (which defaults to ``main``).
        **kwargs: Forwarded to :func:`create_model_card`.

    Returns:
        ``CommitInfo`` from ``huggingface_hub.upload_folder``.
    """
    if not trainer.is_world_process_zero():
        return None

    _require_hub()

    # Fire on_push_begin callback if available.
    if hasattr(trainer._callback_handler, "on_push_begin"):
        trainer._callback_handler.on_push_begin(
            trainer.args, trainer.state, trainer._control
        )

    a = trainer.args

    model_name = kwargs.pop("model_name", None)
    if model_name is None:
        if not hasattr(trainer, "hub_model_id") or trainer.hub_model_id is None:
            model_name = Path(a.output_dir).name
        else:
            model_name = trainer.hub_model_id.split("/")[-1]

    effective_token = token if token is not None else a.hub_token

    # Lazily init repo if the user calls push_to_hub manually without
    # push_to_hub=True having been set at construction time.
    if not hasattr(trainer, "hub_model_id") or trainer.hub_model_id is None:
        init_hf_repo(trainer, token=effective_token)

    # Restore params → save model and processing class to output_dir.
    trainer.save_model(_internal_call=True)

    # Add model-native tags (e.g. "llama") to kwargs if present.
    if getattr(trainer.model, "model_tags", None) is not None:
        if "tags" not in kwargs:
            kwargs["tags"] = []
        if isinstance(kwargs["tags"], str):
            kwargs["tags"] = [kwargs["tags"]]
        for model_tag in trainer.model.model_tags:
            if model_tag not in kwargs["tags"]:
                kwargs["tags"].append(model_tag)

    create_model_card(trainer, model_name=model_name, **kwargs)

    if revision is None:
        revision = a.hub_revision

    from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

    return _upload_folder(
        repo_id=trainer.hub_model_id,
        folder_path=a.output_dir,
        commit_message=commit_message,
        token=effective_token,
        run_as_future=not blocking,
        ignore_patterns=["_*", f"{PREFIX_CHECKPOINT_DIR}-*", "runs/**"],
        revision=revision,
    )


# ---------------------------------------------------------------------------
# create_model_card
# ---------------------------------------------------------------------------

_DP_SECTION_MARKER_BEGIN = "<!-- opaque-dp:begin -->"
_DP_SECTION_MARKER_END = "<!-- opaque-dp:end -->"


def _opaque_version() -> str:
    try:
        from importlib.metadata import version

        return version("opaque-transformers")
    except Exception:
        return "unknown"


_DP_SECTION_TEMPLATE = """\
{begin}
### Privacy budget

This model was trained with differential privacy using
[Opaque](https://github.com/JetBrains-Research/opaque).

- ε (epsilon): {epsilon}
- δ (delta): {delta}
- noise multiplier: {noise_multiplier}
- clipping norm: {clipping_norm}
{end}"""


@dataclass
class DPTrainingSummary(TrainingSummary):
    """Opaque-owned TrainingSummary variant for model-card customization."""

    privacy: dict[str, float | str] | None = field(default=None)

    @classmethod
    def from_trainer(
        cls,
        trainer: Trainer,
        language: str | None = None,
        license: str | None = None,
        tags: str | list[str] | None = None,
        model_name: str | None = None,
        finetuned_from: str | None = None,
        tasks: str | list[str] | None = None,
        dataset_tags: str | list[str] | None = None,
        dataset_metadata: dict[str, Any] | None = None,
        dataset: str | list[str] | None = None,
        dataset_args: str | list[str] | None = None,
    ) -> DPTrainingSummary:
        summary = super().from_trainer(
            trainer,
            language=language,
            license=license,
            tags=tags,
            model_name=model_name,
            finetuned_from=finetuned_from,
            tasks=tasks,
            dataset_tags=dataset_tags,
            dataset_metadata=dataset_metadata,
            dataset=dataset,
            dataset_args=dataset_args,
        )
        summary.privacy = _build_privacy_summary(trainer)
        return summary

    def to_model_card(self) -> str:
        card = super().to_model_card()
        if self.privacy is None:
            return _splice_opaque_framework_version(card)
        return _splice_dp_section(card, _build_dp_section_from_privacy(self.privacy))


def create_model_card(
    trainer: Trainer,
    language: str | None = None,
    license: str | None = None,
    tags: str | list[str] | None = None,
    model_name: str | None = None,
    finetuned_from: str | None = None,
    tasks: str | list[str] | None = None,
    dataset_tags: str | list[str] | None = None,
    dataset: str | list[str] | None = None,
    dataset_args: str | list[str] | None = None,
) -> None:
    """Write ``README.md`` to ``args.output_dir``.

    Delegates to ``DPTrainingSummary.from_trainer`` for the base card (HF
    parity: task tags, dataset metadata, eval metrics, hyperparameters,
    license inference) plus Opaque card customization.

    Opaque-specific additions:
    - Tags ``"differential-privacy"`` and ``"opaque"`` are merged into
      the existing tag list.
    - A ``## Privacy budget`` section lists ε, δ, noise multiplier, and
      clipping norm.  Values are read from the training summary metrics
      injected at the end of ``_inner_training_loop``; if training hasn't
      finished yet the values fall back to the calibrated noise multiplier
      on the training context.
    """
    # Single-process guard — always process-zero for now.
    a = trainer.args
    if a.output_dir is None:
        return

    _require_hub()

    # Inject Opaque tags before delegating to DPTrainingSummary.
    opaque_tags = ["differential-privacy", "opaque"]
    if tags is None:
        tags = opaque_tags
    elif isinstance(tags, str):
        tags = [tags, *opaque_tags]
    else:
        for t in opaque_tags:
            if t not in tags:
                tags.append(t)

    # DPTrainingSummary.from_trainer reads trainer.train_dataset,
    # trainer.eval_dataset, trainer.model, trainer.args, trainer.state.
    # Trainer exposes all of these as public properties.
    training_summary = DPTrainingSummary.from_trainer(
        trainer,
        language=language,
        license=license,
        tags=tags,
        model_name=model_name,
        finetuned_from=finetuned_from,
        tasks=tasks,
        dataset_tags=dataset_tags,
        dataset=dataset,
        dataset_args=dataset_args,
    )
    model_card_content = training_summary.to_model_card()

    output_dir = Path(a.output_dir)
    output_path = output_dir / "README.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write(model_card_content)

    log.info("Model card written to %s", output_path)


def _build_privacy_summary(trainer: Trainer) -> dict[str, float | str]:
    """Extract structured privacy values for model-card rendering."""
    a = trainer.args

    # Prefer values from the training summary metrics (populated at the end
    # of _inner_training_loop) because they reflect the *actual* epsilon
    # consumed, not just the calibrated target.
    epsilon: float | str = "unknown"
    delta: float | str = "unknown"
    noise_multiplier: float | str = "unknown"

    # Pull from the last training log entry if available.
    for entry in reversed(trainer.state.log_history):
        if "privacy_epsilon" in entry:
            epsilon = round(entry["privacy_epsilon"], 4)
        if "privacy_delta" in entry:
            delta = entry["privacy_delta"]
        if "privacy_noise_multiplier" in entry:
            noise_multiplier = round(entry["privacy_noise_multiplier"], 6)
        if epsilon != "unknown":
            break

    # Fall back to live training context values. For a whole-horizon process
    # this is the declared full-run epsilon, not an incomplete-prefix value.
    if epsilon == "unknown" and trainer._ctx is not None:
        ctx = trainer._ctx
        epsilon = round(ctx.accounting.epsilon_at(ctx.target_delta), 4)
        delta = ctx.target_delta
        noise_multiplier = round(ctx.noise_multiplier, 6)

    # Fall back to args.
    st = trainer.state
    if (
        noise_multiplier == "unknown"
        and getattr(st, "privacy_resolved_noise_multiplier", None) is not None
    ):
        noise_multiplier = round(st.privacy_resolved_noise_multiplier, 6)
    elif noise_multiplier == "unknown" and a.privacy_noise_multiplier is not None:
        noise_multiplier = a.privacy_noise_multiplier
    if delta == "unknown" and getattr(st, "privacy_resolved_delta", None) is not None:
        delta = st.privacy_resolved_delta
    elif delta == "unknown" and a.privacy_target_delta is not None:
        delta = a.privacy_target_delta

    clipping_norm: float | str
    cn = trainer._ctx.clip_norm if trainer._ctx is not None else a.clipping_norm
    if cn is None:
        clipping_norm = "unknown"
    elif hasattr(cn, "effective"):
        clipping_norm = round(float(cn.effective), 4)
    elif isinstance(cn, dict):
        clipping_norm = json.dumps(cn, sort_keys=True)
    else:
        clipping_norm = float(cn)

    return {
        "epsilon": epsilon,
        "delta": delta,
        "noise_multiplier": noise_multiplier,
        "clipping_norm": clipping_norm,
    }


def _build_dp_section_from_privacy(privacy: dict[str, float | str]) -> str:
    """Render the DP privacy-budget section from structured values."""
    return _DP_SECTION_TEMPLATE.format(
        begin=_DP_SECTION_MARKER_BEGIN,
        end=_DP_SECTION_MARKER_END,
        epsilon=privacy.get("epsilon", "unknown"),
        delta=privacy.get("delta", "unknown"),
        noise_multiplier=privacy.get("noise_multiplier", "unknown"),
        clipping_norm=privacy.get("clipping_norm", "unknown"),
    )


# Matches the end of the "### Framework versions" subsection: everything
# from the heading up to (but not including) the next ## or end-of-string.
_FRAMEWORK_VERSIONS_RE = re.compile(
    r"(### Framework versions\b.*?)(?=\n## |\Z)",
    re.DOTALL,
)


def _splice_dp_section(card: str, dp_section: str) -> str:
    """Replace or append the ``<!-- opaque-dp:begin/end -->`` block.

    Placement preference:
    1. Replace an existing marker block in-place.
    2. Insert as a new ``### Privacy budget`` subsection immediately after
       ``### Framework versions`` (inside ``## Training procedure``).
    3. Fall back: append at the end of the card.
    """
    card = _splice_opaque_framework_version(card)

    pattern = re.compile(
        re.escape(_DP_SECTION_MARKER_BEGIN) + ".*?" + re.escape(_DP_SECTION_MARKER_END),
        re.DOTALL,
    )
    if pattern.search(card):
        return pattern.sub(dp_section, card)
    # Try to place it right after "### Framework versions".
    m = _FRAMEWORK_VERSIONS_RE.search(card)
    if m:
        insert_pos = m.end()
        return card[:insert_pos] + "\n\n" + dp_section + "\n" + card[insert_pos:]
    # Fallback: append.
    return card.rstrip("\n") + "\n\n" + dp_section + "\n"


def _splice_opaque_framework_version(card: str) -> str:
    """Insert Opaque version into the Framework versions subsection."""
    m = _FRAMEWORK_VERSIONS_RE.search(card)
    if m is None:
        return card

    framework_block = m.group(1)
    if re.search(r"^- Opaque\b", framework_block, re.MULTILINE):
        return card

    opaque_line = f"- Opaque {_opaque_version()}\n"
    insert_pos = m.end()
    return card[:insert_pos] + opaque_line + card[insert_pos:]
