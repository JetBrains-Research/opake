"""ZenML pipelines for infrastructure validation and SFT training."""

from adapters.zenml.steps import (
    probe_environment,
    train_sft_step,
    verify_probe_artifact,
)

from zenml import pipeline


@pipeline(enable_cache=False)
def infrastructure_probe_pipeline(
    require_gpu: bool,
    required_secret_env: list[str],
    check_huggingface: bool,
) -> None:
    """Run the infrastructure probe and verify its materialized artifact."""
    probe_artifact, probe_report = probe_environment(
        require_gpu=require_gpu,
        required_secret_env=required_secret_env,
        check_huggingface=check_huggingface,
    )
    verify_probe_artifact(
        probe_artifact=probe_artifact,
        expected_report=probe_report,
    )


@pipeline(enable_cache=False)
def sft_pipeline(
    config: dict[str, object],
    source_commit_sha: str,
    resume_checkpoint: str | None = None,
) -> None:
    """Run one fully configured SFT training job."""
    train_sft_step(
        config=config,
        source_commit_sha=source_commit_sha,
        resume_checkpoint=resume_checkpoint,
    )
