"""Checkpoint download and path resolution helpers."""

import logging
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import constants, hf_hub_download
from huggingface_hub.file_download import repo_folder_name

logger = logging.getLogger(__name__)

HF_CHECKPOINT_REPO = "microsoft/bioemu2"
# The HF cache keeps each commit's files in <repo folder>/snapshots/<commit>/.
_REPO_FOLDER = repo_folder_name(repo_id=HF_CHECKPOINT_REPO, repo_type="model")
# Checkpoint directories this code version supports. Published directories are never
# overwritten: a changed checkpoint or config gets a new name, added here in the PR that uses it.
SUPPORTED_CHECKPOINTS = ("bioemu2-preview",)
DEFAULT_CHECKPOINT = SUPPORTED_CHECKPOINTS[0]


@dataclass(frozen=True)
class CheckpointFiles:
    """Local paths to a BioEmu2 checkpoint and its config."""

    checkpoint_path: Path
    config_path: Path


def resolve_checkpoint_files(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
) -> CheckpointFiles:
    """Resolve paths to a local checkpoint and config.

    Logs whether the checkpoint lies in a Hugging Face cache snapshot.

    Args:
        checkpoint_path: Path to a local BioEmu2 checkpoint file.
        config_path: Optional path to the ``config.yaml``. Defaults to a
            ``config.yaml`` in the same directory as the checkpoint.

    Returns:
        Paths to the local checkpoint and config files.
    """
    checkpoint = Path(checkpoint_path).expanduser()
    # Check the unresolved path: resolve() follows the symlink into blobs/ and loses the commit.
    _, found, rest = checkpoint.absolute().as_posix().partition(f"{_REPO_FOLDER}/snapshots/")
    if found:
        logger.info(
            "Local checkpoint %s is from %s@%s.", checkpoint, HF_CHECKPOINT_REPO, rest.split("/")[0]
        )
    else:
        logger.info("Local checkpoint %s has no Hugging Face provenance.", checkpoint)
    resolved_checkpoint = checkpoint.resolve()
    resolved_config = (
        Path(config_path).expanduser().resolve()
        if config_path is not None
        else (checkpoint.parent / "config.yaml").resolve()
    )
    return CheckpointFiles(
        checkpoint_path=resolved_checkpoint,
        config_path=resolved_config,
    )


def get_checkpoint_files(name: str, *, cache_dir: str | Path | None = None) -> CheckpointFiles:
    """Load a named checkpoint from the local cache, downloading it if missing.

    Checkpoints follow the deterministic layout ``checkpoints/<name>/checkpoint.ckpt``
    with a matching ``checkpoints/<name>/config.yaml``. Published checkpoint directories
    are never overwritten, so if both files are in the Hugging Face cache, under any
    commit, they are used without contacting the Hub. Otherwise only these two files are
    downloaded, which fails offline.

    Args:
        name: Name of the checkpoint directory, one of ``SUPPORTED_CHECKPOINTS``.
        cache_dir: Hugging Face cache directory to download into and reuse files
            from. Defaults to the standard Hugging Face cache, which ``HF_HUB_CACHE``
            or ``HF_HOME`` can relocate.

    Returns:
        Paths to the cached or downloaded checkpoint and config files.

    Raises:
        ValueError: If ``name`` is not in ``SUPPORTED_CHECKPOINTS``.
    """
    if name not in SUPPORTED_CHECKPOINTS:
        raise ValueError(f"Unsupported checkpoint {name!r}; choose from {SUPPORTED_CHECKPOINTS}.")
    hub_cache = Path(constants.HF_HUB_CACHE if cache_dir is None else cache_dir)
    for snapshot in (hub_cache / _REPO_FOLDER / "snapshots").glob("*"):
        checkpoint = snapshot / "checkpoints" / name / "checkpoint.ckpt"
        config = checkpoint.with_name("config.yaml")
        if checkpoint.is_file() and config.is_file():  # follows symlinks into blobs/
            logger.info(
                "Using cached checkpoint %r from %s at %s.",
                name,
                HF_CHECKPOINT_REPO,
                checkpoint.parent,
            )
            return CheckpointFiles(checkpoint, config)
    checkpoint, config = (
        Path(
            hf_hub_download(
                repo_id=HF_CHECKPOINT_REPO,
                filename=f"checkpoints/{name}/{filename}",
                cache_dir=cache_dir,
            )
        )
        for filename in ("checkpoint.ckpt", "config.yaml")
    )
    logger.info(
        "Downloaded checkpoint %r from %s to %s.", name, HF_CHECKPOINT_REPO, checkpoint.parent
    )
    return CheckpointFiles(checkpoint, config)
