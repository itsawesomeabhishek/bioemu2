"""Persistent cache for single-device trunk embeddings."""

import errno
import hashlib
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

import torch

from bioemu2.biomoa.models.score_model import _EMBEDDINGS_FILE, TrunkEmbeddings
from bioemu2.conditioning.method import SamplingMode
from bioemu2.data_interface import Contact, System
from bioemu2.msa.alignment import MSA

_UNWRITABLE_ERRORS = {errno.EACCES, errno.EPERM, errno.EROFS, errno.ENOSPC, errno.EDQUOT}
# Increment when input preparation or trunk embedding semantics change.
_CACHE_VERSION = 1
logger = logging.getLogger(__name__)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def digest(value: object) -> str:
    """Hash canonical JSON-serializable inputs into a stable cache key component."""
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def system_digest(system: System) -> str:
    return digest(system.model_dump(mode="json", exclude_none=True))


def contacts_digest(contacts: list[Contact] | None) -> str:
    return digest([contact.model_dump(mode="json") for contact in contacts or []])


def msa_digest(msa: MSA) -> str:
    return digest(
        {
            "unpaired": dict(msa.unpaired_a3m),
            "paired": dict(msa.paired_a3m) if msa.paired_a3m is not None else None,
        }
    )


def cache_directory() -> Path | None:
    """Return the persistent embedding cache directory, or None when disabled."""
    configured = os.environ.get("BIOEMU2_EMBEDS_CACHE")
    if not configured:
        return None
    return Path(configured).expanduser()


def cache_writable(root: Path) -> bool:
    """Check that a new entry can be written before computing its embeddings."""
    try:
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=root) as probe:
            probe.write(b"cache")
    except OSError as exc:
        if exc.errno not in _UNWRITABLE_ERRORS:
            raise
        logger.warning(
            "Embedding cache at %s is not writable: %s; continuing without caching.", root, exc
        )
        return False
    return True


def entry_metadata(
    system: System,
    msa: MSA,
    mode: SamplingMode,
    contacts: list[Contact] | None,
    model_fingerprint: str,
    autocast_dtype: torch.dtype | None,
    num_steps_recycle: int,
) -> dict[str, str | int]:
    """Build the versioned inputs used to identify serial trunk embeddings."""
    return {
        "version": _CACHE_VERSION,
        "msa": msa_digest(msa),
        "system": system_digest(system),
        "mode": mode,
        "contacts": contacts_digest(contacts),
        "model": model_fingerprint,
        "autocast_dtype": str(autocast_dtype),
        "num_steps_recycle": num_steps_recycle,
    }


def _entry_path(root: Path, metadata: dict[str, str | int]) -> Path:
    return root / digest(metadata)


def load_embeddings(
    root: Path, metadata: dict[str, str | int], device: torch.device
) -> TrunkEmbeddings | None:
    """Read a complete cache entry, or return None when it is absent or incomplete.

    ``metadata.json`` is written last, so its presence marks a finished entry. An
    entry left without it by an interrupted write is treated as absent and
    recomputed.
    """
    path = _entry_path(root, metadata)
    marker = path / "metadata.json"
    if not marker.exists():
        return None
    recorded = json.loads(marker.read_text(encoding="utf-8"))
    if recorded != metadata:
        raise ValueError(f"Embedding cache metadata mismatch at {path}")
    return TrunkEmbeddings.load(path / _EMBEDDINGS_FILE, device)


def save_embeddings(
    root: Path, metadata: dict[str, str | int], embeddings: TrunkEmbeddings
) -> None:
    """Write a cache entry directly into its final directory.

    Write ``embeddings.pt`` first and the ``metadata.json`` marker last, so a
    reader only trusts the entry once the marker exists. Writing in place, rather
    than staging elsewhere and renaming, avoids a cross-file rename that network
    and blob-backed filesystems implement as a slow copy. A stale ``embeddings.pt``
    from an interrupted write is replaced; a finished entry, identified by its
    marker, is never overwritten.

    A write that fails removes its partial directory and logs a warning with the
    error, leaving the cache root with only finished entries. The in-memory
    embeddings are still returned to the caller, so a cache write failure never
    discards a successful computation. A read-only cache directory is already
    handled earlier by ``cache_writable``.
    """
    path = _entry_path(root, metadata)
    marker = path / "metadata.json"
    if marker.exists():
        raise FileExistsError(f"Embedding cache entry already exists at {path}.")
    path.mkdir(parents=True, exist_ok=True)
    embeddings_file = path / _EMBEDDINGS_FILE
    embeddings_file.unlink(missing_ok=True)
    try:
        embeddings.save(embeddings_file)
        marker.write_bytes(_json_bytes(metadata))
    except (OSError, RuntimeError) as exc:
        shutil.rmtree(path, ignore_errors=True)
        logger.warning(
            "Cannot write embedding cache at %s: %s; continuing without caching.", path, exc
        )
