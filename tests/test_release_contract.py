import logging
import subprocess
import sys
from pathlib import Path

import mdtraj
import pytest

from bioemu2 import cli, load_system_and_contacts
from bioemu2.checkpoints import (
    get_checkpoint_files,
    resolve_checkpoint_files,
)
from bioemu2.msa.colabfold import msa_cache_dir
from bioemu2.paths import ASSETS_DIR, EXAMPLES_DIR
from bioemu2.structure.cif import save_structure_to_cif
from bioemu2.structure.inference_inputs import inference_input_structure_from_inchi
from bioemu2.structure.templates import load_polymer_templates


def test_local_contract_and_packaged_templates(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run" / "checkpoints" / "model.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    config = checkpoint.parent / "config.yaml"
    config.touch()

    checkpoint_files = resolve_checkpoint_files(checkpoint)
    assert checkpoint_files.checkpoint_path == checkpoint
    assert checkpoint_files.config_path == config
    assert len(load_polymer_templates()) == 31
    assert (ASSETS_DIR / "polymer_templates.pt").is_file()


@pytest.mark.parametrize(
    ("name", "has_contacts"),
    [
        ("ake", False),
        ("chignolin", False),
        ("myoglobin_101m", True),
        ("shp2_3tkz", True),
        ("meteorin_like_7tlw", False),
    ],
)
def test_packaged_example_systems(name: str, has_contacts: bool) -> None:
    system, contacts = load_system_and_contacts(EXAMPLES_DIR / "systems" / f"{name}.yaml")
    assert system.entities
    assert bool(contacts) is has_contacts


@pytest.mark.parametrize("explicit_config", [False, True])
def test_local_checkpoint_symlinks_keep_snapshot_config(
    tmp_path: Path, explicit_config: bool
) -> None:
    """Check that a checkpoint stored as a file link can be read with the config
    next to the link target, or with a different config supplied by the caller.
    """
    blobs = tmp_path / "blobs"
    snapshot = tmp_path / "snapshots" / "revision" / "checkpoints" / "preview"
    blobs.mkdir()
    snapshot.mkdir(parents=True)
    checkpoint_blob = blobs / "weights-hash"
    config_blob = blobs / "config-hash"
    checkpoint_blob.write_bytes(b"checkpoint")
    config_blob.write_text("score_model: snapshot\n")
    checkpoint = snapshot / "checkpoint.ckpt"
    checkpoint.symlink_to(checkpoint_blob)
    (snapshot / "config.yaml").symlink_to(config_blob)
    override = tmp_path / "override.yaml"
    override.write_text("score_model: override\n")

    files = resolve_checkpoint_files(checkpoint, override if explicit_config else None)

    assert files.checkpoint_path.read_bytes() == b"checkpoint"
    assert files.config_path.read_text() == (
        "score_model: override\n" if explicit_config else "score_model: snapshot\n"
    )


@pytest.mark.parametrize(
    ("commit", "message"),
    [
        ("0" * 40, f"is from microsoft/bioemu2@{'0' * 40}"),
        (None, "no Hugging Face provenance"),
    ],
)
def test_local_checkpoint_provenance_is_logged(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    commit: str | None,
    message: str,
) -> None:
    snapshots = tmp_path / "models--microsoft--bioemu2" / "snapshots"
    checkpoint_dir = (
        tmp_path / "run"
        if commit is None
        else snapshots / commit / "checkpoints" / "bioemu2-preview"
    )

    with caplog.at_level(logging.INFO, logger="bioemu2.checkpoints"):
        resolve_checkpoint_files(checkpoint_dir / "checkpoint.ckpt")

    assert message in caplog.text


def test_msa_cache_comes_from_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BIOEMU2_MSA_CACHE", str(tmp_path / "first"))
    first = msa_cache_dir()
    monkeypatch.setenv("BIOEMU2_MSA_CACHE", str(tmp_path / "second"))
    second = msa_cache_dir()

    assert first == tmp_path / "first"
    assert second == tmp_path / "second"


@pytest.mark.parametrize("name", ["", "..", "foo/bar", "*", "bioemu2-unknown"])
def test_get_checkpoint_files_rejects_unsupported_name(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    def fail_download(*args, **kwargs):
        raise AssertionError("hf_hub_download should not be called for invalid names")

    monkeypatch.setattr("bioemu2.checkpoints.hf_hub_download", fail_download)

    with pytest.raises(ValueError, match="Unsupported checkpoint"):
        get_checkpoint_files(name)


def test_get_checkpoint_files_uses_cache_from_any_snapshot_without_hub(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    snapshots = tmp_path / "models--microsoft--bioemu2" / "snapshots"
    cached = snapshots / ("1" * 40) / "checkpoints" / "bioemu2-preview"
    cached.mkdir(parents=True)
    (cached / "checkpoint.ckpt").touch()
    (cached / "config.yaml").touch()
    (snapshots / ("2" * 40)).mkdir()  # a newer snapshot without this checkpoint

    def fail_download(*args, **kwargs):
        raise AssertionError("The Hub must not be contacted for a cached checkpoint")

    monkeypatch.setattr("bioemu2.checkpoints.hf_hub_download", fail_download)

    with caplog.at_level(logging.INFO, logger="bioemu2.checkpoints"):
        files = get_checkpoint_files("bioemu2-preview", cache_dir=tmp_path)

    assert files.checkpoint_path == cached / "checkpoint.ckpt"
    assert files.config_path == cached / "config.yaml"
    assert f"Using cached checkpoint 'bioemu2-preview' from microsoft/bioemu2 at {cached}" in (
        caplog.text
    )


def test_get_checkpoint_files_downloads_when_cached_pair_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cached = tmp_path / "models--microsoft--bioemu2" / "snapshots" / ("1" * 40)
    cached = cached / "checkpoints" / "bioemu2-preview"
    cached.mkdir(parents=True)
    (cached / "checkpoint.ckpt").touch()
    (cached / "config.yaml").symlink_to(tmp_path / "missing-blob")  # dangling
    downloads = tmp_path / "downloads"
    downloads.mkdir()

    def fake_download(*, repo_id: str, filename: str, cache_dir: Path) -> str:
        return str(downloads / Path(filename).name)

    monkeypatch.setattr("bioemu2.checkpoints.hf_hub_download", fake_download)

    files = get_checkpoint_files("bioemu2-preview", cache_dir=tmp_path)

    assert files.checkpoint_path.parent == downloads


def test_cli_rejects_checkpoint_cache_dir_with_local_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    system_path = EXAMPLES_DIR / "systems" / "chignolin.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bioemu2",
            "sample-equilibrium",
            "--system",
            str(system_path),
            "--checkpoint-path",
            str(tmp_path / "model.ckpt"),
            "--checkpoint-cache-dir",
            str(tmp_path / "cache"),
            "--out",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        cli.main()
    assert exc_info.value.code == 2
    assert not (tmp_path / "out").exists()


def test_ligand_templates_preserve_empty_and_present_bonds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BIOEMU2_TEMPLATE_CACHE", str(tmp_path / "templates"))

    methane = inference_input_structure_from_inchi("InChI=1S/CH4/h1H4")
    assert tuple(methane.bond_index.shape) == (2, 0)

    ethane = inference_input_structure_from_inchi("InChI=1S/C2H6/c1-2/h1-2H3")
    cif_path = tmp_path / "ethane.cif"
    save_structure_to_cif(ethane, cif_path)
    trajectory = mdtraj.load(str(cif_path))
    assert trajectory.topology.n_bonds == 1


def test_public_import_does_not_load_internal_packages() -> None:
    code = """
import sys
import bioemu2
blocked = {'amlt', 'azure', 'wandb', 'ai4s_filesystems', 'ai4s_logging', 'sqlalchemy'}
loaded = blocked & {name.split('.')[0] for name in sys.modules}
assert not loaded, loaded
"""
    subprocess.run([sys.executable, "-c", code], check=True)
