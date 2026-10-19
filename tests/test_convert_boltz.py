from pathlib import Path

import pytest
from click.testing import CliRunner

from bioemu2 import Contact, Ligand, Protein, System, load_system_and_contacts
from bioemu2.cli import main
from bioemu2.data_interface import ContactToken


def test_convert_boltz_writes_loadable_system(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "examples" / "boltz" / "protein_ligand.yaml"
    out = tmp_path / "converted" / "system.yaml"
    result = CliRunner().invoke(main, ["convert-boltz", str(source), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert System.from_yaml(out) == System(
        entities=[
            Protein(asym_ids=["A", "B"], sequence="RMKQLEDKVEELLSKNYHLENEVARLKKLVGER"),
            Ligand(asym_ids="L", smiles="CCO"),
        ]
    )
    assert "Wrote BioEmu2 system YAML" in result.output
    assert (
        out.read_text()
        == """\
entities:
- entity_type: protein
  asym_ids:
  - A
  - B
  sequence: RMKQLEDKVEELLSKNYHLENEVARLKKLVGER
  mods: []
- entity_type: ligand
  asym_ids:
  - L
  smiles: CCO
bonds: []
asym_id_to_author_id: {}
"""
    )


def test_convert_boltz_reports_discarded_fields(tmp_path: Path) -> None:
    source = tmp_path / "boltz.yaml"
    source.write_text(
        "version: 1\nsequences:\n"
        "  - protein: {id: A, sequence: AC, msa: empty}\n"
        "constraints:\n"
        "  - contact: {token1: [A, 1], token2: [A, 2]}\n"
    )
    out = tmp_path / "system.yaml"
    result = CliRunner().invoke(main, ["convert-boltz", str(source), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "MSA" in result.output
    assert "IGNORED" in result.output
    assert load_system_and_contacts(out) == (
        System(entities=[Protein(asym_ids="A", sequence="AC")]),
        [
            Contact(
                token1=ContactToken(asym_id="A", res_id=1),
                token2=ContactToken(asym_id="A", res_id=2),
            )
        ],
    )


def test_convert_boltz_warns_but_converts_untested_version(tmp_path: Path) -> None:
    source = tmp_path / "boltz.yaml"
    source.write_text("version: 2\nsequences:\n  - protein: {id: A, sequence: AC}\n")
    out = tmp_path / "system.yaml"
    result = CliRunner().invoke(main, ["convert-boltz", str(source), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "Boltz version 2 is not version 1" in result.output
    assert System.from_yaml(out).chain_ids == ["A"]


def test_convert_boltz_ignores_forced_contact_and_writes_output(tmp_path: Path) -> None:
    source = tmp_path / "boltz.yaml"
    source.write_text(
        "version: 1\nsequences:\n"
        "  - protein: {id: A, sequence: AC}\n"
        "constraints:\n"
        "  - contact: {token1: [A, 1], token2: [A, 2], force: true}\n"
    )
    out = tmp_path / "system.yaml"
    result = CliRunner().invoke(main, ["convert-boltz", str(source), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "Ignoring Boltz contact force setting True" in result.output
    assert load_system_and_contacts(out)[1] == [
        Contact(
            token1=ContactToken(asym_id="A", res_id=1),
            token2=ContactToken(asym_id="A", res_id=2),
        )
    ]


@pytest.mark.parametrize(
    ("constraint", "message"),
    [
        ("  - contact: {token1: [A, 1], token2: [L, C8]}\n", "contacts"),
        ("  - bond: {atom1: [A, 1, CA], atom2: [L, 1, C8]}\n", "bonds"),
    ],
)
def test_convert_boltz_rejects_smiles_ligand_atom_references(
    tmp_path: Path, constraint: str, message: str
) -> None:
    source = tmp_path / "boltz.yaml"
    source.write_text(
        "version: 1\nsequences:\n"
        "  - protein: {id: A, sequence: AC}\n"
        "  - ligand: {id: L, smiles: CCO}\n"
        "constraints:\n" + constraint
    )
    out = tmp_path / "system.yaml"
    result = CliRunner().invoke(main, ["convert-boltz", str(source), "--out", str(out)])
    assert result.exit_code != 0
    assert f"Boltz {message} to atoms in SMILES ligand chain 'L' are not supported" in result.output
    assert not out.exists()


@pytest.mark.parametrize(
    "contents",
    [
        "version: 2\n",
        "[invalid",
        "version: 1\nsequences: []",
        "version: 1\nsequences:\n  - protein: {id: A, sequence: AC, cyclic: true}\n",
    ],
)
def test_convert_boltz_failure_does_not_write_output(tmp_path: Path, contents: str) -> None:
    source = tmp_path / "boltz.yaml"
    source.write_text(contents)
    out = tmp_path / "system.yaml"
    result = CliRunner().invoke(main, ["convert-boltz", str(source), "--out", str(out)])
    assert result.exit_code != 0
    assert "Error:" in result.output
    assert not out.exists()


def test_convert_boltz_refuses_to_overwrite_input(tmp_path: Path) -> None:
    source = tmp_path / "boltz.yaml"
    contents = "version: 1\nsequences:\n  - protein: {id: A, sequence: AC}\n"
    source.write_text(contents)
    result = CliRunner().invoke(main, ["convert-boltz", str(source), "--out", str(source)])
    assert result.exit_code != 0
    assert "already exists" in result.output
    assert source.read_text() == contents


def test_convert_boltz_is_discoverable_in_help() -> None:
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "convert-boltz" in result.output
