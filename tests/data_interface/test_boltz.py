"""Boltz YAML decoder."""

import logging
from collections import UserDict
from pathlib import Path

import pytest
import yaml

from bioemu2 import convert_boltz
from bioemu2.data_interface import (
    DNA,
    RNA,
    Atom,
    Bond,
    Contact,
    ContactToken,
    Ligand,
    ModifiedResidue,
    Protein,
    System,
    load_system_and_contacts,
)
from bioemu2.data_interface.boltz import system_from_boltz


def test_from_boltz_round_trips_system_yaml(tmp_path):
    """Decode -> serialize preserves entities + bonds + mods + multi-chain IDs."""
    task = System(
        entities=[
            Protein(
                sequence="MAXDXY",
                asym_ids=["A", "B"],
                mods=[
                    ModifiedResidue(res_id=3, ccd_code="SEP"),
                    ModifiedResidue(res_id=5, ccd_code="TPO"),
                ],
            ),
            Ligand(ccd_code="ATP", asym_ids="L"),
        ],
        bonds=[
            Bond(
                atom1=Atom(name="O2P", res_name="SEP", res_id=3, asym_id="A"),
                atom2=Atom(name="P", res_name="ATP", res_id=1, asym_id="L"),
            )
        ],
    )
    source = tmp_path / "boltz.yaml"
    source.write_text(
        """\
version: 1
sequences:
  - protein:
      id: [A, B]
      sequence: MAXDXY
      modifications:
        - {position: 5, ccd: TPO}
        - {position: 3, ccd: SEP}
  - ligand:
      id: L
      ccd: ATP
constraints:
  - bond:
      atom1: [A, 3, O2P]
      atom2: [L, 1, P]
"""
    )
    converted, contacts = system_from_boltz(yaml.safe_load(source.read_text()))
    assert converted == task
    assert contacts == []
    converted.to_yaml(tmp_path / "system.yaml")
    assert System.from_yaml(tmp_path / "system.yaml") == task


def test_from_boltz_parses_handwritten_dict():
    """Handwritten payload (protein + RNA with mods, CCD ligand, bond) parses cleanly."""
    task, _ = system_from_boltz(
        {
            "version": 1,
            "sequences": [
                {
                    "protein": {
                        "id": "A",
                        "sequence": "AXC",
                        "modifications": [{"position": 2, "ccd": "SEP"}],
                    }
                },
                {
                    "rna": {
                        "id": "R",
                        "sequence": "GN",
                        "modifications": [{"position": 2, "ccd": "5MC"}],
                    }
                },
                {"ligand": {"id": "L", "ccd": "ATP"}},
            ],
            "constraints": [{"bond": {"atom1": ["A", 2, "O2P"], "atom2": ["L", 1, "P"]}}],
        }
    )
    assert task.entities == [
        Protein(asym_ids=["A"], sequence="AXC", mods=[ModifiedResidue(res_id=2, ccd_code="SEP")]),
        RNA(asym_ids=["R"], sequence="GN", mods=[ModifiedResidue(res_id=2, ccd_code="5MC")]),
        Ligand(asym_ids=["L"], ccd_code="ATP"),
    ]
    assert task.bonds == [
        Bond(
            atom1=Atom(name="O2P", res_name="SEP", res_id=2, asym_id="A"),
            atom2=Atom(name="P", res_name="ATP", res_id=1, asym_id="L"),
        )
    ]


def test_from_boltz_recovers_unknown_residues_without_modification_records() -> None:
    task, _ = system_from_boltz(
        {
            "version": 1,
            "sequences": [
                {
                    "protein": {
                        "id": "A",
                        "sequence": "XX",
                        "modifications": [{"position": 2, "ccd": "SEP"}],
                    }
                },
                {"rna": {"id": "R", "sequence": "N"}},
                {"dna": {"id": "D", "sequence": "N"}},
            ],
        }
    )

    assert task.entities == [
        Protein(
            asym_ids=["A"],
            sequence="XX",
            mods=[ModifiedResidue(res_id=2, ccd_code="SEP")],
        ),
        RNA(asym_ids=["R"], sequence="N"),
        DNA(asym_ids=["D"], sequence="N"),
    ]


def test_from_boltz_preserves_biological_sequence_with_modification() -> None:
    task, _ = system_from_boltz(
        {
            "version": 1,
            "sequences": [
                {
                    "protein": {
                        "id": "A",
                        "sequence": "MQXGS",
                        "modifications": [{"position": 5, "ccd": "SEP"}],
                    }
                }
            ],
        }
    )
    assert task.entities == [
        Protein(asym_ids=["A"], sequence="MQXGS", mods=[ModifiedResidue(res_id=5, ccd_code="SEP")])
    ]


def test_from_boltz_warns_for_lossy_fields(caplog):
    """Discarded MSA, pocket, template, and affinity fields each emit a warning."""
    with caplog.at_level(logging.WARNING, logger="bioemu2.data_interface.boltz"):
        system_from_boltz(
            {
                "version": 1,
                "sequences": [{"protein": {"id": "A", "sequence": "AC", "msa": "alignment.a3m"}}],
                "constraints": [{"pocket": {"binder": "A", "contacts": [["A", 1]]}}],
                "templates": [{"cif": "template.cif"}],
                "properties": [{"affinity": {"binder": "A"}}],
            }
        )
    messages = " ".join(r.getMessage() for r in caplog.records)
    for marker in ("MSA", "pocket constraint", "Boltz template", "affinity property"):
        assert marker in messages


@pytest.mark.parametrize("version", [None, 2, True, 1.0])
def test_from_boltz_warns_and_continues_for_non_v1_version(caplog, version):
    source = {"sequences": [{"protein": {"id": "A", "sequence": "AC"}}]}
    if version is not None:
        source["version"] = version

    with caplog.at_level(logging.WARNING, logger="bioemu2.data_interface.boltz"):
        task, _ = system_from_boltz(source)

    assert task.entities == [Protein(asym_ids=["A"], sequence="AC")]
    assert f"Boltz version {version!r} is not version 1" in caplog.text


def test_from_boltz_rejects_cyclic_polymer():
    with pytest.raises(NotImplementedError, match="Cyclic Boltz polymers"):
        system_from_boltz(
            {
                "version": 1,
                "sequences": [{"protein": {"id": "A", "sequence": "AC", "cyclic": True}}],
            }
        )


def test_from_boltz_rejects_missing_sequences():
    with pytest.raises(ValueError):
        system_from_boltz({"version": 1, "sequences": []})


@pytest.mark.parametrize("path_type", [Path, str])
def test_convert_boltz_file_paths(tmp_path: Path, path_type) -> None:
    raw = {"version": 1, "sequences": [{"protein": {"id": "A", "sequence": "AC"}}]}
    path = tmp_path / "boltz.yaml"
    path.write_text(yaml.safe_dump(raw))
    output = tmp_path / "converted" / "system.yaml"
    convert_boltz(path_type(path), path_type(output))
    assert System.from_yaml(output) == System(entities=[Protein(asym_ids="A", sequence="AC")])


def test_system_from_boltz_mapping() -> None:
    raw = UserDict({"version": 1, "sequences": [{"protein": {"id": "A", "sequence": "AC"}}]})
    assert system_from_boltz(raw) == (
        System(entities=[Protein(asym_ids="A", sequence="AC")]),
        [],
    )


def test_convert_boltz_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        convert_boltz(tmp_path / "missing.yaml", tmp_path / "system.yaml")


def test_convert_boltz_does_not_interpret_paths_as_yaml(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        convert_boltz("version: 1", tmp_path / "system.yaml")


@pytest.mark.parametrize(
    "fields",
    [
        {"templates": [{"cif": "template.cif"}]},
        {"properties": [{"affinity": {"binder": "L"}, "unknown_property": True}]},
        {"constraints": [{"pocket": {"binder": "A"}, "unknown_constraint": True}]},
        {"unknown_top_level": True},
    ],
)
def test_from_boltz_warns_for_discarded_top_level_fields(fields, caplog) -> None:
    raw = {"version": 1, "sequences": [{"protein": {"id": "A", "sequence": "AC"}}], **fields}
    with caplog.at_level(logging.WARNING, logger="bioemu2.data_interface.boltz"):
        system, _ = system_from_boltz(raw)
    assert system == System(entities=[Protein(asym_ids="A", sequence="AC")])
    assert caplog.records
    for key in ("unknown_property", "unknown_constraint", "unknown_top_level"):
        if key in str(fields):
            assert key in caplog.text


@pytest.mark.parametrize(
    ("kind", "block", "marker"),
    [
        ("protein", {"sequence": "AC", "msa": "msa.a3m"}, "msa"),
        ("protein", {"sequence": "AC", "msa": "empty"}, "msa"),
        ("rna", {"sequence": "AC", "custom": True}, "custom"),
        ("ligand", {"ccd": "ATP", "custom": True}, "custom"),
        (
            "protein",
            {"sequence": "AS", "modifications": [{"position": 2, "ccd": "SEP", "extra": 1}]},
            "extra",
        ),
    ],
)
def test_from_boltz_warns_for_discarded_entity_fields(kind, block, marker, caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="bioemu2.data_interface.boltz"):
        system_from_boltz({"version": 1, "sequences": [{kind: {"id": "A", **block}}]})
    assert marker in caplog.text
    assert "Dropping Boltz" in caplog.text or "IGNORED" in caplog.text


def test_from_boltz_bond_with_other_constraints_warns_without_losing_bond(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="bioemu2.data_interface.boltz"):
        system, contacts = system_from_boltz(
            {
                "version": 1,
                "sequences": [{"protein": {"id": ["A", "B"], "sequence": "C"}}],
                "constraints": [
                    {
                        "bond": {"atom1": ["A", 1, "SG"], "atom2": ["B", 1, "SG"], "order": 1},
                        "pocket": {"binder": "A"},
                        "contact": {"token1": ["A", 1], "token2": ["B", 1]},
                        "extra": True,
                    }
                ],
            }
        )
    assert len(system.bonds) == 1
    assert system.bonds[0].atom1 == Atom(name="SG", res_name="CYS", res_id=1, asym_id="A")
    assert contacts == [
        Contact(
            token1=ContactToken(asym_id="A", res_id=1), token2=ContactToken(asym_id="B", res_id=1)
        )
    ]
    for marker in ("order", "pocket", "extra"):
        assert marker in caplog.text


def test_convert_boltz_preserves_contacts_in_loadable_yaml(tmp_path: Path, caplog) -> None:
    source = tmp_path / "boltz.yaml"
    source.write_text(
        """\
version: 1
sequences:
  - protein: {id: [A, B], sequence: AC, msa: alignment.a3m}
  - ligand: {id: L, ccd: ATP}
constraints:
  - contact: {token1: [B, 2], token2: [L, P], max_distance: 8.5, force: true}
  - contact: {token1: [A, 1], token2: [B, 2]}
  - pocket: {binder: L, contacts: [[A, 1]]}
"""
    )
    output = tmp_path / "converted.yaml"
    with caplog.at_level(logging.WARNING, logger="bioemu2.data_interface.boltz"):
        convert_boltz(source, output)

    system, contacts = load_system_and_contacts(output)
    assert system == System(
        entities=[
            Protein(asym_ids=["A", "B"], sequence="AC"),
            Ligand(asym_ids="L", ccd_code="ATP"),
        ]
    )
    assert contacts == [
        Contact(
            token1=ContactToken(asym_id="B", res_id=2),
            token2=ContactToken(asym_id="L", atom_name="P"),
            max_distance=8.5,
        ),
        Contact(
            token1=ContactToken(asym_id="A", res_id=1),
            token2=ContactToken(asym_id="B", res_id=2),
        ),
    ]
    assert len(yaml.safe_load(output.read_text())["contacts"]) == 2
    assert "MSA" in caplog.text and "IGNORED" in caplog.text
    assert "pocket constraint" in caplog.text
    assert "Ignoring Boltz contact force setting True" in caplog.text
    assert "Dropping Boltz contact" not in caplog.text


@pytest.mark.parametrize(
    ("contact", "message"),
    [
        ({"token1": ["A", 1], "token2": ["A", 2], "max_distance": 25}, "max_distance"),
        ({"token1": ["A", 1], "token2": ["A", 2], "mode": "hard"}, "mode"),
        ({"token1": ["A", 1, "CA"], "token2": ["A", 2]}, "tokens must"),
        ({"token1": ["A", "CA"], "token2": ["A", 2]}, "residue index"),
        ({"token1": ["L", 1], "token2": ["A", 2]}, "atom name"),
        ({"token1": ["A", 3], "token2": ["A", 2]}, "res_id"),
        ({"token1": ["Z", 1], "token2": ["A", 2]}, "Z"),
    ],
)
def test_convert_boltz_rejects_unrepresentable_contacts(
    tmp_path: Path, contact: dict, message: str
) -> None:
    source = tmp_path / "boltz.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "sequences": [
                    {"protein": {"id": "A", "sequence": "AC"}},
                    {"ligand": {"id": "L", "smiles": "CCO"}},
                ],
                "constraints": [{"contact": contact}],
            }
        )
    )
    output = tmp_path / "system.yaml"
    with pytest.raises((ValueError, KeyError), match=message):
        convert_boltz(source, output)
    assert not output.exists()


def test_from_boltz_ligand_ccd() -> None:
    system, _ = system_from_boltz(
        {"version": 1, "sequences": [{"ligand": {"id": "L", "ccd": "ATP"}}]}
    )
    assert system.entities == [Ligand(asym_ids="L", ccd_code="ATP")]


@pytest.mark.parametrize(
    "block",
    [
        {},
        {"smiles": "CCO", "ccd": "ATP"},
    ],
)
def test_from_boltz_rejects_ambiguous_ligands(block) -> None:
    with pytest.raises(ValueError):
        system_from_boltz({"version": 1, "sequences": [{"ligand": {"id": "L", **block}}]})


@pytest.mark.parametrize(
    "source",
    [
        "[]",
        "",
        {"version": 1, "sequences": {"protein": {"id": "A", "sequence": "AC"}}},
        {"version": 1, "sequences": ["protein"]},
        {"version": 1, "sequences": [{"protein": "AC"}]},
        {"version": 1, "sequences": [{"unknown": {"id": "A"}}]},
    ],
)
def test_from_boltz_rejects_malformed_schema(source) -> None:
    with pytest.raises(ValueError):
        system_from_boltz(source)


@pytest.mark.parametrize("chain_ids", [["A", "A"], ["ABCDE", "B"]])
def test_from_boltz_applies_system_chain_validation(chain_ids) -> None:
    with pytest.raises(ValueError):
        system_from_boltz(
            {"version": 1, "sequences": [{"protein": {"id": chain_ids, "sequence": "AC"}}]}
        )


def test_from_boltz_merges_identical_entities(tmp_path: Path) -> None:
    system, _ = system_from_boltz(
        {
            "version": 1,
            "sequences": [
                {"protein": {"id": "A", "sequence": "AC"}},
                {"protein": {"id": "B", "sequence": "AC"}},
            ],
        }
    )
    assert system.entities == [Protein(asym_ids=["A", "B"], sequence="AC")]
    system.to_yaml(tmp_path / "system.yaml")
    assert System.from_yaml(tmp_path / "system.yaml") == system


@pytest.mark.parametrize(
    ("atom", "error"),
    [
        (["missing", 1, "CA"], KeyError),
        (["A", 0, "CA"], ValueError),
        (["A", 3, "CA"], ValueError),
        (["A", 1, "CA", "extra"], ValueError),
    ],
)
def test_from_boltz_rejects_invalid_bond_references(atom, error) -> None:
    with pytest.raises(error):
        system_from_boltz(
            {
                "version": 1,
                "sequences": [{"protein": {"id": "A", "sequence": "AC"}}],
                "constraints": [{"bond": {"atom1": atom, "atom2": ["A", 2, "SG"]}}],
            }
        )


def test_from_boltz_dna_modification_and_bond() -> None:
    system, _ = system_from_boltz(
        {
            "version": 1,
            "sequences": [
                {
                    "dna": {
                        "id": "D",
                        "sequence": "AC",
                        "modifications": [{"position": 2, "ccd": "5CM"}],
                    }
                },
                {"rna": {"id": "R", "sequence": "A"}},
            ],
            "constraints": [{"bond": {"atom1": ["D", 2, "P"], "atom2": ["R", 1, "P"]}}],
        }
    )
    assert system.entities[0] == DNA(
        asym_ids="D", sequence="AC", mods=[ModifiedResidue(res_id=2, ccd_code="5CM")]
    )
    assert system.bonds == [
        Bond(
            atom1=Atom(name="P", res_name="5CM", res_id=2, asym_id="D"),
            atom2=Atom(name="P", res_name="A", res_id=1, asym_id="R"),
        )
    ]
