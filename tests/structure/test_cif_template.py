import logging
from pathlib import Path

import mdtraj
import pytest
import torch
from pdbecif.mmcif_io import CifFileReader

from bioemu2.data_interface import Ligand, Protein, System
from bioemu2.data_interface.const import BondType
from bioemu2.structure import AllAtomStructure, templates
from bioemu2.structure.cif import save_structure_to_cif
from bioemu2.structure.templates import template_from_cifwrapper

logging.basicConfig(level=logging.INFO)


def check_all_atom_structure_shapes(structure: AllAtomStructure) -> None:
    assert structure.template_coords is not None
    assert structure.template_coords.shape == (structure.num_nodes, 3)
    assert structure.charge.shape[0] == structure.num_nodes
    assert structure.element.shape[0] == structure.num_nodes
    assert structure.name_encoding.shape[0] == structure.num_nodes
    assert structure.bond_index.shape[0] == 2
    assert structure.bond_index.shape[1] == structure.bond_type.shape[0]


def check_bond_type(structure: AllAtomStructure) -> None:
    bond_type_counts = torch.bincount(structure.bond_type, minlength=len(BondType))
    assert torch.unique(structure.bond_type).numel() > 1
    assert bond_type_counts[BondType.OTHER_OR_UNK.value] == 0
    assert structure.bond_index.max() < structure.num_nodes
    assert structure.bond_index.min() >= 0


def test_template_from_cifwrapper():
    cfr = CifFileReader(input="data")
    # get the current file path
    test_file = Path(__file__).parent / "test.cif"
    cif_dictionary = cfr.read(test_file, output="cif_wrapper")
    logging.info("Loaded %d components from CIF file.", len(cif_dictionary))
    for aa in cif_dictionary:
        result = template_from_cifwrapper(cif_dictionary[aa])
        assert isinstance(result, AllAtomStructure), (
            f"Expected AllAtomStructure, got {type(result)}"
        )

        assert result.num_nodes > 0, "Expected non-empty structure."

        check_all_atom_structure_shapes(result)
        check_bond_type(result)


@pytest.mark.parametrize("num_leaving", [0, 1, 2])
def test_heavy_leaving_atom_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, num_leaving: int
) -> None:
    path = tmp_path / "leaving.cif"
    flags = ["Y"] * num_leaving + ["N"] * (2 - num_leaving)
    path.write_text(
        "data_TMP\n"
        "_chem_comp.id TMP\n"
        "loop_\n"
        "_chem_comp_atom.atom_id\n"
        "_chem_comp_atom.type_symbol\n"
        "_chem_comp_atom.charge\n"
        "_chem_comp_atom.pdbx_model_Cartn_x_ideal\n"
        "_chem_comp_atom.pdbx_model_Cartn_y_ideal\n"
        "_chem_comp_atom.pdbx_model_Cartn_z_ideal\n"
        "_chem_comp_atom.pdbx_leaving_atom_flag\n"
        "C C 0 0 0 0 N\n"
        f"O1 O 0 1 0 0 {flags[0]}\n"
        f"O2 O 0 0 1 0 {flags[1]}\n"
        "H H 0 0 0 1 Y\n"
    )
    wrapper = CifFileReader().read(str(path), output="cif_wrapper")["TMP"]

    result = template_from_cifwrapper(wrapper, remove_leaving_atoms=True)

    assert result.num_nodes == 3 - num_leaving
    if num_leaving > 1:
        assert f"Removing {num_leaving} heavy leaving atoms for component TMP" in caplog.text
    else:
        assert "heavy leaving atoms" not in caplog.text


@pytest.mark.parametrize(
    "symbol, expected_number",
    [
        pytest.param("X", mdtraj.element.virtual_site.number, id="unknown"),
        pytest.param("C", mdtraj.element.carbon.number, id="carbon"),
        pytest.param("Xe", mdtraj.element.xenon.number, id="xenon-mixed-case"),
        pytest.param("XE", mdtraj.element.xenon.number, id="xenon-uppercase"),
    ],
)
def test_single_atom_ccd_template_in_system(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    symbol: str,
    expected_number: int,
) -> None:
    (tmp_path / "UNX.cif").write_text(
        "data_UNX\n"
        "_chem_comp.id UNX\n"
        "loop_\n"
        "_chem_comp_atom.atom_id\n"
        "_chem_comp_atom.type_symbol\n"
        "_chem_comp_atom.charge\n"
        "_chem_comp_atom.pdbx_model_Cartn_x_ideal\n"
        "_chem_comp_atom.pdbx_model_Cartn_y_ideal\n"
        "_chem_comp_atom.pdbx_model_Cartn_z_ideal\n"
        f"UNK {symbol} 0 0 0 0\n"
    )
    monkeypatch.setattr(templates, "_get_ccd_path", lambda ccd_code: tmp_path / f"{ccd_code}.cif")
    system = System(
        entities=[
            Protein(sequence="AG", asym_ids=["A"]),
            Ligand(ccd_code="UNX", asym_ids=["B"]),
        ]
    )

    structure = AllAtomStructure.from_system(system)
    assert structure.element[-1].item() == expected_number
    if symbol == "X":
        assert "unknown element X" in caplog.text
    else:
        assert "unknown element X" not in caplog.text
    atom = next(atom for atom in system.topology.atoms if atom.residue.chain.chain_id == "B")
    assert atom.name == "UNK"
    assert atom.element.number == expected_number

    cif_path = tmp_path / "topology.cif"
    save_structure_to_cif(structure, cif_path, renamed_asym_ids=system.chain_ids)
    saved_atom = next(
        atom
        for atom in mdtraj.load_topology(str(cif_path)).atoms
        if atom.residue.chain.chain_id == "B"
    )
    assert saved_atom.name == "UNK"
    assert saved_atom.element.number == expected_number
