"""Test structure builders and fixtures shared across model and structure tests."""

from pathlib import Path

import mdtraj
import pytest
import torch

from bioemu2 import structure
from bioemu2.data_interface.const import (
    PROT_LETTER_TO_TOKEN,
    REF_ATOMS,
    RNA_LETTER_TO_TOKEN,
    BondType,
    MolType,
    Token,
)
from bioemu2.structure import (
    AllAtomStructure,
    get_atom_name_mask_for_name_encoding,
    index_to_resname,
    templates,
)
from bioemu2.structure.inference_inputs import _create_all_edges, _create_bond_features
from bioemu2.structure.templates import load_polymer_templates

LETTER_TO_ATOMS = {k: REF_ATOMS[v] for k, v in PROT_LETTER_TO_TOKEN.items()}


def add_structure_template(
    structure: AllAtomStructure,
    template_cache: dict[str, AllAtomStructure],
) -> AllAtomStructure:
    """
    Add structure template information to an all atom structure in the form of template atom
    positions, template formal atom charges and bond information. This function maps the template
    atom type ordering to the ordering used in the basic all atom structure. Note that MD and AFDB
    data will already have the `bond_index` field set, MD based on topologies and AFDB based on the
    structure templates.

    Args:
        structure: All atom structure which the template information should be added to.
        template_cache: CCD template cache from which templates should be loaded.

    Returns:
        Original all atom structure with the fields `template_coords`, `charge` and
        `template_bond_index` added. These contain template coordinates (in Angstrom), formal atom
        charges and the atom indices where atom are bonded in the template.
    """
    assert structure.mol_type.unique().tolist() == [MolType.PROTEIN.value], (
        "Routine currently only supports single chain proteins."
    )

    target_device = structure.residue_idx.device

    template_positions_tmp: list[torch.Tensor] = []
    template_bond_index_tmp: list[torch.Tensor] = []
    template_bond_type_tmp: list[torch.Tensor] = []
    template_atom_charges_tmp: list[torch.Tensor] = []
    residue_atom_offset: int = 0
    template_residue_idx: int = 0
    previous_c_idx: int = -1

    for idx, residue_type in enumerate(structure.downsample_to_tokens(structure.residue_type)):
        template = template_cache[index_to_resname(residue_type.item())]

        # Check which atoms are present, create a map and mask.
        # First get local name encodings.
        residue_name_encoding = structure.name_encoding[structure.residue_idx == idx]
        template_name_encoding = template.name_encoding

        # NOTE: Sometimes, molecular dynamics systems use `NT` to cap the final residue. These are not
        # present in the templates. In this case, we just tell the template `OXT` to behave like a `NT`.
        # Charges are typically neutral, so this could be acceptable but should be fixed in the future.
        # Optimally we do not need templates of any sort.
        if idx == structure.num_residues - 1:
            residue_has_nt = get_atom_name_mask_for_name_encoding(residue_name_encoding, "NT")
            template_has_oxt = get_atom_name_mask_for_name_encoding(template_name_encoding, "OXT")

            if torch.any(residue_has_nt) and not torch.any(template_has_oxt):
                template_name_encoding[template_has_oxt] = residue_name_encoding[residue_has_nt]

        # Create tensor for mapping between both.
        mapping_mask = torch.all(residue_name_encoding[:, None] == template_name_encoding, dim=-1)
        t2c = torch.nonzero(mapping_mask)
        c2t = torch.nonzero(mapping_mask.transpose(0, 1))

        # Convert bond indices in template to structure indices.
        bond_c2t = torch.full_like(template.element, -1)
        bond_c2t[c2t[:, 0]] = c2t[:, 1]

        assert template.template_coords is not None
        template_positions_tmp.append(template.template_coords[t2c[:, 1]])
        # Connect C of last chain with N of current chain.
        if previous_c_idx > -1:
            # Convert to mapped / masked indices and append.
            n_idx = (
                bond_c2t[
                    torch.where(get_atom_name_mask_for_name_encoding(template_name_encoding, "N"))[
                        0
                    ]
                ]
                + residue_atom_offset
            )
            template_bond_index_tmp.append(
                torch.Tensor([[previous_c_idx, n_idx], [n_idx, previous_c_idx]]).long()
            )
            # Add bond types for peptide bonds (C-N backbone connections are single bonds)
            template_bond_type_tmp.append(torch.Tensor([BondType.SINGLE.value] * 2).long())

        previous_c_idx = (
            bond_c2t[
                torch.where(get_atom_name_mask_for_name_encoding(template_name_encoding, "C"))[0]
            ]
            + residue_atom_offset
        )

        # Add within sidechain bonds. Remove empty mappings (-1) and add offset.
        # There will be bonds to atoms not in structure (mainly capping OXT). Remove them.
        mapped_template_bonds = bond_c2t[template.bond_index]
        mask = torch.logical_and(*(mapped_template_bonds > -1))
        template_bond_index_tmp.append(mapped_template_bonds[:, mask] + residue_atom_offset)
        template_bond_type_tmp.append(template.bond_type[mask])
        template_atom_charges_tmp.append(template.charge[t2c[:, 1]].to(target_device))

        residue_atom_offset += t2c.shape[0]
        template_residue_idx += 1

    template_atom_charges = torch.cat(template_atom_charges_tmp).to(target_device)
    template_positions = torch.cat(template_positions_tmp).to(target_device)
    template_bond_index = torch.cat(template_bond_index_tmp, dim=1).to(target_device)
    # Use proper bond types from CCD templates
    template_bond_type = torch.cat(template_bond_type_tmp).to(target_device)

    return structure.replace(
        template_coords=template_positions,
        bond_index=template_bond_index,
        bond_type=template_bond_type,
        charge=template_atom_charges,
    )


def build_test_structure(test_sequence: str = "YDPE") -> structure.AllAtomStructure:
    """Build the canonical 4-residue test structure (YDPE). Usable outside pytest fixtures."""
    if test_sequence != "YDPE":
        raise ValueError(
            "build_test_structure currently only supports test_sequence='YDPE' (arrays are hard-coded)."
        )
    num_residues = 4
    num_nodes = 36
    residue_idx = torch.LongTensor(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
        ]
    )
    residue_type = torch.LongTensor(
        [
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
        ]
    )
    element = torch.LongTensor(
        [
            7,
            6,
            6,
            8,
            6,
            6,
            6,
            6,
            6,
            6,
            6,
            8,
            7,
            6,
            6,
            8,
            6,
            6,
            8,
            8,
            7,
            6,
            6,
            8,
            6,
            6,
            6,
            7,
            6,
            6,
            8,
            6,
            6,
            6,
            8,
            8,
        ]
    )
    bond_index = torch.LongTensor(
        [
            [0, 1],
            [1, 0],
            [1, 2],
            [1, 4],
            [2, 1],
            [2, 3],
            [3, 2],
            [4, 1],
            [4, 5],
            [5, 4],
            [5, 6],
            [5, 7],
            [6, 5],
            [6, 8],
            [7, 5],
            [7, 9],
            [8, 6],
            [8, 10],
            [9, 7],
            [9, 10],
            [10, 8],
            [10, 9],
            [10, 11],
            [11, 10],
            [2, 12],
            [12, 2],
            [12, 13],
            [13, 12],
            [13, 14],
            [13, 16],
            [14, 13],
            [14, 15],
            [15, 14],
            [16, 13],
            [16, 17],
            [17, 16],
            [17, 18],
            [17, 19],
            [18, 17],
            [19, 17],
            [14, 20],
            [20, 14],
            [20, 21],
            [20, 26],
            [21, 20],
            [21, 22],
            [21, 24],
            [22, 21],
            [22, 23],
            [23, 22],
            [24, 21],
            [24, 25],
            [25, 24],
            [25, 26],
            [26, 20],
            [26, 25],
            [22, 27],
            [27, 22],
            [27, 28],
            [28, 27],
            [28, 29],
            [28, 31],
            [29, 28],
            [29, 30],
            [30, 29],
            [31, 28],
            [31, 32],
            [32, 31],
            [32, 33],
            [33, 32],
            [33, 34],
            [33, 35],
            [34, 33],
            [35, 33],
        ],
    ).transpose(0, 1)

    edge_index = _create_all_edges(num_nodes)
    bond_features = _create_bond_features(num_nodes, bond_index)
    # NOTE: update these for testing when multi-chain is supported for MD data.
    asym_id = torch.zeros((num_residues,), dtype=torch.long)
    sym_id = torch.zeros_like(asym_id)
    entity_id = torch.zeros_like(asym_id)
    mol_type = torch.full_like(residue_idx, MolType.PROTEIN.value)

    atom_names: list[str] = [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
        "OH",
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "OD1",
        "OD2",
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "OE1",
        "OE2",
    ]
    name_encoding = torch.LongTensor(list(map(structure.encode_atom_name, atom_names)))
    residue_labels = [structure.index_to_pdb_resname(ridx) for ridx in residue_type.cpu().tolist()]

    # Use realistic bond types for the test sequence "YDPE"
    # Most bonds are single, but tyrosine (Y) has aromatic bonds in its phenolic ring
    bond_type = torch.full((bond_index.shape[1],), BondType.SINGLE.value, dtype=torch.long)
    TYROSINE_AROMATIC_BONDS = [
        set(["CG", "CD1"]),
        set(["CG", "CD2"]),
        set(["CD1", "CE1"]),
        set(["CD2", "CE2"]),
        set(["CE1", "CZ"]),
        set(["CE2", "CZ"]),
    ]
    CARBONYL_DOUBLE_BONDS = [
        set(["C", "O"]),  # Peptide bond.
        set(["CG", "OD1"]),  # ASP side chain.
        set(["CD", "OE1"]),  # GLU side chain.
    ]

    for itype, (i, j) in enumerate(bond_index.transpose(0, 1)):
        atom_i, atom_j = atom_names[i], atom_names[j]
        if set([atom_i, atom_j]) in TYROSINE_AROMATIC_BONDS:
            bond_type[itype] = BondType.AROMATIC.value
        elif set([atom_i, atom_j]) in CARBONYL_DOUBLE_BONDS:
            bond_type[itype] = BondType.DOUBLE.value  # C=O

    # Token index on CA atoms (also doubles as distogram atom index for testing).
    token_center_atom_index = torch.where(
        torch.Tensor([atom_name == "CA" for atom_name in atom_names])
    )[0]

    base = structure.AllAtomStructure(
        num_nodes=num_nodes,
        num_residues=num_residues,
        num_tokens=num_residues,
        element=element,
        residue_idx=residue_idx,
        residue_type=residue_type,
        edge_index=edge_index,
        edge_is_bonded=bond_features,
        bond_index=bond_index,
        bond_type=bond_type,
        name_encoding=name_encoding,
        atom_mask=torch.ones_like(element).bool(),
        asym_id=asym_id,
        entity_id=entity_id,
        sym_id=sym_id,
        mol_type=mol_type,
        sequence=test_sequence,
        token_center_atom_index=token_center_atom_index,
        token_idx_at_atom=residue_idx,
        distogram_index=token_center_atom_index,
        residue_labels=residue_labels,
    )

    # AlphaFold3/Boltz reference-conformer features (`template_coords`/`charge`, consumed by the
    # model via `feature_converter`). In production these are populated at structure-construction
    # time (from the cached HDF5 conformer for PDB data, or from the polymer template cache for
    # MD/inference). The synthetic structure is hand-built, so we source them from the polymer
    # template cache here while keeping the hand-built bonds unchanged.
    templated = add_structure_template(base, load_polymer_templates())
    return base.replace(template_coords=templated.template_coords, charge=templated.charge)


@pytest.fixture
def test_sequence() -> str:
    """Sequence for testing."""
    return "YDPE"


@pytest.fixture
def test_topology(test_sequence) -> mdtraj.Topology:
    """Topology based on test sequence."""
    topology = mdtraj.Topology()
    chain = topology.add_chain()

    for letter in test_sequence:
        residue_type = structure.letter_to_resname(letter)
        residue = topology.add_residue(residue_type, chain)

        for atom_name in LETTER_TO_ATOMS[letter]:
            atom_element = mdtraj.element.Element._elements_by_symbol[atom_name[0]]
            topology.add_atom(atom_name, atom_element, residue)

    topology.create_standard_bonds()
    return topology


@pytest.fixture
def test_structure(test_sequence) -> structure.AllAtomStructure:
    """Structure object corresponding to the test topology."""
    return build_test_structure(test_sequence)


@pytest.fixture
def test_sequence_rna() -> str:
    """RNA sequence for testing."""
    return "GAC"


@pytest.fixture
def test_topology_rna(test_sequence_rna) -> mdtraj.Topology:
    """RNA topology for structure conversion test."""
    topology = mdtraj.Topology()
    chain = topology.add_chain()

    for letter in test_sequence_rna:
        residue = topology.add_residue(letter, chain)

        for atom_name in REF_ATOMS[RNA_LETTER_TO_TOKEN[letter]]:
            atom_element = mdtraj.element.Element._elements_by_symbol[atom_name[0]]
            topology.add_atom(atom_name, atom_element, residue)

    topology.create_standard_bonds()
    residues = list(topology.residues)
    for first, second in zip(residues[:-1], residues[1:]):
        oxygen = next(atom for atom in first.atoms if atom.name == "O3'")
        phosphorus = next(atom for atom in second.atoms if atom.name == "P")
        topology.add_bond(oxygen, phosphorus)
    return topology


@pytest.fixture
def test_structure_rna(test_sequence_rna) -> structure.AllAtomStructure:
    """Structure object corresponding to the RNA test sequence."""
    num_residues = 3
    num_nodes = 65
    atom_names = [
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "O2'",
        "C1'",
        "N9",
        "C8",
        "N7",
        "C5",
        "C6",
        "O6",
        "N1",
        "C2",
        "N2",
        "N3",
        "C4",
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "O2'",
        "C1'",
        "N9",
        "C8",
        "N7",
        "C5",
        "C6",
        "N6",
        "N1",
        "C2",
        "N3",
        "C4",
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "O2'",
        "C1'",
        "N1",
        "C2",
        "O2",
        "N3",
        "C4",
        "N4",
        "C5",
        "C6",
    ]
    residue_idx = torch.LongTensor(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
        ]
    )
    residue_type = torch.LongTensor(
        [
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            24,
            23,
            23,
            23,
            23,
            23,
            23,
            23,
            23,
            23,
            23,
            23,
            23,
            23,
            23,
            23,
            23,
            23,
            23,
            23,
            23,
            23,
            23,
            25,
            25,
            25,
            25,
            25,
            25,
            25,
            25,
            25,
            25,
            25,
            25,
            25,
            25,
            25,
            25,
            25,
            25,
            25,
            25,
        ]
    )
    element = torch.LongTensor(
        [
            15,
            8,
            8,
            8,
            6,
            6,
            8,
            6,
            8,
            6,
            8,
            6,
            7,
            6,
            7,
            6,
            6,
            8,
            7,
            6,
            7,
            7,
            6,
            15,
            8,
            8,
            8,
            6,
            6,
            8,
            6,
            8,
            6,
            8,
            6,
            7,
            6,
            7,
            6,
            6,
            7,
            7,
            6,
            7,
            6,
            15,
            8,
            8,
            8,
            6,
            6,
            8,
            6,
            8,
            6,
            8,
            6,
            7,
            6,
            8,
            7,
            6,
            7,
            6,
            6,
        ]
    )
    bond_index = torch.LongTensor(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 0],
            [2, 0],
            [3, 0],
            [3, 4],
            [4, 3],
            [4, 5],
            [5, 4],
            [5, 6],
            [5, 7],
            [6, 5],
            [6, 11],
            [7, 5],
            [7, 8],
            [7, 9],
            [8, 7],
            [9, 7],
            [9, 10],
            [9, 11],
            [10, 9],
            [11, 6],
            [11, 9],
            [11, 12],
            [12, 11],
            [12, 13],
            [12, 22],
            [13, 12],
            [13, 14],
            [14, 13],
            [14, 15],
            [15, 14],
            [15, 16],
            [15, 22],
            [16, 15],
            [16, 17],
            [16, 18],
            [17, 16],
            [18, 16],
            [18, 19],
            [19, 18],
            [19, 20],
            [19, 21],
            [20, 19],
            [21, 19],
            [21, 22],
            [22, 12],
            [22, 15],
            [22, 21],
            [8, 23],
            [23, 8],
            [23, 24],
            [23, 25],
            [23, 26],
            [24, 23],
            [25, 23],
            [26, 23],
            [26, 27],
            [27, 26],
            [27, 28],
            [28, 27],
            [28, 29],
            [28, 30],
            [29, 28],
            [29, 34],
            [30, 28],
            [30, 31],
            [30, 32],
            [31, 30],
            [32, 30],
            [32, 33],
            [32, 34],
            [33, 32],
            [34, 29],
            [34, 32],
            [34, 35],
            [35, 34],
            [35, 36],
            [35, 44],
            [36, 35],
            [36, 37],
            [37, 36],
            [37, 38],
            [38, 37],
            [38, 39],
            [38, 44],
            [39, 38],
            [39, 40],
            [39, 41],
            [40, 39],
            [41, 39],
            [41, 42],
            [42, 41],
            [42, 43],
            [43, 42],
            [43, 44],
            [44, 35],
            [44, 38],
            [44, 43],
            [31, 45],
            [45, 31],
            [45, 46],
            [45, 47],
            [45, 48],
            [46, 45],
            [47, 45],
            [48, 45],
            [48, 49],
            [49, 48],
            [49, 50],
            [50, 49],
            [50, 51],
            [50, 52],
            [51, 50],
            [51, 56],
            [52, 50],
            [52, 53],
            [52, 54],
            [53, 52],
            [54, 52],
            [54, 55],
            [54, 56],
            [55, 54],
            [56, 51],
            [56, 54],
            [56, 57],
            [57, 56],
            [57, 58],
            [57, 64],
            [58, 57],
            [58, 59],
            [58, 60],
            [59, 58],
            [60, 58],
            [60, 61],
            [61, 60],
            [61, 62],
            [61, 63],
            [62, 61],
            [63, 61],
            [63, 64],
            [64, 57],
            [64, 63],
        ]
    ).transpose(0, 1)
    edge_index = _create_all_edges(num_nodes)
    bond_features = _create_bond_features(num_nodes, bond_index)
    # NOTE: update these for testing when multi-chain is supported for MD data.
    asym_id = torch.zeros((num_residues,), dtype=torch.long)
    sym_id = torch.zeros_like(asym_id)
    entity_id = torch.zeros_like(asym_id)
    mol_type = torch.full_like(residue_idx, MolType.RNA.value)

    name_encoding = torch.LongTensor(list(map(structure.encode_atom_name, atom_names)))
    residue_labels = [structure.index_to_pdb_resname(ridx) for ridx in residue_type.cpu().tolist()]

    # Token index on C1' atoms (also doubles as distogram atom index for testing).
    token_center_atom_index = torch.where(
        torch.Tensor([atom_name == "C1'" for atom_name in atom_names])
    )[0]

    return structure.AllAtomStructure(
        num_nodes=num_nodes,
        num_residues=num_residues,
        num_tokens=num_residues,
        element=element,
        residue_idx=residue_idx,
        residue_type=residue_type,
        edge_index=edge_index,
        edge_is_bonded=bond_features,
        bond_index=bond_index,
        name_encoding=name_encoding,
        atom_mask=torch.ones_like(element).bool(),
        asym_id=asym_id,
        entity_id=entity_id,
        sym_id=sym_id,
        mol_type=mol_type,
        sequence=test_sequence_rna,
        token_index=token_center_atom_index,
        token_idx_at_atom=residue_idx,
        distogram_index=token_center_atom_index,
        residue_labels=residue_labels,
    )


@pytest.fixture
def multichain_topology() -> mdtraj.Topology:
    """
    Test topology containing multiple chains (two identical for sym ID / entity ID checking) and a
    different chain.
    """

    topology = mdtraj.Topology()

    # Build chains A and B. Both have the same sequence (should be detected by sym ID
    # and entity ID).
    residues_chain_ab = ["ALA", "ARG"]

    chain_a = topology.add_chain()
    for residue in residues_chain_ab:
        res_a = topology.add_residue(residue, chain_a)
        for atom_name in REF_ATOMS[Token[residue]]:
            atom_element = mdtraj.element.Element._elements_by_symbol[atom_name[:1]]
            topology.add_atom(atom_name, atom_element, res_a)

    chain_b = topology.add_chain()
    for residue in residues_chain_ab:
        res_b = topology.add_residue(residue, chain_b)
        for atom_name in REF_ATOMS[Token[residue]]:
            atom_element = mdtraj.element.Element._elements_by_symbol[atom_name[:1]]
            topology.add_atom(atom_name, atom_element, res_b)

    # Build chain C with a different sequence.
    residues_chain_c = ["GLN", "GLU"]
    chain_c = topology.add_chain()

    for residue in residues_chain_c:
        res_c = topology.add_residue(residue, chain_c)
        for atom_name in REF_ATOMS[Token[residue]]:
            atom_element = mdtraj.element.Element._elements_by_symbol[atom_name[:1]]
            topology.add_atom(atom_name, atom_element, res_c)

    # Create bonds.
    topology.create_standard_bonds()

    return topology


@pytest.fixture
def multichain_structure() -> structure.AllAtomStructure:
    """
    Structure corresponding to the `multichain_topology` fixture.
    """
    # Global properties.
    sequence = "AR:AR:QE"
    num_nodes = 50
    num_residues = 6
    num_tokens = 6

    # Token level properties.
    asym_id = torch.LongTensor([0, 0, 1, 1, 2, 2])
    # Only the first two chains should see an incrementation here since they are identical.
    sym_id = torch.LongTensor([0, 0, 1, 1, 0, 0])
    # Fist two chains (four residues) should get same ID since they are the same.
    entity_id = torch.LongTensor([0, 0, 0, 0, 1, 1])
    token_center_atom_index = torch.LongTensor([1, 6, 17, 22, 33, 42])
    distogram_index = torch.LongTensor([4, 9, 20, 25, 36, 45])

    # Atom level properties.
    mol_type = torch.zeros(num_nodes, dtype=torch.long)
    token_idx_at_atom = torch.LongTensor(
        [
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
        ]
    )
    residue_idx = torch.LongTensor(
        [
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
        ]
    )
    residue_type = torch.LongTensor(
        [
            2,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            2,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            7,
            7,
            7,
            7,
            7,
            7,
            7,
            7,
            7,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
        ]
    )

    atom_names = [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "NE",
        "CZ",
        "NH1",
        "NH2",
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "NE",
        "CZ",
        "NH1",
        "NH2",
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "OE1",
        "NE2",
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "OE1",
        "OE2",
    ]
    name_encoding = torch.LongTensor(list(map(structure.encode_atom_name, atom_names)))
    element = torch.LongTensor(
        [
            7,
            6,
            6,
            8,
            6,
            7,
            6,
            6,
            8,
            6,
            6,
            6,
            7,
            6,
            7,
            7,
            7,
            6,
            6,
            8,
            6,
            7,
            6,
            6,
            8,
            6,
            6,
            6,
            7,
            6,
            7,
            7,
            7,
            6,
            6,
            8,
            6,
            6,
            6,
            8,
            7,
            7,
            6,
            6,
            8,
            6,
            6,
            6,
            8,
            8,
        ]
    )
    residue_labels = [structure.index_to_pdb_resname(ridx) for ridx in residue_type.cpu().tolist()]
    return structure.AllAtomStructure(
        num_nodes=num_nodes,
        num_residues=num_residues,
        num_tokens=num_tokens,
        bond_index=torch.zeros((2, 0), dtype=torch.long),
        element=element,
        residue_idx=residue_idx,
        residue_type=residue_type,
        name_encoding=name_encoding,
        atom_mask=torch.ones_like(element).bool(),
        asym_id=asym_id,
        entity_id=entity_id,
        sym_id=sym_id,
        mol_type=mol_type,
        sequence=sequence,
        token_center_atom_index=token_center_atom_index,
        token_idx_at_atom=token_idx_at_atom,
        distogram_index=distogram_index,
        residue_labels=residue_labels,
    )


@pytest.fixture
def multichain_ligand_topology() -> mdtraj.Topology:
    """
    Test topology containing multiple chains (two identical for sym ID / entity ID checking), a
    chain with a modified residue (adding a Seaborgium atom) and a chain with a single atom
    Meitnerium ligand.
    """

    topology = mdtraj.Topology()

    # Build chains A and B. Both have the same sequence (should be detected by sym ID
    # and entity ID).
    residues_chain_ab = ["ALA", "ARG"]

    chain_a = topology.add_chain()
    for residue in residues_chain_ab:
        res_a = topology.add_residue(residue, chain_a)

        for atom_name in REF_ATOMS[Token[residue]]:
            atom_element = mdtraj.element.Element._elements_by_symbol[atom_name[:1]]
            topology.add_atom(atom_name, atom_element, res_a)

    chain_b = topology.add_chain()
    for residue in residues_chain_ab:
        res_b = topology.add_residue(residue, chain_b)

        for atom_name in REF_ATOMS[Token[residue]]:
            atom_element = mdtraj.element.Element._elements_by_symbol[atom_name[:1]]
            topology.add_atom(atom_name, atom_element, res_b)

    # Build chain C with a different sequence.
    residues_chain_c = ["GLN", "GLU"]
    chain_c = topology.add_chain()

    for residue in residues_chain_c:
        res_c = topology.add_residue(residue, chain_c)
        for atom_name in REF_ATOMS[Token[residue]]:
            atom_element = mdtraj.element.Element._elements_by_symbol[atom_name[:1]]
            topology.add_atom(atom_name, atom_element, res_c)

    # Add extra spicy atom so residue should be flagged as modified.
    atom_element = mdtraj.element.Element._elements_by_symbol["SG"]
    topology.add_atom("SG", atom_element, res_c)

    # And chain D is an amazing ligand.
    chain_d = topology.add_chain()
    res_d = topology.add_residue("LIG", chain_d)
    atom_element = mdtraj.element.Element._elements_by_symbol["MT"]
    topology.add_atom("MT", atom_element, res_d)

    # Create bonds.
    topology.create_standard_bonds()

    return topology


@pytest.fixture
def multichain_ligand_structure() -> structure.AllAtomStructure:
    """
    Structure expected for the `test_topology_multichain_ligand` fixture. First modified residue
    should be labeled as `GLU`, second as `LIG`.
    """
    # Global properties.
    sequence = "AR:AR:QX:X"
    num_nodes = 52
    num_residues = 7
    num_tokens = 16

    # Token level properties.
    asym_id = torch.LongTensor([0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3])
    # Only the first two chains should see an incrementation here since they are identical.
    sym_id = torch.LongTensor([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # Fist two chains (four residues) should get same ID since they are the same.
    entity_id = torch.LongTensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2])
    token_center_atom_index = torch.LongTensor(
        [1, 6, 17, 22, 33, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    )
    distogram_index = torch.LongTensor(
        [4, 9, 20, 25, 36, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    )

    # Atom level properties.
    mol_type = torch.zeros(num_nodes, dtype=torch.long)
    mol_type[-1] = MolType.NONPOLYMER.value
    token_idx_at_atom = torch.LongTensor(
        [
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
        ]
    )
    residue_idx = torch.LongTensor(
        [
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            6,
        ]
    )
    residue_type = torch.LongTensor(
        [
            2,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            2,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            7,
            7,
            7,
            7,
            7,
            7,
            7,
            7,
            7,
            22,
            22,
            22,
            22,
            22,
            22,
            22,
            22,
            22,
            22,
            22,
        ]
    )

    atom_names = [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "NE",
        "CZ",
        "NH1",
        "NH2",
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "NE",
        "CZ",
        "NH1",
        "NH2",
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "OE1",
        "NE2",
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "OE1",
        "OE2",
        "SG",
        "MT",
    ]
    name_encoding = torch.LongTensor(list(map(structure.encode_atom_name, atom_names)))
    element = torch.LongTensor(
        [
            7,
            6,
            6,
            8,
            6,
            7,
            6,
            6,
            8,
            6,
            6,
            6,
            7,
            6,
            7,
            7,
            7,
            6,
            6,
            8,
            6,
            7,
            6,
            6,
            8,
            6,
            6,
            6,
            7,
            6,
            7,
            7,
            7,
            6,
            6,
            8,
            6,
            6,
            6,
            8,
            7,
            7,
            6,
            6,
            8,
            6,
            6,
            6,
            8,
            8,
            106,
            109,
        ]
    )
    residue_labels = [structure.index_to_pdb_resname(ridx) for ridx in residue_type.cpu().tolist()]
    residue_labels = [
        "GLU" if resid == 5 else res
        for res, resid in zip(residue_labels, residue_idx.cpu().tolist())
    ]
    residue_labels = [
        "LIG" if resid == 6 else res
        for res, resid in zip(residue_labels, residue_idx.cpu().tolist())
    ]
    return structure.AllAtomStructure(
        num_nodes=num_nodes,
        num_residues=num_residues,
        num_tokens=num_tokens,
        bond_index=torch.zeros((2, 0), dtype=torch.long),
        element=element,
        residue_idx=residue_idx,
        residue_type=residue_type,
        name_encoding=name_encoding,
        atom_mask=torch.ones_like(element).bool(),
        asym_id=asym_id,
        entity_id=entity_id,
        sym_id=sym_id,
        mol_type=mol_type,
        sequence=sequence,
        token_center_atom_index=token_center_atom_index,
        token_idx_at_atom=token_idx_at_atom,
        distogram_index=distogram_index,
        residue_labels=residue_labels,
    )


@pytest.fixture
def multichain_ccd_modified_topology() -> mdtraj.Topology:
    """Topology with multiple chains and modified residue taken from the CCD."""

    topology = mdtraj.Topology()
    atoms_NLE = ["N", "CA", "C", "O", "CB", "CG", "CD", "CE"]

    # Build chains A and B. Both have the same sequence (should be detected by sym ID
    # and entity ID).
    residues_chain_ab = ["ALA", "ARG"]

    chain_a = topology.add_chain()
    for residue in residues_chain_ab:
        res_a = topology.add_residue(residue, chain_a)

        for atom_name in REF_ATOMS[Token[residue]]:
            atom_element = mdtraj.element.Element._elements_by_symbol[atom_name[:1]]
            topology.add_atom(atom_name, atom_element, res_a)

    chain_b = topology.add_chain()
    for residue in residues_chain_ab:
        res_b = topology.add_residue(residue, chain_b)

        for atom_name in REF_ATOMS[Token[residue]]:
            atom_element = mdtraj.element.Element._elements_by_symbol[atom_name[:1]]
            topology.add_atom(atom_name, atom_element, res_b)

    # Build chain C with a different sequence.
    residues_chain_c = ["NLE", "GLN"]
    chain_c = topology.add_chain()

    for residue in residues_chain_c:
        res_c = topology.add_residue(residue, chain_c)

        if residue == "NLE":
            ref_atoms = atoms_NLE
        else:
            ref_atoms = REF_ATOMS[Token[residue]]

        for atom_name in ref_atoms:
            atom_element = mdtraj.element.Element._elements_by_symbol[atom_name[:1]]
            topology.add_atom(atom_name, atom_element, res_c)

    # Create bonds.
    topology.create_standard_bonds()

    return topology


@pytest.fixture
def multichain_ccd_modified_structure() -> structure.AllAtomStructure:
    """
    Structure corresponding to `multichain_ccd_modified_topology`. The modified residue is NLE.
    """
    # Global properties.
    sequence = "AR:AR:XQ"
    num_nodes = 49
    num_residues = 6
    num_tokens = 13

    # Token level properties.
    asym_id = torch.LongTensor([0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    # Only the first two chains should see an incrementation here since they are identical.
    sym_id = torch.LongTensor([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # Fist two chains (four residues) should get same ID since they are the same.
    entity_id = torch.LongTensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    token_center_atom_index = torch.LongTensor([1, 6, 17, 22, 32, 33, 34, 35, 36, 37, 38, 39, 41])
    distogram_index = torch.LongTensor([4, 9, 20, 25, 32, 33, 34, 35, 36, 37, 38, 39, 44])

    # Atom level properties.
    mol_type = torch.zeros(num_nodes, dtype=torch.long)
    token_idx_at_atom = torch.LongTensor(
        [
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
        ]
    )
    residue_idx = torch.LongTensor(
        [
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
        ]
    )
    residue_type = torch.LongTensor(
        [
            2,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            2,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            22,
            22,
            22,
            22,
            22,
            22,
            22,
            22,
            7,
            7,
            7,
            7,
            7,
            7,
            7,
            7,
            7,
        ]
    )

    atom_names = [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "NE",
        "CZ",
        "NH1",
        "NH2",
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "NE",
        "CZ",
        "NH1",
        "NH2",
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "CE",
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "OE1",
        "NE2",
    ]

    name_encoding = torch.LongTensor(list(map(structure.encode_atom_name, atom_names)))
    element = torch.LongTensor(
        [
            7,
            6,
            6,
            8,
            6,
            7,
            6,
            6,
            8,
            6,
            6,
            6,
            7,
            6,
            7,
            7,
            7,
            6,
            6,
            8,
            6,
            7,
            6,
            6,
            8,
            6,
            6,
            6,
            7,
            6,
            7,
            7,
            7,
            6,
            6,
            8,
            6,
            6,
            6,
            6,
            7,
            6,
            6,
            8,
            6,
            6,
            6,
            8,
            7,
        ]
    )
    residue_labels = [structure.index_to_pdb_resname(ridx) for ridx in residue_type.cpu().tolist()]
    residue_labels = [res if not res == "UNK" else "NLE" for res in residue_labels]
    return structure.AllAtomStructure(
        num_nodes=num_nodes,
        num_residues=num_residues,
        num_tokens=num_tokens,
        bond_index=torch.zeros((2, 0), dtype=torch.long),
        element=element,
        residue_idx=residue_idx,
        residue_type=residue_type,
        name_encoding=name_encoding,
        atom_mask=torch.ones_like(element).bool(),
        asym_id=asym_id,
        entity_id=entity_id,
        sym_id=sym_id,
        mol_type=mol_type,
        sequence=sequence,
        token_center_atom_index=token_center_atom_index,
        token_idx_at_atom=token_idx_at_atom,
        distogram_index=distogram_index,
        residue_labels=residue_labels,
    )


CCD_ASSET_DIR = Path(__file__).parent / "ccd"

_load_ccd_cifwrapper = templates._load_ccd_cifwrapper


@pytest.fixture(autouse=True)
def vendored_ccd_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Serve CCD component files from tracked test assets instead of downloading them.

    Redirects CCD component lookups to the vendored ``.cif`` files under ``ccd/`` so the
    CCD-based tests are deterministic and never reach the network. The compiled-template
    npz cache is redirected to a temp dir (its key ignores CIF content), and the CIF parse
    cache is cleared, so the vendored component cannot be shadowed by a previously
    downloaded one.
    """

    def asset_ccd_path(ccd_code: str) -> Path:
        return CCD_ASSET_DIR / f"{ccd_code}.cif"

    monkeypatch.setattr(templates, "_get_ccd_path", asset_ccd_path)
    monkeypatch.setenv("BIOEMU2_TEMPLATE_CACHE", str(tmp_path / "templates"))
    _load_ccd_cifwrapper.cache_clear()
    yield
    _load_ccd_cifwrapper.cache_clear()
