from pathlib import Path

import pytest
import torch
import yaml
from rdkit import Chem

from bioemu2.conditioning.contact import contact_conditioning_info
from bioemu2.conditioning.method import method_types_ids
from bioemu2.data_interface import (
    RNA,
    Contact,
    ContactToken,
    Ligand,
    ModifiedResidue,
    Protein,
    System,
)
from bioemu2.data_interface.const import (
    DNA_LETTER_TO_TOKEN,
    PROT_LETTER_TO_TOKEN,
    REF_ATOMS,
    RNA_LETTER_TO_TOKEN,
    MolType,
    Token,
)
from bioemu2.structure import AllAtomStructure, add_mode_conditioning
from bioemu2.structure.batch import batch_collate
from bioemu2.structure.inference_inputs import topology_from_structure


@pytest.mark.parametrize("input_format", ["yaml", "dict"])
@pytest.mark.parametrize(
    ("ligand_spec", "ligand_smiles"),
    [
        ({"smiles": "CC1=CC=CC=C1"}, "CC1=CC=CC=C1"),
        ({"inchi": "InChI=1S/C2H6/c1-2/h1-2H3"}, "CC"),
        ({"ccd_code": "BEN"}, "NC(=N)c1ccccc1"),
    ],
)
def test_mixed_system_structure(tmp_path: Path, input_format, ligand_spec, ligand_smiles) -> None:
    data = {
        "entities": [
            {"entity_type": "protein", "asym_ids": ["A"], "sequence": "AC"},
            {"entity_type": "protein", "asym_ids": ["B", "C"], "sequence": "GY"},
            {"entity_type": "ligand", "asym_ids": ["D"], **ligand_spec},
            {"entity_type": "rna", "asym_ids": ["E"], "sequence": "AU"},
            {"entity_type": "dna", "asym_ids": ["F"], "sequence": "AT"},
        ]
    }
    if input_format == "yaml":
        path = tmp_path / "system.yaml"
        path.write_text(yaml.safe_dump(data))
        system = System.from_yaml(path)
    else:
        system = System.model_validate(data)
    structure = AllAtomStructure.from_system(system)

    ligand = Chem.MolFromSmiles(ligand_smiles)
    ligand_atoms = ligand.GetNumAtoms()
    residue_tokens = (
        [PROT_LETTER_TO_TOKEN[letter] for letter in "ACGYGY"]
        + [Token.UNK_PROT]
        + [RNA_LETTER_TO_TOKEN[letter] for letter in "AU"]
        + [DNA_LETTER_TO_TOKEN[letter] for letter in "AT"]
    )
    atoms_per_residue = [len(REF_ATOMS[token]) for token in residue_tokens]
    atoms_per_residue[6] = ligand_atoms
    atom_counts = torch.tensor(atoms_per_residue)
    tokens_per_chain = torch.tensor([2, 2, 2, ligand_atoms, 2, 2])
    token_residues = torch.tensor([0, 1, 2, 3, 4, 5] + [6] * ligand_atoms + [7, 8, 9, 10])

    assert structure.num_nodes == sum(atoms_per_residue)
    assert structure.num_residues == 11
    assert structure.num_tokens == 10 + ligand_atoms
    assert structure.sequence == "AC:GY:GY:X:AU:AT"
    assert structure.coords.shape == (structure.num_nodes, 3)
    assert torch.count_nonzero(structure.coords) == 0
    assert structure.template_coords is not None
    assert structure.template_coords.shape == structure.coords.shape
    assert structure.name_encoding.shape == (structure.num_nodes, 4)
    assert structure.charge.shape == (structure.num_nodes,)
    assert structure.atom_mask.shape == (structure.num_nodes,)
    assert torch.all(structure.atom_mask)
    assert structure.bond_index.shape[0] == 2
    assert structure.bond_type.shape == (structure.bond_index.shape[1],)
    torch.testing.assert_close(
        structure.residue_idx, torch.repeat_interleave(torch.arange(11), atom_counts)
    )
    torch.testing.assert_close(
        structure.residue_type,
        torch.repeat_interleave(
            torch.tensor([token.value for token in residue_tokens]), atom_counts
        ),
    )
    torch.testing.assert_close(
        structure.asym_id, torch.repeat_interleave(torch.arange(6), tokens_per_chain)
    )
    torch.testing.assert_close(
        structure.entity_id,
        torch.repeat_interleave(torch.tensor([0, 1, 1, 2, 3, 4]), tokens_per_chain),
    )
    torch.testing.assert_close(
        structure.sym_id,
        torch.repeat_interleave(torch.tensor([0, 0, 1, 0, 0, 0]), tokens_per_chain),
    )
    torch.testing.assert_close(structure.upsample_to_atoms(token_residues), structure.residue_idx)
    torch.testing.assert_close(
        structure.downsample_to_tokens(structure.token_idx_at_atom),
        torch.arange(structure.num_tokens),
    )
    torch.testing.assert_close(
        structure.downsample_to_tokens(structure.mol_type),
        torch.repeat_interleave(
            torch.tensor(
                [MolType.PROTEIN.value] * 3
                + [MolType.NONPOLYMER.value, MolType.RNA.value, MolType.DNA.value]
            ),
            tokens_per_chain,
        ),
    )
    centers = structure.downsample_to_tokens(structure.name_encoding)
    assert centers.shape == (structure.num_tokens, 4)
    assert structure.name_encoding_is_atom("CA")[structure.token_center_atom_index[:6]].all()
    assert structure.name_encoding_is_atom("C1'")[structure.token_center_atom_index[-4:]].all()

    topology = topology_from_structure(structure, renamed_asym_ids=system.chain_ids)
    assert topology.n_atoms == structure.num_nodes
    assert topology.n_residues == structure.num_residues
    assert [chain.chain_id for chain in topology.chains] == system.chain_ids
    ligand_indices = set(structure.residue_idx.eq(6).nonzero().flatten().tolist())
    ligand_bonds = [
        (a, b)
        for a, b in structure.bond_index.T.tolist()
        if a in ligand_indices and b in ligand_indices
    ]
    assert len(ligand_bonds) == ligand.GetNumBonds()


def test_from_system_is_unconditioned_by_default() -> None:
    system = System(entities=[Protein(sequence="AC", asym_ids=["A"])])

    structure = AllAtomStructure.from_system(system)

    assert "method_feature" not in structure
    assert "contact_conditioning" not in structure


def test_from_system_applies_contacts_without_mode() -> None:
    system = System(entities=[Protein(sequence="AC", asym_ids=["A"])])
    contacts = [
        Contact(
            token1=ContactToken(asym_id="A", res_id=1),
            token2=ContactToken(asym_id="A", res_id=2),
            max_distance=8.0,
        )
    ]

    structure = AllAtomStructure.from_system(system, contacts=contacts)

    assert "method_feature" not in structure
    contact = contact_conditioning_info["CONTACT"]
    assert structure.contact_conditioning[0, 1, contact] == 1
    assert structure.contact_conditioning[1, 0, contact] == 1
    assert structure.contact_threshold[0, 1] == 8.0
    assert structure.contact_threshold[1, 0] == 8.0


def test_contacts_resolve_modified_residue_default_and_named_atom() -> None:
    system = System(
        entities=[
            Protein(
                sequence="AT",
                asym_ids=["A"],
                mods=[ModifiedResidue(res_id=2, ccd_code="TPO")],
            )
        ]
    )
    contacts = [
        Contact(
            token1=ContactToken(asym_id="A", res_id=1),
            token2=ContactToken(asym_id="A", res_id=2),
        )
    ]
    structure = AllAtomStructure.from_system(system, contacts=contacts)
    ca = structure.name_encoding_is_atom("CA")[structure.token_center_atom_index]
    modified_ca = torch.where((structure.residue_idx[structure.token_center_atom_index] == 1) & ca)[
        0
    ]
    assert len(modified_ca) == 1
    target = int(modified_ca.item())
    assert structure.contact_conditioning[0, target, contact_conditioning_info["CONTACT"]] == 1

    named = AllAtomStructure.from_system(
        system,
        contacts=[
            Contact(
                token1=ContactToken(asym_id="A", res_id=1),
                token2=ContactToken(asym_id="A", res_id=2, atom_name="OG1"),
            )
        ],
    )
    og = named.name_encoding_is_atom("OG1")[named.token_center_atom_index]
    modified_og = torch.where((named.residue_idx[named.token_center_atom_index] == 1) & og)[0]
    assert len(modified_og) == 1
    assert (
        named.contact_conditioning[0, int(modified_og.item()), contact_conditioning_info["CONTACT"]]
        == 1
    )


def test_contacts_resolve_modified_rna_default_to_c1_prime() -> None:
    system = System(
        entities=[
            RNA(sequence="AA", asym_ids=["R"], mods=[ModifiedResidue(res_id=2, ccd_code="ATP")])
        ]
    )
    structure = AllAtomStructure.from_system(
        system,
        contacts=[
            Contact(
                token1=ContactToken(asym_id="R", res_id=1),
                token2=ContactToken(asym_id="R", res_id=2),
            )
        ],
    )
    token_residue_idx = structure.residue_idx[structure.token_center_atom_index]
    assert (token_residue_idx == 1).sum() > 1
    c1_prime = structure.name_encoding_is_atom("C1'")[structure.token_center_atom_index]
    target = torch.where((token_residue_idx == 1) & c1_prime)[0]
    assert len(target) == 1
    selected = torch.where(
        structure.contact_conditioning[0, :, contact_conditioning_info["CONTACT"]] == 1
    )[0]
    torch.testing.assert_close(selected, target)


def test_contacts_resolve_ligand_atom_at_residue_one() -> None:
    system = System(
        entities=[Protein(sequence="A", asym_ids=["A"]), Ligand(asym_ids="L", smiles="CC")]
    )
    structure = AllAtomStructure.from_system(
        system,
        contacts=[
            Contact(
                token1=ContactToken(asym_id="A", res_id=1),
                token2=ContactToken(asym_id="L", atom_name="C1"),
            )
        ],
    )
    named = structure.name_encoding_is_atom("C1")[structure.token_center_atom_index]
    ligand_c1 = torch.where((structure.asym_id == 1) & named)[0]
    assert len(ligand_c1) == 1
    assert (
        structure.contact_conditioning[
            0, int(ligand_c1.item()), contact_conditioning_info["CONTACT"]
        ]
        == 1
    )


def test_from_system_applies_mode_without_contacts() -> None:
    system = System(entities=[Protein(sequence="AC", asym_ids=["A"])])

    structure = AllAtomStructure.from_system(system, mode="structure")

    assert "contact_conditioning" not in structure
    torch.testing.assert_close(
        structure.method_feature,
        torch.full((structure.num_tokens,), method_types_ids["non-md"], dtype=torch.long),
    )


def test_from_system_validates_explicit_contacts() -> None:
    system = System(entities=[Protein(sequence="AC", asym_ids=["A"])])
    contacts = [
        Contact(
            token1=ContactToken(asym_id="A", res_id=1),
            token2=ContactToken(asym_id="Z", res_id=2),
        )
    ]

    with pytest.raises(ValueError, match="not present in any entity"):
        AllAtomStructure.from_system(system, contacts=contacts)


def _make_single_chain_structure(n: int) -> AllAtomStructure:
    return AllAtomStructure.from_system(
        System(entities=[Protein(sequence="A" * n, asym_ids=["A"])])
    )


@pytest.mark.parametrize("mode", ["x-ray diffraction", "MD"])
def test_add_mode_conditioning(mode: str) -> None:
    structure = _make_single_chain_structure(n=5)
    out = add_mode_conditioning(structure, mode)
    assert out is structure
    torch.testing.assert_close(
        out.method_feature,
        torch.full((5,), method_types_ids[mode.lower()], dtype=torch.long),
    )


def test_method_feature_pads_through_batch_collate() -> None:
    """Per-sample ``(num_tokens,)`` labels pad to ``(B, max_T)`` with zero padding."""
    samples = [add_mode_conditioning(_make_single_chain_structure(n), "afdb") for n in (3, 5, 4)]
    mf = batch_collate(samples).method_feature
    method = method_types_ids["afdb"]
    torch.testing.assert_close(
        mf,
        torch.tensor(
            [
                [method, method, method, 0, 0],
                [method, method, method, method, method],
                [method, method, method, method, 0],
            ]
        ),
    )


def test_add_mode_conditioning_rejects_unknown_mode() -> None:
    structure = _make_single_chain_structure(n=4)
    with pytest.raises(ValueError, match="mode"):
        add_mode_conditioning(structure, "not-a-real-mode")
