"""Construction of model input structures from sequences and ligands."""

import logging
from collections.abc import Collection, Generator, Sequence
from typing import Literal

import einops
import mdtraj
import numpy as np
import torch
from rdkit import Chem

from bioemu2.data_interface.const import (
    CAPPING_ATOMS,
    DNA_LETTER_TO_TOKEN,
    MOLTYPE_TO_UNKNOWN_CCD,
    MOLTYPE_TO_UNKTOKEN,
    PROT_TOKEN_TO_LETTER,
    RNA_LETTER_TO_TOKEN,
    BondType,
    MolType,
    Token,
    get_rdkit_bondtype_from_bondtype,
    is_missing_residue_identity,
    token_to_disto_atom_name,
)
from bioemu2.structure.all_atom_structure import (
    AllAtomStructure,
    ModificationKind,
    PolymerModification,
    decode_atom_name,
    extract_substructure,
    get_atom_name_mask_for_name_encoding,
    index_to_pdb_resname,
    letter_to_resname,
    resname_to_index,
)
from bioemu2.structure.polymer_builder import PolymerStructureBuilder
from bioemu2.structure.templates import (
    LigandTemplateCache,
    _load_ccd_cifwrapper,
    template_from_cifwrapper,
)

LOG = logging.getLogger(__name__)


def broadcast_multiplicity(x: torch.Tensor, multiplicity: int) -> torch.Tensor:
    """
    Broadcast a tensor to the multiplicity dimension.

    Args:
        x: Input tensor.
        multiplicity: Multiplicity factor.

    Returns:
        Tensor with shape expanded to include multiplicity.

    The main application is to broadcast arbitrary dimensional tensors
    with at least one dimension in the first dimension
    """
    return einops.repeat(x, "b ... -> (b m) ...", m=multiplicity)


def _create_all_edges(num_nodes: int) -> torch.LongTensor:
    """
    Create all possible edges given a number of nodes.

    Args:
        num_nodes: Number of nodes.

    Returns:
        Tensor of atom edges with the shape [2, num_nodes**2]
    """
    # Create a meshgrid of indices for the nodes in the i-th graph
    idx = torch.arange(0, num_nodes)
    row, col = torch.meshgrid(idx, idx, indexing="ij")

    # Stack the edge indices the same tensor
    edge_index = torch.stack([row.reshape(-1), col.reshape(-1)], dim=0).long()

    return edge_index


def _create_bond_features(num_nodes: int, bond_index: torch.LongTensor) -> torch.Tensor:
    """
    Create edge features which take the value of one when the edge corresponds to an edge with a
    bond present.

    Args:
        num_nodes: Number of nodes.
        bond_index: Bond edge indices, shape is [2, num_bonds].

    Returns:
        Bonded mask of shape [num_nodes**2, 1].
    """
    bond_features = torch.zeros(num_nodes, num_nodes)
    row, col = bond_index
    bond_features[row, col] = 1.0
    bond_features = bond_features.reshape(-1)

    return bond_features[:, None]


@torch.no_grad()
def inference_input_structure_from_protein_sequence(
    sequence: str,
    template_cache: dict[str, AllAtomStructure],
    modifications: Sequence[PolymerModification] = (),
) -> AllAtomStructure:
    """
    Create an `AllAtomStructure` template object from a protein sequence using CCD templates.

    The template contains zero coordinates, template coordinates and bonds but no edges, which is added by `AddAtomEdges` transform later.

    Args:
        sequence: Original one-letter amino-acid sequence. It could be used for MSA seach, and the
            returned structure retains it as sequence metadata.
        template_cache: Cache of template structures, see `bioemu2.structure.templates` for details.
        modifications: Explicit `(res_id, kind, value)` entries for modified residues. `res_id` is
            1-based and `kind` is `"ccd"` or `"inchi"`. A modification replaces the residue
            structure at its position; the sequence may keep its canonical letter or use `X`.
    Returns:
        `AllAtomStructure` with atom, residue and bonding information. Structure is taken from the
        templates if available.
    """
    if modifications:
        # Contains covalently bound ligands
        return _inference_input_structure_from_polymer_sequence_with_mods(
            sequence=sequence,
            template_cache=template_cache,
            modifications=modifications,
            polymer_type=MolType.PROTEIN,
        )

    coords_tmp: list[torch.Tensor] = []
    residue_type_tmp: list[torch.Tensor] = []
    residue_idx_tmp: list[torch.Tensor] = []
    name_encoding_tmp: list[torch.Tensor] = []
    element_tmp: list[torch.Tensor] = []
    charge_tmp: list[torch.Tensor] = []

    residue_atom_offset: int = 0
    for current_residue_idx, template in enumerate(
        _templates_from_protein_sequence(sequence, template_cache)
    ):
        # Each residue template provides atom-level reference data (template_coords).
        if template.template_coords is None:
            raise ValueError(f"Template coordinates for {current_residue_idx} are missing.")
        coords_tmp.append(template.template_coords)
        name_encoding_tmp.append(template.name_encoding)
        element_tmp.append(template.element)
        charge_tmp.append(template.charge)

        residue_idx_tmp.append(template.residue_idx + current_residue_idx)
        residue_type_tmp.append(template.residue_type)

        residue_atom_offset += template.element.shape[0]

    template_coords = torch.cat(coords_tmp)
    name_encoding = torch.cat(name_encoding_tmp)
    element = torch.cat(element_tmp)
    charge = torch.cat(charge_tmp)
    residue_idx = torch.cat(residue_idx_tmp)
    residue_type = torch.cat(residue_type_tmp)
    num_residues = len(sequence)

    # Set up dummy features for chains.
    asym_id = torch.zeros((num_residues,), dtype=torch.long)
    sym_id = torch.zeros_like(asym_id)
    entity_id = torch.zeros_like(asym_id)
    mol_type = torch.full_like(residue_idx, MolType.PROTEIN.value)

    is_ca_atom = get_atom_name_mask_for_name_encoding(name_encoding, "CA")
    token_center_atom_index = torch.where(is_ca_atom)[0]
    distogram_index = torch.where(
        torch.logical_or(
            get_atom_name_mask_for_name_encoding(name_encoding, "CB"),
            torch.logical_and(residue_type == resname_to_index("GLY"), is_ca_atom),
        )
    )[0]

    residue_labels = [index_to_pdb_resname(res) for res in residue_type.cpu().tolist()]

    return AllAtomStructure(
        coords=torch.zeros_like(template_coords),
        element=element,
        charge=charge,
        residue_idx=residue_idx,
        residue_type=residue_type,
        num_residues=num_residues,
        num_tokens=num_residues,
        bond_index=torch.zeros((2, 0), dtype=torch.long),
        bond_type=torch.zeros((0,), dtype=torch.long),
        num_nodes=element.shape[0],
        name_encoding=name_encoding,
        atom_mask=torch.ones_like(element).bool(),
        asym_id=asym_id,
        entity_id=entity_id,
        sym_id=sym_id,
        mol_type=mol_type,
        sequence=sequence,
        template_coords=template_coords,
        token_center_atom_index=token_center_atom_index,
        token_idx_at_atom=residue_idx,
        distogram_index=distogram_index,
        residue_labels=residue_labels,
    )


def _templates_from_protein_sequence(
    sequence: str, template_cache: dict[str, AllAtomStructure]
) -> Generator[AllAtomStructure, None, None]:
    """
    Create a list of `AllAtomStructure` templates from a protein sequence using CCD templates.

    Args:
        sequence: Protein sequence in one-letter code.
        template_cache: Cache of template structures, see `bioemu2.structure.templates` for details.

    Returns:
        A generator yielding `AllAtomStructure` objects for each residue in the sequence.
    """
    for residue_1 in sequence:
        residue_3 = letter_to_resname(residue_1)
        template = template_cache[residue_3]

        # Terminal OXT atoms are excluded from model features.
        oxt_mask = ~template.name_encoding_is_atom("OXT")
        name_encoding = template.name_encoding[oxt_mask]
        num_atoms = name_encoding.shape[0]

        assert template.template_coords is not None

        yield AllAtomStructure(
            template_coords=template.template_coords[oxt_mask],
            name_encoding=name_encoding,
            element=template.element[oxt_mask],
            charge=template.charge[oxt_mask],
            residue_idx=torch.zeros((num_atoms,), dtype=torch.long),
            residue_type=torch.full((num_atoms,), resname_to_index(residue_3), dtype=torch.long),
        )


@torch.no_grad()
def topology_from_structure(
    structure: AllAtomStructure,
    renamed_asym_ids: Sequence[str] | None = None,
    rename_residues: dict[int, str] | None = None,
) -> mdtraj.Topology:
    """
    Extract residue information etc. from an AllAtomStructure containing a single configuration and construct
    a `mdtraj` topology.

    Args:
        structure: Input AllAtomStructure containing a single configuration, with atoms grouped
            by chain and residue and consecutive polymer residue indices within each chain.
            Residue indices are global and must not restart at chain boundaries.
        renamed_asym_ids (optional): A list of chain identifiers to write chains in the output with.
        rename_residues: Optional dictionary of residue indices (starting from 0) and residue names
          to override the residue names present in `structure.residue_labels`. E.g. `{1 : 'HEM'}`
          can be used to relabel the second residue (not token) to `HEM` in the output topology.
    Returns:
        `mdtraj` topology with 1-based resSeq within each chain; residue.index
        remains 0-based across the topology. Nucleotide O3'-P backbone links
        are repaired after MDTraj creates standard bonds. Explicit structure
        bonds and available protein peptide links are added without duplication.
    """
    num_atoms = structure.element.shape[0]

    # Build base topology and add single chain.
    topology = mdtraj.Topology()
    last_residue_id: int | None = None
    last_asym_id: int | None = None
    chain_residue_number = 0
    atom_chain = structure.allatom_asym_id.cpu().tolist()
    molecule_types = structure.mol_type.cpu().tolist()
    nucleotide_chains: set[int] = set()

    if renamed_asym_ids is not None:
        assert len(renamed_asym_ids) == len(torch.unique(structure.asym_id)), (
            "Unequal number of chains in `asym_ids` and those present in base structure"
        )

    if rename_residues:
        assert max(list(rename_residues.keys())) < structure.num_residues, (
            "Maximum index of residue overrides in `rename_residues` "
            f"exceeds number of residues in structure ({structure.num_residues})."
        )

    for idx in range(num_atoms):
        # Get residue index and type.
        residue_id = structure.residue_idx[idx].detach().cpu().item()

        # Check for custom residue name overrides.
        if rename_residues is not None:
            residue_type = rename_residues.get(residue_id, structure.residue_labels[idx])
        else:
            residue_type = structure.residue_labels[idx]

        asym_id = atom_chain[idx]

        if asym_id != last_asym_id:
            chain_id = renamed_asym_ids[asym_id] if renamed_asym_ids is not None else None
            chain = topology.add_chain(chain_id=chain_id)
            last_asym_id = asym_id
            chain_residue_number = 0

        # If residue index is different from the previous one, start a new residue.
        if residue_id != last_residue_id:
            chain_residue_number += 1
            residue = topology.add_residue(residue_type, chain, resSeq=chain_residue_number)
            if molecule_types[idx] in (MolType.DNA.value, MolType.RNA.value):
                nucleotide_chains.add(chain.index)
            last_residue_id = residue_id

        # Get atom name, element and add atom.
        atom_name = decode_atom_name(structure.name_encoding[idx])
        atom_element = mdtraj.element.Element._elements_by_atomic_number[
            structure.element[idx].detach().cpu().item()
        ]
        topology.add_atom(atom_name, atom_element, residue)

    if not nucleotide_chains:
        topology.create_standard_bonds()
    else:
        # Add the O3'-P bonds MDTraj misses between adjacent nucleic-acid residues and remove
        # the incorrect OP3-P bond it creates between the first and second residues when
        # terminal OP3 is present.
        # TODO: Fix this MDTraj bug upstream in mdtraj/formats/pdb/data/residues.xml:
        # https://github.com/mdtraj/mdtraj/blob/60080d100aa12695f3c7cb1ce7ce8064a736c968/mdtraj/formats/pdb/data/residues.xml#L213
        # The "-" prefix is resolved to the previous residue in create_standard_bonds:
        # https://github.com/mdtraj/mdtraj/blob/60080d100aa12695f3c7cb1ce7ce8064a736c968/mdtraj/core/topology.py#L1130-L1152
        standard = topology.copy()
        standard.create_standard_bonds()
        pairs: set[tuple[int, int]] = set()
        for bond in standard.bonds:
            first, second = bond
            if (
                first.residue != second.residue
                and first.residue.chain.index in nucleotide_chains
                and second.residue.chain.index in nucleotide_chains
                and {first.name, second.name} == {"OP3", "P"}
            ):
                continue
            topology.add_bond(
                topology.atom(first.index),
                topology.atom(second.index),
                type=bond.type,
                order=bond.order,
            )
            pairs.add(tuple(sorted((first.index, second.index))))
        for chain_index in sorted(nucleotide_chains):
            residues = list(topology.chain(chain_index).residues)
            for first, second in zip(residues[:-1], residues[1:]):
                first_atoms = {atom.name: atom for atom in first.atoms}
                second_atoms = {atom.name: atom for atom in second.atoms}
                # Partial or modified nucleotides need not contain both backbone endpoints.
                if "O3'" in first_atoms and "P" in second_atoms:
                    oxygen, phosphorus = first_atoms["O3'"], second_atoms["P"]
                    pair = tuple(sorted((oxygen.index, phosphorus.index)))
                    if pair not in pairs:
                        topology.add_bond(oxygen, phosphorus)
                        pairs.add(pair)

    existing_bonds = {tuple(sorted((atom1.index, atom2.index))) for atom1, atom2 in topology.bonds}
    bonds_to_add = {
        tuple(sorted((atom1, atom2))) for atom1, atom2 in structure.bond_index.T.tolist()
    }
    # MDTraj's peptide rule belongs to the following residue's template, so a
    # nonstandard protein residue with N may need its preceding C-N link added.
    for chain in topology.chains:
        previous_carbon_index: int | None = None
        for residue in chain.residues:
            atom_indices_by_name = {atom.name: atom.index for atom in residue.atoms}
            first_atom_index = next(iter(atom_indices_by_name.values()))
            if structure.mol_type[first_atom_index].item() != MolType.PROTEIN.value:
                previous_carbon_index = None
                continue
            if previous_carbon_index is not None and "N" in atom_indices_by_name:
                bonds_to_add.add(tuple(sorted((previous_carbon_index, atom_indices_by_name["N"]))))
            previous_carbon_index = (
                atom_indices_by_name.get("C") if "N" in atom_indices_by_name else None
            )
    for atom1, atom2 in sorted(bonds_to_add - existing_bonds):
        topology.add_bond(topology.atom(atom1), topology.atom(atom2))

    return topology


@torch.no_grad()
def inference_input_structure_from_template_npz(
    template_npz_uri: str, residue_label: str | None = None
) -> AllAtomStructure:
    """Create an `AllAtomStructure` from a pre-computed ligand template.npz file.

    This is the shared implementation used by ``inference_input_structure_from_ccd``
    and ``inference_input_structure_from_inchi``.
    It can also be called directly when the caller already knows the template URI.

    Args:
        template_npz_uri: Local path to the template.npz file.
        residue_label: Per-atom residue label. Defaults to the UNK_PROT PDB resname.
    """
    with open(template_npz_uri, "rb") as f:
        template_info = dict(np.load(f))

    template_coords: np.ndarray = template_info["template_coords"]
    num_atoms = template_coords.shape[0]
    if residue_label is None:
        residue_label = index_to_pdb_resname(Token.UNK_PROT.value)

    return AllAtomStructure(
        coords=torch.zeros(num_atoms, 3),
        mol_type=torch.full((num_atoms,), MolType.NONPOLYMER.value, dtype=torch.long),
        asym_id=torch.zeros(num_atoms, dtype=int),
        entity_id=torch.zeros(num_atoms, dtype=int),
        sym_id=torch.zeros(num_atoms, dtype=int),
        num_nodes=num_atoms,
        num_residues=1,  # Ligand is a single residue.
        num_tokens=num_atoms,
        element=torch.tensor(template_info["element"], dtype=torch.long),
        residue_type=torch.full((num_atoms,), Token.UNK_PROT.value, dtype=torch.long),
        residue_idx=torch.zeros((num_atoms,), dtype=torch.long),
        name_encoding=torch.tensor(template_info["name_encoding"], dtype=torch.long),
        charge=torch.tensor(template_info["charge"], dtype=torch.int8),
        atom_mask=torch.ones((num_atoms,), dtype=torch.bool),
        token_center_atom_index=torch.arange(num_atoms, dtype=torch.long),
        token_idx_at_atom=torch.arange(num_atoms, dtype=torch.long),
        sequence=PROT_TOKEN_TO_LETTER[Token.UNK_PROT],
        template_coords=torch.from_numpy(template_coords),
        bond_index=torch.from_numpy(template_info["bond_index"]),
        bond_type=torch.from_numpy(template_info["bond_type"]),
        distogram_index=torch.arange(num_atoms, dtype=int),
        residue_labels=[residue_label] * num_atoms,
    )


@torch.no_grad()
def inference_input_structure_from_inchi(inchi: str) -> AllAtomStructure:
    """Create an `AllAtomStructure` template from an InChI string.

    Uses ``LigandTemplateCache`` to load or generate template coordinates keyed
    by the InChI string. InChI and SDF inputs use separate cache entries and can
    produce different template coordinates for the same molecule.
    """
    spec = LigandTemplateCache.from_inchi(inchi)
    return inference_input_structure_from_template_npz(str(spec.npz_file))


@torch.no_grad()
def inference_input_structure_from_ccd(ccd_code: str) -> AllAtomStructure:
    """Create an `AllAtomStructure` template from a CCD code."""
    spec = LigandTemplateCache.from_ccd(ccd_code)
    return inference_input_structure_from_template_npz(
        str(spec.npz_file),
        residue_label=ccd_code,
    )


@torch.no_grad()
def inference_input_structure_from_branched_entity(
    ccd_codes: Collection[str],
    remove_leaving_atoms: Collection[bool] | None = None,
) -> AllAtomStructure:
    """Create an ``AllAtomStructure`` for a branched entity (e.g., a glycan chain).

    Each residue is loaded from its CCD template and concatenated into a single chain with
    intra-residue bonds only. **Inter-residue bonds (glycosidic bonds) are not added here** —
    they should be specified as ``Bond`` entries in ``System.bonds`` and will be wired by
    ``_add_extra_covalent_bonds`` post hoc, which is the same mechanism used for all
    user-defined cross-links.

    Args:
        ccd_codes: List of CCD component codes, one per residue (e.g.,
            ``["NAG", "NAG", "BMA", "MAN", "MAN"]``).
        remove_leaving_atoms: Per-residue leaving-atom removal control.
            - ``False``: no leaving-atom removal for this residue.
            - ``True``: unconditionally remove leaving atoms.
            Must have the same length as ``ccd_codes``. If ``None``, defaults to ``False``
            for every residue.

    Returns:
        ``AllAtomStructure`` with all residues assembled into a single chain.
        ``mol_type`` is set to ``MolType.NONPOLYMER`` and each atom is its own token.
        Only intra-residue (CCD template) bonds are included in ``bond_index``.
    """
    if len(ccd_codes) == 0:
        raise ValueError("ccd_codes must be non-empty.")

    rm_leaving_atoms: list[bool]
    if remove_leaving_atoms is None:
        rm_leaving_atoms = [False] * len(ccd_codes)
    else:
        rm_leaving_atoms = list(remove_leaving_atoms)

    if len(rm_leaving_atoms) != len(ccd_codes):
        raise ValueError(
            f"remove_leaving_atoms length ({len(rm_leaving_atoms)}) must match "
            f"ccd_codes length ({len(ccd_codes)})."
        )

    # Load per-residue templates, conditionally removing leaving atoms.
    residue_templates: list[AllAtomStructure] = []
    for ccd_code, rm_leaving in zip(ccd_codes, rm_leaving_atoms):
        cif_wrapper = _load_ccd_cifwrapper(ccd_code)
        residue_templates.append(
            template_from_cifwrapper(cif_wrapper, remove_leaving_atoms=rm_leaving)
        )

    # Accumulate atom-level data across residues.
    all_template_coords: list[torch.Tensor] = []
    all_name_encoding: list[torch.Tensor] = []
    all_element: list[torch.Tensor] = []
    all_charge: list[torch.Tensor] = []
    all_residue_idx: list[torch.Tensor] = []
    all_bond_index: list[torch.Tensor] = []
    all_bond_type: list[torch.Tensor] = []
    all_residue_labels: list[str] = []

    atom_offset = 0

    for res_i, (ccd_code, template) in enumerate(zip(ccd_codes, residue_templates)):
        num_atoms = template.num_nodes

        assert template.template_coords is not None
        all_template_coords.append(template.template_coords)
        all_name_encoding.append(template.name_encoding)
        all_element.append(template.element)
        all_charge.append(template.charge)
        all_residue_idx.append(torch.full((num_atoms,), res_i, dtype=torch.long))
        all_residue_labels.extend([ccd_code] * num_atoms)

        # Offset intra-residue bonds to global atom indices.
        if template.bond_index.numel() > 0:
            all_bond_index.append(template.bond_index + atom_offset)
            all_bond_type.append(template.bond_type)

        atom_offset += num_atoms

    total_atoms = atom_offset
    num_residues = len(ccd_codes)

    # Concatenate intra-residue bonds only.
    if all_bond_index:
        bond_index = torch.cat(all_bond_index, dim=1)
        bond_type_tensor = torch.cat(all_bond_type)
    else:
        bond_index = torch.zeros((2, 0), dtype=torch.long)
        bond_type_tensor = torch.zeros((0,), dtype=torch.long)

    # Each non-polymer atom corresponds to one token.
    mol_type = torch.full((total_atoms,), MolType.NONPOLYMER.value, dtype=torch.long)
    asym_id = torch.zeros((total_atoms,), dtype=torch.long)
    entity_id = torch.zeros((total_atoms,), dtype=torch.long)
    sym_id = torch.zeros((total_atoms,), dtype=torch.long)
    residue_type = torch.full((total_atoms,), Token.UNK_PROT.value, dtype=torch.long)

    # Build a single-character sequence placeholder per residue (downstream uses len + ':' split).
    sequence = PROT_TOKEN_TO_LETTER[Token.UNK_PROT] * num_residues

    return AllAtomStructure(
        coords=torch.zeros(total_atoms, 3),
        element=torch.cat(all_element),
        charge=torch.cat(all_charge),
        residue_idx=torch.cat(all_residue_idx),
        residue_type=residue_type,
        num_residues=num_residues,
        num_tokens=total_atoms,
        bond_index=bond_index,
        bond_type=bond_type_tensor,
        num_nodes=total_atoms,
        name_encoding=torch.cat(all_name_encoding),
        atom_mask=torch.ones(total_atoms, dtype=torch.bool),
        asym_id=asym_id,
        entity_id=entity_id,
        sym_id=sym_id,
        mol_type=mol_type,
        sequence=sequence,
        template_coords=torch.cat(all_template_coords),
        token_center_atom_index=torch.arange(total_atoms, dtype=torch.long),
        token_idx_at_atom=torch.arange(total_atoms, dtype=torch.long),
        distogram_index=torch.arange(total_atoms, dtype=torch.long),
        residue_labels=all_residue_labels,
    )


def extract_rdkit_mol_from_structure(
    structure: AllAtomStructure, mask: torch.Tensor | None = None
) -> Chem.Mol:
    """Extract an RDKit molecule from an `AllAtomStructure`.

    Args:
        structure: Input AllAtomStructure.
        mask: Optional mask to filter atoms. Shape: [num_atoms].

    Returns:
        RDKit molecule.
    """
    if mask is None:
        mask = torch.ones(structure.num_nodes, dtype=torch.bool)

    nodes = torch.where(mask)[0]
    # maps from atom index to atomic number
    elements = dict(zip(nodes.tolist(), structure.element[mask].tolist()))

    # filter bond indices to only include edges between the selected atoms.
    edge_mask = torch.isin(structure.bond_index, nodes)
    edge_mask = torch.logical_and(edge_mask[0], edge_mask[1])
    assert structure.bond_type is not None
    return _rdkit_mol_from_elements_and_bonds(
        elements=elements,
        bond_index=structure.bond_index[:, edge_mask],
        bond_type=structure.bond_type[edge_mask],
    )


def _rdkit_mol_from_elements_and_bonds(
    elements: dict[int, int],
    bond_index: torch.Tensor,
    bond_type: torch.Tensor,
) -> Chem.Mol:
    """Create an RDKit molecule from element and bond information.

    Args:
        elements: Dictionary mapping atom indices to atomic numbers.
        bond_index: Tensor of shape [2, num_bonds] containing indices of bonded atoms
        bond_type: Tensor of shape [num_bonds] containing bond types.

    Returns:
        RDKit molecule.
    """
    mol = Chem.RWMol()
    atom_map = {}
    for i, elem in elements.items():
        atom = Chem.Atom(elem)
        atom_idx = mol.AddAtom(atom)
        atom_map[i] = atom_idx

    for (i, j), bond_type in zip(bond_index.t().tolist(), bond_type.tolist()):
        rdkit_bond_type = get_rdkit_bondtype_from_bondtype(BondType(bond_type))
        mol.AddBond(atom_map[i], atom_map[j], order=rdkit_bond_type)

    return mol.GetMol()


@torch.no_grad()
def inference_input_structure_from_rna_sequence(
    sequence: str,
    template_cache: dict[str, AllAtomStructure],
    modifications: Sequence[PolymerModification] = (),
) -> AllAtomStructure:
    """
    Create an `AllAtomStructure` object from a RNA sequence.

    Args:
        sequence: Original one-letter RNA sequence. The returned structure retains it as sequence
            metadata.
        template_cache: Cache of template structures, see `bioemu2.structure.templates` for details.
        modifications: Explicit `(res_id, kind, value)` entries for modified residues. `res_id` is
            1-based and `kind` is `"ccd"` or `"inchi"`. A modification replaces the residue
            structure at its position; the sequence may keep its canonical letter or use `N`.

    Returns:
        `AllAtomStructure` with atom and residue information.
    """
    if modifications:
        return _inference_input_structure_from_polymer_sequence_with_mods(
            sequence=sequence,
            modifications=modifications,
            template_cache=template_cache,
            polymer_type=MolType.RNA,
        )
    assert all([s in RNA_LETTER_TO_TOKEN for s in sequence]), "Not a valid RNA sequence"
    return _inference_input_structure_from_nucleotide_sequence(
        sequence=sequence, template_cache=template_cache, nucleotide_type=MolType.RNA
    )


@torch.no_grad()
def inference_input_structure_from_dna_sequence(
    sequence: str,
    template_cache: dict[str, AllAtomStructure],
    modifications: Sequence[PolymerModification] = (),
) -> AllAtomStructure:
    """
    Create an `AllAtomStructure` object from a DNA sequence.

    Args:
        sequence: Original one-letter DNA sequence. The returned structure retains it as sequence
            metadata.
        template_cache: Cache of template structures, see `bioemu2.structure.templates` for details.
        modifications: Explicit `(res_id, kind, value)` entries for modified residues. `res_id` is
            1-based and `kind` is `"ccd"` or `"inchi"`. A modification replaces the residue
            structure at its position; the sequence may keep its canonical letter or use `N`.
    Returns:
        `AllAtomStructure` with atom and residue information.
    """
    if modifications:
        return _inference_input_structure_from_polymer_sequence_with_mods(
            sequence=sequence,
            modifications=modifications,
            template_cache=template_cache,
            polymer_type=MolType.DNA,
        )
    assert all([s in DNA_LETTER_TO_TOKEN for s in sequence]), "Not a valid DNA sequence"
    return _inference_input_structure_from_nucleotide_sequence(
        sequence=sequence, template_cache=template_cache, nucleotide_type=MolType.DNA
    )


@torch.no_grad()
def _inference_input_structure_from_nucleotide_sequence(
    sequence: str,
    template_cache: dict[str, AllAtomStructure],
    nucleotide_type: Literal[MolType.RNA, MolType.DNA],
) -> AllAtomStructure:
    """
    Create an `AllAtomStructure` object from a nucleotide sequence using CCD templates.

    Args:
        sequence: One letter nucleotide sequence.
        template_cache: Cache of template structures, see `bioemu2.structure.templates` for details.
        nucleotide_type: `MolType` specifying whether the sequence is RNA or DNA.

    Returns:
        `AllAtomStructure` with atom and residue information. Structure is taken from the templates
        if available.
    """
    assert nucleotide_type in [
        MolType.DNA,
        MolType.RNA,
    ], "Unsupported type for nucleotide sequences."

    coords_tmp: list[torch.Tensor] = []
    residue_type_tmp: list[int] = []
    residue_idx_tmp: list[int] = []
    name_encoding_tmp: list[torch.Tensor] = []
    element_tmp: list[torch.Tensor] = []
    charge_tmp: list[torch.Tensor] = []
    distogram_mask_tmp: list[torch.Tensor] = []

    residue_atom_offset: int = 0
    for current_residue_idx, template in enumerate(
        _templates_from_nucleotide_sequence(
            sequence, template_cache, nucleotide_type=nucleotide_type
        )
    ):
        coords_tmp.append(template.template_coords)
        name_encoding_tmp.append(template.name_encoding)
        element_tmp.append(template.element)
        charge_tmp.append(template.charge)

        residue_idx_tmp.append(template.residue_idx + current_residue_idx)
        residue_type_tmp.append(template.residue_type)

        # Collect distogram mask here
        residue_token = Token(template.residue_type[0].item())
        distogram_atom_name = token_to_disto_atom_name(residue_token)
        distogram_mask_tmp.append(template.name_encoding_is_atom(distogram_atom_name))

        residue_atom_offset += template.element.shape[0]

    template_coords = torch.cat(coords_tmp)
    name_encoding = torch.cat(name_encoding_tmp)
    element = torch.cat(element_tmp)
    charge = torch.cat(charge_tmp)
    residue_idx = torch.cat(residue_idx_tmp)
    residue_type = torch.cat(residue_type_tmp)
    num_residues = len(sequence)

    # Set up dummy features for chains.
    asym_id = torch.zeros((num_residues,), dtype=torch.long)
    sym_id = torch.zeros_like(asym_id)
    entity_id = torch.zeros_like(asym_id)
    mol_type = torch.full_like(residue_idx, nucleotide_type.value)

    # Token for DNA/RNA is always C1' (sugar atom the nucleobase is bonded to).
    token_center_atom_index = torch.where(
        get_atom_name_mask_for_name_encoding(name_encoding, "C1'")
    )[0]

    # Finalize distogram index on partial masks.
    distogram_index = torch.where(torch.cat(distogram_mask_tmp))[0]

    residue_labels = [index_to_pdb_resname(res) for res in residue_type.cpu().tolist()]

    return AllAtomStructure(
        coords=torch.zeros_like(template_coords),
        element=element,
        charge=charge,
        residue_idx=residue_idx,
        residue_type=residue_type,
        num_residues=num_residues,
        num_tokens=num_residues,
        bond_index=torch.zeros((2, 0), dtype=torch.long),
        bond_type=torch.zeros((0,), dtype=torch.long),
        num_nodes=element.shape[0],
        name_encoding=name_encoding,
        atom_mask=torch.ones_like(element).bool(),
        asym_id=asym_id,
        entity_id=entity_id,
        sym_id=sym_id,
        mol_type=mol_type,
        sequence=sequence,
        template_coords=template_coords,
        token_center_atom_index=token_center_atom_index,
        token_idx_at_atom=residue_idx,
        distogram_index=distogram_index,
        residue_labels=residue_labels,
    )


def _templates_from_nucleotide_sequence(
    sequence: str,
    template_cache: dict[str, AllAtomStructure],
    nucleotide_type: Literal[MolType.RNA, MolType.DNA],
) -> Generator[AllAtomStructure, None, None]:
    """
    Create a list of `AllAtomStructure` templates from a nucleotide sequence using CCD templates.

    Args:
        sequence: Nucleotide sequence in one-letter code.
        template_cache: Cache of template structures, see `bioemu2.structure.templates` for details.
        nucleotide_type: `MolType` specifying whether the sequence is RNA or DNA.

    Returns:
        A generator yielding `AllAtomStructure` objects for each residue in the sequence.
    """
    is_dna = nucleotide_type == MolType.DNA

    for residue_1 in sequence:
        ccd_id = f"D{residue_1}" if is_dna else residue_1
        residue_token = Token.from_single_letter(residue_1, moltype=nucleotide_type)
        template = template_cache[ccd_id]

        # OP3 capping oxygens of the phosphate group are excluded from model features.
        op3_mask = ~template.name_encoding_is_atom("OP3")
        num_atoms = int(op3_mask.sum())
        assert template.template_coords is not None
        yield AllAtomStructure(
            template_coords=template.template_coords[op3_mask],
            name_encoding=template.name_encoding[op3_mask],
            element=template.element[op3_mask],
            charge=template.charge[op3_mask],
            residue_idx=torch.zeros((num_atoms,), dtype=torch.long),
            residue_type=torch.full((num_atoms,), residue_token.value, dtype=torch.long),
        )


def _remove_capping_atoms(modified_residue: AllAtomStructure) -> AllAtomStructure:
    """
    Auxiliary function for removing capping atoms from residues. The removed atoms are OXT and OP3
    which get removed when forming a polymer bond in a peptide and DNA/RNA, respectively. They might
    still be present in the CCD entries of modified amino acids etc and should be removed in that
    case.

    Args:
        modified_residue: All atom structure which should be cleaned of capping atoms.

    Returns:
        All atom structure with capping atoms (and associated bonds) removed.
    """
    # This aggregates the masks with capping atoms in the columns and the atom masks in the rows
    # and then checks across the capping atom dimension to see which atoms are unmasked in both.
    mask_connecting_atoms = torch.all(
        torch.stack(
            [~modified_residue.name_encoding_is_atom(cap) for cap in CAPPING_ATOMS],
            dim=1,
        ),
        dim=1,
    )
    modified_residue = extract_substructure(
        modified_residue, token_mask=None, atom_mask=mask_connecting_atoms
    )
    return modified_residue


@torch.no_grad()
def _inference_input_structure_from_polymer_sequence_with_mods(
    sequence: str,
    template_cache: dict[str, AllAtomStructure],
    polymer_type: MolType,
    modifications: Sequence[PolymerModification] = (),
) -> AllAtomStructure:
    """
    Generates templates for a polymer sequence containing covalently-bound ligands (e.g., PTMs)

    Args:
        sequence: Original one-letter polymer sequence. An explicit modification replaces the
            residue structure at its position, regardless of the sequence letter.
        template_cache: Cache of template structures, see `bioemu2.structure.templates` for details.
        polymer_type: The type of polymer supplied in `sequence`.
        modifications: Explicit `(res_id, kind, value)` modifications.

    Returns:
        `AllAtomStructure` template with the desired covalently bound ligands placed in the sequence.
    """

    if polymer_type == MolType.PROTEIN:
        polymer_template_gen_fun = inference_input_structure_from_protein_sequence
    elif polymer_type == MolType.DNA:
        polymer_template_gen_fun = inference_input_structure_from_dna_sequence
    elif polymer_type == MolType.RNA:
        polymer_template_gen_fun = inference_input_structure_from_rna_sequence
    else:
        raise ValueError("Not recognised polymer type")

    unk_token_char = MOLTYPE_TO_UNKTOKEN[polymer_type].singleletter
    mods_by_position: dict[int, tuple[ModificationKind, str]] = {}
    for position, kind, value in modifications:
        if not 1 <= position <= len(sequence):
            raise ValueError(
                f"Modification position {position} is out of range for sequence {sequence!r}."
            )
        if position in mods_by_position:
            raise ValueError(f"Repeated modification position {position}.")
        if kind not in {"ccd", "inchi"}:
            raise ValueError(f"Unsupported modification kind {kind!r}; expected 'ccd' or 'inchi'.")
        mods_by_position[position] = (kind, value)

    # Build inference structures in the right order
    builder = PolymerStructureBuilder(polymer_type)

    current_polymer_seq = ""
    for position, s in enumerate(sequence, start=1):
        kind_value = mods_by_position.get(position)
        if kind_value is not None or s == unk_token_char:
            if current_polymer_seq != "":
                builder.add_residues(polymer_template_gen_fun(current_polymer_seq, template_cache))
                current_polymer_seq = ""
            if kind_value is None:
                kind, value = "ccd", MOLTYPE_TO_UNKNOWN_CCD[polymer_type]
            else:
                kind, value = kind_value
            if kind == "ccd" and is_missing_residue_identity(polymer_type, value):
                missing_residue = polymer_template_gen_fun(unk_token_char, template_cache)
                missing_residue = missing_residue.replace(
                    residue_labels=[value] * len(missing_residue.name_encoding),
                )
                builder.add_residues(missing_residue)
            else:
                if kind == "inchi":
                    modified_residue = inference_input_structure_from_inchi(value)
                else:
                    modified_residue = inference_input_structure_from_ccd(value)
                modified_residue = _remove_capping_atoms(modified_residue)
                builder.add_residues(modified_residue, is_modified_residue=True)
        else:
            current_polymer_seq += s

    # Last polymer sequence
    if current_polymer_seq != "":
        builder.add_residues(polymer_template_gen_fun(current_polymer_seq, template_cache))
    return builder.build(sequence=sequence)
