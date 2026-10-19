from collections import defaultdict
from collections.abc import Callable

import torch

from bioemu2.structure import AllAtomStructure, decode_atom_name


def repr_by_sequence_and_name_encoding(structure: AllAtomStructure) -> str:
    """Generate a hash string for a structure based on its sequence and name encoding.

    Args:
        structure: The AllAtomStructure to hash.

    Returns:
        A hash string representing the structure.
    """
    name_encoding_str = ":".join(
        [decode_atom_name(name_enc.tolist()) for name_enc in structure.name_encoding]
    )
    return f"(sequence={structure.sequence}, name_encoding={name_encoding_str})"


class StructureBuilder:
    """Assembly of structures from per-residue templates."""

    def __init__(self, system_id: str | None = None):
        self.system_id = system_id
        self.coords: list[torch.Tensor] = []
        self.element: list[torch.Tensor] = []
        self.residue_idx: list[torch.Tensor] = []
        self.residue_type: list[torch.Tensor] = []
        self.num_tokens = 0
        self.num_residues = 0
        self.num_atoms = 0
        self.template_coords: list[torch.Tensor] = []
        self.bond_index: list[torch.Tensor] = []
        self.bond_type: list[torch.Tensor] = []
        self.name_encoding: list[torch.Tensor] = []
        self.charge: list[torch.Tensor] = []
        self.atom_mask: list[torch.Tensor] = []
        self.asym_ids: list[torch.Tensor] = []
        self.entity_ids: list[torch.Tensor] = []
        self.sym_id: list[torch.Tensor] = []
        self.mol_type: list[torch.Tensor] = []
        self.sequence: list[torch.Tensor] = []
        self.token_center_atom_index: list[torch.Tensor] = []
        self.token_idx_at_atom: list[torch.Tensor] = []
        self.distogram_index: list[torch.Tensor] = []
        self.entity_id_map: dict[str, int] = {}
        self.sym_count: dict[int, int] = defaultdict(int)
        self.residue_labels: list[str] = []

    @property
    def num_nodes(self) -> int:
        return self.num_atoms

    def entity_id_for_repr(self, structure_repr: str) -> int:
        """Return the numerical entity ID for a given `structure_repr` string.
        If the string has not been seen before, the next available numerical ID is assigned.
        """
        return self.entity_id_map.setdefault(structure_repr, len(self.entity_id_map))

    def add_chain(
        self,
        structure: AllAtomStructure,
        entity_id: int | None = None,
        structure_repr_func: Callable[[AllAtomStructure], str] = repr_by_sequence_and_name_encoding,
    ) -> None:
        asym_id = len(self.sequence)
        if entity_id is None:
            entity_id = self.entity_id_for_repr(structure_repr_func(structure))
        self.coords.append(structure.coords)
        self.element.append(structure.element)
        # Offset residue index to make global across asym IDs to simplify indexing + bookkeeping.
        self.residue_idx.append(structure.residue_idx + self.num_residues)
        self.residue_type.append(structure.residue_type)
        self.template_coords.append(structure.template_coords)
        self.bond_index.append(structure.bond_index + self.num_atoms)
        self.bond_type.append(structure.bond_type)
        self.name_encoding.append(structure.name_encoding)
        self.charge.append(structure.charge)
        self.atom_mask.append(structure.atom_mask)
        self.asym_ids.append(torch.full((structure.num_tokens,), asym_id, dtype=torch.long))
        self.entity_ids.append(torch.full((structure.num_tokens,), entity_id, dtype=torch.long))
        self.sym_id.append(
            torch.full((structure.num_tokens,), self.sym_count[entity_id], dtype=torch.long)
        )
        self.mol_type.append(structure.mol_type)
        self.sequence.append(structure.sequence)
        self.token_center_atom_index.append(structure.token_center_atom_index + self.num_atoms)
        self.token_idx_at_atom.append(structure.token_idx_at_atom + self.num_tokens)
        self.distogram_index.append(structure.distogram_index + self.num_atoms)
        self.residue_labels += structure.residue_labels

        self.num_tokens += structure.num_tokens
        self.num_residues += structure.num_residues
        self.num_atoms += structure.num_nodes
        self.sym_count[entity_id] += 1

    def build(self) -> AllAtomStructure:
        return AllAtomStructure(
            coords=torch.cat(self.coords, dim=0),
            element=torch.cat(self.element, dim=0),
            residue_idx=torch.cat(self.residue_idx, dim=0),
            residue_type=torch.cat(self.residue_type, dim=0),
            num_residues=self.num_residues,
            num_tokens=self.num_tokens,
            num_nodes=self.num_atoms,
            template_coords=torch.cat(self.template_coords, dim=0),
            bond_index=torch.cat(self.bond_index, dim=1),
            bond_type=torch.cat(self.bond_type, dim=0),
            name_encoding=torch.cat(self.name_encoding, dim=0),
            charge=torch.cat(self.charge, dim=0),
            atom_mask=torch.cat(self.atom_mask, dim=0),
            asym_id=torch.cat(self.asym_ids, dim=0),
            entity_id=torch.cat(self.entity_ids, dim=0),
            sym_id=torch.cat(self.sym_id, dim=0),
            mol_type=torch.cat(self.mol_type, dim=0),
            sequence=":".join(self.sequence),
            token_center_atom_index=torch.cat(self.token_center_atom_index, dim=0),
            token_idx_at_atom=torch.cat(self.token_idx_at_atom, dim=0),
            distogram_index=torch.cat(self.distogram_index, dim=0),
            system_id=self.system_id,
            residue_labels=self.residue_labels,
        )
