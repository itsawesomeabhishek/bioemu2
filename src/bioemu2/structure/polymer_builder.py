import torch

from bioemu2.data_interface.const import MolType
from bioemu2.structure import AllAtomStructure


class PolymerStructureBuilder:
    """Builder class for constructing a single polymer chain (protein, DNA, RNA) structure from multiple AllAtomStructure objects.

    The class supports both standard and modified residues.
    """

    def __init__(self, mol_type: MolType, system_id: str | None = None):
        """Initialize a PolymerStructureBuilder.

        Args:
            mol_type: The molecule type of the polymer chain (e.g., protein, DNA, RNA).
        """
        self.mol_type = mol_type
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
        self.polymer_mol_type: list[torch.Tensor] = []
        self.sequence: list[torch.Tensor] = []
        self.token_center_atom_index: list[torch.Tensor] = []
        self.token_idx_at_atom: list[torch.Tensor] = []
        self.distogram_index: list[torch.Tensor] = []
        self.residue_labels: list[str] = []

    @property
    def num_nodes(self) -> int:
        return self.num_atoms

    def add_residues(
        self,
        structure: AllAtomStructure,
        is_modified_residue: bool = False,
    ) -> None:
        """Add standard or modified residues.

        Args:
            structure: The AllAtomStructure to add.
            is_modified_residue: Whether the residue is a modified residue (default: False).
        """
        assert (
            self.system_id is None
            or structure.system_id is None
            or self.system_id == structure.system_id
        ), "Mismatched system_id between PolymerStructureBuilder and the structure being added."
        if not is_modified_residue:
            # Collect polymer mol types if this is not a modified residue
            self.polymer_mol_type.append(structure.mol_type)

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
        self.sequence.append(structure.sequence)
        self.token_center_atom_index.append(structure.token_center_atom_index + self.num_atoms)
        self.token_idx_at_atom.append(structure.token_idx_at_atom + self.num_tokens)
        self.distogram_index.append(structure.distogram_index + self.num_atoms)
        self.num_tokens += structure.num_tokens
        self.num_residues += structure.num_residues
        self.num_atoms += structure.num_nodes
        self.residue_labels += structure.residue_labels

    def build(self, sequence: str | None = None) -> AllAtomStructure:
        """Build the AllAtomStructure for the polymer chain.

        Args:
            sequence: Optional source sequence to retain as metadata when a residue's
                chemical structure differs from its sequence letter.

        Raises:
            ValueError: If the polymer has no residues.
            AssertionError: If multiple or mismatched unmodified molecule types are found.
        """
        if self.num_residues == 0 or self.num_atoms == 0:
            raise ValueError("Cannot build a polymer without residues and atoms.")
        if self.polymer_mol_type:
            mol_types = torch.unique(torch.cat(self.polymer_mol_type, dim=0))
            assert len(mol_types) == 1, (
                "Unexpected multiple parent molecule types. If the chain contains modified residues, please set `is_modified_residue=True` when adding modified residues."
            )
            assert mol_types[0].item() == self.mol_type.value, "Molecule type mismatch"
        output_sequence = "".join(self.sequence) if sequence is None else sequence
        if len(output_sequence) != self.num_residues:
            raise ValueError(
                f"Sequence length {len(output_sequence)} does not match the built polymer "
                f"length {self.num_residues}."
            )

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
            asym_id=torch.zeros(self.num_tokens, dtype=torch.long),
            entity_id=torch.zeros(self.num_tokens, dtype=torch.long),
            sym_id=torch.zeros(self.num_tokens, dtype=torch.long),
            mol_type=torch.full((self.num_atoms,), self.mol_type.value, dtype=torch.long),
            sequence=output_sequence,
            token_center_atom_index=torch.cat(self.token_center_atom_index, dim=0),
            token_idx_at_atom=torch.cat(self.token_idx_at_atom, dim=0),
            distogram_index=torch.cat(self.distogram_index, dim=0),
            system_id=self.system_id,
            residue_labels=self.residue_labels,
        )
