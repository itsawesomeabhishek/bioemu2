"""Recover a fully-qualified :class:`Atom` from a Boltz bond specification.

Boltz references bonded atoms by ``(chain_id, res_id, atom_name)``.
:class:`bioemu2.data_interface.Atom` additionally requires ``res_name``
(3-letter code or modification CCD code) for validation against the underlying
entity. This module looks up ``res_name`` through the entity's resolved residue
sequence.
"""

from collections.abc import Iterable

from bioemu2.data_interface.structure_hierarchy import (
    DNA,
    RNA,
    Atom,
    Ligand,
    Polymer,
    Protein,
)


def recover_atom(
    *,
    asym_id: str,
    res_id: int,
    atom_name: str,
    entities_by_asym_id: dict[str, Polymer | Ligand],
) -> Atom:
    """Build an :class:`Atom` from a Boltz bond reference.

    Args:
        asym_id: Chain id (e.g. ``"A"``).
        res_id: 1-based residue index within the chain. For a Ligand
            entity this is typically ``1`` (single-residue).
        atom_name: PDB-style atom name (e.g. ``"CA"``, ``"SG"``).
        entities_by_asym_id: Mapping of chain id -> owning entity. The
            caller (a decoder) builds this from the partial
            :class:`System` it is constructing.

    Returns:
        ``Atom(name=atom_name, res_name=<derived>, res_id=res_id, asym_id=asym_id)``.

    Raises:
        KeyError: ``asym_id`` not in *entities_by_asym_id*.
        ValueError: ``res_id`` out of range, or unknown residue letter.
    """
    if asym_id not in entities_by_asym_id:
        raise KeyError(f"Unknown asym_id {asym_id!r} in bond reference")
    entity = entities_by_asym_id[asym_id]

    if isinstance(entity, Ligand):
        # Ligands have a single residue whose name is the ccd_code (or PDB
        # placeholder for SMILES / InChI ligands).
        if entity.ccd_code is not None:
            res_name = entity.ccd_code
        else:
            res_name = "UNL"  # PDB convention for unknown ligand
        return Atom(name=atom_name, res_name=res_name, res_id=res_id, asym_id=asym_id)

    if not isinstance(entity, Protein | DNA | RNA):
        raise TypeError(f"Cannot recover bond atom for entity {type(entity).__name__}")

    if not 1 <= res_id <= len(entity.sequence):
        raise ValueError(
            f"res_id {res_id} out of range for chain {asym_id!r} "
            f"(sequence length {len(entity.sequence)})"
        )

    return Atom(
        name=atom_name,
        res_name=entity.residue_sequence[res_id - 1],
        res_id=res_id,
        asym_id=asym_id,
    )


def build_asym_id_index(entities: Iterable[Polymer | Ligand]) -> dict[str, Polymer | Ligand]:
    """Map every asym_id in *entities* back to its owning entity."""
    out: dict[str, Polymer | Ligand] = {}
    for entity in entities:
        for asym_id in entity.asym_ids:
            out[asym_id] = entity
    return out
