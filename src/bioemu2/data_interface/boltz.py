"""Boltz YAML decoder for :class:`bioemu2.data_interface.System`."""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from bioemu2.data_interface.bond_reference import build_asym_id_index, recover_atom
from bioemu2.data_interface.structure_hierarchy import (
    DNA,
    RNA,
    Bond,
    Contact,
    ContactToken,
    Ligand,
    ModifiedResidue,
    Polymer,
    Protein,
)
from bioemu2.data_interface.system import System

logger = logging.getLogger(__name__)


def _warn_extra_fields(raw: Mapping[str, Any], supported: set[str], context: str) -> None:
    if not isinstance(raw, Mapping):
        raise ValueError(f"Boltz {context} must be a mapping.")
    for field in raw.keys() - supported:
        logger.warning("Dropping Boltz %s field %r.", context, field)


def _asym_ids(block: Mapping[str, Any]) -> str | list[str]:
    raw_id = block["id"]
    if isinstance(raw_id, list):
        return [str(asym_id) for asym_id in raw_id]
    return str(raw_id)


def _mods_from_boltz(raw_mods: list[Mapping[str, Any]]) -> list[ModifiedResidue]:
    for mod in raw_mods:
        _warn_extra_fields(mod, {"position", "ccd"}, "modification")
    return [
        ModifiedResidue(res_id=int(mod["position"]), ccd_code=str(mod["ccd"]))
        for mod in sorted(raw_mods, key=lambda mod: int(mod["position"]))
    ]


def _decode_protein(block: Mapping[str, Any]) -> Protein:
    sequence = str(block["sequence"])
    mods = _mods_from_boltz(list(block.get("modifications", [])))
    return Protein(asym_ids=_asym_ids(block), sequence=sequence, mods=mods)


def _decode_nucleic(block: Mapping[str, Any], entity_type: type[RNA] | type[DNA]) -> RNA | DNA:
    sequence = str(block["sequence"])
    mods = _mods_from_boltz(list(block.get("modifications", [])))
    return entity_type(asym_ids=_asym_ids(block), sequence=sequence, mods=mods)


def _decode_ligand(block: Mapping[str, Any]) -> Ligand:
    identifiers = [key for key in ("smiles", "ccd") if block.get(key) is not None]
    if len(identifiers) != 1:
        raise ValueError("Boltz ligand must define exactly one of smiles or ccd.")
    if identifiers[0] == "smiles":
        return Ligand(asym_ids=_asym_ids(block), smiles=str(block["smiles"]))

    return Ligand(asym_ids=_asym_ids(block), ccd_code=str(block["ccd"]))


def _sequence_block(entry: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    if not isinstance(entry, Mapping):
        raise ValueError("Boltz sequence entry must be a mapping.")
    if len(entry) != 1:
        raise ValueError(
            f"Boltz sequence entry must contain exactly one entity type, got {sorted(entry)}."
        )
    kind, block = next(iter(entry.items()))
    if not isinstance(block, Mapping):
        raise ValueError(f"Boltz {kind} sequence block must be a mapping.")
    return str(kind).lower(), block


def _decode_entity(entry: Mapping[str, Any]) -> Protein | RNA | DNA | Ligand:
    kind, block = _sequence_block(entry)
    if "msa" in block:
        logger.warning(
            "Boltz MSA for chain %r (%s) is IGNORED: conversion does not preserve it, "
            "and BioEmu2 sampling will not use it. Supply an A3M explicitly with "
            "get_msa(system, source=...) or retrieve an MSA.",
            block.get("id"),
            block["msa"],
        )
    if block.get("cyclic") is True:
        raise NotImplementedError("Cyclic Boltz polymers are not supported.")
    supported = (
        {"id", "smiles", "ccd"}
        if kind == "ligand"
        else {"id", "sequence", "modifications", "cyclic", "msa"}
    )
    _warn_extra_fields(block, supported, f"{kind} {block.get('id')!r}")
    if kind == "protein":
        return _decode_protein(block)
    if kind == "rna":
        return _decode_nucleic(block, RNA)
    if kind == "dna":
        return _decode_nucleic(block, DNA)
    if kind == "ligand":
        return _decode_ligand(block)
    raise ValueError(f"Unsupported Boltz entity type: {kind!r}.")


def _decode_bond(
    raw_bond: Mapping[str, Any], entities_by_asym_id: dict[str, Polymer | Ligand]
) -> Bond:
    _warn_extra_fields(raw_bond, {"atom1", "atom2"}, "bond")
    chain1, residue1, name1 = raw_bond["atom1"]
    chain2, residue2, name2 = raw_bond["atom2"]
    for chain in (chain1, chain2):
        entity = entities_by_asym_id.get(str(chain))
        if isinstance(entity, Ligand) and entity.smiles is not None:
            raise NotImplementedError(
                f"Boltz bonds to atoms in SMILES ligand chain {chain!r} are not supported."
            )
    return Bond(
        atom1=recover_atom(
            asym_id=str(chain1),
            res_id=int(residue1),
            atom_name=str(name1),
            entities_by_asym_id=entities_by_asym_id,
        ),
        atom2=recover_atom(
            asym_id=str(chain2),
            res_id=int(residue2),
            atom_name=str(name2),
            entities_by_asym_id=entities_by_asym_id,
        ),
    )


def _decode_contact_token(
    raw_token: list[Any], entities_by_asym_id: dict[str, Polymer | Ligand]
) -> ContactToken:
    if not isinstance(raw_token, list) or len(raw_token) != 2:
        raise ValueError("Boltz contact tokens must be [chain_id, residue_index_or_atom_name].")
    chain_id, index_or_name = raw_token
    asym_id = str(chain_id)
    try:
        entity = entities_by_asym_id[asym_id]
    except KeyError as error:
        raise ValueError(f"Unknown asym_id {asym_id!r} in contact reference") from error
    if isinstance(entity, Ligand):
        if not isinstance(index_or_name, str) or not index_or_name:
            raise ValueError("Boltz ligand contact tokens must specify an atom name.")
        if entity.smiles is not None:
            raise NotImplementedError(
                f"Boltz contacts to atoms in SMILES ligand chain {asym_id!r} are not supported."
            )
        return ContactToken(asym_id=asym_id, atom_name=index_or_name)
    if type(index_or_name) is not int:
        raise ValueError("Boltz polymer contact tokens must specify a 1-based residue index.")
    return ContactToken(asym_id=asym_id, res_id=index_or_name)


def _decode_contact(
    raw_contact: Mapping[str, Any], entities_by_asym_id: dict[str, Polymer | Ligand]
) -> Contact:
    if not isinstance(raw_contact, Mapping):
        raise ValueError("Boltz contact must be a mapping.")
    unsupported = raw_contact.keys() - {"token1", "token2", "max_distance", "force"}
    if unsupported:
        raise ValueError(f"Unsupported Boltz contact fields: {sorted(unsupported)}")
    if "force" in raw_contact:
        logger.warning(
            "Ignoring Boltz contact force setting %r; it does not configure BioEmu2 "
            "contact conditioning or steering.",
            raw_contact["force"],
        )
    return Contact(
        token1=_decode_contact_token(raw_contact["token1"], entities_by_asym_id),
        token2=_decode_contact_token(raw_contact["token2"], entities_by_asym_id),
        max_distance=raw_contact.get("max_distance", 6.0),
    )


def _warn_lossy_fields(raw: Mapping[str, Any]) -> None:
    _warn_extra_fields(
        raw, {"version", "sequences", "constraints", "templates", "properties"}, "input"
    )
    for template in raw.get("templates", []) or []:
        logger.warning("Dropping Boltz template: %s", template)

    for prop in raw.get("properties", []) or []:
        _warn_extra_fields(prop, {"affinity"}, "property")
        if "affinity" in prop:
            logger.warning("Dropping Boltz affinity property: %s", prop["affinity"])


def system_from_boltz(raw: Mapping[str, Any]) -> tuple[System, list[Contact]]:
    """Convert a parsed Boltz YAML mapping into a molecular system and contacts.

    Args:
        raw: Parsed Boltz input mapping.

    Returns:
        A validated System preserving entities and bonds, and a list of contacts
        for model conditioning (empty when absent). Use ``convert_boltz`` to write both to YAML.

    Pocket constraints, templates, properties, MSA settings, and other
    unrecognized fields are discarded with logging warnings. Unsupported
    contact fields, cyclic polymers, multi-CCD ligands, and conflicting
    ligand identifiers raise errors.
    See ``docs/boltz.md`` for supported fields and conversion limitations.
    """
    if not isinstance(raw, Mapping):
        raise ValueError("Boltz input must be a YAML mapping.")

    version = raw.get("version")
    if type(version) is not int or version != 1:
        logger.warning(
            "Boltz version %r is not version 1; attempting conversion anyway.",
            version,
        )

    raw_sequences = raw.get("sequences")
    if not isinstance(raw_sequences, list) or not raw_sequences:
        raise ValueError("Boltz input must contain a non-empty sequences list.")

    entities = [_decode_entity(entry) for entry in raw_sequences]
    entities_by_asym_id = build_asym_id_index(entities)

    bonds: list[Bond] = []
    contacts: list[Contact] = []
    for constraint in raw.get("constraints", []) or []:
        _warn_extra_fields(constraint, {"bond", "pocket", "contact"}, "constraint")
        if "bond" in constraint:
            bonds.append(_decode_bond(constraint["bond"], entities_by_asym_id))
        if "pocket" in constraint:
            logger.warning("Dropping Boltz pocket constraint: %s", constraint["pocket"])
        if "contact" in constraint:
            contacts.append(_decode_contact(constraint["contact"], entities_by_asym_id))

    _warn_lossy_fields(raw)
    system = System(entities=entities, bonds=bonds)
    system.validate_contacts(contacts)
    return system, contacts


def convert_boltz(source: Path | str, destination: Path | str) -> None:
    """Convert a Boltz YAML file to a BioEmu2 system YAML file.

    Args:
        source: Input YAML file path.
        destination: Output YAML file path. Parent directories are created;
            existing files are not overwritten.

    Ignored fields and untested schema versions produce logging warnings.
    Unsupported contact fields raise errors instead of discarding contacts.
    See ``docs/boltz.md`` for supported fields and conversion limitations.
    """
    output = Path(destination)
    if output.exists():
        raise FileExistsError(f"Output file already exists: {output}")
    system, contacts = system_from_boltz(yaml.safe_load(Path(source).read_text()))
    output.parent.mkdir(parents=True, exist_ok=True)
    data = system.model_dump(exclude_none=True)
    if contacts:
        data["contacts"] = [contact.model_dump(exclude_none=True) for contact in contacts]
    output.write_text(yaml.safe_dump(data, sort_keys=False))
