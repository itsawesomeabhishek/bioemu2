"""Load possibly incomplete reference structures for atom mapping.

Only processed benchmark PDB files and mmCIF files are supported because chain
matching requires the CIF ``label_*`` fields. When loading mmCIF structures
with Biotite, set ``use_author_fields=False``.
"""

from collections.abc import Collection
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from biotite.structure import AtomArray
from biotite.structure.io import load_structure

from bioemu2.data_interface.const import TYPICAL_SUBSTITUTION_ATOMS
from bioemu2.structure import AllAtomStructure, decode_atom_name

type ChainId = str
type ResId = int
type ResName = str
type AtomName = str
type AtomIndex = int


class ResInfo(NamedTuple):
    res_name: ResName
    atom_map: dict[AtomName, AtomIndex]


type ResMap = dict[ResId, ResInfo]


def ensure_loaded_ref_struct(ref_struct_or_fpath: AtomArray | Path | str) -> AtomArray:
    """Ensure that a reference structure is loaded as an AtomArray.

    Args:
        ref_struct_or_fpath: Path to a reference structure file (e.g., PDB or CIF) or an AtomArray
            instance.

    Returns:
        AtomArray: The loaded reference structure.
    """
    if isinstance(ref_struct_or_fpath, AtomArray):
        ref_pdb = ref_struct_or_fpath
    else:
        kwargs = {}
        str_fpath = str(ref_struct_or_fpath)
        if any(str_fpath.endswith(ext) for ext in [".cif", ".cif.gz", ".bcif", ".bcif.gz"]):
            # label_* is used when converting from CIF files for PDBDataset
            kwargs["use_author_fields"] = False
        ref_pdb = load_structure(str_fpath, **kwargs)
    if not isinstance(ref_pdb, AtomArray):
        raise ValueError(
            f"Expecting reference structure to be an AtomArray (after loading), but got "
            f"{type(ref_pdb)}."
        )
    return ref_pdb


def get_chain_residue_maps_from_atom_array(
    struct_atom_array: AtomArray, ignore_h: bool = True, ignore_h2o: bool = True
) -> dict[ChainId, ResMap]:
    """Get a mapping from chain IDs to residue maps, where each residue map is a mapping from
    residue IDs to residue names and atom name to atom index mappings.
    This function also checks for duplicate or discontiguous chain and residue IDs, as well as
    duplicate atom names within the same residue, which would indicate issues with the input PDB
    file (e.g. due to alternate locations or missing residues).

    Args:
        struct_atom_array: AtomArray, containing the reference PDB structure. Use
            biotite.structure.io.load_structure to load the PDB/mmCIF file into an AtomArray.
        ignore_h: bool (default True), whether or not to ignore Hydrogen in ref structure (e.g.,
            NMR structures)
        ignore_h2o: bool (default True), whether or not to ignore water molecules in ref structure,
            which are often not included in the system definition and can cause issues with
            matching.

    Returns:
        dict[ChainId, ResMap], where
            ResMap = dict[ResId, ResInfo]
            ResInfo = dict with keys "res_name" (ResName) and "atom_map" (dict[AtomName, AtomIndex])
            type ChainId, ResName, AtomName = str; ResId, AtomIndex = int
    """
    chain_residue_maps: dict[ChainId, ResMap] = {}
    last_chain_id: ChainId | None = None
    last_res_id: ResId | None = None
    for i, (chain_id, res_id, res_name, atom_name, atom_element) in enumerate(
        zip(
            struct_atom_array.chain_id,
            struct_atom_array.res_id,
            struct_atom_array.res_name,
            struct_atom_array.atom_name,
            struct_atom_array.element,
        )
    ):
        if ignore_h and atom_element == "H":
            continue
        if ignore_h2o and res_name == "HOH":
            continue
        if chain_id != last_chain_id:
            if chain_id in chain_residue_maps:
                raise ValueError(f"Duplicate or discontiguous chain with ID {chain_id}")
            chain_residue_maps[chain_id] = {}
            last_chain_id = chain_id
            last_res_id = None
        resmap = chain_residue_maps[chain_id]
        if res_id != last_res_id:
            if res_id in resmap:
                raise ValueError(
                    f"Duplicate or discontiguous residue with ID {res_id} in chain {chain_id}"
                )
            resmap[res_id] = ResInfo(res_name=res_name, atom_map={})
            last_res_id = res_id
        if resmap[res_id].res_name != res_name:
            raise ValueError(
                f"Duplicate residue ID {res_id} with different names: {resmap[res_id]} and {res_name}"
            )
        atom_name = atom_name.strip()
        if atom_name in resmap[res_id].atom_map:
            # this avoids using PDB files with alternate locations
            raise ValueError(f"Duplicate atom name {atom_name} in residue {res_id}")
        else:
            resmap[res_id].atom_map[atom_name] = i
    return chain_residue_maps


def get_residue_maps_from_allatom_chain(allatom_chain: AllAtomStructure) -> ResMap:
    """Get a mapping from residue IDs to residue names and atom name to atom index mappings from an
    AllAtomStructure for a single chain, e.g., output of _get_polymer_structure. The residue IDs
    will be 1-based to match the convention in PDB/CIF files.

    Args:
        allatom_chain: An AllAtomStructure representing a single chain.

    Returns:
        ResMap = dict[ResId, ResInfo] (ResId is 1-based to match PDB convention), where
            ResInfo = dict with keys "res_name" (ResName) and "atom_map" (dict[AtomName, AtomIndex])
            type ResName, AtomName = str; ResId, AtomIndex = int
    """
    if len(torch.unique(allatom_chain.asym_id)) != 1:
        raise ValueError(
            f"Input allatom_chain should contain only one chain, got non-unique asym_ids: "
            f"{torch.unique(allatom_chain.asym_id).tolist()}"
        )
    atom_names = list(map(decode_atom_name, allatom_chain.name_encoding.tolist()))
    residue_map: ResMap = {}
    # since the AllAtomStruture always have the correct format and atom indices are continuous
    # we just take it without checking for contiguity
    for i, (res_id, res_name, atom_name) in enumerate(
        zip(allatom_chain.residue_idx.tolist(), allatom_chain.residue_labels, atom_names)
    ):
        res_id += 1  # match the convention in PDB files where residue IDs start from 1
        if res_id not in residue_map:
            residue_map[res_id] = ResInfo(res_name=res_name, atom_map={})
        if residue_map[res_id].res_name != res_name:
            raise ValueError(
                f"Duplicate residue ID {res_id} with different names: {residue_map[res_id]} and {res_name}"
            )
        atom_name = atom_name.strip()
        if atom_name in residue_map[res_id].atom_map:
            raise ValueError(f"Duplicate atom name {atom_name} in residue {res_id}")
        else:
            residue_map[res_id].atom_map[atom_name] = i
    return residue_map


def match_chain_atoms(
    query_chain_res_map: ResMap,
    full_chain_res_map: ResMap,
    ignorable_atoms: Collection[str] = ("OXT", "OP3", "NT"),
    allow_substitutions: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Map atom indices between query and full chain structure by residue indices and atom names.
    The assumption is that query carries a partial structure as the full (e.g., unresolved atoms
    and residues.). So it complains when the query has extra or mismatch atoms.

    Args:
        query_chain_res_map: dict[ResId, ResInfo] The incomplete structure
        full_chain_res_map: dict[ResId, ResInfo] The complete structure, where the query is mapped onto.
            type ResId, AtomIndex = int; ResName, AtomName = str
        ignorable_atoms: Collection[str] A set of atom names that can be safely ignored if they are
            present in the query but not in the full structure, e.g. terminal oxygen OXT can exist
            for a residue in the reference structure of a shorter length than the system
        allow_substitutions: bool (default True) whether or not to allow typical substitutions of
            residue names and atom names, e.g. MSE -> MET

    Returns:
        Tuple of two np.ndarray, the first array for indices of query atoms, the second array for
            indices of the full chain atoms, where the i-th query atom corresponds to the i-th full
            chain atom.

    Raises ValueError when:
    1. When the query contains extra residue index than the full;
    2. When query residue has different name from the full chain residue of the same index;
    3. When query residue has an atom name that cannot be found in the corresponding the full chain
       residue.
    """
    query_indices = []
    full_indices = []
    for res_id, res_info in query_chain_res_map.items():
        res_name = res_info.res_name
        atom_map = res_info.atom_map
        if res_id not in full_chain_res_map:
            raise ValueError(
                f"Query has residue ID {res_id} ({res_name}) that is not found in the full chain structure"
            )
        if full_chain_res_map[res_id].res_name != res_name:
            if (
                allow_substitutions
                and res_name in TYPICAL_SUBSTITUTION_ATOMS
                and TYPICAL_SUBSTITUTION_ATOMS[res_name][0] == full_chain_res_map[res_id].res_name
            ):
                # allow typical substitutions, e.g. MSE -> MET
                res_name, atom_name_map = TYPICAL_SUBSTITUTION_ATOMS[res_name]
                atom_map = {
                    atom_name_map.get(atom_name, atom_name): atom_idx
                    for atom_name, atom_idx in atom_map.items()
                }
            else:
                raise ValueError(
                    f"Query residue ID {res_id} has name {res_name} that does not match the full "
                    f"chain residue name {full_chain_res_map[res_id].res_name}"
                )
        for atom_name, atom_idx in atom_map.items():
            if atom_name not in full_chain_res_map[res_id].atom_map:
                if atom_name in ignorable_atoms:
                    continue
                raise ValueError(
                    f"Query residue ID {res_id} has atom name {atom_name} that is not found in the full chain residue"
                )
            query_indices.append(atom_idx)
            full_indices.append(full_chain_res_map[res_id].atom_map[atom_name])
    query_indices_arr = np.array(query_indices, dtype=np.int64)
    full_indices_arr = np.array(full_indices, dtype=np.int64)
    if query_indices_arr.size == 0:
        raise ValueError(
            "No matching atoms found between query and full chain structures; "
            "check residue and atom mappings."
        )
    return query_indices_arr, full_indices_arr


def merge_ref_struct_to_allatom_chain(
    allatom_chain: AllAtomStructure,
    ref_chain_res_map: ResMap,
    ref_atomarray: AtomArray,
    *,
    ligand_without_ccd: bool = False,
) -> AllAtomStructure:
    """Copy over the coordinates from the reference structure to the AllAtomStructure, mask them
    as observed and the rest as unobserved. This is used for overfitting and benchmark evaluations
    where we want to have the exact same groundtruth corresponding to the system.

    Args:
        allatom_chain: Complete template structure for one chain.
        ref_chain_res_map: Reference residues and their atom indices.
        ref_atomarray: Reference atom coordinates.
        ligand_without_ccd: If True, accept ``LIG`` in the reference where a
            SMILES/InChI ligand template uses ``UNK``. If False, match residue
            names normally (including supported standard substitutions).
    """
    # ensure the input is single chain
    input_chains = allatom_chain.asym_id.unique()
    if len(input_chains) != 1:
        raise ValueError(f"Input allatom_chain should contain only one chain, got {input_chains}")
    if ligand_without_ccd:
        ref_chain_res_map = {
            res_id: info._replace(res_name="UNK") if info.res_name == "LIG" else info
            for res_id, info in ref_chain_res_map.items()
        }
    query_indices, full_indices = match_chain_atoms(
        ref_chain_res_map, get_residue_maps_from_allatom_chain(allatom_chain)
    )
    coords = torch.zeros_like(allatom_chain.coords)
    atom_mask = torch.zeros_like(allatom_chain.atom_mask)
    coords[full_indices] = torch.from_numpy(ref_atomarray.coord[query_indices])
    atom_mask[full_indices] = True
    return allatom_chain.replace(coords=coords, atom_mask=atom_mask)
