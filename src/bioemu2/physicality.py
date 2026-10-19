"""All-atom physicality checks for filtering sampled structures.

Operates on mdtraj trajectories and flags frames that violate bond-length,
clash, valence, or topology constraints.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Collection
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import mdtraj
import numpy as np
from scipy.spatial import cKDTree

from bioemu2.data_interface.const import PROT_BACKBONE_ATOMS, REF_ATOMS, Token

logger = logging.getLogger(__name__)

BACKBONE_ATOMS = frozenset([*PROT_BACKBONE_ATOMS, "OXT"])

# Accepted (min, max) bond lengths in angstrom. Each window encloses OST's MON_LIB mean -/+ 12 std
# window (the GetBadBonds default) of every standard residue and peptide-link type sharing that
# bond, rounded outward to 0.05 angstrom. Sidechain bonds are keyed by sorted element pair.
MAINCHAIN_BOND_WINDOWS: dict[tuple[str, str], tuple[float, float]] = {
    ("N", "CA"): (1.25, 1.70),
    ("CA", "C"): (1.35, 1.70),
    ("C", "O"): (0.95, 1.50),
    ("C", "OXT"): (0.95, 1.50),
}
PEPTIDE_BOND_WINDOW = (1.05, 1.65)
SIDECHAIN_BOND_WINDOWS: dict[tuple[str, str], tuple[float, float]] = {
    ("C", "C"): (1.10, 1.80),
    ("C", "N"): (1.10, 1.70),
    ("C", "O"): (1.00, 1.60),
    ("C", "S"): (1.55, 2.10),
    # Selenomethionine has no MON_LIB entry; shift the C-S window by the Se-S covalent-radius
    # difference (0.15 angstrom). Deposited MSE Se-C bonds are about 1.95 angstrom.
    ("C", "SE"): (1.70, 2.25),
}
# Disulfides use OST's SG-SG link statistics (2.033 +/- 0.016 angstrom) with the same tolerance.
DISULFIDE_BOND_WINDOW = (1.80, 2.25)
# Other topology bonds outside the windows above (non-standard residues and cross-links) have no
# OST reference, so they use the window of their element pair, falling back to this window.
GENERIC_BOND_WINDOW = (1.00, 1.90)

MAX_HEAVY_NEIGHBORS_BY_ELEMENT: dict[str, int] = {
    "C": 4,
    "N": 4,
    "O": 2,
    "S": 6,
    "P": 5,
}

SIDECHAIN_BONDS_BY_RESIDUE: dict[str, set[tuple[str, str]]] = {
    "ALA": {("CA", "CB")},
    "ARG": {
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD"),
        ("CD", "NE"),
        ("NE", "CZ"),
        ("CZ", "NH1"),
        ("CZ", "NH2"),
    },
    "ASN": {("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "ND2")},
    "ASP": {("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "OD2")},
    "CYS": {("CA", "CB"), ("CB", "SG")},
    "GLN": {("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "NE2")},
    "GLU": {("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "OE2")},
    "GLY": set(),
    "HIS": {
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "ND1"),
        ("CG", "CD2"),
        ("ND1", "CE1"),
        ("CD2", "NE2"),
        ("CE1", "NE2"),
    },
    "ILE": {("CA", "CB"), ("CB", "CG1"), ("CB", "CG2"), ("CG1", "CD1")},
    "LEU": {("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2")},
    "LYS": {("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "CE"), ("CE", "NZ")},
    "MET": {("CA", "CB"), ("CB", "CG"), ("CG", "SD"), ("SD", "CE")},
    "PHE": {
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD1"),
        ("CG", "CD2"),
        ("CD1", "CE1"),
        ("CD2", "CE2"),
        ("CE1", "CZ"),
        ("CE2", "CZ"),
    },
    "PRO": {("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "N")},
    "SER": {("CA", "CB"), ("CB", "OG")},
    "THR": {("CA", "CB"), ("CB", "OG1"), ("CB", "CG2")},
    "TRP": {
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD1"),
        ("CG", "CD2"),
        ("CD1", "NE1"),
        ("NE1", "CE2"),
        ("CD2", "CE2"),
        ("CD2", "CE3"),
        ("CE2", "CZ2"),
        ("CE3", "CZ3"),
        ("CZ2", "CH2"),
        ("CZ3", "CH2"),
    },
    "TYR": {
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD1"),
        ("CG", "CD2"),
        ("CD1", "CE1"),
        ("CD2", "CE2"),
        ("CE1", "CZ"),
        ("CE2", "CZ"),
        ("CZ", "OH"),
    },
    "VAL": {("CA", "CB"), ("CB", "CG1"), ("CB", "CG2")},
}

SIDECHAIN_ATOMS_BY_RESIDUE: dict[str, frozenset[str]] = {
    res_name: frozenset(
        atom
        for atom in REF_ATOMS[Token.from_mdtraj_residue_name(res_name)]
        if atom not in PROT_BACKBONE_ATOMS
    )
    for res_name in SIDECHAIN_BONDS_BY_RESIDUE
}

LIGAND_RESNAMES: tuple[str, ...] = (
    "LIG",
    "L01",
    "L02",
    "L03",
    "L04",
    "L05",
    "L06",
    "L07",
    "L08",
    "L09",
)

# Frames per vectorized geometry batch, bounding pair-distance memory for large systems.
_FRAME_CHUNK_SIZE = 128


@dataclass(frozen=True)
class PhysicalityCutoffs:
    """Distance cutoffs in angstrom for the all-atom physicality checks.

    These distances assume that there are no hydrogens in the structure.  Bond
    lengths use the fixed ``MAINCHAIN_BOND_WINDOWS``, ``PEPTIDE_BOND_WINDOW``,
    and ``SIDECHAIN_BOND_WINDOWS`` instead.
    """

    too_close_distance: float = 1.0  # Any two protein atoms closer than this are too close.
    inferred_bond_distance: float = 1.9  # Neighbors within this count toward valence.
    # Atoms without a neighbor within this, or a bonded neighbor within its bond window, are
    # isolated.
    isolated_atom_distance: float = 2.0
    sidechain_clash_distance: float = 2.0  # Clash threshold for atoms of distinct residues.
    degenerate_geometry_distance: float = 0.5  # Atoms of one residue closer than this fail it.


@dataclass(frozen=True)
class PhysicalityThresholds:
    """Per-check thresholds for deciding whether a frame is physical.

    When ``per_residue=True``, raw violation counts are divided by the number
    of residues before comparing against the threshold.  When
    ``per_residue=False``, raw counts are compared directly.

    ``bad_bonds_sidechain`` counts residues with a failing sidechain,
    ``within_residue_clashes`` counts too-close atom pairs anywhere in the
    protein, and the inter-residue clash fields count unique atom pairs rather
    than residue pairs.
    """

    bad_bonds_backbone: float = 0.0
    bad_bonds_sidechain: float = 0.0
    # Bonded S-S pairs outside DISULFIDE_BOND_WINDOW; no preset tolerates one.
    bad_bonds_sulfur: float = 0.0
    within_residue_clashes: float = 0.0
    valence_exceeded: float = 0.0
    isolated_atoms: float = 0.0
    same_chain_residue_clashes: float = 0.0
    cross_chain_residue_clashes: float = 0.0
    duplicate_atoms: float = 0.0
    atoms_mismatch: float = 0.0
    bonds_mismatch: float = 0.0
    per_residue: bool = False

    def get_thresholds(self) -> dict[str, float]:
        thresholds = asdict(self)
        thresholds.pop("per_residue")
        return thresholds


class PhysicalityFilterMode(StrEnum):
    NO_FILTER = "no_filter"
    NO_VIOLATIONS_ALLOWED = "no_violations"
    HIGH_QUALITY = "high_quality"
    HIGH_QUALITY_PER_RESIDUE = "high_quality_per_residue"


@dataclass(frozen=True)
class PhysicalityReport:
    """Physicality results in input-frame order, without modifying coordinates.

    Attributes:
        passed: Boolean array of shape [num_frames]; True means checks passed.
            For NO_FILTER, every entry is True and no checks were run.
        reasons: First failed check per frame, or None if passed or not checked.
        mode: Requested threshold preset.
        checked: Whether protein physicality checks were performed.
    """

    passed: np.ndarray
    reasons: list[str | None]
    mode: PhysicalityFilterMode
    checked: bool = True

    def save(self, path: str | Path) -> None:
        """Write per-frame statuses and failure reasons to a JSON file."""
        records = []
        for frame, (passed, reason) in enumerate(zip(self.passed, self.reasons, strict=True)):
            status = "passed" if passed else "rejected"
            if not self.checked or self.mode is PhysicalityFilterMode.NO_FILTER:
                status = "not_checked"
            records.append({"frame": frame, "status": status, "reason": reason})
        Path(path).write_text(json.dumps(records, indent=2) + "\n")


def normalize_physicality_mode(
    mode: PhysicalityFilterMode | str,
) -> PhysicalityFilterMode:
    """Resolve a named preset; use ``NO_FILTER`` to disable filtering."""
    return PhysicalityFilterMode(mode)


PHYSICALITY_THRESHOLDS: dict[PhysicalityFilterMode, PhysicalityThresholds] = {
    PhysicalityFilterMode.NO_VIOLATIONS_ALLOWED: PhysicalityThresholds(),
    PhysicalityFilterMode.HIGH_QUALITY: PhysicalityThresholds(
        bad_bonds_backbone=5,
        bad_bonds_sidechain=10,
        within_residue_clashes=5,
        valence_exceeded=10,
        isolated_atoms=1,
        same_chain_residue_clashes=10,
        cross_chain_residue_clashes=5,
        duplicate_atoms=0,
        atoms_mismatch=0,
        bonds_mismatch=0,
        per_residue=False,
    ),
    PhysicalityFilterMode.HIGH_QUALITY_PER_RESIDUE: PhysicalityThresholds(
        bad_bonds_backbone=0.05,
        bad_bonds_sidechain=0.1,
        within_residue_clashes=0.05,
        valence_exceeded=0.1,
        isolated_atoms=0.01,
        same_chain_residue_clashes=0.1,
        cross_chain_residue_clashes=0.05,
        duplicate_atoms=0.0,
        atoms_mismatch=0.0,
        bonds_mismatch=0.0,
        per_residue=True,
    ),
}


def get_protein_atom_indices(topology: mdtraj.Topology) -> np.ndarray:
    """Select multi-residue chains with a standard amino-acid anchor."""
    protein_chains = {
        chain.index
        for chain in topology.chains
        if chain.n_residues > 1
        and any(res.name in SIDECHAIN_BONDS_BY_RESIDUE for res in chain.residues)
    }
    return np.array(
        [atom.index for atom in topology.atoms if atom.residue.chain.index in protein_chains],
        dtype=np.int64,
    )


def _select_expected_extra_bonds(
    expected_extra_bonds: Collection[tuple[int, int]], selected_indices: np.ndarray, n_atoms: int
) -> set[tuple[int, int]]:
    atom_mapping = {int(original): selected for selected, original in enumerate(selected_indices)}
    selected_bonds = set()
    for first, second in expected_extra_bonds:
        if first == second or not (0 <= first < n_atoms and 0 <= second < n_atoms):
            raise ValueError(f"Invalid extra bond atom indices: {(first, second)}")
        if first in atom_mapping and second in atom_mapping:
            selected_bonds.add((atom_mapping[first], atom_mapping[second]))
        elif first in atom_mapping or second in atom_mapping:
            logger.warning(
                f"Extra bond {(first, second)} leaves the checked protein chains and is not checked."
            )
    return selected_bonds


def _ligand_residues(topology: mdtraj.Topology) -> set[int]:
    """Return indices of residues named as ligands or in chains without a standard amino acid."""
    ligand_chains = {
        chain.index
        for chain in topology.chains
        if not any(res.name in SIDECHAIN_BONDS_BY_RESIDUE for res in chain.residues)
    }
    return {
        residue.index
        for residue in topology.residues
        if residue.name in LIGAND_RESNAMES or residue.chain.index in ligand_chains
    }


def _alpha_backbone_residues(topology: mdtraj.Topology) -> set[int]:
    """Return indices of residues that take part in main-chain and peptide-bond checks.

    Standard amino acids always qualify.  Non-standard residues qualify only
    with an alpha-amino-acid backbone, that is, topology bonds N-CA, CA-C, and
    C-O.  Beta- and gamma-amino acids such as BAL or STA place extra atoms
    between N and C, so the main-chain windows do not apply to them; their
    bonds are still checked against element-pair windows.
    """
    backbone_bonds = {("CA", "N"), ("C", "CA"), ("C", "O")}
    bonded_names: dict[int, set[tuple[str, str]]] = {}
    for first, second in topology.bonds:
        if first.residue.index == second.residue.index:
            bonded_names.setdefault(first.residue.index, set()).add(
                _bond_key(first.name, second.name)
            )
    return {
        residue.index
        for residue in topology.residues
        if residue.name in SIDECHAIN_BONDS_BY_RESIDUE
        or backbone_bonds <= bonded_names.get(residue.index, set())
    }


def _bond_key(first: str, second: str) -> tuple[str, str]:
    return (first, second) if first <= second else (second, first)


def _check_topology(
    samples: mdtraj.Trajectory, expected_extra_bonds: Collection[tuple[int, int]] = ()
) -> dict[str, int]:
    """Check if the topology contains the expected atoms, residues, and bonds.

    Checks standard amino-acid topology; skips ligands and nonstandard residue
    atom and sidechain bond expectations. Results are per-system (not per-frame).

    Args:
        samples: Samples to check topology for.
        expected_extra_bonds: Pairs of atom indices in ``samples.topology`` that
            represent declared nonstandard connections.

    Returns:
        Dictionary counting topology violations for ``duplicate_atoms`` (atom name
        twice in residue), ``atoms_mismatch`` (current atoms don't match expected),
        and ``bonds_mismatch`` (current bonds don't match expected).
    """
    ligand_residues = _ligand_residues(samples.topology)
    backbone_residues = _alpha_backbone_residues(samples.topology)
    duplicate_atoms = 0
    atoms_mismatch = 0
    expected_bonds: set[tuple[str, str]] = set()
    for chain in samples.topology.chains:
        previous_peptide_carbon: str | None = None
        for residue in chain.residues:
            if residue.index in ligand_residues:
                previous_peptide_carbon = None
                continue
            residue_atoms = [a.name for a in residue.atoms]
            duplicate_atoms += len(residue_atoms) != len(set(residue_atoms))
            standard = residue.name in SIDECHAIN_BONDS_BY_RESIDUE
            if standard:
                token = Token.from_mdtraj_residue_name(residue.name)
                expected_atoms = set(REF_ATOMS[token])
                observed_atoms = {name for name in residue_atoms if not name.startswith("H")}
                atoms_mismatch += observed_atoms != expected_atoms
            residue_tag = f"{residue.name}{residue.index}"
            if residue.index in backbone_residues:
                backbone_bonds = {("N", "CA"), ("CA", "C"), ("C", "O")}
                bonds = backbone_bonds | (
                    SIDECHAIN_BONDS_BY_RESIDUE[residue.name] if standard else set()
                )
                expected_bonds.update(
                    _bond_key(f"{residue_tag}-{first}", f"{residue_tag}-{second}")
                    for first, second in bonds
                )
                if previous_peptide_carbon is not None:
                    expected_bonds.add(_bond_key(previous_peptide_carbon, f"{residue_tag}-N"))
                previous_peptide_carbon = f"{residue_tag}-C"
            else:
                previous_peptide_carbon = None

    for first_index, second_index in expected_extra_bonds:
        if first_index == second_index or not (
            0 <= first_index < samples.n_atoms and 0 <= second_index < samples.n_atoms
        ):
            raise ValueError(f"Invalid extra bond atom indices: {(first_index, second_index)}")
        first = samples.topology.atom(first_index)
        second = samples.topology.atom(second_index)
        if (
            first.residue.name in SIDECHAIN_BONDS_BY_RESIDUE
            and second.residue.name in SIDECHAIN_BONDS_BY_RESIDUE
        ):
            expected_bonds.add(
                _bond_key(
                    f"{first.residue.name}{first.residue.index}-{first.name}",
                    f"{second.residue.name}{second.residue.index}-{second.name}",
                )
            )
    current_bonds: set[tuple[str, str]] = set()
    for first, second in samples.topology.bonds:
        names = _bond_key(
            f"{first.residue.name}{first.residue.index}-{first.name}",
            f"{second.residue.name}{second.residue.index}-{second.name}",
        )
        if (
            first.residue.name in SIDECHAIN_BONDS_BY_RESIDUE
            and second.residue.name in SIDECHAIN_BONDS_BY_RESIDUE
            and first.residue.index not in ligand_residues
            and second.residue.index not in ligand_residues
        ) or names in expected_bonds:
            current_bonds.add(names)
    bonds_mismatch = len(current_bonds.symmetric_difference(expected_bonds))

    return {
        "duplicate_atoms": duplicate_atoms,
        "atoms_mismatch": atoms_mismatch,
        "bonds_mismatch": bonds_mismatch,
    }


def _pair_distances_sq(coords: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Compute squared distances for atom index pairs.

    Args:
        coords: Coordinates with shape ``[..., n_atoms, 3]``.
        pairs: Atom index pairs with shape ``[n_pairs, 2]``.

    Returns:
        Squared distances with shape ``[..., n_pairs]``.
    """
    deltas = coords[..., pairs[:, 0], :] - coords[..., pairs[:, 1], :]
    return np.einsum("...j,...j->...", deltas, deltas)


def _flag_residues(is_bad: np.ndarray, pair_flags: np.ndarray, pair_residues: np.ndarray) -> None:
    """Mark residues with at least one flagged pair, in place.

    Args:
        is_bad: Per-frame residue flags with shape ``[n_frames, n_residues]``.
        pair_flags: Per-frame pair flags with shape ``[n_frames, n_pairs]``.
        pair_residues: Non-decreasing residue index of each pair with shape ``[n_pairs]``.
    """
    starts = np.flatnonzero(np.r_[True, pair_residues[1:] != pair_residues[:-1]])
    is_bad[:, pair_residues[starts]] |= np.logical_or.reduceat(pair_flags, starts, axis=-1)


@dataclass(frozen=True)
class _BondGeometry:
    """Frame-independent pair lists for the bond and sidechain checks.

    Residue indices count only the non-ligand residues of the topology, in topology order.

    Attributes:
        mainchain_pairs: Atom index pairs of main-chain and peptide bonds. ``int64``,
            shape ``[n_mainchain_bonds, 2]``.
        mainchain_windows: Allowed ``(min, max)`` length of each main-chain bond in angstroms.
            ``float64``, shape ``[n_mainchain_bonds, 2]``.
        sidechain_static_failure: Whether each residue fails the sidechain check regardless of
            coordinates (missing or repeated atom names). ``bool``, shape ``[n_residues]``.
        sidechain_pairs: Atom index pairs of checked sidechain and other bonds. ``int64``,
            shape ``[n_sidechain_bonds, 2]``.
        sidechain_windows: Allowed ``(min, max)`` length of each sidechain bond in angstroms.
            ``float64``, shape ``[n_sidechain_bonds, 2]``.
        sidechain_residues: Non-decreasing residue index of each sidechain bond. ``int64``,
            shape ``[n_sidechain_bonds]``.
        degenerate_pairs: Intra-residue atom index pairs checked for near-overlap. ``int64``,
            shape ``[n_degenerate_pairs, 2]``.
        degenerate_residues: Non-decreasing residue index of each degenerate pair. ``int64``,
            shape ``[n_degenerate_pairs]``.
    """

    mainchain_pairs: np.ndarray
    mainchain_windows: np.ndarray
    sidechain_static_failure: np.ndarray
    sidechain_pairs: np.ndarray
    sidechain_windows: np.ndarray
    sidechain_residues: np.ndarray
    degenerate_pairs: np.ndarray
    degenerate_residues: np.ndarray


def _bond_window(
    atom_i: mdtraj.core.topology.Atom, atom_j: mdtraj.core.topology.Atom
) -> tuple[float, float]:
    elements = sorted((atom_i.element.symbol.upper(), atom_j.element.symbol.upper()))
    if elements == ["S", "S"]:
        return DISULFIDE_BOND_WINDOW
    return SIDECHAIN_BOND_WINDOWS.get((elements[0], elements[1]), GENERIC_BOND_WINDOW)


def _bond_geometry(topology: mdtraj.Topology) -> _BondGeometry:
    """Collect the bonds and residue atom pairs checked in every frame.

    Main-chain bonds cover every non-ligand residue with a backbone (see
    :func:`_alpha_backbone_residues`); peptide bonds link consecutive such residues of a
    chain.  A residue fails the sidechain check when it has a bond outside its
    element-pair window.  Standard amino acids are checked against their
    expected sidechain bonds and also fail when they miss expected sidechain
    atoms, repeat an atom name, or have two atoms closer than the degenerate
    distance.  Any other non-ligand topology bond, such as a bond of a
    non-standard residue or a disulfide, flags the residues of both atoms.
    """
    ligand_residues = _ligand_residues(topology)
    backbone_residues = _alpha_backbone_residues(topology)
    mainchain_pairs, mainchain_windows = [], []
    static_failure: list[bool] = []
    residue_idx_by_index: dict[int, int] = {}
    covered_bonds: set[tuple[int, int]] = set()
    sidechain_entries: list[tuple[int, int, int, tuple[float, float]]] = []
    degenerate_pairs, degenerate_residues = [], []
    for chain in topology.chains:
        previous_carbon = None
        for residue in chain.residues:
            if residue.index in ligand_residues:
                previous_carbon = None
                continue
            residue_idx = len(static_failure)
            residue_idx_by_index[residue.index] = residue_idx
            atom_by_name = {atom.name: atom for atom in residue.atoms}
            expected_atoms = SIDECHAIN_ATOMS_BY_RESIDUE.get(residue.name)
            static_failure.append(
                expected_atoms is not None
                and (
                    bool(expected_atoms - set(atom_by_name)) or len(atom_by_name) != residue.n_atoms
                )
            )
            if residue.index in backbone_residues:
                for (name_i, name_j), window in MAINCHAIN_BOND_WINDOWS.items():
                    if name_i in atom_by_name and name_j in atom_by_name:
                        mainchain_pairs.append(
                            (atom_by_name[name_i].index, atom_by_name[name_j].index)
                        )
                        mainchain_windows.append(window)
                if previous_carbon is not None and "N" in atom_by_name:
                    mainchain_pairs.append((previous_carbon.index, atom_by_name["N"].index))
                    mainchain_windows.append(PEPTIDE_BOND_WINDOW)
                previous_carbon = atom_by_name.get("C")
            else:
                previous_carbon = None

            if expected_atoms is None:
                continue
            for name_i, name_j in SIDECHAIN_BONDS_BY_RESIDUE[residue.name]:
                atom_i = atom_by_name.get(name_i)
                atom_j = atom_by_name.get(name_j)
                if atom_i is None or atom_j is None:
                    continue
                sidechain_entries.append(
                    (residue_idx, atom_i.index, atom_j.index, _bond_window(atom_i, atom_j))
                )
                covered_bonds.add(
                    (min(atom_i.index, atom_j.index), max(atom_i.index, atom_j.index))
                )
            residue_atoms = list(atom_by_name.values())
            for i, atom_i in enumerate(residue_atoms[:-1]):
                for atom_j in residue_atoms[i + 1 :]:
                    degenerate_pairs.append((atom_i.index, atom_j.index))
                    degenerate_residues.append(residue_idx)

    covered_bonds.update((min(i, j), max(i, j)) for i, j in mainchain_pairs)
    for atom_i, atom_j in topology.bonds:
        key = (min(atom_i.index, atom_j.index), max(atom_i.index, atom_j.index))
        residue_i = residue_idx_by_index.get(atom_i.residue.index)
        residue_j = residue_idx_by_index.get(atom_j.residue.index)
        if key in covered_bonds or residue_i is None or residue_j is None:
            continue
        covered_bonds.add(key)
        window = _bond_window(atom_i, atom_j)
        for residue_idx in {residue_i, residue_j}:
            sidechain_entries.append((residue_idx, atom_i.index, atom_j.index, window))
    # Sorting by residue keeps each residue's pairs contiguous for `_flag_residues`.
    sidechain_entries.sort(key=lambda entry: entry[0])

    return _BondGeometry(
        mainchain_pairs=np.array(mainchain_pairs, dtype=np.int64).reshape(-1, 2),
        mainchain_windows=np.array(mainchain_windows, dtype=float).reshape(-1, 2),
        sidechain_static_failure=np.array(static_failure, dtype=bool),
        sidechain_pairs=np.array(
            [entry[1:3] for entry in sidechain_entries], dtype=np.int64
        ).reshape(-1, 2),
        sidechain_windows=np.array([entry[3] for entry in sidechain_entries], dtype=float).reshape(
            -1, 2
        ),
        sidechain_residues=np.array([entry[0] for entry in sidechain_entries], dtype=np.int64),
        degenerate_pairs=np.array(degenerate_pairs, dtype=np.int64).reshape(-1, 2),
        degenerate_residues=np.array(degenerate_residues, dtype=np.int64),
    )


def _outside_windows(coords: np.ndarray, pairs: np.ndarray, windows: np.ndarray) -> np.ndarray:
    distances = np.sqrt(_pair_distances_sq(coords, pairs))
    return (distances < windows[:, 0]) | (distances > windows[:, 1])


def _count_bad_mainchain_bonds(coords: np.ndarray, geometry: _BondGeometry) -> np.ndarray:
    """Count main-chain bonds outside their windows; ``coords`` are angstrom ``[T, N, 3]``."""
    return np.count_nonzero(
        _outside_windows(coords, geometry.mainchain_pairs, geometry.mainchain_windows), axis=-1
    )


def _count_bad_sidechain_conformations(
    coords: np.ndarray, geometry: _BondGeometry, degenerate_distance: float
) -> np.ndarray:
    """Count residues failing sidechain checks; ``coords`` are angstrom ``[T, N, 3]``."""
    is_bad = np.repeat(geometry.sidechain_static_failure[None], coords.shape[0], axis=0)
    if len(geometry.sidechain_pairs):
        is_bad_bond = _outside_windows(coords, geometry.sidechain_pairs, geometry.sidechain_windows)
        _flag_residues(is_bad, is_bad_bond, geometry.sidechain_residues)
    if len(geometry.degenerate_pairs):
        is_degenerate = (
            _pair_distances_sq(coords, geometry.degenerate_pairs) < degenerate_distance**2
        )
        _flag_residues(is_bad, is_degenerate, geometry.degenerate_residues)
    return np.count_nonzero(is_bad, axis=-1)


def _bonded_pair_indices(pairs: np.ndarray, bonded_pairs: np.ndarray) -> np.ndarray:
    """Find each atom pair (``[n_pairs, 2]``, either order) in ``bonded_pairs``.

    Returns:
        Row index into ``bonded_pairs`` per pair, or -1 for unbonded pairs.
    """
    result = np.full(pairs.shape[0], -1, dtype=np.int64)
    if bonded_pairs.shape[0] == 0 or pairs.shape[0] == 0:
        return result
    n_keys = int(max(pairs.max(), bonded_pairs.max())) + 1

    def _keys(index_pairs: np.ndarray) -> np.ndarray:
        low = np.minimum(index_pairs[:, 0], index_pairs[:, 1]).astype(np.int64)
        high = np.maximum(index_pairs[:, 0], index_pairs[:, 1]).astype(np.int64)
        return low * n_keys + high

    bonded_keys = _keys(bonded_pairs)
    order = np.argsort(bonded_keys)
    sorted_keys = bonded_keys[order]
    pair_keys = _keys(pairs)
    positions = np.minimum(np.searchsorted(sorted_keys, pair_keys), len(sorted_keys) - 1)
    found = sorted_keys[positions] == pair_keys
    result[found] = order[positions[found]]
    return result


def _candidate_pairs(coords: np.ndarray, max_distance: float) -> np.ndarray:
    """Collect atom pairs within ``max_distance`` in any frame.

    Args:
        coords: Coordinates with shape ``[n_frames, n_atoms, 3]``.
        max_distance: Neighbor search radius in the units of ``coords``.

    Returns:
        Sorted unique atom index pairs (``i < j``) with shape ``[n_pairs, 2]``.
    """
    n_atoms = coords.shape[1]
    pairs = np.concatenate(
        [
            cKDTree(frame).query_pairs(max_distance, output_type="ndarray").reshape(-1, 2)
            for frame in coords
        ]
    ).astype(np.int64)
    keys = np.unique(pairs[:, 0] * n_atoms + pairs[:, 1])
    return np.stack(np.divmod(keys, n_atoms), axis=-1)


def _count_pair_atoms(pair_mask: np.ndarray, pairs: np.ndarray, n_atoms: int) -> np.ndarray:
    """Count flagged pairs touching each atom.

    Args:
        pair_mask: Per-frame pair flags with shape ``[n_frames, n_pairs]``.
        pairs: Atom index pairs with shape ``[n_pairs, 2]``.
        n_atoms: Number of atoms.

    Returns:
        Per-frame atom counts with shape ``[n_frames, n_atoms]``.
    """
    frame_idx, pair_idx = np.nonzero(pair_mask)
    offsets = frame_idx * n_atoms
    flat_atoms = np.concatenate([offsets + pairs[pair_idx, 0], offsets + pairs[pair_idx, 1]])
    return np.bincount(flat_atoms, minlength=pair_mask.shape[0] * n_atoms).reshape(
        pair_mask.shape[0], n_atoms
    )


def _compute_geometry_stats(
    samples: mdtraj.Trajectory, cutoffs: PhysicalityCutoffs
) -> dict[str, np.ndarray]:
    """Compute the per-frame bond, neighbor, and clash statistics.

    Ligand atoms (``LIGAND_RESNAMES`` residues and chains without a standard
    amino acid) only take part in cross-chain clashes.
    Atom pairs bonded in the topology never count as clashes, and same-chain
    clashes only count sidechain atoms (all non-backbone atoms for
    non-standard residues), because neighboring backbones are always close.
    An unbonded CYS SG-SG pair in the disulfide distance window is not a
    residue clash if neither sulfur has another candidate partner in that frame.

    Frames are evaluated in chunks.  Neighbor checks use the union of atom
    pairs found within the largest neighbor cutoff in any frame of the chunk,
    which contains every pair each frame's own neighbor search would return.
    Frames with non-finite coordinates only report ``non_finite_atoms``; their
    other geometry statistics stay zero.
    """
    topology = samples.topology
    ligand_residues = _ligand_residues(topology)
    is_ligand = np.array(
        [atom.residue.index in ligand_residues for atom in topology.atoms], dtype=bool
    )
    is_protein = ~is_ligand
    protein_indices = np.flatnonzero(is_protein)
    if protein_indices.size == 0:
        raise ValueError("Physicality filtering requires at least one protein atom.")

    geometry = _bond_geometry(topology)
    bonded_pairs = np.array(
        [[bond[0].index, bond[1].index] for bond in topology.bonds], dtype=np.int64
    ).reshape(-1, 2)
    is_sulfur_bond = [a.element == b.element == mdtraj.element.sulfur for a, b in topology.bonds]
    sulfur_pairs = bonded_pairs[np.array(is_sulfur_bond, dtype=bool)]
    # Bonded atoms are neighbors up to the upper bound of their bond window, so long bonds such
    # as Se-C or S-S do not leave their atoms isolated.
    bonded_neighbor_distances = np.array(
        [
            max(cutoffs.isolated_atom_distance, _bond_window(first, second)[1])
            for first, second in topology.bonds
        ],
        dtype=float,
    )
    chain_ids = np.array([atom.residue.chain.index for atom in topology.atoms])
    residue_ids = np.array([atom.residue.index for atom in topology.atoms])
    is_cysteine_sulfur = np.array(
        [
            atom.residue.name == "CYS"
            and atom.name == "SG"
            and atom.element == mdtraj.element.sulfur
            for atom in topology.atoms
        ],
        dtype=bool,
    )
    is_clash_eligible = np.array(
        [
            not is_ligand[atom.index]
            and atom.name not in BACKBONE_ATOMS
            and (
                atom.residue.name not in SIDECHAIN_ATOMS_BY_RESIDUE
                or atom.name in SIDECHAIN_ATOMS_BY_RESIDUE[atom.residue.name]
            )
            for atom in topology.atoms
        ],
        dtype=bool,
    )
    max_neighbors = np.array(
        [
            MAX_HEAVY_NEIGHBORS_BY_ELEMENT.get(atom.element.symbol.upper(), 4)
            for atom in topology.atoms
        ],
        dtype=np.int64,
    )
    max_distance = max(
        cutoffs.too_close_distance,
        cutoffs.inferred_bond_distance,
        cutoffs.sidechain_clash_distance,
        cutoffs.isolated_atom_distance,
        bonded_neighbor_distances.max(initial=0.0),
        DISULFIDE_BOND_WINDOW[1],
    )

    stats = {
        key: np.zeros(len(samples), dtype=np.int64)
        for key in (
            "bad_bonds_backbone",
            "bad_bonds_sidechain",
            "bad_bonds_sulfur",
            "within_residue_clashes",
            "valence_exceeded",
            "isolated_atoms",
            "same_chain_residue_clashes",
            "cross_chain_residue_clashes",
            "non_finite_atoms",
        )
    }
    for start in range(0, len(samples), _FRAME_CHUNK_SIZE):
        chunk_xyz = samples.xyz[start : start + _FRAME_CHUNK_SIZE]
        coords = np.asarray(chunk_xyz, dtype=float) * 10.0  # nanometers to angstrom
        is_finite_atom = np.isfinite(coords).all(axis=-1)
        is_finite_frame = is_finite_atom.all(axis=-1)
        frames = start + np.arange(len(coords))
        stats["non_finite_atoms"][frames] = np.count_nonzero(~is_finite_atom, axis=-1)
        # Neighbor search rejects non-finite coordinates, so only finite frames are checked.
        chunk = frames[is_finite_frame]
        coords = coords[is_finite_frame]
        if len(chunk) == 0:
            continue
        stats["bad_bonds_backbone"][chunk] = _count_bad_mainchain_bonds(coords, geometry)
        stats["bad_bonds_sidechain"][chunk] = _count_bad_sidechain_conformations(
            coords, geometry, cutoffs.degenerate_geometry_distance
        )
        stats["bad_bonds_sulfur"][chunk] = np.count_nonzero(
            _outside_windows(coords, sulfur_pairs, np.array([DISULFIDE_BOND_WINDOW])), axis=-1
        )

        pairs = _candidate_pairs(coords, max_distance)
        distances_sq = _pair_distances_sq(coords, pairs)
        atom_i, atom_j = pairs[:, 0], pairs[:, 1]
        protein_pair = is_protein[atom_i] & is_protein[atom_j]
        same_chain = chain_ids[atom_i] == chain_ids[atom_j]
        bond_indices = _bonded_pair_indices(pairs, bonded_pairs)
        is_bonded = bond_indices >= 0
        clash_pair = (residue_ids[atom_i] != residue_ids[atom_j]) & ~is_bonded
        # Index -1 (unbonded pairs) selects the appended isolated-atom distance.
        neighbor_distance = np.append(bonded_neighbor_distances, cutoffs.isolated_atom_distance)[
            bond_indices
        ]
        is_clash = distances_sq < cutoffs.sidechain_clash_distance**2
        disulfide_candidates = (
            is_cysteine_sulfur[atom_i]
            & is_cysteine_sulfur[atom_j]
            & (residue_ids[atom_i] != residue_ids[atom_j])
        )
        for frame_idx in range(len(coords)):
            possible = np.flatnonzero(
                disulfide_candidates
                & (distances_sq[frame_idx] >= DISULFIDE_BOND_WINDOW[0] ** 2)
                & (distances_sq[frame_idx] <= DISULFIDE_BOND_WINDOW[1] ** 2)
            )
            if len(possible):
                endpoints = pairs[possible]
                partners = np.bincount(endpoints.ravel(), minlength=samples.n_atoms)
                unambiguous = np.all(partners[endpoints] == 1, axis=1)
                is_clash[frame_idx, possible[unambiguous]] = False

        stats["within_residue_clashes"][chunk] = np.count_nonzero(
            protein_pair & (distances_sq < cutoffs.too_close_distance**2), axis=1
        )
        inferred_bonds = _count_pair_atoms(
            protein_pair & (distances_sq <= cutoffs.inferred_bond_distance**2),
            pairs,
            samples.n_atoms,
        )
        stats["valence_exceeded"][chunk] = np.count_nonzero(inferred_bonds > max_neighbors, axis=1)
        neighbor_counts = _count_pair_atoms(
            protein_pair & (distances_sq <= neighbor_distance**2),
            pairs,
            samples.n_atoms,
        )[:, protein_indices]
        stats["isolated_atoms"][chunk] = np.count_nonzero(neighbor_counts == 0, axis=1)
        stats["same_chain_residue_clashes"][chunk] = np.count_nonzero(
            same_chain
            & clash_pair
            & is_clash_eligible[atom_i]
            & is_clash_eligible[atom_j]
            & is_clash,
            axis=1,
        )
        stats["cross_chain_residue_clashes"][chunk] = np.count_nonzero(
            ~same_chain & clash_pair & is_clash, axis=1
        )
    return stats


def compute_physicality_stats(
    samples: mdtraj.Trajectory,
    cutoffs: PhysicalityCutoffs | None = None,
    expected_extra_bonds: Collection[tuple[int, int]] = (),
) -> dict[str, np.ndarray]:
    """Compute physicality statistics for samples.

    Topology results are valid for the whole trajectory but broadcast over
    frames for convenience.  Trajectories should have all hydrogens removed.

    Args:
        samples: Samples to analyze.
        cutoffs: Distance cutoffs for the underlying checks (angstrom).
        expected_extra_bonds: Atom-index pairs for expected nonstandard bonds in
            ``samples.topology`` (e.g., an explicitly declared disulfide bond).

    Returns:
        Dictionary mapping check names to per-frame violation counts:

        - ``duplicate_atoms``: residues with a repeated atom name.
        - ``atoms_mismatch``: residues whose atoms differ from the expected ones.
        - ``bonds_mismatch``: topology bonds differing from the expected ones.
        - ``same_chain_residue_clashes``: unique sidechain atom pairs of
          distinct residues of one chain that clash.
        - ``cross_chain_residue_clashes``: unique heavy-atom pairs of distinct
          chains that clash, including ligand atoms.
        - ``bad_bonds_backbone``: main-chain and peptide bonds outside their windows.
        - ``bad_bonds_sidechain``: residues with an invalid sidechain conformation.
        - ``bad_bonds_sulfur``: bonded S-S pairs outside ``DISULFIDE_BOND_WINDOW``.
        - ``within_residue_clashes``: too-close atom pairs anywhere in the protein.
        - ``valence_exceeded``: atoms with more inferred neighbors than their element allows.
        - ``isolated_atoms``: protein atoms with no neighbor.
        - ``non_finite_atoms``: atoms with a NaN or infinite coordinate.
    """
    cutoffs = cutoffs or PhysicalityCutoffs()
    n_frames = len(samples)

    geometry_stats = _compute_geometry_stats(samples, cutoffs)
    topology_stats = _check_topology(samples, expected_extra_bonds=expected_extra_bonds)
    topology_stats = {
        k: np.full(n_frames, topology_stats[k], dtype=np.int64) for k in topology_stats
    }

    return geometry_stats | topology_stats


def check_allatom_physicality(
    samples: mdtraj.Trajectory,
    cutoffs: PhysicalityCutoffs | None = None,
    *,
    mode: PhysicalityFilterMode,
    thresholds: PhysicalityThresholds | None = None,
    expected_extra_bonds: Collection[tuple[int, int]] = (),
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str | None]]:
    """Flag physically valid frames of an all-atom trajectory.

    The trajectory should contain **protein residues only** (use
    :func:`get_protein_atom_indices` to subset before calling).

    Args:
        samples: All-atom mdtraj trajectory (nanometer coordinates, no hydrogens).
        cutoffs: Distance cutoffs for the underlying checks.
        mode: Named threshold preset.
        thresholds: Explicit thresholds (overrides ``mode``).
        expected_extra_bonds: Atom-index pairs for expected nonstandard bonds in
            ``samples.topology``.

    Returns:
        ``(is_physical, stats, reasons)`` — per-frame boolean keep array, raw
        violation counts, and first exceeded threshold per rejected frame.
        Frames with non-finite coordinates are always rejected with reason
        ``non_finite_atoms``.
    """
    stats = compute_physicality_stats(
        samples, cutoffs=cutoffs, expected_extra_bonds=expected_extra_bonds
    )
    if thresholds is None:
        thresholds = PHYSICALITY_THRESHOLDS[mode]

    n_frames = len(samples)
    num_residues = samples.n_residues
    is_physical = np.ones(n_frames, dtype=bool)
    reasons: list[str | None] = [None] * n_frames

    non_finite = stats["non_finite_atoms"] > 0
    for idx in np.flatnonzero(non_finite):
        reasons[idx] = "non_finite_atoms"
    is_physical &= ~non_finite

    for key, threshold in thresholds.get_thresholds().items():
        values = stats[key] / num_residues if thresholds.per_residue else stats[key]
        failed = values > threshold
        for idx in np.flatnonzero(failed & is_physical):
            reasons[idx] = key
        is_physical &= ~failed

    logger.info(
        f"{int(is_physical.sum())}/{n_frames} frames pass protein physicality checks "
        "(non-protein chains are not checked)."
    )
    return is_physical, stats, reasons


def remove_unphysical_frames(
    trajectory: mdtraj.Trajectory,
    mode: PhysicalityFilterMode | str,
    expected_extra_bonds: Collection[tuple[int, int]] = (),
) -> mdtraj.Trajectory:
    """Return frames passing protein physicality checks.

    Only multi-residue chains containing a standard amino acid are checked.
    Nonstandard residues in those chains participate in generic geometry checks;
    complete N-CA-C-O backbones also participate in peptide-bond checks.
    Ignored chains remain in the returned frames.

    Pass ``expected_extra_bonds`` as pairs of atom indices in the input
    trajectory. Bonds to ignored chains are omitted from the protein-only checks.
    A declared bond is expected in the topology but is not created by this function.
    """
    mode = normalize_physicality_mode(mode)
    if mode is PhysicalityFilterMode.NO_FILTER:
        return trajectory

    protein_indices = get_protein_atom_indices(trajectory.topology)
    if len(protein_indices) == 0:
        raise ValueError("Physicality filtering requires at least one protein atom.")
    protein_traj = trajectory.atom_slice(protein_indices)
    is_physical, _stats, _reasons = check_allatom_physicality(
        protein_traj,
        mode=mode,
        expected_extra_bonds=_select_expected_extra_bonds(
            expected_extra_bonds, protein_indices, trajectory.n_atoms
        ),
    )

    filtered = trajectory[is_physical]
    if len(filtered) == 0:
        raise RuntimeError(
            f"All {len(trajectory)} samples were rejected by physicality checks. "
            "Try increasing num_samples or num_steps."
        )
    return filtered
