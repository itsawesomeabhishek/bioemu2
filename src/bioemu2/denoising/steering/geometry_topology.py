"""Build structural connectivity and select reference bounds for geometry steering."""

import logging
import math
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import NamedTuple

import torch
import yaml

from bioemu2.data_interface.const import PROT_TOKEN_TO_LETTER, MolType, Token
from bioemu2.paths import ASSETS_DIR
from bioemu2.structure import AllAtomStructure
from bioemu2.structure.inference_inputs import topology_from_structure

logger = logging.getLogger(__name__)

# Used for non-bonded penalties: exclude bonds, angle terminals and
# 1-4 pairs (torsions can legitimately bring the latter closer than
# the sum of their VDW radii).
NONBONDED_EXCLUSION_GRAPH_DISTANCE = 3

CANONICAL_PROTEIN_TOKENS = tuple(
    token for token in PROT_TOKEN_TO_LETTER if token not in (Token.UNK_PROT, Token.GAP)
)

# Canonical nucleotide tokens map to CCD names in the MON_LIB reference tables.
# Other nucleotide residues remain in the graph and use fallback bounds when unmatched.
NUCLEIC_ACID_TOKEN_TO_CCD: dict[Token, str] = {
    Token.DNA_A: "DA",
    Token.DNA_C: "DC",
    Token.DNA_G: "DG",
    Token.DNA_T: "DT",
    Token.RNA_A: "A",
    Token.RNA_C: "C",
    Token.RNA_G: "G",
    Token.RNA_U: "U",
}
NUCLEIC_ACID_MOL_TYPES: frozenset[int] = frozenset({MolType.DNA.value, MolType.RNA.value})
# D-amino acids are mirror images of their L enantiomers with the same heavy-atom names, so
# they share bond lengths and angles. Bonds and angles cannot encode chirality.
D_AMINO_ACID_TO_L: dict[str, str] = {
    "DAL": "ALA",
    "DAR": "ARG",
    "DAS": "ASP",
    "DCY": "CYS",
    "DGL": "GLU",
    "DGN": "GLN",
    "DHI": "HIS",
    "DIL": "ILE",
    "DLE": "LEU",
    "DLY": "LYS",
    "DPN": "PHE",
    "DPR": "PRO",
    "DSG": "ASN",
    "DSN": "SER",
    "DTH": "THR",
    "DTR": "TRP",
    "DTY": "TYR",
    "DVA": "VAL",
    "MED": "MET",
}
# Residue key under which OST files the inter-nucleotide phosphodiester geometry. Used for
# selecting reference-table entry for bonds and angles connecting neighboring nucleotides.
NUCLEIC_ACID_LINK_KEY = "NA"

BOND_FALLBACK = (1.1, 2.2, 0.03)  # Angstrom. (lower mean, upper mean, sigma)
ANGLE_FALLBACK = (60.0, 180.0, 3.0)  # degrees
# CB-SG-SG' (mean, sigma) in degrees; Engh & Huber (2001). The reference tables have no entry.
DISULFIDE_ANGLE = (104.2, 2.1)
_PEPTIDE_CONTEXTS = ("PEPTIDE", "GLY", "PRO_TRANS", "PRO_CIS")


def _angles_from_bonds(bonds: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """Enumerate two-bond paths as (outer atom i, outer atom k, center atom)."""
    neighbors: defaultdict[int, set[int]] = defaultdict(set)
    for i, j in bonds:
        neighbors[i].add(j)
        neighbors[j].add(i)
    return [
        (i, k, center)
        for center, bonded in sorted(neighbors.items())
        for i, k in combinations(sorted(bonded), 2)
    ]


@dataclass(frozen=True, eq=False)
class BondedGeometryGraph:
    """Bonds, angles and atom labels used to select geometry references.

    All atom indices refer to the unpadded input structure.

    Args:
        atom_labels: Map from atom index to (residue reference name, atom name). D-amino
            acids are labeled with their L enantiomers' names.
        bonds: Unique bonded atom pairs (i, j), with i < j.
        angles: Two-bond paths stored as (outer i, outer k, center), with i < k.
        peptide_edges: Subset of bonds joining backbone C and N atoms in adjacent
            protein residues in the same chain.
        atom_residue: Map from atom index to (chain ID, residue index), using the
            input structure's zero-based IDs and globally numbered residues.
        atom_mol_type: Map from atom index to its integer MolType value.
        ordinary_x: Atom indices classified as ordinary unknown protein residues
            ("X"), which use pooled references from known amino acids.
    """

    atom_labels: dict[int, tuple[str, str]]
    bonds: list[tuple[int, int]]
    angles: list[tuple[int, int, int]]
    peptide_edges: set[tuple[int, int]]
    atom_residue: dict[int, tuple[int, int]]
    atom_mol_type: dict[int, int]
    ordinary_x: set[int]

    def peptide_following_residue(self, i: int, j: int) -> str | None:
        """Return the residue name following a peptide bond, used to select reference values
        for that bond and its associated angles. `i` and `j` are atom indices in the unpadded
        sampling structure.
        """
        if (min(i, j), max(i, j)) not in self.peptide_edges:
            return None
        residue_i, name_i = self.atom_labels[i]
        return residue_i if name_i == "N" else self.atom_labels[j][0]

    def peptide_angle_following_residue(
        self, outer_i: int, center: int, outer_k: int
    ) -> str | None:
        """Return the following residue name for a peptide-link angle.
        Used for selecting the peptide-link angle's reference.
        """
        following = None
        # A polymer-link angle has one junction arm and one arm within the central residue.
        if self.atom_residue[center] == self.atom_residue[outer_k]:
            following = self.peptide_following_residue(outer_i, center)
        if following is None and self.atom_residue[center] == self.atom_residue[outer_i]:
            following = self.peptide_following_residue(center, outer_k)
        return following

    def is_na_link_bond(self, i: int, j: int) -> bool:
        """Check whether an existing bond i-j is an O3'-P bond joining adjacent nucleotide
        residues in the same chain.
        """
        return (
            self.atom_residue[i] != self.atom_residue[j]
            and self.atom_mol_type[i] == self.atom_mol_type[j]
            and self.atom_mol_type[i] in NUCLEIC_ACID_MOL_TYPES
            and {self.atom_labels[i][1], self.atom_labels[j][1]} == {"O3'", "P"}
            and self.atom_residue[i][0] == self.atom_residue[j][0]
            and abs(self.atom_residue[i][1] - self.atom_residue[j][1]) == 1
        )

    def is_disulfide(self, i: int, j: int) -> bool:
        """Check whether an existing bond i-j joins the SG atoms of two protein cysteines."""
        return (
            self.atom_labels[i] == self.atom_labels[j] == ("CYS", "SG")
            and self.atom_mol_type[i] == self.atom_mol_type[j] == MolType.PROTEIN.value
        )

    def is_na_link_angle(self, outer_i: int, center: int, outer_k: int) -> bool:
        """Check whether an existing angle i-center-k spans an O3'-P bond joining adjacent
        nucleotide residues, with the other bond lying within the central atom's residue.
        """
        atoms = (outer_i, center, outer_k)
        # A nucleotide junction likewise needs one O3'-P arm and an intra-residue second arm.
        return len({self.atom_mol_type[a] for a in atoms}) == 1 and any(
            self.atom_residue[center] == self.atom_residue[other]
            and self.is_na_link_bond(outer, center)
            for outer, other in ((outer_i, outer_k), (outer_k, outer_i))
        )

    def nonbonded_exclusions(
        self, max_bonds: int = NONBONDED_EXCLUSION_GRAPH_DISTANCE
    ) -> set[tuple[int, int]]:
        """Return atom pairs (i, j) separated by at most ``max_bonds`` covalent bonds.

        This is used by our VDW overlap potential and local clash potential.
        Intrachain VDW potential excludes these close covalent neighbors from its overlap
        penalty. Local clash potential uses the same set excluding direct bonds, and pairs
        in the same residue or cross chain.
        """
        neighbors: defaultdict[int, set[int]] = defaultdict(set)
        for i, j in self.bonds:
            neighbors[i].add(j)
            neighbors[j].add(i)
        excluded: set[tuple[int, int]] = set()
        for source in neighbors:
            depth = {source: 0}
            queue = deque([source])
            while queue:
                node = queue.popleft()
                if depth[node] < max_bonds:
                    for other in neighbors[node]:
                        if other not in depth:
                            depth[other] = depth[node] + 1
                            queue.append(other)
            excluded.update((source, other) for other in depth if source < other)
        return excluded


def build_bonded_geometry_graph(structure: AllAtomStructure) -> BondedGeometryGraph:
    """Build bonds, angles and residue labels for selecting geometry references.

    Use the bonds from topology_from_structure(): MDTraj's standard bonds with nucleotide
    backbone corrections, peptide links, and supplied covalent bonds in structure.bond_index.
    Derive angles from pairs of bonds sharing an atom; other missing connections in
    unusual residues are not reconstructed.

    Returns:
        A graph indexed by atoms in the unpadded structure, without reference values
        or restraint bounds; get_bond_reference(), get_angle_reference() and
        compute_geometry_bounds() calculate those separately.
    """
    topology = topology_from_structure(structure)

    atom_chain = structure.allatom_asym_id.tolist()
    atom_mol_type = structure.mol_type.tolist()
    atom_residue_idx = structure.residue_idx.tolist()
    atom_residue_type = structure.residue_type.tolist()
    atom_token_idx = structure.token_idx_at_atom.tolist()
    atoms: dict[int, tuple[str, str]] = {}
    atom_residue: dict[int, tuple[int, int]] = {}
    ordinary_x: set[int] = set()
    # Reference lookup needs residue/atom names; link classification also needs chain/residue IDs.
    for residue in topology.residues:
        indices = [atom.index for atom in residue.atoms]
        first = indices[0]
        residue_token = (
            Token(atom_residue_type[first])
            if atom_mol_type[first] == MolType.PROTEIN.value
            else None
        )
        # We want to distinguish unknown "X" residues from modified residues, which are both
        # labeled as UNK_PROT, but modified residues use one token per atom rather than per residue.
        is_x = residue_token == Token.UNK_PROT and len({atom_token_idx[i] for i in indices}) == 1
        # Canonical nucleotides use CCD residue names in the reference tables, and D-amino
        # acids use their L enantiomers' references.
        residue_name = NUCLEIC_ACID_TOKEN_TO_CCD.get(
            Token(atom_residue_type[first]), D_AMINO_ACID_TO_L.get(residue.name, residue.name)
        )
        for atom in residue.atoms:
            atoms[atom.index] = (
                residue_token.name if is_x and residue_token is not None else residue_name,
                atom.name,
            )
            atom_residue[atom.index] = (atom_chain[first], atom_residue_idx[first])
            if is_x:
                ordinary_x.add(atom.index)
    bonds = sorted({(min(a.index, b.index), max(a.index, b.index)) for a, b in topology.bonds})
    peptide_edges = {
        (i, j)
        for i, j in bonds
        if atom_residue[i] != atom_residue[j]
        and atom_residue[i][0] == atom_residue[j][0]
        and atom_mol_type[i] == atom_mol_type[j] == MolType.PROTEIN.value
        and {atoms[i][1], atoms[j][1]} == {"C", "N"}
        and abs(atom_residue[i][1] - atom_residue[j][1]) == 1
    }
    return BondedGeometryGraph(
        atom_labels=atoms,
        bonds=bonds,
        angles=_angles_from_bonds(bonds),
        peptide_edges=peptide_edges,
        atom_residue=atom_residue,
        atom_mol_type=dict(enumerate(atom_mol_type)),
        ordinary_x=ordinary_x,
    )


_StereoTable = dict[str, dict[tuple[str, ...], tuple[float, float]]]


def build_polymer_steric_graph(
    structure: AllAtomStructure,
) -> tuple[BondedGeometryGraph, torch.Tensor]:
    """Build the full covalent graph and select endpoints for polymer steric penalties.

    Score canonical protein/DNA/RNA atoms and ordinary unknown protein residues (X).
    Omit modified residues, unknown nucleotides and ligands from the endpoint mask,
    but retain their bonds when finding short covalent paths between scored atoms.

    Returns:
        The full bonded graph and a boolean mask of shape ``[N]`` for the unpadded
        structure. ``True`` means an atom is eligible as a scored pair endpoint.
    """
    graph = build_bonded_geometry_graph(structure)
    residue_type = structure.residue_type
    protein = structure.mol_type == MolType.PROTEIN.value
    ordinary_x = torch.zeros_like(protein)
    ordinary_x[torch.tensor(sorted(graph.ordinary_x), dtype=torch.long, device=protein.device)] = (
        True
    )
    canonical_protein = torch.isin(
        residue_type, residue_type.new_tensor([t.value for t in CANONICAL_PROTEIN_TOKENS])
    )
    nucleic_acid = torch.isin(
        structure.mol_type, structure.mol_type.new_tensor(tuple(NUCLEIC_ACID_MOL_TYPES))
    )
    canonical_nucleotide = torch.isin(
        residue_type, residue_type.new_tensor([t.value for t in NUCLEIC_ACID_TOKEN_TO_CCD])
    )
    # Unlike bonded restraints, sterics omit ligands, modifications and unknown nucleotides.
    eligible_protein = protein & (canonical_protein | ordinary_x)
    eligible_nucleotide = nucleic_acid & canonical_nucleotide
    eligible = eligible_protein | eligible_nucleotide
    if (protein & (residue_type == Token.UNK_PROT.value) & ~ordinary_x).any():
        logger.warning("Intrachain steric restraints omit modified protein residues.")
    return graph, eligible


def _canonical_atom_order(atoms: tuple[str, ...]) -> tuple[str, ...]:
    """Choose between a bonded path and its reversal, preserving the center of an angle.

    Used for bond pairs and angle triples: reversing (N, CA, C) to (C, CA, N)
    keeps CA at the center, whereas sorting individual atom names need not.
    """
    return min(atoms, atoms[::-1])


class StereoTables(NamedTuple):
    """Reference means and standard deviations used to prepare bond and angle bounds.

    Separate tables cover bonds/angles within residues and links between residues.
    Entries have the form ``{residue_or_link: {atom_names: (mean, sigma)}}``, with
    canonical atom-name tuples in bonded-path order (outer, center, outer for angles).
    Bond values are in angstroms and angle values in degrees.
    """

    bonds: _StereoTable
    angles: _StereoTable
    link_bonds: _StereoTable
    link_angles: _StereoTable


def load_stereo_tables() -> StereoTables:
    """Load reference tables, canonicalizing atom-name keys and converting (mean, sigma) values."""
    data = yaml.safe_load((ASSETS_DIR / "stereo_chemistry.yaml").read_text())
    raw_tables = {
        "bonds": data["monlib"]["bond_data"],
        "angles": data["monlib"]["angle_data"],
        "link_bonds": data["monlib_link"]["bond_data"],
        "link_angles": data["monlib_link"]["angle_data"],
    }
    return StereoTables(
        **{
            name: {
                residue: {
                    _canonical_atom_order(tuple(key.split("_"))): (float(mean), float(sigma))
                    for key, (mean, sigma) in entries.items()
                }
                for residue, entries in table.items()
            }
            for name, table in raw_tables.items()
        }
    )


def _stereo_lookup(
    table: _StereoTable, residue: str, atoms: tuple[str, ...]
) -> tuple[float, float] | None:
    """Return (mean, sigma) for a bonded path in either direction, or None if unmatched.

    Angle atom names must be in (outer, center, outer) order; canonicalization
    preserves the center. Values use angstroms for bonds or degrees for angles.
    """
    residue_data = table.get(residue)
    if residue_data is None:
        return None
    return residue_data.get(_canonical_atom_order(atoms))


def _peptide_link_key(following_residue: str) -> str:
    """The reference values for peptide-link bonds and angles depend on the residue types."""
    if following_residue == "GLY":
        return "GLY"
    if following_residue == "PRO":
        # PRO_CIS is unreachable: OST's cis/trans split is a coordinate-dependent omega test,
        # while these restraints are static. A genuine cis-proline is therefore restrained to
        # trans geometry (the tables differ only in the nitrogen-centred angles).
        return "PRO_TRANS"
    return "PEPTIDE"


def _pooled_reference(
    table: _StereoTable, names: tuple[str, ...], peptide: bool
) -> tuple[float, float, float] | None:
    """Combine matching known-residue references into an envelope for unknown protein X.

    Return the smallest mean, largest mean and largest sigma for the requested atom
    names across canonical residues, or peptide-link contexts when ``peptide`` is True.
    These supply the lower/upper reference means and sigma used to calculate bounds.
    """
    identities = (
        _PEPTIDE_CONTEXTS if peptide else tuple(token.name for token in CANONICAL_PROTEIN_TOKENS)
    )
    entries = [
        reference
        for identity in identities
        if (reference := _stereo_lookup(table, identity, names)) is not None
    ]
    if not entries:
        return None
    return (
        min(mean for mean, _ in entries),
        max(mean for mean, _ in entries),
        max(sigma for _, sigma in entries),
    )


def get_bond_reference(
    topology: BondedGeometryGraph,
    i: int,
    j: int,
    tables: StereoTables,
    pooled: dict[tuple[tuple[str, ...], bool], tuple[float, float, float] | None],
) -> tuple[float, float, float]:
    """Return bond-length reference means and standard deviation, not the final allowed range.

    Args:
        topology: Bonded graph containing the atom labels and residue identities.
        i: First atom index of an existing bond.
        j: Second atom index of that bond.
        tables: Loaded residue and inter-residue reference tables.
        pooled: Cache of unknown-protein bond references for this set of tables.

    Returns:
        (lowest_mean, highest_mean, sigma), all in angstroms. A matching residue,
        peptide, nucleotide or disulfide reference gives (mean, mean, sigma).
        Ordinary unknown protein X uses the smallest mean, largest mean and largest
        sigma across matching known contexts. Unmatched bonds use BOND_FALLBACK.
        Pass this tuple to compute_geometry_bounds() to apply a tolerance.
    """
    names = (topology.atom_labels[i][1], topology.atom_labels[j][1])
    same_residue = topology.atom_residue[i] == topology.atom_residue[j]
    following = topology.peptide_following_residue(i, j)
    na_link = topology.is_na_link_bond(i, j)
    # For unknown protein residue X, use pooled (envelope) reference values for intra-residue
    # and peptide-link bonds.
    if i in topology.ordinary_x or j in topology.ordinary_x:
        # If reference values exist, use pooled envelope; otherwise, use generous fallback bounds.
        if same_residue or following is not None:
            key = (_canonical_atom_order(names), following is not None)
            if key not in pooled:
                table = tables.link_bonds if following is not None else tables.bonds
                pooled[key] = _pooled_reference(table, names, following is not None)
            return pooled[key] or BOND_FALLBACK
        # For extra cross-residue bonds, use the generous fallback bounds.
        return BOND_FALLBACK
    # Look up reference values for peptide and nucleotide junctions in the tables.
    if following is not None:
        reference = _stereo_lookup(tables.link_bonds, _peptide_link_key(following), names)
    elif na_link:
        reference = _stereo_lookup(tables.link_bonds, NUCLEIC_ACID_LINK_KEY, names)
    # Look up reference values for intra-residue bonds.
    elif same_residue:
        reference = _stereo_lookup(tables.bonds, topology.atom_labels[i][0], names)
    # Look up reference values for disulfide bonds.
    elif topology.is_disulfide(i, j):
        reference = _stereo_lookup(tables.link_bonds, "PEPTIDE", names)
    else:
        reference = None
    # If no reference matches, use the generous fallback bounds.
    return (reference[0], reference[0], reference[1]) if reference is not None else BOND_FALLBACK


def get_angle_reference(
    topology: BondedGeometryGraph,
    outer_i: int,
    center: int,
    outer_k: int,
    tables: StereoTables,
    pooled: dict[tuple[tuple[str, ...], bool], tuple[float, float, float] | None],
) -> tuple[float, float, float]:
    """Return bond-angle reference means and standard deviation, not the final allowed range.

    Args:
        topology: Bonded graph containing the atom labels and residue identities.
        outer_i: First outer atom index of an existing angle.
        center: Central atom index, passed between the outer atoms here even though
            the graph stores it last.
        outer_k: Second outer atom index.
        tables: Loaded residue and inter-residue reference tables.
        pooled: Cache of unknown-protein angle references for this set of tables.

    Returns:
        (lowest_mean, highest_mean, sigma), all in degrees. A matching residue,
        peptide, nucleotide or disulfide reference gives (mean, mean, sigma). Ordinary unknown
        protein X uses the smallest mean, largest mean and largest sigma across
        matching known contexts. Unmatched angles use ANGLE_FALLBACK.
        Pass this tuple to compute_geometry_bounds() to apply a tolerance.
    """
    atoms = (outer_i, center, outer_k)
    names = tuple(topology.atom_labels[a][1] for a in atoms)
    same_residue = len({topology.atom_residue[a] for a in atoms}) == 1
    following = topology.peptide_angle_following_residue(outer_i, center, outer_k)
    peptide = following is not None
    na_link = topology.is_na_link_angle(outer_i, center, outer_k)
    # For unknown protein residue X, use pooled (envelope) reference values for intra-residue
    # and peptide-link angles.
    if any(a in topology.ordinary_x for a in atoms):
        # If reference values exist, use pooled envelope; otherwise, use generous fallback bounds.
        if same_residue or peptide:
            key = (_canonical_atom_order(names), peptide)
            if key not in pooled:
                table = tables.link_angles if peptide else tables.angles
                pooled[key] = _pooled_reference(table, names, peptide)
            return pooled[key] or ANGLE_FALLBACK
        # For extra cross-residue angles, use the generous fallback bounds.
        return ANGLE_FALLBACK
    # Look up reference values for peptide and nucleotide junctions in the tables.
    if following is not None:
        reference = _stereo_lookup(tables.link_angles, _peptide_link_key(following), names)
    elif na_link:
        reference = _stereo_lookup(tables.link_angles, NUCLEIC_ACID_LINK_KEY, names)
    # Look up reference values for intra-residue angles.
    elif same_residue:
        reference = _stereo_lookup(tables.angles, topology.atom_labels[center][0], names)
    # Look up reference values for disulfide CB-SG-SG' angles.
    elif _canonical_atom_order(names) == ("CB", "SG", "SG") and any(
        topology.is_disulfide(center, outer) for outer in (outer_i, outer_k)
    ):
        reference = DISULFIDE_ANGLE
    else:
        reference = None
    # If no reference matches, use the generous fallback bounds.
    return (reference[0], reference[0], reference[1]) if reference is not None else ANGLE_FALLBACK


def compute_geometry_bounds(
    reference: tuple[float, float, float], sigma_tolerance: float, *, angle: bool = False
) -> tuple[float, float]:
    """Expand reference means by a multiple of sigma to obtain an allowed range.

    Args:
        reference: (lowest_mean, highest_mean, sigma), in angstroms for bonds or
            degrees for angles. The means are equal for a single known reference;
            pooled references for unknown X and fallback references span a range.
        sigma_tolerance: Dimensionless multiplier of the reference standard deviation,
            not a distance or an angle.
        angle: If True, convert all three reference values from degrees to radians
            before computing the bounds.

    Returns:
        (lower, upper), computed as lowest_mean - sigma_tolerance * sigma and
        highest_mean + sigma_tolerance * sigma using float32 arithmetic. Bond bounds
        are in angstroms with a nonnegative lower bound; angle bounds are in radians,
        with the lower bound clipped at zero and the upper bound clipped at pi.
    """
    lower_mean, upper_mean, sigma = (
        tuple(math.radians(value) for value in reference) if angle else reference
    )
    lower = torch.tensor(lower_mean, dtype=torch.float32)
    upper = torch.tensor(upper_mean, dtype=torch.float32)
    width = torch.tensor(sigma, dtype=torch.float32) * sigma_tolerance
    return max(0.0, (lower - width).item()), min(
        math.pi if angle else math.inf, (upper + width).item()
    )


def pack_bond_constraints(
    constraints: Sequence[tuple[int, int, tuple[float, float]]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tensorize resolved bond indices and bounds in input order."""
    index = torch.tensor(
        [[bond[row] for bond in constraints] for row in range(2)], dtype=torch.long, device=device
    ).reshape(1, 2, len(constraints))
    lower = torch.tensor(
        [[bounds[0] for _, _, bounds in constraints]], dtype=torch.float32, device=device
    )
    upper = torch.tensor(
        [[bounds[1] for _, _, bounds in constraints]], dtype=torch.float32, device=device
    )
    return index, lower, upper


def pack_angle_constraints(
    constraints: Sequence[tuple[int, int, int, tuple[float, float]]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tensorize resolved (outer-A, outer-C, center-B) indices and radian bounds."""
    index = torch.tensor(
        [[angle[row] for angle in constraints] for row in range(3)], dtype=torch.long, device=device
    ).reshape(1, 3, len(constraints))
    lower = torch.tensor(
        [[bounds[0] for _, _, _, bounds in constraints]], dtype=torch.float32, device=device
    )
    upper = torch.tensor(
        [[bounds[1] for _, _, _, bounds in constraints]], dtype=torch.float32, device=device
    )
    return index, lower, upper
