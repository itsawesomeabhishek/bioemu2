# Portions adapted from Boltz. See NOTICES.md and licenses/LICENSE.BOLTZ-MIT.

"""Molecular-system constants shared across bioemu2.

These include the token / residue / bond schema definitions used by
``bioemu2.data_interface`` together with reference atom names, residue
substitutions, and bond-type conversions.
"""

from enum import Enum
from typing import Final

from rdkit.Chem.rdchem import BondType as RDKitBondType


class MolType(Enum):
    PROTEIN = 0
    DNA = 1
    RNA = 2
    NONPOLYMER = 3


class Token(Enum):
    """Main token class"""

    PAD = 0  # padding
    GAP = 1  # gap ('-')
    ALA = 2
    ARG = 3
    ASN = 4
    ASP = 5
    CYS = 6
    GLN = 7
    GLU = 8
    GLY = 9
    HIS = 10
    ILE = 11
    LEU = 12
    LYS = 13
    MET = 14
    PHE = 15
    PRO = 16
    SER = 17
    THR = 18
    TRP = 19
    TYR = 20
    VAL = 21
    UNK_PROT = 22  # unknown protein token
    RNA_A = 23
    RNA_G = 24
    RNA_C = 25
    RNA_U = 26
    UNK_RNA = 27  # unknown rna token
    DNA_A = 28
    DNA_G = 29
    DNA_C = 30
    DNA_T = 31
    UNK_DNA = 32  # unknown dna token
    # ACE = 33  # cap
    # NME = 34  # cap
    # NHH = 35  # cap
    # NHE = 36  # cap
    # SEC = 37  # Selenocysteine, 21st proteinogenic amino acid.
    # PYL = 38  # Pyrrolysine, 22nd proteinogenic amino acid. Not found in humans.
    # NLE = 39  # Norleucine, Found in DEShaw WW_DOMAIN, isomer of Leucine found in some bacterial strains.
    # MSE = 40  # Selenomethionine, Found in FoldSwitch dataset. https://www.ebi.ac.uk/pdbe/entry/pdb/1ema/modified/MSE

    @staticmethod
    def from_single_letter(l: str, moltype: MolType) -> "Token":
        if moltype == MolType.PROTEIN:
            return PROT_LETTER_TO_TOKEN[l]
        elif moltype == MolType.DNA:
            return DNA_LETTER_TO_TOKEN[l]
        elif moltype == MolType.RNA:
            return RNA_LETTER_TO_TOKEN[l]
        else:
            raise ValueError(f"moltype {moltype} not recognised")

    @property
    def singleletter(self) -> str:
        if self in PROT_TOKEN_TO_LETTER:
            return PROT_TOKEN_TO_LETTER[self]
        elif self in RNA_TOKEN_TO_LETTER:
            return RNA_TOKEN_TO_LETTER[self]
        elif self in DNA_TOKEN_TO_LETTER:
            return DNA_TOKEN_TO_LETTER[self]
        else:
            raise ValueError("Cannot find single letter")

    @property
    def threeletter(self) -> str:
        assert self in PROT_TOKEN_TO_LETTER, "Three-letter not supported for non-protein tokens"
        return self.name

    @staticmethod
    def from_mdtraj_residue_name(residue_name: str) -> "Token":
        """
        Create a token based on residue names extracted from an `mdtraj` topology. Takes care of
        protonated residue conversions and falls back to `UNK_PROT` for unknown residues (ligands,
        PTMs, etc).
        """

        # First check if it is just a protonated amino acid.
        if residue_name in PROTONATED_AA_SUBSTITUTIONS:
            return PROTONATED_AA_SUBSTITUTIONS[residue_name]
        # Check if residue is RNA (single letter code) or DNA (leading 'D' + single letter).
        elif residue_name in RNA_LETTER_TO_TOKEN:
            return RNA_LETTER_TO_TOKEN[residue_name]
        elif residue_name.lstrip("D") in DNA_LETTER_TO_TOKEN:
            return DNA_LETTER_TO_TOKEN[residue_name.lstrip("D")]

        # Otherwise try if it is defined in the Tokens and then return the unknowable.
        try:
            return Token[residue_name]
        except KeyError:
            return Token.UNK_PROT


MOLTYPE_TO_UNKTOKEN: dict[MolType, Token] = {
    MolType.PROTEIN: Token.UNK_PROT,
    MolType.DNA: Token.UNK_DNA,
    MolType.RNA: Token.UNK_RNA,
}

PROT_LETTER_TO_TOKEN: dict[str, Token] = {
    "A": Token.ALA,
    "R": Token.ARG,
    "N": Token.ASN,
    "D": Token.ASP,
    "C": Token.CYS,
    "E": Token.GLU,
    "Q": Token.GLN,
    "G": Token.GLY,
    "H": Token.HIS,
    "I": Token.ILE,
    "L": Token.LEU,
    "K": Token.LYS,
    "M": Token.MET,
    "F": Token.PHE,
    "P": Token.PRO,
    "S": Token.SER,
    "T": Token.THR,
    "W": Token.TRP,
    "Y": Token.TYR,
    "V": Token.VAL,
    "X": Token.UNK_PROT,
    "J": Token.UNK_PROT,
    "B": Token.UNK_PROT,
    "Z": Token.UNK_PROT,
    "O": Token.UNK_PROT,
    "U": Token.UNK_PROT,
    "-": Token.GAP,
}

PROT_TOKEN_TO_LETTER = {v: k for k, v in PROT_LETTER_TO_TOKEN.items()}
PROT_TOKEN_TO_LETTER[Token.UNK_PROT] = "X"

RNA_LETTER_TO_TOKEN = {
    "A": Token.RNA_A,
    "G": Token.RNA_G,
    "C": Token.RNA_C,
    "U": Token.RNA_U,
    "N": Token.UNK_RNA,
}

RNA_TOKEN_TO_LETTER = {v: k for k, v in RNA_LETTER_TO_TOKEN.items()}

DNA_LETTER_TO_TOKEN = {
    "A": Token.DNA_A,
    "G": Token.DNA_G,
    "C": Token.DNA_C,
    "T": Token.DNA_T,
    "N": Token.UNK_DNA,
}

DNA_TOKEN_TO_LETTER = {v: k for k, v in DNA_LETTER_TO_TOKEN.items()}

MOLTYPE_TO_UNKNOWN_CCD: dict[MolType, str] = {
    MolType.PROTEIN: "UNK",
    MolType.DNA: "DN",
    MolType.RNA: "N",
}


def is_missing_residue_identity(polymer_type: MolType, ccd_code: str | None) -> bool:
    """Return whether a CCD code marks an unknown standard polymer residue."""
    return MOLTYPE_TO_UNKNOWN_CCD[polymer_type] == ccd_code


PROTONATED_AA_SUBSTITUTIONS: dict[str, Token] = {
    "ASH": Token.ASP,
    "CYX": Token.CYS,
    "CYM": Token.CYS,
    "GLH": Token.GLU,
    "HID": Token.HIS,
    "HIE": Token.HIS,
    "HIP": Token.HIS,
    "HIN": Token.HIS,
    "LYN": Token.LYS,
}

REF_ATOMS: dict[Token, list[str]] = {
    Token.PAD: [],
    Token.UNK_PROT: ["N", "CA", "C", "O", "CB"],
    Token.GAP: [],
    Token.ALA: ["N", "CA", "C", "O", "CB"],
    Token.ARG: ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    Token.ASN: ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
    Token.ASP: ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
    Token.CYS: ["N", "CA", "C", "O", "CB", "SG"],
    Token.GLN: ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
    Token.GLU: ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
    Token.GLY: ["N", "CA", "C", "O"],
    Token.HIS: ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    Token.ILE: ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
    Token.LEU: ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
    Token.LYS: ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
    Token.MET: ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    Token.PHE: ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    Token.PRO: ["N", "CA", "C", "O", "CB", "CG", "CD"],
    Token.SER: ["N", "CA", "C", "O", "CB", "OG"],
    Token.THR: ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
    Token.TRP: [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "NE1",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
    ],
    Token.TYR: [
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
    ],
    Token.VAL: ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
    Token.RNA_A: [
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
    ],
    Token.RNA_G: [
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
    ],
    Token.RNA_C: [
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
    ],
    Token.RNA_U: [
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
        "O4",
        "C5",
        "C6",
    ],
    Token.UNK_RNA: [
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
    ],
    Token.DNA_A: [
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
    ],
    Token.DNA_G: [
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
    ],
    Token.DNA_C: [
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
        "C1'",
        "N1",
        "C2",
        "O2",
        "N3",
        "C4",
        "N4",
        "C5",
        "C6",
    ],
    Token.DNA_T: [
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
        "C1'",
        "N1",
        "C2",
        "O2",
        "N3",
        "C4",
        "O4",
        "C5",
        "C7",
        "C6",
    ],
    Token.UNK_DNA: [
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
        "C1'",
    ],
    # Token.ACE: ["C", "O", "CH3"],
    # Token.NME: ["N", "C"],
    # Token.NHH: ["N"],
    # Token.NHE: ["N"],
    # Token.SEC: ["N", "CA", "CB", "SE", "C", "O"],
    # Token.PYL: [
    #     "CB2",
    #     "CG2",
    #     "CD2",
    #     "CE2",
    #     "N2",
    #     "CA2",
    #     "C2",
    #     "O2",
    #     "NZ",
    #     "CE",
    #     "CD",
    #     "CG",
    #     "CB",
    #     "CA",
    #     "C",
    #     "O",
    #     "N",
    # ],
    # Token.NLE: ["N", "CA", "C", "O", "CB", "CG", "CD", "CE"],
    # Token.MSE: ["N", "CA", "C", "O", "CB", "CG", "SE", "CE"],
}


def token_to_disto_atom_name(token: Token) -> str:
    rna_disto_atoms = {
        Token.RNA_A: "C4",
        Token.RNA_G: "C4",
        Token.RNA_C: "C2",
        Token.RNA_U: "C2",
        Token.UNK_RNA: "C1'",
    }

    dna_disto_atoms = {
        Token.DNA_A: "C4",
        Token.DNA_G: "C4",
        Token.DNA_C: "C2",
        Token.DNA_T: "C2",
        Token.UNK_DNA: "C1'",
    }

    if token in PROT_TOKEN_TO_LETTER:
        if token == Token.GLY:
            return "CA"
        else:
            return "CB"
    elif token in RNA_TOKEN_TO_LETTER:
        return rna_disto_atoms[token]
    elif token in DNA_TOKEN_TO_LETTER:
        return dna_disto_atoms[token]
    else:
        raise ValueError("Token has undefined disto atom")


class BondType(Enum):
    OTHER_OR_UNK = 0
    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3
    AROMATIC = 4


NUM_TOKENS = len(Token)

PROT_BACKBONE_ATOMS: Final[list[str]] = ["N", "CA", "C", "O"]


# Typical capping atoms used in residues. Proteins use `OXT` for the second oxygen in the carboxylic
# acid group which gets removed upon amide condensation and DNA/RNA uses `OP3` for the oxygen
# capping the phosphate group.
CAPPING_ATOMS: list[str] = ["OXT", "OP3"]

# a mapping from non-standard residue names to their corresponding standard residue names and atom
# name mappings, which is used to handle common substitutions in PDB structures, e.g. MSE
# (selenomethionine) is often used as a substitute for MET (methionine) in X-ray crystallography
# structures, where the SE atom in MSE corresponds to the SD atom in MET.
TYPICAL_SUBSTITUTION_ATOMS: dict[str, tuple[str, dict[str, str]]] = {
    "MSE": ("MET", {"SE": "SD"}),
}


def get_bondtype_from_rdkit_bondtype(bondtype: RDKitBondType) -> BondType:
    try:
        return BondType[str(bondtype)]
    except (ValueError, KeyError):
        return BondType.OTHER_OR_UNK


def get_rdkit_bondtype_from_bondtype(bondtype: BondType) -> RDKitBondType:
    """Convert BondType to RDKit BondType."""
    if bondtype == BondType.SINGLE:
        return RDKitBondType.SINGLE
    elif bondtype == BondType.DOUBLE:
        return RDKitBondType.DOUBLE
    elif bondtype == BondType.TRIPLE:
        return RDKitBondType.TRIPLE
    elif bondtype == BondType.AROMATIC:
        return RDKitBondType.AROMATIC
    else:
        return RDKitBondType.UNSPECIFIED


def get_bondtype_from_pdbx_bondtype(bondtype: str) -> BondType:
    """Attempts to retrieve a BondType from a PDBx `_struct_conn.pdbx_value_order` value."""
    ref_bonds = {
        "SING": BondType.SINGLE,
        "DOUB": BondType.DOUBLE,
        "TRIP": BondType.TRIPLE,
    }
    return ref_bonds.get(bondtype, BondType.OTHER_OR_UNK)
