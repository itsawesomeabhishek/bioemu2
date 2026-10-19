from bioemu2.data_interface import System
from bioemu2.structure import AllAtomStructure
from bioemu2.structure.builder import repr_by_sequence_and_name_encoding


def test_repr_by_sequence_and_name_encoding_protein(test_structure: AllAtomStructure) -> None:
    rendered = repr_by_sequence_and_name_encoding(test_structure)
    assert (
        rendered
        == "(sequence=YDPE, name_encoding=N:CA:C:O:CB:CG:CD1:CD2:CE1:CE2:CZ:OH:N:CA:C:O:CB:CG:OD1:OD2:N:CA:C:O:CB:CG:CD:N:CA:C:O:CB:CG:CD:OE1:OE2)"
    )


def test_repr_by_sequence_and_name_encoding_ligand() -> None:
    ligand_structure = AllAtomStructure.from_system(
        System.from_description(ligand_ccd_codes=["ATP"])
    )
    rendered = repr_by_sequence_and_name_encoding(ligand_structure)
    assert (
        rendered
        == "(sequence=X, name_encoding=PG:O1G:O2G:O3G:PB:O1B:O2B:O3B:PA:O1A:O2A:O3A:O5':C5':C4':O4':C3':O3':C2':O2':C1':N9:C8:N7:C5:C6:N6:N1:C2:N3:C4)"
    )
