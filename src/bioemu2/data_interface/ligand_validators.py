"""Utilities for handling ligands in biomolecular structures."""

from typing import Annotated, Literal

from biotite.structure.info import all_residues
from pydantic import AfterValidator, StringConstraints
from rdkit import Chem

ALL_CCD_RESIDUES = all_residues()

__all__ = [
    "InChIField",
    "SMILESField",
    "CCDCodeField",
]


def get_chem_validator(str_format: Literal["inchi", "smiles", "ccd_code"]):
    RDKIT_VALIDATORS = {
        "inchi": Chem.MolFromInchi,
        "smiles": Chem.MolFromSmiles,
    }

    def _validator(v: str | None) -> str | None:
        if v is not None:
            if str_format == "ccd_code":
                if v not in ALL_CCD_RESIDUES:
                    raise ValueError(f"CCD code {v} not found in CCD database.")
            else:
                m = RDKIT_VALIDATORS[str_format](v)
                if m is None:
                    raise ValueError(f"Invalid chemical string: {v}")
        return v

    return _validator


InChIField = Annotated[
    str | None,
    StringConstraints(min_length=1),
    AfterValidator(get_chem_validator("inchi")),
]  # e.g. "InChI=1S/H2O/h1H2"


SMILESField = Annotated[
    str | None,
    StringConstraints(min_length=1),
    AfterValidator(get_chem_validator("smiles")),
]  # e.g. "O"


CCDCodeField = Annotated[
    str | None,
    StringConstraints(min_length=1, max_length=5, pattern=r"[A-Z0-9]+"),
    AfterValidator(get_chem_validator("ccd_code")),
]  # e.g. "ATP"
