from pathlib import Path

import pytest
import torch

from bioemu2 import MSA, get_msa
from bioemu2.data_interface import Contact, ContactToken, Protein, System
from bioemu2.msa import colabfold as colabfold_module
from bioemu2.structure.preparation import prepare_structure

_SYSTEM = System(entities=[Protein(sequence="GYD", asym_ids=["A"])])
_MISMATCHED_MSA = MSA({"B": ">query\nGYD\n"})


def test_prepare_structure_rejects_msa_chain_mismatch() -> None:
    with pytest.raises(ValueError, match="MSA must contain chains"):
        prepare_structure(_SYSTEM, msa=_MISMATCHED_MSA, mode="equilibrium", contacts=[])


def test_prepare_structure_validates_contacts() -> None:
    bad_contact = Contact(
        token1=ContactToken(asym_id="A", res_id=1),
        token2=ContactToken(asym_id="Z", res_id=2),
    )

    with pytest.raises(ValueError, match="not present in any entity"):
        prepare_structure(_SYSTEM, msa=_MISMATCHED_MSA, mode="equilibrium", contacts=[bad_contact])


def test_prepare_structure_builds_features_from_cached_raw_msa(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BIOEMU2_MSA_CACHE", str(tmp_path))
    monkeypatch.setattr(
        colabfold_module,
        "run_colabfold_msa",
        lambda *_args, **_kwargs: [">query\nGYD\n>hit\nG-D\n"],
    )
    first = prepare_structure(_SYSTEM, msa=get_msa(_SYSTEM), mode="equilibrium", contacts=[])
    monkeypatch.setattr(
        colabfold_module,
        "run_colabfold_msa",
        lambda *_args, **_kwargs: pytest.fail("cache hit must not query ColabFold"),
    )
    second = prepare_structure(_SYSTEM, msa=get_msa(_SYSTEM), mode="equilibrium", contacts=[])

    torch.testing.assert_close(first["msa"], second["msa"])
    assert first["msa"].shape[1] == 2
    assert len(list(tmp_path.iterdir())) == 1
    assert not list(tmp_path.glob("*.npz"))
