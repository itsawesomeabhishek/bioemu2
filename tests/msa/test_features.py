from contextlib import nullcontext
from unittest.mock import Mock

import numpy as np
import pytest

from bioemu2 import get_msa
from bioemu2.data_interface import DNA, RNA, Ligand, Protein, System
from bioemu2.data_interface.const import MolType, Token
from bioemu2.msa import colabfold as colabfold_module
from bioemu2.msa.colabfold import MAX_SEQS, run_colabfold_msa, tokenize_msa
from bioemu2.structure import AllAtomStructure


@pytest.mark.parametrize(
    ("use_pairing", "use_env", "endpoint", "mode"),
    [
        (False, True, "ticket/msa", "env"),
        (True, False, "ticket/pair", "pairgreedy"),
        (True, True, "ticket/pair", "pairgreedy-env"),
    ],
)
def test_run_colabfold_msa_search_mode(
    monkeypatch: pytest.MonkeyPatch,
    use_pairing: bool,
    use_env: bool,
    endpoint: str,
    mode: str,
) -> None:
    response = Mock()
    response.json.return_value = {"status": "ERROR"}
    request = Mock(return_value=response)
    monkeypatch.setattr(colabfold_module.requests, "request", request)

    warning = (
        pytest.warns(UserWarning, match="paired environmental mode")
        if use_pairing and use_env
        else nullcontext()
    )
    with warning, pytest.raises(RuntimeError, match="submission failed"):
        run_colabfold_msa(["GYD"], use_pairing=use_pairing, use_env=use_env)

    assert request.call_args.args[:2] == ("POST", f"{colabfold_module.COLABFOLD_HOST}/{endpoint}")
    assert request.call_args.kwargs["data"]["mode"] == mode


@pytest.mark.slow
@pytest.mark.serial
def test_run_colabfold_msa_adds_environmental_rows() -> None:
    sequences = [
        "AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR",
        "KKAVINGEQIRSISDLHQTLKKELALPEYYGENLDALWDCLTGWVEYPLVLEWRQFEQSKQLTENGAESVLQVFREAKAEGCDITIILS",
    ]

    uniref_a3ms = run_colabfold_msa(sequences, use_env=False)
    environmental_a3ms = run_colabfold_msa(sequences, use_env=True)

    assert len(uniref_a3ms) == len(environmental_a3ms) == len(sequences)
    for query_id, sequence, uniref_a3m, environmental_a3m in zip(
        range(101, 103),
        sequences,
        uniref_a3ms,
        environmental_a3ms,
        strict=True,
    ):
        assert uniref_a3m.splitlines()[:2] == [f">{query_id}", sequence]
        assert environmental_a3m.splitlines()[:2] == [f">{query_id}", sequence]
        assert environmental_a3m.count("\n>") > uniref_a3m.count("\n>")


@pytest.mark.slow
@pytest.mark.serial
def test_system_to_canonical_msa_to_tokenization() -> None:
    barnase = (
        "AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTW"
        "READINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR"
    )
    barstar = (
        "KKAVINGEQIRSISDLHQTLKKELALPEYYGENLDALWDCLTGWVEYPLVLEWRQFEQSKQLTENGAESVLQVFREAKAEGCDITIILS"
    )
    system = System(
        entities=[
            Protein(sequence=barnase, asym_ids=["A"]),
            Protein(sequence=barstar, asym_ids=["B"]),
        ]
    )
    structure = AllAtomStructure.from_system(system)
    canonical_msa = get_msa(
        system,
        source="colabfold",
        pairing_strategy="greedy",
    )
    assert canonical_msa.paired_a3m is not None
    assert all(a3m.count("\n>") > MAX_SEQS for a3m in canonical_msa.unpaired_a3m.values())

    numeric_chain_ids = sorted(np.unique(structure.asym_id.numpy()).tolist())
    chain_ids = dict(zip(canonical_msa.chain_ids, numeric_chain_ids, strict=True))
    msa, deletion, paired = tokenize_msa(
        structure,
        unpaired_a3ms={
            chain_ids[chain_id]: canonical_msa.a3m(chain_id) for chain_id in canonical_msa.chain_ids
        },
        paired_a3ms={
            chain_ids[chain_id]: canonical_msa.a3m(chain_id, paired=True)
            for chain_id in canonical_msa.chain_ids
        },
    )

    expected_query = [
        Token.from_single_letter(residue, MolType.PROTEIN).value
        for sequence in (barnase, barstar)
        for residue in sequence
    ]
    assert msa.shape == deletion.shape == paired.shape
    assert msa.shape == (structure.num_tokens, MAX_SEQS)
    np.testing.assert_array_equal(msa[:, 0], expected_query)
    assert paired[:, 0].all()


def test_tokenize_msa_builds_unpaired_heteromer_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = System(
        entities=[
            Protein(sequence="GYD", asym_ids=["A"]),
            Protein(sequence="AAA", asym_ids=["B"]),
        ]
    )
    structure = AllAtomStructure.from_system(system)
    monkeypatch.setattr(
        colabfold_module,
        "run_colabfold_msa",
        lambda *_args, **_kwargs: pytest.fail("tokenize_msa must not fetch alignments"),
    )

    msa, _, paired = tokenize_msa(
        structure,
        unpaired_a3ms={
            0: ">query\nGYD\n>a1\nG-D\n>a2\nGYD\n",
            1: ">query\nAAA\n>b1\nA-A\n",
        },
    )

    assert msa.shape == (6, 2)
    assert paired[:, 0].all()
    assert not paired[:, 1].any()


def test_tokenize_msa_validates_query_sequence() -> None:
    system = System(entities=[Protein(sequence="GYD", asym_ids=["A"])])
    structure = AllAtomStructure.from_system(system)

    with pytest.raises(ValueError, match="does not match"):
        tokenize_msa(structure, unpaired_a3ms={0: ">query\nAAA\n"})


@pytest.mark.parametrize(
    "system",
    [
        System(entities=[DNA(sequence="ACGT", asym_ids=["D", "E"])]),
        System(entities=[RNA(sequence="ACGU", asym_ids=["R"])]),
        System(
            entities=[
                RNA(sequence="ACGU", asym_ids=["R"]),
                Ligand(smiles="CCO", asym_ids=["L"]),
            ]
        ),
    ],
    ids=["dna", "rna", "rna-ligand"],
)
@pytest.mark.parametrize("paired_a3ms", [None, {}])
def test_tokenize_msa_without_protein(system: System, paired_a3ms: dict[int, str] | None) -> None:
    structure = AllAtomStructure.from_system(system)

    msa, deletion, paired = tokenize_msa(structure, unpaired_a3ms={}, paired_a3ms=paired_a3ms)

    assert msa.shape == deletion.shape == paired.shape == (structure.num_tokens, 1)
    assert msa.dtype == deletion.dtype == paired.dtype == np.int64
    expected_query = structure.downsample_to_tokens(structure.residue_type).numpy()
    np.testing.assert_array_equal(msa[:, 0], expected_query)
    np.testing.assert_array_equal(deletion, 0)
    np.testing.assert_array_equal(paired, 1)


def test_tokenize_msa_combines_separate_paired_and_unpaired_a3ms() -> None:
    system = System(
        entities=[
            Protein(sequence="GY", asym_ids=["A"]),
            Protein(sequence="AA", asym_ids=["B"]),
        ]
    )
    structure = AllAtomStructure.from_system(system)

    msa, _, paired = tokenize_msa(
        structure,
        unpaired_a3ms={
            0: ">query\nGY\n",
            1: ">query\nAA\n",
        },
        paired_a3ms={
            0: ">query\nGY\n>pair\nG-\n",
            1: ">query\nAA\n>pair\nA-\n",
        },
    )

    assert msa.shape == (4, 2)
    assert paired.all()
