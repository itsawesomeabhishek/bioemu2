import json
from pathlib import Path

import numpy as np
import pytest

from bioemu2 import MSA, get_msa, load_a3m_msa
from bioemu2.data_interface import RNA, Protein, System
from bioemu2.msa import colabfold as colabfold_module


def test_msa_is_immutable_and_exposes_raw_a3m() -> None:
    source = {"A": ">query\nGYD\n"}
    msa = MSA(source)
    source["A"] = ">changed\nAAA\n"

    assert msa.a3m() == ">query\nGYD\n"
    with pytest.raises(TypeError):
        msa.unpaired_a3m["A"] = ">changed\nAAA\n"  # type: ignore[index]
    with pytest.raises(ValueError, match="Paired A3M is unavailable"):
        msa.a3m(paired=True)
    with pytest.raises(KeyError, match="Unknown protein chain ID"):
        msa.a3m("B")


def test_load_a3m_msa_parses_and_subsets_rows(tmp_path: Path) -> None:
    system = System(entities=[Protein(sequence="GYD", asym_ids=["A"])])
    path = tmp_path / "a.a3m"
    text = "# metadata\n>query full header\nG.YdD\n>first hit\nG-\nYD\n>second hit\nGY-\n"
    path.write_text(text)

    msa = load_a3m_msa(system, {"A": path})
    subset = msa[np.array([0, 2, 1, 2])]

    assert msa.unpaired_a3m == {"A": text}
    assert msa.paired_a3m is None
    assert msa.chain_ids == ["A"]
    assert msa.sequences == ["G.YdD", "G-YD", "GY-"]
    assert len(msa) == 3
    assert repr(msa) == (
        "MSA(chain_ids=['A'], paired=False, unpaired_shapes={'A': (3, 3)}, paired_shapes=None)"
    )
    assert subset.unpaired_a3m == {
        "A": (">query full header\nG.YdD\n>second hit\nGY-\n>first hit\nG-YD\n>second hit\nGY-\n")
    }
    assert subset.sequences == ["G.YdD", "GY-", "G-YD", "GY-"]


def test_load_a3m_msa_handles_multichain_systems(tmp_path: Path) -> None:
    system = System(
        entities=[
            Protein(sequence="GYD", asym_ids=["A", "B"]),
            Protein(sequence="AAA", asym_ids=["C"]),
        ]
    )
    a = tmp_path / "a.a3m"
    b = tmp_path / "b.a3m"
    c = tmp_path / "c.a3m"
    a.write_text(">query-a\nGYD\n")
    b.write_text(">query-b\nGYD\n>different\nG-D\n")
    c.write_text(">query-c\nAAA\n")

    msa = load_a3m_msa(system, {"A": a, "B": b, "C": c})

    assert msa.unpaired_a3m["A"] != msa.unpaired_a3m["B"]
    assert msa.paired_a3m is None
    assert msa.a3m("B") == b.read_text()
    with pytest.raises(ValueError, match="chain_id is required"):
        msa.a3m()
    with pytest.raises(ValueError, match="single unique protein sequence"):
        _ = msa.sequences
    with pytest.raises(ValueError, match="single unique protein sequence"):
        len(msa)


@pytest.mark.parametrize(
    ("indices", "expected"),
    [
        (slice(1, 3), ">first\nG-D\n>second\nGY-\n"),
        (slice(3, 5), ""),
        (slice(None, None, -1), ">second\nGY-\n>first\nG-D\n>query\nGYD\n"),
    ],
)
def test_msa_slices_allow_inspection_without_query(indices: slice, expected: str) -> None:
    text = ">query\nGYD\n>first\nG-D\n>second\nGY-\n"
    msa = MSA({"A": text})

    assert msa[indices].a3m("A") == expected
    assert msa.a3m() == text


def test_msa_subset_reuses_generator_for_all_alignments() -> None:
    unpaired = {
        "A": ">query-a\nGYD\n>hit-a\nG-D\n>last-a\nGY-\n",
        "B": ">query-b\nAAA\n>hit-b\nA-A\n>last-b\nAA-\n",
    }
    paired = {chain: text.replace("hit-", "paired-") for chain, text in unpaired.items()}
    msa = MSA(unpaired, paired)

    subset = msa[(index for index in [2, 1, 2])]

    assert subset.unpaired_a3m == {
        "A": ">last-a\nGY-\n>hit-a\nG-D\n>last-a\nGY-\n",
        "B": ">last-b\nAA-\n>hit-b\nA-A\n>last-b\nAA-\n",
    }
    assert subset.paired_a3m == {
        "A": ">last-a\nGY-\n>paired-a\nG-D\n>last-a\nGY-\n",
        "B": ">last-b\nAA-\n>paired-b\nA-A\n>last-b\nAA-\n",
    }
    assert msa.unpaired_a3m == unpaired
    assert msa.paired_a3m == paired


def test_homomer_inspection_uses_query_sequence(tmp_path: Path) -> None:
    system = System(entities=[Protein(sequence="GYD", asym_ids=["A", "B"])])
    a = tmp_path / "a.a3m"
    b = tmp_path / "b.a3m"
    a.write_text(">query-a\nGYD\n>hit-a\nG-D\n")
    b.write_text(">query-b\nGYD\n>hit-b\nGY-\n")

    msa = load_a3m_msa(system, {"A": a, "B": b})
    subset = msa[:1]

    assert msa.sequences == ["GYD", "G-D"]
    assert msa.a3m() == a.read_text()
    assert len(msa) == 2
    assert subset.sequences == ["GYD"]


def test_load_a3m_msa_rejects_invalid_inputs(tmp_path: Path) -> None:
    system = System(entities=[Protein(sequence="GYD", asym_ids=["A", "B"])])
    a = tmp_path / "a.a3m"
    b = tmp_path / "b.a3m"
    a.write_text(">query\nGYD\n")
    b.write_text(">query\nGYD\n")

    with pytest.raises(ValueError, match="chains"):
        load_a3m_msa(system, {"A": a})
    a.write_text(">query\nAAA\n")
    with pytest.raises(ValueError, match="does not match"):
        load_a3m_msa(system, {"A": a, "B": a})


def test_get_msa_fetches_paired_and_unpaired_alignments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BIOEMU2_MSA_CACHE", str(tmp_path))
    system = System(
        entities=[
            Protein(sequence="GYD", asym_ids=["A", "B"]),
            Protein(sequence="AAA", asym_ids=["C"]),
        ]
    )
    calls: list[tuple[list[str], bool, str, bool]] = []

    def run(
        sequences: list[str],
        use_pairing: bool = False,
        pairing_strategy: str = "greedy",
        use_env: bool = True,
    ) -> list[str]:
        calls.append((sequences, use_pairing, pairing_strategy, use_env))
        kind = "paired" if use_pairing else "unpaired"
        return [f">query-{kind}\n{sequence}\n" for sequence in sequences]

    monkeypatch.setattr(colabfold_module, "run_colabfold_msa", run)

    msa = get_msa(system, use_env=False, pairing_strategy="complete")

    assert calls == [
        (["GYD", "AAA"], False, "complete", False),
        (["GYD", "AAA"], True, "complete", False),
    ]
    assert msa.unpaired_a3m["A"] == ">query-unpaired\nGYD\n"
    assert msa.paired_a3m == {
        "A": ">query-paired\nGYD\n",
        "B": ">query-paired\nGYD\n",
        "C": ">query-paired\nAAA\n",
    }
    assert msa.a3m("C", paired=True) == ">query-paired\nAAA\n"
    assert repr(msa) == (
        "MSA(chain_ids=['A', 'B', 'C'], paired=True, "
        "unpaired_shapes={'A': (1, 3), 'B': (1, 3), 'C': (1, 3)}, "
        "paired_shapes={'A': (1, 3), 'B': (1, 3), 'C': (1, 3)})"
    )
    monkeypatch.setattr(
        colabfold_module,
        "run_colabfold_msa",
        lambda *_args, **_kwargs: pytest.fail("cache hit must not query ColabFold"),
    )
    assert get_msa(system, use_env=False, pairing_strategy="complete") == msa
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_get_msa_cache_keys_include_sequences_and_search_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BIOEMU2_MSA_CACHE", str(tmp_path))
    # Each call records (sequences, use_pairing, pairing_strategy, use_env).
    calls: list[tuple[tuple[str, ...], bool, str, bool]] = []

    def run(
        sequences: list[str], use_pairing: bool, pairing_strategy: str, use_env: bool
    ) -> list[str]:
        calls.append((tuple(sequences), use_pairing, pairing_strategy, use_env))
        return [f">query\n{sequence}\n" for sequence in sequences]

    monkeypatch.setattr(colabfold_module, "run_colabfold_msa", run)
    first = System(entities=[Protein(sequence="GYD", asym_ids=["A"])])
    second = System(entities=[Protein(sequence="AAA", asym_ids=["B"])])
    heteromer = System(entities=[*first.entities, *second.entities])

    get_msa(first)
    get_msa(second)
    get_msa(first, use_env=False)
    get_msa(heteromer, pairing_strategy="greedy")
    get_msa(heteromer, pairing_strategy="complete")
    get_msa(heteromer, pairing_strategy="complete")
    get_msa(heteromer, use_env=False, pairing_strategy="complete")

    # Switching environmental search only refetches the unpaired alignment.
    assert calls == [
        (("GYD",), False, "greedy", True),
        (("AAA",), False, "greedy", True),
        (("GYD",), False, "greedy", False),
        (("GYD", "AAA"), False, "greedy", True),
        (("GYD", "AAA"), True, "greedy", False),
        (("GYD", "AAA"), False, "complete", True),
        (("GYD", "AAA"), True, "complete", False),
        (("GYD", "AAA"), False, "complete", False),
    ]
    assert len(list(tmp_path.glob("*.json"))) == len(calls)
    requests = [json.loads(path.read_text())["request"] for path in tmp_path.glob("*.json")]
    paired_requests = [request for request in requests if request["use_pairing"]]
    assert len(paired_requests) == 2
    assert all(request["use_env"] is False for request in paired_requests)


@pytest.mark.parametrize(
    "content", ["{", '{"request": {}, "a3ms": []}', "missing", "changed-alignment"]
)
def test_get_msa_rejects_corrupt_cache_without_querying(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, content: str
) -> None:
    monkeypatch.setenv("BIOEMU2_MSA_CACHE", str(tmp_path))
    system = System(entities=[Protein(sequence="GYD", asym_ids=["A"])])
    monkeypatch.setattr(
        colabfold_module,
        "run_colabfold_msa",
        lambda sequences, **_kwargs: [f">query\n{sequence}\n" for sequence in sequences],
    )
    get_msa(system)
    path = next(tmp_path.glob("*.json"))
    if content == "missing":
        payload = json.loads(path.read_text())
        del payload["a3ms"]
        path.write_text(json.dumps(payload))
    elif content == "changed-alignment":
        payload = json.loads(path.read_text())
        payload["a3ms"] = [">query\nGYD\n>new-hit\nGYD\n"]
        path.write_text(json.dumps(payload))
    else:
        path.write_text(content)
    monkeypatch.setattr(
        colabfold_module,
        "run_colabfold_msa",
        lambda *_args, **_kwargs: pytest.fail("corrupt cache must not query ColabFold"),
    )
    with pytest.raises(ValueError, match="Invalid ColabFold MSA"):
        get_msa(system)


def test_get_msa_rejects_wrong_query_without_caching(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BIOEMU2_MSA_CACHE", str(tmp_path))
    monkeypatch.setattr(
        colabfold_module,
        "run_colabfold_msa",
        lambda *_args, **_kwargs: [">query\nAAA\n"],
    )
    system = System(entities=[Protein(sequence="GYD", asym_ids=["A"])])

    with pytest.raises(ValueError, match="query 'AAA' does not match 'GYD'"):
        get_msa(system)

    assert list(tmp_path.glob("*.json")) == []


def test_get_msa_preserves_raw_text_in_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BIOEMU2_MSA_CACHE", str(tmp_path))
    text = "# original metadata\n>query long header\nG\nYD\n>hit\nG-D\n"
    monkeypatch.setattr(colabfold_module, "run_colabfold_msa", lambda *_args, **_kwargs: [text])
    system = System(entities=[Protein(sequence="GYD", asym_ids=["A"])])
    assert get_msa(system).a3m() == text
    monkeypatch.setattr(
        colabfold_module,
        "run_colabfold_msa",
        lambda *_args, **_kwargs: pytest.fail("cache hit must not query ColabFold"),
    )
    assert get_msa(system).a3m() == text


def test_get_msa_reports_incomplete_colabfold_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BIOEMU2_MSA_CACHE", str(tmp_path))
    monkeypatch.setattr(colabfold_module, "run_colabfold_msa", lambda *_args, **_kwargs: [])
    system = System(entities=[Protein(sequence="GYD", asym_ids=["A"])])
    with pytest.raises(RuntimeError, match="returned 0 alignments for 1 sequences"):
        get_msa(system)
    assert not list(tmp_path.glob("*.json"))


def test_msa_apis_validate_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    system = System(entities=[Protein(sequence="GYD", asym_ids=["A"])])
    path = tmp_path / "a.a3m"
    path.write_text(">query\nGYD\n")

    homomer = System(entities=[Protein(sequence="GYD", asym_ids=["A", "B"])])
    assert get_msa(homomer, source=path).unpaired_a3m == {
        "A": ">query\nGYD\n",
        "B": ">query\nGYD\n",
    }

    heteromer = System(
        entities=[
            Protein(sequence="GYD", asym_ids=["A"]),
            Protein(sequence="AAA", asym_ids=["B"]),
        ]
    )
    with pytest.raises(ValueError, match="single MSA path"):
        get_msa(heteromer, source=path)

    monkeypatch.setattr(
        colabfold_module,
        "run_colabfold_msa",
        lambda *_args, **_kwargs: pytest.fail("request should not be made"),
    )
    with pytest.raises(ValueError, match="pairing_strategy"):
        get_msa(system, pairing_strategy="invalid")  # type: ignore[arg-type]

    no_protein = System(entities=[RNA(sequence="AC", asym_ids=["R"])])
    with pytest.raises(ValueError, match="protein"):
        load_a3m_msa(no_protein, {})
