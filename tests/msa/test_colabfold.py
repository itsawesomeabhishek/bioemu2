import tarfile
from collections.abc import Callable
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pytest

from bioemu2.msa import colabfold as colabfold_module
from bioemu2.msa.colabfold import cached_colabfold_msa, run_colabfold_msa


@pytest.fixture
def colabfold_response(monkeypatch: pytest.MonkeyPatch) -> Callable[[dict[str, str]], None]:
    def set_response(files: dict[str, str]) -> None:
        archive = BytesIO()
        with tarfile.open(fileobj=archive, mode="w:gz") as tar:
            for name, text in files.items():
                content = text.encode()
                member = tarfile.TarInfo(name)
                member.size = len(content)
                tar.addfile(member, BytesIO(content))
        submission = Mock()
        submission.json.return_value = {"status": "COMPLETE", "id": "test-job"}
        download = Mock(content=archive.getvalue())
        monkeypatch.setattr(
            colabfold_module.requests,
            "request",
            Mock(side_effect=[submission, download]),
        )

    return set_response


@pytest.mark.parametrize("use_pairing", [False, True])
@pytest.mark.parametrize("use_env", [False, True])
def test_run_colabfold_msa_orders_blocks_by_query_id(
    colabfold_response: Callable[[dict[str, str]], None],
    use_pairing: bool,
    use_env: bool,
) -> None:
    blocks = [">101\nGYD\n>hit_a\nG-D", ">102\nAAA\n>hit_b\nA-A", ">103\nGYD\n>hit_c\nGY-"]
    filename = "pair.a3m" if use_pairing else "uniref.a3m"
    colabfold_response({filename: "\x00".join([blocks[1], blocks[2], blocks[0]]) + "\x00\n"})

    warning = (
        pytest.warns(UserWarning, match="paired environmental mode")
        if use_pairing and use_env
        else nullcontext()
    )
    with warning:
        result = run_colabfold_msa(["GYD", "AAA", "GYD"], use_pairing=use_pairing, use_env=use_env)

    assert result == blocks


def test_cached_colabfold_msa_merges_environment_by_query_id(
    colabfold_response: Callable[[dict[str, str]], None],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BIOEMU2_MSA_CACHE", str(tmp_path))
    uniref = [">101\nGYD\n>uniref_a\nG-D", ">102\nAAA\n>uniref_b\nA-A"]
    environment = [">101\nGYD\n>env_a\nGY-", ">102\nAAA\n>env_b\nAA-"]
    colabfold_response(
        {
            "uniref.a3m": "\x00".join(reversed(uniref)) + "\x00",
            "bfd.mgnify30.metaeuk30.smag30.a3m": "\x00".join(environment) + "\x00",
        }
    )

    result = cached_colabfold_msa(["GYD", "AAA"], use_env=True)

    assert result == [
        "\n".join((uniref[0], environment[0])),
        "\n".join((uniref[1], environment[1])),
    ]
    assert cached_colabfold_msa(["GYD", "AAA"], use_env=True) == result


@pytest.mark.parametrize("sequences", [["GYD"], ["GYD", "AAA", "CCC"]])
def test_run_colabfold_msa_without_environment_file(
    colabfold_response: Callable[[dict[str, str]], None],
    sequences: list[str],
) -> None:
    blocks = [f">{query_id}\n{seq}" for query_id, seq in enumerate(sequences, start=101)]
    colabfold_response({"uniref.a3m": "\x00".join(blocks) + "\x00"})

    assert run_colabfold_msa(sequences, use_env=True) == blocks


def test_run_colabfold_msa_ignores_unrequested_environment(
    colabfold_response: Callable[[dict[str, str]], None],
) -> None:
    colabfold_response(
        {
            "uniref.a3m": ">101\nGYD\n>uniref\nG-D\x00",
            "bfd.mgnify30.metaeuk30.smag30.a3m": ">101\nGYD\n>env\nGY-\x00",
        }
    )

    assert run_colabfold_msa(["GYD"], use_env=False) == [">101\nGYD\n>uniref\nG-D"]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (">101\nGYD\x00", "missing query IDs"),
        (">101\nGYD\x00>101\nGYD\x00>102\nAAA", "duplicate query ID"),
        (">101\nGYD\x00>999\nAAA", "unexpected query ID"),
        ("not-a-header\nGYD\x00>102\nAAA", "invalid query header"),
        (">query\nGYD\x00>102\nAAA", "invalid query header"),
    ],
)
def test_run_colabfold_msa_rejects_invalid_query_ids(
    colabfold_response: Callable[[dict[str, str]], None],
    text: str,
    message: str,
) -> None:
    colabfold_response({"uniref.a3m": text})

    with pytest.raises(RuntimeError, match=message):
        run_colabfold_msa(["GYD", "AAA"], use_env=False)


def test_run_colabfold_msa_rejects_incomplete_environment(
    colabfold_response: Callable[[dict[str, str]], None],
) -> None:
    colabfold_response(
        {
            "uniref.a3m": ">101\nGYD\x00>102\nAAA\x00",
            "bfd.mgnify30.metaeuk30.smag30.a3m": ">101\nGYD\x00",
        }
    )

    with pytest.raises(RuntimeError, match=r"missing query IDs \[102\]"):
        run_colabfold_msa(["GYD", "AAA"], use_env=True)
