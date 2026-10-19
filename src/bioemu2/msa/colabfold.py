"""ColabFold-based MSA computation via the public MMSeqs2 server.

Uses ColabFold's ``ticket/pair`` endpoint for cross-chain paired alignments
and ``ticket/msa`` for independent per-sequence searches.
"""

import hashlib
import json
import logging
import os
import tarfile
import tempfile
import time
import warnings
from collections.abc import Mapping
from io import StringIO
from pathlib import Path

import numpy as np
import numpy.typing as npt
import requests
from Bio import SeqIO
from tenacity import retry, stop_after_attempt, wait_fixed

from bioemu2.data_interface.const import MolType, Token
from bioemu2.msa.arrays import (
    EncodedMSA,
    MSADeletion,
    MSAResidue,
    MSASequence,
    _prepare_msa_arrays,
    make_dummy_msa,
)
from bioemu2.structure import AllAtomStructure

logger = logging.getLogger(__name__)

COLABFOLD_HOST = "https://api.colabfold.com"
_HEADERS = {"User-Agent": "bioemu/dev"}
_TIMEOUT_SUBMIT = 60
_TIMEOUT_DOWNLOAD = 120
_DEFAULT_MAX_RETRIES = 5
_PAIRED_ENV_WARNING = (
    "ColabFold's paired environmental mode does not currently search environmental "
    "databases; use_env=True may change results if the server changes."
)
MAX_SEQS = 4092


def msa_cache_dir() -> Path:
    """Return the directory used for raw ColabFold A3M responses."""
    configured = os.environ.get("BIOEMU2_MSA_CACHE")
    return (
        Path(configured).expanduser()
        if configured is not None
        else Path.home() / ".cache" / "bioemu2" / "colabfold_msa"
    )


def cached_colabfold_msa(
    sequences: list[str],
    *,
    use_env: bool,
    use_pairing: bool = False,
    pairing_strategy: str = "greedy",
) -> list[str]:
    """Read or acquire raw A3M text for an ordered set of unique sequences.

    Uses the search options of ``run_colabfold_msa`` and caches its raw response
    so repeated requests do not contact ColabFold.
    Paired environmental requests warn even when the response is cached.
    Invalid cache entries raise ValueError rather than silently querying again.
    Delete the reported file to request a fresh alignment.
    """
    unique_sequences = list(dict.fromkeys(sequences))
    request = {
        "sequences": unique_sequences,
        "use_pairing": use_pairing,
        "pairing_strategy": pairing_strategy,
        "use_env": use_env,
    }
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode("utf-8")).hexdigest()
    cache_dir = msa_cache_dir()
    path = cache_dir / f"{key}.json"
    cache_hit = path.exists()
    if cache_hit:
        if use_pairing and use_env:
            warnings.warn(_PAIRED_ENV_WARNING, stacklevel=2)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid ColabFold MSA cache entry at {path}: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("request") != request:
            raise ValueError(f"Invalid ColabFold MSA cache entry at {path}: request mismatch.")
        results = payload.get("a3ms")
    else:
        results = run_colabfold_msa(
            unique_sequences,
            use_pairing=use_pairing,
            pairing_strategy=pairing_strategy,
            use_env=use_env,
        )

    if not isinstance(results, list):
        raise ValueError(
            f"Invalid ColabFold MSA response or cache entry at {path}: expected a list."
        )
    if len(results) != len(unique_sequences):
        if not cache_hit:
            raise RuntimeError(
                f"ColabFold returned {len(results)} alignments "
                f"for {len(unique_sequences)} sequences."
            )
        raise ValueError(
            f"Invalid ColabFold MSA cache entry at {path}: "
            f"found {len(results)} alignments for {len(unique_sequences)} sequences."
        )
    digest = hashlib.sha256(json.dumps(results).encode("utf-8")).hexdigest()
    if cache_hit and payload.get("sha256") != digest:
        raise ValueError(f"Invalid ColabFold MSA cache entry at {path}: checksum mismatch.")
    for sequence, a3m in zip(unique_sequences, results, strict=True):
        if not isinstance(a3m, str):
            raise ValueError(
                f"Invalid ColabFold MSA response or cache entry at {path}: invalid A3M."
            )
        try:
            query_record = next(SeqIO.parse(StringIO(a3m), "fasta-pearson"), None)
        except ValueError as exc:
            raise ValueError(
                f"Invalid ColabFold MSA response or cache entry at {path}: {exc}"
            ) from exc
        if query_record is None:
            raise ValueError(f"Invalid ColabFold MSA response or cache entry at {path}: no query.")
        query = "".join(
            char for char in str(query_record.seq) if not char.islower() and char not in {"-", "."}
        ).upper()
        if query != sequence.upper():
            raise ValueError(
                f"Invalid ColabFold MSA response or cache entry at {path}: "
                f"query {query!r} does not match {sequence!r}."
            )

    if not cache_hit:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=cache_dir, suffix=".tmp", delete=False
        ) as file:
            temp_path = Path(file.name)
        try:
            temp_path.write_text(
                json.dumps({"request": request, "a3ms": results, "sha256": digest}),
                encoding="utf-8",
            )
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)
        logger.info("Cached raw ColabFold A3M to %s", path)
    return results


def run_colabfold_msa(
    sequences: list[str],
    *,
    use_env: bool,
    use_pairing: bool = False,
    pairing_strategy: str = "greedy",
    max_retries: int = _DEFAULT_MAX_RETRIES,
    timeout_submit: int = _TIMEOUT_SUBMIT,
    timeout_download: int = _TIMEOUT_DOWNLOAD,
) -> list[str]:
    """Run MMSeqs2 via ColabFold server, return A3M strings per input sequence.

    For pairing mode, returned rows are positionally aligned across chains.

    Args:
        sequences: Protein sequences (one per chain) to search.
        use_pairing: If True, use the ``ticket/pair`` endpoint to produce
            cross-chain paired alignments.  Otherwise use ``ticket/msa`` for
            independent per-sequence searches.
        pairing_strategy: ``"greedy"`` or ``"complete"`` pairing (only relevant
            when ``use_pairing=True``).
        use_env: Include environmental databases (BFD, MGnify, etc.) in the
            search. Pass False for paired searches; True with pairing warns
            because the server currently ignores paired environmental search.
        max_retries: Number of retry attempts for each HTTP request (uses
            exponential back-off via tenacity).
        timeout_submit: Timeout in seconds for job submission and polling
            requests.
        timeout_download: Timeout in seconds for downloading the result
            tarball.

    Returns:
        One A3M-formatted string per input sequence.
    """

    @retry(stop=stop_after_attempt(max_retries), wait=wait_fixed(5), reraise=True)
    def _do_request(method: str, url: str, timeout: int, **kwargs) -> requests.Response:
        resp = requests.request(method, url, timeout=timeout, headers=_HEADERS, **kwargs)
        resp.raise_for_status()
        return resp

    if use_pairing:
        if use_env:
            warnings.warn(_PAIRED_ENV_WARNING, stacklevel=2)
        mode = "pairgreedy" if pairing_strategy == "greedy" else "paircomplete"
        if use_env:
            mode += "-env"
        endpoint = "ticket/pair"
    else:
        mode = "env" if use_env else "all"
        endpoint = "ticket/msa"

    query = "".join(f">{i}\n{seq}\n" for i, seq in enumerate(sequences, start=101))

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        tar_path = tmpdir / "result.tar.gz"

        out = _do_request(
            "POST",
            f"{COLABFOLD_HOST}/{endpoint}",
            timeout=timeout_submit,
            data={"q": query, "mode": mode},
        ).json()
        if out["status"] == "ERROR":
            raise RuntimeError(f"MMSeqs2 submission failed: {out}")

        job_id = out["id"]
        if out["status"] != "COMPLETE":
            logger.info(f"Submitted MMSeqs2 job {job_id}, polling...")
            while True:
                out = _do_request(
                    "GET", f"{COLABFOLD_HOST}/ticket/{job_id}", timeout=timeout_submit
                ).json()
                if out["status"] == "COMPLETE":
                    break
                if out["status"] == "ERROR":
                    raise RuntimeError("MMSeqs2 server returned ERROR")
                time.sleep(5 + int(time.time()) % 5)

        res = _do_request(
            "GET",
            f"{COLABFOLD_HOST}/result/download/{job_id}",
            timeout=timeout_download,
        )
        tar_path.write_bytes(res.content)
        with tarfile.open(tar_path) as tar:
            tar.extractall(tmpdir, filter="data")

        if use_pairing:
            a3m_files = [tmpdir / "pair.a3m"]
        else:
            a3m_files = [tmpdir / "uniref.a3m"]
            env_file = tmpdir / "bfd.mgnify30.metaeuk30.smag30.a3m"
            if use_env and env_file.exists():
                a3m_files.append(env_file)

        query_ids = range(101, 101 + len(sequences))
        a3ms_by_id: dict[int, list[str]] = {query_id: [] for query_id in query_ids}
        for a3m_file in a3m_files:
            blocks_by_id: dict[int, str] = {}
            for block in a3m_file.read_text().split("\x00"):
                block = block.strip()
                if not block:
                    continue
                header = block.splitlines()[0]
                try:
                    if not header.startswith(">"):
                        raise ValueError
                    query_id = int(header[1:].split()[0])
                except (ValueError, IndexError) as exc:
                    raise RuntimeError(
                        f"ColabFold returned an invalid query header {header!r} in {a3m_file.name}."
                    ) from exc
                if query_id not in a3ms_by_id:
                    raise RuntimeError(
                        f"ColabFold returned unexpected query ID {query_id} in {a3m_file.name}."
                    )
                if query_id in blocks_by_id:
                    raise RuntimeError(
                        f"ColabFold returned duplicate query ID {query_id} in {a3m_file.name}."
                    )
                blocks_by_id[query_id] = block
            missing_ids = a3ms_by_id.keys() - blocks_by_id.keys()
            if missing_ids:
                raise RuntimeError(
                    f"ColabFold response {a3m_file.name} is missing query IDs {sorted(missing_ids)}."
                )
            for query_id, block in blocks_by_id.items():
                a3ms_by_id[query_id].append(block)

        return ["\n".join(a3ms_by_id[query_id]) for query_id in query_ids]


def _parse_a3m_sequences(a3m: str) -> list[str]:
    """Extract sequence lines from an A3M string (skip headers)."""
    return [
        line.strip()
        for line in a3m.strip().splitlines()
        if line.strip() and not line.startswith(">") and not line.startswith("#")
    ]


def _build_msa_from_colabfold_seqs(aligned_seqs: list[str]) -> EncodedMSA:
    """Build an EncodedMSA object from aligned A3M sequences (taxonomy=-1)."""
    all_residues: list[int] = []
    all_deletions: list[tuple[int, int]] = []
    sequences_arr: list[tuple] = []
    expected_len: int | None = None

    for seq_idx, seq in enumerate(aligned_seqs):
        residues: list[int] = []
        deletions: list[tuple[int, int]] = []
        count = 0
        res_idx = 0
        for c in seq:
            if c != "-" and c.islower():
                count += 1
                continue
            residues.append(Token.from_single_letter(c, moltype=MolType.PROTEIN).value)
            if count > 0:
                deletions.append((res_idx, count))
                count = 0
            res_idx += 1

        if expected_len is None:
            expected_len = len(residues)
        elif len(residues) != expected_len:
            raise ValueError(
                f"MSA row {seq_idx} has aligned length {len(residues)}, expected {expected_len}"
            )

        res_start = len(all_residues)
        del_start = len(all_deletions)
        sequences_arr.append(
            (
                seq_idx,
                -1,
                res_start,
                res_start + len(residues),
                del_start,
                del_start + len(deletions),
            )
        )
        all_residues.extend(residues)
        all_deletions.extend(deletions)

    return EncodedMSA(
        residues=np.array(all_residues, dtype=MSAResidue),
        deletions=np.array(all_deletions, dtype=MSADeletion),
        sequences=np.array(sequences_arr, dtype=MSASequence),
    )


def _validated_a3m_sequences(a3m: str, sequence: str, max_seqs: int) -> list[str]:
    """Validate one raw A3M and return its aligned sequence rows."""
    rows = _parse_a3m_sequences(a3m)
    if not rows:
        raise ValueError("MSA must contain at least one sequence.")
    query = "".join(
        char for char in rows[0] if not char.islower() and char not in {"-", "."}
    ).upper()
    if query != sequence.upper():
        raise ValueError(
            f"MSA query sequence {query!r} does not match system sequence {sequence!r}."
        )
    return rows[:max_seqs]


def tokenize_msa(
    structure: AllAtomStructure,
    unpaired_a3ms: Mapping[int, str],
    paired_a3ms: Mapping[int, str] | None = None,
    max_seqs: int = MAX_SEQS,
    max_paired_seqs: int = 8192,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Convert pre-fetched A3Ms into token-aligned MSA arrays.

    The structure provides the resolved token layout needed to map residue-level alignments onto
    model tokens and add dummy features for non-protein entities.

    Args:
        structure: Structure whose token layout and protein sequences define the MSA features.
        unpaired_a3ms: Raw unpaired A3M text keyed by protein chain index.
        paired_a3ms: Optional raw paired A3M text keyed by protein chain index.
        max_seqs: Maximum total number of MSA rows.
        max_paired_seqs: Maximum number of paired rows parsed for each protein chain.

    Returns:
        ``(msa, msa_deleted, msa_paired)`` integer arrays, each shaped
        ``[num_tokens, num_msa_rows]``. They contain residue-token indices, deletion counts, and
        paired-row indicators, respectively.
    """
    token_sequences = structure.sequence.split(":")
    allatom_asymid = structure.upsample_to_atoms(structure.asym_id)
    chain_ids = [int(chain_id) for chain_id in np.unique(structure.asym_id.numpy())]

    protein_chainids = [
        int(chain_id)
        for chain_id in np.unique(
            allatom_asymid[structure.mol_type == MolType.PROTEIN.value].numpy()
        )
    ]
    protein_sequences = {chain_id: token_sequences[chain_id] for chain_id in protein_chainids}
    expected_chainids = set(protein_chainids)
    if set(unpaired_a3ms) != expected_chainids:
        raise ValueError(
            f"Unpaired A3Ms must be provided for chains {sorted(expected_chainids)}, "
            f"got {sorted(unpaired_a3ms)}."
        )
    if paired_a3ms is not None and set(paired_a3ms) != expected_chainids:
        raise ValueError(
            f"Paired A3Ms must be provided for chains {sorted(expected_chainids)}, "
            f"got {sorted(paired_a3ms)}."
        )
    is_heteromeric = len(set(protein_sequences.values())) > 1

    if is_heteromeric and paired_a3ms is not None:
        paired_seqs_per_chain = {
            chain_id: [
                sequence
                for sequence in _validated_a3m_sequences(
                    paired_a3ms[chain_id],
                    protein_sequences[chain_id],
                    max_paired_seqs + 1,
                )[1:]
                if sequence.replace("-", "")
            ][:max_paired_seqs]
            for chain_id in protein_chainids
        }
        unpaired_seqs_per_chain = {
            chain_id: _validated_a3m_sequences(
                unpaired_a3ms[chain_id],
                protein_sequences[chain_id],
                max_seqs,
            )
            for chain_id in protein_chainids
        }

        paired_depth = min(
            min(len(sequences) for sequences in paired_seqs_per_chain.values()),
            max_seqs - 1,
        )

        msas_per_asym_id: dict[int, EncodedMSA] = {}
        pairing: list[dict[int, int]] = [{c: 0 for c in chain_ids}]
        is_paired: list[dict[int, int]] = [{c: 1 for c in chain_ids}]

        for chain_id in protein_chainids:
            seq = protein_sequences[chain_id]
            paired = paired_seqs_per_chain[chain_id]
            unpaired = unpaired_seqs_per_chain[chain_id]

            combined = [seq] + paired[:paired_depth]
            unpaired_start = (
                1 if unpaired and unpaired[0].replace("-", "").upper() == seq.upper() else 0
            )
            remaining_budget = max_seqs - len(combined)
            combined += unpaired[unpaired_start : unpaired_start + remaining_budget]
            msas_per_asym_id[chain_id] = _build_msa_from_colabfold_seqs(combined)

        for i in range(1, paired_depth + 1):
            pairing.append({c: (i if c in protein_chainids else -1) for c in chain_ids})
            is_paired.append({c: (1 if c in protein_chainids else 0) for c in chain_ids})

        unpaired_offset = paired_depth + 1
        max_unpaired = max(
            len(msas_per_asym_id[c].sequences) - unpaired_offset for c in protein_chainids
        )
        for i in range(max_unpaired):
            row = {}
            for c in chain_ids:
                if c in protein_chainids:
                    seq_idx = unpaired_offset + i
                    row[c] = seq_idx if seq_idx < len(msas_per_asym_id[c].sequences) else -1
                else:
                    row[c] = -1
            pairing.append(row)
            is_paired.append({c: 0 for c in chain_ids})
            if len(pairing) >= max_seqs:
                break

    else:
        msas_per_asym_id = {
            chain_id: _build_msa_from_colabfold_seqs(
                _validated_a3m_sequences(
                    unpaired_a3ms[chain_id],
                    protein_sequences[chain_id],
                    max_seqs,
                )
            )
            for chain_id in protein_chainids
        }

        is_homomeric = len(set(protein_sequences.values())) == 1 and len(protein_chainids) > 1
        # Without protein chains, use the single query row from the dummy MSAs.
        n_seqs = min(
            min((len(msas_per_asym_id[c].sequences) for c in protein_chainids), default=1),
            max_seqs,
        )
        pairing = []
        is_paired = []
        for i in range(n_seqs):
            paired_flag = 1 if is_homomeric else (1 if i == 0 else 0)
            is_paired.append({c: paired_flag for c in chain_ids})
            pairing.append(
                {c: i if c in protein_chainids else (0 if i == 0 else -1) for c in chain_ids}
            )

    # Dummy MSAs for non-protein entities
    for chain_id in chain_ids:
        if chain_id not in msas_per_asym_id:
            residue_types = structure.downsample_to_residues(structure.residue_type)[
                structure.downsample_to_residues(structure.upsample_to_atoms(structure.asym_id))
                == chain_id
            ].numpy()
            msas_per_asym_id[chain_id] = make_dummy_msa(residue_types)

    # Build deletion map
    deletions: dict[tuple[int, int, int], int] = {}
    for chain_id, chain_msa in msas_per_asym_id.items():
        for sequence in chain_msa.sequences:
            for del_data in chain_msa.deletions[sequence["del_start"] : sequence["del_end"]]:
                deletions[(chain_id, sequence["seq_idx"], del_data["res_idx"])] = del_data[
                    "deletion"
                ]

    return _prepare_msa_arrays(structure, pairing, is_paired, deletions, msas_per_asym_id)
