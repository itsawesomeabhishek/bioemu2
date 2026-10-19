"""Acquire, load, inspect, and subset canonical raw A3M alignments."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Self

from Bio import SeqIO
from Bio.SeqIO.FastaIO import FastaWriter

from bioemu2.data_interface import Protein, System
from bioemu2.msa.colabfold import cached_colabfold_msa


@dataclass(frozen=True)
class MSA:
    """Canonical raw multiple-sequence alignment keyed by protein chain ID.

    ``unpaired_a3m`` stores the independent alignment for each protein chain.
    ``paired_a3m`` stores positionally paired heteromer alignments when they
    were requested. Original A3M text is retained unchanged
    until the MSA is subset.
    """

    unpaired_a3m: Mapping[str, str]
    paired_a3m: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        """Validate, detach, and freeze the raw A3M mappings."""
        if self.paired_a3m is not None and set(self.paired_a3m) != set(self.unpaired_a3m):
            raise ValueError("Paired and unpaired MSA chain IDs must match.")

        object.__setattr__(
            self,
            "unpaired_a3m",
            MappingProxyType(dict(self.unpaired_a3m)),
        )
        if self.paired_a3m is not None:
            object.__setattr__(
                self,
                "paired_a3m",
                MappingProxyType(dict(self.paired_a3m)),
            )

    @property
    def chain_ids(self) -> list[str]:
        """Return the protein chain IDs represented by the MSA."""
        return list(self.unpaired_a3m)

    def a3m(self, chain_id: str | None = None, *, paired: bool = False) -> str:
        """Return the exact raw A3M for one chain and alignment kind.

        ``chain_id`` may be omitted for a monomer or homomer. Paired A3M is
        available only for heteromers acquired from ColabFold.
        """
        a3ms = self.paired_a3m if paired else self.unpaired_a3m
        if a3ms is None:
            raise ValueError("Paired A3M is unavailable.")

        if chain_id is None:
            query_sequences = {
                _query_sequence(str(next(SeqIO.parse(StringIO(text), "fasta-pearson")).seq))
                for text in self.unpaired_a3m.values()
            }
            if len(query_sequences) != 1:
                raise ValueError(
                    "chain_id is required for an MSA with multiple unique protein sequences."
                )
            chain_id = next(iter(a3ms))

        if chain_id not in a3ms:
            raise KeyError(f"Unknown protein chain ID {chain_id!r}.")
        return a3ms[chain_id]

    @property
    def sequences(self) -> list[str]:
        """Return the unpaired sequence rows for a monomer or homomer.

        Raises:
            ValueError: If the MSA contains more than one unique query sequence.
        """
        sequences_by_chain = [
            [str(record.seq) for record in SeqIO.parse(StringIO(a3m), "fasta-pearson")]
            for a3m in self.unpaired_a3m.values()
        ]
        query_sequences = {_query_sequence(sequences[0]) for sequences in sequences_by_chain}
        if len(query_sequences) != 1:
            raise ValueError("This operation requires a single unique protein sequence.")
        return sequences_by_chain[0]

    def __len__(self) -> int:
        """Return the unpaired alignment depth for a monomer or homomer."""
        return len(self.sequences)

    def __repr__(self) -> str:
        """Return a concise summary without embedding the raw A3M text."""
        return (
            f"MSA(chain_ids={self.chain_ids!r}, paired={self.paired_a3m is not None}, "
            f"unpaired_shapes={_a3m_shapes(self.unpaired_a3m)!r}, "
            f"paired_shapes={_a3m_shapes(self.paired_a3m)!r})"
        )

    def __getitem__(self, indices: slice | Iterable[int]) -> Self:
        """Return an MSA containing the selected rows.

        The same indices are applied to every paired and unpaired alignment.
        For subsets intended for inference, retain query row 0 first; embedding
        checks that the first row matches the corresponding system sequence.
        Other subsets are allowed for inspection. Biopython reserializes selected
        records as unwrapped A3M, so comments and the original sequence line
        wrapping are not retained in the subset.

        Args:
            indices: Slice or iterable of integer row indices.
        """
        # Materialize generators so the first alignment does not exhaust indices for the others.
        if not isinstance(indices, slice):
            indices = tuple(indices)
        unpaired_a3m = {
            chain_id: _subset_a3m(a3m, indices) for chain_id, a3m in self.unpaired_a3m.items()
        }
        paired_a3m = None
        if self.paired_a3m is not None:
            paired_a3m = {
                chain_id: _subset_a3m(a3m, indices) for chain_id, a3m in self.paired_a3m.items()
            }
        return type(self)(unpaired_a3m=unpaired_a3m, paired_a3m=paired_a3m)


def get_msa(
    system: System,
    source: Literal["colabfold"] | Path | Mapping[str, Path] = "colabfold",
    *,
    use_env: bool = True,
    pairing_strategy: Literal["greedy", "complete"] = "greedy",
) -> MSA:
    """Acquire a complete canonical MSA.

    ColabFold is used by default. Raw A3M responses are cached in
    ``BIOEMU2_MSA_CACHE`` (by default ``~/.cache/bioemu2/colabfold_msa``).
    Repeated protein sequences are searched once and expanded back to every
    chain. Heteromers receive both independent unpaired alignments and
    positionally paired alignments. A chain-keyed path mapping loads local
    unpaired A3M files instead.

    Args:
        system: Molecular system. Systems without protein chains use an empty MSA
            and do not query ColabFold.
        source: ``"colabfold"``, one local A3M path for a system with one
            unique protein sequence, or one local A3M path per protein chain.
        use_env: Whether ColabFold should search environmental databases for
            unpaired alignments. Paired searches use the non-environmental mode.
        pairing_strategy: ColabFold heteromer pairing strategy.

    Returns:
        The complete, untruncated raw MSA.

    Raises:
        ValueError: If one path is provided for multiple
            unique protein sequences, the pairing strategy is invalid, or a
            cached alignment is incomplete or corrupt.
        RuntimeError: If ColabFold returns an unexpected number of alignments.
    """
    if isinstance(source, Path):
        protein_chains = [
            (chain_id, entity.sequence)
            for entity in system.entities
            if isinstance(entity, Protein)
            for chain_id in entity.asym_ids
        ]
        if len({sequence for _, sequence in protein_chains}) > 1:
            raise ValueError("A single MSA path requires one unique protein sequence.")
        return load_a3m_msa(
            system, dict.fromkeys((chain_id for chain_id, _ in protein_chains), source)
        )
    if source != "colabfold":
        return load_a3m_msa(system, source)

    protein_chains = [
        (chain_id, entity.sequence)
        for entity in system.entities
        if isinstance(entity, Protein)
        for chain_id in entity.asym_ids
    ]
    if not protein_chains:
        return MSA({})
    if pairing_strategy not in {"greedy", "complete"}:
        raise ValueError(
            f"pairing_strategy must be 'greedy' or 'complete', got {pairing_strategy!r}."
        )
    unique_sequences = list(dict.fromkeys(sequence for _, sequence in protein_chains))

    unpaired_results = cached_colabfold_msa(
        unique_sequences,
        use_pairing=False,
        use_env=use_env,
        pairing_strategy=pairing_strategy,
    )
    if len(unpaired_results) != len(unique_sequences):
        raise RuntimeError(
            f"ColabFold returned {len(unpaired_results)} alignments "
            f"for {len(unique_sequences)} sequences."
        )
    unpaired_by_sequence = dict(zip(unique_sequences, unpaired_results))
    unpaired_a3ms = {
        chain_id: unpaired_by_sequence[sequence] for chain_id, sequence in protein_chains
    }

    if len(unique_sequences) == 1:
        return MSA(unpaired_a3m=unpaired_a3ms)

    paired_results = cached_colabfold_msa(
        unique_sequences,
        use_pairing=True,
        use_env=False,
        pairing_strategy=pairing_strategy,
    )
    if len(paired_results) != len(unique_sequences):
        raise RuntimeError(
            f"ColabFold returned {len(paired_results)} alignments "
            f"for {len(unique_sequences)} sequences."
        )
    paired_by_sequence = dict(zip(unique_sequences, paired_results))
    paired_a3ms = {chain_id: paired_by_sequence[sequence] for chain_id, sequence in protein_chains}
    return MSA(unpaired_a3m=unpaired_a3ms, paired_a3m=paired_a3ms)


def load_a3m_msa(system: System, paths: Mapping[str, Path]) -> MSA:
    """Load and validate complete unpaired A3M files.

    Args:
        system: Molecular system whose protein chains define the expected queries.
        paths: One A3M path for every protein chain ID in ``system``.

    Returns:
        A canonical MSA containing the original A3M text.

    Raises:
        ValueError: If protein chains are missing, path keys do not match the
            protein chain IDs, or an A3M query does not match its chain sequence.
    """
    protein_chains = [
        (chain_id, entity.sequence)
        for entity in system.entities
        if isinstance(entity, Protein)
        for chain_id in entity.asym_ids
    ]
    if not protein_chains:
        raise ValueError("System must contain at least one protein chain.")
    chain_ids = [chain_id for chain_id, _ in protein_chains]
    if set(paths) != set(chain_ids):
        raise ValueError(
            f"MSA paths must be provided for chains {sorted(chain_ids)}, got {sorted(paths)}."
        )
    a3ms = {chain_id: paths[chain_id].expanduser().read_text() for chain_id in chain_ids}
    for chain_id, sequence in protein_chains:
        records = SeqIO.parse(StringIO(a3ms[chain_id]), "fasta-pearson")
        query = str(next(records).seq)
        if _query_sequence(query) != sequence.upper():
            raise ValueError(
                f"MSA query for chain {chain_id!r} does not match system sequence {sequence!r}."
            )
    return MSA(a3ms)


def _subset_a3m(text: str, indices: slice | Iterable[int]) -> str:
    """Select and serialize A3M records using Biopython."""
    records = list(SeqIO.parse(StringIO(text), "fasta-pearson"))
    if isinstance(indices, slice):
        selected = records[indices]
    else:
        selected = [records[index] for index in indices]
    output = StringIO()
    FastaWriter(output, wrap=0).write_file(selected)
    return output.getvalue()


def _query_sequence(query: str) -> str:
    """Remove A3M insertions and gaps from a query row."""
    return "".join(char for char in query if not char.islower() and char not in {"-", "."}).upper()


def _a3m_shapes(
    a3ms: Mapping[str, str] | None,
) -> dict[str, tuple[int, int]] | None:
    """Return each A3M block's ``(depth, query length)``."""
    if a3ms is None:
        return None
    shapes = {}
    for chain_id, a3m in a3ms.items():
        records = list(SeqIO.parse(StringIO(a3m), "fasta-pearson"))
        shapes[chain_id] = (len(records), len(_query_sequence(str(records[0].seq))))
    return shapes
