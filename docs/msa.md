# Use multiple sequence alignments

<!-- BEGIN GENERATED TOC -->
## Contents

- [Retrieve the default MSA from ColabFold](#retrieve-the-default-msa-from-colabfold)
- [Use a different or previously saved MSA](#use-a-different-or-previously-saved-msa)
- [Construct an MSA from raw text](#construct-an-msa-from-raw-text)
- [Inspect raw alignments](#inspect-raw-alignments)
- [Select rows and subsample an MSA](#select-rows-and-subsample-an-msa)
- [Use a custom MSA for inference](#use-a-custom-msa-for-inference)
<!-- END GENERATED TOC -->

BioEmu2 uses protein multiple sequence alignments (MSAs) to provide
evolutionary information for both structure prediction and equilibrium
sampling.

Use the `MSA` object to inspect raw A3M text, select rows, and subsample
alignments with ordinary Python indexing. Load alignments from your own
pipeline or retrieve them from ColabFold, then pass the original or modified
MSA directly to an `EmuSession` for inference.

By default, BioEmu2 retrieves A3M alignments from the public ColabFold server
when they are not cached. It obtains an independent unpaired alignment for each
unique protein sequence and, for heteromers, also obtains positionally paired
alignments.

See [`examples/sample_structures.py`](../examples/sample_structures.py) for an
executable workflow that computes an MSA and uses it for structure prediction
and equilibrium sampling.

## Retrieve the default MSA from ColabFold

Pass a `System` to `get_msa()`:

```python
from bioemu2 import System, get_msa
from bioemu2.paths import EXAMPLES_DIR

system = System.from_yaml(EXAMPLES_DIR / "systems/ake.yaml")
msa = get_msa(system)
```

On a cache miss, this queries the public ColabFold server and requires network
access. Raw A3M results are saved in `~/.cache/bioemu2/colabfold_msa` by default;
set `BIOEMU2_MSA_CACHE` to use another directory:

```bash
export BIOEMU2_MSA_CACHE=/path/to/msa-cache
```

Repeated calls with the same protein sequences and search options use the
saved raw alignments without contacting ColabFold. Paired and unpaired searches,
environmental database selection, and pairing strategy have separate entries.
Model-ready MSA features are recomputed from raw alignments, not saved to disk.
If a cache entry is corrupt or incomplete, `get_msa()` reports its path;
delete that file to request a new alignment. Repeated protein sequences are
searched once and expanded back to all matching chains.

When using the public server, follow [ColabFold's guidance](https://github.com/sokrypton/ColabFold#readme)
to query serially from a single IP. If you need many MSAs, use ColabFold's
[local `colabfold_search` workflow](https://github.com/sokrypton/ColabFold#generating-msas-for-large-scale-structurecomplex-predictions)
and [load its A3M outputs](#use-a-different-or-previously-saved-msa) instead
of sending a bulk workload to the public server.

Configure the ColabFold search with:

```python
msa = get_msa(
    system,
    use_env=False,
    pairing_strategy="complete",
)
```

The supported pairing strategies are `"greedy"` and `"complete"`.

## Use a different or previously saved MSA

BioEmu2 accepts local A3M alignments whose first record is the query sequence
for the corresponding protein chain. It does not require ColabFold-specific
provenance, so the file may come from another alignment tool, your own
pipeline, or a saved result from an earlier ColabFold query.

For a monomer, homomer, or any system with one unique protein sequence, pass
one local A3M path:

```python
from pathlib import Path

msa = get_msa(
    system,
    source=Path("alignments/ake.a3m"),
)
```

The first A3M record is the query. `get_msa()` checks that the query,
after removing A3M insertions and gaps, matches the protein sequence in
`System`.

Equivalent protein chains can reuse one alignment:

```python
homodimer = System(
    [
        Protein(sequence="RMKQLEDKVEELLSKNYHLENEVARLKKLVGER", asym_ids=["A", "B"]),  # GCN4 leucine zipper (PDB 2ZTA)
    ]
)
homodimer_msa = get_msa(
    homodimer,
    source=Path("alignments/gcn4.a3m"),
)
```

The resulting `MSA` contains entries for both chains.

For a system with different protein sequences, provide one A3M path per
protein chain:

```python
heteromer = System(
    [
        Protein(sequence="NLYIQWLKDGGPSSGRPPPS", asym_ids=["A"]),
        Protein(sequence="MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF", asym_ids=["B"]),
    ]
)
heteromer_msa = get_msa(
    heteromer,
    source={
        "A": Path("alignments/trp-cage.a3m"),
        "B": Path("alignments/villin.a3m"),
    },
)
```

The mapping keys must exactly match the system's protein chain IDs. Each file's
query is validated against its matching chain sequence.

Saved ColabFold results use the same interface:

```python
saved_msa = get_msa(
    system,
    source=Path("saved-colabfold-results/query.a3m"),
)
```

Loading a saved file is network-free. It also makes the alignment input
explicit and reproducible instead of issuing a new service query.

## Construct an MSA from raw text

If another library or service returns A3M text in memory, construct `MSA`
directly:

```python
from bioemu2 import MSA

a3m_text = """>query
NLYIQWLKDGGPSSGRPPPS
>homolog-1
NLYIQWLKDGGPSSGRPPPS
>homolog-2
NLYIQWLK-GGPSSGRPPPS
"""
msa = MSA({"A": a3m_text})
```

For several protein chains, provide one raw A3M string per chain:

```python
msa = MSA(
    {
        "A": chain_a_a3m,
        "B": chain_b_a3m,
    }
)
```

Direct construction preserves the supplied text but does not compare its query
rows with a `System`. Use `get_msa(..., source=...)` when loading files if
you want that validation.

`MSA` stores raw A3M text for each protein chain, preserving headers, gaps,
lowercase insertions, and row order. MSA entries are required only for protein
chains; ligands, RNA, DNA, and branched entities do not need entries.

## Inspect raw alignments

`chain_ids` lists the represented protein chains:

```python
assert msa.chain_ids == ["A"]
```

Retrieve the exact unpaired A3M text for one chain:

```python
raw_a3m = msa.a3m("A")
```

The chain ID may be omitted for a monomer or homomer:

```python
raw_a3m = msa.a3m()
```

Retrieve a ColabFold-paired heteromer alignment with `paired=True`:

```python
colabfold_heteromer_msa = get_msa(heteromer)
paired_chain_a = colabfold_heteromer_msa.a3m("A", paired=True)
```

Local files are loaded as unpaired alignments. Requesting a paired alignment
when none is present raises `ValueError`.

For an MSA computed for one unique protein sequence, inspect parsed rows with
`.sequences` and the alignment depth with `len()`:

```python
query = msa.sequences[0]
print(f"alignment depth: {len(msa)}")
```

For a heteromer, inspect each chain with `msa.a3m(chain_id)`. `.sequences`
and `len(msa)` require a single unique protein sequence and raise `ValueError`,
even when the chain alignments have equal depths.

## Select rows and subsample an MSA

Indexing returns a new `MSA` containing selected rows:

```python
shallow_msa = msa[:128]
selected_msa = msa[[0, 2, 5, 9]]
```

The same indices are applied to each paired and unpaired alignment, including
when the indices come from a generator. You can inspect any block with a slice
such as `msa[100:120]`; indexing does not require the query row.

For subsets intended for inference, keep row 0 first so the query remains first.
During embedding, BioEmu2 checks that each alignment is nonempty and its first
row's query sequence matches the corresponding protein sequence in `System`.

Slicing with `msa[:128]` subsamples the alignment to at most 128 rows,
including the query. To subsample a monomer or homomer randomly and
reproducibly while retaining the query:

```python
import numpy as np

rng = np.random.default_rng(42)
num_non_query = min(127, len(msa) - 1)
indices = np.r_[
    0,
    rng.choice(np.arange(1, len(msa)), num_non_query, replace=False),
]
subsampled_msa = msa[indices]
```

For a heteromer, `msa[:128]` takes the first 128 rows of each
chain's paired and unpaired alignments. An alignment with fewer than 128 rows
keeps all its rows; the slice does not raise an error for shorter alignments.
By contrast, explicit indices must be valid for each paired and unpaired
alignment or indexing raises `IndexError`.

The original `MSA` is unchanged. Subset text is reserialized by Biopython, so
comments and original line wrapping are not retained.

## Use a custom MSA for inference

Pass an `MSA` to `EmuSession` when you want to supply a custom
alignment constructed from raw A3M text:

```python
from bioemu2 import EmuSession, MSA, System, load_model
from bioemu2.paths import EXAMPLES_DIR

system = System.from_yaml(EXAMPLES_DIR / "systems/ake.yaml")
msa = MSA({"A": ">query\nMRIILLGAPGAGKGTQAQFIMEKYGIPQISTGDMLRAAVKSGSELGKQAKDIMDAGKLVTDELVIALVKERIAQEDCRNGFLLDGFPRTIPQADAMKEAGINVDYVLEFDVPDELIVDRIVGRRVHAPSGRVYHVKFNPPKVEGKDDVTGEELTTRKDDQEETVRKRLVEYHQMTAPLIGYYSKEAEAGNTKYAKVDGTKPVAEVRADLEKILG\n"})
emu = EmuSession(system, model=load_model(), mode="equilibrium", msa=msa)
samples = emu.sample(num_samples=8)
```

Pass a custom `MSA` as `msa=` when creating an `EmuSession`.
For example, retrieve the default alignment and subsample it to at most 128 rows:

```python
from bioemu2 import EmuSession, System, get_msa, load_model
from bioemu2.paths import EXAMPLES_DIR

system = System.from_yaml(EXAMPLES_DIR / "systems/ake.yaml")
model = load_model()
msa = get_msa(system)
emu = EmuSession(
    system,
    model=model,
    mode="equilibrium",
    msa=msa[:128],
)
samples = emu.sample(num_samples=8)
```

You can also pass a [locally loaded MSA](#use-a-different-or-previously-saved-msa),
an MSA constructed from raw text, or the `subsampled_msa` from the random
subsampling example. The session uses the supplied alignment
without querying ColabFold and reuses its embeddings for repeated sampling
and propagation calls. Use `mode="structure"` for structure prediction.
Create a new session to use a different MSA; the session's MSA is fixed.

The MSA chain IDs must exactly match the protein chain IDs in `System`.
If you omit `msa`, `EmuSession` acquires it automatically on first use.
ColabFold responses are cached on disk, so callers do not need to retain an
`MSA` object to avoid repeating searches. For greater control, see the
[low-level functional sampling API](functional-api.md#customize-every-step-with-the-low-level-python-api).
