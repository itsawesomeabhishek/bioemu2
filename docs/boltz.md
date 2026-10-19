# Convert Boltz input YAML

Use `bioemu2 convert-boltz` to convert a Boltz-2 or Boltz-1 input YAML file
into a BioEmu2 system YAML file. Both use `version: 1` for the input schema;
this number does not refer to the Boltz model version. Conversion preserves
only the [supported fields](#check-supported-fields), not Boltz prediction
settings such as affinity. It does not load a model, download checkpoints, or
compute an MSA. It uses the installed BioEmu2 environment described in the
[installation instructions](../README.md#installation).

## Convert a file

Convert a Boltz input file with the installed CLI:

```bash
bioemu2 convert-boltz protein_ligand.yaml \
  --out outputs/boltz-conversion/system.yaml
```

For this example, save the following input as `protein_ligand.yaml` before
running the command:

```yaml
version: 1
sequences:
  - protein:
      id: [A, B]
      sequence: RMKQLEDKVEELLSKNYHLENEVARLKKLVGER  # GCN4 leucine zipper (PDB 2ZTA)
  - ligand:
      id: L
      smiles: CCO
```

The resulting BioEmu2 system YAML is:

```yaml
entities:
- entity_type: protein
  asym_ids:
  - A
  - B
  sequence: RMKQLEDKVEELLSKNYHLENEVARLKKLVGER
  mods: []
- entity_type: ligand
  asym_ids:
  - L
  smiles: CCO
bonds: []
asym_id_to_author_id: {}
```

Read all conversion warnings before using the result: unsupported fields are
not present in the output. The resulting file contains molecular composition,
covalent bonds, and supported contact conditioning, not Boltz prediction
settings. See the [detailed BioEmu2 system-YAML reference](system.md) for
the output schema, modifications, bonds, and contact conditioning.

To predict structures from the converted system:

```bash
bioemu2 predict-structure \
  --system outputs/boltz-conversion/system.yaml \
  --out outputs/boltz-prediction \
  --num-samples 5
```

Prediction, unlike conversion, loads a model and computes an MSA. **Boltz
`msa` entries are ignored**, including A3M paths and `msa: empty`: prediction
does not read or reuse the alignment specified in the Boltz input. To use a
saved A3M, call `get_msa(system, source=...)` explicitly as described
below.

## Convert in Python

```python
from bioemu2 import convert_boltz, load_system_and_contacts

convert_boltz(
    "protein_ligand.yaml",
    "outputs/boltz-python/system.yaml",
)
system, contacts = load_system_and_contacts("outputs/boltz-python/system.yaml")
```

`convert_boltz` takes input and output file paths, as strings or `Path` objects.
`contacts` is `None` when the input has no contact constraints. Pass it as
`contacts=contacts` when creating an `EmuSession` for model contact
conditioning. Imported contacts do not enable steering or create steering
potentials.

For an existing Boltz A3M file, pass its path separately to `get_msa()`:

```python
from pathlib import Path

from bioemu2 import get_msa

msa = get_msa(system, source=Path("alignments/chain-a.a3m"))
```

For multiple distinct protein chains, pass a mapping of chain IDs to A3M
paths; see [use a previously saved MSA](msa.md#use-a-different-or-previously-saved-msa).
Resolve any relative Boltz `msa` paths relative to their original input file,
not the converted output: conversion does not copy or validate those files.

## Check supported fields

| Boltz input | Conversion |
| --- | --- |
| `version: 1` | Boltz YAML schema version used by Boltz-1 and Boltz-2. Missing or other versions emit a warning, then conversion continues using the `version: 1` schema. |
| `sequences[].protein`, `rna`, `dna` | Preserve the biological one-letter `sequence`, including unknown protein `X` and nucleotide `N` residues. |
| Entity `id` | A single chain ID or a list becomes `asym_ids`; IDs must be unique and at most four characters long. Identical entity definitions are merged by `System`. |
| Polymer `modifications` | Each 1-based `position` and `ccd` becomes a `ModifiedResidue`. Modifications are sorted by position; the biological sequence is preserved. |
| Ligand `smiles` | Becomes a SMILES-defined ligand. |
| Ligand `ccd` | Becomes a CCD-defined ligand. Exactly one non-null chemical identifier is required. |
| `constraints[].bond.atom1` and `atom2` | Each `[chain_id, residue_index, atom_name]` becomes a covalent bond endpoint. Residue indices are 1-based; residue names are recovered from the entity, including modifications. |
| `constraints[].contact.token1` and `token2` | Each `[chain_id, residue_index]` for a polymer or `[chain_id, atom_name]` for a ligand becomes a BioEmu2 contact token. Contacts are written in the same YAML as the system and loaded with `load_system_and_contacts()`. |
| `constraints[].contact.max_distance` | Preserved in Angstrom; defaults to 6.0 when omitted. BioEmu2 accepts distances from 4.0 to 20.0. |

Contacts and bonds to atoms in SMILES-defined ligands are unsupported:
conversion raises `NotImplementedError` instead of writing atom references
that might target the wrong atom. SMILES ligand entities without these
constraints can still be converted.

CCD ligand atom names are preserved, but they are not checked against the built
structure during conversion. An unknown CCD atom name can pass conversion and
YAML loading, then raise an error when the structure is built for sampling.

## Review discarded and unsupported fields

| Field | Behavior |
| --- | --- |
| Per-chain `msa`, including paths and `empty` | **Ignored with an explicit warning.** No alignment file is read, copied, or used for sampling. Pass saved A3M paths separately to `get_msa(system, source=...)`, or let BioEmu2 retrieve an MSA. See the [MSA guide](msa.md). |
| `cyclic: true` | Raises `NotImplementedError`; cyclic polymers cannot be represented by this converter. |
| `constraints[].pocket` | Unsupported; discarded with a warning. |
| `constraints[].contact.force` | Ignored with a warning. Boltz force is a potential setting; it does not configure BioEmu2 contact conditioning or steering. |
| `templates` | Discarded with warnings; template files are not read. |
| `properties`, including `affinity` | Discarded with warnings; affinity prediction is not part of the output system. |
| Other top-level, entity, modification, bond, constraint, or property fields | Discarded with warnings naming the field. Extra fields within a wholly discarded template, pocket constraint, or affinity property are covered by its warning. Unsupported fields within a contact raise an error. |
| Unsupported entity types, cyclic polymers, multi-CCD ligands, conflicting ligand identifiers, and invalid systems | Raise errors; the CLI does not write an output file for conversion errors. |

Conversion warnings use Python logging. The CLI displays them automatically.
