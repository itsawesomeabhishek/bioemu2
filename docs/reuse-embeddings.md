# Reuse embeddings

<!-- BEGIN GENERATED TOC -->
## Contents

- [Know when embeddings can be reused](#know-when-embeddings-can-be-reused)
- [Reuse embeddings in memory](#reuse-embeddings-in-memory)
  - [Sample many structures in one job](#sample-many-structures-in-one-job)
  - [Keep embeddings in memory with EmuSession](#keep-embeddings-in-memory-with-emusession)
  - [Reuse embeddings with the functional API](#reuse-embeddings-with-the-functional-api)
- [Save embeddings to disk](#save-embeddings-to-disk)
  - [Save and load specific embeddings](#save-and-load-specific-embeddings)
  - [Cache embeddings to disk automatically](#cache-embeddings-to-disk-automatically)
- [Reuse embeddings with context parallelism](#reuse-embeddings-with-context-parallelism)
<!-- END GENERATED TOC -->

BioEmu2 has two main model components: the trunk processes the system's
sequence, MSA, and task conditioning into embeddings; the diffusion module,
also called the denoiser, uses those embeddings to generate structures. See the
[sampling workflow diagram](speed-and-capacity.md#know-where-the-time-goes).

Especially for larger systems, running the trunk can account for much of the
computation time. Reusing its embeddings avoids running the trunk again for
every call. Once computed, the same embeddings can be used to sample with
different sampling algorithms, settings, or steering potentials, and to
propagate starting structures into time-lagged endpoints.

## Know when embeddings can be reused

Trunk embeddings depend on these inputs:

- the system;
- the MSA;
- the sampling mode (`"structure"` or `"equilibrium"`);
- contacts, if any;
- the model checkpoint;
- the embedding precision, set with `load_model(precision=...)`.

If any of these changes, compute new embeddings. Embeddings do not depend on the
sampler, its settings, steering potentials, the number of samples, the batch
size, the seed, or the propagation lag time, so you can change any of those and
keep the same embeddings.

`EmuSession` and the automatic disk cache check these inputs for you:

- An `EmuSession` raises an error if you modify its system or contacts after it
  computes embeddings, and does not let you reassign its model, MSA, or mode;
  [create a new session](sampling.md#use-emusession) when inputs change.
- The [automatic disk cache](#cache-embeddings-to-disk-automatically) stores embeddings
  under a key computed from all of these inputs, so it only reloads embeddings
  whose inputs match.

Embeddings you [save and load yourself](#save-and-load-specific-embeddings) are
not checked. Loading embeddings computed from different inputs can produce
incorrect samples without an error. If the system differs, the embedding shapes
may not match it and sampling may fail with an error instead.

## Reuse embeddings in memory

### Sample many structures in one job

If sampler settings stay the same, **do** run one job with `num_samples=4000`
and `batch_size=50`: the model computes embeddings once, then denoises 50
structures at a time. **Don't** run 80 separate jobs with `num_samples=50`:
each CLI or functional API job recomputes embeddings unless the
[automatic disk cache](#cache-embeddings-to-disk-automatically) is enabled. Reduce
`batch_size` further if a batch exceeds GPU memory.

### Keep embeddings in memory with EmuSession

An `EmuSession` obtains embeddings on its first `EmuSession.sample()` or
`EmuSession.propagate()` call and retains them on the model's device (GPU
memory for CUDA models), not automatically in host memory. They remain
in memory until you call `EmuSession.cache_clear()` or discard the session.

The following calls sample with two different samplers and then propagate,
all without recomputing trunk embeddings:

```python
from bioemu2 import (
    EDMSampler,
    EmuSession,
    GuidedFKSampler,
    System,
    get_msa,
    load_model,
)
from bioemu2.denoising.steering.potentials import BondLengthPotential

system = System.from_pdb_id("1AKE")
model = load_model()
msa = get_msa(system)
emu = EmuSession(system, model=model, mode="equilibrium", msa=msa)

# Embeddings are computed during this call and cached by the EmuSession.
unsteered_samples = emu.sample(
    num_samples=8,
    sampler=EDMSampler(step_scale=1.0),
    seed=42,
)

# EmuSession automatically retrieves its cached trunk embeddings.
steered_samples = emu.sample(
    num_samples=8,
    sampler=GuidedFKSampler(
        potentials=[
            BondLengthPotential(
                sigma_tolerance=10.0,
                parameters={"guidance_weight": 0.05},
            )
        ],
        step_scale=1.0,
        resampling_interval=None,
    ),
    seed=43,
)

# Propagation also reuses the cached trunk embeddings.
endpoints = emu.propagate(unsteered_samples, lag_time_ns=1.0)
```

Propagating from a session whose mode is not `"equilibrium"` logs a warning.

To release this session's cached embeddings and free their memory when you no
longer need them:

```python
emu.cache_clear()
```

The cache belongs to this instance; there is no shared in-memory cache across
sessions. `cache_clear()` keeps the session, its model, system, and MSA intact.
Its next sampling or propagation call obtains embeddings again, either by
running the trunk or loading a matching entry from the optional disk cache.
It does not clear the disk cache or tensors referenced elsewhere.

### Reuse embeddings with the functional API

To sample or propagate the same system several times, compute embeddings once
with `model.embed()` and pass them to `sample_with_embeddings()` or
`propagate_with_embeddings()`. The high-level functions `predict_structure()`,
`sample_equilibrium()`, and `propagate()` run the trunk on every call unless
the [automatic disk cache](#cache-embeddings-to-disk-automatically) is enabled.

```python
from bioemu2 import EDMSampler, System, get_msa, load_model, sample_with_embeddings

system = System.from_pdb_id("1AKE")
model = load_model()

# Run the trunk once.
embeddings = model.embed(system, msa=get_msa(system), mode="equilibrium")
samples = sample_with_embeddings(
    system, embeddings, model=model, sampler=EDMSampler(), seed=1
)
more_samples = sample_with_embeddings(
    system, embeddings, model=model, sampler=EDMSampler(num_steps=100), seed=2
)
```

For sampling and propagation with the same embeddings, see the
[low-level propagation example](functional-api.md#low-level-functional-api).

## Save embeddings to disk

`TrunkEmbeddings`, the type returned by `model.embed()`, has `save()` and
`load()` methods that write its tensors to a file and read them back. You can
call these yourself, or let BioEmu2 save and reload embeddings automatically.

### Save and load specific embeddings

To control exactly which embeddings persist, save them to a location you
choose and load them back later. Unlike the
[automatic disk cache](#cache-embeddings-to-disk-automatically), you decide what
to keep and when to delete it.

Without a session, save the `TrunkEmbeddings` returned by `model.embed()` to a
file you choose and reload them for `sample_with_embeddings()`:

```python
from pathlib import Path

import torch

from bioemu2 import (
    EDMSampler,
    System,
    TrunkEmbeddings,
    get_msa,
    load_model,
    sample_with_embeddings,
)

system = System.from_pdb_id("1AKE")
model = load_model()
msa = get_msa(system)

path = Path("outputs/adk-embeddings.pt")
embeddings = model.embed(system, msa=msa, mode="equilibrium")
embeddings.save(path)

# Later, reload and sample without recomputing embeddings.
reloaded = TrunkEmbeddings.load(path, device=model.device)
samples = sample_with_embeddings(
    system, reloaded, model=model, num_samples=8, sampler=EDMSampler()
)
```

`TrunkEmbeddings.save()` writes just the embedding tensors to the file you choose
and refuses to overwrite an existing file; it does not manage any metadata.

With an `EmuSession`, call `EmuSession.save_embeddings(directory)` to save
embeddings, computing them first if they are not already computed. The
directory contains `embeddings.pt` alongside `metadata.json`, which records the
input fingerprints (system, MSA, contacts, mode, and model when available) and
the autocast dtype and recycling steps. This is
similar to the automatic cache metadata, but omits its cache-only version field.
BioEmu2 does not check this metadata when loading, but you can inspect it
yourself to confirm which inputs produced the embeddings. Reload the
embeddings with `TrunkEmbeddings.load()` and pass them to a new session through
its `embeddings` argument. The reused session needs no MSA, because it skips the
trunk entirely:

```python
from pathlib import Path

from bioemu2 import EmuSession, System, TrunkEmbeddings, get_msa, load_model

system = System.from_pdb_id("1AKE")
model = load_model()
msa = get_msa(system)

directory = Path("outputs/adk-embeddings")
emu = EmuSession(system, model=model, mode="equilibrium", msa=msa)
emu.save_embeddings(directory)

# Later, or in a new process, reuse the saved embeddings without running the
# trunk. No MSA is needed because the embeddings come from disk.
embeddings = TrunkEmbeddings.load(directory / "embeddings.pt", device=model.device)
new_emu = EmuSession(system, model=model, mode="equilibrium", embeddings=embeddings)
samples = new_emu.sample(num_samples=8, seed=42)
```

`EmuSession.save_embeddings()` refuses to write into a non-empty directory, so
existing embeddings are never silently overwritten; use a separate directory
for each session whose embeddings you save.
Sessions constructed with `embeddings=` cannot save session metadata because
the original embedding inputs are unknown; use `TrunkEmbeddings.save(path)` to
save those tensors without metadata. Loading does not check the embeddings
against the session's inputs, so load ones
produced for the same system, MSA, contacts, mode, and model. Passing a
mismatched embedding produces incorrect samples.

### Cache embeddings to disk automatically

Automatic caching is disabled by default. Set `BIOEMU2_EMBEDS_CACHE`
to a directory to enable it. Later sampling or propagation runs automatically
reload the embeddings instead of recomputing them when the system, MSA,
contacts, sampling mode, model, precision, and recycling settings match.

Set `BIOEMU2_EMBEDS_CACHE` before running the CLI or Python, or supply it for
one CLI command:

```bash
BIOEMU2_EMBEDS_CACHE=/data/bioemu2-embeddings bioemu2 sample-equilibrium \
  --pdb-id 1AKE \
  --out outputs/adk
```

Unset `BIOEMU2_EMBEDS_CACHE` to disable caching. Each keyed directory contains
`metadata.json` with input fingerprints and settings, and `embeddings.pt` with
the tensors. The metadata does not include the raw MSA or System definition.
Checkpoints without a model fingerprint are not cached.

The cache writes a new keyed directory for every distinct combination of
inputs, so its size grows with each variation and it is never pruned. When you
want to keep only the embeddings for specific systems,
[save and load them explicitly](#save-and-load-specific-embeddings) instead.

## Reuse embeddings with context parallelism

`EmuSession`, `BIOEMU2_EMBEDS_CACHE`, and saving or loading embeddings do not
support context parallelism. To reuse embeddings for
[context-parallel sampling](context-parallel-sampling.md), use the
[context-parallel functional API](context-parallel-sampling.md#reuse-prepared-embeddings).
