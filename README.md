<div align="center">

<img src="assets/emu.png" alt="BioEmu2 logo" width="120">

# BioEmu2

### A foundation model for biomolecular structure and dynamics

**Predict folded structures, sample equilibrium ensembles, and propagate structures forward in time.**

[Installation](#installation) · [Quick start](#quick-start) · [Tutorials](#tutorials) · [Documentation](docs/)

<br>

<img src="https://github.com/microsoft/bioemu2/releases/download/readme-media/mcl1_puma_clean_grid_v2.gif" alt="BioEmu2 simulating the PUMA BH3 peptide binding MCL-1" width="420">

<sub>BioEmu2 simulating the PUMA BH3 peptide binding MCL-1.</sub>

</div>

---

BioEmu2, short for "biomolecular emulator", is developed by Microsoft Research
AI for Science.

Key features include:

- molecular systems containing proteins, nucleic acids, ligands, and other
  chemical components;
- equilibrium sampling and structure prediction from the command line or
  Python;
- time-lagged trajectory generation with implicit transfer operator models;
- contact conditioning and inference-time steering toward desired properties;
- context-parallel sampling of structures that do not fit on one GPU.

## System requirements

BioEmu2 requires Linux and an NVIDIA GPU; macOS and Windows are not supported.
The `cuda` extra installs CUDA 13 cuEquivariance, which requires compute
capability 7.5 (Turing) or newer
([CUDA 13 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html#deprecated-architectures)).
GPUs with compute capability 7.0 (Volta) need the CUDA 12
[installation](#installation) instead.
We have tested V100 (compute capability 7.0), T4 (7.5, with the CUDA 12
installation), A100 and RTX A6000 (8.0 and 8.6), and B200 (10.0); other eligible
GPUs have not been verified end to end. Check
your GPU's compute capability with
`nvidia-smi --query-gpu=name,compute_cap --format=csv`.

Context-parallel sampling requires at least four GPUs.

## Installation

Activate a Python 3.12 environment on a supported Linux machine, then install
BioEmu2 from the repository root:

```bash
pip install ".[cuda]"
```

The `cuda` extra supports GPUs with compute capability 7.5 (Turing) or newer.
Compute capability 7.0 (Volta) is the oldest supported; on these GPUs, install
PyTorch built for CUDA 12.6 first, then use the `cuda12` extra:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install ".[cuda12]"
```

GPUs with compute capability below 7.0 are not supported.

For development, see [DEVELOPER.md](DEVELOPER.md).

For Jupyter or VS Code notebooks, follow the [notebook
quickstart](#notebook-quickstart) to install visualization dependencies and
register a kernel.

## Quick start

### CLI

Sample eight equilibrium conformations of adenylate kinase (PDB 1AKE):

```bash
bioemu2 sample-equilibrium \
  --pdb-id 1AKE \
  --out outputs/adk
```

Or predict a folded structure from a PDB entry:

```bash
bioemu2 predict-structure \
  --pdb-id 1UBQ \
  --num-samples 1 \
  --out outputs/ubiquitin
```

Propagate each sampled adenylate kinase conformation by 1 ns:

```bash
bioemu2 propagate \
  --x0-cif outputs/adk/topology.cif \
  --x0-xtc outputs/adk/samples.xtc \
  --lag-time-ns 1 \
  --out outputs/adk-propagated
```

Run `bioemu2 --help` to list all commands.

### Python API using EmuSession

Construct an `EmuSession` to sample and propagate structures:

```python
from pathlib import Path

from bioemu2 import EmuSession, System, load_model, save_frames

model = load_model()

# Predict a folded structure for ubiquitin.
ubiquitin = System.from_pdb_id("1UBQ")
prediction = EmuSession(ubiquitin, model=model, mode="structure").sample(num_samples=1)
save_frames(prediction, Path("outputs/ubiquitin"))

# Sample equilibrium conformations of adenylate kinase.
adk = System.from_pdb_id("1AKE")
emu = EmuSession(adk, model=model, mode="equilibrium")
samples = emu.sample(num_samples=8)
save_frames(samples, Path("outputs/adk"))

# Propagate each adenylate kinase conformation by 1 ns.
endpoints = emu.propagate(samples, lag_time_ns=1.0)
save_frames(endpoints, Path("outputs/adk-propagated"))
```

`save_frames()` writes `samples.xtc` and `topology.cif` to each output
directory.

See [define molecular systems](docs/system.md), [predict and sample
structures](docs/sampling.md), and [propagate structures](docs/propagation.md)
for more options. The functional Python APIs provide
[high-level sampling](docs/functional-api.md#sample-structures) and
[propagation](docs/functional-api.md#propagate-structures) functions similar to
the CLI commands, plus lower-level workflows for explicit control of
embeddings and samplers.

## More workflows

### Steer sampling

Choose a steering sampler to favour structures with desired properties:

```bash
bioemu2 sample-equilibrium \
  --system src/bioemu2/assets/examples/systems/myoglobin_101m.yaml \
  --out outputs/myoglobin \
  --sampler edm_gd
```

See [steer sampling](docs/steering.md) and [define steering
potentials](docs/potentials.md) to select an algorithm and configure its
objective.

### Sample large structures across multiple GPUs

Context parallelism distributes the model's pair representations across a
square mesh of GPUs, reducing the memory required on each device:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc-per-node=4 \
  --module bioemu2.cp sample-equilibrium \
  --pdb-id 1AKE \
  --checkpoint bioemu2-preview \
  --num-samples 3 \
  --out outputs/adk-cp
```

See [sample large structures with context
parallelism](docs/context-parallel-sampling.md) for mesh requirements, the
Python API, and current limitations.

### Inspect and subsample MSAs

Retrieve an alignment from ColabFold or load your own A3M files, inspect the
raw sequences, and select or randomly subsample rows with Python indexing.
Pass the resulting `MSA` directly to `EmuSession` to control the alignment
used for structure prediction, equilibrium sampling, or propagation without
manually constructing embeddings.

See [use multiple sequence alignments](docs/msa.md) for local-file loading,
[MSA subsampling](docs/msa.md#select-rows-and-subsample-an-msa), and
[custom-MSA inference](docs/msa.md#use-a-custom-msa-for-inference).

## Tutorials

- [Define molecular systems](docs/system.md) from PDB IDs, structure files, or sequences.
- [Convert Boltz input YAML](docs/boltz.md) into systems you can sample from.
- [Customise multiple-sequence alignments](docs/msa.md) used to condition sampling.
- [Predict and sample structures](docs/sampling.md) for a system.
- [Improve sampling speed and scale to bigger systems](docs/speed-and-capacity.md) by choosing batch size and precision.
- [Load a model checkpoint](docs/checkpoints.md), choose where checkpoints are cached, or use a local checkpoint.
- [Propagate structures through time](docs/propagation.md).
- [Reuse trunk embeddings](docs/reuse-embeddings.md) across sampling and propagation calls, in memory or on disk.
- [Use the functional Python API](docs/functional-api.md) to sample and propagate without an `EmuSession`.
- [Steer sampling](docs/steering.md) towards structures with desired properties.
- [Define steering potentials](docs/potentials.md) that encode the properties you want to favour.
- [Sample large structures](docs/context-parallel-sampling.md) across multiple GPUs with context parallelism.

## Notebook quickstart

The [`examples`](examples) directory contains complete workflows as regular
Python files split into executable cells with `# %%` markers. This Python
Interactive format provides notebook-style cells while keeping the examples as
plain `.py` files. They require a Linux machine with a supported NVIDIA GPU and
network access.

To run these examples, activate a Python 3.12 environment and install the
CUDA and visualization dependencies from the repository root. The `viz` extra
includes `ipykernel` and `py3Dmol`:

```bash
pip install ".[cuda,viz]"
python -m ipykernel install --user --name bioemu2 --display-name "BioEmu2"
```

Select the **BioEmu2** kernel in Jupyter or VS Code. To check that it uses the
environment you installed into, run `import sys; print(sys.executable)` in a
cell. The examples load system YAML from the installed `bioemu2` package,
regardless of the working directory. Relative `outputs/` paths are written
under the current working directory.

Open a single-GPU example in VS Code and select **Run Cell**, or run **Jupyter:
Run Current File in Interactive Window** from the Command Palette. Launch the
context-parallel example with `torchrun` as described in the
[context-parallel sampling guide](docs/context-parallel-sampling.md#python-api).

To run without VS Code, convert an example with
[Jupytext](https://jupytext.readthedocs.io/) and open it in JupyterLab:

```bash
pip install jupytext jupyterlab
jupytext --to notebook examples/sample_structures.py
jupyter lab examples/sample_structures.ipynb
```

Select the **BioEmu2** kernel, then run individual cells or choose **Run All
Cells** from the **Run** menu.

In VS Code, you can instead run **Jupyter: Export Current Python File as Jupyter
Notebook** from the Command Palette. Generated `.ipynb` files are ignored by
Git.

## License

BioEmu2 is released under the [MIT License](LICENSE). It includes code and
reference data adapted from other open-source projects; the [third-party
notices](NOTICES.md) identify those components and their licenses.
