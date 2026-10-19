# BioEmu-2

BioEmu-2 is a research model that generates ensembles of plausible all-atom 3D structures for biomolecules from their chemical composition. These conformational ensembles can help researchers study how biomolecules move and interact. BioEmu-2 supports multi-chain complexes, ligands, and nucleic acids.

## Model summary

| **Field** | **Value** |
|----|----|
| Developer | Microsoft Research |
| Model architecture | Diffusion-based all-atom score model. |
| Parameters | Approximately 500M parameters |
| Inputs | Biomolecular sequence(s) plus an optional multiple sequence alignment (MSA), contact conditioning, and method conditioning. |
| Outputs | A generative ensemble of all-atom 3D biomolecular structures in Cartesian coordinates, sampled by reverse diffusion. |
| Training dates | August–September 2026 |
| Release date | October 2026 |
| License | MIT |
| Related assets | [Source code](https://github.com/microsoft/bioemu2) and [model checkpoints](https://huggingface.co/microsoft/bioemu2) |

## Key model capabilities

- Predicts 3D structures of biomolecules given their chemical composition.

- Models multiple conformations rather than returning only one structure.

- Structure prediction can optionally be guided by user-provided information about distances between specific atoms.

## Usage

### Primary use cases

BioEmu-2 is intended for structural-biology and computational-chemistry researchers. It can predict 3D structures of biomolecules and biomolecular complexes given their composition.

### Out-of-scope use cases

BioEmu-2 is not intended or validated for:

- clinical, diagnostic, treatment, dosing, or other patient decisions;

- designing toxins, pathogens, gain-of-function constructs, or other hazardous biological agents;

- quantitative binding-affinity or absolute free-energy prediction;

- de novo small-molecule design;

- autonomous or high-risk decision-making (e.g. law enforcement, legal, finance, healthcare);

- sensitive domains where inaccurate outputs could lead to injury or affect someone's legal, financial, or life opportunities;

- uses that violate applicable law, export controls, or the Microsoft Acceptable Use Policy.

BioEmu-2 predicts structural ensembles for sequences supplied by the user. It does not generate sequences, and it does not predict binding affinities. Predicted structures are not experimentally validated measurements and should not be treated as such.

BioEmu-2 is released for research purposes. We do not recommend using it in commercial or real-world applications without further testing and development.

It was not designed or evaluated for every possible downstream purpose. Developers should weigh its inherent limitations when selecting use cases, and should evaluate and mitigate accuracy, safety, and fairness concerns specific to each intended use.

### Technical requirements and integration guidance

BioEmu-2 provides open-source inference code with a Python API and command-line interface. See the repository README for installation, configuration, and usage instructions.

BioEmu-2 has been tested using NVIDIA GPUs, with an 80 GB A100 being the smallest tested device. Generated structures should be treated as computational hypotheses rather than experimentally validated measurements and should be assessed using appropriate computational or experimental validation before informing downstream research.

### Known limitations

Outputs are plausible hypotheses, not measurements. The following limitations have not been quantitatively characterized:

- **MSA dependence:** quality is expected to fall for shallow or poor MSAs, including orphan proteins, engineered constructs, and some viral proteins.

- **Training-distribution bias:** PDB, AFDB, and Microsoft-generated MD data over-represent well-studied protein families. Performance on under-represented sequence classes is uncertain.

- **Uncalibrated ensembles:** sampling frequency must not be interpreted as a calibrated Boltzmann probability or quantitative free energy.

- **Ligand scope:** supplied bound ligands are supported, but de novo ligand design and binding-affinity prediction are out of scope.

- **Physicality of large systems:** the model may struggle to produce physically plausible structures for extremely large complexes (over 10k tokens).

### Responsible AI considerations

Combining the model with sequence-generation or autonomous design tools may create risks not assessed for the standalone model.

## Evaluation methodology

The evaluation covers: (i) structure prediction using standard public benchmarks, (ii) multiconformational sampling performance using a curated benchmark, and (iii) molecular dynamics (MD) emulation performance using an MD benchmark suite. It also examines the model's ability to scale to large complexes at inference time.

We use the same model checkpoint to evaluate all benchmarks.

Folding quality is evaluated in terms of root-mean-square deviation (RMSD), TM-score, local distance difference test (lDDT), physicality, and interface DockQ scores on the Boltz-1 test set. Average metrics are compared against a range of state-of-the-art cofolding models.

Multiconformational sampling performance is evaluated on an in-house curated subset of highly similar Protein Data Bank (PDB) structures paired to contrast the effect of cofolding partners, including small-molecule ligands and other polymer chains. Success is determined by whether the model can distinguish and correctly generate samples for the designated states corresponding to the input system definitions.

MD emulation performance is evaluated by sampling a holdout set from the MD datasets described in the Training data section and computing state coverage and weight-distribution divergences. Better coverage and lower divergence indicate better model performance.

The ability to sample large complexes is evaluated on a subset of complexes exceeding 1,500 tokens and compared against reference PDB structures. The selected systems span a range of sizes to demonstrate memory consumption, runtime scaling, and feasibility for multi-GPU deployment. Success is determined by correct overall chain placement, structural physicality, and standard folding quality metrics.

## Training data

The model was trained on the following data.

| **Dataset** | **Modality** | **Train sequences/systems** |
|----|----|----|
| PDB, Boltz-1 preprocessed | Experimental structure | 189,745 |
| MSR_cath2 | MD trajectories | 886 |
| MSR_cath3 | MD trajectories | 4,136 |
| MSR_cath_dimers | MD trajectories | 7,164 |
| MSR_heterodimers | MD trajectories | 1,272 |
| MSR_pdbbind2020 | MD trajectories | 3,036 |
| MSR_pdbbind2020_apo_free | MD trajectories | 2,770 |
| MSR_megasim_merge | MD trajectories | 170 |
| AFDB monomer | AI-predicted structure | 7,231,107 |
| AFDB homodimer multiconformation | AI-predicted structure | 753,271 |
| AFDB heterodimer multiconformation | AI-predicted structure | 22,157 |
| Teddymer | AI-predicted structure | 50,651 |

BioEmu-2 was trained only on publicly available data and data generated in-house by Microsoft; it did not use datasets specially licensed to Microsoft. The training data combines experimental structures from the Protein Data Bank (PDB), AI-predicted structures from the AlphaFold Protein Structure Database (AFDB) and Teddymer, and Microsoft-generated molecular-dynamics trajectories. The AFDB datasets contain AlphaFold-family predictions, and Teddymer is derived from AlphaFold2 predictions.

## Contact

Requests for additional information can be directed to MSFTAIActRequest@microsoft.com.

Authorized representative: Microsoft Ireland Operations Limited, 70 Sir John Rogerson's Quay, Dublin 2, D02 R296, Ireland.
