# DataLoaders

This directory contains the dataset-specific loaders used throughout the project. Their main purpose is to provide a uniform interface to heterogeneous neuroimaging datasets while hiding the underlying storage format and directory organization.

The rest of the library never interacts directly with dataset files. Instead, all analyses, simulations and machine learning algorithms communicate with a common DataLoader interface, making it straightforward to switch between datasets without modifying downstream code.

⸻

## Design philosophy

Neuroimaging datasets differ substantially in:

* folder organization;
* file formats (.mat, .npy, HDF5, etc.);
* metadata conventions;
* subject identifiers;
* available modalities;
* preprocessing pipelines.

Rather than adapting every analysis to each dataset individually, every dataset is represented by a dedicated DataLoader class inheriting from the common DataLoader superclass.

Each loader is responsible for:

* locating the dataset files;
* loading all required modalities;
* organizing subjects into diagnostic groups;
* matching imaging data with subject metadata;
* exposing the data through a common API.

As a consequence, higher-level algorithms are completely independent of the original dataset organization.

⸻

## Example datasets (current)

### ADNI_B

Original cross-sectional ADNI dataset.

Provides:

* resting-state fMRI timeseries
* Amyloid PET burden
* Tau PET burden
* subject metadata
* diagnostic groups

This loader serves as the original implementation upon which most subsequent loaders are based.

⸻

### ADNI_G

Cross-sectional ADNI dataset using the DBS80 parcellation.

Main differences with respect to ADNI_B:

* one MATLAB file per subject;
* resting-state timeseries stored in the ts variable;
* DBS80 atlas;
* identical PET organization to ADNI_B.

The public API remains identical to ADNI_B.

⸻

### ADNI_Long

Longitudinal ADNI dataset containing two imaging sessions per subject.

Available modalities include:

* resting-state fMRI
* Amyloid PET
* Tau PET
* regional gray matter volume (GMV)
* longitudinal GMV annualized percentage change (GMV-APC)

The loader supports multiple:

* parcellations;
* PET preprocessing options;
* anatomical templates;
* fMRI preprocessing pipelines.

Whenever several versions are available, the default configuration uses

* Partial Volume Correction (PVC),
* Centiloid (CL) Amyloid PET,
* Schaefer400 parcellation,
* template_all anatomical template.

Unlike the cross-sectional datasets, this loader returns data organized by subject and imaging session.

### Others

There is a number of other dataLoaders that I use, ventually, for other dataset, including:

* HCP
* MEG
* ADNI A, E, etc...

⸻

## Common API

All loaders expose a common interface regardless of the original dataset.

Typical methods include:

loader = ADNI_G()
loader.get_subjectData(subjectID)
loader.get_groupLabels()
loader.get_subject_count()
loader.N()
loader.TR()
loader.name()

Higher-level analyses should rely exclusively on these methods instead of accessing dataset files directly.

⸻

## Extending the framework

Adding a new dataset only requires implementing a new class derived from DataLoader.

A new loader should be responsible for:

1. defining the dataset paths;
2. loading metadata;
3. loading imaging modalities;
4. matching metadata and imaging files;
5. organizing subjects into groups;
6. exposing the standard API.

No modifications to the remaining library should be necessary.

⸻

## Design goals

The DataLoader framework was designed with the following objectives:

* Dataset independence for downstream analyses.
* Reproducibility, by encapsulating dataset-specific preprocessing choices.
* Maintainability, through a common object-oriented interface.
* Extensibility, allowing new datasets to be incorporated with minimal effort.
* Consistency, ensuring identical access patterns across all neuroimaging cohorts.

This architecture allows statistical analyses, computational models, machine learning algorithms and visualization tools to operate transparently across multiple neuroimaging datasets without requiring dataset-specific code.

(Code by Gustavo Patow, and many many sources, which I quote when I remember to)
