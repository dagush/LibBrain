# Neuroreduce

A Python library for dimensionality reduction of fMRI BOLD signals and
structural connectivity data, developed for computational neuroscience research.

## What it does

Neuroreduce provides a unified interface for a family of techniques that
reduce high-dimensional brain signals (N parcels × T timepoints) into a
low-dimensional representation (k dimensions × T timepoints), making it
easier to analyse brain dynamics, compare methods, and reproduce results
across studies.

All methods share the same interface — you instantiate a reducer, call
`fit()`, and then `transform()` any BOLD signal:

```python
from Neuroreduce import PCAReducer, CHARMReducer, ConnectomeHarmonicsReducer

reducer = PCAReducer(k=10)
reducer.fit(X)                  # X: (N, T)
Z = reducer.transform(X)        # Z: (k, T)
W = reducer.get_basis()         # W: (N, k)  — spatial modes
```

## Implemented methods

| Class | Method | Input | Reference |
|---|---|---|---|
| `PCAReducer` | Principal Component Analysis | BOLD | — |
| `CHARMReducer` | Complex HARMonics (BOLD) | BOLD timeseries | Deco et al. (2025) |
| `CHARMSCReducer` | Complex HARMonics (geometry) | Parcel coordinates | Deco et al. (2025) |
| `ConnectomeHarmonicsReducer` | Connectome Harmonics | SC matrix | Atasoy et al. (2016) |
| `FunctionalHarmonicsReducer` | Functional Harmonics | BOLD (FC computed internally) | Glomb et al. (2021) |

## Conventions

- BOLD input: `np.ndarray`, shape `(N, T)` — parcels × timepoints
- SC input: `np.ndarray`, shape `(N, N)`
- Output: `np.ndarray`, shape `(k, T)` — reduced dimensions × timepoints
- Basis: `np.ndarray`, shape `(N, k)` — spatial modes

## Analysis utilities

- `PCASpectrumAnalyzer` — explained variance spectrum and component selection
- `CHARMAnalysis` — metastability, lagged FC, trophic coherence, SVM classification
- `HarmonicAnalysis` — RSN projection, mutual information, reconstruction error
- `ECMPlotter` — edge-centric metastability comparison plots

## References

- Atasoy et al. (2016). *Human brain networks function in connectome-specific
  harmonic waves.* Nature Communications 7, 10340.
- Glomb et al. (2021). *Functional harmonics reveal multi-dimensional basis
  functions underlying cortical organization.* Cell Reports 36(8).
- Deco, Sanz Perl & Kringelbach (2025). *Complex harmonics reveal low-dimensional
  manifolds of critical brain dynamics.* Physical Review E 111(1).
