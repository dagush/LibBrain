"""
Deco2025_CHARM_SC
-------
Standalone project implementing the geometry-based brain models from:

    Deco, G., Sanz Perl, Y., & Kringelbach, M. L. (2025). Complex harmonics
    reveal low-dimensional manifolds of critical brain dynamics.
    Physical Review E, 111(1). https://doi.org/10.1103/physreve.111.014410

NOT part of Neuroreduce — this is a generative modelling project, not
dimensionality reduction. The two share no code.

Modules
-------
geometry    : HARM and CHARM-SC diffusion models on parcel geometry
simulation  : Random-walk BOLD generator + FC computation
empirical   : Empirical parcel state distributions from BOLD events
"""