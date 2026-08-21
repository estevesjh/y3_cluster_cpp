# DES Y3 Cluster Cosmology

Documentation of the DES Y3 cluster-cosmology CosmoSIS pipeline: how to
run the reference analysis, and what every module loads, reads, computes,
and writes.

If the code is already built on your machine, follow {doc}`running`;
otherwise start with {doc}`installation`. Then follow the per-module
pages. Each module page answers, in order: what it computes, which
script implements it, where that script lives, how it is configured,
what DataBlock values it reads and writes, and which module consumes
its outputs.

For development, {doc}`pipeline_organization` explains the additive
`src/pipelines/des_y3` layout and how its reference and alternative
implementations relate to the path-stable production modules.

```{toctree}
:maxdepth: 1
:caption: Getting started

installation
building_macos
pipeline_organization
testing
running
```

```{toctree}
:maxdepth: 1
:caption: Cluster observables

observables/number_counts
modules/richness_mass
observables/shear_halo
observables/second_halo_term
observables/shear_projection
observables/likelihood
```

```{toctree}
:maxdepth: 1
:caption: Cosmology quantities

cosmology/consistency
cosmology/growth_factor
cosmology/cp_camb
cosmology/mf_tinker
cosmology/halo_model
cosmology/halo_bias
cosmology/sigma_crit_inv
```

```{toctree}
:maxdepth: 1
:caption: Selection effects

selection/sel_function
selection/bsel
modules/survey_area
```

```{toctree}
:maxdepth: 1
:caption: Modules

modules/redshift_kernel
```

```{toctree}
:maxdepth: 1
:caption: Mathematical framework

math/index
```

```{toctree}
:maxdepth: 1
:caption: API reference

api/index
```

```{toctree}
:maxdepth: 1
:caption: Variants and history

variants
modules/historical
```

```{toctree}
:maxdepth: 1
:caption: Background chapters

overview
numerics/index
data/index
```

## References

### Papers

The main published reference for the pipeline — the cluster
number-count and population-averaged lensing forward model and the
CosmoSIS/CUBA software framework — is
[DES Cluster et al. 2023](https://ui.adsabs.harvard.edu/abs/2023arXiv230906593A/abstract)
(arXiv:[2309.06593](https://arxiv.org/abs/2309.06593)).

The optical selection-bias and projection-lensing model layered on it
(the `sel_function` / `b_sel_marg` / `bsel` / projection-shear branch)
is [Costanzi et al. 2026, PhRvD 113, 103508](https://ui.adsabs.harvard.edu/abs/2026PhRvD.113j3508C/abstract)
(arXiv:[2604.05833](https://arxiv.org/abs/2604.05833)).

### Archival documents

The LaTeX documents below are the archival, paper-grade record from which
the background chapters are ported. Where this site and a PDF disagree,
the site is the living reference. The sources live under `docs/` (PDFs
are built locally with `pdflatex`; they are deliberately not tracked in
git):

- [pipeline_modules.tex](https://github.com/estevesjh/y3_cluster_cpp/blob/master/docs/pipeline_modules.tex)
  — wired-pipeline algorithms, DataBlock contracts, timing audit,
  quadrature knob cheat-sheet.
- [projection_lensing_paper.tex](https://github.com/estevesjh/y3_cluster_cpp/blob/master/docs/projection_lensing_paper.tex)
  — the optical-projection lensing model.
- [emulator_validation.tex](https://github.com/estevesjh/y3_cluster_cpp/blob/master/docs/emulator_validation.tex)
  — validation of the `cp_camb` linear-$P(k)$ emulator against CAMB.
- [shear1h_radial_factorization.tex](https://github.com/estevesjh/y3_cluster_cpp/blob/master/docs/shear1h_radial_factorization.tex)
  — factorisation strategies for the one-halo shear mass integral.

The selection-model derivations below live in the `RichnessSelection`
repository and are the source of truth for the model chapters here:

- [richness_selection_function.tex](https://github.com/estevesjh/RichnessSelection/blob/main/docs/richness_selection_function.tex)
  — selection functions and richness–mass models.
- [richness_selection.tex](https://github.com/estevesjh/RichnessSelection/blob/main/docs/richness_selection.tex)
  — projection lensing.
- [delta_sigma_prj_derivation.tex](https://github.com/estevesjh/RichnessSelection/blob/main/docs/delta_sigma_prj_derivation.tex)
  — projection lensing.
