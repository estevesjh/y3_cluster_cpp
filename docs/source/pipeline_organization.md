# DES Y3 source organization

New DES Y3 observable implementations live under `src/pipelines/des_y3`.
This namespace complements the production CosmoSIS modules under
`src/modules`; it does not replace them. The reference
`mock_mcmc_buzzard.ini` configuration therefore continues to load the
existing production paths documented in {doc}`running`.

```{admonition} Compatibility boundary
:class: important
The reorganization is additive. Existing module labels, DataBlock sections,
units, array ordering, Python entry points, and compiled-library paths remain
public interfaces. A new implementation becomes a production entry point only
after a separate, explicit cutover.
```

## Directory rule

The maintained namespace is organized as

```text
observable -> integration strategy -> language/backend
```

The implemented tree is:

```text
src/pipelines/des_y3/
├── shared/
│   ├── datablock_models.py
│   ├── full_ltmz_core.py
│   ├── lensing_profiles.py
│   ├── sel_function.py
│   ├── sel_kernels.py
│   └── z_kernel.py
└── observables/
    ├── number_counts/
    │   ├── fast_mass/python/
    │   └── full_ltmz/{python,cpp,cuda}/
    ├── shear_1h2h/
    │   ├── fast_mass/{python,cpp,cuda}/
    │   ├── full_ltmz/{python,cpp,cuda}/
    │   └── radial_series/{python,cpp}/
    └── shear_projection/
        ├── fast_mass/{python,cpp,cuda}/
        └── full_ltmz/cuda/
```

Directories are created only for runnable implementations or substantive
design material. The absence of a backend directory means that backend is not
currently implemented; it is not an empty placeholder.

The `shared` package is the Python model layer used by the namespace
implementations and validators. It reproduces the maintained HOD, HMF,
volume, selection, redshift-kernel, and lensing conventions rather than
letting each observable carry its own copy. As of 2026-08-12,
`shared/sel_function.py` and `src/modules/sel_function/sel_function.py` are
identical staged copies. The production ini still loads the latter, while
namespace code imports the former through `shared/sel_kernels.py`.

## Integration strategies

`full_ltmz`
: Evaluates the selection kernels explicitly over true richness
  $\lambda_{\rm true}$, log mass $\ln M$, and true redshift $z$. It is the
  accuracy reference and may use fixed GL, adaptive Cuhre, or PAGANI depending
  on the backend. Projection shear also retains the angular coordinate needed
  by its observable definition.

`fast_mass`
: Contracts the redshift-dependent population weight on fixed GL nodes before
  the final mass/radial operator. For number counts, the operator is $f=1$.
  For shear, the redshift contraction is exact for the implemented observable;
  `fast_mass` does not by itself mean frozen redshift physics. A backend whose
  physics is frozen is labelled explicitly.

`radial_series`
: Contracts redshift exactly, computes the sample-dependent moments of
  $y=\ln r_s(M)$, and evaluates an offline radial expansion through at most
  $\ell=3$. The reusable unit-profile tables live in `data/radial_series` and
  are never regenerated during an MCMC sample. This strategy is implemented
  for one-halo miscentred shear; the projection counterpart remains planned.

## Implemented observable matrix

| Observable | Strategy | Backends | Role |
|---|---|---|---|
| Number counts | `full_ltmz` | Python, C++, CUDA | Explicit-selection accuracy references |
| Number counts | `fast_mass` | Python; production C++ by identity | Fast redshift-contracted calculation |
| One-halo miscentred shear | `full_ltmz` | Python, C++, CUDA | Explicit-selection accuracy references |
| One-halo miscentred shear | `fast_mass` | Python, C++ | Exact mass-sum path; C++ is bitwise-equivalent to production |
| Traditional 1h+2h max model | `fast_mass` | Python, C++, CUDA | Implemented variant; two-halo profile has an open debugging flag |
| One-halo miscentred shear | `radial_series` | Python, C++ | Candidate moment-expansion implementation |
| Projection shear | `full_ltmz` | CUDA | Adaptive reference with an open wall-edge convergence study |
| Projection shear | `fast_mass` | Python, C++, CUDA | Exact-$z$ Python/C++; CUDA reproduces the frozen production machinery |
| Projection shear | `radial_series` | — | Planned |

The per-implementation `README.md` files record DataBlock contracts,
quadrature, validation tolerances, timing, and known limitations. In
particular, the traditional max model is not promoted while its
$\Delta\Sigma_{hh}$ input is under investigation, and the projection
`full_ltmz` CUDA backend has not closed its innermost-radius convergence
study.

## Production and reference choices

The currently selected reference backend for day-to-day comparisons is the
`fast_mass` C++ cell for each observable:

| Observable | Selected backend |
|---|---|
| Number counts | Production `NumCountsSel.so` (the same algorithm as `fast_mass`) |
| One-halo miscentred shear | `Shear1hFastMass.so` |
| Traditional 1h+2h max model | `Shear1h2hMax.so` |
| Projection shear | `ShearPrjFastMass.so` |

On GPU nodes, `Shear1h2hMaxGpu.so` and `ShearPrjFrozenGpu.so` provide CUDA
alternatives for the max-model and frozen-projection arms respectively.
These choices identify comparison cells; they do not change which modules the
pinned production ini loads.

## Validation policy

Numerical accuracy is quoted against the corresponding explicit
`full_ltmz` fiducial after that fiducial has passed its own convergence and
cross-backend checks. Agreement with a production module is reported
separately as an algorithm-identity or compatibility check, because production
can intentionally include selection tabulation or frozen-physics
approximations.

The namespace-wide report is
`src/pipelines/des_y3/validate_against_fiducial.py`. Targeted validators live
beside each implementation. The implementation matrix and measured results
are maintained in `src/pipelines/des_y3/README.md`; the numerical algorithms
are described in {doc}`numerics/index`.

## What remains under `src/modules`

The reference ini continues to load the production selection, counts,
one-halo shear, selection-bias, and projection entry points from their stable
locations:

| Stage | Stable production source |
|---|---|
| Selection tensor | `src/modules/sel_function/sel_function.py` |
| Number counts | `src/modules/num_counts_sel/NumCounts.cc` |
| One-halo miscentred shear | `src/modules/num_counts_sel/Shear1hMis.cc` |
| Selection-bias marginalization | `src/modules/b_sel_marg_cpu/BSelMargIntegrand.cc` |
| Projection shear | `src/modules/sigma_prj_cpu/ShearPrjFrozenPhysics.cc` |

The path-stable support stages `cp_camb`, `halo_model`,
`average_sigma_crit_inv`, and `likelihoods` also remain outside the new
observable tree. Shared C++ models stay under `src/models` and infrastructure
under `src/utils`; the new layout does not reorganize either layer.
