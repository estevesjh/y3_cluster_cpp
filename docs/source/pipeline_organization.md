# DES Y3 source organization

New DES Y3 observable implementations live under `src/pipelines/des_y3`.
This namespace complements the production CosmoSIS modules under
`src/modules`; it does not replace them. The reference `des_y3.ini`
({doc}`running`) loads the `0d` C++ backends from here; the DES Y1
`mock_mcmc_buzzard.ini` continues to load the `src/modules` paths
({doc}`variants`).

```{admonition} Compatibility boundary
:class: important
The reorganization is additive. Existing module labels, DataBlock sections,
units, array ordering, Python entry points, and compiled-library paths remain
public interfaces. A new implementation becomes a production entry point only
after a separate, explicit cutover.
```

## Directory rule

The maintained namespaces are organized as

```text
observable -> integration strategy -> language/backend
```

The implemented tree is:

```text
src/pipelines/
├── shared/                         # reusable numerical/datablock layer
├── cosmology/                      # halo-model physics
├── systematics/                    # canonical systematics layer
│   ├── selection_richness/python/  # sel_function, sel_kernels
│   ├── selection_bias/{python,cpp,cuda/3d}/  # bsel, BSelBins, PAGANI b_sel_marg
│   ├── selection_function/         # prj_params (EMG coefficients)
│   ├── costanzi_bprj/{python,cpp}/ # B_prj(R) max-model correction
│   ├── boost_factor/               # McClintock+19 B(R) (published, unconsumed)
│   └── shear_prj/cpp/              # ShearPrjCore + frozen twins
├── buzzard/likelihoods/            # likelihood_cp.py
└── des_y3/                         # survey observable compositions
    ├── number_counts/
    │   ├── cpp/{0d,3d}/
    │   ├── cuda/3d/
    │   └── python/0d/
    ├── shear_1h2h/
    │   ├── cpp/{0d,3d}/
    │   ├── cuda/{0d,3d}/
    │   └── python/0d/
    └── shear_projection/
        ├── cpp/{0d,2d}/
        ├── cuda/{0d,3d}/
        └── python/0d/
```

Directories are created only for runnable implementations or substantive
design material. The absence of a backend directory means that backend is not
currently implemented; it is not an empty placeholder.

The `shared` package is the lower-level Python model layer used by the
implementations and validators. It provides the HOD, HMF, volume,
redshift-kernel, and lensing utilities. Selection-systematics entry points
are now owned by `systematics/`; the older files under `shared/`,
`cosmology/`, and `src/models/` remain compatibility references and are not
deleted by this migration.

## Integration strategies (adaptive-dimension tags)

Strategy folders count **adaptive integration dimensions** (Cuhre/Vegas
on CPU, PAGANI on GPU) — the quantity that drives per-sample cost. `0d`
means no adaptive integration at all: fixed Gauss--Legendre sums and
offline tables, fast and MCMC-viable. The maximum-dimension folder of
each observable is always the adaptive reference, and every lower folder
is a documented dimension reduction from it. Fixed-GL and offline
dimensions do not count, whatever their number.

`3d`
: Adaptive quadrature over the full explicit volume — true richness
  $\lambda_{\rm true}$, log mass $\ln M$, and true redshift $z$ for
  counts and one-halo/max shear (formerly the `full_ltmz` C++/CUDA
  backends), or the fully coupled $(\theta, z, \ln M)$ PAGANI diagnostic
  for projection shear. The accuracy reference of each observable.

`2d` (projection shear only)
: `ShearPrjCuhre`: the outer $\theta$ integral is feature-split
  fixed-GL, adaptive Cuhre/Vegas handles the inner two dimensions
  $(z, \ln M)$. Two adaptive dimensions — the tag counts only those.

`0d`
: Everything with zero adaptive dimensions, merged per observable:
  the $S_{ij}$-tabulated fixed-GL sums (formerly `fast_mass`), the
  explicit fixed-GL Python references (formerly the `full_ltmz` Python
  backends), the offline $U_\ell$ radial-series expansion (formerly
  `radial_series`, one-halo shear only; tables in `data/radial_series`),
  and for projection shear the region-split fixed-GL path (exact-$z$
  core plus the frozen-physics CUDA port).

Only the folders carry the new names: module labels, class names, file
names, and DataBlock output sections keep their historical strings
(`NumCounts3d`, `shear1h_gl/vals`, `numcounts_explicit_gl.py`,
`shared/explicit_grid_core.py`, ...), so existing ini files and validators
reference them unchanged. The only file renames are the merged `0d`
validators, qualified as `validate_fast_vs_production.py` and
`validate_explicit_vs_production.py`.

## Implemented observable matrix

| Dims | Observable | Backends | Role |
|---|---|---|---|
| `3d` | Number counts | C++, CUDA | Adaptive explicit-selection accuracy references |
| `0d` | Number counts | Python (explicit GL + fast), C++ (production by identity) | Explicit 3-dim GL reference and the $S_{ij}$-tabulated 2-dim GL fast sum |
| `3d` | One-halo miscentred shear | C++, CUDA | Adaptive explicit-selection accuracy references |
| `0d` | One-halo miscentred shear | Python, C++ | Explicit 3-dim GL reference, the exact $z$-contracted 1-dim GL mass sum (C++ bitwise-equivalent to production), and the $U_\ell$ moment expansion |
| `3d` | Traditional 1h+2h max model | C++ | Adaptive explicit reference for the max model |
| `0d` | Traditional 1h+2h max model | Python, C++, CUDA | $z$-resolved 2-dim GL sum; two-halo profile has an open debugging flag |
| `3d` | Projection shear | CUDA | Fully-coupled adaptive diagnostic with an open wall-edge convergence study |
| `2d` | Projection shear | C++ | `ShearPrjCuhre`: feature-split $\theta$ GL, adaptive $(z, \ln M)$ |
| `0d` | Projection shear | Python, C++, CUDA | Exact-$z$ region-split 3-dim GL Python/C++; CUDA reproduces the frozen production machinery |

The per-implementation `README.md` files record DataBlock contracts,
quadrature, validation tolerances, timing, and known limitations. In
particular, the traditional max model is not promoted while its
$\Delta\Sigma_{hh}$ input is under investigation, and the projection
`3d` CUDA backend has not closed its innermost-radius convergence
study.

## Production and reference choices

The currently selected reference backend for day-to-day comparisons is the
fast (`0d`, fixed-GL) C++ cell for each observable:

| Observable | Selected backend |
|---|---|
| Number counts | Production `NumCountsSel.so` (the same algorithm as the `0d` fast backend) |
| One-halo miscentred shear | `Shear1hGl.so` |
| Traditional 1h+2h max model | `Shear1h2hMax.so` |
| Projection shear | `ShearPrjGl.so` |

On GPU nodes, `Shear1h2hMaxGpu.so` and `ShearPrjFrozenGpu.so` provide CUDA
alternatives for the max-model and frozen-projection arms respectively.
These choices identify comparison cells; they do not change which modules the
pinned production ini loads.

## Validation policy

Numerical accuracy is quoted against the corresponding explicit
`3d` fiducial after that fiducial has passed its own convergence and
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

The DES Y1 ini continues to load the production selection, counts,
one-halo shear, selection-bias, and projection entry points from their stable
locations:

| Stage | Stable production source |
|---|---|
| Selection tensor | `src/modules/sel_function/sel_function.py` (shim over `systematics/selection_richness`) |
| Number counts | `src/modules/num_counts_sel/NumCounts.cc` (`NumCountsSel`) |
| One-halo miscentred shear | `src/modules/num_counts_sel/Shear1hMis.cc` (`Shear1hMisSel`) |
| Selection-bias marginalization | `src/modules/b_sel_marg_cpu/BSelMargIntegrand.cc` (also used by `des_y3.ini`) |
| Projection shear | `src/modules/sigma_prj_cpu/ShearPrjFrozenPhysics.cc` |

The path-stable support stages `cp_camb`, `halo_model`, and
`average_sigma_crit_inv` also remain outside the new observable tree;
the likelihood lives in `src/pipelines/buzzard/likelihoods/`. Shared C++ models stay under `src/models` and infrastructure
under `src/utils`; the new layout does not reorganize either layer.
