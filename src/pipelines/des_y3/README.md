# DES Y3 observable implementations

This directory contains the maintained DES Y3 implementations. The layout is

```text
observable / numerical strategy / language or backend
```

The strategy-level READMEs are the main entry points. Each one explains the
mathematics first and then describes the Python, C++, and CUDA implementations
that realize the same observable or approximation.

## The three model families

### Cluster number counts

The count in observed richness bin \(i\) and observed-redshift bin \(j\) is

$$
N_{ij} = \int dz\,d\ln M\,d\lambda_{\rm tr}\;
n(M,z)\frac{dV}{d\Omega dz}\Omega(z)
S_j(z)S_i(\lambda_{\rm tr},z)
P_{\rm HOD}(\lambda_{\rm tr}\mid M,z).
$$

The kernels describe the halo population, the richness--mass relation, the
mapping from true richness to observed richness, and the photometric-redshift
selection. The count model has no radial coordinate. Start with
[number-count strategies](observables/number_counts/README.md).

### One-halo and traditional one-plus-two-halo shear

The maintained one-halo shear uses a production miscentred mixture,

$$
\Delta\Sigma_{1h}(R,M,z) =
(1-f_{\rm mis})\Delta\Sigma_{\rm cen}(R,M,z)
 + f_{\rm mis}\Delta\Sigma_{\rm mis}(R,M,z).
$$

The lensing observable inserts this profile into the count integral and
weights it by the inverse critical surface density,

$$
O_{ij}(R)=\int dz\,d\ln M\,d\lambda_{\rm tr}\;
\mathcal W_{ij}(M,z,\lambda_{\rm tr})
\Sigma_{\rm crit}^{-1}(z)\Delta\Sigma_{1h}(R,M,z).
$$

The `fast_mass` directory also contains the traditional max model used for a
one-plus-two-halo comparison,

$$
\Delta\Sigma_{\rm max}(R,M,z)=
\max\left[\Delta\Sigma_{1h}(R,M,z),
b(M,z)\Delta\Sigma_{hh}(R,z)\right].
$$

This is separate from the selection-affected projection observable below.
Start with [one-halo shear strategies](observables/shear_1h2h/README.md).

### Projection shear

Projection shear models correlated line-of-sight structure around the lens.
The exact implementation keeps the angular offset \(\theta\), true redshift,
and halo mass coupled through the selection-affected bias, nonlinear
correlation function, slab exclusion, and miscentred profile:

$$
\Delta\Sigma_{\rm prj,ij}(R)=
\int d\theta\;2\pi\sin\theta\int d\ln M\;
\left[
W^{\rm rnd}_{ij}(M)
+b_{{\rm sel},ij}(\theta)W^{\rm cl}_{ij}(\theta,M)
\right]
\Delta\Sigma_{\rm mis}(R,\tau;M),
\qquad
\tau=\theta D_A(z_{{\rm ob},j}).
$$

The weights \(W^{\rm rnd}_{ij}\) and \(W^{\rm cl}_{ij}\) are defined in the
projection strategy README. They contain the random and clustered channels,
the selection-dependent \(b_{{\rm sel},ij}(\theta)\), the line-of-sight
correlation, photo-\(z\) weighting, and the exclusion geometry. These weights
are distinct from the number-count \(S_i,S_j\) kernels. The path is named
`shear_projection`; module names use `ShearPrj`. Start with
[projection-shear strategies](observables/shear_projection/README.md).

## Numerical strategies

| Strategy | Numerical idea | Main use |
| --- | --- | --- |
| `full_ltmz` | Keep \((\lambda_{\rm tr},z,\ln M)\) explicit. Use fixed Gauss--Legendre, adaptive Cuhre, or adaptive PAGANI depending on backend. | Independent reference and convergence checks. |
| `fast_mass` | Build the redshift/selection weight first, then integrate or sum over mass. For projection, retain the coupled angular geometry in the shared core. | Maintained CPU path and fast Python reference. |
| `radial_series` | Expand a fixed radial profile around the population mean of \(y=\ln r_s\), with precomputed \(U_\ell\) functions. | One-halo shear candidate only. |

The first two strategies are exact representations of their stated
observable definitions, up to quadrature and interpolation. `radial_series`
is a fixed-profile approximation: its current table assumes \(c=4\) for all
mass and redshift and therefore must not be compared with a varying
production concentration relation as if it were exact.

## Language and backend inventory

| Observable / strategy | Python | C++ | CUDA |
| --- | --- | --- | --- |
| Number counts / `full_ltmz` | Fixed-GL reference; adaptive mass reference in shared code | Adaptive Cuhre | Adaptive PAGANI |
| Number counts / `fast_mass` | Importable production-algorithm replica | Thin `SelGLCore` wrapper, production identity | Not warranted for this 1-D mass contraction |
| One-halo shear / `full_ltmz` | Fixed-GL reference; adaptive mass reference in shared code | Adaptive Cuhre | Adaptive PAGANI |
| One-halo shear / `fast_mass` | Production-algorithm replica | Thin `SelGLCore` wrapper | CUDA max-model contraction |
| One-halo shear / `radial_series` | Table generator and evaluator | Table reader and evaluator | Not implemented |
| Projection shear / `full_ltmz` | Not implemented in this namespace | Not implemented in this namespace | Adaptive PAGANI |
| Projection shear / `fast_mass` | Exact-\(z\) reference | Shared exact core | Frozen-production GPU contraction |

The CUDA rows are not always the same mathematical observable as the CPU
reference: the projection GPU path intentionally ports the frozen production
algorithm, while the projection full-ltmz GPU path is the independent adaptive
reference.

## Folder inventory

```text
src/pipelines/
├── shared/                     survey-agnostic selection/mass-weight layer
│   ├── datablock_models.py     HMF, volume, area, lensing weights, GL nodes
│   ├── full_ltmz_core.py       explicit selection contraction
│   ├── lensing_profiles.py     production profile adapters
│   ├── sel_function.py         maintained richness/photo-z selection tensor
│   ├── sel_kernels.py          maintained richness/photo-z kernels
│   └── z_kernel.py             projection photo-z kernel
├── cosmology/                  survey-agnostic halo-model physics
│   ├── halo_model.py           1h+2h lensing, Tinker et al. bias
│   ├── concentration.py        mass-concentration relations
│   ├── nfw_model.py            analytic Wright & Brainerd NFW
│   ├── prj_params.py           frozen Costanzi-2026 EMG coefficients
│   ├── bsel.py                 selection-bias closure
│   └── sigma_crit_inv.py       Sigma_crit^-1(z_lens, R)
└── des_y3/
    └── observables/
        ├── README.md
        ├── number_counts/
        │   ├── README.md
        │   ├── fast_mass/README.md
        │   └── full_ltmz/README.md
        ├── shear_1h2h/
        │   ├── README.md
        │   ├── fast_mass/README.md
        │   ├── full_ltmz/README.md
        │   └── radial_series/README.md
        └── shear_projection/
            ├── README.md
            ├── fast_mass/README.md
            └── full_ltmz/README.md
```

`shared/` and `cosmology/` are siblings of `des_y3/`, not nested under it —
both predate this namespace conceptually (they mirror production model
classes 1:1) and are meant to be reused as-is by future non-Y3 pipelines;
see `src/pipelines/cosmology/README.md` for that directory's contents and
why it copies rather than moves its `y3_buzzard/` sources.

Each strategy README contains a second table that inventories the actual
source files in its Python, C++, and CUDA directories. A missing language
directory means that backend is not implemented, not that documentation was
omitted.

## Reference pipeline choices

The maintained smoke pipeline uses the following CPU backends:

| Observable | Strategy | Backend |
| --- | --- | --- |
| Number counts | `fast_mass` | C++ `NumCountsSel.so` identity path |
| One-halo miscentred shear | `fast_mass` | C++ `Shear1hFastMass.so`, identity with production |
| Projection shear | `fast_mass` | C++ `ShearPrjFastMass.so`, exact-\(z\) reference |

The traditional max model and the frozen projection GPU path are optional
performance variants, not replacements for the exact reference definitions.

## Production reference stages (maintenance manifest)

Everything in this namespace is validated against six existing production
stages, none of which move or change as part of any work here (see
"Rules that bite if ignored" in the top-level `CLAUDE.md`). This is the
current validation baseline, audited against `y3_cluster_cpp` commit
`29949bdfecda00eea938eae194ac9f3a1d5fad1e` (2026-08-11) and the external
pipeline pin below; **the module source is authoritative for exact
DataBlock keys, shapes, and units** — treat this table as a map of where
to look, not a substitute for reading `NumCounts.cc`/`Shear1hMis.cc`/etc.
directly.

| Pipeline stage | Language | Source | Build target / runtime file |
| --- | --- | --- | --- |
| `sel_function` | Python | `src/modules/sel_function/sel_function.py` | No CMake target; run in place |
| `NumCountsSel` | C++ | `NumCounts.cc` driving `NumCountsSelGL` (`n_operator_sel_gl_t.hh`) | `src/modules/num_counts_sel/` → `NumCountsSel.so` |
| `Shear1hMisSel` | C++ | `Shear1hMis.cc` driving `Shear1hMisSelGL` (`n_operator_sel_gl_t.hh`) | `src/modules/num_counts_sel/` → `Shear1hMisSel.so` |
| `b_sel_marg` | C++ | `BSelMargIntegrand.cc` driving `P_operator` (`p_operator_t.hh`) | `src/modules/b_sel_marg_cpu/` → `BSelMargIntegrand.so` |
| `bsel` | Python | `y3_buzzard/bsel.py` (physics copy: `src/pipelines/cosmology/bsel.py`) | No CMake target; run in place |
| `shear_prj_frozen_physics` | C++ | `ShearPrjFrozenPhysics.cc` driving `ShearPrjFrozenPhysics` (`sigma_prj_frozen_t.hh`) | `src/modules/sigma_prj_cpu/` → `ShearPrjFrozenPhysics.so` |

The pinned module order for the authoritative external pipeline is
`consistency GrowthFactor cp_camb MfTinker halo_model
average_sigma_crit_inv sel_function NumCountsSel Shear1hMisSel
b_sel_marg bsel shear_prj_frozen_physics likelihoods`. `sel_function`,
`b_sel_marg`, and `bsel` are prerequisite selection stages, not
competing integration strategies for the final count or shear observable.

External pipeline pin (run-management repo `estevesjh/des-nersc-cluster-scripts`,
commit `9fd24ddc075d394af4e20241bda716ac4d529fcb`):
`cosmosis-models/mock_mcmc_buzzard.ini` (pipeline) and
`cosmosis-models/mock_mcmc_widePlanck_values.ini` (values; no separate
priors file — the values file carries the sampled boxes). The full commit
hash, not the branch name, defines the pin.

Configuration-grid contract at that pin (setup-time options, not
per-sample DataBlock values):

| Stage | Grid |
| --- | --- |
| `sel_function` | Twelve equal-length `lam_min`/`lam_max`/`zob_min`/`zob_max`/`sigma_z`; shared `(lnM,z)` table, `n_lnm=192`, `n_z_shared=64` |
| `NumCountsSel` | `bin_index=0..11`; fixed-GL `n_lnm=96`, `n_z=64`; output length 12 |
| `Shear1hMisSel` | 12 `bin_index` × 15 `r_perp` (bin slow, radius fast); output length 180; `method=exact` |
| `b_sel_marg` | Zipped 12-point `(zo_low,zo_high,lambda_bin)` wall, richness fastest; length 12 per section |
| `bsel` | `lob=(25,37.5,52.5,130)`, `zob=(0.275,0.425,0.575)`, legacy 32-node `theta` |
| `shear_prj_frozen_physics` | Zipped 180-point `(lambda_bin,zo_low,zo_high,radii)` wall; 12×15, bin slow/radius fast |

This snapshot supersedes the retired `docs/module_reorganization_plan.md`
(design proposal, approved 2026-08-11, fully implemented — its Phase 1/2
process content is no longer forward-looking) and
`docs/des_y3_maintenance_manifest.md` (the original, more detailed
per-stage DataBlock I/O audit); both are kept in `archive/` for history,
not tracked by git.

## Known follow-up work

Tracked backlog surfaced by review, not yet implemented in this namespace:

- **Upstream CUDA sync**: reuse the device headers/kernels from
  `marcpaterno/y3_cluster_cpp#3` (`b_sel.cuh`, `emg_des_t.cuh`,
  `gamma_1h_nfw.cuh`, `mor_shifted_poisson_t.cuh`, `nfw_dsigma_mis.cuh`,
  `nfw_sigma_mis.cuh`, `p_operator_gpu_t.cuh`, `sigma_photoz_table_t.cuh`,
  the `gpu_prj_costanzi2026` module) in this namespace's CUDA drivers,
  once merged. Exception: keep the CPU `bsel`/`BSelMargIntegrand`
  fixed-GL path for anything needing b_sel — do not adopt `b_sel.cuh`,
  it's ~10^3x slower for this integrand.
- **Concentration**: `radial_series`'s offline `U_ell` table is valid for
  any concentration (it's purely a function of the dimensionless
  `x`/`x_mis` coordinates), but `nfw_profile_family.py`'s
  `r_s_of_lnM`/`y_of_lnM` and `lensing_profiles.py`'s `CONC=4.0` hardcode
  the *consumer's* M→r_s mapping. Plumb the real per-sample concentration
  (e.g. `haloModel/concentration`) into that mapping instead.
- **`generate_radial_series_tables.py`**: remove `sbar_off_grid`/the
  cumulative off-center integration (biased below `R_min`); use the
  existing `data/nfw_off_center` lookup/analytic profile directly.
  Re-benchmark the finite-difference stencils against spline
  differentiation or direct analytic derivatives.
- **`validate_radial_series.py`**: check the centered case against the
  closed-form analytic NFW at high precision, and the miscentered case
  against one hardcoded high-precision reference grid value.
- **`shear1h_radial_series_t.hh`**: make `ell_max` a CosmoSIS-config int
  (already true in the Python backend); document `rho_mult`/`mu2_`/`mu3_`.
- Drop the unused `#include "models/p_operator_cuhre_t.hh"` in
  `Shear1h2hMax.cc` (it only needs the `gl_nodes` helper from that
  header) once a smaller GL-utility header is available.
- **HMF**: not yet ported into `src/pipelines/cosmology/`; explicitly
  deferred (see that directory's README).

## Unified timing and precision table

These measurements are from the pinned 12-bin DES Y3 wall at the fiducial
point. Counts use 12 bins; one-halo shear uses 12 bins × 10 radii; projection
uses 180 wall points. Times are milliseconds per sample unless stated
otherwise. “Error” always means error against the reference named in the
comparison column; identity numbers are separate backend-equivalence checks.

| Observable | Strategy / backend | Language / hardware | Cost | Precision or comparison |
| --- | --- | --- | ---: | --- |
| Counts | `full_ltmz`, adaptive | Python | 25 s | Fiducial reference; reported error \(\le 10^{-6}\) |
| Counts | `full_ltmz`, fixed GL | Python | 83 ms | \(3.5\times10^{-5}\) vs adaptive |
| Counts | `full_ltmz`, Cuhre | C++ CPU | 3.1 s | \(4.9\times10^{-4}\) vs fixed-GL reference |
| Counts | `full_ltmz`, PAGANI | CUDA / A100 | 2.0 s | \(5.1\times10^{-4}\) vs fixed-GL reference |
| Counts | `fast_mass` | Python | 5 ms | \(7.6\times10^{-4}\) vs adaptive; \(2.4\times10^{-15}\) vs production |
| Counts | `fast_mass` | C++ CPU | 6 ms | \(7.6\times10^{-4}\) vs adaptive; production identity |
| One-halo shear | `full_ltmz`, adaptive | Python | 35 s | Fiducial reference; reported error \(\le 10^{-6}\) |
| One-halo shear | `full_ltmz`, fixed GL | Python | 149 ms | \(4.9\times10^{-5}\) vs adaptive |
| One-halo shear | `full_ltmz`, Cuhre | C++ CPU | 51 s | \(3.3\times10^{-4}\) vs Python reference |
| One-halo shear | `full_ltmz`, PAGANI | CUDA / A100 | 32 s | \(3.4\times10^{-4}\) vs C++ reference |
| One-halo shear | `fast_mass` | Python | 74 ms | \(8.4\times10^{-4}\) vs adaptive; \(3.1\times10^{-15}\) vs production |
| One-halo shear | `fast_mass` | C++ CPU | 9 ms | \(8.4\times10^{-4}\) vs adaptive; production identity |
| One-halo shear | `radial_series` | Python | 6 ms | \(3.7\times10^{-3}\) for its fixed-profile fiducial |
| One-halo shear | `radial_series` | C++ CPU | 7 ms | \(3.7\times10^{-3}\) plus \(1.6\times10^{-4}\) interpolation difference |
| Max model | `fast_mass` | C++ CPU | 11 ms | \(8.3\times10^{-4}\) vs adaptive; \(6.0\times10^{-15}\) vs Python |
| Max model | `fast_mass` | CUDA / A100 | 8 ms | \(6.4\times10^{-15}\) vs C++ twin |
| Projection | `full_ltmz`, PAGANI | CUDA / A100 | 95 s | Median \(9.5\times10^{-4}\), maximum \(2.2\%\) vs refined GL; convergence open |
| Projection | `fast_mass`, exact \(z\) | Python | 270 ms | \(1.6\times10^{-11}\) vs exact evaluator; \(5.5\times10^{-5}\) vs frozen production |
| Projection | `fast_mass`, exact \(z\) | C++ CPU | 154 ms | \(9.9\times10^{-12}\) vs exact evaluator |
| Projection | frozen GPU path | CUDA / A100 | 8.3 ms | \(1.5\times10^{-11}\) vs frozen production; not the exact-\(z\) reference |

The table is a benchmark record, not a universal performance guarantee. CPU
timings depend on the Perlmutter node and build; GPU timings depend on the
device and occupancy. Full validation notes and the exact comparison policy
are kept in the strategy READMEs and the validation documents under `docs/`.
