# DES Y3 observable implementations

This directory contains the maintained DES Y3 implementations for cluster
number counts, one-halo lensing, traditional one-plus-two-halo lensing, and
selection-affected projection lensing.

The directory is organized as:

```text
observable / numerical strategy / language or backend
```

Strategy folders count **adaptive integration dimensions** — the quantity
that drives per-sample computing time. `0d` means no adaptive integration at
all (fixed Gauss--Legendre sums and offline tables — fast, MCMC-viable);
`Nd` means N adaptive (Cuhre/Vegas/PAGANI) dimensions. The
maximum-dimension folder of each observable is always the adaptive
reference, and every lower folder is a documented dimension reduction from
it. Code identifiers follow the same quadrature scheme: `SijGl` marks the
S_ij-tabulated fixed-GL fast paths (`NumCountsSijGl`, `Shear1hGl`,
`ShearPrjGl`), `explicit_gl` the explicit fixed-GL Python references
(`numcounts_explicit_gl.py`, `shear1h_explicit_gl.py`), and `3d` the
adaptive references (`NumCounts3d`, `Shear1h3d`, `Shear1h2hMax3d`,
`*3dGpu`). Module labels are the ini `[section]` names and the labels
double as DataBlock output sections (`numcounts_sij_gl/vals`,
`shear1h_gl/vals`), so the local inis were updated in the same commit;
the retired `fast_mass`/`full_ltmz` strings survive only in "formerly"
notes like the ones below.

The observable READMEs are the detailed documentation — each leads with
the physics (the integral being computed and what each dims tag
approximates), then the per-dims backend sections including the
Gauss--Legendre node placement, with the run mechanics last. This file
is the map of the directory and the short decision guide for choosing a
numerical method.

## Folder structure

The layout is `<observable>/<language>/<dims>/`, with one README per
observable carrying the physics, the per-dims sections, and the run
mechanics:

```text
src/pipelines/des_y3/
├── README.md
├── number_counts/
│   ├── README.md
│   ├── cpp/{0d, 3d}/     0d: S_ij-tab fast path (formerly fast_mass);
│   │                     3d: adaptive explicit Cuhre reference
│   ├── cuda/3d/          adaptive explicit PAGANI reference
│   └── python/0d/        fast path + explicit fixed-GL reference
│                         (formerly full_ltmz Python) + validators
├── shear_1h2h/
│   ├── README.md
│   ├── cpp/{0d, 3d}/     0d: 1h z-contracted + max z-resolved +
│   │                     radial series; 3d: adaptive explicit (1h, max)
│   ├── cuda/{0d, 3d}/    0d: max model GPU; 3d: adaptive explicit GPU
│   └── python/0d/        all fixed-GL replicas, explicit fixed-GL
│                         references, radial series + validators
└── shear_projection/
    ├── README.md
    ├── cpp/{0d, 2d}/     0d: region-split GL exact-z; 2d: ShearPrjCuhre
    │                     (fixed log-GL angle, adaptive (z, lnM))
    ├── cuda/{0d, 3d}/    0d: frozen-physics GPU; 3d: fully-coupled
    │                     adaptive PAGANI diagnostic
    └── python/0d/        exact-z reference + validator
```

`<dims>` counts the adaptive integration dimensions of the backends in
that directory. A missing `<language>/<dims>/` directory means that the
backend is not implemented.

The observable implementations use the sibling layers
[`../systematics/`](../systematics/README.md),
[`../cosmology/`](../cosmology/README.md), and `../shared/`. The systematics
layer owns richness selection, selection bias, projection parameters, and
the projection-shear C++ cores. The cosmology layer contains halo-model
physics; `shared/` contains lower-level datablock models, mass/redshift
weights, and common integration utilities.

## Observable entry points

Start with the observable README, then follow its links to the numerical
strategies and language backends.

| Observable | Physical quantity | Documentation |
| --- | --- | --- |
| Cluster number counts | Counts in observed richness and redshift bins | [`number_counts/`](number_counts/README.md) |
| One-halo shear | Miscentered target-cluster NFW lensing | [`shear_1h2h/`](shear_1h2h/README.md) |
| Traditional one-plus-two-halo model | Maximum of one-halo and conventional two-halo terms | [`shear_1h2h/`](shear_1h2h/README.md) |
| Projection shear | Lensing from correlated line-of-sight structure with selection-affected bias | [`shear_projection/`](shear_projection/README.md) |

## Numerical strategies

| Strategy tag | Numerical idea | Role |
| --- | --- | --- |
| `3d` | All three integration variables handled by adaptive Cuhre/Vegas/PAGANI quadrature. | Independent reference and convergence tool; the maximum-dimension adaptive reference of each observable. |
| `2d` (projection only) | Fixed log-GL angular grid; adaptive quadrature over the inner (z, lnM). | Adaptive comparison backend (`ShearPrjCuhre`). |
| `0d` | No adaptive integration: fixed Gauss--Legendre sums (with the redshift/selection contractions of the former `fast_mass` paths, or the full explicit composition on fixed nodes) and offline tables + moments (the former `radial_series`). | Maintained production methods; fast and MCMC-viable. |

The `3d` adaptive implementations define the accuracy baseline
("Precision vs 3d" in every table); fixed Gauss--Legendre backends are
evaluated against that baseline. Agreement with an existing production
module is a separate backend-identity check, not an accuracy measurement.

The radial-series `0d` algorithm is not an exact replacement for the
varying-concentration one-halo model: its current profile uses a fixed
concentration and is not scientifically interchangeable with the other
implementations.

## Precision and cost overview

The measurements below are pinned fiducial DES Y3 benchmarks, not universal
performance guarantees. Costs are per sample. The benchmark uses 12 count
bins, 12 one-halo bins with 10 radii each, and 180 projection wall points.
CPU timings depend on the Perlmutter node and build; GPU timings use an A100.

The precision column is quoted against the `3d` adaptive reference of each
observable ("Precision vs 3d"); where a number was measured against a
different baseline (production identity, backend twins, the exact
evaluator), the baseline is stated in the cell:

| Dims | Observable | Method and backend | Cost | Precision vs 3d |
| --- | --- | --- | ---: | --- |
| `3d` | Counts | [`3d`](number_counts/README.md#the-3d-backends), adaptive Python | 25 s | Reference (3d); reported integration error at or below 1e-6 |
| `3d` | Counts | [`3d`](number_counts/README.md#the-3d-backends), Cuhre C++ | 3.1 s | 1.1e-4 direct vs the 3d Python reference (re-measured 2026-08-26, same fiducial point as `real_pipeline_extract_output`; supersedes the older 4.9e-4 proxy-via-0d-Python figure) |
| `3d` | Counts | [`3d`](number_counts/README.md#the-3d-backends), PAGANI CUDA/A100 | 2.0 s | 1.2e-4 direct vs the 3d Python reference (re-measured 2026-08-26); CUDA vs C++ twin 2.1e-5 (separate baseline, shared A100) |
| `0d` | Counts | [`0d`](number_counts/README.md#explicit-fixed-gl-python-reference), explicit Python (3-dim GL) | 83 ms | 3.5e-5 |
| `0d` | Counts | [`0d`](number_counts/README.md#redshift-contracted-fast-path), fast path Python (2-dim GL, S_ij tab) | 5 ms | 7.6e-4; also 2.4e-15 vs production (separate baseline); 1.1e-3 direct vs cuda-3d (re-measured 2026-08-26, identical to the C++ row -- numerically identical output) |
| `0d` | Counts | [`0d`](number_counts/README.md#redshift-contracted-fast-path), fast path C++ (2-dim GL, S_ij tab) | 6 ms | 7.6e-4; also identity with production (separate baseline); 1.1e-3 direct vs cuda-3d (re-measured 2026-08-26) |
| `3d` | One-halo shear | [`3d`](shear_1h2h/README.md#the-3d-backends), adaptive Python | 35 s | Reference (3d); reported integration error at or below 1e-6 |
| `3d` | One-halo shear | [`3d`](shear_1h2h/README.md#the-3d-backends), Cuhre C++ | 51 s | 2.6e-4 direct vs the 3d Python reference (re-measured 2026-08-26 on the reduced 6-pt corner wall, same fiducial point) |
| `3d` | One-halo shear | [`3d`](shear_1h2h/README.md#the-3d-backends), PAGANI CUDA/A100 | 32 s | 3.0e-4 direct vs the 3d Python reference (re-measured 2026-08-26); CUDA vs C++ twin 4.3e-5 (separate baseline, shared A100) |
| `0d` | One-halo shear | [`0d`](shear_1h2h/README.md#the-0d-backends), explicit Python (3-dim GL) | 149 ms | 4.9e-5 |
| `0d` | One-halo shear | [`0d`](shear_1h2h/README.md#the-0d-backends), 1h C++ (1-dim GL, z contracted) | 9 ms | 8.4e-4; also identity with production (separate baseline); 2.3e-4 direct vs cuda-3d on the reduced 6-pt wall (re-measured 2026-08-26, python twin gave numerically identical values) |
| `0d` | One-halo shear | [`0d`](shear_1h2h/README.md#the-0d-backends), radial series (tables + moments) | 6--7 ms | 56--86% (known fixed-c=4 defect); 3.7e-3 internal fixed-profile consistency (separate baseline) |
| `0d` | Max model | [`0d`](shear_1h2h/README.md#the-0d-backends), max C++/CUDA (2-dim GL, z-resolved) | 11 / 8 ms | 8.3e-4; CUDA vs C++ twin 6.4e-15 (separate baseline) |
| `3d` | Projection shear | [`3d`](shear_projection/README.md#the-3d-backend), PAGANI on A100, eps_rel=1e-3 | 95 s | Reference-class diagnostic (3d); convergence open — median 9.5e-4, maximum 2.2% vs region-split GL (separate baseline) |
| `2d` | Projection shear | [`2d`](shear_projection/README.md#the-2d-backend), `ShearPrjCuhre` C++ | ~72 s/pt (measured 2026-08-26, 3-pt sample; full 180-pt wall ≈ 3.6 h, not run interactively) | not yet measured |
| `0d` | Projection shear | [`0d`](shear_projection/README.md#the-0d-backends), exact-z Python (3-dim region-split GL) | 270 ms | median 9.5e-4, max 2.2% vs the 3d diagnostic (its convergence is open); 1.6e-11 vs exact evaluator, 5.5e-5 vs frozen production (separate baselines) |
| `0d` | Projection shear | [`0d`](shear_projection/README.md#the-0d-backends), exact-z C++ (3-dim region-split GL) | 154 ms | median 9.5e-4, max 2.2% vs the 3d diagnostic (its convergence is open); 1e-11 vs exact evaluator (separate baseline) |
| `0d` | Projection shear | [`0d`](shear_projection/README.md#the-0d-backends), frozen GPU path | 16 ms (measured 2026-08-26, full 180-pt wall) | broken: `cl` channel is uninitialized/mis-indexed device memory (up to 100% off, some NaN/denormal), not a faithful port — [known defect](../../../docs/known_issues/frozen_physics_signed_rnd_defect.md) |

The observable READMEs contain the complete backend tables, grid
settings (including the "GL nodes and weights" quadrature sections),
tolerances, and comparison definitions:

- [Number counts](number_counts/README.md)
- [One-halo and traditional one-plus-two-halo shear](shear_1h2h/README.md)
- [Projection shear](shear_projection/README.md)

## Prior-volume robustness (apriori sampling)

The table above is a single fiducial-point measurement. To check that
cost is stable and the pipeline doesn't silently misbehave away from
the fiducial point, `cosmosis-models/des_y3_cpp0d_fast_apriori.ini`
(1000 draws) and `cosmosis-models/des_y3_cpp3d_slow_reference_apriori.ini`
(10 draws) run the same two module chains under CosmoSIS's `apriori`
sampler, which draws uniformly from the full prior in
`mock_mcmc_widePlanck_values_mis.ini` (not just the fiducial point) and
reports module-by-module wall-clock for every draw
(`timing = T`). `cp_camb`'s CAMB-emulator grid stays at `nz = 50` in
both (matching the base `des_y3_cpp0d_fast.ini`/`des_y3_cpp3d_slow_reference.ini`
inis) -- `nz = 400` measurably slows every sample for no benefit to
this 0d/3d fixed-node chain.

**Fast chain, 1000 draws (2026-08-26, shared login-GPU node, CPU-only
modules)** -- per-module wall-clock across all draws that completed:

| Module | Median | Mean | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| `halo_model` | 454 ms | 467 ms | 335 ms | 721 ms |
| `sel_function` | 65 ms | 67 ms | 51 ms | 1322 ms |
| `ShearPrjGl` | 590 ms | 590 ms | 539 ms | 865 ms |
| `NumCountsSel` | 8 ms | 8 ms | 7 ms | 11 ms |
| `Shear1hMisSel` | 11 ms | 11 ms | 9 ms | 14 ms |
| `Shear1h2hMax` | 12 ms | 12 ms | 11 ms | 21 ms |
| **Total pipeline** | **1.39 s** | **1.44 s** | **1.18 s** | **2.78 s** |

The single-fiducial-point number in the table above (2.56 s) sits
inside this distribution, close to the mean -- consistent, not a
different regime. `sel_function`'s 1.3 s outlier is the one-time numba
JIT-compile cost on the *first* draw of the process (see the Perf notes
below); every other draw pays 50-80 ms.

**Prior-domain robustness finding**: at least 285 of the 1000 draws
(28.5%; the true count is likely somewhat higher -- some successful
completions get mis-attributed by stdout/stderr interleaving when
parsing the log, see the script's own note) made `cosmosis` report
`Pipeline failed on these parameters`, almost always inside `cp_camb`
or the NFW profile evaluation (`y3_buzzard/nfwModel.py:62: RuntimeWarning:
divide by zero encountered in arctanh`). The declared prior box in
`mock_mcmc_widePlanck_values_mis.ini` (e.g. `omega_m in U(0.11, 1.0)`,
`sigma8 in U(0.5, 1.5)`) is wider than the CAMB emulator's trained
bounding box and the domain where the fixed-`c=4`/analytic NFW
profiles stay well-conditioned. This is invisible at the single
fiducial point and only shows up under prior sampling -- worth fixing
before this prior is used for an actual MCMC run (either narrow the
prior to the emulator's valid box, or make the affected modules fail
soft instead of raising).

**Slow chain, 10 draws (2026-08-27)**: only 3 of 10 draws parsed as
clean single-pass successes (the same stdout/stderr caveat as above
applies, plus the adaptive backends below can raise their own
convergence errors independent of the `cp_camb`/NFW failure mode). The
striking result is the **cost variance** of the adaptive Cuhre
backends across the prior -- something a single fiducial-point
measurement cannot see at all:

| Module | Median | Mean | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| `NumCounts3d` | 0.83 s | 16.2 s | 0.73 s | **47.2 s** |
| `Shear1h3d` | 0.85 s | 9.2 s | 0.71 s | **26.1 s** |
| `Shear1h2hMax3d` | 0.92 s | 10.3 s | 0.74 s | **29.3 s** |
| `shear_prj_cuhre` | 220.8 s | 214.2 s | 199.9 s | 221.9 s |
| **Total pipeline** | **224 s** | **251 s** | **204 s** | **325 s** |

`NumCounts3d`/`Shear1h3d`/`Shear1h2hMax3d` each vary by **~40-60x**
between their cheapest and most expensive prior draw (0.7-0.9 s at
most points, up to 26-47 s at a handful of them) -- some parameter
draws push the near-delta richness ridge or the mass/redshift
integrand into a shape Cuhre needs far more subdivisions to resolve at
`eps_rel=1e-4`. `shear_prj_cuhre` stays comparatively stable (200-222
s) because it was already run on a reduced 3-point wall dominated by
one expensive angular integral rather than the adaptive mass/redshift
integral. This variance is invisible at the single fiducial point
(where all three read as a tidy few seconds) and matters for anyone
budgeting a batch job around these backends: size the wall time on the
*worst* draw in the relevant prior region, not the fiducial-point
number.

## Recommended methods

The maintained smoke/reference pipeline uses:

| Dims | Observable | Strategy | Backend |
| --- | --- | --- | --- |
| `0d` | Number counts | fast path (2-dim GL, S_ij tab) | `NumCountsSijGl.so` |
| `0d` | One-halo miscentered shear | 1h z-contracted (1-dim GL) | `Shear1hGl.so` |
| `0d` | Projection shear | region-split GL exact-z | `ShearPrjGl.so` |

These are the production or reference choices for the DES Y3 implementations.
The corresponding `3d` adaptive methods are validation tools. The traditional
max model and the frozen projection GPU path (both `0d`) are optional
variants.

## Important limitations

- The radial-series `0d` algorithm currently uses a fixed concentration and
  is not validated as an accurate replacement for the varying-concentration
  one-halo model.
- The projection `3d` GPU calculation has unresolved wall-edge
  convergence; do not call its default result fully converged.
- The traditional max model depends on the known `haloModel/dSigma_hh` data
  defect and is provisional as a scientific result. See
  [`docs/known_issues/dsigma_hh_debug_flag.md`](../../../docs/known_issues/dsigma_hh_debug_flag.md).
- The benchmark values are fiducial measurements. They are useful for method
  selection and regression detection, not as hardware-independent guarantees.

For the detailed validation policy and current maintenance backlog, use the
validation documents under `docs/` rather than treating this README as the
source of module-level DataBlock contracts.
