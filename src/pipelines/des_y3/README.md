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
| `0d` | Counts | [`0d`](number_counts/README.md#the-0d-backends), explicit Python (3-dim GL) | 83 ms | 3.5e-5 |
| `0d` | Counts | [`0d`](number_counts/README.md#the-0d-backends), fast path C++ (2-dim GL, S_ij tab) | 6 ms | 7.6e-4; also identity with production (separate baseline) |
| `3d` | One-halo shear | [`3d`](shear_1h2h/README.md#the-3d-backends), adaptive Python | 35 s | Reference (3d); reported integration error at or below 1e-6 |
| `0d` | One-halo shear | [`0d`](shear_1h2h/README.md#the-0d-backends), explicit Python (3-dim GL) | 149 ms | 4.9e-5 |
| `0d` | One-halo shear | [`0d`](shear_1h2h/README.md#the-0d-backends), 1h C++ (1-dim GL, z contracted) | 9 ms | 8.4e-4; also identity with production (separate baseline) |
| `0d` | One-halo shear | [`0d`](shear_1h2h/README.md#the-0d-backends), radial series (tables + moments) | 6--7 ms | 56--86% (known fixed-c=4 defect); 3.7e-3 internal fixed-profile consistency (separate baseline) |
| `0d` | Max model | [`0d`](shear_1h2h/README.md#the-0d-backends), max C++/CUDA (2-dim GL, z-resolved) | 11 / 8 ms | 8.3e-4; CUDA vs C++ twin 6.4e-15 (separate baseline) |
| `3d` | Projection shear | [`3d`](shear_projection/README.md#the-3d-backend), PAGANI on A100 | 95 s | Reference-class diagnostic (3d); convergence open — median 9.5e-4, maximum 2.2% vs region-split GL (separate baseline) |
| `2d` | Projection shear | [`2d`](shear_projection/README.md#the-2d-backend), `ShearPrjCuhre` C++ | minutes | pending (Perlmutter re-run, issue #23 task) |
| `0d` | Projection shear | [`0d`](shear_projection/README.md#the-0d-backends), exact-z C++ (3-dim region-split GL) | 154 ms | median 9.5e-4, max 2.2% vs the 3d diagnostic (its convergence is open); 1e-11 vs exact evaluator (separate baseline) |
| `0d` | Projection shear | [`0d`](shear_projection/README.md#the-0d-backends), frozen GPU path | 8.3 ms | pending (Perlmutter re-run, issue #23 task); faithful acceleration of frozen production (separate baseline) |

The observable READMEs contain the complete backend tables, grid
settings (including the "GL nodes and weights" quadrature sections),
tolerances, and comparison definitions:

- [Number counts](number_counts/README.md)
- [One-halo and traditional one-plus-two-halo shear](shear_1h2h/README.md)
- [Projection shear](shear_projection/README.md)

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
