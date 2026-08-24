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
it. Only the folders carry these names: module labels, class names,
source-file names, and DataBlock output sections keep their historical
strings (for example `NumCountsFullLtmz`, `shear1h_fast_mass/vals`,
`numcounts_full_ltmz.py`), so inis and validators reference them
unchanged.

The observable and strategy READMEs are the detailed documentation. This file
is the map of the directory and the short decision guide for choosing a
numerical method.

## Folder structure

```text
src/pipelines/des_y3/
├── README.md
├── number_counts/
│   ├── README.md
│   ├── 0d/   fixed-GL: S_ij-tab fast path (formerly fast_mass) +
│   │         explicit fixed-GL Python (formerly full_ltmz Python)
│   └── 3d/   adaptive explicit C++/CUDA references (formerly full_ltmz)
├── shear_1h2h/
│   ├── README.md
│   ├── 0d/   fixed-GL/tables: 1h z-contracted + max-model z-resolved
│   │         (both formerly fast_mass), explicit fixed-GL Python
│   │         (formerly full_ltmz Python), radial series
│   │         (formerly radial_series)
│   └── 3d/   adaptive explicit C++/CUDA references (formerly full_ltmz)
└── shear_projection/
    ├── README.md
    ├── 0d/   region-split fixed GL (formerly fast_mass)
    ├── 2d/   ShearPrjCuhre: fixed log-GL angle, adaptive (z, lnM)
    └── 3d/   fully-coupled adaptive PAGANI diagnostic
              (formerly full_ltmz)
```

`python/`, `cpp/`, and `cuda/` directories under a strategy contain
implementations of that strategy. A missing language directory means that the
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
| `3d` | Counts | [`3d`](number_counts/3d/README.md), adaptive Python | 25 s | Reference (3d); reported integration error at or below 1e-6 |
| `0d` | Counts | [`0d`](number_counts/0d/README.md), explicit Python (3-dim GL) | 83 ms | 3.5e-5 |
| `0d` | Counts | [`0d`](number_counts/0d/README.md), fast path C++ (2-dim GL, S_ij tab) | 6 ms | 7.6e-4; also identity with production (separate baseline) |
| `3d` | One-halo shear | [`3d`](shear_1h2h/3d/README.md), adaptive Python | 35 s | Reference (3d); reported integration error at or below 1e-6 |
| `0d` | One-halo shear | [`0d`](shear_1h2h/0d/README.md), explicit Python (3-dim GL) | 149 ms | 4.9e-5 |
| `0d` | One-halo shear | [`0d`](shear_1h2h/0d/README.md), 1h C++ (1-dim GL, z contracted) | 9 ms | 8.4e-4; also identity with production (separate baseline) |
| `0d` | One-halo shear | [`0d`](shear_1h2h/0d/README.md), radial series (tables + moments) | 6--7 ms | 56--86% (known fixed-c=4 defect); 3.7e-3 internal fixed-profile consistency (separate baseline) |
| `0d` | Max model | [`0d`](shear_1h2h/0d/README.md), max C++/CUDA (2-dim GL, z-resolved) | 11 / 8 ms | 8.3e-4; CUDA vs C++ twin 6.4e-15 (separate baseline) |
| `3d` | Projection shear | [`3d`](shear_projection/3d/README.md), PAGANI on A100 | 95 s | Reference-class diagnostic (3d); convergence open — median 9.5e-4, maximum 2.2% vs region-split GL (separate baseline) |
| `2d` | Projection shear | [`2d`](shear_projection/2d/README.md), `ShearPrjCuhre` C++ | minutes | pending (Perlmutter re-run, issue #23 task) |
| `0d` | Projection shear | [`0d`](shear_projection/0d/README.md), exact-z C++ (3-dim region-split GL) | 154 ms | median 9.5e-4, max 2.2% vs the 3d diagnostic (its convergence is open); 1e-11 vs exact evaluator (separate baseline) |
| `0d` | Projection shear | [`0d`](shear_projection/0d/README.md), frozen GPU path | 8.3 ms | pending (Perlmutter re-run, issue #23 task); faithful acceleration of frozen production (separate baseline) |

The detailed strategy READMEs contain the complete backend tables, grid
settings, tolerances, and comparison definitions:

- [Number-count strategies](number_counts/README.md)
- [One-halo and traditional one-plus-two-halo strategies](shear_1h2h/README.md)
- [Projection-shear strategies](shear_projection/README.md)

## Recommended methods

The maintained smoke/reference pipeline uses:

| Dims | Observable | Strategy | Backend |
| --- | --- | --- | --- |
| `0d` | Number counts | fast path (2-dim GL, S_ij tab) | `NumCountsFastMass.so` |
| `0d` | One-halo miscentered shear | 1h z-contracted (1-dim GL) | `Shear1hFastMass.so` |
| `0d` | Projection shear | region-split GL exact-z | `ShearPrjFastMass.so` |

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
