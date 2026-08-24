# DES Y3 observable implementations

This directory contains the maintained DES Y3 implementations for cluster
number counts, one-halo lensing, traditional one-plus-two-halo lensing, and
selection-affected projection lensing.

The directory is organized as:

```text
observable / numerical strategy / language or backend
```

The observable and strategy READMEs are the detailed documentation. This file
is the map of the directory and the short decision guide for choosing a
numerical method.

## Folder structure

```text
src/pipelines/des_y3/
├── README.md
├── number_counts/
│   ├── README.md
│   ├── full_ltmz/
│   │   └── README.md
│   └── fast_mass/
│       └── README.md
├── shear_1h2h/
│   ├── README.md
│   ├── full_ltmz/
│   │   └── README.md
│   ├── fast_mass/
│   │   └── README.md
│   └── radial_series/
│       └── README.md
└── shear_projection/
    ├── README.md
    ├── full_ltmz/
    │   └── README.md
    └── fast_mass/
        └── README.md
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

| Strategy | Numerical idea | Role |
| --- | --- | --- |
| [`full_ltmz`](number_counts/full_ltmz/README.md) | Keep true richness, redshift, and halo mass explicit in the integral. | Independent reference and convergence tool. |
| [`fast_mass`](number_counts/fast_mass/README.md) | Contract the redshift and selection dependence before the mass integration, using fixed grids or the shared exact core. | Maintained production method. |
| [`radial_series`](shear_1h2h/radial_series/README.md) | Approximate the population-averaged radial profile with a moment expansion and precomputed radial functions. | One-halo candidate approximation only. |

`full_ltmz` is the explicit reference formulation. Its adaptive Python
implementation defines the accuracy baseline; fixed Gauss--Legendre, Cuhre,
and PAGANI implementations are evaluated against that baseline.

`fast_mass` is a contracted production implementation. It is much faster
than the explicit reference and has a measured residual relative to it. Its
agreement with an existing production module is a separate backend-identity
check, not an accuracy measurement.

`radial_series` is not an exact replacement for the varying-concentration
one-halo model. Its current profile uses a fixed concentration and is not
scientifically interchangeable with `full_ltmz` or `fast_mass`.

## Precision and cost overview

The measurements below are pinned fiducial DES Y3 benchmarks, not universal
performance guarantees. Costs are per sample. The benchmark uses 12 count
bins, 12 one-halo bins with 10 radii each, and 180 projection wall points.
CPU timings depend on the Perlmutter node and build; GPU timings use an A100.

The comparison labels mean:

- **Reference error**: discrepancy relative to the adaptive `full_ltmz`
  calculation or, for projection, the exact evaluator named in the row.
- **Production identity**: agreement with an existing production backend. This
  checks implementation equivalence, not physical accuracy.
- **Internal consistency**: agreement between implementations of the same
  approximation. It is not validation against the physical reference model.

| Observable | Method and backend | Cost | Precision or status |
| --- | --- | ---: | --- |
| Counts | [`full_ltmz`](number_counts/full_ltmz/README.md), adaptive Python | 25 s | Reference; reported integration error at or below 1e-6 |
| Counts | [`full_ltmz`](number_counts/full_ltmz/README.md), fixed GL Python | 83 ms | 3.5e-5 vs adaptive reference |
| Counts | [`fast_mass`](number_counts/fast_mass/README.md), C++ | 6 ms | 7.6e-4 vs adaptive reference; identity with production |
| One-halo shear | [`full_ltmz`](shear_1h2h/full_ltmz/README.md), adaptive Python | 35 s | Reference; reported integration error at or below 1e-6 |
| One-halo shear | [`full_ltmz`](shear_1h2h/full_ltmz/README.md), fixed GL Python | 149 ms | 4.9e-5 vs adaptive reference |
| One-halo shear | [`fast_mass`](shear_1h2h/fast_mass/README.md), C++ | 9 ms | 8.4e-4 vs adaptive reference; identity with production |
| One-halo shear | [`radial_series`](shear_1h2h/radial_series/README.md) | 6--7 ms | Internal fixed-profile consistency only; not a production-accuracy result |
| Projection shear | [`full_ltmz`](shear_projection/full_ltmz/README.md), PAGANI on A100 | 95 s | Convergence remains open; median 9.5e-4 and maximum 2.2% vs refined GL |
| Projection shear | [`fast_mass`](shear_projection/fast_mass/README.md), exact-z C++ | 154 ms | 1e-11 vs exact evaluator |
| Projection shear | [`fast_mass`](shear_projection/fast_mass/README.md), frozen GPU path | 8.3 ms | Faithful acceleration of frozen production; not the exact-z reference |

The detailed strategy READMEs contain the complete backend tables, grid
settings, tolerances, and comparison definitions:

- [Number-count strategies](number_counts/README.md)
- [One-halo and traditional one-plus-two-halo strategies](shear_1h2h/README.md)
- [Projection-shear strategies](shear_projection/README.md)

## Recommended methods

The maintained smoke/reference pipeline uses:

| Observable | Strategy | Backend |
| --- | --- | --- |
| Number counts | `fast_mass` | `NumCountsFastMass.so` |
| One-halo miscentered shear | `fast_mass` | `Shear1hFastMass.so` |
| Projection shear | `fast_mass` | `ShearPrjFastMass.so` exact-z CPU path |

These are the production or reference choices for the DES Y3 implementations.
The corresponding `full_ltmz` methods are validation tools. The traditional
max model and frozen projection GPU path are optional variants.

## Important limitations

- `radial_series` currently uses a fixed concentration and is not validated as
  an accurate replacement for the varying-concentration one-halo model.
- The projection `full_ltmz` GPU calculation has unresolved wall-edge
  convergence; do not call its default result fully converged.
- The traditional max model depends on the known `haloModel/dSigma_hh` data
  defect and is provisional as a scientific result. See
  [`docs/known_issues/dsigma_hh_debug_flag.md`](../../../docs/known_issues/dsigma_hh_debug_flag.md).
- The benchmark values are fiducial measurements. They are useful for method
  selection and regression detection, not as hardware-independent guarantees.

For the detailed validation policy and current maintenance backlog, use the
validation documents under `docs/` rather than treating this README as the
source of module-level DataBlock contracts.
