# Adaptive projection integration (`3d`)

This strategy (formerly `full_ltmz`; three adaptive dimensions) is the
independent full-dimensional adaptive diagnostic for the projection
observable. It is not the precision reference: the precision path
([`../0d/`](../0d/README.md)) explicitly splits the redshift integral into
foreground, exclusion-ring, and background regions. This namespace implements
the fully-coupled diagnostic in CUDA only; the partially adaptive C++
comparison backend `ShearPrjCuhre` has two adaptive dimensions and lives in
[`../2d/`](../2d/README.md).

## Numerical definition

The integrand keeps the angular, redshift, and mass variables coupled,

$$
\Delta\Sigma_{\rm prj}(R)=
\int d\ln\theta\,dz\,d\ln M\;
J(\theta,z,M;R),
$$

where `J` contains the random and clustered projection channels, the
selection-dependent bias, nonlinear correlation, photo-z factors, slab
exclusion, and the miscentred profile. The angular coordinate is integrated in
logarithmic form because the miscentred profile has a narrow feature near

$$
\theta_R=R/D_A(z).
$$

Using a linear-theta volume can make PAGANI report convergence while
missing this feature, especially at the smallest radii. The same problem
applies to the redshift direction: sharp cusps and exclusion boundaries are
not reliably resolved by one global adaptive volume.

## Common algorithm

1. Build device-resident interpolators for the fixed input tables.
2. Map the wall point to a log-theta integration domain.
3. Evaluate the HMF, bias, correlation, selection, photo-z, and exclusion
   factors at each adaptive node.
4. Evaluate the miscentred profile and sum the random and clustered channels.
5. Let PAGANI adaptively subdivide the three-dimensional domain for each wall
   point and return values, estimated errors, and status flags.

## Language implementation

| Language | Algorithm and source file | Status |
| --- | --- | --- |
| Python | No module in this namespace. The Python exact-z `0d` implementation is the readable reference. | Not implemented |
| C++ | No fully-coupled 3-D module. The partially adaptive `ShearPrjCuhre` backend (outer theta fixed log-GL, Cuhre/Vegas over the inner (z, lnM) — two adaptive dimensions) lives in [`../2d/`](../2d/README.md). | See `../2d/` |
| CUDA | `cuda/DSigmaPrjFullLtmzGpu.cu` (thin driver, physics in `cuda/dsigma_prj_full_ltmz_gpu_t.cuh`) uses the standard CUDA integration template and PAGANI; interpolation tables are evaluated on the device. | Implemented diagnostic backend |

## Precision and cost

The 2026-08-12 180-point A100 benchmark reported. These numbers are useful
for cost and failure-mode studies, not as a precision certificate:

| Dims | Setting | Cost | Precision vs 3d |
| --- | --- | ---: | --- |
| `3d` | `eps_rel=1e-3` | 95 s/sample | Reference-class diagnostic (3d); median (9.5\times10^{-4}), maximum (2.2\% ) vs region-split GL (separate baseline); not a precision reference |
| `3d` | `eps_rel=1e-4` | 463 s/sample | More expensive run; lowering the tolerance does not guarantee that sharp features were sampled |

The production fixed-grid settings under-resolve some innermost and outermost
wall points by up to about 2.3%. This backend exposed that issue, but its own
adaptive error estimate does not certify the full-domain integral. The
region-split exact-z calculation remains the highest-precision reference.

This three-dimensional PAGANI calculation is distinct from the historical
three-dimensional Cuhre path, which had the same full-box feature-resolution
problem, and from the current C++ `ShearPrjCuhre` comparison backend
([`../2d/`](../2d/README.md)): the latter uses a fixed log-Gauss--Legendre
angular grid and applies Cuhre or Vegas only to the inner `(z, lnM)`
integral — which is exactly what protects it from the missed-cusp failure
mode of this fully-coupled diagnostic. The fixed-grid and GSL alternatives
are summarized in the
[projection numerical-method map](../README.md#what-the-numerical-methods-actually-integrate).
