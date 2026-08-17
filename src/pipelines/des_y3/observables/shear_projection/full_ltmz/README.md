# Adaptive projection integration

This strategy is the independent adaptive reference for the projection
observable. The current namespace implements it in CUDA only.

## Numerical definition

The integrand keeps the angular, redshift, and mass variables coupled,

$$
\Delta\Sigma_{\rm prj}(R)=
\int d\ln\theta\,dz\,d\ln M\;
J(\theta,z,M;R),
$$

where \(J\) contains the random and clustered projection channels, the
selection-dependent bias, nonlinear correlation, photo-\(z\) factors, slab
exclusion, and the miscentred profile. The angular coordinate is integrated in
logarithmic form because the miscentred profile has a narrow feature near

$$
\theta_R=R/D_A(z).
$$

Using a linear-(\theta) volume can make PAGANI report convergence while
missing this feature, especially at the smallest radii.

## Common algorithm

1. Build device-resident interpolators for the fixed input tables.
2. Map the wall point to a log-(\theta) integration domain.
3. Evaluate the HMF, bias, correlation, selection, photo-\(z\), and exclusion
   factors at each adaptive node.
4. Evaluate the miscentred profile and sum the random and clustered channels.
5. Let PAGANI adaptively subdivide the three-dimensional domain for each wall
   point and return values, estimated errors, and status flags.

## Language implementation

| Language | Algorithm and source file | Status |
| --- | --- | --- |
| Python | No module in this namespace. The Python exact-\(z\) fast-mass implementation is the readable reference. | Not implemented |
| C++ | No module in this namespace. | Not implemented |
| CUDA | `cuda/DSigmaPrjFullLtmzGpu.cu` uses the standard CUDA integration template and PAGANI; interpolation tables are evaluated on the device. | Implemented reference backend |

## Precision and cost

The 2026-08-12 180-point A100 benchmark reported:

| Setting | Cost | Comparison |
| --- | ---: | --- |
| `eps_rel=1e-3` | 95 s/sample | Median (9.5\times10^{-4}), maximum (2.2\% ) vs refined GL |
| `eps_rel=1e-4` | 463 s/sample | More expensive convergence study; wall-edge convergence remains open |

The production fixed-grid settings under-resolve some innermost and outermost
wall points by up to about 2.3%. This backend exposed that issue; neither the
adaptive result at the default tolerance nor the production grid should yet be
called a fully converged wall-edge reference.
