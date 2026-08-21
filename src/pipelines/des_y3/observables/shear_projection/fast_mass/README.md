# Fixed-grid projection contraction

This strategy evaluates the projection observable on a fixed angular wall and
per-slice redshift/mass grids. It has two related implementations: an
exact-\(z\) Python/C++ core and a CUDA port of the frozen production
machinery.

## Definition of the projection weights

Let \(i\) label the observed richness wall bin and let \(j\) label the
observed lens-redshift wall slice, with observed redshift
\(z_{{\rm ob},j}\). These labels do **not** mean that projection shear uses
the number-count kernels \(S_i(\lambda_{\rm tr},z)\) and \(S_j(z)\).
Projection shear receives a wall-selection slice, a photo-\(z\) weight, and a
selection-dependent bias \(b_{{\rm sel},ij}(\theta)\) as inputs.

Define the redshift factor

$$
C_j(z)=
\frac{dV}{d\Omega\,dz}(z)\,
w_{\rm phot}(z;z_{{\rm ob},j}),
$$

where the fixed-grid quadrature weights are applied when this integral is
evaluated numerically. For a wall slice \((i,j)\), define the two distinct
mass weights

$$
\mathcal{P}^{\rm rnd}_{ij}(M)
=\int dz\;C_j(z)\,n(M,z),
$$

$$
\mathcal{P}^{\rm cl}_{ij}(\theta,M)
=\int dz\;C_j(z)\,
\xi_{\rm NL}\!\left(\left|\Delta\chi(\theta,z)\right|,
z_{{\rm ob},j}\right)
n(M,z)b(M,z)\,
\mathbf 1\!\left[\theta>\theta_{{\rm excl},ij}(z)\right].
$$

The random weight is independent of \(\theta\) after the redshift
contraction. The clustered weight remains a \(\theta\times M\) object because
the nonlinear correlation and the line-of-sight exclusion depend on the
angular offset. The combined projection weight is

$$
\mathcal{P}_{ij}(\theta,M)
=\mathcal{P}^{\rm rnd}_{ij}(M)
+b_{{\rm sel},ij}(\theta)\,
\mathcal{P}^{\rm cl}_{ij}(\theta,M).
$$

The profile offset is the physical coordinate

$$
\tau=\theta D_A(z_{{\rm ob},j}),
$$

so the projected surface-density channels are

$$
\Delta\Sigma^{\rm rnd}_{ij}(R)
=\int d\theta\;2\pi\sin\theta
\int d\ln M\;
\mathcal{P}^{\rm rnd}_{ij}(M)\,
\Delta\Sigma_{\rm mis}(R,\tau;M),
$$

$$
\Delta\Sigma^{\rm cl}_{ij}(R)
=\int d\theta\;2\pi\sin\theta\,
b_{{\rm sel},ij}(\theta)
\int d\ln M\;
\mathcal{P}^{\rm cl}_{ij}(\theta,M)\,
\Delta\Sigma_{\rm mis}(R,\tau;M),
$$

with

$$
\Delta\Sigma^{\rm prj}_{ij}(R)
=\Delta\Sigma^{\rm rnd}_{ij}(R)
+\Delta\Sigma^{\rm cl}_{ij}(R).
$$

The returned shear uses

$$
\gamma^{\rm prj}_{t,ij}(R)
=\left\langle\Sigma_{\rm crit}^{-1}\right\rangle_j
\Delta\Sigma^{\rm prj}_{ij}(R).
$$

The \(\Omega_m\) factor used by the NFW miscentred profile is a profile
normalization called rho_mult; it is not \(\mathcal{P}^{\rm rnd}_{ij}\), not
\(\mathcal{P}^{\rm cl}_{ij}\), and not a replacement for either weight.

These weights are therefore different from the number-count selection factor
\(S_{ij}(M,z)\): projection weights retain the angular dependence from
\(\xi_{\rm NL}\), halo bias, and the exclusion mask, while the outer
\(2\pi\sin\theta\,d\theta\) measure supplies the angular geometry. Richness
and photo-\(z\) inputs enter through the wall slice and its supplied weights.

## Common algorithm

1. Construct the per-slice angular grid from the wall breakpoints and log-GL
   segments.
2. Construct the exclusion-ring and log-distance wing redshift nodes; invert
   the comoving-distance relation for the wings.
3. Evaluate the HMF, bias, nonlinear correlation, photo-\(z\) weights, and
   selection-dependent \(b_{{\rm sel},ij}(\theta)\).
4. Build the mass weights for the `rnd` and `cl` channels.
5. Evaluate the cached miscentred NFW profile at every wall radius and mass,
   then contract the mass and angular grids.
6. Publish both projected surface-density and shear triples.

## Language implementations

| Language | Algorithm and source files | Status |
| --- | --- | --- |
| Python | `python/shear_prj_fast_mass.py` is a convention-exact port of `models/sigma_prj_t.hh`; `python/validate_vs_production.py` replays saved pipeline data. | Exact-\(z\) readable reference |
| C++ | `cpp/ShearPrjFastMass.cc` wraps the immutable `sp_detail::ShearPrjCore` and computes both observables in one pass. | Exact-\(z\) CPU reference |
| CUDA | `cuda/ShearPrjFrozenGpu.cu` keeps the frozen host grids and contracts the heavy miscentred-NFW cache in one device kernel. | Frozen-production optimization |

## Precision and cost

Pinned 180-point wall at the fiducial point:

| Backend | Cost | Comparison |
| --- | ---: | --- |
| Python exact-\(z\) | 270 ms/sample | \(1.6\times10^{-11}\) vs exact evaluator; \(5.5\times10^{-5}\) vs frozen production |
| C++ exact-\(z\) | 154 ms/sample | \(9.9\times10^{-12}\) vs exact evaluator |
| CUDA frozen path / A100 | 8.3 ms/sample | \(1.5\times10^{-11}\) vs frozen production |

The frozen production CPU path is about 82 ms/sample. The CUDA result is
therefore a faithful acceleration of that frozen definition, not a validation
of the exact-\(z\) definition.
