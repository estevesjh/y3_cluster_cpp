# Selection-affected projection shear

The projection observable models correlated line-of-sight structure around a
cluster. The code path is named `shear_projection`; the module names use the
short form `ShearPrj`.

## Numerical definition

For wall slice \((i,j)\), where \(i\) labels the observed richness wall bin
and \(j\) labels the observed lens-redshift slice, let \(z_{{\rm ob},j}\) be
the observed lens redshift. Projection shear uses the dedicated
projection-weight notation
\(\mathcal{P}\), not the number-count weight \(W_{ij}(M)\).

Define

$$
C_j(z)=
\frac{dV}{d\Omega\,dz}(z)\,
w_{\rm phot}(z;z_{{\rm ob},j}),
$$

and the two redshift-contracted mass weights

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
\mathbf{1}\!\left[\theta>\theta_{{\rm excl},ij}(z)\right].
$$

The random weight is independent of \(\theta\) after the redshift
contraction. The clustered weight remains dependent on both \(\theta\) and
mass because of the nonlinear correlation and exclusion mask. The combined
projection weight is

$$
\mathcal{P}_{ij}(\theta,M)
=\mathcal{P}^{\rm rnd}_{ij}(M)
+b_{{\rm sel},ij}(\theta)
\mathcal{P}^{\rm cl}_{ij}(\theta,M).
$$

The projected excess surface density is

$$
\Delta\Sigma^{\rm prj}_{ij}(R)
=\int d\theta\;2\pi\sin\theta\int d\ln M\;
\mathcal{P}_{ij}(\theta,M)
\Delta\Sigma_{\rm mis}(R,\tau;M),
$$

where \(\tau=\theta D_A(z_{{\rm ob},j})\). The exact definitions of
\(\mathcal{P}^{\rm rnd}_{ij}\) and \(\mathcal{P}^{\rm cl}_{ij}\) are in the
[fixed-grid strategy](fast_mass/README.md).

The returned dimensionless shear is formed from the projected surface-density
channels using the same lensing conventions as the rest of the pipeline.

## Strategies and implementations

| Strategy | Numerical method | Python | C++ | CUDA |
| --- | --- | --- | --- | --- |
| [`fast_mass`](fast_mass/README.md) | Exact-\(z\) fixed-grid contraction over the angular wall and mass | Yes | Yes | Frozen-production contraction |
| [`full_ltmz`](full_ltmz/README.md) | Adaptive integral over the continuous angular, redshift, and mass variables | No | No | PAGANI |
| `radial_series` | Not implemented; the angular coordinate remains coupled to the projection geometry | — | — | — |

The exact-\(z\) Python/C++ path is the reference for the current observable.
The fast CUDA path is an optimization of the frozen production algorithm and
must be compared with that frozen definition, not silently described as the
exact-\(z\) calculation.
