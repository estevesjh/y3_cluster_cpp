# Selection-affected projection shear

This observable models lensing from correlated line-of-sight structure around
the cluster. The code path is named `shear_projection`; module names use
`ShearPrj`.

## Numerical definition

For richness wall bin `i` and observed lens-redshift slice `j`, define

$$
C_j(z)=
\frac{dV}{d\Omega\,dz}(z)
w_{\rm phot}(z;z_{{\rm ob},j}),
$$

$$
\mathcal{P}^{\rm rnd}_{ij}(M)
=\int dz\;C_j(z)n(M,z),
$$

$$
\mathcal{P}^{\rm cl}_{ij}(\theta,M)
=\int dz\;C_j(z)
\xi_{\rm NL}\!\left(\left|\Delta\chi(\theta,z)\right|,
z_{{\rm ob},j}\right)
n(M,z)b(M,z)
\mathbf{1}\!\left[\theta>\theta_{{\rm excl},ij}(z)\right].
$$

The combined projection weight is

$$
\mathcal{P}_{ij}(\theta,M)=
\mathcal{P}^{\rm rnd}_{ij}(M)
+b_{{\rm sel},ij}(\theta)\mathcal{P}^{\rm cl}_{ij}(\theta,M).
$$

The projected excess surface density is

$$
\Delta\Sigma^{\rm prj}_{ij}(R)=
\int d\theta\;2\pi\sin\theta\int d\ln M\;
\mathcal{P}_{ij}(\theta,M)
\Delta\Sigma_{\rm mis}(R,\tau;M),
\qquad
\tau=\theta D_A(z_{{\rm ob},j}).
$$

The detailed weight construction is documented in the
[`fast_mass` strategy README](fast_mass/README.md).

## Precision and cost

Pinned 180-point wall fiducial measurements; cost is per sample.

| Method and backend | Cost | Comparison or status |
| --- | ---: | --- |
| [`fast_mass`](fast_mass/README.md), exact-z Python | 270 ms | 1.6e-11 vs exact evaluator; 5.5e-5 vs frozen production |
| [`fast_mass`](fast_mass/README.md), exact-z C++ | 154 ms | 9.9e-12 vs exact evaluator |
| [`fast_mass`](fast_mass/README.md), frozen CUDA/A100 | 8.3 ms | Faithful acceleration of frozen production; not exact-z reference |
| [`full_ltmz`](full_ltmz/README.md), PAGANI CUDA/A100, eps_rel=1e-3 | 95 s | Median 9.5e-4, maximum 2.2% vs refined GL; convergence open |
| [`full_ltmz`](full_ltmz/README.md), PAGANI CUDA/A100, eps_rel=1e-4 | 463 s | More expensive convergence study; wall-edge convergence remains open |

The exact-z Python/C++ path is the current observable reference. The frozen
GPU path is a production algorithm variant, not the exact-z calculation.
