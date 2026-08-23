# One-halo and traditional one-plus-two-halo shear

This tree contains two related calculations:

1. the maintained miscentred one-halo lensing profile; and
2. the traditional max model, which compares the one-halo profile with a
   two-halo term at each mass, redshift, and radius.

The selection, HMF, volume, and lensing-weight conventions are shared with
the count observable.

## One-halo definition

The profile is a mixture of centred and miscentred terms,

$$
\Delta\Sigma_{1h}(R,M,z)=
(1-f_{\rm mis})\Delta\Sigma_{\rm cen}(R,M,z)
+f_{\rm mis}\Delta\Sigma_{\rm mis}(R,M,z).
$$

The stacked observable is

$$
O_{ij}(R)=\int dz\,d\ln M\,d\lambda_{\rm tr}\;
n\frac{dV}{d\Omega dz}\Omega
\Sigma_{\rm crit}^{-1}(z)
S_jS_iP_{\rm HOD}\Delta\Sigma_{1h}(R,M,z).
$$

The production profile uses the maintained halo-model centred table and the
miscentred NFW/gamma lookup convention. The profile details are intentionally
kept in `shared/lensing_profiles.py` and the immutable model headers.

## Traditional max model

The optional traditional model retains the redshift dependence of the
two-halo term and forms

$$
\Delta\Sigma_{\rm max}(R,M,z)=
\max\left[\Delta\Sigma_{1h}(R,M,z),
b(M,z)\Delta\Sigma_{hh}(R,z)\right].
$$

It therefore cannot use the same redshift-free profile contraction as the
one-halo term. The max-model implementation returns a characterization and
performance path; its two-halo input has a separately documented low-radius
debugging issue.

## Strategies and implementations

| Strategy | Numerical method | Python | C++ | CUDA |
| --- | --- | --- | --- | --- |
| [`full_ltmz`](full_ltmz/README.md) | Explicit selection integral, then radial profile contraction | Yes | Cuhre | PAGANI |
| [`fast_mass`](fast_mass/README.md) | Exact fixed-GL redshift contraction, then mass sum; z-resolved for max model | Yes | Yes | Max model only |
| [`radial_series`](radial_series/README.md) | Moment expansion using precomputed (U_\ell) for a fixed profile | Generator and evaluator | Table reader | No |

The production one-halo identity path is `fast_mass` C++. The radial-series
path is a candidate approximation and has a fixed-(c=4) shape limitation.
