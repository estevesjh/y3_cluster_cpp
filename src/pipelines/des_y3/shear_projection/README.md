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

## What the numerical methods actually integrate

These names describe different things:

- **Gauss--Legendre** is a fixed quadrature rule: its nodes and weights are
  chosen before the calculation.
- **GSL** is a numerical library. `ShearPrjGsl` uses GSL's adaptive `QAGP`,
  which applies a Gauss--Kronrod rule on adaptively subdivided redshift
  intervals. It is not a Gauss--Legendre integrator.
- **Cuhre** and **Vegas** are adaptive multidimensional integrators from the
  Cuba/cubacpp layer. The variables they integrate depend on the class.

The three comparison paths therefore have this map:

| Path | Redshift | Mass | Angular variable |
| --- | --- | --- | --- |
| `ShearPrjFrozenCuhre` | Fixed Gauss--Legendre reduction over the foreground, exclusion ring around `z_ob`, and background | Cuhre/Vegas | Cuhre/Vegas in `u = ln(theta)` |
| `ShearPrjGsl` | Adaptive GSL `QAGP` (Gauss--Kronrod), with `z_ob` as an explicit breakpoint | Fixed Gauss--Legendre | Fixed Gauss--Legendre in `ln(theta)` |
| `ShearPrjCuhre` | Cuhre/Vegas | Cuhre/Vegas | Fixed log-Gauss--Legendre grid |

The related `ShearPrjEvaluator` and `fast_mass` exact-z path use fixed
Gauss--Legendre reductions in mass, angle, and the explicitly split redshift
regions; they have no adaptive final contraction and are separate from the
three comparison classes above.

### Precision hierarchy

The precision reference is the exact-z path with explicit redshift regions:
the foreground wing, the exclusion ring around `z_ob`, and the background
wing. Each region receives its own quadrature nodes; the foreground and
background wings are placed in logarithmic line-of-sight distance so that the
sharp structure near the exclusion boundary is resolved.

This region split is essential. A full-box adaptive integral can return a small
internal Cuhre/PAGANI error while still missing narrow profile peaks, cusps,
or exclusion boundaries. The reported adaptive error therefore does not by
itself establish physical precision for this observable.

The corresponding implementations are in
[`sigma_prj_t.hh`](../../../models/sigma_prj_t.hh) and
[`sigma_prj_frozen_interp_t.hh`](../../../models/sigma_prj_frozen_interp_t.hh).

### `ShearPrjFrozenCuhre`

`set_sample()` first performs the redshift reduction with fixed
Gauss--Legendre nodes. The redshift interval is split into the foreground, an
exclusion ring around `z_ob`, and the background. The result is tabulated as
`W_rnd(lnM)`, `W_cl(lnM)`, and
`correlation_func(theta) = W_cls(theta)`. `evaluate()` then integrates only
over `u = ln(theta)` and `lnM`:

$$
\Delta\Sigma_{\rm rnd}(R) =
\int du\,d\ln M\,
[2\pi\sin(\theta)\theta]
W_{\rm rnd}(\ln M)
\Delta\Sigma_{\rm NFW}^{\rm mis}(R,\theta D_A,M),
$$

$$
\Delta\Sigma_{\rm corr}(R) =
\int du\,d\ln M\,
[2\pi\sin(\theta)\theta]
b_{\rm sel}(\theta)W_{\rm cls}(\theta)W_{\rm cl}(\ln M)
\Delta\Sigma_{\rm NFW}^{\rm mis}(R,\theta D_A,M).
$$

Thus, the statement “Cuhre integrates over `(z, lnM)`” applies to
`ShearPrjCuhre`, not to `ShearPrjFrozenCuhre`: its redshift integral has
already been reduced before Cuhre/Vegas is called.

### `ShearPrjGsl`

This path uses fixed Gauss--Legendre nodes for `lnM` and for `ln(theta)`.
Only the redshift integral is adaptive: GSL `QAGP` applies its
Gauss--Kronrod rule and receives `z_ob` as an explicit breakpoint. The point
is treated as a non-smooth breakpoint; it should be called singular only if
the integrand actually diverges there.

### `ShearPrjCuhre`

This path keeps the outer angular integral on a fixed log-Gauss--Legendre
grid. Cuhre or Vegas handles the inner two-dimensional integral over
`(z, lnM)` and returns the random and correlated channels together:

$$
\left[
\Sigma_{\rm rnd},\Sigma_{\rm corr},
\Delta\Sigma_{\rm rnd},\Delta\Sigma_{\rm corr}
\right]
=
\int dz\,d\ln M\;\mathbf{F}(z,\ln M).
$$

This is the unfrozen adaptive comparison path. Its `(z, lnM)` domain is
different from the `(u, lnM)` domain of `ShearPrjFrozenCuhre`.

### Historical full 3-D Cuhre path

An earlier projection implementation integrated the continuous observable in
all three variables at once:

$$
\Delta\Sigma^{\rm prj}(R)
= \int dz\,d\ln M\,d\theta\;
I_{\Delta\Sigma}(z,\ln M,\theta\mid \lambda_{\rm bin},z_{\rm ob},R).
$$

This was the original full-box Cuhre path, using the legacy code labels
`Sigma_prj_integrand_cuhre` and `DSigma_prj_integrand`. It was unreliable:
the global adaptive domain did not expose the redshift regions and narrow
profile peaks to the integrator. A nominal Cuhre convergence result was not
evidence of accurate projection shear. It was removed from the built code on
2026-05-06. It should not be confused with the maintained `ShearPrjCuhre`
backend above, which keeps a fixed angular GL grid and applies Cuhre only to
`(z, lnM)`.

The maintained full three-dimensional diagnostic is now the
[`full_ltmz`](full_ltmz/README.md) CUDA/PAGANI backend. In both cases the
physical output is projected tangential shear,

$$
\gamma_t^{\rm prj}(R)
= \Delta\Sigma^{\rm prj}(R)\,\Sigma_{\rm crit}^{-1}(z_{\rm ob}),
$$

with the non-unity `Sigma_crit^{-1}` factor supplied by the sample.

## Precision and cost

Pinned 180-point wall fiducial measurements; cost is per sample.

| Method and backend | Cost | Comparison or status |
| --- | ---: | --- |
| [`fast_mass`](fast_mass/README.md), exact-z Python | 270 ms | 1.6e-11 vs exact evaluator; 5.5e-5 vs frozen production |
| [`fast_mass`](fast_mass/README.md), exact-z C++ | 154 ms | 9.9e-12 vs exact evaluator |
| [`fast_mass`](fast_mass/README.md), frozen CUDA/A100 | 8.3 ms | Faithful acceleration of frozen production; not exact-z reference |
| [`full_ltmz`](full_ltmz/README.md), PAGANI CUDA/A100, eps_rel=1e-3 | 95 s | Diagnostic only: median 9.5e-4, maximum 2.2% vs region-split GL; convergence open |
| [`full_ltmz`](full_ltmz/README.md), PAGANI CUDA/A100, eps_rel=1e-4 | 463 s | Lower requested tolerance does not remove the missed-feature risk |

The exact-z Python/C++ region-split path is the current observable precision
reference. The frozen GPU path is a production algorithm variant, while the
full-box Cuhre/PAGANI paths are diagnostics and convergence stress tests.
