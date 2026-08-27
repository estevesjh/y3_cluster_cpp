# Selection-affected projection shear

This observable models lensing from correlated line-of-sight structure
around the cluster — the two-halo term sourced by projected neighbours,
carrying the selection-affected bias $b_{\rm sel}(\theta)$ of
Costanzi et al. (2026). The code path is named `shear_projection`;
module names use `ShearPrj`.

## The physics

For richness wall bin $i$ and observed lens-redshift slice $j$, define
the redshift factor

$$
C_j(z)=
\frac{dV}{d\Omega\,dz}(z)\,
w_{\rm phot}(z;z_{{\rm ob},j}),
$$

and the two distinct mass weights

$$
\mathcal{P}^{\rm rnd}_{ij}(M)
=\int dz\;C_j(z)\,n(M,z),
$$

$$
\mathcal{P}^{\rm cl}_{ij}(\theta,M)
=\int dz\;C_j(z)\,
\xi_{\rm NL}\!\left(\left|\Delta\chi(\theta,z)\right|,
z_{{\rm ob},j}\right)
n(M,z)\,b(M,z)\,
\mathbf 1\!\left[\theta>\theta_{{\rm excl},ij}(z)\right].
$$

The random weight is independent of $\theta$ after the redshift
contraction; the clustered weight remains a $\theta\times M$ object
because the nonlinear correlation and the line-of-sight slab exclusion
depend on the angular offset. The combined projection weight is

$$
\mathcal{P}_{ij}(\theta,M)=
\mathcal{P}^{\rm rnd}_{ij}(M)
+b_{{\rm sel},ij}(\theta)\,\mathcal{P}^{\rm cl}_{ij}(\theta,M),
$$

and the projected excess surface density is

$$
\Delta\Sigma^{\rm prj}_{ij}(R)=
\int d\theta\;2\pi\sin\theta\int d\ln M\;
\mathcal{P}_{ij}(\theta,M)\,
\Delta\Sigma_{\rm mis}(R,\tau;M),
\qquad
\tau=\theta D_A(z_{{\rm ob},j}),
$$

with $\gamma_t^{\rm prj}(R) = \langle\Sigma_{\rm crit}^{-1}\rangle_j\,
\Delta\Sigma^{\rm prj}_{ij}(R)$. These weights are **not** the
number-count kernels: projection shear receives a wall-selection slice,
a photo-$z$ weight, and the selection-dependent bias
$b_{{\rm sel},ij}(\theta)$ as inputs, and the projection weights retain
angular dependence from $\xi_{\rm NL}$, halo bias, and the exclusion
mask. The $\Omega_m$ factor inside the miscentred NFW profile is a
profile normalization (rho_mult), not a mass weight.

The integrand has two sharp features that dominate every numerical
choice here: the miscentred profile peaks near
$\theta_R = R/D_A(z_{\rm ob})$, and the slab exclusion cuts the
clustered channel at the ring around $z_{\rm ob}$. The dims tags state
how each backend confronts them:

- **`0d`** — no adaptive integration: all three integrals are fixed-GL
  sums on grids whose nodes are *placed on the features*
  (see [GL nodes and weights](#gl-nodes-and-weights)). This region
  split is the precision side of every comparison below.
- **`2d`** — the angular integral stays on the feature-split fixed
  log-GL grid; adaptive Cuhre/Vegas handles only the inner
  $(z, \ln M)$ — protected from the missed-cusp failure by
  construction.
- **`3d`** — fully-coupled adaptive PAGANI over
  $(\ln\theta, z, \ln M)$: the independent diagnostic. A global
  adaptive volume can report a small internal error while missing the
  $\theta_R$ cusp or the exclusion boundary, so its own error estimate
  does not certify physical precision.

## Precision and cost

Pinned 180-point wall fiducial measurements; cost is per sample.

| Dims | Method and backend | Cost | Precision vs 3d |
| --- | --- | ---: | --- |
| `0d` | [exact-z Python (3-dim region-split GL)](#the-0d-backends) | 270 ms | median 9.5e-4, max 2.2% vs the 3d diagnostic (whose own convergence is open — the region split is the higher-precision side); 1.6e-11 vs exact evaluator, 5.5e-5 vs frozen production (separate baselines) |
| `0d` | [exact-z C++ (3-dim region-split GL)](#the-0d-backends) | 154 ms | same vs-3d relation as the Python row; 9.9e-12 vs exact evaluator (separate baseline) |
| `0d` | [frozen CUDA/A100](#the-0d-backends) | 16 ms (measured 2026-08-26, full 180-pt wall) | broken: `cl` channel is uninitialized/mis-indexed device memory (up to 100% off, some NaN/denormal) — [known defect](../../../docs/known_issues/frozen_physics_signed_rnd_defect.md); `rnd` matches the CPU frozen module to 2.6e-12 |
| `2d` | [`ShearPrjCuhre` C++ (fixed log-GL angle, adaptive (z, lnM))](#the-2d-backend) | ~72 s/pt (measured 2026-08-26, 3-pt sample; full 180-pt wall ≈ 3.6 h, not run interactively) | not yet measured |
| `3d` | [PAGANI CUDA/A100, eps_rel=1e-3](#the-3d-backend) | 95 s | Reference-class diagnostic (3d); median 9.5e-4, maximum 2.2% vs region-split GL; convergence open |
| `3d` | [PAGANI CUDA/A100, eps_rel=1e-4](#the-3d-backend) | 463 s | Lower requested tolerance does not remove the missed-feature risk |

The exact-z region-split path is the current observable precision
reference. The frozen GPU path is a production algorithm variant; the
full-box Cuhre/PAGANI paths are diagnostics and convergence stress
tests.

## The 0d backends

(Formerly `fast_mass`; zero adaptive dimensions — all three integrals
are region-split fixed-GL sums.) Two related implementations: an
exact-$z$ Python/C++ core and a CUDA port of the frozen production
machinery.

Algorithm: construct the per-slice angular grid from the wall
breakpoints and log-GL segments; construct the exclusion-ring and
log-distance wing redshift nodes (inverting the comoving-distance
relation for the wings); evaluate the HMF, bias, nonlinear correlation,
photo-$z$ weights, and $b_{{\rm sel},ij}(\theta)$; build the `rnd` and
`cl` mass weights; evaluate the cached miscentred NFW profile at every
wall radius and mass; contract the mass and angular grids; publish both
projected surface-density and shear triples.

| Dims | Backend | Cost | Precision vs 3d |
| --- | --- | ---: | --- |
| `0d` | Python exact-z (3-dim region-split GL) | 270 ms/sample | median 9.5e-4, max 2.2% vs the 3d diagnostic (its convergence is open; the region split is the higher-precision side); $1.6\times10^{-11}$ vs exact evaluator, $5.5\times10^{-5}$ vs frozen production (separate baselines) |
| `0d` | C++ exact-z (3-dim region-split GL) | 154 ms/sample | same vs-3d relation as the Python row; $9.9\times10^{-12}$ vs exact evaluator (separate baseline) |
| `0d` | CUDA frozen path / A100 | 16 ms/sample (measured 2026-08-26, full 180-pt wall) | broken: see below |

The frozen production CPU path is about 82 ms/sample. The CUDA port's
`rnd` channel matches the CPU frozen module to 2.6e-12 (the device
ΔΣ_mis cache and mean-field sweep are correct), but its `cl` channel
reads from uninitialized/mis-indexed device memory (NaNs on part of the
wall, denormal garbage elsewhere, up to 100% off where finite) — it is
NOT currently a faithful acceleration of the frozen definition. See
[docs/known_issues/frozen_physics_signed_rnd_defect.md](../../../docs/known_issues/frozen_physics_signed_rnd_defect.md#second-observation-2026-08-24-the-gpu-ports-cl-channel-is-broken-outright).

### GL nodes and weights

The angular grid is fixed log-Gauss–Legendre on segments split at
feature breakpoints
(`src/models/sigma_prj_t.hh::sp_detail::build_theta_grid`, mirrored in
`src/pipelines/systematics/shear_prj/cpp/sigma_prj_t.hh`): the segment
boundaries are
$\{\theta_{\rm lower},\ \theta_{{\rm excl},o},\
\theta_R(R_i)\ \forall R_i,\ \theta_\lambda,\ 2\theta_\lambda,\
\theta_{\max}\}$, with `n_per_seg` GL nodes per segment. The per-radius
breakpoint is load-bearing: each requested $R$ lands a node cluster at
its $\Sigma_{\rm mis}$ peak $\theta_R = R/D_A(z_{\rm ob})$, which is
exactly the feature a single global adaptive volume misses.

The redshift integral is split around the lens slice: an exclusion ring
of `n_zring` nodes (default 20) around $z_{\rm ob}$, plus foreground
and background wings of `n_zouter` nodes (default 20) each, placed in
logarithmic line-of-sight distance $|\Delta\chi|$ so the sharp
structure at the exclusion boundary is resolved
(`sigma_prj_t.hh`, ring + fg/bg log-$|\Delta\chi|$ grid). Mass uses a
plain fixed GL grid of `n_lnm` nodes. All node/weight sets come from
the same Gauss–Legendre constructor as the other observables
(`shared/datablock_models.py::gl_nodes` on the Python side;
`y3_pipelines::gl_nodes` in the pipeline-owned
`shared/sel_gl_weights.hh` for the pipeline drivers, with the
projection core in `sigma_prj_t.hh` applying the identical rule),
affine-mapped per segment or region. The composed weights are exactly the $C_j(z)$-weighted
contractions defined above, with the quadrature weights folded in at
evaluation.

## What the numerical methods actually integrate

These names describe different things:

- **Gauss--Legendre** is a fixed quadrature rule: its nodes and weights
  are chosen before the calculation.
- **GSL** is a numerical library. `ShearPrjGsl` uses GSL's adaptive
  `QAGP`, which applies a Gauss--Kronrod rule on adaptively subdivided
  redshift intervals. It is not a Gauss--Legendre integrator.
- **Cuhre** and **Vegas** are adaptive multidimensional integrators
  from the Cuba/cubacpp layer. The variables they integrate depend on
  the class.

| Path | Redshift | Mass | Angular variable |
| --- | --- | --- | --- |
| `ShearPrjFrozenCuhre` | Fixed Gauss--Legendre reduction over the foreground, exclusion ring around `z_ob`, and background | Cuhre/Vegas | Cuhre/Vegas in `u = ln(theta)` |
| `ShearPrjGsl` | Adaptive GSL `QAGP` (Gauss--Kronrod), with `z_ob` as an explicit breakpoint | Fixed Gauss--Legendre | Fixed Gauss--Legendre in `ln(theta)` |
| `ShearPrjCuhre` | Cuhre/Vegas | Cuhre/Vegas | Fixed log-Gauss--Legendre grid |

The related `ShearPrjEvaluator` and the `0d` exact-z path use fixed
Gauss--Legendre reductions in mass, angle, and the explicitly split
redshift regions; they have no adaptive final contraction and are
separate from the three comparison classes above. `ShearPrjFrozenCuhre`
reduces the redshift integral *before* Cuhre sees anything — "Cuhre
integrates over `(z, lnM)`" applies to `ShearPrjCuhre` only.

A historical fully-coupled 3-D Cuhre path (legacy labels
`Sigma_prj_integrand_cuhre`, `DSigma_prj_integrand`) integrated all
three variables in one global adaptive box; it was unreliable for
exactly the missed-feature reason above and was removed from the built
code on 2026-05-06. The maintained fully-coupled diagnostic is the
[3d PAGANI backend](#the-3d-backend); the maintained partially adaptive
comparison backend is [`ShearPrjCuhre` (2d)](#the-2d-backend).

## The 2d backend

Two adaptive dimensions: the outer angular integral stays on the fixed
log-GL feature-split grid (shared with the `0d` evaluators), and
adaptive Cuhre or Vegas handles only the inner $(z, \ln M)$ integral.
The feature-split angular treatment protects this backend from the
missed-cusp failure mode of the fully-coupled 3d diagnostic, while the
adaptive inner integral makes it an independent convergence check on
the `0d` region-split redshift/mass grids.

`cpp/2d/ShearPrjCuhre.cc` (moved here from `src/modules/sigma_prj_cpu/`)
is a one-line instantiation over the immutable `models/sigma_prj_t.hh`
core (`y3_cluster::ShearPrjCuhre`). It is a diagnostic/comparison
backend, not a production entry point.

## The 3d backend

(Formerly `full_ltmz`; three adaptive dimensions, CUDA only.) The
independent fully-coupled diagnostic,

$$
\Delta\Sigma_{\rm prj}(R)=
\int d\ln\theta\,dz\,d\ln M\;
J(\theta,z,M;R),
$$

where $J$ contains both projection channels, the selection-dependent
bias, nonlinear correlation, photo-$z$ factors, slab exclusion, and the
miscentred profile. The angular coordinate is integrated in logarithmic
form because of the narrow $\theta_R$ feature; even so, a linear or
under-resolved volume can make PAGANI report convergence while missing
it, especially at the smallest radii — which is why this backend is a
diagnostic, not the precision reference. It exposed that the production
fixed-grid settings under-resolve some innermost and outermost wall
points by up to about 2.3%, but its own adaptive error estimate does
not certify the full-domain integral.

## Running the backends

| Dims | Language | Sources | Module / output |
| --- | --- | --- | --- |
| `0d` | Python | `python/0d/shear_prj_gl.py` (+ `validate_vs_production.py`) | `dsigma_prj_gl/{vals,rnd,cl}`, `shear_prj_gl/{vals,rnd,cl}` |
| `0d` | C++ | `cpp/0d/ShearPrjGl.cc` (physics `cpp/0d/shear_prj_gl_t.hh`, over `sp_detail::ShearPrjCore`) | `ShearPrjGl.so` in `release-build/src/modules/des_y3_shear_prj_0d_cpp/` |
| `0d` | CUDA | `cuda/0d/ShearPrjFrozenGpu.cu` (physics `cuda/0d/shear_prj_frozen_gpu_t.cuh`) | `ShearPrjFrozenGpu.so` in `des_y3_shear_prj_0d_cuda/`, sections `dsigma_prj_frozen_gpu` / `shear_prj_frozen_gpu` |
| `2d` | C++ | `cpp/2d/ShearPrjCuhre.cc` (over immutable `models/sigma_prj_t.hh`) | `ShearPrjCuhre.so` in `des_y3_shear_prj_2d_cpp/`, sections `sigma_prj_cuhre` / `dsigma_prj_cuhre` / `shear_prj_cuhre` (unchanged by the move) |
| `3d` | CUDA | `cuda/3d/DSigmaPrj3dGpu.cu` (physics `cuda/3d/dsigma_prj_3d_gpu_t.cuh`) | `DSigmaPrj3dGpu.so` in `des_y3_shear_prj_3d_cuda/`, section `dsigmaprj3dgpu` |

Module labels are the ini `[section]` names (`[ShearPrjGl]`,
`[ShearPrjFrozenGpu]`, `[ShearPrjCuhre]`, ...), so pipelines drive
these backends by pointing those sections at the `.so` paths above.
