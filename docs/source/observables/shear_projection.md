# Shear Projection

`C++` · `y3_cluster_cpp` (`src/pipelines/des_y3`) · `Cluster observable` · module `ShearPrjGl` · `~154 ms/sample`

Computes $\Sigma^{\rm prj}$ and its lensing observables — in the paper's
language, **the two-halo term sourced by correlated line-of-sight
structure**, carrying the selection-affected bias $b_{\rm sel}(\theta)$:
the surface density and tangential shear contributed by the distinct
neighbouring haloes whose presence is correlated with the target cluster
and, through projection effects, with its optical selection. It replaces
the conventional (unselected-bias) two-halo term in the reference shear
composition $\gamma_t^{\rm theory} = \langle\gamma_t^{1h}\rangle +
\gamma_t^{\rm prj}$. The model is
[Costanzi et al. 2026, PhRvD 113, 103508](https://ui.adsabs.harvard.edu/abs/2026PhRvD.113j3508C/abstract)
(arXiv:[2604.05833](https://arxiv.org/abs/2604.05833)), built on the
population-integral framework of
[DES Cluster et al. 2023](https://ui.adsabs.harvard.edu/abs/2023arXiv230906593A/abstract)
(arXiv:[2309.06593](https://arxiv.org/abs/2309.06593)).

## Script

- Model: [`src/pipelines/systematics/shear_prj/cpp/sigma_prj_t.hh`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/pipelines/systematics/shear_prj/cpp/sigma_prj_t.hh)
  (`sp_detail::ShearPrjCore` — the caching $\theta$-grid/$z$-grid core
  shared with DES Y1's `ShearPrjEvaluator`/`ShearPrjFrozenPhysics`,
  {doc}`../variants`; the copy under `src/models/` is the frozen legacy
  twin). Exact-row $b_{\rm sel}$ lookup via
  `systematics/selection_bias/cpp/bsel_bins_t.hh`.
- Module driver: [`src/pipelines/des_y3/shear_projection/cpp/0d/ShearPrjGl.cc`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/pipelines/des_y3/shear_projection/cpp/0d/ShearPrjGl.cc)
  + [`shear_prj_gl_t.hh`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/pipelines/des_y3/shear_projection/cpp/0d/shear_prj_gl_t.hh)
  — des_y3-namespaced wrapper over the core, own module label and
  output sections.
- Compiled library loaded by CosmoSIS:
  `${Y3_CLUSTER_CPP_DIR}/release-build/src/modules/des_y3_shear_prj_0d_cpp/ShearPrjGl.so`.
- Disk tables (loaded once at construction; **single**/delta offset
  kernel — not the gamma kernel {doc}`shear_halo` uses):
  `data/nfw_off_center/table_1000_1e-03_5e+03_single_{logx, logxmis}.txt`,
  `…_log_deltasigma_single.txt`, with the concentration taken from
  `haloModel/concentration` (the concentration–mass relation, never a
  fixed $c$).

## Numerical framework

The full integral — per $(\lambda^{\rm ob}, z^{\rm ob}, R)$ wall point,
the projected excess surface density of the line-of-sight structure
(Costanzi 2026 Eq. 13, derived in {doc}`../math/index`):

$$\Sigma^{\rm prj}(R) = 2\pi\!\int\! d\theta\,\sin\theta
\int\! dz\; w_z(z, z^{\rm ob})\, \frac{dV}{d\Omega\,dz}
\int\! d\ln M\; n(M, z)\;
b(M,z)\, b_{\rm sel}(\theta)\, \xi_{\rm NL}\big(r(\theta,z), z^{\rm ob}\big)\,
E(\theta, z)\;
\Sigma_{\rm mis}\big(R \mid M, z, R_{\rm off}=\theta\,\chi(z^{\rm ob})\big),$$

with the same weight and $\Delta\Sigma_{\rm mis}$ in place of
$\Sigma_{\rm mis}$ for $\Delta\Sigma^{\rm prj}$, and
$\gamma_t^{\rm prj} = \Delta\Sigma^{\rm prj}\,
\langle\Sigma_{\rm crit}^{-1}\rangle(z^{\rm ob})$. Reading the factors
from the outside in:

- **$\theta$** is the angular separation of the neighbour; its
  comoving projected offset $R_{\rm off} = \theta\,\chi(z^{\rm ob})$ is
  the argument of the neighbour's **single-offset miscentred NFW**
  profile $\Sigma_{\rm mis}$ (azimuth-averaged, $1/2\pi$-normalised) —
  *neighbouring-halo miscentering*, a geometric ingredient with no
  nuisance parameters, as opposed to the gamma-kernel
  *target-cluster miscentering* of {doc}`shear_halo`.
- **$w_z$** is the parabolic photo-$z$ window
  $\max(0, 1-u^2)$, $u=(z-z^{\rm ob})/\sigma_z(z)$: which
  line-of-sight neighbours get projected onto the cluster
  ({doc}`../modules/redshift_kernel`).
- **$b(M,z)\,b_{\rm sel}(\theta)\,\xi_{\rm NL}(r)$** is the modelled
  halo–halo correlation on the exact 3-D chord
  $r^2 = \chi^2(z)+\chi_{\rm c}^2-2\chi(z)\chi_{\rm c}\cos\theta$.
  $b_{\rm sel}(\theta) = B_{\rm small} + (B_{\rm large} - B_{\rm
  small})\,\sigma(\theta)$ is the analytic sigmoid between the two
  plateaus that `bsel` closes per bin ({doc}`../systematics/bsel`) —
  the scale-dependent bias of the optically selected population that
  is the whole point of the Costanzi-2026 model.
- **$E(\theta,z)$** is the halo-exclusion mask: $\xi_{\rm NL}$ is zeroed
  for neighbour centres inside the comoving ball
  $R_{\rm excl} = R_\lambda(\lambda^{\rm ob})(1+z^{\rm ob})$ around the
  cluster, evaluated per $z$ as the angular cap $\theta <
  \theta_{\rm excl}(z)$ with $\cos\theta_{\rm excl} =
  [\chi^2(z)+\chi_{\rm c}^2-R_{\rm excl}^2]/[2\chi(z)\chi_{\rm c}]$.
  The exclusion counterterm of the full pair-distribution treatment is
  not evaluated (a $\lesssim 0.6\%$ effect at $R\to0$; see the
  exclusion subsection of {doc}`../math/index`).

Two things the operator deliberately lacks, compared with the
$N_i[f]$ population operator of {doc}`number_counts`: no $S_{ij}$ and no
$N_i[1]$ division (it is already per $(\lambda^{\rm ob}, z^{\rm ob})$
bin), and no $\Omega(z)$ (cancels for a surface density). The
`ShearPrjCore` has no `include_omega_z` knob at all — unlike DES Y1's
frozen core, where the option exists and is set to 0 in the ini
({doc}`../modules/survey_area`).

### The two channels

The module accumulates two integrals sharing the same
$(\theta, z, \ln M)$ grid and the same $\Sigma_{\rm mis}$ tables:

- **`cl`** — the clustered channel: the integrand above. This *is*
  $\Sigma^{\rm prj}$ (and $\Delta\Sigma^{\rm prj}$, $\gamma_t^{\rm prj}$),
  the correlated excess a random-point-subtracted measurement contains.
  **The likelihood consumes `cl`** ({doc}`likelihood`).
- **`rnd`** — the random-point channel: the same integral with the
  bracket replaced by $1$ and no exclusion, i.e. the mean projected
  matter column $\Sigma_{\rm bkg}$ in the photo-$z$ window. It is
  near-uniform, selection-blind, and integrates to
  $\Delta\Sigma_{\rm bkg} \approx 0$; published as a diagnostic and for
  raw (non-subtracted) mock comparisons.
- **`vals`** $=$ `rnd` $+$ `cl`, the raw column of distinct neighbours.

### Evaluation: region-split fixed Gauss–Legendre (`0d`)

Nothing adaptive; every integral is a fixed sum on a grid whose nodes
are *placed on the integrand's features*:

1. **$\theta$** — log-GL on segments split at the breakpoints
   $\{\theta_{\min},\ \theta_{\rm excl}(z^{\rm ob}),\ \theta_R = R/\chi_{\rm c}$
   for every wall radius on the slice$,\ \theta_\lambda,\ 2\theta_\lambda,\
   \theta_{\max} = R_{\max}/\chi_{\rm c}\}$, `n_per_seg` nodes per
   segment. The per-$R$ breakpoint is load-bearing: $\Sigma_{\rm mis}$
   peaks at $R_{\rm off}\approx R$.
2. **$z$** — an exclusion-ring band around $z^{\rm ob}$ (`n_zring`
   nodes, where $\theta_{\rm excl}(z)$ varies fastest) plus
   foreground/background wings in log $|\Delta\chi|$ (`n_zouter` nodes
   each), inside the photo-$z$ support $[z_{\rm lo}, z_{\rm hi}]$.
3. **$\ln M$** — fixed GL with `n_lnm` nodes.

The $z$ sum is contracted once per $(\lambda^{\rm ob}, z^{\rm ob})$ slice
into mass vectors — $w^{\rm rnd}(M)$ and, because $\xi_{\rm NL}$ and the
exclusion depend on the angle, $w^{\rm cl}(\theta, M)$ — and every wall
radius is then a $(\theta, M)$ dot product against the cached
$\Sigma_{\rm mis}(R\mid\theta,M)$ and $\Delta\Sigma_{\rm mis}$ tables,
producing $\Sigma$, $\Delta\Sigma$, and $\gamma_t$ in one pass. The
slice caches are keyed on the sample, so the 15 radii of a bin share
all grid construction. **The complete step-by-step recipe — grid
construction, exclusion mask, channel contractions, table lookups, cost
— lives in {doc}`../numerics/index`, §"The shear-projection recipe,
step by step".**

This module keeps the **exact redshift dependence** (HMF, bias, volume
element, $\xi_{\rm NL}$ all evaluated at each $z$ node). DES Y1's
`shear_prj_frozen_physics` speeds the same core up by freezing the
clustered channel at $z^{\rm ob}$ with an $r_s(M)$-anchored drift
amplitude ($\sim 3\times$ faster, $<0.2\%$ deviation, $5.5\times10^{-5}$
on the wall at the fiducial point); `ShearPrjFrozenGpu` is its CUDA
port. Selection-bias inputs: {doc}`../systematics/bsel`; validation
backends (`ShearPrjEvaluator`, `ShearPrjGsl`, `ShearPrjCuhre`):
{doc}`../variants`.

## CosmoSIS setup

```ini
[ShearPrjGl]
file = ${Y3_CLUSTER_CPP_DIR}/release-build/src/modules/des_y3_shear_prj_0d_cpp/ShearPrjGl.so
zt_low      = 0.10
zt_high     = 0.75
lnm_low     = 29.9336
lnm_high    = 35.6814
R_max_cMpch = 35.0
lambda_bin  = <180-entry wall: 15 radii per (richness, zob) bin>
zo_low      = <180 entries>
zo_high     = <180 entries>
radii       = <180 entries: 0.0426 … 24.8771 cMpc/h per bin>
```

(The four wall arrays are 180 entries each — 12 bins × 15 radii, bin
slow / radius fast; full arrays in
[`cosmosis-models/des_y3.ini`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/cosmosis-models/des_y3.ini).)

- Ordering: **after `bsel`** (needs the bias plateaus), `halo_model`,
  `MfTinker`, `cp_camb`, `average_sigma_crit_inv`; before `likelihoods`.
- The ini section name must be `ShearPrjGl` — it is the hard-coded
  `module_label()`.
- Left at the core's class defaults for `n_lnm`/`n_per_seg`/
  `n_zring`/`n_zouter` (24/30/20/20); the frozen DES Y1 module reduces
  `n_lnm` to 16 and `n_per_seg` to 10.

## Configuration options

| Option | Meaning | Units | Reference value |
|---|---|---|---|
| `lambda_bin`, `zo_low`, `zo_high`, `radii` | zipped wall grid (4 equal-length arrays) | $R$: cMpc/$h$ | 180 entries |
| `zt_low`, `zt_high` | true-redshift envelope of the line-of-sight integral | — | 0.10, 0.75 |
| `lnm_low`, `lnm_high` | mass GL limits | $\ln(M_\odot/h)$ | 29.9336, 35.6814 |
| `n_lnm` | GL nodes in $\ln M$ | — | 24 (default, left unset) |
| `n_per_seg` | log-GL nodes per $\theta$ segment | — | 30 (default, left unset) |
| `n_zring`, `n_zouter` | ring-band / per-wing redshift nodes | — | 20, 20 (defaults) |
| `R_max_cMpch` | sets $\theta_{\max} = R_{\max}/\chi(z^{\rm ob})$ | cMpc/$h$ | 35.0 |
| `lob_centers` | richness centres for $R_\lambda$, $R_{\rm excl}$, $\theta_\lambda$ | — | 25 37.5 52.5 130 (default) |

## DataBlock inputs

| DataBlock input | Meaning | Units / shape | Produced by |
|---|---|---|---|
| `mass_function/*`, `cluster_abundance/{hmf_s, hmf_q}` | HMF via `HMF_t` (applies the $\ln(\Omega_m-\Omega_\nu)$ mass-axis shift) | — | `MfTinker`, values file |
| `haloModel/{lnM, z, bias}` | halo bias $b(M,z)$ | `(50, 100)` | `halo_model` |
| `haloModel/{lnM, concentration}` | concentration–mass relation for the neighbour NFW | `(100,)` | `halo_model` |
| `haloModel/rho_m_ref` | reference density of the NFW boundary and amplitude (unified $\rho_m$ convention) | $M_\odot h^2/{\rm Mpc}^3$ | `halo_model` |
| `xi_nl/{r, z, xi_nl}` | nonlinear correlation function | `(50, 128)` | `halo_model` |
| `distances/{z, d_a, d_c}` | volume element via `DV_DO_DZ_t`; comoving distance ($\times h_0$ → cMpc/$h$) | Mpc | `cp_camb` |
| `b_sel_marginalised/{lambda_bin, zob, lob, b_small, b_large}` | selection-bias plateaus, exact `(lambda_bin, zob)` row lookup | `(12,)` each | `bsel` |
| `average_sigma_crit_inv/{zlense, sci_average}` | $\langle\Sigma_{\rm crit}^{-1}\rangle$ — **optional**: if absent, all $\gamma_t$ outputs are identically zero | `(50,)` | `average_sigma_crit_inv` |
| `cosmological_parameters/{omega_m, h0}` | unit conversions | scalars | `consistency` |

The photo-$z$ width $\sigma_z(z)$ comes from the compiled-in table
`src/models/z_kernel_data.hh`, not the DataBlock.

## DataBlock outputs

Nine arrays, each of length 180 (bin slow / radius fast):

| DataBlock output | Meaning | Units / shape | Consumed by |
|---|---|---|---|
| `sigma_prj_gl/{vals, rnd, cl}` | $\Sigma$: raw column, background $\Sigma_{\rm bkg}$, clustered $\Sigma^{\rm prj}$ | $M_\odot/(h\,\mathrm{pc}^2)$, `(180,)` | diagnostics |
| `dsigma_prj_gl/{vals, rnd, cl}` | the same for $\Delta\Sigma$ | $M_\odot/(h\,\mathrm{pc}^2)$, `(180,)` | diagnostics, $\Delta\Sigma$-mode likelihoods |
| `shear_prj_gl/{vals, rnd, cl}` | $\gamma_t = \Delta\Sigma\,\langle\Sigma_{\rm crit}^{-1}\rangle(z^{\rm ob})$ | dimensionless, `(180,)` | `likelihoods` reads **`cl`** |

Unlike DES Y1's `shear_prj_frozen_physics` (which additionally aliases
its output to `shear_prj/*`), this module writes only its own sections;
`des_y3.ini` points the likelihood at `shear_prj_gl` explicitly. The
two modules' sections never collide, so both can run in one pipeline
for comparison.

## DES Y3 implementations

This module is the `0d` cell (zero adaptive dimensions; formerly
`fast_mass`) — the reference pipeline's choice per the "Recommended
methods" table of `src/pipelines/des_y3/README.md`. The namespace under
`src/pipelines/des_y3/shear_projection` also contains
({doc}`../pipeline_organization`); costs are per sample on the
180-point wall (Perlmutter CPU, A100 for CUDA):

| Dims | Backend | Implementation and status | Cost | Precision |
|---|---|---|---:|---|
| `3d` | CUDA | `DSigmaPrj3dGpu.so`; fully-coupled adaptive $(\ln\theta,z,\ln M)$ PAGANI diagnostic at `eps_rel = 1e-3`; the innermost-radius convergence study is still open (a global adaptive volume can miss the $\theta_R$ cusp while reporting a small internal error) | 95 s (463 s at 1e-4) | median 9.5e-4, max 2.2% vs the region-split GL — the GL side is the higher-precision one |
| `2d` | C++ | `ShearPrjCuhre.so`; feature-split $\theta$ log-GL with adaptive Cuhre/Vegas over $(z,\ln M)$ | ~72 s/point (full wall ≈ 3.6 h) | not yet measured |
| `0d` | Python | `shear_prj_gl.py` — exact-$z$ region-split GL port of `ShearPrjCore` | 270 ms | 1.6e-11 vs the exact evaluator; 5.5e-5 vs frozen production |
| `0d` | **C++ (this page)** | `ShearPrjGl.so`; exact-$z$ core emitting $\Sigma$, $\Delta\Sigma$ and shear in one pass | 154 ms | 9.9e-12 vs the exact-$z$ evaluator (same core) |
| `0d` | CUDA | `ShearPrjFrozenGpu.so`; CUDA port of the frozen DES Y1 machinery (issue #24 closed 2026-08-28) | 16 ms | ~1e-10 vs the CPU frozen module on `vals`/`rnd`/`cl` |
| radial series | — | not applicable / not planned | — | — |

The exact-$z$ region-split path is the current precision reference of
this observable; the `3d` PAGANI run is a diagnostic whose own error
estimate does not certify the cusp. Validator:
`python/0d/validate_vs_production.py`; tests in
{doc}`../testing/src_pipelines_des_y3`.
