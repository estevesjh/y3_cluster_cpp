# One-Halo Lensing

`C++` · `y3_cluster_cpp` (`src/pipelines/des_y3`) · `Cluster observable` · module `Shear1hGl` · `~28 ms/sample`

Computes the population-integrated one-halo tangential shear
$N_i[\gamma_t^{1h,\rm full}](R)$ — the centred + miscentred NFW profile
weighted by the same halo population as the number counts. The likelihood
divides by `numcounts_sij_gl/vals` to form the stacked per-cluster
profile and adds the projection term.

## Script

- Model: [`src/pipelines/shared/sel_gl_weights.hh`](https://github.com/estevesjh/y3_cluster_cpp/blob/pipelines/des_y3/src/pipelines/shared/sel_gl_weights.hh)
  (`y3_pipelines::SelGlWeights`, shared with {doc}`number_counts` and
  identity-certified against the production `SelGLCore`) +
  [`src/pipelines/shared/lensing_helpers.hh`](https://github.com/estevesjh/y3_cluster_cpp/blob/pipelines/des_y3/src/pipelines/shared/lensing_helpers.hh)
  (pipeline-owned lensing helpers; production `Shear1hMisSel.so` keeps
  its own `SelGLCore` + `lensing_weights.hh` pair, untouched).
- Module driver: [`src/pipelines/des_y3/shear_1h2h/cpp/0d/Shear1hGl.cc`](https://github.com/estevesjh/y3_cluster_cpp/blob/pipelines/des_y3/src/pipelines/des_y3/shear_1h2h/cpp/0d/Shear1hGl.cc)
  (`DEFINE_COSMOSIS_SCALAR_EVALUATOR_MODULE`) — bitwise-equivalent to
  DES Y1's `Shear1hMisSel.so` ({doc}`../variants`), own module label
  and output section.
- Compiled library loaded by CosmoSIS:
  `${Y3_CLUSTER_CPP_DIR}/release-build/src/modules/des_y3_shear1h_0d_cpp/Shear1hGl.so`.
- Disk tables: `data/nfw_off_center/*gamma*` — $1000 \times 1000$ log-log
  grids of the gamma-kernel miscentred NFW in
  $(R/r_s, R_{\rm mis}/r_s)$, loaded once at module construction, built
  at a single **frozen $\rho_{\rm ref}=\rho_{m,0}$** — see "Evaluating the
  frozen table at any redshift" below for how a per-sample $z$ is folded
  in without rebuilding it.

## DES Y3 implementations

This module is the `0d` $z$-contracted cell (formerly `fast_mass`; zero adaptive dimensions) — the reference pipeline's choice
per `src/pipelines/des_y3/README.md`'s own "Reference pipeline choices"
table. Other implementations below
`src/pipelines/des_y3/shear_1h2h` provide the reference and
alternative cells described in {doc}`../pipeline_organization`:

| Dims | Backends | Implementation and status | Precision vs 3d |
|---|---|---|---|
| `3d` | C++, CUDA | Explicit $(\lambda_{\rm true},\ln M,z)$ adaptive one-halo miscentred references | 3.3e-4 / 3.4e-4 (vs the adaptive certifier) |
| `0d` | Python | Explicit fixed-GL 3-dim grid | 4.9e-5 |
| `0d` | **C++ (this page)**, Python | Exact redshift contraction, 1-dim GL mass sum; `Shear1hGl.so` is bitwise-equivalent to `Shear1hMisSel.so` (identity recorded separately) | 8.4e-4 |
| `0d` | Python, C++ | Offline $U_\ell$ radial-series tables plus per-sample population moments; a review comment flags a possible double-counted miscentering term here — see {doc}`../variants` | 3.7e-3 truncation vs same-profile fiducial; the 56–86% amplitude offset was a density-convention mismatch, RESOLVED 2026-08-24 (unified `rho_m_ref`); residual -10.6%–+3.9% is the fixed-c=4-family-vs-Child18 concentration mismatch, accepted per the fixed-concentration policy decision — see `docs/known_issues/radial_series_vs_full_ltmz_defect.md` |
| `0d` | Python, C++, CUDA | `Shear1h2hMax` — traditional $\max(1h,b\,2h)$ model, $z$-resolved 2-dim GL sum, not part of this reference pipeline — see {doc}`../variants` | 8.3e-4 (through the Python max chain) |

The radial-series tables are versioned under `data/radial_series` and are
loaded, never regenerated, during sampling. The `3d` cells are the
accuracy references; the fast and series paths retain their own documented
physics and interpolation approximations.

## Numerical framework

The full integral — the population operator $N_i[f]$
({doc}`number_counts`) with the miscentering-mixture shear weight:

$$N_i[\gamma_t^{1h,\rm full}](R) = \int d\ln M \int dz\;
\Omega(z)\,\frac{dV}{d\Omega\,dz}\,\frac{dn}{d\ln M}(M,z)\,
S_{ij}(\ln M, z)\;\gamma_t^{1h,\rm full}(R; M, z),$$

$$\gamma_t^{1h,\rm full}(R; M, z) =
\Big[(1 - f_{\rm mis})\,\Delta\Sigma_{\rm NFW}(R, M)
+ f_{\rm mis}\,\Delta\Sigma_{\rm mis}\big(R, M;\, \tau_{\rm mis} R_\lambda\big)\Big]\,
\langle\Sigma_{\rm crit}^{-1}\rangle(z),$$

The count-weighted lensing operator and its miscentering treatment
follow
[DES Cluster et al. 2023](https://ui.adsabs.harvard.edu/abs/2023arXiv230906593A/abstract)
(arXiv:[2309.06593](https://arxiv.org/abs/2309.06593)), the reference
paper for this software suite, using the DES Y3 redMaPPer miscentring
calibration of
[Kelly et al. 2024, MNRAS 533, 572](https://ui.adsabs.harvard.edu/abs/2024MNRAS.533..572K/abstract)
(arXiv:[2310.13207](https://arxiv.org/abs/2310.13207)) — the Gamma
offset kernel, an updated analysis of the DES Y1 calibration of
[Zhang et al. 2019, MNRAS 487, 2578](https://ui.adsabs.harvard.edu/abs/2019MNRAS.487.2578Z/abstract)
(arXiv:[1901.07119](https://arxiv.org/abs/1901.07119)) —

with $R_\lambda = (\lambda/100)^{0.2}\,h^{-1}$Mpc per richness bin.
This is **target-cluster miscentering** — the assigned redMaPPer centre
offset from the true halo centre, with $(f_{\rm mis}, \tau_{\rm mis})$
as nuisance parameters — distinct from the parameter-free
neighbouring-halo offset inside the two-halo term
({doc}`shear_projection`). Both mixture pieces are linear in
$\Delta\Sigma$, so the one-halo + projection sum in the likelihood is
exact (tangential shear, not reduced shear —
{doc}`../math/index`).

Evaluation is fixed Gauss–Legendre with the redshift axis contracted
once per sample into mass weights $W_{ij}(\ln M)$; each of the 180 wall
points is then one 1-D mass sum ($\sim 16\times$ faster than the
retired per-(bin, $R$) Cuhre path, deterministic cost). **The complete
step-by-step recipe lives in {doc}`../numerics/index`,
§"The number-counts and one-halo lensing recipe, step by step".**

Setting `miscentering/f_mis = 0` recovers the centred-only `Shear1hSel`
result ({doc}`../variants`). Model derivation: {doc}`../math/index`;
miscentering model: {doc}`../math/index`.

### Evaluating the frozen table at any redshift (`one_halo_physical_density`)

The NFW disk tables are built once, at a single fixed reference density
$\rho_{\rm ref}=\rho_{m,0}$ (comoving, frozen at $z=0$). Rebuilding them
per-sample at the halo's actual physical mean density,
$\rho_m(z)=\rho_{m,0}(1+z)^3$ — the convention Buzzard's own halo
catalog uses (issue #22) — would mean tabulating over mass **and**
dozens of $z$ values instead of one table. NFW self-similarity avoids
that entirely: the profile has exactly one length scale, $r_s$, and
every lensing observable has the Wright & Brainerd (2000) form
$(\Sigma,\bar\Sigma) = \Sigma_0\cdot(\text{dimensionless function of }R/r_s)$
with the **same** prefactor $\Sigma_0 \equiv 2\rho_s r_s$ for both, so
$\Delta\Sigma(R) = \Sigma_0\cdot F(R/r_s)$ too. Only $\rho_{\rm ref}$
changes between conventions ($M$, $c$ are the same halo); at fixed
$(M,c)$,

$$r_{200}=\Big[\frac{3M}{4\pi\cdot200\cdot\rho_{\rm ref}}\Big]^{1/3},\quad
r_s=\frac{r_{200}}{c},\quad \rho_s=\delta_c(c)\,\rho_{\rm ref}
\;\;\Longrightarrow\;\;
\Sigma_0 = 2\rho_s r_s \propto \rho_{\rm ref}^{2/3}.$$

($\rho_s\propto\rho_{\rm ref}^{1}$, $r_s\propto\rho_{\rm ref}^{-1/3}$;
the product nets the $2/3$ power.) Plugging in the physical/frozen
density ratio, $\rho_{\rm ref,phys}/\rho_{\rm ref,frozen}=(1+z)^3$,
gives $\Sigma_0$'s redshift dependence directly:
$[(1+z)^3]^{2/3}=(1+z)^2$. Combined with the $r_s\propto\rho_{\rm
ref}^{-1/3}$ shrink — smaller $r_s$ at fixed physical $R$ is the same
argument as querying the **frozen**-table shape at a bigger radius,
$R(1+z)$ — the exact identity is

$$\Delta\Sigma_{\rm phys}(R\mid z) = (1+z)^2\;\Delta\Sigma_{\rm frozen}\big(R\,(1+z)\big).$$

So the physical-density evaluation costs **zero new tables**: query the
one frozen-$z{=}0$ disk table at $R(1+z)$ instead of $R$, multiply by
$(1+z)^2$. Opt-in via `[halo_model] one_halo_physical_density = T`
(incompatible with `one_halo_z_density \neq 0`); implemented as
`z_amp_power=2` folded into the $z$-integration weight
(`sel_gl_weights.hh::build_weights`) and the `R\to R(1+z)` query
rescale at the bin's $z_{\rm eff}$ (`shear1h_gl_t.hh::evaluate`). Not
yet implemented for `Shear1h2hMaxGpu` or the Python explicit/max
mirrors (fails loudly there). See GitHub issue #22 for the Buzzard
validation this identity answers.

## CosmoSIS setup

```ini
[Shear1hGl]
file = ${Y3_CLUSTER_CPP_DIR}/release-build/src/modules/des_y3_shear1h_0d_cpp/Shear1hGl.so
bin_index = 0 1 2 3 4 5 6 7 8 9 10 11
r_perp = 0.0426 0.0669 0.1045 0.1652 0.2607 0.4117 0.6505 1.0257 1.6181 2.5537 4.0265 6.3490 10.0107 15.7832 24.8771
zt_low  = 0.05
zt_high = 0.80
lnm_low  = 29.9336
lnm_high = 36.7300
n_lnm = 96
n_z   = 64
lob_centers = 25.0 37.5 52.5 130.0
```

- Ordering: after `sel_function`, `halo_model` (with
  `compute_lensing_1h = T` — it reads the NFW $\Delta\Sigma$ table),
  `average_sigma_crit_inv`, `MfTinker`, `cp_camb`.
- Grid: 12 bins × 15 radii = **180** points, matching the Y1 WL covariance
  layout (`wl_cov.txt`) asserted by the likelihood.
- No adaptive-Cuhre knobs — fixed Gauss-Legendre only, same `n_lnm`/`n_z`
  convention as {doc}`number_counts`.

## Configuration options

| Option | Meaning | Units | Reference value |
|---|---|---|---|
| `bin_index` × `r_perp` | Cartesian wall grid (bin slow, $R$ fast) | $R$: cMpc/$h$ | 12 × 15 |
| `zt_low`, `zt_high` | true-redshift limits | — | 0.05, 0.80 |
| `lnm_low`, `lnm_high` | mass limits | $\ln(M_\odot/h)$ | 29.9336, 36.7300 |
| `n_lnm`, `n_z` | GL nodes | — | 96, 64 (defaults) |
| `lob_centers` | richness-bin centres driving $R_\lambda$ | — | 25 37.5 52.5 130 (default) |

DES Y1's `Shear1hMisSel.so` additionally supports `method = idea2` (a
2nd-order moment expansion); `Shear1hGl.so` always does the exact
GL mass sum ({doc}`../variants`).

## DataBlock inputs

Everything {doc}`NumCountsSijGl <number_counts>` reads, plus:

| DataBlock input | Meaning | Units / shape | Produced by |
|---|---|---|---|
| `haloModel/{r_sigma, lnM, dSigma_nfw}` | centred NFW $\Delta\Sigma(R, M)$ spline | cMpc/$h$; `(100, 128)` | `halo_model` (`compute_lensing_1h = T`) |
| `average_sigma_crit_inv/{zlense, sci_average}` | $\langle\Sigma_{\rm crit}^{-1}\rangle(z)$, folded into the $z$ weight | `(50,)` | `average_sigma_crit_inv` |
| `miscentering/f_mis`, `miscentering/tau_mis` | miscentred fraction and offset scale | scalars | sampler if declared; in-code defaults 0.22 / 0.17 |
| `cosmological_parameters/omega_m` | $\bar\rho_m$ multiplier of the miscentred table | scalar | `consistency` |

## DataBlock outputs

| DataBlock output | Meaning | Units / shape | Consumed by |
|---|---|---|---|
| `shear1h_gl/vals` | $N_i[\gamma_t^{1h,\rm full}](R)$, bin slow / radius fast | `(180,)` | `likelihoods` |

DES Y1's `Shear1hMisSel.so` writes `shear1hmissel/vals` instead — the
two sections never collide ({doc}`../variants`).
