# One-Halo Lensing

Computes the population-integrated one-halo tangential shear
$N_i[\gamma_t^{1h,\rm full}](R)$ — the centred + miscentred NFW profile
weighted by the same halo population as the number counts. The likelihood
divides by `numcounts_fast_mass/vals` to form the stacked per-cluster
profile and adds the projection term.

## Numerical framework

**The operator: ** The full integral — the population operator $N_i[f]$
({doc}`number_counts`) with the miscentering-mixture shear weight:

$$N_i[\gamma_t^{1h,\rm full}](R) = \int d\ln M \int dz\;
\Omega(z)\,\frac{dV}{d\Omega\,dz}\,\frac{dn}{d\ln M}(M,z)\,
S_{ij}(\ln M, z)\;\gamma_t^{1h,\rm full}(R; M, z),$$

$$\gamma_t^{1h,\rm full}(R; M, z) =
\Big[(1 - f_{\rm mis})\,\Delta\Sigma_{\rm NFW}(R, M)
+ f_{\rm mis}\,\Delta\Sigma_{\rm mis}\big(R, M;\, \tau_{\rm mis} R_\lambda\big)\Big]\,
\langle\Sigma_{\rm crit}^{-1}\rangle(z),$$

**Source and calibration: ** The count-weighted lensing operator follows
[DES Cluster et al. 2023](https://ui.adsabs.harvard.edu/abs/2023arXiv230906593A/abstract)
(arXiv:[2309.06593](https://arxiv.org/abs/2309.06593)), the reference
paper for this software suite, using the DES Y3 redMaPPer miscentring
calibration of Kelly et al. 2024 (an updated analysis of Zhang et al.
2019). Full citations and the gamma offset kernel derivation:
{doc}`../math/index`, §"One-halo lensing and miscentering".

**Target-cluster vs. neighbouring-halo miscentering: ** This module uses
**target-cluster miscentering** — the assigned redMaPPer centre offset
from the true halo centre, with $(f_{\rm mis}, \tau_{\rm mis})$ as
nuisance parameters — distinct from the parameter-free neighbouring-halo
offset inside the two-halo term ({doc}`shear_projection`). Both mixture
pieces are linear in $\Delta\Sigma$, so the one-halo + projection sum in
the likelihood is exact (tangential shear, not reduced shear —
{doc}`../math/index`).

**Performance: ** Evaluation is fixed Gauss–Legendre with the redshift
axis contracted once per sample into mass weights $W_{ij}(\ln M)$; each
of the 180 wall points is then one 1-D mass sum ($\sim 16\times$ faster
than the retired per-(bin, $R$) Cuhre path, deterministic cost). The
complete step-by-step recipe lives in {doc}`../numerics/index`,
§"The number-counts and one-halo lensing recipe, step by step". Setting
`miscentering/f_mis = 0` recovers the centred-only `Shear1hSel` result
({doc}`../variants`).

## DES Y3 implementations

This module is the `fast_mass` cell — the reference pipeline's choice
per `src/pipelines/des_y3/README.md`'s own "Reference pipeline choices"
table. Other implementations below
`src/pipelines/des_y3/observables/shear_1h2h` provide the reference and
alternative cells described in {doc}`../pipeline_organization`:

| Strategy | Backends | Implementation and status |
|---|---|---|
| `full_ltmz` | Python, C++, CUDA | Explicit $(\lambda_{\rm true},\ln M,z)$ one-halo miscentred references |
| `fast_mass` | **C++ (this page)**, Python | Exact redshift contraction and direct mass sum; `Shear1hFastMass.so` is bitwise-equivalent to `Shear1hMisSel.so` |
| `radial_series` | Python, C++ | Offline $U_\ell$ tables plus per-sample population moments; a review comment flags a possible double-counted miscentering term here — see {doc}`../variants` |
| `fast_mass` max model | Python, C++, CUDA | `Shear1h2hMax` — traditional $\max(1h,b\,2h)$ model option, not part of this reference pipeline — see {doc}`../variants` |

The radial-series tables are versioned under `data/radial_series` and are
loaded, never regenerated, during sampling. The `full_ltmz` cells are the
accuracy references; the fast and series paths retain their own documented
physics and interpolation approximations.


## Script

- Model: [`src/models/n_operator_sel_gl_t.hh`](https://github.com/estevesjh/y3_cluster_cpp/blob/pipelines/des_y3/src/models/n_operator_sel_gl_t.hh)
  (`nosel_gl_detail::SelGLCore`, shared with {doc}`number_counts`) +
  [`src/modules/num_counts_sel/lensing_weights.hh`](https://github.com/estevesjh/y3_cluster_cpp/blob/pipelines/des_y3/src/modules/num_counts_sel/lensing_weights.hh)
  (unchanged production model, immutable dependency).
- Module driver: [`src/pipelines/des_y3/observables/shear_1h2h/fast_mass/cpp/Shear1hFastMass.cc`](https://github.com/estevesjh/y3_cluster_cpp/blob/pipelines/des_y3/src/pipelines/des_y3/observables/shear_1h2h/fast_mass/cpp/Shear1hFastMass.cc)
  (`DEFINE_COSMOSIS_SCALAR_EVALUATOR_MODULE`) — bitwise-equivalent to
  DES Y1's `Shear1hMisSel.so` ({doc}`../variants`), own module label
  and output section.
- Compiled library loaded by CosmoSIS:
  `${Y3_CLUSTER_CPP_DIR}/release-build/src/modules/des_y3_shear_fast_mass_cpp/Shear1hFastMass.so`.
- Disk tables: `data/nfw_off_center/*gamma*` — $1000 \times 1000$ log-log
  grids of the gamma-kernel miscentred NFW in
  $(R/r_s, R_{\rm mis}/r_s)$, loaded once at module construction.

## CosmoSIS setup

```ini
[Shear1hFastMass]
file = ${Y3_CLUSTER_CPP_DIR}/release-build/src/modules/des_y3_shear_fast_mass_cpp/Shear1hFastMass.so
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
2nd-order moment expansion); `Shear1hFastMass.so` always does the exact
GL mass sum ({doc}`../variants`).

## DataBlock inputs

Everything {doc}`NumCountsFastMass <number_counts>` reads, plus:

| DataBlock input | Meaning | Units / shape | Produced by |
|---|---|---|---|
| `haloModel/{r_sigma, lnM, dSigma_nfw}` | centred NFW $\Delta\Sigma(R, M)$ spline | cMpc/$h$; `(100, 128)` | `halo_model` (`compute_lensing_1h = T`) |
| `average_sigma_crit_inv/{zlense, sci_average}` | $\langle\Sigma_{\rm crit}^{-1}\rangle(z)$, folded into the $z$ weight | `(50,)` | `average_sigma_crit_inv` |
| `miscentering/f_mis`, `miscentering/tau_mis` | miscentred fraction and offset scale | scalars | sampler if declared; in-code defaults 0.22 / 0.17 |
| `cosmological_parameters/omega_m` | $\bar\rho_m$ multiplier of the miscentred table | scalar | `consistency` |

## DataBlock outputs

| DataBlock output | Meaning | Units / shape | Consumed by |
|---|---|---|---|
| `shear1h_fast_mass/vals` | $N_i[\gamma_t^{1h,\rm full}](R)$, bin slow / radius fast | `(180,)` | `likelihoods` |

DES Y1's `Shear1hMisSel.so` writes `shear1hmissel/vals` instead — the
two sections never collide ({doc}`../variants`).
