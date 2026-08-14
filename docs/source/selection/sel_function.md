# Selection Function

`Python` · `y3_cluster_cpp` · `Selection` · module `sel_function` · `197 ms/sample`

Pre-tabulates, once per sample, the joint richness + photo-$z$ selection
function $S_{ij}(\ln M, z)$ for all 12 observed bins on one shared grid.
`NumCountsSel` and `Shear1hMisSel` slice their bin's plane from the packed
tensor and interpolate it inside their population integrals.

## Script

- Source: [`src/modules/sel_function/sel_function.py`](https://github.com/estevesjh/y3_cluster_cpp/blob/d7feb7504ed5dfcad84f99a1791af8a55c858aa0/src/modules/sel_function/sel_function.py)
  (`y3_cluster_cpp` @ `d7feb75`).
- Loaded by CosmoSIS as a Python module.

The maintained DES Y3 namespace also contains
`src/pipelines/des_y3/shared/sel_function.py`. As of 2026-08-12 it is an
exact staged copy of this production entry point, and namespace validators
load it through `shared/sel_kernels.py`. This is not yet a runtime cutover:
the reference ini continues to load `src/modules/sel_function/sel_function.py`.
See {doc}`../pipeline_organization` for the compatibility boundary.

## Numerical framework

The full integral: the **richness selection function** — the probability
that a halo of mass $M$ at true redshift $z^{\rm tr}$ is observed inside
the $(i, j)$ bin — is the redMaPPer selection function
$\mathcal S_{ij}$ integrated against the intrinsic richness–mass
relation,

$$S_{ij}(M, z^{\rm tr}) =
\int_{\Delta\lambda_i}\! d\lambda^{\rm ob}
\int_{\Delta z_j}\! dz^{\rm ob}
\int_0^\infty\! d\lambda^{\rm tr}\;
P(\lambda^{\rm ob}\mid\lambda^{\rm tr}, z^{\rm tr})\,
P(z^{\rm ob}\mid z^{\rm tr}, \Delta\lambda_i)\,
P(\lambda^{\rm tr}\mid M, z^{\rm tr}).$$

Both bin integrals are analytic, so the tabulated tensor is

$$S_{ij}(\ln M, z) = S_i(\ln M, z)\,\mathcal S_j(z)
= \Big[\textstyle\sum_k W_k\,
\mathcal S_i(\lambda_k, z)\, P_{\rm HOD}(\lambda_k \mid M, z)\Big]\,
\mathcal S_j(z),$$

with the **observed-richness kernel** $\mathcal S_i$ (the closed-form
Gaussian + EMG CDF difference at the bin edges,
{doc}`../modules/richness_mass`), the **observed-redshift kernel**
$\mathcal S_j$ (a Gaussian CDF difference of width `sigma_z`,
{doc}`../modules/redshift_kernel`), and the shifted-Poisson
$P_{\rm HOD}$. Only the $\lambda^{\rm tr}$ integral is numerical.

The recipe, per sample:

1. Per $(\ln M, z)$ cell, place $N_q = 32$ Gauss–Legendre nodes in
   $\lambda^{\rm tr}$ on the adaptive bracket
   $[\mu_{\rm eff} - L\sigma_{\rm eff},\, \mu_{\rm eff} +
   L\sigma_{\rm eff}]$, with
   $\sigma_{\rm eff} = \sqrt{\mu_{\rm sat} +
   (\sigma_\lambda\mu_{\rm sat})^2}$ and $L = 6$; evaluate
   $P_{\rm HOD}$ on the full $(192, 64, 32)$ tensor in a single
   `gammaln` call (narrow-Gaussian fallback where
   $\mu_{\rm sat} \le 10^{-8}$).
2. Evaluate the 8 EMG coefficient splines on the 1-D $z$ grid only
   (saves $\sim 130$ ms/sample), broadcast to
   $(\mu, \sigma, \tau, f^{\rm prj})$ at each node.
3. Compute the EMG CDF via `erfcx` at the **5 unique bin edges**
   $\{20, 30, 45, 60, 200\}$ and difference, giving all four
   $\mathcal S_i$ tables at once.
4. Per bin: contract
   $S_i = \sum_k W_k\, \mathcal S_i\, P_{\rm HOD}$, multiply by
   $\mathcal S_j(z)$, pack into `S_stack`.

The Python kernels match the C++ models
(`src/models/mor_hod_t.hh`, `src/models/richness_kernel_t.hh`)
line-for-line. Full derivation: {doc}`../math/index`.

## CosmoSIS setup

```ini
[sel_function]
file = ${Y3_CLUSTER_CPP_DIR}/src/modules/sel_function/sel_function.py
lam_min = 20.0  30.0  45.0  60.0   20.0  30.0  45.0  60.0   20.0  30.0  45.0  60.0
lam_max = 30.0  45.0  60.0  200.0  30.0  45.0  60.0  200.0  30.0  45.0  60.0  200.0
zob_min = 0.20  0.20  0.20  0.20   0.35  0.35  0.35  0.35   0.50  0.50  0.50  0.50
zob_max = 0.35  0.35  0.35  0.35   0.50  0.50  0.50  0.50   0.65  0.65  0.65  0.65
sigma_z = 0.03  0.03  0.03  0.03   0.03  0.03  0.03  0.03   0.03  0.03  0.03  0.03
zt_low   = 0.05
zt_high  = 0.80
lnm_low  = 29.9336
lnm_high = 36.8414
n_lnm = 192
n_z   = 20
n_z_shared = 64
L_z    = 6.0
L_lam  = 6.0
N_q    = 32
```

- Requires `Y3_CLUSTER_CPP_DIR`.
- Ordering: after `consistency` (MOR parameters live in the sampled
  sections); before `NumCountsSel` and `Shear1hMisSel`.
- The 12 array entries define the bin wall: 4 richness bins
  $\{[20,30), [30,45), [45,60), [60,200)\}$ × 3 photo-$z$ bins
  $\{[0.20,0.35), [0.35,0.50), [0.50,0.65)\}$, richness index fastest.

## Configuration options

| Option | Meaning | Units | Reference value |
|---|---|---|---|
| `lam_min`, `lam_max` | per-bin observed-richness edges | — | 12-entry wall |
| `zob_min`, `zob_max` | per-bin observed-redshift edges | — | 12-entry wall |
| `sigma_z` | photo-$z$ scatter in the redshift kernel $\mathcal S_j$ | — | 0.03 (all bins) |
| `zt_low`, `zt_high` | shared true-$z$ grid envelope | — | 0.05, 0.80 |
| `lnm_low`, `lnm_high` | shared $\ln M$ grid envelope | $\ln(M_\odot/h)$ | 29.9336, 36.8414 |
| `n_lnm` | $\ln M$ nodes — whole-pipeline optimum 192; **64 is pathological** (GL resonance, 4.5% drift on counts) | — | 192 |
| `n_z_shared` | shared $z$ nodes | — | 64 |
| `N_q` | Gauss–Legendre nodes of the $\lambda^{\rm tr}$ quadrature | — | 32 |
| `L_lam`, `L_z` | quadrature bracket half-widths | units of $\sigma$ | 6.0, 6.0 |

### Numerical precision: choosing `n_lnm`

`n_lnm = 192` is a **whole-pipeline** optimum, not a per-module one:
coarsening the $S_{ij}$ mass grid forces the downstream Cuhre-based
`NumCountsSel` to refine harder, which cancels the time this module
saves. Measured sweep (2026-05-07, wall-clock per sample):

| `n_lnm` | `sel_function` | `NumCountsSel` | Total | Note |
|---:|---:|---:|---:|---|
| 256 | 0.38 s | 0.08 s | 1.36 s | accuracy ceiling |
| **192** | 0.22 s | 0.12 s | **1.15 s** | sweet spot (reference value) |
| 128 | 0.15 s | 0.27 s | 1.22 s | Cuhre refinement eats the savings |
| 64 | — | — | — | **pathological**: GL resonance, 4.5% drift on `NumCountsSel` — avoid |

Accuracy at `n_lnm = 192` vs `256`: $<0.05\%$ on `NumCountsSel` and
`Shear1hMisSel`.

## DataBlock inputs

| DataBlock input | Meaning | Units / shape | Produced by |
|---|---|---|---|
| `cluster_mor/{log10_Mmin, log10_M1 \| log10_ratio, alpha, epsilon, sigma_lambda}` | shifted-Poisson HOD richness–mass parameters | — | sampler (values file) |
| `plob_ltr_params/*` | EMG projection-kernel coefficient splines (optional; falls back to the table embedded in `y3_buzzard/prj_params.py`) | — | retired `prj_params` module / in-code fallback |

## DataBlock outputs

| DataBlock output | Meaning | Units / shape | Consumed by |
|---|---|---|---|
| `sel_function/lnM` | shared mass grid | `(192,)` | `NumCountsSel`, `Shear1hMisSel` |
| `sel_function/z` | shared redshift grid | `(64,)` | same |
| `sel_function/S_stack` | packed selection tensor $S_{ij}(\ln M, z)$, layout `(bin, z, lnM)`, C-contiguous | `(12, 64, 192)` | same |
